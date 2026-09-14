"""Command line entry point.

    uv run hvm health                    # can we reach the MLX server?
    uv run hvm calibrate                 # difficulty check BEFORE spending compute
    uv run hvm run                       # the sweep (resumable)
    uv run hvm report                    # markdown tables
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys

from .analyze import (difficulty_bands, ladder_health, ladder_report, load,
                      report, select, summary_json)
from .harnesses import DecodeSettings
from .providers.mlx_provider import ModelMismatch
from .runner import build_provider, run_sweep
from .serving import RuntimeSwitcher, SwitchError, server_models

DEFAULT_CONFIG = "config.json"


def load_config(path: str) -> dict:
    with open(path) as fh:
        if path.endswith((".yaml", ".yml")):
            import yaml  # optional
            return yaml.safe_load(fh)
        return json.load(fh)


def _decode(cfg: dict) -> DecodeSettings:
    d = cfg.get("decode", {})
    return DecodeSettings(
        temperature=d.get("temperature", 0.6),
        max_tokens=d.get("max_tokens", 900),
    )


def _base_url(cfg: dict) -> str:
    return cfg.get("runtime", {}).get("base_url", "http://127.0.0.1:8080/v1")


def _models(cfg: dict, args) -> dict:
    """Honour --only-model so one model can be run per server lifetime."""
    models = cfg["models"]
    only = getattr(args, "only_model", None)
    if not only:
        return models
    missing = [k for k in only if k not in models]
    if missing:
        raise SystemExit(f"unknown model key(s): {missing}. Known: {list(models)}")
    return {k: models[k] for k in only}


def _switcher(cfg: dict, args) -> RuntimeSwitcher | None:
    if getattr(args, "no_switch", False):
        return None
    rt = cfg.get("runtime", {})
    sw = RuntimeSwitcher(rt.get("switch"), _base_url(cfg))
    if not sw.enabled:
        return None
    ok, msg = sw.available()
    if not ok:
        raise SystemExit(f"cannot switch models: {msg}")
    return sw


def cmd_health(cfg: dict, args) -> int:
    base = _base_url(cfg)
    loaded = server_models(base)
    print(f"  endpoint       {base}")
    print(f"  /v1/models     {loaded or 'NO RESPONSE — is `agent-serve status` up?'}")

    if getattr(args, "no_switch", False):
        print("  model switch   DISABLED (--no-switch) — you drive agent-model yourself")
        switch_ok = True
    else:
        sw = RuntimeSwitcher(cfg.get("runtime", {}).get("switch"), base)
        switch_ok, msg = sw.available()
        print(f"  model switch   {'OK' if switch_ok else 'UNAVAILABLE'} — {msg}")

    selected = _models(cfg, args)
    if len(selected) < len(cfg["models"]):
        print(f"  checking       {list(selected)} (--only-model)")
    else:
        print("\n  Only the ACTIVE model answers; the rest are expected to fail until")
        print("  the runner swaps them in. That is the one-model-resident design.")
    print()

    any_ok = False
    for key, mc in selected.items():
        p = build_provider(key, mc, base)
        try:
            if hasattr(p, "resolve"):
                p.resolve(verbose=False)
                if p.served_model != p.model:
                    print(f"  {key:<16} config id {p.model!r} is not served; "
                          f"using {p.served_model!r}")
            out = p.healthcheck() if hasattr(p, "healthcheck") else "(mock)"
            print(f"  {key:<16} OK   -> {out!r}")
            any_ok = True
        except Exception as exc:
            print(f"  {key:<16} --   -> {str(exc)[:400]}")
    return 0 if (any_ok or not loaded) and switch_ok else 1


def cmd_calibrate(cfg: dict, args) -> int:
    """Single-shot baseline on E1. Run this BEFORE freezing the task set.

    We are looking for a mid-range baseline. Tasks pinned at 0% or 100% for
    every model carry no information about either factor, and a task set that is
    mostly pinned will make whichever factor we favour look decisive.
    """
    out_path = args.out or "results/calibration.jsonl"
    selected = _models(cfg, args)
    run_sweep(
        models=selected,
        harnesses=["h1_naive"],
        evals=["e1_code"],
        seeds=list(range(cfg.get("calibration_seeds", 3))),
        out_path=out_path,
        decode=_decode(cfg),
        limit=args.limit,
        switcher=_switcher(cfg, args),
        base_url=_base_url(cfg),
        e1_tiers=cfg.get("e1_tiers"),
    )
    rows = load(out_path)
    models = list(selected)
    print("\nPer-task single-shot baseline (H1, E1):\n")
    print(f"{'task':<22}" + "".join(f"{m:>14}" for m in models))
    pinned_low = pinned_high = 0
    task_ids = sorted({r["task_id"] for r in rows})
    for tid in task_ids:
        cells = []
        for m in models:
            rs = select(rows, task_id=tid, model=m, harness="h1_naive")
            cells.append(sum(1 for r in rs if r["passed"]) / len(rs) if rs else float("nan"))
        best = max([c for c in cells if c == c], default=0.0)
        worst = min([c for c in cells if c == c], default=0.0)
        if best == 0.0:
            pinned_low += 1
        if worst == 1.0:
            pinned_high += 1
        flag = "  <- floor" if best == 0.0 else ("  <- ceiling" if worst == 1.0 else "")
        print(f"{tid:<22}" + "".join(f"{100*c:13.0f}%" for c in cells) + flag)

    overall = [
        statistics.fmean([1.0 if r["passed"] else 0.0
                          for r in select(rows, model=m, harness="h1_naive")] or [0])
        for m in models
    ]
    print("\n" + "-" * (22 + 14 * len(models)))
    print(f"{'MEAN':<22}" + "".join(f"{100*o:13.1f}%" for o in overall))
    print(f"\nfloor-pinned tasks (0% for every model): {pinned_low}/{len(task_ids)}")
    print(f"ceiling-pinned tasks (100% for every model): {pinned_high}/{len(task_ids)}")
    lo, hi = cfg.get("target_baseline", [0.30, 0.60])
    print(f"\nPer-model baseline (guidance only — the GATE is the ladder, below):")
    for m, o in zip(models, overall):
        note = "mid-range" if lo <= o <= hi else ("high" if o > hi else "low")
        print(f"  {m:<14} {100*o:5.1f}%  {note}")
    print(
        "\nA rung reading 'high' is NOT a reason to drop that model. What matters is\n"
        "whether the ladder as a whole leaves room on both axes."
    )

    # The real gate, over every model present in the file -- including ones from
    # earlier runs with a different config.
    all_models = sorted({r["model"] for r in rows})
    print("\n" + "=" * 72)
    print(ladder_report(rows, all_models))
    return 0


def cmd_ladder(cfg: dict, args) -> int:
    """Read-only ladder verdict over calibration data already on disk."""
    path = args.out or "results/calibration.jsonl"
    if not os.path.exists(path):
        print(f"no calibration data at {path} — run `uv run hvm calibrate` first",
              file=sys.stderr)
        return 1
    rows = load(path)
    all_models = sorted({r["model"] for r in rows})
    print(ladder_report(rows, all_models))
    h = ladder_health(rows, all_models)
    return 0 if h["usable"] else 1


def cmd_run(cfg: dict, args) -> int:
    out_path = args.out or "results/results.jsonl"
    run_sweep(
        models=_models(cfg, args),
        harnesses=cfg.get("harnesses", ["h1_naive", "h2_loop", "bon_control"]),
        evals=cfg.get("evals", ["e1_code", "e2a_knowledge", "e2b_traps"]),
        seeds=list(range(cfg.get("seeds", 5))),
        out_path=out_path,
        decode=_decode(cfg),
        h2_rounds=cfg.get("h2_rounds", 3),
        bon_n=cfg.get("bon_n", 3),
        limit=args.limit,
        switcher=_switcher(cfg, args),
        base_url=_base_url(cfg),
        e1_tiers=cfg.get("e1_tiers"),
    )
    return 0


def cmd_report(cfg: dict, args) -> int:
    path = args.out or "results/results.jsonl"
    if not os.path.exists(path):
        print(f"no results at {path}", file=sys.stderr)
        return 1
    # Report over every configured model regardless of --only-model: a partial
    # sweep should still render, with the missing cells simply absent.
    models = list(cfg["models"])
    evals = cfg.get("evals", ["e1_code", "e2a_knowledge", "e2b_traps"])
    text = report(path, models, evals)
    print(text)
    if args.markdown:
        os.makedirs(os.path.dirname(args.markdown) or ".", exist_ok=True)
        with open(args.markdown, "w") as fh:
            fh.write(text + "\n")
        with open(args.markdown.replace(".md", ".json"), "w") as fh:
            json.dump(summary_json(path, models, evals), fh, indent=2)
        print(f"\n[written] {args.markdown}", file=sys.stderr)
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="hvm", description="Harness vs model experiment")
    ap.add_argument("-c", "--config", default=DEFAULT_CONFIG)
    ap.add_argument("--out", default=None, help="results jsonl path")
    ap.add_argument("--limit", type=int, default=None, help="first N tasks per eval")
    ap.add_argument(
        "--only-model", nargs="+", default=None, metavar="KEY",
        help="run only these model keys (repeatable). Use when you would rather "
             "drive `agent-model use` / `agent-serve restart` by hand.",
    )
    ap.add_argument(
        "--no-switch", action="store_true",
        help="never shell out to agent-model/agent-serve; assume the right model "
             "is already resident.",
    )
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("health", "calibrate", "run", "ladder"):
        sub.add_parser(name)
    rp = sub.add_parser("report")
    rp.add_argument("--markdown", default=None, help="also write markdown here")

    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    try:
        return {
            "health": cmd_health, "calibrate": cmd_calibrate,
            "run": cmd_run, "report": cmd_report, "ladder": cmd_ladder,
        }[args.cmd](cfg, args)
    except (ModelMismatch, SwitchError) as exc:
        # Operator errors, not bugs -- a traceback buries the one line that
        # says what to do about it.
        print(f"\n{exc}\n", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
