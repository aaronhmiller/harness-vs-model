"""Effect decomposition and report generation.

Everything here is designed around one question: does the RANKING of the two
factors flip between evals? A harness win on E1 alone proves little. A sign
change across evals, with the models, prompts and loop held fixed, is the
result that cannot be produced by rigging either factor.

Confidence intervals bootstrap over TASKS, not over individual attempts. Seeds
within a task are correlated -- treating 5 seeds on one task as 5 independent
observations would shrink every interval by roughly sqrt(5) and manufacture
significance that is not there.
"""
from __future__ import annotations

import json
import random
import statistics
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Iterable

N_BOOT = 2000


# ------------------------------------------------------------------ data access
def load(path: str) -> list[dict]:
    rows = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _by_task(rows: Iterable[dict]) -> dict[str, list[dict]]:
    d: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        d[r["task_id"]].append(r)
    return d


def select(rows: list[dict], **kw) -> list[dict]:
    return [r for r in rows if all(r.get(k) == v for k, v in kw.items())]


def rate(rows: list[dict]) -> float:
    """Pass rate, averaging within task first so every task carries equal weight."""
    if not rows:
        return float("nan")
    per_task = [
        sum(1 for r in rs if r["passed"]) / len(rs)
        for rs in _by_task(rows).values()
    ]
    return statistics.fmean(per_task) if per_task else float("nan")


# ------------------------------------------------------------------- bootstrap
def boot_ci(rows: list[dict], fn=rate, n: int = N_BOOT, alpha: float = 0.05,
            rng: random.Random | None = None) -> tuple[float, float]:
    rng = rng or random.Random(0)
    groups = list(_by_task(rows).values())
    if len(groups) < 2:
        return (float("nan"), float("nan"))
    stats = []
    for _ in range(n):
        sample = [g for _ in range(len(groups)) for g in (rng.choice(groups),)]
        flat = [r for g in sample for r in g]
        stats.append(fn(flat))
    stats.sort()
    lo = stats[int(alpha / 2 * len(stats))]
    hi = stats[int((1 - alpha / 2) * len(stats)) - 1]
    return (lo, hi)


def boot_diff_ci(rows_a: list[dict], rows_b: list[dict], n: int = N_BOOT,
                 alpha: float = 0.05, seed: int = 0) -> tuple[float, float]:
    """CI for rate(a) - rate(b), resampling the SAME task ids in both arms."""
    rng = random.Random(seed)
    ga, gb = _by_task(rows_a), _by_task(rows_b)
    ids = sorted(set(ga) & set(gb))
    if len(ids) < 2:
        return (float("nan"), float("nan"))
    stats = []
    for _ in range(n):
        pick = [rng.choice(ids) for _ in ids]
        fa = [r for i in pick for r in ga[i]]
        fb = [r for i in pick for r in gb[i]]
        stats.append(rate(fa) - rate(fb))
    stats.sort()
    return (stats[int(alpha / 2 * len(stats))],
            stats[int((1 - alpha / 2) * len(stats)) - 1])


# ----------------------------------------------------------------------- stats
@dataclass
class Cell:
    model: str
    harness: str
    eval_id: str
    n: int
    pass_rate: float
    ci: tuple[float, float]
    mean_tokens: float
    mean_rounds: float
    pass_per_1k: float
    errors: int


def cell(rows: list[dict], model: str, harness: str, eval_id: str) -> Cell:
    rs = select(rows, model=model, harness=harness, eval_id=eval_id)
    pr = rate(rs)
    tok = statistics.fmean([r["prompt_tokens"] + r["completion_tokens"] for r in rs]) if rs else 0.0
    rounds = statistics.fmean([r["rounds"] for r in rs]) if rs else 0.0
    return Cell(
        model=model, harness=harness, eval_id=eval_id, n=len(rs),
        pass_rate=pr, ci=boot_ci(rs) if rs else (float("nan"),) * 2,
        mean_tokens=tok, mean_rounds=rounds,
        pass_per_1k=(pr / tok * 1000) if tok else float("nan"),
        errors=sum(1 for r in rs if r.get("error")),
    )


def effects(rows: list[dict], models: list[str], eval_id: str,
            h1: str = "h1_naive", h2: str = "h2_loop") -> dict[str, Any]:
    """Main effects, interaction, and the harness gain expressed in scale steps."""
    small, large = models[0], models[-1]

    def r(m, h):
        return rate(select(rows, model=m, harness=h, eval_id=eval_id))

    per_model_harness_gain = {m: r(m, h2) - r(m, h1) for m in models}
    d_harness = statistics.fmean(per_model_harness_gain.values())

    per_harness_model_gain = {h: r(large, h) - r(small, h) for h in (h1, h2)}
    d_model = statistics.fmean(per_harness_model_gain.values())

    steps = max(1, len(models) - 1)
    per_step = d_model / steps

    ci_h = boot_diff_ci(
        [x for m in models for x in select(rows, model=m, harness=h2, eval_id=eval_id)],
        [x for m in models for x in select(rows, model=m, harness=h1, eval_id=eval_id)],
        seed=1,
    )
    ci_m = boot_diff_ci(
        [x for h in (h1, h2) for x in select(rows, model=large, harness=h, eval_id=eval_id)],
        [x for h in (h1, h2) for x in select(rows, model=small, harness=h, eval_id=eval_id)],
        seed=2,
    )

    return {
        "eval_id": eval_id,
        "delta_harness": d_harness,
        "delta_harness_ci": ci_h,
        "delta_model": d_model,
        "delta_model_ci": ci_m,
        "interaction": per_model_harness_gain[small] - per_model_harness_gain[large],
        "harness_gain_by_model": per_model_harness_gain,
        "model_gain_by_harness": per_harness_model_gain,
        "model_gain_per_step": per_step,
        "harness_in_scale_steps": (d_harness / per_step) if abs(per_step) > 1e-9 else float("nan"),
        "winner": "harness" if d_harness > d_model else "model",
    }


def difficulty_bands(rows: list[dict], models: list[str], eval_id: str = "e1_code",
                     h1: str = "h1_naive", h2: str = "h2_loop") -> list[dict]:
    """Tests the 'non-zero base rate' clause: harness gain vs baseline difficulty.

    Prediction is an inverted U -- no gain where the model cannot start, no gain
    where it already succeeds, large gain in between.

    NOTE: this section is meaningless on mock data. The mock repairs with a flat
    probability regardless of task difficulty, so it cannot exhibit a floor
    effect. Teaching it to would be building in the result. Real runs only.
    """
    base = _by_task(select(rows, harness=h1, eval_id=eval_id))
    loop = _by_task(select(rows, harness=h2, eval_id=eval_id))
    per_task = []
    for tid, rs in base.items():
        if tid not in loop:
            continue
        b = sum(1 for r in rs if r["passed"]) / len(rs)
        l = sum(1 for r in loop[tid] if r["passed"]) / len(loop[tid])
        per_task.append((tid, b, l))

    bands = [("floor: model cannot start (0-15%)", 0.0, 0.15),
             ("workable middle (15-85%)", 0.15, 0.85),
             ("ceiling: already solved (85-100%)", 0.85, 1.01)]
    out = []
    for label, lo, hi in bands:
        sel = [(t, b, l) for (t, b, l) in per_task if lo <= b < hi]
        out.append({
            "band": label,
            "n_tasks": len(sel),
            "baseline": statistics.fmean([b for _, b, _ in sel]) if sel else float("nan"),
            "with_loop": statistics.fmean([l for _, _, l in sel]) if sel else float("nan"),
            "gain": (statistics.fmean([l - b for _, b, l in sel]) if sel else float("nan")),
        })
    return out


def ladder_health(rows: list[dict], models: list[str],
                  harness: str = "h1_naive", eval_id: str = "e1_code") -> dict:
    """Judge the LADDER, not each rung.

    The 30-60% target is about keeping both axes sensitive, and it applies to
    the ladder as a whole. Reading it per-model leads somewhere badly wrong:
    reject every rung that scores above 60%, keep only the smallest model, and
    you have an experiment with no model axis at all -- Δ model is undefined,
    so "harness beats model" compares a number against nothing.

    Three things actually have to be true:
      bottom rung not on the floor   -- the repair loop needs something to repair
      top rung not on the ceiling    -- the repair loop needs room to improve
      a real gap between them        -- otherwise Δ model ≈ 0 by construction and
                                        the harness wins vacuously
    """
    ERROR_LIMIT = 0.20

    per_model, broken = [], []
    for m in models:
        rs = select(rows, model=m, harness=harness, eval_id=eval_id)
        if not rs:
            continue
        n_err = sum(1 for r in rs if r.get("error"))
        err_rate = n_err / len(rs)
        by_task = _by_task(rs)
        task_rates = {t: sum(1 for r in v if r["passed"]) / len(v)
                      for t, v in by_task.items()}
        entry = {
            "model": m,
            "baseline": statistics.fmean(task_rates.values()) if task_rates else float("nan"),
            "n_tasks": len(task_rates),
            "n_attempts": len(rs),
            "errors": n_err,
            "error_rate": err_rate,
            "floor": sum(1 for v in task_rates.values() if v == 0.0),
            "ceiling": sum(1 for v in task_rates.values() if v == 1.0),
            "workable": sum(1 for v in task_rates.values() if 0.0 < v < 1.0),
        }
        # A cell that mostly errored measured NOTHING. Reporting it as a 0%
        # baseline is a category error: "the model failed" and "the request
        # never reached a model" are different facts, and averaging the second
        # into a capability curve silently poisons the model axis.
        (broken if err_rate > ERROR_LIMIT else per_model).append(entry)

    if not per_model:
        return {"models": [], "broken": broken, "usable": False,
                "checks": [], "reasons": [
                    "no usable calibration rows"
                    + (f" ({len(broken)} model(s) excluded as mostly errors)" if broken else "")]}

    per_model.sort(key=lambda d: d["baseline"])
    bottom, top = per_model[0], per_model[-1]
    slope = top["baseline"] - bottom["baseline"]
    thinnest = min(per_model, key=lambda d: d["workable"])

    checks = [
        ("no cell is mostly errors", not broken,
         f"{len(broken)} excluded", "0 excluded",
         "an all-errors cell is infrastructure failure, not model capability — "
         "delete those rows and re-run that model"),
        ("bottom rung off the floor", bottom["baseline"] >= 0.10,
         f"{100*bottom['baseline']:.1f}% ({bottom['model']})", ">= 10%",
         "at 0% there is nothing for a repair loop to repair"),
        ("top rung off the ceiling", top["baseline"] <= 0.85,
         f"{100*top['baseline']:.1f}% ({top['model']})", "<= 85%",
         "near 100% no harness can show a gain"),
        ("model axis alive", slope >= 0.15,
         f"{100*slope:+.1f} pts", ">= 15 pts",
         "without a real model gap, 'harness > model' is vacuous"),
        # Every rung needs partially-solved tasks, not just the bottom one: the
        # harness effect at a given model is measured only on that model's
        # workable tasks, so a saturated top rung has almost no signal left.
        ("every rung has usable tasks", thinnest["workable"] >= 8,
         f"{thinnest['workable']} ({thinnest['model']})", ">= 8",
         "a rung with few partially-solved tasks cannot show a harness effect"),
    ]
    if len(per_model) < 2:
        checks.insert(1, (
            "at least two models", False, f"{len(per_model)} model", ">= 2",
            "one model means no model axis: Δ model is undefined and the "
            "whole comparison is impossible",
        ))

    return {
        "models": per_model, "broken": broken,
        "bottom": bottom, "top": top, "slope": slope,
        "checks": checks,
        "usable": all(ok for _, ok, _, _, _ in checks),
        "reasons": [why for _, ok, _, _, why in checks if not ok],
    }


def ladder_report(rows: list[dict], models: list[str]) -> str:
    h = ladder_health(rows, models)
    out = ["# Ladder health — E1, single-shot (H1)\n"]
    if not h["models"]:
        return "No calibration rows found. Run `uv run hvm calibrate` first."

    out.append(md_table(
        ["model", "baseline", "tasks", "floor (0%)", "workable", "ceiling (100%)", "errors"],
        [[m["model"], _pct(m["baseline"]), m["n_tasks"],
          m["floor"], m["workable"], m["ceiling"],
          f"{m['errors']} ({100*m['error_rate']:.0f}%)"] for m in h["models"]]))

    if h.get("broken"):
        out.append("\n### ⚠ Excluded — these cells measured nothing\n")
        out.append(md_table(
            ["model", "attempts", "errors", "error rate"],
            [[b["model"], b["n_attempts"], b["errors"], f"{100*b['error_rate']:.0f}%"]
             for b in h["broken"]]))
        out.append(
            "\nA cell that mostly errored is an **infrastructure failure, not a "
            "capability measurement** — the requests never reached a model. It is "
            "excluded from the gate rather than averaged in as a 0% baseline.\n\n"
            "Find the cause, then delete those rows and re-run that model:\n\n"
            "```bash\n"
            "python3 - <<'PY'\n"
            "import json\n"
            "keep = [l for l in open('results/calibration.jsonl')\n"
            "        if json.loads(l)['model'] not in {"
            + ", ".join(repr(b["model"]) for b in h["broken"]) + "}]\n"
            "open('results/calibration.jsonl','w').writelines(keep)\n"
            "PY\n"
            "```"
        )

    out.append("\n## Gate\n")
    out.append(md_table(
        ["check", "observed", "needs", "verdict"],
        [[name, obs, need, "PASS" if ok else "**FAIL**"]
         for name, ok, obs, need, _ in h["checks"]]))

    if h["usable"]:
        out.append(
            f"\n**LADDER USABLE.** Bottom {_pct(h['bottom']['baseline']).strip()} "
            f"({h['bottom']['model']}), top {_pct(h['top']['baseline']).strip()} "
            f"({h['top']['model']}), slope {100*h['slope']:+.1f} pts. "
            f"Freeze the task set now and run the sweep — changing tasks after "
            f"scoring is p-hacking."
        )
    else:
        out.append("\n**LADDER NOT USABLE YET.** Why:\n")
        for r in h["reasons"]:
            out.append(f"- {r}")
        out.append(
            "\nFix by moving the whole ladder, not by discarding rungs: shift every "
            "model one size down (or up), or adjust task difficulty. Keep three "
            "models — dropping to one removes the axis you are trying to measure."
        )
    return "\n".join(out)


def trap_capture(rows: list[dict], specs: list[dict], models: list[str],
                 harnesses: list[str]) -> list[dict]:
    """Fraction of E2b answers that land on the ATTRACTIVE WRONG answer.

    If the loop raises this, it did not merely fail to help -- it amplified a
    counterfeit signal. That is the strongest single piece of evidence for
    scenario Y.
    """
    from .scoring import matches

    trap_by_id = {f"e2b:{s['id']}": (s.get("trap", []), bool(s.get("numeric"))) for s in specs}
    out = []
    for m in models:
        for h in harnesses:
            rs = select(rows, model=m, harness=h, eval_id="e2b_traps")
            if not rs:
                continue
            hits = 0
            for r in rs:
                traps, numeric = trap_by_id.get(r["task_id"], ([], False))
                if traps and matches(r["final_answer"], traps, numeric):
                    hits += 1
            out.append({"model": m, "harness": h, "n": len(rs),
                        "trap_rate": hits / len(rs)})
    return out


# --------------------------------------------------------------------- rendering
def _pct(x: float) -> str:
    return "  n/a" if x != x else f"{100 * x:5.1f}%"


def _signed(x: float) -> str:
    return " n/a" if x != x else f"{100 * x:+5.1f}"


def md_table(headers: list[str], rows: list[list[str]]) -> str:
    out = ["| " + " | ".join(headers) + " |",
           "|" + "|".join("---" for _ in headers) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(c) for c in r) + " |")
    return "\n".join(out)


def report(path: str, models: list[str], evals: list[str],
           harnesses: list[str] = ("h1_naive", "h2_loop", "bon_control")) -> str:
    rows = load(path)
    if not rows:
        return "No results found."

    present_h = [h for h in harnesses if any(r["harness"] == h for r in rows)]
    out: list[str] = ["# Harness vs Model — results\n"]

    # 0. provenance audit, before any number is believed
    served: dict[str, set[str]] = defaultdict(set)
    for r in rows:
        if r.get("served_model"):
            served[r["model"]].add(r["served_model"])
    suspect = {m: s for m, s in served.items() if len(s) > 1}
    if served:
        out.append("## 0. Which model actually answered\n")
        out.append(md_table(
            ["model key", "served by", "attempts"],
            [[m, ", ".join(sorted(s)), sum(1 for r in rows if r["model"] == m)]
             for m, s in sorted(served.items())]))
        if suspect:
            out.append(
                "\n> **⚠ MODEL AXIS COMPROMISED.** "
                + "; ".join(f"`{m}` was served by {len(s)} different ids ({', '.join(sorted(s))})"
                            for m, s in suspect.items())
                + ". Results filed under that key mix different weights, so every "
                  "Δ model below is meaningless. Delete those rows and re-run "
                  "that model before reporting anything."
            )

    # 1. raw cells
    out.append("## 1. Pass rate by cell\n")
    body = []
    for e in evals:
        for m in models:
            for h in present_h:
                c = cell(rows, m, h, e)
                if not c.n:
                    continue
                body.append([e, m, h, c.n, _pct(c.pass_rate),
                             f"[{_pct(c.ci[0]).strip()}, {_pct(c.ci[1]).strip()}]",
                             f"{c.mean_tokens:,.0f}", f"{c.mean_rounds:.1f}",
                             f"{c.pass_per_1k:.3f}", c.errors])
    out.append(md_table(
        ["eval", "model", "harness", "n", "pass", "95% CI", "tok/task",
         "calls", "pass/1k tok", "err"], body))

    # 2. effect decomposition -- the headline
    out.append("\n## 2. Which factor moved the needle\n")
    eff = {e: effects(rows, models, e) for e in evals}
    body = []
    for e in evals:
        x = eff[e]
        body.append([
            e,
            f"{_signed(x['delta_harness'])} [{_signed(x['delta_harness_ci'][0]).strip()}, {_signed(x['delta_harness_ci'][1]).strip()}]",
            f"{_signed(x['delta_model'])} [{_signed(x['delta_model_ci'][0]).strip()}, {_signed(x['delta_model_ci'][1]).strip()}]",
            _signed(x["interaction"]),
            "n/a" if x["harness_in_scale_steps"] != x["harness_in_scale_steps"]
            else f"{x['harness_in_scale_steps']:.2f}",
            x["winner"].upper(),
        ])
    out.append(md_table(
        ["eval", "Δ harness (pts)", "Δ model (pts)", "interaction",
         "harness in scale-steps", "winner"], body))
    out.append(
        "\n*Δ harness* = H2 − H1 averaged over models. *Δ model* = largest − smallest "
        "averaged over harnesses. *Interaction* > 0 means the loop helped the SMALL "
        "model more, i.e. the harness substitutes for capability."
    )

    # 3. the flip
    winners = {e: eff[e]["winner"] for e in evals}
    flipped = len(set(winners.values())) > 1
    out.append("\n## 3. Did the ranking flip?\n")
    out.append(md_table(["eval", "winner"], [[e, winners[e].upper()] for e in evals]))
    out.append(
        f"\n**{'FLIP OBSERVED' if flipped else 'NO FLIP'}** — "
        + ("the same models, prompts and loop produce opposite rankings on different "
           "evals, which is the signature of a conditional effect rather than a rigged "
           "comparison."
           if flipped else
           "one factor won everywhere. The conditional claim is NOT supported by this "
           "run; report it as such rather than reaching for the E1 number alone.")
    )

    # 4. compute-matched control
    if "bon_control" in present_h:
        out.append("\n## 4. Was it the feedback or just the tokens?\n")
        body = []
        for e in evals:
            for m in models:
                h2c, bonc = cell(rows, m, "h2_loop", e), cell(rows, m, "bon_control", e)
                if not h2c.n or not bonc.n:
                    continue
                body.append([e, m, _pct(h2c.pass_rate), _pct(bonc.pass_rate),
                             _signed(h2c.pass_rate - bonc.pass_rate),
                             f"{h2c.mean_tokens:,.0f}", f"{bonc.mean_tokens:,.0f}"])
        out.append(md_table(
            ["eval", "model", "H2 loop", "best-of-N", "Δ", "H2 tok", "BoN tok"], body))
        out.append(
            "\nBest-of-N spends a comparable budget with NO external signal. A positive Δ "
            "means the structure did the work; a Δ near zero means we bought the gain "
            "with tokens and should say so."
        )

    # 5. inverted U
    if "e1_code" in evals:
        out.append("\n## 5. Where the loop can and cannot help (E1)\n")
        out.append(md_table(
            ["baseline band", "tasks", "H1 baseline", "H2 with loop", "gain"],
            [[b["band"], b["n_tasks"], _pct(b["baseline"]),
              _pct(b["with_loop"]), _signed(b["gain"])]
             for b in difficulty_bands(rows, models)]))
        out.append(
            "\nThe 'non-zero base rate' clause, measured: a repair loop needs something "
            "to repair at one end and something left to fix at the other."
        )

    # 6. trap capture
    if "e2b_traps" in evals:
        from .tasks import E2B_SPECS
        tc = trap_capture(rows, E2B_SPECS, models, present_h)
        if tc:
            out.append("\n## 6. Trap capture rate (E2b)\n")
            out.append(md_table(
                ["model", "harness", "n", "landed on the attractive wrong answer"],
                [[t["model"], t["harness"], t["n"], _pct(t["trap_rate"])] for t in tc]))
            out.append(
                "\nIf H2's rate exceeds H1's, self-verification did not catch the error — "
                "it ratified it."
            )

    return "\n".join(out)


def summary_json(path: str, models: list[str], evals: list[str]) -> dict:
    rows = load(path)
    return {
        "models": models,
        "evals": evals,
        "cells": [dataclasses_asdict(cell(rows, m, h, e))
                  for e in evals for m in models
                  for h in ("h1_naive", "h2_loop", "bon_control")
                  if select(rows, model=m, harness=h, eval_id=e)],
        "effects": {e: effects(rows, models, e) for e in evals},
        "bands": difficulty_bands(rows, models) if "e1_code" in evals else [],
    }


def dataclasses_asdict(c: Cell) -> dict:
    import dataclasses as _d
    return _d.asdict(c)
