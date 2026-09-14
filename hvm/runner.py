"""Cell orchestration.

One "cell" is (model x harness x eval). Every cell runs every task at every
seed. Results stream to JSONL and the runner is resumable: re-running skips
work already on disk, so a long sweep survives a closed laptop.

MODEL IS THE OUTER LOOP, and deliberately so. The server holds one model
resident (`--max-loaded-models 1`), so changing model means swapping several GB
of weights. Grouping every harness and eval under one model means exactly
`len(models)` swaps per sweep instead of one per cell.
"""
from __future__ import annotations

import dataclasses
import json
import os
import sys
import time
from typing import Iterable, Iterator

from .harnesses import HARNESSES, DecodeSettings
from .providers import MLXProvider, MockProvider
from .serving import RuntimeSwitcher
from .tasks import EvalItem, load_eval
from .types import Attempt, Completion, Message, Provider


# --------------------------------------------------------------------- mock shim
class _MockItemProvider:
    """Injects per-task ground truth into the mock's context.

    Only the mock reads these sentinels; MLXProvider never sees them because the
    shim is only applied when the underlying provider is a MockProvider. This
    keeps harness code free of any knowledge that mocks exist.
    """

    def __init__(self, inner: MockProvider, hints: dict[str, str]) -> None:
        self.inner = inner
        self.name = inner.name
        self.hints = hints

    served_model = "mock"

    def generate(self, messages: list[Message], **kw) -> Completion:
        tag = (
            f"<<MOCK_KIND>>{self.hints.get('kind','code')}<</MOCK_KIND>>"
            f"<<MOCK_TRUTH>>{self.hints.get('truth','')}<</MOCK_TRUTH>>"
            f"<<MOCK_TRAP>>{self.hints.get('trap','')}<</MOCK_TRAP>>"
        )
        return self.inner.generate(
            [Message("system", tag)] + list(messages), **kw
        )


def _wrap(provider: Provider, item: EvalItem) -> Provider:
    if isinstance(provider, MockProvider):
        return _MockItemProvider(provider, item.mock_hints)  # type: ignore[return-value]
    return provider


# ------------------------------------------------------------------- providers
def build_provider(key: str, cfg: dict, base_url: str | None = None) -> Provider:
    kind = cfg.get("provider", "mlx")
    if kind == "mock":
        return MockProvider(cfg.get("model", key))
    return MLXProvider(
        model=cfg["model"],
        base_url=cfg.get("base_url") or base_url or "http://127.0.0.1:8080/v1",
        name=key,
        registry_key=cfg.get("registry_key", key),
    )


# ---------------------------------------------------------------------- runner
def _key(a: dict) -> tuple:
    return (a["model"], a["harness"], a["eval_id"], a["task_id"], a["seed"])


def load_done(path: str) -> set[tuple]:
    done: set[tuple] = set()
    if not os.path.exists(path):
        return done
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                done.add(_key(json.loads(line)))
            except (json.JSONDecodeError, KeyError):
                continue
    return done


def iter_cells(models: dict, harnesses: Iterable[str], evals: Iterable[str]) -> Iterator[tuple]:
    for model_key in models:
        for h in harnesses:
            for e in evals:
                yield model_key, h, e


def _has_work(model_key: str, harnesses, evals, eval_items, seeds, done: set) -> bool:
    return any(
        (model_key, h, e, item.id, s) not in done
        for h in harnesses for e in evals
        for item in eval_items[e] for s in seeds
    )


def run_sweep(
    models: dict,
    harnesses: list[str],
    evals: list[str],
    seeds: list[int],
    out_path: str,
    decode: DecodeSettings,
    h2_rounds: int = 3,
    bon_n: int = 3,
    limit: int | None = None,
    verbose: bool = True,
    switcher: RuntimeSwitcher | None = None,
    base_url: str | None = None,
    e1_tiers: list[str] | None = None,
) -> str:
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    done = load_done(out_path)
    if verbose and done:
        print(f"[resume] {len(done)} attempts already on disk", file=sys.stderr)

    eval_items = {e: load_eval(e, e1_tiers) for e in evals}
    if limit:
        eval_items = {e: items[:limit] for e, items in eval_items.items()}

    total = sum(
        len(eval_items[e]) * len(seeds)
        for _ in models for h in harnesses for e in evals
    )
    n_done = 0
    t_start = time.time()

    with open(out_path, "a") as fh:
        for model_key, hname, eval_id in iter_cells(models, harnesses, evals):
            mcfg = models[model_key]
            provider = build_provider(model_key, mcfg, base_url)

            # Swap the resident weights only when the model actually changes AND
            # there is still work for it -- on a resume, a fully finished model
            # should not cost a multi-GB reload to do nothing.
            is_real = mcfg.get("provider", "mlx") != "mock"
            has_work = _has_work(model_key, harnesses, evals, eval_items, seeds, done)

            if switcher is not None and is_real and has_work:
                switcher.ensure(model_key, mcfg.get("registry_key", model_key),
                                verbose=verbose)

            # Reconcile the configured id with what the server answers to, and
            # refuse to run if the parameter counts disagree. Doing this once
            # per cell (it is cached per provider instance) means a wrong-model
            # situation fails immediately instead of after 600 silent attempts.
            if is_real and has_work and hasattr(provider, "resolve"):
                provider.resolve(verbose=verbose)

            cls = HARNESSES[hname]
            harness = (
                cls(max_rounds=h2_rounds, decode=decode) if hname == "h2_loop"
                else cls(n=bon_n, decode=decode) if hname == "bon_control"
                else cls(decode=decode)
            )

            for item in eval_items[eval_id]:
                for seed in seeds:
                    n_done += 1
                    k = (model_key, hname, eval_id, item.id, seed)
                    if k in done:
                        continue
                    try:
                        att = harness.run(_wrap(provider, item), item, seed)
                        att.model = model_key      # log the config key, not the repo id
                    except Exception as exc:       # keep the sweep alive
                        att = Attempt(
                            task_id=item.id, eval_id=eval_id, model=model_key,
                            harness=hname, seed=seed, passed=False, partial=0.0,
                            rounds=0, prompt_tokens=0, completion_tokens=0,
                            latency_s=0.0, final_answer="", error=repr(exc)[:500],
                            served_model=getattr(provider, "served_model", "") or "",
                        )
                    fh.write(json.dumps(dataclasses.asdict(att)) + "\n")
                    fh.flush()

                    if verbose and n_done % 25 == 0:
                        el = time.time() - t_start
                        print(f"  {n_done}/{total}  ({el:6.1f}s)  "
                              f"{model_key} / {hname} / {eval_id}", file=sys.stderr)

    if verbose:
        print(f"[done] {out_path} in {time.time() - t_start:.1f}s", file=sys.stderr)
    return out_path
