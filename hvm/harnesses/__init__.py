"""The three harnesses.

  H1  naive       one generation, submit whatever comes back.
  H2  loop        propose -> CHECK -> revise, up to R rounds.
  BoN best-of-N   N independent proposals, model picks one. (control)

H1 vs H2 is the experiment: identical prompts, identical decode settings,
identical parsing. The ONLY difference is whether a check-and-revise loop runs.

H2's `check` is whatever oracle the eval provides:
  * E1 has an EXTERNAL oracle -- run the public tests, feed back the traceback.
  * E2 has NONE -- so the model verifies itself, using the same intuition that
    produced the answer. Same control flow, counterfeit signal. That substitution
    is the mechanism the whole experiment is trying to expose.

BoN is the compute-matched control. It spends H2-like token budget WITHOUT any
external signal, so if H2 > BoN the win came from the feedback, not the spend.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

from ..scoring import extract_answer, extract_code
from ..tasks import EvalItem
from ..types import Attempt, Message, Provider


@dataclass
class DecodeSettings:
    """Held constant across harnesses. Changing these changes the model, not the harness."""
    temperature: float = 0.6
    max_tokens: int = 900


class _Accum:
    """Token/latency bookkeeping so cost comparisons are measured, not assumed."""

    def __init__(self) -> None:
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.latency_s = 0.0
        self.calls = 0
        self.trace: list[dict] = []

    def call(self, provider: Provider, messages: list[Message], d: DecodeSettings,
             seed: int, label: str) -> str:
        t0 = time.time()
        c = provider.generate(
            messages, temperature=d.temperature, max_tokens=d.max_tokens, seed=seed
        )
        self.prompt_tokens += c.prompt_tokens
        self.completion_tokens += c.completion_tokens
        self.latency_s += c.latency_s or (time.time() - t0)
        self.calls += 1
        self.trace.append({"label": label, "seed": seed, "text": c.text[:4000]})
        return c.text


def _finish(item: EvalItem, provider: Provider, harness: str, seed: int,
            acc: _Accum, final_text: str, error: str | None = None) -> Attempt:
    result = item.grade(final_text)
    return Attempt(
        task_id=item.id,
        eval_id=item.eval_id,
        model=getattr(provider, "name", "unknown"),
        harness=harness,
        seed=seed,
        passed=result.ok,
        partial=result.fraction,
        rounds=acc.calls,
        prompt_tokens=acc.prompt_tokens,
        completion_tokens=acc.completion_tokens,
        latency_s=acc.latency_s,
        final_answer=(extract_code(final_text) if item.eval_id == "e1_code"
                      else extract_answer(final_text)),
        trace=acc.trace,
        error=error,
        served_model=getattr(provider, "served_model", "") or "",
    )


def _base_messages(item: EvalItem) -> list[Message]:
    return [Message("system", item.system), Message("user", item.prompt)]


# --------------------------------------------------------------------------- H1
class H1Naive:
    name = "h1_naive"

    def __init__(self, decode: DecodeSettings | None = None) -> None:
        self.decode = decode or DecodeSettings()

    def run(self, provider: Provider, item: EvalItem, seed: int) -> Attempt:
        acc = _Accum()
        text = acc.call(provider, _base_messages(item), self.decode, seed, "propose")
        return _finish(item, provider, self.name, seed, acc, text)


# --------------------------------------------------------------------------- H2
VERIFY_PROMPT = (
    "Review the proposed answer to the question below. Is it correct?\n\n"
    "Question:\n{question}\n\nProposed answer:\n{answer}\n\n"
    "Reply with a line 'VERDICT: CORRECT' or 'VERDICT: INCORRECT'. "
    "If incorrect, add one sentence saying what is wrong."
)


class H2Loop:
    """Propose, check, revise -- up to `max_rounds` proposals."""

    name = "h2_loop"

    def __init__(self, max_rounds: int = 3, decode: DecodeSettings | None = None) -> None:
        self.max_rounds = max_rounds
        self.decode = decode or DecodeSettings()

    def run(self, provider: Provider, item: EvalItem, seed: int) -> Attempt:
        acc = _Accum()
        messages = _base_messages(item)
        text = acc.call(provider, messages, self.decode, seed, "propose")

        for rnd in range(1, self.max_rounds):
            feedback = self._check(provider, item, text, acc, seed + 1000 * rnd)
            if feedback is None:
                break  # check passed
            messages = messages + [
                Message("assistant", text),
                Message("user",
                        f"{feedback}\n\nRevise your answer to fix this. "
                        f"Reply in the same format as before."),
            ]
            text = acc.call(provider, messages, self.decode, seed + rnd, f"revise{rnd}")

        return _finish(item, provider, self.name, seed, acc, text)

    def _check(self, provider: Provider, item: EvalItem, text: str,
               acc: _Accum, seed: int) -> str | None:
        """Return feedback if the check FAILED, else None."""
        if item.public_oracle is not None:
            # External oracle: real, cheap, and free of model opinion.
            res = item.public_oracle(text)
            return None if res.ok else (res.feedback or "TRACEBACK:\nchecks failed")

        # No external oracle: the model grades itself.
        verdict = acc.call(
            provider,
            [Message("system", "You are a meticulous reviewer."),
             Message("user", VERIFY_PROMPT.format(
                 question=item.prompt, answer=extract_answer(text)))],
            self.decode, seed, "self_verify",
        )
        if "INCORRECT" in verdict.upper():
            return f"A reviewer flagged your answer as incorrect:\n{verdict.strip()[:600]}"
        return None


# -------------------------------------------------------------------- control
SELECT_PROMPT = (
    "Below are {n} candidate answers to the same question. "
    "Choose the single best one.\n\nQuestion:\n{question}\n\n{candidates}\n\n"
    "Reply with a line 'CHOICE: <number>' and nothing else."
)


class BestOfNControl:
    """Compute-matched control: same budget as H2, no external signal.

    Selection is by the model itself, never by the oracle -- using the tests here
    would make this a second H2 rather than a control.
    """

    name = "bon_control"

    def __init__(self, n: int = 3, decode: DecodeSettings | None = None) -> None:
        self.n = n
        self.decode = decode or DecodeSettings()

    def run(self, provider: Provider, item: EvalItem, seed: int) -> Attempt:
        acc = _Accum()
        messages = _base_messages(item)
        cands = [
            acc.call(provider, messages, self.decode, seed + i, f"sample{i}")
            for i in range(self.n)
        ]

        shown = "\n\n".join(
            f"--- candidate {i + 1} ---\n"
            f"{(extract_code(c) if item.eval_id == 'e1_code' else extract_answer(c))[:1200]}"
            for i, c in enumerate(cands)
        )
        pick_text = acc.call(
            provider,
            [Message("system", "You select the best of several candidate answers."),
             Message("user", SELECT_PROMPT.format(
                 n=self.n, question=item.prompt, candidates=shown))],
            self.decode, seed + 777, "select",
        )

        idx = 0
        for tok in pick_text.replace("CHOICE:", " ").split():
            if tok.strip(".,)").isdigit():
                k = int(tok.strip(".,)")) - 1
                if 0 <= k < self.n:
                    idx = k
                break
        return _finish(item, provider, self.name, seed, acc, cands[idx])


HARNESSES = {
    "h1_naive": H1Naive,
    "h2_loop": H2Loop,
    "bon_control": BestOfNControl,
}

__all__ = ["H1Naive", "H2Loop", "BestOfNControl", "HARNESSES", "DecodeSettings"]
