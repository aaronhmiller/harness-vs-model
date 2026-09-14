"""A deterministic fake model, used to verify the pipeline without a GPU.

This exists so the plumbing -- scoring, repair loops, resumability, the effect
decomposition -- can be tested end to end before any real compute is spent.

It is a *simulator with knobs*, not a model. Each mock "size" has:

  * `code_skill`     -- P(emit correct code on a fresh attempt)
  * `repair_skill`   -- P(fix it given a real traceback)  [E1 only]
  * `knowledge`      -- P(know a closed-book fact)        [E2a]
  * `trap_resist`    -- P(avoid the attractive wrong answer) [E2b]

Crucially the mock reproduces the *mechanism* we are claiming: repair_skill only
applies when a real traceback is available, and on E2b the mock is
*consistently* wrong, so self-consistency voting cannot rescue it. If the
analysis code cannot recover the known planted effects from mock data, the
analysis is broken -- that is the point of this file.
"""
from __future__ import annotations

import hashlib
import random
import re

from ..types import Completion, Message

MOCK_PROFILES: dict[str, dict[str, float]] = {
    # `self_ratify` is the probability the mock approves its OWN answer when asked
    # to review it. It is high for every size on purpose: a weak self-verifier
    # mostly rubber-stamps, which is what makes E2's loop inert.
    # `talked_out` is the chance a correct answer is abandoned after criticism --
    # the mechanism by which a loop can make things actively worse.
    "mock-small": dict(
        code_skill=0.15, repair_skill=0.40, knowledge=0.20,
        trap_resist=0.15, self_ratify=0.90, talked_out=0.35,
    ),
    "mock-medium": dict(
        code_skill=0.28, repair_skill=0.50, knowledge=0.45,
        trap_resist=0.25, self_ratify=0.85, talked_out=0.25,
    ),
    "mock-large": dict(
        code_skill=0.45, repair_skill=0.62, knowledge=0.70,
        trap_resist=0.40, self_ratify=0.80, talked_out=0.15,
    ),
}


class MockProvider:
    """Emits answers whose *correctness* is drawn from the profile above.

    The emitted text is real enough to flow through the real scorers: correct
    code that actually passes the real hidden tests, or the literal correct
    answer string for the QA evals.
    """

    def __init__(self, name: str = "mock-medium") -> None:
        if name not in MOCK_PROFILES:
            raise ValueError(f"unknown mock profile {name!r}")
        self.name = name
        self.profile = MOCK_PROFILES[name]

    # -- deterministic per (model, prompt, seed, attempt-index) --------------
    def _rng(self, messages: list[Message], seed: int) -> random.Random:
        h = hashlib.sha256()
        h.update(self.name.encode())
        h.update(str(seed).encode())
        for m in messages:
            h.update(m.role.encode())
            h.update(m.content.encode())
        return random.Random(int.from_bytes(h.digest()[:8], "big"))

    def generate(
        self,
        messages: list[Message],
        *,
        temperature: float,
        max_tokens: int,
        seed: int,
    ) -> Completion:
        rng = self._rng(messages, seed)
        convo = "\n".join(m.content for m in messages)

        # The harness smuggles the ground truth into the mock via a sentinel the
        # real providers never see. Only the mock reads it.
        truth = _extract(convo, "<<MOCK_TRUTH>>", "<</MOCK_TRUTH>>")
        wrong = _extract(convo, "<<MOCK_TRAP>>", "<</MOCK_TRAP>>")
        kind = _extract(convo, "<<MOCK_KIND>>", "<</MOCK_KIND>>") or "code"
        has_traceback = "TRACEBACK:" in convo
        criticised = "reviewer flagged" in convo.lower()
        p = self.profile

        # -- meta-calls the harness makes: self-verification and BoN selection --
        if "VERDICT:" in convo:
            ratify = rng.random() < p["self_ratify"]
            return _mk(
                "VERDICT: CORRECT" if ratify
                else "VERDICT: INCORRECT\nThe reasoning looks off to me.",
                convo,
            )
        if "CHOICE:" in convo:
            return _mk(f"CHOICE: {rng.randint(1, 3)}", convo)

        # -- the actual answer -------------------------------------------------
        if kind == "code":
            prob = p["repair_skill"] if has_traceback else p["code_skill"]
            good = rng.random() < prob
            text = _fence(truth) if good else _fence(_break(truth, rng))
        else:
            # Latent ability is a property of the weights, so it is STABLE across
            # seeds for a given item: resampling cannot reach a fact the model
            # never learned, or undo an intuition it reliably trusts.
            trait = "knowledge" if kind == "knowledge" else "trap_resist"
            stable = random.Random(
                hashlib.sha256((self.name + trait + (truth or "")).encode()).digest()[:8]
            )
            good = stable.random() < p[trait]
            if good and criticised and rng.random() < p["talked_out"]:
                good = False  # criticism talks a correct answer out of itself
            # When wrong, always the SAME attractive wrong answer -- which is
            # precisely why voting and self-review cannot rescue it.
            text = f"ANSWER: {truth if good else (wrong or _plausible(truth, rng))}"

        return Completion(
            text=text,
            prompt_tokens=max(1, len(convo) // 4),
            completion_tokens=max(1, len(text) // 4),
            latency_s=0.0,
        )


def _mk(text: str, convo: str) -> Completion:
    return Completion(
        text=text,
        prompt_tokens=max(1, len(convo) // 4),
        completion_tokens=max(1, len(text) // 4),
        latency_s=0.0,
    )


def _extract(s: str, a: str, b: str) -> str:
    i, j = s.find(a), s.find(b)
    if i == -1 or j == -1:
        return ""
    return s[i + len(a) : j].strip()


def _fence(code: str) -> str:
    return f"Here is the implementation:\n\n```python\n{code}\n```\n"


DEF = re.compile(r"^def\s+(\w+)\s*\(", re.MULTILINE)


def _break(code: str, rng: random.Random) -> str:
    """Introduce a defect that RELIABLY fails the tests.

    Earlier versions mutated source text hopefully (`return` -> `return None or`,
    `range(` -> `range(1 + `). Both are no-ops on plenty of solutions -- the
    first because `None or x` is just `x` for truthy x, the second because many
    solutions contain no `range(` at all. The mock therefore looked far more
    capable than its profile claimed and every baseline was inflated.

    Wrapping the real function guarantees a wrong result while keeping the code
    runnable, which is also the realistic failure mode: it executes, it is wrong,
    and an assertion -- not a syntax error -- is what H2 gets to read.
    """
    m = DEF.search(code)
    if not m:
        return "def _hvm_broken():\n    raise RuntimeError('no function produced')\n"
    fn = m.group(1)

    if rng.random() < 0.25:  # a quarter of failures don't even load
        return code.replace("return ", "return _hvm_undefined_name + ", 1)

    return code + f'''

_hvm_real_{fn} = {fn}

def {fn}(*a, **k):
    r = _hvm_real_{fn}(*a, **k)
    if isinstance(r, bool):
        return not r
    if isinstance(r, (int, float)):
        return r + 1
    if isinstance(r, str):
        return r + "x"
    if isinstance(r, dict):
        return {{}}
    if isinstance(r, tuple):
        return ()
    if isinstance(r, list):
        return r[:-1]
    return None
'''


def _plausible(truth: str, rng: random.Random) -> str:
    return f"{truth[::-1][:12] if truth else 'unknown'}"
