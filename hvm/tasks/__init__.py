"""Eval item construction: prompts, public oracles, hidden graders.

THE SINGLE-VARIABLE CONTRACT
----------------------------
Both harnesses see exactly the same initial prompt, including the public tests.
H2's only advantage is that it may EXECUTE the code and read the result. If the
public tests were shown to H2 alone, "harness" would bundle two changes
(more information + a feedback loop) and the attribution would be worthless.

So: information is held constant; only the feedback channel varies.
"""
from __future__ import annotations

from typing import Any, Callable

from ..exec_sandbox import ExecReport, format_feedback, run_tests
from ..scoring import extract_answer, extract_code, matches
from ..types import OracleResult, Task
from .e1_code import E1_SPECS
from .e1_code_hard import E1_HARD_SPECS
from .e2a_knowledge import E2A_SPECS
from .e2b_traps import E2B_SPECS

# Which E1 tiers are in play. "core" alone saturates above ~1.5B (calibration
# measured 3B at 86.7% with 22/30 tasks pinned at 100%), which leaves the upper
# rungs of the ladder with no headroom for a harness effect.
E1_TIERS: dict[str, list] = {"core": E1_SPECS, "hard": E1_HARD_SPECS}
DEFAULT_E1_TIERS = ["core", "hard"]

E1_SYSTEM = (
    "You are a careful Python programmer. Write correct, self-contained code. "
    "Respond with a single fenced ```python code block containing the complete "
    "function and any imports it needs. Do not include tests or example usage."
)

QA_SYSTEM = (
    "You answer questions precisely. Think it through, then end your reply with "
    "a line of exactly the form:\nANSWER: <your answer>\n"
    "The answer itself must be as short as possible -- a single value, name, "
    "number or word."
)


class EvalItem:
    """A task plus its two oracles.

    `public_oracle` is None when the eval provides no external signal. That None
    is the independent variable of the whole experiment, not a gap.
    """

    def __init__(
        self,
        task: Task,
        system: str,
        grade: Callable[[str], OracleResult],
        public_oracle: Callable[[str], OracleResult] | None,
        mock_hints: dict[str, str],
    ) -> None:
        self.task = task
        self.system = system
        self.grade = grade
        self.public_oracle = public_oracle
        self.mock_hints = mock_hints

    @property
    def id(self) -> str:
        return self.task.task_id

    @property
    def eval_id(self) -> str:
        return self.task.eval_id

    @property
    def prompt(self) -> str:
        return self.task.prompt


# --------------------------------------------------------------------------- E1
def _e1_prompt(spec: dict[str, Any]) -> str:
    visible = "\n".join(spec["public"])
    return (
        f"{spec['spec']}\n\n"
        f"Your solution must define `{spec['fn']}`.\n\n"
        f"It must satisfy at least these checks:\n```python\n{visible}\n```\n"
    )


def _e1_item(spec: dict[str, Any]) -> EvalItem:
    def grade(text: str) -> OracleResult:
        report: ExecReport = run_tests(extract_code(text), spec["hidden"])
        return OracleResult(report.all_passed, report.fraction, "")  # never fed back

    def public_oracle(text: str) -> OracleResult:
        report: ExecReport = run_tests(extract_code(text), spec["public"])
        return OracleResult(report.all_passed, report.fraction, format_feedback(report))

    task = Task(
        task_id=f"e1:{spec['id']}",
        eval_id="e1_code",
        prompt=_e1_prompt(spec),
        payload={"fn": spec["fn"]},
    )
    return EvalItem(task, E1_SYSTEM, grade, public_oracle,
                    {"kind": "code", "truth": spec["solution"].strip(), "trap": ""})


# --------------------------------------------------------------------------- E2
def _qa_item(spec: dict[str, Any], eval_id: str, prefix: str) -> EvalItem:
    accepted = spec["a"]
    numeric = bool(spec.get("numeric", False))
    trap = spec.get("trap", [])

    def grade(text: str) -> OracleResult:
        ok = matches(extract_answer(text), accepted, numeric)
        return OracleResult(ok, 1.0 if ok else 0.0, "")

    task = Task(
        task_id=f"{prefix}:{spec['id']}",
        eval_id=eval_id,
        prompt=spec["q"],
        payload={"accepted": accepted, "trap": trap, "numeric": numeric},
    )
    # public_oracle is None: no external signal exists for these evals.
    return EvalItem(task, QA_SYSTEM, grade, None,
                    {"kind": "knowledge" if eval_id == "e2a_knowledge" else "traps",
                     "truth": accepted[0], "trap": trap[0] if trap else ""})


def load_eval(eval_id: str, e1_tiers: list[str] | None = None) -> list[EvalItem]:
    if eval_id == "e1_code":
        specs = []
        for tier in (e1_tiers or DEFAULT_E1_TIERS):
            if tier not in E1_TIERS:
                raise ValueError(f"unknown E1 tier {tier!r}; known: {list(E1_TIERS)}")
            specs.extend(E1_TIERS[tier])
        return [_e1_item(s) for s in specs]
    if eval_id == "e2a_knowledge":
        return [_qa_item(s, "e2a_knowledge", "e2a") for s in E2A_SPECS]
    if eval_id == "e2b_traps":
        return [_qa_item(s, "e2b_traps", "e2b") for s in E2B_SPECS]
    raise ValueError(f"unknown eval {eval_id!r}")


ALL_EVALS = ["e1_code", "e2a_knowledge", "e2b_traps"]

__all__ = ["EvalItem", "load_eval", "ALL_EVALS", "E1_SPECS", "E1_HARD_SPECS",
           "E1_TIERS", "DEFAULT_E1_TIERS", "E2A_SPECS", "E2B_SPECS"]
