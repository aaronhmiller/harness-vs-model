"""Core datatypes shared across providers, tasks, harnesses and scoring."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Literal, Protocol

Role = Literal["system", "user", "assistant"]


@dataclass
class Message:
    role: Role
    content: str

    def as_dict(self) -> dict[str, str]:
        return {"role": self.role, "content": self.content}


@dataclass
class Completion:
    """One model call's result, with token accounting."""
    text: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_s: float = 0.0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


class Provider(Protocol):
    """Anything that can turn messages into text.

    Implementations must be deterministic given the same `seed`, to the extent
    the backend allows. MLX honours seeds; sampling is still the dominant source
    of within-cell variance, which is why we run K seeds per cell.
    """

    name: str

    def generate(
        self,
        messages: list[Message],
        *,
        temperature: float,
        max_tokens: int,
        seed: int,
    ) -> Completion: ...


@dataclass
class Task:
    """A single eval item.

    `check_public` is the harness-visible oracle. It is None for evals that have
    no external signal -- that absence is the independent variable in E2, not an
    oversight.
    """
    task_id: str
    eval_id: str            # "e1_code" | "e2a_knowledge" | "e2b_traps"
    prompt: str
    payload: dict[str, Any] = field(default_factory=dict)
    # Set by the eval module; see hvm.tasks for the concrete implementations.
    difficulty_hint: str = "unknown"


@dataclass
class Attempt:
    """One harness's full trajectory on one task."""
    task_id: str
    eval_id: str
    model: str
    harness: str
    seed: int
    passed: bool
    partial: float               # fraction of hidden checks passed (0..1)
    rounds: int                  # model calls actually made
    prompt_tokens: int
    completion_tokens: int
    latency_s: float
    final_answer: str
    trace: list[dict[str, Any]] = field(default_factory=list)
    error: str | None = None
    # The id the server actually answered to. Audited in the report: if one
    # model key was ever served by two different ids, the model axis is
    # untrustworthy and the report must say so rather than average over it.
    served_model: str = ""

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


@dataclass
class OracleResult:
    """Outcome of running a task's verifier against a candidate answer."""
    ok: bool
    fraction: float
    feedback: str  # what the harness is allowed to see on failure


# A grader closes over hidden tests; a public checker closes over visible ones.
Grader = Callable[[str], OracleResult]
