"""Run model-written code against assertions in an isolated subprocess.

This is isolation, not security hardening: a fresh interpreter, a hard timeout,
CPU/address-space rlimits, no inherited stdin, and a scratch cwd. That is
appropriate for code written by a local 1.5B model on your own machine. Do not
point this at untrusted third-party code.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import textwrap
from dataclasses import dataclass

DEFAULT_TIMEOUT_S = 10.0
MEM_LIMIT_BYTES = 2 * 1024 * 1024 * 1024  # 2 GiB

_TEMPLATE = r'''
import json, sys, traceback

CANDIDATE_SRC = {candidate!r}
TESTS = json.loads({tests_json!r})

ns = {{}}
results = []
try:
    exec(compile(CANDIDATE_SRC, "<candidate>", "exec"), ns)
except BaseException:
    print(json.dumps({{
        "load_error": traceback.format_exc(limit=3),
        "results": [],
    }}))
    sys.exit(0)

for t in TESTS:
    try:
        exec(compile(t, "<test>", "exec"), dict(ns))
        results.append({{"test": t, "ok": True, "err": None}})
    except BaseException:
        tb = traceback.format_exc(limit=2)
        results.append({{"test": t, "ok": False, "err": tb[-600:]}})

print(json.dumps({{"load_error": None, "results": results}}))
'''


@dataclass
class ExecReport:
    n_pass: int
    n_total: int
    load_error: str | None
    failures: list[tuple[str, str]]          # (test source, traceback tail)
    timed_out: bool = False

    @property
    def fraction(self) -> float:
        return self.n_pass / self.n_total if self.n_total else 0.0

    @property
    def all_passed(self) -> bool:
        return self.n_total > 0 and self.n_pass == self.n_total and not self.load_error


def _limits() -> None:  # pragma: no cover - child process only
    try:
        import resource

        resource.setrlimit(resource.RLIMIT_AS, (MEM_LIMIT_BYTES, MEM_LIMIT_BYTES))
        resource.setrlimit(resource.RLIMIT_CPU, (15, 15))
        resource.setrlimit(resource.RLIMIT_NPROC, (64, 64))
    except Exception:
        pass


def run_tests(
    candidate_src: str,
    tests: list[str],
    timeout_s: float = DEFAULT_TIMEOUT_S,
) -> ExecReport:
    if not candidate_src.strip():
        return ExecReport(0, len(tests), "empty candidate", [])

    script = _TEMPLATE.format(
        candidate=textwrap.dedent(candidate_src),
        tests_json=json.dumps(tests),
    )
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "_run.py")
        with open(path, "w") as fh:
            fh.write(script)
        try:
            proc = subprocess.run(
                [sys.executable, "-I", "-S", path],
                capture_output=True,
                text=True,
                timeout=timeout_s,
                cwd=td,
                stdin=subprocess.DEVNULL,
                preexec_fn=_limits if os.name == "posix" else None,
                env={"PATH": "/usr/bin:/bin", "PYTHONHASHSEED": "0"},
            )
        except subprocess.TimeoutExpired:
            return ExecReport(0, len(tests), "timeout", [], timed_out=True)

    out = (proc.stdout or "").strip().splitlines()
    if not out:
        return ExecReport(0, len(tests), (proc.stderr or "no output")[-600:], [])
    try:
        data = json.loads(out[-1])
    except json.JSONDecodeError:
        return ExecReport(0, len(tests), (proc.stdout or "")[-600:], [])

    if data["load_error"]:
        return ExecReport(0, len(tests), data["load_error"], [])

    results = data["results"]
    failures = [(r["test"], r["err"] or "") for r in results if not r["ok"]]
    n_pass = sum(1 for r in results if r["ok"])
    return ExecReport(n_pass, len(results), None, failures)


def format_feedback(report: ExecReport, max_failures: int = 3) -> str:
    """The ONLY channel through which H2 learns anything. Public tests only."""
    if report.timed_out:
        return "TRACEBACK:\nExecution timed out (likely an infinite loop)."
    if report.load_error:
        return f"TRACEBACK:\nYour code failed to load:\n{report.load_error}"
    if report.all_passed:
        return ""
    lines = [f"TRACEBACK:\n{report.n_pass}/{report.n_total} visible tests passed. Failures:"]
    for test, err in report.failures[:max_failures]:
        lines.append(f"\n--- failing check ---\n{test}\n{err}")
    return "\n".join(lines)
