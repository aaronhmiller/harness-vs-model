"""Guardrails for the things that would silently invalidate the experiment.

Run with:  uv run --group dev pytest -q    (or: uv run tests/test_pipeline.py)
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hvm.exec_sandbox import run_tests
from hvm.providers.mlx_provider import ModelMismatch, MLXProvider, size_tokens
from hvm.scoring import extract_answer, extract_code, matches
from hvm.tasks import load_eval
from hvm.tasks.e1_code import E1_SPECS
from hvm.tasks.e2a_knowledge import E2A_SPECS
from hvm.tasks.e2b_traps import E2B_SPECS


def test_reference_solutions_pass_every_test():
    """If a reference solution fails, the task is unsolvable and scores nothing."""
    for spec in E1_SPECS:
        rep = run_tests(spec["solution"], spec["public"] + spec["hidden"])
        assert rep.all_passed, f"{spec['id']}: {rep.load_error or rep.failures[:1]}"


def test_task_ids_unique():
    for specs in (E1_SPECS, E2A_SPECS, E2B_SPECS):
        ids = [s["id"] for s in specs]
        assert len(ids) == len(set(ids))


def test_hidden_tests_are_not_public_tests():
    """Leakage check: if the graded tests were visible, we would measure overfitting."""
    for spec in E1_SPECS:
        assert not (set(spec["public"]) & set(spec["hidden"])), spec["id"]


def test_e2_has_no_public_oracle():
    """The absence of an external signal on E2 is the independent variable."""
    for eval_id in ("e2a_knowledge", "e2b_traps"):
        for item in load_eval(eval_id):
            assert item.public_oracle is None, item.id
    for item in load_eval("e1_code"):
        assert item.public_oracle is not None, item.id


def test_trap_answers_are_actually_wrong():
    """A 'trap' that matches the correct answer would silently corrupt the metric."""
    for spec in E2B_SPECS:
        for t in spec.get("trap", []):
            assert not matches(t, spec["a"], bool(spec.get("numeric"))), spec["id"]


def test_public_tests_appear_in_the_prompt():
    """Both harnesses must see the same information; only feedback may differ."""
    for spec, item in zip(E1_SPECS, load_eval("e1_code")):
        for t in spec["public"]:
            assert t.strip().splitlines()[0] in item.prompt, spec["id"]


def test_broken_code_is_scored_zero():
    rep = run_tests("def f():\n    return 1\n", ["assert f() == 2"])
    assert not rep.all_passed and rep.n_pass == 0


def test_infinite_loop_times_out():
    rep = run_tests("def f():\n    while True: pass\n", ["f()"], timeout_s=3.0)
    assert rep.timed_out and not rep.all_passed


def test_extraction():
    assert extract_code("blah\n```python\nx = 1\n```\n") == "x = 1"
    assert extract_answer("reasoning...\nANSWER: 42") == "42"
    assert extract_answer("ANSWER: a\nmore\nANSWER: b") == "b"


def test_matching():
    assert matches("ANSWER: $0.30", ["0.30"], numeric=True)
    assert matches("about 0.3 dollars", ["0.30"], numeric=True)
    assert not matches("0.60", ["0.30"], numeric=True)
    assert matches("Second", ["second"], numeric=False)
    assert not matches("secondary school", ["second"], numeric=False)
    assert matches("Marie Curie", ["marie curie"], numeric=False)


def test_size_tokens():
    """Quantisation suffixes must not be read as parameter counts."""
    assert size_tokens("mlx-community/Qwen2.5-1.5B-Instruct-4bit") == {"1.5"}
    assert size_tokens("Qwen/Qwen2.5-1.5B-Instruct") == {"1.5"}
    assert size_tokens("mlx-community/Qwen3-8B-4bit") == {"8"}   # not {"8","4"}
    assert size_tokens("mlx-community/Qwen2.5-14B-Instruct-8bit") == {"14"}
    assert size_tokens("some/model") == set()


def _provider(configured, available):
    p = MLXProvider(model=configured, name="k", registry_key="k")
    p.list_models = lambda: available
    return p


def test_resolve_adopts_served_id_when_sizes_agree():
    """The exact failure seen against rMLX: config names the MLX conversion,
    the server answers to the upstream repo id."""
    p = _provider("mlx-community/Qwen2.5-1.5B-Instruct-4bit",
                  ["Qwen/Qwen2.5-1.5B-Instruct"])
    assert p.resolve(verbose=False) == "Qwen/Qwen2.5-1.5B-Instruct"
    assert p.served_model == "Qwen/Qwen2.5-1.5B-Instruct"


def test_resolve_refuses_when_sizes_disagree():
    """The dangerous case: quietly filing 3B results under the 1.5B key."""
    p = _provider("mlx-community/Qwen2.5-1.5B-Instruct-4bit",
                  ["Qwen/Qwen2.5-3B-Instruct"])
    try:
        p.resolve(verbose=False)
        raise AssertionError("expected ModelMismatch")
    except ModelMismatch as e:
        assert "REFUSING TO RUN" in str(e)


def test_resolve_exact_match_is_untouched():
    p = _provider("repo/x-7b", ["repo/x-7b", "other/y-3b"])
    assert p.resolve(verbose=False) == "repo/x-7b"


def test_resolve_refuses_ambiguous_server():
    p = _provider("repo/absent-7b", ["a/one-7b", "b/two-7b"])
    try:
        p.resolve(verbose=False)
        raise AssertionError("expected ModelMismatch")
    except ModelMismatch as e:
        assert "cannot pick" in str(e)


def test_resolve_errors_when_server_is_down():
    p = _provider("repo/x-7b", [])
    try:
        p.resolve(verbose=False)
        raise AssertionError("expected ModelMismatch")
    except ModelMismatch as e:
        assert "agent-serve status" in str(e)


if __name__ == "__main__":
    fails = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"  PASS  {name}")
            except AssertionError as e:
                fails += 1
                print(f"  FAIL  {name}: {e}")
    print(f"\n{'all green' if not fails else str(fails) + ' failing'}")
    raise SystemExit(1 if fails else 0)
