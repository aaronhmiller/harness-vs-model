# harness-vs-model

A falsifiable experiment on whether the agent harness matters more than the model.

The claim under test is **not** "harness beats model." It is conditional:

> A harness is a **feedback amplifier, not a knowledge source.** It multiplies
> whatever signal the task provides. Harness dominates when the task is
> verifiable, decomposable, and has a non-zero base success rate. Model dominates
> when the task is gated by latent knowledge or by one-shot reasoning the model
> cannot check.

The load-bearing evidence is a **sign flip**: the same three models, the same
prompts and the same loop produce opposite rankings on different evals. A rigged
comparison cannot produce a flip — rigging either factor moves every eval the
same direction.

## Design

| Factor | Levels | Held constant |
|---|---|---|
| Model | Qwen2.5-Instruct 1.5B / 3B / 7B (4-bit), served by rMLX | one family, one quantisation — isolates scale from post-training |
| Harness | H1 naive · H2 propose-check-revise | identical prompts, decode settings, parser |
| Eval | E1 code+hidden tests · E2a knowledge · E2b traps | same runner, same scorer |
| Control | best-of-N self-selection | H2-like budget, no external signal |

**The single-variable contract.** Both harnesses see the same prompt, *including
the public tests*. H2's only advantage is that it may execute the code and read
the result. Showing the tests to H2 alone would bundle two changes — more
information plus a feedback loop — and destroy the attribution.

**The oracle substitution.** H2 is one control flow. On E1 its check is an
external oracle (run the public tests). On E2 no oracle exists, so the check
becomes the model reviewing itself with the same intuition that produced the
answer. Same loop, counterfeit signal. That substitution *is* the experiment.

## Why three model sizes

Two sizes give one delta, and that delta is a free parameter you chose when you
picked the pair. Three give a **slope**, which converts the harness effect into a
unit anyone can hold: *one harness upgrade = N model scale-steps*. Reported as
`harness_in_scale_steps`.

## Evals

| id | what | role | expected |
|---|---|---|---|
| `e1_code` | 30 functions, public tests visible, 154 hidden tests grade | verifiable | harness wins |
| `e2a_knowledge` | 40 closed-book facts | **scenario X** — info isn't in the weights | model wins, Δharness ≈ 0 |
| `e2b_traps` | 25 items with attractive wrong answers | **scenario Y** — the verifier is the model | model wins, Δharness ≤ 0 |

E2b numbers are varied from the folklore versions (bat-and-ball is $2.60/$2.00
here). A model that merely memorised "5 cents" scores zero — without this, E2b
would quietly become a second knowledge eval.

E2b also records each item's trap answer, so the report shows **trap capture
rate**. If H2's exceeds H1's, self-review did not catch the error — it ratified
it. That is the strongest available evidence for scenario Y.

## Serving: rMLX via local-coding-agent

Models are served by
[local-coding-agent](https://github.com/aaronhmiller/local-coding-agent), whose
default runtime is **rMLX** — a single Rust binary linking MLX's C ABI, no Python
in the serving path.

Nothing in `hvm/` knows which runtime is active. rMLX and `mlx_lm.server` expose
the same OpenAI-compatible `/v1/chat/completions` on the same port, so
`agent-serve runtime mlx-lm && agent-serve restart` is a fallback that changes
no code and no config here.

**Switching models is a restart, not a request field.** This is the one thing
that shapes the runner. rMLX runs with `--max-loaded-models 1` so a second set of
weights cannot arrive while the first is resident, which means the model axis
costs a multi-GB reload. So:

- **model is the outer loop** — every harness × eval for one model runs before
  the next swap, giving exactly `len(models)` reloads per sweep;
- between models the runner shells out to `agent-model use <key>` then
  `agent-serve restart`, and polls `/v1/models` until the weights are loaded;
- on a resume, a model with no outstanding work is **not** reloaded.

Those commands are config (`runtime.switch` in `config.json`), not hard-coded.
`--no-switch` disables the shell-outs entirely and `--only-model KEY ...` runs a
subset, for when you would rather drive `agent-model` by hand.

One thing this experiment does *not* need: **`tool_calls`**. The harnesses only
use text completions, so `agent-model test` — the gate that decides whether a
model can drive OpenCode or Pi — does not gate model choice here. A model that
narrates instead of emitting structured calls is still a perfectly good subject.

Do keep `--max-ctx` at the registry's 32768 rather than rMLX's unset default of
`min(capacity, 4096)`. H2 accumulates a proposal, a traceback and a revision in
one conversation, and a 4k window would truncate exactly the feedback the
experiment is measuring.

## Model ladder and memory

`config.json` ships **Qwen2.5-Instruct 1.5B / 3B / 7B, 4-bit** — roughly
0.9 / 1.8 / 4.2 GB. On a 16GB machine (~10GB usable after macOS, a browser and an
editor) each loads comfortably, one at a time.

If you have the headroom, `1.5B / 7B / 14B` gives a wider slope and a cleaner
scale-step number — 14B-4bit is ~8.5GB, fine on 32GB+ and tight on 16GB. Register
whichever ladder you use before running:

```bash
agent-model add qwen2.5-1.5b --repo mlx-community/Qwen2.5-1.5B-Instruct-4bit --size 1
agent-model add qwen2.5-3b   --repo mlx-community/Qwen2.5-3B-Instruct-4bit   --size 2
agent-model add qwen2.5-7b   --repo mlx-community/Qwen2.5-7B-Instruct-4bit   --size 5
agent-model pull qwen2.5-1.5b && agent-model pull qwen2.5-3b && agent-model pull qwen2.5-7b
```

The `registry_key` field in `config.json` is what `agent-model use` receives; the
`model` field is the Hugging Face repo sent as the request's `model`.

## Running it

Managed with [uv](https://docs.astral.sh/uv/). `uv run` resolves and installs the
environment on first use, so there is no activate step.

```bash
uv sync                      # optional; uv run does it too

# 0. verify the plumbing with no GPU at all (mock models with known planted effects)
uv run hvm -c config.mock.json --out results/mock.jsonl run
uv run hvm -c config.mock.json --out results/mock.jsonl report

# 1. bring the server up (rMLX by default)
agent-serve start && agent-serve status

# 2. check the endpoint, the switch commands, and which model is resident
uv run hvm health

# 3. CALIBRATE FIRST -- then freeze the task set
uv run hvm calibrate
uv run hvm ladder            # re-print the gate without re-running

# 4. the sweep (resumable; safe to Ctrl-C)
uv run hvm run

# 5. tables
uv run hvm report --markdown results/report.md
```

Driving the swaps yourself instead:

```bash
agent-model use qwen2.5-1.5b && agent-serve restart
uv run hvm run --no-switch --only-model qwen2.5-1.5b
agent-model use qwen2.5-3b && agent-serve restart
uv run hvm run --no-switch --only-model qwen2.5-3b
```

Results append to the same JSONL and `report` renders whatever is present, so a
sweep split across sessions costs nothing.

Tests:

```bash
uv run --group dev pytest -q
```

`uv run hvm ...` and `uv run python -m hvm ...` are equivalent — the `hvm` console
script is declared in `pyproject.toml`. Without uv, `pip install -e .` gives the
same command.

## Model ids: the config name is not the served name

`agent-serve status` reports what the server actually answers to, and it is
often **not** the id in `config.json`:

```
agent-serve status   ->  Qwen/Qwen2.5-1.5B-Instruct
config.json          ->  mlx-community/Qwen2.5-1.5B-Instruct-4bit
```

Sending the configured id then gets `404 Not Found` from
`/v1/chat/completions` even though the server is healthy and the right weights
are loaded. Since exactly one model is resident, the runner reconciles this
automatically: it reads `/v1/models` and adopts the served id.

But **only after checking the parameter counts agree.** Silently adopting a
mismatched id would file 3B results under the 1.5B key and destroy the model
axis — the one failure this experiment cannot survive. So:

```
REFUSING TO RUN: config wants 'mlx-community/Qwen2.5-3B-Instruct-4bit' (~3B)
but the server is serving 'Qwen/Qwen2.5-1.5B-Instruct' (~1.5B).
Fix with: agent-model use qwen2.5-3b && agent-serve restart
```

Every attempt records the `served_model` it actually got, and section 0 of the
report audits it. If one model key was ever answered by two different ids, the
report says so in bold and tells you those Δ model numbers are meaningless.

## Troubleshooting

| symptom | cause |
|---|---|
| `404 Not Found` on `/v1/chat/completions` while `/v1/models` works | config id ≠ served id — handled automatically now; if it still appears, the sizes disagreed and the run was refused on purpose |
| `'agent-model' is not on PATH` | `~/.local/bin` missing from PATH — local-coding-agent's `install.sh` prints the line. Use `--no-switch` meanwhile. |
| connection refused / hangs against a server that is definitely up | `HTTP_PROXY`/`ALL_PROXY` set without `127.0.0.1` in `NO_PROXY`. Requests to loopback now bypass proxies explicitly. |
| server did not answer within the timeout after a restart | cold start reading several GB. Raise `runtime.switch.ready_timeout_s`. |

## What the mock can and cannot verify

`config.mock.json` runs the whole matrix in ~20s with no GPU, against simulated
models with *known planted effects*. It verifies the plumbing: scoring, the
repair loop, resumability, the effect decomposition, the flip detector, and the
NO-FLIP guard (fed harness-wins-everywhere data, the report refuses to endorse
the thesis).

It **cannot** validate section 5, the inverted-U. The mock's `repair_skill` is a
flat probability that ignores how hard a task is, so its floor band shows a large
gain — the opposite of the prediction. That is a property of the simulator, not a
finding. Making the mock reproduce the U would mean building in the very result
the experiment is supposed to discover, so it is deliberately left naive.
**Only the real run can test that clause.**

## Calibration is not optional

`uv run hvm calibrate` reports the single-shot baseline per task per model, then
prints the **ladder gate**. `uv run hvm ladder` re-prints that verdict from data
already on disk without re-running anything (exit 0 = usable, 1 = not).

**Judge the ladder, never a single rung.** A model reading "high" on its own is
not a reason to drop it. Four things have to hold together:

| check | needs | why |
|---|---|---|
| at least two models | ≥ 2 | one model means Δ model is *undefined* — there is nothing to compare the harness against |
| bottom rung off the floor | ≥ 10% | at 0% there is nothing for a repair loop to repair |
| top rung off the ceiling | ≤ 85% | near 100% no harness can show a gain |
| model axis alive | ≥ 15 pts | without a real model gap, "harness > model" is vacuous |

The trap worth naming: rejecting every rung that scores above 60% and keeping
only the smallest model. That reads like a success — one model, comfortably
mid-range — and it silently removes the axis the whole experiment exists to
measure. **If the rungs are too easy, move the whole ladder down a size; do not
discard rungs.**

- Everything at ~95%: no headroom, no harness can help, Δharness ≈ 0 by construction.
- Everything at ~2%: nothing to repair, the loop iterates on garbage.
- **Flat model slope on E1 is a validity failure, not a result.** If 1.5B → 14B
  doesn't move E1, "the harness beat the model upgrade" is vacuous: you beat a
  step that doesn't exist. Fix the task set before scoring.

Adjust the task set *before* the scored run. Adjusting it after seeing results is
p-hacking; if you do it anyway, say so in the report.

## What would falsify the thesis

| Observed | Meaning |
|---|---|
| Harness wins on **both** E1 and E2 | The second half is dead — checking isn't the mechanism. (First verify E2 didn't leak a verifiable signal.) |
| Model wins on **both** | No harness dominance at this scale gap; the model pair was too far apart. |
| Flat model slope on E1 | Validity failure — the comparison is vacuous. |
| Harness gain identical across all three sizes | Harness is a constant offset, not a substitute for capability. |
| `bon_control` ≈ `h2_loop` on E1 | The gain was bought with tokens, not structure. Report it that way. |

The report prints **NO FLIP** in bold when the rankings don't reverse, and says
to report that rather than reaching for the E1 number alone. That is deliberate.

## Statistics

Confidence intervals bootstrap over **tasks**, not attempts. Seeds within a task
are correlated; treating 5 seeds on one task as 5 independent observations
shrinks every interval by roughly √5 and manufactures significance.

Pass rates average within task first, so all tasks carry equal weight regardless
of how many seeds completed.

## Layout

```
hvm/
  types.py           dataclasses + the Provider protocol
  providers/         mlx_provider.py (real) · mock_provider.py (planted effects)
  tasks/             e1_code.py · e2a_knowledge.py · e2b_traps.py · __init__.py (prompts + oracles)
  harnesses/         H1 · H2 · best-of-N control
  exec_sandbox.py    subprocess test runner, timeout + rlimits
  scoring.py         extraction and matching, identical across harnesses
  serving.py         model swaps via agent-model/agent-serve, readiness polling
  runner.py          cell orchestration, resumable JSONL
  analyze.py         effect decomposition, bootstrap CIs, report
  cli.py             health · calibrate · run · report
```

## A note on the sandbox

`exec_sandbox.py` is isolation, not security hardening: fresh interpreter, hard
timeout, CPU/memory rlimits, scratch cwd. That is right for code written by a
local 1.5B model on your own machine. Do not point it at untrusted code.
