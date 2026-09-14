# Harness vs Model — results

## 1. Pass rate by cell

| eval | model | harness | n | pass | 95% CI | tok/task | calls | pass/1k tok | err |
|---|---|---|---|---|---|---|---|---|---|
| e1_code | mock-small | h1_naive | 150 |  20.0% | [14.0%, 26.0%] | 372 | 1.0 | 0.537 | 0 |
| e1_code | mock-small | h2_loop | 150 |  62.7% | [57.6%, 68.0%] | 1,442 | 2.4 | 0.434 | 0 |
| e1_code | mock-small | bon_control | 150 |  16.7% | [11.1%, 22.4%] | 1,747 | 4.0 | 0.095 | 0 |
| e1_code | mock-medium | h1_naive | 150 |  31.3% | [25.3%, 37.1%] | 359 | 1.0 | 0.874 | 0 |
| e1_code | mock-medium | h2_loop | 150 |  81.3% | [76.5%, 86.2%] | 1,131 | 2.1 | 0.719 | 0 |
| e1_code | mock-medium | bon_control | 150 |  31.3% | [24.2%, 38.8%] | 1,677 | 4.0 | 0.187 | 0 |
| e1_code | mock-large | h1_naive | 150 |  46.0% | [40.0%, 52.2%] | 347 | 1.0 | 1.325 | 0 |
| e1_code | mock-large | h2_loop | 150 |  88.0% | [84.4%, 91.6%] | 921 | 1.8 | 0.956 | 0 |
| e1_code | mock-large | bon_control | 150 |  41.3% | [33.8%, 48.9%] | 1,603 | 4.0 | 0.258 | 0 |
| e2a_knowledge | mock-small | h1_naive | 200 |  30.0% | [18.5%, 40.7%] | 94 | 1.0 | 3.179 | 0 |
| e2a_knowledge | mock-small | h2_loop | 200 |  28.0% | [17.1%, 37.6%] | 222 | 2.2 | 1.259 | 0 |
| e2a_knowledge | mock-small | bon_control | 200 |  30.0% | [18.5%, 40.7%] | 394 | 4.0 | 0.760 | 0 |
| e2a_knowledge | mock-medium | h1_naive | 200 |  50.0% | [38.1%, 60.9%] | 94 | 1.0 | 5.291 | 0 |
| e2a_knowledge | mock-medium | h2_loop | 200 |  45.5% | [34.4%, 56.0%] | 234 | 2.3 | 1.948 | 0 |
| e2a_knowledge | mock-medium | bon_control | 200 |  50.0% | [38.1%, 60.9%] | 395 | 4.0 | 1.265 | 0 |
| e2a_knowledge | mock-large | h1_naive | 200 |  65.0% | [53.8%, 76.0%] | 95 | 1.0 | 6.869 | 0 |
| e2a_knowledge | mock-large | h2_loop | 200 |  53.0% | [43.2%, 62.5%] | 257 | 2.4 | 2.064 | 0 |
| e2a_knowledge | mock-large | bon_control | 200 |  65.0% | [53.8%, 76.0%] | 396 | 4.0 | 1.641 | 0 |
| e2b_traps | mock-small | h1_naive | 125 |  16.0% | [5.9%, 26.7%] | 107 | 1.0 | 1.496 | 0 |
| e2b_traps | mock-small | h2_loop | 125 |  13.6% | [4.7%, 22.7%] | 263 | 2.3 | 0.516 | 0 |
| e2b_traps | mock-small | bon_control | 125 |  16.0% | [5.9%, 26.7%] | 442 | 4.0 | 0.362 | 0 |
| e2b_traps | mock-medium | h1_naive | 125 |  12.0% | [0.0%, 20.0%] | 107 | 1.0 | 1.122 | 0 |
| e2b_traps | mock-medium | h2_loop | 125 |  11.2% | [0.0%, 18.7%] | 260 | 2.2 | 0.432 | 0 |
| e2b_traps | mock-medium | bon_control | 125 |  12.0% | [0.0%, 20.0%] | 442 | 4.0 | 0.271 | 0 |
| e2b_traps | mock-large | h1_naive | 125 |  40.0% | [25.0%, 55.6%] | 107 | 1.0 | 3.740 | 0 |
| e2b_traps | mock-large | h2_loop | 125 |  34.4% | [21.2%, 47.7%] | 295 | 2.5 | 1.164 | 0 |
| e2b_traps | mock-large | bon_control | 125 |  40.0% | [25.0%, 55.6%] | 442 | 4.0 | 0.905 | 0 |

## 2. Which factor moved the needle

| eval | Δ harness (pts) | Δ model (pts) | interaction | harness in scale-steps | winner |
|---|---|---|---|---|---|
| e1_code | +44.9 [+40.8, +49.0] | +25.7 [+20.0, +31.7] |  +0.7 | 3.50 | HARNESS |
| e2a_knowledge |  -6.2 [-7.7, -4.6] | +30.0 [+18.3, +42.6] | +10.0 | -0.41 | MODEL |
| e2b_traps |  -2.9 [-4.2, -1.6] | +22.4 [+6.2, +38.4] |  +3.2 | -0.26 | MODEL |

*Δ harness* = H2 − H1 averaged over models. *Δ model* = largest − smallest averaged over harnesses. *Interaction* > 0 means the loop helped the SMALL model more, i.e. the harness substitutes for capability.

## 3. Did the ranking flip?

| eval | winner |
|---|---|
| e1_code | HARNESS |
| e2a_knowledge | MODEL |
| e2b_traps | MODEL |

**FLIP OBSERVED** — the same models, prompts and loop produce opposite rankings on different evals, which is the signature of a conditional effect rather than a rigged comparison.

## 4. Was it the feedback or just the tokens?

| eval | model | H2 loop | best-of-N | Δ | H2 tok | BoN tok |
|---|---|---|---|---|---|---|
| e1_code | mock-small |  62.7% |  16.7% | +46.0 | 1,442 | 1,747 |
| e1_code | mock-medium |  81.3% |  31.3% | +50.0 | 1,131 | 1,677 |
| e1_code | mock-large |  88.0% |  41.3% | +46.7 | 921 | 1,603 |
| e2a_knowledge | mock-small |  28.0% |  30.0% |  -2.0 | 222 | 394 |
| e2a_knowledge | mock-medium |  45.5% |  50.0% |  -4.5 | 234 | 395 |
| e2a_knowledge | mock-large |  53.0% |  65.0% | -12.0 | 257 | 396 |
| e2b_traps | mock-small |  13.6% |  16.0% |  -2.4 | 263 | 442 |
| e2b_traps | mock-medium |  11.2% |  12.0% |  -0.8 | 260 | 442 |
| e2b_traps | mock-large |  34.4% |  40.0% |  -5.6 | 295 | 442 |

Best-of-N spends a comparable budget with NO external signal. A positive Δ means the structure did the work; a Δ near zero means we bought the gain with tokens and should say so.

## 5. Where the loop can and cannot help (E1)

| baseline band | tasks | H1 baseline | H2 with loop | gain |
|---|---|---|---|---|
| floor: model cannot start (0-15%) | 3 |  13.3% |  82.2% | +68.9 |
| workable middle (15-85%) | 27 |  34.6% |  76.8% | +42.2 |
| ceiling: already solved (85-100%) | 0 |   n/a |   n/a |  n/a |

The 'non-zero base rate' clause, measured: a repair loop needs something to repair at one end and something left to fix at the other.

## 6. Trap capture rate (E2b)

| model | harness | n | landed on the attractive wrong answer |
|---|---|---|---|
| mock-small | h1_naive | 125 |  84.0% |
| mock-small | h2_loop | 125 |  72.8% |
| mock-small | bon_control | 125 |  84.0% |
| mock-medium | h1_naive | 125 |  88.0% |
| mock-medium | h2_loop | 125 |  78.4% |
| mock-medium | bon_control | 125 |  88.0% |
| mock-large | h1_naive | 125 |  60.0% |
| mock-large | h2_loop | 125 |  44.0% |
| mock-large | bon_control | 125 |  60.0% |

If H2's rate exceeds H1's, self-verification did not catch the error — it ratified it.
