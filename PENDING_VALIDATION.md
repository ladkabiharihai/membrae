# Pending 1B validation (GPU busy with Unreal; batch-test when free)

Code written but NOT yet validated on the live 1B. All changes are additive/dormant (new methods,
not wired into the hot path) unless noted, so they can't break the working system. Batch-test each
with the 1B (learn=False on GPU) when the GPU frees up.

| Task | What | How to validate |
|------|------|-----------------|
| T1.1 | `_semantic_entropy(q,k,n)` in brain.py | known Q -> low entropy, nonsense -> high; compare vs `_self_consistency`. `/tmp/sement.py`. |
| T1.2 | `_rephrasing_stability(q)` in brain.py | known Q -> high stability, nonsense -> low (answers drift across rephrasings). |
| T1.4 | `verify_against_source(answer,query)` in brain.py | factual answer + source -> supported=True; wrong claim -> False/None. Needs internet. |

| T2.5 | `latent_think(prompt,steps)` in brain.py | ponder K continuous steps then answer; check it runs + compare answer vs greedy on a multi-hop Q (does pondering help?). |
| T2.1 | `workspace_vector()` + `generate_with_workspace(prompt,inject=)` | inject on/off ablation; NOTE: frozen-model prototype, likely needs prefix-tuning to help -- measure the on/off delta. |

| T1.5 | `eval_all.py [label]` | run it -> eval_registry/<label>.json with ppl curve + graded multi-hop + calibration separation. One source of truth. |
| T1.6 | `faculty_ablate.py` | run it -> tools/calibration/deliberation ON-vs-OFF deltas. Confirms each faculty helps. |
| T1.7 | (in eval_all.py MULTIHOP) | graded 1/2/3-hop accuracy; run across snapshots for the token curve. |

| T3.3 | `_mine_corpus_questions()` wired into `_route_intent` | re-run router probe across checkpoints; _other now auto-mines real corpus questions (should stop the per-checkpoint misroutes). |
| T5.1 | `prepare_sft.py` (GPU-free tokenize) | run -> data/sft.bin; then the low-LR SFT recipe on H100 (in the file). Fixes 'Hi -> WW2 text'. |

| T2.2 | `goal_vector()` in brain.py | goal as a d-vector for conditioning (pooled into workspace already); check it returns for an active goal. |
| T2.6 | `background_tick()` in brain.py | call repeatedly (learn=True) -> ongoing thought writing to workspace/memory/affect; check mood/topic evolve across ticks. |



## BATCH-TEST RESULTS (GPU freed)
- T1.1 semantic entropy: **FAILS fundamentally** -- the small model confabulates CONSISTENTLY (same made-up
  answer every sample -> low entropy), so entropy detects uncertainty, not confident-consistent-confabulation.
  Kept as a documented negative result; the real calibration path is grounding (T1.4) + activations (T1.3).
- T1.4 grounding verification: **WORKS** (Paris/Jupiter supported=True; London/Mars supported=False) after the
  entity-level check. The calibration signal that works for lookupable facts.
- T1.2 rephrasing: weak (0.68 vs 0.65). T2.5 latent_think: pondering surfaces the answer ('Paris') but decode
  is weak even with rep-penalty. T2.6 monologue, T2.2 goal-vec, T3.3 router: PASS. T2.1 injection: garbage on
  the frozen model (as flagged -- needs prefix-tuning).
- T1.3 activation probe: **WORKS** (train sep 1.29; held-out nonsense all 0.0, known mean 0.19). The internal
  state signals 'unknown' on nonsense even when the OUTPUT confabulates confidently -- catches what semantic
  entropy missed. Conservative (some known flagged low = safe abstain). THIS is the calibration path.
- CALIBRATION VERDICT: semantic entropy FAILS (consistent confabulation); grounding (T1.4) WORKS for lookupable
  facts; activation probe (T1.3) WORKS for internal knows/doesn't-know. Wire T1.3+T1.4 as the honesty gate,
  retire semantic entropy for this model.

## HONESTY GATE WIRED (hot path, validated)
interact() now gates answers on the T1.3 activation probe (calibrated at init, chat-form anchors, floor=0.5
midpoint): answer only if output-consistent AND internal state doesn't flag confabulation; clear nonsense
(knows < 0.3) abstains immediately. VALIDATED: 'flarn of a quix' -> 'I'm not sure I reliably know this' (was a
confident confabulation); 'capital of France' -> Paris; arithmetic -> calc; uncertain -> abstains. The gate
fixes WHEN it speaks. What it does NOT fix: generation QUALITY (self-corpus bleed, rambling, 'Hi'->WW2) -- that
is the SFT/training issue (T5.1, prepped). The gate makes it HONEST; the SFT makes it FLUENT.

## Batch-test command sketch
```
CUDA on, Unreal closed:
  python3 /tmp/sement.py    # semantic-entropy discrimination
  python3 eval_all.py wweighted   # canonical metrics -> registry
  python3 faculty_ablate.py       # faculty-value deltas
  python3 /tmp/sement.py          # semantic-entropy discrimination
```
