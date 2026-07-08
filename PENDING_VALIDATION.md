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

## Batch-test command sketch
```
CUDA on, Unreal closed:
  python3 /tmp/sement.py    # semantic-entropy discrimination
  python3 eval_all.py wweighted   # canonical metrics -> registry
  python3 faculty_ablate.py       # faculty-value deltas
  python3 /tmp/sement.py          # semantic-entropy discrimination
```
