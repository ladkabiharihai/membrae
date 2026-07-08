# Pending 1B validation (GPU busy with Unreal; batch-test when free)

Code written but NOT yet validated on the live 1B. All changes are additive/dormant (new methods,
not wired into the hot path) unless noted, so they can't break the working system. Batch-test each
with the 1B (learn=False on GPU) when the GPU frees up.

| Task | What | How to validate |
|------|------|-----------------|
| T1.1 | `_semantic_entropy(q,k,n)` in brain.py | known Q -> low entropy, nonsense -> high; compare vs `_self_consistency` (should separate where it fails). `/tmp/sement.py`. |

## Batch-test command sketch
```
CUDA on, Unreal closed:
  python3 /tmp/sement.py    # semantic-entropy discrimination
  (add more probes here as tasks land)
```
