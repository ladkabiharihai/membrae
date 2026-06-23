# Pragnosia — Handover for Claude on the H100

Read this whole file first. It is the single source of truth for *where we are, the blunder we made,
the correct path, what to keep, and what to do next*. Repo: branch `growth-refine-1.4b`, pull before
starting (`git pull --ff-only`). Code at `/opt/code/membrae` (server) / `~/Downloads/membrae` (laptop).

---

## 0. What this is
**Pragnosia = a "spinning brain"**: a language model whose *core* is meant to be a **spin carrier** — a
diagonal-complex LRU/S5-style recurrence `h_t = λ⊙h_{t-1} + b_t` (a parallel associative scan, matmul-free,
O(T·d)) — with **attention only as a helper for generation fluency**. On top of the LM, `brain.py` is the
**living controller**: honest (answers what it knows, says IDK otherwise), curious (asks its own questions),
learns continually (teach + replay, no forgetting), grows itself when saturated, has an in-weights
subconscious memory, and an autonomous cognition loop. The vision: raise it like a child → adult brain.

User's standing directives (escalated repeatedly): **(1) NO HARDCODING** — every scale/threshold is
self-derived from data or the model's own behaviour and re-derived as it grows. **(2) Spin is the core**,
attention is the helper. **(3) Honest evaluation** — measure, don't assert.

---

## 1. ⚠️ THE BLUNDER (and the measurement that caught it)
We scaled to a **1.4B model — but built it ATTENTION-DOMINANT**, the *inverse* of the intended design.
The spin carrier was left as a **single fixed ~5.25M layer after the transformer stack**, so its share
**collapsed as we scaled**: 5.76% (21M) → 1.90% (276M) → **0.37% (1.4B)**. The brain-swap causal test at
1.4B (fp32) still passes directionally (`own < swapped < random`) but the causal signal is only
**0.00086 NLL = 0.028% of the loss** — down ~20× from 0.016 at 276M. **The spin carrier became
vestigial at scale.** An external reviewer correctly flagged this. The "answer lives in the phase /
fundamentally new reasoning paradigm" claim is **NOT supported at scale** and has been removed from the docs.

`pragnosia_best.pt` (local, bf16) and the pristine fp32 on the H100 are this **wrong-build 1.4B** — keep
them as a reference, but they are not the path forward.

---

## 2. ✅ THE CORRECT PATH — spin-dominant (validated)
The intended design: **spin IS the token-mixer core; attention only every 4th layer as a helper.**
Implemented as `carrier="spin_dominant"` (a `SpinBlock` = the spin carrier as the mixer replacing attention,
strong gate; a normal attention `Block` every 4th layer). **Param-matched ablation (same tokens/budget,
d=512, ~37–46M, on the laptop):**

| architecture | params | val ppl |
|---|---|---|
| attention-only (transformer) | 35.7M | 279 |
| transformer-only, param-matched | 45.2M | 270 |
| per-block hybrid | 46.2M | 228 |
| **spin-dominant (intended)** | **37.3M** | **219** ✅ |

**Spin-dominant wins by ~21% at matched params** — the carrier earns its weights when it is the core.
Carrier modes (`none`/`single`/`per_block`/`spin_dominant`) are wired through `pragnosia.json` →
`train_pragnosia.py` → `grow.py` (growth is function-preserving for spin blocks too).

**CAVEAT (be honest):** small-scale / short-budget / single-seed. **This MUST be replicated at scale.**

**Confirmed in production:** the laptop spin-dominant run **grew itself 24M (4L) → 27.3M (5L)** (probe-confirmed
saturation, function-preserving), then capped at the ~25M data budget — self-governing growth works on spin blocks.

**Speed:** spin-dominant is *slower* at ctx=256 (the scan does more than attention on short sequences; it
can't be in-place-optimized — that breaks autograd). Its advantage is **long context** (O(T) vs attention
O(T²)) and **inference** (recurrent → O(1)/token, no KV cache). `torch.compile` **fuses the scan → ~2×**
(measured **49K → ~100K tok/s at bs=24 on the laptop RTX 4060**). The old compile asserts on the 4060 were
the **Prefetcher CUDA race (now fixed)**, not compile — so compile is ON by default; use `NOCOMPILE=1` only
if it misbehaves.

---

## 3. 🎯 TOP PRIORITY ON THE H100: validate spin-dominant at scale
Run, on the new CoT-enriched corpus (`window2_train`), at ~200M–1B, full budget:
1. `carrier:"spin_dominant"` (the candidate), and
2. `carrier:"none"` (transformer-only baseline) — **same size, same tokens.**
If spin-dominant holds its ~15–21% edge at scale, that is a real, publishable hybrid result. Set the
config's `"carrier"` field; the trainer + growth honour it. Use `run_fast.py` (compile on; works on H100).
Watch with `bash ops.sh status`. **A high derived LR can destabilise a fresh run** — `find_lr` now applies
Smith's ÷10 margin (`--lr 0` derives it); if a run's ppl *rises* after warmup, the LR was still too hot. On
**`--resume` the peak LR is RESTORED from the sidecar** (find_lr is unreliable on trained weights — it once
re-derived ~1e-6 and trained frozen), so resume continues at the right LR, not re-derived.

---

## 4. The brain.py living loop (the whole brain now — the toy is GONE)
The 302K `unified_brain` toy is removed (it was never in the live loop; its "14/14" ran on the toy, not the
real model). `brain.py` = the spin-dominant LM + self-derived controller:
- **Honesty** by self-consistency (`consistency_min` via two-distribution EER); answers known, says IDK.
- **Curiosity** `wonder()` (own surprise) → `search()` (Wikipedia) → `teach()` (+ generative replay).
- **Self-governing growth** — probe-confirmed saturation, NO hardcoded caps (data-budget/VRAM limited).
- **In-weights SUBCONSCIOUS memory** (`FastWeightMemory` in s6_hybrid): a hippocampal episodic store — the
  model's own (context→next-token) traces, surprise-gated write, **attention recall** (selective, no
  cross-talk), **uncertainty-gated** so it never contaminates confident outputs, decay = forgetting,
  consolidate-to-slow-weights = sleep. NOT RAG. Functional; verbatim recall is model-quality-limited.
- **Cognition**: `_introspect` (metacognition — its own confidence), `_deliberate` (step-by-step
  scratchpad), `think_aloud` (autonomous internal monologue) — wired into `interact()`.
All mechanisms fire; **quality follows the model** (both models are undertrained — the CoT corpus is the fix
for the weak axis: strong single-step, weak multi-hop = undertraining-for-size).

**How to test every faculty + feature on the new paradigm:**
- `python3 brain.py test` → faculties + memory + cognition on the spin-dominant LM (language, honesty,
  learn/seek, SUBCONSCIOUS write→consolidate→forget, metacog/deliberate/monologue). No toy.
- `python3 faculty_test.py <ckpt>` → LM-only battery, CPU, READ-ONLY (snapshot a live checkpoint; never
  touches training). Builds with `carrier=cfg['carrier']` so it loads spin-dominant checkpoints.
- `python3 brain.py chat` (inference) / `python3 brain.py learn` (continuous learning).

---

## 5. Repo map — what to KEEP (everything else was removed)
| File | Job |
|---|---|
| `brain.py` | **The whole mind**: spin-dominant LM + honesty/curiosity/learn/grow + subconscious memory + cognition |
| `s6_hybrid.py` | The LM: `SpinAttentionLM`, `SpinCarrier`, `SpinBlock`, `FastWeightMemory`, carrier modes, loaders, the brain-swap `swap()` gate |
| `grow.py` | Function-preserving growth (carrier-aware: depth/width over attention or spin blocks) |
| `train_pragnosia.py` | Self-governing trainer (derived LR, probe-confirmed-saturation growth, lowmem 1.4B-on-8GB, true-resume sidecar) |
| `prepare_data.py` → `prepare_posttrain.py` → `prepare_scale.py` | The data-build chain (`prepare_scale` adds the CoT corpus via `cot_stream`) |
| `cot_build.sh` | Builds + appends the ~18B CoT corpus into window1/window2 |
| `run_fast.py` | Compile wrapper (use on the H100 for the fused scan) |
| `faculty_test.py` | LM faculty battery (CPU, read-only) |
| `ops.sh` | H100 ops: `status` / `resume` / `test` |
| `pragnosia.json` | Active config (incl. `"carrier"`). `pragnosia.json.1p4B` = the 1.4B arch (for loading `pragnosia_best.pt`) |
| `HANDOVER.md` / `RUNBOOK.md` / `TRAINING_NOTES.md` | This / how-to-run / the training story |
| `paper/`, `site/` | Paper + explainer site (now tell the honest spin-dominant story; regen PDF: `cd paper && weasyprint paper.html ../Pragnosia_paper.pdf`) |

**REMOVED (rubbish, do not recreate):** `unified_brain.py` + `unified_brain.pt` (the dead 302K toy);
`README.md` (was the most outdated); `CLAUDE_CODE_HANDOFF.md`, `PRAGNOSIA_COMPLETE.md`, `S1/S2/S3_RESULTS.md`
(superseded stage logs); duplicate checkpoint backups + old attention-scratch configs.

**Workspace hygiene (do regularly):** checkpoints/bins are gitignored — don't commit `*.pt`/`*.bin`. Keep
only the live run's checkpoints + the bins + `pragnosia_best.pt` (the 1.4B reference). Remove stale backups,
old logs, and superseded configs as you go. `pragnosia.json` is rewritten by the trainer on growth — expect
it dirty during a run; that's runtime state, not a leak.

---

## 6. ⚠️ CRITICAL FOOTGUN: the data bins
`data/window2_train.bin` + `data/big_valid.bin` + `data/bpe.json` are **load-bearing runtime files**, not
just training artifacts. `brain.py`'s `teach()` replays `window2_train` so learning a fact doesn't erase
skills — **the wrong/dirty corpus corrupts the model on every teach** (cost a long debug once). The bins
must be the **same corpus + same tokenizer** the checkpoint trained on. They're gitignored; carry them with
the checkpoint or rebuild with `prepare_scale.py`. Sanity: a slice decodes to clean prose/math, and
`H.val_ppl(base_lm, big_valid)` lands near the training value (not ~2 = contaminated).

---

## 7. State as of this handover
A **spin-dominant from-scratch run** trained on the laptop (`pragnosia_spin.pt`): it **self-grew 24M(4L)→27.3M(5L)**,
reached **~it 53,500, val ppl ~40**, at **~85K tok/s** (compile, bs=24). Probe of the 27.3M: **fluent** generation,
**weak** faculties (it's a 27M model — can't store world facts), **faint emergence** (solved a theory-of-mind
false-belief + an in-context binding case). The laptop now holds a **fresh ~1 GB strided CoT-inclusive subsample**
(500M tokens) pulled from the H100's 330 GB corpus. The 1.4B (`pragnosia_best.pt`, attention-dominant, the wrong
build) is the reference. The decisive experiment — spin-dominant vs transformer-only at 200M–1B on the full CoT
corpus — **is the H100's job (§3).** Deferred no-hardcode items (need teach-regression tests): teach
`plasticity`/`target` clamps, trainer `warm`/`target_eff`. Key memory for future Claude sessions lives in
the user's `memory/` dir (`spinning-brain-project.md`).
