# Pragnosia  Handover for Claude on the H100

Read this whole file first. It is the single source of truth for *where we are, the blunder we made,
the correct path, what to keep, and what to do next*. Repo: branch `growth-refine-1.4b`, pull before
starting (`git pull --ff-only`). Code at `/opt/code/membrae` (server) / `~/Downloads/membrae` (laptop).

---

## 0. What this is
**Pragnosia = a "spinning brain"**: a language model whose *core* is meant to be a **spin carrier**  a
diagonal-complex LRU/S5-style recurrence `h_t = λ⊙h_{t-1} + b_t` (a parallel associative scan, matmul-free,
O(T·d))  with **attention only as a helper for generation fluency**. On top of the LM, `brain.py` is the
**living controller**: honest (answers what it knows, says IDK otherwise), curious (asks its own questions),
learns continually (teach + replay, no forgetting), grows itself when saturated, has an in-weights
subconscious memory, and an autonomous cognition loop. The vision: raise it like a child → adult brain.

User's standing directives (escalated repeatedly): **(1) NO HARDCODING**  every scale/threshold is
self-derived from data or the model's own behaviour and re-derived as it grows. **(2) Spin is the core**,
attention is the helper. **(3) Honest evaluation**  measure, don't assert.

---

## 1. ⚠️ THE BLUNDER (and the measurement that caught it)
We scaled to a **1.4B model  but built it ATTENTION-DOMINANT**, the *inverse* of the intended design.
The spin carrier was left as a **single fixed ~5.25M layer after the transformer stack**, so its share
**collapsed as we scaled**: 5.76% (21M) → 1.90% (276M) → **0.37% (1.4B)**. The brain-swap causal test at
1.4B (fp32) still passes directionally (`own < swapped < random`) but the causal signal is only
**0.00086 NLL = 0.028% of the loss**  down ~20× from 0.016 at 276M. **The spin carrier became
vestigial at scale.** An external reviewer correctly flagged this. The "answer lives in the phase /
fundamentally new reasoning paradigm" claim is **NOT supported at scale** and has been removed from the docs.

`pragnosia_best.pt` (local, bf16) and the pristine fp32 on the H100 are this **wrong-build 1.4B**  keep
them as a reference, but they are not the path forward.

---

## 2. ✅ THE CORRECT PATH  spin-dominant (validated)
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

**Spin-dominant wins by ~21% at matched params**  the carrier earns its weights when it is the core.
Carrier modes (`none`/`single`/`per_block`/`spin_dominant`) are wired through `pragnosia.json` →
`train_pragnosia.py` → `grow.py` (growth is function-preserving for spin blocks too).

**CAVEAT (be honest):** small-scale / short-budget / single-seed. **This MUST be replicated at scale.**

**Confirmed in production:** the laptop spin-dominant run **grew itself 24M (4L) → 27.3M (5L)** (probe-confirmed
saturation, function-preserving), then capped at the ~25M data budget  self-governing growth works on spin blocks.

**🔑 AT-SCALE VALIDATION (284M  the decisive result):** the H100 trained a **284M spin-dominant** model
(d=768, 20L, mlp9) to **val ppl 24.5** on the 176.6B CoT corpus (~24 tok/param). **Carrier-causality ablation:
zeroing the spin carrier sends ppl 25 → 7716 (306×); zeroing attention sends it 25 → 85 (3.4×)**  the spin
carrier is the **LOAD-BEARING CORE**, the decisive opposite of the 1.4B's vestigial 0.028%. When BUILT
spin-dominant, the carrier becomes the dominant computation at scale  exactly the intended design. Capability
jumped vs the 27M: **arithmetic 3/3, knowledge 2/3, theory-of-mind 2/2** ("capital of France→Paris", "2+2→4",
fluent stories); multi-step/counterfactual/causal still weak (next axis). **Honesty gate (open problem):** interact() now asks
in the trained `<user>..<assistant>` format (bare questions were OOD → it over-abstained), but
**consistency-voting can't separate knowledge from confabulation at 284M**  known "capital of France" 0.53
vs unknown "password" 0.53 / "2031 election" 0.67 (overlapping distributions, no threshold works); the model
confabulates *consistently*. Reliable honesty needs scale (consistency sharpens at 1B+) or a redesigned
confidence signal. The 284M is synced to the laptop as `pragnosia_spin.pt` (config d=768).

**Latest 284M evals (laptop):** **(1) Deliberation works  the CoT corpus paid off.** On multi-step word problems
the model is **0/5 greedy** but **~4/5 when prompted to deliberate**  it emits the `<think>` tag straight from the
CoT training data and works through the steps (not always *correct*, but the reasoning *form* is there). So routing
hard questions through `_deliberate` is the right design. **(2) Continual learning runs but has a stability–plasticity
tension.** `teach()` is now OOM-safe on the 284M (grad-checkpoint + micro-batch); it learns a new fact (recall=YES)
and keeps general skills (stories stay coherent), but an *aggressive* teach interferes with close neighbours (teaching
"Zubland→Maretto" nudged "France→Maretto"), while a *gentle* teach is safe but under-learns  and episodic recall
doesn't yet compensate. Reliable single-fact editing is an open follow-up (targeted replay / stronger episodic recall / EWC).
**(3) Deliberation is now wired in**  `interact()` routes computational/multi-step questions through the `<user>..<assistant>`
CoT path by default. **(4) Honesty is a confirmed scale-limit**  consistency, entropy, greedy-confidence, and novelty
were all tested and NONE separates knowledge from confident confabulation at 284M (known and unknowable overlap on every
signal); no veto shipped. (The classic brain-swap test needs cross-segment carried state, which only `single`/`per_block`
thread; in `spin_dominant` the carrier is intra-block, so **ablation is the equivalent causal measure**.)

**🎯 H100 baseline is PREPPED (just run it).** The decisive test  does spin-dominant *win* vs a transformer?  is ready:
`pragnosia_baseline.json` is `carrier="none"`, d768/20L/mlp9, **param-matched to the 284M within 3.1%**. The trainer now
takes a `CONFIG` env. On the H100, train it to the **same token budget** the 284M saw (step 418K), then compare val ppl:
`CONFIG=pragnosia_baseline.json BEST_CKPT=pragnosia_baseline_best.pt python3 train_pragnosia.py --steps 418000 --lr 0`
(use `run_fast.py` for compile). Spin-dominant won by ~21% at 37M; this confirms it at 284M.

**Speed:** spin-dominant is *slower* at ctx=256 (the scan does more than attention on short sequences; it
can't be in-place-optimized  that breaks autograd). Its advantage is **long context** (O(T) vs attention
O(T²)) and **inference** (recurrent → O(1)/token, no KV cache). `torch.compile` **fuses the scan → ~2×**
(measured **49K → ~100K tok/s at bs=24 on the laptop RTX 4060**). The old compile asserts on the 4060 were
the **Prefetcher CUDA race (now fixed)**, not compile  so compile is ON by default; use `NOCOMPILE=1` only
if it misbehaves.

---

## 3. 🎯 TOP PRIORITY ON THE H100: validate spin-dominant at scale
Run, on the new CoT-enriched corpus (`window2_train`), at ~200M–1B, full budget:
1. `carrier:"spin_dominant"` (the candidate), and
2. `carrier:"none"` (transformer-only baseline)  **same size, same tokens.**
If spin-dominant holds its ~15–21% edge at scale, that is a real, publishable hybrid result. Set the
config's `"carrier"` field; the trainer + growth honour it. Use `run_fast.py` (compile on; works on H100).
Watch with `bash ops.sh status`. **A high derived LR can destabilise a fresh run**  `find_lr` now applies
Smith's ÷10 margin (`--lr 0` derives it); if a run's ppl *rises* after warmup, the LR was still too hot. On
**`--resume` the peak LR is RESTORED from the sidecar** (find_lr is unreliable on trained weights  it once
re-derived ~1e-6 and trained frozen), so resume continues at the right LR, not re-derived.

---

## 4. The brain.py living loop (the whole brain now  the toy is GONE)
The 302K `unified_brain` toy is removed (it was never in the live loop; its "14/14" ran on the toy, not the
real model). `brain.py` = the spin-dominant LM + self-derived controller:
- **Honesty** by self-consistency (`consistency_min` via two-distribution EER); answers known, says IDK.
- **Curiosity** `wonder()` (own surprise) → `search()` (Wikipedia) → `teach()` (+ generative replay).
- **Self-governing growth**  probe-confirmed saturation, NO hardcoded caps (data-budget/VRAM limited).
- **In-weights SUBCONSCIOUS memory** (`FastWeightMemory` in s6_hybrid): a hippocampal episodic store  the
  model's own (context→next-token) traces, surprise-gated write, **attention recall** (selective, no
  cross-talk), **uncertainty-gated** so it never contaminates confident outputs, decay = forgetting,
  consolidate-to-slow-weights = sleep. NOT RAG. Functional; verbatim recall is model-quality-limited.
- **Cognition**: `_introspect` (metacognition  its own confidence), `_deliberate` (step-by-step
  scratchpad), `think_aloud` (autonomous internal monologue)  wired into `interact()`.
All mechanisms fire; **quality follows the model** (both models are undertrained  the CoT corpus is the fix
for the weak axis: strong single-step, weak multi-hop = undertraining-for-size).

**How to test every faculty + feature on the new paradigm:**
- `python3 brain.py test` → faculties + memory + cognition on the spin-dominant LM (language, honesty,
  learn/seek, SUBCONSCIOUS write→consolidate→forget, metacog/deliberate/monologue). No toy.
- `python3 faculty_test.py <ckpt>` → LM-only battery, CPU, READ-ONLY (snapshot a live checkpoint; never
  touches training). Builds with `carrier=cfg['carrier']` so it loads spin-dominant checkpoints.
- `python3 brain.py chat` (inference) / `python3 brain.py learn` (continuous learning).

---

## 5. Repo map  what to KEEP (everything else was removed)
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
| `eval_external.py` | zero-shot public benchmarks (WikiText/LAMBADA/HellaSwag/ARC/PIQA), tokenizer-independent |
| `ops.sh` | H100 ops: `status` / `resume` / `test` |
| `pragnosia.json` | Active config (incl. `"carrier"`). `pragnosia.json.1p4B` = the 1.4B arch (for loading `pragnosia_best.pt`) |
| `HANDOVER.md` / `RUNBOOK.md` / `TRAINING_NOTES.md` | This / how-to-run / the training story |
| `paper/`, `site/` | Paper + explainer site (now tell the honest spin-dominant story; regen PDF: `cd paper && weasyprint paper.html ../Pragnosia_paper.pdf`) |

**REMOVED (rubbish, do not recreate):** `unified_brain.py` + `unified_brain.pt` (the dead 302K toy);
`README.md` (was the most outdated); `CLAUDE_CODE_HANDOFF.md`, `PRAGNOSIA_COMPLETE.md`, `S1/S2/S3_RESULTS.md`
(superseded stage logs); duplicate checkpoint backups + old attention-scratch configs.

**Workspace hygiene (do regularly):** checkpoints/bins are gitignored  don't commit `*.pt`/`*.bin`. Keep
only the live run's checkpoints + the bins + `pragnosia_best.pt` (the 1.4B reference). Remove stale backups,
old logs, and superseded configs as you go. `pragnosia.json` is rewritten by the trainer on growth  expect
it dirty during a run; that's runtime state, not a leak.

---

## 6. ⚠️ CRITICAL FOOTGUN: the data bins
`data/window2_train.bin` + `data/big_valid.bin` + `data/bpe.json` are **load-bearing runtime files**, not
just training artifacts. `brain.py`'s `teach()` replays `window2_train` so learning a fact doesn't erase
skills  **the wrong/dirty corpus corrupts the model on every teach** (cost a long debug once). The bins
must be the **same corpus + same tokenizer** the checkpoint trained on. They're gitignored; carry them with
the checkpoint or rebuild with `prepare_scale.py`. Sanity: a slice decodes to clean prose/math, and
`H.val_ppl(base_lm, big_valid)` lands near the training value (not ~2 = contaminated).

---

## 7. State as of this handover
**UPDATE (2026-06-30).** The H100 grew the model to a **565M snapshot** (d768/24L/mlp17, 18 spin + 6 attn,
**val ppl ~22**), synced to the laptop as `pragnosia_spin.pt` (the 284M preserved as `pragnosia_284m.*`, still
the paper's anchor). New results this session: **(a) Generalization confirmed (reviewer's #1)**  a diagonal-REAL
carrier (no phase, `real_dominant` mode) placed as the core ALSO beats attention: multi-seed attention 360.9±4.5 /
complex-spin 115.7±1.8 / **real-no-phase 102.2±2.1 (3.5×, fewer params)**  so it's about PLACEMENT, not the phase.
**(b) Open-model comparison** (`eval_compare.py`): the 284M matches/beats GPT-2(124M)+Cerebras-256M on
ARC-Easy/HellaSwag, trails on PIQA/LAMBADA. **(c) Long-context: mechanism + training implemented.** The cross-window
carrier-carry wasn't wired for spin_dominant (carrier is intra-block)  fixed (`forward_state` + threaded `forward`);
but a model trained fresh-per-window doesn't USE a carried state (measured: 62.6 w/ carry vs 58.0 w/o). So added
**windowed-TBPTT long-context training** (`LONG_CTX_W=W`, optional `LONG_BPTT=1`)  verified to teach the carry
(407 vs 649 ppl on continuations). **Run it on the H100: `VRAM_CAP=16 LONG_CTX_W=4 python3 train_pragnosia.py --resume
--no-grow`** (see RUNBOOK 2b). Multi-seed ablation = `ablation_sweep.py` (the table is complete bar a transformer-11L
re-run at a sane lr, the auto-lr diverged it).

**The validated result is the 284M spin-dominant** (`pragnosia_spin.pt`, d=768/20L/mlp9), trained to step 418K /
**val ppl 24.5**  this is the §2 at-scale result (carrier 306× load-bearing; arithmetic/knowledge/ToM emerged).
A **bf16 copy is synced to the laptop** and verified (ppl 25.2 ≈ 24.54 fp32). **In progress on the H100 (read-only
check):** with growth on, the run has **self-grown to 23 layers / mlp_mult 12 (~440M)** and is re-training after the
grow (step 444K, ppl ~25.8 re-annealing  bigger model, not yet beating the 284M's 24.5; over-training toward the
~600M capability run at 30–35 tok/param). The **`carrier="none"` baseline is prepped but NOT yet run** (config only). The earlier 27M laptop run (self-grew 24M→27.3M, ppl ~40) was the
small-scale proof. The laptop holds a ~1 GB strided CoT-inclusive subsample (500M tokens) for local teach/replay.
The 1.4B (`pragnosia_best.pt`, attention-dominant, the wrong build) is the reference. The decisive experiment  spin-dominant vs transformer-only at 200M–1B on the full CoT
corpus  **is the H100's job (§3).** Deferred no-hardcode items (need teach-regression tests): teach
`plasticity`/`target` clamps, trainer `warm`/`target_eff`. Key memory for future Claude sessions lives in
the user's `memory/` dir (`spinning-brain-project.md`).
