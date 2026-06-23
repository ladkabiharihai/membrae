# Pragnosia — parallel-spin (production training notes)

> ## ⚡ HEADLINE FINDING (this session, measured) — the carrier must be the CORE, not a side-channel
> The scaled 1.4B model **drifted to attention-dominant**. The spin carrier is a single fixed
> ~5.25M-param layer, so as the model grew its **parameter share collapsed**:
> **5.76% (21M) → 1.90% (276M) → 0.37% (1.4B)**. The brain-swap causal test at 1.4B (fp32) still
> passes *directionally* (own < swapped < random), but the causal signal is only
> **0.00086 NLL = 0.028% of the loss** — down ~20× from 0.016 at 276M. So **as built, the spin
> carrier is causally negligible at scale.** (An external reviewer correctly flagged this.)
>
> The *intended* design was always **spin-dominant** — spin as the core token-mixer, attention only a
> small helper for generation. The implementation simply drifted to the inverse. A clean
> **param-matched ablation** (this session, same tokens/budget, d=512) confirms the intended design
> wins: a new `carrier="spin_dominant"` (SpinBlock = the spin carrier as the token mixer replacing
> attention, with an attention Block only every 4th layer) reaches **val ppl 219.1 vs 279.0 for
> attention-only and 228.0 for the per-block hybrid** — spin-dominant wins by **~21% at matched
> params** (37.3M vs 35.7M / 46.2M), and beats a 45M param-matched transformer-only (270). The
> carrier **earns its weights when it is the core, not a side-channel.**
> **Caveat:** small-scale / short-budget / single-seed. Must be replicated at 200M–1B before it's
> proven; transformers sometimes close SSM gaps with more scale/training. This is a strong,
> properly-controlled signal, **not** a final result.

Pretraining checkpoint: from-scratch 228M, **val PPL 20.76** (~5B digit-tokenized tokens, 8.3h).
**Current best: `pragnosia_best.pt` — 1.4B params (48L), val PPL 17.35** after the Jun 19–21
freed-GPU growth+refine run (this is the *attention-dominant* drift instance — see HEADLINE above).
`pragnosia.json.1p4B` tracks that arch (d=1024, layers=48, mlp_mult=12, ctx=256, vocab 16384).
The live `pragnosia.json` now points at the **`spin_dominant`** arch (d=512, mlp4,
`carrier="spin_dominant"`, ckpt `pragnosia_spin.pt`) — the intended design under validation.
It was born at 4 layers and **self-grew to 5 layers** in production (24M→27.3M), then capped
at the data budget (no hardcoded size cap).

## Jun 19–21 freed-GPU GROWTH + REFINE run (the 276M → 1.4B story)
After post-training (228M → 276M beside prod), prod services were stopped for a 3-day window
and the model trained full-throttle on a freed H100 with **grow-as-you-train enabled**.

**Self-growth fired repeatedly to the cap:** 276M → 305M → 878M → … → **1.4B (48 layers,
mlp_mult=12)**, where it hit the safety cap (`max_layers=48`). `grow.py` is function-preserving
(near-identity new blocks); each grow re-tunes the batch to fit. The model grew ~5× by adding
its own capacity on saturation — no separate scale run.

**Over-growth was real and we caught it:** growing faster than it could train left the fresh 1.4B
**undertrained** — val ppl drifted *up* (19.76 → 22.8) and capability/emergence slid. Root cause:
the original cosine schedule held lr near peak (2e-4), over-writing faster than it refined.

**Fixes (now in `train_pragnosia.py`), all in the project's self-adjusting spirit:**
- **ADAPTIVE learning rate** (no hardcoded schedule): warm up, then the **validation signal drives
  lr** — *auto-halve* on degradation (val > best+2%), *auto-ease* ×0.7 on plateau. Mirrors
  grow-on-saturation; `--lr` is just the initial seed, so a too-high start self-corrects.
- **Save-on-BEST** (`pragnosia_best.pt`): lowest-ppl weights are preserved alongside the resumable
  `latest` — refinement can no longer silently overwrite the good model.
- **Background-thread prefetcher** (`s6_hybrid.Prefetcher`): overlaps the memmap
  gather + pinned H2D copy with compute (GPU util 90% → 99%). The prefetcher CUDA race
  that used to trip `torch.compile` device-side asserts is now fixed, so **compile can stay
  ON** (it fuses the spin scan for ~2× throughput).

**Refine run result:** with the adaptive lr the drift fully reversed and the model **improved past
its old peak → val PPL 17.35** (best ever; was 20.5 → 19.65 → **17.35**).

### Final test results (best 1.4B reference model, `pragnosia_best.pt`, ppl 17.35)
*(historical, attention-dominant drift instance — measured at the time on the old self-test;
the old "14 proven faculties / unified_brain" self-test has since been removed. The current
self-test is `python3 brain.py test` on the spin-dominant model — see "Run / test" below.)*
- **Faculties wired** + teach/recall (continual learning) ✓
- **Arithmetic: 10/10** — `23+45=68 · 100−37=63 · 12×12=144 · 250+250=500 · 9×7=63 · 144/12=12 · 1000−1=999`
- **Knowledge ~7/8** — Paris, Tokyo, Shakespeare, **Jupiter**, H2O, Everest, Portuguese (miss: continents)
- **Human-cognition emergence 9/20 (45%)** — **Theory-of-Mind 2/2** (Sally-Anne false belief),
  **Counterfactual 2/2**, **Causal 2/2**, **Metacognition 1/1**, working-memory, category-formation
- **Unmemorizable in-context emergence ~30%** — binds made-up words/entities/facts, abstracts numeric rules
- **Honest gaps:** relational analogy (0/3), compositional/systematic generalization (0/2), formal
  deduction; precise long-tail facts still wobble (the undertraining-for-size signature — Chinchilla
  ≈28B tokens for 1.4B, the window fed far less). Identity is NOT in the weights (it bleeds when
  injected — see `identity_sentences.txt`); it is installed at runtime by `brain.py`.

### Two-window scaling data (`prepare_scale.py`, on the 3.2 TB NVMe)
Built two purpose-proportioned corpora, each containing ALL requested data types:
- **window1** (~20B, beside-prod): weighted to new skills — instruction 12% · math 12% ·
  **code 14% · science 14% · reasoning 16%** · knowledge 32%.
- **window2** (freed-GPU growth): knowledge-heavy fuel — 64% knowledge, every skill at scale.
  The live CoT-enriched build is **`/mnt/kv_cache/pragnosia_data/window2_train.bin` — 330 GB /
  176.6B tokens** on the H100. The laptop carries a ~1 GB strided CoT-inclusive subsample
  (~500M tokens) for replay (see RUNBOOK STEP 1).
- **Science** (physics/astro/particle/cosmology/bio/chem): `common-pile/arxiv_papers_filtered`
  full papers (bulk) + `camel-ai` physics/chem/bio + arXiv abstracts. **Code:** glaive-code-assistant
  + Magicoder + evol-codealpaca (the-stack/starcoder are gated; codeparrot dies on bad parquet rows).
  **Reasoning (“GI reasoning”):** synthetic **kinship** (random family trees, true relation via LCA) +
  abstract **pattern/analogy** + **ARC-AGI** small grids + bAbI + BBH + ANLI + RuleTaker + Open-Platypus.
- **Build engineering:** pipelined `encode_fast` (producer thread overlaps stream-decode with
  `encode_batch`), sharded multi-worker builds (`split_dataset_by_node`), resilient per-source
  try/except. Helpers: `ops.sh status`, `ops.sh resume` (adaptive-lr resume), `ops.sh test`
  (CPU-only, snapshot, never interrupts training).

## Carrier modes + the spin-dominant ablation (this session)
`SpinAttentionLM` now takes a `carrier=` mode (wired through `pragnosia.json` →
`train_pragnosia.py` → `grow.py`):
- **`none`** — transformer-only baseline (no carrier).
- **`single`** (the 1.4B used this) — one `SpinCarrier` after all blocks. Its share → 0 as depth
  grows; this is exactly the drift that made it causally negligible at 1.4B.
- **`per_block`** — one carrier per block, so the share stays constant with depth.
- **`spin_dominant`** (intended design) — `SpinBlock` (the carrier as the **token mixer**, strong
  gate `sigmoid(2)≈0.88`, in place of attention) for every layer except every 4th, which is a
  standard attention `Block` (the helper). Spin is the core compute.

**Param-matched ablation** (same 6.1M tokens, same budget, d=512, 8L, single seed):

| Design | Params | Val ppl |
|---|---|---|
| attention-dominant (transformer-only) | 35.7M | 279.0 |
| **SPIN-DOMINANT (intended)** | **37.3M** | **219.1** (−21.5%) |
| attn + per-block spin hybrid | 46.2M | 228.0 |
| param-matched transformer-only (11L) | 45M | 270 |

Spin-dominant is the **best and most parameter-efficient** — it beats the 46M hybrid (228) and the
45M transformer-only (270) with only 37M. **The spinning-brain design is sound; the scaled model
was simply built backwards** (attention-dominant with the spin reduced to a side-channel).
*Honest caveat:* small-scale, short-budget, single-seed — needs larger-scale, longer, multi-seed
replication before it's proven at 1B+; transformers sometimes close SSM-style gaps with more scale.

**Speed framing (not a free lunch at short context):** spin-dominant is **not** faster than
attention on 256 tokens — the scan does more work there. Its advantage is *structural*: long
context (spin `O(T·d)` vs attention `O(T²·d)`) and inference (the carrier is recurrent → `O(1)`/token,
constant memory, **no KV-cache growth**). `torch.compile` fuses the spin scan — measured
**49K → ~100K tok/s at bs=24 on an RTX 4060 (~2×)**; bf16 inference.

## The model — PARALLEL spin
SpinAttentionLM (`s6_hybrid.py`): d=1024, 16 heads, ctx=256, vocab 16384 (digit-aware BPE,
`data/bpe.json`). The 1.4B was born at 16L/mlp4 (228M); **grown by grow-as-you-train to 48L/mlp12
(1.4B)** — depth/width are the only things growth changes (d fixed). Because the `single` carrier is
one fixed `d`-shaped layer, that growth is exactly what diluted it from 5.76% → 0.37% of params.
`pragnosia.json.1p4B` tracks that arch; the live `pragnosia.json` tracks the spin-dominant arch.

The spin carrier is now a **diagonal-complex (LRU/S5-style) recurrence**
`h_t = lambda (.) h_{t-1} + b_t`, `lambda_j = exp(-exp(nu_j))·exp(i·phi_j)` — each channel a
damped complex oscillator at its own frequency ("the answer lives in the phase", per
channel). Diagonal ⇒ **matmul-free (O(T·d)) and an associative scan**, so it runs as a
log-depth **parallel scan** instead of the original 256-step sequential tanh loop:
- **~2× faster training** (168K vs 109K tok/s beside prod); the 228M model trained faster
  than the old 176M and reached the **same perplexity** (20.76 vs 20.47).
- **Dynamics directionally preserved**: brain-swap control holds (`own < swapped < random`);
  `brain.py test` reports **WIRED**. But honestly, the carried-state signal is small and
  *shrinks as a fraction of the loss as the model grows* (0.016 at 276M → **0.00086 = 0.028% of
  the loss at 1.4B**), because the single fixed carrier is diluted by depth. The fix is to make
  spin the core mixer (`spin_dominant`), not to keep it as a side-channel.
- Arithmetic on the finished model: `5+7=12 · 9+6=15 · 20+30=50 · 7×8=56` ✓ (4/6).

This replaces the dense-tanh sequential `SpinStep` carrier (kept in git history + the
paper as the anchor). It resolves the paper's limitation (iii) "the spinning core is
sequential and cannot parallelize across time."

## Prior model
The old 176M dense-tanh instance (val PPL 20.47, 12 layers) is superseded by the above.

## Data we trained on  (~5.4B tokens, math-heavy)
Built by `prepare_scale.py --tokens 6e9 --web-config sample-100BT` (parallel
`encode_batch` tokenization). Final mix (`built:` counts):
- **math 1,807,524 docs** — GSM8K + Orca-Math + **MetaMathQA**, with full chain-of-thought
  worked solutions, **oversampled 3×** and placed FIRST in the mix.
- reason 4,300,907 docs — OpenOrca + Alpaca + Dolly (general instruction/reasoning).
- code 38,628 · chat 9,846 (OpenAssistant) · grammar 69,071 (CoEdit).
- web 2,716,000 docs — FineWeb-Edu (`sample-100BT`).
- Totals: curated 2.24B + web 3.16B = **5.40B tokens** (~41% reasoning/math, vs ~2% before).

## What changed in the files (the "make reasoning work" fixes #3/#4/#5)
- **`prepare_data.py`**
  - **#3 digit-aware tokenizer**: pre-tokenizer `Sequence([ByteLevel, Digits(individual_digits=True)])`
    so numbers split per-digit (`127` → `1`,`2`,`7`) — place-value aligned, clean roundtrip.
  - **#4/#5 math/CoT diet**: `curated()` restructured — math/CoT (incl. MetaMathQA) yielded
    FIRST and oversampled (`MATH_OVERSAMPLE=3`); huge OpenOrca demoted to last filler so math
    isn't drowned out. Added `--tokens` / `--web-config` flags (decouple budget from arch).
- **`s6_hybrid.py`** — `load()`/`batch()` now memory-map the token bin (int16) and cast each
  batch to long on the fly → RAM stays ~flat instead of loading the whole corpus as int64.
- **`train_pragnosia.py`** — `MEM_STOP_GB=2.0` self-stop guard: if free GPU VRAM ≤ 2 GB it saves
  the checkpoint and stops cleanly (safe to co-run with the prod GPU services). Unchanged otherwise.
- **`run_fast.py`** (new) — wrapper that force-enables `torch.compile` (fuses the spin recurrence)
  WITHOUT editing the trainer; ~1.3–1.8× speedup.
- **`prepare_scale.py`** (new) — parallel `encode_batch` corpus builder (all cores).
- **`prepare_scale.py`** (new) — multiprocess sharded builder (for very large corpora).

## Result
Old 176M (old tokenizer, ~2% math): could not do arithmetic (`5 + 7 = 8`).
This model: basic arithmetic mostly correct — `5 + 7 = 12`, `9 + 6 = 15`, `20 + 30 = 50`,
`100 - 25 = 75` (✓), though multi-digit non-round still slips (`12 + 13 = 45`) and the
chain-of-thought prose is often confabulated. The 176M size is the remaining ceiling.

## Run / test
```
python3 brain.py test            # full self-test on the spin-dominant model:
                                 #   language + honesty + learn/seek + SUBCONSCIOUS memory
                                 #   + COGNITION (metacognition / deliberation / monologue)
python3 brain.py "5 + 7 ="       # one-shot: say anything to it, see what it does
python3 brain.py                 # it LIVES: talk to it (answers, learns, wonders, looks up, grows)

python3 faculty_test.py pragnosia_spin.pt   # CPU-only / read-only LM capability battery
                                            # (builds with carrier=cfg['carrier'])
```
The old "14 proven faculties / unified_brain" self-test has been removed (along with
`unified_brain.py`/`unified_brain.pt`). The self-test now runs against the spin-dominant LM.

## ⚠️ Ship the data bins with the checkpoint (continual learning depends on it)
The checkpoint alone is **not enough** to run `brain.py` correctly — you also need the
train bin, `data/big_valid.bin`, and `data/bpe.json`, all from THIS run:
- The train bin (`data/window2_train.bin` locally, or `window2_train` per the config's
  `train_bin`) = replay source for `teach()`. Continual learning interleaves replay from
  it so new facts don't erase old skills. Replaying the **wrong/dirty** corpus (e.g. one
  rebuilt locally with HTML markup) corrupts the model on every teach — including the
  startup identity install. It must be the **same corpus, same tokenizer** as training.
  (The H100 source is `/mnt/kv_cache/pragnosia_data/window2_train.bin`, 330 GB / 176.6B
  tokens; the laptop ships a ~1 GB / ~500M-token strided subsample.)
- `big_valid.bin` = what the abstention boundary / seek-match / answer-confidence
  self-calibrate from. A fake valid set (decodes as garbage, ppl ~2) gives wrong boundaries.

These bins are gitignored. Verify a transferred bin: it should decode to clean
prose/math under `bpe.json`, and base-model val ppl on the valid set should land near the
checkpoint's reported training value, not ~2. See RUNBOOK.md STEP 1 for the full rationale.
