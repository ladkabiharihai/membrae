# Pragnosia — parallel-spin (production training notes)

Pretraining checkpoint: from-scratch 228M, **val PPL 20.76** (~5B digit-tokenized tokens, 8.3h).
**Current best: `pragnosia_best.pt` — 1.4B params (48L), val PPL 17.35** after the Jun 19–21
freed-GPU growth+refine run. `pragnosia.json` tracks the live arch (d=1024, **layers=48,
mlp_mult=12**, ctx=256, vocab 16384).

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
- **Background-thread prefetcher** (`s6_hybrid.Prefetcher`, `PREFETCH=1`): overlaps the memmap
  gather + pinned H2D copy with compute (GPU util 90% → 99%; the model is compute/bandwidth-bound
  so wall-clock gain is small, but utilization is clean).

**Refine run result:** with the adaptive lr the drift fully reversed and the model **improved past
its old peak → val PPL 17.35** (best ever; was 20.5 → 19.65 → **17.35**).

### Final test results (best 1.4B model, `pragnosia_best.pt`, ppl 17.35)
- **Faculties: 14/14 wired** + teach/recall (continual learning) ✓
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
- **window2** (~160B, freed-GPU growth): knowledge-heavy fuel — 64% knowledge, every skill at scale.
- **Science** (physics/astro/particle/cosmology/bio/chem): `common-pile/arxiv_papers_filtered`
  full papers (bulk) + `camel-ai` physics/chem/bio + arXiv abstracts. **Code:** glaive-code-assistant
  + Magicoder + evol-codealpaca (the-stack/starcoder are gated; codeparrot dies on bad parquet rows).
  **Reasoning (“GI reasoning”):** synthetic **kinship** (random family trees, true relation via LCA) +
  abstract **pattern/analogy** + **ARC-AGI** small grids + bAbI + BBH + ANLI + RuleTaker + Open-Platypus.
- **Build engineering:** pipelined `encode_fast` (producer thread overlaps stream-decode with
  `encode_batch`), sharded multi-worker builds (`split_dataset_by_node`), resilient per-source
  try/except. Helpers: `status.sh`, `resume.sh` (adaptive-lr resume), `faculty_test.sh`
  (CPU-only, snapshot, never interrupts training).

## The model — PARALLEL spin
SpinAttentionLM (`s6_hybrid.py`): d=1024, 16 heads, ctx=256, vocab 16384 (digit-aware BPE,
`data/bpe.json`). Born at 16L/mlp4 (228M); **grown by grow-as-you-train to 48L/mlp12 (1.4B)** —
depth/width are the only things growth changes (d fixed), so the parallel carrier is unaffected.
Config `pragnosia.json` always tracks the live arch (now layers=48, mlp_mult=12).

The spin carrier is now a **diagonal-complex (LRU/S5-style) recurrence**
`h_t = lambda (.) h_{t-1} + b_t`, `lambda_j = exp(-exp(nu_j))·exp(i·phi_j)` — each channel a
damped complex oscillator at its own frequency ("the answer lives in the phase", per
channel). Diagonal ⇒ **matmul-free (O(T·d)) and an associative scan**, so it runs as a
log-depth **parallel scan** instead of the original 256-step sequential tanh loop:
- **~2× faster training** (168K vs 109K tok/s beside prod); the 228M model trained faster
  than the old 176M and reached the **same perplexity** (20.76 vs 20.47).
- **Proven dynamics preserved**: brain-swap control holds (`own < swapped < random`,
  signal grew 0.001→0.016 with training); `brain.py test` = **14/14 + WIRED**.
- Arithmetic on the finished model: `5+7=12 · 9+6=15 · 20+30=50 · 7×8=56` ✓ (4/6).

This replaces the dense-tanh sequential `SpinStep` carrier (kept in git history + the
paper as the anchor). It resolves the paper's limitation (iii) "the spinning core is
sequential and cannot parallelize across time."

## Prior model
The old 176M dense-tanh instance (val PPL 20.47, 12 layers) is superseded by the above.

## Data we trained on  (~5.4B tokens, math-heavy)
Built by `prepare_data_fast.py --tokens 6e9 --web-config sample-100BT` (parallel
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
- **`prepare_data_fast.py`** (new) — parallel `encode_batch` corpus builder (all cores).
- **`prepare_data_shard.py`** (new) — multiprocess sharded builder (for very large corpora).

## Result
Old 176M (old tokenizer, ~2% math): could not do arithmetic (`5 + 7 = 8`).
This model: basic arithmetic mostly correct — `5 + 7 = 12`, `9 + 6 = 15`, `20 + 30 = 50`,
`100 - 25 = 75` (✓), though multi-digit non-round still slips (`12 + 13 = 45`) and the
chain-of-thought prose is often confabulated. The 176M size is the remaining ceiling.

## Run / test
```
python3 brain.py test            # full self-test: every faculty + language
python3 brain.py "5 + 7 ="       # one-shot: say anything to it, see what it does
python3 brain.py                 # it LIVES: talk to it (answers, learns, wonders, looks up, grows)
```

## ⚠️ Ship the data bins with the checkpoint (continual learning depends on it)
`pragnosia.pt` alone is **not enough** to run `brain.py` correctly — you also need
`data/big_train.bin`, `data/big_valid.bin`, and `data/bpe.json`, all from THIS run:
- `big_train.bin` = replay source for `teach()`. Continual learning interleaves
  replay from it so new facts don't erase old skills. Replaying the **wrong/dirty**
  corpus (e.g. one rebuilt locally with HTML markup) corrupts the model on every
  teach — including the startup identity install — so `brain.py` looks broken while
  `brain.py probe` (which never teaches) stays fine. It must be the **same corpus, same
  tokenizer** as training.
- `big_valid.bin` = what the abstention boundary / seek-match / answer-confidence
  self-calibrate from. A fake valid set (decodes as garbage, ppl ~2 instead of ~18)
  gives wrong boundaries.

These bins are gitignored. Verify a transferred bin: it should decode to clean
prose/math under `bpe.json`, and base-model val ppl on `big_valid` ≈ 18–22 (this
run's training value), not ~2. See RUNBOOK.md STEP 1 for the full rationale.
