# Pragnosia 176M — reasoning rebuild (training notes)

Best checkpoint: `pragnosia.pt`, **val PPL 20.47** (176M params, from scratch).

## The model
SpinAttentionLM (`s6_hybrid.py`): d=1024, 12 layers, 16 heads, mlp_mult=4, ctx=256,
vocab 16384. Config in `pragnosia.json`. Tokenizer `data/bpe.json` (digit-aware BPE).

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
