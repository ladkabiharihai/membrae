# Pragnosia — Runbook (train once, then it grows by itself)

## STEP 1 — Build the rebalanced corpus  (downloads several GB, ~once)
```
python3 prepare_data.py
```
Produces: `data/corpus_big.txt`, `data/bpe16384.json`, `data/big_train.bin`,
`data/big_valid.bin`, and `pragnosia.json` (the model config).
Corpus mix (stories deliberately only ~10%): Simple Wikipedia (knowledge) +
OpenOrca/Alpaca/Dolly (reasoning) + GSM8K/Orca-Math (mathematics) +
CodeAlpaca/Python-instructions (coding) + OpenAssistant (multi-turn chat) +
CoEdit (English grammar) + TinyStories (10%, fluency).

> ### ⚠️ CRITICAL — the data bins must match the model, and travel with it
> `data/big_train.bin` and `data/big_valid.bin` are **not optional runtime files** —
> they are load-bearing:
> - **`big_train.bin` is the replay source for continual learning.** Every `teach()`
>   / curiosity-learn (and the startup identity install) interleaves replay batches
>   from it so learning a new fact does **not** erase old skills. Without replay,
>   teaching is catastrophic (val ppl 22 → 1000+). With replay from the **wrong or
>   dirty** corpus, every teach drags the model off its trained distribution and
>   leaks junk into generations — the model looks broken even though the base
>   checkpoint is fine. The bin **must be the same corpus the checkpoint was trained
>   on**, tokenized with the **same `data/bpe.json`**.
> - **`big_valid.bin` is what every internal scale self-calibrates from** (abstention
>   boundary, seek-match, answer-confidence). A non-representative valid set gives
>   wrong boundaries → the controller mis-routes (abstains/over-learns).
>
> These bins are **gitignored** (large, regenerable). When you move a trained
> `pragnosia.pt` to another machine, **carry the matching `big_train.bin` +
> `big_valid.bin` + `bpe.json` with it**, or regenerate them there with
> `prepare_data_fast.py` using the *same* tokenizer. Never rebuild them from a
> different/uncleaned corpus. Sanity check before use: a slice should decode to clean
> prose/math (not HTML markup), and `H.val_ppl(base_lm, big_valid)` should land near
> the checkpoint's reported training val ppl (~18–22 for the 176M run), not ~2.

## STEP 2 — Train the 176M brain  (ONE long run; GPU-ADAPTIVE)
```
nohup python3 train_pragnosia.py > pragnosia_train.log 2>&1 &
tail -f pragnosia_train.log
```
- 176M params (d=1024, 12 layers, vocab 16384), best checkpoint -> `pragnosia_168m.pt`.
- **GPU-adaptive**: it auto-detects the GPU's VRAM and tunes batch size, gradient
  accumulation, precision (bf16 on capable GPUs), and compile. Move to a bigger
  GPU and just (re)start or `--resume` — it picks up the new hardware and trains
  optimally. No edits.
- Resume on ANY GPU:  `python3 train_pragnosia.py --resume`
- When `pragnosia_168m.pt` exists, **brain.py automatically uses it**.

## STEP 3 — Use it  (after training; this is the part you asked for)
The brain decides everything itself; nothing is hardcoded.
Everything you do with the brain is a mode of the ONE file `brain.py` (no other scripts):
```
python3 brain.py test                          # full self-test: every faculty + language
python3 brain.py chat                           # talk to it (it decides: answer/seek/abstain/learn)
python3 brain.py child                          # raise it like a child: it wonders its own
                                                #   questions, looks up what it doesn't know
                                                #   (Wikipedia), learns it, and grows when saturated
python3 brain.py ask "What is the capital of France?"
python3 brain.py teach "Mount Everest is the tallest mountain."   # teaches + PERSISTS
python3 brain.py probe "any prompt"             # quick one-shot decision trace
```

## STEP 4 — It grows by itself (NO retraining, ever)
- In `chat`, when you tell it something new, it **decides on its own to learn it**
  (curiosity = its own uncertainty) and **saves it on exit** — it remembers next session.
- `teach` persists immediately (weights + memory in `learned_memory.json`).
- After it has grown a lot, re-tune its internal scales (optional, no retraining):
  ```python
  from brain import Brain; b = Brain(); b.recalibrate()
  ```

## What it can do once trained (all in one model, all real, none hardcoded)
- **Reason** about general things (commonsense/causal) — stronger at 168M than 20M
- **Generate / understand language**, answer questions
- **Know what it doesn't know** and say "I don't know" instead of fabricating
- **Seek** answers from its memory
- **Learn new facts continuously** without forgetting — and **persist** them
- **Be curious** — act on its own uncertainty
- **Know itself** (name Pragnosia, its nature) and decide every action itself

## Honest expectation
168M is GPT-2-small/medium class. It will be clearly better than the 20M model —
more reliable reasoning, more facts, sharper "I don't know" — but it is still small;
it will not match large LLMs. Strength comes from scale + reasoning data, both of
which this run adds as far as one laptop allows.
```
