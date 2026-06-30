# Pragnosia  Runbook (train once, then it grows by itself)

> ### ⚡ Carrier mode (the intended + validated design)
> `pragnosia.json` has a **`carrier`** field selecting the token-mixing design:
> `none` (transformer-only) · `single` (one spin carrier after all blocks  the 1.4B
> `pragnosia_best.pt` used this and it drifted to a causally-negligible side-channel,
> carrier 0.37% of params) · `per_block` · **`spin_dominant`** (the intended design 
> a diagonal-complex spin-carrier as the token mixer replacing attention in 3 of every
> 4 layers, with a normal attention block every 4th layer). A param-matched ablation
> found **spin-dominant wins ~21%** (219 vs 279 ppl). The live `pragnosia.json` is set to
> `spin_dominant` (d=512, carrier `pragnosia_spin.pt`); the 1.4B attention-dominant arch
> is preserved in `pragnosia.json.1p4B` as a reference, not the path forward. To train a
> design, point `pragnosia.json` at it and run STEP 2  `train_pragnosia.py` and `grow.py`
> read the carrier mode from the config. See `TRAINING_NOTES.md` for the full ablation
> table and the honest small-scale/single-seed caveat.

## STEP 1  Build the corpus  (downloads several GB, ~once)
```
python3 prepare_data.py
```
Produces: `data/bpe.json` (digit-aware BPE tokenizer), `data/big_train.bin`,
`data/big_valid.bin`, and `pragnosia.json` (the model config).
Corpus mix (stories deliberately only ~10%): Simple Wikipedia (knowledge) +
OpenOrca/Alpaca/Dolly (reasoning) + GSM8K/Orca-Math (mathematics) +
CodeAlpaca/Python-instructions (coding) + OpenAssistant (multi-turn chat) +
CoEdit (English grammar) + TinyStories (10%, fluency).

For large-scale corpora (the window1/window2 scaling builds), use `prepare_scale.py`
(parallel/sharded tokenizer); for post-training data use `prepare_posttrain.py`.

> ### ⚠️ CRITICAL  the data bins must match the model, and travel with it
> `data/big_train.bin` (or the larger `window2_train.bin`) and `data/big_valid.bin` are
> **not optional runtime files**  they are load-bearing:
> - **The train bin is the replay source for continual learning.** Every `teach()`
>   / curiosity-learn (and the startup identity install) interleaves replay batches
>   from it so learning a new fact does **not** erase old skills. Without replay,
>   teaching is catastrophic. With replay from the **wrong or dirty** corpus, every
>   teach drags the model off its trained distribution and leaks junk into generations
>    the model looks broken even though the base checkpoint is fine. The bin **must be
>   the same corpus the checkpoint was trained on**, tokenized with the **same
>   `data/bpe.json`**. `brain.py` replays the train bin during teach.
> - **`big_valid.bin` is what every internal scale self-calibrates from** (abstention
>   boundary, seek-match, answer-confidence). A non-representative valid set gives
>   wrong boundaries → the controller mis-routes (abstains/over-learns).
>
> These bins are **gitignored** (large, regenerable runtime state). When you move a
> trained checkpoint to another machine, **carry the matching train bin +
> `big_valid.bin` + `bpe.json` with it**, or regenerate them there with `prepare_data.py`
> / `prepare_scale.py` using the *same* tokenizer. Never rebuild them from a
> different/uncleaned corpus. Sanity check before use: a slice should decode to clean
> prose/math (not HTML markup), and base-model val ppl on the valid set should land near
> the checkpoint's reported training val ppl, not ~2.
>
> **Compact replay for easy transfer (lossless).** Replay only needs *distribution
> coverage*, not full size. The full H100 `window2_train.bin` is 330 GB / 176.6B tokens
> (CoT-enriched) at `/mnt/kv_cache/pragnosia_data/window2_train.bin`; the laptop ships a
> **~1 GB strided CoT-inclusive subsample (~500M tokens)** instead of the whole thing.
> To make one, **stride-sample chunks across the WHOLE corpus** (e.g. one 4096-token
> chunk every Nth)  never a contiguous slice, which would miss whole regions (the corpus
> is ordered math→web). Identity / word-importance caches regenerate on first run.

## STEP 2  Train the brain  (ONE long run; GPU-ADAPTIVE)
```
nohup python3 train_pragnosia.py --resume --bs 24 --lr 0 > pragnosia_train.log 2>&1 &
tail -f pragnosia_train.log
```
- The arch comes from `pragnosia.json` (d/layers/carrier); best checkpoint → the config's
  `ckpt` (e.g. `pragnosia_spin.pt` for the spin-dominant run).
- **GPU-adaptive**: it auto-detects the GPU's VRAM and tunes batch size, gradient
  accumulation, precision (bf16 on capable GPUs), and `torch.compile`. **compile ON gives
  ~2× throughput** (49K→~100K tok/s at bs=24 on an RTX 4060); the prefetcher CUDA race that
  caused compile asserts is fixed. On the laptop set `NOCOMPILE=1` only if compile misbehaves.
- **LR**: `--lr 0` on a **fresh** run derives the peak LR via the Smith range-test (÷10).
  On `--resume` it **RESTORES** the peak LR from the sidecar (find_lr is unreliable on
  trained weights). After warm-up the validation signal drives LR (auto-halve on
  degradation, ease ×0.7 on plateau).
- **True-resume**: the sidecar `pragnosia_<ckpt>_state.json` carries
  `{step, lr_scale, lr_wait, best, lr}`. `--resume` **continues from that step at the
  restored LR  it does NOT restart from 0.** The sidecar is gitignored (runtime state).
- **`TOK_PER_PARAM`** env var (default 20) = how many tokens/param to train each size
  before it may grow. Raise to 80 for an inference-optimal / leaner model (it does NOT
  speed training).
- Other flags: `--steps N`, `--no-grow` (disable grow-as-you-train).
- When the config's `ckpt` exists, **brain.py automatically uses it**.

### STEP 2b  Long-context training (teach the carrier to carry across windows)
The spin/real carrier is recurrent (O(T)), so it *can* hold context past the trained 256-token
window  but a model trained on fresh 256-windows never learns to **use** a cross-window state
(measured: feeding the 565M one didn't help, next-window ppl 62.6 w/ carry vs 58.0 w/o). Fix =
train with the windowed carry:
```
# resume + train on long context. W windows of ctx are carried (attention stays at ctx):
VRAM_CAP=16 LONG_CTX_W=4 python3 train_pragnosia.py --resume --no-grow   # 4*256 = 1024 eff ctx
```
- `LONG_CTX_W=W` (or `"long_ctx_windows"` in `pragnosia.json`): feed each sample as W windows,
  carrying the carrier state across them. Attention + positions stay inside the trained ctx.
- **Default = detached TBPTT**: per-window backward, memory bounded to **one window** (safe on a
  shared GPU; auto-sizes). `LONG_BPTT=1` = full BPTT across windows (carrier also learns to WRITE
  cross-window state  stronger, grad-checkpointed, ~½ speed; set `--bs` small as fit_batch only
  measures one window).
- Validation prints `VAL_PPL` measured **through the carry**  watch it drop = the model learning
  long context. Growth is auto-disabled in this mode (refining, not growing).
- If the restored peak LR spikes the long-ctx ppl early, add `--lr 3e-4` for a gentler pass.

## STEP 3  Use it  (after training; this is the part you asked for)
The brain decides everything itself; nothing is hardcoded.
Talking to it IS how it works -- answering, learning, curiosity, looking things up and
growing are all intrinsic to the ONE file `brain.py`, not separate commands:
```
python3 brain.py            # it LIVES: talk to it; it answers what it knows, learns what
                            #   you tell it, wonders its own questions, looks up what it
                            #   doesn't know (Wikipedia), and grows itself when it saturates.
                            #   (press Enter alone to let it think; 'quit' saves + exits)
python3 brain.py "tell or ask it anything"     # one-shot version of the same
python3 brain.py test                          # full self-test on the spin-dominant model
```
`python3 brain.py test` exercises the whole spin-dominant model: language, honesty,
learn/seek, the in-weights **SUBCONSCIOUS** memory, and **COGNITION**
(metacognition / deliberation / autonomous monologue).

For a read-only, CPU-only LM capability battery against any checkpoint (never touches a
running training job):
```
python3 faculty_test.py pragnosia_spin.pt      # builds with carrier=cfg['carrier']
```

## STEP 4  It grows by itself (NO retraining, ever)
- Growth is **self-governing and function-preserving**, fired only on probe-confirmed
  saturation. The only caps are the **DATA budget** (tokens/20) and **VRAM**  there are
  no hardcoded size caps. Confirmed in production: a run grew 24M(4L)→27.3M(5L) by itself,
  then capped at the data budget.
- `grow.py` modes (all carrier-aware): `grow_depth` / `grow_width` / `shrink_width`.
- In chat, when you tell it something new, it **decides on its own to learn it**
  (curiosity = its own uncertainty) and **saves it on exit**  it remembers next session.
- `teach` persists immediately (weights + memory in `learned_memory.json`).
- After it has grown a lot, re-tune its internal scales (optional, no retraining):
  ```python
  from brain import Brain; b = Brain(); b.recalibrate()
  ```

## What it can do once trained (all in one model, all real, none hardcoded)
- **Reason** about general things (commonsense/causal)
- **Generate / understand language**, answer questions
- **Know what it doesn't know** and say "I don't know" instead of fabricating
- **Seek** answers from its memory
- **Learn new facts continuously** without forgetting  and **persist** them
- **Be curious**  act on its own uncertainty
- **Know itself** (name Pragnosia, its nature) and decide every action itself

## Honest expectation
These are small models (the spin-dominant ablation is ~37M; the 1.4B reference is
GPT-2-large class). They will be clearly better than a 20M toy  more reliable reasoning,
more facts, sharper "I don't know"  but still small; they will not match large LLMs.
Strength comes from scale + reasoning data, both of which these runs add as far as the
hardware allows. The spin-dominant design's real payoff is **structural**: long-context
`O(T)` mixing and recurrent `O(1)`/token inference (no KV-cache growth).
