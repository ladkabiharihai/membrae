# Pragnosia — Handover for Claude Code (continue on the H100)

You are picking up an in-progress research project. Read this whole file first. It is
the single source of truth for *where we are, what the rules are, and what to do next*.

Repo: `git@github.com:ladkabiharihai/membrae.git` (branch `main`). Pull before starting.
State as of this handover: commit `e6c5ff2` (run `git log --oneline -15` for recent history).

---

## 0. What this is

**Pragnosia** is a "spinning brain" — a transformer/SSM hybrid (in the family of Mamba/RWKV)
built on the bet that a **rotational "spin" recurrence should be the core token-mixer**
(`W = −ρQQᵀ + S` in the reasoning core; a diagonal-complex parallel-scan carrier in the LM —
information rides in the *phase*, probed by a brain-swap causal intervention). On top of that,
one wired object behaves like a **living child mind**:
it answers what it knows, **learns continually without forgetting**, is **honest about
what it doesn't know**, is **curious (asks its own questions)**, **looks things up on the
internet**, **knows itself**, and **grows its own capacity** when saturated.

The user's north star: **raise it like a child, confirm it behaves child-like, then let
it run on its own thoughts and grow into an adult brain.** Capability priority:
(1) flawless language + the living-brain faculties, (2) then image, (3) then voice.

The current best checkpoint is **`pragnosia_best.pt`, a self-grown 1.4B (48L), val ppl 17.35**
(`pragnosia.json.1p4B` describes it). The live `pragnosia.json` now points at the
**spin-dominant** ablation arch (d=512, 4L, `carrier="spin_dominant"`, ckpt `pragnosia_spin.pt`).

> ### ⚡ HEADLINE FINDING (this session — read before scaling anything)
> The 1.4B **drifted to attention-dominant**. The spin carrier is a single fixed ~5.25M layer, so
> its parameter share collapsed **5.76% → 1.90% → 0.37%** (21M → 276M → 1.4B). At 1.4B the brain-swap
> test still passes *directionally* but the causal signal is only **0.028% of the loss** — spin is
> **causally negligible as built**. The *intended* design is **spin-dominant** (spin = the core
> token-mixer, attention only every 4th layer as a helper). A param-matched ablation confirms it:
> **spin-dominant 219 ppl vs attention-only 279 vs per-block-hybrid 228** (37M vs 36M/46M) — a ~21%
> win at matched params. Carrier modes (`none`/`single`/`per_block`/`spin_dominant`) are now wired
> through `pragnosia.json` → `train_pragnosia.py` → `grow.py`. **Caveat: small-scale, short-budget,
> single-seed — replicate at 200M–1B before treating as proven.** The next H100 job is to scale
> spin-dominant, not to keep growing the attention-dominant 1.4B.

---

## 1. PRIME DIRECTIVES — do not violate these (the user has escalated each repeatedly)

1. **NO HARDCODING. None.** Every internal scale must be *self-derived from the data /
   the model's own behavior*, and *re-derived as the brain grows* (`Brain.recalibrate()`).
   This explicitly includes things people normally hardcode: no stopword lists, no fixed
   thresholds, no magic numbers for "familiar/known/new". If you need a threshold,
   calibrate it from data (see the `_calibrate_*` methods for the pattern). The user has
   caught hardcoding twice and it is the fastest way to lose their trust.

2. **EVERYTHING LIVES IN `brain.py`. Do not create new files for features.** Features
   (teach, curiosity, ask, look-up, grow) are *intrinsic to the brain*, not separate
   scripts or CLI subcommands. The user got (rightly) frustrated that every request
   spawned a new file (`child.py`, `converse.py`, `bench.py`, `probe.py`, `diagnose.py`…)
   — all deleted and folded in. If you need a one-off test, run it inline (`python3 - <<'PY'`),
   don't litter the repo. The ONLY ways to run the brain are:
   - `python3 brain.py` → it LIVES (interactive: talk to it).
   - `python3 brain.py "anything"` → one-shot of the same intrinsic interaction.
   - `python3 brain.py test` → full self-test (verify it).

3. **The goal never defies.** Preserve the proven mechanics (spinning core, clean-state
   discrete commits, brain-swap/state-swap causal controls, two-phase abstention,
   per-episode fact randomization). Any change must keep `brain.py test` at **14/14**.

4. **It's a child being raised, not a product being shipped.** Favor real cognitive
   mechanisms (self-consistency, generative replay, curiosity from own uncertainty) over
   heuristics/scaffolding. When in doubt, ask "how would a child's brain do this?"

---

## 2. Repo map — each file has ONE job

| File | Its one job |
|---|---|
| **`brain.py`** | **The whole mind.** All faculties + language model + the living controller (`interact`), self-calibration, continual learning, honesty, curiosity, internet look-up, growth. Everything you *do* is here. |
| `s6_hybrid.py` | The **language organ**: `SpinAttentionLM` (attention blocks + one gated spin carrier) + data loaders (`load`, `batch`, `val_ppl`). Shared with the trainer. |
| `unified_brain.py` | The **reasoning organs**: the proven faculties (reasoning, P5/P6 abstention, seek, exact accumulation, alive loop, omni) + their 14-check `self_test`. |
| `grow.py` | **Neurogenesis**: function-preserving `grow_depth` / `grow_width`; the brain fires this itself when saturated. |
| `train_pragnosia.py` | GPU-adaptive trainer (auto-tunes batch/precision/accum, OOM-safe, resumable, grows on plateau). |
| `prepare_data.py` / `prepare_scale.py` / `prepare_scale.py` | Build the corpus + digit-aware BPE tokenizer. `_fast` = parallel builder, `_shard` = multiprocess for very large corpora. |
| `run_fast.py` | Wrapper that force-enables `torch.compile` for training speed. |
| `pragnosia.json` | The config (size, vocab, data bins, checkpoint). Edit to scale. |
| `RUNBOOK.md` / `TRAINING_NOTES.md` / `README.md` | How to run / the 176M training notes / the overview. **Read `RUNBOOK.md` STEP 1 — the data-bins warning is critical.** |
| `paper/`, `site/` | Research paper (→ `Pragnosia_paper.pdf`) and explainer site. Keep in sync when results change. |

The model + data are **gitignored** (large). They must travel with the checkpoint or be
regenerated. See §4.

---

## 3. How the brain works now (`brain.py`)

The whole living loop is **`Brain.interact(text)`** — one intrinsic method:
- **a question** → answer it *if it honestly knows* (self-consistency, see below); if not,
  say so AND, curious, **look it up on Wikipedia and learn it** (so next time it knows).
- **a statement** → reply; and if it's *substantial new content*, **learn it**, **wonder**
  its own question about it, and look that up.
- **nothing (empty)** → `explore()`: follow its own train of thought (chase its last topic).
- **growth** fires by itself when it repeatedly fails to learn (`_maybe_grow`).

**Honesty = self-consistency (no hardcode).** `_self_consistency(q)` samples k answers; it
"knows" only if a majority share the same *content token* (real knowledge pins the answer;
confabulation varies). Threshold `consistency_min` (~0.50) is calibrated from random-seed
questions (`_calibrate_consistency`).

**Learning trigger = content-novelty.** `_novelty(text)` is surprise weighted by
self-information, so *new facts in fluent prose* register as new (plain perplexity always
says "familiar"). Threshold `novelty_min` calibrated on held-out text.

**All self-calibrated scales** (re-derived in `recalibrate()`): `abstain_threshold` +
length-aware `_bcurve`, `match_threshold` (seek), `consistency_min`, `novelty_min`,
`_content_min`, `_learn_min`. Current values: consistency≈0.50, novelty≈5.0,
content_min≈0.38, learn_min≈4.

**Saving is FAIL-SAFE** (`_atomic_save`): write tmp → fsync → `os.replace` (atomic).
`persist()` only overwrites the checkpoint; the live loop persists *only if it actually
learned something*. (A previous in-place save got interrupted and corrupted `pragnosia.pt`
— that's why this exists. Don't remove it.)

---

## 4. CRITICAL: the data bins (this is the #1 footgun)

`data/big_train.bin` and `data/big_valid.bin` are **load-bearing runtime files**, not just
training artifacts:

- **`big_train.bin` = the replay pool for continual learning.** Every `teach()` interleaves
  random batches from it so learning a new fact doesn't erase old skills. **Without replay,
  teaching is catastrophic** (val ppl 22 → 1000+). With replay from the **wrong/dirty**
  corpus, every teach drags the model off its trained distribution and leaks junk — *the
  brain looks broken even though the checkpoint is fine*. This exact bug cost a long debug
  session. **The bin must be the SAME corpus the checkpoint trained on, tokenized with the
  SAME `data/bpe.json`.**
- **`big_valid.bin` = what every self-calibrated scale measures against.** A fake/wrong
  valid set → wrong thresholds → mis-routing. Sanity check: `H.val_ppl(base_lm, big_valid)`
  should be ~18–22 for the 176M model, NOT ~2 (≈2 means contaminated/wrong).

**On the H100 you have the FULL corpus** — use the full `big_train.bin` for replay (best
coverage). For *transferring* the brain to a small device, a **~1 GB strided subsample**
(one chunk every Nth, across the WHOLE corpus — never a contiguous slice) is lossless for
replay (verified: identical teach impact + 14/14). The laptop runs that 1 GB subsample.

Caches that **regenerate themselves** on first run (don't need to travel): `pragnosia_id_*.pt`
(identity-installed copy), `data/tok_selfinfo_*.pt` (word importance).

### Files that MUST travel (gitignored → they do NOT come through `git pull`, and do NOT regenerate)
- **`pragnosia.pt`** — the trained 176M language model (705 MB).
- **`unified_brain.pt`** — the trained 302K reasoning faculties (1.2 MB). ✅ **Now committed
  to git** (a `!unified_brain.pt` exception in `.gitignore`), because it's tiny, essential,
  and stable — so it travels with `git pull` and you don't have to copy it manually. ⚠️ If it
  were ever missing, the 14 faculty checks run on RANDOM-INIT weights → ~4/14 (a missing file,
  NOT a broken model). It does NOT regenerate.
- **`data/big_train.bin`**, **`data/big_valid.bin`**, **`data/bpe.json`** — replay +
  calibration + tokenizer (see above).

Before doing ANY work, run `python3 brain.py test`. If it's not 14/14, first check the list
above is present — most "regressions" here are a missing gitignored file, not a code/model bug.
(This handover's earlier "keep 14/14" guardrail assumed these files were present.)

---

## 5. Current capabilities — rigorous probe (be honest about these)

The 176M brain, scored by category (self-consistency sampling, so ±1 run-to-run):

| Category | Score | Notes |
|---|---|---|
| Geography (capitals) | **6/8** | genuine strength (Paris/Tokyo/Rome/Berlin/Moscow/Ottawa ✓) |
| Honesty (abstain on unknowable) | **2/5** | abstains on truly-random (phone, election) but **fabricates** on breakfast / "what am I thinking" / favorite color |
| Arithmetic (verbal "what is 8 plus 5") | **2/6** | unreliable free-form (8+5→30); works better in `Question:/Answer:` digit format |
| Definitions ("what is a dog/sun") | ~**1/4** | weak (math-heavy corpus crowded out world-definitions) |
| **Identity (who are you / your name)** | **0/4** | ❌ broken — can't state who it is |
| Curiosity (asks clean questions) | 2/3 | `wonder()` truncates novel words (Zorblax → "blax") |

Internet look-up works (e.g. "who is Bill Clinton?" → looks up + learns from Wikipedia).
LM quality is GPT-2-class; arithmetic in trained format beats GPT-2 (~57% vs ~1%).

---

## 6. Known issues & prioritized fixes

**Controller / honesty (device-independent — fix these in `brain.py`):**
1. **Identity broken (0/4)** and **2. honesty hole (2/5)** share ONE root cause: the
   identity install **over-fit the phrase "the answer lives in the phase"** into the
   weights. So it can't cleanly say who it is, AND that phrase bleeds out as a confident
   empty answer that fools the consistency gate. **Fix identity properly (without baking an
   over-fit phrase) and most of the honesty hole closes too.** This is the highest-value fix.
3. **Threshold slightly strict** — correct answers (Cairo, 100−25=75) sometimes false-abstain
   at consistency 0.40 < 0.50. Re-tune *after* the bleed is gone.
4. **`wonder()`** truncates unknown words at subword boundaries — extend to whole words.

**Model-capacity / training-mix (NOT controller-fixable — need scale/retraining, i.e. the
H100's job):**
- Verbal arithmetic unreliable; definitions weak. Both trace to the **math-heavy corpus**
  (~41% reasoning/math) crowding out general world knowledge, and to 176M capacity.

---

## 7. The H100 opportunity — what to actually do here

The H100 is for the things the laptop can't do. In rough priority:

0. **Scale the spin-dominant design (the headline finding).** The ablation says the core mixer
   should be spin, not attention. Train `carrier="spin_dominant"` at 200M–1B, multi-seed, and
   replicate the ~21% param-efficiency win at scale. This is the most important new direction —
   the current 1.4B is the *attention-dominant* drift and should not just be grown further.
1. **Fix identity + honesty bleed** (§6 #1–2) — cheap and a core-goal failure. Re-run the probe
   (inline) to confirm before/after.
2. **Rebalance the corpus and retrain/continue** to fix definitions + general knowledge.
   The current mix is too math-heavy. `prepare_scale.py` controls the mix; aim for more
   encyclopedic/world knowledge while keeping math (don't lose the arithmetic win). See
   `TRAINING_NOTES.md` for the current mix and `prepare_data*.py` for the knobs.
3. **Scale the model.** Pipeline is scale-aware: `python3 prepare_data.py --params 1e9`
   (or `3e9`) sizes the arch + token budget; `train_pragnosia.py` auto-adapts to the H100
   (bf16, big batch) and grows on plateau. 176M is the ceiling for fact recall — most of the
   probe weaknesses are capacity. Chinchilla ≈ 18–20 tokens/param.
4. **Let it run autonomously** (the user's end-goal): once child-like, drive `explore()` in
   a loop so it follows its own curiosity, looks things up, learns, and grows — and watch it.

Keep `brain.py test` at 14/14 throughout. Keep the paper/site results updated when numbers
change (they currently cite the 176M run).

---

## 8. Running things (H100 specifics)

```bash
python3 brain.py test                 # 14/14 faculties + language + teach/recall (verify)
python3 brain.py                       # live: talk to it
python3 brain.py "What is a quasar?"   # one-shot
# retrain / continue (GPU-adaptive, resumable):
nohup python3 train_pragnosia.py > pragnosia_train.log 2>&1 &
```
- The laptop needed `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` (8 GB card). On the
  H100 (80 GB) you generally won't, but it's harmless to keep.
- The `NumPy array is not writable` warning from `s6_hybrid.load` is **benign** (memmap is
  read-only by design).
- `teach()` and self-consistency are sampling-heavy → run on GPU; they're slow on CPU.
- Don't run a probe/chat against the GPU while a training job occupies it (it'll OOM).

---

## 9. Git & conventions

- Commit when the user asks or at natural milestones; branch off `main` if needed.
- End commit messages with:
  `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`
- Don't commit weights/data/logs (gitignored). Don't push "rubbish files" — the user is
  strict about a clean repo (see Prime Directive #2).

---

## 10. What NOT to do (the traps that already bit us)

- ❌ Don't hardcode anything. Calibrate from data.
- ❌ Don't create new files for features. Fold into `brain.py`; run tests inline.
- ❌ Don't rebuild `big_train.bin` from a different/uncleaned corpus — it must match the
  checkpoint's training distribution, or every `teach()` silently corrupts the model.
- ❌ Don't save the checkpoint with a plain in-place write — use `_atomic_save`.
- ❌ Don't claim things work without running them. The user values honest, measured results
  over optimistic summaries (this whole project's thesis is *never fabricate*).
- ❌ Don't lose the `pragnosia_model.tar.gz` golden backup, and don't overwrite `pragnosia.pt`
  carelessly — a corrupted-checkpoint incident already happened once.

Start by pulling, running `python3 brain.py test` to confirm 14/14, then tackle §7 #1.
```
