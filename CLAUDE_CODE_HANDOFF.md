# SPINNING BRAIN — Build & Scale Handoff (for Claude Code)

You are inheriting a **proven research prototype**: a novel "spinning" (phase-based)
reasoning substrate with 14/14 capabilities verified on a single unified model.
Your job: **scale it toward a usable model** while preserving what's proven.
This document is self-contained — everything you need is here or in the files listed.

---

## 1. FILES YOU RECEIVE

| File | What it is |
|---|---|
| `unified_brain.py` | THE deliverable. One `UnifiedBrain` class with all capabilities + trainer + self-test. Start here. |
| `unified_brain.pt` | Trained checkpoint of the unified brain (302,383 params, 14/14 self-test PASS). |
| `train_unified.py` | Phase-split trainer (checkpointed; phases a/b/c) used to train the .pt. |
| `spinning_brain.py` | The per-capability test suite (10 independent tests, 10/10 PASS) — use as regression reference. |
| `stage5_langseek.py`, `stage6_explore.py`, `scale_language.py` | Standalone verified scripts for seeking, the alive-loop, and the scale-strain measurement. |
| `Spinning_Brain_Complete_Status_v2.pdf` | Full status doc: goals, results, honest caveats, method history. |

Run `python unified_brain.py` to retrain from scratch + self-test, or load the .pt
and call `self_test(brain)`.

---

## 2. THE ARCHITECTURE (do not break these invariants)

### Core dynamics
```
F(h, x) = (1 - eta) * h + eta * tanh(W h + U x + b)
W = -rho * (Q Q^T) + S        # Q orthonormal (QR of learned A), S skew-symmetric
```
- The skew part makes dynamics **rotational** (orbits, never fixed points).
- **Proven property:** the trained model reasons by *spinning, not settling* —
  the answer is encoded in the **phase** of a non-converging oscillation.
  Causally validated by the **brain-swap test**: swap the mid-trajectory state
  from a different-answer input → answer flips 100%; random-state swap → chance.
- **Invariant to preserve when scaling:** keep the brain-swap test as a regression
  check. If a scaled model passes accuracy but fails brain-swap, it has degraded
  to memorization — that is a regression even if accuracy looks fine.

### Skill-conditioned spinning (multi-task without interference)
Per-skill `gain` and `bias` vectors modulate the update:
`h' = (1-eta)h + eta * (gain[sk] * tanh(... + bias[sk]))`.
**Why it exists:** naive shared core+readout caused easy skills to starve hard
ones (proven failure). This is the fix that consolidated 3 diverse skills in one
core. When adding skills at scale, extend `nskill` — do NOT revert to a plain
shared core.

### Clean-state recurrence (the ceiling fix) — `CleanStateRecurrence`
The bare spin **cannot** learn many-wrap exact arithmetic (e.g. running sum mod 5
over ≥4 terms). Proven root cause: **state drift** carrying a discrete count in
continuous state (the atomic op (r+d)%5 learns at 100%; chaining it in continuous
state fails). Fix: after each step, **commit** to the discrete argmax (straight-
through estimator) and feed it back. Verified: exact accumulation at lengths 8
AND 16 (scales), ~1K params. **Rule:** any task needing exact counting/state-
tracking routes through this module; do not try to make the bare spin do it.

### Two-phase training for abstention (P6) — ORDER IS MANDATORY
Teach the **skill first**, THEN the abstention incentive (payoff: correct +1,
wrong −4, abstain −0.2). Co-training from scratch provably collapses to
always-abstain (the model is wrong at init, so abstaining is the safe move
from step 1 and it never learns the skill). `train_unified.py` phase "a"
encodes this order.

### Non-memorizable facts for seeking (P7/Stage 5)
Facts in the seek-world are **randomized every episode**. With fixed facts the
model memorizes them into weights and "retrieval" becomes decoration (proven:
95% accuracy with NO retrieval when facts were fixed). Keep randomization in any
scaled retrieval setup; the control "answer with random fetched values must
collapse" is the regression test.

---

## 3. WHAT IS PROVEN (your regression baseline)

One UnifiedBrain (302K params, ~1.2 MB), 14/14:

| Capability | Measured | Threshold |
|---|---|---|
| Reasoning ×3 skills (skill-conditioned) | 0.90–0.93 | >0.9 |
| P5 confidence gap (knowable vs unanswerable) | 0.34 | >0.3 |
| P6 abstains: known / unknowable | 0.19 / 1.00 | <0.25 / >0.6 |
| Language parse, held-out novel | 1.00 | >0.8 |
| Language generate, held-out novel, free-run | 1.00 | >0.8 |
| Seek: generated query language exact | 1.00 | >0.9 |
| Seek: answer with loop / random-value ctrl | 1.00 / 0.14 | >0.85 / <0.45 |
| Exact accumulation L=16 | 1.00 | >0.9 |
| Alive loop: final / retention gap / probes | 1.00 / 0.00 / 9 | >0.95 / <0.15 / < random(≈20) |

Note: the 3 reasoning skills run 0.90–0.93 in the unified brain vs 1.00 when
trained alone — mild cross-faculty interference. Acceptable now; watch it at scale.

---

## 4. KNOWN LIMITS (honest; design around them)

1. **Scaled compositional generalization is PARTIAL.** At 35-word vocab + recursive
   grammar (depth 1–3, ≤11 tokens): train whole-parse 1.00, **held-out 0.65**
   (slots 0.85–1.00, climbing slowly with budget). First real train/held gap of
   the project. This is your **primary scaling problem**.
2. **Relational vision partial** (~0.76 on "is A above B"); localization and
   aggregation are 1.00. Theme: direct-readout-strong, multi-element-comparison-weak.
3. **Bare-spin exact arithmetic** — permanent; route through CleanStateRecurrence.
4. Everything is **toy scale** (synthetic tones, 8×8 grids, ≤35-word grammars).
5. Sequential per-token Python loops are slow on CPU; the QR per forward is fine
   (0.18s/500 at 96×96) but you should batch/vectorize the token loop when scaling.

---

## 5. SCALING ROADMAP (priority order, with acceptance criteria)

### Phase S1 — crack scaled compositional generalization  [HIGHEST PRIORITY]
The 0.65 held-out is the gate to everything else.
- Sweep: width D ∈ {256, 512}, T_inner ∈ {2,4}, EMB ∈ {64,128}, training budget ≥50K steps, data ≥20K unique sentences.
- Try: curriculum over recursion depth (1→2→3, *growing data*, never shifting rule on fixed input — a shifting-rule curriculum provably fails).
- Try: the clean-state idea applied to parsing — commit intermediate parse decisions discretely (a "clean parse stack").
- **Accept when:** held-out whole-parse >0.95 at vocab ≥100, depth ≥3. **Regression:** tiny-grammar held-out stays 1.00.

### Phase S2 — real tokenization & corpus language
- Move from hand-built grammars to a small real corpus (e.g. TinyStories-class), BPE/char tokenizer, vocab 1–8K.
- Architecture: keep `SpinSeqReader` shape (embed → spin-per-token → carry state); vectorize the inner loop; consider GPU.
- **Accept when:** next-token perplexity competitive with a parameter-matched GRU/LSTM baseline on the same data, AND the brain-swap causal test still passes on probe tasks.

### Phase S3 — scale generation
- Extend `generate_language` to open-ended next-token generation on the S2 corpus.
- Keep free-running evaluation and per-position drift tracking (drift was zero at toy scale; verify it stays bounded).
- **Accept when:** ≥100-token free-running generation with bounded drift and grammaticality by held-out perplexity.

### Phase S4 — scale seeking (RAG-style, language-mediated)
- Replace the toy store with a real retrieval index (e.g. embedding search over documents); the brain still **generates the query in language** (this is the proven mechanism — keep it).
- Keep: per-episode fact randomization in training tasks + the random-value collapse control.
- **Accept when:** answer accuracy with retrieval ≫ without on held-out questions whose answers are only in the store.

### Phase S5 — scale the alive loop
- Bigger worlds (≥1000 facts), uncertainty-driven exploration, online learning with self-replay buffer (the proven recipe), measure forgetting curves over long horizons.
- **Accept when:** retention gap <0.05 after 10× the facts, curiosity ≥1.5× more sample-efficient than random.

### Phase S6 — integration & product hardening
- One scaled UnifiedBrain passing the full self-test at scale; export ONNX/TorchScript; simple chat-style interface wiring parse→(seek if low-confidence)→reason→generate, with P6 abstention surfaced as "I don't know" to the user.

### Cross-cutting regression suite (run after every change)
1. `spinning_brain.py` full suite (10/10 must hold at toy scale).
2. Brain-swap causal test on at least one reasoning task (anti-memorization).
3. Seek random-value collapse control.
4. First-vs-last retention gap in the alive loop.
5. P6: abstain-on-unknowable ≥0.6 with needless-abstain ≤0.25.

---

## 6. METHOD RULES (these were earned the hard way — follow them)

1. **Isolate before fixing.** If a metric is constant across multiple different
   interventions, you are solving the wrong problem. (This caught: an unlearnable
   task misdiagnosed as 4 consolidation failures; a dataset infinite-loop; an
   off-by-one where 0.00-on-training revealed a bug, not a learning failure.)
2. **Skill first, then abstention.** Never co-train from scratch.
3. **Curricula grow the input/data; never shift the rule on fixed input.**
4. **Every claim gets a checkable number; every control gets run.** A capability
   without its collapse-control (blank a modality, randomize the facts, random-
   state swap) is not proven.
5. **No hard-coded values.** Anything memorizable will be memorized; randomize it.
6. **Report negative results.** The project's value includes its falsified
   hypotheses (spinning does NOT resist forgetting; capacity did NOT fix
   consolidation; gating/periodic-readout/cyclic-phase did NOT fix the ceiling).

---

## 7. GOAL AUDIT (so you know what "done" means)

Original goal: cheap ✅(1.2 MB total) · reasons like a brain ✅(causally proven) ·
omni ✅(3 modalities + fusion at toy scale) · alive ✅(curiosity + online learning,
zero forgetting) · knows-what-it-doesn't-know ✅ · never fabricates ✅(abstains) ·
seeks in language ✅ · generates language ✅.

**All mechanisms proven. NOT yet achieved: scale.** The model is usable as a
research substrate today; it is not yet a usable *product* model. Phases S1–S6
above are the distance between those two. The single hardest open problem is S1
(scaled compositional generalization, currently 0.65) — start there.
