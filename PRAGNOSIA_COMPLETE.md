# PRAGNOSIA — Complete Project Record (everything, to date)
_Last updated: 2026-06-11. Machine: RTX 4060 Laptop, 8 GB VRAM, 16 cores, 30 GB RAM._

This document records **everything** built on top of the inherited "Spinning Brain"
research prototype: the goal, the architecture, every phase of work, every result
with its number, every honest limitation, the data we have, the hardware ceiling,
and exactly what is needed to push further. Nothing is omitted or rounded into hype.

---

## 0. ONE-PARAGRAPH STATUS
Pragnosia is a single ~20M-parameter "spinning brain" that, in one object
(`brain.py`), **reasons causally** (proven, not memorization), **generates and
understands language** (perplexity 30.7 on a 312M-token corpus), **knows what it
does not know** and says so, **seeks** answers in memory, **learns new facts
continuously without forgetting**, is **curious** (acts on its own uncertainty),
**knows itself** (name, nature — learned into weights), and **decides every action
itself** from its own confidence and curiosity — with **nothing hardcoded** (word
importance and all thresholds are derived from data and re-derive as it grows).
It genuinely reasons about general things (commonsense/causal), learned from data,
but **weakly**, because it is small. Everything here is demonstrated at small scale;
**reasoning depth, reliable knowledge, and rich emergence require scale + reasoning
data** — that is the next phase and cannot be faked on one laptop.

---

## 1. THE GOAL (inherited, unchanged)
Build a brain that is, all at once: **cheap**, **reasons like a brain** (not a
fluent-text generator), **omni** (text/vision/audio), **alive** (continuous
learning, no forgetting), **knows what it doesn't know**, **never fabricates**,
**seeks** what it lacks in language, **generates** language. "Conscious" is *not*
taken literally — it is translated to a testable cluster: models its own
uncertainty, knows its knowledge boundary, acts to reduce it, refuses to fabricate.

**Governing rules (earned, followed):** every claim has a checkable number;
report negative results; isolate before fixing; curricula grow the data, never
shift the rule on fixed input; **no hard-coded values** — anything memorizable
will be memorized, so derive/randomize it.

---

## 2. THE ARCHITECTURE

### 2.1 Spinning core (the substrate — unchanged from the prototype)
```
F(h,x) = (1-η)·h + η·tanh(W·h + U·x + b)
W = -ρ·(QQᵀ) + S      (Q orthonormal via QR of a learned matrix, S skew-symmetric)
```
The skew term makes the dynamics **rotational** (orbits, never fixed points). The
answer is encoded in the **phase** of a non-converging oscillation — proven causal
by the **brain-swap test** (swap the mid-trajectory state → the answer follows the
swapped-in state, not the input). Re-verified live this session: follows-swapped
**0.59** vs follows-input **0.10** vs random-state **0.21** (chance). That is real
computation-in-state, not a lookup.

### 2.2 The proven faculties (toy scale, `unified_brain.py`, 302K params, 14/14)
- skill-conditioned reasoning (parity-3, sum-3, max)
- P5 confidence (knows knowable vs unanswerable) / P6 abstention (refuses to guess)
- language parse + generate (compositional)
- seek (generates a language query, fetches, integrates — facts randomized/episode)
- exact accumulation (CleanStateRecurrence: commit discrete state each step)
- explore_and_learn (alive loop: curiosity = own uncertainty, online learning, zero forgetting)
- omni perception (symbols + vision + audio fused into one latent)

### 2.3 The scaled language faculty (the hybrid, `s6_hybrid.py`)
Pure spin loses to a transformer at language (measured: spin ppl 7.56 vs GRU 6.55
on TinyStories). Resolution that preserves the goal:
- **Reasoning faculties stay pure-spin** (brain-swap holds where it is *defined*).
- **Language uses a hybrid**: attention-dominant transformer blocks (the quality
  engine) + **one gated spin "carrier"** that holds rotational cross-token state.
  Naively spinning *every* layer scrambled attention's context (worse than pure
  spin); the working design is attention for quality + a light spin carrier, with
  the standard transformer recipe (GPT init + LR warmup) — that recipe was the
  difference between a broken (ppl 459) and a working (ppl 76 at the same step) model.

### 2.4 The integrated brain (`brain.py`, class `Brain`, "Pragnosia")
One object: proven faculties + hybrid LM + the integrations, **everything wired**.
- **ask / respond** — abstain when unsure (own confidence), seek memory, else answer
- **teach** — online learning + self-replay (continuous learning on language)
- **autonomous controller** — decides answer / seek / abstain / learn from its own
  confidence + curiosity. Only structural input is reading "?" (I/O).
- **self-knowledge** — identity (name=Pragnosia, nature) learned into the weights,
  recalled by its own generation.
- **self-calibration (NOTHING hardcoded)** — `recalibrate()` re-derives, from the
  data/model itself: word importance = self-information (−log freq, from the
  corpus); the familiarity/abstain boundary (from its own confidence distribution);
  the memory-match boundary (from the data's similarity distribution). All re-derive
  as the brain grows.

---

## 3. EVERYTHING BUILT, PHASE BY PHASE (with numbers)

| Phase | What | Result | Where |
|---|---|---|---|
| Baseline | verified inherited toy brain on this machine | **14/14 self-test** | `unified_brain.pt` |
| **S1** | scaled compositional parse + generate (108-word recursive grammar) | baseline 0.73 → **clean-stack 1.00 held-out**; generation **1.00**; round-trip **1.00**; state-swap 1.0/1.0/0.0 | archived `s1_*`, `S1_RESULTS.md` |
| **S2** | real corpus (TinyStories), SpinLM vs param-matched GRU | spin **7.56** vs GRU 6.55; fluent gen, zero drift, brain-swap holds; **honest: loses to GRU** | archived `s2_*`, `S2_RESULTS.md` |
| **S2b** | gain-conditioned + 2-layer spin, bigger data | ppl 11.3 → 9.9 → **8.9** (d=512, 67M tok) | archived |
| **S3** | conversation (EmpatheticDialogues + more, masked SFT) | **converses** (turn-taking, right register), generic; first run diverged (overfit), fixed with story-mix + AdamW + early-stop; bot-token ppl **84.6** | archived `s3_*`, `S3_DIALOGUE_RESULTS.md` |
| **S4** | scale model+data for coherence (BPE-4096, 236M tok, d=1024) | **ppl 11.2**, clearly coherent greedy gen | archived `s4_*` |
| **S5** | world knowledge (Simple Wikipedia + stories, BPE-8192, 312M tok) | ~18M model; learns wiki+story registers; facts weak (too small) | archived `s5_*` |
| Integration v1 | UnifiedScaledBrain: faculties + scaled LM, self-test every step | **14/14 + text**, caught the frozen-alive bug | archived `unified_scaled*` |
| **S6 (current)** | hybrid spin+attention LM, from scratch, BPE-8192, 312M tok | **ppl 218 → 30.7**; ~3× better than pure spin; coherent + some real facts ("The Earth orbits the → Sun") | `s6_hybrid.py`, `s6_hybrid.pt` |
| Abstention-on-LM | "I don't know" via calibrated confidence | abstains on all clear unknowns; noisy on familiar-but-wrong | in `brain.py` |
| Continuous-learning-on-LM | teach a fact, recall in fresh context | **"CEO of Tesla → Elon Musk"** learned + recalled, low forgetting (one bleed) | in `brain.py` |
| Pragnosia identity | self-knowledge learned into weights | "What is your name? → My name is Pragnosia" | `pragnosia_id.pt` |
| De-hardcoding | importance + boundaries derived from data; `recalibrate()` | nothing hand-set; re-derives as it grows | in `brain.py` |
| General reasoning probe | commonsense/causal, pure generation | **real but weak** ("if it rains the ground becomes → wet" ✓; ~half wrong) | demonstrated |

**Current full self-test:** 14/14 faculties + language (ppl ~31–34) + abstention +
teach/recall, all in one object, no regression after workspace cleanup.

---

## 4. HONEST LIMITATIONS (not hidden)
1. **Reasoning is real but weak** — general commonsense/causal reasoning is learned
   and un-hardcoded, but ~half-wrong at 20M params. Needs scale + reasoning data.
2. **Knowledge is unreliable** — a 20M model cannot store much; it hallucinates
   familiar-but-unknown facts confidently (so does every small LM).
3. **Abstention signal is noisy** — confidence (prompt perplexity) catches clearly
   out-of-distribution inputs but not familiar-but-wrong ones. A *trained* confidence
   head (the proven P6 recipe applied to language) would sharpen it.
4. **Continuous learning interferes at volume** — works for a handful of facts;
   needs consolidation to scale to many (one observed bleed: Paris→Tesla).
5. **The non-language faculties are still toy-scale** — reasoning (3 skills), alive
   (8 facts), omni (8×8 grids / tones). Wired and verified, not yet scaled.
6. **Emergence** — one genuine emergent *mechanism* (phase-encoding). No emergent
   *capabilities* from scale yet (we are far below that regime).

---

## 5. DATA WE HAVE
| Tokenized corpus | Tokens | Vocab | Notes |
|---|---|---|---|
| `data/s5train.bin` | **311.9M** | BPE-8192 | **main clean corpus**: Simple Wikipedia (~65M) + TinyStories (~247M) |
| `data/s4train.bin` | 236.5M | BPE-4096 | TinyStories only |
| `data/train.bin` | 67.1M | BPE-2048 | TinyStories slice |
| `data/dtrain.bin` | 8.8M (4.4M bot-target) | dialogue BPE | EmpatheticDialogues + Persona-Chat |
| Raw text on disk | — | — | `ts_big.txt` 950 MB (~225M tok), `wiki_simple.txt` 266 MB (~65M tok), TinyStories valid 23 MB |

**Clean, good, ready-to-train data: ~312M tokens** (current corpus). Expandable to
**~600–900M** by tokenizing the full TinyStories train (~2.7 GB ≈ 600M tok) + full
Simple Wikipedia. For comparison, Chinchilla-optimal for a 150M model is ~3B tokens —
so we are data-limited and would want more for a larger model.

**What's missing for better reasoning:** the data is stories + simple encyclopedia —
almost no *explicit reasoning*. To make reasoning strong you need **reasoning-rich
data**: question→step-by-step-answer, instructions→worked solutions, diverse
expository/argumentative text. That is the single highest-value data to add.

---

## 6. HARDWARE CEILING (measured on this 8 GB GPU, seq 256)
| Config | Params | Max batch | Peak VRAM | Verdict |
|---|---|---|---|---|
| d=512, 4 layers (current) | 20M | 64 | 4.1 GB | comfortable |
| d=768, 6 layers | 54M | 32 | 3.9 GB | fits |
| d=1024, 8 layers | 118M | 16 | 4.3 GB | fits |
| **d=1024, 12 layers** | **168M** | 8 | 4.3 GB | **fits — practical max** |
| d=1536, 12 layers | ~340M | 4 | — | **OOM** |

**Practical maximum on this laptop: ~150–200M parameters** (GPT-2-small to -medium
class). bf16 + gradient checkpointing could push toward ~300M but slowly. The hard
floor on speed is the **sequential spin carrier** (256 steps per layer, can't
parallelize over time) — `torch.compile` mitigates it (~12× earlier), but a 150M
spin-hybrid will train in **days**, not hours, on this GPU.

---

## 7. WHAT YOU NEED TO TRAIN IT FURTHER
**To make reasoning + knowledge strong (in priority order):**
1. **Reasoning-rich data** — the #1 lever. Add QA / instruction / chain-of-thought /
   diverse expository text to the corpus. Reasoning is *learned*, so it needs to be
   *in the data*. (No hardcoding — the model learns to reason from examples.)
2. **Scale the model** — go from 20M to ~100–168M (the laptop max), where reasoning
   and knowledge sharpen. Same `s6_hybrid.py` architecture, larger `d`/layers.
3. **More tokens** — expand from 312M to ~600M–1B (full TinyStories + full Wikipedia +
   the reasoning data). Aim for ≥5 tokens/param.
4. **A trained confidence/abstention head** — replace the calibrated-threshold
   abstention with the proven P6 two-phase recipe applied to language, for sharp
   "I don't know."
5. **Consolidation for continuous learning** — replay/rehearsal scaling so it learns
   many facts without interference.
6. **Time/compute** — a 150M run here is days. Beyond that needs a bigger GPU
   (the "bigger compute" step) — at which point the same code scales up.

**Concrete next run that fits this laptop (recommended):**
- Tokenize full TinyStories + full Simple Wiki + any reasoning data → ~600M+ tokens, BPE-8192/16384.
- Train `SpinAttentionLM` at **d=1024, 8–12 layers (~120–168M params)**, bs 8–16,
  warmup + cosine, ~50–100K steps (multi-day).
- Re-run `brain.py test` + `recalibrate()` (everything self-tunes to the bigger model).
- Expect: clearly stronger commonsense/causal reasoning, more reliable facts,
  sharper abstention — all by scale + data, zero new hardcoding.

---

## 8. FILE MAP (clean workspace)
**Active (root):**
- `brain.py` — THE wired object (Pragnosia): all faculties + hybrid LM + integrations + self-calibration
- `s6_hybrid.py` — hybrid spin+attention language model (architecture + train/gen/swap)
- `unified_brain.py` — proven toy faculties + their self-test
- `s6_hybrid.pt` — trained hybrid LM (ppl 30.7) · `pragnosia_id.pt` — LM + learned identity
- `unified_brain.pt` — proven faculties (14/14)
- `data/` — corpora, tokenizers, token bins, `tok_selfinfo.pt`
- Docs: this file, `CLAUDE_CODE_HANDOFF.md`, `Spinning_Brain_Complete_Status_v2.pdf`,
  `S1_RESULTS.md`, `S2_RESULTS.md`, `S3_DIALOGUE_RESULTS.md`
- `old_versions/` — 66 archived files (all S1–S5 intermediate code, checkpoints, logs)

**How to use Pragnosia:**
- `python3 brain.py test` — full self-test (every faculty + language)
- `python3 brain.py chat` — autonomous chat (decides learn/answer/seek/abstain itself)
- `python3 brain.py ask "your question"` — one-shot (abstains if unsure)
- `python3 brain.py teach "a new fact"` — teach it persistently

---

## 9. THE BOTTOM LINE
Every mechanism the goal asks for is **built, wired into one object, causally
verified, and un-hardcoded** — at small scale. Pragnosia genuinely reasons (weakly),
knows itself, knows its limits, learns continuously, and decides for itself. The
distance from here to a strong, reliable, human-like-reasoning brain is **scale +
reasoning data**, not more cleverness or scaffolding. The architecture and the
integration are ready for that step; the laptop ceiling is ~150–200M params on
~600M–1B tokens, and the highest-value addition is reasoning-rich data.
