# Pragnosia — a spinning neural substrate

A compact "spinning brain" built on the opposite bet from large language models.
Its core uses **rotational** recurrent dynamics (`W = −ρQQᵀ + S`), so representations
*orbit* rather than settle, and an answer is encoded in the **phase** of a
non-converging oscillation — verified causally by a brain-swap test. On this
substrate, eight goal faculties are integrated into one model that **reasons**,
**knows what it doesn't know** (abstains), **seeks**, **learns continually without
forgetting** (surprise-modulated plasticity), **knows itself**, and **decides its
own actions** — with **nothing hardcoded** (every internal scale is derived from
data and re-derived as the brain grows).

> Honest scope: at ≤176M parameters this is GPT-2 class — it reasons and recalls
> *weakly* relative to frontier models. The point is the properties a frozen model
> lacks (alive, honest, causal), demonstrated and measured. See the paper.

## What's here
| File | What it is |
|---|---|
| `brain.py` | **The single wired object** (Pragnosia): all faculties + the hybrid language model + autonomous controller, self-calibration, continual learning. CLI: `test` / `chat` / `ask` / `teach`. |
| `s6_hybrid.py` | The spin–attention hybrid language model (architecture + train/gen/brain-swap). |
| `unified_brain.py` | The proven toy faculties (reasoning, P5/P6 abstention, seek, exact accumulation, alive loop, omni) + their 14-check self-test. |
| `train_pragnosia.py` | GPU-adaptive trainer — auto-tunes batch / precision / accumulation to the card, OOM-safe, resumable. |
| `prepare_data.py` | Builds the rebalanced corpus (knowledge + reasoning + math + code + chat + grammar, stories ≤10%) and BPE tokenizer. |
| `pragnosia.json` | Model config (size, vocab, data, checkpoint). Edit to scale the brain. |
| `paper/` | The research paper (`paper.html`, `make_figures.py`, figures) → `Pragnosia_paper.pdf`. |
| `site/` | Animated explainer site (`index.html`) + honest LLM comparison (`compare.html`). |
| `*_RESULTS.md`, `PRAGNOSIA_COMPLETE.md`, `RUNBOOK.md` | Phase results, the complete record, and how to run. |

Trained weights, the corpus, and logs are **not** committed (they're large and
regenerable). Reproduce them with `prepare_data.py` then `train_pragnosia.py`.

## Quick start
```bash
python3 prepare_data.py --laptop        # small ~200M-token corpus for a laptop
python3 train_pragnosia.py              # train (GPU-adaptive, OOM-safe, resumable: --resume)
python3 brain.py test                   # full self-test: every faculty + language
python3 brain.py chat                   # talk to it — it decides answer/seek/abstain/learn itself
python3 brain.py teach "a new fact"     # teach it persistently (it grows by itself)
```

## Scaling (one knob)
The pipeline is scale-aware. Tell `prepare_data.py` the target model size; it sizes
the architecture (writes `pragnosia.json`), computes a ~18-tokens/param budget, and
streams the data from FineWeb-Edu (on top of the curated reasoning/math/code/chat/
grammar sets), stories capped at 10%:
```bash
python3 prepare_data.py --params 176e6  #  ~176M model,  ~3.2B-token budget
python3 prepare_data.py --params 1e9    #  ~1.1B model,  ~18B-token budget
python3 prepare_data.py --params 3e9    #  ~2.6B model,  ~54B-token budget
```
`train_pragnosia.py` then auto-adapts batch / precision / accumulation to whatever
GPU it runs on (OOM-safe), so the same commands train a 176M model on a laptop or a
multi-billion-parameter model on an H100. Resume on any card with `--resume`.

## Architecture
Reasoning faculties stay **pure spin** (the brain-swap invariant holds where it's
defined); **language** uses attention for fluency plus a single gated spin *carrier*
for rotational cross-token memory. One autonomous controller routes every input from
the model's own confidence and curiosity. See `paper/` and `PRAGNOSIA_COMPLETE.md`.
