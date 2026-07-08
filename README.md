# Pragnosia

**Making a diagonal-complex "spin" recurrence the load-bearing core of a small language model.**

Pragnosia is a transformer/SSM hybrid in which a rotational **spin carrier** -- a diagonal-complex
linear-recurrent unit run as a parallel associative scan -- is the *core token-mixer*, with attention only a
periodic helper (every 4th layer). On top of the model runs a self-calibrating controller (`brain.py`) with
continual learning, an in-weights episodic memory, an autobiographical timeline, working memory, tool-use
agency (act→observe→learn), a theory-of-mind user model, reflection, introspection, self-directed goals, and
function-preserving self-growth. Every *calibration threshold* is derived from the model's own data and re-derived as it grows; the
controller's structural constants (buffer sizes, affect timescales) are documented engineering choices, not tuned thresholds.

📄 Paper: `paper/paper.html` (→ `Pragnosia_paper.pdf`)  ·  🌐 Site: `site/index.html`  ·  🛠 Train: `RUNBOOK.md`

---

## The central finding: placement, not phase

The one result the whole project turns on -- **where you put a recurrence decides whether it does the
computation:**

| | Result |
|---|---|
| Carrier as a **side-channel** after a transformer stack (our first 1.4B) | vestigial -- **0.028% of the loss** |
| Carrier as the **core mixer**, param-matched ablation (~37M) | **219 vs 279 ppl** -- a quality win when spin is the core |
| A **different** recurrence (real, no phase) as the core | also beats attention (3.5×) -- so it's *placement*, not the complex phase |
| **Causal dominance grows with scale** (ablate the carrier → × worse) | **306× (284M) → 1081× (706M) → 1886× (1B)** |

The carrier's causal dominance **strengthens** as the model grows -- it is not a small-scale artifact.

## Progression across grown sizes

| Metric | 284M | 565M | 706M | ~1B |
|---|---|---|---|---|
| Carrier ablation (× worse) | 306 | -- | 1081 | **1886** |
| PIQA | 59.5 | 60.7 | 61.8 | **63.3** |
| ARC-Easy | 34.7 | 34.1 | 35.3 | **37.2** |
| HellaSwag | 27.9 | 29.3 | 29.4 | **29.6** |
| LAMBADA | 19.0 | 21.7 | 24.4 | **25.3** |
| Multi-hop battery | 2/10 | 3/10 | 4/11 | 4/11 |

Long context is realized at scale (perplexity drops with the carry), and 8k generation runs at **constant
O(1)/token** with no KV-cache.

## Honest status

- The spin-dominant *quality* win is shown at small scale (3 seeds); the *causal-dominance* property is shown
  at 284M→1B and **strengthens with scale** (needs no baseline). What we do **not** claim is that the design
  beats a transformer at 1B -- a **matched 1B transformer is outside our compute budget**; the affordable crux
  is a matched `carrier="none"` transformer at 284M, still to run.
- The models are undertrained for their size; multi-hop reasoning is weak. A published transformer of similar
  size (Cerebras-GPT-1.3B) is *also* at chance on ARC-Challenge, which is **consistent with** a scale reading -- 
  but that is a **non-controlled** comparison (different corpus/tokenizer/recipe), reported as illustration only,
  not evidence for the design.
- The honesty signal is functional but does not yet cleanly separate knowledge from confident confabulation.
- We report every measured number, including the ones that don't flatter the design.

## Toward a mind (experimental faculties, honestly labelled)

Built on the controller and validated on the 1B -- each labelled for exactly what it is, never more:

- **Functional emotion** (`appraise`/`feel`): appraisal → valence/arousal/mood → it changes behaviour, all
  derived from its own signals. The *mechanism* of emotion; whether it is *felt* is unknowable for any system,
  so we neither claim nor deny it. (It inherits the confidence miscalibration -- it can feel good about a wrong
  answer.)
- **Grounding** (`search`/`crawl`/`provenance`): web search + page-crawl that records the **source URL**, so
  *"how do you know?"* is answerable -- epistemic grounding for looked-up facts.
- **Global workspace** (`broadcast`/`integration`): the attended state bound across all faculties, with an
  integration metric -- the substrate consciousness could **emerge** from. **Access-level, not experience; we
  build the conditions and measure, we never claim.**
- **Embodiment sandbox** (`world.py`, `experience`): a grid world it perceives, acts in, and learns from by
  consequence -- closing perceive → act → observe → reward → learn (its affect is now grounded in outcomes).
- **Multimodal adapter** (`perception.py`, `perceive_multimodal`): a projection adapter feeds image/audio
  features into the **frozen** LM's token stream (LLaVA-style). Architecture + interface are live; the adapter
  is **untrained** -- making it actually *see* is a later GPU job. We built the socket, not the eye.

## Quick start

```bash
python3 brain.py chat             # talk to the model (inference, light)
python3 brain.py learn            # the full controller: memory, self-directed goals, growth
python3 brain.py "Who are you?"   # one-shot

python3 eval_compare.py ours      # zero-shot benchmarks vs open models, one harness
# training: see RUNBOOK.md  (STEP 2b covers long-context; PRAGNOSIA_SELF= bakes in the persona)
```

The active model config lives in `pragnosia.json`; the checkpoint is gitignored (large). For weight-level
learning on a small GPU use a smaller model (the 1B's backward needs more VRAM; it degrades to episodic-only).

## Repo map

| File | What |
|---|---|
| `s6_hybrid.py` | the LM: `SpinAttentionLM`, the spin carrier + parallel scan, `FastWeightMemory` |
| `brain.py` | the controller: honesty, continual learning, episodic + autobiographical + working memory, tool-use agency, theory-of-mind, reflection, introspection, self-model, goals, growth |
| `train_pragnosia.py` | GPU-adaptive trainer (self-governing growth, long-context modes, identity/persona injection) |
| `grow.py` | function-preserving growth (depth/width) |
| `world.py` | the embodiment sandbox (GridWorld) the brain acts in and learns from |
| `perception.py` | the multimodal perception adapter (frozen-LM, LLaVA-style) + placeholder encoder |
| `eval_compare.py` / `eval_external.py` | zero-shot benchmark harnesses |
| `paper/`, `site/` | the write-up and the explainer site |
| `RUNBOOK.md`, `HANDOVER.md`, `TRAINING_NOTES.md` | how to train, project handover, training notes |

## What Pragnosia is *not*

Not a competitive small LM (it lags well-trained open models on general-English fluency), not conscious, and
not a solved system. It is a measured demonstration that a recurrence placed as the core token-mixer carries
the computation -- and a substrate for building continual, self-aware, goal-directed behaviour on top of it.
