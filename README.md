---
license: apache-2.0
language:
- en
library_name: pytorch
pipeline_tag: text-generation
tags:
- language-model
- linear-attention
- recurrent
- associative-recall
- intrinsic-memory
- research
---

# Pragnosia — an intrinsic brain on a fast recall core

**A small, compute-efficient recurrent model whose brain faculties — memory, forget-free online
learning, agency, and constant-memory streaming — are *intrinsic to the forward pass*, not bolted on
as external code, and validated at small scale.**

This repository documents a research pivot. It began as a **spin-dominant** language model (a
diagonal-complex "spin" recurrence as the core token-mixer) with faculties provided by an external
Python controller (`brain.py`). Honest evaluation retired both of those choices:

- **Spin was ruled out as the substrate.** The diagonal-complex carrier is fast but **cannot do
  associative recall** (MQAR 26%→2% vs attention ~100%), so it cannot support memory or
  online-learning. See `RESULTS_MEASURED.md` #30.
- **The faculties moved *into* the model.** They are no longer `self.store`/`teach()` scaffolding in
  `brain.py`; they are properties of the architecture and the forward pass.

The pre-pivot spin work is preserved, unmodified, in **`v1_spin/`** (and the papers) — it stands as an
honest negative result: *at matched parameters, a spin-dominant recurrence is a statistical **tie**
with attention, not a win.*

---

## The core: `fastcore_v.py` — `VChunkRecall`

A vectorized, chunk-parallel **linear-attention core with a 2nd-order Taylor feature map** (softmax-like
sharpness → associative recall). Pure PyTorch, no custom kernels, no external deps.

| property | measured |
|---|---|
| associative recall (MQAR, 16 pairs) | **100%** |
| speed vs flash-attention @64K ctx | **~13–17×** (O(T) vs O(T²); gap grows with length) |
| speed vs the spin carrier @32K | **~5×** |
| memory of a fact | the fast-weight state `S = Σ φ(k)ᵀv`; recall is `o = φ(q)·S` |

The memory **is** the state; **writing the state is learning** — both happen in the forward pass.

## The four faculties — in the model, proven small (`RESULTS_MEASURED.md` #31–#37)

| faculty | result | how it's intrinsic |
|---|---|---|
| **online learning, no forgetting** | new facts kept, prior skill **100%** (weight-teaching → 0%) | knowledge enters the fast-weight state; weights untouched |
| **agency (learning from consequence)** | **85%** exploit-after-discovery (chance 25%) | the state adapts to reward, zero weight updates |
| **O(1) streaming** | constant **73 MB** to 100K tokens; recall to trained length | recurrent state, no growing KV cache |
| **per-FLOP efficiency** | **~13–17×** attention at long context | linear-time core |

A single **2.4M-param unified model** (`unified_brain.py`) does all four at once, and **grows itself**
(`.grow()`, function-preserving) — proven at ≤100M, the standing rule of this project.

## Self-scaling (`scale_train.py`, `self_grow.py`, `RESEARCH_selfgrowth.md`)

Growth is a developmental controller with an **intrinsic trigger**: the model grows when *its own
predictive entropy* (label-free) saturates — not when a supervised loss plateaus. The *allocation* of
new parameters is irreducibly external (a fixed weight set cannot `malloc` new weights from a forward
pass); the *decision* is the model's own.

The genuinely scaffolding-free capacity axis is **adaptive compute**: chain-of-thought with intrinsic
halting solves multi-step reasoning the parallel forward can't (13% → **92%**, with compute-steps that
track difficulty exactly). It belongs in a reasoning fine-tune, not pretraining (PonderNet does not
pretrain stably — `ponder_smoke.py`).

## Honest gaps (not yet closed)

- **Language transfer.** The faculties are proven on synthetic tokens. In-context recall does **not**
  yet transfer to natural prose at ≤45M — it's a scale threshold, not a mechanism failure. A from-scratch
  language run (`scale_train.py`) is the open experiment.
- **Real scale.** The core has not been trained to a useful size; beside co-resident production the
  VRAM cap limits growth.
- **Fused kernel.** The vectorized core is fast, but a Triton kernel would extend the efficiency win to
  short context.

## Layout

```
fastcore_v.py        the fast recall core (VChunkRecall)     ← the substrate
gated_core.py        + per-token gate (forgetting)
unified_brain.py     one model, all four faculties + growth
self_grow.py         intrinsic (own-signal) self-growth
scale_train.py       from-scratch self-scaling language run
onforget_demo.py     forget-free online learning demo
prove_*.py           per-faculty proofs (agency, streaming, efficiency)
chain_cot.py         adaptive-compute / CoT self-growth
bench3.py            3-way speed: ours vs attention vs spin
RESEARCH_selfgrowth.md   scaffolding-free self-growth study
RESULTS_MEASURED.md  the durable, honest lab record (every number)
v1_spin/             the pre-pivot spin-dominant model + trainer + controller
paper/, manuscript*/ the spin-era papers (historical; a brain paper is TBD)
```

Every claim here is backed by a numbered entry in `RESULTS_MEASURED.md`, including the negatives.
