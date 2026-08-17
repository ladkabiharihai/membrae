# Research program: a cheaper, brain-like, *growing* model (faculties in the weights, not scaffolding)

## Thesis
Not a product, not an LLM to out-scale. A research question: **can a small, compute-efficient model be brain-like —
its faculties intrinsic to the forward pass (not external Python scaffolding), and can it GROW developmentally
(start small/cheap, grow into capacity) rather than being trained big from scratch?** Small scale is the point: we
prove *principles*, not benchmark scores. A 216M model useless for coding is irrelevant — the deliverable is the
method and the evidence.

## Testable claims (brain-like must become numbers)
| Brain property | Testable claim | Status |
|---|---|---|
| Faculties intrinsic | recall / online-learning-no-forget / agency / streaming emerge from the forward pass, no `brain.py` scaffolding | ✅ #31–#37 |
| Efficient | O(1)-memory streaming, linear-time core, per-FLOP win vs attention | ✅ #36 |
| Faculties survive growth | recall stays ~100% (MQAR) as the model grows (function-preserving) | ✅ #49/#50 (to d=512) |
| Growth is intrinsic | model triggers its OWN growth from its OWN predictive entropy (label-free), no external schedule | ✅ #38 + scale_train.maybe_grow |
| **Growth is CHEAPER** | **grow-small-then-big reaches a target loss for LESS total compute than from-scratch** | ❓ **testing now (grow_efficiency.py)** |
| Growth adds capability | a grown bigger model is more capable than the small one | ❌ negative so far (#48/#56): growth ≠ free capability |

## The crux (what actually decides the thesis)
Earlier we wrongly hoped **growth = free capability** and got 4 negatives — correctly, because a bigger net still
needs training. The RIGHT brain-like claim is **growth = efficiency**: a brain learns cheaply while small, then grows.
Metric: **loss vs cumulative COMPUTE (params × tokens), not loss vs steps.** If early cheap-small steps let a grown
model hit a target loss for fewer FLOPs than from-scratch, growth is genuinely brain-like efficient. This is the
experiment running now (from-scratch 6L vs staged 2L→4L→6L, compute-to-target).

## Prior art to build on (grounding, not reinventing)
- **Net2Net** (Chen 2016): function-preserving width/depth growth (our depth+feat operators are this family).
- **Gradual stacking / bert2BERT / LiGO** (Gong 2019 / Chen 2022 / Wang 2023): growing a model *warm-starts* a bigger
  one and cuts pretraining compute 20–50% — the exact "growth is cheaper" claim, shown at real scale. Our contribution
  = doing it on an *efficient recurrent brain core with intrinsic faculties*, and letting the model self-trigger growth.
- **Developmental / neurogenesis analogies**: capacity added where/when the system saturates — our entropy trigger.

## Roadmap (small-scale, decisive, cheap)
1. **Growth EFFICIENCY** (compute-to-target, grow vs scratch) — RUNNING. If cheaper → thesis validated; if not → why.
2. **Growth OPERATOR** — identity-block grow (current) vs STACKING (copy trained layers, known better warm-start) vs
   LiGO-style learned init. Which reaches target for least compute?
3. **Growth SCHEDULE** — how much/when to grow (the intrinsic entropy trigger + a converge-before-grow gate).
4. **Faculty preservation through growth** — recall/online-learning/agency intact across every grow (function-preserving).
5. **Width axis** — needs a function-preserving width-grow (LayerNorm-on-residual subtlety, cf. GroupFeatNorm #50).

## Results (roadmap #1-4 DONE, loss-vs-compute, small-budget so orderings are the robust claim)
- **#1 growth is CHEAPER** (#61): grow 2L→4L→6L reaches target for **17% less compute** than from-scratch (identity op).
- **#2 operator** (#62): stacking (copy trained layers) > identity at equal schedule (22% vs 17%).
- **#3 schedule** (#65): EARLY-heavy (front-load cheap small-size learning) is best: **36% cheaper**; late worst (17%)
  → developmental timing confirmed. Schedule is the DOMINANT compute lever.
- **#4 capstone operator** (#66): at the early schedule all operators tie on compute-to-target (~34%); the operator
  choice governs FUNCTION-PRESERVATION + final quality. **HYBRID 'zero-gated copy'** (copy trained block × learnable
  α init 0 = ReZero/LayerScale-for-growth) is the winner: function-preserving (grow-jump |Δloss|=0, faculties intact
  every step, like identity) AND warm-started (better final loss than identity, near stacking).

## THE RECIPE (quantified)
**cheaper brain-like growth = EARLY-heavy schedule (~34-36% less compute) + HYBRID operator (function-preserving,
faculties intact every step, warm).** On an efficient recurrent brain core (VChunkRecall), faculties intrinsic,
growth self-triggered by own entropy. Novel combination; in the LiGO/bert2BERT 20-50% band but self-growing + on a
brain core, provable at small scale.

## Honest limits + next
- Small budget (4500 steps, d=384): RELATIVE orderings robust, absolute % indicative. A larger-budget replication
  would firm the numbers (real compute).
- Depth axis only for growth here; WIDTH-grow (function-preserving, LayerNorm-on-residual subtlety) untested = open.
- Growth ≠ free capability stays true; growth = EFFICIENCY is the validated claim.
- Consolidate #61/#62/#65/#66 + the faculty proofs into a writeup = the deliverable.
