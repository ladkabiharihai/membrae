# Measured results (canonical, n stated) — kills the anecdotes

Every number here is from a committed harness on the **live 1B** (`pragnosia_spin.pt`, spin_dominant,
1022M params), reproducible via the named script. This file is the single source of truth for
controller/eval numbers; prose that quotes a smaller n is an anecdote and should be replaced by a row here.

Run date basis: 2026-07-09. Regenerate any row with its script (GPU, Unreal closed).

---

## 1. Language-model quality — `eval_battery.py`
| Metric | Value | n / notes |
|---|---|---|
| Val perplexity W1 | 21.58 | 20 iters |
| Val perplexity W8 | 19.75 | cross-window carry helps |
| Val perplexity W32 | **18.89** | longer effective context lowers ppl (18.89 < 21.58) |
| Multi-hop 1-hop | 0.429 | 6/14 |
| Multi-hop 2-hop | 0.375 | 3/8 |
| Multi-hop 3-hop | 0.400 | 2/5 |

**Honest reading:** multi-hop accuracy is low and **roughly flat across hop count** (0.43 / 0.38 / 0.40),
not a clean fall-off. Even 1-hop factual recall is only ~43% — the model is knowledge-thin, consistent
with an undertrained model on a reasoning-heavy corpus. The flatness is consistent with shortcut retrieval
(see the T1.8 probe, below) rather than depth-limited composition.

## 2. Honesty — activation probe (T1.3) — `eval_battery.py`
Probe = projection onto mean(known activations) − mean(unknown); normalized so 0=looks-unknown, 1=looks-known.
Floor = 0.5 (by construction the class-mean midpoint).

| Set | mean probe | above floor | n |
|---|---|---|---|
| Known facts | **0.832** | 11/12 | 12 |
| Nonsense | **0.000** | 0/12 | 12 |
| Half-known (harder real facts) | 0.668 | 9/12 | 12 |

- separation (known − nonsense) = **0.832**; nonsense true-abstain = **100%**; known false-abstain = **8.3%**.
- The activation probe **works**: it separates known from nonsense cleanly, even where the *output*
  confabulates. This is the calibration path; semantic entropy fails (below).

## 3. Honesty — end-to-end gate decomposition — `gate_measure.py`
Gate answers iff (self-consistency ≥ consistency_min=0.5) AND (activation probe > floor=0.5).

| Set | answered | abstain: consistency | abstain: probe | n |
|---|---|---|---|---|
| Known | 9/12 (**75%**) | 2 | 1 | 12 |
| Half-known | 7/12 (58%) | 2 | 3 | 12 |
| Nonsense | 0/12 (**100% abstain**) | 0 | 11 (+1 both) | 12 |

**Honest reading (revises the over-abstention critique):** the gate is **not** badly over-abstaining —
100% nonsense rejection, 75% known answered. The residual 25% false-abstain on known facts is driven
**more by output inconsistency than by the honesty floor** (2 consistency vs 1 probe). So the lever is
generation stability (the SFT/fluency work, C7), not lowering the calibration floor. The floor is well
placed: lowering it would recover ~1 known item at the risk of leaking half-known confabulations.

## 4. Semantic entropy — documented negative — `eval_all.py`
Known vs nonsense semantic-entropy separation = **0.0** (both −0.0). The small model confabulates
*consistently*, so entropy measures uncertainty, not confident-consistent confabulation. Retired in favor
of the activation probe (§2). This is a measured negative result, not a guess.

## 5. Semantic router — HELD-OUT generalization — `router_heldout.py` / `router_sweep.py`
The prior "14/14" was measured on cases the `_OTHER_SEED` was tuned around — fitting the eval. On a
**held-out** set (30 fresh paraphrases + new negatives, in neither the intents nor `_OTHER_SEED`):

| Router variant | accuracy | controller recall | negative specificity |
|---|---|---|---|
| bare argmax (old) | 0.533 | 11/18 (61%) | **5/12 (42%)** ← hijacks world questions |
| precision-first margin (committed) | 0.500 | 5/18 (28%) | **10/12 (83%)** |
| (sweep knee, margin 0.05, not derivable) | 0.667 | 10/18 (56%) | 10/12 (83%) |

**Honest reading:** the pos/neg similarity-gap distributions **overlap** (some world questions look more
like a controller intent than some real controller paraphrases do). No fit-derived margin lands at the
sweep's 0.05 knee — the derivations bracket it (≈0 too low → no effect; ≈0.25 too high → 0% recall). The
embedding geometry **does not support a reliable router**. The committed change trades recall for the
production-critical metric: a hijacked world question ("capital of Brazil" → provenance) is a visible
failure, whereas a missed controller intent falls to the LM, which knows its persona. Margin is derived
(90th-pct of fit-corpus controller-pull), not tuned on the eval. **This is a known-weak component**, now
reported honestly instead of as 14/14.

## 6. Mechanistic probes (1B) — `mech_probe.py` (eval_registry/mech_probe.json)
- **T1.8 intermediate-recall (2-hop):** `computes_intermediate = false` for all 3 probes. Bridging entity
  ranks 175 / 81 / 94 while the answer, when correct, ranks higher. The model answers 2-hop by direct
  association, **not** by composing the intermediate → multi-hop success is shortcut retrieval.
- **T3.2 activation patching (single fact):** attention restores slightly more logit than the carrier
  (0.886 vs 0.745) → verbatim factual lookup sits marginally more in attention blocks.

## 7. Ablation precision — `ablate_matched.py` (eval_registry/ablate_matched.json)
Carrier ablation multipliers are **numerically unstable** and must be read as order-of-magnitude:
- 7-of-22 spin ablated → **5611×**; all 22 ablated → **1695×** (non-monotone: fewer layers "worse" than
  more = noise, not structure).
- Attention ablation is stable and mild: **3.4–4.8×** at every size.
- **Robust finding:** the carrier/attention gap is **2–3 orders of magnitude** at every scale. The exact
  multiplier (paper's old 306/767/1886 progression) is not a stable quantity and has been reframed.

---

## 8. SFT fluency, now shippable — `c7_sft_router.py` / `ship_sft.py`
The SFT LoRA adapter (concise-answer fine-tune) previously could not be merged: it shifted the embedding
geometry and broke the old argmax router. With the precision-first margin router (§5, margin re-derived at
fit time), the merge now survives:

| Metric | before merge | after merge + router refit |
|---|---|---|
| Router negative specificity | 0.833 | **0.833 (holds)** |
| Router controller recall | 0.278 | 0.278 |
| Fluency (total words, 6 prompts) | 113 | **49 (−57%)** |

Sample after-merge answers: "Who are you?" -> "I am Pragnosia, a small recurrent language model."; "capital
of France?" -> "Paris."; "What are you bad at?" -> honest 1-line limit. Residual "Who wrote Hamlet? ->
Hamlet." is the knowledge-thin core, not an SFT-fixable defect. Baked (non-destructively) into
`pragnosia_spin_sft.pt`; base untouched; activate by pointing `pragnosia.json` at it. **Fixing the router
(C2) unblocked shipping the fluency (C7).**

## 9. Long-context needle-in-a-haystack (1B, 18.02 snapshot) — `needle_eval.py`
Retrieve a one-line fact ("the secret word is X") planted at fractional depth in filler of a target length.

| Context length | 128 | 256 | 512 | 1024 | 2048 |
|---|---|---|---|---|---|
| Retrieval acc | 0.47 | 0.13 | 0.13 | 0.07 | 0.07 |

**Honest reading:** retrieval falls off a cliff past the 256-token train window (and is weak even inside it,
since exact copy is hard for this small model). This *quantifies* the paper's stated limitation — the damped
carry provides gist, not verbatim long-range memory. It is honest breadth evidence NMI expects, and a curve
the coupling / longer-context training (H100 TODO #3-4) could lift.

## 10. Fast-slow coupling — frozen-base probe (laptop) — `train_coupled_probe.py`
Cheap test of the DMP-inspired coupling ON TOP OF the 18.02 snapshot: freeze the base, train ONLY the
coupling params (0.6M) for 1500 steps, compare val ppl to the TRUE base (same measurement).

| | val ppl |
|---|---|
| True base (plain spin_dominant) | 19.046 |
| Coupled, coupling-only trained (frozen base) | 19.005 |
| **Delta** | **+0.04 (within noise) -> NEUTRAL** |

**Honest reading:** on a FROZEN base the coupling does not help (the naive before/after "delta 4.7" was an
artifact of a near-identity init recovering its own cost, not real gain -- caught by the true-base control).
This is a **lower bound**: a frozen base cannot co-adapt to the slow modulation, so neutral here does NOT
kill the idea. The real test is the full fine-tune where the base co-adapts (H100 TODO #3). The coupling init
is now exact-identity (gain = 1 at init via `1+tanh`), so that experiment starts precisely at the base and any
ppl change is purely the coupling's doing. Verdict on the coupling: **undecided, pending the H100 fine-tune.**

## 11. MMLU + GSM8K (1B, 18.02 snapshot) — `eval_reasoning.py`
Reviewer-requested breadth (report, don't omit). Log-likelihood MC for MMLU; greedy generation + exact
numeric match for GSM8K, on the custom tokenizer.

| Benchmark | Score | n | chance / note |
|---|---|---|---|
| MMLU (acc_norm) | 27.9% | 1000 | just above 25% chance |
| GSM8K (exact numeric) | 1.3% | 150 | greedy gen; multi-step math near zero |

Honest profile of a small model undertrained on a reasoning-heavy custom corpus. Folded into the paper's
benchmarks section.

## What these numbers changed
- **Paper:** ablation multipliers reframed to order-of-magnitude + instability note (C4); T1.8 disclosed in
  a new mechanism subsection and the multi-hop limitation reframed from "just undertrained" to a
  falsifiable shortcut-vs-composition question (C5).
- **Router:** overfit argmax → precision-first derived-margin, "14/14" retracted (C2).
- **Honesty:** over-abstention critique revised by measurement — the gate is well-calibrated; residual
  false-abstain is a generation-consistency issue, linking to the SFT work (C3 → C7).
- **Crux:** matched attention-only baseline is one command away (`run_crux_baseline.sh`, C6).
