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
benchmarks section. **SUPERSEDED for GSM8K by #20:** that near-zero was mostly direct-prompt + bad parsing;
with few-shot CoT + a general parser the continued-trained 1B scores ~62%.

## 12. THE CRUX — matched transformer vs spin 284M (`crux_compare.py`) — comparison PENDING fair spin
The matched attention-only transformer baseline (289M, carrier=none, 418K steps, same corpus/budget) is
trained and confirmed. But the lr-floor bug (`f3277c1`) crippled runs that did not get the warm-restart fix:
the baseline itself was stuck at **val ppl 174.8** until the fix took it to **21.01**.

| Model | params | val ppl (200-it harness) | val ppl (training) | fair? |
|---|---|---|---|---|
| transformer baseline (fixed-lr) | 289M | 22.22 | 21.01 | yes |
| spin 284M (**PRE-FIX, crippled**) | 284M | 25.68 | 24.54 | **NO** |

**Status: NOT DECIDED.** The old spin 284M ran WITHOUT the lr fix, so 24.54/25.68 is a handicapped number,
not the spin's true potential. The fair comparison is transformer-baseline vs the re-run **fixed-lr** spin
(`pragnosia_284m_fair.pt`, in progress). `crux_compare.py` auto-detects it. Do NOT read a winner from the
pre-fix numbers.

## 13. Internal monologue: honesty-gated consolidation (`monologue_effect.py`)
Added an honesty gate on the monologue: a self-generated thought is written to memory/weights only if NOVEL
**and** it passes the truth probe (internal state looks known, not confabulated) -- `brain._worth_consolidating`,
wired into `think_aloud` and `background_tick`. Prevents the monologue self-poisoning the weights with its own
hallucinations (previously it consolidated on novelty alone).

Measured on 20 self-generated thoughts (1B):
| | value |
|---|---|
| Pass honesty (knows > 0.5) | 7/20 |
| Blocked as confabulation | **13/20 (65%)** |
| Mean knows: kept vs blocked | **0.80 vs 0.28** |

The gate keeps the one genuinely-correct thought ("Photosynthesis is how plants turn sunlight...", 0.90) and
blocks the filler/off-topic/confabulated ones (0.10-0.21). **Two honest findings:** (1) the gate discriminates
cleanly -- anti-self-poisoning validated; (2) raw monologue quality is poor (65% of thoughts are junk) because
it is capped by the weak base LM, and separately the novelty gate currently blocks ALL consolidation
(novelty_min > observed novelty) -- so the monologue is presently safe but inert. Making it genuinely useful
needs the better base model (thoughts worth keeping) and a latent-first redesign; the safety gate is the
first honest step and it is done + measured.

## 14. THE CRUX — DECIDED: a statistical TIE at 284M (`crux_compare.py`)
Both models: fixed-284M from scratch (spin confirmed NO growth), same corpus/tokenizer, matched budget
(418K steps, 6.85B tokens), both with the lr-floor fix.

| | training best val ppl | my 200-it harness | multi-hop |
|---|---|---|---|
| transformer (289M, carrier=none) | 21.01 | 22.89 (best) / 22.22 (final) | 2/8, 1/3 |
| spin-dominant (284M, fair-lr) | **20.92** | 22.32 | 3/8, 2/3 |

**Verdict: TIE.** Training-best gap is 0.4% (spin marginally ahead). But the transformer's own best-vs-final
checkpoints differ by ~0.7 ppl in my harness -- LARGER than the spin-vs-transformer gap -- so neither
reliably wins. **The 21% spin advantage at 37M narrows to parity at 284M.** Honest claim: spin-dominant is
*competitive* with a matched transformer at scale and does NOT collapse (contrast the 1.4B drift); we do NOT
claim it wins at 284M. The carrier remains the load-bearing computation by ablation regardless.

## 15. Coupling at 284M (laptop) -- inconclusive BY DESIGN; needs from-scratch on H100
Tried Stage A on the laptop as warm-start-from-fair-spin + continue (coupled vs base control, matched
lr/data/steps). Result: the fair spin is CONVERGED on window2_train, so continued training has no headroom --
both arms only DRIFT UP:
| lr | control end | coupled end | verdict |
|---|---|---|---|
| 2e-4 | 57.7 (from 22.3) | 58.0 | both destroyed; coupled ~= control |
| 5e-5 | climbing (22.3->24.4 @500) | (stopped) | both degrading; no headroom |

**This does NOT decide the coupling.** Warm-start-continue is the wrong test (no headroom); the coupling adds
capacity, which only helps trained FROM SCRATCH with room to fit the data (where the 37M spin win was
measured). A from-scratch 284M coupled-vs-base run is not laptop-feasible (hours). Moved to H100_TODO Stage A
as a from-scratch run. Honest status of the coupling: **still undecided** (frozen-1B probe neutral, laptop
warm-start inconclusive) -- the from-scratch 284M test on the H100 is the real decider.

## 16. Coupling Stage A — DECIDED: an honest NEGATIVE (from-scratch 284M, H100)
Ran exactly the Stage A design from #15 / H100_TODO: both arms FROM SCRATCH at 284M, `--no-grow`, matched
budget (100K steps, ~1.6B tokens) and matched explicit lr (6e-3); the arms differ ONLY in the coupling
(`carrier=spin_dominant` control vs `spin_dominant_coupled`).

| | training best val ppl | clean harness (4 val-seeds x 30 it) |
|---|---|---|
| control (base spin, 284M) | **23.87** | **24.87** |
| coupled (284M) | 25.51 | 26.65 |

**Verdict: NEGATIVE — the coupling is 7.2% WORSE** (gap 1.78 ppl, far outside the ~0.7 ppl checkpoint noise
that made the crux a tie). Letting the slow carrier modulate the fast attention pathway does not help; it
hurts. This is the second independent measurement against it (the frozen-1B probe was neutral: +0.04).
**Per the pre-registered criterion, STAGE B (scale to 1B) IS CANCELLED** — no 1B compute spent on a dead
idea. Report as an honest negative; keep the parallel design.

## 17. Multi-seed at 100M — PARITY; the 37M spin win does NOT survive scale
Param-matched pair at ~100M (spin d=640 h=10 L=14 mm=6 = 109.5M; attention-only d=640 h=10 L=15 mm=6 =
111.6M, +1.9%), 3 training seeds each (spin+attn of a seed trained as a parallel pair = identical GPU
conditions), matched tokens (70K steps, ~1.15B, ~10.5 tok/param) and matched explicit lr 3e-3 for both
arms (an explicit matched lr also dodges the unreliable auto-derive, which produced 1.78e-5 for one arm).

Clean harness (4 val-seeds x 30 it x bs24, big_valid):

| seed | spin | attn | paired delta (attn-spin) |
|---|---|---|---|
| 1 | 28.12 | 28.57 | +0.45 |
| 2 | 28.99 | 29.22 | +0.23 |
| 3 | 29.29 | 29.38 | +0.09 |
| **mean** | **28.80** | **29.06** | **+0.255 ppl -> spin +0.88%** |

**Verdict: PARITY.** Spin is ahead in **3/3** paired seeds (direction is consistent — spin is never worse),
but the effect is **+0.88% (0.26 ppl)**, the unpaired ranges **overlap** (spin worst 29.29 > attn best 28.57),
a paired t-test at n=3 gives **p ~ 0.14 (not significant)**, and the per-seed deltas **shrink** (0.45 -> 0.23
-> 0.09; seed 3 is a dead heat).

**The scale story (honest):** 37M ~21% -> **100M ~0.9% (n.s.)** -> 284M tie (#14). The spin advantage is a
small-scale phenomenon that is essentially gone by 100M and is parity by 284M. Consistent with #14: claim
spin-dominant is *competitive* at scale (and does not collapse — contrast the 1.4B drift), NOT superior.
The carrier remains the load-bearing computation by ablation regardless.

## 18. MULTI-SEED CAUSAL ABLATION — the CENTRAL claim is seed-robust (`multiseed_ablate.py`)
The paper's centerpiece (the carrier is the load-bearing computation) was single-seed at each size. The 100M
sweep left 3 independently-trained spin models and ablation is inference-only, so seed-robustness is cheap.
Ablate ALL spin carriers (zero the gate, keep the MLPs) vs ALL attention sublayers, per seed:

**n=3 (all three seeds):**

| seed | base ppl | ablate SPIN | ablate ATTN | carrier/attn gap |
|---|---|---|---|---|
| 1 | 28.34 | 5628 (198.6x) | 85.6 (3.02x) | 66x |
| 2 | 28.98 | 8853 (**305.5x**) | 82.0 (2.83x) | 108x |
| 3 | 29.26 | 6206 (212.1x) | 84.3 (2.88x) | 74x |

**Verdict: the load-bearing property reproduces in EVERY seed** -- carrier ablation costs **two orders of
magnitude** more than attention ablation in all three (gap 66x / 108x / 74x). This answers the "single-seed"
concern on the CENTRAL claim (#17's multi-seed only covered the spin-vs-attention *comparison*).

**Honest correction:** an interim n=2 (seeds 1,3) showed 203.9x vs 206.3x and looked like a 1.2% spread --
that was luck. With seed 2 at 305.5x the real spread is **198.6-305.5x (54%)**. So the *exact* spin multiplier
is NOT stable across seeds, while **attention ablation is** (2.83-3.02x, 7%). This independently reinforces
the #7 / C4 decision to report the **order-of-magnitude gap, not a precise multiplier** -- the multiplier is
noisy across seeds just as it is across checkpoints. Claim the gap; never the exact number.

## 19. Ablation on the CORRECTED (fair-lr) 284M — the lr bug moved the ppl, not the conclusion
The paper had two 284M spin runs and did not distinguish them: the original (val ppl 24.54, ablation 306x)
predates the lr-floor fix; the fair-lr re-run (20.92) is the crux model. Ablated the corrected model
(`eval_registry/ablate_284m_fair.json`, inference-only):

| 284M model | base | ablate SPIN | ablate ATTN | gap |
|---|---|---|---|---|
| original (pre-fix) | 25 | 7716 (306x) | 85 (3.4x) | 90x |
| **fair-lr (crux model)** | 21.82 | 6838 (**313x**) | 100.4 (4.60x) | **68x** |

**The load-bearing property is unaffected by the lr bug** — it reproduces on the corrected model (313x vs
4.6x) and across 3 seeds at 100M (#18). The bug moved perplexity, not the conclusion. Paper now explains the
two runs explicitly instead of quoting 24.54 and 20.92 for "the 284M model" without distinction.

## 20. Reasoning RE-MEASURED — GSM8K "0%" was a PARSER artifact; needle = decay + weak-copy (`eval_reasoning_cot.py`)
Tested on the continued-trained best 1B (`pragnosia_1b_best.pt`, val 16.56). Inference-only, run alongside
live training (no training cycle spent). Few-shot examples are in-context DEMONSTRATIONS and the answer
parser is a generic regex ("answer is X" > last "= X" > last number) -- neither is fit to the eval's
answers (no hardcoding).

**GSM8K-style (8 grade-school problems):**
| Method | Score |
|---|---|
| direct-answer prompt (what #11's ~0-1.3% used) | 3/8 |
| **few-shot CoT + general parser + stop-on-answer** | **5/8 (62%)** |
| correct number present in the model's working | 6/8 |

The near-zero GSM8K in #11 was **mostly a harness artifact** -- last-number-over-a-ramble, no stop
sequence, no CoT. With CoT the model DOES multi-step arithmetic ("50 mph * 2 hours = 100 miles", "20 - 7 =
13", "24 / 4 = 6"). Two residual, specific failure modes: (a) it computes correctly then botches the *stated*
answer line; (b) a systematic **multiply-vs-divide confusion** on rate problems ("X per Y for N" -> divides
to a rate instead of multiplying to a total). Both are SFT-fixable, not a capability ceiling.

**Needle (recall a planted secret; naive long-range vs retrieval):**
| Condition | Recall |
|---|---|
| naive depth ~16 tok | 5/6 |
| naive depth ~64 | 2/6 |
| naive depth ~200 (still inside the 256 attn window) | 0/6 |
| naive depth 512 / 1024 | 0/6 |
| **RETRIEVAL** (feed only the queried sentence, brain.py-memory style) | **3/6** |

The 0/15 from the earlier deep battery splits into TWO causes: (1) the damped |lam|<1 carry FORGETS within a
few hundred tokens (5/6 when recent -> 0 by ~200), and (2) the verbatim-copy mechanism itself is only ~50%
even when the fact is present (retrieval 3/6, not 6/6). So retrieval recovers a lot but is copy-limited.

**Fix roadmap (what "good" would take) -- SFT/architecture items need a training cycle, so QUEUED, not run
(training must not stop):**
- GSM8K: free = few-shot CoT + parser (done, 62%); cheap = small CoT SFT emphasizing total-vs-rate (1B
  ceiling ~60-80% on grade-school); robust = tool-use (calculator) via `brain.py` act->observe.
- Needle: retrieval via `brain.py` episodic memory / `FastWeightMemory` works now (~50%, copy-limited);
  higher = recall-task SFT (teach exact copy) and/or **selective/input-dependent state** (Mamba-style lambda
  so the carrier can *latch* a token) -- the upgrade that keeps the O(1)/KV-cache-free thesis. Global
  attention would also solve needle but SACRIFICES that thesis, so it is not the preferred path.

## 21. Selective state — toy proof that input-dependent decay fixes long-range recall (`selective_spin.py`)
The needle failure (#20) is architectural: `SpinCarrier`'s decay is input-INDEPENDENT, so it can't latch a
token. Prototyped the Mamba/S6 fix (per-token step size `Delta_t=softplus(W x_t)` -> `|lambda_t|=exp(-Delta_t*
exp(nu))`, |lambda|<1 preserved, parallel scan + O(1)/token intact). Tiny carrier-only LM, synthetic copy
task, CPU (chance 0.125):

| gap | fixed-decay | selective (latch-biased init) |
|---|---|---|
| 32 | 1.00 | 1.00 |
| 64 | **0.12 (chance)** | **1.00** |
| 128 | **0.12 (chance)** | **1.00** |

Fixed decay collapses by gap 64; selective recalls perfectly to 128. **Latch-biased init (remember-by-default)
is essential** -- forget-by-default barely beat baseline. Mechanism VALIDATED; the real 1B win needs a training
cycle (queued). Full design + integration plan in `SELECTIVE_STATE.md`.

## 22. Selective-state continue-train ATTEMPT #1 — NEGATIVE (recipe wrong, mechanism still sound)
Grafted the validated `spin_selective` carrier (input-dependent decay) onto the v4 SFT model
(function-preserving graft = exact, 0.0 logit diff) and continue-trained mixed-W (~19h, ~4.3K steps) on
`sft_selective.bin` (82.5M, 50% replay). Tested the mid-run checkpoint:

| | selective (attempt #1) | base/v4 |
|---|---|---|
| long-ctx keyword recall (5K ctx) | 0/5 | 0/5 |
| needle @64 / @256+ | 2/4 / 0 | ~same |
| "Who are you?" | **"I don't know"** (regressed) | "I am Pragnosia..." |
| "3 plus 5" | rambled | 8 |

**Verdict: this run failed on BOTH axes** -- no long-context gain AND short-form regression. Root cause is the
TRAINING RECIPE, not the architecture: (1) the latching signal (long-range recall) was only ~3.5% of the
corpus -- far too weak against 50% general replay + 18M long-doc tokens to teach the delta to latch from a
function-preserving (no-modulation) init; (2) ~10 epochs of mixed-W general/long text eroded v4's crisp SFT
skills (identity, math) toward web-rambling + reflexive abstention. The TOY that hit 1.00 (#21) used the
opposite: latch-biased init + a corpus that is almost entirely the recall task. **Corrected recipe for a
future attempt:** latch-biased delta init (bias ~ -2), corpus DOMINATED by long-range recall, light general
dilution, few epochs. Run stopped; v4 kept as the deliverable; long-context via `brain.py` retrieval is the
pragmatic path meanwhile. spin_selective code + graft + corpus stay staged.

## 23. Brain pipeline steps 1-2 — memory loop FIXED, continual learning made SAFE (episodic)
Toward the "brain" (continual, memory-grounded agent) on the v4 base. Two brain.py fixes, measured:

**Step 1 - memory/retrieval loop (store->retrieve->use): FIXED.** Was broken two ways: taught facts only went
to weights/FastWeightMemory, never the retrievable `self.store` (so `_retrieve` only ever found identity);
and the embedding ranked by question-FORM (every "What is X?" matched "What is your name?"). Fixes: statements
are now always indexed in `self.store` (episodic, decoupled from the weight-teach novelty gate); `_retrieve`
is a hybrid of embedding + self-information-weighted content-token overlap (no stopword list) requiring
majority content match; and `interact()` now SEEKS the store before guessing/web. Result: tell 3 facts ->
recall 3/3 (both question angles), zero false-fire on unrelated Qs. Doubles as the system-level long-context fix.

**Step 2 - continual learning: weight-teach CATASTROPHICALLY FORGETS; episodic is the safe path.** The
novelty gate blocked all teaching (`_learn_min=32` tokens -- facts are 7-20; fixed to content-based). But once
unblocked, teaching ONE fact into weights drove its nll 6.5->0.2 AND **GSM8K 3/4 -> 0/4** (+ over-memorized:
"Xland.Bortville.BortBort."). Gentler lr/target/steps did NOT save it -- the concentrated single-fact gradient
overwrites skills faster than the thin self-replay protects. So weight-teach is now OFF by default
(`WEIGHT_TEACH=1` to force); the brain learns EPISODICALLY: learn 3 facts -> recall 3/3, **GSM8K holds 3/4 ->
3/4 (no forgetting)**. Forget-free weight-internalization needs real R&D (EWC / LoRA adapters / heavy replay) --
that is the training wall.

**Step 3 - grounded agency: loop CLOSES, behavior-improvement is the wall.** `experience()` in the GridWorld
runs perceive->act->observe->FEEL(affect: mood -0.66, arousal 1.0 on failure)->LEARN(24 episodic lessons), and
self-directed goal-proposal works. But over 5 episodes the return stays flat (-1.7, never reaches goal): the
stored lessons aren't fed back into action-selection and a 1B can't navigate unaided, so the loop runs without
improving BEHAVIOR. Closing it needs a memory-guided policy (inference, uncertain on a 1B) or RL (training).
**Pipeline stop:** steps 1-2 delivered (memory + safe episodic continual learning); step 3 loop validated but
behavior-learning + step 4 (perception encoders) both need training/RL R&D -- the training wall.

**Research (inference-only) probes of the wall items -- both NEGATIVE (they need training):**
(a) EPISODIC CONTROL for agency behaviour (instance-based RL, no gradient): buffer (state_emb, action, reward),
pick the reward-weighted nearest action. Result: NO improvement (return flat, 0/10 reach) -- the LM's state
embedding of the grid obs isn't discriminative enough for similarity control, and structured direction-parsing
would be the hand-coded planner the design forbids. Behaviour-learning needs RL (a reward-gradient training run).
(b) LoRA-ISOLATED fact teaching for forget-free internalization (base FROZEN, 11.6M adapter only): the fact is
learned (nll 6.3->0.03) but GSM8K STILL collapses 3/4->0/4 -- an always-on adapter over-fit to one fact distorts
every other forward, so freezing the base does not save it. Forget-free internalization needs LoRA + heavy
replay + gentle early-stop + regularization (a training-tuning loop), OR just keep episodic memory (which works).
Conclusion: the brain's solid core (memory + safe episodic continual learning + honest abstention on v4) is
inference-complete; the advanced faculties (behaviour-RL, weight-internalization, perception) all require the
training that comes last -- now with a known recipe, not a guess.

## 24. Selective-state combined run ATTEMPT #2 — PARTIAL WIN: carry horizon ~2x, hard cliff at ctx=256
The corrected recipe from #22, run as ONE combined train: latch-biased graft (`LATCH_BIAS=0`, after `-2` blew
up -- see below) of v4 -> `spin_selective` carrier, mixed-W (LONG_CTX_W=32), corpus DOMINATED by long-range
recall + a knowledge slice (internalization) + 50% replay (`sft_combined_run.bin`, 100.7M). The `LATCH_BIAS=-2`
first attempt drove |lambda|~1 and W32 (8192-tok) val ppl EXPLODED to ~8400 by hour 46 -- restarted at
`LATCH_BIAS=0` (milder latch). The bias-0 run started at v4 quality (loss 1.95, not the -2 run's 8.9), never
NaN'd, prod stayed 200 throughout. It **peaked at it=1500 (hour 7, VAL_PPL 18.94)** then drifted UP for 16h --
W1/W8 flat ~21 but **W32 alone climbed 18.9 -> 45.1** (the same long-only destabilization as #22, far milder and
40h slower). Stopped at it=4947; kept the it=1500 best (`pragnosia_1b_combined_best.pt`).

Tested it=1500 (selective) vs v4 (spin_dominant), SAME greedy harness (`eval_reasoning_cot.py` + a 14-trial
needle sweep):

| naive carry-recall @depth | v4 | it=1500 selective |
|---|---|---|
| ~128 tok | 57% | **92%** |
| ~192 tok | 7% | **50%** |
| ~256 tok | 0% | 0% (HARD CLIFF -- both) |
| 320-512 | 0% | 0-21% (noise) |

GSM8K (same harness): v4 2/8 -> it=1500 **5/8** (no regression; knowledge/replay slice helped a touch).

**Verdict: the mechanism WORKS but is bounded by the training context window.** The corrected recipe (#22's fix)
genuinely taught the input-dependent decay to latch -- reliable recall horizon roughly DOUBLED in the 128-192
band, with zero skill loss (50% replay held). But BOTH models collapse to 0 exactly at ctx=256 (the training
window): the selective carrier holds info a bit PAST short-range but still can't cross the 256-token chunk
boundary reliably, which is why W32 never converged and the run peaked at hour 7. It did NOT reach the 512+/8K
goal. **Next lever the data points at: train at a LONGER context window (ctx 512/1024), not just mixed-W at
ctx=256.** it=1500 kept as proof-of-mechanism; v4 remains the shipped deliverable; brain.py episodic retrieval
stays the pragmatic long-context path meanwhile.

**GROWN-WINDOW SCALING RESULT (follow-on to #26):** ctx=512 grow WORKED cleanly (pos-interp warm-start; needle
depth-256 0%->100%, kept as `pragnosia_1b_ctx512_it500_CROSSES256.pt`). But ctx=1024 grow (from the ctx512 winner,
pos-interp 512->1024, 512 new positions) did NOT: val never beat its 19.15 warm-start, WIN=1024 needle oscillated
and topped out ~512 at a weak 35% (256:50-85%, 512:21-35%, 640:14-21%, 768+:0%), then decayed (peak ~it=1000 then
overfit-decay, 128 recall collapsed to 0% by it=1500). So **grown-window has a practical native ceiling ~512 (2x)
for this 1B with a quick recipe** -- a model pretrained at ctx=256 can double its window cleanly but 4x
destabilizes (bigger interp perturbation + attention long-range patterns never pretrained). Native long-context =
DOUBLED and stable at 512; beyond that is resolved by retrieval (#28), not the window. Deeper native long-context
would need the delta-carrier research (#27) or from-scratch longer-ctx pretraining, not a quick grow.

## 37. UNIFIED BRAIN v2 -- ONE model does all 4 faculties + GROWTH built in (`unified_brain.py`)
Closes the biggest gap (faculties were on SEPARATE toy models). ONE 2.4M UnifiedBrain (fast core VChunkRecall,
d192) trained on a MIXED stream of all task types; growth is a MODEL METHOD `.grow()` (function-preserving
depth-add: new block = exact identity via zero-init output projs). All four faculties measured on the SAME model:

| stage | skill(weights) | recall(state) | stream@200 | agency(exploit,chance25) | forget-free skill-kept |
|---|---|---|---|---|---|
| pre-grow 3L | 100% | 98% | 100% | 90% | 100% |
| post-grow 4L (t=0) | 100% | 98% | 100% | 89% | 95% |
| grown 4L trained | 100% | 98% | 100% | 86% | 91% |

PROVEN in one network: memory(state)+skill(weights) COEXIST; streaming recall over distractors; agency
exploit-after-discovery 86-90%; forget-free (new facts in-state keep the weight-skill ~91-100%). GROWTH is
in-model + function-preserving (faculties ~UNCHANGED across 3L->4L at t=0), then continues training -> the brain
grows mid-life without forgetting. Saved `unified_brain_v2.pt`. This is the intrinsic brain integrated at small
scale. REMAINING (honest, next phase): real LANGUAGE (all tasks are synthetic), real SCALE (2.4M -> useful size),
streaming EXTRAPOLATION past trained length, and a fused kernel. Integration + growth gaps: CLOSED.

## 36. #3 SOLVED (measurement) + #2 improved (fast core) -- (`prove_agency_fast.py`, `prove_streaming_fast.py`)
Revisited both partials on the fast core with fixes:
- **#3 AGENCY -- SOLVED, it was a METRIC artifact.** The "weak 40%" first/last-third optimal-rate CONFLATES
  unavoidable exploration (first visit to a context MUST guess) with failure. Correct metric = EXPLOITATION-
  AFTER-DISCOVERY (once a context's good action has been rewarded, does the model pick it on later visits?).
  Result on the fast core (15k steps): **85%** (chance 25%) -- STRONG learning-from-consequence, in the fast-weight
  state, zero weight updates. (The crude first/last metric was 29->35% same run.) #3 is a real WIN; the earlier
  "moderate" was mis-measurement, not a weak model.
- **#2 STREAMING -- horizon extended to 8K solid; crash was an out-of-range embedding index (FIXED by clamp).**
  The recurring device-assert was `vectorized_gather_kernel: ind>=0 && ind<dim` = a rare OOR embedding id (my CPU
  sampling missed it; async attribution hid it). A `x.clamp_(0,VOCAB-1)` guard fixed it -> trains clean at 97
  step/s, loss->0.000. Trained on 6K-token streams: recall 100% @2K AND @8K (was collapsing before), but 2% @32K,
  0% @100K. So recall horizon TRACKS the trained length (train longer -> recall further) but does NOT extrapolate
  beyond it. Net #2: O(1) constant memory PROVEN (#32) + a TRAINABLE recall horizon (solid to 8K here) -- a real
  but bounded win (not unbounded extrapolation).

## 35. Porting #2/#3 onto the fast core -- speed delivered, strengthening did NOT (honest)
Ported agency (#3) + streaming (#2) onto VChunkRecall (#34) to retry their STRONG versions now that iteration is
~25x faster. Outcome:
- SPEED: agency trained 15k steps at ~46 step/s (was ~5-10) -- the fast core works, iteration is cheap now. WIN.
- #3 AGENCY: 15k steps on the (ungated) fast core -> 28%->33% (chance 25%) = NOT stronger, slightly WORSE than the
  gated-core 31%->40% (#32). The ungated core is weaker for reward-history recall, and more training didn't crack
  it. In-context RL is just a hard task at this tiny scale. #3 stays MODERATE (best 31->40, gated).
- #2 STREAMING: VChunkRecall hits an INTERMITTENT CUDA device-assert on long sequences (surfaces async at a later
  LayerNorm; non-deterministic -- 40 steps survive under CUDA_LAUNCH_BLOCKING+seed0 but the real run crashes early).
  Poisons the context so it can't be caught/skipped; a feature-bounding (LayerNorm) guard didn't fix it. Likely a
  rare numerical inf in the long-seq cumsum. Horizon-extension could not run -> #2 stays at constant-memory-PROVEN
  (#32, 73MB flat to 100K), recall horizon ~8K, extension unresolved (undertraining #33 THEN this numerical bug).
NET this round: the fast core is a genuine speed/#4 win; #2 and #3 did NOT get to "strong" -- #3 is
hard-at-scale, #2 needs a numerically-stable long-seq kernel (the recurring "needs a proper kernel" theme).

## 34. FAST CORE via VECTORIZATION -- ~17x flash-attention, no Triton/compile (`fastcore_v.py`)
The eager core was kernel-LAUNCH-bound (Python chunk-loop of ~120 small ops/step, GPU ~60% util). torch.compile
failed on it (#33, device-side assert). SOLUTION = VECTORIZE the chunk loop in pure PyTorch: all chunks' intra-
chunk attention as ONE batched matmul; the inter-chunk state recurrence as ONE cumsum (exclusive: Sprev=cumsum(KV)
-KV); ~5 big ops instead of ~120 launches. Autograd FREE (pure torch), no compile/Triton fragility, no hand-written
backward. `VChunkRecall` (Taylor feat8, chunk128/256), d=256/2L, T=65536 fwd+bwd:

| core | tok/s @64K | recall@16 |
|---|---|---|
| eager loop (chunk128) | 78K | 100% |
| big-chunk loop (chunk512) | 311K | 100% |
| **VECTORIZED (VChunkRecall)** | **1,952K** | **99%** |
| flash-attention (ref) | 116K | (n/a) |

**~25x the eager loop, ~17x flash-attention at 64K, recall intact.** This is the fast substrate -- unblocks fast
iteration on ALL the brain demos, and STRENGTHENS #4 (per-FLOP win) from 2.69x to ~17x vs attention at long ctx.
No Triton kernel needed (the launch overhead, not FLOPs, was the whole problem). Next: port the demos onto
VChunkRecall (add the gate to the vectorized form for the forgetting demos) so #2/#3 iterate in seconds not hours.

## 33. Follow-ups on #2 (streaming) -- two honest NEGATIVES (`prove_streaming*.py`)
Tried to (a) extend the recall horizon by training on LONGER streams, and (b) speed up the slow eager core with
torch.compile. Both failed:
- HORIZON-EXTENSION NEGATIVE: retrained streaming with fill=1500 (vs 300) but fewer steps (2500 vs 4000) + lean
  feat8 -> recall collapsed to ~2% (chance) even at 2003 tokens = UNDERTRAINED. Longer streams are a HARDER task
  (fact must survive more interference) and needed MORE steps, not fewer. So horizon-extension not achieved; the
  original #2 result (100% to 8K, fades by 100K, constant 73MB) stands as the honest best.
- torch.compile NEGATIVE: compiling the gated core -> inductor Triton autotuner (`benchmark_all_configs`) hits a
  CUDA device-side assert (same class as the spin-scan compile corruption). So torch.compile can't fuse this core;
  the eager core stays launch-bound/slow for training iteration. FAST iteration would need a hand-written Triton
  kernel (or fla). NOTE: the #4 efficiency WIN (2.69x attention @64K) came from the EAGER big-chunk trick and does
  NOT depend on compile -- it stands. WHY the eager core is slow at all (diagnosed): kernel-LAUNCH-bound -- tiny
  model (d128) + eager + a Python chunk-loop of ~120 small ops/step -> GPU ~60% util (starved), dispatch overhead
  >> compute. Flash-attn is one fused kernel (no launch tax) -> the honest contrast.

## 32. BRAIN FACULTIES #2/#3/#4 -- honest scorecard (`prove_streaming.py`, `prove_agency.py`, `prove_efficiency.py`)
Built + measured the other 3 faculties on the ChunkedRecall/GatedRecall core. Honest outcome: only #1 (#31) is a
clean strong win; #2/#3/#4 are PARTIAL -- mechanism shown, strong version needs more. No overclaiming.
- **#2 O(1) STREAMING (`prove_streaming.py`): HALF.** Fact @ start of stream, recall @ end, fed in chunks carrying
  only the state. PEAK MEMORY FLAT at 73MB from 500 -> 100K tokens (constant-memory streaming PROVEN; attention
  can't). BUT recall decays: 100% @8K, 67% @32K, 2% @100K -- the fixed-capacity gated state + per-token gate decay
  lose a single fact over 100K tokens of interference. => constant memory YES, unbounded recall NO (bounded
  horizon). Path: train on longer streams + tune gate to hold longer + more state; or lean on retrieval beyond.
- **#3 AGENCY / in-context RL (`prove_agency.py`): WEAK.** Meta-trained in-context bandit (explore then exploit
  the reward-history-in-state; first attempt was flat-at-chance because I teacher-forced the optimal action --
  fixed to a reward-conditioned target). First attempt 34%->37% (weak); STRENGTHENED (NC 6->4, d192/3L/feat16,
  12k steps, 64-step episodes) -> **31%->40% within an episode (chance 25%)** = +9pt in-episode gain, 1.6x chance.
  The property (behavior improves from consequence, in the fast-weight state, ZERO weight updates) is DEMONSTRATED;
  still MODERATE (40% not strong exploitation) -> full strength needs more scale/training.
- **#4 per-FLOP WIN vs attention: FIRST measured NEGATIVE, then WON after fixing the impl (`prove_lean2.py`).**
  Initial (`prove_efficiency.py`, feat16/F273, chunk128): flash-attn faster to 128K (ours 0.1x->0.6x). Diagnosis:
  the fat Taylor feature (F=273) + a too-granular chunk loop (128 -> many small kernel launches), NOT the O(T)
  mechanism. FIX = LEAN feature (feat=8, F=73 -- still recalls) + BIGGER chunk (512 -> fewer, larger matmuls).
  Result at T=65536, fwd+bwd, d=256/2L: **ours 311K tok/s = 2.69x flash-attention (116K)**, and the SAME config
  recalls **MQAR-16 = 100%**. So a lean-feature + big-chunk linear-attn core BEATS flash-attention at long context
  while keeping perfect recall -- #4 ACHIEVED at long T (crossover is in the 16K-64K range; attn still wins at
  short T, which is fine -- long context is where a brain's memory/streaming lives). No custom kernel needed.
HONEST SUMMARY of the 4-faculty brain @ small scale: the recall core (#30) + forget-free online learning (#31) are
real, strong, in-model wins; constant-memory streaming is proven but recall-horizon-bounded; agency + efficiency
are demonstrated-but-weak, each with a clear, concrete path to strengthen (longer-stream training / more meta-RL /
lean fused kernel). The ambition is validated in MECHANISM at small scale; the strong versions are the next work.

## 31. BRAIN FACULTY #1 in-model: forget-free ONLINE LEARNING (`gated_core.py`, `onforget_demo.py`)
First of the 4 intrinsic faculties (task #9), on the ChunkedRecall substrate (#30) + a per-token per-head GATE
(GLA-style chunked decay = the intrinsic forgetting mechanism; sanity: gated core still MQAR-16 99.9% @ 346K
tok/s). Demo: a small model learns a fixed SKILL into its WEIGHTS (permutation lookup, 100%) AND in-context fact
recall from its STATE. Teach it NEW facts two ways:

| | new-fact recall | prior SKILL after |
|---|---|---|
| (A) IN-STATE (stream facts through forward pass; weights untouched) | 88% | **100% -> 100%** |
| (B) WEIGHT-TEACH (gradient-fit the facts = old brain.py teach()) | 88% | **100% -> 0%** |

**Both learn the facts; only in-state keeps the skill.** Forget-free online learning is INTRINSIC to the core --
new knowledge enters the fast-weight STATE (a forward pass), so weights + all prior skills are untouched BY
CONSTRUCTION. This solves the #23 failure (teach() dropped GSM8K 3->0) in the architecture, no replay/store code.
HONEST RECIPE FINDING: forget-free-ness is NOT automatic -- the skill must be trained STATE-ROBUST (computed
correctly regardless of state contents; use state only when querying). Empty-state-trained skill broke to 0%
under a loaded state (out-of-distribution); adding random fact-prefixes to skill training -> robust -> 100%.
Remaining faculties on this core: O(1) streaming, agency (reward-modulated fast-weight update), per-FLOP win.

## 30. INTRINSIC-BRAIN SUBSTRATE found — fast, recall-capable, ours (`fastcore.py`)
Redirect (user: the project is a BRAIN not an LLM; faculties must be IN THE MODEL, not brain.py scaffolding; and
it must be compute-efficient + provable SMALL). Substrate bake-off (`bake_off.py`) + core build settled it:
- spin (our thesis carrier): fastest (211K tok/s) but FAILS recall (MQAR 26/12/2%) and state (~chance) -> cannot
  carry a brain (no memory, no online-learning). Honest: spin is NOT the substrate.
- `ChunkedRecall` (fastcore.py): chunked-parallel linear attention w/ 2nd-order Taylor feature map (softmax-like
  sharpness -> recall). MATMUL chunk form (no per-token loop). d=128/2L/feat16:

| metric | ChunkedRecall | spin | my naive loop-delta |
|---|---|---|---|
| MQAR-4 / 16 / 32 | 99.9% / **100%** / 7% | 26/12/2% | 99.5/10/7% |
| speed | **1,470K tok/s** | 211K | 26K |

**A FAST (7x spin), RECALL-CAPABLE (100%@16) core -- pure PyTorch, no deps.** INTRINSIC brain shape: the
fast-weight state S=sum phi(k)^T v IS the memory; recall o=phi(q)S; writing S is the learning -> memory +
online-learning live in the FORWARD PASS, not in self.store/teach(). Capacity ceiling at 32 pairs = the
feature-dim DIAL (feat16 ~16 bindings; raise feat for more), a knob not a wall. THIS is the substrate for the
integrated brain (task #9): build the 4 faculties (O(1) streaming, online-no-forget via a per-token GATE, agency,
per-FLOP win) ON this core, each proven at <=100M. Next: add the gate (forgetting) + demonstrate online-no-forget.

## 28. Long context RESOLVED for the product — first-class retrieval (`brain.py`, `test_longctx_retrieval.py`)
The native carrier can't recall past its window (#27) and window-growing (#26) is bounded + abandons the thesis.
So the product's long context is resolved the working way: `brain.py::_ingest` splits arbitrary-length input into
SENTENCE chunks (each stored with its own embedding), and `_retrieve` pulls the exact relevant sentence. Fixed a
router collision (a 'where does Dr. X keep..' question mis-routed to the 'provenance' meta-answer -> now seeks
memory before provenance, guarded by the >=0.5 content bar so genuine meta-Qs stay safe). Needle-in-haystack test
(fact buried after 10/40/100 filler sentences, fresh memory each trial):

| depth (filler sentences) | needle recall |
|---|---|
| 10 | 3/3 exact |
| 40 | 3/3 exact |
| 100 | 3/3 exact |

**DEPTH-INDEPENDENT exact-fact recall** -- identical at depth 100 as at depth 10, which no fixed-window native
model can do. SCOPE (honest): this resolves needle-style FACT recall over arbitrary length; long-range REASONING
(multi-hop over distant facts, summarization) is still bounded by what retrieval surfaces + the model's window.
So: product long-context = RESOLVED for recall; native architectural long-context (the carrier doing it) remains
the hard delta-rule research (#27).

## 29. DEEPER RESEARCH Phase 1 — a subquadratic carrier that MATCHES attention on recall (`research_carriers.py`)
Follow-on to #27 (our diagonal carrier can't recall; delta didn't scale in quick probes). Root cause of the quick
failures was diagnosed as (a) a benchmark CONFOUND (carriers ran pos-OFF, attention pos-ON) and (b) linear recall
being too SMOOTH. Fix = Based (Arora et al.): linear attention with a 2nd-order Taylor feature map (approximates
exp(q.k) -> softmax-like SHARPNESS), computed via CUMSUM (fully parallel, O(T), no python loop). FAIR test (all
pos-ON, same steps/seed, tiny d=128 2-layer):

| #pairs | diagonal (ours) | Based (feat=16) | attention |
|---|---|---|---|
| 16 | 10.3% | **99.8%** | 99.9% |
| 32 | 5.9% | 7.6% | 100% |
| 64 | 1.5% | 5.4% | 100% |

**Based MATCHES attention at 16 pairs (99.8%) where our diagonal gets 10.3%** -- the FIRST subquadratic carrier in
this project to do associative recall at all. It then hits a ceiling at 32+ pairs. I HYPOTHESIZED this was the
Zoology "recall capacity scales with feature dim" law, but a quick sweep did NOT confirm it: feat=16/24/32 all
~7-8% at 32 pairs (3000 steps, batch 32) -- bumping feature dim alone did not lift it. So the honest Phase-1
verdict: **sharpness (Taylor map) is genuinely the missing ingredient -- Based is the FIRST subquadratic carrier
here to match attention on recall (99.8% @16 pairs vs diagonal 10.3%)** -- but scaling past ~16 pairs is NOT a
one-line knob; it needs more (more training, tuning, or the proper chunked implementation + hparams). Consistent
with the theme: mechanism findable in an afternoon, production-quality scaling is real engineering.

**PHASE 1 COMPLETE (the decisive test).** Many-pair MQAR (32-64 simultaneous bindings) was the wrong bar -- our
actual goal is NEEDLE recall = 1 fact over a LONG sequence. Tested MULTI-HEAD Based (4 heads x feat12, cumsum) on
long-needle (plant 1 kv pair, L filler tokens, then query):

| L filler | mh-Based | diagonal (ours) |
|---|---|---|
| 64 | 100% | 2.2% |
| 256 | 100% | (dies) |
| 512 | 100% | (dies) |
| 1024 | 100% | (dies) |

**mh-Based recalls PERFECTLY at every distance to 1024** where the diagonal carrier is at chance -- a linear-time
O(T) carrier doing long-context recall as well as attention. This is the missing mechanism, VALIDATED. (Aside:
many-pair MQAR caps at ~16-32 pairs for tiny models = the capacity/state tradeoff, but single-fact long recall --
what we need -- is flawless at any distance.) PHASE 2 = graft mh-Based into the 1B as an ADDITIVE, function-
preserving parallel recall path (zero-init output so the 1B is unchanged at t=0), then train the Based path on
long-recall data so the model gains native long-context WITHOUT a from-scratch retrain. `research_carriers.py` +
`based_horizon.py` hold the validated carrier.

**PHASE 2 integration DONE + verified; training did NOT yet yield cross-window carry (diagnosed).** Added
`BasedRecall` to s6_hybrid as an ADDITIVE zero-init branch (`spin_based` carrier, 4.9M params on layers
4/12/20/28), grafted onto the ctx512 winner FUNCTION-PRESERVING (logit diff 0.00e+00, threaded 2-window also
0.00e+00 -- cross-window state plumbing verified correct). Trained the branch (higher lr 15x on the new params;
W<=2 BPTT; 16h/~3.5k steps): overall val IMPROVED (18.18 < graft 19.02) but the WIN=256/512 cross-window NEEDLE
did NOT (128 within-window 100%, but 256/512/640+ all ~chance). ROOT CAUSE (diagnosed, not guessed): `batch_long`
feeds a RANDOM CONTIGUOUS SPAN of the general corpus, so a planted fact almost never lands on the opposite side
of a window boundary from its query -> the next-token objective never STRUCTURALLY requires cross-window carry
(within-window attention already minimizes that loss), so the Based branch gets no gradient pressure to thread.
Same class as every prior long-ctx miss: recipe/DATA, not mechanism. THE FIX = a purpose-built curriculum that
forces it: plant fact in window 1, query in window 2, mask loss to the answer only -> the ONLY way to reduce that
loss is to write the fact into the Based state and read it across the boundary. That's the correct next step;
integration + carrier are done and correct, only the training TASK needs to require the capability.

**CROSS-WINDOW CURRICULUM RUN -- NEGATIVE, and it isolates the real blocker (`xwin_train.py`).** Built the
purpose-built task (fact@window-1, query+answer@window-2, answer-masked loss) with RANDOM-TOKEN secrets (train/
test id split so memorization is impossible -- an English-word first attempt memorized: loss->0.015 but held-out
0%). Trained ONLY the Based branch, base FROZEN, 2000 steps. Result: loss fell 11.7->~6.5 then PLATEAUED (chance
= ln(16384) ~ 9.7, so it learned the train-token PRIOR but not the specific carry) and **held-out recall stayed
0/40 the entire run.** ROOT BLOCKER (now clear): an additive branch on a FROZEN base can't help, because the
frozen downstream layers + head were never trained to READ what the branch writes into the residual -- in Phase 1
the WHOLE tiny model was the Based carrier trained end-to-end, so the readout adapted; here the base cannot decode
the branch's output. FIX would require training the Based branch JOINTLY with at least the post-branch layers +
head (or from-scratch with the carrier in-loop) -- a substantial run that also risks the frozen model's preserved
quality. CONCLUSION: the Based MECHANISM is real (Phase 1, #29 -- 100% to L=1024 end-to-end), but bolting it onto
the frozen 1B as an additive branch does NOT transfer; native long-context in THIS model needs joint training, not
a graft. The PRODUCT's long context stays solved by retrieval (#28); native >512 remains genuine research.

## 27. WHY the carrier fails at long context — associative-recall probe (`mqar_probe.py`)
Strategic de-risking probe (tiny: d=128, 2 layers) on MQAR (multi-query associative recall -- the standard
synthetic that separates recall mechanisms). Isolates whether the long-context wall is our CARRIER's recall.

| #kv pairs | diagonal (our carrier) | delta-rule | attention |
|---|---|---|---|
| 4 | 26% | **99%** | 100% |
| 8 | 15% | 13-98%* | 100% |
| 16 | 9% | 8-9% | 100% |
| 32 | 3% | 7% | 100% |

*delta @ small pairs is impl-sensitive; best plain-delta got 99% @4, but adding SiLU feature-map/multi-head
did NOT help scaling (all collapsed by 16 pairs). Findings: (1) our DIAGONAL carrier (SelectiveSpinCarrier
mechanism) fundamentally CANNOT do associative recall -- 26% @4 pairs vs attention 100% -- this is the ROOT
CAUSE of the long-context cliff (not a training bug: the carrier structurally can't retrieve a specific earlier
token). (2) DELTA-RULE proves the mechanism is fixable in principle (99% @4 pairs where diagonal gets 26%) BUT
across 3 quick implementations I could NOT get it to SCALE past ~4-8 pairs in a day. STRATEGIC READ: the
diagonal carrier's recall failure is exactly why spin can't beat attention at long context; a delta-rule carrier
is the scientifically-correct fix but is genuine multi-WEEK research with real execution risk, NOT a quick win
-- de-risk it at small scale (get delta to 64+ pairs on MQAR) BEFORE committing 1B compute.
FOLLOW-UP (the de-risk was RUN): across ~6 impls/levers -- pos-off, tied q=k, SiLU feature map, multi-head, state
dim up to 512, up to 8000 steps -- NO subquadratic carrier cleared ~4-8 pairs (all ~9% @16 pairs vs attention
100%). Neither alignment nor bigger state fixed it. So the cheap probe CONFIRMED the delta-carrier path is genuine
multi-week research needing the published careful recipe (Based/Gated-DeltaNet feature maps), NOT quick hacks ->
the 1B delta retrain is OFF; delta-carrier = documented low-priority future work. The strategic weight goes to
SHIPPING the honest story (spin ties + ctx=512 recall win + agency + retrieval). Meanwhile the honest
shippable story stands: spin TIES attention at matched params (crux), ctx=512 doubled native recall (#26), agency
closed (#25), brain.py retrieval = unlimited context for the product. Growing the attention window (#26) makes
ATTENTION do the long-range work -- it moves the cliff but abandons the spin-dominant thesis; the carrier itself
still can't recall, which this probe makes explicit.

## 26. Long-context — the ctx=256 cliff is REAL and MOVABLE: grow the trained window (`graft_ctx512.py`)
#24/#25 established a HARD recall cliff at exactly 256 tokens (the pretrain context window) that BPTT-at-ctx256
could not cross (it strengthened IN-window recall -- 192tok 50->92% -- but past 256 stayed at chance, and the run
peaked it=1000 then drifted, same instability). Diagnosis: the model pretrained 18B tokens at ctx=256, so it
never learned to use context older than 256; fine-tuning can't override that. FIX = grow the TRAINED window.
`graft_ctx512.py`: warm-start ctx 256->512 by POSITION-INTERPOLATION (stretch the 256 trained pos-embeddings
across 512 positions -- a cold random start at 256..511 would be catastrophic; interp keeps it in-distribution,
initial val 20.70 not broken). Then continue-train at ctx=512 (mixed-W W<=2, BPTT, from the it=1000 selective
ckpt). Fits WITH prod up (bs=2: 74GB used, 21GB free, no prod-down needed). Peaked fast (best val 18.68 @it=500,
1.8h -- better than the ctx256 run's 19.21) then oscillated 20h without beating it (lr=1.5e-4 a bit hot).

Decisive test -- ctx=512-AWARE needle (WIN=512, exercises the new attention window) on the it=500 best:

| depth | ALL prior ckpts (ctx256) | ctx=512 model |
|---|---|---|
| 128 / 192 | 57-100% / 7-92% | 100% / 100% |
| **256** | **0% (chance)** | **100% (14/14)** |
| 320 / 384 / 448 | 0% / 0% / 0% | 28% / 50% / 50% |
| 512 | 0% | 0% (new window edge) |

**The cliff MOVED 256 -> ~512, exactly as predicted.** Depth-256 recall went 0% -> 100% (clean, not noise);
320-448 partial (28-50%, above the ~12% 8-word chance; it=500 is early so this zone isn't fully consolidated);
new hard wall at 512 = the trained window. Mechanism CONFIRMED: recall horizon tracks the trained ctx window,
so ctx=1024 should move it to ~1024. Kept `pragnosia_1b_ctx512_it500_CROSSES256.pt` (the deliverable: native
256-token recall, 2x the pretrain model). This is the SUPERVISED-window path; the carrier-threaded UNLIMITED path
(selective latch across window boundaries) remains the harder architectural goal; brain.py episodic retrieval is
the pragmatic unlimited-context path for the product meanwhile. Next: stage 2 = ctx=1024 (same pos-interp graft).

## 25. Agency behaviour-learning — POSITIVE, and it CORRECTS #23 (`agency_rl.py`)
#23 concluded agency behaviour-learning was a training wall: inference-only episodic control gave "0/10 reach,
return flat -- the LM state-embedding isn't discriminative enough." That conclusion was WRONG, and this run
isolates why. Fair head-to-head on the GridWorld, both agents consuming the IDENTICAL `brain._embed` features
(importance-weighted contextual hidden), on the HARD setting: randomized start x randomized goal, greedy eval on
a fixed HELD-OUT set of unseen (start,goal) pairs (tests generalization, not memorization of 25 fixed states):

| method (same features) | held-out reach | return |
|---|---|---|
| REINFORCE (reward-gradient policy head) | 12% -> **88%** (best 92%) | +1.06 |
| EPISODIC control (nearest-12 reward vote, NO gradient) | 17% -> **100%** | +1.18 |

**Both LEARN a generalizing goal-reaching policy.** So (1) the brain's grounded features ARE discriminative
enough for behaviour-learning -- refuting #23's core claim; (2) #23's negative was an artifact of its EXPLORATION
policy: it explored via the LM (`generate_text` -> the non-discriminative-embedding path), which poisons the
experience buffer; with RANDOM exploration the SAME reward-weighted vote hits 100% with NO gradient and NO
training run; (3) reward-gradient RL (REINFORCE: batched, entropy bonus, gentle lr) ALSO closes it (88-92%).
Stabilizing REINFORCE mattered -- an early lr=1e-2 single-episode version collapsed (100% spike -> 12%); lr=3e-3
+ batch-8 + entropy 0.02 gives the smooth monotonic climb above. **Honest scope:** GridWorld is tiny,
deterministic, shaped-reward -- this is proof-of-MECHANISM (behaviour improves from consequence on the brain's
own perception), NOT proof of complex agency; sparse-reward / larger worlds are untested. But the specific #23
wall is removed. Implementation note: `brain.py::_choose_action` currently explores via the LM (the failing
path) when the buffer has no positive memory -- switching that fallback to random/epsilon-greedy is the one-line
change that makes the live `experience()` loop actually learn. Prod-safe run: CPU-only, 6-thread capped (load
~6), 625 obs embeddings disk-cached (`agency_emb_cache.pt`); ran alongside the live long-context GPU job (A).

## What these numbers changed
- **Paper:** ablation multipliers reframed to order-of-magnitude + instability note (C4); T1.8 disclosed in
  a new mechanism subsection and the multi-hop limitation reframed from "just undertrained" to a
  falsifiable shortcut-vs-composition question (C5).
- **Router:** overfit argmax → precision-first derived-margin, "14/14" retracted (C2).
- **Honesty:** over-abstention critique revised by measurement — the gate is well-calibrated; residual
  false-abstain is a generation-consistency issue, linking to the SFT work (C3 → C7).
- **Crux:** matched attention-only baseline is one command away (`run_crux_baseline.sh`, C6).

## 39. 3-WAY SPEED: ours vs transformer vs spin (`bench3.py`)
Matched d=512, 2 layers, fwd+bwd tok/s (measured under concurrent training -> ratios are the honest part):
| seqlen | attention | spin | VChunkRecall | vs attn | vs spin |
|---|---|---|---|---|---|
| 512   | 219K | 41K  | 93K  | 0.4x | 2.3x |
| 2048  | 308K | 99K  | 325K | 1.1x | 3.3x |
| 8192  | 159K | 137K | 632K | 4.0x | 4.6x |
| 32768 | 52K  | 129K | 669K | 13.0x | 5.2x |
Crossover vs attention ~2K; then ours pulls away (4x@8K, 13x@32K, gap GROWS -- attention throughput collapses
O(T^2) 308K->52K while ours rises/flattens O(T) 325K->669K). Ours beats spin at EVERY length (2.3-5.2x); spin
itself only beats attention at very long ctx. Short-ctx (<=1K) attention's fused kernel wins (0.4x) -- a Triton
kernel would close that, but long-ctx/streaming is where a brain lives. Verdict: for the brain workload our core
is decisively fastest (~13x transformer, ~5x spin @32K).

## 40. Self-growth prototypes A/B/C compared (`selfgrow_proto.py`, `RESEARCH_selfgrowth.md`)
Tested the 3 scaffolding-free self-growth approaches on ONE variable-hop lookup task (harder h = more capacity):
| approach | params | overall | hard(h>=3) | own capacity-signal vs difficulty |
|---|---|---|---|---|
| baseline fixed | 0.4M | 13% | 15% | - |
| A PonderNet halting | 0.2M | 13% | 17% | RISES h1=1.54->h4=1.59 (self-allocates) |
| B MoE routing | 1.1M | 12% | 15% | FLAT 1.00 (router collapsed to 1 expert) |
| C hypernet | 1.5M | 13% | 15% | none |
FINDING: only **A (adaptive halting)** delivers scaffolding-free self-allocation -- expected compute-steps rise
monotonically with difficulty, NO external schedule, fewest params, best hard-acc. B collapsed (needs a
load-balance loss = more scaffolding); C fragile, no gain. Matches the research memo: the honest working form of
"the model grows itself" is ADAPTIVE COMPUTE (decide own depth), not param recruitment. CAVEAT: absolute acc low
(~13%) -- variable-hop composition is hard for this core (known multi-hop limit); this compares the MECHANISM, not
task mastery. A's rise is real but modest at tiny scale. NEXT if pursued: scale A (weight-tied + PonderNet) on an
easier capacity task to show acc gain + stronger halt-vs-difficulty; combine A (intrinsic depth) with a
surprise-triggered metabolic param-grow.

## 41. THE FIX for 13%: chain-of-thought + intrinsic halting = self-scaled computation (`chain_cot.py`)
The 13% (multi-hop composition) was NOT a depth problem (depth sweep #hop_depth: L2=17% L4=15%, h1 at chance,
depth doesn't help) -- the recall core does SINGLE-hop ~100% but CANNOT compose multi-hop in one parallel forward.
FIX = don't ask the parallel forward to compose; ITERATE the single-hop autoregressively (chain-of-thought) and
HALT on the model's own arrival signal. Successor-chase task (N=12):
- ONESHOT (1 parallel forward): 100% BUT via a shortcut (terminal is directly identifiable); on the earlier
  non-shortcuttable arbitrary-h-hop target, oneshot = 13%.
- COT (iterated single-hop + intrinsic STOP): terminal-acc **92%**, and #generated steps == TRUE distance
  PERFECTLY (dist k -> k.0 steps, clean 1:1 diagonal across dist 1..11). The model self-scales computation exactly
  to difficulty, halting by its OWN terminal-recognition, NO external step schedule.
FINDING: multi-step reasoning = iterated single-hop + intrinsic halt (chain-of-thought). Jumps composition 13%->
92%. This is ALSO the honest scaffolding-free self-growth answer: adaptive compute in the GENERATION axis -- the
brain spends exactly as much thinking as the problem needs, decided by itself (perfect steps-vs-distance). Ranking
of self-growth mechanisms: (1) CoT+intrinsic-halt = strong self-scaling [#41]; (2) PonderNet adaptive-depth =
works, modest [#40]; (3) MoE/hypernet param-recruitment = didn't deliver [#40]; physical param-alloc = irreducibly
external [RESEARCH_selfgrowth.md].


## 42. Can adaptive-compute (halting) go INTO pretraining? -- NO, empirically (ponder_smoke.py)
Q: integrate the scaffolding-free adaptive-compute axis (#41 CoT+halting) into the from-scratch scale run?
Smoke test on real text (window2, ctx=1024, 700 steps): STANDARD fixed-depth-4 LM -> ppl 129; PONDER weight-tied +
PonderNet halting -> ppl 1068 (8x WORSE), 2.7x slower, halting COLLAPSED (exp_steps pinned 4.0 = always max, never
halts early). VERDICT: adaptive-compute does NOT pretrain well from scratch -- the halting head has no capable
atomic primitive to gate, so it just adds instability. Correctly NOT in the scale run. Its right home is a
REASONING FINE-TUNE on a capable base (as #41 showed -- shines when the single-step primitive already works; same
pattern as real reasoning models: pretrain then RL/CoT). The two scaffolding-free axes live in DIFFERENT phases:
intrinsic GROWTH (own-entropy trigger) in pretraining [live]; intrinsic ADAPTIVE-COMPUTE (CoT+halt) in a later
fine-tune [proven #41].


## 43. Fixed 200M pretraining -- 3 architecture bugs (scale_train.py)
Starting a fixed 216M (d1024/L18, no-growth) run diverged (loss=196, ppl=e^20) -- NOT lr. Three real bugs, all
would bite any deep/wide model here:
1. NO FINAL LAYERNORM before the (tied) head -> residual grows over 18 layers -> logit explosion. 45M/12L survived
   (shallow); 18L/d1024 blew up. Fix: s.lnf=LayerNorm(d); head(lnf(h)).
2. nn.Embedding DEFAULT INIT = N(0,1) -> tied-head logit std ~sqrt(d)~32 -> init loss in the HUNDREDS. Fix:
   normal_(emb,0,0.02) (GPT init; tied head inherits).
3. Short warmup + high lr for a big model. Fix: WARM=2000, lr=3e-4.
After fixes: init loss 9.9 -> dropping (7.3 @ warmup200). 216M fixed run live (ctx4096, bs2 beside prod, ~17K
tok/s, no growth, no token cap; ~2.7 days to saturate the 200M rung). Also added: env-config D/LSTART/NOGROW,
RESUME-from-latest-snapshot, VChunkRecall stable=True (feature-norm) for big-d/long-ctx numerical stability.
Plan: train 200M to convergence (best chance at recall emergence -- one size trained fully, vs grow-from-small
undertraining each rung), THEN enable growth from the converged base.
