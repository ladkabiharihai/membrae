# Theory draft — properties of the diagonal-complex spin carrier

Purpose: close the Tier-1 "one theoretical result" gap the editor flagged, honestly. Three modest but
correct propositions about the carrier. Two are unconditional (stability, memory horizon); the third ("why
phase") is framed as a falsifiable prediction that is *consistent with* our current empirical finding that
at the tested 256-token budget placement dominates and the real-carrier control also works.

Carrier recurrence (per channel j, complex state h):
    h_t = λ_j h_{t-1} + b_t,   λ_j = exp(-exp(ν_j)) · exp(i θ_j),   ν_j, θ_j ∈ ℝ learned.
Write r_j := |λ_j| = exp(-exp(ν_j)).

---

## Proposition 1 (Unconditional stability)
For every finite ν_j, r_j ∈ (0,1) strictly; hence every pole lies inside the open unit disk and the carrier
is BIBO-stable *for all parameter values*, with no constraint, clipping, or projection.

**Proof.** ν_j ∈ ℝ ⟹ exp(ν_j) ∈ (0,∞) ⟹ r_j = exp(-exp(ν_j)) ∈ (0,1). Unrolling from h_{-1}=0,
h_t = Σ_{k=0}^{t} λ_j^k b_{t-k}, so |h_t| ≤ Σ_{k≥0} r_j^k |b_{t-k}| ≤ (sup_s |b_s|)/(1-r_j) < ∞: bounded
input ⟹ bounded state. The BPTT sensitivity ∂h_t/∂b_{t-k} = λ_j^k has modulus r_j^k, which is summable, so
gradients through the associative scan can neither explode nor accumulate unboundedly. ∎

**Why it matters.** The `exp(-exp(·))` parametrization makes stability a *structural guarantee*, not a
training outcome. This is the mechanistic reason the design trains cleanly at 284M–1B where an
unconstrained real recurrence would risk an escaping eigenvalue. (Contrast: an additive/tanh recurrence has
no such guarantee.)

---

## Proposition 2 (Per-channel memory horizon)
The lag-k influence of an input on the state decays as r_j^k. Defining the memory horizon τ_j as the lag at
which this magnitude falls to 1/e gives τ_j = exp(-ν_j). Because ν enters through a double exponential, a
bounded parameter range ν_j ∈ [a,b] realizes horizons τ_j ∈ [e^{-b}, e^{-a}] spanning e^{(b-a)} in ratio,
i.e. many orders of magnitude of timescale in one channel bank.

**Proof.** From Prop 1 the contribution kernel is w_k = r_j^k = exp(-k exp(ν_j)). Solving w_τ = e^{-1} gives
τ_j exp(ν_j) = 1, so τ_j = exp(-ν_j), monotone decreasing in ν_j; the range statement is immediate. ∎

**Why it matters.** A single learned scalar per channel places that channel anywhere on a logarithmic
timescale axis; the bank is therefore a *learnable multi-timescale memory*, which is exactly the property a
fixed convolution or a single-timescale RNN lacks. E.g. ν ∈ [-4,4] ⟹ τ ∈ [0.018, 54.6] steps.

---

## Proposition 3 (Phase is a lossless isometry: retention and indexing decouple)
Write λ_j = r_j e^{iθ_j}. The per-step update factors into a real attenuation by r_j (the only lossy part)
and a rotation z ↦ e^{iθ_j} z, which is a norm-preserving isometry of ℂ. Consequently a lag-k input appears
in the state as r_j^k e^{i k θ_j} b: same magnitude decay as a real recurrence, but rotated to phase kθ_j.
Inputs at distinct lags k ≠ k′ with kθ_j ≢ k′θ_j (mod 2π) occupy distinct phase angles and remain linearly
separable by the read-out C even when r_j^k ≈ r_j^{k′}. A real recurrence (θ_j = 0) collapses every lag onto
the real axis, discarding the lag index. Hence at matched decay r_j the complex channel has strictly greater
recoverable capacity.

**Proof.** |e^{iθ} z| = |z| and e^{iθ} is invertible (inverse e^{-iθ}), so the rotation neither contracts
nor loses information; only the scalar r_j < 1 attenuates. Over k steps the state term is (r_j e^{iθ_j})^k b
= r_j^k e^{ikθ_j} b. Two lags map to r_j^k e^{ikθ_j} b and r_j^{k′} e^{ik′θ_j} b′; if the phases differ these
are linearly independent directions in ℂ ≅ ℝ², recoverable by a linear C, whereas at θ_j = 0 both lie on ℝ
and, when r_j^k ≈ r_j^{k′}, are near-collinear. ∎

**Honest framing (this is the key move).** Prop 3 does *not* claim phase is necessary or that it wins at any
budget. It predicts a phase advantage that **grows with the required memory horizon / context length**. This
is fully consistent with — and sharpens — our empirical result that at the tested 256-token budget the
real-carrier control (`real_dominant`) also beats attention, so "placement, not phase" holds *at that
budget*. The theory turns this into a falsifiable experiment: **train spin vs real carriers at increasing
context length; Prop 3 predicts the phase gap opens as horizon grows.** That is a concrete, honest next
experiment, not an overclaim.

**Remark (frequency view, connects to S4/LRU).** As r_j → 1 a channel approaches a pure oscillator
e^{iθ_j t}; the bank is thus a learnable *damped Fourier-like filterbank*, linking the carrier to the
HiPPO/structured-SSM lineage while making the damping and frequency per-channel and input-gated.

---

## How this maps to the editor's asks
- "Can you prove stability?" -> Prop 1 (unconditional, structural).
- "memory capacity?" -> Prop 2 (explicit horizon τ = e^{-ν}, multi-timescale).
- "why phase should preserve information?" -> Prop 3 (isometry; retention/indexing decoupling).
- "frequency decomposition?" -> Remark (damped filterbank, S4/LRU link).
- Bonus: Prop 3 yields a *new falsifiable experiment* (phase gap vs context length), which is itself a
  contribution and keeps the paper's honesty intact.

## Integration plan
Replace the informal "Why phase" subsection with a formal "Properties of the carrier" subsection carrying
Props 1–3 + the remark, keep the intuition as one-line glosses, add `\usepackage{amsthm}` +
`\newtheorem{proposition}`. ~0.4 column. Apply to `manuscript_ieee/main.tex` (source of truth), then
re-sync `manuscript_zenodo/`.
