# Phase S2 — real-corpus language modeling (TinyStories) — RESULTS (2026-06-10)

## Verdict (honest)
The spinning substrate **learns real language**: fluent, grammatical free-running
generation for 150 tokens with ZERO positional drift, and the brain-swap probe
confirms continuations are carried causally by the spinning state. But the
acceptance gate **"perplexity competitive with a parameter-matched GRU" FAILS**:
best spin variant 8.89 vs GRU 6.55 (~36% higher), and the gap was robust to
gain-conditioning and depth. Recorded as a negative result per method rule #6.

## Setup
TinyStories (real corpus), 21.5M train / 6.0M valid BPE-2048 tokens,
context 128, identical data/steps/batch/lr per comparison. SpinStep dynamics
unchanged (W = −ρQQᵀ + skew, QR per forward). torch.compile gave 12×
throughput (52K → 718K tok/s) with no change to dynamics. Code: `s2_corpus.py`.

## Numbers (final, 80-batch val ppl)

| Model | Mechanism | Params | Val ppl |
|---|---|---|---|
| SpinLM | plain spin reader | 1.90M | 11.29 |
| GainSpinLM | + token-conditioned gain/bias (proven Stage-1 mechanism family) | 2.04M | 9.90 |
| GainSpin2LM | + second spin layer | 2.23M | **9.11** (best-ckpt 8.89) |
| GRU (cuDNN) | gated baseline | 1.96M | **6.58** |

Each proven-mechanic addition helped (11.3 → 9.9 → 8.9), trend ~–0.9 ppl per
addition, but extrapolation says the gap is mechanism-level (GRU's multiplica-
tive state gating), not budget-level: GRU matched spin's final ppl using ~10×
less compute.

## What works
- **Generation**: 150-token free-running samples are locally fluent and
  grammatical ("Once upon a time, there was a little girl named Mia…"); story
  thread drifts (names/objects switch) — a coherence weakness, not a fluency one.
- **No positional drift** (S3 invariant): teacher-forced NLL by quartile =
  2.29 / 2.15 / 2.15 / 2.13 — flat-to-improving over the full window.
- **State causality (brain-swap probe)**: continuation NLL with own state
  2.18 < swapped 2.37 < random 2.91 — the prefix lives in the state, causally.
- **Regression**: unified 14/14 PASS after all S2 work.

## Limits / next options
1. Coherence (name/thread persistence) is the visible failure of leaky
   integration; the clean-state mechanism (discrete commits) is the proven fix
   for binding drift — apply it to discrete story-state at the LM scale.
2. If ppl parity matters, the GRU-style multiplicative state gate would need
   adopting into the spin core — flagged for a decision, since it changes the
   core update (goal-preservation question, not a code question).

## Artifacts
`s2_corpus.py`, `data/bpe2048.json`, `data/{train,valid}.bin`,
`s2_{spin,gspin,gspin2,gru}.pt`, `s2_*_result.json`, `s2_eval.json`, logs.
