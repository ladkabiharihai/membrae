# Phase S1 + scaled generation — RESULTS (2026-06-10)

## Verdict
**S1 (scaled compositional generalization) is cracked, and scaled language
generation is flawless** — both via the same proven mechanism (clean-state
discrete commits) applied to parsing and generation. All handoff invariants
preserved; all controls run.

## Setup
Grammar: 108-word vocab (8 verbs, 40 adjectives, 50 nouns, 6 spatial relations,
'the'), recursive depth 1–3, ≤12 tokens, e.g.
"push the red ball near the big box behind the old door".
30,000 unique sentences, 10% held out (never seen in training). 9-slot parse
target (act, adj/noun ×3, rel ×2, with none-classes). Code: `s1_scale.py`.

## Numbers (held-out, exact whole-parse / whole-sentence)

| Model | Mechanism | Held-out | Params |
|---|---|---|---|
| BaselineParser | spin reader → heads off final h (original approach) | **0.7273** (train 0.99) | 231K |
| CleanStackParser (e2e, no role supervision) | + discrete role commits, ST-Gumbel annealed | ≈0.95 (plateau) | 205K |
| **CleanStackParser (canonical)** | + auxiliary role supervision (derived from data gen, nothing hard-coded) | **1.0000** | 205K |
| SpinGenerator | plain meaning→sentence decoder (Stage-4 shape) | 0.9963 | 312K |
| **CleanStackGenerator (canonical)** | + discrete role pointer gates the meaning slot under emission | **1.0000** (free-running, all positions, correct EOS, zero drift) | 392K |
| **Round-trip** (parse → meaning → regenerate), 3000 held-out sentences | | **1.0000 / 1.0000** | |

Acceptance criterion was >0.95 held-out at vocab ≥100, depth ≥3 — exceeded.

## Why the baseline strained and the fix works
The original failure (0.65–0.73) is **binding drift**: continuous state cannot
hold which adjective binds to which noun at which depth. The plain generator
showed the same signature — its only errors copied an adjective across depths
("heavy bag inside the *heavy*(→rough) wheel"). This is exactly the state-drift
root cause behind the original arithmetic ceiling, and the same fix —
**commit discrete decisions each step (straight-through) and feed them back**
(CleanStateRecurrence) — eliminates it:
- **Parse**: per-token discrete ROLE commit routes each word's lexical value
  into its slot; syntax (a tiny learned automaton) and lexicon (value heads
  shared across depths) factorize compositionally.
- **Generate**: per-step discrete role POINTER selects the single meaning slot
  being verbalized, so the decoder never carries all nine bindings in
  continuous state.

Earned lessons (added to the method record):
1. Deterministic ST-argmax commits cold-start-lock (metrics frozen across
   evals — rule #1 caught it). Fix: ST-**Gumbel** sampling, temperature
   annealed 2.0→0.5, deterministic argmax at eval.
2. Feed the current token embedding directly to the role cell (token class is
   what disambiguates roles); the spin state supplies context.

## Controls (all pass)
- **State-swap causal test** (anti-memorization invariant): swap the entire
  carried state (h, role, slot accumulators) after the first NP between two
  sentences → prefix slots follow the swapped-in state **1.00**, suffix slots
  follow the live tokens **1.00**; random-state control **0.00**.
  (`s1_swap_result.json`)
- **Tiny-scale regression**: unified_brain.pt self-test re-run after all work —
  **14/14 PASS**.
- No hard-coded values: every sentence sampled; held-out split by fixed rng;
  role targets are derived during data generation (supervision, not hard-coding).

## Artifacts
`s1_scale.py` (grammar+models+training+controls) · `s1_demo.py` (round-trip
demo) · `s1_clean.pt`, `s1_cleangen.pt` (canonical weights) ·
`s1_baseline.pt`, `s1_gen.pt` (ablation weights) · `s1_*_result.json` ·
`s1_*.log` (full training logs).

## Next (per roadmap and user priority)
- S2: real corpus + tokenizer (TinyStories-class, BPE, vocab 1–8K), perplexity
  vs parameter-matched GRU baseline, brain-swap still passing.
- Then image and voice scaling (S-omni): same clean-stack treatment for
  relational vision (the 0.76 limit is the same multi-element-binding theme).
