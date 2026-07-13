# Zenodo upload — Pragnosia paper

**File to upload:** `pragnosia_nature.pdf` (Nature single-column format, line-numbered, all measured numbers).
Source: `manuscript_nature/main.tex` (recompile with `~/bin/tectonic main.tex`). The SAME PDF is the NMI
submission manuscript and the Zenodo preprint (NMI is single-blind, so the named-author PDF is correct for
both). IEEE format has been dropped per the decision to target Nature Machine Intelligence only.

## Suggested Zenodo metadata

- **Upload type:** Publication -> Preprint
- **Title:** Pragnosia: Making a Diagonal-Complex Spin Recurrence the Load-Bearing Core of a Language Model
- **Authors:** Kumar, Ashish (Independent Researcher)
- **Contact email:** ashish99anonymous@gmail.com
- **Description (abstract):**
  > A transformer/state-space hybrid in which a rotational "spin" recurrence (a diagonal-complex
  > linear-recurrent unit run as a parallel associative scan) is the core token-mixer, with attention only
  > a periodic helper. A single carrier bolted after a transformer stack drifts to a causally-negligible
  > side-channel as the model scales (0.028% of the loss at 1.4B); a parameter-matched ablation shows the
  > intended spin-dominant design wins by ~21% at 37M; and at 284M, ablating the carrier is catastrophic
  > (two-to-three orders of magnitude in perplexity) versus 3.4x for attention, establishing the carrier as
  > the load-bearing core. Numbers are reported as measured, with the exact ablation multiplier treated as
  > order-of-magnitude only (a catastrophically ablated model's perplexity is numerically unstable). The
  > open crux (a matched attention-only transformer at 284M) is stated as the key experiment still to run.
- **Keywords:** state-space models; linear recurrence; sequence modeling; language models; causal ablation;
  mechanistic interpretability; continual learning
- **License:** CC BY 4.0 (recommended for a preprint) or MIT if bundling code
- **Language:** English
- **Version:** v1 (2026-07)

## Honest status note (for a possible v2)
- Numbers reflect the checkpoints as of the writeup: the 1B progression row is the 20.1-ppl snapshot. The
  newer H100 snapshot (val ppl 18.02) and the in-progress replay-mixed SFT are NOT yet folded in; fold them
  into a v2 once the SFT lands and is promoted.
- The single highest-value addition for v2 is the crux baseline (matched carrier="none" transformer at
  284M, `run_crux_baseline.sh`) -- it converts "the carrier is used" into "the design wins."
