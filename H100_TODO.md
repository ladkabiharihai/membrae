# H100 runbook — run these when the GPU frees (SFT done) — NMI target

Ordered by scientific value. The H100 is READ-ONLY for the laptop session; these are for the on-box
session to execute. All code + configs are committed and tested on the laptop (tiny-config smoke tests pass).
Compare every result to `RESULTS_MEASURED.md` and write new numbers into `eval_registry/`.

Precondition: GPU free of the inference stack (VLLM/parakeet/whisper/kokoro ~90 GB) OR a window where it is.
At the shared-GPU rate (~20K tok/s) training crawls; free the card first.

---

## 0. Promote the SFT'd model (SFT already running -> pragnosia_sft.pt, 15% replay)
When SFT reaches ~1 epoch / converges:
```
# laptop will pull pragnosia_sft.pt and run: eval_battery / router_heldout / gate_measure / mech_probe
# promote criterion: knowledge holds (1-hop battery not < base), gate known-answer recovers toward 0.75,
# router negative-specificity stays ~0.83. If so, set pragnosia.json ckpt -> pragnosia_sft.pt.
```

## 1. THE CRUX (highest value) — matched attention-only transformer @284M
Turns "the carrier is used" into "spin-dominant WINS". One command; corpus/tokenizer/budget already matched
(carrier=none, 21 layers = 288.5M, +1.4% of the 284M spin model):
```
STEPS=418000 bash run_crux_baseline.sh
# then: CONFIG=pragnosia_baseline.json python3 eval_all.py crux_baseline
#       CONFIG=pragnosia_baseline.json python3 eval_battery.py crux_baseline
# WIN CRITERION: baseline val ppl >= 24.54 (the 284M spin ppl) => spin-as-core wins at scale.
#   If baseline is clearly lower, report it honestly (ablation was necessity, not superiority).
```

## 2. Multi-seed the 284M causal result (single-seed -> a property)
Retrain the 284M spin-dominant at 2-3 seeds; re-run the ablation each time. Reviewers demand this.
```
# for SEED in 1 2 3: train the 284M spin config (d=768,20L,mlp9,carrier=spin_dominant) with that seed,
#   then ablation_sweep.py / ablate_matched.py -> confirm carrier ablation catastrophic every seed.
# Report mean +/- range of the carrier/attention gap (the ORDER-OF-MAGNITUDE gap, not exact multiplier).
```

## 3. Fast-slow COUPLING experiment (the DMP-inspired upside; could make it "a principle")
Does letting the slow carrier MODULATE the fast attention pathway beat the parallel design? Warm-started, so
cheap. Code: `s6_hybrid.py` CoupledBlock + carrier="spin_dominant_coupled" (smoke-tested, 0.1% param overhead,
inits near-identity).
```
python3 init_coupled.py pragnosia_sft.pt          # or pragnosia_spin.pt -> warm-start pragnosia_coupled.pt
CONFIG=pragnosia_coupled.json python3 train_pragnosia.py --resume --steps 30000
# CONTROL: continue the SAME base for 30000 steps too (no coupling), same data/lr.
# compare: CONFIG=pragnosia_coupled.json python3 eval_battery.py coupled
#          (base) python3 eval_battery.py base_control
# WIN: coupled val ppl / multi-hop / long-ctx beats the control at matched steps -> a fast-slow contribution.
# If no gain, report as a negative result (honest) and keep the parallel design.
```
NOTE (laptop probe already run): a FROZEN-base coupling probe on the 18.02 snapshot was NEUTRAL (19.005 vs
true base 19.046, +0.04 within noise; eval_registry/coupled_probe.json). That is a lower bound -- a frozen
base cannot co-adapt to the slow modulation -- so it does not decide the idea; the full fine-tune above is the
real test. The coupling init is now EXACT identity (gain=1 via 1+tanh), so the fine-tune starts precisely at
the base and any ppl change is purely the coupling's doing.

## 4. Long-range evidence (NMI expects it) — re-run on the promoted / crux models
```
python3 needle_eval.py <ckpt>          # retrieval vs context length (harness done; laptop baseline below)
python3 eval_battery.py <label>        # ppl W1/W8/W32 + multi-hop + activation probe
```
Laptop baseline already measured on the 18.02 snapshot (eval_registry/needle_pragnosia_spin_h100.json):
retrieval by length L128/256/512/1024/2048 = 0.47/0.13/0.13/0.07/0.07 -> sharp retrieval falls off past the
256 train window (supports the paper's "gist not verbatim" limitation). If the coupling/longer-ctx training
lifts this curve, that is a real long-context result to report.

---

## Paper status (laptop side, done, no GPU)
- Reformatted to **Nature single-column** (manuscript_nature/, pragnosia_nature.pdf). IEEE dropped.
- Theory: "Properties of the carrier" (Props 1-2 cited to LRU/S4D as background; Prop 3 = phase-decoupling,
  ours, framed as a falsifiable phase-vs-context prediction).
- Positioning: cite the NMI review (tiezzi2025recurrent) + the DMP fast-slow paper (sun2026dualmemory);
  cast the paper as the causal test of the fast-slow hypothesis at LM scale.
- Needle harness + coupling architecture prepared and tested.

## When results land, fold into the paper
- Crux ppl -> the crux row + retract the "single highest-value experiment still to run" caveat if it wins.
- Multi-seed -> "single-seed" limitation removed; report mean +/- range.
- Coupling -> if it wins, a new fast-slow result section; if not, an honest negative in limitations.
- Needle curve + long-ctx -> the long-context subsection (quantifies the gist-not-verbatim limit).
