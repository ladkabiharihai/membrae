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

## 0b. Re-bake the identity (paper-aligned) into the weights
The runtime self-model (`brain._derive_self`) is already paper-aligned (spin-dominant / diagonal-complex /
load-bearing / fast-slow). But the model's GENERATED self-intro still says the old "small recurrent language
model" because that lives in the WEIGHTS. `pragnosia_self.txt` + `identity_sentences.txt` are updated
(spin-dominant, fast-slow, load-bearing core; honesty kept). The currently-running SFT does NOT include these
edits (its corpus was tokenized before). So, on the NEXT bake:
```
python3 prepare_sft.py          # re-tokenize -> picks up the updated identity corpora
# then include it in the next SFT / continued-training run so the paper-aligned persona bakes into the weights
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
STAGED PLAN (test cheap at 284M first, scale to 1B ONLY on success):

STAGE A -- test the coupling at 284M (cheap, clean control, base co-adapts):
```
CONFIG=pragnosia_coupled_284m.json python3 init_coupled.py pragnosia_284m_fair.pt   # warm-start (recipe verified)
CONFIG=pragnosia_coupled_284m.json python3 train_pragnosia.py --resume --steps 30000 --lr <fair-spin lr>
# CONTROL (required): continue the SAME fair spin base for 30000 steps, no coupling, same data/lr.
#   CONFIG=pragnosia_284m_fair.json python3 train_pragnosia.py --resume --steps 30000
# compare COUPLED-final vs CONTROL-final (NOT vs coupled@init): val ppl + multi-hop + needle.
# WIN = coupled beats the base CONTROL at matched steps. If neutral/loses -> honest negative, STOP (don't
#   waste 1B compute; the frozen-1B probe was already neutral, so 284M is the decider).
```

STAGE B -- ONLY if Stage A wins: scale 284M->1B with coupling.
```
# grow the WINNING coupled-284M to 1B (function-preserving growth) + mixed-window continued training.
# CAVEAT: grow.py must carry the CoupledBlock's extra params (to_slow/slow/from_slow) through a grow step
#   -- verify it does before launching (a plain layer/width grow may drop them). Flag if it needs a patch.
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

## 5. Continue training the 1B further (long-context carry convergence) -- READ BEFORE RESUMING
When resuming continued training of the 1B to strengthen the cross-window carry, apply these three rules
(they matter more than the raw token count):

1. **Raise the LR first, or it won't learn the new objective.** At 7.5e-5 with the schedule floored, the
   cross-window carry will only crawl in. Learning a genuinely new capability (writing/using cross-window
   state) wants ~1-2e-4 -- let the LR warm back up rather than resume at the floor. This is the single
   biggest lever, more than the token budget.

2. **Train to convergence of the long-context metrics, not a fixed step target.** Eval every ~50-100K steps
   on W1/W8/W32 ppl + the needle curve, and stop when BOTH:
   - W32 ppl and the needle curve stop improving for 2-3 consecutive evals, AND
   - W1 ppl has NOT regressed vs the 18.02 base (short-context skill intact -- the 30% W1 in the mix
     protects this; watch it).

3. **Ballpark budget:** on the order of 1-3B additional tokens for the carry to converge (the 706M learned
   it in a comparable budget). At the shared rate ~20K tok/s that's ~14h-2 days; free the GPU of the
   inference stack (~100-200K tok/s) and it's a few hours.

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
