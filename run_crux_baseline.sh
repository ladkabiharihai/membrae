#!/usr/bin/env bash
# ============================================================================
# THE CRUX EXPERIMENT (C6): a parameter-matched attention-only transformer at
# ~284M, same corpus / tokenizer / budget as the 284M spin-dominant model.
#
# This is THE experiment that turns "the carrier is USED" (306x ablation) into
# "the spin-dominant design WINS": if the matched transformer's validation
# perplexity is >= the spin model's 24.54, spin-as-core wins at scale; if it is
# clearly lower, the ablation was necessity-not-superiority and we say so.
#
# Matched-parameter derivation (computed, not guessed):
#   spin-dominant 284M : d=768, 20 layers, mlp_mult=9, carrier=spin_dominant -> 284.4M params
#   this baseline      : d=768, 21 layers, mlp_mult=9, carrier=none          -> 288.5M params (+1.4%)
#   (swapping SpinBlock->attention Block drops the carrier weights, so one extra
#    attention layer restores the count; 21 layers is the closest integer match.)
#
# EVERYTHING ELSE IS HELD IDENTICAL to the spin run for a clean comparison:
#   same vocab (16384), same tokenizer (data/bpe.json), same train/valid bins
#   (window2_train / big_valid), same context (256), same optimizer/schedule
#   (train_pragnosia.py derives its own lr; no hand-set seed), same token budget.
#
# NOTE: this needs the H100 (the 284M spin run was ~418k steps). It CANNOT run on
# the local 8GB card in reasonable time. Launch on the training box, READ-ONLY
# repo excepted. One command:
# ============================================================================
set -euo pipefail
cd "$(dirname "$0")"

STEPS="${STEPS:-418000}"          # match the 284M spin run's budget (override with STEPS=...)
export CONFIG=pragnosia_baseline.json

echo "[crux] matched attention-only baseline vs 284M spin-dominant"
python3 - <<'PY'
import json, s6_hybrid as H
c = json.load(open("pragnosia_baseline.json"))
H.VOC, H.L = c["vocab"], c["ctx"]
m = H.SpinAttentionLM(c["vocab"], c["d"], c["heads"], c["layers"],
                      mlp_mult=c["mlp_mult"], carrier=c["carrier"])
p = sum(x.numel() for x in m.parameters())
print(f"[crux] carrier={c['carrier']} d={c['d']} L={c['layers']} mm={c['mlp_mult']} -> {p/1e6:.1f}M params (target 284.4M spin)")
assert c["carrier"] == "none", "baseline must be attention-only"
assert abs(p - 284.4e6) / 284.4e6 < 0.03, "param match drifted >3%; retune layers"
print("[crux] param match OK (<3%). Launching matched training...")
PY

# same entry point, same self-derived lr / schedule / long-context curriculum as the spin run
python3 train_pragnosia.py --steps "$STEPS"

echo "[crux] done. Compare val ppl to the spin model's 24.54:"
echo "  CONFIG=pragnosia_baseline.json python3 eval_all.py crux_baseline"
echo "  CONFIG=pragnosia_baseline.json python3 eval_battery.py crux_baseline"
echo "  spin_dominant 24.54  vs  baseline <ppl>  ->  >=24.54 means spin-as-core WINS"
