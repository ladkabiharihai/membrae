import os, torch, json, sys
import s6_hybrid as H
src = sys.argv[1] if len(sys.argv) > 1 else "pragnosia_1b_sft.pt"   # v4 final
dst = os.environ.get("GRAFT_DST", "pragnosia_1b_selective.pt")
CFG = json.load(open("pragnosia_1b_combined.json"))
sd = torch.load(src, map_location="cpu", weights_only=True)
m = H.SpinAttentionLM(16384, 768, 12, CFG["layers"], mlp_mult=CFG["mlp_mult"], carrier="spin_selective")
r = m.load_state_dict(sd, strict=False)
bad = [k for k in r.missing_keys if "delta" not in k]
assert not bad and not r.unexpected_keys, f"graft mismatch: missing={bad} unexpected={list(r.unexpected_keys)}"
# CORRECTED RECIPE (RESULTS #22): LATCH-BIASED delta init (default bias -2 -> |lambda|~1 = remember-by-default,
# LEARN to forget) instead of the function-preserving 0.5413 that failed to learn latching in attempt #1.
# This perturbs the graft at t=0 (the mixed-W run + 50% replay recover v4's quality) but is what teaches latching.
import torch.nn as nn
bias = float(os.environ.get("LATCH_BIAS", "-2.0"))
for blk in m.blocks:
    c = getattr(blk, "carrier", None)
    if c is not None and hasattr(c, "delta"): nn.init.constant_(c.delta.bias, bias)
torch.save(m.state_dict(), dst)
print(f"grafted {src} -> {dst}: spin_selective, LATCH-biased delta (bias={bias}) ({len(r.missing_keys)} delta params)", flush=True)
