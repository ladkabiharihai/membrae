"""Grow the TRAINED context window 256 -> 512 (the real fix for the ctx=256 cliff, RESULTS #24/#25).
The model pretrained at ctx=256, so pos-embeddings 256..511 are UNTRAINED (random) -> a cold jump to ctx=512
would be lost above 256. Warm-start via POSITION INTERPOLATION: stretch the 256 trained pos-embeddings across
512 positions (linear interp between neighbours), so training starts IN-DISTRIBUTION. Everything else (emb,
carrier, attention, head) carries over unchanged. Source = the best long-ctx checkpoint (it=1000)."""
import os, json, sys, torch, torch.nn as nn
import s6_hybrid as H
SRC = sys.argv[1] if len(sys.argv) > 1 else "pragnosia_1b_longctx_it1000_best.pt"
DST = os.environ.get("DST", "pragnosia_1b_ctx512.pt")
NEWCTX = int(os.environ.get("NEWCTX", "512"))
CFG = json.load(open("pragnosia_1b_longctx.json"))
sd = torch.load(SRC, map_location="cpu", weights_only=True)

# interpolate the pos-embedding table's first OLDCTX trained rows across NEWCTX positions
pos = sd["pos.weight"]                       # [4096, d] -- only rows 0..OLDCTX-1 are trained
OLDCTX = int(os.environ.get("OLDCTX", "256"))
trained = pos[:OLDCTX]                        # [OLDCTX, d]
# linear position-interpolation: new row i maps to fractional old index i*(OLDCTX-1)/(NEWCTX-1)
idx = torch.linspace(0, OLDCTX - 1, NEWCTX)
lo = idx.floor().long().clamp(0, OLDCTX - 1); hi = idx.ceil().long().clamp(0, OLDCTX - 1)
frac = (idx - lo.float()).unsqueeze(1)
interp = trained[lo] * (1 - frac) + trained[hi] * frac      # [NEWCTX, d] in-distribution
pos_new = pos.clone(); pos_new[:NEWCTX] = interp
sd["pos.weight"] = pos_new

# sanity: rebuild the model and load
m = H.SpinAttentionLM(CFG["vocab"], CFG["d"], CFG["heads"], CFG["layers"],
                      mlp_mult=CFG["mlp_mult"], carrier=CFG["carrier"])
r = m.load_state_dict(sd, strict=True)
torch.save(m.state_dict(), DST)
print(f"grafted {SRC} -> {DST}: pos-interp {OLDCTX}->{NEWCTX} (in-distribution warm-start), "
      f"carrier={CFG['carrier']}, {sum(p.numel() for p in m.parameters())/1e6:.0f}M params", flush=True)
