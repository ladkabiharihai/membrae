import sys; sys.path.insert(0, "/opt/code/membrae")
import json, torch, s6_hybrid as H
CFG = json.load(open("pragnosia_1b_based.json"))
# FRESH graft from the ctx512 winner (NOT the trained pragnosia_1b_based.pt)
sd = torch.load("pragnosia_1b_ctx512_it500_CROSSES256.pt", map_location="cpu", weights_only=True)
torch.manual_seed(0)
base = H.SpinAttentionLM(CFG["vocab"], CFG["d"], CFG["heads"], CFG["layers"], mlp_mult=CFG["mlp_mult"], carrier="spin_selective").eval()
base.load_state_dict(sd, strict=True)
graft = H.SpinAttentionLM(CFG["vocab"], CFG["d"], CFG["heads"], CFG["layers"], mlp_mult=CFG["mlp_mult"], carrier="spin_based").eval()
graft.load_state_dict(sd, strict=False)   # based.* stay zero-init
x = torch.randint(0, CFG["vocab"], (2, 300))
with torch.no_grad():
    diff1 = (base(x) - graft(x)).abs().max().item()
    x1, x2 = x[:, :256], x[:, 256:]
    lb, sb = base(x1, return_state=True); lb2, _ = base(x2, state=sb, return_state=True)
    lg, sg = graft(x1, return_state=True); lg2, _ = graft(x2, state=sg, return_state=True)
    diff2 = (lb2 - lg2).abs().max().item()
    ncarr = sum(1 for b in graft.blocks if hasattr(b, "forward_state"))
print("FRESH graft (based zero-init):")
print("  non-threaded  logit diff: %.2e" % diff1)
print("  THREADED 2-win logit diff: %.2e" % diff2)
print("  state: %d (%d carrier + %d Based)" % (len(sg), ncarr, len(sg) - ncarr))
print("VERDICT:", "OK function-preserving + threading works" if diff1 < 1e-4 and diff2 < 1e-4 else "BUG")
