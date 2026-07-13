"""CHEAP laptop probe of the fast-slow coupling ON TOP OF the latest model: warm-start the coupled model from
a base checkpoint (coupling inits near-identity -> starts EQUAL to the base), FREEZE the base, train ONLY the
coupling params (to_slow / slow / from_slow), and measure val ppl before vs after. Any drop is attributable to
the coupling. This is a LOWER BOUND on the coupling's value (the base cannot co-adapt); a gain here is a strong
signal to run the full fine-tune on the H100.

Run (GPU, Unreal closed):  python3 train_coupled_probe.py [base_ckpt] [steps]
"""
import sys, os, json, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch, torch.nn.functional as F
import s6_hybrid as H

BASE = sys.argv[1] if len(sys.argv) > 1 else "pragnosia_spin_h100.pt"
STEPS = int(sys.argv[2]) if len(sys.argv) > 2 else 1500
DEV = "cuda"
c = json.load(open("pragnosia_coupled.json")); H.VOC, H.L = c["vocab"], c["ctx"]; CTX = c["ctx"]
torch.manual_seed(0); np.random.seed(0)
P = lambda *a: print(*a, flush=True)

# build coupled, warm-start from base (coupling params stay at near-identity init)
m = H.SpinAttentionLM(c["vocab"], c["d"], c["heads"], c["layers"], mlp_mult=c["mlp_mult"], carrier=c["carrier"])
missing, unexpected = m.load_state_dict(torch.load(BASE, map_location="cpu", weights_only=True), strict=False)
assert not unexpected, f"name mismatch: {unexpected[:3]}"
m = m.bfloat16().to(DEV); m.grad_checkpoint = False

# freeze everything except the coupling params
COUP = ("to_slow", "from_slow", ".slow.")
trn = [p for n, p in m.named_parameters() if any(t in n for t in COUP)]
for n, p in m.named_parameters():
    p.requires_grad = any(t in n for t in COUP)
P(f"warm-started coupled from {BASE}: {sum(p.numel() for p in trn)/1e6:.2f}M trainable coupling params "
  f"(base frozen), {len(missing)} new keys")

tr = np.memmap("data/window2_train.bin", dtype=np.int16, mode="r")
vd = np.memmap("data/big_valid.bin", dtype=np.int16, mode="r")
def batch(data, bs=2):
    ix = np.random.randint(0, len(data) - CTX - 1, bs)
    x = torch.tensor(np.stack([np.asarray(data[i:i+CTX]) for i in ix]), dtype=torch.long, device=DEV)
    y = torch.tensor(np.stack([np.asarray(data[i+1:i+CTX+1]) for i in ix]), dtype=torch.long, device=DEV)
    return x, y

@torch.no_grad()
def val_ppl(iters=40):
    m.eval(); tot = 0.0
    g = torch.Generator().manual_seed(123)
    for _ in range(iters):
        i = int(torch.randint(0, len(vd) - CTX - 1, (1,), generator=g))
        x = torch.tensor(np.asarray(vd[i:i+CTX])[None], dtype=torch.long, device=DEV)
        y = torch.tensor(np.asarray(vd[i+1:i+CTX+1])[None], dtype=torch.long, device=DEV)
        tot += F.cross_entropy(m(x).reshape(-1, c["vocab"]), y.reshape(-1)).item()
    m.train(); return float(np.exp(tot / iters))

ppl0 = val_ppl(); P(f"val ppl BEFORE (coupled@init ~= base): {ppl0:.3f}")
opt = torch.optim.AdamW(trn, lr=3e-4, betas=(0.9, 0.95))
m.train()
for step in range(STEPS):
    x, y = batch(tr)
    loss = F.cross_entropy(m(x).reshape(-1, c["vocab"]), y.reshape(-1))
    opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(trn, 1.0); opt.step()
    if step % 250 == 0:
        P(f"  step {step:4d}  loss {loss.item():.3f}  peakGPU {torch.cuda.max_memory_allocated()/1e9:.1f}GB")
ppl1 = val_ppl(); P(f"val ppl AFTER coupling-only training: {ppl1:.3f}")

delta = ppl0 - ppl1
out = {"base": BASE, "steps": STEPS, "trainable_M": round(sum(p.numel() for p in trn)/1e6, 3),
       "val_ppl_before": round(ppl0, 3), "val_ppl_after": round(ppl1, 3), "delta": round(delta, 3),
       "coupling_helps_frozen_base": delta > 0.05}
os.makedirs("eval_registry", exist_ok=True)
json.dump(out, open("eval_registry/coupled_probe.json", "w"), indent=2)
P(json.dumps(out, indent=2))
P("VERDICT: " + ("coupling helps even with a FROZEN base -> strong signal for the full H100 fine-tune"
                if delta > 0.05 else
                "no gain with a frozen base -> inconclusive; the coupling may need base co-adaptation (H100)"))
