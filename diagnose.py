"""
diagnose.py -- isolate WHY brain.py degrades while probe.py is fine.
Hypothesis: teach() replays from a mismatched local corpus, so every teach
(including the startup identity install) drags the model off its trained
distribution. We measure val perplexity + key generations at each stage.
"""
import os, math, torch, torch.nn.functional as F, json
import s6_hybrid as H
from tokenizers import Tokenizer
DEV = "cuda" if torch.cuda.is_available() else "cpu"
cfg = json.load(open("pragnosia.json")); H.VOC, H.L = cfg["vocab"], cfg["ctx"]
tok = Tokenizer.from_file(cfg["tokenizer"])
vd = H.load(cfg["valid_bin"])

def fresh():
    m = H.SpinAttentionLM(cfg["vocab"], cfg["d"], cfg["heads"], cfg["layers"],
                          mlp_mult=cfg.get("mlp_mult",4)).to(DEV)
    m.load_state_dict(torch.load(cfg["ckpt"], map_location=DEV, weights_only=True)); m.eval()
    return m

@torch.no_grad()
def gen(m, p, n=16):
    ids = tok.encode(p).ids; s=len(ids)
    for _ in range(n):
        lo = m(torch.tensor([ids[-256:]],device=DEV))[0,-1].float()
        for t in set(ids[-20:]): lo[t]/=1.3
        nx=lo.argmax().item()
        if nx==0: break
        ids.append(nx)
    return tok.decode(ids[s:]).strip()

@torch.no_grad()
def nll(m, text):
    ids = tok.encode(text).ids
    return F.cross_entropy(m(torch.tensor([ids],device=DEV))[0,:-1], torch.tensor(ids[1:],device=DEV)).item()

PROBES = ["def add(a, b):", "Question: What is 5 plus 7?\nAnswer:", "once upon a time"]
def report(tag, m):
    p = H.val_ppl(m, vd, iters=10, bs=4)
    print(f"\n[{tag}] val_ppl={p:.1f}")
    for q in PROBES:
        print(f"   {q[:30]!r:34} nll={nll(m,q):4.1f}  gen-> {gen(m,q)[:70]!r}")

# one teach step, exactly like brain.py but replay toggleable
def teach(m, fact, replay=True, steps=40, lr=2e-4):
    fid = torch.tensor([tok.encode(fact).ids], device=DEV)
    rep = H.load(cfg["train_bin"]) if replay else None
    opt = torch.optim.AdamW(m.parameters(), lr=lr); m.train()
    for _ in range(steps):
        lf = F.cross_entropy(m(fid[:,:-1]).reshape(-1,H.VOC), fid[:,1:].reshape(-1))
        loss = lf
        if rep is not None:
            xr,yr = H.batch(rep, 8); loss = lf + F.cross_entropy(m(xr).reshape(-1,H.VOC), yr.reshape(-1))
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(),1.0); opt.step()
    m.eval()

PHYS = ("A black hole is a region of spacetime where gravity is so strong that nothing, not even "
        "light, can escape. For a non-rotating mass M the horizon lies at the Schwarzschild radius "
        "r_s = 2 G M / c^2.")

print("="*70); print("DIAGNOSIS: does teach() (local replay) degrade the good base model?")
print("="*70)
m = fresh(); report("BASE (pristine pragnosia.pt)", m)

m = fresh(); teach(m, PHYS, replay=True);  report("after 1 physics teach WITH local replay", m)
m = fresh(); teach(m, PHYS, replay=False); report("after 1 physics teach NO replay", m)

print("\n--- identity install effect (6 statements, local replay) ---")
m = fresh()
for s in ["My name is Pragnosia.", "I am Pragnosia, a spinning brain that reasons by phase.",
          "I learn new things continuously without forgetting.",
          "I know the limits of my knowledge, and I say so when I do not know.",
          "I am curious, and I explore whatever I am most uncertain about.",
          "I think by spinning; my answer lives in the phase."]:
    teach(m, s, replay=True, steps=40)
report("after identity install (6x teach, local replay)", m)
