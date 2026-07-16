"""STAGE A (H100): does the fast-slow coupling help at 284M? Two arms trained FROM SCRATCH (random init),
SAME data order, SAME lr/steps:
  COUPLED  = spin_dominant_coupled
  CONTROL  = spin_dominant (base)
Compare COUPLED-final vs CONTROL-final val ppl (+ multi-hop). Win = coupled beats control at matched steps.

WHY FROM SCRATCH (not warm-start-continue): warm-starting the CONVERGED fair spin and continuing on its own
training data has NO HEADROOM -- both arms only drift up, so the coupling cannot show benefit (verified on the
laptop: lr 2e-4 -> both ~58 ppl; lr 5e-5 -> both climbing). The coupling adds CAPACITY, which only pays off
with room to fit the data better, i.e. from scratch (where the 37M spin win was measured). This needs the
H100: a matched budget of ~100K-418K steps per arm is NOT laptop-feasible.

Run (H100, GPU free):  python3 train_coupled_284.py [steps]   # e.g. 100000 for an early signal
"""
import sys, os, json, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch, torch.nn.functional as F, s6_hybrid as H
from tokenizers import Tokenizer

STEPS = int(sys.argv[1]) if len(sys.argv) > 1 else 2500
DEV = "cuda"; CTX = 256; VOC = 16384; LR = 3e-4; BS = 4; WARMUP = 500   # from-scratch lr (not fine-tune)
# NOTE: for the REAL H100 run prefer train_pragnosia.py (derived lr + schedule + lr-floor fix) with
# CONFIG=pragnosia_coupled_284m.json vs a from-scratch spin config; this standalone is a simpler reference.
H.VOC, H.L = VOC, CTX
BASE = "pragnosia_284m_fair.pt"
tok = Tokenizer.from_file("data/bpe.json")
tr = np.memmap("data/window2_train.bin", dtype=np.int16, mode="r")
vd = np.memmap("data/big_valid.bin", dtype=np.int16, mode="r")
P = lambda *a: print(*a, flush=True)

MH = [("The capital of France is", "paris"), ("The largest planet in the solar system is", "jupiter"),
      ("The capital of Japan is", "tokyo"), ("Water is made of hydrogen and", "oxygen"),
      ("The capital of the country where the Eiffel Tower stands is", "paris"),
      ("The planet closest to the star at the center of our solar system is", "mercury")]


def batch(step):                                    # SAME data for both arms: seed by step only
    g = np.random.RandomState(1000 + step)
    ix = g.randint(0, len(tr) - CTX - 1, BS)
    x = torch.tensor(np.stack([np.asarray(tr[i:i+CTX]) for i in ix]), dtype=torch.long, device=DEV)
    y = torch.tensor(np.stack([np.asarray(tr[i+1:i+CTX+1]) for i in ix]), dtype=torch.long, device=DEV)
    return x, y

@torch.no_grad()
def val_ppl(m, iters=200):
    m.eval(); tot = 0.0; g = torch.Generator().manual_seed(123)
    for _ in range(iters):
        i = int(torch.randint(0, len(vd) - CTX - 1, (1,), generator=g))
        x = torch.tensor(np.asarray(vd[i:i+CTX])[None], dtype=torch.long, device=DEV)
        y = torch.tensor(np.asarray(vd[i+1:i+CTX+1])[None], dtype=torch.long, device=DEV)
        tot += F.cross_entropy(m(x).reshape(-1, VOC), y.reshape(-1)).item()
    m.train(); return float(np.exp(tot / iters))

@torch.no_grad()
def multihop(m):
    m.eval(); hit = 0
    for q, a in MH:
        ids = tok.encode(q).ids[-CTX:]
        gen = tok.decode(m.generate(ids, n_new=8, window=CTX, temp=0.0, rep=1.3)).lower()
        hit += (a in gen[len(q):] if len(gen) > len(q) else a in gen)
    m.train(); return f"{hit}/{len(MH)}"


def train_arm(name, carrier, layers):
    torch.manual_seed(0)                                # SAME init seed for a fair coupled-vs-base comparison
    m = H.SpinAttentionLM(VOC, 768, 12, layers, mlp_mult=9, carrier=carrier)   # FROM SCRATCH (no warm-start)
    m = m.to(DEV); m.grad_checkpoint = True; m.train()
    opt = torch.optim.AdamW(m.parameters(), lr=LR, betas=(0.9, 0.95), weight_decay=0.05)
    p0 = val_ppl(m)
    P(f"[{name}] {sum(p.numel() for p in m.parameters())/1e6:.1f}M  val_ppl@0 {p0:.3f}")
    for step in range(STEPS):
        for g in opt.param_groups: g["lr"] = LR * min(1.0, (step + 1) / WARMUP)   # linear warmup
        x, y = batch(step)
        loss = F.cross_entropy(m(x).reshape(-1, VOC), y.reshape(-1))
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
        if step % 500 == 0:
            vp = val_ppl(m)                                                       # REAL metric each log (catch divergence)
            P(f"  [{name}] step {step:4d} loss {loss.item():.3f} val_ppl {vp:.2f} peakGPU {torch.cuda.max_memory_allocated()/1e9:.1f}GB")
    pf = val_ppl(m); mh = multihop(m)
    P(f"[{name}] val_ppl@{STEPS} {pf:.3f}  multihop {mh}")
    del m, opt; torch.cuda.empty_cache()
    return {"val_ppl_start": round(p0, 3), "val_ppl_end": round(pf, 3), "multihop": mh}


P(f"=== STAGE A: coupling at 284M, {STEPS} steps/arm, lr {LR}, from {BASE} ===")
control = train_arm("CONTROL (base spin)", "spin_dominant", 20)
coupled = train_arm("COUPLED (fast-slow)", "spin_dominant_coupled", 20)
delta = control["val_ppl_end"] - coupled["val_ppl_end"]        # >0 => coupled better
res = {"steps": STEPS, "lr": LR, "control": control, "coupled": coupled,
       "coupled_minus_control_ppl": round(-delta, 3), "coupling_helps": delta > 0.1}
os.makedirs("eval_registry", exist_ok=True)
json.dump(res, open("eval_registry/coupled_284_stageA.json", "w"), indent=2)
P("\n=== VERDICT ===")
P(json.dumps(res, indent=2))
P(("COUPLING HELPS at 284M -> worth the 1B run" if delta > 0.1 else
   "coupling NEUTRAL/worse at 284M -> do NOT scale to 1B (honest negative)"))
