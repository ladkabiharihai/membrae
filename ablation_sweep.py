"""Multi-seed param-matched ablation, FAIR version: each architecture gets its OWN learning rate via a
Smith range-test (the recipe the original 219/279 used), then is trained at matched budget across seeds.
Only ARCHITECTURE and SEED vary the model; lr is derived per architecture. Reports mean ± std val ppl."""
import json, math, time, os
import numpy as np, torch, torch.nn.functional as F, s6_hybrid as H
DEV = "cuda"; c = json.load(open("pragnosia.json")); H.VOC, H.L = c["vocab"], c["ctx"]
VOC, CTX = c["vocab"], c["ctx"]
td, vd = H.load("window2_train"), H.load("big_valid")
ARCHS = [("attention-only", "none", 8), ("spin-dominant", "spin_dominant", 8),
         ("real-dominant", "real_dominant", 8),                  # generalization control: a DIFFERENT recurrence as core
         ("per-block", "per_block", 8), ("transformer-11L", "none", 11)]
SEEDS = [0, 1, 2]
STEPS, BS, WARM = 5000, 24, 250

def mk(car, L):
    return H.SpinAttentionLM(VOC, 512, 8, L, mlp_mult=4, carrier=car).to(DEV).bfloat16()

@torch.no_grad()
def _noop(): pass
def find_lr(car, L, lo=1e-5, hi=0.3, n=70):
    """Smith LR range-test: steepest-descent point / 10 (the trainer's recipe)."""
    torch.manual_seed(0); np.random.seed(0)
    m = mk(car, L); opt = torch.optim.AdamW(m.parameters(), lr=lo, betas=(0.9, 0.95), fused=True); m.train()
    lrs = np.exp(np.linspace(np.log(lo), np.log(hi), n)); loss_hist = []
    for lr in lrs:
        for g in opt.param_groups: g["lr"] = float(lr)
        x, y = H.batch(td, BS)
        loss = F.cross_entropy(m(x).reshape(-1, VOC), y.reshape(-1))
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
        loss_hist.append(loss.item())
    del m, opt; torch.cuda.empty_cache()
    sm = np.convolve(loss_hist, np.ones(5) / 5, "valid")
    idx = int(np.argmin(np.diff(sm)))
    return max(1e-4, float(lrs[idx]) / 10.0)

def lr_at(t, peak):
    return peak * t / WARM if t < WARM else 0.5 * peak * (1 + math.cos(math.pi * (t - WARM) / (STEPS - WARM)))

RES = "ablation_results.json"
res = json.load(open(RES)) if os.path.exists(RES) else {}
for name, car, L in ARCHS:
    res.setdefault(name, {})
    if "_lr" not in res[name]:
        res[name]["_lr"] = round(find_lr(car, L), 6)
        print(f"  {name:16} derived lr = {res[name]['_lr']:.1e}", flush=True)
        json.dump(res, open(RES, "w"), indent=2)
    peak = res[name]["_lr"]
    for seed in SEEDS:
        k = str(seed)
        if k in res[name]: continue
        try:
            t0 = time.time(); torch.manual_seed(seed); np.random.seed(seed)
            m = mk(car, L); npar = sum(p.numel() for p in m.parameters())
            opt = torch.optim.AdamW(m.parameters(), lr=peak, betas=(0.9, 0.95), weight_decay=0.05, fused=True); m.train()
            for step in range(STEPS):
                for g in opt.param_groups: g["lr"] = lr_at(step, peak)
                x, y = H.batch(td, BS)
                loss = F.cross_entropy(m(x).reshape(-1, VOC), y.reshape(-1))
                opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
            ppl = float(H.val_ppl(m, vd, iters=80))
            res[name][k] = {"ppl": round(ppl, 2), "params_M": round(npar / 1e6, 1)}
            print(f"  {name:16} seed{seed}: ppl {ppl:6.1f}  ({npar/1e6:.1f}M, lr {peak:.1e}, {time.time()-t0:.0f}s)", flush=True)
            del m, opt; torch.cuda.empty_cache(); json.dump(res, open(RES, "w"), indent=2)
        except Exception as e:
            print(f"  {name} seed{seed}: FAILED {str(e)[:60]}", flush=True); torch.cuda.empty_cache()

print("\n=== SUMMARY (mean ± std val ppl, per-arch derived lr) ===", flush=True)
for name, _, _ in ARCHS:
    d = res.get(name, {}); ppls = [v["ppl"] for kk, v in d.items() if kk != "_lr"]
    if ppls:
        pm = [v["params_M"] for kk, v in d.items() if kk != "_lr"][0]
        print(f"  {name:16} {np.mean(ppls):6.1f} ± {np.std(ppls):4.1f}  (n={len(ppls)}, {pm}M, lr {d['_lr']:.1e})", flush=True)
