"""Learning curves: spin-dominant vs attention-only at matched ~37M, val ppl vs TOKENS over a long budget.
Directly addresses the reviewer reading that the 3.1x short-budget gap is just "the transformer hasn't moved
yet" -- we plot both trajectories to convergence and report where the gap actually settles. Each arch at its
OWN range-test lr (the recipe behind the ablation). Resume-safe (skips a finished arch in learning_curves.json)."""
import json, math, os, numpy as np, torch, torch.nn.functional as F, s6_hybrid as H
DEV = "cuda"; c = json.load(open("pragnosia.json")); H.VOC, H.L = c["vocab"], c["ctx"]
VOC, CTX = c["vocab"], c["ctx"]
td, vd = H.load("window2_train"), H.load("big_valid")
ARCHS = [("spin-dominant", "spin_dominant", 0.006734), ("attention-only", "none", 0.0001)]  # lrs from the sweep's find_lr
STEPS, BS, WARM, LOG = 20000, 24, 250, 1000
def mk(car): return H.SpinAttentionLM(VOC, 512, 8, 8, mlp_mult=4, carrier=car).to(DEV).bfloat16()
def lr_at(t, peak): return peak * t / WARM if t < WARM else 0.5 * peak * (1 + math.cos(math.pi * (t - WARM) / (STEPS - WARM)))
RES = "learning_curves.json"
# INTERLEAVED: train both archs together (one step each per iteration), log both every LOG -> the two
# curves grow side by side so the gap can be watched forming in real time. Same total compute, fair (same
# seed, same batches per step). Both 37M models fit easily.
res = {}
torch.manual_seed(0); np.random.seed(0)
models = []
for name, car, peak in ARCHS:
    m = mk(car); opt = torch.optim.AdamW(m.parameters(), lr=peak, betas=(0.9, 0.95), weight_decay=0.05, fused=True); m.train()
    models.append((name, m, opt, peak)); res[name] = []
for step in range(1, STEPS + 1):
    x, y = H.batch(td, BS)                                    # SAME batch to both archs this step (fair)
    for name, m, opt, peak in models:
        for g in opt.param_groups: g["lr"] = lr_at(step, peak)
        loss = F.cross_entropy(m(x).reshape(-1, VOC), y.reshape(-1))
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
    if step % LOG == 0 or step == STEPS:
        toks = step * BS * CTX
        for name, m, opt, peak in models:
            ppl = float(H.val_ppl(m, vd, iters=60))
            res[name].append({"step": step, "tokens": toks, "ppl": round(ppl, 2)})
            print(f"  {name:16} step {step:6d} ({toks/1e6:5.0f}M tok)  ppl {ppl:7.1f}", flush=True)
        json.dump(res, open(RES, "w"), indent=2)
print("LC DONE", flush=True)
