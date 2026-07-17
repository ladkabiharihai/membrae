"""MULTI-SEED CAUSAL ABLATION -- the paper's CENTRAL claim (the carrier is the load-bearing computation) is
single-seed at each size. The 100M multi-seed sweep left us 3 independently-trained spin models, and ablation
is inference-only, so we can test seed-robustness cheaply: for each seed, zero the spin carriers (keep the
MLPs) and, separately, zero the attention sublayers, and compare the cost.

Robust claim = the carrier/attention gap holds in EVERY seed (order-of-magnitude, not an exact multiplier).

Run (GPU):  python3 multiseed_ablate.py
"""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch, s6_hybrid as H

CFG = dict(vocab=16384, d=640, heads=10, layers=14, mlp_mult=6, carrier="spin_dominant", ctx=256)
H.VOC, H.L = CFG["vocab"], CFG["ctx"]
DEV = "cuda"
SEEDS = [int(x) for x in sys.argv[1:]] or [1, 2, 3]      # e.g. `multiseed_ablate.py 1 3`
P = lambda *a: print(*a, flush=True)
vd = H.load("big_valid")


def run(ckpt):
    m = H.SpinAttentionLM(CFG["vocab"], CFG["d"], CFG["heads"], CFG["layers"],
                          mlp_mult=CFG["mlp_mult"], carrier=CFG["carrier"])
    m.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=True))
    m = m.bfloat16().to(DEV).eval()
    spin = [i for i, b in enumerate(m.blocks) if isinstance(b, H.SpinBlock)]
    attn = [i for i, b in enumerate(m.blocks) if isinstance(b, H.Block)]
    base = float(H.val_ppl(m, vd, iters=60))

    # --- ablate ALL spin carriers (zero the carrier gate, keep the MLP) ---
    sv = {}
    for i in spin:
        sv[i] = float(m.blocks[i].carrier.gate.item())
        with torch.no_grad(): m.blocks[i].carrier.gate.fill_(-30.0)
    ppl_spin = float(H.val_ppl(m, vd, iters=60))
    for i, v in sv.items():
        with torch.no_grad(): m.blocks[i].carrier.gate.fill_(v)

    # --- ablate ALL attention sublayers (zero the attn output, keep the MLP) ---
    saved = {}
    for i in attn:
        saved[i] = m.blocks[i].attn.forward
        m.blocks[i].attn.forward = (lambda x: torch.zeros_like(x))
    ppl_attn = float(H.val_ppl(m, vd, iters=60))
    for i, f in saved.items():
        m.blocks[i].attn.forward = f

    del m; torch.cuda.empty_cache()
    return {"base_ppl": round(base, 2), "n_spin": len(spin), "n_attn": len(attn),
            "ablate_spin_ppl": round(ppl_spin, 1), "ablate_attn_ppl": round(ppl_attn, 2),
            "spin_x": round(ppl_spin / base, 1), "attn_x": round(ppl_attn / base, 2)}


res = {}
for s in SEEDS:
    ck = f"p100_spin_s{s}.pt"
    if not os.path.exists(ck):
        P(f"seed {s}: MISSING {ck}"); continue
    r = run(ck); res[f"seed{s}"] = r
    P(f"seed {s}: base {r['base_ppl']:.2f} | ablate SPIN -> {r['ablate_spin_ppl']:.1f} ({r['spin_x']}x) | "
      f"ablate ATTN -> {r['ablate_attn_ppl']:.2f} ({r['attn_x']}x)")

if res:
    sx = [r["spin_x"] for r in res.values()]; ax = [r["attn_x"] for r in res.values()]
    summary = {"n_seeds": len(res),
               "spin_x_range": [min(sx), max(sx)], "attn_x_range": [min(ax), max(ax)],
               "carrier_load_bearing_every_seed": all(s > 10 * a for s, a in zip(sx, ax))}
    res["summary"] = summary
    os.makedirs("eval_registry", exist_ok=True)
    json.dump(res, open("eval_registry/multiseed_ablation_100m.json", "w"), indent=2)
    P("\n=== SUMMARY ===")
    P(json.dumps(summary, indent=2))
    P("carrier ablation is catastrophic in EVERY seed" if summary["carrier_load_bearing_every_seed"]
      else "NOT consistent across seeds -- the central claim is seed-dependent (report honestly)")
