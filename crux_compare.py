"""Crux comparison under ONE identical harness: matched attention-only transformer (fixed-lr, 21.01 in
training) vs the 284M spin-dominant. Same val set, same measurement, many iters for a stable estimate.

IMPORTANT (fairness): the transformer baseline ran WITH the lr-floor fix; the OLD spin (pragnosia_284m.pt)
ran WITHOUT it, so its number is CRIPPLED. This script labels the old spin PRE-FIX / provisional. The fair
comparison is transformer vs the re-run fixed-lr spin (pragnosia_284m_fair.pt) -- swap it in when ready.

Run:  python3 crux_compare.py
"""
import sys, os, json, re
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, torch, torch.nn.functional as F, s6_hybrid as H
from tokenizers import Tokenizer

DEV = "cuda"; CTX = 256; VOC = 16384
H.VOC, H.L = VOC, CTX
tok = Tokenizer.from_file("data/bpe.json")
vd = np.memmap("data/big_valid.bin", dtype=np.int16, mode="r")

TF = dict(d=768, heads=12, layers=21, mlp_mult=9, carrier="none")
SP = dict(d=768, heads=12, layers=20, mlp_mult=9, carrier="spin_dominant")
# THE FAIR CRUX: best-vs-best, both fixed-lr, both from-scratch fixed-284M, matched 418K steps / 6.85B tokens.
MODELS = []
if os.path.exists("pragnosia_baseline_best.pt"):
    MODELS.append(("transformer_baseline (fixed-lr, BEST)", "pragnosia_baseline_best.pt", TF))
if os.path.exists("pragnosia_284m_fair.pt"):
    MODELS.append(("spin_dominant_284M (fixed-lr, BEST, FAIR)", "pragnosia_284m_fair.pt", SP))
# references (not the fair comparison)
if os.path.exists("pragnosia_baseline.pt"):
    MODELS.append(("transformer_baseline (final ckpt, ref)", "pragnosia_baseline.pt", TF))
if os.path.exists("pragnosia_284m.pt"):
    MODELS.append(("spin_284M (PRE-FIX, crippled ref)", "pragnosia_284m.pt", SP))

MULTIHOP = {
    "1hop": [("The capital of France is", "paris"), ("The largest planet in the solar system is", "jupiter"),
             ("The chemical symbol for oxygen is", "o"), ("Water is made of hydrogen and", "oxygen"),
             ("The capital of Japan is", "tokyo"), ("The first president of the United States was", "washington"),
             ("The freezing point of water in Celsius is", "0"), ("The opposite of hot is", "cold")],
    "2hop": [("The capital of the country where the Eiffel Tower stands is", "paris"),
             ("The planet closest to the star at the center of our solar system is", "mercury"),
             ("The currency of the country whose capital is London is", "pound")],
}


def load(path, cfg):
    m = H.SpinAttentionLM(VOC, cfg["d"], cfg["heads"], cfg["layers"], mlp_mult=cfg["mlp_mult"], carrier=cfg["carrier"])
    m.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    return m.bfloat16().to(DEV).eval()


@torch.no_grad()
def val_ppl(m, iters=200):
    tot = 0.0; g = torch.Generator().manual_seed(123)
    for _ in range(iters):
        i = int(torch.randint(0, len(vd) - CTX - 1, (1,), generator=g))
        x = torch.tensor(np.asarray(vd[i:i+CTX])[None], dtype=torch.long, device=DEV)
        y = torch.tensor(np.asarray(vd[i+1:i+CTX+1])[None], dtype=torch.long, device=DEV)
        tot += F.cross_entropy(m(x).reshape(-1, VOC), y.reshape(-1)).item()
    return float(np.exp(tot / iters))


@torch.no_grad()
def multihop(m):
    out = {}
    for lvl, probes in MULTIHOP.items():
        hit = 0
        for q, a in probes:
            ids = tok.encode(q).ids[-CTX:]
            gen = tok.decode(m.generate(ids, n_new=8, window=CTX, temp=0.0, rep=1.3)).lower()
            hit += (a in gen[len(q):] if len(gen) > len(q) else a in gen)
        out[lvl] = f"{hit}/{len(probes)}"
    return out


res = {}
for name, path, cfg in MODELS:
    if not os.path.exists(path):
        print(f"  {name}: MISSING {path}", flush=True); continue
    m = load(path, cfg)
    n = sum(p.numel() for p in m.parameters()) / 1e6
    ppl = val_ppl(m); mh = multihop(m)
    res[name] = {"ckpt": path, "params_M": round(n, 1), "val_ppl_200it": round(ppl, 3), "multihop": mh}
    print(f"  {name:42} {n:5.0f}M  val_ppl {ppl:6.3f}  multihop {mh}", flush=True)
    del m; torch.cuda.empty_cache()

os.makedirs("eval_registry", exist_ok=True)
json.dump(res, open("eval_registry/crux_compare.json", "w"), indent=2)
print("\nNOTE: transformer=fixed-lr; spin_284M(PRE-FIX)=crippled by the lr floor -> NOT a fair comparison.")
print("Fair comparison = transformer vs pragnosia_284m_fair.pt (re-run in progress).", flush=True)
