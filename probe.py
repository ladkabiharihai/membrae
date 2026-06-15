"""
probe.py -- inspect the WHOLE brain on CPU while it trains on the GPU.

Forces CPU (never touches the training GPU). Loads the language model (at
whatever size it has grown to -- reads pragnosia.json) AND the proven faculties
(unified_brain.pt), and shows, for every answer, WHAT it decides and WHY, plus a
health check of every wired faculty so you can see what's working.

  python3 probe.py                  # full report: faculty health + language + decision trace
  python3 probe.py "your prompt"    # one prompt: confidence -> decision -> answer
  python3 probe.py --quick "p"      # skip the faculty health panel (faster)
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""        # CPU only -> cannot disturb the GPU run
import sys, json, shutil, re, torch, torch.nn.functional as F, numpy as np
import unified_brain as U
import s6_hybrid as H
from tokenizers import Tokenizer

args = [a for a in sys.argv[1:] if a != "--quick"]
quick = "--quick" in sys.argv
cfg = json.load(open("pragnosia.json"))
H.VOC, H.L = cfg["vocab"], cfg["ctx"]          # match the s6_hybrid helpers to this model

# ---------- language faculty (the scaled model, snapshotted for safe reading) ----------
if not os.path.exists(cfg["ckpt"]):
    sys.exit(f"no checkpoint yet at {cfg['ckpt']} -- let the trainer save one first.")
shutil.copy(cfg["ckpt"], "/tmp/_probe_ckpt.pt")
tok = Tokenizer.from_file(cfg["tokenizer"])
lm = H.SpinAttentionLM(cfg["vocab"], cfg["d"], cfg["heads"], cfg["layers"], mlp_mult=cfg.get("mlp_mult", 4))
lm.load_state_dict(torch.load("/tmp/_probe_ckpt.pt", map_location="cpu", weights_only=True)); lm.eval()

# ---------- the proven faculties (tiny, fast on CPU) ----------
fac = U.UnifiedBrain(d=128)
if os.path.exists("unified_brain.pt"):
    fac.load_state_dict(torch.load("unified_brain.pt", map_location="cpu", weights_only=True))
fac.eval()

# ---------- self-calibrated familiarity boundary (its own confidence) ----------
@torch.no_grad()
def nll(ids):
    return F.cross_entropy(lm(torch.tensor([ids]))[0, :-1], torch.tensor(ids[1:])).item()
boundary, vd = 4.5, None
try:
    vd = H.load(cfg["valid_bin"])
    g = torch.Generator().manual_seed(0); xs = []
    for _ in range(60):
        i = int(torch.randint(0, vd.size(0) - 40, (1,), generator=g)); xs.append(nll(vd[i:i+32].tolist()))
    xs.sort(); boundary = xs[int(0.9 * len(xs))]
except Exception:
    pass

last = ""
try:
    vp = [l for l in open("pragnosia_train.log") if "VAL_PPL" in l]
    last = "| " + re.findall(r"it=\s*\d+\s+VAL_PPL=[\d.]+", vp[-1])[-1] if vp else ""
except Exception:
    pass

# ============================ helpers ============================
@torch.no_grad()
def conf(p):
    ids = tok.encode(p).ids
    return 0.0 if len(ids) < 2 else nll(ids)

@torch.no_grad()
def gen(prompt, n=36, rep=1.3):
    ids = tok.encode(prompt).ids; start = len(ids)
    for _ in range(n):
        lo = lm(torch.tensor([ids[-cfg["ctx"]:]]))[0, -1].float()
        for t in set(ids[-40:]): lo[t] /= rep
        nx = lo.argmax().item()
        if nx == 0: break
        ids.append(nx)
    return tok.decode(ids[start:]).strip()

def answer(p):
    """Show the brain's decision trace: confidence -> route -> reply."""
    c = conf(p)
    is_q = p.strip().endswith("?")
    if c > boundary:
        return (f"  confidence : {c:.1f}  (> boundary {boundary:.1f}: unfamiliar)\n"
                f"  DECISION   : ABSTAIN — too unsure to be honest\n"
                f"  reply      : \"I don't know.\"\n"
                f"  (raw model, if forced: '{gen(p, n=14)[:60]}...')")
    return (f"  confidence : {c:.1f}  (<= boundary {boundary:.1f}: familiar)\n"
            f"  DECISION   : ANSWER ({'question' if is_q else 'continuation'})\n"
            f"  reply      : {gen(p)[:140]}")

# ============================ faculty health (toy faculties, CPU) ============================
def faculty_health():
    print("=" * 64); print("FACULTY HEALTH  (is every wired faculty working?)"); print("=" * 64)
    R = U.self_test(fac)
    rows = [("reasoning (parity/sum/max)", min(R['reason_parity3'],R['reason_sum3'],R['reason_max']), lambda v: v>0.85),
            ("knows-what-it-doesnt (P5 gap)", R['p5_gap'], lambda v: v>0.3),
            ("never-fabricate (P6 abstain)", R['p6_abstain_unknowable'], lambda v: v>0.6),
            ("seek (answer w/ retrieval)", R['seek_answer'], lambda v: v>0.85),
            ("seek anti-memorize control", R['seek_random_ctrl'], lambda v: v<0.45),
            ("exact counting (len 16)", R['exact_L16'], lambda v: v>0.9),
            ("alive: learns, no forgetting", R['alive_final'], lambda v: v>0.95),
            ("language parse + generate", min(R['lang_parse_held'],R['lang_gen_held']), lambda v: v>0.8)]
    for nm, v, ok in rows:
        print(f"  [{'OK ' if ok(v) else 'XX'}] {nm:<32} {v:.2f}")

# ============================ main ============================
print(f"\nPragnosia {sum(p.numel() for p in lm.parameters())/1e6:.0f}M lang + {sum(p.numel() for p in fac.parameters())/1e3:.0f}K faculties"
      f" | {cfg['layers']}L d={cfg['d']} mlp_mult={cfg.get('mlp_mult',4)} | CPU {last}")
print(f"abstention boundary = {boundary:.1f}\n")

if args:
    p = " ".join(args)
    print(f"PROMPT: {p}"); print(answer(p))
else:
    if not quick:
        faculty_health(); print()
    print("=" * 64); print("LANGUAGE + DECISIONS  (what happens when it answers)"); print("=" * 64)
    ppl = H.val_ppl(lm, vd, iters=6, bs=4) if vd is not None else float('nan')
    print(f"  language perplexity: {ppl:.1f}\n")
    for p in ["Once upon a time", "def add(a, b):", "Question: What is 5 plus 7?\nAnswer:",
              "who is the president of india", "what is earth", "The quantum entanglement equation is"]:
        print(f"  PROMPT: {p!r}"); print(answer(p)); print()
