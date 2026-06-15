"""
probe.py -- safely test Pragnosia on CPU while it trains on the GPU.

Forces CPU (zero GPU contention -> never disturbs a running trainer), snapshots
the live checkpoint (safe to read while the trainer is writing it), and loads it
at WHATEVER size it currently is -- it reads pragnosia.json, so it works after
grow-as-you-train has added layers/width too. Prints generations and the model's
OWN confidence on each prompt (low = familiar/sure, high = unsure).

  python3 probe.py                     # a default set of probes
  python3 probe.py "your prompt here"  # a custom prompt
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""        # CPU only -> cannot touch the training GPU
import sys, json, shutil, re, torch, torch.nn.functional as F
import s6_hybrid as H
from tokenizers import Tokenizer

cfg = json.load(open("pragnosia.json"))
if not os.path.exists(cfg["ckpt"]):
    sys.exit(f"no checkpoint yet at {cfg['ckpt']} -- let the trainer save one first.")
snap = "/tmp/_probe_ckpt.pt"; shutil.copy(cfg["ckpt"], snap)        # snapshot: safe vs. concurrent writes

tok = Tokenizer.from_file(cfg["tokenizer"])
m = H.SpinAttentionLM(cfg["vocab"], cfg["d"], cfg["heads"], cfg["layers"], mlp_mult=cfg.get("mlp_mult", 4))
m.load_state_dict(torch.load(snap, map_location="cpu", weights_only=True)); m.eval()
P = sum(p.numel() for p in m.parameters())

last = ""                                                          # show training progress if a log is present
try:
    vp = [l for l in open("pragnosia_train.log") if "VAL_PPL" in l]
    last = "| " + re.findall(r"it=\s*\d+\s+VAL_PPL=[\d.]+", vp[-1])[-1] if vp else ""
except Exception:
    pass
print(f"Pragnosia {P/1e6:.0f}M | {cfg['layers']} layers, d={cfg['d']}, mlp_mult={cfg.get('mlp_mult',4)}, "
      f"vocab={cfg['vocab']} | CPU {last}\n")

@torch.no_grad()
def conf(p):                                                       # the model's own surprise on the prompt
    ids = tok.encode(p).ids
    return 0.0 if len(ids) < 2 else F.cross_entropy(m(torch.tensor([ids]))[0, :-1], torch.tensor(ids[1:])).item()

@torch.no_grad()
def gen(prompt, n=28, rep=1.3):
    ids = tok.encode(prompt).ids; start = len(ids)
    for _ in range(n):
        lo = m(torch.tensor([ids[-cfg["ctx"]:]]))[0, -1].float()
        for t in set(ids[-40:]): lo[t] /= rep                      # mild repetition penalty
        nx = lo.argmax().item()
        if nx == 0: break
        ids.append(nx)
    return tok.decode(ids[start:]).strip()

if len(sys.argv) > 1:
    p = " ".join(sys.argv[1:])
    print(f"[confidence {conf(p):.1f}]  {p}\n  -> {gen(p, n=48)}")
else:
    print("=== probes (confidence = its own surprise; lower = more sure) ===")
    for p in ["Once upon a time", "Question: What is 5 plus 7?\nAnswer:", "The capital of Japan is",
              "def add(a, b):", "Photosynthesis is the process by which", "The opposite of hot is"]:
        print(f"  [conf {conf(p):.1f}] {p!r}\n     -> {gen(p)[:110]}\n")
