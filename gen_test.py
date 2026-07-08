import sys, time, os; sys.path.insert(0, "/home/ashish/Downloads/membrae")
import torch, brain
b = brain.Brain(learn=False)
os.makedirs("generations", exist_ok=True)
P = lambda *a: print(*a, flush=True)
torch.manual_seed(0)
P("INIT OK  (%.0fM, val-ppl target ~18.6)" % (b.n_params() / 1e6))

# ---- IDENTITY: what does the RAW LM say BY ITSELF (bypassing the controller self-model)? ----
P("\n=== RAW-LM IDENTITY PROBE (the weights talking, not the controller) ===")
idlog = []
for prompt in ["<user> Who are you? <assistant>", "<user> What is your name? <assistant>",
               "My name is", "I am Pragnosia, a", "The name of this model is"]:
    out = b.generate_text(prompt, n=30).strip()
    line = f"  {prompt!r} -> {out[:130]!r}"
    P(line); idlog.append(line)
open("generations/identity_probe.log", "w").write("\n".join(idlog))

# ---- LONG GENERATION 256 -> 8k (windowed, O(1)/token) ----
P("\n=== LONG GENERATION (windowed carrier, temp=0.8) ===")
seed = "The history of science is a long story that begins"
sids = b.tok.encode(seed).ids
for n in [256, 512, 1024, 2048, 4096, 8000]:
    t0 = time.time()
    out = list(b.lm.generate(sids, n_new=n, window=256, overlap=64, temp=0.8, rep=1.3, eos=-999))
    dt = time.time() - t0
    txt = b.tok.decode(out)
    open(f"generations/gen_{n}.log", "w").write(txt)
    words = txt.split()
    tail = words[-200:] if len(words) > 200 else words
    uniq = len(set(w.lower() for w in tail)) / max(1, len(tail))   # low -> looping/repetition
    P(f"  {n:5d} tok  {dt:5.0f}s  {n/max(dt,1):4.0f} tok/s  chars={len(txt):6d}  tail-uniq={uniq:.2f}  -> generations/gen_{n}.log")
P("GEN DONE")
