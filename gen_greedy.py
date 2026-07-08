import sys, time, os; sys.path.insert(0, "/home/ashish/Downloads/membrae")
import torch, brain
b = brain.Brain(learn=False)
os.makedirs("generations", exist_ok=True)
P = lambda *a: print(*a, flush=True)
P("INIT OK  (GREEDY temp=0 sweep, same prompt/rep as the temp=0.8 run)")
seed = "The history of science is a long story that begins"
sids = b.tok.encode(seed).ids
for n in [256, 512, 1024, 2048, 4096, 8000]:
    t0 = time.time()
    out = list(b.lm.generate(sids, n_new=n, window=256, overlap=64, temp=0.0, rep=1.3, eos=-999))
    dt = time.time() - t0
    txt = b.tok.decode(out)
    open(f"generations/gen_greedy_{n}.log", "w").write(txt)
    words = txt.split()
    tail = words[-200:] if len(words) > 200 else words
    uniq = len(set(w.lower() for w in tail)) / max(1, len(tail))   # low -> greedy loop
    P(f"  {n:5d} tok  {dt:5.0f}s  {n/max(dt,1):4.0f} tok/s  chars={len(txt):6d}  tail-uniq={uniq:.2f}  -> generations/gen_greedy_{n}.log")
P("GREEDY DONE")
