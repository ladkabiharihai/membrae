"""DECISIVE test behind #48: is recall capacity bound by STATE (feat) or DEPTH (layers)? Uses the KNOWN-GOOD MQAR
harness from fastcore_v (untied head, the one that hit ~100%@16) -- NOT a reimplementation. Push to n=32 stored
pairs to create a real capacity bottleneck, then:
  (A) fix depth=2, vary feat  -> if recall CLIMBS with feat, recall is STATE-bound.
  (B) fix feat=2 (bottlenecked), vary depth -> if recall stays FLAT/low, depth cannot buy recall capacity.
That pair of curves is the proof (or refutation) that depth-growth was the wrong axis for recall (#48).
"""
import warnings,os; warnings.filterwarnings("ignore"); os.environ["PYTHONWARNINGS"]="ignore"
import sys; sys.path.insert(0,"/opt/code/membrae")
from fastcore_v import recall, VChunkRecall
d=128; N=16; STEPS=3500   # n=16 = converging regime (n=32 floors on shared GPU, under-converged)
print(f"MQAR recall via reference harness (n={N} stored pairs, d={d}, {STEPS} steps). Axis of recall?\n",flush=True)

print("(A) FIX depth=2, vary feat (STATE axis):",flush=True)
Ares=[]
for feat in [1,2,4,8,16,32]:
    acc=recall(lambda feat=feat: VChunkRecall(d,heads=4,feat=feat,chunk=64), n=N, d=d, L=2, steps=STEPS)
    Ares.append((feat,acc)); print(f"    feat={feat:2d}  depth=2   recall={acc*100:5.1f}%",flush=True)

print("\n(B) FIX feat=2 (bottlenecked), vary depth (DEPTH axis):",flush=True)
Bres=[]
for L in [1,2,4,8]:
    acc=recall(lambda: VChunkRecall(d,heads=4,feat=2,chunk=64), n=N, d=d, L=L, steps=STEPS)
    Bres.append((L,acc)); print(f"    feat= 2  depth={L}   recall={acc*100:5.1f}%",flush=True)

aglow,aghi=Ares[0][1], max(a for _,a in Ares); bghi=max(a for _,a in Bres)
print(f"\nfeat 1->32 moved recall {Ares[0][1]*100:.0f}%->{aghi*100:.0f}%  |  depth 1->8 (feat2) moved recall {Bres[0][1]*100:.0f}%->{bghi*100:.0f}%",flush=True)
print("VERDICT: if the feat sweep spans a big range and the depth sweep is flat -> recall is STATE-bound (confirms #48).",flush=True)
