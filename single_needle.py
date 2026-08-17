"""DIAGNOSTIC for the #52 long-context wall: what rescues LARGE-NEEDLE recall? One key->value needle, then D noise
tokens (distinct filler vocab), then the key again -> predict the value across distance D. This mirrors the live
needle@1792=0% failure. Compare 3 cores trained identically on mixed distance, eval recall vs D:
  (1) ungated VChunkRecall feat=8   = the LIVE core
  (2) GATED GatedRecall  feat=8     = per-head decay (learn to forget noise, keep the fact)
  (3) ungated VChunkRecall feat=32  = pure capacity (bigger state)
Whichever holds recall as D grows is the lever we need for large needle.
"""
import warnings,os; warnings.filterwarnings("ignore"); os.environ["PYTHONWARNINGS"]="ignore"
import sys,random,numpy as np; sys.path.insert(0,"/opt/code/membrae")
import torch,torch.nn as nn,torch.nn.functional as F
from fastcore_v import VChunkRecall
from gated_core import GatedRecall
DEV="cuda"
Kv=32; Vv=32; Fn=96; BASE=1+Kv+Vv; V=BASE+Fn                 # keys 1..32, values 33..64, filler 65..160
def seed(s): torch.manual_seed(s); random.seed(s); np.random.seed(s)
def needle(B,dist):
    T=dist+3; x=torch.zeros(B,T,dtype=torch.long); y=torch.full((B,T),-100,dtype=torch.long)
    for b in range(B):
        k=random.randint(1,Kv); v=random.randint(Kv+1,Kv+Vv)
        fill=[random.randint(BASE,BASE+Fn-1) for _ in range(dist)]     # D tokens of pure noise
        s=[k,v]+fill+[k]; x[b,:len(s)]=torch.tensor(s[:T]); y[b,T-1]=v # query=key at end -> predict value
    return x.to(DEV),y.to(DEV)
class Blk(nn.Module):
    def __init__(s,d,mk): super().__init__(); s.n1=nn.LayerNorm(d); s.mix=mk(d); s.n2=nn.LayerNorm(d); s.mlp=nn.Sequential(nn.Linear(d,4*d),nn.GELU(),nn.Linear(4*d,d))
    def forward(s,x): m=s.mix(s.n1(x)); m=m[0] if isinstance(m,tuple) else m; x=x+m; return x+s.mlp(s.n2(x))
class LM(nn.Module):
    def __init__(s,d,L,mk): super().__init__(); s.emb=nn.Embedding(V,d); s.pos=nn.Embedding(4096,d); s.blocks=nn.ModuleList([Blk(d,mk) for _ in range(L)]); s.head=nn.Linear(d,V)
    def forward(s,x):
        h=s.emb(x)+s.pos(torch.arange(x.shape[1],device=x.device))
        for b in s.blocks: h=b(h)
        return s.head(h)
DISTS=[64,256,512,1024]
def run(name,mk,steps=3500,B=16):
    seed(0); m=LM(128,2,mk).to(DEV); opt=torch.optim.AdamW(m.parameters(),lr=2e-3)
    for it in range(steps):
        d=random.choice(DISTS); x,y=needle(B,d)
        F.cross_entropy(m(x).reshape(-1,V),y.reshape(-1),ignore_index=-100).backward()
        nn.utils.clip_grad_norm_(m.parameters(),1.0); opt.step(); opt.zero_grad()
    m.eval(); res={}
    with torch.no_grad():
        for d in DISTS:
            c=t=0
            for _ in range(25):
                x,y=needle(B,d); p=m(x).argmax(-1); msk=y!=-100; c+=(p[msk]==y[msk]).sum().item(); t+=msk.sum().item()
            res[d]=100.0*c/t
    print(f"  {name:28s} "+"  ".join(f'@{d}={res[d]:4.0f}%' for d in DISTS),flush=True); return res
if __name__=="__main__":
    print(f"SINGLE-NEEDLE vs distance (chance={100/Vv:.0f}%). one k->v, then D noise, then query:\n",flush=True)
    print(f"  {'core':28s} "+"  ".join(f'  @{d}' for d in DISTS),flush=True)
    r1=run("ungated feat=8 (LIVE core)", lambda d: VChunkRecall(d,heads=4,feat=8,chunk=128,stable=False))
    r2=run("GATED feat=8", lambda d: GatedRecall(d,heads=4,feat=8,chunk=128))
    r3=run("ungated feat=32 (capacity)", lambda d: VChunkRecall(d,heads=4,feat=32,chunk=128,stable=False))
    print(f"\nVERDICT: lever = whichever holds recall as D->1024. gate@1024={r2[1024]:.0f}% vs ungated@1024={r1[1024]:.0f}% vs cap@1024={r3[1024]:.0f}%",flush=True)
    print("DONE",flush=True)
