"""DISAMBIGUATE the long-context wall. #52: live needle@1792=0% on REAL prose. single_needle.py showed the ungated
core does a single needle in NOISE at 100%@1024 -> DISTANCE is not the wall. So the wall is either (a) DISTRACTOR
density (many competing bindings, like real prose) or (b) LANGUAGE grounding. This isolates (a): needle k0->v0 first,
then D clean DISTRACTOR pairs, then query k0 -> predict v0. If cores fail as D grows -> distractor density is the
wall (and which core holds = the lever). If they hold -> the live failure is LANGUAGE, not the mechanism.
"""
import warnings,os; warnings.filterwarnings("ignore"); os.environ["PYTHONWARNINGS"]="ignore"
import sys,random,numpy as np; sys.path.insert(0,"/opt/code/membrae")
import torch,torch.nn as nn,torch.nn.functional as F
from fastcore_v import VChunkRecall
from gated_core import GatedRecall
DEV="cuda"
K=256; Vv=256; V=1+K+Vv
def seed(s): torch.manual_seed(s); random.seed(s); np.random.seed(s)
def batch(B,D):                                              # needle + D distractor pairs, query needle
    T=2*(D+1)+1; x=torch.zeros(B,T,dtype=torch.long); y=torch.full((B,T),-100,dtype=torch.long)
    for b in range(B):
        ks=random.sample(range(1,K+1),D+1); vs=[random.randint(K+1,K+Vv) for _ in ks]
        seq=[]
        for a,c in zip(ks,vs): seq+=[a,c]                    # needle is the FIRST pair (max distance)
        seq+=[ks[0]]                                         # query the needle key
        x[b,:len(seq)]=torch.tensor(seq[:T]); y[b,len(seq)-1]=vs[0]
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
DS=[8,32,64,128]
def run(name,mk,steps=4000,B=16):
    seed(0); m=LM(128,2,mk).to(DEV); opt=torch.optim.AdamW(m.parameters(),lr=2e-3)
    for it in range(steps):
        D=random.choice(DS); x,y=batch(B,D)
        F.cross_entropy(m(x).reshape(-1,V),y.reshape(-1),ignore_index=-100).backward()
        nn.utils.clip_grad_norm_(m.parameters(),1.0); opt.step(); opt.zero_grad()
    m.eval(); res={}
    with torch.no_grad():
        for D in DS:
            c=t=0
            for _ in range(25):
                x,y=batch(B,D); p=m(x).argmax(-1); msk=y!=-100; c+=(p[msk]==y[msk]).sum().item(); t+=msk.sum().item()
            res[D]=100.0*c/t
    print(f"  {name:28s} "+"  ".join(f'D={D}:{res[D]:4.0f}%' for D in DS),flush=True); return res
if __name__=="__main__":
    print(f"NEEDLE among D clean DISTRACTOR pairs (Fd feat8=73, feat32=1057; chance={100/Vv:.1f}%):\n",flush=True)
    r1=run("ungated feat=8 (LIVE core)", lambda d: VChunkRecall(d,heads=4,feat=8,chunk=128,stable=False))
    r2=run("GATED feat=8", lambda d: GatedRecall(d,heads=4,feat=8,chunk=128))
    r3=run("ungated feat=32 (capacity)", lambda d: VChunkRecall(d,heads=4,feat=32,chunk=128,stable=False))
    print(f"\nVERDICT @D=128: ungated8={r1[128]:.0f}% gated8={r2[128]:.0f}% cap32={r3[128]:.0f}%  -> lever = whichever survives distractors",flush=True)
    print("DONE",flush=True)
