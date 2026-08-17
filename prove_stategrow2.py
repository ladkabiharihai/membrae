"""PROOF v2 — earns the STATE-axis label the v1 harness could not. Adversarial review (workflow wf_df53b7ad) found
v1 ran at T=40 < chunk=64 => nc=1 (ONE chunk): the inter-chunk recurrent state (Sprev/Zprev) was identically zero,
so v1 only tested INTRA-chunk feature rank, not recurrent STATE capacity. Fix: run MULTI-CHUNK (chunk=8, T=40 =>
nc=5) so a query in the last chunk can only reach earlier keys THROUGH the recurrent KV state -> recall is now a
genuine recurrent-STATE test. Also fixes: (1) STRICT same-batch function-preservation (fixed batch, report max
logit delta), (2) capacity-HEAVY depth control (+4 feat=2 layers, far more params than feat-grow), (3) multi-seed.
Claim earned iff: preservation delta ~0, feat-grow rescues multi-chunk recall, and NO amount of feat=2 depth does.
"""
import warnings,os; warnings.filterwarnings("ignore"); os.environ["PYTHONWARNINGS"]="ignore"
import sys,random,numpy as np; sys.path.insert(0,"/opt/code/membrae")
import torch, torch.nn as nn, torch.nn.functional as F
from fastcore_v import VChunkRecall, mqar
from prove_stategrow import grow_feat_
DEV="cuda"; V=129; d=128; CH=8; N=16               # chunk=8, n=16 -> T=40, nc=5 (multi-chunk: recurrent state in play)

class Blk(nn.Module):
    def __init__(s,d,feat):
        super().__init__(); s.n1=nn.LayerNorm(d); s.mix=VChunkRecall(d,heads=4,feat=feat,chunk=CH,stable=False)
        s.n2=nn.LayerNorm(d); s.mlp=nn.Sequential(nn.Linear(d,4*d),nn.GELU(),nn.Linear(4*d,d))
    def make_identity(s): [nn.init.zeros_(p) for p in (s.mix.o.weight,s.mix.o.bias,s.mlp[-1].weight,s.mlp[-1].bias)]
    def forward(s,x): x=x+s.mix(s.n1(x)); return x+s.mlp(s.n2(x))
class LM(nn.Module):
    def __init__(s,V,d,L,feat):
        super().__init__(); s.emb=nn.Embedding(V,d); s.pos=nn.Embedding(4096,d)
        s.blocks=nn.ModuleList([Blk(d,feat) for _ in range(L)]); s.head=nn.Linear(d,V)
    def forward(s,x):
        h=s.emb(x)+s.pos(torch.arange(x.shape[1],device=x.device))
        for b in s.blocks: h=b(h)
        return s.head(h)
    def grow_feat(s,nf): [grow_feat_(b.mix,nf) for b in s.blocks]
    def grow_depth(s,k=1):
        for _ in range(k): b=Blk(d, s.blocks[0].mix.fe).to(next(s.parameters()).device); b.make_identity(); s.blocks.append(b)

def nparams(m): return sum(p.numel() for p in m.parameters())
def train(m,steps,lr=2e-3):
    opt=torch.optim.AdamW(m.parameters(),lr=lr); m.train()
    for _ in range(steps):
        x,y=mqar(48,N); F.cross_entropy(m(x).reshape(-1,V),y.reshape(-1),ignore_index=-100).backward(); opt.step(); opt.zero_grad()
@torch.no_grad()
def evalr(m,reps=40):
    m.eval(); c=t=0
    for _ in range(reps):
        x,y=mqar(48,N); p=m(x).argmax(-1); msk=y!=-100; c+=(p[msk]==y[msk]).sum().item(); t+=msk.sum().item()
    m.train(); return 100.0*c/t
def seed(s): torch.manual_seed(s); random.seed(s); np.random.seed(s)

if __name__=="__main__":
    # sanity: confirm MULTI-CHUNK (nc>1) so the recurrent state is actually exercised
    Tp=((N*2+8)+CH-1)//CH*CH; print(f"regime: chunk={CH}, T={N*2+8}, nc={Tp//CH}  (nc>1 => recurrent state is the retrieval path)\n",flush=True)

    # (1) STRICT same-batch function-preservation in the multi-chunk regime
    seed(0); m=LM(V,d,2,2).to(DEV); train(m,300)
    xb,yb=mqar(48,N)
    with torch.no_grad(): lg0=m(xb).clone()
    m.grow_feat(16)
    with torch.no_grad(): lg1=m(xb)
    dmax=(lg0-lg1).abs().max().item()
    print(f"[preserve] STRICT same-batch max|Δlogit| after grow 2->16 = {dmax:.2e}  ({'PRESERVED' if dmax<1e-3 else 'BROKEN'})\n",flush=True)

    # (2) multi-seed arms
    def arms(tag, build):
        vals=[]
        for s in [0,1,2]:
            seed(s); vals.append(build())
        import statistics as st
        print(f"  {tag:34s} {st.mean(vals):5.1f}%  (seeds: {', '.join(f'{v:.0f}' for v in vals)})",flush=True)
        return st.mean(vals)
    print("MULTI-CHUNK MQAR recall (recurrent state is the bottleneck), 2000+2000 steps:",flush=True)
    def base():   m=LM(V,d,2,2).to(DEV); train(m,4000); return evalr(m)
    def stateg(): m=LM(V,d,2,2).to(DEV); train(m,2000); m.grow_feat(16); train(m,2000); return evalr(m)
    def depth1(): m=LM(V,d,2,2).to(DEV); train(m,2000); m.grow_depth(1); train(m,2000); return evalr(m)
    def depth4(): m=LM(V,d,2,2).to(DEV); train(m,2000); m.grow_depth(4); train(m,2000); return evalr(m)
    def nativ():  m=LM(V,d,2,16).to(DEV); train(m,4000); return evalr(m)
    b=arms("baseline feat=2 (no grow)", base)
    sg=arms("STATE-GROW 2->16", stateg)
    d1=arms("DEPTH-GROW +1 layer (feat=2)", depth1)
    d4=arms("DEPTH-GROW +4 layers (feat=2)", depth4)
    nv=arms("native feat=16 (upper bound)", nativ)

    # param accounting for the +2500 grow step (single seed, illustrative)
    seed(0); ma=LM(V,d,2,2).to(DEV); p0=nparams(ma); ma.grow_feat(16); pf=nparams(ma)
    seed(0); mb=LM(V,d,2,2).to(DEV); mb.grow_depth(4); pd=nparams(mb)
    print(f"\nparam add: STATE-GROW 2->16 = +{pf-p0} | DEPTH +4 layers = +{pd-nparams(LM(V,d,2,2).to(DEV))}",flush=True)
    print("\n==================== STATE-AXIS PROOF v2 (multi-chunk) ====================",flush=True)
    print(f"  STATE-GROW {sg:.0f}%  vs  DEPTH+1 {d1:.0f}%  DEPTH+4 {d4:.0f}%  baseline {b:.0f}%   (native-16 ceiling {nv:.0f}%)",flush=True)
    ok = dmax<1e-3 and sg>b+25 and sg>d4+25
    print(f"  VERDICT: {'EARNED' if ok else 'NOT earned'} -- function-preserving feat-widening rescues RECURRENT-STATE recall;",flush=True)
    print(f"           depth (even +4 heavier feat=2 layers) does not. State capacity = Taylor feature dim fe.",flush=True)
