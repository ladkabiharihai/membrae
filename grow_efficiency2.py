"""RESEARCH roadmap #2 -- growth OPERATOR: does a better warm-start beat identity-block grow (and push the #61 ~15%
compute saving higher)? Three arms, same target (d=384 L=6), same val set, loss-vs-CUMULATIVE-COMPUTE:
  SCRATCH        6L random init.
  GROW-IDENTITY  2->4->6, new blocks zero-init (FUNCTION-PRESERVING; faculties intact at t=0). [reproduces #61]
  GROW-STACK     2->4->6, new blocks = COPIES of trained blocks (warm-start; NOT function-preserving -- output jumps
                 at grow, recovers fast). The classic gradual-stacking / bert2BERT operator.
Tests the trade-off: strict function-preservation (identity) vs efficiency (stacking). Report compute-to-target each.
"""
import warnings,os,copy; warnings.filterwarnings("ignore"); os.environ["PYTHONWARNINGS"]="ignore"; os.environ.setdefault("RECALL_FRAC","0")
import sys,numpy as np,torch,torch.nn as nn,torch.nn.functional as F
sys.path.insert(0,"/opt/code/membrae")
import scale_train as S
from fastcore_v import VChunkRecall
DEV="cuda"; DATA=S.DATA; VOCAB=S.VOCAB; CTX=512; d=384; H=6
rng=np.random.RandomState(0); VAL_I=rng.randint(0,len(DATA)-CTX-1,size=64)
def vx():
    x=np.stack([DATA[j:j+CTX] for j in VAL_I]).astype(np.int64); y=np.stack([DATA[j+1:j+CTX+1] for j in VAL_I]).astype(np.int64)
    return torch.from_numpy(x).to(DEV),torch.from_numpy(y).to(DEV)
class Blk(nn.Module):
    def __init__(s): super().__init__(); s.n1=nn.LayerNorm(d); s.mix=VChunkRecall(d,heads=H,feat=8,chunk=128,stable=True); s.n2=nn.LayerNorm(d); s.mlp=nn.Sequential(nn.Linear(d,4*d),nn.GELU(),nn.Linear(4*d,d))
    def idty(s): [nn.init.zeros_(p) for p in (s.mix.o.weight,s.mix.o.bias,s.mlp[-1].weight,s.mlp[-1].bias)]
    def forward(s,x): x=x+s.mix(s.n1(x)); return x+s.mlp(s.n2(x))
class LM(nn.Module):
    def __init__(s,L):
        super().__init__(); s.emb=nn.Embedding(VOCAB,d); s.pos=nn.Embedding(8192,d)
        s.blocks=nn.ModuleList([Blk() for _ in range(L)]); s.lnf=nn.LayerNorm(d); s.head=nn.Linear(d,VOCAB,bias=False); s.head.weight=s.emb.weight
        nn.init.normal_(s.emb.weight,0,0.02); nn.init.normal_(s.pos.weight,0,0.02)
    def forward(s,x):
        h=s.emb(x)+s.pos(torch.arange(x.shape[1],device=x.device)[None])
        for b in s.blocks: h=b(h)
        return s.head(s.lnf(h))
    def grow(s,mode):
        for _ in range(2):
            b=Blk().to(DEV)
            if mode=="identity": b.idty()
            else: b.load_state_dict(s.blocks[-1].state_dict())   # STACK: copy the last trained block
            s.blocks.append(b)
def npar(m): return sum(p.numel() for p in m.parameters())
@torch.no_grad()
def vloss(m): m.eval(); x,y=vx(); l=F.cross_entropy(m(x).reshape(-1,VOCAB),y.reshape(-1)).item(); m.train(); return l
def bs(B):
    i=np.random.randint(0,len(DATA)-CTX-1,size=B); x=np.stack([DATA[j:j+CTX] for j in i]).astype(np.int64); y=np.stack([DATA[j+1:j+CTX+1] for j in i]).astype(np.int64)
    return torch.from_numpy(x).to(DEV),torch.from_numpy(y).to(DEV)
def train(m,steps,cum,B=8,lr=6e-4,warm=150):
    opt=torch.optim.AdamW(m.parameters(),lr=lr,betas=(0.9,0.95),weight_decay=0.1); curve=[]
    for it in range(1,steps+1):
        for gp in opt.param_groups: gp["lr"]=lr*min(1.0,it/warm)
        x,y=bs(B); loss=F.cross_entropy(m(x).reshape(-1,VOCAB),y.reshape(-1))
        if torch.isfinite(loss): opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(m.parameters(),1.0); opt.step()
        cum+=npar(m)*B*CTX
        if it%100==0: curve.append((cum,vloss(m)))
    return cum,curve
def run_grow(mode,TOT):
    torch.manual_seed(0); m=LM(2).to(DEV); cum=0; C=[]
    cum,c=train(m,TOT//3,cum); C+=c
    m.grow(mode); cum,c=train(m,TOT//3,cum); C+=c
    m.grow(mode); cum,c=train(m,TOT//3,cum); C+=c
    return C
def c2t(curve,L):
    for cc,l in curve:
        if l<=L: return cc
    return None
if __name__=="__main__":
    TOT=4500
    torch.manual_seed(0); mS=LM(6).to(DEV); _,cS=train(mS,TOT,0)
    cI=run_grow("identity",TOT); cK=run_grow("stack",TOT)
    tgt=max(cS[-1][1],cI[-1][1],cK[-1][1])+0.05
    print(f"GROWTH-OPERATOR (target vloss {tgt:.3f}):\n",flush=True)
    for name,cv in [("SCRATCH 6L",cS),("GROW-identity",cI),("GROW-stack",cK)]:
        ct=c2t(cv,tgt); print(f"  {name:16s} final={cv[-1][1]:.3f}@{cv[-1][0]/1e12:.0f}e12   reaches-tgt@ {ct/1e12 if ct else float('nan'):.0f}e12",flush=True)
    cs=c2t(cS,tgt)
    for name,cv in [("GROW-identity",cI),("GROW-stack",cK)]:
        ct=c2t(cv,tgt)
        if ct and cs: print(f"  {name}: {100*(1-ct/cs):+.0f}% compute vs scratch",flush=True)
    print("DONE",flush=True)
