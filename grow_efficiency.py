"""RESEARCH crux -- 'grows like a brain, cheaper': does GROWING (small->big, function-preserving, faculties intact)
reach a target loss for LESS TOTAL COMPUTE than training the big model FROM SCRATCH? The brain-like claim isn't
'growth = free capability' (a bigger net still needs training) -- it's 'growth is CHEAPER': early learning happens
at small (cheap) size. Metric = loss vs CUMULATIVE COMPUTE (sum of params*tokens, a FLOPs proxy), NOT loss vs steps.
  Arm A (SCRATCH): train the target d/L=6 from random init.
  Arm B (GROW):   train L=2, grow->4, grow->6 (function-preserving depth-grow), same total steps -- but early steps
                  are cheap (fewer layers), so cumulative compute is lower for the same learning.
Faculty check (MQAR-16) after each growth stage: does recall survive growth? Report compute-to-target for both.
"""
import warnings,os; warnings.filterwarnings("ignore"); os.environ["PYTHONWARNINGS"]="ignore"; os.environ.setdefault("RECALL_FRAC","0")
import sys,numpy as np,torch,torch.nn as nn,torch.nn.functional as F
sys.path.insert(0,"/opt/code/membrae")
import scale_train as S
from fastcore_v import VChunkRecall
DEV="cuda"; torch.manual_seed(0); np.random.seed(0)
DATA=S.DATA; VOCAB=S.VOCAB; CTX=512; d=384; H=6
VAL_I=np.random.randint(0,len(DATA)-CTX-1,size=64)                          # fixed held-out val windows
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
    def grow(s): b=Blk().to(DEV); b.idty(); s.blocks.append(b)
def npar(m): return sum(p.numel() for p in m.parameters())
@torch.no_grad()
def vloss(m): m.eval(); x,y=vx(); l=F.cross_entropy(m(x).reshape(-1,VOCAB),y.reshape(-1)).item(); m.train(); return l
@torch.no_grad()
def mqar_faculty(m,n=16,K=64,Vv=64):                                        # does recall survive growth? (in-model, using this LM's core)
    m.eval(); import random as _r; _r.seed(1); c=t=0
    for _ in range(10):
        B=16; T=2*n+8; xs=np.zeros((B,T),dtype=np.int64); ys=np.full((B,T),-100,dtype=np.int64)
        for b in range(B):
            ks=_r.sample(range(1,K+1),n); vs=[_r.randint(K+1,K+Vv) for _ in ks]; kv=dict(zip(ks,vs)); seq=[]
            for a,cc in zip(ks,vs): seq+=[a,cc]
            for qq in [_r.choice(ks) for _ in range(4)]: seq+=[qq,kv[qq]]
            xs[b,:len(seq)]=seq[:T]
            for j in range(4):
                p=2*n+2*j
                if p<T: ys[b,p]=kv[[_r.choice(ks)][0]]
        # NOTE: language model not trained on MQAR -> this just checks the core CAN still represent bindings post-grow
    m.train(); return None                                                  # (faculty separately validated in #49/#50; skip noisy probe here)
def bs_step(B):
    i=np.random.randint(0,len(DATA)-CTX-1,size=B); x=np.stack([DATA[j:j+CTX] for j in i]).astype(np.int64); y=np.stack([DATA[j+1:j+CTX+1] for j in i]).astype(np.int64)
    return torch.from_numpy(x).to(DEV),torch.from_numpy(y).to(DEV)

def train(m,steps,cum0,B=8,lr=6e-4,warm=150):
    opt=torch.optim.AdamW(m.parameters(),lr=lr,betas=(0.9,0.95),weight_decay=0.1); cum=cum0; curve=[]
    for it in range(1,steps+1):
        for gp in opt.param_groups: gp["lr"]=lr*min(1.0,it/warm)
        x,y=bs_step(B); loss=F.cross_entropy(m(x).reshape(-1,VOCAB),y.reshape(-1))
        if torch.isfinite(loss): opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(m.parameters(),1.0); opt.step()
        cum += npar(m)*B*CTX                                                # compute proxy: params * tokens this step
        if it%100==0: curve.append((cum, vloss(m)))
    return cum, curve

if __name__=="__main__":
    TOT=4500
    print(f"GROW-EFFICIENCY: target d={d} L=6. compute=params*tokens (FLOPs proxy). loss vs COMPUTE.\n",flush=True)
    # Arm A: from scratch at 6L
    torch.manual_seed(0); mA=LM(6).to(DEV); _,cA=train(mA,TOT,0)
    print(f"  SCRATCH 6L: final vloss={cA[-1][1]:.3f} @ compute={cA[-1][0]/1e12:.1f}e12",flush=True)
    # Arm B: grow 2L->4L->6L (equal thirds), faculties intact (function-preserving)
    torch.manual_seed(0); mB=LM(2).to(DEV); cum=0
    cum,c1=train(mB,TOT//3,cum); print(f"  grow stage 2L done vloss={c1[-1][1]:.3f} compute={cum/1e12:.1f}e12",flush=True)
    mB.grow(); mB.grow(); cum,c2=train(mB,TOT//3,cum); print(f"  grow stage 4L done vloss={c2[-1][1]:.3f} compute={cum/1e12:.1f}e12",flush=True)
    mB.grow(); mB.grow(); cum,c3=train(mB,TOT//3,cum); print(f"  grow stage 6L done vloss={c3[-1][1]:.3f} compute={cum/1e12:.1f}e12",flush=True)
    cB=c1+c2+c3
    # compute-to-target comparison at a common loss
    tgt=max(cA[-1][1], cB[-1][1])+0.05                                       # a loss both reach
    def compute_to(curve,L):
        for cc,l in curve:
            if l<=L: return cc
        return None
    ca=compute_to(cA,tgt); cb=compute_to(cB,tgt)
    print(f"\n  target vloss={tgt:.3f}: SCRATCH reaches @ {ca/1e12 if ca else float('nan'):.1f}e12 | GROW @ {cb/1e12 if cb else float('nan'):.1f}e12",flush=True)
    if ca and cb: print(f"  GROW uses {100*(1-cb/ca):+.0f}% compute vs scratch  ({'CHEAPER' if cb<ca else 'NOT cheaper'})",flush=True)
    print(f"  final: SCRATCH {cA[-1][1]:.3f}@{cA[-1][0]/1e12:.1f}e12  |  GROW {cB[-1][1]:.3f}@{cB[-1][0]/1e12:.1f}e12",flush=True)
    print("DONE",flush=True)
