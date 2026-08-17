"""RESEARCH roadmap #3 -- growth SCHEDULE: how much time at each (cheap) small size before growing? Uses the best
operator from #62 (STACKING). Fixed total steps; vary the step split across 2L/4L/6L phases. Hypothesis: EARLY-heavy
(more learning while small+cheap) reaches target for least compute -- the developmental 'grows like a brain' timing.
  scratch  : 6L from init
  equal    : [1/3,1/3,1/3]  (the #62 baseline, -22%)
  early    : [1/2,1/4,1/4]  (max cheap-size learning)
  late     : [1/6,1/6,2/3]  (little cheap learning -> should approach scratch)
compute = params*tokens; report compute-to-target for each.
"""
import warnings,os; warnings.filterwarnings("ignore"); os.environ["PYTHONWARNINGS"]="ignore"; os.environ.setdefault("RECALL_FRAC","0")
import sys,numpy as np,torch,torch.nn as nn,torch.nn.functional as F
sys.path.insert(0,"/opt/code/membrae")
import scale_train as S
from fastcore_v import VChunkRecall
DEV="cuda"; DATA=S.DATA; VOCAB=S.VOCAB; CTX=512; d=384; H=6
VAL_I=np.random.RandomState(0).randint(0,len(DATA)-CTX-1,size=64)
def vx():
    x=np.stack([DATA[j:j+CTX] for j in VAL_I]).astype(np.int64); y=np.stack([DATA[j+1:j+CTX+1] for j in VAL_I]).astype(np.int64)
    return torch.from_numpy(x).to(DEV),torch.from_numpy(y).to(DEV)
class Blk(nn.Module):
    def __init__(s): super().__init__(); s.n1=nn.LayerNorm(d); s.mix=VChunkRecall(d,heads=H,feat=8,chunk=128,stable=True); s.n2=nn.LayerNorm(d); s.mlp=nn.Sequential(nn.Linear(d,4*d),nn.GELU(),nn.Linear(4*d,d))
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
    def grow_stack(s):
        for _ in range(2): b=Blk().to(DEV); b.load_state_dict(s.blocks[-1].state_dict()); s.blocks.append(b)
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
def sched(split,TOT):
    torch.manual_seed(0); m=LM(2).to(DEV); cum=0; C=[]
    s2,s4,s6=[int(TOT*f) for f in split]
    cum,c=train(m,s2,cum); C+=c
    m.grow_stack(); cum,c=train(m,s4,cum); C+=c
    m.grow_stack(); cum,c=train(m,s6,cum); C+=c
    return C
def c2t(cv,L):
    for cc,l in cv:
        if l<=L: return cc
    return None
if __name__=="__main__":
    TOT=4500
    torch.manual_seed(0); mS=LM(6).to(DEV); _,cS=train(mS,TOT,0)
    res={"scratch":cS,"equal":sched((1/3,1/3,1/3),TOT),"early":sched((1/2,1/4,1/4),TOT),"late":sched((1/6,1/6,2/3),TOT)}
    tgt=max(cv[-1][1] for cv in res.values())+0.05
    print(f"GROWTH-SCHEDULE (stacking operator, target vloss {tgt:.3f}):\n",flush=True)
    cs=c2t(cS,tgt)
    for name,cv in res.items():
        ct=c2t(cv,tgt); sav=f"{100*(1-ct/cs):+.0f}% vs scratch" if (ct and cs and name!='scratch') else ""
        print(f"  {name:8s} final={cv[-1][1]:.3f}@{cv[-1][0]/1e12:.0f}e12  tgt@{ct/1e12 if ct else float('nan'):.0f}e12  {sav}",flush=True)
    print("DONE",flush=True)
