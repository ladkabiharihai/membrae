"""SCALE TEST -- does the growth-efficiency result (36% cheaper, #65/#66) HOLD at a larger budget? Small study was
d=384/L6/4500 steps. Here: d=512, target L=8, 9000 steps (~3x compute), the WINNING recipe (early-heavy schedule +
HYBRID operator) vs from-scratch. Measure compute-to-target. Faculty (MQAR-16) check at the end (recall survives scale
+ growth). If ~30%+ saving holds at 3x budget, the headline number is real, not a small-budget artifact.
"""
import warnings,os; warnings.filterwarnings("ignore"); os.environ["PYTHONWARNINGS"]="ignore"; os.environ.setdefault("RECALL_FRAC","0")
import sys,numpy as np,torch,torch.nn as nn,torch.nn.functional as F
sys.path.insert(0,"/opt/code/membrae")
import scale_train as S
from fastcore_v import VChunkRecall, recall
DEV="cuda"; DATA=S.DATA; VOCAB=S.VOCAB; CTX=512; d=512; H=8
VAL_I=np.random.RandomState(0).randint(0,len(DATA)-CTX-1,size=64)
def vx():
    x=np.stack([DATA[j:j+CTX] for j in VAL_I]).astype(np.int64); y=np.stack([DATA[j+1:j+CTX+1] for j in VAL_I]).astype(np.int64)
    return torch.from_numpy(x).to(DEV),torch.from_numpy(y).to(DEV)
class Blk(nn.Module):
    def __init__(s): super().__init__(); s.n1=nn.LayerNorm(d); s.mix=VChunkRecall(d,heads=H,feat=8,chunk=128,stable=True); s.n2=nn.LayerNorm(d); s.mlp=nn.Sequential(nn.Linear(d,4*d),nn.GELU(),nn.Linear(4*d,d)); s.alpha=nn.Parameter(torch.ones(1))
    def forward(s,x): h=x+s.mix(s.n1(x)); h=h+s.mlp(s.n2(h)); return x+s.alpha*(h-x)
class LM(nn.Module):
    def __init__(s,L):
        super().__init__(); s.emb=nn.Embedding(VOCAB,d); s.pos=nn.Embedding(8192,d)
        s.blocks=nn.ModuleList([Blk() for _ in range(L)]); s.lnf=nn.LayerNorm(d); s.head=nn.Linear(d,VOCAB,bias=False); s.head.weight=s.emb.weight
        nn.init.normal_(s.emb.weight,0,0.02); nn.init.normal_(s.pos.weight,0,0.02)
    def forward(s,x):
        h=s.emb(x)+s.pos(torch.arange(x.shape[1],device=x.device)[None])
        for b in s.blocks: h=b(h)
        return s.head(s.lnf(h))
    def grow_hybrid(s):
        for _ in range(2): b=Blk().to(DEV); b.load_state_dict(s.blocks[-1].state_dict()); b.alpha.data.zero_(); s.blocks.append(b)
def npar(m): return sum(p.numel() for p in m.parameters())
@torch.no_grad()
def vloss(m): m.eval(); x,y=vx(); l=F.cross_entropy(m(x).reshape(-1,VOCAB),y.reshape(-1)).item(); m.train(); return l
def bs(B):
    i=np.random.randint(0,len(DATA)-CTX-1,size=B); x=np.stack([DATA[j:j+CTX] for j in i]).astype(np.int64); y=np.stack([DATA[j+1:j+CTX+1] for j in i]).astype(np.int64)
    return torch.from_numpy(x).to(DEV),torch.from_numpy(y).to(DEV)
def train(m,steps,cum,B=8,lr=6e-4,warm=200):
    opt=torch.optim.AdamW(m.parameters(),lr=lr,betas=(0.9,0.95),weight_decay=0.1); curve=[]
    for it in range(1,steps+1):
        for gp in opt.param_groups: gp["lr"]=lr*min(1.0,it/warm)
        x,y=bs(B); loss=F.cross_entropy(m(x).reshape(-1,VOCAB),y.reshape(-1))
        if torch.isfinite(loss): opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(m.parameters(),1.0); opt.step()
        cum+=npar(m)*B*CTX
        if it%150==0: curve.append((cum,vloss(m)))
    return cum,curve
def c2t(cv,L):
    for cc,l in cv:
        if l<=L: return cc
    return None
if __name__=="__main__":
    TOT=9000
    print(f"SCALE TEST: d={d} target L=8, {TOT} steps (~3x). recipe=early+hybrid vs scratch.\n",flush=True)
    torch.manual_seed(0); mS=LM(8).to(DEV); _,cS=train(mS,TOT,0); print(f"  scratch 8L final={cS[-1][1]:.3f}@{cS[-1][0]/1e12:.0f}e12",flush=True)
    # recipe: grow 2->4->6->8, EARLY-heavy [0.40,0.25,0.20,0.15], HYBRID operator
    torch.manual_seed(0); mB=LM(2).to(DEV); cum=0; C=[]
    for i,frac in enumerate([0.40,0.25,0.20,0.15]):
        if i>0: mB.grow_hybrid()
        cum,c=train(mB,int(TOT*frac),cum); C+=c
        print(f"  recipe {2+2*i}L done vloss={C[-1][1]:.3f} compute={cum/1e12:.0f}e12",flush=True)
    tgt=max(cS[-1][1],C[-1][1])+0.05; cs=c2t(cS,tgt); cb=c2t(C,tgt)
    print(f"\n  target vloss={tgt:.3f}: scratch@{cs/1e12 if cs else float('nan'):.0f}e12  recipe@{cb/1e12 if cb else float('nan'):.0f}e12",flush=True)
    if cs and cb: print(f"  RECIPE {100*(1-cb/cs):+.0f}% compute vs scratch @ 3x budget ({'HOLDS' if cb<cs else 'does NOT hold'})",flush=True)
    mq=recall(lambda: VChunkRecall(d,heads=H,feat=8,chunk=64), n=16, d=d, L=2, steps=2500)
    print(f"  faculty check MQAR-16 @ d={d}: {mq*100:.0f}%",flush=True)
    print("DONE",flush=True)
