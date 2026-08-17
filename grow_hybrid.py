"""RESEARCH roadmap #4 -- capstone growth OPERATOR: can we get stacking's efficiency AND identity's instantaneous
function-preservation (faculties intact at every step)? HYBRID 'zero-gated copy': new block = COPY of a trained block
(warm weights) but its residual contribution is scaled by a learnable gate alpha INITIALIZED TO 0 -> at t=0 the block
is exact identity (function-preserving, no output jump), yet underneath it holds trained computation, so as alpha
grows it immediately adds USEFUL work (not learning from dead-zero like identity). ReZero/LayerScale applied to growth.
Compare identity / stack / hybrid at the best (early-heavy) schedule: compute-to-target + the output JUMP at each grow.
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
    def __init__(s): super().__init__(); s.n1=nn.LayerNorm(d); s.mix=VChunkRecall(d,heads=H,feat=8,chunk=128,stable=True); s.n2=nn.LayerNorm(d); s.mlp=nn.Sequential(nn.Linear(d,4*d),nn.GELU(),nn.Linear(4*d,d)); s.alpha=nn.Parameter(torch.ones(1))
    def forward(s,x):
        h=x+s.mix(s.n1(x)); h=h+s.mlp(s.n2(h)); return x+s.alpha*(h-x)     # gated residual: alpha=1 normal, alpha=0 identity
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
            if mode=="identity": [nn.init.zeros_(p) for p in (b.mix.o.weight,b.mix.o.bias,b.mlp[-1].weight,b.mlp[-1].bias)]
            elif mode=="stack":  b.load_state_dict(s.blocks[-1].state_dict())                 # copy, alpha=1 -> output JUMPS
            elif mode=="hybrid": b.load_state_dict(s.blocks[-1].state_dict()); b.alpha.data.zero_()  # copy + alpha=0 -> identity at t0, warm underneath
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
def run(mode,TOT):                                                          # early-heavy schedule (best, #65)
    torch.manual_seed(0); m=LM(2).to(DEV); cum=0; C=[]; jumps=[]
    cum,c=train(m,TOT//2,cum); C+=c
    pre=vloss(m); m.grow(mode); jumps.append(vloss(m)-pre); cum,c=train(m,TOT//4,cum); C+=c
    pre=vloss(m); m.grow(mode); jumps.append(vloss(m)-pre); cum,c=train(m,TOT//4,cum); C+=c
    return C,jumps
def c2t(cv,L):
    for cc,l in cv:
        if l<=L: return cc
    return None
if __name__=="__main__":
    TOT=4500
    torch.manual_seed(0); mS=LM(6).to(DEV); _,cS=train(mS,TOT,0)
    R={m:run(m,TOT) for m in ["identity","stack","hybrid"]}
    tgt=max([cS[-1][1]]+[R[m][0][-1][1] for m in R])+0.05
    cs=c2t(cS,tgt)
    print(f"GROWTH-OPERATOR CAPSTONE (early schedule, target {tgt:.3f}):  scratch tgt@{cs/1e12:.0f}e12\n",flush=True)
    for m in ["identity","stack","hybrid"]:
        cv,jumps=R[m]; ct=c2t(cv,tgt)
        sav=f"{100*(1-ct/cs):+.0f}%" if (ct and cs) else "n/a"
        print(f"  {m:9s} tgt@{ct/1e12 if ct else float('nan'):>4.0f}e12 ({sav} vs scratch)  final={cv[-1][1]:.3f}  grow-jumps(|Δloss|)={[round(j,3) for j in jumps]}",flush=True)
    print("\nHYBRID wins if: ~stack efficiency AND grow-jumps ~0 (function-preserving like identity).",flush=True)
    print("DONE",flush=True)
