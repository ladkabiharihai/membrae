"""THE FRONTIER (user: 'train as large a model as possible WITHOUT increasing compute; compute stays same at any
param'). Dense growth can't -- every param runs every token. The ONLY way is SPARSE activation: many params, few
active per token = Mixture-of-Experts (brain-like: 86B neurons, a sparse subset fires per thought). Growth adds
EXPERTS (top-1 routed); TOTAL params scale freely, ACTIVE params/token stay CONSTANT. Prove: val loss improves as
experts (total params) grow while active-compute is FLAT -- 'bigger at constant compute'. + function-preserving expert
growth (add expert routed-off at t=0). On the VChunkRecall brain core (shared mix = faculties intact).
"""
import warnings,os; warnings.filterwarnings("ignore"); os.environ["PYTHONWARNINGS"]="ignore"; os.environ.setdefault("RECALL_FRAC","0")
import sys,numpy as np,torch,torch.nn as nn,torch.nn.functional as F
sys.path.insert(0,"/opt/code/membrae")
import scale_train as S
from fastcore_v import VChunkRecall
DEV="cuda"; DATA=S.DATA; V=S.VOCAB; CTX=384; d=256; H=4
VAL_I=np.random.RandomState(0).randint(0,len(DATA)-CTX-1,size=48)
def vx():
    x=np.stack([DATA[j:j+CTX] for j in VAL_I]).astype(np.int64); y=np.stack([DATA[j+1:j+CTX+1] for j in VAL_I]).astype(np.int64)
    return torch.from_numpy(x).to(DEV),torch.from_numpy(y).to(DEV)
class MoEMLP(nn.Module):                                             # top-1 routed experts: ACTIVE compute = 1 expert, always
    def __init__(s,E=1):
        super().__init__(); s.E=E; s.router=nn.Linear(d,E)
        s.experts=nn.ModuleList([nn.Sequential(nn.Linear(d,4*d),nn.GELU(),nn.Linear(4*d,d)) for _ in range(E)])
    def forward(s,x):
        B,T,_=x.shape; xf=x.reshape(-1,d); lg=s.router(xf); g=lg.softmax(-1)
        top=lg.argmax(-1); out=torch.zeros_like(xf)
        for e in range(s.E):                                        # each token runs through EXACTLY ONE expert
            m=top==e
            if m.any(): out[m]=s.experts[e](xf[m])*g[m,e:e+1]
        return out.reshape(B,T,d)
    def grow(s):                                                    # function-preserving: add expert, router logit for it = very negative -> unrouted at t=0
        ne=nn.Sequential(nn.Linear(d,4*d),nn.GELU(),nn.Linear(4*d,d)).to(DEV); s.experts.append(ne)
        nr=nn.Linear(d,s.E+1).to(DEV)
        with torch.no_grad():
            nr.weight[:s.E]=s.router.weight; nr.bias[:s.E]=s.router.bias
            nr.weight[s.E].zero_(); nr.bias[s.E]=-30.0              # new expert unreachable until router learns
        s.router=nr; s.E+=1
class Blk(nn.Module):
    def __init__(s,E): super().__init__(); s.n1=nn.LayerNorm(d); s.mix=VChunkRecall(d,heads=H,feat=8,chunk=128,stable=True); s.n2=nn.LayerNorm(d); s.moe=MoEMLP(E)
    def forward(s,x): x=x+s.mix(s.n1(x)); return x+s.moe(s.n2(x))
class LM(nn.Module):
    def __init__(s,E,L=3):
        super().__init__(); s.emb=nn.Embedding(V,d); s.pos=nn.Embedding(8192,d)
        s.blocks=nn.ModuleList([Blk(E) for _ in range(L)]); s.lnf=nn.LayerNorm(d); s.head=nn.Linear(d,V,bias=False); s.head.weight=s.emb.weight
        nn.init.normal_(s.emb.weight,0,0.02); nn.init.normal_(s.pos.weight,0,0.02)
    def forward(s,x):
        h=s.emb(x)+s.pos(torch.arange(x.shape[1],device=x.device)[None])
        for b in s.blocks: h=b(h)
        return s.head(s.lnf(h))
    def grow_experts(s): [b.moe.grow() for b in s.blocks]
def total_params(m): return sum(p.numel() for p in m.parameters())
def active_params(m):                                               # params actually USED per token (shared + 1 expert/block)
    E=m.blocks[0].moe.E; tot=total_params(m); one_exp=sum(p.numel() for p in m.blocks[0].moe.experts[0].parameters())
    return tot - sum((b.moe.E-1)*one_exp for b in m.blocks)         # subtract the inactive experts
@torch.no_grad()
def vloss(m): m.eval(); x,y=vx(); l=F.cross_entropy(m(x).reshape(-1,V),y.reshape(-1)).item(); m.train(); return l
def bs(B):
    i=np.random.randint(0,len(DATA)-CTX-1,size=B); x=np.stack([DATA[j:j+CTX] for j in i]).astype(np.int64); y=np.stack([DATA[j+1:j+CTX+1] for j in i]).astype(np.int64)
    return torch.from_numpy(x).to(DEV),torch.from_numpy(y).to(DEV)
def train(m,steps,B=16,lr=1e-3):
    opt=torch.optim.AdamW(m.parameters(),lr=lr,weight_decay=0.05)
    for it in range(1,steps+1):
        for gp in opt.param_groups: gp["lr"]=lr*min(1.0,it/100)
        x,y=bs(B); loss=F.cross_entropy(m(x).reshape(-1,V),y.reshape(-1))
        if torch.isfinite(loss): opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(m.parameters(),1.0); opt.step()
    return vloss(m)
if __name__=="__main__":
    ST=2500
    print("CONSTANT-COMPUTE via SPARSE (top-1 expert) growth. val loss vs TOTAL params @ FLAT active-compute:\n",flush=True)
    print(f"  {'experts':>7} | {'total params':>13} | {'ACTIVE params/token':>20} | {'val loss':>9}",flush=True)
    print("  "+"-"*60,flush=True)
    for E in [1,2,4,8]:
        torch.manual_seed(0); m=LM(E).to(DEV); vl=train(m,ST)
        print(f"  {E:>7} | {total_params(m)/1e6:>11.1f}M | {active_params(m)/1e6:>18.1f}M | {vl:>9.3f}",flush=True)
    print("\n  GROW experts 1->8 (function-preserving, same ACTIVE compute throughout):",flush=True)
    torch.manual_seed(0); m=LM(1).to(DEV); vl=train(m,ST//2)
    print(f"    E=1 trained: total={total_params(m)/1e6:.1f}M active={active_params(m)/1e6:.1f}M vloss={vl:.3f}",flush=True)
    for _ in range(3):
        pre=vloss(m); m.grow_experts(); post=vloss(m)              # function-preserving check at each grow
        vl=train(m,ST//2)
        print(f"    grow->E={m.blocks[0].moe.E}: preserve Δ={post-pre:+.3f}  total={total_params(m)/1e6:.1f}M active={active_params(m)/1e6:.1f}M  vloss={vl:.3f}",flush=True)
    print("\n  PROOF: if val loss FALLS as total params grow while ACTIVE params/token stay FLAT -> larger model at",flush=True)
    print("  CONSTANT compute. Sparse expert growth = 'as large as possible without more compute'. Brain-like sparsity.",flush=True)
    print("DONE",flush=True)
