"""SOLVE-SCALING step 1: does the VChunkRecall brain scale with WIDTH (d) -- the axis depth/feat did NOT -- WITH
faculties intact? Depth-growth was dead (#48/#50/#56); width is the untested, parameter-efficient, easier-to-train
axis. Train the SAME core (faculties architectural) at increasing d on REAL language (window2), fixed token budget,
and measure: (A) language loss/ppl vs d (does capability scale with width?), (B) MQAR-16 recall at each d (does the
recall FACULTY stay intact as we scale?). Clean scaling curve + intact faculty => width is the scaling recipe.
"""
import warnings,os; warnings.filterwarnings("ignore"); os.environ["PYTHONWARNINGS"]="ignore"; os.environ.setdefault("RECALL_FRAC","0")
import sys,time,numpy as np,torch,torch.nn as nn,torch.nn.functional as F
sys.path.insert(0,"/opt/code/membrae")
import scale_train as S
from fastcore_v import VChunkRecall, recall
DEV="cuda"; torch.manual_seed(0); np.random.seed(0)
DATA=S.DATA; VOCAB=S.VOCAB; CTX=512

class Blk(nn.Module):
    def __init__(s,d,H): super().__init__(); s.n1=nn.LayerNorm(d); s.mix=VChunkRecall(d,heads=H,feat=8,chunk=128,stable=True); s.n2=nn.LayerNorm(d); s.mlp=nn.Sequential(nn.Linear(d,4*d),nn.GELU(),nn.Linear(4*d,d))
    def forward(s,x): x=x+s.mix(s.n1(x)); return x+s.mlp(s.n2(x))
class LM(nn.Module):
    def __init__(s,d,L):
        super().__init__(); H=max(2,d//64); s.emb=nn.Embedding(VOCAB,d); s.pos=nn.Embedding(8192,d)
        s.blocks=nn.ModuleList([Blk(d,H) for _ in range(L)]); s.lnf=nn.LayerNorm(d); s.head=nn.Linear(d,VOCAB,bias=False); s.head.weight=s.emb.weight
        nn.init.normal_(s.emb.weight,0,0.02); nn.init.normal_(s.pos.weight,0,0.02)
    def forward(s,x):
        h=s.emb(x)+s.pos(torch.arange(x.shape[1],device=x.device)[None])
        for b in s.blocks: h=b(h)
        return s.head(s.lnf(h))

def batch(B):
    i=np.random.randint(0,len(DATA)-CTX-1,size=B)
    x=np.stack([DATA[j:j+CTX] for j in i]).astype(np.int64); y=np.stack([DATA[j+1:j+CTX+1] for j in i]).astype(np.int64)
    return torch.from_numpy(x).to(DEV),torch.from_numpy(y).to(DEV)

def train_lang(d,L=6,steps=3000,B=8,lr=6e-4):
    m=LM(d,L).to(DEV); opt=torch.optim.AdamW(m.parameters(),lr=lr,betas=(0.9,0.95),weight_decay=0.1); ema=None
    for it in range(1,steps+1):
        for gp in opt.param_groups: gp["lr"]=lr*min(1.0,it/200)
        x,y=batch(B); loss=F.cross_entropy(m(x).reshape(-1,VOCAB),y.reshape(-1))
        if not torch.isfinite(loss): opt.zero_grad(set_to_none=True); continue
        opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(m.parameters(),1.0); opt.step()
        ema=loss.item() if ema is None else 0.98*ema+0.02*loss.item()
    P=sum(p.numel() for p in m.parameters())/1e6
    return P,ema

if __name__=="__main__":
    print(f"WIDTH-SCALING (fixed L=6, {3000} steps, ctx{CTX}, window2 real language). Does capability scale with d?\n",flush=True)
    print(f"  {'d':>4} {'params':>8}  {'lang-loss':>9} {'ppl':>7}   {'MQAR-16 (faculty)':>18}",flush=True)
    for d in [192,320,512,768]:
        P,ll=train_lang(d)
        mq=recall(lambda dd=d: VChunkRecall(dd,heads=max(2,dd//64),feat=8,chunk=64), n=16, d=d, L=2, steps=2500)  # faculty intact at this d?
        print(f"  {d:>4} {P:>6.1f}M  {ll:>9.3f} {np.exp(min(20,ll)):>7.1f}   {mq*100:>16.0f}%",flush=True)
    print("\nVERDICT: if lang-ppl FALLS monotonically with d AND MQAR stays ~100%, WIDTH is the scaling axis w/ faculties intact.",flush=True)
    print("DONE",flush=True)
