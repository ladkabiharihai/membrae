"""FACULTY #2 on the FAST core (VChunkRecall): train on LONG streams (now cheap) so the model LEARNS TO IGNORE
filler and keep the fact -> extend the recall horizon past 8K. Constant-memory is already proven (#32); this
tests horizon. Logs live."""
import warnings,os; warnings.filterwarnings("ignore"); os.environ["PYTHONWARNINGS"]="ignore"
import sys,time,random,torch,torch.nn as nn,torch.nn.functional as F
sys.path.insert(0,"/opt/code/membrae")
from fastcore_v import VChunkRecall
DEV="cuda"; torch.manual_seed(0); random.seed(0)
K=64; Vv=64; PAD=0; Kt=lambda i:1+i; Vt=lambda i:1+K+i; QT=1+K+Vv; FILL=2+K+Vv; VOCAB=3+K+Vv
class Blk(nn.Module):
    def __init__(s,d): super().__init__(); s.n1=nn.LayerNorm(d); s.mix=VChunkRecall(d,feat=8,chunk=128,stable=True); s.n2=nn.LayerNorm(d); s.mlp=nn.Sequential(nn.Linear(d,4*d),nn.GELU(),nn.Linear(4*d,d))
    def forward(s,x): x=x+s.mix(s.n1(x)); return x+s.mlp(s.n2(x))
class LM(nn.Module):
    def __init__(s,d=128,L=2): super().__init__(); s.emb=nn.Embedding(VOCAB,d); s.blocks=nn.ModuleList([Blk(d) for _ in range(L)]); s.head=nn.Linear(d,VOCAB)   # no pos-emb -> unbounded
    def forward(s,x):
        x=x.clamp_(0,VOCAB-1)                      # defensive: guarantee in-range emb lookup (rule out index assert)
        h=s.emb(x)
        for b in s.blocks: h=b(h)
        return s.head(h)
def seqof(fl):
    a=random.randrange(K); b=random.randrange(Vv); return [Kt(a),Vt(b)]+[FILL]*fl+[QT], Vt(b)
def make_batch(B,fill):
    rows=[seqof(random.randint(fill//4,fill)) for _ in range(B)]; T=max(len(s) for s,_ in rows)
    x=torch.full((B,T),PAD,dtype=torch.long); y=torch.full((B,T),-100,dtype=torch.long)
    for bi,(s,t) in enumerate(rows): x[bi,:len(s)]=torch.tensor(s); y[bi,len(s)-1]=t
    return x.to(DEV),y.to(DEV)
@torch.no_grad()
def ev(m,fl,n=40):
    m.eval(); h=0
    for _ in range(n):
        a=random.randrange(K); b=random.randrange(Vv); seq=[Kt(a),Vt(b)]+[FILL]*fl+[QT]
        lg=m(torch.tensor([seq],device=DEV)); h+= int(lg[0,-1].argmax())==Vt(b)
    m.train(); return h/n
if __name__=="__main__":
    m=LM().to(DEV); opt=torch.optim.AdamW(m.parameters(),lr=2e-3); m.train()
    STEPS=6000; FILL=6000; BS=4; t0=time.time()
    print(f"[stream-fast] training on streams up to {FILL} tokens, {STEPS} steps (fast core)...",flush=True)
    skipped=0
    for it in range(1,STEPS+1):
        x,y=make_batch(BS,FILL); lg=m(x)
        loss=F.cross_entropy(lg.reshape(-1,VOCAB),y.reshape(-1),ignore_index=-100)
        if not torch.isfinite(loss):                          # rare bad batch -> SKIP (don't poison training)
            opt.zero_grad(set_to_none=True); skipped+=1; continue
        opt.zero_grad(); loss.backward()
        # clip also guards: if grads non-finite, clip_grad_norm returns inf total_norm -> skip the step
        tn=nn.utils.clip_grad_norm_(m.parameters(),1.)
        if not torch.isfinite(tn): opt.zero_grad(set_to_none=True); skipped+=1; continue
        opt.step()
        if it%500==0: print(f"  step {it}/{STEPS} loss={loss.item():.3f} ({it/(time.time()-t0):.1f} step/s, skipped={skipped})",flush=True)
    print(f"\nFACULTY #2 (fast core, trained to {FILL}) -- recall horizon:",flush=True)
    for fl in [2000,8000,32000,100000]:
        print(f"  stream {fl+3}: recall {ev(m,fl)*100:.0f}%",flush=True)
    print("[done]",flush=True)
