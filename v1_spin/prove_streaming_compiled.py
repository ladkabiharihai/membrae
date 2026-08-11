"""COMPILED streaming run: same lean core (feat=8, chunk=128) but torch.compile()'d -> fuses the ~120 eager
kernel launches per step into a few, killing the launch overhead that made the eager run slow. Logs step TIMING
so we can directly compare to the eager run. Clean LIVE log: prints straight to stdout (-u, no grep/tee), warnings
suppressed. Runs alongside the eager run (different filename, no pkill needed)."""
import warnings, os; warnings.filterwarnings("ignore"); os.environ["PYTHONWARNINGS"]="ignore"
import sys, time, random, torch, torch.nn as nn, torch.nn.functional as F
sys.path.insert(0, "/opt/code/membrae")
from gated_core import GatedRecall
DEV="cuda"; torch.manual_seed(0); random.seed(0)
K=64; V=64; PAD=0; Kt=lambda i:1+i; Vt=lambda i:1+K+i; QT=1+K+V; FILL=2+K+V; VOCAB=3+K+V

class Blk(nn.Module):
    def __init__(s,d): super().__init__(); s.n1=nn.LayerNorm(d); s.mix=GatedRecall(d,feat=8,chunk=128); s.n2=nn.LayerNorm(d); s.mlp=nn.Sequential(nn.Linear(d,4*d),nn.GELU(),nn.Linear(4*d,d))
    def forward(s,x): m,_=s.mix(s.n1(x)); x=x+m; return x+s.mlp(s.n2(x))
class LM(nn.Module):
    def __init__(s,d=128,L=2): super().__init__(); s.emb=nn.Embedding(VOCAB,d); s.blocks=nn.ModuleList([Blk(d) for _ in range(L)]); s.head=nn.Linear(d,VOCAB)
    def forward(s,x):
        h=s.emb(x)
        for b in s.blocks: h=b(h)
        return s.head(h)

def train_seq(fill_len):
    a=random.randrange(K); b=random.randrange(V)
    return [Kt(a),Vt(b)]+[FILL]*fill_len+[QT], Vt(b)
def make_batch(B,fill):
    rows=[train_seq(random.randint(fill//2,fill)) for _ in range(B)]; T=max(len(s) for s,_ in rows)
    x=torch.full((B,T),PAD,dtype=torch.long); y=torch.full((B,T),-100,dtype=torch.long)
    for bi,(s,tgt) in enumerate(rows): x[bi,:len(s)]=torch.tensor(s); y[bi,len(s)-1]=tgt
    return x.to(DEV),y.to(DEV)

if __name__=="__main__":
    STEPS=2500; FILL=1500; BS=16
    m=LM().to(DEV)
    print(f"[compiled] building model, then torch.compile (first step compiles, ~1-2 min)...",flush=True)
    mc=torch.compile(m)
    opt=torch.optim.AdamW(m.parameters(),lr=2e-3); m.train()
    t_warm=time.time()
    # warmup / compile on a couple steps (not timed)
    for _ in range(2):
        x,y=make_batch(BS,FILL); F.cross_entropy(mc(x).reshape(-1,VOCAB),y.reshape(-1),ignore_index=-100).backward(); opt.zero_grad()
    torch.cuda.synchronize(); print(f"[compiled] compile+warmup done in {time.time()-t_warm:.0f}s; now timing...",flush=True)
    t0=time.time(); last=t0
    for it in range(1,STEPS+1):
        x,y=make_batch(BS,FILL); loss=F.cross_entropy(mc(x).reshape(-1,VOCAB),y.reshape(-1),ignore_index=-100)
        opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(m.parameters(),1.); opt.step()
        if it%100==0:
            torch.cuda.synchronize(); now=time.time(); sps=100/(now-last); last=now
            print(f"  step {it}/{STEPS}  loss={loss.item():.3f}  {sps:.1f} step/s  ({1000/sps:.0f} ms/step)",flush=True)
    torch.cuda.synchronize(); print(f"[compiled] {STEPS} steps in {time.time()-t0:.0f}s = {STEPS/(time.time()-t0):.1f} step/s",flush=True)
    # quick horizon eval -- single forward (core chunks internally; no positional emb -> any length ok)
    @torch.no_grad()
    def ev(fl,n=30):
        m.eval(); hit=0
        for _ in range(n):
            a=random.randrange(K); b=random.randrange(V); seq=[Kt(a),Vt(b)]+[FILL]*fl+[QT]
            lg=m(torch.tensor([seq],device=DEV)); hit+= int(lg[0,-1].argmax())==Vt(b)
        m.train(); return hit/n
    print("\nhorizon (recall of start-fact after fill tokens):",flush=True)
    for fl in [500,2000,8000,32000]:
        print(f"  stream {fl+3}: recall {ev(fl)*100:.0f}%",flush=True)
    print("[done]",flush=True)
