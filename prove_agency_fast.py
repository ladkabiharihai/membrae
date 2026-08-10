"""FACULTY #3 on the FAST core (VChunkRecall). Same in-context bandit, but trains 5x longer (now cheap) to get
STRONG exploitation. Logs live (direct write)."""
import warnings,os; warnings.filterwarnings("ignore"); os.environ["PYTHONWARNINGS"]="ignore"
import sys,time,random,torch,torch.nn as nn,torch.nn.functional as F
sys.path.insert(0,"/opt/code/membrae")
from fastcore_v import VChunkRecall
DEV="cuda"; torch.manual_seed(0); random.seed(0)
NC=4; NA=4; PAD=0; CTX=lambda c:1+c; ACT=lambda a:1+NC+a; RGOOD=1+NC+NA; RBAD=2+NC+NA; V=3+NC+NA
class Blk(nn.Module):
    def __init__(s,d): super().__init__(); s.n1=nn.LayerNorm(d); s.mix=VChunkRecall(d,feat=8,chunk=64); s.n2=nn.LayerNorm(d); s.mlp=nn.Sequential(nn.Linear(d,4*d),nn.GELU(),nn.Linear(4*d,d))
    def forward(s,x): x=x+s.mix(s.n1(x)); return x+s.mlp(s.n2(x))
class Policy(nn.Module):
    def __init__(s,d=192,L=3): super().__init__(); s.emb=nn.Embedding(V,d); s.pos=nn.Embedding(8192,d); s.blocks=nn.ModuleList([Blk(d) for _ in range(L)]); s.head=nn.Linear(d,V)
    def forward(s,x):
        h=s.emb(x)+s.pos(torch.arange(x.shape[1],device=x.device))
        for b in s.blocks: h=b(h)
        return s.head(h)
def episode(steps=64,explore=0.5):
    best={c:random.randrange(NA) for c in range(NC)}; seen={}; seq=[]; ap=[]; at=[]
    for _ in range(steps):
        c=random.randrange(NC); known=seen.get(c)
        a=known if (known is not None and random.random()>explore) else random.randrange(NA)
        if known is not None: ap.append(len(seq)+1); at.append(ACT(known))
        r=RGOOD if a==best[c] else RBAD
        if a==best[c]: seen[c]=a
        seq+=[CTX(c),ACT(a),r]
    return seq,ap,at
def make_batch(B,steps=64):
    rows=[episode(steps) for _ in range(B)]; T=max(len(r[0]) for r in rows)
    x=torch.zeros(B,T,dtype=torch.long); y=torch.full((B,T),-100,dtype=torch.long)
    for b,(s,ap,at) in enumerate(rows):
        x[b,:len(s)]=torch.tensor(s)
        for p,t in zip(ap,at): y[b,p]=t
    return x.to(DEV),y.to(DEV)
@torch.no_grad()
def evaluate(m,steps=64,n=200):
    """Two metrics: (1) crude first/last-third optimal-rate (conflates exploration cost); (2) EXPLOITATION-AFTER-
    DISCOVERY -- once a context's good action has been rewarded, does the model pick it on later visits? -- which
    ISOLATES learning-from-consequence from exploration."""
    m.eval(); early=late=en=ln=0; post_c=post_t=0
    for _ in range(n):
        best={c:random.randrange(NA) for c in range(NC)}; seen=set(); seq=[]; hits=[]
        for t in range(steps):
            c=random.randrange(NC); seq.append(CTX(c))
            a=int(m(torch.tensor([seq],device=DEV))[0,-1].argmax())-(1+NC); a=a if 0<=a<NA else 0
            correct=(a==best[c])
            if c in seen: post_t+=1; post_c+=correct          # visit AFTER good action was discovered
            r=RGOOD if correct else RBAD
            if correct: seen.add(c)
            seq+=[ACT(a),r]; hits.append(correct)
        k=steps//3; early+=sum(hits[:k]); en+=k; late+=sum(hits[-k:]); ln+=k
    return early/en, late/ln, (post_c/max(1,post_t))
if __name__=="__main__":
    m=Policy().to(DEV); opt=torch.optim.AdamW(m.parameters(),lr=2e-3); m.train()
    STEPS=15000; t0=time.time()
    print(f"[agency-fast] meta-training {STEPS} steps on fast core...",flush=True)
    for it in range(1,STEPS+1):
        x,y=make_batch(64); loss=F.cross_entropy(m(x).reshape(-1,V),y.reshape(-1),ignore_index=-100)
        opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(m.parameters(),1.); opt.step()
        if it%1000==0: print(f"  step {it}/{STEPS} loss={loss.item():.3f} ({it/(time.time()-t0):.0f} step/s)",flush=True)
    e,l,expl=evaluate(m)
    print(f"\nFACULTY #3 (fast core, 15k steps) -- in-context learning from consequence (chance=25%):",flush=True)
    print(f"  optimal-action rate: FIRST {e*100:.0f}% -> LAST {l*100:.0f}%  (conflates exploration)",flush=True)
    print(f"  EXPLOITATION-after-discovery: {expl*100:.0f}%  (isolates learning-from-consequence; chance 25%)",flush=True)
    print("[done]",flush=True)
