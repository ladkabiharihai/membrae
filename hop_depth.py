"""DIAGNOSTIC: is variable-hop composition a DEPTH problem (adaptive compute can win) or a CAPABILITY problem
(the core can't chain hops at all -> need a scratchpad)? Train FIXED-depth models L=2/4/6/8 on variable-hop
lookup; report per-hop accuracy. If deeper -> solves it, adaptive-depth is the lever. If even L=8 fails on h>=2,
the core can't compose in-parallel -> the real fix is a SCRATCHPAD (write intermediate hop, re-read). Longer
training (6000 steps), small N so single-hop is easy."""
import warnings,os; warnings.filterwarnings("ignore"); os.environ["PYTHONWARNINGS"]="ignore"
import sys,random,collections,torch,torch.nn as nn,torch.nn.functional as F
sys.path.insert(0,"/opt/code/membrae")
from fastcore_v import VChunkRecall
DEV="cuda"; torch.manual_seed(0); random.seed(0)
N=12; HMAX=4; PAD=0
SYM=lambda i:1+i; ARROW=1+N; Q=2+N; HOP=lambda h:2+N+h; VOCAB=2+N+HMAX+1
def sample():
    perm=list(range(N)); random.shuffle(perm); seq=[]; order=list(range(N)); random.shuffle(order)
    for i in order: seq+=[SYM(i),ARROW,SYM(perm[i])]
    h=random.randint(1,HMAX); start=random.randrange(N); cur=start
    for _ in range(h): cur=perm[cur]
    seq+=[Q,SYM(start),HOP(h)]; return seq,SYM(cur),h
def batch(B):
    rows=[sample() for _ in range(B)]; T=max(len(s) for s,_,_ in rows)
    x=torch.full((B,T),PAD,dtype=torch.long); y=torch.full((B,T),-100,dtype=torch.long); hs=[]
    for b,(s,t,h) in enumerate(rows): x[b,:len(s)]=torch.tensor(s); y[b,len(s)-1]=t; hs.append(h)
    return x.to(DEV),y.to(DEV),torch.tensor(hs)
class FF(nn.Module):
    def __init__(s,d): super().__init__(); s.n1=nn.LayerNorm(d); s.mix=VChunkRecall(d,heads=4,feat=8,chunk=64); s.n2=nn.LayerNorm(d); s.mlp=nn.Sequential(nn.Linear(d,4*d),nn.GELU(),nn.Linear(4*d,d))
    def forward(s,x): x=x+s.mix(s.n1(x)); return x+s.mlp(s.n2(x))
class LM(nn.Module):
    def __init__(s,d=160,L=2): super().__init__(); s.emb=nn.Embedding(VOCAB,d); s.pos=nn.Embedding(256,d); s.blocks=nn.ModuleList([FF(d) for _ in range(L)]); s.head=nn.Linear(d,VOCAB)
    def forward(s,x):
        h=s.emb(x)+s.pos(torch.arange(x.shape[1],device=x.device)[None])
        for b in s.blocks: h=b(h)
        return s.head(h)
def run(L,steps=6000):
    torch.manual_seed(0); m=LM(160,L).to(DEV); opt=torch.optim.AdamW(m.parameters(),lr=2e-3); m.train()
    for _ in range(steps):
        x,y,_=batch(64); F.cross_entropy(m(x).reshape(-1,VOCAB),y.reshape(-1),ignore_index=-100).backward(); opt.step(); opt.zero_grad()
    m.eval(); cor=collections.defaultdict(int); tot=collections.defaultdict(int)
    with torch.no_grad():
        for _ in range(50):
            x,y,hs=batch(64); pred=m(x).argmax(-1)
            for b in range(x.shape[0]):
                p=(y[b]!=-100).nonzero()[0].item(); h=int(hs[b]); cor[h]+=int(pred[b,p]==y[b,p]); tot[h]+=1
    per={h:cor[h]/tot[h] for h in sorted(tot)}; ov=sum(cor.values())/sum(tot.values())
    return ov,per,sum(p.numel() for p in m.parameters())/1e6
if __name__=="__main__":
    print(f"DEPTH SWEEP on variable-hop (N={N}, HMAX={HMAX}) -- per-hop accuracy\n",flush=True)
    print(f"{'depth':>6} {'params':>7} {'overall':>8}   per-hop (h1..h4)",flush=True)
    for L in [2,4,6,8]:
        ov,per,pm=run(L); ph=" ".join(f"h{h}={per[h]*100:.0f}%" for h in per)
        print(f"{L:>5}L {pm:6.1f}M {ov*100:7.0f}%   {ph}",flush=True)
    print("\nREAD: acc rising with depth AND h1 high but hX dropping = DEPTH-bound (adaptive-compute wins).",flush=True)
    print("      even L=8 failing h>=2 = CAPABILITY gap -> need a SCRATCHPAD (write+reread intermediate hops).",flush=True)
