"""THE FIX for 13%: multi-step reasoning as ITERATED single-hop recall (chain-of-thought) with INTRINSIC halting.
Task: a successor-chain ending in a self-loop TERMINAL. Given a start, reach the terminal.
  ONESHOT : predict the terminal in ONE parallel forward (what scored ~13% -- the core can't compose).
  COT     : GENERATE the hop chain autoregressively (each step = single-hop recall, which the core does ~100%),
            and HALT (emit STOP) when it recognizes the terminal -- the model's OWN decision of how many steps.
This is adaptive compute in the GENERATION axis: #steps = distance, decided by the model itself. Expect COT >> ONESHOT,
and #generated-steps to track the true distance = scaffolding-free self-scaling of computation.
"""
import warnings,os; warnings.filterwarnings("ignore"); os.environ["PYTHONWARNINGS"]="ignore"
import sys,random,collections,torch,torch.nn as nn,torch.nn.functional as F
sys.path.insert(0,"/opt/code/membrae")
from fastcore_v import VChunkRecall
DEV="cuda"; torch.manual_seed(0); random.seed(0)
N=12; PAD=0
SYM=lambda i:1+i; ARROW=1+N; Q=2+N; STOP=3+N; VOCAB=4+N
def make_map():
    order=list(range(N)); random.shuffle(order)
    succ={order[i]:order[i+1] for i in range(N-1)}; succ[order[-1]]=order[-1]  # terminal self-loop
    term=order[-1]; return succ,term,order
def context(succ):
    order=list(range(N)); random.shuffle(order); seq=[]
    for a in order: seq+=[SYM(a),ARROW,SYM(succ[a])]
    return seq
def chase(succ,start,term):
    path=[]; cur=start
    while cur!=term and len(path)<N: cur=succ[cur]; path.append(cur)
    return path  # [s1,...,term]
def oneshot_batch(B):   # predict terminal in one shot at the query position
    xs=[]; ys=[]; T=0; rows=[]
    for _ in range(B):
        succ,term,_=make_map(); start=random.randrange(N)
        seq=context(succ)+[Q,SYM(start)]; rows.append((seq,SYM(term))); T=max(T,len(seq))
    x=torch.full((B,T),PAD,dtype=torch.long); y=torch.full((B,T),-100,dtype=torch.long)
    for b,(s,t) in enumerate(rows): x[b,:len(s)]=torch.tensor(s); y[b,len(s)-1]=t
    return x.to(DEV),y.to(DEV)
def cot_batch(B):       # teacher-force the generated hop chain + STOP
    rows=[]; T=0
    for _ in range(B):
        succ,term,_=make_map(); start=random.randrange(N); path=chase(succ,start,term)
        gen=[SYM(s) for s in path]+[STOP]
        seq=context(succ)+[Q,SYM(start)]+gen; rows.append((seq,len(context(succ))+2)); T=max(T,len(seq))
    x=torch.full((B,T),PAD,dtype=torch.long); y=torch.full((B,T),-100,dtype=torch.long)
    for b,(s,qend) in enumerate(rows):
        x[b,:len(s)]=torch.tensor(s)
        for p in range(qend-1,len(s)-1): y[b,p]=s[p+1]   # predict each next hop token (incl STOP), over the gen region
    return x.to(DEV),y.to(DEV)
class FF(nn.Module):
    def __init__(s,d): super().__init__(); s.n1=nn.LayerNorm(d); s.mix=VChunkRecall(d,heads=4,feat=8,chunk=64); s.n2=nn.LayerNorm(d); s.mlp=nn.Sequential(nn.Linear(d,4*d),nn.GELU(),nn.Linear(4*d,d))
    def forward(s,x): x=x+s.mix(s.n1(x)); return x+s.mlp(s.n2(x))
class LM(nn.Module):
    def __init__(s,d=160,L=3): super().__init__(); s.emb=nn.Embedding(VOCAB,d); s.pos=nn.Embedding(512,d); s.blocks=nn.ModuleList([FF(d) for _ in range(L)]); s.head=nn.Linear(d,VOCAB)
    def forward(s,x):
        h=s.emb(x)+s.pos(torch.arange(x.shape[1],device=x.device)[None])
        for b in s.blocks: h=b(h)
        return s.head(h)
def train(m,bfn,steps=6000,lr=2e-3):
    opt=torch.optim.AdamW(m.parameters(),lr=lr); m.train()
    for _ in range(steps):
        x,y=bfn(64); F.cross_entropy(m(x).reshape(-1,VOCAB),y.reshape(-1),ignore_index=-100).backward(); opt.step(); opt.zero_grad()
@torch.no_grad()
def eval_oneshot(m,n=60):
    m.eval(); c=t=0
    for _ in range(n):
        x,y=oneshot_batch(64); pred=m(x).argmax(-1)
        for b in range(64):
            p=(y[b]!=-100).nonzero()[0].item(); c+=int(pred[b,p]==y[b,p]); t+=1
    m.train(); return c/t
@torch.no_grad()
def eval_cot(m,n=400):
    """free-generate hops until STOP; correct if the symbol before STOP == terminal. Track steps vs true distance."""
    m.eval(); c=0; steps_by_dist=collections.defaultdict(list)
    for _ in range(n):
        succ,term,_=make_map(); start=random.randrange(N); true=chase(succ,term=term,start=start); dist=len(true)
        seq=context(succ)+[Q,SYM(start)]; gen=[]
        for _ in range(N+2):
            nx=int(m(torch.tensor([seq+gen],device=DEV))[0,-1].argmax())
            if nx==STOP: break
            gen.append(nx)
        reached = len(gen)>0 and gen[-1]==SYM(term)
        c+=int(reached); steps_by_dist[dist].append(len(gen))
    m.train(); return c/n, steps_by_dist
if __name__=="__main__":
    print(f"CHAIN-OF-THOUGHT vs ONESHOT on successor-chase (N={N}) -- multi-step reasoning\n",flush=True)
    m1=LM().to(DEV); train(m1,oneshot_batch); a1=eval_oneshot(m1)
    print(f"  ONESHOT (1 parallel forward): terminal-acc = {a1*100:.0f}%   <- the ~13% failure mode",flush=True)
    m2=LM().to(DEV); train(m2,cot_batch); a2,sbd=eval_cot(m2)
    print(f"  COT (iterated single-hop + intrinsic STOP): terminal-acc = {a2*100:.0f}%",flush=True)
    print("  adaptive compute -- #generated steps vs TRUE distance (should track):",flush=True)
    for d in sorted(sbd):
        import statistics as st
        print(f"    dist={d}: mean steps={st.mean(sbd[d]):.1f} (n={len(sbd[d])})",flush=True)
    print("\nWIN = COT >> ONESHOT and steps track distance = the model self-scales its OWN computation to difficulty.",flush=True)
