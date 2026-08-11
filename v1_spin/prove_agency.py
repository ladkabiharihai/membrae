"""FACULTY #3: AGENCY / learning-from-consequence, IN THE MODEL. In-context RL: a contextual-bandit stream --
each step the model sees a CONTEXT token, emits an ACTION, then sees the REWARD token for that action. Its
fast-weight STATE accumulates which action pays off per context, so action quality RISES within a single episode
-- learning from consequence in the FORWARD PASS, zero weight updates at test. A frozen transformer just predicts;
this ACTS and improves. (Meta-trained: slow weights learn HOW the fast-weight state should adapt to reward.)"""
import sys, random, torch, torch.nn as nn, torch.nn.functional as F
sys.path.insert(0, "/opt/code/membrae")
from gated_core import GatedRecall
DEV="cuda"; torch.manual_seed(0); random.seed(0)

NC=4; NA=4                                   # contexts, actions
PAD=0; CTX=lambda c:1+c; ACT=lambda a:1+NC+a; RGOOD=1+NC+NA; RBAD=2+NC+NA
V=3+NC+NA

class Blk(nn.Module):
    def __init__(s,d): super().__init__(); s.n1=nn.LayerNorm(d); s.mix=GatedRecall(d,feat=16); s.n2=nn.LayerNorm(d); s.mlp=nn.Sequential(nn.Linear(d,4*d),nn.GELU(),nn.Linear(4*d,d))
    def forward(s,x): m,_=s.mix(s.n1(x)); x=x+m; return x+s.mlp(s.n2(x))
class Policy(nn.Module):
    def __init__(s,d=192,L=3): super().__init__(); s.emb=nn.Embedding(V,d); s.pos=nn.Embedding(4096,d); s.blocks=nn.ModuleList([Blk(d) for _ in range(L)]); s.head=nn.Linear(d,V)
    def forward(s,x): h=s.emb(x)+s.pos(torch.arange(x.shape[1],device=x.device)); [None for b in s.blocks if (h:=b(h)) is None] ; return s.head(h)

def _fwd(s,x):
    h=s.emb(x)+s.pos(torch.arange(x.shape[1],device=x.device))
    for b in s.blocks: h=b(h)
    return s.head(h)
Policy.forward=_fwd

def episode(steps=64, explore=0.5):
    """Meta-RL in-context bandit: random per-episode reward table. Actions EXPLORE (random) until a context's
    good action is discovered (RGOOD), then EXPLOIT. LOSS only where the good action is ALREADY known from PRIOR
    reward history -> the model must READ its state (past rewards) to choose -> learns from consequence."""
    best={c:random.randrange(NA) for c in range(NC)}; seen={}
    seq=[]; act_pos=[]; act_tgt=[]
    for _ in range(steps):
        c=random.randrange(NC); known=seen.get(c)                    # good action known from PRIOR steps (in state)
        a=known if (known is not None and random.random()>explore) else random.randrange(NA)
        if known is not None:                                        # supervise ONLY when discoverable from history
            act_pos.append(len(seq)+1); act_tgt.append(ACT(known))
        r=RGOOD if a==best[c] else RBAD
        if a==best[c]: seen[c]=a                                     # discovered -> remember for future steps
        seq+=[CTX(c),ACT(a),r]
    return seq,act_pos,act_tgt,best

def make_batch(B,steps=64):
    rows=[episode(steps) for _ in range(B)]; T=max(len(r[0]) for r in rows)
    x=torch.zeros(B,T,dtype=torch.long); y=torch.full((B,T),-100,dtype=torch.long)
    for b,(seq,ap,at,_) in enumerate(rows):
        x[b,:len(seq)]=torch.tensor(seq)
        for p,t in zip(ap,at): y[b,p]=t
    return x.to(DEV),y.to(DEV)

def train(m,steps=12000,lr=2e-3):
    opt=torch.optim.AdamW(m.parameters(),lr=lr); m.train()
    for _ in range(steps):
        x,y=make_batch(64); loss=F.cross_entropy(m(x).reshape(-1,V),y.reshape(-1),ignore_index=-100)
        opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(m.parameters(),1.); opt.step()

@torch.no_grad()
def evaluate(m,steps=64,n=200):
    """TEST: model chooses actions itself (greedy); reward fed back from a fresh random table. Measure optimal-
    action rate in the FIRST third vs LAST third of the episode -> does it improve from consequence in-state?"""
    m.eval(); early=late=e_n=l_n=0
    for _ in range(n):
        best={c:random.randrange(NA) for c in range(NC)}; seq=[]; hits=[]
        for t in range(steps):
            c=random.randrange(NC); seq.append(CTX(c))
            a=int(m(torch.tensor([seq],device=DEV))[0,-1].argmax())-(1+NC)   # model's chosen action
            a=a if 0<=a<NA else 0
            r=RGOOD if a==best[c] else RBAD
            seq+=[ACT(a),r]; hits.append(a==best[c])
        k=steps//3
        early+=sum(hits[:k]); e_n+=k; late+=sum(hits[-k:]); l_n+=k
    return early/e_n, late/l_n

if __name__=="__main__":
    m=Policy().to(DEV)
    print("meta-training policy (learn to map context+reward-history-in-state -> good action)...",flush=True)
    train(m)
    e,l=evaluate(m)
    chance=1/NA
    print(f"\nFACULTY #3 -- in-context learning from consequence (chance={chance*100:.0f}%):")
    print(f"  optimal-action rate: FIRST third {e*100:.0f}%  ->  LAST third {l*100:.0f}%",flush=True)
    print(f"\nWIN = LAST >> FIRST (and >> chance): the model improved its behavior from reward WITHIN one episode,")
    print("       in its fast-weight state, ZERO weight updates -> agency is intrinsic to the forward pass.")
