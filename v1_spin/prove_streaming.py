"""FACULTY #2: O(1)-memory STREAMING. Plant a fact at the START of a long stream, feed the whole stream in
fixed-size CHUNKS carrying only the O(1) fast-weight state (constant memory, no growing KV cache), then recall
the fact at the END. Proof = recall stays high as stream length grows to lengths a transformer can't hold, while
peak memory stays FLAT. A frozen transformer's context is bounded + O(T^2); this streams unboundedly in O(1)."""
import sys, random, torch, torch.nn as nn, torch.nn.functional as F
sys.path.insert(0, "/opt/code/membrae")
from gated_core import GatedRecall
DEV="cuda"; torch.manual_seed(0); random.seed(0)

K=64; V=64; PAD=0; Kt=lambda i:1+i; Vt=lambda i:1+K+i; QT=1+K+V; FILL=2+K+V; VOCAB=3+K+V
CHUNK=256

class Blk(nn.Module):
    def __init__(s,d): super().__init__(); s.n1=nn.LayerNorm(d); s.mix=GatedRecall(d,feat=16,chunk=64); s.n2=nn.LayerNorm(d); s.mlp=nn.Sequential(nn.Linear(d,4*d),nn.GELU(),nn.Linear(4*d,d))
    def forward(s,x,st=None): m,ns=s.mix(s.n1(x),st); x=x+m; return x+s.mlp(s.n2(x)), ns
class LM(nn.Module):
    def __init__(s,d=128,L=2): super().__init__(); s.emb=nn.Embedding(VOCAB,d); s.blocks=nn.ModuleList([Blk(d) for _ in range(L)]); s.head=nn.Linear(d,VOCAB)   # NO positional emb -> unbounded streaming
    def forward(s,x,states=None):
        h=s.emb(x); ns=[]
        for i,b in enumerate(s.blocks):
            h,st=b(h, None if states is None else states[i]); ns.append(st)
        return s.head(h), ns

def train_seq(fill_len):                              # [Kt(a) Vt(b)  filler...  QT] -> predict Vt(b)
    a=random.randrange(K); b=random.randrange(V)
    seq=[Kt(a),Vt(b)]+[FILL]*fill_len+[QT]
    return seq, Vt(b)
def make_batch(B,fill_len):
    rows=[train_seq(random.randint(fill_len//2,fill_len)) for _ in range(B)]; T=max(len(s) for s,_ in rows)
    x=torch.full((B,T),PAD,dtype=torch.long); y=torch.full((B,T),-100,dtype=torch.long)
    for bi,(s,tgt) in enumerate(rows):
        x[bi,:len(s)]=torch.tensor(s); y[bi,len(s)-1]=tgt
    return x.to(DEV),y.to(DEV)

def train(m,steps=6000,lr=2e-3,fill=4000):
    opt=torch.optim.AdamW(m.parameters(),lr=lr); m.train()
    for _ in range(steps):
        x,y=make_batch(8,fill); lg,_=m(x); loss=F.cross_entropy(lg.reshape(-1,VOCAB),y.reshape(-1),ignore_index=-100)
        opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(m.parameters(),1.); opt.step()

@torch.no_grad()
def stream_eval(m,fill_len,n=60):
    """Feed the sequence in CHUNKS carrying only state (O(1) mem). Recall the START fact at the END."""
    m.eval(); torch.cuda.reset_peak_memory_stats(); hits=0
    for _ in range(n):
        a=random.randrange(K); b=random.randrange(V)
        seq=[Kt(a),Vt(b)]+[FILL]*fill_len+[QT]
        states=None; last=None
        for c0 in range(0,len(seq),CHUNK):
            xc=torch.tensor([seq[c0:c0+CHUNK]],device=DEV)
            last,states=m(xc,states)
        hits+= int(last[0,-1].argmax())==Vt(b)
    peak=torch.cuda.max_memory_allocated()/2**20
    return hits/n, peak

if __name__=="__main__":
    m=LM().to(DEV)
    print("training streaming recaller (fact @ start, recall @ end)...",flush=True)
    train(m)
    print(f"\nFACULTY #2 -- O(1)-memory streaming recall (chunked, state-only carry):")
    print(f"{'stream len':>11} | {'recall':>7} | {'peak MB':>8}")
    for fl in [500, 2000, 8000, 32000, 100000]:
        acc,mb=stream_eval(m,fl)
        print(f"{fl+3:>11} | {acc*100:6.0f}% | {mb:7.0f}",flush=True)
    print("\nWIN = recall stays high while peak memory stays ~FLAT as stream len -> 100K (attention: bounded + O(T^2)).")
