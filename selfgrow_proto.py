"""PROTOTYPE + COMPARE the 3 scaffolding-free self-growth approaches on ONE variable-difficulty testbed.
Task = VARIABLE-HOP lookup: context gives a successor map (X->succ(X) for all X); query (start, HOPS=h) -> apply
succ h times. Harder h needs MORE capacity/compute -> lets each approach show capacity-ON-DEMAND by its OWN signal.
  BASELINE : fixed 2-layer VChunkRecall (no adaptivity) -- control.
  A PONDER : weight-tied block iterated with a LEARNED HALTING unit (PonderNet). Depth-on-demand, zero scaffold.
  B MoE    : pool of expert-blocks + LEARNED ROUTER recruits them. Width-on-demand, zero scaffold (pre-set ceiling).
  C HYPER  : a controller GENERATES the mixing weights per-input from a state summary. Self-generated capacity.
Metrics: overall acc, acc on HARD (h>=3), and does the model's OWN capacity signal track difficulty h
(A: expected halt-steps vs h; B: router spread vs h) = the intrinsic-decision check.
"""
import warnings,os; warnings.filterwarnings("ignore"); os.environ["PYTHONWARNINGS"]="ignore"
import sys,time,random,torch,torch.nn as nn,torch.nn.functional as F
sys.path.insert(0,"/opt/code/membrae")
from fastcore_v import VChunkRecall
DEV="cuda" if torch.cuda.is_available() else "cpu"; torch.manual_seed(0); random.seed(0)

N=16; HMAX=4; PAD=0
SYM=lambda i:1+i; ARROW=1+N; Q=2+N; HOP=lambda h:2+N+h; VOCAB=2+N+HMAX+1
def sample():
    perm=list(range(N)); random.shuffle(perm)                 # succ map: i -> perm[i]
    seq=[]
    order=list(range(N)); random.shuffle(order)
    for i in order: seq+=[SYM(i),ARROW,SYM(perm[i])]
    h=random.randint(1,HMAX); start=random.randrange(N); cur=start
    for _ in range(h): cur=perm[cur]
    seq+=[Q,SYM(start),HOP(h)]
    return seq,SYM(cur),h
def batch(B):
    rows=[sample() for _ in range(B)]; T=max(len(s) for s,_,_ in rows)
    x=torch.full((B,T),PAD,dtype=torch.long); y=torch.full((B,T),-100,dtype=torch.long); hs=[]
    for b,(s,t,h) in enumerate(rows): x[b,:len(s)]=torch.tensor(s); y[b,len(s)-1]=t; hs.append(h)
    return x.to(DEV),y.to(DEV),torch.tensor(hs)

def mix_block(d): return VChunkRecall(d,heads=4,feat=8,chunk=64)
class FF(nn.Module):
    def __init__(s,d): super().__init__(); s.n1=nn.LayerNorm(d); s.mix=mix_block(d); s.n2=nn.LayerNorm(d); s.mlp=nn.Sequential(nn.Linear(d,4*d),nn.GELU(),nn.Linear(4*d,d))
    def forward(s,x): x=x+s.mix(s.n1(x)); return x+s.mlp(s.n2(x))

# ---------- BASELINE ----------
class Baseline(nn.Module):
    def __init__(s,d=128,L=2): super().__init__(); s.emb=nn.Embedding(VOCAB,d); s.pos=nn.Embedding(512,d); s.blocks=nn.ModuleList([FF(d) for _ in range(L)]); s.head=nn.Linear(d,VOCAB)
    def forward(s,x,ret=False):
        h=s.emb(x)+s.pos(torch.arange(x.shape[1],device=x.device)[None])
        for b in s.blocks: h=b(h)
        return (s.head(h),{}) if ret else s.head(h)

# ---------- A: PonderNet adaptive halting (weight-tied block, learned #steps) ----------
class Ponder(nn.Module):
    def __init__(s,d=128,nmax=6,prior=0.4): super().__init__(); s.emb=nn.Embedding(VOCAB,d); s.pos=nn.Embedding(512,d); s.block=FF(d); s.halt=nn.Linear(d,1); s.head=nn.Linear(d,VOCAB); s.nmax=nmax; s.prior=prior
    def forward(s,x,ret=False):
        h=s.emb(x)+s.pos(torch.arange(x.shape[1],device=x.device)[None])
        B,T,d=h.shape; still=torch.ones(B,device=h.device); halts=[]; logits=[]; ps=[]
        for n in range(s.nmax):
            h=s.block(h)
            lam=torch.sigmoid(s.halt(h[:,-1,0:1]) if False else s.halt(h[:,-1]).squeeze(-1))   # per-seq halt prob
            lam=torch.sigmoid(s.halt(h[:,-1]).squeeze(-1))
            p = still*lam if n<s.nmax-1 else still                # last step forced
            ps.append(p); logits.append(s.head(h)); still=still*(1-lam)
        P=torch.stack(ps,1)                                       # [B,nmax] halting distribution
        LG=torch.stack(logits,1)                                  # [B,nmax,T,V]
        exp_steps=(P*torch.arange(1,s.nmax+1,device=h.device)[None]).sum(1)
        if ret: return LG,{"P":P,"exp_steps":exp_steps}
        return LG,P
    def loss(s,x,y):
        LG,P=s(x)
        B,nmax,T,V=LG.shape
        ce=torch.stack([F.cross_entropy(LG[:,n].reshape(-1,V),y.reshape(-1),ignore_index=-100,reduction='none').reshape(B,T).sum(1) for n in range(nmax)],1)
        pond=(P*ce).sum(1).mean()
        pri=torch.full_like(P, 0.0);
        g=torch.tensor([s.prior*((1-s.prior)**n) for n in range(nmax)],device=x.device); g=g/g.sum()
        kl=(P*((P+1e-8).log()-(g[None]+1e-8).log())).sum(1).mean()
        return pond+0.01*kl
    @torch.no_grad()
    def predict(s,x): LG,info=s(x,ret=True); return LG[:, -1] if False else torch.einsum('bn,bntv->btv',info["P"],LG), info

# ---------- B: MoE learned recruitment ----------
class MoE(nn.Module):
    def __init__(s,d=128,E=6,k=2,L=2): super().__init__(); s.emb=nn.Embedding(VOCAB,d); s.pos=nn.Embedding(512,d); s.experts=nn.ModuleList([FF(d) for _ in range(E)]); s.router=nn.Linear(d,E); s.E=E; s.k=k; s.L=L; s.norm=nn.ModuleList([nn.LayerNorm(d) for _ in range(L)]); s.head=nn.Linear(d,VOCAB)
    def forward(s,x,ret=False):
        h=s.emb(x)+s.pos(torch.arange(x.shape[1],device=x.device)[None]); loads=[]
        for l in range(s.L):
            g=torch.softmax(s.router(s.norm[l](h).mean(1)),-1)   # per-seq routing over experts
            topv,topi=g.topk(s.k,-1); topv=topv/topv.sum(-1,keepdim=True)
            out=torch.zeros_like(h)
            for j in range(s.k):
                for e in range(s.E):
                    m=(topi[:,j]==e)
                    if m.any(): out[m]=out[m]+topv[m,j:j+1,None]*s.experts[e](h[m])
            h=out; loads.append(g)
        L0=torch.stack(loads,1).mean(1)                          # [B,E] avg expert usage
        info={"load":L0,"nactive":(L0>0.1).float().sum(-1)}
        return (s.head(h),info) if ret else s.head(h)

# ---------- C: Hypernet self-generated mixing weights ----------
class Hyper(nn.Module):
    def __init__(s,d=128,r=16,L=2):
        super().__init__(); s.emb=nn.Embedding(VOCAB,d); s.pos=nn.Embedding(512,d); s.d=d; s.r=r; s.L=L
        # controller reads a pooled summary -> generates low-rank per-input mixing (U:[d,r], V:[r,d])
        s.ctrl=nn.ModuleList([nn.Linear(d,2*d*r) for _ in range(L)]); s.norm=nn.ModuleList([nn.LayerNorm(d) for _ in range(L)])
        s.mix=nn.ModuleList([mix_block(d) for _ in range(L)]); s.n2=nn.ModuleList([nn.LayerNorm(d) for _ in range(L)])
        s.mlp=nn.ModuleList([nn.Sequential(nn.Linear(d,4*d),nn.GELU(),nn.Linear(4*d,d)) for _ in range(L)]); s.head=nn.Linear(d,VOCAB)
    def forward(s,x,ret=False):
        h=s.emb(x)+s.pos(torch.arange(x.shape[1],device=x.device)[None]); B,T,d=h.shape
        for l in range(s.L):
            summ=s.norm[l](h).mean(1)                            # [B,d]
            w=s.ctrl[l](summ).view(B,2,d,s.r); U=w[:,0]; Vv=w[:,1]  # per-input low-rank
            gen=torch.einsum('btd,bdr->btr',h,U); gen=torch.einsum('btr,bdr->btd',gen,Vv)/ (s.r**0.5)  # self-generated transform
            h=h+gen + s.mix[l](s.n2[l](h))                        # generated mix + fixed recall mix
            h=h+s.mlp[l](s.n2[l](h))
        return (s.head(h),{}) if ret else s.head(h)

# ---------- train / eval ----------
def train_eval(name,build,steps=3000,lr=2e-3):
    torch.manual_seed(0); m=build().to(DEV); opt=torch.optim.AdamW(m.parameters(),lr=lr); m.train()
    for it in range(1,steps+1):
        x,y,_=batch(64)
        if name=="A": loss=m.loss(x,y)
        else: loss=F.cross_entropy(m(x).reshape(-1,VOCAB),y.reshape(-1),ignore_index=-100)
        opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(m.parameters(),1.); opt.step()
    m.eval(); import collections
    correct=collections.defaultdict(int); tot=collections.defaultdict(int); cap=collections.defaultdict(list)
    with torch.no_grad():
        for _ in range(40):
            x,y,hs=batch(64)
            if name=="A":
                lg,info=m.predict(x); pred=lg.argmax(-1); es=info["exp_steps"].cpu()
            else:
                out=m(x,ret=True); lg=out[0] if isinstance(out,tuple) else out; info=out[1] if isinstance(out,tuple) else {}; pred=lg.argmax(-1)
            for b in range(x.shape[0]):
                p=(y[b]!=-100).nonzero()[0].item(); ok=int(pred[b,p]==y[b,p]); h=int(hs[b])
                correct[h]+=ok; tot[h]+=1
                if name=="A": cap[h].append(float(es[b]))
                elif name=="B": cap[h].append(float(info["nactive"][b]))
    ov=sum(correct.values())/sum(tot.values()); hard=(correct[3]+correct[4])/max(1,tot[3]+tot[4])
    capstr=""
    if cap: capstr=" | capacity vs hop: "+", ".join(f"h{h}={sum(cap[h])/len(cap[h]):.2f}" for h in sorted(cap))
    npar=sum(p.numel() for p in m.parameters())/1e6
    print(f"  {name:9s} {npar:4.1f}M  overall={ov*100:4.0f}%  hard(h>=3)={hard*100:4.0f}%{capstr}",flush=True)
    return ov,hard

if __name__=="__main__":
    print(f"SELF-GROWTH PROTOTYPES on variable-hop lookup (N={N}, HMAX={HMAX}, VOCAB={VOCAB}) on {DEV}\n",flush=True)
    print("  model      params  overall   hard      [intrinsic: does own capacity signal track difficulty?]",flush=True)
    train_eval("BASELINE",lambda:Baseline(128,2))
    train_eval("A",lambda:Ponder(128,nmax=6))
    train_eval("B",lambda:MoE(128,E=6,k=2,L=2))
    train_eval("C",lambda:Hyper(128,r=16,L=2))
    print("\nWIN = higher hard-acc AND (A/B) capacity signal RISES with hop h (model self-allocates by its own signal).",flush=True)
    print("[done]",flush=True)
