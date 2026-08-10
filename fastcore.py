"""FAST INTRINSIC CORE (a2): chunked-parallel linear attention with a Taylor feature map (softmax-like SHARPNESS
-> associative recall, validated in RESULTS #29). The fast-weight state S = sum phi(k)^T v IS the memory; recall
is o = phi(q) S; reading writes S -> learning is the forward pass. FAST via CHUNKING with MATMULS (no per-token
python loop): intra-chunk parallel (causal-masked phi(Q)phi(K)^T @ V), inter-chunk carries the single state S.
Pure PyTorch, no deps, no compat risk. Add a per-token GATE later for forgetting; first prove fast + recall."""
import math, time, random, torch, torch.nn as nn, torch.nn.functional as F
DEV = "cuda" if torch.cuda.is_available() else "cpu"

class ChunkedRecall(nn.Module):
    def __init__(self, d, heads=4, feat=16, chunk=128):
        super().__init__(); self.H=heads; self.fe=feat; self.dv=d//heads; self.C=chunk
        self.q=nn.Linear(d,heads*feat); self.k=nn.Linear(d,heads*feat); self.v=nn.Linear(d,d); self.o=nn.Linear(d,d)
    def taylor(self, x):                                  # [B,L,H,fe] -> [B,L,H,1+fe+fe^2]  (2nd-order Taylor of exp)
        B,L,H,fd=x.shape; x2=(x.unsqueeze(-1)*x.unsqueeze(-2)).reshape(B,L,H,fd*fd)/(2**.5)
        return torch.cat([torch.ones(B,L,H,1,device=x.device,dtype=x.dtype), x, x2], -1)
    def forward(self, x, state=None):
        B,T,d=x.shape; H,fe,dv,C=self.H,self.fe,self.dv,self.C
        q=self.taylor(self.q(x).view(B,T,H,fe)); k=self.taylor(self.k(x).view(B,T,H,fe)); v=self.v(x).view(B,T,H,dv)
        Fd=q.shape[-1]
        if state is None: S=torch.zeros(B,H,Fd,dv,device=x.device,dtype=x.dtype); z=torch.zeros(B,H,Fd,device=x.device,dtype=x.dtype)
        else: S,z=state
        outs=[]
        for c0 in range(0,T,C):                           # CHUNK loop: T/C iters (not T) -- matmuls inside
            c1=min(c0+C,T); L=c1-c0
            qc=q[:,c0:c1].transpose(1,2); kc=k[:,c0:c1].transpose(1,2); vc=v[:,c0:c1].transpose(1,2)  # [B,H,L,*]
            A=torch.matmul(qc, kc.transpose(-1,-2))       # [B,H,L,L] = phi(q).phi(k)
            causal=torch.tril(torch.ones(L,L,device=x.device,dtype=x.dtype))
            A=A*causal
            intra=torch.matmul(A, vc)                     # [B,H,L,dv] within-chunk (parallel, causal)
            inter=torch.matmul(qc, S)                     # [B,H,L,dv] from carried state
            den_intra=A.sum(-1)                           # [B,H,L]
            den_inter=torch.einsum('bhlf,bhf->bhl', qc, z)
            den=(den_intra+den_inter).clamp_min(1e-4).unsqueeze(-1)
            oc=((intra+inter)/den).transpose(1,2)         # [B,L,H,dv]
            outs.append(oc)
            S=S + torch.matmul(kc.transpose(-1,-2), vc)   # update state with this chunk: sum phi(k)^T v
            z=z + kc.sum(2)                               # [B,H,Fd]
        return self.o(torch.cat(outs,1).reshape(B,T,d)), (S,z)

# ---- tiny LM to test recall + speed ----
class Block(nn.Module):
    def __init__(s,d,feat): super().__init__(); s.n1=nn.LayerNorm(d); s.mix=ChunkedRecall(d,feat=feat); s.n2=nn.LayerNorm(d); s.mlp=nn.Sequential(nn.Linear(d,4*d),nn.GELU(),nn.Linear(4*d,d))
    def forward(s,x): m,_=s.mix(s.n1(x)); x=x+m; return x+s.mlp(s.n2(x))
class LM(nn.Module):
    def __init__(s,V,d,L,feat=16,pos=True): super().__init__(); s.emb=nn.Embedding(V,d); s.pos=nn.Embedding(4096,d) if pos else None; s.blocks=nn.ModuleList([Block(d,feat) for _ in range(L)]); s.head=nn.Linear(d,V)
    def forward(s,x):
        h=s.emb(x); h=h+s.pos(torch.arange(x.shape[1],device=x.device)) if s.pos is not None else h
        for b in s.blocks: h=b(h)
        return s.head(h)

def mqar_batch(B,npairs,K=64,V=64):
    T=2*npairs+8; x=torch.zeros(B,T,dtype=torch.long); y=torch.full((B,T),-100,dtype=torch.long)
    for b in range(B):
        ks=random.sample(range(1,K+1),npairs); vs=[random.randint(K+1,K+V) for _ in range(npairs)]; kv=dict(zip(ks,vs)); seq=[]
        for k2,v2 in zip(ks,vs): seq+=[k2,v2]
        qs=[random.choice(ks) for _ in range(4)]
        for qq in qs: seq+=[qq,kv[qq]]
        x[b,:len(seq)]=torch.tensor(seq[:T])
        for j in range(4):
            p=2*npairs+2*j
            if p<T: y[b,p]=kv[qs[j]]
    return x.to(DEV),y.to(DEV)
def test_recall(npairs,d=128,L=2,feat=16,steps=2000):
    V=129; torch.manual_seed(0); m=LM(V,d,L,feat).to(DEV); opt=torch.optim.AdamW(m.parameters(),lr=2e-3); m.train()
    for _ in range(steps):
        x,y=mqar_batch(48,npairs); loss=F.cross_entropy(m(x).reshape(-1,V),y.reshape(-1),ignore_index=-100)
        opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(m.parameters(),1.); opt.step()
    m.eval(); c=t=0
    with torch.no_grad():
        for _ in range(15):
            x,y=mqar_batch(48,npairs); p=m(x).argmax(-1); msk=y!=-100; c+=(p[msk]==y[msk]).sum().item(); t+=msk.sum().item()
    return c/t
def test_speed(d=128,L=2,T=512,B=32,feat=16):
    m=LM(129,d,L,feat).to(DEV); x=torch.randint(0,129,(B,T),device=DEV); m.train()
    for _ in range(3): m(x).sum().backward(); m.zero_grad()
    if DEV=="cuda": torch.cuda.synchronize()
    t0=time.time()
    for _ in range(10): m(x).sum().backward(); m.zero_grad()
    if DEV=="cuda": torch.cuda.synchronize()
    return B*T*10/(time.time()-t0)

if __name__=="__main__":
    print(f"FAST CHUNKED CORE (a2) on {DEV} | d=128, 2L, feat=16, chunk=128\n")
    for n in (4,16,32): print(f"  MQAR-{n:<2}: {test_recall(n)*100:5.1f}%",flush=True)
    print(f"  speed:   {test_speed()/1e3:.0f}K tok/s  (spin 211K, my slow loop-core 26K)",flush=True)
    print("\nWIN = recalls (>=16 pairs like #29) AND fast (>>26K) -> the intrinsic-brain substrate, ours.")
