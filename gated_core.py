"""GATED intrinsic core: ChunkedRecall + a per-token, per-head GATE (decay) -> the network learns WHAT TO KEEP vs
forget, in the forward pass. The gate is the intrinsic forgetting mechanism (vs brain.py replay code). GLA-style
chunked math (per-head scalar decay), pure PyTorch matmuls. Sanity: must still recall fast with the gate added."""
import math, time, random, torch, torch.nn as nn, torch.nn.functional as F
DEV = "cuda" if torch.cuda.is_available() else "cpu"

class GatedRecall(nn.Module):
    def __init__(self, d, heads=4, feat=16, chunk=32):
        super().__init__(); self.H=heads; self.fe=feat; self.dv=d//heads; self.C=chunk
        self.q=nn.Linear(d,heads*feat); self.k=nn.Linear(d,heads*feat); self.v=nn.Linear(d,d)
        self.g=nn.Linear(d,heads); self.o=nn.Linear(d,d)
        nn.init.constant_(self.g.bias, 4.0)              # start ~keep (sigmoid(4)=0.982): don't forget by default
    def taylor(self,x):
        B,L,H,fd=x.shape; x2=(x.unsqueeze(-1)*x.unsqueeze(-2)).reshape(B,L,H,fd*fd)/(2**.5)
        return torch.cat([torch.ones(B,L,H,1,device=x.device,dtype=x.dtype),x,x2],-1)
    def forward(self,x,state=None,return_gate=False):
        B,T,d=x.shape; H,fe,dv,C=self.H,self.fe,self.dv,self.C
        q=self.taylor(self.q(x).view(B,T,H,fe)); k=self.taylor(self.k(x).view(B,T,H,fe)); v=self.v(x).view(B,T,H,dv)
        g=torch.sigmoid(self.g(x))                        # [B,T,H] per-head keep-prob in (0,1)
        Fd=q.shape[-1]
        if state is None: S=torch.zeros(B,H,Fd,dv,device=x.device,dtype=x.dtype); zc=torch.zeros(B,H,Fd,device=x.device,dtype=x.dtype)
        else: S,zc=state
        outs=[]
        for c0 in range(0,T,C):
            c1=min(c0+C,T); L=c1-c0
            qc=q[:,c0:c1].transpose(1,2); kc=k[:,c0:c1].transpose(1,2); vc=v[:,c0:c1].transpose(1,2)  # [B,H,L,*]
            gc=g[:,c0:c1].transpose(1,2).clamp(1e-4,1-1e-6)                                            # [B,H,L]
            Gam=torch.cumprod(gc,dim=-1)                  # prefix decay from chunk start [B,H,L]
            Gi=Gam.unsqueeze(-1)                          # [B,H,L,1]
            qp=qc*Gi; kp=kc/Gi.clamp_min(1e-6)            # GLA reparam: q'=Gamma q, k'=k/Gamma
            A=torch.tril(torch.matmul(qp,kp.transpose(-1,-2)))                # [B,H,L,L] causal, decay-weighted
            intra=torch.matmul(A,vc)                      # within-chunk
            inter=torch.matmul(qp,S)                      # from carried (decayed) state
            den_in=A.sum(-1)                              # normalizer (intra)
            den_ex=torch.einsum('bhlf,bhf->bhl',qp,zc)    # normalizer (state)
            den=(den_in+den_ex).clamp_min(1e-3).unsqueeze(-1)
            outs.append(((intra+inter)/den).transpose(1,2))
            GL=Gam[...,-1:]                               # total chunk decay [B,H,1]
            kpp=kc*(GL.unsqueeze(-1)/Gi.clamp_min(1e-6))  # k''=(Gamma_L/Gamma) k
            S=GL.unsqueeze(-1)*S + torch.matmul(kpp.transpose(-1,-2),vc)
            zc=GL*zc + kpp.sum(2)
        return (self.o(torch.cat(outs,1).reshape(B,T,d)), (S,zc)) if not return_gate else (self.o(torch.cat(outs,1).reshape(B,T,d)),(S,zc),g)

class Block(nn.Module):
    def __init__(s,d,feat): super().__init__(); s.n1=nn.LayerNorm(d); s.mix=GatedRecall(d,feat=feat); s.n2=nn.LayerNorm(d); s.mlp=nn.Sequential(nn.Linear(d,4*d),nn.GELU(),nn.Linear(4*d,d))
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
        for a,c in zip(ks,vs): seq+=[a,c]
        qs=[random.choice(ks) for _ in range(4)]
        for qq in qs: seq+=[qq,kv[qq]]
        x[b,:len(seq)]=torch.tensor(seq[:T])
        for j in range(4):
            p=2*npairs+2*j
            if p<T: y[b,p]=kv[qs[j]]
    return x.to(DEV),y.to(DEV)
def test_recall(npairs,d=128,L=2,steps=2000):
    V=129; torch.manual_seed(0); m=LM(V,d,L).to(DEV); opt=torch.optim.AdamW(m.parameters(),lr=2e-3); m.train()
    for _ in range(steps):
        x,y=mqar_batch(48,npairs); loss=F.cross_entropy(m(x).reshape(-1,V),y.reshape(-1),ignore_index=-100)
        opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(m.parameters(),1.); opt.step()
    m.eval(); c=t=0
    with torch.no_grad():
        for _ in range(15):
            x,y=mqar_batch(48,npairs); p=m(x).argmax(-1); msk=y!=-100; c+=(p[msk]==y[msk]).sum().item(); t+=msk.sum().item()
    return c/t
def test_speed(d=128,L=2,T=512,B=32):
    m=LM(129,d,L).to(DEV); x=torch.randint(0,129,(B,T),device=DEV); m.train()
    for _ in range(3): m(x).sum().backward(); m.zero_grad()
    if DEV=="cuda": torch.cuda.synchronize()
    t0=time.time()
    for _ in range(10): m(x).sum().backward(); m.zero_grad()
    if DEV=="cuda": torch.cuda.synchronize()
    return B*T*10/(time.time()-t0)

if __name__=="__main__":
    print(f"GATED CORE sanity on {DEV} | d=128 2L feat16 chunk32\n")
    for n in (4,16): print(f"  MQAR-{n:<2}: {test_recall(n)*100:5.1f}%",flush=True)
    print(f"  speed:  {test_speed()/1e3:.0f}K tok/s",flush=True)
    print("\n(gate added, must still recall ~100%@16 + stay fast; then -> online-no-forget demo)")
