"""SUBSTRATE BAKE-OFF (step 0 of the integrated brain program). The brain's core token-mixer must support ALL
four brain-properties at once: associative RECALL (for online weight-learning), O(1) long-range STATE (for
constant-memory streaming), TRAINABILITY, and SPEED/compilability (efficiency). Score candidate cores at a FIXED
tiny budget on capability-per-FLOP -- pick the core that can carry the whole brain, and honestly settle whether
spin is it. Small + fast on purpose (the whole thesis is: prove it small).

Cores: spin (diagonal-complex LRU, our thesis) | based (Taylor linear-attn) | gdelta (gated DeltaNet) | attn (ref).
Tests: (1) MQAR associative recall @ {4,16,32} pairs  (2) long-range STATE-TRACKING @ seq {512,2048}  (3) throughput.
"""
import os, sys, math, random, time, torch, torch.nn as nn, torch.nn.functional as F
DEV = "cuda" if torch.cuda.is_available() else "cpu"; torch.manual_seed(0)

# ---------------- cores (drop-in token mixers, x[B,T,d]->[B,T,d]) ----------------
class Spin(nn.Module):                       # diagonal-complex LRU (our carrier, real-valued proxy of s6_hybrid)
    def __init__(s, d):
        super().__init__(); s.b=nn.Linear(d,d); s.dt=nn.Linear(d,d); s.C=nn.Linear(d,d); s.A=nn.Parameter(torch.zeros(d))
    def forward(s,x):
        B,T,d=x.shape; b=s.b(x); lam=torch.exp(-F.softplus(s.dt(x))*F.softplus(s.A)); h=torch.zeros(B,d,device=x.device); ys=[]
        for t in range(T): h=lam[:,t]*h+b[:,t]; ys.append(h)
        return s.C(torch.stack(ys,1))
class Based(nn.Module):                      # Taylor linear attention (parallel cumsum)
    def __init__(s,d,H=4,fe=8):
        super().__init__(); s.H=H; s.fe=fe; s.dh=d//H
        s.q=nn.Linear(d,H*fe); s.k=nn.Linear(d,H*fe); s.v=nn.Linear(d,d); s.o=nn.Linear(d,d)
    def tay(s,x):
        B,T,H,fd=x.shape; x2=(x.unsqueeze(-1)*x.unsqueeze(-2)).reshape(B,T,H,fd*fd)/(2**.5)
        return torch.cat([torch.ones(B,T,H,1,device=x.device),x,x2],-1)
    def forward(s,x):
        B,T,d=x.shape; H,fe,dh=s.H,s.fe,s.dh
        q=s.tay(s.q(x).view(B,T,H,fe)); k=s.tay(s.k(x).view(B,T,H,fe)); v=s.v(x).view(B,T,H,dh)
        S=torch.cumsum(k.unsqueeze(-1)*v.unsqueeze(-2),1); Z=torch.cumsum(k,1)
        num=torch.einsum('bthf,bthfd->bthd',q,S); den=torch.einsum('bthf,bthf->bth',q,Z).clamp_min(1e-4).unsqueeze(-1)
        return s.o((num/den).reshape(B,T,d))
class GDelta(nn.Module):                      # gated DeltaNet (delta rule + forget gate)
    def __init__(s,d):
        super().__init__(); s.k=nn.Linear(d,d); s.v=nn.Linear(d,d); s.q=nn.Linear(d,d); s.b=nn.Linear(d,1); s.g=nn.Linear(d,1); s.o=nn.Linear(d,d)
    def forward(s,x):
        B,T,d=x.shape; k=F.normalize(s.k(x),dim=-1); v=s.v(x); q=s.q(x); beta=torch.sigmoid(s.b(x)); gate=torch.sigmoid(s.g(x))
        St=torch.zeros(B,d,d,device=x.device); out=[]
        for t in range(T):
            kt,vt,qt,bt,gt=k[:,t],v[:,t],q[:,t],beta[:,t],gate[:,t]
            St=gt.unsqueeze(-1)*St                                   # forget gate
            Sk=torch.einsum('bvk,bk->bv',St,kt)
            St=St+bt.unsqueeze(-1)*torch.einsum('bv,bk->bvk',vt-Sk,kt)
            out.append(torch.einsum('bvk,bk->bv',St,qt))
        return s.o(torch.stack(out,1))
class Attn(nn.Module):                        # causal attention reference (O(T^2))
    def __init__(s,d,H=4):
        super().__init__(); s.H=H; s.qkv=nn.Linear(d,3*d); s.o=nn.Linear(d,d)
    def forward(s,x):
        B,T,d=x.shape; qkv=s.qkv(x).view(B,T,3,s.H,d//s.H).permute(2,0,3,1,4); q,k,v=qkv[0],qkv[1],qkv[2]
        return s.o(F.scaled_dot_product_attention(q,k,v,is_causal=True).transpose(1,2).reshape(B,T,d))
CORES={"spin":Spin,"based":Based,"gdelta":GDelta,"attn":Attn}

class Block(nn.Module):
    def __init__(s,d,cls): super().__init__(); s.n1=nn.LayerNorm(d); s.mix=cls(d); s.n2=nn.LayerNorm(d); s.mlp=nn.Sequential(nn.Linear(d,4*d),nn.GELU(),nn.Linear(4*d,d))
    def forward(s,x): x=x+s.mix(s.n1(x)); return x+s.mlp(s.n2(x))
class TinyLM(nn.Module):
    def __init__(s,V,d,L,cls,pos=True):
        super().__init__(); s.emb=nn.Embedding(V,d); s.pos=nn.Embedding(4096,d) if pos else None
        s.blocks=nn.ModuleList([Block(d,cls) for _ in range(L)]); s.head=nn.Linear(d,V)
    def forward(s,x):
        h=s.emb(x); h=h+s.pos(torch.arange(x.shape[1],device=x.device)) if s.pos is not None else h
        for b in s.blocks: h=b(h)
        return s.head(h)

def nparams(m): return sum(p.numel() for p in m.parameters())

# ---------------- test 1: MQAR associative recall ----------------
def mqar_batch(B,npairs,K=64,V=64):
    T=2*npairs+8; x=torch.zeros(B,T,dtype=torch.long); y=torch.full((B,T),-100,dtype=torch.long)
    for b in range(B):
        ks=random.sample(range(1,K+1),npairs); vs=[random.randint(K+1,K+V) for _ in range(npairs)]; kv=dict(zip(ks,vs)); seq=[]
        for k,v in zip(ks,vs): seq+=[k,v]
        qs=[random.choice(ks) for _ in range(4)]
        for q in qs: seq+=[q,kv[q]]
        x[b,:len(seq)]=torch.tensor(seq[:T])
        for j in range(4):
            p=2*npairs+2*j
            if p<T: y[b,p]=kv[qs[j]]
    return x.to(DEV),y.to(DEV)
def test_mqar(cls,npairs,d=96,L=2,steps=1500):
    V=129; torch.manual_seed(0); m=TinyLM(V,d,L,cls).to(DEV); opt=torch.optim.AdamW(m.parameters(),lr=2e-3); m.train()
    for _ in range(steps):
        x,y=mqar_batch(48,npairs); loss=F.cross_entropy(m(x).reshape(-1,V),y.reshape(-1),ignore_index=-100)
        opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(m.parameters(),1.); opt.step()
    m.eval(); c=t=0
    with torch.no_grad():
        for _ in range(15):
            x,y=mqar_batch(48,npairs); p=m(x).argmax(-1); msk=y!=-100; c+=(p[msk]==y[msk]).sum().item(); t+=msk.sum().item()
    return c/t

# ---------------- test 2: long-range STATE-TRACKING (running parity of a rare marker) ----------------
# tokens: 0 filler, 1 = 'flip' marker (rare). Target at each step = parity of #flips so far (2/3). O(1)-state task;
# a transformer must attend over the whole prefix, a good recurrent core keeps it in state -> tests constant-memory state.
def state_batch(B,T,pflip=0.03):
    x=(torch.rand(B,T)<pflip).long()                      # 1 = flip
    par=(torch.cumsum(x,1)%2)                              # running parity
    y=par.clone()                                         # predict parity at every position
    inp=x.clone()                                         # model sees only the flip markers
    return inp.to(DEV), (y+2).to(DEV)                    # labels 2/3 (disjoint from input 0/1)
def test_state(cls,T,d=96,L=2,steps=1200):
    V=4; torch.manual_seed(0); m=TinyLM(V,d,L,cls,pos=(T<=512)).to(DEV); opt=torch.optim.AdamW(m.parameters(),lr=2e-3); m.train()
    for _ in range(steps):
        x,y=state_batch(24,T); loss=F.cross_entropy(m(x).reshape(-1,V),y.reshape(-1))
        opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(m.parameters(),1.); opt.step()
    m.eval(); c=t=0
    with torch.no_grad():
        for _ in range(10):
            x,y=state_batch(24,T); p=m(x).argmax(-1); c+=(p[:,-64:]==y[:,-64:]).sum().item(); t+=y[:,-64:].numel()   # accuracy at the END (needs full-seq state)
    return c/t

# ---------------- test 3: throughput ----------------
def test_speed(cls,d=96,L=2,T=512,B=32):
    m=TinyLM(129,d,L,cls).to(DEV); x=torch.randint(0,129,(B,T),device=DEV); m.train()
    for _ in range(3):
        m(x).sum().backward(); m.zero_grad()
    if DEV=="cuda": torch.cuda.synchronize()
    t0=time.time()
    for _ in range(10): m(x).sum().backward(); m.zero_grad()
    if DEV=="cuda": torch.cuda.synchronize()
    return B*T*10/(time.time()-t0)

if __name__=="__main__":
    print(f"BAKE-OFF on {DEV} | tiny fixed budget (d=96, 2 layers) | capability-per-FLOP\n")
    print(f"{'core':>8} | {'params':>7} | {'MQAR-4':>6} {'MQAR-16':>7} {'MQAR-32':>7} | {'state-512':>9} {'state-2048':>10} | {'tok/s':>8}")
    print("-"*92)
    for name,cls in CORES.items():
        pm=nparams(TinyLM(129,96,2,cls))/1e6
        r4,r16,r32=[test_mqar(cls,n) for n in (4,16,32)]
        s512=test_state(cls,512); s2048=test_state(cls,2048)
        sp=test_speed(cls)
        print(f"{name:>8} | {pm:6.2f}M | {r4*100:5.0f}% {r16*100:6.0f}% {r32*100:6.0f}% | {s512*100:8.0f}% {s2048*100:9.0f}% | {sp/1e3:6.0f}K",flush=True)
    print("\nWIN = the core strongest across recall + long-range state + speed -> the substrate to build the brain on.")
