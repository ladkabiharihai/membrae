"""FAST core v2: VECTORIZED chunk-parallel linear attention (Taylor feature -> recall). Kills the eager Python
chunk-loop: all chunks' intra-chunk attention is ONE batched matmul; the inter-chunk state recurrence is ONE
cumsum. ~5 big ops total instead of ~120 small launches -> launch overhead gone, autograd free (pure PyTorch),
no compile/Triton fragility. Must still recall ~100%@16 AND be much faster than the loop core."""
import math, time, random, torch, torch.nn as nn, torch.nn.functional as F
DEV="cuda" if torch.cuda.is_available() else "cpu"

class GroupFeatNorm(nn.Module):
    """Function-preserving replacement for the per-feature LayerNorm when feat grows under stable=True: the OLD feat
    dims keep the ORIGINAL LayerNorm (byte-identical old behavior) and the NEW dims get a separate LayerNorm whose
    bias is zeroed, so a zero-valued new KEY feature -> 0 out -> zero attention contribution. (RESULTS #50)"""
    def __init__(self, old_norm, old_dim, new_dim):
        super().__init__(); self.old=old_norm; self.split=old_dim; self.new=nn.LayerNorm(new_dim); nn.init.zeros_(self.new.bias)
    def forward(self, x):
        return torch.cat([self.old(x[..., :self.split]), self.new(x[..., self.split:])], -1)

class VChunkRecall(nn.Module):
    def __init__(self, d, heads=4, feat=8, chunk=128, stable=False):
        super().__init__(); self.H=heads; self.fe=feat; self.dv=d//heads; self.C=chunk; self.stable=stable
        self.q=nn.Linear(d,heads*feat); self.k=nn.Linear(d,heads*feat); self.v=nn.Linear(d,d); self.o=nn.Linear(d,d)
        if stable: self.qn=nn.LayerNorm(feat); self.kn=nn.LayerNorm(feat)   # bound feature inputs -> no cumsum overflow over long seq
        # SYMMETRIC (deduplicated) 2nd-order Taylor: the outer product x (x) x is symmetric, so only the upper triangle
        # is unique. Using it gives the IDENTICAL inner product phi(q).phi(k) with Fd=1+fe+fe(fe+1)/2 (=45 at fe=8) vs
        # fe^2 (=73) -> ~1.4x fewer feature dims, ZERO capability change (RESULTS #77). iu/diag precomputed per fe.
        iu=torch.triu_indices(feat,feat,offset=1); self.register_buffer("_iu_i",iu[0],persistent=False); self.register_buffer("_iu_j",iu[1],persistent=False)
    def taylor(self,x):                                    # [...,fe] -> [...,1+fe+fe(fe+1)/2]  (symmetric, exact)
        pre=x.shape[:-1]; ones=torch.ones(*pre,1,device=x.device,dtype=x.dtype)
        diag=(x*x)/(2**.5)                                 # i==j terms, coeff 1/sqrt(2)
        off=x[...,self._iu_i]*x[...,self._iu_j]            # i<j terms, coeff 1 (accounts for the 2x from (i,j)+(j,i))
        return torch.cat([ones,x,diag,off],-1)
    def forward(self,x):
        B,T,d=x.shape; H,fe,dv,C=self.H,self.fe,self.dv,self.C
        pad=(C-T%C)%C; Tp=T+pad
        q=self.q(x).view(B,T,H,fe); k=self.k(x).view(B,T,H,fe); v=self.v(x).view(B,T,H,dv)
        if self.stable: q=self.qn(q); k=self.kn(k)                  # bounded feature inputs (long-seq stability)
        if pad: q=F.pad(q,(0,0,0,0,0,pad)); k=F.pad(k,(0,0,0,0,0,pad)); v=F.pad(v,(0,0,0,0,0,pad))
        valid=torch.ones(B,Tp,1,1,device=x.device,dtype=x.dtype);
        if pad: valid[:,T:]=0
        qf=self.taylor(q); kf=self.taylor(k)*valid                 # zero padded keys -> no contribution
        vv=v*valid
        nc=Tp//C; Fd=qf.shape[-1]
        # reshape to chunks: [B,H,nc,C,*]
        def ch(t): return t.view(B,nc,C,H,-1).permute(0,3,1,2,4)    # [B,H,nc,C,X]
        qc=ch(qf); kc=ch(kf); vc=ch(vv)
        # intra-chunk (causal within chunk): A=[B,H,nc,C,C]
        A=torch.matmul(qc,kc.transpose(-1,-2)); cm=torch.tril(torch.ones(C,C,device=x.device,dtype=x.dtype)); A=A*cm
        intra=torch.matmul(A,vc)                                   # [B,H,nc,C,dv]
        din=A.sum(-1)                                              # [B,H,nc,C]
        # per-chunk KV and Z
        KV=torch.einsum('bhncf,bhncd->bhnfd',kc,vc)                # [B,H,nc,Fd,dv]
        Z =kc.sum(3)                                               # [B,H,nc,Fd]
        # EXCLUSIVE cumulative state entering each chunk (sum of PRIOR chunks)
        Sprev=torch.cumsum(KV,2)-KV                                # [B,H,nc,Fd,dv]
        Zprev=torch.cumsum(Z,2)-Z                                  # [B,H,nc,Fd]
        inter=torch.einsum('bhncf,bhnfd->bhncd',qc,Sprev)          # [B,H,nc,C,dv]
        dex=torch.einsum('bhncf,bhnf->bhnc',qc,Zprev)              # [B,H,nc,C]
        out=(intra+inter)/(din+dex).clamp_min(1e-4).unsqueeze(-1)  # [B,H,nc,C,dv]
        out=out.permute(0,2,3,1,4).reshape(B,Tp,d)[:,:T]
        return self.o(out)
    def grow_feat(self, new_feat):
        """STATE-axis function-preserving growth: widen the Taylor feature dim fe (= the recall-capacity dial,
        RESULTS #49/#50). New KEY rows zero-init -> exactly 0 attention contribution at t=0 (output identical);
        new QUERY rows small-random -> break symmetry so the new key dims receive gradient (q-zero+k-zero is a dead
        saddle). v/o untouched (independent of fe). stable=True handled via GroupFeatNorm (old dims keep exact norm)."""
        assert new_feat>self.fe; H,fe=self.H,self.fe; din=self.q.in_features
        dev=self.q.weight.device; dt=self.q.weight.dtype
        def expand(lin, rand_new):
            ow=lin.weight.data.view(H,fe,din); ob=lin.bias.data.view(H,fe)
            nl=nn.Linear(din,H*new_feat).to(dev,dt); nw=nl.weight.data.view(H,new_feat,din); nb=nl.bias.data.view(H,new_feat)
            nw.zero_(); nb.zero_(); nw[:,:fe,:]=ow; nb[:,:fe]=ob
            if rand_new: nw[:,fe:,:].normal_(0,0.02)
            return nl
        self.k=expand(self.k, False); self.q=expand(self.q, True)
        if self.stable:
            self.qn=GroupFeatNorm(self.qn, fe, new_feat-fe).to(dev,dt)
            self.kn=GroupFeatNorm(self.kn, fe, new_feat-fe).to(dev,dt)
        self.fe=new_feat

# ---- test harness ----
class Blk(nn.Module):
    def __init__(s,d,mix): super().__init__(); s.n1=nn.LayerNorm(d); s.mix=mix; s.n2=nn.LayerNorm(d); s.mlp=nn.Sequential(nn.Linear(d,4*d),nn.GELU(),nn.Linear(4*d,d))
    def forward(s,x): m=s.mix(s.n1(x)); m=m[0] if isinstance(m,tuple) else m; x=x+m; return x+s.mlp(s.n2(x))
class LM(nn.Module):
    def __init__(s,V,d,L,mk): super().__init__(); s.emb=nn.Embedding(V,d); s.pos=nn.Embedding(4096,d); s.blocks=nn.ModuleList([Blk(d,mk()) for _ in range(L)]); s.head=nn.Linear(d,V)
    def forward(s,x):
        h=s.emb(x)+s.pos(torch.arange(x.shape[1],device=x.device))
        for b in s.blocks: h=b(h)
        return s.head(h)
def mqar(B,n,K=64,Vv=64):
    T=2*n+8; x=torch.zeros(B,T,dtype=torch.long); y=torch.full((B,T),-100,dtype=torch.long)
    for b in range(B):
        ks=random.sample(range(1,K+1),n); vs=[random.randint(K+1,K+Vv) for _ in ks]; kv=dict(zip(ks,vs)); s=[]
        for a,c in zip(ks,vs): s+=[a,c]
        qs=[random.choice(ks) for _ in range(4)]
        for qq in qs: s+=[qq,kv[qq]]
        x[b,:len(s)]=torch.tensor(s[:T])
        for j in range(4):
            p=2*n+2*j
            if p<T: y[b,p]=kv[qs[j]]
    return x.to(DEV),y.to(DEV)
def recall(mk,n=16,d=128,L=2,steps=2000):
    Vc=129; torch.manual_seed(0); m=LM(Vc,d,L,mk).to(DEV); opt=torch.optim.AdamW(m.parameters(),lr=2e-3); m.train()
    for _ in range(steps):
        x,y=mqar(48,n); F.cross_entropy(m(x).reshape(-1,Vc),y.reshape(-1),ignore_index=-100).backward(); opt.step(); opt.zero_grad()
    m.eval(); c=t=0
    with torch.no_grad():
        for _ in range(15):
            x,y=mqar(48,n); p=m(x).argmax(-1); msk=y!=-100; c+=(p[msk]==y[msk]).sum().item(); t+=msk.sum().item()
    return c/t
def speed(mk,d=256,L=2,T=65536,B=1):
    m=nn.Sequential(*[Blk(d,mk()) for _ in range(L)]).to(DEV); x=torch.randn(B,T,d,device=DEV,requires_grad=True)
    try:
        for _ in range(3): m(x).sum().backward(); m.zero_grad(); x.grad=None
        torch.cuda.synchronize(); t0=time.time()
        for _ in range(3): m(x).sum().backward(); m.zero_grad(); x.grad=None
        torch.cuda.synchronize(); return B*T*3/(time.time()-t0)
    except RuntimeError as e:
        return None if "out of memory" in str(e).lower() else (_ for _ in ()).throw(e)

if __name__=="__main__":
    print(f"VECTORIZED chunk-parallel core on {DEV}\n")
    mk=lambda: VChunkRecall(128,feat=8,chunk=128)
    print(f"  recall @16: {recall(mk)*100:.0f}%",flush=True)
    mk256=lambda: VChunkRecall(256,feat=8,chunk=256)
    print(f"  speed @64K (d=256): {speed(mk256)/1e3:.0f}K tok/s  (loop-core was 78K; big-chunk-loop 311K)",flush=True)
    print("\nWIN = recall ~100%@16 AND much faster than the loop core (no per-chunk python launches).")
