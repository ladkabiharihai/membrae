"""Efficient MoE dispatch (pure PyTorch, no Triton): capacity-based routing + ONE batched bmm for ALL experts,
instead of the naive per-expert masked loop (#75, which ran at dense speed). Experts stacked as [E,d,4d]/[E,4d,d];
dispatch tokens into [E,C,d] buffer (C=capacity), one bmm -> one bmm, scatter back. Recovers the sparse FLOP saving
in wall-clock. Includes function-preserving expert growth (append expert routed-off). Measures MoE-1B throughput.
"""
import warnings,os,time; warnings.filterwarnings("ignore"); os.environ["PYTHONWARNINGS"]="ignore"
import sys,torch,torch.nn as nn,torch.nn.functional as F
sys.path.insert(0,"/opt/code/membrae")
from fastcore_v import VChunkRecall

class FastMoE(nn.Module):
    def __init__(s,d,E=8,cap=1.5):
        super().__init__(); s.d=d; s.E=E; s.cap=cap; s.router=nn.Linear(d,E)
        s.w1=nn.Parameter(torch.empty(E,d,4*d)); s.w2=nn.Parameter(torch.empty(E,4*d,d))
        s.b1=nn.Parameter(torch.zeros(E,4*d)); s.b2=nn.Parameter(torch.zeros(E,d))
        for e in range(E): nn.init.kaiming_uniform_(s.w1[e],a=5**0.5); nn.init.kaiming_uniform_(s.w2[e],a=5**0.5)
        s.register_buffer("route_sat",torch.ones(1),persistent=False)   # model's own routing-saturation signal (set in forward)
    def forward(s,x):
        B,T,d=x.shape; N=B*T; xf=x.reshape(N,d)
        lg=s.router(xf); pr=lg.softmax(-1); top=lg.argmax(-1); gate=pr.gather(1,top[:,None])   # [N,1]
        C=max(1,int(s.cap*N/s.E)); oneh=F.one_hot(top,s.E)                                       # [N,E]
        # INTRINSIC saturation signal (the model's OWN routing): mean top-1 confidence x load-balance. High+stable =
        # experts specialized and evenly full = the model itself signalling "capacity saturated, need another expert".
        with torch.no_grad():
            load=oneh.float().mean(0); bal=1.0-(load*s.E-1.0).abs().mean()                       # 1=perfectly balanced
            s.route_sat=(gate.mean()*bal).detach()                                               # in (0,1); read by the grower
        posin=(oneh.cumsum(0)-1).gather(1,top[:,None]).squeeze(1)                                # rank within expert
        keep=posin<C
        buf=torch.zeros(s.E,C,d,device=x.device,dtype=x.dtype)
        ek=top[keep]; ck=posin[keep].clamp(max=C-1); buf[ek,ck]=xf[keep].to(buf.dtype)
        h=torch.bmm(buf,s.w1)+s.b1[:,None]; h=F.gelu(h); ob=torch.bmm(h,s.w2)+s.b2[:,None]        # [E,C,d] ONE bmm each
        out=torch.zeros(N,d,device=x.device,dtype=x.dtype); out[keep]=ob[ek,ck].to(out.dtype)*gate[keep].to(out.dtype)
        return out.reshape(B,T,d)
    def grow(s):                                                                                  # function-preserving: add expert, router logit -30 -> unrouted at t=0
        E=s.E; dev=s.w1.device
        nw1=nn.Parameter(torch.empty(E+1,s.d,4*s.d,device=dev)); nw2=nn.Parameter(torch.empty(E+1,4*s.d,s.d,device=dev))
        nb1=nn.Parameter(torch.zeros(E+1,4*s.d,device=dev)); nb2=nn.Parameter(torch.zeros(E+1,s.d,device=dev))
        with torch.no_grad():
            nw1[:E]=s.w1; nw2[:E]=s.w2; nb1[:E]=s.b1; nb2[:E]=s.b2
            nn.init.kaiming_uniform_(nw1[E],a=5**0.5); nn.init.kaiming_uniform_(nw2[E],a=5**0.5)
            nr=nn.Linear(s.d,E+1).to(dev); nr.weight[:E]=s.router.weight; nr.bias[:E]=s.router.bias; nr.weight[E].zero_(); nr.bias[E]=-30.0
        s.w1,s.w2,s.b1,s.b2,s.router,s.E=nw1,nw2,nb1,nb2,nr,E+1

if __name__=="__main__":
    from torch.utils.checkpoint import checkpoint
    DEV="cuda"; V=16384; d=1024; H=8; E=8; L=14
    class Blk(nn.Module):
        def __init__(s): super().__init__(); s.n1=nn.LayerNorm(d); s.mix=VChunkRecall(d,heads=H,feat=8,chunk=128,stable=True); s.n2=nn.LayerNorm(d); s.moe=FastMoE(d,E)
        def forward(s,x): x=x+s.mix(s.n1(x)); return x+s.moe(s.n2(x))
    class Net(nn.Module):
        def __init__(s): super().__init__(); s.emb=nn.Embedding(V,d); s.blocks=nn.ModuleList([Blk() for _ in range(L)]); s.head=nn.Linear(d,V,bias=False); s.head.weight=s.emb.weight
        def forward(s,x):
            h=s.emb(x)
            for b in s.blocks: h=checkpoint(b,h,use_reentrant=False)
            return s.head(h)
    m=Net().to(DEV); opt=torch.optim.AdamW(m.parameters(),lr=1e-4)
    tot=sum(p.numel() for p in m.parameters())/1e9
    print(f"FAST-MoE-1B: {tot:.2f}B total (E={E} top-1, d={d} L={L})",flush=True)
    x=torch.randint(0,V,(4,2048),device=DEV); y=torch.randint(0,V,(4,2048),device=DEV)
    def step():
        opt.zero_grad()
        with torch.autocast("cuda",dtype=torch.bfloat16): loss=F.cross_entropy(m(x).reshape(-1,V),y.reshape(-1))
        loss.backward(); opt.step()
    step(); torch.cuda.synchronize(); t0=time.time()
    for _ in range(3): step()
    torch.cuda.synchronize(); dt=(time.time()-t0)/3; tps=4*2048/dt
    print(f"  FAST dispatch: {tps:.0f} tok/s  ({dt*1e3:.0f}ms/step)  vs naive-MoE/dense 22.7K",flush=True)
    print(f"  -> days to 20B tok: {20e9/tps/86400:.1f}d  (dense-1B ~10d)",flush=True)
    print("DONE",flush=True)
