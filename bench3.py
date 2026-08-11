"""3-WAY SPEED: attention (SDPA/flash) vs spin (s6_hybrid parallel-scan carrier) vs VChunkRecall (our brain core).
Matched d/layers, fwd+bwd tok/s across sequence lengths. Reports tok/s + ratio vs attention and vs spin.
(Runs alongside the training run -> contention lowers ALL three equally, so the RATIOS are fair.)"""
import warnings,os; warnings.filterwarnings("ignore"); os.environ["PYTHONWARNINGS"]="ignore"
import sys,time,torch,torch.nn as nn,torch.nn.functional as F
sys.path.insert(0,"/opt/code/membrae")
from fastcore_v import VChunkRecall
from s6_hybrid import SpinCarrier
DEV="cuda"
class Attn(nn.Module):
    def __init__(s,d,H=8): super().__init__(); s.H=H; s.qkv=nn.Linear(d,3*d); s.o=nn.Linear(d,d)
    def forward(s,x):
        B,T,d=x.shape; qkv=s.qkv(x).view(B,T,3,s.H,d//s.H).permute(2,0,3,1,4); q,k,v=qkv[0],qkv[1],qkv[2]
        return s.o(F.scaled_dot_product_attention(q,k,v,is_causal=True).transpose(1,2).reshape(B,T,d))
class SpinMix(nn.Module):
    def __init__(s,d): super().__init__(); s.c=SpinCarrier(d)
    def forward(s,x): o=s.c(x); return o[0] if isinstance(o,tuple) else o
class Blk(nn.Module):
    def __init__(s,d,mk): super().__init__(); s.n1=nn.LayerNorm(d); s.mix=mk(d); s.n2=nn.LayerNorm(d); s.mlp=nn.Sequential(nn.Linear(d,4*d),nn.GELU(),nn.Linear(4*d,d))
    def forward(s,x): m=s.mix(s.n1(x)); m=m[0] if isinstance(m,tuple) else m; x=x+m; return x+s.mlp(s.n2(x))
def net(mk,d=512,L=2): return nn.Sequential(*[Blk(d,mk) for _ in range(L)]).to(DEV)
CORES={"attention":lambda d:Attn(d), "spin":lambda d:SpinMix(d), "VChunkRecall":lambda d:VChunkRecall(d,heads=8,feat=8,chunk=128)}
def speed(mk,T,B=1,d=512):
    torch.cuda.empty_cache()
    try:
        m=net(mk,d); x=torch.randn(B,T,d,device=DEV,requires_grad=True)
        for _ in range(2): m(x).sum().backward(); m.zero_grad(); x.grad=None
        torch.cuda.synchronize(); t0=time.time()
        for _ in range(3): m(x).sum().backward(); m.zero_grad(); x.grad=None
        torch.cuda.synchronize(); return B*T*3/(time.time()-t0)
    except RuntimeError as e:
        if "out of memory" in str(e).lower(): torch.cuda.empty_cache(); return None
        raise
if __name__=="__main__":
    print(f"3-WAY fwd+bwd tok/s (d=512, 2 layers) on {DEV} -- higher=faster\n",flush=True)
    print(f"{'seqlen':>7} | {'attention':>10} | {'spin':>10} | {'VChunkRecall':>12} | {'vs attn':>8} {'vs spin':>8}")
    print("-"*72)
    for T in [512,2048,8192,32768]:
        r={}
        for name,mk in CORES.items(): r[name]=speed(mk,T)
        def f(v): return f"{v/1e3:.0f}K" if v else "OOM/slow"
        va=f"{r['VChunkRecall']/r['attention']:.1f}x" if r['VChunkRecall'] and r['attention'] else "-"
        vs=f"{r['VChunkRecall']/r['spin']:.1f}x" if r['VChunkRecall'] and r['spin'] else "-"
        print(f"{T:>7} | {f(r['attention']):>10} | {f(r['spin']):>10} | {f(r['VChunkRecall']):>12} | {va:>8} {vs:>8}",flush=True)
    print("\n(ratios are the honest answer -- absolute tok/s is depressed by the concurrent training run)",flush=True)
