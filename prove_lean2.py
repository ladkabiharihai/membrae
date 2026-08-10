"""FACULTY #4 push: the bottleneck is the eager Python chunk-loop vs FUSED flash-attn. Fuse ours too:
lean feat=8 + bigger chunks + torch.compile (compile corrupted the spin SCAN, but this is a linear-attn matmul
loop -- should fuse fine). Benchmark @65536 vs flash-attention. WIN = ours > attention at 64K."""
import sys, time, torch, torch.nn as nn, torch.nn.functional as F
sys.path.insert(0, "/opt/code/membrae")
from fastcore import ChunkedRecall
DEV="cuda"
class Attn(nn.Module):
    def __init__(s,d,H=4): super().__init__(); s.H=H; s.qkv=nn.Linear(d,3*d); s.o=nn.Linear(d,d)
    def forward(s,x):
        B,T,d=x.shape; qkv=s.qkv(x).view(B,T,3,s.H,d//s.H).permute(2,0,3,1,4); q,k,v=qkv[0],qkv[1],qkv[2]
        return s.o(F.scaled_dot_product_attention(q,k,v,is_causal=True).transpose(1,2).reshape(B,T,d))
class Blk(nn.Module):
    def __init__(s,d,mix): super().__init__(); s.n1=nn.LayerNorm(d); s.mix=mix; s.n2=nn.LayerNorm(d); s.mlp=nn.Sequential(nn.Linear(d,4*d),nn.GELU(),nn.Linear(4*d,d))
    def forward(s,x): m=s.mix(s.n1(x)); m=m[0] if isinstance(m,tuple) else m; x=x+m; return x+s.mlp(s.n2(x))
def net(mixfn,d=256,L=2): return nn.Sequential(*[Blk(d,mixfn()) for _ in range(L)]).to(DEV)
def speed(m,T=65536,B=1,d=256,compile=False):
    x=torch.randn(B,T,d,device=DEV,requires_grad=True)
    if compile: m=torch.compile(m)
    try:
        for _ in range(3): m(x).sum().backward(); m.zero_grad(); x.grad=None    # warmup (compile builds here)
        torch.cuda.synchronize(); t0=time.time()
        for _ in range(3): m(x).sum().backward(); m.zero_grad(); x.grad=None
        torch.cuda.synchronize(); return B*T*3/(time.time()-t0)
    except RuntimeError as e:
        if "out of memory" in str(e).lower(): torch.cuda.empty_cache(); return None
        raise

if __name__=="__main__":
    a=speed(net(lambda: Attn(256))); print(f"flash-attention @64K: {a/1e3:.0f}K tok/s\n",flush=True)
    configs=[("lean feat8 chunk128 eager", lambda: ChunkedRecall(256,feat=8,chunk=128), False),
             ("lean feat8 chunk512 eager", lambda: ChunkedRecall(256,feat=8,chunk=512), False),
             ("lean feat8 chunk256 COMPILE", lambda: ChunkedRecall(256,feat=8,chunk=256), True),
             ("lean feat8 chunk512 COMPILE", lambda: ChunkedRecall(256,feat=8,chunk=512), True)]
    print(f"{'config':>30} | {'tok/s':>8} | {'vs attn':>7}")
    print("-"*52)
    for name,mk,comp in configs:
        try: ts=speed(net(mk),compile=comp)
        except Exception as e: ts=None; print(f"  ({name}: {str(e)[:40]})")
        vs=f"{ts/a:.2f}x" if ts else "FAIL"
        print(f"{name:>30} | {ts/1e3 if ts else 0:7.0f}K | {vs:>7}",flush=True)
    print("\nWIN = any config > 1.0x -> #4 achieved (lean+fused beats attention at 64K).")
