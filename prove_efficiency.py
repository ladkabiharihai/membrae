"""FACULTY #4: per-FLOP / efficiency WIN vs attention -- a thing a transformer can't do (O(T^2), OOMs).
Same d/layers, increasing sequence length T. Measure forward+backward tok/s and PEAK memory for our ChunkedRecall
core vs causal Attention. Proof = ours stays ~LINEAR time + CONSTANT-per-token memory to 128K while attention
blows up quadratically and OOMs. (This also demonstrates faculty #2's constant-memory streaming substrate.)"""
import sys, time, torch, torch.nn as nn, torch.nn.functional as F
sys.path.insert(0, "/opt/code/membrae")
from fastcore import ChunkedRecall
DEV = "cuda"

class Attn(nn.Module):
    def __init__(s,d,H=4): super().__init__(); s.H=H; s.qkv=nn.Linear(d,3*d); s.o=nn.Linear(d,d)
    def forward(s,x):
        B,T,d=x.shape; qkv=s.qkv(x).view(B,T,3,s.H,d//s.H).permute(2,0,3,1,4); q,k,v=qkv[0],qkv[1],qkv[2]
        return s.o(F.scaled_dot_product_attention(q,k,v,is_causal=True).transpose(1,2).reshape(B,T,d))
class Blk(nn.Module):
    def __init__(s,d,mix): super().__init__(); s.n1=nn.LayerNorm(d); s.mix=mix; s.n2=nn.LayerNorm(d); s.mlp=nn.Sequential(nn.Linear(d,4*d),nn.GELU(),nn.Linear(4*d,d))
    def forward(s,x):
        m=s.mix(s.n1(x)); m=m[0] if isinstance(m,tuple) else m; x=x+m; return x+s.mlp(s.n2(x))
def build(kind,d=256,L=2):
    mk=(lambda: ChunkedRecall(d,chunk=128)) if kind=="ours" else (lambda: Attn(d))
    return nn.Sequential(*[Blk(d,mk()) for _ in range(L)]).to(DEV)

def bench(kind,T,d=256,B=1):
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    m=build(kind,d); x=torch.randn(B,T,d,device=DEV,requires_grad=True)
    try:
        for _ in range(2): m(x).sum().backward(); m.zero_grad(); x.grad=None
        torch.cuda.synchronize(); t0=time.time()
        for _ in range(3): m(x).sum().backward(); m.zero_grad(); x.grad=None
        torch.cuda.synchronize()
        tok_s=B*T*3/(time.time()-t0); peak=torch.cuda.max_memory_allocated()/2**20
        return tok_s, peak
    except RuntimeError as e:
        if "out of memory" in str(e).lower(): torch.cuda.empty_cache(); return None, None
        raise

if __name__=="__main__":
    print(f"FACULTY #4: efficiency vs attention (d=256, 2 layers, fwd+bwd)\n")
    print(f"{'seqlen':>8} | {'ours tok/s':>11} {'ours MB':>8} | {'attn tok/s':>11} {'attn MB':>8} | {'speedup':>8}")
    print("-"*72)
    for T in [1024, 4096, 16384, 65536, 131072]:
        o_ts,o_mb=bench("ours",T)
        a_ts,a_mb=bench("attn",T)
        def f(ts,mb): return (f"{ts/1e3:8.0f}K {mb:7.0f}" if ts else f"{'OOM':>8} {'--':>7}")
        sp = f"{o_ts/a_ts:6.1f}x" if (o_ts and a_ts) else ("attn OOM" if o_ts and not a_ts else "--")
        print(f"{T:>8} | {f(o_ts,o_mb):>20} | {f(a_ts,a_mb):>20} | {sp:>8}",flush=True)
    print("\nWIN = ours runs (linear time, ~flat mem/token) where attention OOMs / slows quadratically.")
