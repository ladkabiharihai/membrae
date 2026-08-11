"""FACULTY #4 attempt: find the LEANEST feature that still RECALLS, then check it beats attention at long T.
The F=273 Taylor (feat=16) was the cost. Sweep feat -> for each: recall@16 (must stay high) AND fwd+bwd tok/s
at T=65536 vs flash-attention. WIN = a feat that recalls ~100%@16 AND is faster than attention at 64K."""
import sys, time, random, torch, torch.nn as nn, torch.nn.functional as F
sys.path.insert(0, "/opt/code/membrae")
from fastcore import ChunkedRecall
DEV="cuda"; torch.manual_seed(0); random.seed(0)

class Attn(nn.Module):
    def __init__(s,d,H=4): super().__init__(); s.H=H; s.qkv=nn.Linear(d,3*d); s.o=nn.Linear(d,d)
    def forward(s,x):
        B,T,d=x.shape; qkv=s.qkv(x).view(B,T,3,s.H,d//s.H).permute(2,0,3,1,4); q,k,v=qkv[0],qkv[1],qkv[2]
        return s.o(F.scaled_dot_product_attention(q,k,v,is_causal=True).transpose(1,2).reshape(B,T,d))
class Blk(nn.Module):
    def __init__(s,d,mix): super().__init__(); s.n1=nn.LayerNorm(d); s.mix=mix; s.n2=nn.LayerNorm(d); s.mlp=nn.Sequential(nn.Linear(d,4*d),nn.GELU(),nn.Linear(4*d,d))
    def forward(s,x): m=s.mix(s.n1(x)); m=m[0] if isinstance(m,tuple) else m; x=x+m; return x+s.mlp(s.n2(x))

def speed(mixfn,d=256,L=2,T=65536,B=1):
    torch.cuda.empty_cache(); m=nn.Sequential(*[Blk(d,mixfn()) for _ in range(L)]).to(DEV); x=torch.randn(B,T,d,device=DEV,requires_grad=True)
    try:
        for _ in range(2): m(x).sum().backward(); m.zero_grad(); x.grad=None
        torch.cuda.synchronize(); t0=time.time()
        for _ in range(3): m(x).sum().backward(); m.zero_grad(); x.grad=None
        torch.cuda.synchronize(); return B*T*3/(time.time()-t0)
    except RuntimeError as e:
        if "out of memory" in str(e).lower(): torch.cuda.empty_cache(); return None
        raise

# recall harness (reuse fastcore's LM path)
from fastcore import LM, mqar_batch
def recall16(feat,d=128,L=2,steps=2000):
    V=129; torch.manual_seed(0); m=LM(V,d,L,feat).to(DEV); opt=torch.optim.AdamW(m.parameters(),lr=2e-3); m.train()
    for _ in range(steps):
        x,y=mqar_batch(48,16); loss=F.cross_entropy(m(x).reshape(-1,V),y.reshape(-1),ignore_index=-100)
        opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(m.parameters(),1.); opt.step()
    m.eval(); c=t=0
    with torch.no_grad():
        for _ in range(15):
            x,y=mqar_batch(48,16); p=m(x).argmax(-1); msk=y!=-100; c+=(p[msk]==y[msk]).sum().item(); t+=msk.sum().item()
    return c/t

if __name__=="__main__":
    a_ts=speed(lambda: Attn(256)); print(f"attention @64K: {a_ts/1e3:.0f}K tok/s (the baseline to beat)\n",flush=True)
    print(f"{'feat':>4} | {'F dim':>5} | {'recall@16':>9} | {'ours@64K':>9} | {'vs attn':>7}")
    print("-"*48)
    for feat in [2,4,8,16]:
        Fdim=1+feat+feat*feat
        r=recall16(feat)
        ts=speed(lambda f=feat: ChunkedRecall(256,feat=f,chunk=128))
        vs=f"{ts/a_ts:.2f}x" if ts else "OOM"
        print(f"{feat:>4} | {Fdim:>5} | {r*100:8.0f}% | {ts/1e3 if ts else 0:8.0f}K | {vs:>7}",flush=True)
    print("\nWIN = a feat with recall ~100%@16 AND ours@64K > attention (>1.0x). If none -> need delta/lean kernel.")
