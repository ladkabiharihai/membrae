"""SOLVE the training-MEMORY bottleneck (the 'space' inefficiency #training_efficiency found: brain uses 1.4x
attention's training memory because the Taylor feature Fd=73 blows up activations). FIX = gradient CHECKPOINTING:
recompute each block in backward instead of storing its activations -- trades a little recompute for big memory
savings, and the chunked recurrent block is ideal for it. Measure training mem+time: attention vs brain vs
brain+checkpoint, pushing context to where un-checkpointed OOMs. If brain+ckpt trains longer/cheaper (memory) than
attention AND keeps the linear-time win -> training efficiency SOLVED -> fixed budget trains larger models.
"""
import warnings,os,time; warnings.filterwarnings("ignore"); os.environ["PYTHONWARNINGS"]="ignore"
import sys,torch,torch.nn as nn,torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
sys.path.insert(0,"/opt/code/membrae")
from fastcore_v import VChunkRecall
DEV="cuda"; d=512; H=8; L=6; V=4096
class Attn(nn.Module):
    def __init__(s): super().__init__(); s.qkv=nn.Linear(d,3*d); s.o=nn.Linear(d,d)
    def forward(s,x):
        B,T,_=x.shape; qkv=s.qkv(x).view(B,T,3,H,d//H).permute(2,0,3,1,4); q,k,v=qkv[0],qkv[1],qkv[2]
        return s.o(F.scaled_dot_product_attention(q,k,v,is_causal=True).transpose(1,2).reshape(B,T,d))
class Blk(nn.Module):
    def __init__(s,mk): super().__init__(); s.n1=nn.LayerNorm(d); s.mix=mk(); s.n2=nn.LayerNorm(d); s.mlp=nn.Sequential(nn.Linear(d,4*d),nn.GELU(),nn.Linear(4*d,d))
    def forward(s,x): x=x+s.mix(s.n1(x)); return x+s.mlp(s.n2(x))
class Net(nn.Module):
    def __init__(s,mk,ckpt=False): super().__init__(); s.emb=nn.Embedding(V,d); s.blocks=nn.ModuleList([Blk(mk) for _ in range(L)]); s.head=nn.Linear(d,V); s.ckpt=ckpt
    def forward(s,x):
        h=s.emb(x)
        for b in s.blocks:
            h=checkpoint(b,h,use_reentrant=False) if s.ckpt else b(h)
        return s.head(h)
def train_step(mk,T,ckpt=False,B=1):
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    try:
        m=Net(mk,ckpt).to(DEV); opt=torch.optim.AdamW(m.parameters(),lr=1e-4)
        x=torch.randint(0,V,(B,T),device=DEV); y=torch.randint(0,V,(B,T),device=DEV)
        def step():
            opt.zero_grad(); loss=F.cross_entropy(m(x).reshape(-1,V),y.reshape(-1)); loss.backward(); opt.step()
        step(); torch.cuda.synchronize(); t0=time.time()
        for _ in range(3): step()
        torch.cuda.synchronize(); dt=(time.time()-t0)/3
        peak=torch.cuda.max_memory_allocated()/2**20; del m,opt,x,y; return peak,dt
    except RuntimeError as e:
        if "out of memory" in str(e).lower(): torch.cuda.empty_cache(); return None,None
        raise
if __name__=="__main__":
    brain=lambda:VChunkRecall(d,heads=H,feat=8,chunk=128,stable=True); attn=lambda:Attn()
    print(f"TRAINING MEMORY+TIME with gradient CHECKPOINTING -- solving the 'space' bottleneck. d={d} L={L} batch1.\n",flush=True)
    print(f"  {'ctx':>7} | {'attention':>16} | {'brain (no ckpt)':>18} | {'brain +CKPT':>18} | {'brain+ckpt vs attn':>20}",flush=True)
    print("  "+"-"*90,flush=True)
    for T in [4096,16384,32768,65536,131072]:
        am,at=train_step(attn,T); bm,bt=train_step(brain,T,ckpt=False); cm,ct=train_step(brain,T,ckpt=True)
        f=lambda mm,tt: (f"{mm:.0f}MB {tt*1e3:.0f}ms" if mm else "OOM")
        win=""
        if cm and am: win=f"{am/cm:.1f}x mem {at/ct:.1f}x time"
        elif cm and not am: win="attn OOM; BRAIN OK"
        print(f"  {T:>7} | {f(am,at):>16} | {f(bm,bt):>18} | {f(cm,ct):>18} | {win:>20}",flush=True)
    print("\n  If brain+CKPT memory < attention AND still faster -> training efficiency SOLVED (power+space).",flush=True)
    print("  Then a fixed GPU budget trains LARGER models / LONGER context on the brain than on a transformer. => the unlock.",flush=True)
    print("DONE",flush=True)
