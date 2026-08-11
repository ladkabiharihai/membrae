"""SMOKE TEST: can adaptive-compute (weight-tied block + PonderNet halting) PRETRAIN STABLY as an LM on real
text? If ppl drops comparably to a standard fixed-depth LM, adaptive-compute can be integrated into the scale run
(scaffolding-free capacity axis IN training). If it destabilizes / ppl stalls, keep it as a fine-tune-later.
Small + short (runs beside the scale run)."""
import warnings,os; warnings.filterwarnings("ignore"); os.environ["PYTHONWARNINGS"]="ignore"
import sys,time,numpy as np,torch,torch.nn as nn,torch.nn.functional as F
sys.path.insert(0,"/opt/code/membrae")
from fastcore_v import VChunkRecall
from tokenizers import Tokenizer
DEV="cuda"; torch.manual_seed(0)
tok=Tokenizer.from_file("data/bpe.json"); VOCAB=tok.get_vocab_size()
DATA=np.memmap("/mnt/kv_cache/pragnosia_data/window2_train.bin",dtype=np.uint16,mode="r")
CTX=1024
def batch(B):
    ix=np.random.randint(0,len(DATA)-CTX-1,size=B)
    x=torch.from_numpy(np.stack([DATA[i:i+CTX] for i in ix]).astype(np.int64)); y=torch.from_numpy(np.stack([DATA[i+1:i+CTX+1] for i in ix]).astype(np.int64))
    return x.to(DEV),y.to(DEV)
class Blk(nn.Module):
    def __init__(s,d): super().__init__(); s.n1=nn.LayerNorm(d); s.mix=VChunkRecall(d,heads=4,feat=8,chunk=128); s.n2=nn.LayerNorm(d); s.mlp=nn.Sequential(nn.Linear(d,4*d),nn.GELU(),nn.Linear(4*d,d))
    def forward(s,x): x=x+s.mix(s.n1(x)); return x+s.mlp(s.n2(x))
class StdLM(nn.Module):   # fixed depth L
    def __init__(s,d=256,L=4): super().__init__(); s.emb=nn.Embedding(VOCAB,d); s.pos=nn.Embedding(4096,d); s.blocks=nn.ModuleList([Blk(d) for _ in range(L)]); s.head=nn.Linear(d,VOCAB,bias=False); s.head.weight=s.emb.weight
    def forward(s,x):
        h=s.emb(x)+s.pos(torch.arange(x.shape[1],device=x.device)[None])
        for b in s.blocks: h=b(h)
        return s.head(h)
class PonderLM(nn.Module):  # ONE weight-tied block, applied up to nmax times, PonderNet per-token halting
    def __init__(s,d=256,nmax=4,prior=0.5): super().__init__(); s.emb=nn.Embedding(VOCAB,d); s.pos=nn.Embedding(4096,d); s.block=Blk(d); s.halt=nn.Linear(d,1); s.head=nn.Linear(d,VOCAB,bias=False); s.head.weight=s.emb.weight; s.nmax=nmax; s.prior=prior
    def forward_steps(s,x):
        h=s.emb(x)+s.pos(torch.arange(x.shape[1],device=x.device)[None]); B,T,d=h.shape
        still=torch.ones(B,T,device=h.device); ps=[]; lgs=[]
        for n in range(s.nmax):
            h=s.block(h); lam=torch.sigmoid(s.halt(h).squeeze(-1))          # per-token halt prob
            p=still*(lam if n<s.nmax-1 else torch.ones_like(lam)); ps.append(p); lgs.append(s.head(h)); still=still*(1-lam)
        return torch.stack(ps,0),torch.stack(lgs,0)   # [nmax,B,T], [nmax,B,T,V]
    def loss(s,x,y):
        P,LG=s.forward_steps(x); nmax=P.shape[0]
        ce=torch.stack([F.cross_entropy(LG[n].reshape(-1,VOCAB),y.reshape(-1),reduction='none').reshape(y.shape) for n in range(nmax)],0)
        pond=(P*ce).sum(0).mean()
        g=torch.tensor([s.prior*((1-s.prior)**n) for n in range(nmax)],device=x.device); g=g/g.sum()
        kl=(P*((P+1e-8).log()-g[:,None,None].log())).sum(0).mean()
        exp_steps=(P*torch.arange(1,nmax+1,device=x.device)[:,None,None]).sum(0).mean()
        return pond+0.01*kl, exp_steps
def run(kind,steps=700,lr=6e-4):
    torch.manual_seed(0); m=(StdLM(256,4) if kind=="std" else PonderLM(256,nmax=4)).to(DEV)
    opt=torch.optim.AdamW(m.parameters(),lr=lr,betas=(0.9,0.95),weight_decay=0.1); m.train(); t0=time.time()
    for it in range(1,steps+1):
        for gp in opt.param_groups: gp["lr"]=lr*min(1.0,it/200)
        x,y=batch(8)
        if kind=="std": loss=F.cross_entropy(m(x).reshape(-1,VOCAB),y.reshape(-1)); es=torch.tensor(4.0)
        else: loss,es=m.loss(x,y)
        opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(m.parameters(),1.); opt.step()
        if it%100==0: print(f"    {kind:6s} step {it}/{steps} loss={loss.item():.3f} ppl={np.exp(min(20,loss.item())):.0f} exp_steps={float(es):.2f} ({it/(time.time()-t0):.1f} st/s)",flush=True)
    return
if __name__=="__main__":
    print(f"PONDER SMOKE: can weight-tied + halting pretrain stably on real text (ctx={CTX})?\n",flush=True)
    print("  STANDARD fixed-depth-4 LM:",flush=True); run("std")
    print("  PONDER weight-tied + halting LM (exp_steps = model's own chosen depth):",flush=True); run("ponder")
    print("\nREAD: if ponder ppl tracks std ppl -> adaptive-compute pretrains stably -> integrate into scale run.",flush=True)
    print("      if ponder ppl stalls/diverges -> keep as fine-tune-later.",flush=True)
