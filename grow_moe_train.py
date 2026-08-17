"""GROW-MoE training: a brain-core LM whose MLPs are FastMoE (top-1 experts). It GROWS by adding EXPERTS on its own
entropy saturation -> total params (capacity) scale while ACTIVE compute/token stays ~constant (the 'as large as
possible without more compute' thesis). bf16 + gradient-checkpointing (fits + fast). window2 176B + recall mix.
Snapshots + resume. Faculties intact (recall in the shared VChunkRecall mix). Fast dispatch (#fast_moe).
"""
import warnings,os; warnings.filterwarnings("ignore"); os.environ["PYTHONWARNINGS"]="ignore"
import sys,time,glob,re,numpy as np,torch,torch.nn as nn,torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
sys.path.insert(0,"/opt/code/membrae")
from fastcore_v import VChunkRecall
from fast_moe import FastMoE
from tokenizers import Tokenizer
DEV="cuda"; torch.manual_seed(0)
tok=Tokenizer.from_file("data/bpe.json"); VOCAB=tok.get_vocab_size()
DATA=np.memmap("/mnt/kv_cache/pragnosia_data/window2_train.bin",dtype=np.uint16,mode="r")
CTX=2048; SNAP="/mnt/kv_cache/pragnosia_data/snaps"; d=int(os.environ.get("D","1024")); H=8
L=int(os.environ.get("L","14")); E0=int(os.environ.get("E0","4")); EMAX=int(os.environ.get("EMAX","16"))
BS=int(os.environ.get("BS","4")); STEPS=int(os.environ.get("STEPS","100000000")); GROW_EVERY=int(os.environ.get("GROW_EVERY","30000"))
RECALL_FRAC=float(os.environ.get("RECALL_FRAC","0.15")); VRAM_STOP=float(os.environ.get("VRAM_STOP_GB","14"))
RDATA=RSTARTS=None; _rp="/mnt/kv_cache/pragnosia_data/recall_train.bin"
if RECALL_FRAC>0 and os.path.exists(_rp):
    RDATA=np.memmap(_rp,dtype=np.uint16,mode="r"); _eos=np.where(np.asarray(RDATA)==0)[0]
    RSTARTS=np.concatenate([[0],_eos+1]); RSTARTS=RSTARTS[RSTARTS<len(RDATA)-CTX-1]
class Blk(nn.Module):
    def __init__(s,E): super().__init__(); s.n1=nn.LayerNorm(d); s.mix=VChunkRecall(d,heads=H,feat=8,chunk=128,stable=True); s.n2=nn.LayerNorm(d); s.moe=FastMoE(d,E)
    def forward(s,x): x=x+s.mix(s.n1(x)); return x+s.moe(s.n2(x))
class MoELM(nn.Module):
    def __init__(s,E):
        super().__init__(); s.emb=nn.Embedding(VOCAB,d); s.pos=nn.Embedding(8192,d)
        s.blocks=nn.ModuleList([Blk(E) for _ in range(L)]); s.lnf=nn.LayerNorm(d); s.head=nn.Linear(d,VOCAB,bias=False); s.head.weight=s.emb.weight
        nn.init.normal_(s.emb.weight,0,0.02); nn.init.normal_(s.pos.weight,0,0.02); s.hist=[]; s.last_grow=0
    def forward(s,x):
        x=x.clamp(0,VOCAB-1); h=s.emb(x)+s.pos(torch.arange(x.shape[1],device=x.device)[None])
        for b in s.blocks: h=checkpoint(b,h,use_reentrant=False) if s.training else b(h)
        return s.head(s.lnf(h))
    def grow_experts(s): [b.moe.grow() for b in s.blocks]; return s.blocks[0].moe.E
    def maybe_grow(s,ema,step):
        s.hist.append(ema)
        if len(s.hist)<20 or step-s.last_grow<GROW_EVERY: return None
        rel=(s.hist[-20]-ema)/max(1e-6,s.hist[-20]); free=torch.cuda.mem_get_info()[0]/2**30
        if rel<0.003 and free>VRAM_STOP and s.blocks[0].moe.E<EMAX:
            E=s.grow_experts(); s.last_grow=step; s.hist=[]; return E
        return None
def nact(m):
    tot=sum(p.numel() for p in m.parameters()); oneexp=(m.blocks[0].moe.w1[0].numel()+m.blocks[0].moe.w2[0].numel()+m.blocks[0].moe.b1[0].numel()+m.blocks[0].moe.b2[0].numel())
    return tot, tot-sum((b.moe.E-1)*oneexp for b in m.blocks)
def batch(B):
    nr=np.random.binomial(B,RECALL_FRAC) if RSTARTS is not None else 0; rows=[]
    for _ in range(B-nr): i=np.random.randint(0,len(DATA)-CTX-1); rows.append((DATA,i))
    for _ in range(nr): rows.append((RDATA,int(RSTARTS[np.random.randint(len(RSTARTS))])))
    x=np.stack([a[i:i+CTX] for a,i in rows]).astype(np.int64); y=np.stack([a[i+1:i+CTX+1] for a,i in rows]).astype(np.int64)
    return torch.from_numpy(x).to(DEV),torch.from_numpy(y).to(DEV)
if __name__=="__main__":
    start=0; m=MoELM(E0).to(DEV)
    sn=glob.glob(f"{SNAP}/moe_*.pt")
    if os.environ.get("RESUME")=="1" and sn:
        latest=max(sn,key=lambda p:int(re.search(r'moe_(\d+)_',p).group(1))); start=int(re.search(r'moe_(\d+)_',latest).group(1))
        Es=int(re.search(r'_E(\d+)\.pt',latest).group(1))
        while m.blocks[0].moe.E<Es: m.grow_experts()
        m.load_state_dict(torch.load(latest,map_location=DEV)); print(f"RESUMED {latest} E={Es} step{start}",flush=True)
    opt=torch.optim.AdamW(m.parameters(),lr=3e-4,betas=(0.9,0.95),weight_decay=0.1)
    tt,ac=nact(m); print(f"GROW-MoE: {tt/1e6:.0f}M total / {ac/1e6:.0f}M active, {L}L E={m.blocks[0].moe.E}->{EMAX}, bf16+ckpt, ctx{CTX}",flush=True)
    if RSTARTS is not None: print(f"[recall-mix {RECALL_FRAC:.0%}]",flush=True)
    ema=None; t0=time.time(); base=start; WARM=(start+300) if start else 2000
    for it in range(start+1,STEPS+1):
        for g in opt.param_groups: g["lr"]=3e-4*min(1.0,it/WARM)
        x,y=batch(BS)
        with torch.autocast("cuda",dtype=torch.bfloat16): lg=m(x); loss=F.cross_entropy(lg.reshape(-1,VOCAB),y.reshape(-1))
        if not torch.isfinite(loss): opt.zero_grad(set_to_none=True); continue
        with torch.no_grad(): pp=lg.float().softmax(-1); ent=(-(pp*(pp+1e-9).log()).sum(-1)).mean().item()
        opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(m.parameters(),1.); opt.step()
        if it<=WARM:
            if it%200==0: print(f"  [warmup {it}/{WARM}] loss={loss.item():.3f}",flush=True)
            continue
        ema=ent if ema is None else 0.995*ema+0.005*ent
        if it%500==0:
            tt,ac=nact(m); print(f"  step {it} loss={loss.item():.3f} ppl={np.exp(min(20,loss.item())):.1f} uncert={ema:.3f} E={m.blocks[0].moe.E} {tt/1e6:.0f}M/{ac/1e6:.0f}Mact {BS*CTX*(it-base)/(time.time()-t0)/1e3:.0f}Kt/s free={torch.cuda.mem_get_info()[0]/2**30:.0f}G",flush=True)
        if it%1500==0:
            g=m.maybe_grow(ema,it)
            if g: opt=torch.optim.AdamW(m.parameters(),lr=3e-4,betas=(0.9,0.95),weight_decay=0.1); base=it; t0=time.time(); print(f"  >>> GREW EXPERTS -> E={g} (constant active-compute)",flush=True)
        if it%10000==0:
            tt,ac=nact(m); torch.save(m.state_dict(),f"{SNAP}/moe_{it}_{tt/1e6:.0f}M_E{m.blocks[0].moe.E}.pt"); print(f"  [SNAP {it}] {tt/1e6:.0f}M/{ac/1e6:.0f}Mact E={m.blocks[0].moe.E}",flush=True)
    print("[done]",flush=True)
