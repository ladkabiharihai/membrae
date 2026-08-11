"""SELF-SCALING LANGUAGE RUN: start ~20M on the intrinsic-brain fast core (VChunkRecall), train on the 176B
window2 corpus (real diverse text), and let the model GROW ITSELF when its own loss saturates (intrinsic
depth-grow, function-preserving). Save SNAPSHOTS at each size + a light in-prose RECALL eval, so we can watch
whether the memory faculty EMERGES on language as it scales. prod-safe (VRAM cap, eager). Resumable-ish via snaps.
"""
import warnings,os; warnings.filterwarnings("ignore"); os.environ["PYTHONWARNINGS"]="ignore"
import sys,time,random,numpy as np,torch,torch.nn as nn,torch.nn.functional as F
sys.path.insert(0,"/opt/code/membrae")
from fastcore_v import VChunkRecall
from tokenizers import Tokenizer
DEV="cuda"; torch.manual_seed(0); random.seed(0)
tok=Tokenizer.from_file("data/bpe.json"); VOCAB=tok.get_vocab_size()
DATA=np.memmap("/mnt/kv_cache/pragnosia_data/window2_train.bin",dtype=np.uint16,mode="r")
CTX=2048; SNAPDIR="/mnt/kv_cache/pragnosia_data/snaps"; os.makedirs(SNAPDIR,exist_ok=True)
VRAM_STOP_GROW_GB=float(os.environ.get("VRAM_STOP_GROW_GB","12"))   # don't grow if free VRAM below this (protect prod)

class Blk(nn.Module):
    def __init__(s,d): super().__init__(); s.n1=nn.LayerNorm(d); s.mix=VChunkRecall(d,heads=8,feat=8,chunk=128,stable=True); s.n2=nn.LayerNorm(d); s.mlp=nn.Sequential(nn.Linear(d,4*d),nn.GELU(),nn.Linear(4*d,d))
    def forward(s,x): x=x+s.mix(s.n1(x)); return x+s.mlp(s.n2(x))
    def make_identity(s): nn.init.zeros_(s.mix.o.weight); nn.init.zeros_(s.mix.o.bias); nn.init.zeros_(s.mlp[-1].weight); nn.init.zeros_(s.mlp[-1].bias)
class BrainLM(nn.Module):
    def __init__(s,d=512,L=4):
        super().__init__(); s.d=d; s.emb=nn.Embedding(VOCAB,d); s.pos=nn.Embedding(8192,d)
        s.blocks=nn.ModuleList([Blk(d) for _ in range(L)]); s.lnf=nn.LayerNorm(d); s.head=nn.Linear(d,VOCAB,bias=False); s.head.weight=s.emb.weight
        nn.init.normal_(s.emb.weight,0.0,0.02); nn.init.normal_(s.pos.weight,0.0,0.02)   # GPT-style init (tied head inherits emb)
        # intrinsic saturation sensor
        s.hist=[]; s.last_grow=0
    def forward(s,x):
        x=x.clamp(0,VOCAB-1); h=s.emb(x)+s.pos(torch.arange(x.shape[1],device=x.device)[None])
        for b in s.blocks: h=b(h)
        return s.head(s.lnf(h))                                   # FINAL NORM before head (deep model logit stability)
    def grow(s):
        b=Blk(s.d).to(next(s.parameters()).device); b.make_identity(); s.blocks.append(b); return len(s.blocks)
    def maybe_grow(s,ema,step):
        """INTRINSIC trigger: `ema` is the EMA of the model's OWN predictive ENTROPY (label-free, from its own
        output distribution) -- NOT the supervised loss. When the model's own uncertainty stops falling (it has
        extracted what it can at this capacity), IT signals saturation -> grow. Allocation stays external (a
        metabolic controller; irreducible -- weights can't malloc weights, see RESEARCH_selfgrowth.md)."""
        s.hist.append(ema)
        if len(s.hist)<6 or step-s.last_grow<4000: return None
        prev=s.hist[-6]; rel=(prev-ema)/max(1e-6,prev)          # relative improvement over ~6 checks
        free=torch.cuda.mem_get_info()[0]/2**30
        if rel<0.005 and free>VRAM_STOP_GROW_GB:
            n=s.grow(); s.last_grow=step; s.hist=[]; return n
        return None

def batch(B):
    ix=np.random.randint(0,len(DATA)-CTX-1,size=B)
    x=torch.from_numpy(np.stack([DATA[i:i+CTX] for i in ix]).astype(np.int64))
    y=torch.from_numpy(np.stack([DATA[i+1:i+CTX+1] for i in ix]).astype(np.int64))
    return x.to(DEV),y.to(DEV)

@torch.no_grad()
def recall_prose(m,n=40):
    m.eval(); names=["Dr. Rowan","Mr. Alvarez","Captain Vance","Ms. Okafor","Professor Lin","Sergeant Boyd","Elena","Marcus"]
    objs=["a brass key","a red notebook","an old compass","a silver coin","a folded map","a glass vial"]
    fill=("The afternoon was quiet and the corridor smelled faintly of rain. A clock ticked while papers "
          "rustled on the desk. Outside, the market carried on with its usual murmur and clatter. ")
    hit=0
    for _ in range(n):
        who=random.choice(names); what=random.choice(objs)
        pr=f"{who} kept {what} in the top drawer. "+fill*random.randint(2,5)+f"The item that {who} kept in the drawer was"
        ids=tok.encode(pr).ids; out=[]
        for _ in range(6):
            lg=m(torch.tensor([(ids+out)[-CTX:]],device=DEV))[0,-1]; out.append(int(lg.argmax()))
        hit+= what.split()[-1] in tok.decode(out).lower()
    m.train(); return hit/n

def nparams(m): return sum(p.numel() for p in m.parameters())

if __name__=="__main__":
    import glob,re
    start_step=0
    if os.environ.get("RESUME","")=="1":                          # resume from latest snapshot (keep training + grow further)
        snaps=glob.glob(f"{SNAPDIR}/scale_*.pt")
        latest=max(snaps,key=lambda p:int(re.search(r'scale_(\d+)_',p).group(1)))
        L0=int(re.search(r'_(\d+)L\.pt',latest).group(1)); start_step=int(re.search(r'scale_(\d+)_',latest).group(1))
        m=BrainLM(d=512,L=L0).to(DEV); m.load_state_dict(torch.load(latest,map_location=DEV))
        print(f"RESUMED from {latest}: {L0}L {nparams(m)/1e6:.0f}M at step {start_step}",flush=True)
    else:
        m=BrainLM(d=int(os.environ.get("D","512")),L=int(os.environ.get("LSTART","4"))).to(DEV)
    opt=torch.optim.AdamW(m.parameters(),lr=6e-4,betas=(0.9,0.95),weight_decay=0.1)
    BS=int(os.environ.get("BS","24")); STEPS=int(os.environ.get("STEPS","400000"))
    print(f"SELF-SCALING RUN: {nparams(m)/1e6:.0f}M ({len(m.blocks)}L), window2 176B, ctx{CTX}, bs{BS}",flush=True)
    # fresh run: long warmup (big models diverge on short warmup); resume: short re-warmup. lr lower for big models.
    ema=None; t0=time.time(); base=start_step
    WARM=(start_step+300) if start_step>0 else int(os.environ.get("WARM","2000"))
    PEAK=float(os.environ.get("LR","3e-4"))
    for it in range(start_step+1,STEPS+1):
        for gp in opt.param_groups: gp["lr"]=PEAK*min(1.0,it/WARM)   # LR warmup -> clean early loss / saturation signal
        x,y=batch(BS); lg=m(x); loss=F.cross_entropy(lg.reshape(-1,VOCAB),y.reshape(-1))
        if not torch.isfinite(loss): opt.zero_grad(set_to_none=True); continue
        with torch.no_grad():                                        # the model's OWN predictive uncertainty
            pp=lg.softmax(-1); ent=(-(pp*(pp+1e-9).log()).sum(-1)).mean().item()   # entropy, LABEL-FREE
        opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(m.parameters(),1.); opt.step()
        if it%200==0 and it<=WARM: print(f"  [warmup {it}/{WARM}] loss={loss.item():.3f} ppl={np.exp(min(20,loss.item())):.0f}",flush=True)
        if it<=WARM: continue                                        # don't feed warmup steps into the sensor
        L=loss.item(); ema=ent if ema is None else 0.995*ema+0.005*ent   # EMA of the model's OWN entropy (intrinsic)
        if it%500==0:
            tps=BS*CTX*(it-base)/(time.time()-t0)
            print(f"  step {it} loss={L:.3f} ppl={np.exp(min(20,L)):.1f} own_uncert={ema:.3f} {len(m.blocks)}L {nparams(m)/1e6:.0f}M {tps/1e3:.0f}Ktok/s free={torch.cuda.mem_get_info()[0]/2**30:.0f}G",flush=True)
        if it%1500==0 and os.environ.get("NOGROW","0")!="1":         # NOGROW=1 -> fixed size, no self-grow
            g=m.maybe_grow(ema,it)
            if g:
                opt=torch.optim.AdamW(m.parameters(),lr=6e-4,betas=(0.9,0.95),weight_decay=0.1); base=it; t0=time.time()
                print(f"  >>> SELF-GREW at step {it}: saturated -> {g}L, {nparams(m)/1e6:.0f}M",flush=True)
        if it%10000==0:                                              # RECALL eval + snapshot every 10k (expensive: autoregressive gen)
            r=recall_prose(m); P=nparams(m)/1e6
            print(f"  [SNAP step {it}] {P:.0f}M {len(m.blocks)}L  ppl={np.exp(min(20,L)):.1f}  own_uncert={ema:.3f}  in-prose RECALL={r*100:.0f}%",flush=True)
            torch.save(m.state_dict(),f"{SNAPDIR}/scale_{it}_{P:.0f}M_{len(m.blocks)}L.pt")
    print("[done]",flush=True)
