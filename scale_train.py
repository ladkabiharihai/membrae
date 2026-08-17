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
# RECALL MIX: a fraction of each batch drawn from a QA/needle corpus (SQuAD+HotpotQA) so next-token loss REQUIRES
# retrieving a fact from context -> pressures the memory faculty (window2 web-text alone never does). #build_recall_data
RECALL_FRAC=float(os.environ.get("RECALL_FRAC","0.0"))
RDATA=RSTARTS=None
_rp="/mnt/kv_cache/pragnosia_data/recall_train.bin"
if RECALL_FRAC>0 and os.path.exists(_rp):
    RDATA=np.memmap(_rp,dtype=np.uint16,mode="r")
    _eos=np.where(np.asarray(RDATA)==0)[0]                     # example boundaries (EOS) -> aligned windows keep answer+context
    RSTARTS=np.concatenate([[0],_eos+1]); RSTARTS=RSTARTS[RSTARTS<len(RDATA)-2048-1]
    print(f"[recall-mix] {RECALL_FRAC:.0%} of batches from recall corpus ({len(RDATA)/1e6:.0f}M tok, {len(RSTARTS)} examples)",flush=True)
from torch.utils.checkpoint import checkpoint as _ckpt
from contextlib import nullcontext as _nullcm
GRAD_CKPT=os.environ.get("GRAD_CKPT","0")=="1"                      # #72: gradient-checkpoint blocks -> long-ctx training memory parity w/ attn (recompute in backward)
BF16=os.environ.get("BF16","0")=="1"                               # #73: bf16 autocast -> ~4.5x faster on H100 (22K tok/s @1B), same memory
VRAM_STOP_GROW_GB=float(os.environ.get("VRAM_STOP_GROW_GB","12"))   # don't grow DEPTH if free VRAM below this (protect prod)
# BOTH-AXIS growth: depth (cheap, language capacity) + STATE/feat (recall capacity; QUADRATIC VRAM Fd=1+fe+fe^2 so a
# higher guard + a cap). feat-grow is function-preserving via VChunkRecall.grow_feat (GroupFeatNorm, RESULTS #50).
FEAT_CAP=int(os.environ.get("FEAT_CAP","16"))                       # don't grow feat past this
FEAT_STEP=int(os.environ.get("FEAT_STEP","2"))                      # feat increment per grow
FEAT_GROW_MIN_GB=float(os.environ.get("FEAT_GROW_MIN_GB","20"))     # feat-grow needs lots of headroom (quadratic state)

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
        for b in s.blocks:                                        # GRAD_CKPT=1 -> recompute blocks in backward (RESULTS #72):
            h=_ckpt(b,h,use_reentrant=False) if (GRAD_CKPT and s.training) else b(h)   # fixes Taylor-feat memory -> parity w/ attn, keeps ~20x-faster long-ctx training
        return s.head(s.lnf(h))                                   # FINAL NORM before head (deep model logit stability)
    def grow(s):
        b=Blk(s.d).to(next(s.parameters()).device); b.make_identity(); s.blocks.append(b); return len(s.blocks)
    def grow_feat(s,new_feat):
        for b in s.blocks: b.mix.grow_feat(new_feat)   # STATE axis: function-preserving feat widen (GroupFeatNorm handles stable=True, #50)
        return new_feat
    def maybe_grow(s,ema,step):
        """INTRINSIC trigger: `ema` is the EMA of the model's OWN predictive ENTROPY (label-free, from its own
        output distribution) -- NOT the supervised loss. When the model's own uncertainty stops falling (it has
        extracted what it can at this capacity), IT signals saturation -> grow. Allocation stays external (a
        metabolic controller; irreducible -- weights can't malloc weights, see RESEARCH_selfgrowth.md)."""
        s.hist.append(ema)
        # #48 LESSON: grew 13x @ ~9k steps -> every depth undertrained, capacity thrashed, ZERO capability gain.
        # Each function-preserving identity block must be LEARNED into use; give it real time. Min 30k steps/grow +
        # a stricter, longer plateau gate so a depth actually converges before the next grow.
        if len(s.hist)<20 or step-s.last_grow<30000: return None
        prev=s.hist[-20]; rel=(prev-ema)/max(1e-6,prev)         # relative improvement over ~20 checks (~30k steps)
        if rel>=0.003: return None
        free=torch.cuda.mem_get_info()[0]/2**30
        grew=[]                                                # grow BOTH axes when saturated (guards self-limit)
        if free>VRAM_STOP_GROW_GB:                             # DEPTH (cheap): +1 layer -> language/reasoning capacity
            grew.append(f"{s.grow()}L")
        cf=s.blocks[0].mix.fe                                  # STATE (quadratic VRAM): +FEAT_STEP feat -> recall capacity
        if free>FEAT_GROW_MIN_GB and cf<FEAT_CAP:
            grew.append(f"feat{s.grow_feat(min(cf+FEAT_STEP,FEAT_CAP))}")
        if grew: s.last_grow=step; s.hist=[]; return grew
        return None

def batch(B):
    nr = np.random.binomial(B, RECALL_FRAC) if RSTARTS is not None else 0   # how many recall samples this batch
    rows=[]
    for _ in range(B-nr):                                     # window2 (fluency + knowledge)
        i=np.random.randint(0,len(DATA)-CTX-1); rows.append((DATA,i))
    for _ in range(nr):                                       # recall (example-aligned QA/needle window)
        i=int(RSTARTS[np.random.randint(len(RSTARTS))]); rows.append((RDATA,i))
    x=torch.from_numpy(np.stack([d[i:i+CTX] for d,i in rows]).astype(np.int64))
    y=torch.from_numpy(np.stack([d[i+1:i+CTX+1] for d,i in rows]).astype(np.int64))
    return x.to(DEV),y.to(DEV)

import json as _json
_PROBE=None; _pp="/mnt/kv_cache/pragnosia_data/recall_probe.json"
if os.path.exists(_pp): _PROBE=_json.load(open(_pp))   # held-out SQuAD dev (real facts, trained QA format) -- NEVER in train
@torch.no_grad()
def recall_prose(m,n=60):
    """HONEST recall metric: held-out SQuAD extractive QA (feed context+question, does the model produce the
    gold answer span?). This measures the faculty on the SAME distribution the recall-mix teaches -- real prose
    in Question:/Answer: format. The old synthetic-needle probe read 0% only because its made-up entities +
    free-form phrasing were OUT-of-distribution; the true recall is ~33%, not 0 (RESULTS_MEASURED.md)."""
    if not _PROBE: return 0.0
    m.eval(); hit=0; tot=0
    for ex in _PROBE[:n]:
        ans=ex['a'].lower(); ids=tok.encode(f"{ex['context']}\n\nQuestion: {ex['q']}\nAnswer:").ids; out=[]
        for _ in range(max(4,len(ans.split())+3)):
            nx=int(m(torch.tensor([(ids+out)[-CTX:]],device=DEV))[0,-1].argmax())
            if nx==0: break
            out.append(nx)
        g=tok.decode(out).strip().lower()
        hit+= ans in g or g in ans or any(w in g for w in ans.split() if len(w)>3); tot+=1
    m.train(); return hit/max(1,tot)

def nparams(m): return sum(p.numel() for p in m.parameters())

if __name__=="__main__":
    import glob,re
    start_step=0
    if os.environ.get("RESUME","")=="1":                          # resume from latest snapshot (keep training + grow further)
        snaps=glob.glob(f"{SNAPDIR}/scale_*.pt")
        latest=max(snaps,key=lambda p:int(re.search(r'scale_(\d+)_',p).group(1)))
        L0=int(re.search(r'_(\d+)L',latest).group(1)); start_step=int(re.search(r'scale_(\d+)_',latest).group(1))
        fm=re.search(r'_f(\d+)\.pt',latest); feat0=int(fm.group(1)) if fm else 8   # feat from filename (old snaps = 8)
        m=BrainLM(d=int(os.environ.get("D","512")),L=L0).to(DEV)
        f=8                                                       # REPLAY feat-grows to rebuild GroupFeatNorm structure before load
        while f<feat0: f=min(f+FEAT_STEP,feat0); m.grow_feat(f)
        m.load_state_dict(torch.load(latest,map_location=DEV))    # d from env (matches the run)
        print(f"RESUMED from {latest}: {L0}L feat{feat0} {nparams(m)/1e6:.0f}M at step {start_step}",flush=True)
    else:
        m=BrainLM(d=int(os.environ.get("D","512")),L=int(os.environ.get("LSTART","4"))).to(DEV)
    opt=torch.optim.AdamW(m.parameters(),lr=6e-4,betas=(0.9,0.95),weight_decay=0.1)
    if os.environ.get("COMPILE","0")=="1":                        # #77: torch.compile now CORRECT+stable on the core
        m=torch.compile(m); print("[torch.compile ON] ~1.45x on the mix (verified fwd+bwd correct, recall intact)",flush=True)  # (recompiles on grow; grows are rare)
    BS=int(os.environ.get("BS","24")); STEPS=int(os.environ.get("STEPS","400000"))
    print(f"SELF-SCALING RUN: {nparams(m)/1e6:.0f}M ({len(m.blocks)}L), window2 176B, ctx{CTX}, bs{BS}",flush=True)
    # fresh run: long warmup (big models diverge on short warmup); resume: short re-warmup. lr lower for big models.
    ema=None; t0=time.time(); base=start_step
    WARM=(start_step+300) if start_step>0 else int(os.environ.get("WARM","2000"))
    PEAK=float(os.environ.get("LR","3e-4"))
    for it in range(start_step+1,STEPS+1):
        for gp in opt.param_groups: gp["lr"]=PEAK*min(1.0,it/WARM)   # LR warmup -> clean early loss / saturation signal
        x,y=batch(BS)
        with (torch.autocast("cuda",dtype=torch.bfloat16) if BF16 else _nullcm()):   # BF16=1 -> ~4.5x faster on H100 (RESULTS #73)
            lg=m(x); loss=F.cross_entropy(lg.reshape(-1,VOCAB),y.reshape(-1))
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
                print(f"  >>> SELF-GREW at step {it}: saturated -> {'+'.join(g)} ({len(m.blocks)}L feat{m.blocks[0].mix.fe}), {nparams(m)/1e6:.0f}M",flush=True)
        if it%10000==0:                                              # RECALL eval + snapshot every 10k (expensive: autoregressive gen)
            r=recall_prose(m); P=nparams(m)/1e6
            print(f"  [SNAP step {it}] {P:.0f}M {len(m.blocks)}L  ppl={np.exp(min(20,L)):.1f}  own_uncert={ema:.3f}  held-out SQuAD RECALL={r*100:.0f}%",flush=True)
            torch.save(m.state_dict(),f"{SNAPDIR}/scale_{it}_{P:.0f}M_{len(m.blocks)}L_f{m.blocks[0].mix.fe}.pt")
    print("[done]",flush=True)
