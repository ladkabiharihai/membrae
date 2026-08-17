"""PROOF: a STATE-axis growth operator that rescues recall where depth-growth cannot (closes RESULTS #48/#49).
grow_feat_ widens the Taylor-feature dim fe->fe' FUNCTION-PRESERVINGLY: the NEW KEY feature rows are zero-init
(weight+bias) so the key's new dims are exactly 0 -> every attention-score Taylor term touching a new dim is 0 ->
output identical at t=0. NEW QUERY rows are small-random so symmetry is broken and the new key dims get gradient and
train up. v/o are untouched (independent of fe). stable=False (LayerNorm-on-feat would break preservation when fe
grows; that's a documented follow-up for the live stable=True core).

Rigorous 5-arm design on MQAR-16 (K=64,V=64), d=128, using the reference untied-head harness (fastcore_v.mqar):
  base       feat=2, 2500 steps                          -> bottlenecked baseline (~24% per #49)
  PRESERVE   grow_feat 2->16, measure recall JUST BEFORE vs JUST AFTER  -> must be ~identical (function-preserving)
  STATE-GROW ...then +2500 steps                          -> should climb toward ~100% (native feat=16 = 99.5%)
  DEPTH-GROW feat=2, +identity layer at 2500, +2500 steps -> CONTROL, same opt-reset + added params, other axis
  NO-GROW    feat=2, 5000 steps straight                  -> CONTROL, "more steps alone" does not rescue
  OPT-RESET  feat=2, 2500 + fresh optimizer + 2500        -> CONTROL, optimizer reset alone does not rescue
The airtight comparison is STATE-GROW vs DEPTH-GROW: both reset the optimizer and add params at the same step; the
ONLY difference is the axis. If state->~100% and depth->~24%, recall capacity lives on the STATE axis. QED.
"""
import warnings,os; warnings.filterwarnings("ignore"); os.environ["PYTHONWARNINGS"]="ignore"
import sys; sys.path.insert(0,"/opt/code/membrae")
import torch, torch.nn as nn, torch.nn.functional as F
from fastcore_v import VChunkRecall, mqar
DEV="cuda"; V=129; d=128

def grow_feat_(mix, new_feat):
    """Function-preserving STATE-axis growth of a VChunkRecall (stable=False). Mutates mix in place."""
    H, fe = mix.H, mix.fe; din = mix.q.in_features
    dev = mix.q.weight.device; dt = mix.q.weight.dtype
    def expand(lin, rand_new):
        ow = lin.weight.data.view(H, fe, din); ob = lin.bias.data.view(H, fe)   # head-major: row = h*fe+f
        nl = nn.Linear(din, H*new_feat).to(dev, dt)
        nw = nl.weight.data.view(H, new_feat, din); nb = nl.bias.data.view(H, new_feat)
        nw.zero_(); nb.zero_()
        nw[:, :fe, :] = ow; nb[:, :fe] = ob                                     # keep old features EXACTLY
        if rand_new: nw[:, fe:, :].normal_(0, 0.02)                             # new query rows: break symmetry
        return nl
    mix.k = expand(mix.k, rand_new=False)   # new KEY dims = 0 -> zero contribution -> output preserved
    mix.q = expand(mix.q, rand_new=True)    # new QUERY dims small-random -> key-new-dims receive gradient
    mix.fe = new_feat

class Blk(nn.Module):
    def __init__(s,d,feat):
        super().__init__(); s.n1=nn.LayerNorm(d); s.mix=VChunkRecall(d,heads=4,feat=feat,chunk=64,stable=False)
        s.n2=nn.LayerNorm(d); s.mlp=nn.Sequential(nn.Linear(d,4*d),nn.GELU(),nn.Linear(4*d,d))
    def make_identity(s): [nn.init.zeros_(p) for p in (s.mix.o.weight,s.mix.o.bias,s.mlp[-1].weight,s.mlp[-1].bias)]
    def forward(s,x): x=x+s.mix(s.n1(x)); return x+s.mlp(s.n2(x))
class LM(nn.Module):
    def __init__(s,V,d,L,feat):
        super().__init__(); s.emb=nn.Embedding(V,d); s.pos=nn.Embedding(4096,d)
        s.blocks=nn.ModuleList([Blk(d,feat) for _ in range(L)]); s.head=nn.Linear(d,V)
    def forward(s,x):
        h=s.emb(x)+s.pos(torch.arange(x.shape[1],device=x.device))
        for b in s.blocks: h=b(h)
        return s.head(h)
    def grow_feat(s,nf): [grow_feat_(b.mix,nf) for b in s.blocks]
    def grow_depth(s):
        b=Blk(d, s.blocks[0].mix.fe).to(next(s.parameters()).device); b.make_identity(); s.blocks.append(b)

def train(m,steps,lr=2e-3,n=16):
    opt=torch.optim.AdamW(m.parameters(),lr=lr); m.train()
    for _ in range(steps):
        x,y=mqar(48,n); F.cross_entropy(m(x).reshape(-1,V),y.reshape(-1),ignore_index=-100).backward(); opt.step(); opt.zero_grad()
@torch.no_grad()
def evalr(m,n=16,reps=30):
    m.eval(); c=t=0
    for _ in range(reps):
        x,y=mqar(48,n); p=m(x).argmax(-1); msk=y!=-100; c+=(p[msk]==y[msk]).sum().item(); t+=msk.sum().item()
    m.train(); return 100.0*c/t

if __name__=="__main__":
    R={}
    torch.manual_seed(0)
    print("[base] feat=2, 2500 steps...",flush=True)
    m=LM(V,d,2,2).to(DEV); train(m,2500); R["base feat=2"]=evalr(m)
    print(f"  base recall={R['base feat=2']:.1f}%",flush=True)
    before=evalr(m); m.grow_feat(16); after=evalr(m)          # FUNCTION-PRESERVATION check
    R["preserve before->after"]=(before,after)
    print(f"[preserve] before grow={before:.1f}%  after grow={after:.1f}%  (delta {after-before:+.1f})",flush=True)
    train(m,2500); R["STATE-GROW 2->16"]=evalr(m)
    print(f"[STATE-GROW] recall={R['STATE-GROW 2->16']:.1f}%",flush=True)

    torch.manual_seed(0); print("[depth-grow control] feat=2 +layer...",flush=True)
    mB=LM(V,d,2,2).to(DEV); train(mB,2500); mB.grow_depth(); train(mB,2500); R["DEPTH-GROW +layer"]=evalr(mB)
    print(f"[DEPTH-GROW] recall={R['DEPTH-GROW +layer']:.1f}%",flush=True)

    torch.manual_seed(0); print("[no-grow control] feat=2 x5000...",flush=True)
    mC=LM(V,d,2,2).to(DEV); train(mC,5000); R["NO-GROW feat=2 x5000"]=evalr(mC)
    print(f"[NO-GROW] recall={R['NO-GROW feat=2 x5000']:.1f}%",flush=True)

    torch.manual_seed(0); print("[opt-reset control] feat=2, reset opt, no grow...",flush=True)
    mD=LM(V,d,2,2).to(DEV); train(mD,2500); train(mD,2500); R["OPT-RESET only"]=evalr(mD)
    print(f"[OPT-RESET] recall={R['OPT-RESET only']:.1f}%",flush=True)

    print("\n==================== STATE-AXIS GROWTH PROOF ====================",flush=True)
    print(f"  baseline feat=2 (2500 steps)      : {R['base feat=2']:.1f}%",flush=True)
    print(f"  function-preservation (grow 2->16): {R['preserve before->after'][0]:.1f}% -> {R['preserve before->after'][1]:.1f}%  (must be ~equal)",flush=True)
    print(f"  STATE-GROW 2->16 then +2500       : {R['STATE-GROW 2->16']:.1f}%   <-- rescued?",flush=True)
    print(f"  DEPTH-GROW +layer then +2500      : {R['DEPTH-GROW +layer']:.1f}%   (control: other axis)",flush=True)
    print(f"  NO-GROW feat=2 x5000              : {R['NO-GROW feat=2 x5000']:.1f}%   (control: more steps)",flush=True)
    print(f"  OPT-RESET only                    : {R['OPT-RESET only']:.1f}%   (control: reset alone)",flush=True)
    print("VERDICT: if STATE-GROW >> DEPTH-GROW ~ NO-GROW ~ OPT-RESET ~ baseline, and preserve-delta ~0,",flush=True)
    print("         then a function-preserving STATE-axis grow is the recall lever depth-growth is not. QED.",flush=True)
