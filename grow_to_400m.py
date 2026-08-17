"""Manually grow the 18L/feat8/216M checkpoint to ~400M FUNCTION-PRESERVINGLY on both axes (depth + state/feat),
save it as a new checkpoint to train from. Depth carries the param count (~36L); feat->12 adds recall-state capacity.
At t=0 the 400M model outputs are IDENTICAL to the 216M (new layers identity, new feat dims zero-contribution), so
NO capability is lost -- it just has room to grow into with sustained training. VRAM-tested before saving.
"""
import warnings,os; warnings.filterwarnings("ignore"); os.environ["PYTHONWARNINGS"]="ignore"; os.environ.setdefault("RECALL_FRAC","0")
import torch, torch.nn.functional as F, scale_train as S
DEV="cuda"; TARGET=float(os.environ.get("TARGET_M","400"))*1e6; FEAT=int(os.environ.get("TGT_FEAT","12"))
CK="/mnt/kv_cache/pragnosia_data/snaps/scale_490000_216M_18L.pt"
m=S.BrainLM(d=1024,L=18).to(DEV); m.load_state_dict(torch.load(CK,map_location=DEV))
def npar(): return sum(p.numel() for p in m.parameters())
print(f"start: {npar()/1e6:.0f}M {len(m.blocks)}L feat{m.blocks[0].mix.fe}",flush=True)

# function-preservation check (before/after the whole grow) on a fixed batch
m.eval(); xb=torch.randint(0,S.VOCAB,(1,256),device=DEV)
with torch.no_grad(): pre=m(xb).clone()

while npar() < TARGET - 6e6: m.grow()                 # DEPTH first (all new layers feat8) until ~target
f=8
while f<FEAT: f=min(f+2,FEAT); m.grow_feat(f)         # THEN feat 8->FEAT uniformly across ALL layers
L=len(m.blocks); feat=m.blocks[0].mix.fe; P=npar()/1e6
print(f"grown: {P:.0f}M {L}L feat{feat}",flush=True)
with torch.no_grad(): post=m(xb)
print(f"function-preservation max|Δlogit| 216M->400M = {(pre-post).abs().max().item():.2e}",flush=True)

# VRAM test: real fwd+bwd at the training config (BS2, ctx2048) alongside prod
m.train(); torch.cuda.reset_peak_memory_stats()
x=torch.randint(0,S.VOCAB,(2,2048),device=DEV); y=torch.randint(0,S.VOCAB,(2,2048),device=DEV)
lg=m(x); loss=F.cross_entropy(lg.reshape(-1,S.VOCAB),y.reshape(-1)); loss.backward()
peak=torch.cuda.max_memory_allocated()/2**30; free=torch.cuda.mem_get_info()[0]/2**30
print(f"fwd+bwd BS2 ctx2048: peak_alloc={peak:.1f}G  free_after={free:.1f}G",flush=True)

out=f"/mnt/kv_cache/pragnosia_data/snaps/scale_490000_{P:.0f}M_{L}L_f{feat}.pt"
torch.save(m.state_dict(), out); print(f"SAVED {out}",flush=True)
print("DONE",flush=True)
