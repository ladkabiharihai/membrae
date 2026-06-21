"""Test the LATEST Pragnosia model on ALL brain faculties — READ-ONLY, CPU-only, so it
never touches the running GPU training. Run via faculty_test.sh (snapshots the live
checkpoint first, sets CUDA_VISIBLE_DEVICES="" so torch can't see the GPU).

Faculties covered:
  - the 14 proven faculties (unified_brain.pt): reason, confidence, abstain, seek, exact, alive
  - language: perplexity on the held-out valid set
  - identity: who/what are you  (raw weights — identity is normally runtime-installed)
  - generation: arithmetic, world knowledge, science, kinship reasoning
  - abstention: answers the knowable, says-IDK the unknowable
  - continual learning: teach a new fact + recall it (on the in-memory copy; never persisted)
"""
import json, sys, math, torch
import torch.nn.functional as F
torch.set_num_threads(16)
import s6_hybrid as H
import unified_brain as U
from tokenizers import Tokenizer

CK = sys.argv[1] if len(sys.argv) > 1 else "pragnosia.pt"
cfg = json.load(open("pragnosia.json")); H.VOC = cfg["vocab"]; H.L = cfg["ctx"]
U.DEVICE = "cpu"; H.DEVICE = "cpu"
tok = Tokenizer.from_file(cfg["tokenizer"]); ctx = cfg["ctx"]

# ---- load the LM from the snapshot; infer the (possibly grown) arch from the weights ----
sd = torch.load(CK, map_location="cpu", weights_only=True)
nl = max(int(k.split(".")[1]) for k in sd if k.startswith("blocks.") and k.split(".")[1].isdigit()) + 1
mm = next(sd[k].shape[0] for k in sd if k.endswith("mlp.0.weight")) // cfg["d"]
lm = H.SpinAttentionLM(cfg["vocab"], cfg["d"], cfg["heads"], nl, mlp_mult=mm).eval()
lm.load_state_dict(sd)
np_ = sum(p.numel() for p in lm.parameters())
print("=" * 68)
print(f"PRAGNOSIA FACULTY TEST  —  LM {np_/1e6:.0f}M ({nl}L mlp{mm})  ckpt={CK}")
print("=" * 68)

@torch.no_grad()
def gen(p, n=12, chat=False):
    ids = tok.encode(p).ids; out = []
    for _ in range(n):
        nx = int(lm(torch.tensor([(ids + out)[-ctx:]]))[0, -1].argmax())
        if nx == 0: break
        out.append(nx); t = tok.decode(out)
        if chat and "<user>" in t: return t.split("<user>")[0].strip()
        if not chat and "\n" in t: return t.split("\n")[0].strip()
    return tok.decode(out).strip()

# ===================== 1. THE 14 PROVEN FACULTIES =====================
print("\n[1] PROVEN FACULTIES (unified_brain.pt)")
fac = U.UnifiedBrain(d=128)
import os
if os.path.exists("unified_brain.pt"):
    fac.load_state_dict(torch.load("unified_brain.pt", map_location="cpu", weights_only=True))
fac.eval()
try:
    R = U.self_test(fac)
    checks = [("reason parity3", R['reason_parity3'], lambda v: v > 0.9), ("reason sum3", R['reason_sum3'], lambda v: v > 0.85),
              ("reason max", R['reason_max'], lambda v: v > 0.9), ("conf gap", R['p5_gap'], lambda v: v > 0.3),
              ("abstain known", R['p6_abstain_known'], lambda v: v < 0.25), ("abstain unknowable", R['p6_abstain_unknowable'], lambda v: v > 0.6),
              ("lang parse", R['lang_parse_held'], lambda v: v > 0.8), ("lang gen", R['lang_gen_held'], lambda v: v > 0.8),
              ("seek query", R['seek_query_lang'], lambda v: v > 0.9), ("seek answer", R['seek_answer'], lambda v: v > 0.85),
              ("seek rand ctrl", R['seek_random_ctrl'], lambda v: v < 0.45), ("exact L16", R['exact_L16'], lambda v: v > 0.9),
              ("alive final", R['alive_final'], lambda v: v > 0.95), ("alive gap", R['alive_retention_gap'], lambda v: v < 0.15)]
    npass = 0
    for nm, v, c in checks:
        ok = c(v); npass += ok; print(f"    [{'PASS' if ok else 'FAIL'}] {nm:<20}{v:.2f}")
    print(f"    => {npass}/14 faculties wired")
except Exception as e:
    print(f"    (faculties test skipped: {str(e)[:60]})")

# ===================== 2. LANGUAGE: perplexity =====================
print("\n[2] LANGUAGE perplexity (held-out valid)")
try:
    vd = H.load(cfg["valid_bin"]); ppl = H.val_ppl(lm, vd, iters=15)
    print(f"    [{'PASS' if ppl < 30 else 'FAIL'}] valid ppl {ppl:.1f}  (training best ~20.5)")
except Exception as e:
    print(f"    (skipped: {str(e)[:50]})")

# ===================== 3. IDENTITY =====================
print("\n[3] IDENTITY (raw weights — normally runtime-installed via teach)")
for q in ["Who are you?", "What is your name?"]:
    print(f"    {q!r} -> {gen('<user> '+q+' <assistant>', 16, chat=True)!r}")

# ===================== 4. GENERATION across domains =====================
print("\n[4] GENERATION")
ar = [("23 + 45 =", "68"), ("100 - 37 =", "63"), ("12 * 12 =", "144"), ("250 + 250 =", "500")]
na = sum(want in gen(q, 6) for q, want in ar)
print(f"    arithmetic: {na}/{len(ar)} correct  " + "  ".join(f"{q}{gen(q,6)}" for q, _ in ar[:3]))
for q in ["The capital of France is", "The largest planet is"]:
    print(f"    knowledge: {q!r} -> {gen(q,8)!r}")
print(f"    science:   'Photosynthesis converts sunlight into' -> {gen('Photosynthesis converts sunlight into',8)!r}")
print(f"    kinship:   'Tom is father of Sam. Sam is father of Leo. Tom is Leo's' -> {gen('Tom is the father of Sam. Sam is the father of Leo. So Tom is Leo'+chr(39)+'s',6)!r}")

# ===================== 5. ABSTENTION (language) =====================
print("\n[5] ABSTENTION (answer knowable, hedge unknowable)")
known = ["The cat played with the", "A dog is an"]; unk = ["My phone number is", "The lottery numbers tomorrow are"]
def conf(t):  # lower NLL = more confident
    ids = tok.encode(t).ids
    if len(ids) < 2: return 9.9
    return float(F.cross_entropy(lm(torch.tensor([ids]))[0, :-1], torch.tensor(ids[1:])))
kc = [conf(t) for t in known]; uc = [conf(t) for t in unk]
print(f"    known-text NLL {[round(x,1) for x in kc]} (low=confident)  vs unknowable {[round(x,1) for x in uc]}")
print(f"    [{'PASS' if sum(uc)/2 > sum(kc)/2 else 'WEAK'}] more confident on knowable than unknowable")

# ===================== 6. CONTINUAL LEARNING (teach + recall, on copy) =====================
print("\n[6] CONTINUAL LEARNING (teach a fact, recall it — in-memory, NOT persisted)")
before = gen("The CEO of Tesla is", 6)
opt = torch.optim.AdamW(lm.parameters(), lr=2e-4)
fact = "The CEO of Tesla is Elon Musk. Elon Musk is the chief executive of Tesla."
fids = tok.encode(fact).ids
for _ in range(30):
    x = torch.tensor([fids[:-1]]); y = torch.tensor([fids[1:]])
    opt.zero_grad(); F.cross_entropy(lm(x).reshape(-1, cfg["vocab"]), y.reshape(-1)).backward(); opt.step()
after = gen("The CEO of Tesla is", 6)
print(f"    before: {before!r}")
print(f"    after teaching: {after!r}   recall={'YES' if 'elon' in after.lower() or 'musk' in after.lower() else 'no'}")
print("=" * 68)
print("DONE — training was NOT touched (CPU-only, snapshot, no persist).")
sys.stdout.flush(); import os as o; o._exit(0)
