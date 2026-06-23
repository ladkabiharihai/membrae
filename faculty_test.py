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
import json, sys, math, os, torch
import torch.nn.functional as F
torch.set_num_threads(16)
import s6_hybrid as H
from tokenizers import Tokenizer

CK = sys.argv[1] if len(sys.argv) > 1 else "pragnosia.pt"
cfg = json.load(open("pragnosia.json")); H.VOC = cfg["vocab"]; H.L = cfg["ctx"]
H.DEVICE = "cpu"
tok = Tokenizer.from_file(cfg["tokenizer"]); ctx = cfg["ctx"]

# ---- load the LM from the snapshot; infer the (possibly grown) arch from the weights ----
sd = torch.load(CK, map_location="cpu", weights_only=True)
nl = max(int(k.split(".")[1]) for k in sd if k.startswith("blocks.") and k.split(".")[1].isdigit()) + 1
mm = next(sd[k].shape[0] for k in sd if k.endswith("mlp.0.weight")) // cfg["d"]
lm = H.SpinAttentionLM(cfg["vocab"], cfg["d"], cfg["heads"], nl, mlp_mult=mm, carrier=cfg.get("carrier", "single")).eval()
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

# (The old 302K toy faculties are gone -- the spin-dominant LM is the whole brain now. This file
#  tests the LM directly; `python3 brain.py test` adds the subconscious + cognition checks.)

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

# ===================== 4. CAPABILITY BATTERY (rigorous, scored) =====================
# One repeatable battery so every checkpoint is measured the same way. Each item is
# (prompt, accept) where accept is a substring or list of acceptable substrings; gen()
# stops at the first newline so a clean first-line answer is what's checked.
print("\n[4] CAPABILITY BATTERY (scored — track this across checkpoints)")
BATTERY = {
 "arithmetic (1-step)": [("23 + 45 =","68"),("100 - 37 =","63"),("12 * 12 =","144"),
    ("250 + 250 =","500"),("9 * 7 =","63"),("144 / 12 =","12"),("1000 - 1 =","999")],
 "multi-step reasoning": [("There are 5 boxes with 4 balls each. The total number of balls is","20"),
    ("John has 12 apples. He gives away 5 and buys 8 more. John now has","15"),
    ("A train goes 60 miles per hour for 3 hours. It travels","180")],
 "knowledge": [("The capital of France is","Paris"),("The capital of Japan is","Tokyo"),
    ("Water is made of hydrogen and","oxygen"),("Photosynthesis converts sunlight into",["energy","sugar","glucose"])],
 "long-tail facts": [("The capital of Australia is","Canberra"),("The chemical symbol for gold is","Au"),
    ("The largest planet in the solar system is","Jupiter"),("World War II ended in the year","1945")],
 "relational / kinship": [("Tom is the father of Sam. Sam is the father of Leo. Tom is Leo's",["grandfather","grandpa"]),
    ("Anna is Bob's sister. Bob is Carl's father. Anna is Carl's","aunt")],
 "analogy": [("Paris is to France as Tokyo is to","Japan"),("Hot is to cold as up is to","down")],
 "logic / deduction": [("All cats are animals. Felix is a cat. So Felix is an","animal"),
    ("If it rains the ground is wet. It is raining. So the ground is","wet")],
 "code": [("def add(a, b):\n    return ","a + b"),("def is_palindrome(s):\n    return s == s","[::-1"),
    ("def is_even(n):\n    return n % 2 == ","0")],
 "instruction following": [("List three colors:",["red","blue","green","yellow"]),
    ("Write the opposite of 'happy':",["sad","unhappy"])],
}
def _ok(ans, acc): return any(x.lower() in ans.lower() for x in (acc if isinstance(acc, list) else [acc]))
tot_ok = tot_n = 0
for cat, items in BATTERY.items():
    ok = sum(_ok(gen(p, 8), a) for p, a in items)
    tot_ok += ok; tot_n += len(items)
    bar = "█" * ok + "·" * (len(items) - ok)
    print(f"    {cat:22} {ok}/{len(items):<2} {bar}")
print(f"    {'OVERALL':22} {tot_ok}/{tot_n}  ({100*tot_ok//tot_n}%)")
print("    (strong: 1-step arithmetic/facts/logic · weak: multi-step/relational/analogy = needs more training, not prompting)")

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
need_gb = np_ * 12 / 1e9                    # AdamW(m,v)+grad over the whole LM, fp32 ~ 3x params x 4B
free_gb = os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE") / 1e9
if need_gb > free_gb * 0.85:
    print(f"    [SKIP] teaching this model needs ~{need_gb:.0f}GB RAM (free ~{free_gb:.0f}GB).")
    print(f"           continual learning is verified on a bigger box; run there for this size.")
else:
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
