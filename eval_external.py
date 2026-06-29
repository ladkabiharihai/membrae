"""Zero-shot evaluation of the spin-dominant LM on standard public benchmarks.
Custom 16K tokenizer -> we report WORD-normalized perplexity (tokenizer-independent) and
log-likelihood multiple-choice accuracy (also tokenizer-independent). CPU/GPU, read-only."""
import os, sys, math, json, time
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
import torch, torch.nn.functional as F, s6_hybrid as H
from tokenizers import Tokenizer
CKPT = sys.argv[1] if len(sys.argv) > 1 else "pragnosia_spin.pt"
DEV = "cuda" if torch.cuda.is_available() else "cpu"
c = json.load(open("pragnosia.json")); H.VOC, H.L = c["vocab"], c["ctx"]; CTX = c["ctx"]
tok = Tokenizer.from_file(c["tokenizer"])
m = H.SpinAttentionLM(c["vocab"], c["d"], c["heads"], c["layers"], mlp_mult=c["mlp_mult"], carrier=c["carrier"]).to(DEV)
m.load_state_dict(torch.load(CKPT, map_location="cpu", weights_only=True))
m = m.bfloat16().eval() if DEV == "cuda" else m.eval()
print(f"loaded {sys.exit if False else ''}{H.n_params(m)/1e6:.0f}M {c['carrier']} on {DEV}\n", flush=True)

@torch.no_grad()
def cont_lp(ctx, cont):
    """sum log p(cont | ctx) and #cont-tokens, context truncated to the model window."""
    full = tok.encode(ctx + cont).ids
    ci = len(tok.encode(ctx).ids)
    n = len(full) - ci
    if n <= 0: return -1e9, 1
    full = full[-CTX:]                       # cap to window (cont is at the tail)
    n = min(n, len(full) - 1)
    x = torch.tensor([full], device=DEV)
    lg = m(x)[0].float().log_softmax(-1)
    lp = sum(lg[j - 1, full[j]].item() for j in range(len(full) - n, len(full)))
    return lp, n

def mc(name, items, norm=True, cap=1000):
    """items: list of (context, [options], gold_idx). Returns acc."""
    import random; random.seed(0)
    if len(items) > cap: items = random.sample(items, cap)
    ok = 0
    for ctx, opts, gold in items:
        sc = []
        for o in opts:
            lp, nt = cont_lp(ctx, o)
            sc.append(lp / nt if norm else lp)
        if max(range(len(sc)), key=lambda i: sc[i]) == gold: ok += 1
    print(f"  {name:14} {ok/len(items)*100:5.1f}%  (n={len(items)}, chance {100/len(items[0][1]):.0f}%)", flush=True)
    return ok / len(items)

from datasets import load_dataset
res = {}

# ---- WikiText-2 word-perplexity ----
try:
    t0 = time.time(); wt = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n".join(r for r in wt["text"] if r.strip())
    words = len(text.split()); ids = tok.encode(text).ids; tot = 0.0
    for i in range(0, len(ids) - 1, CTX):
        ch = ids[i:i + CTX + 1]
        if len(ch) < 2: break
        x = torch.tensor([ch[:-1]], device=DEV)
        lg = m(x)[0].float()
        tot += F.cross_entropy(lg, torch.tensor(ch[1:], device=DEV), reduction="sum").item()
    res["wikitext2_word_ppl"] = math.exp(tot / words)
    print(f"  WikiText-2     word-ppl {res['wikitext2_word_ppl']:.1f}  ({words/1e3:.0f}K words, {time.time()-t0:.0f}s)", flush=True)
except Exception as e: print("  WikiText-2     skipped:", str(e)[:60], flush=True)

# ---- LAMBADA (last-word accuracy) ----
try:
    lb = load_dataset("EleutherAI/lambada_openai", "en", split="test")
    items = []
    for r in lb:
        s = r["text"].strip(); k = s.rfind(" ")
        if k > 0: items.append((s[:k], [s[k:]], 0))   # predict the last word
    # accuracy = greedy last token matches; approximate via: the gold continuation is the only option,
    # so instead score whether argmax-continuation equals the word -> use a small generate
    import random; random.seed(0); sub = random.sample(items, min(1000, len(items)))
    ok = 0
    for ctx, (gold,), _ in sub:
        gid = tok.encode(ctx + gold).ids[len(tok.encode(ctx).ids):]
        if not gid: continue
        full = tok.encode(ctx).ids[-CTX:]
        pred = int(m(torch.tensor([full], device=DEV))[0, -1].argmax())
        ok += (pred == gid[0])
    res["lambada_acc"] = ok / len(sub)
    print(f"  LAMBADA        {res['lambada_acc']*100:5.1f}%  (first-token of last word, n={len(sub)})", flush=True)
except Exception as e: print("  LAMBADA        skipped:", str(e)[:60], flush=True)

# ---- HellaSwag (acc_norm) ----
try:
    hs = load_dataset("Rowan/hellaswag", split="validation")
    items = [(r["ctx"], r["endings"], int(r["label"])) for r in hs if r["label"] != ""]
    res["hellaswag_accn"] = mc("HellaSwag", items, norm=True)
except Exception as e: print("  HellaSwag      skipped:", str(e)[:60], flush=True)

# ---- ARC-Easy (acc_norm) ----
try:
    ar = load_dataset("allenai/ai2_arc", "ARC-Easy", split="test")
    items = []
    for r in ar:
        ch = r["choices"]["text"]; lab = r["choices"]["label"]
        if r["answerKey"] in lab: items.append(("Question: " + r["question"] + "\nAnswer: ", ch, lab.index(r["answerKey"])))
    res["arc_easy_accn"] = mc("ARC-Easy", items, norm=True)
except Exception as e: print("  ARC-Easy       skipped:", str(e)[:60], flush=True)

# ---- ARC-Challenge (acc_norm) ----
try:
    ac = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test")
    items = []
    for r in ac:
        ch = r["choices"]["text"]; lab = r["choices"]["label"]
        if r["answerKey"] in lab: items.append(("Question: " + r["question"] + "\nAnswer: ", ch, lab.index(r["answerKey"])))
    res["arc_challenge_accn"] = mc("ARC-Challenge", items, norm=True)
except Exception as e: print("  ARC-Challenge  skipped:", str(e)[:60], flush=True)

# ---- PIQA (acc) ----
try:
    pq = load_dataset("piqa", split="validation", revision="refs/convert/parquet")
    items = [(r["goal"] + " ", [r["sol1"], r["sol2"]], r["label"]) for r in pq if r["label"] in (0, 1)]
    res["piqa_acc"] = mc("PIQA", items, norm=False)
except Exception as e: print("  PIQA           skipped:", str(e)[:60], flush=True)

print("\nJSON:", json.dumps({k: round(v, 3) for k, v in res.items()}), flush=True)
