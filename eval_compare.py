"""Apples-to-apples zero-shot comparison: OUR 284M spin-dominant vs open models of similar size,
through the SAME harness (same example subsamples, same log-likelihood scoring). Tokenizer-independent
metrics (multiple-choice accuracy + LAMBADA), so the custom-vs-standard vocab difference doesn't matter.
Usage: python3 eval_compare.py <model>   where <model> is 'ours' or an HF id. Appends to compare_results.json."""
import os, sys, json, math, time
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
import torch, torch.nn.functional as F
NAME = sys.argv[1] if len(sys.argv) > 1 else "ours"
DEV = "cuda" if torch.cuda.is_available() else "cpu"

def load(name):
    if name == "ours":
        import s6_hybrid as H
        from tokenizers import Tokenizer
        c = json.load(open("pragnosia.json")); H.VOC, H.L = c["vocab"], c["ctx"]
        m = H.SpinAttentionLM(c["vocab"], c["d"], c["heads"], c["layers"], mlp_mult=c["mlp_mult"], carrier=c["carrier"]).to(DEV)
        m.load_state_dict(torch.load("pragnosia_spin.pt", map_location="cpu", weights_only=True))
        m = m.bfloat16().eval(); tk = Tokenizer.from_file(c["tokenizer"])
        enc = lambda s: tk.encode(s).ids
        logp = lambda ids: m(torch.tensor([ids], device=DEV))[0].float().log_softmax(-1)
        np_ = sum(p.numel() for p in m.parameters())
        return enc, logp, c["ctx"], f"ours-{np_/1e6:.0f}M"
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tk = AutoTokenizer.from_pretrained(name)
    m = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.bfloat16).to(DEV).eval()
    ctx = min(getattr(m.config, "n_positions", None) or getattr(m.config, "max_position_embeddings", 1024) or 1024, 1024)
    enc = lambda s: tk(s).input_ids
    logp = lambda ids: m(torch.tensor([ids], device=DEV)).logits[0].float().log_softmax(-1)
    np_ = sum(p.numel() for p in m.parameters())
    return enc, logp, ctx, f"{name} ({np_/1e6:.0f}M)"

enc, logp, CTX, label = load(NAME)
print(f"loaded {label} on {DEV}", flush=True)

@torch.no_grad()
def cont_lp(ctx, cont):
    full = enc(ctx + cont); ci = len(enc(ctx)); n = len(full) - ci
    if n <= 0: return -1e9, 1
    full = full[-CTX:]; n = min(n, len(full) - 1)
    if n <= 0: return -1e9, 1
    lg = logp(full)
    return sum(lg[j - 1, full[j]].item() for j in range(len(full) - n, len(full))), n

def mc(name, items, norm=True, cap=1000):
    import random; random.seed(0)
    if len(items) > cap: items = random.sample(items, cap)
    ok = sum(max(range(len(o)), key=lambda i: (lambda lp, nt: lp / nt if norm else lp)(*cont_lp(ctx, o[i])))
             == g for ctx, o, g in items)
    print(f"  {name:14} {ok/len(items)*100:5.1f}%", flush=True); return round(ok / len(items), 4)

from datasets import load_dataset
res = {}
try:
    lb = load_dataset("EleutherAI/lambada_openai", "en", split="test")
    import random; random.seed(0)
    sub = random.sample([r["text"].strip() for r in lb], 1000)
    ok = 0
    with torch.no_grad():
        for s in sub:
            k = s.rfind(" ");  ctx, gold = s[:k], s[k:]
            gid = enc(ctx + gold)[len(enc(ctx)):]
            if not gid: continue
            pred = int(logp(enc(ctx)[-CTX:])[-1].argmax())
            ok += (pred == gid[0])
    res["lambada"] = round(ok / len(sub), 4); print(f"  {'LAMBADA':14} {res['lambada']*100:5.1f}%", flush=True)
except Exception as e: print("  LAMBADA skipped:", str(e)[:50], flush=True)
try:
    hs = load_dataset("Rowan/hellaswag", split="validation")
    res["hellaswag"] = mc("HellaSwag", [(r["ctx"], r["endings"], int(r["label"])) for r in hs if r["label"] != ""])
except Exception as e: print("  HellaSwag skipped:", str(e)[:50], flush=True)
for cfg, key in [("ARC-Easy", "arc_easy"), ("ARC-Challenge", "arc_challenge")]:
    try:
        ar = load_dataset("allenai/ai2_arc", cfg, split="test")
        items = [("Question: " + r["question"] + "\nAnswer: ", r["choices"]["text"], r["choices"]["label"].index(r["answerKey"]))
                 for r in ar if r["answerKey"] in r["choices"]["label"]]
        res[key] = mc(cfg, items)
    except Exception as e: print(f"  {cfg} skipped:", str(e)[:50], flush=True)
try:
    pq = load_dataset("piqa", split="validation", revision="refs/convert/parquet")
    res["piqa"] = mc("PIQA", [(r["goal"] + " ", [r["sol1"], r["sol2"]], r["label"]) for r in pq if r["label"] in (0, 1)], norm=False)
except Exception as e: print("  PIQA skipped:", str(e)[:50], flush=True)

allr = json.load(open("compare_results.json")) if os.path.exists("compare_results.json") else {}
allr[label] = res; json.dump(allr, open("compare_results.json", "w"), indent=2)
print("\n", label, json.dumps(res), flush=True)
