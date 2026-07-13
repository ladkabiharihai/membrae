"""Reviewer item C: add MMLU (knowledge, 4-choice) and GSM8K (grade-school math, generation) on the custom
tokenizer. MMLU via length-normalized log-likelihood MC (tokenizer-independent, like the other benchmarks);
GSM8K via greedy generation + numeric-answer match. Scores will be modest for a small undertrained model --
reporting them honestly is the point (better than omitting).

Run (GPU, Unreal closed):  python3 eval_reasoning.py [ckpt] [mmlu_cap] [gsm8k_cap]
"""
import os, sys, json, re, random
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch, s6_hybrid as H
from tokenizers import Tokenizer
from datasets import load_dataset

CKPT = sys.argv[1] if len(sys.argv) > 1 else "pragnosia_spin_h100.pt"
MMLU_CAP = int(sys.argv[2]) if len(sys.argv) > 2 else 1000
GSM_CAP = int(sys.argv[3]) if len(sys.argv) > 3 else 150
DEV = "cuda" if torch.cuda.is_available() else "cpu"
c = json.load(open("pragnosia.json")); H.VOC, H.L = c["vocab"], c["ctx"]; CTX = c["ctx"]
tok = Tokenizer.from_file(c["tokenizer"])
m = H.SpinAttentionLM(c["vocab"], c["d"], c["heads"], c["layers"], mlp_mult=c["mlp_mult"], carrier=c["carrier"]).to(DEV)
m.load_state_dict(torch.load(CKPT, map_location="cpu", weights_only=True))
m = m.bfloat16().eval() if DEV == "cuda" else m.eval()
P = lambda *a: print(*a, flush=True)
P(f"loaded {CKPT} ({H.n_params(m)/1e6:.0f}M) on {DEV}")


@torch.no_grad()
def cont_lp(ctx, cont):
    full = tok.encode(ctx + cont).ids; ci = len(tok.encode(ctx).ids)
    n = len(full) - ci
    if n <= 0: return -1e9, 1
    full = full[-CTX:]; n = min(n, len(full) - 1)
    lg = m(torch.tensor([full], device=DEV))[0].float().log_softmax(-1)
    lp = sum(lg[j - 1, full[j]].item() for j in range(len(full) - n, len(full)))
    return lp, n


def mc(items, norm=True):
    ok = 0
    for ctx, opts, gold in items:
        sc = [cont_lp(ctx, o)[0] / (cont_lp(ctx, o)[1] if norm else 1) for o in opts]
        ok += (max(range(len(sc)), key=lambda i: sc[i]) == gold)
    return ok / len(items)


res = {}

# ---- MMLU (acc_norm, 4-choice) ----
try:
    ds = load_dataset("cais/mmlu", "all", split="test")
    LET = ["A", "B", "C", "D"]
    items = []
    for r in ds:
        q, ch, a = r["question"], r["choices"], r["answer"]
        if len(ch) == 4 and 0 <= a < 4:
            ctx = "Question: " + q.strip() + "\nAnswer: "
            items.append((ctx, [str(o) for o in ch], a))
    random.seed(0); random.shuffle(items); items = items[:MMLU_CAP]
    res["mmlu_accn"] = round(mc(items, norm=True), 4)
    P(f"  MMLU        {res['mmlu_accn']*100:5.1f}%  (n={len(items)}, chance 25%)")
except Exception as e:
    P("  MMLU        skipped:", str(e)[:80])


# ---- GSM8K (generation, exact numeric match) ----
def last_num(s):
    nums = re.findall(r"-?\d[\d,]*\.?\d*", s.replace(",", ""))
    return nums[-1] if nums else None

try:
    gs = load_dataset("gsm8k", "main", split="test")
    items = [(r["question"], r["answer"].split("####")[-1].strip().replace(",", "")) for r in gs]
    random.seed(0); random.shuffle(items); items = items[:GSM_CAP]
    ok = 0
    for q, gold in items:
        prompt = f"Question: {q}\nAnswer:"
        ids = tok.encode(prompt).ids[-CTX:]
        with torch.no_grad():
            out = m.generate(ids, n_new=100, window=CTX, temp=0.0, rep=1.3)
        gen = tok.decode(out[len(ids):] if len(out) > len(ids) else out)
        pred = last_num(gen)
        ok += (pred is not None and pred == last_num(gold))
    res["gsm8k_em"] = round(ok / len(items), 4)
    P(f"  GSM8K       {res['gsm8k_em']*100:5.1f}%  (n={len(items)}, greedy gen, exact numeric match)")
except Exception as e:
    P("  GSM8K       skipped:", str(e)[:80])

os.makedirs("eval_registry", exist_ok=True)
out = {"ckpt": CKPT, **res}
json.dump(out, open("eval_registry/reasoning_bench.json", "w"), indent=2)
P("\nJSON: " + json.dumps(out))
