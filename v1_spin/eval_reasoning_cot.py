"""Reasoning eval that separates the MODEL's ability from the HARNESS's scoring — the earlier "GSM8K 0%"
was a parser artifact (last-number-over-a-ramble), not a model failure. Two honest measurements:

  1. GSM8K-style: few-shot chain-of-thought + a GENERAL answer parser (prefer "answer is X"; else the
     last "= X"; else the last number) with a stop-once-answered rule. Few-shot examples are in-context
     DEMONSTRATIONS, and the parser is a generic regex -- neither is fit to the eval's answers (no hardcoding).
  2. Needle: naive long-range (feed the whole context through the damped carry) vs RETRIEVAL (feed only the
     sentence the query asks about, as brain.py's episodic memory would) -- isolates carry-decay from copy.

Inference-only; safe to run alongside live training. Checkpoint via env:
  OURS_CKPT=pragnosia_1b_best.pt OURS_CONFIG=pragnosia_1b_combined.json python3 eval_reasoning_cot.py
"""
import os, re, json, random, torch, torch.nn.functional as F
import s6_hybrid as H
from tokenizers import Tokenizer

DEV = "cuda" if torch.cuda.is_available() else "cpu"
H.VOC, H.L, H.DEVICE = 16384, 256, DEV
CKPT = os.environ.get("OURS_CKPT", "pragnosia_1b_best.pt")
CFG = json.load(open(os.environ.get("OURS_CONFIG", "pragnosia_1b_combined.json")))
tok = Tokenizer.from_file(CFG["tokenizer"])
AC = torch.autocast("cuda", dtype=torch.bfloat16) if DEV == "cuda" else torch.autocast("cpu", enabled=False)

sd = torch.load(CKPT, map_location="cpu", weights_only=True)
m = H.SpinAttentionLM(CFG["vocab"], CFG["d"], CFG["heads"], CFG["layers"],
                      mlp_mult=CFG["mlp_mult"], carrier=CFG["carrier"]).to(DEV).eval()
m.load_state_dict(sd)
if DEV == "cuda": m = m.bfloat16()
print(f"[eval] {CKPT}: {sum(p.numel() for p in m.parameters())/1e6:.0f}M params on {DEV}")


@torch.no_grad()
def gen(prompt, max_new=64, stop_answer=True):
    ids = tok.encode(prompt).ids; out = []
    for _ in range(max_new):
        with AC:
            nx = int(m(torch.tensor([(ids + out)[-256:]], device=DEV))[0, -1].argmax())
        if nx == 0: break
        out.append(nx); dec = tok.decode(out)
        if stop_answer and re.search(r'answer is\s*-?\d', dec): break     # stop once it states an answer
        if dec.count("\n") >= 1: break
    return tok.decode(out)


def parse_answer(text):
    """GENERAL numeric parser (not tuned to any answer): explicit 'answer is X' > last '= X' > last number."""
    t = text.replace(",", "")
    ma = re.search(r'answer is\s*(-?\d+(?:\.\d+)?)', t)
    if ma: return ma.group(1)
    eqs = re.findall(r'=\s*(-?\d+(?:\.\d+)?)', t)
    if eqs: return eqs[-1]
    ns = re.findall(r'-?\d+(?:\.\d+)?', t)
    return ns[-1] if ns else None


# In-context CoT demonstrations (generic worked examples; NOT any test item's answer).
FEWSHOT = ("Q: A shop has 6 boxes with 3 pens each. How many pens total?\n"
           "A: Each box has 3 pens. 6 times 3 = 18. The answer is 18.\n"
           "Q: Ann had 15 dollars and spent 4. How much is left?\n"
           "A: She had 15 and spent 4. 15 - 4 = 11. The answer is 11.\n")

GSM = [("Sarah has 3 apples and buys 5 more. How many apples does she have?", "8"),
       ("A book costs 12 dollars. John buys 4 books. What is the total cost?", "48"),
       ("There are 24 students split into 4 equal groups. How many in each group?", "6"),
       ("Tom had 20 dollars and spent 7. How much does he have left?", "13"),
       ("A car travels 50 miles per hour for 2 hours. How far does it go?", "100"),
       ("A baker makes 8 cakes a day for 5 days. How many cakes total?", "40"),
       ("Lisa reads 15 pages a day. In 3 days how many pages?", "45"),
       ("A team scored 6 goals in each of 3 games. Total goals?", "18")]


def gsm8k():
    print("\n=== GSM8K-style: direct vs few-shot CoT (general parser) ===")
    direct = comp = correct = 0
    for q, a in GSM:
        gd = gen("Q: " + q + "\nA:", 12); direct += (parse_answer(gd) == a)
        g = gen(FEWSHOT + f"Q: {q}\nA:", 64).split("Q:")[0]
        pred = parse_answer(g); correct += (pred == a)
        comp += (a in re.findall(r'-?\d+', g.replace(",", "")))   # did the RIGHT number appear in its working?
        first = (g.strip().splitlines() or [""])[0]
        print(f"  {'OK' if pred==a else 'x '} ans={a:>4} pred={str(pred):>5} :: {first[:64]!r}")
    n = len(GSM)
    print(f"  direct: {direct}/{n}  |  few-shot CoT (parsed): {correct}/{n} ({100*correct//n}%)  "
          f"|  correct number present in working: {comp}/{n}")


def needle():
    print("\n=== Needle: naive long-range vs RETRIEVAL (brain.py-style memory) ===")
    random.seed(0); filler = " The weather today is mild and the streets are quiet."
    words = ["apple", "river", "copper", "violet", "tiger", "maple"]

    def run(depth, retrieve):
        hits = trials = 0
        for _ in range(6):
            secret = random.choice(words)
            if retrieve:
                ids = tok.encode(f"The secret word is {secret}. The secret word was").ids
            else:
                fids = tok.encode(filler * 200).ids[:depth]
                ids = tok.encode(f"The secret word is {secret}.").ids + fids + tok.encode(" The secret word was").ids
            with torch.no_grad():
                S = None
                for i in range(0, max(1, len(ids) - 1), 256):
                    with AC: _, S = m(torch.tensor([ids[i:i + 256]], device=DEV), state=S, return_state=True)
                o = []
                for _ in range(6):
                    with AC: nx = int(m(torch.tensor([(ids + o)[-256:]], device=DEV))[0, -1].argmax()); o.append(nx)
            hits += secret in tok.decode(o).lower(); trials += 1
        return hits, trials
    for depth in [16, 64, 200, 512, 1024]:
        h, t = run(depth, False); print(f"  naive depth ~{depth:4d}: {h}/{t}")
    h, t = run(0, True); print(f"  RETRIEVAL           : {h}/{t}")


if __name__ == "__main__":
    gsm8k(); needle(); print("\n[done]")
