"""
Post-training corpus for Pragnosia: an instruction/chat-heavy mix on top of the SAME
digit-aware tokenizer (data/bpe.json), with EXPLICIT per-category token budgets so the
final blend is what we ask for. Built for continued training (post-train) of the 228M
parallel-spin model with growth enabled. Tokenization is parallel (encode_batch).

Target blend (of --tokens, default 5e9):
  instruction + chat   40%   OpenOrca, Alpaca, Dolly, OpenAssistant (chat oversampled)
  reasoning + math     20%   OpenOrca-reason + GSM8K, Orca-Math, MetaMathQA (CoT)
  physics + science    12%   SciQ, OpenBookQA, ARC
  world knowledge      18%   Wikipedia + FineWeb-Edu
  commonsense          10%   CommonsenseQA, PIQA

Writes data/big_train.bin + data/big_valid.bin (the model's new distribution + replay
pool). Each source is wrapped in try/except so a missing dataset is skipped, not fatal.

  python3 prepare_posttrain.py --tokens 5e9
"""
import argparse, json, os, sys, time
import numpy as np
from datasets import load_dataset
from tokenizers import Tokenizer
import prepare_data as P            # qa(), web_stream()

os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")

def _cap(it, n):
    for i, x in enumerate(it):
        if n and i >= n: break
        yield x

def _repeat(gen, k):                      # oversample a small source by replaying it
    for _ in range(k):
        for x in gen(): yield x

_ROLE = {"human": "<user> ", "user": "<user> ", "gpt": "<assistant> ", "assistant": "<assistant> ",
         "system": "<system> ", "tool": "<tool> "}
def _conv_fv(convs):                      # OpenHermes / SlimOrca: [{from, value}]
    return "\n".join(_ROLE.get(m.get("from", ""), "") + " ".join((m.get("value") or "").split()) for m in convs)
def _conv_rc(msgs):                       # UltraChat / Tulu-3: [{role, content}]
    return "\n".join(_ROLE.get(m.get("role", ""), "") + " ".join((m.get("content") or "").split()) for m in msgs)

def instruction_chat():
    # gold-standard SFT / chat sets that real LLMs train on (high quality, large)
    try:
        for r in load_dataset("teknium/OpenHermes-2.5", split="train", streaming=True):
            if r.get("conversations"): yield _conv_fv(r["conversations"])
    except Exception as e: print("skip openhermes", str(e)[:50])
    try:
        for r in load_dataset("Open-Orca/SlimOrca", split="train", streaming=True):
            if r.get("conversations"): yield _conv_fv(r["conversations"])
    except Exception as e: print("skip slimorca", str(e)[:50])
    try:
        for r in load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft", streaming=True):
            if r.get("messages"): yield _conv_rc(r["messages"])
    except Exception as e: print("skip ultrachat", str(e)[:50])
    try:
        for r in load_dataset("allenai/tulu-3-sft-mixture", split="train", streaming=True):
            if r.get("messages"): yield _conv_rc(r["messages"])
    except Exception as e: print("skip tulu3", str(e)[:50])
    for repo, im, ic, om in [("tatsu-lab/alpaca","instruction","input","output"),
                             ("databricks/databricks-dolly-15k","instruction","context","response")]:
        try:
            for r in load_dataset(repo, split="train"):
                if r[im] and r[om]: yield P.qa(r[im], r[om], r.get(ic) or "")
        except Exception as e: print("skip", repo, str(e)[:40])

def math_reasoning():
    try:
        for r in load_dataset("AI-MO/NuminaMath-CoT", split="train", streaming=True):
            if r.get("problem") and r.get("solution"): yield P.qa(r["problem"], r["solution"])
    except Exception as e: print("skip numina", str(e)[:50])
    try:
        for r in _cap(load_dataset("nvidia/OpenMathInstruct-2", split="train", streaming=True), 2_000_000):
            if r.get("problem") and r.get("generated_solution"): yield P.qa(r["problem"], r["generated_solution"])
    except Exception as e: print("skip openmathinstruct", str(e)[:50])
    for _ in range(2):                                       # oversample the smaller CoT sets
        try:
            for r in load_dataset("openai/gsm8k","main",split="train"): yield P.qa(r["question"], r["answer"])
        except Exception as e: print("skip gsm8k", str(e)[:40])
        try:
            for r in load_dataset("meta-math/MetaMathQA", split="train"):
                q = r.get("query") or r.get("original_question"); a = r.get("response")
                if q and a: yield P.qa(q, a)
        except Exception as e: print("skip metamath", str(e)[:40])

def physics_science():
    try:
        for r in load_dataset("sciq", split="train"):
            a = r.get("correct_answer"); s = r.get("support") or ""
            if r.get("question") and a: yield P.qa(r["question"], (a + ". " + s).strip())
    except Exception as e: print("skip sciq", str(e)[:40])
    try:
        for r in load_dataset("openbookqa","main",split="train"):
            ch = r["choices"]; ans = dict(zip(ch["label"], ch["text"])).get(r["answerKey"])
            if r.get("question_stem") and ans: yield P.qa(r["question_stem"], ans)
    except Exception as e: print("skip openbookqa", str(e)[:40])
    for cfg in ["ARC-Easy", "ARC-Challenge"]:
        try:
            for r in load_dataset("ai2_arc", cfg, split="train"):
                ch = r["choices"]; ans = dict(zip(ch["label"], ch["text"])).get(r["answerKey"])
                if r.get("question") and ans: yield P.qa(r["question"], ans)
        except Exception as e: print("skip arc", cfg, str(e)[:30])

def world_knowledge():
    # Cosmopedia (synthetic textbooks — what SmolLM trains on) + Wikipedia + FineWeb-Edu
    for cfg in ["web_samples_v2", "stanford", "khanacademy", "wikihow"]:
        try:
            for r in load_dataset("HuggingFaceTB/cosmopedia", cfg, split="train", streaming=True):
                t = (r.get("text") or "").strip()
                if len(t) > 300: yield t
        except Exception as e: print("skip cosmopedia", cfg, str(e)[:40])
    try:
        for r in load_dataset("wikimedia/wikipedia", "20231101.en", split="train", streaming=True):
            t = (r.get("text") or "").strip()
            if len(t) > 400: yield t[:4000]
    except Exception as e: print("skip wikipedia", str(e)[:50])
    for _, t in P.web_stream("sample-100BT"):                # FineWeb-Edu fills the rest
        yield t

def commonsense():
    for _ in range(4):                                       # small sets, oversample
        try:
            for r in load_dataset("tau/commonsense_qa", split="train"):
                ans = dict(zip(r["choices"]["label"], r["choices"]["text"])).get(r["answerKey"])
                if r.get("question") and ans: yield P.qa(r["question"], ans)
        except Exception as e: print("skip csqa", str(e)[:40])
        try:
            for r in load_dataset("piqa", split="train", trust_remote_code=True):
                sol = r["sol1"] if r["label"] == 0 else r["sol2"]
                if r.get("goal") and sol: yield P.qa(r["goal"], sol)
        except Exception as e: print("skip piqa", str(e)[:40])

CATS = {"instruction": instruction_chat, "math": math_reasoning, "physics": physics_science,
        "world": world_knowledge, "commonsense": commonsense}

def encode_to(f, src, budget, tok, t0, label, batch=4000):
    written, buf = 0, []
    def flush(b):
        nonlocal written
        for e in tok.encode_batch(b):
            ids = np.fromiter(e.ids, dtype=np.uint16, count=len(e.ids))
            f.write(ids.tobytes()); f.write(b"\x00\x00"); written += len(e.ids) + 1
    for txt in src:
        buf.append(txt)
        if len(buf) >= batch:
            flush(buf); buf = []
            if written - 0 and written % (batch * 50) < batch:
                print(f"    [{label}] {written/1e9:.2f}B / {budget/1e9:.2f}B  {written/max(time.time()-t0,1)/1e6:.1f}M tok/s", flush=True)
            if written >= budget: return written
    if buf: flush(buf)
    return written

def build(target):
    assert os.path.exists("data/bpe.json"), "need data/bpe.json (digit tokenizer)"
    tok = Tokenizer.from_file("data/bpe.json"); t0 = time.time(); counts = {}
    print(f"post-training corpus ~{target/1e9:.1f}B tokens (instruction ~40%, world fills remainder)", flush=True)
    with open("data/big_train.bin", "wb") as f:
        # instruction ~40% (OpenOrca oversampled 2x to reach it); math; tiny physics/commonsense
        # take all they have; WORLD (Wikipedia + FineWeb, unlimited) fills the rest to target.
        plan = [("instruction", _repeat(instruction_chat, 2), 0.40),
                ("math",        math_reasoning(),             0.20),
                ("physics",     _repeat(physics_science, 3),  0.12),
                ("commonsense", commonsense(),                0.10)]
        for cat, src, frac in plan:
            b = int(frac * target)
            n = encode_to(f, src, b, tok, t0, cat)
            counts[cat] = n; print(f"  {cat:12} {n/1e9:.3f}B  (cap {b/1e9:.2f}B)", flush=True)
        rem = max(0, target - sum(counts.values()))               # world fills to the target
        counts["world"] = encode_to(f, world_knowledge(), rem, tok, t0, "world")
        print(f"  {'world':12} {counts['world']/1e9:.3f}B  (fills remainder)", flush=True)
    # held-out valid: fresh FineWeb docs
    vt = 0
    with open("data/big_valid.bin", "wb") as f:
        for _, t in P.web_stream("sample-100BT"):
            ids = np.array(tok.encode(t).ids + [0], dtype=np.uint16); f.write(ids.tobytes()); vt += len(ids)
            if vt > 1_000_000: break
    tot = sum(counts.values())
    print(f"\nbuilt {tot/1e9:.2f}B tokens: " + " ".join(f"{k}={100*v/tot:.0f}%" for k, v in counts.items()), flush=True)
    print(f"valid {vt/1e6:.1f}M  in {(time.time()-t0)/60:.1f}m", flush=True)

if __name__ == "__main__":
    pa = argparse.ArgumentParser(); pa.add_argument("--tokens", type=float, default=5e9); a = pa.parse_args()
    build(int(a.tokens)); print("DONE.", flush=True)
    sys.stdout.flush(); os._exit(0)
