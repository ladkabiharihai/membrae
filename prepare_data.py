"""
Build the REBALANCED corpus for Pragnosia: stories only ~10%, the rest refined
world-knowledge + reasoning + multi-turn chat + mathematics + coding + grammar.
Trains BPE-16384, tokenizes to big_train/big_valid, writes pragnosia.json.

  python3 prepare_data.py
"""
import json, os, numpy as np
from datasets import load_dataset
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders

VOCAB = 16384
RAW = "data/corpus_big.txt"
SEP = "\n<|endoftext|>\n"

def qa(q, a, ctx=""):
    return f"Question: {q.strip()}\n" + (f"{ctx.strip()}\n" if ctx.strip() else "") + f"Answer: {a.strip()}"

def collect():
    """Yield (category, text) for every non-story document, capped per source."""
    # --- world knowledge: Simple Wikipedia (all of it) ---
    if os.path.exists("data/wiki_simple.txt"):
        for c in open("data/wiki_simple.txt").read().split("<|endoftext|>"):
            if len(c.strip()) > 200: yield "wiki", c.strip()
    # --- reasoning: OpenOrca (slice) + Alpaca + Dolly ---
    try:
        for r in load_dataset("Open-Orca/OpenOrca", split="train[:120000]"):
            if r["question"] and r["response"]: yield "reason", qa(r["question"], r["response"])
    except Exception as e: print("skip orca", str(e)[:40])
    for repo, im, ic, om in [("tatsu-lab/alpaca","instruction","input","output"),
                             ("databricks/databricks-dolly-15k","instruction","context","response")]:
        try:
            for r in load_dataset(repo, split="train"):
                if r[im] and r[om]: yield "reason", qa(r[im], r[om], r.get(ic) or "")
        except Exception as e: print("skip", repo, str(e)[:40])
    # --- mathematics: GSM8K (with reasoning) + Orca-Math ---
    try:
        for r in load_dataset("openai/gsm8k","main",split="train"):
            yield "math", qa(r["question"], r["answer"])
    except Exception as e: print("skip gsm8k", str(e)[:40])
    try:
        for r in load_dataset("microsoft/orca-math-word-problems-200k", split="train[:120000]"):
            yield "math", qa(r["question"], r["answer"])
    except Exception as e: print("skip orca-math", str(e)[:40])
    # --- coding ---
    for repo in ["sahil2801/CodeAlpaca-20k","iamtarun/python_code_instructions_18k_alpaca"]:
        try:
            for r in load_dataset(repo, split="train"):
                if r.get("output"): yield "code", qa(r["instruction"], r["output"], r.get("input") or "")
        except Exception as e: print("skip", repo, str(e)[:40])
    # --- multi-turn chat: OpenAssistant (reconstruct best conversation thread) ---
    try:
        ds = load_dataset("OpenAssistant/oasst1", split="train")
        msgs, kids = {}, {}
        for r in ds:
            msgs[r["message_id"]] = r
            kids.setdefault(r["parent_id"], []).append(r["message_id"])
        for root in kids.get(None, []):
            conv, cur = [], root
            while cur is not None:
                m = msgs[cur]
                role = "<user>" if m["role"] == "prompter" else "<assistant>"
                conv.append(f"{role} {' '.join(m['text'].split())}")
                ch = kids.get(cur, [])
                ch = sorted(ch, key=lambda c: (msgs[c]["rank"] if msgs[c]["rank"] is not None else 9))
                cur = ch[0] if ch else None
            if len(conv) >= 2: yield "chat", "\n".join(conv)
    except Exception as e: print("skip oasst", str(e)[:40])
    # --- english grammar: CoEdit (correction pairs) ---
    try:
        for r in load_dataset("grammarly/coedit", split="train"):
            if r.get("src") and r.get("tgt"):
                yield "grammar", f"Fix the grammar: {r['src'].strip()}\nCorrected: {r['tgt'].strip()}"
    except Exception as e: print("skip coedit", str(e)[:40])

def build_raw():
    counts, chars = {}, 0
    with open(RAW, "w") as f:
        for cat, text in collect():
            f.write(text + SEP); counts[cat] = counts.get(cat, 0) + 1; chars += len(text)
        # stories last: add exactly ~10% of the final corpus by characters
        story_budget = chars / 9.0           # stories will be 1/(9+1) = 10%
        used = 0; sc = 0
        if os.path.exists("data/ts_big.txt"):
            for c in open("data/ts_big.txt").read().split("<|endoftext|>"):
                c = c.strip()
                if not c: continue
                f.write(c + SEP); used += len(c); sc += 1
                if used >= story_budget: break
        counts["story"] = sc
    print("documents per category:", counts, flush=True)
    print(f"non-story chars={chars/1e6:.0f}M  story chars={used/1e6:.0f}M "
          f"(stories ~{100*used/(chars+used):.0f}%)", flush=True)

def tokenize():
    tok = Tokenizer(models.BPE(unk_token=None))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)
    tok.decoder = decoders.ByteLevel()
    tok.train([RAW], trainers.BpeTrainer(vocab_size=VOCAB, special_tokens=["<|endoftext|>"]))
    tok.save("data/bpe16384.json")
    docs = [c.strip() for c in open(RAW).read().split("<|endoftext|>") if c.strip()]
    rng = np.random.default_rng(0); rng.shuffle(docs)
    for name, chunk in [("big_valid", docs[:3000]), ("big_train", docs[3000:])]:
        ids = []
        for c in chunk: ids.extend(tok.encode(c).ids + [0])
        np.array(ids, dtype=np.uint16).tofile(f"data/{name}.bin")
        print(name, f"{len(ids):,}", "tokens", flush=True)

def write_config():
    json.dump({"vocab": VOCAB, "d": 1024, "heads": 16, "layers": 12, "ctx": 256,
               "tokenizer": "data/bpe16384.json", "train_bin": "big_train",
               "valid_bin": "big_valid", "ckpt": "pragnosia_168m.pt"},
              open("pragnosia.json", "w"), indent=2)
    print("wrote pragnosia.json", flush=True)

if __name__ == "__main__":
    build_raw(); tokenize(); write_config()
    print("DONE.", flush=True)
