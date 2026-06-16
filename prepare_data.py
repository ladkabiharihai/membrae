"""
Scale-aware corpus builder for Pragnosia. You give it a target model size; it
sizes the architecture (writes pragnosia.json), computes a Chinchilla-style token
budget (~18 tokens/param), pulls the curated reasoning/knowledge/math/code/chat/
grammar sets, and STREAMS the remainder from FineWeb-Edu (high-quality web text) up
to the budget. Stories are capped at ~10%. Tokenization is incremental, so it can
build anything from ~200M tokens (laptop) to tens of billions (for a 3B model on a
big machine) without holding the corpus in memory.

  python3 prepare_data.py                       # default 176M model (~3.5B-token budget)
  python3 prepare_data.py --params 176e6 --laptop   # small ~200M-token build for a laptop
  python3 prepare_data.py --params 1e9          # ~1B model  (~18B tokens)
  python3 prepare_data.py --params 3e9          # ~3B model  (~54B tokens)
"""
import argparse, json, os, numpy as np
from datasets import load_dataset
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders

SEP = "\n<|endoftext|>\n"; RAW_SAMPLE = "data/tok_sample.txt"
MATH_OVERSAMPLE = 3          # repeat math/CoT this many times in the corpus (a small model
                            # needs to see step-by-step reasoning many times to pick it up)

def size_for(params):
    """Pick (d, heads, layers, vocab) so the model ~= target params, and a token budget."""
    table = [   # (max_params, d, heads, layers, vocab)
        (3.0e8, 1024, 16, 12, 16384),    # ~176M
        (7.0e8, 1536, 16, 16, 32768),    # ~500M
        (1.5e9, 2048, 16, 20, 32768),    # ~1B
        (4.0e9, 2560, 20, 32, 32768),    # ~3B
    ]
    for cap, d, h, l, v in table:
        if params <= cap:
            return dict(d=d, heads=h, layers=l, vocab=v, target_tokens=int(18 * params))
    d, h, l, v = table[-1][1:]
    return dict(d=d, heads=h, layers=l, vocab=v, target_tokens=int(18 * params))

def qa(q, a, ctx=""):
    return f"Question: {q.strip()}\n" + (f"{ctx.strip()}\n" if ctx.strip() else "") + f"Answer: {a.strip()}"

def curated(cap_each=None):
    """Yield (category, text) from curated sources. MATH + chain-of-thought goes
    FIRST and is OVERSAMPLED (so a small model sees step-by-step reasoning many
    times); the huge general set (OpenOrca) is LAST so it only fills leftover budget
    after every smaller high-signal set is already in."""
    def capped(it, n=cap_each):
        for i, x in enumerate(it):
            if n and i >= n: break
            yield x
    # ---- #4/#5 MATH + chain-of-thought: the reasoning lever, oversampled, FIRST ----
    def math_cot():
        try:
            for r in capped(load_dataset("openai/gsm8k","main",split="train")):
                yield "math", qa(r["question"], r["answer"])          # answers include worked steps
        except Exception as e: print("skip gsm8k", str(e)[:40])
        try:
            for r in capped(load_dataset("microsoft/orca-math-word-problems-200k", split="train")):
                yield "math", qa(r["question"], r["answer"])
        except Exception as e: print("skip orca-math", str(e)[:40])
        try:
            for r in capped(load_dataset("meta-math/MetaMathQA", split="train")):
                q = r.get("query") or r.get("original_question"); a = r.get("response")
                if q and a: yield "math", qa(q, a)                    # CoT solutions
        except Exception as e: print("skip metamath", str(e)[:40])
    for _ in range(1 if cap_each else MATH_OVERSAMPLE):               # don't oversample the tokenizer sample
        for cat, t in math_cot(): yield cat, t
    # ---- smaller high-signal sets, kept ahead of the big filler ----
    for repo, im, ic, om in [("tatsu-lab/alpaca","instruction","input","output"),
                             ("databricks/databricks-dolly-15k","instruction","context","response")]:
        try:
            for r in load_dataset(repo, split="train"):
                if r[im] and r[om]: yield "reason", qa(r[im], r[om], r.get(ic) or "")
        except Exception as e: print("skip", repo, str(e)[:40])
    for repo in ["sahil2801/CodeAlpaca-20k","iamtarun/python_code_instructions_18k_alpaca"]:
        try:
            for r in load_dataset(repo, split="train"):
                if r.get("output"): yield "code", qa(r["instruction"], r["output"], r.get("input") or "")
        except Exception as e: print("skip", repo, str(e)[:40])
    try:
        ds = load_dataset("OpenAssistant/oasst1", split="train"); msgs, kids = {}, {}
        for r in ds: msgs[r["message_id"]]=r; kids.setdefault(r["parent_id"],[]).append(r["message_id"])
        for root in kids.get(None, []):
            conv, cur = [], root
            while cur is not None:
                m=msgs[cur]; conv.append(("<user> " if m["role"]=="prompter" else "<assistant> ")+" ".join(m["text"].split()))
                ch=sorted(kids.get(cur,[]),key=lambda c:(msgs[c]["rank"] if msgs[c]["rank"] is not None else 9)); cur=ch[0] if ch else None
            if len(conv)>=2: yield "chat", "\n".join(conv)
    except Exception as e: print("skip oasst", str(e)[:40])
    try:
        for r in load_dataset("grammarly/coedit", split="train"):
            if r.get("src") and r.get("tgt"): yield "grammar", f"Fix the grammar: {r['src'].strip()}\nCorrected: {r['tgt'].strip()}"
    except Exception as e: print("skip coedit", str(e)[:40])
    if os.path.exists("data/wiki_simple.txt"):
        for c in open("data/wiki_simple.txt").read().split("<|endoftext|>"):
            if len(c.strip()) > 200: yield "wiki", c.strip()
    # ---- big general reasoning/instruction filler, LAST (fills remaining curated budget) ----
    try:
        for r in capped(load_dataset("Open-Orca/OpenOrca", split="train", streaming=True)):
            if r.get("question") and r.get("response"): yield "reason", qa(r["question"], r["response"])
    except Exception as e: print("skip orca", str(e)[:40])

def web_stream(config="sample-10BT"):
    """High-quality educational web text, streamed (scales to ~1.3T tokens).
    `config` selects the FineWeb-Edu shard: sample-10BT (~10B tok), sample-100BT
    (~100B), sample-350BT, or default (~1.3T). Pick one big enough for the budget."""
    ds = load_dataset("HuggingFaceFW/fineweb-edu", config, split="train", streaming=True)
    for r in ds:
        t = (r.get("text") or "").strip()
        if len(t) > 300: yield "web", t

def build(params, laptop, tokens=0, web_config="sample-10BT"):
    cfg = size_for(params); VOCAB = cfg["vocab"]; target = cfg["target_tokens"]
    if laptop: target = min(target, 200_000_000)
    if tokens: target = int(tokens)        # decouple token budget from model size
    approx_tok = lambda s: len(s) // 4                 # ~4 chars/token estimate
    print(f"target model ~{params/1e6:.0f}M params -> d={cfg['d']} layers={cfg['layers']} "
          f"vocab={VOCAB}; token budget ~{target/1e9:.2f}B", flush=True)

    # 1. tokenizer: train on a representative SAMPLE (curated + a little web)
    if not os.path.exists("data/bpe.json"):
        with open(RAW_SAMPLE, "w") as f:
            chars = 0
            for _, t in curated(cap_each=20000):
                f.write(t + SEP); chars += len(t)
                if chars > 300_000_000: break
            wc = 0
            for _, t in web_stream(web_config):
                f.write(t + SEP); wc += len(t)
                if wc > 200_000_000: break
        tok = Tokenizer(models.BPE(unk_token=None))
        # #3 DIGIT-AWARE: split every digit into its own token so numbers are never
        # merged into ragged multi-digit chunks -> place-value aligned, arithmetic learnable.
        tok.pre_tokenizer = pre_tokenizers.Sequence([
            pre_tokenizers.ByteLevel(add_prefix_space=True, use_regex=True),
            pre_tokenizers.Digits(individual_digits=True),     # isolate each digit (clean roundtrip)
        ])
        tok.decoder = decoders.ByteLevel()
        tok.train([RAW_SAMPLE], trainers.BpeTrainer(vocab_size=VOCAB, special_tokens=["<|endoftext|>"]))
        tok.save("data/bpe.json"); os.remove(RAW_SAMPLE)
        print(f"trained BPE-{VOCAB}", flush=True)
    tok = Tokenizer.from_file("data/bpe.json")

    # 2. incremental tokenization to bins, stories capped at 10%, web fills to budget
    def write_stream(path, source, budget):
        n = 0
        with open(f"data/{path}.bin", "wb") as f:
            for cat, text in source():
                ids = np.array(tok.encode(text).ids + [0], dtype=np.uint16)
                f.write(ids.tobytes()); n += len(ids)
                if n >= budget: break
        return n
    counts = {}
    # curated first (a few B tokens at most), then web to fill, then stories ≤10%
    cur_tok = 0
    with open("data/big_train.bin", "wb") as f:
        for cat, text in curated():
            ids = np.array(tok.encode(text).ids + [0], dtype=np.uint16); f.write(ids.tobytes())
            cur_tok += len(ids); counts[cat] = counts.get(cat, 0) + 1
            if cur_tok >= 0.45 * target: break          # cap curated at ~45% of budget
        story_budget = 0.10 * target; web_budget = target - cur_tok - story_budget
        wtok = 0
        for cat, text in web_stream(web_config):
            ids = np.array(tok.encode(text).ids + [0], dtype=np.uint16); f.write(ids.tobytes())
            wtok += len(ids); counts["web"] = counts.get("web", 0) + 1
            if wtok >= web_budget: break
        stok = 0
        if os.path.exists("data/ts_big.txt"):
            for c in open("data/ts_big.txt").read().split("<|endoftext|>"):
                c = c.strip()
                if not c: continue
                ids = np.array(tok.encode(c).ids + [0], dtype=np.uint16); f.write(ids.tobytes())
                stok += len(ids); counts["story"] = counts.get("story", 0) + 1
                if stok >= story_budget: break
    total = cur_tok + wtok + stok
    # small held-out valid from the web stream (fresh docs)
    vtok = 0
    with open("data/big_valid.bin", "wb") as f:
        for cat, text in web_stream(web_config):
            ids = np.array(tok.encode(text).ids + [0], dtype=np.uint16); f.write(ids.tobytes()); vtok += len(ids)
            if vtok > 1_000_000: break
    print(f"built: {counts}", flush=True)
    print(f"train tokens ~{total/1e9:.2f}B  (curated {cur_tok/1e9:.2f}B + web {wtok/1e9:.2f}B + "
          f"story {stok/1e9:.2f}B = {100*stok/max(total,1):.0f}%), valid {vtok/1e6:.1f}M", flush=True)

    json.dump({"vocab": VOCAB, "d": cfg["d"], "heads": cfg["heads"], "layers": cfg["layers"],
               "mlp_mult": 4, "ctx": 256, "tokenizer": "data/bpe.json", "train_bin": "big_train",
               "valid_bin": "big_valid", "ckpt": "pragnosia.pt"}, open("pragnosia.json", "w"), indent=2)
    print("wrote pragnosia.json", flush=True)

if __name__ == "__main__":
    pa = argparse.ArgumentParser()
    pa.add_argument("--params", type=float, default=176e6, help="target model size (e.g. 1e9, 3e9)")
    pa.add_argument("--laptop", action="store_true", help="cap the build at ~200M tokens")
    pa.add_argument("--tokens", type=float, default=0, help="override token budget, decoupled from model size (e.g. 20e9)")
    pa.add_argument("--web-config", default="sample-10BT", help="FineWeb-Edu shard: sample-10BT/sample-100BT/sample-350BT/default")
    a = pa.parse_args()
    build(a.params, a.laptop, tokens=a.tokens, web_config=a.web_config)
    print("DONE.", flush=True)
