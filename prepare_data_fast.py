"""
Parallel corpus builder for Pragnosia — same data sources & budget logic as
prepare_data.py, but tokenization runs through the Rust tokenizer's multithreaded
`encode_batch`, saturating all CPU cores (≈10-30x faster than the serial per-doc
encode). It REUSES prepare_data.py's proven sources (curated/web_stream/size_for),
so only the tokenize-and-write hot loop is new. The trainer is untouched.

Requires data/bpe.json (the trained tokenizer) to already exist.

  python3 prepare_data_fast.py --tokens 20e9 --web-config sample-100BT
"""
import argparse, json, os, sys, time
import numpy as np
from tokenizers import Tokenizer
import prepare_data as P            # curated(), web_stream(), size_for(), SEP

os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")

def batched(it, n):
    buf = []
    for x in it:
        buf.append(x)
        if len(buf) >= n:
            yield buf; buf = []
    if buf:
        yield buf

def encode_to(f, it, budget, tok, counts, label, t0, batch=4000):
    """Stream (cat,text) pairs, tokenize each batch on ALL cores, write ids+sep
    in order, stop once `budget` tokens written. Returns tokens written."""
    written = last = 0
    for chunk in batched(it, batch):
        encs = tok.encode_batch([t for _, t in chunk])      # multithreaded (rayon)
        for (cat, _), e in zip(chunk, encs):
            ids = np.fromiter(e.ids, dtype=np.uint16, count=len(e.ids))
            f.write(ids.tobytes()); f.write(b"\x00\x00")     # uint16 0 separator
            written += len(e.ids) + 1
            counts[cat] = counts.get(cat, 0) + 1
        if written - last >= 200_000_000:
            last = written; el = time.time() - t0
            print(f"  [{label}] {written/1e9:.2f}B tok  {written/max(el,1)/1e6:.2f}M tok/s  "
                  f"{el/60:.1f}m", flush=True)
        if written >= budget:
            break
    return written

def build(target, web_config, batch):
    cfg = P.size_for(176e6); VOCAB = cfg["vocab"]
    assert os.path.exists("data/bpe.json"), "run prepare_data.py first to train data/bpe.json"
    tok = Tokenizer.from_file("data/bpe.json")
    print(f"target ~20B-style budget {target/1e9:.2f}B tok | web shard {web_config} | "
          f"arch d={cfg['d']} layers={cfg['layers']} vocab={VOCAB} | batch={batch}", flush=True)
    counts = {}; t0 = time.time()
    web_it = iter(P.web_stream(web_config))               # ONE web iterator...
    with open("data/big_train.bin", "wb") as f:
        cur_tok = encode_to(f, P.curated(), int(0.45 * target), tok, counts, "curated", t0, batch)
        story_budget = int(0.10 * target)
        web_budget = max(0, target - cur_tok - story_budget)
        wtok = encode_to(f, web_it, web_budget, tok, counts, "web", t0, batch)
        stok = 0
        if os.path.exists("data/ts_big.txt"):
            def story_src():
                for c in open("data/ts_big.txt").read().split("<|endoftext|>"):
                    c = c.strip()
                    if c: yield "story", c
            stok = encode_to(f, story_src(), story_budget, tok, counts, "story", t0, batch)
    total = cur_tok + wtok + stok
    # held-out valid: CONTINUE the same web iterator => fresh docs never seen in train
    with open("data/big_valid.bin", "wb") as f:
        vtok = encode_to(f, web_it, 1_000_000, tok, {}, "valid", t0, batch=500)
    print(f"built: {counts}", flush=True)
    print(f"train tokens ~{total/1e9:.3f}B  (curated {cur_tok/1e9:.3f}B + web {wtok/1e9:.3f}B + "
          f"story {stok/1e9:.3f}B = {100*stok/max(total,1):.0f}%), valid {vtok/1e6:.1f}M  "
          f"in {(time.time()-t0)/60:.1f}m", flush=True)
    json.dump({"vocab": VOCAB, "d": cfg["d"], "heads": cfg["heads"], "layers": cfg["layers"],
               "mlp_mult": 4, "ctx": 256, "tokenizer": "data/bpe.json", "train_bin": "big_train",
               "valid_bin": "big_valid", "ckpt": "pragnosia.pt"}, open("pragnosia.json", "w"), indent=2)
    print("wrote pragnosia.json", flush=True)

if __name__ == "__main__":
    pa = argparse.ArgumentParser()
    pa.add_argument("--tokens", type=float, default=20e9)
    pa.add_argument("--web-config", default="sample-100BT")
    pa.add_argument("--batch", type=int, default=4000)
    a = pa.parse_args()
    build(int(a.tokens), a.web_config, a.batch)
    print("DONE.", flush=True)
    # Bins + json are already flushed/closed; skip Python finalization to dodge the
    # benign rayon/datasets thread-teardown segfault at interpreter shutdown.
    sys.stdout.flush(); sys.stderr.flush(); os._exit(0)
