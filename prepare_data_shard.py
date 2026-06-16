"""
SHARDED parallel corpus builder — real producer parallelism. N worker processes
each stream a DISJOINT subset of FineWeb-Edu parquet shards (via
datasets.distributed.split_dataset_by_node), tokenize with the multithreaded Rust
tokenizer, and write their own web shard bin. Curated runs concurrently in its own
process. Main then concatenates all shards into data/big_train.bin. Held-out valid
comes from an extra disjoint shard (index = workers), so NO train/valid leak.

Reuses prepare_data.py's data sources + budget logic; trainer untouched.
Requires data/bpe.json.

  python3 prepare_data_shard.py --tokens 20e9 --web-config sample-100BT --workers 16
"""
import argparse, json, os, sys, time, glob, shutil
import multiprocessing as mp
import numpy as np
from tokenizers import Tokenizer
import prepare_data as P

os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")
TMP = "data/_shard"

def _encode_stream(rows, tok, f, budget, batch=2000):
    """rows: iterator of text strings. Tokenize in batches, write ids+0 sep."""
    written = 0; buf = []
    def flush(b):
        nonlocal written
        for e in tok.encode_batch(b):
            ids = np.fromiter(e.ids, dtype=np.uint16, count=len(e.ids))
            f.write(ids.tobytes()); f.write(b"\x00\x00"); written += len(e.ids) + 1
    for t in rows:
        buf.append(t)
        if len(buf) >= batch:
            flush(buf); buf = []
            if written >= budget: return written
    if buf: flush(buf)
    return written

def web_worker(rank, world, web_config, budget, batch, out):
    from datasets.distributed import split_dataset_by_node
    from datasets import load_dataset
    tok = Tokenizer.from_file("data/bpe.json")
    ds = load_dataset("HuggingFaceFW/fineweb-edu", web_config, split="train", streaming=True)
    ds = split_dataset_by_node(ds, rank=rank, world_size=world)
    def rows():
        for r in ds:
            t = (r.get("text") or "").strip()
            if len(t) > 300: yield t
    t0 = time.time()
    with open(out, "wb") as f:
        n = _encode_stream(rows(), tok, f, budget, batch)
    print(f"  [web{rank}] {n/1e6:.0f}M tok in {(time.time()-t0)/60:.1f}m -> {out}", flush=True)
    sys.stdout.flush(); os._exit(0)

def curated_worker(budget, batch, out):
    tok = Tokenizer.from_file("data/bpe.json")
    counts = {}
    def rows():
        for cat, text in P.curated():
            counts[cat] = counts.get(cat, 0) + 1; yield text
    t0 = time.time()
    with open(out, "wb") as f:
        n = _encode_stream(rows(), tok, f, budget, batch)
    print(f"  [curated] {n/1e6:.0f}M tok in {(time.time()-t0)/60:.1f}m  {counts}", flush=True)
    sys.stdout.flush(); os._exit(0)

def valid_worker(rank, world, web_config, out):
    """Disjoint shard (index=workers) -> held-out, never in any train shard."""
    from datasets.distributed import split_dataset_by_node
    from datasets import load_dataset
    tok = Tokenizer.from_file("data/bpe.json")
    ds = load_dataset("HuggingFaceFW/fineweb-edu", web_config, split="train", streaming=True)
    ds = split_dataset_by_node(ds, rank=rank, world_size=world)
    def rows():
        for r in ds:
            t = (r.get("text") or "").strip()
            if len(t) > 300: yield t
    with open(out, "wb") as f:
        n = _encode_stream(rows(), tok, f, 1_000_000, 200)
    print(f"  [valid] {n/1e6:.1f}M tok -> {out}", flush=True)
    sys.stdout.flush(); os._exit(0)

def build(target, web_config, workers, batch):
    assert os.path.exists("data/bpe.json"), "run prepare_data.py first to train data/bpe.json"
    cfg = P.size_for(176e6); VOCAB = cfg["vocab"]
    os.makedirs(TMP, exist_ok=True)
    for f in glob.glob(f"{TMP}/*"): os.remove(f)
    cur_budget = int(0.45 * target); story_budget = int(0.10 * target)
    web_budget = max(0, target - cur_budget - story_budget)
    per = web_budget // workers + 1
    print(f"budget {target/1e9:.1f}B | curated<= {cur_budget/1e9:.1f}B | web {web_budget/1e9:.1f}B "
          f"across {workers} workers ({per/1e9:.2f}B each) | shard {web_config}", flush=True)
    t0 = time.time()
    # valid uses world_size = workers+1, index = workers (disjoint from all train shards)
    world = workers + 1
    procs = [mp.Process(target=curated_worker, args=(cur_budget, batch, f"{TMP}/curated.bin"))]
    for i in range(workers):
        procs.append(mp.Process(target=web_worker, args=(i, world, web_config, per, batch, f"{TMP}/web_{i:03d}.bin")))
    procs.append(mp.Process(target=valid_worker, args=(workers, world, web_config, f"{TMP}/valid.bin")))
    for p in procs: p.start()
    for p in procs: p.join()
    # merge: curated + all web shards -> big_train.bin  (order irrelevant for LM pretraining)
    parts = [f"{TMP}/curated.bin"] + sorted(glob.glob(f"{TMP}/web_*.bin"))
    tot = 0
    with open("data/big_train.bin", "wb") as out:
        for part in parts:
            if os.path.exists(part):
                tot += os.path.getsize(part)
                with open(part, "rb") as f: shutil.copyfileobj(f, out, 1 << 24)
    shutil.copyfile(f"{TMP}/valid.bin", "data/big_valid.bin")
    vt = os.path.getsize("data/big_valid.bin") // 2
    print(f"MERGED train ~{tot/2/1e9:.3f}B tokens, valid ~{vt/1e6:.1f}M  in {(time.time()-t0)/60:.1f}m", flush=True)
    json.dump({"vocab": VOCAB, "d": cfg["d"], "heads": cfg["heads"], "layers": cfg["layers"],
               "mlp_mult": 4, "ctx": 256, "tokenizer": "data/bpe.json", "train_bin": "big_train",
               "valid_bin": "big_valid", "ckpt": "pragnosia.pt"}, open("pragnosia.json", "w"), indent=2)
    for f in glob.glob(f"{TMP}/*"): os.remove(f)
    print("wrote pragnosia.json", flush=True)

if __name__ == "__main__":
    mp.set_start_method("spawn")
    pa = argparse.ArgumentParser()
    pa.add_argument("--tokens", type=float, default=20e9)
    pa.add_argument("--web-config", default="sample-100BT")
    pa.add_argument("--workers", type=int, default=16)
    pa.add_argument("--batch", type=int, default=2000)
    a = pa.parse_args()
    build(int(a.tokens), a.web_config, a.workers, a.batch)
    print("DONE.", flush=True)
    sys.stdout.flush(); sys.stderr.flush(); os._exit(0)
