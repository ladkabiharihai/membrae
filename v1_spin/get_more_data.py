"""Acquire a BALANCED fresh corpus to expand the 176.6B training data (only ~7% has been trained on, so this
adds NEW signal, esp. the general-web fluency we lag on). Same BPE tokenizer + <user>/<assistant> chat format
as the existing corpus, so it mixes cleanly. Streams from HF (no full download), tokenizes pipelined (Rust,
GIL-released), writes uint16 + \x00\x00 EOS to /mnt/kv_cache. CPU/network only -> does not touch GPU training.

  python3 get_more_data.py --tokens 60   # ~60B tokens, balanced
"""
import argparse, os, time, threading, queue as _queue
import numpy as np
from tokenizers import Tokenizer
from datasets import load_dataset

OUT = "/mnt/kv_cache/pragnosia_data"
os.environ.setdefault("HF_HOME", "/mnt/kv_cache/hf_home")
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"


def _txt(x):
    return x.strip() if isinstance(x, str) else ""


def fineweb_edu():
    """Best general web text (educational-quality filtered) — the fluency lever (LAMBADA/HellaSwag).
    sample-100BT = ~100B tokens (10x the 10BT config that ran dry at 9.5B)."""
    for r in load_dataset("HuggingFaceFW/fineweb-edu", "sample-100BT", split="train", streaming=True):
        t = _txt(r.get("text"))
        if len(t) > 200: yield t[:16000]


def _messages_stream(ds, cfg=None, split="train", key="messages"):
    """Turn a multi-turn `messages` list into one <user>/<assistant> transcript."""
    for r in load_dataset(ds, cfg, split=split, streaming=True):
        msgs = r.get(key) or r.get("conversation") or r.get("conversations")
        if not msgs: continue
        parts = []
        for m in msgs:
            role = (m.get("role") or m.get("from") or "").lower()
            content = _txt(m.get("content") or m.get("value") or "")
            if not content: continue
            tag = "assistant" if role in ("assistant", "gpt", "bot") else "user"
            parts.append(f"<{tag}> {content}")
        if len(parts) >= 2:
            yield "\n".join(parts)[:16000]


def ultrachat():   # clean, high-quality multi-turn chat (the standard)
    yield from _messages_stream("HuggingFaceH4/ultrachat_200k", split="train_sft")

def smoltalk():    # curated diverse conversation/instruction mix
    yield from _messages_stream("HuggingFaceTB/smoltalk", "all")

def lmsys():       # real-user multi-turn conversations
    yield from _messages_stream("lmsys/lmsys-chat-1m", key="conversation")

def tulu():        # top-tier curated instruction/chat mixture
    yield from _messages_stream("allenai/tulu-3-sft-mixture")


def wildchat():    # real-world multi-turn user<->assistant conversations (1M, larger than ultrachat)
    yield from _messages_stream("allenai/WildChat-1M", key="conversation")

def cosmopedia():  # large synthetic textbook/web knowledge (world knowledge, billions of tokens)
    for r in load_dataset("HuggingFaceTB/cosmopedia-100k", split="train", streaming=True):
        t = _txt(r.get("text"))
        if len(t) > 200: yield t[:16000]

# web-heavy for fluency/world-knowledge; chat sources drained fully then web backfills to the target.
# budgets are CAPS; small sets finish early, fineweb (100B available) absorbs the target.
SOURCES = {
    "fineweb-edu":   (fineweb_edu, 0.70),   # ~100B available -> the bulk (general web / world knowledge)
    "wildchat":      (wildchat,    0.12),   # real multi-turn chat (large)
    "ultrachat":     (ultrachat,   0.08),   # clean multi-turn chat
    "smoltalk":      (smoltalk,    0.06),   # diverse conversation/instruction
    "tulu3":         (tulu,        0.04),   # curated instruction/chat
}


def encode_fast(f, src, budget, tok, t0, label, batch=2000):
    q = _queue.Queue(maxsize=6); done = threading.Event()
    def producer():
        buf = []
        for txt in src:
            if done.is_set(): return
            buf.append(txt)
            if len(buf) >= batch: q.put(buf); buf = []
        if buf: q.put(buf)
        q.put(None)
    threading.Thread(target=producer, daemon=True).start()
    written = nb = 0
    while True:
        b = q.get()
        if b is None: break
        try:
            enc = tok.encode_batch(b)
        except Exception:
            continue
        for e in enc:
            f.write(np.asarray(e.ids, dtype=np.uint16).tobytes()); f.write(b"\x00\x00")
            written += len(e.ids) + 1
        nb += 1
        if nb % 25 == 0:
            print(f"    [{label}] {written/1e9:.3f}B / {budget/1e9:.2f}B  {written/max(time.time()-t0,1)/1e6:.2f}M tok/s", flush=True)
        if written >= budget:
            done.set()
            break
    return written


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", type=float, default=60.0, help="target billions of tokens")
    a = ap.parse_args()
    total = a.tokens * 1e9
    tok = Tokenizer.from_file("data/bpe.json"); t0 = time.time()
    out = f"{OUT}/expanded2_train.bin"
    print(f"[expand] target {a.tokens:.0f}B balanced tokens -> {out}", flush=True)
    grand = 0
    with open(out, "wb") as f:
        for name, (fn, frac) in SOURCES.items():
            budget = int(total * frac)
            print(f"  == {name}: target {budget/1e9:.1f}B ({frac*100:.0f}%) ==", flush=True)
            try:
                w = encode_fast(f, fn(), budget, tok, t0, name)
                grand += w
                print(f"  == {name} done: {w/1e9:.2f}B (total {grand/1e9:.2f}B) ==", flush=True)
            except Exception as e:
                print(f"  == {name} FAILED: {str(e)[:120]} -- skipping ==", flush=True)
    print(f"[expand] DONE: {grand/1e9:.2f}B tokens -> {out} ({(time.time()-t0)/3600:.1f}h)", flush=True)
    open(f"{OUT}/expanded2_DONE.flag", "w").write(f"{grand}\n")


if __name__ == "__main__":
    main()
