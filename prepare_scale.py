"""
LARGE knowledge-scaling corpus, written to the 3.2 TB NVMe (/mnt/kv_cache) so disk is
not the limit. Instruction + math are capped (small sets, ~8B each, oversampled); the
bulk is knowledge: Cosmopedia (all configs) + Wikipedia + FineWeb-Edu (sample-350BT).
Same digit tokenizer. Runs continuously toward --tokens; the .bin is a valid corpus at
ANY size, so it can be stopped whenever and trained on. Symlinked into data/ for the trainer.

  python3 prepare_scale.py --tokens 300e9
"""
import argparse, os, sys, time, random, itertools, ast, threading, queue as _queue
import numpy as np
from datasets import load_dataset
from tokenizers import Tokenizer
import prepare_posttrain as PT          # instruction_chat, math_reasoning, encode_to, _repeat
import prepare_data as P                # web_stream

os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")
OUT = "/mnt/kv_cache/pragnosia_data"

# arXiv categories -> the physics domains the user wants at scale:
#   astro-ph = astrophysics+cosmology, gr-qc = general relativity/cosmology,
#   hep-ph/hep-th = particle physics, physics.* = general physics, q-bio = biology
PHYS_CATS = ("astro-ph", "gr-qc", "hep-ph", "hep-th", "hep-ex", "nucl",
             "physics", "cond-mat", "quant-ph", "q-bio")

def _qa_stream(ds, fa, fb):
    """Yield <user>/<assistant> pairs from a code/science QA dataset, resilient to bad rows."""
    try:
        for r in load_dataset(ds, split="train", streaming=True):
            a = (r.get(fa) or "").strip(); b = (r.get(fb) or "").strip()
            if a and b: yield f"<user> {a}\n<assistant> {b}"
    except Exception as e: print("skip", ds, str(e)[:40], flush=True)

def code_stream():
    """Coding at scale. Bulk = glaive-code-assistant (~140K large Q&A) + Magicoder/evol;
    codeparrot-clean was dropped (streams die on a JSON-parse bad row)."""
    yield from _qa_stream("glaiveai/glaive-code-assistant", "question", "answer")
    yield from _qa_stream("ise-uiuc/Magicoder-OSS-Instruct-75K", "problem", "solution")
    yield from _qa_stream("theblackcat102/evol-codealpaca-v1", "instruction", "output")
    yield from _qa_stream("sahil2801/CodeAlpaca-20k", "instruction", "output")

def science_stream():
    """Physics / astrophysics / particle physics / cosmology / biology / chemistry, at scale.
    BULK FIRST = common-pile arXiv full papers (~11K tok/row, the scale source) + science wiki +
    arxiver; camel-ai clean domain Q&A LAST (tiny ~30M tok and 429-prone, so it must never gate
    the bulk). (arxiv-abstracts-2021 dropped: dies on a JSON bad row.)"""
    # clip is GENEROUS for full arXiv papers (~66K chars / ~16K tok each): trainer samples
    # 256-tok windows, so long docs are fine and multiply science volume ~8x vs a 6K clip.
    for ds, field, clip in [("common-pile/arxiv_papers_filtered", "text", 60000),
                            ("millawell/wikipedia_field_of_science", "text", 8000),
                            ("neuralwork/arxiver", "abstract", 6000)]:
        try:
            for r in load_dataset(ds, split="train", streaming=True):
                t = (r.get(field) or r.get("text") or "").strip()
                if len(t) > 300: yield t[:clip]
        except Exception as e: print("skip", ds, str(e)[:40], flush=True)
    for dom in ["physics", "chemistry", "biology"]:   # seasoning, last so a 429 can't block the bulk
        yield from _qa_stream2(f"camel-ai/{dom}", "message_1", "message_2")

def _qa_stream2(ds, fa, fb):
    try:
        for r in load_dataset(ds, split="train", streaming=True):
            a = (r.get(fa) or "").strip(); b = (r.get(fb) or "").strip()
            if a and b: yield f"<user> {a}\n<assistant> {b}"
    except Exception as e: print("skip", ds, str(e)[:40], flush=True)

# ---------------- "GI reasoning" section -------------------------------------
# Family/kinship + abstract pattern (visual-style) are SYNTHESIZED (CLUTRR is dead),
# so they are guaranteed correct and on-target; the rest stream from HF reasoning sets.
_M = "James John Robert Michael William David Tom Liam Noah Henry Jack Leo Sam Paul Mark George Carl Adam Eric Luke".split()
_F = "Mary Linda Sara Emma Olivia Ava Mia Anna Grace Lucy Rose Jane Kate Nina Ruth Beth Clara Iris Lily Nora".split()

def _kin(p, q, gy):                       # p,q = steps subject/target to LCA; gy = target gender
    g = lambda m, f: m if gy == "m" else f
    if p == 0:                            # target is a descendant of the subject
        if q == 1: return g("son", "daughter")
        return "great-" * (q - 2) + g("grandson", "granddaughter")
    if q == 0:                            # target is an ancestor of the subject
        if p == 1: return g("father", "mother")
        return "great-" * (p - 2) + g("grandfather", "grandmother")
    if p == 1 and q == 1: return g("brother", "sister")
    if p == 2 and q == 1: return g("uncle", "aunt")
    if p == 1 and q == 2: return g("nephew", "niece")
    if p == 2 and q == 2: return "cousin"
    return None                           # skip deeper lateral (grand-uncle, 2nd cousin) to stay exact

def family_stream():
    """Synthetic kinship reasoning: state parent facts, ask the composed relation (true via LCA)."""
    while True:
        n = random.randint(5, 9); names = random.sample(_M + _F, n)
        gen = {nm: ("m" if nm in _M else "f") for nm in names}
        parent = {}; depth = {names[0]: 0}
        for i in range(1, n):
            par = random.choice(names[:i]); parent[names[i]] = par; depth[names[i]] = depth[par] + 1
        facts = [f"{p} is the {'father' if gen[p]=='m' else 'mother'} of {c}." for c, p in parent.items()]
        random.shuffle(facts)
        def anc(x):
            path = [x]
            while x in parent: x = parent[x]; path.append(x)
            return path
        pairs = [(a, b) for a in names for b in names if a != b]; random.shuffle(pairs); asked = 0
        for a, b in pairs:
            pa, pb = anc(a), anc(b); common = set(pa) & set(pb)
            if not common: continue
            lca = max(common, key=lambda x: depth[x]); rel = _kin(pa.index(lca), pb.index(lca), gen[b])
            if rel is None: continue
            yield " ".join(facts) + f"\nQuestion: Who is {b} to {a}? Answer: {b} is {a}'s {rel}."
            asked += 1
            if asked >= 3: break

def pattern_stream():
    """Synthetic abstract/visual-style reasoning: number/letter sequences + analogies."""
    while True:
        k = random.choice(["arith", "geom", "fib", "letter", "analogy"])
        if k == "arith":
            a = random.randint(1, 9); d = random.randint(2, 9); s = [a + d * i for i in range(5)]
            yield f"Find the next number: {', '.join(map(str, s[:4]))}, ? Answer: {s[4]}."
        elif k == "geom":
            a = random.randint(2, 4); r = random.choice([2, 3]); s = [a * r ** i for i in range(4)]
            yield f"Find the next number: {', '.join(map(str, s[:3]))}, ? Answer: {s[3]}."
        elif k == "fib":
            s = [random.randint(1, 4), random.randint(1, 4)]
            for _ in range(3): s.append(s[-1] + s[-2])
            yield f"Find the next number: {', '.join(map(str, s[:4]))}, ? Answer: {s[4]}."
        elif k == "letter":
            st = random.randint(0, 14); sp = random.randint(1, 3); s = [chr(65 + st + sp * i) for i in range(5)]
            yield f"Find the next letter: {', '.join(s[:4])}, ? Answer: {s[4]}."
        else:
            a = random.randint(2, 9); m = random.randint(2, 4)
            yield f"Analogy: {a} is to {a*m} as {a+1} is to ? Answer: {(a+1)*m}."

def _grid(g): return "\n".join("".join(str(c) for c in row) for row in g)
def arc_stream():
    """ARC-AGI abstract visual reasoning — only small grids that fit a 256 context."""
    try:
        for r in load_dataset("lordspline/arc-agi", split="training", streaming=True):
            tr, te = r.get("train"), r.get("test")
            tr = ast.literal_eval(tr) if isinstance(tr, str) else tr
            te = ast.literal_eval(te) if isinstance(te, str) else te
            if not tr or not te: continue
            def sm(x): return len(x) <= 6 and all(len(row) <= 6 for row in x)
            if not all(sm(p["input"]) and sm(p["output"]) for p in tr[:3]): continue
            if not sm(te[0]["input"]) or "output" not in te[0] or not sm(te[0]["output"]): continue
            s = "Find the pattern and complete the last grid.\n"
            for p in tr[:3]: s += "Input:\n" + _grid(p["input"]) + "\nOutput:\n" + _grid(p["output"]) + "\n"
            s += "Input:\n" + _grid(te[0]["input"]) + "\nOutput:\n" + _grid(te[0]["output"])
            if len(s) < 900: yield s
    except Exception as e: print("skip arc", str(e)[:40], flush=True)

def hf_reasoning():
    try:                                   # bAbI: deduction/induction/counting/path/positional(spatial)
        for r in load_dataset("Muennighoff/babi", split="train", streaming=True):
            p = (r.get("passage") or "").strip(); q = (r.get("question") or "").strip(); a = str(r.get("answer") or "").strip()
            if p and q and a: yield f"{p}\nQuestion: {q} Answer: {a}"
    except Exception as e: print("skip babi", str(e)[:40], flush=True)
    try:                                   # logical rule-based deduction
        for r in load_dataset("tasksource/ruletaker", split="train", streaming=True):
            c = (r.get("context") or "").strip(); q = (r.get("question") or "").strip(); a = str(r.get("label") or "").strip()
            if c and q: yield f"{c}\nQuestion: {q} Answer: {a}"
    except Exception as e: print("skip ruletaker", str(e)[:40], flush=True)
    for sp in ["train_r1", "train_r2", "train_r3"]:   # adversarial NLI with rationale
        try:
            for r in load_dataset("facebook/anli", split=sp, streaming=True):
                pr = (r.get("premise") or "").strip(); h = (r.get("hypothesis") or "").strip()
                lab = {0: "entailment", 1: "neutral", 2: "contradiction"}.get(r.get("label"), "")
                rs = (r.get("reason") or "").strip()
                if pr and h: yield f"Premise: {pr}\nHypothesis: {h}\nRelation: {lab}." + (f" Because: {rs}" if rs else "")
        except Exception as e: print("skip anli", sp, str(e)[:30], flush=True)
    for cfg in ["logical_deduction_seven_objects", "logical_deduction_five_objects", "tracking_shuffled_objects_three_objects",
                "reasoning_about_colored_objects", "geometric_shapes", "navigate", "date_understanding", "temporal_sequences"]:
        try:                               # BIG-Bench-Hard: logic / spatial / object tracking
            for r in load_dataset("lukaemon/bbh", cfg, split="test", streaming=True):
                i = (r.get("input") or "").strip(); t = str(r.get("target") or "").strip()
                if i: yield f"{i}\nAnswer: {t}"
        except Exception as e: print("skip bbh", cfg, str(e)[:30], flush=True)
    try:                                   # STEM/logic CoT
        for r in load_dataset("garage-bAInd/Open-Platypus", split="train", streaming=True):
            ins = (r.get("instruction") or "").strip(); o = (r.get("output") or "").strip()
            if ins and o: yield f"<user> {ins}\n<assistant> {o}"
    except Exception as e: print("skip platypus", str(e)[:40], flush=True)
    try:                                   # long reasoning traces
        for r in load_dataset("open-thoughts/OpenThoughts-114k", split="train", streaming=True):
            cv = r.get("conversations") or []
            if len(cv) >= 2:
                u = (cv[0].get("value") or cv[0].get("content") or "").strip()
                a = (cv[1].get("value") or cv[1].get("content") or "").strip()
                if u and a: yield f"<user> {u}\n<assistant> {a}"[:6000]
    except Exception as e: print("skip openthoughts", str(e)[:40], flush=True)

def reasoning_stream():
    """HF reasoning sets + synthetic family/kinship + abstract pattern + ARC visual, INTERLEAVED.
    HF sets (weight) lead while available; the infinite synthetic streams fill any larger cap."""
    return interleave([(hf_reasoning(), 4), (family_stream(), 2), (pattern_stream(), 1), (arc_stream(), 1)])

def cot_stream():
    """Long multi-step CHAIN-OF-THOUGHT at scale — the model's weakest axis. Bulk is
    OpenMathReasoning (~3.2M problems, full ~5K-token worked solutions) + OpenThoughts2 reasoning
    traces + OpenMathInstruct-2 (full, uncapped) + NuminaMath-CoT. Clipped at 16K chars (~4K tok)
    so the full reasoning CHAIN survives (the chain is the signal); trainer samples 256-tok windows."""
    try:
        for r in load_dataset("nvidia/OpenMathReasoning", split="cot", streaming=True):
            q = (r.get("problem") or "").strip(); a = (r.get("generated_solution") or "").strip()
            if q and a: yield f"<user> {q}\n<assistant> {a}"[:16000]
    except Exception as e: print("skip OpenMathReasoning", str(e)[:40], flush=True)
    try:
        for r in load_dataset("open-thoughts/OpenThoughts2-1M", split="train", streaming=True):
            cv = r.get("conversations") or []
            if len(cv) >= 2:
                u = (cv[0].get("value") or cv[0].get("content") or "").strip()
                a = (cv[1].get("value") or cv[1].get("content") or "").strip()
                if u and a: yield f"<user> {u}\n<assistant> {a}"[:16000]
    except Exception as e: print("skip OpenThoughts2", str(e)[:40], flush=True)
    try:
        for r in load_dataset("nvidia/OpenMathInstruct-2", split="train", streaming=True):
            q = (r.get("problem") or "").strip(); a = (r.get("generated_solution") or "").strip()
            if q and a: yield f"<user> {q}\n<assistant> {a}"[:16000]
    except Exception as e: print("skip OpenMathInstruct-2", str(e)[:40], flush=True)
    try:
        for r in load_dataset("AI-MO/NuminaMath-CoT", split="train", streaming=True):
            q = (r.get("problem") or "").strip(); a = (r.get("solution") or "").strip()
            if q and a: yield f"<user> {q}\n<assistant> {a}"[:16000]
    except Exception as e: print("skip NuminaMath-CoT", str(e)[:40], flush=True)

def interleave(weighted):
    """Round-robin generators by integer weight so ANY prefix keeps the target proportions.
    Exhausted (finite) generators drop out; infinite ones fill the remainder of a large cap."""
    gens = [[iter(g), w] for g, w in weighted]
    while gens:
        for slot in list(gens):
            g, w = slot
            for _ in range(w):
                try: yield next(g)
                except StopIteration:
                    if slot in gens: gens.remove(slot)
                    break

def cosmopedia_stream():
    for cfg in ["web_samples_v2", "web_samples_v1", "stanford", "khanacademy", "wikihow", "openstax", "stories"]:
        try:
            for r in load_dataset("HuggingFaceTB/cosmopedia", cfg, split="train", streaming=True):
                t = (r.get("text") or "").strip()
                if len(t) > 300: yield t
        except Exception as e: print("skip cosmopedia", cfg, str(e)[:40], flush=True)

def wiki_stream():
    try:
        for r in load_dataset("wikimedia/wikipedia", "20231101.en", split="train", streaming=True):
            t = (r.get("text") or "").strip()
            if len(t) > 400: yield t[:4000]
    except Exception as e: print("skip wiki", str(e)[:40], flush=True)

def fineweb_stream(cfg):
    try:
        for _, t in P.web_stream(cfg): yield t
    except Exception as e: print("skip fineweb", cfg, str(e)[:40], flush=True)

def world_mix(web_cfg):
    """Knowledge, interleaved: textbook (Cosmopedia) + encyclopedic (Wikipedia) + educational web."""
    return interleave([(cosmopedia_stream(), 3), (wiki_stream(), 2), (fineweb_stream(web_cfg), 6)])

# Every source as a fresh generator (math oversampled like instruction so it can fill a big cap).
SRC = {
    "instruction": lambda web: PT._repeat(PT.instruction_chat, 4),
    "math":        lambda web: PT._repeat(PT.math_reasoning, 3),
    "code":        lambda web: code_stream(),
    "science":     lambda web: science_stream(),
    "reasoning":   lambda web: reasoning_stream(),
    "cot":         lambda web: cot_stream(),
    "world":       lambda web: world_mix(web),
}

# TWO purpose-built datasets. Both contain ALL the requested data; proportions differ by purpose.
WINDOWS = {
    # now -> Jun 19, trains BESIDE prod (~120K tok/s ~= ~20B in 2 days). Weighted to the
    # newly-added skills (code/science/reasoning) to sharpen them, knowledge/instr/math solid.
    "window1": dict(total=20e9, web="sample-100BT",
                    props=dict(instruction=.12, math=.12, code=.14, science=.14, reasoning=.16, world=.32)),
    # Jun 19 -> 21, FREED GPU + growth. Knowledge-heavy scaling fuel, every skill present at scale.
    "window2": dict(total=180e9, web="sample-350BT",
                    props=dict(instruction=.06, math=.06, code=.08, science=.08, reasoning=.08, world=.64)),
    # science+code top-up to repair Window 2 (its science/code streams died on bad parquet rows).
    # Appended onto window2_train.bin so the growth run gets physics/code at real scale.
    # RUN SINGLE-PROCESS (no --shards): common-pile/arXiv is large but split_dataset_by_node
    # gives each shard a thin file-slice that exhausts early; single-process streams it fully.
    # code is capped low — non-gated code-at-scale is scarce (the-stack gated, codeparrot bad rows).
    "topup": dict(total=10e9, web="sample-100BT",
                  props=dict(science=.80, code=.20)),
    # long multi-step CoT top-up — the weak axis. ~18B of OpenMathReasoning + OpenThoughts2 +
    # OpenMathInstruct. Built once, then APPENDED into the existing window1/window2 corpora.
    "cot": dict(total=18e9, web="sample-100BT", props=dict(cot=1.0)),
}

def encode_fast(f, src, budget, tok, t0, label, batch=2000):
    """Pipelined encoder: a producer thread streams+formats the next batch while the main
    thread runs encode_batch (Rust, releases the GIL) on the current one -> decode and
    tokenize OVERLAP instead of running serially. ~1.5-2x over the serial encode_to."""
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
        for e in tok.encode_batch(b):
            f.write(np.asarray(e.ids, dtype=np.uint16).tobytes()); f.write(b"\x00\x00")
            written += len(e.ids) + 1
        nb += 1
        if nb % 50 == 0:
            print(f"    [{label}] {written/1e9:.2f}B / {budget/1e9:.2f}B  {written/max(time.time()-t0,1)/1e6:.1f}M tok/s", flush=True)
        if written >= budget:
            done.set()
            try:
                while True: q.get_nowait()
            except _queue.Empty: pass
            break
    return written

def build_window(name, total, props, web):
    tok = Tokenizer.from_file("data/bpe.json"); t0 = time.time(); counts = {}
    out = f"{OUT}/{name}_s{SHARD_ID}.bin" if SHARDS > 1 else f"{OUT}/{name}_train.bin"
    print(f"[{name}] shard {SHARD_ID+1}/{SHARDS} target {total/1e9:.0f}B  web={web}  -> {out}", flush=True)
    with open(out, "wb") as f:
        for src, frac in props.items():
            cap = int(total * frac / SHARDS)        # each shard does 1/SHARDS of the section
            counts[src] = encode_fast(f, SRC[src](web), cap, tok, t0, f"{name}:{src}:s{SHARD_ID}")
            print(f"  [{name}] {src} {counts[src]/1e9:.2f}B / {cap/1e9:.1f}B cap", flush=True)
    tot = sum(counts.values())
    print(f"[{name}] BUILT {tot/1e9:.2f}B: " + " ".join(f"{k}={100*v/tot:.0f}%" for k, v in counts.items())
          + f"  in {(time.time()-t0)/60:.0f}m", flush=True)
    return tot

def build_valid():
    tok = Tokenizer.from_file("data/bpe.json"); vt = 0
    with open(f"{OUT}/scale_valid.bin", "wb") as f:
        for _, t in P.web_stream("sample-100BT"):
            ids = np.array(tok.encode(t).ids + [0], dtype=np.uint16); f.write(ids.tobytes()); vt += len(ids)
            if vt > 1_500_000: break
    print("valid built", flush=True)

SHARDS, SHARD_ID = 1, 0
def _apply_shard():
    """When SHARDS>1, give this worker a DISJOINT slice of every streaming source
    (split_dataset_by_node assigns files per rank) and a distinct synthetic-data seed."""
    if SHARDS <= 1: return
    random.seed(1000 + SHARD_ID)
    from datasets.distributed import split_dataset_by_node
    from datasets import IterableDataset
    def wrap(orig):
        def f(*a, **k):
            d = orig(*a, **k)
            if isinstance(d, IterableDataset):
                try: d = split_dataset_by_node(d, rank=SHARD_ID, world_size=SHARDS)
                except Exception: pass
            return d
        return f
    g = globals(); g["load_dataset"] = wrap(g["load_dataset"])
    PT.load_dataset = wrap(PT.load_dataset); P.load_dataset = wrap(P.load_dataset)

if __name__ == "__main__":
    pa = argparse.ArgumentParser()
    pa.add_argument("--only", default=None, help="window1 | window2 (default: both, window1 first)")
    pa.add_argument("--shards", type=int, default=1)
    pa.add_argument("--shard-id", type=int, default=0)
    a = pa.parse_args()
    SHARDS, SHARD_ID = a.shards, a.shard_id
    _apply_shard()
    os.makedirs(OUT, exist_ok=True)
    if SHARDS == 1 and not os.path.exists(f"{OUT}/scale_valid.bin"):
        build_valid()
    for nm in ([a.only] if a.only else ["window1", "window2"]):   # window1 first: ready for the till-19 run
        w = WINDOWS[nm]; build_window(nm, int(w["total"]), w["props"], w["web"])
    print("DONE.", flush=True); sys.stdout.flush(); os._exit(0)
