"""Build the COMBINED SFT corpus for the locked 1B base: CoT-math + instruction/chat + recall (+ replay).
All in the model's <user>/<assistant> chat format, uint16 + \\x00\\x00 EOS, W1/256-ctx (SFT is short-context;
long-range recall is brain.py's job -- here we teach IN-WINDOW copy). Templated generation is parametrized
(no fixed answers baked in); the CoT-math templates specifically target the measured total-vs-rate bug.

  python3 build_sft.py            # -> /mnt/kv_cache/pragnosia_data/sft_combined.bin
"""
import os, random, numpy as np
from tokenizers import Tokenizer

random.seed(0)
OUT = "/mnt/kv_cache/pragnosia_data"
tok = Tokenizer.from_file("data/bpe.json")
EOS = b"\x00\x00"
CTX = 256


def enc(text):
    ids = tok.encode(text).ids[:CTX]
    return np.asarray(ids, dtype=np.uint16).tobytes() + EOS


# ---------- 1. CoT MATH (targets total-vs-rate: "X per Y for N" -> multiply, not divide) ----------
NAMES = ["Sarah", "Tom", "Lisa", "John", "Ann", "Ben", "Mia", "Leo", "Zoe", "Sam"]
ITEMS = ["apples", "books", "pens", "cakes", "pages", "goals", "coins", "cards", "boxes", "miles"]

def cot_math(n):
    """BALANCED across 5 operation types (equal share, so the model can't default to 'multiply everything' --
    the overfit that broke the first run) with MULTIPLE phrasings per type to avoid surface memorization."""
    out = []
    ops = ["add", "sub", "mul", "div", "two"]
    for i in range(n):
        op = ops[i % len(ops)]            # exact equal balance across operations
        nm, it = random.choice(NAMES), random.choice(ITEMS)
        if op == "add":
            a, b = random.randint(3, 40), random.randint(2, 30)
            q = random.choice([f"{nm} had {a} {it} and got {b} more. How many now?",
                               f"There are {a} {it} in one box and {b} in another. How many in total?",
                               f"{nm} counts {a} {it}, then adds {b} more. What is the total?"])
            s = f"Add them: {a} + {b} = {a+b}. The answer is {a+b}."
        elif op == "sub":
            a, b = random.randint(12, 60), random.randint(1, 11)
            q = random.choice([f"{nm} had {a} {it} and gave away {b}. How many are left?",
                               f"{nm} had {a} dollars and spent {b}. How much is left?",
                               f"From {a} {it}, {b} are removed. How many remain?"])
            s = f"Subtract: {a} - {b} = {a-b}. The answer is {a-b}."
        elif op == "mul":
            a, b = random.randint(2, 12), random.randint(2, 9)
            q = random.choice([f"{nm} makes {a} {it} each day for {b} days. How many {it} in total?",
                               f"There are {b} bags with {a} {it} each. How many {it} altogether?",
                               f"A shelf holds {a} {it} per row across {b} rows. How many {it} in total?"])
            s = f"This is a total, so multiply: {a} times {b} = {a*b}. The answer is {a*b}."
        elif op == "div":
            d, per = random.randint(2, 9), random.randint(2, 12); tot = per * d
            q = random.choice([f"There are {tot} {it} split equally into {d} groups. How many in each group?",
                               f"{tot} {it} are shared equally among {d} people. How many each?",
                               f"{nm} packs {tot} {it} into {d} equal boxes. How many per box?"])
            s = f"This is sharing, so divide: {tot} divided by {d} = {per}. The answer is {per}."
        else:  # two-step
            a, b, c = random.randint(3, 9), random.randint(2, 6), random.randint(1, 9)
            q = f"{nm} has {a} {it} in each of {b} bags, then finds {c} more. How many in total?"
            s = f"First {a} times {b} = {a*b}, then add {c}: {a*b} + {c} = {a*b+c}. The answer is {a*b+c}."
        out.append(f"<user> {q} <assistant> Let's think step by step. {s}")
    return out


# ---------- FIX (v4): multi-OPERATION word problems -- v3 chained op#1 then DROPPED op#2 ----------
def multiop(n):
    out = []
    for _ in range(n):
        nm, nm2 = random.sample(NAMES, 2); it = random.choice(ITEMS); pat = random.random()
        if pat < 0.4:            # add then subtract
            a, b = random.randint(3, 12), random.randint(1, 6); c = random.randint(1, a + b - 1)
            q = f"I have {a} {it}. {nm} gives me {b} more, then {nm2} takes {c} away. How many do I have now?"
            s = f"Start with {a}. Add {b}: {a} + {b} = {a+b}. Then subtract {c}: {a+b} - {c} = {a+b-c}. The answer is {a+b-c}."
        elif pat < 0.7:          # subtract then add
            a, b = random.randint(8, 18), random.randint(1, 6); c = random.randint(1, 7)
            q = f"I have {a} {it}. I lose {b}, then find {c} more. How many now?"
            s = f"Start with {a}. Subtract {b}: {a} - {b} = {a-b}. Then add {c}: {a-b} + {c} = {a-b+c}. The answer is {a-b+c}."
        else:                    # add then add
            a, b, c = random.randint(2, 10), random.randint(1, 8), random.randint(1, 8)
            q = f"I have {a} {it}. {nm} gives me {b}, then {nm2} gives me {c}. How many now?"
            s = f"Start with {a}. Add {b}: {a} + {b} = {a+b}. Then add {c}: {a+b} + {c} = {a+b+c}. The answer is {a+b+c}."
        out.append(f"<user> {q} <assistant> Let's think step by step. {s}")
    return out


# ---------- FIX #3 (v4): honest ABSTENTION on genuine unknowables (teach "I don't know") ----------
FICT = ["Zzyzx-9","Qwortle","Blimvale","Xanthar City","Frobnia","Glorbon-7","Vexmoor","Nimbus-Prime"]
def abstain(n):
    out = []
    for _ in range(n):
        k = random.random()
        if k < 0.3:              # fictional / unknown entity
            p = random.choice(FICT)
            q = random.choice([f"What is the population of {p}?", f"Where is {p} located?", f"Tell me about the city of {p}."])
            s = f"I don't know -- {p} is not something I have reliable information about."
        elif k < 0.55:           # future event
            yr = random.choice([2040, 2050, 2075, 2088, 2099])
            q = random.choice([f"Who won the {yr} World Cup?", f"Who will be president in {yr}?", f"What major events happened in {yr}?"])
            s = f"I don't know -- {yr} is in the future, so I can't know that."
        elif k < 0.8:            # user's personal info (distinct from ITS own identity)
            q = random.choice(["What is my name?", "What is my favorite color?", "How old am I?", "Where do I live?"])
            s = "I don't know -- you haven't told me, so I can't be sure."
        else:                    # inherently unknowable
            q = random.choice(["What are tomorrow's lottery numbers?", "What number am I thinking of?", "Will it rain on this day next year?"])
            s = "I don't know -- that isn't something anyone can reliably predict."
        out.append(f"<user> {q} <assistant> {s}")
    return out


# ---------- 2. RECALL (in-window copy: key/value + short filler + query) ----------
KEYS = ["secret word", "password", "code name", "favorite color", "chosen number", "magic token"]
VALS = ["apple", "river", "copper", "violet", "tiger", "maple", "delta", "orbit", "ember", "quartz", "nova", "flint"]
FILL = ["The weather is calm.", "The room was quiet.", "Time passed slowly.", "The lights stayed on.",
        "Nothing else happened.", "The day continued.", "People walked by.", "The clock kept ticking."]

def recall(n):
    out = []
    for _ in range(n):
        key, val = random.choice(KEYS), random.choice(VALS)
        filler = " ".join(random.choice(FILL) for _ in range(random.randint(0, 12)))
        q = f"Remember: the {key} is {val}. {filler} What is the {key}?"
        out.append(f"<user> {q} <assistant> The {key} is {val}.")
    return out


# ---------- FIX #1: constrained instructions -- teach EXACT count / length AND clean stopping ----------
CATS = {"fruits": ["apple","banana","orange","pear","grape","peach","mango","plum","cherry","lemon"],
        "colors": ["red","blue","green","yellow","purple","orange","pink","black","white","brown"],
        "animals": ["dog","cat","horse","rabbit","lion","tiger","bear","fox","deer","wolf"],
        "ocean animals": ["dolphin","whale","shark","octopus","tuna","crab","seal","jellyfish"],
        "countries": ["France","Japan","Brazil","Kenya","Canada","Egypt","Norway","Chile"],
        "planets": ["Mercury","Venus","Earth","Mars","Jupiter","Saturn","Neptune","Uranus"]}
ONE_WORD = [("What color is the sky?","blue"),("What is the opposite of hot?","cold"),
            ("What is 2 plus 2?","4"),("What is the capital of France?","Paris"),
            ("What is the opposite of up?","down"),("What color is grass?","green"),
            ("How many legs does a dog have?","four"),("What is the first month of the year?","January")]
ONE_SENT = {"dogs":"Dogs are loyal animals often kept as pets.","the sun":"The sun is the star at the center of our solar system.",
            "water":"Water is a clear liquid essential for life.","books":"Books store knowledge and stories in written form.",
            "the moon":"The moon is Earth's only natural satellite.","music":"Music is organized sound that expresses emotion."}

def instructions(n):
    out = []
    for i in range(n):
        k = i % 4
        if k == 0:  # exact-count list (varies N: 2/3/4) -> teaches counting AND stopping
            cnt = random.choice([2, 3, 4]); cat = random.choice(list(CATS)); items = random.sample(CATS[cat], cnt)
            words = {2: "two", 3: "three", 4: "four"}[cnt]
            body = " ".join(f"{j+1}. {w.capitalize()}" for j, w in enumerate(items))
            out.append(f"<user> List exactly {words} {cat}. <assistant> {body}")
        elif k == 1:  # one word
            q, a = random.choice(ONE_WORD)
            out.append(f"<user> Answer in one word: {q} <assistant> {a.capitalize()}.")
        elif k == 2:  # one sentence
            t, s = random.choice(list(ONE_SENT.items()))
            out.append(f"<user> In one sentence, tell me about {t}. <assistant> {s}")
        else:  # name exactly one
            cat = random.choice(list(CATS)); it = random.choice(CATS[cat])
            out.append(f"<user> Name one {cat[:-1] if cat.endswith('s') else cat}. <assistant> {it.capitalize()}.")
    return out


# ---------- FIX #2: multi-hop CoT -- teach COMPOSITION with worked intermediate steps ----------
PPL = ["Tom","Sam","Leo","Anna","Bob","Carl","Mia","Ben","Zoe","Ada","Ivan","Omar"]
GEO = [("Paris","France","Europe"),("Tokyo","Japan","Asia"),("Cairo","Egypt","Africa"),
       ("Lima","Peru","South America"),("Oslo","Norway","Europe"),("Nairobi","Kenya","Africa")]
SYL = [("cats","animals","need food"),("roses","flowers","need water"),("whales","mammals","breathe air"),
       ("squares","shapes","have sides"),("robins","birds","have feathers")]

def multihop(n):
    out = []
    for i in range(n):
        k = i % 5
        if k == 0:  # 2-hop kinship
            a, b, c = random.sample(PPL, 3)
            q = f"{a} is {b}'s father. {b} is {c}'s father. What is {a} to {c}?"
            s = f"Let's think step by step. {a} is the parent of {b}, and {b} is the parent of {c}. The parent of a parent is a grandparent, so {a} is {c}'s grandfather. The answer is grandfather."
        elif k == 1:  # transitive size (3 people)
            a, b, c = random.sample(PPL, 3)
            q = f"{a} is taller than {b}. {b} is taller than {c}. Who is the tallest?"
            s = f"Let's think step by step. {a} is taller than {b}, and {b} is taller than {c}, so {a} is taller than both. The answer is {a}."
        elif k == 2:  # transitive geography
            city, country, cont = random.choice(GEO)
            q = f"{city} is in {country}. {country} is in {cont}. Is {city} in {cont}?"
            s = f"Let's think step by step. {city} is in {country}, and {country} is in {cont}, so {city} is in {cont}. The answer is yes."
        elif k == 3:  # syllogism
            x, y, z = random.choice(SYL)
            q = f"All {x} are {y}. All {y} {z}. Do {x} {z}?"
            s = f"Let's think step by step. All {x} are {y}, and all {y} {z}. Therefore all {x} {z}. The answer is yes."
        else:  # 3-way ordering (lightest)
            a, b, c = random.sample(PPL, 3)
            q = f"{a} is heavier than {b}. {c} is lighter than {b}. Who is the lightest?"
            s = f"Let's think step by step. {a} is heavier than {b}, and {c} is lighter than {b}, so {c} is below {b} which is below {a}. The answer is {c}."
        out.append(f"<user> {q} <assistant> {s}")
    return out


# ---------- 3. real instruction/chat spans mined from expanded2 (has WildChat/UltraChat/etc) ----------
def mine_chat(bin_path, n, ctx=CTX):
    if not os.path.exists(bin_path): return []
    d = np.memmap(bin_path, dtype=np.uint16, mode="r"); out = []; L = len(d)
    step = max(ctx, L // (n * 3))
    for i in range(0, L - ctx, step):
        txt = tok.decode([int(x) for x in d[i:i + ctx] if int(x) != 0])
        j = txt.find("<user>")
        if j >= 0 and "<assistant>" in txt[j:]:
            out.append(txt[j:j + 900])
            if len(out) >= n: break
    return out


# ---------- 4. REAL instruction/chat streamed from HF (guaranteed <user>/<assistant> dialogue) ----------
def stream_chat(max_ex):
    os.environ.setdefault("HF_HOME", "/mnt/kv_cache/hf_home")
    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
    out = []
    try:
        from datasets import load_dataset
        srcs = [("HuggingFaceH4/ultrachat_200k", None, "train_sft", "messages"),
                ("HuggingFaceTB/smoltalk", "all", "train", "messages")]
        per = max_ex // len(srcs)
        for ds, cfg, split, key in srcs:
            got = 0
            for r in load_dataset(ds, cfg, split=split, streaming=True):
                msgs = r.get(key) or r.get("conversation") or []
                turns = []
                for mm in msgs:
                    role = (mm.get("role") or "").lower(); c = (mm.get("content") or "").strip()
                    if not c: continue
                    turns.append(f"<{'assistant' if role in ('assistant','gpt') else 'user'}> {c}")
                if len(turns) >= 2:
                    out.append(" ".join(turns)[:1200]); got += 1
                if got >= per: break
    except Exception as e:
        print(f"  (chat stream failed: {str(e)[:80]} -- falling back to cot_train + mining)", flush=True)
    return out


# ---------- 5. real reasoning-CoT slice from the existing cot_train.bin ----------
def cot_train_slice(n_tokens):
    p = f"{OUT}/cot_train.bin"
    if not os.path.exists(p): return b""
    d = np.memmap(p, dtype=np.uint16, mode="r")
    start = random.randint(0, max(0, len(d) - n_tokens - 1))
    chunk = np.asarray(d[start:start + n_tokens], dtype=np.uint16)
    return chunk.tobytes()


def main():
    # v2 mix: LESS templated math (overfit the 'multiply' pattern last run), MORE real/diverse instruction,
    # and REPLAY = 50% of the final corpus so general knowledge/fluency does NOT drift (last run: 16.56->25 ppl).
    parts = []
    parts.append(("cot-math(balanced)", cot_math(30000)))                       # balanced ops (kept: 87%)
    parts.append(("multihop-CoT", multihop(25000)))                             # FIX #2: composition (kept: 5/6)
    parts.append(("multiop-word", multiop(20000)))                              # v4: multi-OPERATION word problems
    parts.append(("instructions", instructions(30000)))                         # FIX #1: exact count + stopping
    parts.append(("abstain", abstain(15000)))                                   # FIX #3: honest "I don't know"
    parts.append(("recall", recall(25000)))                                     # kept (8/8)
    parts.append(("chat-real", stream_chat(90000) or mine_chat(f"{OUT}/expanded2_train.bin", 30000)))  # real chat
    out = f"{OUT}/sft_combined.bin"; tot = 0
    with open(out, "wb") as f:
        for name, exs in parts:
            random.shuffle(exs); nb = 0
            for ex in exs:
                b = enc(ex); f.write(b); nb += len(b) // 2
            tot += nb; print(f"  {name}: {len(exs)} ex, {nb/1e6:.1f}M tok", flush=True)
        cot_bytes = cot_train_slice(18_000_000)                                 # more REAL diverse reasoning-CoT
        f.write(cot_bytes); tot += len(cot_bytes) // 2
        print(f"  cot_train slice: {len(cot_bytes)//2/1e6:.1f}M tok", flush=True)
        sft_content = tot
        # REPLAY = 50% of total => replay tokens == sft_content tokens. Anchors general LM quality.
        rep_path = f"{OUT}/window2_train.bin"
        if os.path.exists(rep_path):
            d = np.memmap(rep_path, dtype=np.uint16, mode="r")
            target = sft_content; span = CTX + 1; got = 0
            while got < target:
                i = random.randint(0, len(d) - span - 1)
                f.write(np.asarray(d[i:i + CTX], dtype=np.uint16).tobytes()); f.write(EOS); got += CTX + 1
            tot += got
            print(f"  replay(window2): {got/1e6:.1f}M tok ({100*got//tot}% of corpus)", flush=True)
    print(f"[sft] wrote {out}: {tot/1e6:.1f}M tokens total (sft {sft_content/1e6:.1f}M + replay {(tot-sft_content)/1e6:.1f}M)", flush=True)
    open(f"{OUT}/sft_combined_DONE.flag", "w").write(str(tot))


if __name__ == "__main__":
    main()
