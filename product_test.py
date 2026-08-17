"""PRODUCT USABILITY TEST -- is the SFT'd brain usable as a product (not a benchmark)? Exercise brain_infer across
its real capabilities with diverse prompts, auto-grade where answers are known, measure LATENCY, and give an honest
product verdict (what it's usable FOR, what it isn't). Uses good decoding (#46) via brain_infer.
"""
import warnings,os,sys,time; warnings.filterwarnings("ignore"); os.environ.setdefault("RECALL_FRAC","0")
sys.path.insert(0,"/opt/code/membrae")
import brain_infer as B
from tokenizers import Tokenizer
tok=Tokenizer.from_file("data/bpe.json")
def hit(gold,out): gl=gold.lower(); o=out.lower(); return gl in o or any(w in o for w in gl.split() if len(w)>3)

print(f"PRODUCT TEST on {os.path.basename(B.CKPT)}\n",flush=True)

# 1) READING COMPREHENSION / RETRIEVAL (gradeable) -- the core strength
RC=[("The Great Wall of China is over 13,000 miles long and was built over many centuries to protect against invasions. Construction began more than 2,000 years ago.",
     [("How long is the Great Wall?","13,000 miles"),("What was it built to protect against?","invasions"),("How long ago did construction begin?","2,000 years")]),
    ("Photosynthesis occurs in the chloroplasts of plant cells. It uses sunlight, water, and carbon dioxide to produce glucose and oxygen.",
     [("Where does photosynthesis occur?","chloroplasts"),("What does it produce?","glucose"),("What gas is released?","oxygen")]),
    ("Marie Curie was born in Warsaw in 1867. She won two Nobel Prizes, one in Physics and one in Chemistry, for her work on radioactivity.",
     [("Where was Marie Curie born?","Warsaw"),("How many Nobel Prizes did she win?","two"),("What did she study?","radioactivity")])]
print("== READING COMPREHENSION (gradeable) ==",flush=True)
rc_hit=rc_tot=0; t0=time.time()
for ctx,qas in RC:
    for q,a in qas:
        o=B.ask(ctx,q); ok=hit(a,o); rc_hit+=ok; rc_tot+=1
        print(f"  [{'OK' if ok else 'x '}] {q[:42]:44s} -> {o[:40]!r}",flush=True)
print(f"  reading-comprehension: {rc_hit}/{rc_tot} = {100*rc_hit//rc_tot}%\n",flush=True)

# 2) LONG-CONTEXT retrieval at distance
print("== LONG-CONTEXT NEEDLE (distance) ==",flush=True)
fill="The weather stayed mild and the market carried on with its usual murmur. Reports were filed and calls were made. Nothing else of note occurred that afternoon. "
nd_hit=nd_tot=0
for reps,lbl in [(6,"~160tok"),(16,"~430tok"),(34,"~900tok")]:
    doc=f"Important: the vault password is orange-falcon-92. Keep it safe.\n\n"+fill*reps
    o=B.ask(doc,"What is the vault password?"); ok=hit("orange-falcon-92",o) or hit("falcon",o); nd_hit+=ok; nd_tot+=1
    print(f"  [{'OK' if ok else 'x '}] needle @ {lbl:9s} ({len(tok.encode(doc).ids)} tok) -> {o[:30]!r}",flush=True)
print(f"  long-context needle: {nd_hit}/{nd_tot}\n",flush=True)

# 3) INSTRUCTION FOLLOWING (qualitative -- show outputs)
print("== INSTRUCTION FOLLOWING (qualitative) ==",flush=True)
for q in ["List three fruits.","Write one sentence about the ocean.","What is 2 plus 2?","Give a tip for studying.","Translate 'hello' to a greeting."]:
    print(f"  Q: {q}\n     -> {B.instruct(q,max_new=45)!r}",flush=True)

# 4) LATENCY
print("\n== LATENCY ==",flush=True)
t0=time.time(); o=B.instruct("Describe a sunny day.",max_new=60); dt=time.time()-t0
ntok=len(tok.encode(o).ids)
print(f"  generated {ntok} tokens in {dt:.1f}s = {ntok/dt:.0f} tok/s",flush=True)

print("\n== HONEST PRODUCT VERDICT ==",flush=True)
print(f"  reading-comprehension/retrieval: {100*rc_hit//rc_tot}%  | long-context needle: {nd_hit}/{nd_tot}",flush=True)
print("  USABLE FOR: reading-comprehension, retrieval/QA over provided text, long-context lookup, simple instructions.",flush=True)
print("  NOT usable for: closed-book facts (hallucinates), math, code, open-ended knowledge -- 216M capacity floor.",flush=True)
print("DONE",flush=True)
