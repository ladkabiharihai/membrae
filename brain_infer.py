"""USABLE inference for the SFT'd brain. Turns eval-numbers into something you can actually use: proper decoding
(sampling + repetition-penalty + no-repeat-ngram, the #46 fix that kills the 92% greedy degeneration) and the exact
prompt formats the model was SFT'd on. NOT a general/coding assistant (216M capacity floor) -- a usable long-context
retrieval / QA / instruction model, which is what it's genuinely good at (multi-hop 45%, extractive-QA 38%).
Usage: brain_infer.py [--demo] ; or import and call ask(context,q) / instruct(text) / chat().
"""
import warnings,os,sys,argparse; warnings.filterwarnings("ignore"); os.environ.setdefault("RECALL_FRAC","0")
sys.path.insert(0,"/opt/code/membrae")
import torch,torch.nn.functional as F, scale_train as S
from tokenizers import Tokenizer
DEV="cuda"; tok=Tokenizer.from_file("data/bpe.json"); VOCAB=S.VOCAB; CTX=2048
CKPT=os.environ.get("CKPT","/mnt/kv_cache/pragnosia_data/snaps/sft_final_216M_18L.pt")
_m=None
def model():
    global _m
    if _m is None:
        _m=S.BrainLM(d=1024,L=18).to(DEV).eval(); _m.load_state_dict(torch.load(CKPT,map_location=DEV))
    return _m

@torch.no_grad()
def generate(prompt, max_new=128, temp=0.7, top_p=0.9, rep_pen=1.3, no_repeat=3, stop_nl=False, greedy=False):
    """Proven-good decoding (#46): sampling + CTRL repetition-penalty + no-repeat-ngram. Kills greedy degeneration."""
    m=model(); ids=tok.encode(prompt).ids; out=list(ids); gen=[]; nl=tok.encode("\n").ids[0]
    for _ in range(max_new):
        logits=m(torch.tensor([out[-CTX:]],device=DEV))[0,-1].float()
        if greedy: nx=int(logits.argmax())
        else:
            for t in set(gen):                                   # repetition penalty
                logits[t]/= rep_pen if logits[t]>0 else (1/rep_pen)
            if no_repeat and len(gen)>=no_repeat-1:              # no-repeat n-gram
                pref=tuple(gen[-(no_repeat-1):])
                for i in range(len(gen)-no_repeat+1):
                    if tuple(gen[i:i+no_repeat-1])==pref: logits[gen[i+no_repeat-1]]=-1e9
            p=(logits/temp).softmax(-1); sp,si=p.sort(descending=True); cdf=sp.cumsum(-1)
            sp[cdf-sp>top_p]=0; sp/=sp.sum(); nx=int(si[torch.multinomial(sp,1)])
        if nx==0 or (stop_nl and nx==nl): break
        out.append(nx); gen.append(nx)
    return tok.decode(gen).strip()

def ask(context, question, **kw):
    """Long-context retrieval / reading comprehension. GREEDY (+no-repeat) -- factual answers need the most-likely
    token, not sampling (sampling corrupts spans, e.g. orange->yellow). Short, stop at newline."""
    kw.setdefault("greedy", True)
    return generate(f"{context}\n\nQuestion: {question}\nAnswer:", max_new=kw.pop("max_new",24), stop_nl=True, **kw)
def instruct(text, **kw):
    """Instruction-following (Alpaca format the model was SFT'd on)."""
    return generate(f"Instruction: {text}\nResponse:", max_new=kw.pop("max_new",120), **kw)

def chat():
    print("brain chat (retrieval/QA/instruction). Ctrl-D to exit.\nPrefix with 'ctx:' then a line 'q:' for reading-comprehension.",flush=True)
    import sys
    for line in sys.stdin:
        t=line.strip()
        if not t: continue
        print("  >", instruct(t), flush=True)

if __name__=="__main__":
    ap=argparse.ArgumentParser(); ap.add_argument("--demo",action="store_true"); ap.add_argument("--chat",action="store_true"); a=ap.parse_args()
    print(f"loaded {os.path.basename(CKPT)}\n",flush=True)
    if a.chat: chat(); sys.exit()
    # DEMO: show it's usable on its real strengths, with good decoding
    print("== INSTRUCTION ==",flush=True)
    for q in ["Give two tips for staying healthy.","What is the capital of France?","Explain what photosynthesis is."]:
        print(f"  Q: {q}\n  A: {instruct(q, max_new=60)!r}\n",flush=True)
    print("== READING COMPREHENSION / RETRIEVAL ==",flush=True)
    ctx=("The Amazon rainforest is located in South America and spans nine countries. It produces about 20 percent of "
         "the world's oxygen and is home to over 390 billion trees. The Amazon River runs through it.")
    for q in ["Where is the Amazon rainforest located?","How much of the world's oxygen does it produce?","How many trees are in it?"]:
        print(f"  Q: {q}\n  A: {ask(ctx,q)!r}\n",flush=True)
    print("== LONG-CONTEXT NEEDLE ==",flush=True)
    fill="The weather was mild and the market carried on with its usual murmur. Reports were filed and calls were made. "
    doc=f"Important note: Dr. Rowan's access code is 7391. Remember it.\n\n"+fill*20
    print(f"  (needle buried ~{len(tok.encode(doc).ids)} tokens back)",flush=True)
    print(f"  A: {ask(doc,'What is Dr. Rowan access code?')!r}",flush=True)
