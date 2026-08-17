"""NO-TRAINING FIXES: which faculty failures are DECODING/PROMPTING artifacts (fixable at inference) vs genuine
capability gaps (need training)? Test each fix and MEASURE the before/after, honestly.
  FIX 1  long-text degeneration (92% 4-gram repeat)  -> sampling + repetition penalty + no-repeat-ngram  (decoding)
  FIX 2  extractive-QA / knowledge format             -> few-shot exemplars in the prompt                (prompting)
  FIX 3  answer over-generation                        -> stop at newline + strip                          (decoding)
Uses the latest snapshot. Prints greedy(before) vs fixed(after) for each.
"""
import warnings,os; warnings.filterwarnings("ignore"); os.environ["PYTHONWARNINGS"]="ignore"; os.environ["HF_HUB_DISABLE_PROGRESS_BARS"]="1"
import sys,re,glob,random,numpy as np,torch,torch.nn.functional as F
sys.path.insert(0,"/opt/code/membrae")
import scale_train as S
from tokenizers import Tokenizer
from datasets import load_dataset
DEV="cuda"; random.seed(0); torch.manual_seed(0)
tok=Tokenizer.from_file("data/bpe.json"); CTX=2048
SNAP=sorted(glob.glob("/mnt/kv_cache/pragnosia_data/snaps/scale_*.pt"),key=lambda p:int(re.search(r'scale_(\d+)_',p).group(1)))[-1]
L0=int(re.search(r'_(\d+)L\.pt',SNAP).group(1))
m=S.BrainLM(d=1024,L=L0).to(DEV).eval(); m.load_state_dict(torch.load(SNAP,map_location=DEV))
print(f"=== NO-TRAINING FIXES on {os.path.basename(SNAP)} ===\n",flush=True)

@torch.no_grad()
def decode(prompt, n=256, greedy=True, temp=0.8, top_p=0.95, rep_pen=1.3, no_repeat=3, stop_nl=False):
    ids=tok.encode(prompt).ids; out=list(ids); gen=[]
    for _ in range(n):
        logits=m(torch.tensor([out[-CTX:]],device=DEV))[0,-1].float()
        if greedy:
            nx=int(logits.argmax())
        else:
            # repetition penalty (CTRL-style): divide logit of already-seen tokens
            if rep_pen and gen:
                for t in set(gen):
                    logits[t]/= rep_pen if logits[t]>0 else 1.0
                    if logits[t]<0: logits[t]*=rep_pen
            # no-repeat n-gram: ban tokens that would complete a seen n-gram
            if no_repeat and len(gen)>=no_repeat-1:
                pref=tuple(gen[-(no_repeat-1):])
                for i in range(len(gen)-no_repeat+1):
                    if tuple(gen[i:i+no_repeat-1])==pref: logits[gen[i+no_repeat-1]]=-1e9
            probs=(logits/temp).softmax(-1)
            sp,si=probs.sort(descending=True); cdf=sp.cumsum(-1)
            sp[cdf-sp>top_p]=0; sp/=sp.sum()
            nx=int(si[torch.multinomial(sp,1)])
        if nx==0 or (stop_nl and nx==tok.encode("\n").ids[0]): break
        out.append(nx); gen.append(nx)
    return tok.decode(gen).strip()

def degen(toks):
    d1=len(set(toks))/max(1,len(toks)); fg=list(zip(toks,toks[1:],toks[2:],toks[3:]))
    rep=1-len(set(fg))/max(1,len(fg)); return d1,rep

# ===================== FIX 1: long-text degeneration (decoding) =====================
print("--- FIX 1: long-text generation  (greedy vs sampling+rep-penalty+no-repeat-3gram) ---",flush=True)
prompts=["The history of the Roman Empire began when","Photosynthesis is the process by which plants",
         "In the beginning of the story, the young explorer"]
for p in prompts:
    g=decode(p,120,greedy=True); s=decode(p,120,greedy=False)
    gd1,grep=degen(tok.encode(g).ids); sd1,srep=degen(tok.encode(s).ids)
    print(f"\n  PROMPT: {p!r}",flush=True)
    print(f"   GREEDY   distinct-1={gd1:.2f} 4gram-rep={grep:.2f}: {g[:150]!r}",flush=True)
    print(f"   SAMPLED  distinct-1={sd1:.2f} 4gram-rep={srep:.2f}: {s[:150]!r}",flush=True)

# ===================== FIX 2: extractive-QA via few-shot (prompting) =====================
print("\n\n--- FIX 2: extractive-QA  (0-shot vs 2-shot exemplars) ---",flush=True)
FEWSHOT=("Context: The Eiffel Tower is located in Paris and was completed in 1889.\n"
         "Question: In what city is the Eiffel Tower?\nAnswer: Paris\n\n"
         "Context: Water is composed of hydrogen and oxygen atoms.\n"
         "Question: What atoms make up water?\nAnswer: hydrogen and oxygen\n\n")
def span_hit(gold,g): gl=gold.lower();g=g.lower(); return gl in g or g in gl or any(w in g for w in gl.split() if len(w)>3)
def ans(g): return g.split("\n")[0].strip()
try:
    sq=[e for e in load_dataset('rajpurkar/squad_v2',split='validation') if e['answers']['text']][:60]
    h0=hf=0
    for e in sq:
        base=f"Context: {e['context']}\nQuestion: {e['question']}\nAnswer:"
        g0=ans(decode(base,8,greedy=True,stop_nl=True))
        gf=ans(decode(FEWSHOT+base,8,greedy=True,stop_nl=True))
        h0+=span_hit(e['answers']['text'][0],g0); hf+=span_hit(e['answers']['text'][0],gf)
    print(f"  0-shot extractive-QA: {h0}/60 = {100*h0//60}%",flush=True)
    print(f"  2-shot extractive-QA: {hf}/60 = {100*hf//60}%   (few-shot delta = {100*hf//60-100*h0//60:+d}pt)",flush=True)
except Exception as e: print(f"  FIX2 SKIP {e}",flush=True)

# ===================== FIX 3: knowledge MC with instruction prefix (prompting) =====================
print("\n--- FIX 3: MMLU history  (bare vs few-shot MC prompt, loglik) ---",flush=True)
@torch.no_grad()
def nll(context,cont):
    cids=tok.encode(context).ids; kids=tok.encode(cont).ids
    if not kids: return 1e9
    ids=(cids+kids)[-CTX:]; n=min(len(kids),len(ids)-1)
    logp=m(torch.tensor([ids],device=DEV))[0].log_softmax(-1); tot=0.0
    for j in range(n): tot+=logp[len(ids)-n+j-1, ids[len(ids)-n+j]].item()
    return -tot/n
try:
    hist=load_dataset('cais/mmlu','high_school_world_history',split='test')
    ex=[hist[i] for i in range(40)]
    fs=("Question: What is the capital of France?\nAnswer: Paris\n\n"
        "Question: In what year did World War II end?\nAnswer: 1945\n\n")
    b=f=0
    for e in ex:
        gd=int(e['answer']); ch=e['choices']
        b+= int(np.argmin([nll(f"Question: {e['question']}\nAnswer:",c) for c in ch])==gd)
        f+= int(np.argmin([nll(fs+f"Question: {e['question']}\nAnswer:",c) for c in ch])==gd)
    print(f"  bare      MMLU-history: {b}/40 = {100*b//40}%   (chance 25%)",flush=True)
    print(f"  few-shot  MMLU-history: {f}/40 = {100*f//40}%   (delta = {100*f//40-100*b//40:+d}pt)",flush=True)
except Exception as e: print(f"  FIX3 SKIP {e}",flush=True)
print("\n[done]",flush=True)
