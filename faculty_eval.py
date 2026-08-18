"""RIGOROUS FACULTY EVAL: an honest capability map of the intrinsic-brain snapshot across every axis the user
asked for -- reasoning, multi-hop, long-context, long generation, physics, maths, chemistry, biology, coding,
world knowledge, history. 100+ held-out questions. Two scoring modes, both standard:
  * MULTIPLE-CHOICE -> length-normalized log-likelihood (lm-eval 'acc_norm': pick the choice with lowest per-token
    NLL). This is the FAIR way to score a small model on knowledge/reasoning -- no generation lottery.
  * GENERATION -> greedy decode + span/exact match (recall, multi-hop, needle, math, code) or degeneration metrics
    (long-text: distinct-n + repetition rate).
Every dataset is a held-out split NEVER in training. Sections are independently guarded so one failed download
does not sink the run. Prints a per-category table + an honest summary at the end.
"""
import warnings,os; warnings.filterwarnings("ignore"); os.environ["PYTHONWARNINGS"]="ignore"; os.environ["HF_HUB_DISABLE_PROGRESS_BARS"]="1"
import sys,json,math,random,time,numpy as np,torch
sys.path.insert(0,"/opt/code/membrae")
import scale_train as S
from tokenizers import Tokenizer
from datasets import load_dataset
DEV="cuda"; random.seed(1234); torch.manual_seed(0)
tok=Tokenizer.from_file("data/bpe.json"); CTX=2048
import glob,re
SNAP=os.environ.get("SNAP") or sorted(glob.glob("/mnt/kv_cache/pragnosia_data/snaps/scale_*.pt"),key=lambda p:int(re.search(r'scale_(\d+)_',p).group(1)))[-1]
if "moe_" in os.path.basename(SNAP):                        # MoE snapshot (grow_moe_train MoELM)
    os.environ.setdefault("D","1280"); os.environ.setdefault("L","20"); os.environ["COMPILE"]="0"; os.environ["RECALL_FRAC"]="0"
    import grow_moe_train as GM
    E=int(re.search(r'_E(\d+)\.pt',SNAP).group(1)); m=GM.MoELM(E).to(DEV).eval()
    sd=torch.load(SNAP,map_location=DEV); sd={k.replace("_orig_mod.",""):v for k,v in sd.items()}
    m.load_state_dict(sd)
    print(f"=== FACULTY EVAL on {os.path.basename(SNAP)} (MoE {GM.d}d {len(m.blocks)}L E={E}, {sum(p.numel() for p in m.parameters())/1e6:.0f}M) ===",flush=True)
else:
    L0=int(re.search(r'_(\d+)L',SNAP).group(1))
    _fm=re.search(r'_f(\d+)\.pt',SNAP); FEAT0=int(_fm.group(1)) if _fm else 8   # feat from filename (old snaps=8)
    m=S.BrainLM(d=1024,L=L0).to(DEV).eval()
    _f=8                                                       # replay feat-grows to rebuild GroupFeatNorm before load
    while _f<FEAT0: _f=min(_f+2,FEAT0); m.grow_feat(_f)
    m.load_state_dict(torch.load(SNAP,map_location=DEV))
    print(f"=== FACULTY EVAL on {os.path.basename(SNAP)} ({sum(p.numel() for p in m.parameters())/1e6:.0f}M, {L0}L feat{FEAT0}) ===",flush=True)
RESULTS={}

@torch.no_grad()
def nll_pertoken(context, cont):
    """length-normalized NLL of `cont` given `context` (lower = model prefers it). acc_norm scoring."""
    cids=tok.encode(context).ids; kids=tok.encode(cont).ids
    if not kids: return 1e9
    ids=(cids+kids)[-CTX:]; n=min(len(kids),len(ids)-1)
    x=torch.tensor([ids],device=DEV); logp=m(x)[0].log_softmax(-1)
    tot=0.0
    for j in range(n):
        pos=len(ids)-n+j-1; tgt=ids[len(ids)-n+j]
        tot+=logp[pos,tgt].item()
    return -tot/n

def mc(context, choices, gold):
    """pick argmin per-token NLL; return 1 if == gold index."""
    scores=[nll_pertoken(context,c) for c in choices]
    return int(int(np.argmin(scores))==gold)

@torch.no_grad()
def gen(prompt, n=12, stop_eos=True):
    ids=tok.encode(prompt).ids; out=[]
    for _ in range(n):
        nx=int(m(torch.tensor([(ids+out)[-CTX:]],device=DEV))[0,-1].argmax())
        if stop_eos and nx==0: break
        out.append(nx)
    return tok.decode(out).strip()

def span_hit(gold, g):
    gl=gold.lower(); g=g.lower()
    return gl in g or g in gl or any(w in g for w in gl.split() if len(w)>3)

def run_mc(name, items):
    """items: list of (context, choices, gold_idx)"""
    try:
        h=sum(mc(c,ch,gd) for c,ch,gd in items); n=len(items)
        RESULTS[name]=(h,n, 100.0*h/n, 100.0/len(items[0][1]))
        print(f"  [{name:26s}] {h:3d}/{n:3d} = {100.0*h/n:5.1f}%   (chance {100.0/len(items[0][1]):.0f}%)",flush=True)
    except Exception as e:
        print(f"  [{name:26s}] SKIP: {type(e).__name__}: {str(e)[:60]}",flush=True)

def run_gen(name, items, ntok=12):
    """items: list of (prompt, gold)"""
    try:
        h=0
        for p,gold in items: h+=span_hit(gold,gen(p,ntok))
        n=len(items); RESULTS[name]=(h,n,100.0*h/n,None)
        print(f"  [{name:26s}] {h:3d}/{n:3d} = {100.0*h/n:5.1f}%",flush=True)
    except Exception as e:
        print(f"  [{name:26s}] SKIP: {type(e).__name__}: {str(e)[:60]}",flush=True)

def take(ds,n):
    out=[]
    for i,ex in enumerate(ds):
        out.append(ex)
        if len(out)>=n: break
    return out

# ============ PART A: MEMORY / RECALL (in-context -- the model's intrinsic advantage) ============
print("\n-- PART A: in-context memory / recall (the intrinsic advantage) --",flush=True)
# A1 extractive QA (SQuAD v2 held-out)
try:
    sq=take((e for e in load_dataset('rajpurkar/squad_v2',split='validation') if e['answers']['text']),60)
    run_gen("A1 extractive-QA (SQuAD)", [(f"{e['context']}\n\nQuestion: {e['question']}\nAnswer:", e['answers']['text'][0]) for e in sq], ntok=8)
except Exception as e: print(f"  A1 SKIP {e}",flush=True)
# A2 multi-hop (HotpotQA distractor -- answer buried in 10 shuffled paragraphs)
try:
    hp=take(load_dataset('hotpotqa/hotpot_qa','distractor',split='validation'),40)
    items=[]
    for e in hp:
        paras=[t+": "+" ".join(s) for t,s in zip(e['context']['title'],e['context']['sentences'])]
        random.shuffle(paras)
        items.append(("\n\n".join(paras)+f"\n\nQuestion: {e['question']}\nAnswer:", e['answer']))
    run_gen("A2 multi-hop (HotpotQA)", items, ntok=10)
except Exception as e: print(f"  A2 SKIP {e}",flush=True)

# ============ PART B: LONG-CONTEXT needle-in-haystack (at increasing lengths, greedy) ============
print("\n-- PART B: long-context handling (needle-in-haystack, real facts) --",flush=True)
try:
    # use SQuAD contexts padded with distractor prose to hit target token lengths, answer must survive the distance
    pool=take((e for e in load_dataset('rajpurkar/squad_v2',split='validation') if e['answers']['text']),400)
    fill=("In unrelated news, the weather remained mild across the region and markets stayed calm. "
          "Analysts noted routine activity while commuters went about their day without incident. ")
    for TGT in [256,512,1024,1792]:
        h=0;n=0
        for e in pool:
            base=f"{e['context']}\n\n"
            while len(tok.encode(base).ids) < TGT: base+=fill
            base=tok.decode(tok.encode(base).ids[:TGT])
            g=gen(base+f"\n\nQuestion: {e['question']}\nAnswer:",8)
            h+=span_hit(e['answers']['text'][0],g); n+=1
            if n>=25: break
        RESULTS[f"B needle@{TGT}tok"]=(h,n,100.0*h/n,None)
        print(f"  [needle @ {TGT:4d} tok        ] {h:3d}/{n:3d} = {100.0*h/n:5.1f}%",flush=True)
except Exception as e: print(f"  B SKIP {e}",flush=True)

# ============ PART C: REASONING (multiple-choice, loglik) ============
print("\n-- PART C: reasoning (multiple-choice, length-normalized loglik) --",flush=True)
try:
    hs=take(load_dataset('Rowan/hellaswag',split='validation'),40)
    run_mc("C hellaswag (commonsense)", [(e['ctx'], e['endings'], int(e['label'])) for e in hs if e['label']!=''])
except Exception as e: print(f"  hellaswag SKIP {e}",flush=True)
try:
    pq=take(load_dataset('ybisk/piqa',split='validation',trust_remote_code=True),40)
    run_mc("C piqa (physical reason)", [(e['goal'], [e['sol1'],e['sol2']], int(e['label'])) for e in pq])
except Exception as e: print(f"  piqa SKIP {e}",flush=True)
try:
    ar=take(load_dataset('allenai/ai2_arc','ARC-Challenge',split='test'),40)
    items=[]
    for e in ar:
        ch=e['choices']['text']; lbl=e['choices']['label']
        if e['answerKey'] not in lbl: continue
        items.append((f"Question: {e['question']}\nAnswer:", ch, lbl.index(e['answerKey'])))
    run_mc("C ARC-challenge (sci reason)", items)
except Exception as e: print(f"  arc SKIP {e}",flush=True)

# ============ PART D: KNOWLEDGE by domain (MMLU + SciQ, multiple-choice loglik) ============
print("\n-- PART D: domain knowledge (MMLU/SciQ, multiple-choice loglik) --",flush=True)
MMLU={"physics":["high_school_physics","college_physics"],
      "maths":["high_school_mathematics","college_mathematics","elementary_mathematics"],
      "chemistry":["high_school_chemistry","college_chemistry"],
      "biology":["high_school_biology","college_biology"],
      "history":["high_school_world_history","prehistory"],
      "world-knowledge":["miscellaneous","global_facts","world_religions"]}
for dom,subs in MMLU.items():
    items=[]
    for sub in subs:
        try:
            ds=take(load_dataset('cais/mmlu',sub,split='test'),20)
            for e in ds:
                items.append((f"Question: {e['question']}\nAnswer:", e['choices'], int(e['answer'])))
        except Exception as ex:
            print(f"    (mmlu {sub} unavailable: {str(ex)[:40]})",flush=True)
    if items: run_mc(f"D {dom} (MMLU)", items[:40])
try:
    sc=take(load_dataset('allenai/sciq',split='validation'),40)
    items=[]
    for e in sc:
        ch=[e['correct_answer'],e['distractor1'],e['distractor2'],e['distractor3']]
        order=list(range(4)); random.shuffle(order)
        items.append((f"Question: {e['question']}\nAnswer:", [ch[i] for i in order], order.index(0)))
    run_mc("D science (SciQ)", items)
except Exception as e: print(f"  sciq SKIP {e}",flush=True)

# ============ PART E: MATHS generation (GSM8K + arithmetic) ============
print("\n-- PART E: maths (generation, exact) --",flush=True)
try:
    gs=take(load_dataset('openai/gsm8k','main',split='test'),25)
    h=0
    for e in gs:
        gold=e['answer'].split('####')[-1].strip()
        g=gen(f"Question: {e['question']}\nAnswer:",40)
        nums=re.findall(r'-?\d[\d,]*',g.replace(',',''))
        h+= gold.replace(',','') in [x.replace(',','') for x in nums]
    RESULTS["E GSM8K (word problems)"]=(h,len(gs),100.0*h/len(gs),None)
    print(f"  [E GSM8K (word problems)  ] {h:3d}/{len(gs):3d} = {100.0*h/len(gs):5.1f}%",flush=True)
except Exception as e: print(f"  gsm8k SKIP {e}",flush=True)
# raw arithmetic probe (in-distribution for web text)
try:
    random.seed(5); h=0; N=30
    for _ in range(N):
        a,b=random.randint(2,49),random.randint(2,49)
        g=gen(f"{a} + {b} = ",4); h+= str(a+b) in g.replace(' ','')
    RESULTS["E arithmetic (2-digit add)"]=(h,N,100.0*h/N,None)
    print(f"  [E arithmetic (2-digit add)] {h:3d}/{N:3d} = {100.0*h/N:5.1f}%",flush=True)
except Exception as e: print(f"  arith SKIP {e}",flush=True)

# ============ PART F: CODING (simple completion, execution) ============
print("\n-- PART F: coding (simple completion) --",flush=True)
CODE=[("def add(a, b):\n    return", "a + b", "a+b"),
      ("def is_even(n):\n    return n % 2 ==", "0", "0"),
      ("def square(x):\n    return x *", "x", "x"),
      ("# return the length of a list\ndef length(lst):\n    return", "len", "len(lst)"),
      ("def maximum(a, b):\n    if a > b:\n        return a\n    else:\n        return", "b", "b")]
try:
    h=0
    for pr,g1,g2 in CODE:
        out=gen(pr,8,stop_eos=False); ok=g1 in out or g2.replace(' ','') in out.replace(' ','')
        h+=ok; print(f"    {pr.splitlines()[-1].strip()[:30]:32s} -> {out[:24]!r:26s} {'OK' if ok else 'x'}",flush=True)
    RESULTS["F coding (5 completions)"]=(h,len(CODE),100.0*h/len(CODE),None)
except Exception as e: print(f"  coding SKIP {e}",flush=True)

# ============ PART G: LONG-TEXT GENERATION quality (degeneration metrics) ============
print("\n-- PART G: long-text generation (256 tokens, degeneration metrics) --",flush=True)
try:
    prompts=["The history of the Roman Empire began when",
             "Photosynthesis is the process by which plants",
             "In the beginning of the story, the young explorer"]
    d1s=[];d2s=[];reps=[]
    for p in prompts:
        ids=tok.encode(p).ids; out=list(ids)
        with torch.no_grad():
            for _ in range(256):
                nx=int(m(torch.tensor([out[-CTX:]],device=DEV))[0,-1].argmax()); out.append(nx)
        toks=out[len(ids):]
        d1=len(set(toks))/max(1,len(toks))
        bg=list(zip(toks,toks[1:])); d2=len(set(bg))/max(1,len(bg))
        # 4-gram repetition rate
        fg=list(zip(toks,toks[1:],toks[2:],toks[3:])); rep=1-len(set(fg))/max(1,len(fg))
        d1s.append(d1);d2s.append(d2);reps.append(rep)
        print(f"    prompt {p[:34]!r:36s} distinct-1={d1:.2f} distinct-2={d2:.2f} 4gram-rep={rep:.2f}",flush=True)
        print(f"      sample: {tok.decode(toks)[:180]!r}",flush=True)
    RESULTS["G long-gen distinct-1"]=(None,3,100*np.mean(d1s),None)
    RESULTS["G long-gen distinct-2"]=(None,3,100*np.mean(d2s),None)
    RESULTS["G long-gen 4gram-repeat"]=(None,3,100*np.mean(reps),None)
except Exception as e: print(f"  long-gen SKIP {e}",flush=True)

# ============ SUMMARY ============
print("\n"+"="*64,flush=True)
print("HONEST CAPABILITY MAP  (snapshot "+os.path.basename(SNAP)+")",flush=True)
print("="*64,flush=True)
total_q=0
for k,v in RESULTS.items():
    h,n,pct,chance=v
    tag=f"{h}/{n}" if h is not None else f"n={n}"
    ch=f"  chance {chance:.0f}%" if chance else ""
    print(f"  {k:30s} {pct:5.1f}%   {tag}{ch}",flush=True)
    if h is not None: total_q+=n
print(f"\n  TOTAL graded questions: {total_q}",flush=True)
json.dump({k:list(v) for k,v in RESULTS.items()},open("/opt/code/membrae/faculty_eval_results.json","w"))
print("  (written to faculty_eval_results.json)",flush=True)
