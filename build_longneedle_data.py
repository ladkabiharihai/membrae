"""LONG-NEEDLE corpus: the fix for the #45 long-context collapse (recall 8%@512tok -> 0%@1k). Each example states
a FACT early, then 700-1900 tokens of REAL-TEXT distractors, then asks for it at the END -> next-token loss on the
answer REQUIRES carrying the fact across ~1-2k tokens (the O(1)-state long-context pressure our short SQuAD needles
never applied). Two kinds, both built in TOKEN space so the answer at the tail is NEVER truncated and distance is
controlled:
  (A) synthetic key-value  ("Dr. Rowan's access code is 7391" ... filler ... "Question: ...code? Answer: 7391")
  (B) real-fact (SQuAD)     answer-bearing context FIRST (needle at front = max distance), distractors, question last.
Filler = real SQuAD contexts (natural English -> does NOT teach degeneration). Merge into recall_train.bin.
"""
import warnings,os; warnings.filterwarnings("ignore"); os.environ["PYTHONWARNINGS"]="ignore"; os.environ["HF_HUB_DISABLE_PROGRESS_BARS"]="1"
import numpy as np, random
from datasets import load_dataset
from tokenizers import Tokenizer
tok=Tokenizer.from_file("data/bpe.json"); random.seed(0)
OUT="/mnt/kv_cache/pragnosia_data/longneedle_train.bin"; EOS=0; CTX=2048; NL=tok.encode("\n\n").ids

print("loading SQuAD pool...",flush=True)
sq=load_dataset('rajpurkar/squad_v2',split='train')
seen=set(); qa=[]
for ex in sq:
    c=ex['context']; seen.add(c)
    if ex['answers']['text']: qa.append((c,ex['question'],ex['answers']['text'][0]))
contexts=list(seen)
POOL=[tok.encode(c).ids for c in contexts[:9000]]          # pre-tokenized filler pool (natural text)
print(f"  {len(contexts)} contexts, {len(qa)} QA, pool={len(POOL)}",flush=True)

names=["Dr. Rowan","Captain Vance","Ms. Okafor","Professor Lin","Sergeant Boyd","Elena Marsh","Marcus Reed","Ada Fenn","Colonel Hart","Nadia Vale"]
attrs=[("access code",4),("badge number",3),("favorite color",0),("locker number",2),("codeword",0),("room number",3)]
vals_word=["crimson","teal","amber","violet","olive","indigo","scarlet","turquoise","falcon","lantern","harbor","meridian","cobalt","thistle","zephyr","granite","willow","quartz"]

buf=[]
def emit(prefix_ids, suffix_ids, total_target):
    budget=max(50, total_target-len(prefix_ids)-len(suffix_ids))
    fill=[]
    while len(fill)<budget: fill+=random.choice(POOL)+NL
    fill=fill[:budget]
    doc=(prefix_ids+fill+suffix_ids)[:CTX-1]                # tail (answer) preserved; only over-long filler trims
    buf.extend(doc); buf.append(EOS)

N_SYN=42000; N_REAL=32000
print(f"synthetic key-value long needles x{N_SYN}...",flush=True)
for _ in range(N_SYN):
    who=random.choice(names); attr,ndig=random.choice(attrs)
    val=("".join(str(random.randint(0,9)) for _ in range(ndig)) if ndig else random.choice(vals_word))
    pre=tok.encode(f"Important note: {who}'s {attr} is {val}. Remember it for later.").ids
    suf=tok.encode(f"\n\nQuestion: What is {who}'s {attr}?\nAnswer: {val}").ids
    emit(pre,suf,random.randint(750,1900))
print(f"  running tokens={len(buf)/1e6:.0f}M",flush=True)

print(f"real-fact (SQuAD) long needles x{N_REAL}...",flush=True)
for _ in range(N_REAL):
    c,q,a=random.choice(qa)
    pre=tok.encode(c).ids                                  # answer-bearing context at the FRONT (max distance)
    suf=tok.encode(f"\n\nQuestion: {q}\nAnswer: {a}").ids
    emit(pre,suf,random.randint(750,1800))
print(f"  running tokens={len(buf)/1e6:.0f}M",flush=True)

arr=np.array(buf,dtype=np.uint16); arr.tofile(OUT)
print(f"\nWROTE {OUT}: {len(arr)/1e6:.0f}M tokens ({N_SYN} synthetic + {N_REAL} real-fact long needles, dist 750-1900 tok)",flush=True)
