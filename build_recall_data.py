"""Build a high-quality RECALL corpus (recall_train.bin) that PRESSURES the memory faculty: reading-comprehension
QA where predicting the answer REQUIRES retrieving a fact from the context. SQuAD v2 (short retrieval) + HotpotQA
distractor (answer buried in 10 paragraphs = needle-in-haystack + multi-hop). Tokenized with our 16K BPE to match
window2. Mix this into scale_train.py so next-token loss finally depends on carrying a fact across distance."""
import warnings,os; warnings.filterwarnings("ignore"); os.environ["HF_HUB_DISABLE_PROGRESS_BARS"]="1"
import numpy as np, random
from datasets import load_dataset
from tokenizers import Tokenizer
tok=Tokenizer.from_file("data/bpe.json"); random.seed(0)
OUT="/mnt/kv_cache/pragnosia_data/recall_train.bin"
EOS=0
buf=[]
def add(text):
    ids=tok.encode(text).ids
    buf.extend(ids); buf.append(EOS)

print("SQuAD v2 -> QA format...",flush=True)
sq=load_dataset('rajpurkar/squad_v2',split='train')
n_sq=0
for ex in sq:
    ans=ex['answers']['text']
    if ans:                                                   # answerable: teaches retrieval
        text=f"{ex['context']}\n\nQuestion: {ex['question']}\nAnswer: {ans[0]}"
        add(text); n_sq+=1
    elif random.random()<0.15:                                # a few unanswerable -> teaches abstention
        add(f"{ex['context']}\n\nQuestion: {ex['question']}\nAnswer: unanswerable"); n_sq+=1
print(f"  SQuAD: {n_sq} QA, running tokens={len(buf)/1e6:.0f}M",flush=True)

print("HotpotQA distractor -> needle-in-10-paragraphs + multi-hop...",flush=True)
hp=load_dataset('hotpotqa/hotpot_qa','distractor',split='train')
n_hp=0
for ex in hp:
    ctx=ex['context']                                         # {'title':[...], 'sentences':[[...],...]}
    paras=[]
    for title,sents in zip(ctx['title'],ctx['sentences']):
        paras.append(title+": "+" ".join(sents))
    random.shuffle(paras)                                     # answer paragraph at random depth (needle)
    doc="\n\n".join(paras)
    text=f"{doc}\n\nQuestion: {ex['question']}\nAnswer: {ex['answer']}"
    add(text); n_hp+=1
print(f"  HotpotQA: {n_hp} multi-hop, running tokens={len(buf)/1e6:.0f}M",flush=True)

arr=np.array(buf,dtype=np.uint16)
arr.tofile(OUT)
print(f"\nWROTE {OUT}: {len(arr)/1e6:.0f}M tokens ({n_sq} SQuAD + {n_hp} HotpotQA)",flush=True)
print("Mix ~15% of scale_train batches from this -> next-token loss now REQUIRES recall.",flush=True)
