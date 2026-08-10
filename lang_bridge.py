"""LANGUAGE BRIDGE: put the intrinsic-brain fast core (VChunkRecall) on REAL TEXT. Train a small LM on the
16K-BPE corpus, then test (a) does it produce coherent language, and (b) does the MEMORY faculty work on NATURAL
LANGUAGE -- an in-context needle: state a fact in real prose, then ask, does the state recall it. Converts the
synthetic faculty proofs to real language. Small + fast (provable-small rule)."""
import warnings,os; warnings.filterwarnings("ignore"); os.environ["PYTHONWARNINGS"]="ignore"
import sys,time,random,numpy as np,torch,torch.nn as nn,torch.nn.functional as F
sys.path.insert(0,"/opt/code/membrae")
from fastcore_v import VChunkRecall
from tokenizers import Tokenizer
DEV="cuda"; torch.manual_seed(0); random.seed(0)
tok=Tokenizer.from_file("data/bpe.json"); VOCAB=tok.get_vocab_size()
DATA=np.memmap("data/big_train.bin",dtype=np.uint16,mode="r")   # 8B tok real text

class Blk(nn.Module):
    def __init__(s,d): super().__init__(); s.n1=nn.LayerNorm(d); s.mix=VChunkRecall(d,heads=6,feat=8,chunk=128); s.n2=nn.LayerNorm(d); s.mlp=nn.Sequential(nn.Linear(d,4*d),nn.GELU(),nn.Linear(4*d,d))
    def forward(s,x): x=x+s.mix(s.n1(x)); return x+s.mlp(s.n2(x))
class LM(nn.Module):
    def __init__(s,d=384,L=6,ctx=512): super().__init__(); s.emb=nn.Embedding(VOCAB,d); s.pos=nn.Embedding(8192,d); s.blocks=nn.ModuleList([Blk(d) for _ in range(L)]); s.head=nn.Linear(d,VOCAB,bias=False); s.head.weight=s.emb.weight
    def forward(s,x):
        h=s.emb(x)+s.pos(torch.arange(x.shape[1],device=x.device)[None])
        for b in s.blocks: h=b(h)
        return s.head(h)

CTX=512
def batch(B):
    ix=torch.randint(0,len(DATA)-CTX-1,(B,))
    x=torch.stack([torch.from_numpy(DATA[i:i+CTX].astype(np.int64)) for i in ix])
    y=torch.stack([torch.from_numpy(DATA[i+1:i+CTX+1].astype(np.int64)) for i in ix])
    return x.to(DEV),y.to(DEV)

@torch.no_grad()
def gen(m,prompt,n=40,rep=1.3):
    m.eval(); ids=tok.encode(prompt).ids; out=[]
    for _ in range(n):
        lg=m(torch.tensor([(ids+out)[-CTX:]],device=DEV))[0,-1].float()
        for t in set(out[-32:]): lg[t]/=rep
        nx=int(lg.argmax());
        if nx==0: break
        out.append(nx)
    m.train(); return tok.decode(out)

@torch.no_grad()
def needle_lang(m,n=40):
    """MEMORY faculty ON LANGUAGE: a fact in real-ish prose, distractor sentences, then a question. Does the
    model's STATE recall the planted entity? Uses varied names/values so it can't be memorized."""
    m.eval(); names=["Dr. Rowan","Mr. Alvarez","Captain Vance","Ms. Okafor","Professor Lin","Sergeant Boyd"]
    objs=["a brass key","a red notebook","an old compass","a silver coin","a folded map","a glass vial"]
    fill=("The afternoon was quiet and the corridor smelled faintly of rain. "
          "Somewhere a clock ticked while papers rustled on the desk. "
          "Outside, the market carried on with its usual murmur and clatter. ")
    hit=0
    for _ in range(n):
        who=random.choice(names); what=random.choice(objs)
        text=f"{who} kept {what} in the top drawer. "+fill*random.randint(2,6)+f"The item that {who} kept in the drawer was"
        g=gen(m,text,n=8)
        hit+= what.split()[-1] in g.lower()      # the key noun recalled
    m.train(); return hit/n

def train(m,steps,lr=3e-4):
    opt=torch.optim.AdamW(m.parameters(),lr=lr,betas=(0.9,0.95),weight_decay=0.1); m.train(); t0=time.time()
    for it in range(1,steps+1):
        x,y=batch(24); loss=F.cross_entropy(m(x).reshape(-1,VOCAB),y.reshape(-1))
        opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(m.parameters(),1.); opt.step()
        if it%500==0: print(f"  step {it}/{steps} loss={loss.item():.3f} ppl={torch.exp(loss).item():.1f} ({it/(time.time()-t0):.0f} st/s)",flush=True)

if __name__=="__main__":
    m=LM(d=384,L=6,ctx=CTX).to(DEV)
    P=sum(p.numel() for p in m.parameters())/1e6
    print(f"LANGUAGE BRIDGE: VChunkRecall LM, {P:.0f}M params, real 16K-BPE text\n",flush=True)
    STEPS=int(os.environ.get("STEPS","12000"))
    print(f"training {STEPS} steps on real text (data/big_train)...",flush=True)
    train(m,STEPS)
    print("\n=== coherence (greedy) ===",flush=True)
    for p in ["The capital of France is","Water is made of","In the morning, she"]:
        print(f"  {p!r} -> {gen(m,p,28)!r}",flush=True)
    print("\n=== MEMORY faculty ON LANGUAGE (in-context needle in prose) ===",flush=True)
    print(f"  recall of planted entity through distractor text: {needle_lang(m)*100:.0f}%",flush=True)
    torch.save(m.state_dict(),"lang_bridge.pt")
    print(f"\n[saved lang_bridge.pt] {P:.0f}M params. [done]",flush=True)
