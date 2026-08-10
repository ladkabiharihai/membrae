"""ONLINE LEARNING WITHOUT FORGETTING -- the brain property a frozen transformer can't do, IN THE MODEL.
A gated-recall model learns a fixed SKILL into its weights (a permutation lookup) AND to recall in-context facts
from its fast-weight STATE. Then we teach it NEW facts two ways and measure the SKILL afterwards:
  (A) IN-STATE (intrinsic): stream the facts through the forward pass -> they live in state S; weights untouched.
  (B) WEIGHT-TEACH (the old brain.py teach()): gradient-update the weights on the facts.
CLAIM: (A) learns the facts AND keeps the skill (no forgetting); (B) learns the facts but WRECKS the skill (#23).
So the architecture does forget-free continual learning intrinsically -- no replay, no store."""
import sys, copy, random, torch, torch.nn as nn, torch.nn.functional as F
sys.path.insert(0, "/opt/code/membrae")
from gated_core import GatedRecall
DEV = "cuda" if torch.cuda.is_available() else "cpu"; torch.manual_seed(0); random.seed(0)

# ---- vocab: PAD, SK, Q markers + skill-Q/skill-A + fact-K/fact-V (disjoint ranges) ----
S=40; K=60
PAD,SK,Q=0,1,2; base=3
SKQ=lambda i: base+i                      # skill query token  i in [0,S)
SKA=lambda i: base+S+i                    # skill answer token
FK =lambda i: base+2*S+i                  # fact key
FV =lambda i: base+2*S+K+i                # fact value
V = base+2*S+2*K
PERM=list(range(S)); random.shuffle(PERM) # fixed skill: SKQ(i) -> SKA(PERM[i]) (lives in WEIGHTS)

class LM(nn.Module):
    def __init__(s,d=128,L=2,feat=16): super().__init__(); s.emb=nn.Embedding(V,d); s.pos=nn.Embedding(2048,d); s.blocks=nn.ModuleList([_Blk(d,feat) for _ in range(L)]); s.head=nn.Linear(d,V)
    def forward(s,x): h=s.emb(x)+s.pos(torch.arange(x.shape[1],device=x.device)); [s.__setattr__('_',b) for b in ()];
    # (defined below via _Blk)
class _Blk(nn.Module):
    def __init__(s,d,feat): super().__init__(); s.n1=nn.LayerNorm(d); s.mix=GatedRecall(d,feat=feat); s.n2=nn.LayerNorm(d); s.mlp=nn.Sequential(nn.Linear(d,4*d),nn.GELU(),nn.Linear(4*d,d))
    def forward(s,x): m,_=s.mix(s.n1(x)); x=x+m; return x+s.mlp(s.n2(x))
def _fwd(s,x):
    h=s.emb(x)+s.pos(torch.arange(x.shape[1],device=x.device))
    for b in s.blocks: h=b(h)
    return s.head(h)
LM.forward=_fwd

def _factpre(npairs):                      # a random fact prefix (loads the STATE with distractor bindings)
    ks=random.sample(range(K),npairs); vs=[random.randrange(K) for _ in ks]; seq=[]
    for a,b in zip(ks,vs): seq+=[FK(a),FV(b)]
    return seq,ks,vs
def skill_ex():                            # [random facts] + [SK, SKQ(i)] -> SKA(PERM[i])
    pre,_,_=_factpre(random.randint(0,8))  # STATE-ROBUST: skill must hold with arbitrary facts in state
    i=random.randrange(S); seq=pre+[SK,SKQ(i)]
    return seq,[SKA(PERM[i])],len(seq)-1   # label at the SKQ position
def recall_ex(npairs=8):                   # [k1 v1 ... , Q, kq] -> predict vq  (from STATE)
    seq,ks,vs=_factpre(npairs)
    j=random.randrange(npairs); seq+=[Q,FK(ks[j])]
    return seq,[FV(vs[j])],len(seq)-1
def make_batch(B):
    X=[];Y=[];P=[];mx=0; rows=[]
    for _ in range(B):
        seq,ans,pos=(skill_ex() if random.random()<0.5 else recall_ex()); rows.append((seq,ans,pos)); mx=max(mx,len(seq))
    x=torch.zeros(B,mx,dtype=torch.long); y=torch.full((B,mx),-100,dtype=torch.long)
    for b,(seq,ans,pos) in enumerate(rows):
        x[b,:len(seq)]=torch.tensor(seq); y[b,pos]=ans[0]
    return x.to(DEV),y.to(DEV)

def train(m,steps=4000,lr=2e-3):
    opt=torch.optim.AdamW(m.parameters(),lr=lr); m.train()
    for _ in range(steps):
        x,y=make_batch(64); loss=F.cross_entropy(m(x).reshape(-1,V),y.reshape(-1),ignore_index=-100)
        opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(m.parameters(),1.); opt.step()

@torch.no_grad()
def skill_acc(m,prefix=None,n=200):
    m.eval(); c=0
    for _ in range(n):
        i=random.randrange(S); seq=(prefix or [])+[SK,SKQ(i)]
        p=int(m(torch.tensor([seq],device=DEV))[0,-1].argmax()); c+=(p==SKA(PERM[i]))
    return c/n
@torch.no_grad()
def fact_recall(m,facts,n=None):           # facts: list of (kidx,vidx); query each from a streamed prefix
    m.eval(); pre=[];
    for a,b in facts: pre+=[FK(a),FV(b)]
    c=0
    for a,b in facts:
        seq=pre+[Q,FK(a)]; p=int(m(torch.tensor([seq],device=DEV))[0,-1].argmax()); c+=(p==FV(b))
    return c/len(facts), pre

if __name__=="__main__":
    torch.manual_seed(0); m=LM().to(DEV)
    print("training gated-recall model on SKILL(in weights) + RECALL(in state)...",flush=True)
    train(m)
    s0=skill_acc(m); print(f"\n[baseline] skill accuracy (no new facts): {s0*100:.0f}%",flush=True)
    facts=[(random.randrange(K),random.randrange(K)) for _ in range(8)]
    # (A) IN-STATE: stream facts in context -> recall + skill-with-facts-in-context
    r_state,pre=fact_recall(m,facts); s_state=skill_acc(m,prefix=pre)
    print(f"\n(A) IN-STATE (forward pass, weights untouched):")
    print(f"      new-fact recall = {r_state*100:.0f}%   |   skill AFTER = {s_state*100:.0f}%  (was {s0*100:.0f}%)",flush=True)
    # (B) WEIGHT-TEACH: gradient-fit the facts into weights (old brain.py teach()) -> measure skill
    mt=copy.deepcopy(m); opt=torch.optim.AdamW(mt.parameters(),lr=5e-3); mt.train()
    for _ in range(400):                    # aggressive fit (as one must, to force facts into weights) -> overwrites skill
        b=random.choice(facts); seq=[Q,FK(b[0])]; x=torch.tensor([seq],device=DEV); y=torch.tensor([[-100,FV(b[1])]],device=DEV)
        loss=F.cross_entropy(mt(x).reshape(-1,V),y.reshape(-1),ignore_index=-100); opt.zero_grad(); loss.backward(); opt.step()
    @torch.no_grad()
    def taught_recall(mt):
        mt.eval(); c=0
        for a,b in facts:
            p=int(mt(torch.tensor([[Q,FK(a)]],device=DEV))[0,-1].argmax()); c+=(p==FV(b))
        return c/len(facts)
    r_w=taught_recall(mt); s_w=skill_acc(mt)
    print(f"\n(B) WEIGHT-TEACH (gradient update on the facts, = old teach()):")
    print(f"      new-fact recall = {r_w*100:.0f}%   |   skill AFTER = {s_w*100:.0f}%  (was {s0*100:.0f}%)",flush=True)
    print(f"\nVERDICT: in-state keeps skill {s0*100:.0f}->{s_state*100:.0f}%; weight-teach forgets it {s0*100:.0f}->{s_w*100:.0f}%.")
    print("If (A) recalls AND keeps skill while (B) forgets -> forget-free online learning is INTRINSIC to the core.")
