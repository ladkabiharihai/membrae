"""UNIFIED BRAIN v2 -- ONE model, ALL FOUR faculties in a single network, with GROWTH built in.
Closes the biggest gap: the faculties were proven on SEPARATE toy models; here they live in ONE model trained on
a MIX of all task types, on the fast recall core (VChunkRecall, ~17x flash-attn). Growth is a MODEL METHOD
(function-preserving depth-grow: new block = exact identity via zero-init output projs) -> the brain GROWS
mid-life without forgetting. Proves: (1) memory/recall, (2) O(1) streaming, (3) agency/consequence,
(4) forget-free online learning, (G) growth preserves all faculties. All in-model, small scale.
"""
import warnings,os; warnings.filterwarnings("ignore"); os.environ["PYTHONWARNINGS"]="ignore"
import sys,time,random,copy,torch,torch.nn as nn,torch.nn.functional as F
sys.path.insert(0,"/opt/code/membrae")
from fastcore_v import VChunkRecall
DEV="cuda" if torch.cuda.is_available() else "cpu"; torch.manual_seed(0); random.seed(0)

# ---- one vocab spanning all faculties ----
NS=20; NK=32; NV=32; NCTX=4; NACT=4
PAD=0
SKQ=lambda i:1+i                         # skill query   (skill lives in WEIGHTS)
SKA=lambda i:1+NS+i                       # skill answer
KEY=lambda i:1+2*NS+i                     # memory key    (facts live in STATE)
VAL=lambda i:1+2*NS+NK+i                  # memory value
QUERY=1+2*NS+NK+NV                        # recall marker
ACTX=lambda c:QUERY+1+c                   # agency context
AACT=lambda a:QUERY+1+NCTX+a              # agency action
RGOOD=QUERY+1+NCTX+NACT; RBAD=RGOOD+1     # agency rewards
FILL=RBAD+1                               # streaming filler
VOCAB=FILL+1
PERM=list(range(NS)); random.shuffle(PERM)   # the fixed skill (permutation) -> must be learned into WEIGHTS

# ---- model: fast core + function-preserving growth ----
class Blk(nn.Module):
    def __init__(s,d):
        super().__init__(); s.n1=nn.LayerNorm(d); s.mix=VChunkRecall(d,feat=8,chunk=128); s.n2=nn.LayerNorm(d)
        s.mlp=nn.Sequential(nn.Linear(d,4*d),nn.GELU(),nn.Linear(4*d,d))
    def forward(s,x): x=x+s.mix(s.n1(x)); return x+s.mlp(s.n2(x))
    def make_identity(s):                                   # zero output projs -> block is EXACT identity
        nn.init.zeros_(s.mix.o.weight); nn.init.zeros_(s.mix.o.bias)
        nn.init.zeros_(s.mlp[-1].weight); nn.init.zeros_(s.mlp[-1].bias)
class UnifiedBrain(nn.Module):
    def __init__(s,d=192,L=3):
        super().__init__(); s.d=d; s.emb=nn.Embedding(VOCAB,d); s.pos=nn.Embedding(4096,d)
        s.blocks=nn.ModuleList([Blk(d) for _ in range(L)]); s.head=nn.Linear(d,VOCAB)
    def forward(s,x):
        x=x.clamp(0,VOCAB-1); h=s.emb(x)+s.pos(torch.arange(x.shape[1],device=x.device)[None])
        for b in s.blocks: h=b(h)
        return s.head(h)
    def grow(s):                                            # GROWTH IN THE MODEL: add a function-preserving block
        b=Blk(s.d).to(next(s.parameters()).device); b.make_identity(); s.blocks.append(b)
        return len(s.blocks)

# ---- task builders (each -> (seq, [(pos,target)...]) ) ----
def t_skill():                                             # random facts (distractors) + skill query -> answer
    pre=[];
    for _ in range(random.randint(0,4)):
        pre+=[KEY(random.randrange(NK)),VAL(random.randrange(NV))]
    i=random.randrange(NS); seq=pre+[SKQ(i)]; return seq,[(len(seq)-1,SKA(PERM[i]))]
def t_recall(npairs=8):
    ks=random.sample(range(NK),npairs); vs=[random.randrange(NV) for _ in ks]; seq=[]
    for k,v in zip(ks,vs): seq+=[KEY(k),VAL(v)]
    j=random.randrange(npairs); seq+=[QUERY,KEY(ks[j])]; return seq,[(len(seq)-1,VAL(vs[j]))]
def t_stream(fill=200):
    k=random.randrange(NK); v=random.randrange(NV)
    seq=[KEY(k),VAL(v)]+[FILL]*random.randint(fill//2,fill)+[QUERY,KEY(k)]; return seq,[(len(seq)-1,VAL(v))]
def t_agency(steps=24,explore=0.5):
    best={c:random.randrange(NACT) for c in range(NCTX)}; seen={}; seq=[]; tg=[]
    for _ in range(steps):
        c=random.randrange(NCTX); known=seen.get(c)
        a=known if (known is not None and random.random()>explore) else random.randrange(NACT)
        if known is not None: tg.append((len(seq)+1,AACT(known)))
        r=RGOOD if a==best[c] else RBAD
        if a==best[c]: seen[c]=a
        seq+=[ACTX(c),AACT(a),r]
    return seq,tg
BUILDERS=[t_skill,t_recall,t_stream,t_agency]
def make_batch(B):
    rows=[random.choice(BUILDERS)() for _ in range(B)]; T=max(len(s) for s,_ in rows)
    x=torch.full((B,T),PAD,dtype=torch.long); y=torch.full((B,T),-100,dtype=torch.long)
    for b,(s,tg) in enumerate(rows):
        x[b,:len(s)]=torch.tensor(s)
        for p,t in tg:
            if p<T: y[b,p]=t
    return x.to(DEV),y.to(DEV)

# ---- per-faculty evals (all on the SAME model) ----
@torch.no_grad()
def ev_skill(m,n=200):
    m.eval(); c=0
    for _ in range(n):
        pre=[];
        for _ in range(random.randint(0,4)): pre+=[KEY(random.randrange(NK)),VAL(random.randrange(NV))]
        i=random.randrange(NS); seq=pre+[SKQ(i)]
        c+= int(m(torch.tensor([seq],device=DEV))[0,-1].argmax())==SKA(PERM[i])
    m.train(); return c/n
@torch.no_grad()
def ev_recall(m,n=200,npairs=8):
    m.eval(); c=0
    for _ in range(n):
        s,tg=t_recall(npairs); p,t=tg[0]
        c+= int(m(torch.tensor([s],device=DEV))[0,p].argmax())==t
    m.train(); return c/n
@torch.no_grad()
def ev_stream(m,fill,n=60):
    m.eval(); c=0
    for _ in range(n):
        k=random.randrange(NK); v=random.randrange(NV)
        seq=[KEY(k),VAL(v)]+[FILL]*fill+[QUERY,KEY(k)]
        c+= int(m(torch.tensor([seq],device=DEV))[0,-1].argmax())==VAL(v)
    m.train(); return c/n
@torch.no_grad()
def ev_agency(m,steps=24,n=150):
    m.eval(); post_c=post_t=0
    for _ in range(n):
        best={c:random.randrange(NACT) for c in range(NCTX)}; seen=set(); seq=[]
        for _ in range(steps):
            c=random.randrange(NCTX); seq.append(ACTX(c))
            a=int(m(torch.tensor([seq],device=DEV))[0,-1].argmax())-(QUERY+1+NCTX); a=a if 0<=a<NACT else 0
            ok=(a==best[c])
            if c in seen: post_t+=1; post_c+=ok
            if ok: seen.add(c)
            seq+=[AACT(a),RGOOD if ok else RBAD]
    m.train(); return post_c/max(1,post_t)
@torch.no_grad()
def ev_forgetfree(m):
    """#1: learn NEW facts IN-STATE (context) -> does the weight-based SKILL survive? (vs weight-teach = 0)."""
    facts=[(random.randrange(NK),random.randrange(NV)) for _ in range(6)]; pre=[]
    for k,v in facts: pre+=[KEY(k),VAL(v)]
    # recall the new facts from state
    rc=0
    for k,v in facts:
        rc+= int(m(torch.tensor([pre+[QUERY,KEY(k)]],device=DEV))[0,-1].argmax())==VAL(v)
    # skill WITH those facts loaded in context (state) -> must be unchanged
    sc=0
    for _ in range(100):
        i=random.randrange(NS); sc+= int(m(torch.tensor([pre+[SKQ(i)]],device=DEV))[0,-1].argmax())==SKA(PERM[i])
    return rc/len(facts), sc/100

def report(m,tag):
    sk=ev_skill(m); rc=ev_recall(m); st=ev_stream(m,200); ag=ev_agency(m); ff_r,ff_s=ev_forgetfree(m)
    print(f"  [{tag}] skill(weights)={sk*100:.0f}%  recall(state)={rc*100:.0f}%  stream@200={st*100:.0f}%  "
          f"agency(exploit)={ag*100:.0f}%  | forget-free: newfacts={ff_r*100:.0f}% skill-kept={ff_s*100:.0f}%",flush=True)
    return dict(skill=sk,recall=rc,stream=st,agency=ag,ff_skill=ff_s)

def train(m,steps,lr=2e-3,t0=None):
    opt=torch.optim.AdamW(m.parameters(),lr=lr); m.train(); t0=t0 or time.time()
    for it in range(1,steps+1):
        x,y=make_batch(48); loss=F.cross_entropy(m(x).reshape(-1,VOCAB),y.reshape(-1),ignore_index=-100)
        opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(m.parameters(),1.); opt.step()
        if it%1000==0: print(f"    step {it}/{steps} loss={loss.item():.3f} ({it/(time.time()-t0):.0f} st/s)",flush=True)

if __name__=="__main__":
    print(f"UNIFIED BRAIN v2 -- one model, 4 faculties + growth. VOCAB={VOCAB}, start d=192/3L on {DEV}\n",flush=True)
    m=UnifiedBrain(d=192,L=3).to(DEV)
    print("PHASE 1: train the single model on the MIXED faculty stream (6000 steps)...",flush=True)
    train(m,6000)
    print("\nALL FOUR FACULTIES, one model, BEFORE growth:",flush=True); report(m,"pre-grow 3L")
    print(f"\nGROWTH (in-model, function-preserving): 3L -> {m.grow()}L. Faculties immediately after grow (should be UNCHANGED):",flush=True)
    report(m,"post-grow 4L (t=0)")
    print("\nPHASE 2: continue training the grown model (3000 steps)...",flush=True)
    train(m,3000)
    print("\nALL FOUR FACULTIES, grown model, AFTER continued training:",flush=True); r=report(m,"grown 4L trained")
    torch.save(m.state_dict(),"unified_brain_v2.pt")
    print(f"\n[saved unified_brain_v2.pt] params={sum(p.numel() for p in m.parameters())/1e6:.1f}M",flush=True)
    print("[done]",flush=True)
