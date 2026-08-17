"""Clean answer to 'does adding MoE break the recall faculty?' -- use the REFERENCE MQAR harness structure (the one
that hits ~100%: untied head, no final-norm, stable=False, chunk=64) and swap ONLY the MLP for a top-1 MoE. If PLAIN
=~100% and MoE=~100%, the faculty is intact with MoE (recall lives in the VChunkRecall mix, untouched by the MLP swap).
"""
import warnings,os,random; warnings.filterwarnings("ignore"); os.environ["PYTHONWARNINGS"]="ignore"
import sys,torch,torch.nn as nn,torch.nn.functional as F; sys.path.insert(0,"/opt/code/membrae")
from fastcore_v import VChunkRecall
DEV="cuda"; d=128; H=4; Vc=129
class MoEMLP(nn.Module):
    def __init__(s,E):
        super().__init__(); s.E=E; s.router=nn.Linear(d,E)
        s.experts=nn.ModuleList([nn.Sequential(nn.Linear(d,4*d),nn.GELU(),nn.Linear(4*d,d)) for _ in range(E)])
    def forward(s,x):
        B,T,_=x.shape; xf=x.reshape(-1,d); lg=s.router(xf); pr=lg.softmax(-1); top=lg.argmax(-1); out=torch.zeros_like(xf)
        for e in range(s.E):
            m=top==e
            if m.any(): out[m]=s.experts[e](xf[m])*pr[m,e:e+1]
        return out.reshape(B,T,d)
class Blk(nn.Module):
    def __init__(s,E):
        super().__init__(); s.n1=nn.LayerNorm(d); s.mix=VChunkRecall(d,heads=H,feat=8,chunk=64); s.n2=nn.LayerNorm(d)  # stable=False (reference)
        s.mlp=(nn.Sequential(nn.Linear(d,4*d),nn.GELU(),nn.Linear(4*d,d)) if E==0 else MoEMLP(E))
    def forward(s,x): x=x+s.mix(s.n1(x)); return x+s.mlp(s.n2(x))
class LM(nn.Module):   # reference structure: untied head, NO final norm
    def __init__(s,E,L=2): super().__init__(); s.emb=nn.Embedding(Vc,d); s.pos=nn.Embedding(4096,d); s.blocks=nn.ModuleList([Blk(E) for _ in range(L)]); s.head=nn.Linear(d,Vc)
    def forward(s,x):
        h=s.emb(x)+s.pos(torch.arange(x.shape[1],device=x.device))
        for b in s.blocks: h=b(h)
        return s.head(h)
def mqar(B,n=16,K=64,Vv=64):
    T=2*n+8; x=torch.zeros(B,T,dtype=torch.long); y=torch.full((B,T),-100,dtype=torch.long)
    for b in range(B):
        ks=random.sample(range(1,K+1),n); vs=[random.randint(K+1,K+Vv) for _ in ks]; kv=dict(zip(ks,vs)); s=[]
        for a,c in zip(ks,vs): s+=[a,c]
        qs=[random.choice(ks) for _ in range(4)]
        for qq in qs: s+=[qq,kv[qq]]
        x[b,:len(s)]=torch.tensor(s[:T])
        for j in range(4):
            p=2*n+2*j
            if p<T: y[b,p]=kv[qs[j]]
    return x.to(DEV),y.to(DEV)
def rec(E,steps=2000):
    random.seed(0); torch.manual_seed(0); m=LM(E).to(DEV); opt=torch.optim.AdamW(m.parameters(),lr=2e-3)
    for _ in range(steps):
        x,y=mqar(48); loss=F.cross_entropy(m(x).reshape(-1,Vc),y.reshape(-1),ignore_index=-100)
        if torch.isfinite(loss): opt.zero_grad(); loss.backward(); opt.step()
    m.eval(); c=t=0
    with torch.no_grad():
        for _ in range(15):
            x,y=mqar(48); p=m(x).argmax(-1); msk=y!=-100; c+=(p[msk]==y[msk]).sum().item(); t+=msk.sum().item()
    return 100*c/t
if __name__=="__main__":
    print("FACULTY CHECK (reference harness): does MoE break recall?\n",flush=True)
    for E,name in [(0,"plain MLP (reference)"),(4,"MoE E=4"),(8,"MoE E=8")]:
        print(f"  {name:22s} MQAR-16 recall = {rec(E):.0f}%",flush=True)
    print("  -> if plain ~100% AND MoE ~100%: faculty INTACT with MoE.",flush=True); print("DONE",flush=True)
