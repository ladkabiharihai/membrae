import warnings; warnings.filterwarnings("ignore")
import sys; sys.path.insert(0,"/opt/code/membrae")
import torch,numpy as np,torch.nn as nn,torch.nn.functional as F
import scale_train as S
def probe(CTX,BS,d=512,L=4):
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    m=S.BrainLM(d=d,L=L).cuda(); opt=torch.optim.AdamW(m.parameters(),lr=6e-4)
    def batch(B):
        ix=np.random.randint(0,len(S.DATA)-CTX-1,size=B)
        x=torch.from_numpy(np.stack([S.DATA[i:i+CTX] for i in ix]).astype(np.int64)); y=torch.from_numpy(np.stack([S.DATA[i+1:i+CTX+1] for i in ix]).astype(np.int64))
        return x.cuda(),y.cuda()
    try:
        for _ in range(3):
            x,y=batch(BS); loss=F.cross_entropy(m(x).reshape(-1,S.VOCAB),y.reshape(-1)); opt.zero_grad(); loss.backward(); opt.step()
        torch.cuda.synchronize()
        print(f"  ctx={CTX:5d} bs={BS:2d}  peak_train={torch.cuda.max_memory_allocated()/2**30:4.1f}G  free_now={torch.cuda.mem_get_info()[0]/2**30:2.0f}G",flush=True)
    except RuntimeError as e:
        print(f"  ctx={CTX:5d} bs={BS:2d}  {'OOM' if 'out of memory' in str(e).lower() else str(e)[:40]}",flush=True); torch.cuda.empty_cache()
    del m,opt; torch.cuda.empty_cache()
if __name__=="__main__":
    print(f"VRAM probe (start 23M/d512/L4; note it GROWS later). prod uses the rest.",flush=True)
    for CTX,BS in [(2048,8),(4096,16),(8192,8),(4096,24),(8192,16)]: probe(CTX,BS)
