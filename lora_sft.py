import sys; sys.path.insert(0,"/home/ashish/Downloads/membrae")
import json, numpy as np, torch, torch.nn.functional as F, os
import s6_hybrid as H, lora
from tokenizers import Tokenizer
P=lambda *a:print(*a,flush=True)
c=json.load(open("pragnosia.json")); H.VOC,H.L=c["vocab"],c["ctx"]; DEV="cuda"; CTX=c["ctx"]
torch.manual_seed(0); np.random.seed(0)
tok=Tokenizer.from_file(c["tokenizer"])
m=H.SpinAttentionLM(c["vocab"],c["d"],c["heads"],c["layers"],mlp_mult=c["mlp_mult"],carrier=c["carrier"])
m.load_state_dict(torch.load(c["ckpt"],map_location="cpu",weights_only=True)); m=m.bfloat16().to(DEV); m.grad_checkpoint=False
def ask(q,n=24):
    ids=tok.encode(f"<user> {q} <assistant>").ids
    return tok.decode(m.generate(ids,n_new=n,window=CTX,temp=0.0,rep=1.3,eos=0)).strip()
TESTS=["Hi","Who are you?","What is the capital of France?","Who wrote Hamlet?","What is photosynthesis?"]
m.eval(); P("=== BEFORE SFT ==="); [P(f"  {q!r:26} -> {ask(q)[:60]!r}") for q in TESTS]
# LoRA SFT
n=lora.inject_lora(m,r=16); lp=lora.lora_params(m)
P(f"injected {n} LoRA layers, {sum(p.numel() for p in lp)/1e6:.1f}M trainable")
opt=torch.optim.AdamW(lp,lr=2e-4,betas=(0.9,0.95))
sft=np.memmap("data/sft.bin",dtype=np.int16,mode="r")
rep=np.memmap(f"data/{c['train_bin']}.bin",dtype=np.int16,mode="r") if os.path.exists(f"data/{c['train_bin']}.bin") else sft
def batch(data,bs=2):
    ix=np.random.randint(0,len(data)-CTX-1,bs)
    x=torch.tensor(np.stack([np.asarray(data[i:i+CTX]) for i in ix]),dtype=torch.long,device=DEV)
    y=torch.tensor(np.stack([np.asarray(data[i+1:i+CTX+1]) for i in ix]),dtype=torch.long,device=DEV)
    return x,y
m.train()
for step in range(500):
    data=sft if np.random.rand()<0.7 else rep     # 70% chat SFT, 30% replay (don't forget)
    x,y=batch(data)
    loss=F.cross_entropy(m(x).reshape(-1,c["vocab"]),y.reshape(-1))
    opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(lp,1.0); opt.step()
    if step%100==0: P(f"  step {step} loss {loss.item():.2f}  peakGPU {torch.cuda.max_memory_allocated()/1e9:.1f}GB")
m.eval(); P("=== AFTER SFT ==="); [P(f"  {q!r:26} -> {ask(q)[:60]!r}") for q in TESTS]
# save the adapter (does NOT overwrite the base checkpoint)
torch.save({k:v.detach().cpu() for k,v in m.state_dict().items() if k.endswith(".A") or k.endswith(".B")}, "pragnosia_sft_lora.pt")
P("saved LoRA adapter -> pragnosia_sft_lora.pt (base checkpoint untouched)")
P("SFT OK")
