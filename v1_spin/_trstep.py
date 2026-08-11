import torch, json, torch.nn.functional as F
import s6_hybrid as H
dev="cuda"; H.VOC,H.L,H.DEVICE=16384,256,dev
CFG=json.load(open("pragnosia_1b_combined.json")); td=H.load(CFG["train_bin"]); torch.manual_seed(0)
def trial(use_bf16, steps=10, W=32, lr=2e-4):
    sd=torch.load("pragnosia_spin.pt",map_location="cpu",weights_only=True)
    m=H.SpinAttentionLM(16384,768,12,CFG["layers"],mlp_mult=CFG["mlp_mult"],carrier="spin_dominant").to(dev).train()
    m.load_state_dict(sd)
    opt=torch.optim.AdamW(m.parameters(),lr=lr,betas=(0.9,0.95),fused=True)
    actx=torch.autocast("cuda",dtype=torch.bfloat16) if use_bf16 else torch.autocast("cuda",enabled=False)
    tag="bf16" if use_bf16 else "fp32"
    for it in range(steps):
        opt.zero_grad(); x,y=H.batch_long(td,2,W); S=None; badfwd=-1; L=0.0
        for w in range(W):
            xw=x[:,w*256:(w+1)*256]; yw=y[:,w*256:(w+1)*256]
            with actx:
                lg,S=m(xw,state=S,return_state=True); lw=F.cross_entropy(lg.reshape(-1,16384),yw.reshape(-1))
            if not torch.isfinite(lg).all(): badfwd=w
            if not torch.isfinite(lw): badfwd=w; break
            (lw/W).backward(); S=[s.detach() for s in S]; L+=lw.item()
        gnorm=torch.nn.utils.clip_grad_norm_(m.parameters(),1.0); gfin=bool(torch.isfinite(gnorm))
        if badfwd>=0 or not gfin:
            where = "FORWARD overflow @window "+str(badfwd) if badfwd>=0 else "BACKWARD overflow (forward was finite)"
            print(f"  [{tag}] step {it}: NON-FINITE -> {where}; gnorm={float(gnorm):.2e}", flush=True); return
        opt.step()
        print(f"  [{tag}] step {it}: loss={L/W:.3f} gnorm={float(gnorm):.2f} OK", flush=True)
    print(f"  [{tag}] {steps} steps CLEAN", flush=True)
for bf in (True, False):
    print(f"=== {'bf16 autocast' if bf else 'fp32'} ===", flush=True); trial(bf)
