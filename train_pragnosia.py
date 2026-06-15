"""
Train Pragnosia's language faculty from pragnosia.json. GPU-ADAPTIVE: it detects
the GPU's VRAM and auto-tunes batch size, gradient accumulation, and precision —
so if you move to a bigger GPU and (re)start or --resume, it just picks the new
hardware up and trains optimally. Resumable from the auto-saved checkpoint.

  python3 train_pragnosia.py            # from scratch (new data)
  python3 train_pragnosia.py --resume   # continue from the checkpoint (any GPU)
"""
import argparse, json, math, os, sys, time
import numpy as np, torch, torch.nn.functional as F
import s6_hybrid as H
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
DEVICE = H.DEVICE
CFG = json.load(open("pragnosia.json"))
H.VOC, H.L = CFG["vocab"], CFG["ctx"]
VOC = CFG["vocab"]

def autotune():
    """Pick batch / grad-accum / precision / compile from the actual GPU."""
    if DEVICE != "cuda":
        return dict(bs=4, accum=4, bf16=False, compile=False, gpu="cpu", vram=0)
    p = torch.cuda.get_device_properties(0); vram = p.total_memory / 2**30
    bf16 = torch.cuda.is_bf16_supported()
    # micro-batch that fits, scaled to VRAM (measured: 176M ~ bs8 @ 8GB)
    bs = max(4, int(vram // 1.0))                 # ~1 GB per micro-batch unit
    bs = min(bs, 256)
    target_eff = 64                                # keep a sane effective batch
    accum = max(1, target_eff // bs)
    # Optimal per GPU class (measured): small GPU -> fp32 + COMPILE (compile fuses
    # the sequential spin loop, fastest here). Big GPU -> bf16, no compile (big
    # batch + bf16 throughput; avoids the compile+tied-weights dtype bug).
    small = vram <= 12
    use_compile = small
    use_bf16 = bf16 and not small
    return dict(bs=bs, accum=accum, bf16=use_bf16, compile=use_compile, gpu=p.name, vram=round(vram, 1))

def fit_batch(m, td, start_bs, bf16):
    """Find the largest micro-batch that actually fits THIS model on THIS GPU --
    so the trainer adapts to any model size + any card, never OOMs mid-run."""
    ac = torch.autocast("cuda", dtype=torch.bfloat16) if bf16 else torch.autocast("cuda", enabled=False)
    bs = max(1, start_bs)
    while bs > 1:
        try:
            torch.cuda.empty_cache();
            x, y = H.batch(td, bs)
            with ac:
                loss = F.cross_entropy(m(x).reshape(-1, VOC), y.reshape(-1))
            loss.backward()
            for pa in m.parameters(): pa.grad = None
            torch.cuda.empty_cache()
            return max(1, int(bs * 0.9))            # 10% headroom for compile/eval
        except RuntimeError as e:
            if "out of memory" not in str(e).lower(): raise
            for pa in m.parameters(): pa.grad = None
            torch.cuda.empty_cache()
            bs = int(bs * 0.7)
    return 1

def main(steps, lr, resume, override_bs, grow_enabled):
    torch.manual_seed(0); np.random.seed(0)
    cfg = autotune()
    td, vd = H.load(CFG["train_bin"]), H.load(CFG["valid_bin"])
    m = H.SpinAttentionLM(VOC, CFG["d"], CFG["heads"], CFG["layers"]).to(DEVICE)
    p = sum(x.numel() for x in m.parameters())
    if resume and os.path.exists(CFG["ckpt"]):
        m.load_state_dict(torch.load(CFG["ckpt"], map_location=DEVICE, weights_only=True)); print("resumed", flush=True)
    # adapt batch to the ACTUAL model+GPU (robust to any size), then keep eff batch ~64
    if override_bs:
        cfg["bs"] = override_bs
    else:
        cfg["bs"] = fit_batch(m, td, cfg["bs"], cfg["bf16"])
    cfg["accum"] = max(1, 64 // cfg["bs"])
    fwd = torch.compile(m, dynamic=False) if cfg["compile"] else m
    bs, accum, bf16 = cfg["bs"], cfg["accum"], cfg["bf16"]
    print(f"[pragnosia] {p/1e6:.0f}M params | GPU {cfg['gpu']} {cfg['vram']}GB | "
          f"bs={bs} accum={accum} (eff {bs*accum}) bf16={bf16} compile={cfg['compile']} | "
          f"train_toks={td.size(0):,}", flush=True)
    opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=0.05, betas=(0.9, 0.95))
    warm = 2000
    def lr_at(it):
        if it < warm: return it / warm
        pr = (it - warm) / max(1, steps - warm); return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * pr))
    actx = torch.autocast("cuda", dtype=torch.bfloat16) if bf16 else torch.autocast("cuda", enabled=False)
    from tqdm import tqdm
    best, t0, run_loss = 1e9, time.time(), None
    no_improve, grow_patience, max_layers = 0, 5, len(m.blocks) * 2   # grow-as-you-train
    pbar = tqdm(range(1, steps + 1), desc="pragnosia", dynamic_ncols=True, mininterval=4,
                file=sys.stdout, smoothing=.05)
    for it in pbar:
        clr = lr * lr_at(it)
        for g in opt.param_groups: g["lr"] = clr
        opt.zero_grad()
        for _ in range(accum):                       # gradient accumulation
            x, y = H.batch(td, bs)
            with actx:
                loss = F.cross_entropy(fwd(x).reshape(-1, VOC), y.reshape(-1)) / accum
            loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
        l = loss.item() * accum
        run_loss = l if run_loss is None else 0.92 * run_loss + 0.08 * l
        toks = it * bs * accum * CFG["ctx"]; tps = toks / (time.time() - t0)
        # live progress bar (updates every step; throttled write to log)
        pbar.set_postfix_str(f"loss={run_loss:.3f} lr={clr:.1e} ppl*={best if best<1e8 else 0:.1f} "
                             f"{tps/1e3:.0f}Ktok/s epoch={toks/td.size(0):.2f}")
        if it % 100 == 0:                            # richer periodic log line
            pbar.write(f"  [step {it:6d}/{steps}] loss={run_loss:.3f} lr={clr:.2e} "
                       f"eff_batch={bs*accum} {tps/1e3:.0f}K tok/s  epoch {toks/td.size(0):.2f}  "
                       f"elapsed {(time.time()-t0)/60:.0f}m")
        if it % 500 == 0 or it == steps:            # validation + checkpoint
            m.eval(); tot = n = 0
            with torch.no_grad(), actx:
                for _ in range(30):
                    a, b = H.batch(vd, max(4, bs // 2))
                    tot += F.cross_entropy(m(a).reshape(-1, VOC), b.reshape(-1)).item() * b.numel(); n += b.numel()
            ppl = math.exp(tot / n); m.train()
            star = "  *** new best, checkpoint saved" if ppl < best else ""
            pbar.write(f"  >> it={it:6d}  VAL_PPL={ppl:.2f}  (best {min(best,ppl):.2f})  "
                       f"loss={run_loss:.3f}  ({(time.time()-t0)/3600:.2f}h){star}")
            if ppl < best:
                best = ppl; no_improve = 0; torch.save(m.state_dict(), CFG["ckpt"])
            else:
                no_improve += 1
            # GROW-AS-YOU-TRAIN: plateau = the model has extracted what it can at
            # this size. If there's VRAM headroom, add a layer (function-preserving,
            # no loss in quality at the moment of growth) and keep training. The
            # free-VRAM guard makes this safe -- it never risks the running process.
            if grow_enabled and no_improve >= grow_patience and len(m.blocks) < max_layers:
                free = (torch.cuda.mem_get_info()[0] / 2**30) if DEVICE == "cuda" else 99
                if free <= 4.0:
                    grow_enabled = False
                    pbar.write(f"  ## saturated, but only {free:.1f}GB free — growth needs a bigger "
                               f"GPU; continuing at {len(m.blocks)} layers")
                else:
                    import grow as G
                    cand = G.grow_depth(m, 1).to(DEVICE)
                    nb = fit_batch(cand, td, cfg["bs"], cfg["bf16"])
                    if nb >= 2:                                  # the grown model fits -> adopt it
                        m = cand; CFG["layers"] = len(m.blocks)
                        json.dump(CFG, open("pragnosia.json", "w"), indent=2)   # keep config in sync for --resume
                        bs, accum = nb, max(1, 64 // nb)
                        opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=0.05, betas=(0.9, 0.95))
                        fwd = torch.compile(m, dynamic=False) if cfg["compile"] else m
                        torch.save(m.state_dict(), CFG["ckpt"]); no_improve = 0; best = ppl
                        pbar.write(f"  ## GREW: +1 layer -> {len(m.blocks)} layers, "
                                   f"{G.n_params(m)/1e6:.0f}M params (saturation detected), bs={bs}")
                    else:
                        del cand; torch.cuda.empty_cache(); grow_enabled = False
                        pbar.write("  ## saturated, but the grown model won't fit — staying at current size")
    pbar.close()
    print(f"[pragnosia] DONE best val ppl={best:.2f} saved {CFG['ckpt']}", flush=True)

if __name__ == "__main__":
    pa = argparse.ArgumentParser()
    pa.add_argument("--steps", type=int, default=150000)
    pa.add_argument("--lr", type=float, default=6e-4)
    pa.add_argument("--bs", type=int, default=0)      # 0 = auto-tune from GPU
    pa.add_argument("--resume", action="store_true")
    pa.add_argument("--no-grow", action="store_true", help="disable grow-as-you-train")
    a = pa.parse_args()
    main(a.steps, a.lr, a.resume, a.bs, grow_enabled=not a.no_grow)
