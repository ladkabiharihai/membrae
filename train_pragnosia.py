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
MEM_STOP_GB = 2.0      # background safety floor: if free GPU VRAM drops to/below this,
                       # save the checkpoint and stop cleanly (protects the co-resident
                       # prod services from OOM). Resume later with --resume.

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

# ---- optional identity injection: feed Pragnosia's name-binding sentences gently every
# few steps, interleaved with the main corpus, so a stable self-concept settles into the
# weights without dominating (mirrors brain.py's gentle, many-phrasing teach). Env-gated:
#   PRAGNOSIA_IDENTITY=identity_sentences.txt  IDENTITY_EVERY=50  IDENTITY_LR=2e-5
def _load_identity(path):
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(CFG["tokenizer"])
    seqs = []
    for line in open(path):
        s = line.strip()
        if not s or s.startswith("#"): continue
        ids = tok.encode(s).ids[:CFG["ctx"]]
        if len(ids) >= 2: seqs.append(ids)
    print(f"[identity] loaded {len(seqs)} sentences from {path}", flush=True)
    return seqs

def _identity_batch(seqs, bs):
    import random
    pick = random.sample(seqs, min(bs, len(seqs)))
    L = max(len(s) for s in pick)
    x = torch.zeros(len(pick), L - 1, dtype=torch.long, device=DEVICE)
    y = torch.full((len(pick), L - 1), -100, dtype=torch.long, device=DEVICE)   # -100 = ignore pad
    for i, s in enumerate(pick):
        xi, yi = s[:-1], s[1:]
        x[i, :len(xi)] = torch.tensor(xi, device=DEVICE)
        y[i, :len(yi)] = torch.tensor(yi, device=DEVICE)
    return x, y

def main(steps, lr, resume, override_bs, grow_enabled):
    torch.manual_seed(0); np.random.seed(0)
    cfg = autotune()
    td, vd = H.load(CFG["train_bin"]), H.load(CFG["valid_bin"])
    m = H.SpinAttentionLM(VOC, CFG["d"], CFG["heads"], CFG["layers"], mlp_mult=CFG.get("mlp_mult", 4))  # build on CPU
    p = sum(x.numel() for x in m.parameters())
    if resume and os.path.exists(CFG["ckpt"]):                          # load weights on CPU (no 2x GPU spike)
        m.load_state_dict(torch.load(CFG["ckpt"], map_location="cpu", weights_only=True)); print("resumed", flush=True)
    # LOW-MEMORY full-param training -- a BIG model on a SMALL GPU (e.g. 1.4B on 8GB):
    #   bf16 weights (halve) + gradient checkpointing (recompute acts in backward) keep the
    #   GPU to ~weights+grads+tiny-acts (~6-7GB); PagedAdamW8bit holds the optimizer state in
    #   8-bit and AUTO-PAGES it to CPU RAM, so the 11.5GB AdamW state never sits on the GPU.
    #   Same params, no LoRA -- just memory-relocated. Auto-on for big-model/small-GPU; LAPTOP=1 forces.
    lowmem = os.environ.get("LAPTOP", "0") == "1" or (DEVICE == "cuda" and p > 7e8 and cfg["vram"] < 16)
    bnb = None
    if lowmem:
        import bitsandbytes as bnb
        m = m.bfloat16(); m.grad_checkpoint = True
        cfg.update(bs=1, accum=64, bf16=False, compile=False)     # bf16 weights -> no autocast; ckpt+compile clash
        grow_enabled = False                                      # growth re-allocs the optimizer -> keep it off here
        print(f"[LOWMEM] full {p/1e6:.0f}M on {cfg['vram']}GB: bf16 weights + grad-checkpoint + PagedAdamW8bit "
              f"(optimizer state pages to CPU)", flush=True)
    m = m.to(DEVICE)                                                    # move ONCE (bf16 if lowmem -> ~half)
    # adapt batch to the ACTUAL model+GPU (robust to any size), then keep eff batch ~64
    if override_bs:
        cfg["bs"] = override_bs
    elif not lowmem:
        cfg["bs"] = fit_batch(m, td, cfg["bs"], cfg["bf16"])
    cfg["accum"] = max(1, 64 // cfg["bs"])
    fwd = torch.compile(m, dynamic=False) if cfg["compile"] else m
    bs, accum, bf16 = cfg["bs"], cfg["accum"], cfg["bf16"]
    print(f"[pragnosia] {p/1e6:.0f}M params | GPU {cfg['gpu']} {cfg['vram']}GB | "
          f"bs={bs} accum={accum} (eff {bs*accum}) bf16={bf16} compile={cfg['compile']} | "
          f"train_toks={td.size(0):,}", flush=True)
    def _mkopt(params, lr, wd=0.05):                                   # fused AdamW kernel on GPU (~5% faster
        if lowmem: return bnb.optim.PagedAdamW8bit(params, lr=lr, weight_decay=wd, betas=(0.9, 0.95))
        return torch.optim.AdamW(params, lr=lr, weight_decay=wd, betas=(0.9, 0.95), fused=(DEVICE == "cuda"))
    opt = _mkopt(m.parameters(), lr)
    # identity injection (optional, env-gated)
    id_path = os.environ.get("PRAGNOSIA_IDENTITY")
    id_seqs = _load_identity(id_path) if id_path and os.path.exists(id_path) else None
    id_every = int(os.environ.get("IDENTITY_EVERY", "50"))
    id_lr = float(os.environ.get("IDENTITY_LR", "1e-4"))                      # brain.py's proven gentle teach lr
    id_opt = torch.optim.AdamW(m.parameters(), lr=id_lr, betas=(0.9, 0.95)) if (id_seqs and not lowmem) else None
    # background-thread batch prefetcher (overlaps data prep with compute). PREFETCH=0 to disable.
    use_prefetch = os.environ.get("PREFETCH", "1") != "0"
    pf = H.Prefetcher(td, bs, depth=4) if use_prefetch else None
    def get_batch():
        return pf.next() if pf else H.batch(td, bs)
    warm = 2000
    # ADAPTIVE lr (no fixed schedule): warm up, then let the VAL signal drive it — halve on
    # degradation, ease on plateau (mirrors grow-on-saturation). lr_scale self-tunes; --lr is
    # just the initial peak. So a too-high start auto-corrects instead of silently degrading.
    lr_scale, lr_wait, lr_patience = 1.0, 0, 3
    def lr_at(it):
        return min(1.0, it / warm) * lr_scale
    actx = torch.autocast("cuda", dtype=torch.bfloat16) if bf16 else torch.autocast("cuda", enabled=False)
    from tqdm import tqdm
    # init `best` from the RESUMED model's own val ppl, so an early step can never overwrite a
    # known-good checkpoint -- save-on-best only fires on a real improvement.
    best = H.val_ppl(m, vd, iters=15) if (resume and os.path.exists(CFG["ckpt"])) else 1e9
    if best < 1e9: print(f"resumed model val_ppl={best:.2f} -- will only save if training beats it", flush=True)
    t0, run_loss = time.time(), None
    no_improve, grow_patience, grow_count = 0, 5, 0                   # grow-as-you-train
    max_layers, max_mlp_mult = 48, 4                                 # FLOP-efficient shape: grow by DEPTH,
                                                                     # keep the MLP lean (mlp_mult<=4). A wide
                                                                     # mlp_mult=12 just inflates params (=FLOPs)
                                                                     # for little gain; depth at mlp4 is the
                                                                     # better capability-per-FLOP trade.
    pbar = tqdm(range(1, steps + 1), desc="pragnosia", dynamic_ncols=True, mininterval=4,
                file=sys.stdout, smoothing=.05)
    for it in pbar:
        clr = lr * lr_at(it)
        for g in opt.param_groups: g["lr"] = clr
        opt.zero_grad()
        for _ in range(accum):                       # gradient accumulation
            x, y = get_batch()
            with actx:
                loss = F.cross_entropy(fwd(x).reshape(-1, VOC), y.reshape(-1)) / accum
            loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
        l = loss.item() * accum
        run_loss = l if run_loss is None else 0.92 * run_loss + 0.08 * l
        if id_seqs and it % id_every == 0:           # gentle identity nudge, interleaved
            id_opt.zero_grad()
            xi, yi = _identity_batch(id_seqs, min(16, bs))
            with actx:
                il = F.cross_entropy(fwd(xi).reshape(-1, VOC), yi.reshape(-1), ignore_index=-100)
            il.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); id_opt.step()
        toks = it * bs * accum * CFG["ctx"]; tps = toks / (time.time() - t0)
        # live progress bar (updates every step; throttled write to log)
        pbar.set_postfix_str(f"loss={run_loss:.3f} lr={clr:.1e} ppl*={best if best<1e8 else 0:.1f} "
                             f"{tps/1e3:.0f}Ktok/s epoch={toks/td.size(0):.2f}")
        if it % 100 == 0:                            # richer periodic log line
            pbar.write(f"  [step {it:6d}/{steps}] loss={run_loss:.3f} lr={clr:.2e} "
                       f"eff_batch={bs*accum} {tps/1e3:.0f}K tok/s  epoch {toks/td.size(0):.2f}  "
                       f"elapsed {(time.time()-t0)/60:.0f}m")
            if DEVICE == "cuda":                     # SELF-STOP safety floor (background-safe)
                free_gb = torch.cuda.mem_get_info()[0] / 2**30
                if free_gb <= MEM_STOP_GB:
                    torch.save(m.state_dict(), CFG["ckpt"])
                    pbar.write(f"  !! free GPU VRAM {free_gb:.2f}GB <= {MEM_STOP_GB}GB floor — "
                               f"checkpoint saved, stopping cleanly (resume with --resume)")
                    break
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
            torch.save(m.state_dict(), CFG["ckpt"])      # save LATEST every val (resumable)
            if ppl < best:                                # ALSO protect the BEST: refinement runs can
                best = ppl; no_improve = 0; lr_wait = 0   # drift worse at too-high lr, and latest-only
                torch.save(m.state_dict(), "pragnosia_best.pt")   # would overwrite the good weights.
            else:                                         # pragnosia_best.pt = lowest-ppl checkpoint.
                no_improve += 1; lr_wait += 1
                # ADAPTIVE lr: react to the val signal instead of a hardcoded value
                if it > warm and ppl > best * 1.02 and lr_scale > 0.02:        # clearly degrading -> halve
                    lr_scale *= 0.5; lr_wait = 0
                    pbar.write(f"  ~~ lr auto-CUT (val {ppl:.2f} > best {best:.2f}+2%) -> lr {lr*lr_scale:.2e} (scale {lr_scale:.3f})")
                elif lr_wait >= lr_patience and lr_scale > 0.02:               # plateaued -> ease down
                    lr_scale *= 0.7; lr_wait = 0
                    pbar.write(f"  ~~ lr auto-EASED (plateau) -> lr {lr*lr_scale:.2e} (scale {lr_scale:.3f})")
            # GROW-AS-YOU-TRAIN: plateau = the model has extracted what it can at
            # this size. With VRAM headroom, grow -- ALTERNATING depth (add a layer)
            # and width (widen every MLP), function-preserving (no quality loss at
            # growth), and keep training. So from a small start it auto-scales into a
            # balanced larger brain, sized by the data and the GPU. The free-VRAM
            # guard + fit check make it safe -- it never risks the running process,
            # and it stops when the card is full (VRAM is the real cap).
            if grow_enabled and no_improve >= grow_patience:
                import grow as G
                free = (torch.cuda.mem_get_info()[0] / 2**30) if DEVICE == "cuda" else 99
                can_depth = len(m.blocks) < max_layers; can_width = m.mlp_mult < max_mlp_mult
                want_width = can_width and not can_depth               # DEPTH-FIRST; widen only if depth is capped
                mode = ("width" if (want_width and can_width) or not can_depth else "depth") if (can_depth or can_width) else None
                if free <= 4.0 or mode is None:
                    grow_enabled = False
                    why = f"only {free:.1f}GB free" if free <= 4.0 else "at growth cap"
                    pbar.write(f"  ## saturated, {why} — staying at {len(m.blocks)}L mlp_mult={m.mlp_mult} "
                               f"({G.n_params(m)/1e6:.0f}M)")
                else:
                    cand = (G.grow_depth(m, 1) if mode == "depth" else G.grow_width(m, 1)).to(DEVICE)
                    nb = fit_batch(cand, td, cfg["bs"], cfg["bf16"])
                    if nb >= 2:                                        # the grown model fits -> adopt it
                        m = cand; CFG["layers"] = len(m.blocks); CFG["mlp_mult"] = m.mlp_mult
                        json.dump(CFG, open("pragnosia.json", "w"), indent=2)   # in sync for --resume
                        bs, accum = nb, max(1, 64 // nb)
                        opt = _mkopt(m.parameters(), lr)
                        if id_seqs: id_opt = torch.optim.AdamW(m.parameters(), lr=id_lr, betas=(0.9, 0.95))  # retarget grown params
                        if pf: pf._stop = True; pf = H.Prefetcher(td, bs, depth=4)   # rebuild for new bs
                        def get_batch(): return pf.next() if pf else H.batch(td, bs)
                        fwd = torch.compile(m, dynamic=False) if cfg["compile"] else m
                        torch.save(m.state_dict(), CFG["ckpt"]); no_improve = 0; best = ppl; grow_count += 1
                        pbar.write(f"  ## GREW ({mode}) -> {len(m.blocks)}L mlp_mult={m.mlp_mult}  "
                                   f"{G.n_params(m)/1e6:.0f}M params (saturation), bs={bs}")
                    else:
                        del cand; torch.cuda.empty_cache(); grow_enabled = False
                        pbar.write(f"  ## saturated, {mode} growth won't fit — staying at current size")
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
