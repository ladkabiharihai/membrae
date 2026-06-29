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
CFG = json.load(open(os.environ.get("CONFIG", "pragnosia.json")))   # CONFIG=pragnosia_baseline.json for the carrier=none baseline
H.VOC, H.L = CFG["vocab"], CFG["ctx"]
VOC = CFG["vocab"]
MEM_STOP_GB = 2.0      # background safety floor: if free GPU VRAM drops to/below this,
                       # save the checkpoint and stop cleanly (protects the co-resident
                       # prod services from OOM). Resume later with --resume.

def autotune():
    """Pick batch / grad-accum / precision / compile from the actual GPU."""
    if DEVICE != "cuda":
        return dict(bs=4, accum=4, bf16=False, compile=False, gpu="cpu", vram=0)
    p = torch.cuda.get_device_properties(0); total = p.total_memory / 2**30
    cap = float(os.environ.get("VRAM_CAP", "0"))      # shared GPU: VRAM_CAP=16 -> size to (and hard-cap at) the
    vram = min(total, cap) if cap > 0 else total      # FREE memory, not the card's total (else autotune OOMs prod)
    if cap > 0: torch.cuda.set_per_process_memory_fraction(min(1.0, cap / total))   # never touch co-resident jobs
    bf16 = torch.cuda.is_bf16_supported()
    # micro-batch that fits, scaled to VRAM (measured: 176M ~ bs8 @ 8GB)
    bs = max(4, int(vram // 1.0))                 # ~1 GB per micro-batch unit
    bs = min(bs, 256)
    target_eff = 64                                # keep a sane effective batch
    accum = max(1, target_eff // bs)
    # FASTEST per GPU class. bf16 everywhere (tensor cores; the old bf16+compile+tied-weights
    # bug is gone on torch 2.8, verified) + compile on the small GPU to fuse the spin scan.
    small = vram <= 12
    # compile fuses the spin scan, but inductor is unreliable on small consumer GPUs (e.g. the
    # 4060: "not enough SMs" -> intermittent device-side asserts in long runs). NOCOMPILE=1 to disable.
    use_compile = small and os.environ.get("NOCOMPILE", "0") != "1"
    use_bf16 = bf16
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
    m = H.SpinAttentionLM(VOC, CFG["d"], CFG["heads"], CFG["layers"], mlp_mult=CFG.get("mlp_mult", 4),
                          carrier=CFG.get("carrier", "single"))                                    # build on CPU
    p = sum(x.numel() for x in m.parameters())
    if resume and os.path.exists(CFG["ckpt"]):                          # load weights on CPU (no 2x GPU spike)
        m.load_state_dict(torch.load(CFG["ckpt"], map_location="cpu", weights_only=True)); print("resumed", flush=True)
    # LOW-MEMORY full-param training -- a BIG model on a SMALL GPU (e.g. 1.4B on 8GB):
    #   bf16 weights (halve) + gradient checkpointing (recompute acts in backward) keep the
    #   GPU to ~weights+grads+tiny-acts (~6-7GB); PagedAdamW8bit holds the optimizer state in
    #   8-bit and AUTO-PAGES it to CPU RAM, so the 11.5GB AdamW state never sits on the GPU.
    #   Same params, no LoRA -- just memory-relocated. Auto-on for big-model/small-GPU; LAPTOP=1 forces.
    lowmem = os.environ.get("LAPTOP", "0") == "1" or (DEVICE == "cuda" and p > 7e8 and cfg["vram"] <= 16)
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
    # RESUME CONTINUES, not restarts: restore the step counter + adaptive-lr state from a sidecar
    # so --resume picks up where it left (no re-warmup from 0, no lr_scale reset). Weights already
    # loaded above; best is re-validated above so it's never lost. Data is sampled at RANDOM offsets
    # across the whole corpus (see H.batch), so any stop point has already seen all data types.
    state_path = CFG["ckpt"].rsplit(".", 1)[0] + "_state.json"
    start_it, step_base = 1, 0
    if resume and os.path.exists(state_path):
        try:
            st = json.load(open(state_path)); start_it = int(st.get("step", 0)) + 1; step_base = start_it - 1
            lr_scale = float(st.get("lr_scale", lr_scale)); lr_wait = int(st.get("lr_wait", lr_wait))
            if "lr" in st and lr <= 0: lr = float(st["lr"])    # restore the DERIVED peak lr (find_lr is unreliable
            print(f"resumed STATE: continuing at step {start_it} (lr_scale {lr_scale:.3f}, "  # on trained weights -> don't re-run it)
                  f"peak_lr {lr:.2e}) -- not restarting from 0", flush=True)
        except Exception as e: print("state restore skipped:", str(e)[:50], flush=True)
    t0, run_loss = time.time(), None
    no_improve, grow_patience, grow_count = 0, int(os.environ.get("GROW_PATIENCE", "5")), 0   # grow-as-you-train
    val_every = int(os.environ.get("VAL_EVERY", "500"))              # validation/checkpoint cadence (tunable)
    best_ckpt = os.environ.get("BEST_CKPT", "pragnosia_best.pt")     # save-on-best path (env so tests don't clobber)
    # NO hardcoded size caps. The DATA sets the ceiling: Chinchilla ~20 tokens/param is
    # compute-optimal, so the model may grow only while it's below tokens/20 -- past that, extra
    # params simply can't be trained to maturity on this corpus. VRAM is the other (physical) cap.
    budget_params = td.size(0) / 20.0
    # TOK_PER_PARAM: train each grown size to >= this many tokens/param BEFORE it may grow again.
    # 20 = Chinchilla (compute-optimal); raise it to OVERTRAIN each rung (e.g. 120 = 6x). With no
    # fixed size cap, the model then grows until VRAM is near-full and STOPS there (memory is the cap).
    TPP = float(os.environ.get("TOK_PER_PARAM", "20"))
    PROBE_STEPS = int(os.environ.get("PROBE_STEPS", "60"))           # steps to test whether capacity truly helps
    val_hist = []                                                    # recent val ppls -> derive the "is it noise?" bar
    print(f"[growth] {td.size(0)/1e6:.0f}M tokens | TOK_PER_PARAM={TPP:.0f} (train each rung this thoroughly "
          f"before growing) | NO fixed size cap -> grow until VRAM near-full then stop. "
          f"Grow only on PROBE-CONFIRMED saturation.", flush=True)
    # DERIVE the peak LR from the model's OWN loss-vs-lr curve (Smith range test) instead of a
    # hardcoded seed: sweep lr exponentially over a short probe, pick the steepest-descent point.
    # lo/hi/n are just the search grid (structural); the chosen lr is the model+data's answer.
    def find_lr(lo=1e-6, hi=1.0, n=80):
        import copy
        st = copy.deepcopy(m.state_dict()); o = _mkopt(m.parameters(), lo)
        mult = (hi / lo) ** (1.0 / n); cur = lo; losses = []; lrs = []
        m.train()
        for _ in range(n):
            for g in o.param_groups: g["lr"] = cur
            o.zero_grad(); x, y = get_batch()
            with actx: ls = F.cross_entropy(fwd(x).reshape(-1, VOC), y.reshape(-1))
            ls.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); o.step()
            losses.append(ls.item()); lrs.append(cur); cur *= mult
            if not math.isfinite(losses[-1]) or losses[-1] > 4 * losses[0]: break   # diverged -> stop
        m.load_state_dict(st)                                          # restore: the sweep was only a probe
        if len(losses) < 8: return lrs[len(losses) // 2]
        sm = np.convolve(np.array(losses), np.ones(5) / 5, mode="valid")
        return float(lrs[int(np.argmin(np.diff(sm)))]) / 10.0          # steepest-descent / 10: the steepest point is
                                                                       # the INSTABILITY edge (training there degrades --
                                                                       # observed ppl 225->310 at raw 0.075); Smith's
                                                                       # margin puts the peak safely below it.
    if lr <= 0:
        lr = find_lr()
        print(f"[lr] DERIVED peak lr = {lr:.2e} (steepest descent of the model's own lr sweep -- not a hardcoded seed)", flush=True)
    pbar = tqdm(range(start_it, steps + 1), desc="pragnosia", dynamic_ncols=True, mininterval=4,
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
        toks = it * bs * accum * CFG["ctx"]                                  # cumulative (for epoch)
        tps = (it - step_base) * bs * accum * CFG["ctx"] / (time.time() - t0)  # rate THIS run (resume-correct)
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
        if it % val_every == 0 or it == steps:            # validation + checkpoint
            m.eval(); tot = n = 0
            with torch.no_grad(), actx:
                for _ in range(30):
                    a, b = H.batch(vd, max(4, bs // 2))
                    tot += F.cross_entropy(m(a).reshape(-1, VOC), b.reshape(-1)).item() * b.numel(); n += b.numel()
            ppl = math.exp(tot / n); m.train(); val_hist.append(ppl)
            star = "  *** new best, checkpoint saved" if ppl < best else ""
            pbar.write(f"  >> it={it:6d}  VAL_PPL={ppl:.2f}  (best {min(best,ppl):.2f})  "
                       f"loss={run_loss:.3f}  ({(time.time()-t0)/3600:.2f}h){star}")
            torch.save(m.state_dict(), CFG["ckpt"])      # save LATEST every val (resumable)
            json.dump({"step": it, "lr_scale": lr_scale, "lr_wait": lr_wait, "best": best, "lr": lr},
                      open(state_path, "w"))             # sidecar -> resume CONTINUES from here, not 0
            if ppl < best:                                # ALSO protect the BEST: refinement runs can
                best = ppl; no_improve = 0; lr_wait = 0   # drift worse at too-high lr, and latest-only
                torch.save(m.state_dict(), best_ckpt)   # would overwrite the good weights.
            else:                                         # pragnosia_best.pt = lowest-ppl checkpoint.
                no_improve += 1; lr_wait += 1
                # ADAPTIVE lr from the val signal. "Degrading" is measured against the val NOISE
                # itself (the same std/mean bar the growth probe uses), not a hardcoded 2%: a real
                # degradation is one that clears 2 sigma of the recent val wobble.
                vnoise = float(np.std(val_hist[-4:]) / max(1e-6, np.mean(val_hist[-4:]))) if len(val_hist) >= 3 else 0.02
                if it > warm and ppl > best * (1 + 2 * vnoise) and lr_scale > 0.02:   # degraded past the noise -> cut
                    lr_scale *= 0.5; lr_wait = 0
                    pbar.write(f"  ~~ lr auto-CUT (val {ppl:.2f} > best +{2*vnoise*100:.1f}% noise) -> lr {lr*lr_scale:.2e} (scale {lr_scale:.3f})")
                elif lr_wait >= lr_patience and lr_scale > 0.02:               # plateaued -> ease down
                    lr_scale *= 0.7; lr_wait = 0
                    pbar.write(f"  ~~ lr auto-EASED (plateau) -> lr {lr*lr_scale:.2e} (scale {lr_scale:.3f})")
            # SELF-GOVERNING GROWTH (no hardcoded caps). A plateau is only a HINT, not a
            # licence to grow -- a high-loss plateau is just stuck on lr/data, NOT saturation
            # (that exact confusion is how an undertrained model ballooned to 1.4B). So on a
            # plateau we PROBE: grow a copy by depth AND by width, train each briefly, and grow
            # only if the extra capacity DEMONSTRABLY lowers val loss past the val noise. The
            # shape (depth vs width) is whichever probe helps more. Caps are the DATA budget
            # (Chinchilla tokens/20) and VRAM -- both derived, none hardcoded.
            if grow_enabled and no_improve >= grow_patience and toks < TPP * H.n_params(m):
                pbar.write(f"  .. {H.n_params(m)/1e6:.0f}M plateaued at {toks/H.n_params(m):.0f} tok/param "
                           f"(< {TPP:.0f} target) -- training this size more before considering growth")
            elif grow_enabled and no_improve >= grow_patience:       # TPP-trained AND plateaued -> consider growth
                import grow as G
                free = (torch.cuda.mem_get_info()[0] / 2**30) if DEVICE == "cuda" else 99
                if G.n_params(m) >= budget_params:                    # DATA budget reached (derived)
                    grow_enabled = False
                    pbar.write(f"  ## DATA-BUDGET cap: {G.n_params(m)/1e6:.1f}M >= {budget_params/1e6:.1f}M "
                               f"(tokens/20) -- a bigger model can't be trained to maturity on this corpus. Staying.")
                elif free <= 4.0:                                     # physical VRAM cap
                    grow_enabled = False
                    pbar.write(f"  ## VRAM cap ({free:.1f}GB free) -- staying at {G.n_params(m)/1e6:.0f}M")
                else:
                    # SATURATED: TPP-trained AND plateaued, with VRAM/data headroom. A CONVERGED size
                    # cannot be beaten by a short probe -- the benefit of extra capacity needs THOUSANDS of
                    # steps, not 60 -- so saturation ITSELF is the grow signal (the old "must beat X% in 60
                    # steps" probe could never confirm, and got stuck). A short GENTLE probe now only PICKS
                    # the shape (depth vs width); we adopt the winner and the LR-RESET trains it up. Over-
                    # growth is held by the TPP floor + lr reset (each rung really trains) + data/VRAM caps
                    # + save-on-best (the 1.4B ballooned only because the lr DIDN'T reset, so grown layers
                    # never trained and ppl drifted up -- fixed here).
                    def _try(gfn):
                        # build+train a CANDIDATE beside the live model -- needs ~2x memory transiently. fit_batch
                        # can't see the candidate's optimizer states, so it can under-estimate and OOM mid-probe.
                        # Catch that: an OOM just means "this shape doesn't fit beside the live model+prod" -> decline
                        # gracefully (return None) so the caller treats it as the VRAM cap instead of CRASHING the run.
                        c = po = None
                        try:
                            c = gfn(m, 1).to(DEVICE); c.train()
                            nb = fit_batch(c, td, cfg["bs"], cfg["bf16"])
                            if nb < 2: return None, 1e9, 0
                            po = _mkopt(c.parameters(), lr * 0.1)      # GENTLE: compares shapes, doesn't destabilize
                            for _ in range(PROBE_STEPS):
                                po.zero_grad(); xa, ya = H.batch(td, nb)
                                with actx: pl = F.cross_entropy(c(xa).reshape(-1, VOC), ya.reshape(-1))
                                pl.backward(); torch.nn.utils.clip_grad_norm_(c.parameters(), 1.0); po.step()
                            return c, H.val_ppl(c, vd, iters=15), nb
                        except torch.cuda.OutOfMemoryError:
                            c = None; return None, 1e9, 0
                        finally:
                            po = None; torch.cuda.empty_cache()
                    pbar.write(f"  .. saturated at {G.n_params(m)/1e6:.0f}M -> GROWING (probe picks depth vs width)")
                    opts_ = [(md, *_try(gfn)) for gfn, md in ((G.grow_depth, "depth"), (G.grow_width, "width"))]
                    opts_ = [(md, c, p, nb) for md, c, p, nb in opts_ if c is not None]
                    if not opts_:
                        grow_enabled = False
                        pbar.write(f"  ## VRAM cap -- neither shape fits; staying at {G.n_params(m)/1e6:.0f}M")
                    else:
                        best_mode, best_c, best_p, best_nb = min(opts_, key=lambda t: t[2])
                        for md, c, p, nb in opts_:
                            if c is not best_c: del c
                        torch.cuda.empty_cache()
                        m = best_c; CFG["layers"] = len(m.blocks); CFG["mlp_mult"] = m.mlp_mult
                        json.dump(CFG, open("pragnosia.json", "w"), indent=2)
                        bs, accum = best_nb, max(1, 64 // best_nb)
                        opt = _mkopt(m.parameters(), lr)
                        if id_seqs: id_opt = torch.optim.AdamW(m.parameters(), lr=id_lr, betas=(0.9, 0.95))
                        if pf: pf._stop = True; pf = H.Prefetcher(td, bs, depth=4)
                        def get_batch(): return pf.next() if pf else H.batch(td, bs)
                        fwd = torch.compile(m, dynamic=False) if cfg["compile"] else m
                        best = best_p; torch.save(m.state_dict(), CFG["ckpt"]); no_improve = 0; grow_count += 1
                        # reset adaptive lr to a gentle ABSOLUTE level (~3e-4), not a fraction of peak:
                        # the function-preserving grow starts AT the converged model, so a big jump (0.3*peak
                        # = 2.3e-3 when the derived peak is a hot 7.5e-3) spikes ppl 28->52 and wastes ~10 min
                        # re-annealing. Targeting an absolute ~3e-4 trains the new capacity without the spike,
                        # and is peak-independent (capped at 0.3*peak so a low-peak run still gets a real lr).
                        reset_lr = min(0.3, 3e-4 / lr)
                        lr_scale, lr_wait = reset_lr, 0
                        pbar.write(f"  ## GREW ({best_mode}) on saturation -> {len(m.blocks)}L mlp{m.mlp_mult} "
                                   f"{G.n_params(m)/1e6:.0f}M, bs={bs} (lr -> {lr*reset_lr:.1e}, gently re-anneals)")
    pbar.close()
    print(f"[pragnosia] DONE best val ppl={best:.2f} saved {CFG['ckpt']}", flush=True)

if __name__ == "__main__":
    pa = argparse.ArgumentParser()
    pa.add_argument("--steps", type=int, default=150000)
    pa.add_argument("--lr", type=float, default=0.0, help="0 = DERIVE the peak lr (Smith range test); >0 overrides")
    pa.add_argument("--bs", type=int, default=0)      # 0 = auto-tune from GPU
    pa.add_argument("--resume", action="store_true")
    pa.add_argument("--no-grow", action="store_true", help="disable grow-as-you-train")
    a = pa.parse_args()
    main(a.steps, a.lr, a.resume, a.bs, grow_enabled=not a.no_grow)
