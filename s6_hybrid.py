"""
================================================================================
The Pragnosia language organ -- attention blocks + a PARALLEL spin carrier.
================================================================================
Causal self-attention enriches each token's input (the quality lever); one spin
carrier threads ROTATIONAL cross-token state through the stream (so the brain-swap /
state-causality invariant survives).

The carrier is a DIAGONAL-COMPLEX (LRU/S5-style) recurrence -- each channel is a damped
complex oscillator at its own frequency, "the answer lives in the phase", per channel:
             h_t = lambda (.) h_{t-1} + b_t ,   lambda_j = exp(-exp(nu_j)) * exp(i*phi_j)
Because lambda is diagonal the recurrence is matmul-free (O(T*d)) AND associative, so it
runs as a log-depth PARALLEL scan -- ~2x faster to train than the original dense-tanh
sequential SpinStep recurrence, with the SAME proven dynamics (verified: 228M model
reaches the same ppl, and the brain-swap control still gives own < swapped < random).
The old dense-tanh carrier (W = -rho QQ^T + skew, sequential tanh loop) lives in git
history and the paper as the anchor; this file IS the production model.

MANDATORY GATE: swapping the carried spin state mid-sequence must still degrade the
continuation (own < swapped < random). Pass quality but FAIL this = regression; say so.
================================================================================
"""
import argparse, json, math, time, os
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
import torch.utils.checkpoint   # gradient checkpointing (low-memory training of big models)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cuda.matmul.allow_tf32 = True
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
VOC, L = 8192, 256

class CausalSelfAttention(nn.Module):
    def __init__(self, d, n_head):
        super().__init__()
        self.n_head, self.d = n_head, d
        self.qkv = nn.Linear(d, 3 * d)
        self.proj = nn.Linear(d, d)
    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(self.d, dim=2)
        hd = C // self.n_head
        q = q.view(B, T, self.n_head, hd).transpose(1, 2)
        k = k.view(B, T, self.n_head, hd).transpose(1, 2)
        v = v.view(B, T, self.n_head, hd).transpose(1, 2)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.proj(y.transpose(1, 2).contiguous().view(B, T, C))

class Block(nn.Module):
    """Standard transformer block (attention + MLP) -- the quality engine.
    mlp_mult lets the MLP hidden width grow (width growth survives --resume)."""
    def __init__(self, d, n_head, mlp_mult=4):
        super().__init__()
        self.ln1 = nn.LayerNorm(d); self.attn = CausalSelfAttention(d, n_head)
        self.ln2 = nn.LayerNorm(d)
        self.mlp = nn.Sequential(nn.Linear(d, mlp_mult * d), nn.GELU(), nn.Linear(mlp_mult * d, d))
    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

class SpinBlock(nn.Module):
    """SPIN-DOMINANT block: the spin carrier IS the token mixer (causal recurrence, in place of
    attention) + an MLP for channel mixing. This is the intended design -- spin is the core compute,
    attention is reserved for a few 'helper' layers. The carrier here mixes strongly (not a light
    side-channel), so cross-token information flows through the spin dynamics."""
    def __init__(self, d, mlp_mult=4):
        super().__init__()
        self.carrier = SpinCarrier(d)
        with torch.no_grad(): self.carrier.gate.fill_(2.0)        # strong mixer (sigmoid(2)=0.88), not 0.12 side-channel
        self.ln2 = nn.LayerNorm(d)
        self.mlp = nn.Sequential(nn.Linear(d, mlp_mult * d), nn.GELU(), nn.Linear(mlp_mult * d, d))
    def forward(self, x):
        x, _ = self.carrier(x)                                    # token mixing (carrier does its own ln + gated residual)
        x = x + self.mlp(self.ln2(x))                             # channel mixing
        return x
    def forward_state(self, x, h0=None):                          # stateful path: carry the carrier state across
        x, hs = self.carrier(x, h0)                               # forward calls -> O(T) cross-window long context
        x = x + self.mlp(self.ln2(x))
        return x, hs

def _lru_scan(lr, li, br, bi, h0r=None, h0i=None):
    """Hillis-Steele inclusive associative scan (log2 T steps, fully parallel) of the
    diagonal-complex recurrence  h_t = lam (.) h_{t-1} + b_t , in REAL arithmetic (re/im)
    so it torch.compiles. No division -> numerically stable for |lam| < 1."""
    B, T, D = br.shape
    Ar = lr.view(1, 1, D).expand(B, T, D).clone(); Ai = li.view(1, 1, D).expand(B, T, D).clone()
    Hr = br.clone(); Hi = bi.clone()
    if h0r is not None:                                    # fold initial state into position 0
        Hr[:, 0] = Hr[:, 0] + lr * h0r - li * h0i; Hi[:, 0] = Hi[:, 0] + lr * h0i + li * h0r
    d = 1
    while d < T:
        arr, ari = Ar[:, d:], Ai[:, d:]; alr, ali = Ar[:, :T - d], Ai[:, :T - d]
        hlr, hli = Hr[:, :T - d], Hi[:, :T - d]
        tHr = arr * hlr - ari * hli; tHi = arr * hli + ari * hlr                  # Ar*Hl
        Hr = torch.cat([Hr[:, :d], Hr[:, d:] + tHr], 1); Hi = torch.cat([Hi[:, :d], Hi[:, d:] + tHi], 1)
        nAr = arr * alr - ari * ali; nAi = arr * ali + ari * alr                  # Ar*Al
        Ar = torch.cat([Ar[:, :d], nAr], 1); Ai = torch.cat([Ai[:, :d], nAi], 1)
        d *= 2
    return Hr, Hi

class SpinCarrier(nn.Module):
    """PARALLEL spin: a diagonal-complex (LRU/S5-style) recurrence carrying cross-token
    state, added to the stream through a learned gate. Each channel is a damped complex
    oscillator at its OWN frequency -- "the answer lives in the phase", per channel. The
    diagonal lambda makes the recurrence matmul-free (O(T*d), not O(T*d^2)) AND an
    associative scan, so it runs as a log-depth PARALLEL scan instead of a T-step
    sequential loop: ~2x faster to train, and the carried state stays causal (verified
    own < swapped < random on the brain-swap test). Replaces the original dense-tanh
    sequential SpinStep recurrence (kept in git history / the paper as the anchor)."""
    def __init__(self, d):
        super().__init__()
        self.d = d
        self.ln = nn.LayerNorm(d)
        r = torch.rand(d)
        self.nu = nn.Parameter(torch.log(-torch.log(0.5 + 0.4 * r)))   # |lam| ~ U(0.5,0.9), stable
        self.theta = nn.Parameter(math.pi * torch.rand(d))            # per-channel frequency spread
        self.U_re = nn.Linear(d, d); self.U_im = nn.Linear(d, d)       # real input -> complex
        self.gain = nn.Linear(d, d)                                   # input-dependent input gate
        self.C = nn.Linear(2 * d, d)                                  # complex state -> real out
        self.gate = nn.Parameter(torch.tensor(-2.0))                  # sigmoid(-2)=0.12: light
    def _lam_re_im(self):
        mag = torch.exp(-torch.exp(self.nu)); ph = torch.exp(self.theta)
        return mag * torch.cos(ph), mag * torch.sin(ph)
    def forward(self, x, h0=None):
        yn = self.ln(x)
        g = torch.sigmoid(self.gain(yn))
        br = self.U_re(yn) * g; bi = self.U_im(yn) * g                # gated complex input (re,im)
        lr, li = self._lam_re_im()
        h0r = h0i = None
        if h0 is not None: h0r, h0i = h0[:, 0], h0[:, 1]             # state carried as [B,2,d]
        hr, hi = _lru_scan(lr.to(br.dtype), li.to(br.dtype), br, bi, h0r, h0i)
        out = self.C(torch.cat([hr, hi], -1))                        # phase + magnitude -> real
        return x + torch.sigmoid(self.gate) * out, torch.stack([hr[:, -1], hi[:, -1]], 1)

class RealCarrier(nn.Module):
    """Diagonal-REAL linear recurrence (no phase): h_t = lam (.) h_{t-1} + b_t, lam in (0,1). A DIFFERENT
    recurrence family from the complex spin carrier -- used to test whether PLACEMENT, not the complex
    phase, is what makes a recurrent mixer load-bearing (the generalization control). Same parallel scan
    with the imaginary parts zeroed; gated residual like the spin carrier."""
    def __init__(self, d):
        super().__init__()
        self.d = d
        self.ln = nn.LayerNorm(d)
        r = torch.rand(d)
        self.nu = nn.Parameter(torch.log(-torch.log(0.5 + 0.4 * r)))   # |lam| ~ U(0.5,0.9), stable real decay
        self.gain = nn.Linear(d, d)                                    # input-dependent input gate
        self.U = nn.Linear(d, d)                                       # real input
        self.C = nn.Linear(d, d)                                       # real state -> real out
        self.gate = nn.Parameter(torch.tensor(-2.0))
    def forward(self, x, h0=None):
        yn = self.ln(x)
        g = torch.sigmoid(self.gain(yn))
        br = self.U(yn) * g
        lr = torch.exp(-torch.exp(self.nu))                            # real lambda in (0,1)
        li = torch.zeros_like(lr); bi = torch.zeros_like(br)
        h0r = h0i = None
        if h0 is not None: h0r, h0i = h0[:, 0], h0[:, 1]
        hr, _ = _lru_scan(lr.to(br.dtype), li.to(br.dtype), br, bi, h0r, h0i)
        out = self.C(hr)
        return x + torch.sigmoid(self.gate) * out, torch.stack([hr[:, -1], torch.zeros_like(hr[:, -1])], 1)

class RealBlock(nn.Module):
    """Like SpinBlock but with the REAL carrier as the token-mixer -- for the generalization control."""
    def __init__(self, d, mlp_mult=4):
        super().__init__()
        self.carrier = RealCarrier(d)
        with torch.no_grad(): self.carrier.gate.fill_(2.0)            # strong mixer, not a side-channel
        self.ln2 = nn.LayerNorm(d)
        self.mlp = nn.Sequential(nn.Linear(d, mlp_mult * d), nn.GELU(), nn.Linear(mlp_mult * d, d))
    def forward(self, x):
        x, _ = self.carrier(x)
        x = x + self.mlp(self.ln2(x))
        return x
    def forward_state(self, x, h0=None):
        x, hs = self.carrier(x, h0)
        x = x + self.mlp(self.ln2(x))
        return x, hs

class SpinAttentionLM(nn.Module):
    """Attention-dominant for quality + one spin carrier for cross-token state."""
    def __init__(self, vocab=VOC, d=512, n_head=8, n_layer=3, mlp_mult=4, carrier="single"):
        super().__init__()
        self.d = d; self.mlp_mult = mlp_mult; self.n_head = n_head; self.carrier_mode = carrier
        self.emb = nn.Embedding(vocab, d)
        self.pos = nn.Embedding(4096, d)
        if carrier == "spin_dominant":                                   # INTENDED design: SPIN is the token-mixer
            self.blocks = nn.ModuleList([Block(d, n_head, mlp_mult) if i % 4 == 3 else SpinBlock(d, mlp_mult)
                                         for i in range(n_layer)])        #   core; attention only every 4th (helper)
        elif carrier == "real_dominant":                                 # GENERALIZATION control: a DIFFERENT
            self.blocks = nn.ModuleList([Block(d, n_head, mlp_mult) if i % 4 == 3 else RealBlock(d, mlp_mult)
                                         for i in range(n_layer)])        #   (real, no-phase) recurrence as the core
        else:
            self.blocks = nn.ModuleList([Block(d, n_head, mlp_mult) for _ in range(n_layer)])
        if carrier == "single":                                          # one carrier after all blocks (default):
            self.carrier = SpinCarrier(d)                                #   its share -> 0 as depth grows
        elif carrier == "per_block":                                     # one carrier per block: share stays constant
            self.carriers = nn.ModuleList([SpinCarrier(d) for _ in range(n_layer)])
        # carrier == "none": transformer-only baseline; "spin_dominant": spin-mixer blocks (built above)
        self.lnf = nn.LayerNorm(d)
        self.head = nn.Linear(d, vocab, bias=False)
        self.head.weight = self.emb.weight            # tied
        self.apply(self._init)
        for nm, p in self.named_parameters():         # GPT residual-proj scaling
            if nm.endswith('proj.weight') or nm.endswith('mlp.2.weight'):
                nn.init.normal_(p, 0.0, 0.02 / math.sqrt(2 * n_layer))
    def _init(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0.0, 0.02)
            if m.bias is not None: nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, 0.0, 0.02)
    def forward(self, x, state=None, return_state=False):
        B, T = x.shape
        pos = torch.arange(T, device=x.device)
        h = self.emb(x) + self.pos(pos)[None]
        ck = self.training and getattr(self, "grad_checkpoint", False)   # recompute acts in backward
        mode = getattr(self, "carrier_mode", "single"); states = []
        # generation path: carry the intra-block carrier state across forward calls (windows), so
        # spin_dominant / real_dominant get O(T) cross-window long context. Off in training (state=None,
        # return_state=False) so checkpointing/compile are untouched.
        thread = (return_state or state is not None) and mode in ("spin_dominant", "real_dominant")
        si = 0
        for i, b in enumerate(self.blocks):                              # -> trains a big model on a small GPU
            if thread and hasattr(b, "forward_state"):                   # SpinBlock/RealBlock -> thread its carrier state
                h0 = None if state is None else state[si]
                h, hs = (torch.utils.checkpoint.checkpoint(b.forward_state, h, h0, use_reentrant=False)
                         if ck else b.forward_state(h, h0)); states.append(hs); si += 1   # ck -> bounded long-ctx train
            else:
                h = torch.utils.checkpoint.checkpoint(b, h, use_reentrant=False) if ck else b(h)
            if mode == "per_block":                                      # carrier inside every block
                h, hs = self.carriers[i](h, None if state is None else state[i]); states.append(hs)
        if mode == "single":                                            # one carrier after all blocks
            h, hs = self.carrier(h, None if state is None else state[0]); states = [hs]
        lg = self.head(self.lnf(h))
        return (lg, states) if return_state else lg

    @torch.no_grad()
    def generate(self, ids, n_new=64, window=256, overlap=64, temp=0.0, rep=1.3, eos=0):
        """O(T) LONG-context generation. Attention only ever sees `window` tokens (the
        trained context) so positions never exceed the cap and quality stays in-distribution;
        the spin carrier carries the running cross-window state, so the prompt AND the output
        can be arbitrarily long without growing per-token compute. Long prompt -> processed in
        window-blocks (carry state). Generation -> when the live window fills, the older part is
        committed into the carrier and an `overlap` tail is kept for local attention continuity."""
        dev = self.emb.weight.device
        ids = list(ids); out = []; S = None
        keep_from = max(0, len(ids) - window)                  # commit everything before the last window
        j = 0
        while j < keep_from:
            blk = ids[j:min(j + window, keep_from)]
            _, S = self.forward(torch.tensor([blk], device=dev), state=S, return_state=True)
            j += window
        win = ids[keep_from:] or [eos]
        for _ in range(n_new):
            lg, _ = self.forward(torch.tensor([win], device=dev), state=S, return_state=True)
            logits = lg[0, -1].float()
            for t in set((win + out)[-40:]): logits[t] /= rep
            nx = int(logits.argmax()) if temp <= 0 else int(torch.multinomial(torch.softmax(logits / temp, -1), 1))
            if nx == eos: break
            out.append(nx); win = win + [nx]
            if len(win) >= window:                             # commit older tokens, keep overlap for context
                commit, win = win[:-overlap], win[-overlap:]
                _, S = self.forward(torch.tensor([commit], device=dev), state=S, return_state=True)
        return out

    def represent(self, x):
        """The brain's own contextual understanding of a sequence (learned hidden
        states, pre-head). Used as the meaning key for seek/retrieval."""
        B, T = x.shape
        pos = torch.arange(T, device=x.device)
        h = self.emb(x) + self.pos(pos)[None]
        mode = getattr(self, "carrier_mode", "single")             # mirror forward(): carrier placement varies
        for i, b in enumerate(self.blocks):
            h = b(h)
            if mode == "per_block": h, _ = self.carriers[i](h, None)
        if mode == "single": h, _ = self.carrier(h, None)
        return self.lnf(h)                       # (B,T,d) contextual representation

class FastWeightMemory(nn.Module):
    """In-model SUBCONSCIOUS = a hippocampal EPISODIC store (NOT RAG): it holds the brain's OWN
    hidden states as (key=context-rep -> value=next-token) traces, written by SURPRISE, and recalled
    by the model's own ATTENTION (softmax over the keys -> SELECTIVE, so distinct facts don't
    cross-talk). Traces decay (forgetting) and are consolidated into the slow weights ('neocortex')
    during sleep. Stores the model's representations, not external text -> in-model, not retrieval."""
    def __init__(self, d, cap=4096, decay=0.97):
        super().__init__()
        self.d = d; self.cap = cap; self.decay = decay
        self.register_buffer("K", torch.zeros(0, d))           # unit context keys
        self.register_buffer("V", torch.zeros(0, d))           # next-token value embeddings
        self.register_buffer("S", torch.zeros(0))              # trace strength (surprise, decaying)
        self.gate = nn.Parameter(torch.tensor(1.0))            # recall strength when something MATCHES
        self.sharp = nn.Parameter(torch.tensor(4.0))           # attention selectivity (softmax temperature)
    @torch.no_grad()
    def write(self, key, value, surprise):
        k = (key / (key.norm() + 1e-6)).unsqueeze(0).to(self.K)
        self.K = torch.cat([self.K, k]); self.V = torch.cat([self.V, value.unsqueeze(0).to(self.V)])
        self.S = torch.cat([self.S, torch.tensor([float(surprise)], device=self.S.device)])
        if self.K.size(0) > self.cap:                          # evict the weakest trace at capacity
            keep = self.S.argsort(descending=True)[:self.cap]
            self.K, self.V, self.S = self.K[keep], self.V[keep], self.S[keep]
    def read(self, h):
        if self.K.size(0) == 0: return h
        hn = h / (h.norm() + 1e-6)
        sims = self.K @ hn                                      # cosine of h to each stored key
        w = torch.softmax(sims * F.softplus(self.sharp), 0) * self.S    # SELECTIVE, strength-weighted
        recall = w @ self.V                                    # the matched trace's value (not a blur of all)
        recall = recall / (recall.norm() + 1e-6) * h.norm()    # scale to h's magnitude so it can steer the head
        conf = sims.max().clamp(min=0.0)                       # strong recall ONLY when h really matches a key
        return h + torch.sigmoid(self.gate) * conf * recall
    @torch.no_grad()
    def tick(self):                                            # time passes -> traces fade; drop the faded
        if self.S.numel() == 0: return
        self.S = self.S * self.decay
        keep = self.S > 0.05 * float(self.S.max())
        self.K, self.V, self.S = self.K[keep], self.V[keep], self.S[keep]
    @torch.no_grad()
    def energy(self): return float(self.S.sum()) if self.S.numel() else 0.0
    @torch.no_grad()
    def reset(self):
        self.K, self.V, self.S = self.K[:0], self.V[:0], self.S[:0]

def n_params(m): return sum(p.numel() for p in m.parameters())

# ----------------------------------------------------------------- data/eval
def load(name):
    # memory-map the token bin: stays on disk, batches page in lazily (chunked) ->
    # RAM stays ~flat instead of loading the whole corpus as int64. Read as int16
    # (identical bytes to the uint16 ids since vocab < 32768; torch-native dtype).
    return torch.from_numpy(np.memmap(f"data/{name}.bin", dtype=np.int16, mode="r"))
def batch(data, bs):
    ix = torch.randint(0, data.size(0) - L - 1, (bs,))
    x = torch.stack([data[i:i+L] for i in ix]).to(DEVICE).long()       # cast only this batch
    y = torch.stack([data[i+1:i+L+1] for i in ix]).to(DEVICE).long()
    return x, y
def batch_long(data, bs, W):
    """A batch of bs CONTIGUOUS sequences of W*L tokens, for windowed long-context training: the
    sequence is fed to the model in W windows of L, carrying the carrier state across them (the
    recurrence sees >L tokens of context while attention stays inside its trained L-window)."""
    span = W * L
    ix = torch.randint(0, data.size(0) - span - 1, (bs,))
    x = torch.stack([data[i:i+span] for i in ix]).to(DEVICE).long()
    y = torch.stack([data[i+1:i+span+1] for i in ix]).to(DEVICE).long()
    return x, y


class Prefetcher:
    """Background-thread batch loader: the thread does only the slow CPU work (memmap gather +
    pin) WHILE the GPU computes the current batch; the H2D copy happens in next() on the MAIN
    (default) stream, serialized right before the forward that consumes it. The old version did
    the copy on a side CUDA stream and handed the tensors off without record_stream() -- the
    caching allocator could then recycle that memory while the main stream still read it ->
    intermittent illegal-memory-access / device-side assert. Keeping all GPU work on the consumer
    stream removes the hazard while preserving the gather/compute overlap. Drop-in via next()."""
    def __init__(self, data, bs, depth=3):
        import threading, queue
        self.data, self.bs = data, bs
        self.q = queue.Queue(maxsize=depth)
        self.cuda = (DEVICE == "cuda")
        self._stop = False
        self.t = threading.Thread(target=self._worker, daemon=True); self.t.start()
    def _worker(self):
        n = self.data.size(0) - L - 1
        while not self._stop:
            ix = torch.randint(0, n, (self.bs,))
            xc = torch.stack([self.data[i:i+L] for i in ix])           # int16 CPU gather (the slow part)
            yc = torch.stack([self.data[i+1:i+L+1] for i in ix])
            if self.cuda:
                xc, yc = xc.pin_memory(), yc.pin_memory()              # pin so next()'s H2D is fast
            self.q.put((xc, yc))                                       # hand off CPU tensors -- NO GPU op in the thread
    def next(self):
        xc, yc = self.q.get()
        if not self.cuda:
            return xc.long(), yc.long()
        return (xc.to(DEVICE, non_blocking=True).long(),               # H2D on the consumer/default stream:
                yc.to(DEVICE, non_blocking=True).long())               # identical to batch(), serialized before forward
@torch.no_grad()
def val_ppl(model, vd, iters=40, bs=24):
    model.eval(); tot = n = 0
    for _ in range(iters):
        x, y = batch(vd, bs)
        tot += F.cross_entropy(model(x).reshape(-1, VOC), y.reshape(-1)).item() * y.numel(); n += y.numel()
    model.train(); return math.exp(tot / n)

def train(steps, bs, lr, d, n_head, n_layer, compile_=True):
    torch.manual_seed(0); np.random.seed(0)
    td, vd = load("s5train"), load("s5valid")
    model = SpinAttentionLM(VOC, d, n_head, n_layer).to(DEVICE)
    fwd = torch.compile(model, dynamic=False) if compile_ else model
    print(f"[hybrid] params={n_params(model):,} d={d} heads={n_head} layers={n_layer}", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.05, betas=(0.9, 0.95))
    warmup = 500
    def lr_at(it):
        if it < warmup: return it / warmup
        prog = (it - warmup) / max(1, steps - warmup)
        return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * prog))
    best, t0 = 1e9, time.time()
    for it in range(1, steps + 1):
        for grp in opt.param_groups: grp['lr'] = lr * lr_at(it)
        x, y = batch(td, bs)
        loss = F.cross_entropy(fwd(x).reshape(-1, VOC), y.reshape(-1))
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        if it % 500 == 0 or it == steps:
            p = val_ppl(model, vd)
            print(f"  it={it:6d} loss={loss.item():.3f} val_ppl={p:.2f} ({time.time()-t0:.0f}s)", flush=True)
            if p < best:
                best = p; torch.save(model.state_dict(), f"s6_hybrid.pt")
    json.dump({'params': n_params(model), 'd': d, 'heads': n_head, 'layers': n_layer,
               'best_val_ppl': best, 'steps': steps}, open("s6_hybrid_result.json", "w"), indent=2)
    print(f"[hybrid] BEST val ppl = {best:.2f}", flush=True)

@torch.no_grad()
def gen(d, n_head, n_layer):
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file("data/bpe8192.json")
    model = SpinAttentionLM(VOC, d, n_head, n_layer).to(DEVICE)
    model.load_state_dict(torch.load("s6_hybrid.pt", map_location=DEVICE, weights_only=True)); model.eval()
    for p in ["The Sun is", "Water is made of", "Paris is the capital of",
              "The Earth orbits the", "Albert Einstein was", "Once upon a time"]:
        ids = tok.encode(p).ids
        for _ in range(28):
            x = torch.tensor([ids[-L:]], device=DEVICE)
            nxt = model(x)[0, -1].argmax().item()
            if nxt == 0: break
            ids.append(nxt)
        print(f"  {p} ...-> {tok.decode(ids[len(tok.encode(p).ids):]).strip()[:90]}")

@torch.no_grad()
def swap(d, n_head, n_layer):
    """THE GATE: swap carried spin state mid-sequence; own<swapped<random => state causal."""
    model = SpinAttentionLM(VOC, d, n_head, n_layer).to(DEVICE)
    model.load_state_dict(torch.load("s6_hybrid.pt", map_location=DEVICE, weights_only=True)); model.eval()
    vd = load("s5valid"); torch.manual_seed(0)
    xa, _ = batch(vd, 128); xb, _ = batch(vd, 128)
    pre, cont = 128, 64
    _, sa = model(xa[:, :pre], return_state=True)
    _, sb = model(xb[:, :pre], return_state=True)
    def cnll(state, xc):
        lg = model(xc[:, pre-1:pre-1+cont], state=state)
        return F.cross_entropy(lg.reshape(-1, VOC), xc[:, pre:pre+cont].reshape(-1)).item()
    rnd = [torch.randn_like(s) for s in sb]
    own, swp, rndv = cnll(sb, xb), cnll(sa, xb), cnll(rnd, xb)
    res = {"own_state": own, "swapped_state": swp, "random_state": rndv,
           "state_is_causal": own < swp < rndv}
    print(json.dumps(res, indent=2), flush=True)
    json.dump(res, open("s6_swap_result.json", "w"), indent=2)

if __name__ == "__main__":
    pa = argparse.ArgumentParser()
    pa.add_argument("mode", choices=["train", "gen", "swap"])
    pa.add_argument("--steps", type=int, default=30000)
    pa.add_argument("--bs", type=int, default=32)
    pa.add_argument("--lr", type=float, default=8e-4)
    pa.add_argument("--d", type=int, default=512)
    pa.add_argument("--heads", type=int, default=8)
    pa.add_argument("--layers", type=int, default=3)
    a = pa.parse_args()
    if a.mode == "train": train(a.steps, a.bs, a.lr, a.d, a.heads, a.layers)
    elif a.mode == "gen": gen(a.d, a.heads, a.layers)
    else: swap(a.d, a.heads, a.layers)
