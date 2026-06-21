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

class SpinAttentionLM(nn.Module):
    """Attention-dominant for quality + one spin carrier for cross-token state."""
    def __init__(self, vocab=VOC, d=512, n_head=8, n_layer=3, mlp_mult=4):
        super().__init__()
        self.d = d; self.mlp_mult = mlp_mult; self.n_head = n_head
        self.emb = nn.Embedding(vocab, d)
        self.pos = nn.Embedding(4096, d)
        self.blocks = nn.ModuleList([Block(d, n_head, mlp_mult) for _ in range(n_layer)])
        self.carrier = SpinCarrier(d)
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
        for b in self.blocks:
            h = b(h)
        h, hs = self.carrier(h, None if state is None else state[0])
        lg = self.head(self.lnf(h))
        return (lg, [hs]) if return_state else lg

    def represent(self, x):
        """The brain's own contextual understanding of a sequence (learned hidden
        states, pre-head). Used as the meaning key for seek/retrieval."""
        B, T = x.shape
        pos = torch.arange(T, device=x.device)
        h = self.emb(x) + self.pos(pos)[None]
        for b in self.blocks:
            h = b(h)
        h, _ = self.carrier(h, None)
        return self.lnf(h)                       # (B,T,d) contextual representation

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


class Prefetcher:
    """Background-thread batch loader: builds the next batch (memmap gather + pinned H2D
    copy on a side stream) WHILE the GPU computes the current one, hiding the synchronous
    data-prep stall behind compute. Same sampling distribution as batch(); drop-in via next()."""
    def __init__(self, data, bs, depth=3):
        import threading, queue
        self.data, self.bs = data, bs
        self.q = queue.Queue(maxsize=depth)
        self.cuda = (DEVICE == "cuda")
        self.stream = torch.cuda.Stream() if self.cuda else None
        self._stop = False
        self.t = threading.Thread(target=self._worker, daemon=True); self.t.start()
    def _worker(self):
        n = self.data.size(0) - L - 1
        while not self._stop:
            ix = torch.randint(0, n, (self.bs,))
            xc = torch.stack([self.data[i:i+L] for i in ix])           # int16 CPU gather
            yc = torch.stack([self.data[i+1:i+L+1] for i in ix])
            if self.cuda:
                xc, yc = xc.pin_memory(), yc.pin_memory()
                with torch.cuda.stream(self.stream):
                    x = xc.to(DEVICE, non_blocking=True).long()        # async H2D + cast on side stream
                    y = yc.to(DEVICE, non_blocking=True).long()
                self.stream.synchronize()                              # copy done -> safe to hand off
                self.q.put((x, y))
            else:
                self.q.put((xc.long(), yc.long()))
    def next(self):
        return self.q.get()
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
