"""
================================================================================
PHASE S6 -- HYBRID spin + attention language model (quality, gated by brain-swap)
================================================================================
The pure spinning recurrence is on the wrong side of the efficiency curve for
language. This hybrid keeps the ROTATIONAL SPIN STATE as the cross-token carrier
(so the brain-swap / state-causality invariant can survive) but adds causal
self-attention to enrich each token's input (the quality lever).

Per block:   y = x + Attn(LN x)               # parallel context (quality)
             h_t = (1-eta)h_{t-1} + eta*g(y_t)*tanh(W h_{t-1} + U y_t + b)   # spin carrier
             x = y + h                          # residual: attention + spun state
W = -rho QQ^T + skew  -> rotational, unchanged from the proven SpinStep.

MANDATORY GATE (per handoff): the state-swap causal control must still pass --
swapping the carried spin state mid-sequence must degrade the continuation
(own < swapped < random). If it passes quality but FAILS this, it has degraded
to memorization = regression, and we say so.

Modes:
  train  from scratch on data/s5{train,valid}.bin (312M tok, BPE-8192)
  gen    greedy / factual probes
  swap   state-swap causal control (THE gate)
================================================================================
"""
import argparse, json, math, time, os
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from unified_brain import SpinStep
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
    """Standard transformer block (attention + MLP) -- the quality engine."""
    def __init__(self, d, n_head):
        super().__init__()
        self.ln1 = nn.LayerNorm(d); self.attn = CausalSelfAttention(d, n_head)
        self.ln2 = nn.LayerNorm(d)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))
    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

class SpinCarrier(nn.Module):
    """A single rotational spin recurrence that carries cross-token state, added
    to the stream through a learned gate. Keeps the model spin-flavored and gives
    a recurrent state we can probe -- without scrambling attention's context."""
    def __init__(self, d):
        super().__init__()
        self.ln = nn.LayerNorm(d)
        self.spin = SpinStep(d, d)
        self.gain = nn.Linear(d, d)
        self.gate = nn.Parameter(torch.tensor(-2.0))   # sigmoid(-2)=0.12: light by default
    def forward(self, x, h0=None):
        B, T, C = x.shape
        yn = self.ln(x); W = self.spin.W()
        h = torch.zeros(B, C, device=x.device) if h0 is None else h0
        eu = yn @ self.spin.U.t() + self.spin.b
        g = torch.sigmoid(self.gain(yn)) * 2.0
        eta = self.spin.eta
        outs = []
        for t in range(T):
            h = (1 - eta) * h + eta * g[:, t] * torch.tanh(h @ W.t() + eu[:, t])
            outs.append(h)
        return x + torch.sigmoid(self.gate) * torch.stack(outs, 1), h

class SpinAttentionLM(nn.Module):
    """Attention-dominant for quality + one spin carrier for cross-token state."""
    def __init__(self, vocab=VOC, d=512, n_head=8, n_layer=3):
        super().__init__()
        self.d = d
        self.emb = nn.Embedding(vocab, d)
        self.pos = nn.Embedding(4096, d)
        self.blocks = nn.ModuleList([Block(d, n_head) for _ in range(n_layer)])
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
    return torch.from_numpy(np.fromfile(f"data/{name}.bin", dtype=np.uint16).astype(np.int64))
def batch(data, bs):
    ix = torch.randint(0, data.size(0) - L - 1, (bs,))
    x = torch.stack([data[i:i+L] for i in ix]).to(DEVICE)
    y = torch.stack([data[i+1:i+L+1] for i in ix]).to(DEVICE)
    return x, y
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
