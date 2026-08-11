"""RESEARCH PROTOTYPE (does not touch the live model/training): a SELECTIVE spin carrier.

Current `SpinCarrier` (s6_hybrid.py) uses an INPUT-INDEPENDENT decay: lambda = exp(-exp(nu))*exp(i*theta),
where nu/theta are fixed learned parameters -- the SAME decay for every token. So the state forgets on a
FIXED schedule and the model cannot CHOOSE to hold a specific token -> the measured needle failure (recall
collapses by ~200 tokens even in-window; RESULTS_MEASURED #20).

Mamba/S6 fix, adapted to the complex spin carrier: make the decay MAGNITUDE input-dependent via a per-token
step size Delta_t = softplus(W_delta x_t):
        |lambda_t| = exp(-Delta_t * exp(nu))            (still in (0,1) -> stable; |lambda|<1 preserved)
When the model wants to LATCH a token it drives Delta_t -> 0 (|lambda_t| -> 1, remember); to forget it drives
Delta_t large (|lambda_t| -> 0). Phase theta stays per-channel (rotation), input-independent, for stability.

KEY: this keeps the O(T*d) PARALLEL ASSOCIATIVE SCAN and the O(1)/token, KV-cache-free inference -- the whole
project thesis. The existing Hillis-Steele scan already carries a PER-POSITION A_t, so it needs per-timestep
lambda_t instead of a broadcast constant; `_scan_tv` below is that generalization (one line different).

This file trains a TINY model on a synthetic COPY task on CPU (zero GPU/SFT/prod impact) to show the
selective carrier can recall where the fixed one cannot.
"""
import math, torch, torch.nn as nn, torch.nn.functional as F


def _scan_tv(lr, li, br, bi):
    """Time-varying diagonal-complex associative scan: lr,li,br,bi all [B,T,D]. Same Hillis-Steele combine
    as s6_hybrid._lru_scan, but A_t is per-timestep (selective) instead of a broadcast constant."""
    B, T, D = br.shape
    Ar, Ai = lr.clone(), li.clone()
    Hr, Hi = br.clone(), bi.clone()
    d = 1
    while d < T:
        arr, ari = Ar[:, d:], Ai[:, d:]; alr, ali = Ar[:, :T - d], Ai[:, :T - d]
        hlr, hli = Hr[:, :T - d], Hi[:, :T - d]
        tHr = arr * hlr - ari * hli; tHi = arr * hli + ari * hlr
        Hr = torch.cat([Hr[:, :d], Hr[:, d:] + tHr], 1); Hi = torch.cat([Hi[:, :d], Hi[:, d:] + tHi], 1)
        nAr = arr * alr - ari * ali; nAi = arr * ali + ari * alr
        Ar = torch.cat([Ar[:, :d], nAr], 1); Ai = torch.cat([Ai[:, :d], nAi], 1)
        d *= 2
    return Hr, Hi


class SelectiveSpinCarrier(nn.Module):
    """Input-dependent-decay spin carrier. Drop-in shape-compatible with SpinCarrier.forward(x)->(x,state)."""
    def __init__(self, d, selective=True):
        super().__init__()
        self.d, self.selective = d, selective
        self.ln = nn.LayerNorm(d)
        r = torch.rand(d)
        self.nu = nn.Parameter(torch.log(-torch.log(0.5 + 0.4 * r)))   # base decay, same init as SpinCarrier
        self.theta = nn.Parameter(math.pi * torch.rand(d))
        self.U_re = nn.Linear(d, d); self.U_im = nn.Linear(d, d)
        self.gain = nn.Linear(d, d)
        self.C = nn.Linear(2 * d, d)
        self.gate = nn.Parameter(torch.tensor(-2.0))
        if selective:
            self.delta = nn.Linear(d, d)                                # per-token step size -> selective decay
            # LATCH-BIASED init: bias<0 -> softplus small -> |lambda|~1 (remember by default; LEARN to forget).
            nn.init.zeros_(self.delta.weight); nn.init.constant_(self.delta.bias, -2.0)

    def forward(self, x):
        yn = self.ln(x)
        g = torch.sigmoid(self.gain(yn))
        br = self.U_re(yn) * g; bi = self.U_im(yn) * g
        base = torch.exp(self.nu)                                       # >0
        if self.selective:
            dt = F.softplus(self.delta(yn))                            # [B,T,D] > 0 ; small -> latch, large -> forget
            mag = torch.exp(-dt * base)                                # per-token |lambda_t| in (0,1)
        else:
            mag = torch.exp(-base).view(1, 1, -1).expand_as(br)        # fixed decay (baseline)
        ph = torch.exp(self.theta).view(1, 1, -1)
        lr = mag * torch.cos(ph); li = mag * torch.sin(ph)
        hr, hi = _scan_tv(lr, li, br, bi)
        out = self.C(torch.cat([hr, hi], -1))
        return x + torch.sigmoid(self.gate) * out


class ToyLM(nn.Module):
    """Minimal carrier-only LM (no attention) to isolate the carrier's recall ability."""
    def __init__(self, vocab, d, layers, selective):
        super().__init__()
        self.emb = nn.Embedding(vocab, d)
        self.blocks = nn.ModuleList([SelectiveSpinCarrier(d, selective) for _ in range(layers)])
        self.ln = nn.LayerNorm(d); self.head = nn.Linear(d, vocab)
    def forward(self, x):
        h = self.emb(x)
        for b in self.blocks: h = b(h)
        return self.head(self.ln(h))


def copy_task(bs, T, vocab, n_vals):
    """Sequence: [WRITE, value, filler.... , QUERY] ; target at the QUERY position = value.
    WRITE=vocab-2, QUERY=vocab-1, fillers in [1, vocab-n_vals-2], values in [1, n_vals]."""
    WRITE, QUERY = vocab - 2, vocab - 1
    x = torch.randint(n_vals + 1, vocab - 2, (bs, T))
    vals = torch.randint(1, n_vals + 1, (bs,))
    x[:, 0] = WRITE; x[:, 1] = vals; x[:, -1] = QUERY
    y = torch.full((bs, T), -100)
    y[:, -1] = vals                                                    # only score the recall position
    return x, y


def run(selective, T, steps=800, vocab=40, d=64, layers=2, n_vals=8, seed=0):
    torch.manual_seed(seed)
    m = ToyLM(vocab, d, layers, selective); opt = torch.optim.AdamW(m.parameters(), lr=3e-3)
    for _ in range(steps):
        x, y = copy_task(64, T, vocab, n_vals)
        loss = F.cross_entropy(m(x).reshape(-1, vocab), y.reshape(-1), ignore_index=-100)
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
    with torch.no_grad():
        x, y = copy_task(512, T, vocab, n_vals)
        pred = m(x)[:, -1].argmax(-1)
        acc = (pred == y[:, -1]).float().mean().item()
    return acc


if __name__ == "__main__":
    print("Synthetic copy task: recall a value planted at position 1, queried at position T-1.")
    print("(carrier-only tiny LM, CPU; fixed-decay baseline vs input-dependent selective decay)\n")
    print(f"  {'gap T':>6} | {'fixed-decay':>11} | {'selective':>10}")
    for T in [8, 32, 64, 128]:
        af = run(False, T); as_ = run(True, T)
        print(f"  {T:>6} | {af:>11.2f} | {as_:>10.2f}")
    print("\n(chance = 1/8 = 0.125; higher = better recall. Selective should hold recall as the gap grows.)")
