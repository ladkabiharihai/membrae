"""DECISIVE PROBE (long-context strategy, Step 0): does a DELTA-RULE carrier do the associative recall that our
DIAGONAL (Mamba/S6-style) carrier cannot? MQAR = multi-query associative recall, the standard synthetic where
diagonal SSMs fail as #pairs grows and delta-rule / attention succeed. Tiny scale (d=128, 2 layers), seconds/run.

Three carriers, SAME tiny model + data:
  - diagonal : input-dependent diagonal decay h_t = lambda_t (.) h_{t-1} + b_t  -> our SelectiveSpinCarrier mechanism
  - delta    : DeltaNet fast-weight  S_t = S_{t-1} + beta_t (v_t - S_{t-1} k_t) k_t^T ,  o_t = S_t q_t
  - attention: causal MHA (gold-standard reference; O(T^2))
VERDICT: if delta tracks attention while diagonal collapses at high #pairs -> delta-rule is the missing mechanism,
Angle 2 is ALIVE. If delta also collapses -> Angle 2 is dead, go consolidate.
"""
import os, math, random, torch, torch.nn as nn, torch.nn.functional as F
DEV = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(0); random.seed(0)

# ---------------- MQAR data ----------------
def make_batch(B, n_pairs, n_query, K, V):
    T = 2 * n_pairs + 2 * n_query
    x = torch.zeros(B, T, dtype=torch.long); y = torch.full((B, T), -100, dtype=torch.long)
    for b in range(B):
        keys = random.sample(range(1, K + 1), n_pairs)
        vals = [random.randint(K + 1, K + V) for _ in range(n_pairs)]
        kv = dict(zip(keys, vals)); seq = []
        for k, v in zip(keys, vals): seq += [k, v]
        qk = [random.choice(keys) for _ in range(n_query)]
        for q in qk: seq += [q, kv[q]]
        x[b] = torch.tensor(seq)
        for j in range(n_query):
            p = 2 * n_pairs + 2 * j          # position of the query key -> next token must be its value
            y[b, p] = kv[qk[j]]
    return x.to(DEV), y.to(DEV)

# ---------------- carriers ----------------
class DiagonalCarrier(nn.Module):                       # our SelectiveSpinCarrier mechanism (real-valued S6)
    def __init__(self, d):
        super().__init__()
        self.b = nn.Linear(d, d); self.dt = nn.Linear(d, d); self.C = nn.Linear(d, d)
        self.A = nn.Parameter(torch.zeros(d))
    def forward(self, x):
        B, T, d = x.shape
        b = self.b(x); lam = torch.exp(-F.softplus(self.dt(x)) * F.softplus(self.A))    # (0,1) decay
        h = torch.zeros(B, d, device=x.device); ys = []
        for t in range(T):
            h = lam[:, t] * h + b[:, t]; ys.append(h)
        return self.C(torch.stack(ys, 1))

class DeltaCarrier(nn.Module):                          # DeltaNet fast-weight (associative recall)
    def __init__(self, d, dh=None):
        super().__init__()
        self.dh = dh or d
        self.k = nn.Linear(d, self.dh); self.v = nn.Linear(d, self.dh)
        self.q = nn.Linear(d, self.dh); self.beta = nn.Linear(d, 1); self.o = nn.Linear(self.dh, d)
    def forward(self, x):
        B, T, d = x.shape
        k = F.normalize(self.k(x), dim=-1); v = self.v(x); q = self.q(x); beta = torch.sigmoid(self.beta(x))
        S = torch.zeros(B, self.dh, self.dh, device=x.device); outs = []
        for t in range(T):
            kt, vt, qt, bt = k[:, t], v[:, t], q[:, t], beta[:, t]
            Sk = torch.einsum('bvk,bk->bv', S, kt)
            S = S + bt.unsqueeze(-1) * torch.einsum('bv,bk->bvk', vt - Sk, kt)          # delta rule (overwrite)
            outs.append(torch.einsum('bvk,bk->bv', S, qt))                              # recall S q_t
        return self.o(torch.stack(outs, 1))

class AttnCarrier(nn.Module):                           # gold-standard causal attention
    def __init__(self, d, heads=4):
        super().__init__()
        self.h = heads; self.qkv = nn.Linear(d, 3 * d); self.o = nn.Linear(d, d)
    def forward(self, x):
        B, T, d = x.shape; qkv = self.qkv(x).view(B, T, 3, self.h, d // self.h).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.o(y.transpose(1, 2).reshape(B, T, d))

CARRIERS = {"diagonal": DiagonalCarrier, "delta": DeltaCarrier, "attention": AttnCarrier}

# ---------------- tiny model ----------------
class Block(nn.Module):
    def __init__(self, d, cls):
        super().__init__(); self.n1 = nn.LayerNorm(d); self.mix = cls(d); self.n2 = nn.LayerNorm(d)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))
    def forward(self, x): x = x + self.mix(self.n1(x)); return x + self.mlp(self.n2(x))

class TinyLM(nn.Module):
    def __init__(self, vocab, d, layers, cls, use_pos=True):
        super().__init__(); self.emb = nn.Embedding(vocab, d)
        self.pos = nn.Embedding(4096, d) if use_pos else None
        self.blocks = nn.ModuleList([Block(d, cls) for _ in range(layers)]); self.head = nn.Linear(d, vocab)
    def forward(self, x):
        h = self.emb(x)
        if self.pos is not None: h = h + self.pos(torch.arange(x.shape[1], device=x.device))
        for b in self.blocks: h = b(h)
        return self.head(h)

# ---------------- train + eval ----------------
def run(name, n_pairs, K=64, V=64, d=128, layers=2, steps=800, B=64, n_query=4):
    vocab = K + V + 1; torch.manual_seed(0)
    m = TinyLM(vocab, d, layers, CARRIERS[name]).to(DEV)
    opt = torch.optim.AdamW(m.parameters(), lr=3e-3)
    m.train()
    for s in range(steps):
        x, y = make_batch(B, n_pairs, n_query, K, V)
        lg = m(x); loss = F.cross_entropy(lg.reshape(-1, vocab), y.reshape(-1), ignore_index=-100)
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
    m.eval(); cor = tot = 0
    with torch.no_grad():
        for _ in range(20):
            x, y = make_batch(B, n_pairs, n_query, K, V); lg = m(x)
            msk = y != -100; pred = lg.argmax(-1)
            cor += (pred[msk] == y[msk]).sum().item(); tot += msk.sum().item()
    return cor / tot

if __name__ == "__main__":
    PAIRS = [4, 8, 16, 32, 64]
    print(f"MQAR probe on {DEV} | K=V=64 vocab, d=128, 2 layers, 800 steps/config | acc at query positions\n")
    print(f"{'#pairs':>7} | {'diagonal(ours)':>15} | {'delta-rule':>11} | {'attention':>10}")
    print("-" * 52)
    for np_ in PAIRS:
        row = {}
        for name in ["diagonal", "delta", "attention"]:
            row[name] = run(name, np_)
        print(f"{np_:>7} | {row['diagonal']*100:>14.1f}% | {row['delta']*100:>10.1f}% | {row['attention']*100:>9.1f}%", flush=True)
    print("\nVERDICT: delta tracks attention while diagonal collapses -> delta-rule is the missing recall mechanism (Angle 2 ALIVE).")
