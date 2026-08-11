import sys; sys.path.insert(0, "/opt/code/membrae")
import mqar_probe as M
import torch, torch.nn as nn, torch.nn.functional as F

class DeltaTied(nn.Module):     # delta with TIED q=k (same token -> same key/query) + configurable state dim
    def __init__(self, d, dh=None):
        super().__init__(); self.dh = dh or d
        self.k = nn.Linear(d, self.dh); self.v = nn.Linear(d, self.dh)
        self.beta = nn.Linear(d, 1); self.o = nn.Linear(self.dh, d)
    def forward(self, x):
        B, T, d = x.shape
        kq = F.normalize(self.k(x), dim=-1); v = self.v(x); beta = torch.sigmoid(self.beta(x))
        S = torch.zeros(B, self.dh, self.dh, device=x.device); outs = []
        for t in range(T):
            kt, vt, bt = kq[:, t], v[:, t], beta[:, t]
            Sk = torch.einsum('bvk,bk->bv', S, kt)
            S = S + bt.unsqueeze(-1) * torch.einsum('bv,bk->bvk', vt - Sk, kt)
            outs.append(torch.einsum('bvk,bk->bv', S, kt))     # query with the SAME kt (tied)
        return self.o(torch.stack(outs, 1))

class DiagBig(nn.Module):       # our diagonal carrier with a configurable (bigger) state dim
    def __init__(self, d, dh=None):
        super().__init__(); self.dh = dh or d
        self.b = nn.Linear(d, self.dh); self.dt = nn.Linear(d, self.dh); self.C = nn.Linear(self.dh, d)
        self.A = nn.Parameter(torch.zeros(self.dh))
    def forward(self, x):
        B, T, d = x.shape; bb = self.b(x); lam = torch.exp(-F.softplus(self.dt(x)) * F.softplus(self.A))
        h = torch.zeros(B, self.dh, device=x.device); ys = []
        for t in range(T):
            h = lam[:, t] * h + bb[:, t]; ys.append(h)
        return self.C(torch.stack(ys, 1))

def mk(cls, dh):
    return lambda d: cls(d, dh)

def run(carrier_fn, np_, steps=3000, lr=1.5e-3):
    vocab = 64 + 64 + 1; torch.manual_seed(0)
    m = M.TinyLM(vocab, 128, 2, carrier_fn, use_pos=False).to(M.DEV)
    opt = torch.optim.AdamW(m.parameters(), lr=lr); m.train()
    for s in range(steps):
        x, y = M.make_batch(64, np_, 4, 64, 64); lg = m(x)
        loss = F.cross_entropy(lg.reshape(-1, vocab), y.reshape(-1), ignore_index=-100)
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
    m.eval(); c = t = 0
    with torch.no_grad():
        for _ in range(20):
            x, y = M.make_batch(64, np_, 4, 64, 64); p = m(x).argmax(-1); msk = y != -100
            c += (p[msk] == y[msk]).sum().item(); t += msk.sum().item()
    return c / t

print("levers at the FAILURE point (16 pairs), pos off, 3000 steps:")
print(f"  delta tied-qk, state=128 : {run(mk(DeltaTied,128),16)*100:5.1f}%", flush=True)
print(f"  delta tied-qk, state=256 : {run(mk(DeltaTied,256),16)*100:5.1f}%", flush=True)
print(f"  delta tied-qk, state=512 : {run(mk(DeltaTied,512),16)*100:5.1f}%", flush=True)
print(f"  diagonal(ours) state=512 : {run(mk(DiagBig,512),16)*100:5.1f}%", flush=True)
