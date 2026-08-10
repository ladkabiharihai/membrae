"""PHASE 2, Step 1 (refocused): the REAL goal is needle recall = 1 fact over a LONG sequence, not 32-64
simultaneous bindings. Test the carriers on LONG-NEEDLE: plant 1 key-value pair, then L filler tokens, then
query -> predict the value. This is exactly our long-context task. If Based recalls at long L where the diagonal
(ours) dies, it's the carrier we need -> integrate + graft into the 1B."""
import sys; sys.path.insert(0, "/opt/code/membrae")
import mqar_probe as M
import torch, torch.nn as nn, torch.nn.functional as F
DEV = M.DEV

class MHBased(nn.Module):
    def __init__(self, d, heads=4, feat=12):
        super().__init__(); self.h = heads; self.fe = feat; self.dh = d // heads
        self.q = nn.Linear(d, heads * feat); self.k = nn.Linear(d, heads * feat); self.v = nn.Linear(d, d); self.o = nn.Linear(d, d)
    def taylor(self, x):
        B, T, H, fd = x.shape
        x2 = (x.unsqueeze(-1) * x.unsqueeze(-2)).reshape(B, T, H, fd * fd) / (2 ** 0.5)
        return torch.cat([torch.ones(B, T, H, 1, device=x.device), x, x2], -1)
    def forward(self, x):
        B, T, d = x.shape; H, fe, dh = self.h, self.fe, self.dh
        q = self.taylor(self.q(x).view(B, T, H, fe)); k = self.taylor(self.k(x).view(B, T, H, fe)); v = self.v(x).view(B, T, H, dh)
        kv = k.unsqueeze(-1) * v.unsqueeze(-2); S = torch.cumsum(kv, 1); Z = torch.cumsum(k, 1)
        num = torch.einsum('bthf,bthfd->bthd', q, S); den = torch.einsum('bthf,bthf->bth', q, Z).clamp_min(1e-4).unsqueeze(-1)
        return self.o((num / den).reshape(B, T, d))

# long-needle batch: [k, v, filler x L, k] -> predict v at the final key position
def needle_batch(B, L, K=32, V=32, Fr=64):    # Fr = filler vocab (distinct from keys/values)
    T = 2 + L + 1
    x = torch.zeros(B, T, dtype=torch.long); y = torch.full((B, T), -100, dtype=torch.long)
    import random
    for b in range(B):
        k = random.randint(1, K); v = random.randint(K + 1, K + V)
        fill = [random.randint(K + V + 1, K + V + Fr) for _ in range(L)]
        x[b] = torch.tensor([k, v] + fill + [k]); y[b, -1] = v      # predict v after the requeried key
    return x.to(DEV), y.to(DEV)

def run_needle(carrier, L, steps=3000, lr=2e-3, B=48):
    vocab = 32 + 32 + 64 + 1; torch.manual_seed(0); M.CARRIERS["c"] = carrier
    m = M.TinyLM(vocab, 128, 2, M.CARRIERS["c"], use_pos=True).to(DEV)
    opt = torch.optim.AdamW(m.parameters(), lr=lr); m.train()
    for s in range(steps):
        x, y = needle_batch(B, L); lg = m(x)
        loss = F.cross_entropy(lg.reshape(-1, vocab), y.reshape(-1), ignore_index=-100)
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
    m.eval(); c = t = 0
    with torch.no_grad():
        for _ in range(20):
            x, y = needle_batch(B, L); p = m(x).argmax(-1); msk = y != -100
            c += (p[msk] == y[msk]).sum().item(); t += msk.sum().item()
    return c / t

mh = lambda d: MHBased(d, 4, 12)
print("LONG-NEEDLE (1 fact, L filler tokens) -- the actual long-context task, pos-ON:")
print(f"{'L filler':>9} | {'mh-Based':>9} | {'diagonal':>9}")
for L in [64, 256, 512, 1024]:
    print(f"{L:>9} | {run_needle(mh, L)*100:8.1f}% | {run_needle(M.CARRIERS['diagonal'], L)*100:8.1f}%", flush=True)
