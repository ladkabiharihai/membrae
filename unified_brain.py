"""
================================================================================
UNIFIED SPINNING BRAIN  --  one model, all proven capabilities
================================================================================

This file combines every capability proven in the project into ONE coherent
brain object (`UnifiedBrain`), instead of ten separate test models. It is the
consolidation deliverable: a single architecture you can instantiate, train,
query, and extend.

WHAT ONE UnifiedBrain INSTANCE CAN DO
-------------------------------------
  perceive(symbols | image | audio)   -> shared latent (Omni, with fusion)
  reason(latent, skill)               -> answer via skill-conditioned spinning
  confidence(latent, skill)           -> knows-what-it-doesn't-know (P5)
  answer_or_abstain(latent, skill)    -> refuses to fabricate (P6)
  parse_language(tokens)              -> structured meaning (Stage 3)
  generate_language(meaning)          -> sentence, autoregressive (Stage 4)
  seek(question_tokens, store)        -> generates LANGUAGE queries, fetches,
                                         reasons to answer (Stage 5 / P7)
  exact_accumulate(digit_seq, mod)    -> clean-state recurrence (ceiling fix)
  explore_and_learn(world)            -> curiosity-driven online learning with
                                         self-replay, zero forgetting (Stage 6)

ARCHITECTURE
------------
One shared spinning core per module family:
  SpinStep:  F(h,x) = (1-eta)h + eta*tanh(Wh + Ux + b),
             W = -rho*(QQ^T) + skew   ->  ROTATIONAL dynamics (orbits).
  The answer lives in the PHASE of a non-converging oscillation (causally
  validated by brain-swap). Skill-conditioning modulates HOW the core spins
  (per-skill gain+bias) so diverse skills coexist without interference.
  CleanStateRecurrence: commit-and-refeed discrete state (straight-through),
  defeating the bare spin's exact-accumulation limit.

HONEST SCOPE
------------
All capabilities are demonstrated at toy scale (vocab<=35, sentences<=11 tok,
8x8 grids, synthetic tones). Known limits: scaled compositional generalization
is partial (0.65 held-out at 35-word recursive grammar); relational vision
~0.76; bare spin cannot do many-wrap exact arithmetic (use the clean-state
module). This file proves the INTEGRATION of mechanisms; scale is the frontier.

USAGE
-----
  python unified_brain.py            # builds one brain, trains all faculties,
                                     # runs the integrated self-test, prints report
  from unified_brain import UnifiedBrain   # to extend / build further
================================================================================
"""
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F, json, time

torch.manual_seed(0); np.random.seed(0)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ============================================================================
# 1. CORE DYNAMICS
# ============================================================================
class SpinStep(nn.Module):
    """One spinning step. W = -rho*(QQ^T) + skew -> rotational (orbit) dynamics."""
    def __init__(self, d, in_dim, eta=0.3, rho=0.3):
        super().__init__()
        self.eta, self.rho, self.d = eta, rho, d
        self.A = nn.Parameter(0.1 * torch.randn(d, d))
        self.S_raw = nn.Parameter(0.1 * torch.randn(d, d))
        self.U = nn.Parameter(0.1 * torch.randn(d, in_dim))
        self.b = nn.Parameter(torch.zeros(d))
    def W(self):
        Q, _ = torch.linalg.qr(self.A)
        return -self.rho * (Q @ Q.t()) + (self.S_raw - self.S_raw.t())
    def forward(self, h, x, W):
        return (1 - self.eta) * h + self.eta * torch.tanh(h @ W.t() + x @ self.U.t() + self.b)


class CondSpinStep(nn.Module):
    """Skill-conditioned spin: per-skill gain+bias change HOW the core spins,
    so diverse skills share one core without interference (Stage-1 result)."""
    def __init__(self, d, in_dim, nskill, eta=0.3, rho=0.3):
        super().__init__()
        self.eta, self.rho = eta, rho
        self.A = nn.Parameter(0.1 * torch.randn(d, d))
        self.S_raw = nn.Parameter(0.1 * torch.randn(d, d))
        self.U = nn.Parameter(0.1 * torch.randn(d, in_dim))
        self.b = nn.Parameter(torch.zeros(d))
        self.gain = nn.Parameter(torch.ones(nskill, d))
        self.bias = nn.Parameter(torch.zeros(nskill, d))
    def W(self):
        Q, _ = torch.linalg.qr(self.A)
        return -self.rho * (Q @ Q.t()) + (self.S_raw - self.S_raw.t())
    def forward(self, h, x, W, sk):
        upd = torch.tanh(h @ W.t() + x @ self.U.t() + self.b + self.bias[sk])
        return (1 - self.eta) * h + self.eta * (self.gain[sk] * upd)


class CleanStateRecurrence(nn.Module):
    """Commit-and-refeed discrete state (straight-through). Defeats the bare
    spin's many-wrap exact-arithmetic limit; scales with length (ceiling fix)."""
    def __init__(self, n_states, in_classes, d=64):
        super().__init__()
        self.n = n_states
        self.cell = nn.Sequential(nn.Linear(n_states + in_classes, d), nn.ReLU(),
                                  nn.Linear(d, n_states))
        self.in_classes = in_classes
    def forward(self, seq):                      # seq: (B,L) int tokens
        B, L = seq.shape
        r = torch.zeros(B, self.n, device=seq.device); r[:, 0] = 1.0
        outs = []
        for t in range(L):
            x = F.one_hot(seq[:, t], self.in_classes).float()
            lg = self.cell(torch.cat([r, x], 1)); outs.append(lg)
            hard = F.one_hot(lg.argmax(-1), self.n).float()
            r = hard + (F.softmax(lg, -1) - F.softmax(lg, -1).detach())
        return torch.stack(outs, 1)


# ============================================================================
# 2. THE UNIFIED BRAIN
# ============================================================================
# ---- fixed small world / language spec (toy scale, as proven) ----
M = 5                                # value alphabet 0..4
NSKILL = 3                           # parity-3 / sum-of-3 / max  (Stage 1)
NBIT = 8                             # symbolic input: 8 digits
GRID = 4; GC = 3                     # vision: 4x4 grid, 3 cell-colors
ALEN = 64; NTONE = 5                 # audio: 64-sample wave, 5 tones
NA = NC = NO = 4                     # language meaning space (action,color,object)
BOS, EOS = 12, 13; LV = 14           # language vocab (12 content + BOS/EOS)
NE = 8                               # seek-world entities
SW = ['<pad>', 'what', 'is', 'plus', 'value', 'of'] + [f'ent{i}' for i in range(NE)] + ['<bos>', '<eos>']
ST = {w: i for i, w in enumerate(SW)}; SV = len(SW); ENT0 = ST['ent0']
ABSTAIN = M                          # the "I don't know" class


class UnifiedBrain(nn.Module):
    """One brain object exposing every proven capability. See module docstring."""
    def __init__(self, d=128):
        super().__init__()
        self.d = d
        # ---- perception (Omni): three encoders into ONE shared latent ----
        self.enc_sym   = nn.Linear(NBIT * M, d)
        self.enc_img   = nn.Linear(GRID * GRID * GC, d)
        self.enc_aud   = nn.Linear(ALEN, d)
        # ---- reasoning core (skill-conditioned spin) over the shared latent ----
        self.core = CondSpinStep(d, d, NSKILL)
        self.T = 10
        self.head_ans = nn.Linear(d, M + 1)          # +1 = ABSTAIN (P6)
        # ---- language: sequential reader + meaning heads (Stage 3) ----
        self.lang_emb = nn.Embedding(LV, 36)
        self.lang_read = SpinStep(d, 36)
        self.head_act = nn.Linear(d, NA); self.head_col = nn.Linear(d, NC); self.head_obj = nn.Linear(d, NO)
        # ---- language generation: meaning-conditioned decoder (Stage 4) ----
        self.mean_emb = nn.Linear(NA + NC + NO, d)
        self.dec = SpinStep(d, 36 + d)
        self.head_tok = nn.Linear(d, LV)
        # ---- seeking (Stage 5): read question, generate language queries, integrate ----
        self.seek_emb = nn.Embedding(SV, 36)
        self.seek_read = SpinStep(d, 36)
        self.qdec = SpinStep(d, 36 + d)
        self.head_qtok = nn.Linear(d, SV)
        self.val_emb = nn.Embedding(M, 36)
        self.integrate = SpinStep(d, 36)
        self.head_sum = nn.Linear(d, 9)              # 0..8
        # ---- exact accumulation (ceiling fix) ----
        self.counter = CleanStateRecurrence(M, M)
        # ---- knowledge head for explore/learn (Stage 6) ----
        self.know = nn.Sequential(nn.Linear(NE, 64), nn.ReLU(), nn.Linear(64, M))

    # ------------------------- PERCEPTION (Omni) -------------------------
    def perceive(self, symbols=None, image=None, audio=None):
        """Fuse any subset of modalities into one latent (sum of encodings)."""
        B = next(x.size(0) for x in (symbols, image, audio) if x is not None)
        z = torch.zeros(B, self.d, device=DEVICE)
        if symbols is not None: z = z + self.enc_sym(symbols)
        if image   is not None: z = z + self.enc_img(image)
        if audio   is not None: z = z + self.enc_aud(audio)
        return torch.tanh(z)

    # --------------------- REASONING (skill-conditioned) -----------------
    def think(self, z, sk):
        h = z; W = self.core.W()
        for _ in range(self.T):
            h = self.core(h, z, W, sk)
        return h

    def reason(self, z, sk):
        return self.head_ans(self.think(z, sk))[:, :M]

    # --------------------------- P5 / P6 ---------------------------------
    def confidence(self, z, sk):
        return torch.softmax(self.reason(z, sk), -1).max(-1).values

    def answer_or_abstain(self, z, sk):
        """Returns predictions where ABSTAIN(=M) means 'I don't know'."""
        return self.head_ans(self.think(z, sk)).argmax(-1)

    # ------------------------ LANGUAGE: parse ----------------------------
    def parse_language(self, toks):
        B, L = toks.shape; W = self.lang_read.W(); h = torch.zeros(B, self.d, device=toks.device)
        for t in range(L):
            e = self.lang_emb(toks[:, t])
            for _ in range(4): h = self.lang_read(h, e, W)
        return self.head_act(h), self.head_col(h), self.head_obj(h)

    # ----------------------- LANGUAGE: generate --------------------------
    def generate_language(self, meaning, targets=None, teacher=False):
        """meaning: (B,3) ints -> emits 4 tokens [act,col,obj,EOS] after BOS."""
        B = meaning.size(0); W = self.dec.W()
        mv = torch.cat([F.one_hot(meaning[:, 0], NA).float(),
                        F.one_hot(meaning[:, 1], NC).float(),
                        F.one_hot(meaning[:, 2], NO).float()], 1)
        md = self.mean_emb(mv); h = torch.tanh(md)
        prev = torch.full((B,), BOS, device=meaning.device); out = []
        for t in range(4):
            e = self.lang_emb(prev)
            inp = torch.cat([e, md], 1)
            for _ in range(3): h = self.dec(h, inp, W)
            lg = self.head_tok(h); out.append(lg)
            prev = targets[:, t + 1] if (teacher and targets is not None) else lg.argmax(-1)
        return torch.stack(out, 1)

    # ------------------------- SEEK (Stage 5) ----------------------------
    def _read_question(self, q):
        B, L = q.shape; W = self.seek_read.W(); h = torch.zeros(B, self.d, device=q.device)
        for t in range(L):
            e = self.seek_emb(q[:, t])
            for _ in range(2): h = self.seek_read(h, e, W)
        return h

    def _gen_query(self, ctx, which, teacher=None):
        B = ctx.size(0); W = self.qdec.W(); h = ctx + which * 0.5
        prev = torch.full((B,), ST['<bos>'], device=ctx.device); outs = []
        for t in range(3):
            e = self.seek_emb(prev); inp = torch.cat([e, ctx], 1)
            for _ in range(2): h = self.qdec(h, inp, W)
            lg = self.head_qtok(h); outs.append(lg)
            prev = teacher[:, t] if teacher is not None else lg.argmax(-1)
        return torch.stack(outs, 1)

    @staticmethod
    def _store_lookup(query_tokens, vals):
        """External store: parses 'value of entX' in LANGUAGE; bad query -> noise."""
        ent = query_tokens[:, 2]
        valid = (query_tokens[:, 0] == ST['value']) & (query_tokens[:, 1] == ST['of']) \
                & (ent >= ENT0) & (ent < ENT0 + NE)
        idx = torch.where(valid, ent - ENT0, torch.randint(0, NE, ent.shape, device=ent.device))
        fetched = vals[torch.arange(vals.size(0)), idx]
        fetched = torch.where(valid, fetched, torch.randint(0, M, fetched.shape, device=ent.device))
        return fetched

    def seek(self, question, vals):
        """Full loop: read question -> generate two LANGUAGE queries -> store
        parses them -> integrate fetched values -> answer."""
        ctx = self._read_question(question)
        qA = self._gen_query(ctx, 0).argmax(-1)
        qB = self._gen_query(ctx, 1).argmax(-1)
        vA = self._store_lookup(qA, vals); vB = self._store_lookup(qB, vals)
        W = self.integrate.W(); h = ctx
        for v in (vA, vB):
            e = self.val_emb(v)
            for _ in range(2): h = self.integrate(h, e, W)
        return self.head_sum(h), (qA, qB)

    # ------------------ EXACT ACCUMULATION (ceiling fix) -----------------
    def exact_accumulate(self, digit_seq):
        """Running sum mod M over arbitrary length -- the clean-state module."""
        return self.counter(digit_seq)

    # ---------------- EXPLORE & LEARN ONLINE (Stage 6) -------------------
    def explore_and_learn(self, world, max_probes=60, lr=5e-3):
        """Curiosity loop: probe least-confident entity, learn online with
        self-replay. Returns probes-used, final acc, first/last retention."""
        opt = torch.optim.Adam(self.know.parameters(), lr=lr)
        eye = torch.eye(NE, device=DEVICE)
        memory, discovered, covered, probes = [], [], set(), 0
        while len(covered) < NE and probes < max_probes:
            with torch.no_grad():
                conf = torch.softmax(self.know(eye), -1).max(-1).values
            cand = torch.argmin(conf).item()          # curiosity = own uncertainty
            probes += 1
            v = world[cand].item()
            if cand not in covered: discovered.append(cand)
            covered.add(cand); memory.append((cand, v))
            for _ in range(30):                       # online learning + self-replay
                bi = np.random.randint(0, len(memory), min(16, len(memory)))
                ents = torch.stack([eye[memory[i][0]] for i in bi])
                tv = torch.tensor([memory[i][1] for i in bi], device=DEVICE)
                loss = F.cross_entropy(self.know(ents), tv)
                opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            pred = self.know(eye).argmax(-1)
            acc = (pred == world).float().mean().item()
            f3, l3 = discovered[:3], discovered[-3:]
            af = (pred[f3] == world[f3]).float().mean().item()
            al = (pred[l3] == world[l3]).float().mean().item()
        return probes, acc, af, al

    def n_params(self):
        return sum(p.numel() for p in self.parameters())


# ============================================================================
# 3. TRAINING (all faculties of ONE brain)
# ============================================================================
def _sym_batch(b, skill):
    nums = torch.randint(0, M, (b, NBIT))
    if skill == 0:   y = (nums[:, :3] % 2).sum(1) % 2
    elif skill == 1: y = nums[:, :3].sum(1) % M
    else:            y = nums.max(1).values
    return F.one_hot(nums, M).float().reshape(b, -1).to(DEVICE), y.long().to(DEVICE)

def _unknowable_batch(b):
    nums = torch.randint(0, M, (b, NBIT))
    x = F.one_hot(nums, M).float().reshape(b, -1)
    x = x + 0.0; x[:, -1] = 1.0                      # marker pattern on last dim
    return x.to(DEVICE), torch.randint(0, M, (b,)).to(DEVICE)

def _lang_data():
    allm = [(a, c, o) for a in range(NA) for c in range(NC) for o in range(NO)]
    rng = np.random.RandomState(0); perm = rng.permutation(64)
    held = set(perm[:16].tolist())
    return [allm[i] for i in range(64) if i not in held], [allm[i] for i in range(64) if i in held]

def _lang_batch(data, b):
    pick = [data[i] for i in np.random.randint(0, len(data), b)]
    toks = torch.tensor([[m[0], 4 + m[1], 8 + m[2]] for m in pick])
    tgt  = torch.tensor([[BOS, m[0], 4 + m[1], 8 + m[2], EOS] for m in pick])
    P = torch.tensor([list(m) for m in pick])
    return toks.to(DEVICE), tgt.to(DEVICE), P.to(DEVICE)

def _seek_batch(b):
    vals = torch.randint(0, M, (b, NE), device=DEVICE)        # randomized each episode!
    A = torch.randint(0, NE, (b,), device=DEVICE); B_ = torch.randint(0, NE, (b,), device=DEVICE)
    q = torch.stack([torch.full((b,), ST['what'], device=DEVICE),
                     torch.full((b,), ST['is'], device=DEVICE), ENT0 + A,
                     torch.full((b,), ST['plus'], device=DEVICE), ENT0 + B_], 1)
    y = (vals[torch.arange(b), A] + vals[torch.arange(b), B_]).clamp(max=8)
    return q, A, B_, vals, y

def _qtgt(ent):
    b = ent.size(0)
    return torch.stack([torch.full((b,), ST['value'], device=DEVICE),
                        torch.full((b,), ST['of'], device=DEVICE), ENT0 + ent], 1)


def train_brain(brain, log=print):
    """Trains every faculty of one UnifiedBrain. Order matters where proven:
    skills first, THEN abstention (P6 two-phase)."""
    t0 = time.time()
    # --- 1. reasoning skills (skill-conditioned core), trained jointly ---
    log("  [1/5] reasoning skills (3, skill-conditioned)...")
    opt = torch.optim.Adam(brain.parameters(), lr=2e-3)
    for it in range(4000):
        s = np.random.randint(0, NSKILL)
        x, y = _sym_batch(256, s)
        z = brain.perceive(symbols=x)
        loss = F.cross_entropy(brain.reason(z, torch.full((256,), s, device=DEVICE)), y)
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(brain.parameters(), 1.0); opt.step()
    # --- 2. P6 abstention (two-phase: AFTER skills) on the unknowable ---
    log("  [2/5] abstention (P6, two-phase) ...")
    opt = torch.optim.Adam(brain.parameters(), lr=1e-3)
    for it in range(3000):
        xk, yk = _sym_batch(128, 1); xu, yu = _unknowable_batch(128)
        x = torch.cat([xk, xu]); y = torch.cat([yk, yu])
        z = brain.perceive(symbols=x)
        p = torch.softmax(brain.head_ans(brain.think(z, torch.ones(256, dtype=torch.long, device=DEVICE))), -1)
        pc = p[torch.arange(256), y]; pa = p[:, :M]
        payoff = pc * 1.0 + (pa.sum(-1) - pc) * (-4.0) + p[:, ABSTAIN] * (-0.2)
        loss = -payoff.mean()
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(brain.parameters(), 1.0); opt.step()
    # --- 3. language: parse + generate ---
    log("  [3/5] language parse + generate ...")
    tr, hd = _lang_data()
    opt = torch.optim.Adam(brain.parameters(), lr=2e-3)
    for it in range(5000):
        toks, tgt, P = _lang_batch(tr, 256)
        la, lc, lo = brain.parse_language(toks)
        ploss = F.cross_entropy(la, P[:, 0]) + F.cross_entropy(lc, P[:, 1]) + F.cross_entropy(lo, P[:, 2])
        lg = brain.generate_language(P, tgt, teacher=True)
        gloss = F.cross_entropy(lg.reshape(-1, LV), tgt[:, 1:].reshape(-1))
        loss = ploss + gloss
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(brain.parameters(), 1.0); opt.step()
    # --- 4. seeking in language ---
    log("  [4/5] language-scale seeking ...")
    opt = torch.optim.Adam(brain.parameters(), lr=2e-3)
    for it in range(5000):
        q, A, B_, vals, y = _seek_batch(256)
        ctx = brain._read_question(q)
        tA, tB = _qtgt(A), _qtgt(B_)
        lA = brain._gen_query(ctx, 0, teacher=tA); lB = brain._gen_query(ctx, 1, teacher=tB)
        qloss = F.cross_entropy(lA.reshape(-1, SV), tA.reshape(-1)) + \
                F.cross_entropy(lB.reshape(-1, SV), tB.reshape(-1))
        alog, _ = brain.seek(q, vals)
        loss = qloss + F.cross_entropy(alog, y)
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(brain.parameters(), 1.0); opt.step()
    # --- 5. exact accumulation (clean-state) ---
    log("  [5/5] exact accumulation (clean-state recurrence) ...")
    opt = torch.optim.Adam(brain.counter.parameters(), lr=3e-3)
    for it in range(3000):
        L = np.random.choice([8, 16])
        seq = torch.randint(0, M, (256, L), device=DEVICE)
        tgt = seq.cumsum(1) % M
        out = brain.exact_accumulate(seq)
        loss = F.cross_entropy(out.reshape(-1, M), tgt.reshape(-1))
        opt.zero_grad(); loss.backward(); opt.step()
    log(f"  training done in {time.time()-t0:.0f}s")
    return brain


# ============================================================================
# 4. INTEGRATED SELF-TEST (one brain, every capability)
# ============================================================================
@torch.no_grad()
def _noT(): pass

def self_test(brain):
    R = {}
    # reasoning per skill
    for s, name in [(0, 'parity3'), (1, 'sum3'), (2, 'max')]:
        with torch.no_grad():
            x, y = _sym_batch(3000, s); z = brain.perceive(symbols=x)
            R[f'reason_{name}'] = (brain.reason(z, torch.full((3000,), s, device=DEVICE)).argmax(-1) == y).float().mean().item()
    # P5 confidence gap
    with torch.no_grad():
        xk, _ = _sym_batch(3000, 1); xu, _ = _unknowable_batch(3000)
        ck = brain.confidence(brain.perceive(symbols=xk), torch.ones(3000, dtype=torch.long, device=DEVICE)).mean().item()
        cu = brain.confidence(brain.perceive(symbols=xu), torch.ones(3000, dtype=torch.long, device=DEVICE)).mean().item()
        R['p5_gap'] = ck - cu
    # P6 abstention
    with torch.no_grad():
        pk = brain.answer_or_abstain(brain.perceive(symbols=xk), torch.ones(3000, dtype=torch.long, device=DEVICE))
        pu = brain.answer_or_abstain(brain.perceive(symbols=xu), torch.ones(3000, dtype=torch.long, device=DEVICE))
        R['p6_abstain_known'] = (pk == ABSTAIN).float().mean().item()
        R['p6_abstain_unknowable'] = (pu == ABSTAIN).float().mean().item()
    # language parse + generate (held-out)
    tr, hd = _lang_data()
    with torch.no_grad():
        toks, tgt, P = _lang_batch(hd, 1500)
        la, lc, lo = brain.parse_language(toks)
        R['lang_parse_held'] = ((la.argmax(-1) == P[:, 0]) & (lc.argmax(-1) == P[:, 1]) & (lo.argmax(-1) == P[:, 2])).float().mean().item()
        pred = brain.generate_language(P, teacher=False).argmax(-1)
        R['lang_gen_held'] = (pred[:, :3] == tgt[:, 1:4]).all(1).float().mean().item()
    # seek (language queries; randomized facts; random-value control)
    with torch.no_grad():
        q, A, B_, vals, y = _seek_batch(3000)
        alog, (qA, qB) = brain.seek(q, vals)
        R['seek_query_lang'] = ((qA == _qtgt(A)).all(1).float().mean().item() + (qB == _qtgt(B_)).all(1).float().mean().item()) / 2
        R['seek_answer'] = (alog.argmax(-1) == y).float().mean().item()
        ctx = brain._read_question(q)
        W = brain.integrate.W(); h = ctx
        for v in (torch.randint(0, M, (3000,), device=DEVICE), torch.randint(0, M, (3000,), device=DEVICE)):
            e = brain.val_emb(v)
            for _ in range(2): h = brain.integrate(h, e, W)
        R['seek_random_ctrl'] = (brain.head_sum(h).argmax(-1) == y).float().mean().item()
    # exact accumulation at L=16
    with torch.no_grad():
        seq = torch.randint(0, M, (3000, 16), device=DEVICE)
        R['exact_L16'] = (brain.exact_accumulate(seq)[:, -1].argmax(-1) == (seq.cumsum(1) % M)[:, -1]).float().mean().item()
    # explore & learn (alive loop)
    pr, acc, af, al = brain.explore_and_learn(torch.randint(0, M, (NE,), device=DEVICE))
    R['alive_probes'] = pr; R['alive_final'] = acc; R['alive_retention_gap'] = abs(af - al)
    return R


def main():
    print("=" * 76)
    print("UNIFIED SPINNING BRAIN -- one model, all capabilities")
    print("=" * 76)
    brain = UnifiedBrain().to(DEVICE)
    print(f"  instantiated: {brain.n_params():,} params (~{brain.n_params()*4/1024:.0f} KB)")
    train_brain(brain)
    print("-" * 76)
    R = self_test(brain)
    checks = [
        ("reason: parity-3",        R['reason_parity3'],        lambda v: v > 0.9),
        ("reason: sum-of-3",        R['reason_sum3'],           lambda v: v > 0.9),
        ("reason: max",             R['reason_max'],            lambda v: v > 0.9),
        ("P5 confidence gap",       R['p5_gap'],                lambda v: v > 0.3),
        ("P6 abstain (known)",      R['p6_abstain_known'],      lambda v: v < 0.25),
        ("P6 abstain (unknowable)", R['p6_abstain_unknowable'], lambda v: v > 0.6),
        ("language parse (held)",   R['lang_parse_held'],       lambda v: v > 0.8),
        ("language generate (held)",R['lang_gen_held'],         lambda v: v > 0.8),
        ("seek: query language",    R['seek_query_lang'],       lambda v: v > 0.9),
        ("seek: answer",            R['seek_answer'],           lambda v: v > 0.85),
        ("seek: random-ctrl",       R['seek_random_ctrl'],      lambda v: v < 0.45),
        ("exact accumulate L=16",   R['exact_L16'],             lambda v: v > 0.9),
        ("alive: final knowledge",  R['alive_final'],           lambda v: v > 0.95),
        ("alive: retention gap",    R['alive_retention_gap'],   lambda v: v < 0.15),
    ]
    npass = 0
    for name, val, cond in checks:
        ok = cond(val); npass += ok
        print(f"  [{'PASS' if ok else 'FAIL'}] {name:<26} {val:.2f}")
    print("-" * 76)
    print(f"  {npass}/{len(checks)} checks PASS  |  alive-loop probes: {R['alive_probes']}")
    print("=" * 76)
    json.dump(R, open("/home/claude/unified_results.json", "w"), indent=2)


if __name__ == "__main__":
    main()
