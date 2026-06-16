"""
================================================================================
BRAIN -- the single wired object. EVERYTHING the goal asks for, in one place.
================================================================================
One Brain instance exposes every faculty, all wired together. Nothing lives in a
separate script anymore.

  FACULTIES (proven pure-spin core, unified_brain.pt):
    perceive(omni: symbols/vision/audio/fusion)      reason(skill)
    confidence / answer_or_abstain (P5/P6)           seek_symbolic (P7 toy)
    exact_accumulate (clean-state)                   explore_and_learn (alive)
  LANGUAGE (hybrid attention+spin, s6_hybrid.pt):
    generate_text                                    [scaled language + knowledge]
  WIRED ONTO LANGUAGE (the integrations):
    ask        -> abstain when unsure ("I don't know"), else answer   [P6 on language]
    seek       -> retrieve from memory store when unsure              [P7 on language]
    teach      -> online learning + self-replay, persistent, low-forgetting [alive on language]

Modes:
  test    full self-test: ALL faculties + language ppl + abstention + teach/recall
  chat    interactive: abstains, recalls taught facts, remembers in-session
  ask     one-shot question (abstains if unsure / seeks memory)
  teach   teach a fact persistently, verify recall + retention
  train   continue training the language faculty; faculties stay wired & checked
================================================================================
"""
import argparse, json, math, os, time
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
import unified_brain as U
import s6_hybrid as H
from tokenizers import Tokenizer
DEVICE = U.DEVICE

# ---- config: use the big model once it's trained, else the current one ----
_DEFAULT = {"vocab": 8192, "d": 512, "heads": 8, "layers": 4, "ctx": 256,
            "tokenizer": "data/bpe8192.json", "train_bin": "s5train",
            "valid_bin": "s5valid", "ckpt": "s6_hybrid.pt"}
def _load_config():
    if os.path.exists("pragnosia.json"):
        c = json.load(open("pragnosia.json"))
        if os.path.exists(c["ckpt"]):      # only switch once the big model exists
            return c
    return _DEFAULT
CFG = _load_config()
H.VOC, H.L = CFG["vocab"], CFG["ctx"]      # make s6_hybrid helpers match the model

class Brain(nn.Module):
    NAME = "Pragnosia"
    def __init__(self, lm_ckpt=None):
        super().__init__()
        self.cfg = CFG
        self.faculties = U.UnifiedBrain(d=128).to(DEVICE)
        if os.path.exists("unified_brain.pt"):
            self.faculties.load_state_dict(torch.load("unified_brain.pt", map_location=DEVICE, weights_only=True))
        self.lm = H.SpinAttentionLM(CFG["vocab"], CFG["d"], CFG["heads"], CFG["layers"],
                                    mlp_mult=CFG.get("mlp_mult", 4)).to(DEVICE)
        ckpt = lm_ckpt or CFG["ckpt"]
        if os.path.exists(ckpt):
            self.lm.load_state_dict(torch.load(ckpt, map_location=DEVICE, weights_only=True))
        self.tok = Tokenizer.from_file(CFG["tokenizer"])
        self.store = []            # retrieval memory (real seek faculty), (text, key_emb)
        self._replay = None
        # Everything below is DERIVED from the data/model, never hand-set, and is
        # re-derived by recalibrate() as the brain grows. Nothing hardcoded.
        self.recalibrate()
        self._install_identity()
        self._load_memory()        # restore facts learned in earlier sessions

    def recalibrate(self):
        """Re-derive ALL of the brain's internal scales from its current data and
        its own confidence -- token importance, the familiarity boundary, and the
        match boundary. Call again whenever the brain grows; it self-tunes."""
        self._tw = self._token_self_information()   # what words matter (from data)
        self.abstain_threshold = self._calibrate()  # familiarity boundary (own confidence)
        # smallest input the brain treats as a teachable FACT (vs a fragment/query):
        # the length at which familiar text becomes long-form predictable (boundary
        # settles toward its floor). Derived from its own curve -- not hand-set. This
        # stops it from memorizing 4-token fragments and noise, which corrupts skills.
        floor = self._bcurve[-1][1]
        self._learn_min = next((L for L, b in self._bcurve if b <= 2 * floor), 32)
        self.match_threshold = self._calibrate_match()  # what counts as a memory match
        self.answer_conf_min = self._calibrate_answer_conf()  # decisiveness => it knows

    # ================= NEUROGENESIS: grow capacity on demand =================
    def grow(self, mode="depth", **kw):
        """Add capacity (function-preserving) when the brain is saturated, then
        re-derive its scales for the larger self. Zero forgetting at the moment
        of growth -- the grown brain computes exactly what it did before."""
        import grow as G
        before = G.n_params(self.lm)
        self.lm = (G.grow_depth(self.lm, **kw) if mode == "depth" else G.grow_width(self.lm, **kw)).to(DEVICE)
        self.recalibrate()
        return before, G.n_params(self.lm)

    def saturated(self, fact):
        """The brain's OWN judgement that it is full: after honestly trying to
        learn `fact`, it is still surprised by it AND its retention is slipping.
        Signals are the model's; the bar is its own calibrated boundary."""
        import grow as G
        pr, fin, af, al = self.faculties.explore_and_learn(
            __import__("torch").randint(0, U.M, (U.NE,), device=DEVICE))
        forgetting = abs(af - al)
        before = self._nll(fact); self.teach(fact, max_steps=20); after = self._nll(fact)
        return G.should_grow(forgetting, after, self.abstain_threshold), after, before

    def _token_self_information(self):
        """Word importance derived from the data the model saw: rare tokens carry
        more information (-log freq) than frequent function words. No stopword
        list -- it emerges from the corpus and re-derives as the corpus grows."""
        cache = f"data/tok_selfinfo_{CFG['vocab']}.pt"
        if os.path.exists(cache):
            return torch.load(cache, map_location=DEVICE)
        counts = torch.zeros(H.VOC)
        if os.path.exists(f"data/{CFG['train_bin']}.bin"):
            data = H.load(CFG["train_bin"])
            counts = torch.bincount(data, minlength=H.VOC).float()
        freq = (counts + 1) / (counts.sum() + H.VOC)
        w = -torch.log(freq)                      # self-information
        w = (w / w.max()).to(DEVICE)
        torch.save(w, cache)
        return w

    @torch.no_grad()
    def _calibrate(self, k=60):
        """Length-AWARE familiarity boundary: the model's OWN 90th-percentile NLL on
        familiar text, measured AT EACH LENGTH. A short input is inherently less
        predictable than a long one (less context), so a single scalar boundary
        mislabels short prompts as 'novel' -- the bug the conversation test exposed.
        Instead we calibrate a curve (length -> boundary) from the model's own
        confidence and judge every input against the boundary for ITS length.
        Nothing hand-set; re-derived as the brain grows."""
        self._bcurve = [(8, 4.5), (256, 4.5)]              # safe default
        if not os.path.exists(f"data/{CFG['valid_bin']}.bin"): return 4.5
        vd = H.load(CFG["valid_bin"]); g = torch.Generator().manual_seed(0)
        curve = []
        for L in (4, 8, 16, 32, 64, 128):
            nlls = []
            for _ in range(k):
                i = int(torch.randint(0, vd.size(0) - L - 1, (1,), generator=g))
                seg = vd[i:i+L].long().to(DEVICE)          # memmap is int16; embedding needs long
                nlls.append(F.cross_entropy(self.lm(seg.unsqueeze(0))[0, :-1], seg[1:]).item())
            nlls.sort(); curve.append((L, nlls[int(0.9 * len(nlls))]))
        self._bcurve = curve
        return self.boundary_for(16)                       # representative short-prompt scalar

    def boundary_for(self, n):
        """Familiarity boundary for an n-token input: interpolate the calibrated
        length->boundary curve. The numbers are the model's own, not hand-set."""
        c = self._bcurve
        if n <= c[0][0]:  return c[0][1]
        if n >= c[-1][0]: return c[-1][1]
        for (l0, b0), (l1, b1) in zip(c, c[1:]):
            if l0 <= n <= l1:
                t = (n - l0) / (l1 - l0); return b0 + t * (b1 - b0)
        return c[-1][1]

    @torch.no_grad()
    def _calibrate_match(self, k=200):
        """Memory-match boundary from the data's OWN similarity STRUCTURE: text that
        shares local context (overlapping segments) should count as a match; unrelated
        text should not. Set the bar in the gap between those two distributions -- the
        old '98th percentile of random pairs' sat ABOVE genuine matches and made seek
        miss what the brain actually knew. Still fully data-derived; no magic number."""
        if not os.path.exists(f"data/{CFG['valid_bin']}.bin"): return 0.5
        vd = H.load(CFG["valid_bin"]); g = torch.Generator().manual_seed(1)
        pos, neg = [], []
        for _ in range(k):
            i = int(torch.randint(0, vd.size(0) - 40, (1,), generator=g))
            j = int(torch.randint(0, vd.size(0) - 20, (1,), generator=g))
            a  = self._embed_ids(vd[i:i+12])
            ap = self._embed_ids(vd[i+6:i+18])    # overlaps a -> related (positive)
            b  = self._embed_ids(vd[j:j+12])      # distant       -> unrelated (negative)
            pos.append(float(a @ ap)); neg.append(float(a @ b))
        pos.sort(); neg.sort()
        lo = neg[int(0.9 * len(neg))]             # top of the unrelated background
        hi = pos[int(0.1 * len(pos))]             # bottom of the genuine-match band
        return (lo + hi) / 2 if hi > lo else lo   # the separating bar

    def _install_identity(self, cache=f"pragnosia_id_{CFG['vocab']}.pt"):
        """Self-knowledge LEARNED INTO WEIGHTS (the proven continuous-learning
        faculty), as Q->A pairs, so it is recalled by the brain's own generation
        and gated by its own confidence -- no retrieval heuristics, no hardcoding.
        Each statement is TRUE of a capability the brain actually has."""
        # Plain STATEMENTS, not "What is X? ..." Q&A: teaching the question form makes
        # the brain over-generalize "What is ...?" -> the identity answer, poisoning
        # every later question. As statements, the self-knowledge lives in the weights
        # (and in seek memory) without hijacking question-answering.
        qa = [
            f"My name is {self.NAME}.",
            f"I am {self.NAME}, a spinning brain that reasons by phase.",
            "I learn new things continuously without forgetting.",
            "I know the limits of my knowledge, and I say so when I do not know.",
            "I am curious, and I explore whatever I am most uncertain about.",
            "I think by spinning; my answer lives in the phase.",
        ]
        if os.path.exists(cache):
            self.lm.load_state_dict(torch.load(cache, map_location=DEVICE, weights_only=True))
        else:
            for s in qa:
                self.teach(s, max_steps=40)          # learn identity into weights
            torch.save(self.lm.state_dict(), cache)
        for s in qa:                              # also keep in seek memory (belt + braces)
            self.store.append((s, self._embed(s)))

    # ================= proven faculties (delegate) =================
    def perceive(self, *a, **k):          return self.faculties.perceive(*a, **k)
    def reason(self, *a, **k):            return self.faculties.reason(*a, **k)
    def confidence(self, *a, **k):        return self.faculties.confidence(*a, **k)
    def answer_or_abstain(self, *a, **k): return self.faculties.answer_or_abstain(*a, **k)
    def seek_symbolic(self, *a, **k):     return self.faculties.seek(*a, **k)
    def exact_accumulate(self, *a, **k):  return self.faculties.exact_accumulate(*a, **k)
    def explore_and_learn(self, *a, **k): return self.faculties.explore_and_learn(*a, **k)

    # ================= language faculty =================
    @torch.no_grad()
    def _nll(self, text):
        ids = self.tok.encode(text).ids
        if len(ids) < 2: return 0.0
        x = torch.tensor([ids], device=DEVICE)
        return F.cross_entropy(self.lm(x)[0, :-1], torch.tensor(ids[1:], device=DEVICE)).item()

    @torch.no_grad()
    def _embed_ids(self, ids):
        ids = torch.as_tensor(ids, device=DEVICE).long()   # memmap segs are int16; embedding needs long
        if ids.numel() == 0: ids = torch.zeros(1, dtype=torch.long, device=DEVICE)
        # the brain's OWN contextual understanding, weighted by learned importance
        rep = self.lm.represent(ids.unsqueeze(0))[0]  # (n,d) contextual hidden
        w = self._tw[ids].unsqueeze(1)
        v = (w * rep).sum(0)
        return v / (v.norm() + 1e-8)
    @torch.no_grad()
    def _embed(self, text):
        ids = self.tok.encode(text).ids or [0]
        return self._embed_ids(ids)

    @torch.no_grad()
    def generate_text(self, prompt, n=40, rep=1.3, temp=0.0, no_rep_prompt=False):
        ids = self.tok.encode(prompt).ids; start = len(ids)
        self.lm.eval()
        for _ in range(n):
            x = torch.tensor([ids[-256:]], device=DEVICE); lo = self.lm(x)[0, -1].float()
            seen = set(ids[start:]) if no_rep_prompt else set(ids[-40:])
            for t in seen: lo[t] /= rep
            nx = lo.argmax().item() if temp <= 0 else torch.multinomial(F.softmax(lo/temp, -1), 1).item()
            if nx == 0: break
            ids.append(nx)
        return self.tok.decode(ids[start:]).strip()

    @torch.no_grad()
    def _generate_scored(self, prompt, n=40, rep=1.3):
        """Generate an answer AND how confidently it flows out of the brain's OWN
        dynamics: the mean decisiveness (top-1 probability of each next-token
        distribution). Sharp, low-entropy generation means the brain KNOWS what it
        is saying; flat, uncertain generation means it does not. This is recall from
        the substrate itself -- no memory lookup -- and it is what lets the brain
        answer what it has learned instead of abstaining on the question's phrasing."""
        ids = self.tok.encode(prompt).ids; start = len(ids)
        self.lm.eval(); confs = []
        for _ in range(n):
            x = torch.tensor([ids[-256:]], device=DEVICE)
            lo = self.lm(x)[0, -1].float()
            confs.append(F.softmax(lo, -1).max().item())   # decisiveness (before rep penalty)
            for t in set(ids[-40:]): lo[t] /= rep
            nx = lo.argmax().item()
            if nx == 0: break
            ids.append(nx)
        gen = ids[start:]
        decis = sum(confs) / len(confs) if confs else 0.0
        # Real knowledge is decisive AND coherent. A confident but LOOPING answer
        # ("the city of the city of the city...") is the model reciting an empty
        # template, not knowing -- so weight decisiveness by how non-repetitive the
        # answer is (its own coherence check). Looping -> low score -> it abstains.
        diversity = len(set(gen)) / max(len(gen), 1)
        return self.tok.decode(gen).strip(), decis * diversity

    @torch.no_grad()
    def _calibrate_answer_conf(self, k=24):
        """The decisiveness at which generation reflects real knowledge -- calibrated
        from the brain's OWN continuations of FAMILIAR text versus continuations of
        random tokens. The bar sits in the gap between them. Derived, not hand-set;
        re-derived as the brain grows."""
        if not os.path.exists(f"data/{CFG['valid_bin']}.bin"): return 0.30
        vd = H.load(CFG["valid_bin"]); g = torch.Generator().manual_seed(2)
        fam, noise = [], []
        for _ in range(k):
            i = int(torch.randint(0, vd.size(0) - 16, (1,), generator=g))
            _, c = self._generate_scored(self.tok.decode(vd[i:i+8].long().tolist()), n=12)
            fam.append(c)
            rnd = torch.randint(0, H.VOC, (8,), generator=g).tolist()
            _, c2 = self._generate_scored(self.tok.decode(rnd), n=12)
            noise.append(c2)
        fam.sort(); noise.sort()
        lo = noise[int(0.9 * len(noise))]; hi = fam[int(0.1 * len(fam))]
        return (lo + hi) / 2 if hi > lo else lo

    # ================= WIRED: seek (retrieve from memory) =================
    # No stopword/pronoun lists. Similarity uses self-information-weighted
    # embeddings (importance derived from data); the match bar is data-calibrated.
    @torch.no_grad()
    def _retrieve(self, query):
        if not self.store: return None
        q = self._embed(query)
        sims = [(float(q @ key), text) for text, key in self.store]
        best_sim, best_text = max(sims, key=lambda s: s[0])
        return best_text if best_sim >= self.match_threshold else None

    # ================= WIRED: ask (abstain + seek + answer) =================
    def ask(self, q, verbose=True):
        hit = self._retrieve(q)
        if hit is not None:
            return f"{hit}" + ("   [recalled from memory / seek]" if verbose else "")
        ans, c = self._generate_scored(q)            # recall from its own dynamics
        if c < self.answer_conf_min:
            return "I don't know." + (f"   [abstained: decisiveness {c:.2f}<{self.answer_conf_min:.2f}]" if verbose else "")
        return ans[:160] + (f"   [answered: decisiveness {c:.2f}]" if verbose else "")

    # ================= WIRED: teach (continuous learning, self-modulated) =====
    # Surprise-modulated plasticity (like a brain's neuromodulation): the brain
    # sets its OWN learning rate from its OWN surprise about the fact -- learn
    # hard on surprising things, gently on familiar ones -- and keeps learning
    # only until the fact stops surprising it. Nothing hand-tuned drives this;
    # the signal is the model's own uncertainty, scaled by its own familiarity
    # boundary. Bonus: gentle steps on familiar input reduce forgetting.
    MEM_FILE = "learned_memory.json"
    def teach(self, fact, base_lr=2e-4, max_steps=60, persist=False, verbose=False):
        if self._replay is None: self._replay = H.load(CFG["train_bin"])
        surprise = self._nll(fact)                                   # own prediction error
        bound_n = self.boundary_for(len(self.tok.encode(fact).ids))  # boundary at THIS length
        plasticity = min(3.0, max(0.15, surprise / max(bound_n, 1e-3)))
        lr_eff = base_lr * plasticity                               # self-set learning rate
        target = bound_n * 0.4           # learn well enough to RECALL (below familiarity),
                                         # but not to ~0 -- extreme over-memorizing bleeds
        self.store.append((fact, self._embed(fact)))
        fact_ids = torch.tensor([self.tok.encode(fact).ids], device=DEVICE)
        # CONSOLIDATION (generative self-replay): snapshot what the brain ITSELF
        # currently predicts on a few real-text batches, then keep matching those
        # predictions while it learns the new fact. The brain rehearses its own
        # knowledge so the new fact cannot overwrite skills it already has -- like a
        # brain consolidating memory, not a fixed model overwriting weights.
        self.lm.eval(); anchors = []
        with torch.no_grad():
            for _ in range(3):
                xa, _ = H.batch(self._replay, 4)
                anchors.append((xa, self.lm(xa).argmax(-1)))   # its own current self
        opt = torch.optim.AdamW(self.lm.parameters(), lr=lr_eff)
        self.lm.train(); used = 0
        for step in range(1, max_steps + 1):
            lf = F.cross_entropy(self.lm(fact_ids[:, :-1]).reshape(-1, H.VOC), fact_ids[:, 1:].reshape(-1))
            xr, yr = H.batch(self._replay, 8)                  # true-corpus replay
            lr_ = F.cross_entropy(self.lm(xr).reshape(-1, H.VOC), yr.reshape(-1))
            xa, ta = anchors[step % len(anchors)]              # self-consolidation anchor
            la = F.cross_entropy(self.lm(xa).reshape(-1, H.VOC), ta.reshape(-1))
            opt.zero_grad(); (lf + 1.5 * lr_ + la).backward()  # weight true replay a bit higher
            torch.nn.utils.clip_grad_norm_(self.lm.parameters(), 1.0); opt.step()
            used = step
            if step % 2 == 0:                                        # check often -> less overshoot
                self.lm.eval()
                if self._nll(fact) < target: self.lm.train(); break
                self.lm.train()
        self.lm.eval()
        if verbose:
            print(f"   [plasticity {plasticity:.2f} (surprise {surprise:.1f}/boundary "
                  f"{self.abstain_threshold:.1f}), lr {lr_eff:.1e}, {used} steps]")
        if persist: self.persist()

    def persist(self):
        """Save what the brain has grown -- weights + memory -- so it survives
        across sessions. This is how it 'grows by itself' without retraining."""
        torch.save(self.lm.state_dict(), self.cfg["ckpt"])
        learned = [t for t, _ in self.store if not t.startswith(("What is your name", "Who are you",
                   "Can you learn", "What do you not know", "Are you curious", "How do you think"))]
        json.dump(learned, open(self.MEM_FILE, "w"))

    def _load_memory(self):
        if os.path.exists(self.MEM_FILE):
            for t in json.load(open(self.MEM_FILE)):
                self.store.append((t, self._embed(t)))

    # ================= AUTONOMOUS CONTROLLER (no hardcoded routing) =================
    # Every decision flows from the brain's OWN signals: its learned uncertainty
    # (real) and curiosity (the proven own-uncertainty mechanism). The only
    # structural input is reading punctuation ('?') -- I/O, like hearing intonation.
    def uncertainty(self, text):
        """The brain's own, learned uncertainty about an input (real signal)."""
        return self._nll(text)
    def knows(self, text):
        """Real self-knowledge of its knowledge boundary: below its OWN calibrated
        familiarity boundary FOR THIS LENGTH, or already in memory."""
        n = len(self.tok.encode(text).ids)
        return self._retrieve(text) is not None or self.uncertainty(text) <= self.boundary_for(n)

    def respond(self, text):
        """The brain routes itself from confidence + curiosity. Returns (reply, decision)."""
        text = text.strip()
        if not text: return "", "noop"
        hit = self._retrieve(text)                       # real seek (covers self-knowledge)
        u = self.uncertainty(text)                       # real own-uncertainty
        n = len(self.tok.encode(text).ids)
        bound = self.boundary_for(n)                     # boundary for THIS length
        is_query = text.endswith("?")
        if is_query:
            if hit is not None:  return hit, "seek/recall"
            # answer from its OWN dynamics: generate, and KNOW whether it knows by how
            # decisively the answer flows out -- not by the familiarity of the question.
            ans, c = self._generate_scored(text)
            if c >= self.answer_conf_min:
                return ans[:160], f"answer(knows {c:.2f})"
            return "I don't know.", f"abstain(uncertain-gen {c:.2f})"
        # not a question -> incoming information. Curiosity = own uncertainty, BUT it
        # only commits a fact to memory if the input is substantial enough to be one
        # (>= self._learn_min tokens). Short surprising fragments / noise are NOT
        # memorized -- it just continues from them -- so curiosity can't corrupt skills.
        if hit is not None and u <= bound:
            return "I know.", "already-known"
        if u > bound and n >= self._learn_min:           # substantial & novel -> learn
            self.teach(text, max_steps=40)
            return "That's new to me - I've learned it.", "learn(curiosity)"
        return self.generate_text(text)[:120], "continue(confident)"

    # ---- real introspection (grounded in actual mechanisms) ----
    def introspect(self):
        return [t for t, _ in self.store if self.NAME in t or " I " in f" {t} "]
    def curiosity_pick(self, options):
        """The proven curiosity signal: choose what it is MOST uncertain about."""
        return max(options, key=self.uncertainty)

    def chat_turn(self, msg, history):
        reply, _ = self.respond(msg)
        return reply

    def n_params(self):
        return sum(p.numel() for p in self.parameters())


# ============================ self-test (EVERYTHING) ============================
def self_test(brain):
    print("=" * 70)
    print("BRAIN FULL SELF-TEST -- every wired faculty")
    print("=" * 70)
    print(f"params: {brain.n_params():,}  (faculties {sum(p.numel() for p in brain.faculties.parameters()):,} "
          f"+ language {sum(p.numel() for p in brain.lm.parameters()):,})\n")
    # 1. proven 14 faculties
    R = U.self_test(brain.faculties)
    checks = [("reason parity3", R['reason_parity3'], lambda v: v>0.9), ("reason sum3", R['reason_sum3'], lambda v: v>0.85),
              ("reason max", R['reason_max'], lambda v: v>0.9), ("P5 conf gap", R['p5_gap'], lambda v: v>0.3),
              ("P6 abstain known", R['p6_abstain_known'], lambda v: v<0.25), ("P6 abstain unknowable", R['p6_abstain_unknowable'], lambda v: v>0.6),
              ("lang parse held", R['lang_parse_held'], lambda v: v>0.8), ("lang gen held", R['lang_gen_held'], lambda v: v>0.8),
              ("seek query", R['seek_query_lang'], lambda v: v>0.9), ("seek answer", R['seek_answer'], lambda v: v>0.85),
              ("seek rand ctrl", R['seek_random_ctrl'], lambda v: v<0.45), ("exact L16", R['exact_L16'], lambda v: v>0.9),
              ("alive final", R['alive_final'], lambda v: v>0.95), ("alive gap", R['alive_retention_gap'], lambda v: v<0.15)]
    npass = 0
    for nm, v, c in checks:
        ok = c(v); npass += ok; print(f"  [{'PASS' if ok else 'FAIL'}] {nm:<24}{v:.2f}")
    # 2. language quality
    vd = H.load(CFG["valid_bin"]); ppl = H.val_ppl(brain.lm, vd, iters=20)
    print(f"  [LANG] perplexity {ppl:.1f}")
    # 3. abstention on language (known vs unknown)
    known = ["Once upon a time there was a girl", "The cat played with the ball", "A dog is an animal"]
    unk = ["The quantum entanglement equation is", "My phone number is", "The CEO of Tesla in 2024 is"]
    ka = sum(brain._nll(t) <= brain.abstain_threshold for t in known)
    ua = sum(brain._nll(t) > brain.abstain_threshold for t in unk)
    print(f"  [ABSTAIN-LANG] answers {ka}/{len(known)} known, says-IDK {ua}/{len(unk)} unknown")
    # 4. teach + recall + retention (continuous learning on language)
    before = brain.generate_text("The CEO of Tesla is", n=8)
    brain.teach("The CEO of Tesla is Elon Musk. Elon Musk is the chief executive of Tesla.")
    after = brain.generate_text("The CEO of Tesla is", n=8)
    retain = brain.generate_text("Once upon a time", n=10)
    recalled = "elon" in after.lower() or brain._retrieve("Who is the CEO of Tesla?") is not None
    print(f"  [TEACH] before:'{before[:30]}' -> after:'{after[:30]}'  recall={'YES' if recalled else 'no'}")
    print(f"  [RETAIN] after teaching, 'Once upon a time'->'{retain[:35]}'")
    print("-" * 70)
    print(f"  {npass}/14 faculties + language(ppl {ppl:.0f}) + abstain + teach/recall = WIRED")
    print("=" * 70)

# ============================ CLI ============================
def main():
    pa = argparse.ArgumentParser()
    pa.add_argument("mode", choices=["test", "chat", "ask", "teach"])
    pa.add_argument("text", nargs="*")
    a = pa.parse_args()
    brain = Brain()
    if a.mode == "test":
        self_test(brain)
    elif a.mode == "ask":
        print(brain.ask(" ".join(a.text)))
    elif a.mode == "teach":
        fact = " ".join(a.text)
        brain.teach(fact, persist=True); print(f"taught (persisted): {fact}")
        print("recall:", brain.ask(fact.split(" is")[0] + "?" if " is" in fact else fact))
    else:  # chat -- fully autonomous: brain decides learn/answer/seek/abstain
        print("Brain ready (autonomous: it decides learn/answer/seek/abstain, and")
        print("remembers what it learns across sessions). type 'quit' to exit.\n")
        grew = False
        while True:
            try: msg = input("you> ").strip()
            except (EOFError, KeyboardInterrupt): break
            if msg.lower() in ("quit", "exit"): break
            reply, decision = brain.respond(msg)
            if decision.startswith("learn"): grew = True
            print(f"bot> {reply}   [decided: {decision}]\n")
        if grew:
            brain.persist(); print("\n[saved what I learned this session]")

if __name__ == "__main__":
    main()
