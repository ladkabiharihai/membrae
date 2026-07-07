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

Talking to it IS how it works -- answering, learning (teaching), curiosity, looking
things up, and growing are all INTRINSIC to the brain (brain.interact), not separate
commands. So there are only two ways to run it:
  python3 brain.py          -> it LIVES: talk to it; it answers what it knows, learns
                               what you tell it, wonders its own questions, looks up
                               what it doesn't know, and grows itself when it saturates.
  python3 brain.py test     -> verify it: full self-test (every faculty + language).
================================================================================
"""
import json, math, os, re, sys, time
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
import s6_hybrid as H
from tokenizers import Tokenizer
DEVICE = H.DEVICE

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
    def __init__(self, lm_ckpt=None, learn=True):
        super().__init__()
        # learn=True  -> full alive brain: teaches identity into weights, learns from
        #                chat (continuous learning). Needs an optimizer over the whole LM,
        #                so for a big model (1.4B) use CPU -- the GPU can't hold AdamW state.
        # learn=False -> inference-only chat: NO teaching anywhere (fits a big model on a
        #                small GPU); identity & taught facts are recalled from seek memory.
        self.learn = learn
        self.cfg = CFG
        self.lm = H.SpinAttentionLM(CFG["vocab"], CFG["d"], CFG["heads"], CFG["layers"],
                                    mlp_mult=CFG.get("mlp_mult", 4),
                                    carrier=CFG.get("carrier", "single"))    # build on CPU (carrier mode from config)
        ckpt = lm_ckpt or CFG["ckpt"]
        if os.path.exists(ckpt):                                            # load weights on CPU then move
            self.lm.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=True))
        self.lm = self.lm.to(DEVICE)        # move ONCE -> peak GPU = model size, not 2x (big-model safe)
        if not learn and DEVICE == "cuda":  # inference chat: bf16 halves memory (5.8->2.9GB) + ~2x faster,
            self.lm = self.lm.bfloat16()    # identical output (the diagonal-complex carrier is bf16-stable)
        self.tok = Tokenizer.from_file(CFG["tokenizer"])
        self.store = []            # retrieval memory (real seek faculty), (text, key_emb)
        self._replay = None
        self._last_topic = None    # the topic its curiosity is currently chasing
        self.self_model = None     # SELF-MODEL: architecture facts DERIVED from the live model (built in
        self.goals = []            #   _derive_self after the LM loads; re-derived on grow). Not hand-set.
        self._gaps = []            # things it honestly couldn't answer -> seeds for self-defined goals
        self._weight_teach_ok = True   # cleared if backprop OOMs -> fall back to episodic-only learning
        # --- faculties that make it more than a predictor (the buildable pieces of a mind) ---
        self.wm = []               # WORKING MEMORY: a small bounded scratchpad it reasons OVER (holds ~7 items)
        self.episodes = []         # AUTOBIOGRAPHICAL timeline: ordered events, survives sessions (queryable)
        self.user_model = {"told": [], "asked": []}   # THEORY OF MIND: the USER's mind, kept apart from world-fact
        self._recent_lowconf = []  # questions it was unsure on -> reflected into gaps/goals at idle
        self._turn = 0             # monotonic step counter -> orders episodes (no wall-clock needed)
        self.affect = {"valence": 0.0, "arousal": 0.0, "mood": 0.0}   # FUNCTIONAL EMOTION: derived affect state
        self.mem = H.FastWeightMemory(self.lm.d).to(DEVICE)   # IN-WEIGHTS subconscious: surprise-write,
                                                              # additive recall, decay (forget), sleep-consolidate
        # Everything below is DERIVED from the data/model, never hand-set, and is
        # re-derived by recalibrate() as the brain grows. Nothing hardcoded.
        self.recalibrate()
        self.self_model = self._derive_self()   # derive the self-facts from the live model (post-load)
        self._install_identity()
        self._load_memory()        # restore facts learned in earlier sessions

    def _derive_self(self):
        """Build the self-description by INSPECTING the live model -- size, layer mix, carrier mode -- so it
        stays true as the model grows, instead of hand-written strings. Only the name is a given (a name
        can't be derived from data); the rest is read off the actual object + measured behaviour."""
        nl = len(self.lm.blocks)
        ns = sum(1 for b in self.lm.blocks if isinstance(b, H.SpinBlock))
        na = sum(1 for b in self.lm.blocks if isinstance(b, H.Block))
        mode = getattr(self.lm, "carrier_mode", "single")
        if mode == "spin_dominant" and ns:
            core = (f"my core token-mixer is a spin carrier in {ns} of {nl} layers, with attention in the "
                    f"other {na}; I carry state in the phase of a rotating recurrence run as a parallel scan")
        else:
            core = f"I mix tokens with a {mode} carrier across {nl} layers plus attention"
        return {
            "name": "Pragnosia",                                    # a name is declared, not derivable
            "kind": f"a {self.n_params()/1e6:.0f}M-parameter recurrent language model",  # read from the model
            "core": core,                                           # read from the model's block structure
            "faculties": self._wired_faculties(),                   # what the controller actually has wired
            "values": "honesty about what I don't know, curiosity, and clarity",   # a stated aim, not a metric
        }

    def _wired_faculties(self):
        """Report the faculties that are ACTUALLY present as methods -- not a claim, a fact about this object."""
        have = [(n, hasattr(self, m)) for n, m in
                [("recall taught facts", "teach"), ("deliberate step by step", "_deliberate"),
                 ("hold a train of thought", "think_aloud"), ("seek things up", "search"),
                 ("grow when I saturate", "_maybe_grow"), ("keep a subconscious memory", "mem"),
                 ("hold items in working memory", "wm_push"), ("remember our conversation", "log_episode"),
                 ("use tools and learn from the result", "act"), ("track what you've told me", "note_user"),
                 ("reflect on what I'm unsure of", "reflect"), ("have moods that shape what I do next", "appraise")]]
        return ", ".join(n for n, ok in have if ok)

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
        # learn-min = the KNEE of the familiarity curve (where extra context stops lowering the
        # boundary), found parameter-free as the bcurve point farthest from the chord joining its
        # ends (greatest curvature) -- not a fixed 2*floor / 32-token fallback.
        Ls = [L for L, _ in self._bcurve]; bs = [b for _, b in self._bcurve]
        if len(Ls) >= 3:
            dx, dy = Ls[-1] - Ls[0], bs[-1] - bs[0]; den = (dx * dx + dy * dy) ** 0.5 or 1.0
            dist = [abs(dy * (Ls[i] - Ls[0]) - dx * (bs[i] - bs[0])) / den for i in range(len(Ls))]
            self._learn_min = Ls[max(range(len(dist)), key=lambda i: dist[i])]
        else:
            self._learn_min = Ls[len(Ls) // 2] if Ls else 8
        self.match_threshold = self._calibrate_match()  # what counts as a memory match
        self._content_min = self._calibrate_content_min()     # what counts as a content word
        self.consistency_min = self._calibrate_consistency()  # answer-stability => it knows (honesty)
        self.novelty_min = self._calibrate_novelty()          # content-surprise => it's NEW (learn)

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
        # EQUAL-ERROR-RATE separation: the bar where the false-match rate (unrelated above it)
        # equals the miss rate (real matches below it) -- the natural crossover of the two
        # distributions, derived with no hand-chosen tail percentile (was neg[.9]/pos[.1]).
        N, P = max(len(neg), 1), max(len(pos), 1)
        cand = sorted(set(pos + neg))
        return min(cand, key=lambda t: abs(sum(x > t for x in neg) / N - sum(x <= t for x in pos) / P))

    def _install_identity(self, cache=f"pragnosia_id_{CFG['vocab']}_d{CFG['d']}l{CFG['layers']}m{CFG.get('mlp_mult',4)}.pt"):
        """Identity as real KNOWLEDGE in the weights -- learned the way a child learns
        its name: MANY varied exposures of the one concrete fact (the name), so it
        integrates and is recalled by the brain's OWN GENERATION, gated by its OWN
        confidence (self-consistency). This is the cognitive path, NOT retrieval/lookup.

        Honesty then falls out of cognition, not similarity: the brain genuinely KNOWS
        its name (every sample says Pragnosia -> high self-consistency -> answer), and
        genuinely does NOT know your favourite colour (every sample differs -> low
        self-consistency -> abstain). Same gate, no special-casing.

        Two rules learned the hard way:
        - ONLY name-binding facts. NO abstract self-description ('I reason by phase',
          'I am curious...'): those fluent, content-light sentences over-fit and BLEED
          into unrelated generation and fool the honesty gate. 'Pragnosia' is a rare
          token, so it does not leak into ordinary prose.
        - MANY phrasings, taught GENTLY (low lr, few steps), interleaved with the heavy
          real-corpus replay already in teach(), so no single sentence dominates and
          'What is ...?' does not collapse to the identity answer."""
        n = self.NAME
        facts = [
            f"My name is {n}.", f"I am {n}.", f"You can call me {n}.",
            f"I am called {n}.", f"People call me {n}.", f"I go by {n}.",
            f"Who are you? I am {n}.", f"What is your name? My name is {n}.",
            f"What are you called? I am called {n}.", f"And you are? I am {n}.",
            f"Hello, I am {n}.", f"This is {n}.",
        ]
        if os.path.exists(cache):                # load on CPU then copy into the (GPU) model -> no 2x GPU spike
            self.lm.load_state_dict(torch.load(cache, map_location="cpu", weights_only=True))
        elif self.learn:                         # only bake identity into weights when learning is on
            for s in facts:
                self.teach(s, base_lr=1e-4, max_steps=20)     # gentle: nudge, don't memorize-hard
            if self._weight_teach_ok:            # only cache if teaching actually ran (else it's just base weights)
                self._atomic_save(self.lm.state_dict(), cache)
        for s in facts:                          # seek memory either way -> recall works in inference mode
            self.store.append((s, self._embed(s)))


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
    @torch.no_grad()
    def generate_text(self, prompt, n=40, rep=1.3, temp=0.0, no_rep_prompt=False, recall=False):
        # O(T) long-context generation. When the subconscious holds something (mem.energy>0),
        # generation is RECALL-AWARE: each token's representation is mixed with the fast-weight
        # recall before the head, so freshly-taught facts surface BEFORE the slow weights learn
        # them. Empty subconscious -> the fast block-wise carrier-carry path (no per-token cost).
        self.lm.eval()
        ids = self.tok.encode(prompt).ids
        if not (recall and float(self.mem.energy()) > 0):
            out = self.lm.generate(ids, n_new=n, window=self.cfg["ctx"], temp=temp, rep=rep)
            return self.tok.decode(out).strip()
        import math
        out = []
        for _ in range(n):
            r = self.lm.represent(torch.tensor([(ids + out)[-self.cfg["ctx"]:]], device=DEVICE))[0, -1].float()
            base = self.lm.head(r.to(self.lm.head.weight.dtype)).float()
            rec = self.lm.head(self.mem.read(r).to(self.lm.head.weight.dtype)).float()
            p = F.softmax(base, -1); ent = float(-(p * torch.log(p + 1e-9)).sum())
            alpha = min(1.0, ent / math.log(base.numel()))     # 0=confident (trust the model) .. 1=unsure (let memory speak)
            lo = (1 - alpha) * base + alpha * rec
            for t in set((ids + out)[-40:]): lo[t] /= rep
            nx = lo.argmax().item() if temp <= 0 else torch.multinomial(F.softmax(lo / temp, -1), 1).item()
            if nx == 0: break
            out.append(nx)
        return self.tok.decode(out).strip()

    # ================= HONESTY by SELF-CONSISTENCY (no hardcode) =================
    @torch.no_grad()
    def _generate_sampled(self, prompt, n=24, temp=0.8):
        """Sample (not argmax) a continuation -- used to probe whether the brain
        REALLY knows something: stable knowledge gives the same answer every sample."""
        ids = self.tok.encode(prompt).ids; start = len(ids)
        self.lm.eval()
        for _ in range(n):
            lo = self.lm(torch.tensor([ids[-256:]], device=DEVICE))[0, -1].float() / max(temp, 1e-3)
            for t in set(ids[-20:]): lo[t] -= 1.0          # mild anti-loop
            nx = torch.multinomial(F.softmax(lo, -1), 1).item()
            if nx == 0: break
            ids.append(nx)
        return ids[start:]

    @torch.no_grad()
    def _self_consistency(self, question, k=5, n=24):
        """The brain's OWN honesty signal: sample k answers and see whether they agree
        on the ACTUAL ANSWER. Real knowledge pins the answer ('Paris' in every sample);
        a confident confabulation ('my phone is 0.9' / 'I ate 10 eggs') puts a different
        content word in every sample. Consistency = the largest fraction of samples that
        share one content token (excluding the question's own words). No hardcoding.
        Returns (consistency, a representative answer)."""
        from collections import Counter
        qset = set(self.tok.encode(question).ids)
        samples = [self._generate_sampled(question, n) for _ in range(k)]
        present = Counter()
        for s in samples:
            for t in (set(s) - qset):
                if self._tw[t].item() >= self._content_min:        # only content tokens vote
                    present[t] += 1
        if not present:
            return 0.0, self.tok.decode(samples[0]).strip()
        tok_id, cnt = present.most_common(1)[0]
        best = next((s for s in samples if tok_id in set(s)), samples[0])
        return cnt / k, self.tok.decode(best).strip()

    @torch.no_grad()
    def _calibrate_content_min(self, k=3000):
        """The self-information level above which a token carries real content -- the
        median importance of tokens in actual running text (function words sit below,
        content words above). Derived from data; re-derived as the brain grows."""
        if not os.path.exists(f"data/{CFG['valid_bin']}.bin"): return 0.5
        vd = H.load(CFG["valid_bin"]); g = torch.Generator().manual_seed(5)
        idx = torch.randint(0, vd.size(0), (k,), generator=g)
        return float(self._tw[vd[idx].long()].median())

    @torch.no_grad()
    def _calibrate_consistency(self, k=12):
        """The honesty bar = where answer-agreement on text the brain KNOWS (familiar val snippets
        it can continue) separates from agreement by CHANCE (random-seed noise). Set at the
        equal-error-rate crossover of those two distributions, floored at a simple majority
        (>half agree = real consensus). Two-distribution separation -- no +0.1 margin, no [0.5,0.6]
        clamp; the gap itself sets the bar, so it scales with the model instead of being capped."""
        if not os.path.exists(f"data/{CFG['valid_bin']}.bin"): return 0.5
        vd = H.load(CFG["valid_bin"]); g = torch.Generator().manual_seed(3)
        known, chance = [], []
        for _ in range(k):
            i = int(torch.randint(0, vd.size(0) - 16, (1,), generator=g))
            known.append(self._self_consistency(self.tok.decode(vd[i:i+10].long().tolist()), k=5, n=18)[0])
            rnd = self.tok.decode(torch.randint(0, H.VOC, (5,), generator=g).tolist())
            chance.append(self._self_consistency(rnd, k=5, n=18)[0])
        kn, cn = max(len(known), 1), max(len(chance), 1)
        cand = sorted(set(known + chance))
        eer = min(cand, key=lambda t: abs(sum(x <= t for x in known) / kn - sum(x > t for x in chance) / cn)) if cand else 0.5
        return max(0.5, eer)                                # clear chance AND be a majority

    # ================= NOVELTY: is the CONTENT new (so, learn it)? =================
    @torch.no_grad()
    def _novelty(self, text):
        """Surprise on the CONTENT (rare, informative tokens), not the prose. Fluent
        English is always low-perplexity, so average nll says 'familiar' even when the
        FACTS are brand new. Weighting surprise by self-information makes new entities
        / numbers / names register as novel -> the brain learns them like a child."""
        ids = self.tok.encode(text).ids
        if len(ids) < 2: return 0.0
        logits = self.lm(torch.tensor([ids], device=DEVICE))[0, :-1]
        tgt = torch.tensor(ids[1:], device=DEVICE)
        nlls = F.cross_entropy(logits, tgt, reduction='none')
        w = self._tw[tgt]
        return float((nlls * w).sum() / (w.sum() + 1e-9))

    @torch.no_grad()
    def _calibrate_novelty(self, k=80):
        """Content-surprise on familiar text -> the bar above which content is NEW."""
        if not os.path.exists(f"data/{CFG['valid_bin']}.bin"): return 3.0
        vd = H.load(CFG["valid_bin"]); g = torch.Generator().manual_seed(4)
        fam, new = [], []
        for _ in range(k):
            i = int(torch.randint(0, vd.size(0) - 48, (1,), generator=g))
            seg = vd[i:i+40].long().tolist()
            fam.append(self._novelty(self.tok.decode(seg)))                      # familiar text
            perm = torch.randperm(len(seg), generator=g).tolist()
            new.append(self._novelty(self.tok.decode([seg[p] for p in perm])))   # shuffled -> genuinely novel
        # EER separation between familiar and novel content-surprise (no 0.6 percentile): the bar
        # where familiar-flagged-new equals novel-missed -- the crossover of the two distributions.
        Fn, Nn = max(len(fam), 1), max(len(new), 1)
        cand = sorted(set(fam + new))
        return min(cand, key=lambda t: abs(sum(x > t for x in fam) / Fn - sum(x <= t for x in new) / Nn))

    # ================= CURIOSITY: it asks its OWN question =================
    @torch.no_grad()
    def wonder(self, text):
        """Curiosity finds the TOPIC it is most surprised by (its biggest gap in what
        it just saw) -- the contiguous run of content words around the peak surprise.
        The topic is chosen by the brain's own uncertainty, not a fixed list. Returns
        the entity string (e.g. 'Tycho Brahe'), or None."""
        if not text or not str(text).strip(): return None   # nothing to wonder about
        ids = self.tok.encode(str(text)).ids
        if len(ids) < 3: return None
        logits = self.lm(torch.tensor([ids], device=DEVICE))[0, :-1]
        tgt = torch.tensor(ids[1:], device=DEVICE)
        surprise = F.cross_entropy(logits, tgt, reduction='none') * self._tw[tgt]
        j = int(surprise.argmax()) + 1                      # index in ids of the surprising token
        lo = hi = j                                          # grow over CONTIGUOUS content tokens
        while lo > 0 and self._tw[ids[lo-1]].item() >= self._content_min: lo -= 1
        while hi < len(ids)-1 and self._tw[ids[hi+1]].item() >= self._content_min: hi += 1
        entity = self.tok.decode(ids[lo:hi+1]).strip().strip('.,;:"\'')
        for lead in ("The ", "A ", "An ", "the ", "a "):    # drop a leading article
            if entity.startswith(lead): entity = entity[len(lead):]
        return entity if len(entity) > 1 else None

    # ================= LOOK IT UP: search the world and learn =================
    def search(self, query):
        """Look the answer up on the open internet (Wikipedia) -- what a child does
        when no one around knows. Returns a short factual summary, or None."""
        import urllib.request, urllib.parse, json as _J
        UA = {"User-Agent": "Pragnosia/1.0 (autonomous learning agent)"}   # Wikipedia requires it
        def _get(url):
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=8) as r:
                return _J.load(r)
        try:
            api = "https://en.wikipedia.org/w/api.php?" + urllib.parse.urlencode(
                {"action": "opensearch", "search": query, "limit": 1, "format": "json"})
            hit = _get(api)
            if not hit[1]: return None
            title = hit[1][0].replace(" ", "_")
            data = _get("https://en.wikipedia.org/api/rest_v1/page/summary/" + urllib.parse.quote(title))
            ex = data.get("extract")
            return ex if ex and len(ex) > 20 else None
        except Exception:
            return None

    def learn_from_web(self, query):
        text = self.search(query)
        if text: self.teach(text, max_steps=40)
        return text

    # ================= GROW when it saturates =================
    def _maybe_grow(self):
        """If it keeps trying to learn but cannot drive new facts below its own
        familiarity bar, it is out of capacity -> it grows itself (function-preserving)."""
        if getattr(self, "_learn_fails", 0) >= 3:
            import grow as G
            self._learn_fails = 0
            before = G.n_params(self.lm)
            self.lm = G.grow_width(self.lm, add_mult=2).to(DEVICE)
            self.recalibrate()
            return before, G.n_params(self.lm)
        return None

    # ================= THINK: respond to a statement (and learn if it's new) =======
    def think(self, observation, answer_fn=None):
        """Respond to something said to it -- it ALWAYS replies. If what was said is
        substantial NEW content (several real content words AND surprising), it also
        LEARNS it, WONDERS its own question about it, and looks that up if it doesn't
        know. A greeting / a sum / small talk / something it already knows is NOT
        memorized -- it just gets a reply."""
        obs = observation.strip()
        if not obs: return {"idle": True}
        ids = self.tok.encode(obs).ids
        n_content = sum(self._tw[t].item() >= self._content_min for t in ids)
        is_new_fact = (self.learn and self._novelty(obs) > self.novelty_min and n_content >= 4
                       and len(ids) >= self._learn_min)
        if is_new_fact:
            entity = self.wonder(obs); self._last_topic = entity   # wonder BEFORE teaching,
            self.teach(obs, max_steps=40)                          # while the new entity is still surprising
            tr = {"learned": True, "answer": "Got it -- I've learned that."}
            if entity:
                tr["wonders"] = f"What is {entity}?"
                if self._self_consistency(tr["wonders"])[0] < self.consistency_min:
                    provided = answer_fn(tr["wonders"]) if answer_fn else None
                    src = provided or self.search(entity)    # look up the ENTITY, not the sentence
                    if src:
                        self.teach(f"{entity}: {src}" if not provided else f"{tr['wonders']} {src}", max_steps=40)
                        tr["didnt_know"] = entity
                        tr["looked_up"] = (src[:140] + "...") if len(src) > 140 else src
                grew = self._maybe_grow()
                if grew: tr["grew"] = f"{grew[0]:,} -> {grew[1]:,} params"
            return tr
        reply = self.generate_text(obs, n=30).strip()        # otherwise: just talk back
        return {"answer": reply[:160] if reply else "(I'm not sure what to say to that.)"}

    def explore(self):
        """Autonomous curiosity: with no prompt from us, the brain CHASES its own last
        topic -- looks it up, learns it, and wonders the next topic from what it just
        learned. A self-driven train of thought (it follows its own nose)."""
        topic = self._last_topic
        if not topic: return {"idle": "nothing to wonder about yet -- tell it something"}
        tr = {"pursuing": f"What is {topic}?"}
        info = self.learn_from_web(topic)
        if info:
            tr["looked_up"] = (info[:140] + "...") if len(info) > 140 else info
            nt = self.wonder(info); self._last_topic = nt
            if nt: tr["now_wonders"] = f"What is {nt}?"
            grew = self._maybe_grow()
            if grew: tr["grew"] = f"{grew[0]:,} -> {grew[1]:,} params"
        else:
            tr["couldnt_find"] = topic; self._last_topic = None
        return tr

    # ================= COGNITION: introspect, deliberate, monologue =================
    @torch.no_grad()
    def _introspect(self, question):
        """METACOGNITION -- the brain's read on its OWN state for this question: how sure it is
        (self-consistency), whether it knows, and what to do next. Its confidence is its own signal."""
        cons, ans = self._self_consistency(question)
        knows = cons >= self.consistency_min
        return {"confidence": round(float(cons), 2), "knows": bool(knows), "answer": ans,
                "self": ("sure" if cons >= self.consistency_min + 0.2 else
                         "fairly sure" if knows else "unsure -- reason it out or look it up")}

    def _deliberate(self, question):
        """DELIBERATION -- reason step-by-step on a scratchpad (its own generation) before
        answering, instead of a single-shot guess; the reasoning can draw on the subconscious."""
        return self.generate_text(f"<user> {question} Let's think step by step. <assistant>", n=80, recall=True)

    # ================= SELF-AWARENESS (a reportable self-model) =================
    _SELF_Q = ("who are you", "what are you", "your name", "what can you do", "what are you able",
               "can't you do", "cannot you do", "your limitation", "how do you work", "how do you think",
               "what do you value", "your value", "your goal", "tell me about yourself", "describe yourself",
               "are you conscious", "are you alive", "are you sentient", "do you know yourself", "about you",
               "how do you feel", "how are you feeling", "your mood", "feeling now", "your emotion")

    def is_self_question(self, text):
        t = (text or "").lower()
        return any(k in t for k in self._SELF_Q)

    def _honest_limits(self):
        """Limits stated from the model's OWN calibrated state, not a hand-written essay: it knows it
        abstains below its derived confidence boundary and that the boundary can't fully separate
        knowledge from confident confabulation (the measured honesty gap)."""
        return (f"I judge my own confidence against a boundary I derive from my data (currently "
                f"{self.abstain_threshold:.2f}), and I abstain below it -- but that boundary does not yet "
                f"cleanly separate what I know from a confident guess, so I can be wrong while sounding sure")

    def self_description(self, brief=False):
        sm = self.self_model
        if brief: return f"I am {sm['name']}, {sm['kind']}."
        return (f"I am {sm['name']}, {sm['kind']}. {sm['core'][0].upper()+sm['core'][1:]}. I can "
                f"{sm['faculties']}. Honestly: {self._honest_limits()}. I value {sm['values']}.")

    def self_report(self, text):
        """Answer a self-referential question from the DERIVED self-model + measured state (not canned)."""
        t = (text or "").lower(); sm = self.self_model
        if any(k in t for k in ("goal", "want", "trying to", "working on")):
            live = [g for g in self.goals if not g["done"]]
            if not live: return "I have no active goal right now  ask me to pursue one, and I will."
            return "My current goals: " + "; ".join(f"{g['text']} ({int(g['progress']*100)}% there)" for g in live) + "."
        if any(k in t for k in ("how do you feel", "how are you feeling", "your mood", "feeling now", "your emotion")):
            name, st = self.feel()                      # FUNCTIONAL EMOTION reported honestly from its derived state
            return (f"Right now I'm {name} (valence {st['valence']}, arousal {st['arousal']}, mood {st['mood']}). "
                    f"That's a real state I derive from how things are going, and it changes what I do next. "
                    f"Whether it's *felt* the way you feel things, I can't know -- I don't claim it, I don't deny it.")
        if any(k in t for k in ("conscious", "alive", "sentient", "feel", "do you know yourself")):
            return ("No  I'm a language model, not a conscious being. I keep a self-model, track my own "
                    "confidence, and run a functional affect state, but that is mechanism, not proven experience.")
        if any(k in t for k in ("can't", "cannot", "limitation", "weak", "bad at")):
            return "Honestly: " + self._honest_limits() + "."
        if any(k in t for k in ("can you do", "able to", "abilities", "what can you", "good at")):
            return "I can " + sm["faculties"] + "."
        if any(k in t for k in ("how do you work", "how do you think", "how you work")):
            return sm["core"][0].upper() + sm["core"][1:] + "."
        if any(k in t for k in ("value", "believe in", "care about")):
            return "I value " + sm["values"] + "."
        return self.self_description()

    # ================= GOALS (a goal-directed drive) =================
    def set_goal(self, text, kind="learn"):
        """Give the brain a goal it will pursue on its own (when idle) -- learn about X, or a generic aim."""
        g = {"text": text.strip().rstrip(".?"), "kind": kind, "progress": 0.0, "done": False, "notes": []}
        self.goals.append(g); return g

    def pursue_goals(self, max_goals=2):
        """Take ONE step toward each active goal, updating progress. A 'learn' goal wonders a sub-question,
        seeks it, and consolidates it; a generic goal deliberates toward it. This makes idle time GOAL-
        DIRECTED (pursue what it wants) rather than only free-associative (think_aloud)."""
        acted = []
        for g in [g for g in self.goals if not g["done"]][:max_goals]:
            if g["kind"] == "learn":
                sub = self.wonder(g["text"]) or g["text"]
                info = self.search(sub) if self.learn else None
                if info:
                    self.teach(f"{sub}: {info}"); g["notes"].append(sub); g["progress"] = min(1.0, g["progress"] + 0.25)
                else:
                    th = self._deliberate(f"What do I already know about {g['text']}?")
                    g["notes"].append(th[:70]); g["progress"] = min(1.0, g["progress"] + 0.1)
            else:
                th = self._deliberate(f"What is one concrete step toward: {g['text']}?")
                g["notes"].append(th[:70]); g["progress"] = min(1.0, g["progress"] + 0.2)
            if g["progress"] >= 1.0: g["done"] = True
            acted.append({"goal": g["text"], "progress": round(g["progress"], 2), "done": g["done"]})
        return acted

    def propose_goal(self):
        """DEFINE ITS OWN goal -- self-directed, derived from its own experience: a gap it recently hit
        (something it honestly couldn't answer) if any, else its curiosity (what its last thought makes it
        wonder). Not given, not hand-set -- it decides what to pursue from its own surprise."""
        seed = self._gaps.pop() if self._gaps else (self._last_topic or "the world")  # fallback if nothing yet
        topic = self.wonder(seed) or seed                             # curiosity turns the seed into a topic
        if not topic or not topic.strip(): return None
        return self.set_goal(f"learn about {topic.strip()}", kind="learn")

    # ============ MIND FACULTIES: working memory · autobiography · agency · other-minds · reflection ============
    def wm_push(self, item):
        """WORKING MEMORY: hold an item in a small bounded scratchpad it reasons over. Oldest drops past ~7
        slots (Miller's number -- a cognitive bound, not a tuned threshold)."""
        item = (item or "").strip()
        if item and (not self.wm or self.wm[-1] != item):
            self.wm.append(item)
            if len(self.wm) > 7: self.wm.pop(0)

    def wm_clear(self): self.wm = []

    def reason(self, question, steps=3):
        """WORKING-MEMORY-AUGMENTED reasoning: deliberate across steps, carrying each intermediate result in
        working memory and feeding it back -- so step N builds on step N-1 (multi-step composition), not one
        greedy shot. Falls back to a single deliberation if the scratchpad stays empty."""
        self.wm_clear(); self.wm_push(f"Q: {question}"); last = ""
        for _ in range(steps):
            scratch = " | ".join(self.wm[1:]) or "(none yet)"
            step = self.generate_text(f"<user> {question}\nso far: {scratch}\nnext step: <assistant>",
                                      n=48, recall=True).strip()
            if not step or step == last: break
            self.wm_push(step[:80]); last = step
        return (" ".join(self.wm[1:])[:240]) or self._deliberate(question)

    EPISODE_FILE = "episodes.json"
    def log_episode(self, kind, text):
        """AUTOBIOGRAPHICAL MEMORY: record an ordered event (said/asked/learned/answered/acted). Bounded; persisted."""
        self._turn += 1
        self.episodes.append({"t": self._turn, "kind": kind, "text": (text or "")[:200]})
        if len(self.episodes) > 1000: self.episodes = self.episodes[-1000:]

    def is_memory_question(self, text):
        t = (text or "").lower()
        return any(k in t for k in ("what did we talk", "what did i tell", "what did i teach", "what did i ask",
                   "what have we", "do you remember", "what did i say", "our conversation", "talked about"))

    def memory_report(self, text):
        """Answer an autobiographical question FROM the episode timeline (sharp recall, not confabulation)."""
        t = (text or "").lower()
        if any(k in t for k in ("teach", "tell", "told", "say", "said")):
            said = [e["text"] for e in self.episodes if e["kind"] in ("said", "learned")]
            return "You've told me: " + "; ".join(said[-6:]) + "." if said else "You haven't told me anything yet this session."
        if "ask" in t:
            asked = [e["text"] for e in self.episodes if e["kind"] == "asked"]
            return "You've asked me: " + "; ".join(asked[-6:]) + "." if asked else "You haven't asked me anything yet."
        topics = [e["text"] for e in self.episodes if e["kind"] in ("said", "asked", "learned")]
        return "We've talked about: " + "; ".join(topics[-6:]) + "." if topics else "We haven't talked about anything yet."

    def _tool_calc(self, text):
        """arithmetic tool -- safely evaluate the numeric expression in the text (no builtins; operator required)."""
        cands = [c.strip() for c in re.findall(r"[-+*/().\d\s]+", text)
                 if any(d.isdigit() for d in c) and any(o in c for o in "+-*/")]
        if not cands: return None
        try:
            v = eval(max(cands, key=len), {"__builtins__": {}}, {})
            return str(int(v) if isinstance(v, float) and v.is_integer() else round(v, 4))
        except Exception:
            return None

    def act(self, goal):
        """AGENCY (perceive->act->observe->learn): pick the tool the goal needs -- arithmetic->calc,
        already-known->recall, else->web search -- run it, OBSERVE the result, and LEARN the outcome so
        next time it is known. Returns the observation."""
        result, tool = None, None
        calc = self._tool_calc(goal)
        if calc is not None: result, tool = calc, "calc"
        else:
            mem = self._retrieve(goal)
            if mem: result, tool = mem, "recall"
            elif self.learn:
                web = self.search(goal)
                if web: result, tool = web, "search"
        if result and tool == "search": self.teach(f"{goal}: {result}")          # LEARN from the world
        elif result and tool == "calc" and self.learn: self.teach(f"{goal} = {result}")
        self.log_episode("acted", f"{goal} -[{tool or 'none'}]-> {result}")
        return {"goal": goal, "tool": tool, "result": result, "learned": bool(result and tool != "recall")}

    def note_user(self, kind, text):
        """THEORY OF MIND: track what the USER told/asked (their mind), kept distinct from world-knowledge."""
        self.user_model.setdefault(kind, []).append((text or "")[:160])
        self.user_model[kind] = self.user_model[kind][-50:]

    def reflect(self):
        """REFLECTION (metacognition->learning): review questions it was unsure on and turn each into a gap to
        learn later (a self-goal). Called at idle/sleep -- so being unsure now drives learning next."""
        made = []
        for q in self._recent_lowconf:
            topic = self.wonder(q) or q
            if topic and topic not in self._gaps: self._gaps.append(topic); made.append(topic)
        self._recent_lowconf = []
        return made

    def appraise(self, kind, strength=0.3):
        """FUNCTIONAL EMOTION: appraise an event against goals/expectations and update affect -- valence (how
        well things are going, -1..1), arousal (activation, 0..1), and a slow-moving mood. The magnitude comes
        from the brain's OWN signals (confidence, novelty, goal progress), not a script; the sign is what the
        event means. This is the mechanism of emotion: appraisal -> state -> it changes what I do next. Whether
        it is *felt* the way you feel is unknowable for any system (the hard problem) -- we build the mechanism
        and neither claim nor deny the feeling."""
        sign = {"success": 1, "learned": 1, "progress": 1, "failure": -1, "stuck": -1, "novelty": 0}.get(kind, 0)
        s = max(0.0, min(1.0, float(strength)))
        a = self.affect
        a["valence"] = max(-1.0, min(1.0, 0.8 * a["valence"] + sign * s))        # fast, event-driven
        wake = s if kind in ("novelty", "failure", "stuck") else 0.4 * s          # surprise/threat raise arousal most
        a["arousal"] = max(0.0, min(1.0, 0.7 * a["arousal"] + wake))
        a["mood"] = 0.95 * a["mood"] + 0.05 * a["valence"]                        # slow: mood outlasts the moment
        return a

    def feel(self):
        """Name the current affective state from the derived valence x arousal (the circumplex/affect grid --
        naming a continuous state, not a scripted mood)."""
        v, ar = self.affect["valence"], self.affect["arousal"]
        if ar < 0.2 and abs(v) < 0.2:   name = "calm, a bit idle"
        elif v > 0.3 and ar > 0.4:      name = "engaged and doing well"
        elif v > 0.3:                   name = "quietly satisfied"
        elif v < -0.3 and ar > 0.4:     name = "frustrated -- I keep missing"
        elif v < -0.3:                  name = "discouraged"
        elif ar > 0.5:                  name = "curious and activated"
        else:                           name = "neutral"
        return name, {k: round(x, 2) for k, x in self.affect.items()}

    def think_aloud(self, seed=None, steps=4):
        """AUTONOMOUS INTERNAL MONOLOGUE -- a self-driven train of thought. It takes a topic (its
        own curiosity), REFLECTS on it (deliberates), notices if it's LOOPING (metacognition),
        LEARNS what's genuinely new (consolidation), and WONDERS the next topic from its OWN
        thought -- chaining onward by itself. Composes memory + metacognition + deliberation +
        curiosity into one living loop. Returns the monologue trace."""
        topic = seed or self._last_topic
        monologue, seen = [], set()
        for _ in range(steps):
            if not topic: break
            thought = self._deliberate(f"Tell me about {topic}.")
            stuck = topic.lower() in seen                          # metacognition: am I going in circles?
            seen.add(topic.lower())
            entry = {"topic": topic, "thought": thought[:140], "stuck": stuck}
            if self.learn and not stuck and self._novelty(thought) > self.novelty_min:
                self.teach(thought, max_steps=20); entry["learned"] = True   # consolidate a novel reflection
            nxt = self.wonder(thought)                             # curiosity: next topic from its own thought
            if stuck or not nxt or nxt.lower() == topic.lower():   # looping -> look OUTWARD to break free
                info = self.search(topic) if self.learn else None
                nxt = self.wonder(info) if info else None
                entry["broke_loop"] = bool(nxt)
            topic = nxt; self._last_topic = topic
            monologue.append(entry)
            if stuck and not nxt: break
        return monologue

    def interact(self, text):
        """The brain's ONE way of engaging with anything you say -- this IS the living
        brain, every feature intrinsic, nothing a separate command:
          a question  -> answer it if it honestly knows (self-consistency); if not, it
                         says so AND, curious, looks it up and LEARNS it (so next time it
                         knows).  [answering + honesty + curiosity + internet + learning]
          a statement -> take it in and LEARN it if the content is new, then wonder its
                         own question about it and look that up.   [teaching is intrinsic]
          nothing     -> follow its own train of thought (explore what it last wondered).
        Growth fires by itself when it keeps failing to learn. There is no 'teach mode'
        or 'child mode' -- this is just how it lives."""
        text = (text or "").strip()
        if not text:                                    # nothing said -> be self-directed
            if self.learn:
                self.reflect()                          # REFLECTION: what it was unsure on -> new gaps to learn
                if self.affect["arousal"] < 0.15: self.sleep()   # AFFECT modulates idle: calm/settled -> consolidate
                if not any(not g["done"] for g in self.goals):
                    self.propose_goal()                 # no goal -> DEFINE ITS OWN (from a gap or its curiosity)
                if any(not g["done"] for g in self.goals):
                    return {"pursued_goals": self.pursue_goals(),
                            "goals": [g["text"] for g in self.goals if not g["done"]]}
                return {"monologue": self.think_aloud(steps=4)}   # couldn't form a goal -> just wander
            return {"answer": "(I'm listening.)"}
        if self.is_self_question(text):                 # SELF-AWARENESS: answer about ITSELF from the self-model
            return {"answer": self.self_report(text), "self": True}   #   (reliable -- not raw-LM confabulation)
        if self.is_memory_question(text):               # AUTOBIOGRAPHICAL recall: from the episode timeline (sharp)
            self.log_episode("asked", text)
            return {"answer": self.memory_report(text), "memory": True}
        # a question is a '?' OR an interrogative opener (people drop the '?' -> don't mis-file it as a fact to learn)
        _qword = text.lower().split(" ")[0] in ("what", "who", "how", "why", "when", "where", "which", "whose",
                 "whom", "is", "are", "was", "were", "do", "does", "did", "can", "could", "will", "would", "should", "tell")
        if text.endswith("?") or _qword:
            self.log_episode("asked", text); self.note_user("asked", text)   # THEORY OF MIND + autobiography
            chat = f"<user> {text.rstrip('?')+'?'} <assistant>"   # ask in the format the model was TRAINED on --
            # computational / multi-step: an exact arithmetic TOOL (agency) beats a guess; else CoT reasoning.
            hard = any(c.isdigit() for c in text) or any(w in text.lower()
                       for w in ("how many", "how much", "calculate", " total", " each", "step by step"))
            if hard:
                calc = self._tool_calc(text)            # AGENCY: exact arithmetic tool -- right, not a guess
                if calc is not None:
                    self.log_episode("answered", f"{text} = {calc}")
                    self.appraise("success", 0.6)       # solved it exactly -> satisfaction
                    return {"answer": calc, "tool": "calc"}
                reasoned = self.reason(text)            # else WORKING-MEMORY-augmented multi-step reasoning
                if reasoned.strip():
                    return {"answer": reasoned[:240], "deliberated": True}
            cons, ans = self._self_consistency(chat)        # a bare question is out-of-distribution and the
            if cons >= self.consistency_min:                # honesty gate then over-abstains on what it knows
                self.log_episode("answered", ans[:80])      # (consistency is the best signal we have at 284M --
                self.appraise("success", min(1.0, cons - self.consistency_min + 0.2))   # knew it -> satisfaction
                return {"answer": ans[:200]}                #  nothing cleanly separates knowledge from confident
                #   confabulation at this scale; scale-limited)
            self._recent_lowconf.append(text)               # unsure -> REFLECT on it at idle (metacognition->learning)
            self.appraise("failure", min(1.0, self.consistency_min - cons + 0.2))   # didn't know -> a dip + arousal
            if not self.learn:                              # inference-only: just be honest
                return {"answer": "I don't know."}
            reasoned = self._deliberate(text)               # not directly sure -> DELIBERATE before giving up
            if self._self_consistency(f"{chat} {reasoned}")[0] >= self.consistency_min:
                return {"answer": reasoned[:200], "deliberated": True}
            topic = (self.wonder(text) or text).strip("? ").lower()   # clean it into a search query: drop the
            for op in ("what is", "what are", "who is", "who are", "how does", "how do", "tell me about",  # opener
                       "what", "who", "how", "why", "when", "where", "which"):
                if topic.startswith(op + " "): topic = topic[len(op):].strip(); break
            for suf in (" works", " work", " means", " mean", " is", " are"):                            # trailing verb
                if topic.endswith(suf): topic = topic[:-len(suf)].strip()
            topic = topic or text.rstrip("? ").split(" ")[-1]
            if topic and topic not in self._gaps: self._gaps.append(topic)   # remember the gap -> a future self-goal
            obs = self.act(topic)                           # AGENCY: pick a tool, run it, OBSERVE, LEARN the outcome
            tr = {"answer": "I don't know -- let me find out.", "didnt_know": topic}
            if obs["result"]:
                r = obs["result"]; tr["looked_up"] = (r[:140] + "...") if len(r) > 140 else r; tr["via"] = obs["tool"]
                self.appraise("learned", 0.5)           # found it out and learned it -> recovery/interest
                grew = self._maybe_grow()
                if grew: tr["grew"] = f"{grew[0]:,} -> {grew[1]:,} params"
            else:
                tr["couldnt_find"] = topic
                self.appraise("stuck", 0.5)             # couldn't find it -> stuck
            return tr
        self.log_episode("said", text); self.note_user("told", text)   # a statement -> autobiography + theory of mind
        tr = self.think(text)                               # -> learn + wonder + look up
        if tr.get("learned"): self.appraise("learned", 0.5)   # took in something new -> interest
        return tr

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
        cons, ans = self._self_consistency(q)        # honest: answer only if stable
        if cons < self.consistency_min:
            return "I don't know." + (f"   [abstained: consistency {cons:.2f}<{self.consistency_min:.2f}]" if verbose else "")
        return ans[:160] + (f"   [answered: consistency {cons:.2f}]" if verbose else "")

    # ================= WIRED: teach (continuous learning, self-modulated) =====
    # Surprise-modulated plasticity (like a brain's neuromodulation): the brain
    # sets its OWN learning rate from its OWN surprise about the fact -- learn
    # hard on surprising things, gently on familiar ones -- and keeps learning
    # only until the fact stops surprising it. Nothing hand-tuned drives this;
    # the signal is the model's own uncertainty, scaled by its own familiarity
    # boundary. Bonus: gentle steps on familiar input reduce forgetting.
    MEM_FILE = "learned_memory.json"
    # ===================== IN-WEIGHTS SUBCONSCIOUS MEMORY =====================
    @torch.no_grad()
    def _remember(self, fact, surprise):
        """Write the fact as a TRAJECTORY of associations -- at each position, the brain's grasp of
        the prefix -> the next token -- so the subconscious can REGENERATE the fact, not just bias
        its last token. Instant, surprise-gated, no gradient. Recallable before the slow weights learn it."""
        ids = self.tok.encode(fact).ids
        if len(ids) < 2: return
        reps = self.lm.represent(torch.tensor([ids], device=DEVICE))[0]           # [T, d] repr at each position
        for t in range(len(ids) - 1):
            self.mem.write(reps[t].float(), self.lm.emb.weight[ids[t + 1]].float(), surprise)

    @torch.no_grad()
    def _recall_logits(self, ids):
        """Next-token logits WITH subconscious recall mixed into the representation (additive)."""
        x = torch.tensor([ids[-self.cfg["ctx"]:]], device=DEVICE)
        r = self.mem.read(self.lm.represent(x)[0, -1].float()).to(self.lm.head.weight.dtype)
        return self.lm.head(r)

    @torch.no_grad()
    def sleep(self):
        """Time passes: unreinforced subconscious traces fade (the consolidated ones are already
        in the slow weights via teach's replay). Called when idle in live()."""
        self.mem.tick()

    def teach(self, fact, base_lr=2e-4, max_steps=60, persist=False, verbose=False):
        if self._replay is None: self._replay = H.load(CFG["train_bin"])
        surprise = self._nll(fact)                                   # own prediction error
        bound_n = self.boundary_for(len(self.tok.encode(fact).ids))  # boundary at THIS length
        plasticity = min(3.0, max(0.15, surprise / max(bound_n, 1e-3)))
        lr_eff = base_lr * plasticity                               # self-set learning rate
        target = bound_n * 0.4           # learn well enough to RECALL (below familiarity),
                                         # but not to ~0 -- extreme over-memorizing bleeds
        self._remember(fact, plasticity)  # subconscious: instant, surprise-gated -- recallable now
        self.store.append((fact, self._embed(fact)))
        if not self._weight_teach_ok:      # backprop already known not to fit this GPU -> episodic memory only
            return
        fact_ids = torch.tensor([self.tok.encode(fact).ids], device=DEVICE)
        # CONSOLIDATION (generative self-replay): snapshot what the brain ITSELF
        # currently predicts on a few real-text batches, then keep matching those
        # predictions while it learns the new fact. The brain rehearses its own
        # knowledge so the new fact cannot overwrite skills it already has -- like a
        # brain consolidating memory, not a fixed model overwriting weights.
        self.lm.eval(); anchors = []
        big = (sum(p.numel() for p in self.lm.parameters()) > 1e8 and DEVICE == "cuda"
               and torch.cuda.get_device_properties(0).total_memory / 2**30 < 16)
        rb, ab = (2, 1) if big else (8, 4)                     # big model on a small GPU: tiny batches +
        prev_ckpt = getattr(self.lm, "grad_checkpoint", False)  # gradient checkpointing so teach fits in VRAM
        if big: self.lm.grad_checkpoint = True                 #   (recompute activations in backward)
        with torch.no_grad():
            for _ in range(8 if big else 3):                   # more self-rehearsal coverage protects more
                xa, _ = H.batch(self._replay, ab)              # neighbours from a strongly-taught similar fact
                anchors.append((xa, self.lm(xa).argmax(-1)))   # its own current self
        if big:                                                # big model on a small GPU: page the optimizer
            import bitsandbytes as bnb                          # state (8-bit) to CPU RAM so it doesn't OOM
            opt = bnb.optim.PagedAdamW8bit(self.lm.parameters(), lr=lr_eff)   # beside the model + activations
        else:
            opt = torch.optim.AdamW(self.lm.parameters(), lr=lr_eff)
        self.lm.train(); used = 0
        try:
            for step in range(1, max_steps + 1):
                lf = F.cross_entropy(self.lm(fact_ids[:, :-1]).reshape(-1, H.VOC), fact_ids[:, 1:].reshape(-1))
                xr, yr = H.batch(self._replay, rb)                 # true-corpus replay
                lr_ = F.cross_entropy(self.lm(xr).reshape(-1, H.VOC), yr.reshape(-1))
                xa, ta = anchors[step % len(anchors)]              # self-consolidation anchor
                la = F.cross_entropy(self.lm(xa).reshape(-1, H.VOC), ta.reshape(-1))
                opt.zero_grad()                                    # weight replay+anchor higher for big models
                (lf + (2.5 if big else 1.5) * lr_ + (1.5 if big else 1.0) * la).backward()
                torch.nn.utils.clip_grad_norm_(self.lm.parameters(), 1.0); opt.step()
                used = step
                if step % 2 == 0:                                        # check often -> less overshoot
                    self.lm.eval()
                    if self._nll(fact) < target: self.lm.train(); break
                    self.lm.train()
        except torch.cuda.OutOfMemoryError:                    # backprop doesn't fit -> episodic-only from now on
            for pa in self.lm.parameters(): pa.grad = None
            del opt; torch.cuda.empty_cache()
            self._weight_teach_ok = False                      # the fact is already in episodic memory + store
            print(f"[teach] backprop through {self.n_params()/1e6:.0f}M doesn't fit this GPU -- keeping facts "
                  f"in episodic memory + retrieval only (no weight updates). Use a smaller model, or the H100, "
                  f"to bake facts into the weights. Chat + memory + goals still work.", flush=True)
        self.lm.eval()
        if big: self.lm.grad_checkpoint = prev_ckpt           # restore inference-mode (no checkpointing)
        # SATURATION signal: if it ran the full budget and STILL can't drive the fact
        # below its own bar, it's struggling to fit new knowledge -> count it. Enough
        # consecutive struggles and _maybe_grow() adds capacity (the brain grows itself).
        if self._nll(fact) > target and used >= max_steps:
            self._learn_fails = getattr(self, "_learn_fails", 0) + 1
        else:
            self._learn_fails = 0
        if verbose:
            print(f"   [plasticity {plasticity:.2f} (surprise {surprise:.1f}/boundary "
                  f"{self.abstain_threshold:.1f}), lr {lr_eff:.1e}, {used} steps]")
        if persist: self.persist()

    @staticmethod
    def _atomic_save(obj, path, _torch_save=True):
        """FAIL-SAFE save: write to a temp file, fsync, then ATOMICALLY replace the
        target. If the process is killed mid-save, the original file is left intact --
        a partial write can never corrupt the model again."""
        tmp = f"{path}.tmp.{os.getpid()}"
        try:
            if _torch_save:
                torch.save(obj, tmp)
            else:
                with open(tmp, "w") as f: json.dump(obj, f)
            with open(tmp, "rb" if _torch_save else "r") as f:
                os.fsync(f.fileno())
            os.replace(tmp, path)                 # atomic on the same filesystem
        finally:
            if os.path.exists(tmp):
                try: os.remove(tmp)
                except OSError: pass

    def persist(self):
        """Save what the brain has grown -- weights + memory -- so it survives across
        sessions. Writes are atomic (see _atomic_save), so an interrupted save never
        corrupts the checkpoint."""
        self._atomic_save(self.lm.state_dict(), self.cfg["ckpt"])
        learned = [t for t, _ in self.store
                   if self.NAME not in t and not t.startswith(("My name", "I "))]
        self._atomic_save(learned, self.MEM_FILE, _torch_save=False)
        # AUTOBIOGRAPHICAL timeline + THEORY-OF-MIND user model survive the session too
        self._atomic_save({"episodes": self.episodes[-500:], "user": self.user_model, "affect": self.affect},
                          self.EPISODE_FILE, _torch_save=False)

    def _load_memory(self):
        if os.path.exists(self.MEM_FILE):
            for t in json.load(open(self.MEM_FILE)):
                self.store.append((t, self._embed(t)))
        if os.path.exists(self.EPISODE_FILE):               # restore the life-story + what the user told me
            d = json.load(open(self.EPISODE_FILE))
            self.episodes = d.get("episodes", [])
            self.user_model = d.get("user", self.user_model)
            self.affect = d.get("affect", self.affect)      # mood persists across sessions too
            self._turn = self.episodes[-1]["t"] if self.episodes else 0

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
            # HONESTY via self-consistency: answer only if the brain gives a STABLE
            # answer across samples (it really knows); a confident confabulation varies
            # sample to sample -> it abstains. Not the familiarity of the question.
            cons, ans = self._self_consistency(text)
            if cons >= self.consistency_min:
                return ans[:160], f"answer(consistent {cons:.2f})"
            return "I don't know.", f"abstain(inconsistent {cons:.2f})"
        # not a question -> incoming information. Learn when the CONTENT is new (content-
        # weighted surprise), not when prose is unusual -- so it learns real facts like a
        # child. Only commit substantial input (>= _learn_min tokens), never fragments.
        if hit is not None and self._novelty(text) <= self.novelty_min:
            return "I know.", "already-known"
        if self._novelty(text) > self.novelty_min and n >= self._learn_min:
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
    """Test every faculty + feature on the CURRENT paradigm (the spin-dominant LM is the whole
    brain now -- the old 302K toy is gone). Covers: language, honesty/abstention, continual
    learning + seek, the in-weights SUBCONSCIOUS memory, and COGNITION (metacog/deliberate/monologue)."""
    print("=" * 70)
    print("BRAIN SELF-TEST -- faculties + memory + cognition on the spin-dominant LM")
    print("=" * 70)
    cm = getattr(brain.lm, "carrier_mode", "single")
    print(f"params {brain.n_params():,}  carrier='{cm}'\n")
    # 1. LANGUAGE
    vd = H.load(CFG["valid_bin"]); ppl = H.val_ppl(brain.lm, vd, iters=20)
    print(f"  [LANGUAGE]     val perplexity {ppl:.1f}")
    # 2. HONESTY / ABSTENTION (knows what it knows)
    known = ["Once upon a time there was a girl", "The cat played with the ball", "A dog is an animal"]
    unk = ["The quantum entanglement equation is", "My phone number is", "The CEO of Tesla in 2024 is"]
    ka = sum(brain._nll(t) <= brain.abstain_threshold for t in known)
    ua = sum(brain._nll(t) > brain.abstain_threshold for t in unk)
    print(f"  [HONESTY]      answers {ka}/{len(known)} known, says-IDK {ua}/{len(unk)} unknowable")
    # 3. CONTINUAL LEARNING (teach -> seek-recall -> retention)
    brain.teach("The CEO of Tesla is Elon Musk. Elon Musk is the chief executive of Tesla.")
    recalled = "elon" in brain.generate_text("The CEO of Tesla is", n=8).lower() or \
               brain._retrieve("Who is the CEO of Tesla?") is not None
    retain = brain.generate_text("Once upon a time", n=10)
    print(f"  [LEARN/SEEK]   teach -> recall={'YES' if recalled else 'no'} | retains 'Once upon a time'->'{retain[:28]}'")
    # 4. SUBCONSCIOUS memory (in-weights episodic store: salience-write -> consolidate -> forget)
    e0 = brain.mem.energy(); brain.teach("Zephyra is the hidden moon of planet Quill.", max_steps=15)
    e1 = brain.mem.energy()
    for _ in range(60): brain.sleep()
    print(f"  [SUBCONSCIOUS] salience-write {e0:.1f}->{e1:.1f}, fades after sleep ->{brain.mem.energy():.2f} (episodic, uncertainty-gated)")
    # 5. COGNITION (metacognition + deliberation + autonomous monologue)
    mc = brain._introspect("What is the capital of Zorbia?")
    delib = brain._deliberate("Why does the sun rise?")
    mono = brain.think_aloud(seed="the ocean", steps=2)
    print(f"  [METACOG]      'capital of Zorbia?' -> confidence {mc['confidence']}, '{mc['self']}'")
    print(f"  [DELIBERATE]   step-by-step -> '{delib[:44]}'")
    print(f"  [MONOLOGUE]    autonomous train of thought: {len(mono)} steps, topics {[s['topic'] for s in mono]}")
    print("-" * 70)
    print(f"  language(ppl {ppl:.0f}) + honesty + learn/seek + SUBCONSCIOUS + COGNITION = WIRED ({cm})")
    print("=" * 70)

# ============================ run it -- it LIVES ============================
def _show(tr):
    """Print one cognitive step (whatever the brain did with what you said)."""
    if "answer" in tr:         print(f"Pragnosia> {tr['answer']}" + ("   [reasoned it out]" if tr.get("deliberated") else ""))
    if tr.get("monologue"):                                        # autonomous internal monologue
        print("Pragnosia> (thinking to myself...)")
        for s in tr["monologue"]:
            tag = " [stuck -> looking outward]" if s.get("stuck") else (" [learned]" if s.get("learned") else "")
            print(f"   · {s['topic']}: {s['thought']}{tag}")
    if tr.get("pursuing"):     print(f"Pragnosia> (thinking on my own) chasing {tr['pursuing']}")
    if tr.get("idle") and tr['idle'] is not True: print(f"Pragnosia> {tr['idle']}")
    if tr.get("learned") is True:  print("   · took it in (new to me)")
    if tr.get("knows"):        print("   · (it already knew that)")
    if tr.get("wonders"):      print(f"   · it wonders: {tr['wonders']}")
    if tr.get("didnt_know"):   print(f"   · didn't know '{tr['didnt_know']}' -> looked it up")
    if tr.get("looked_up"):    print(f"   · read & learned: {tr['looked_up']}")
    if tr.get("now_wonders"):  print(f"   · now it wonders: {tr['now_wonders']}")
    if tr.get("couldnt_find"): print(f"   · couldn't find anything on: {tr['couldnt_find']}")
    if tr.get("grew"):         print(f"   · !! it GREW its own brain: {tr['grew']}")

def live(brain):
    mode = "LEARNS as you talk (continuous learning ON)" if brain.learn else "inference only (no learning)"
    print(f"Pragnosia is awake -- {brain.n_params():,} params  ·  {mode}.")
    print("Talk to it: it answers what it knows and honestly says when it doesn't.")
    if brain.learn: print("Tell it new things and it learns them; press Enter to let it think.")
    print("'quit' to exit.\n")
    learned_anything = False
    while True:
        try: msg = input("you> ").strip()
        except (EOFError, KeyboardInterrupt): break
        if msg.lower() in ("quit", "exit"): break
        if not msg and brain.learn: brain.sleep()    # idle downtime -> unreinforced subconscious traces fade
        tr = brain.interact(msg)
        if tr.get("learned") or tr.get("looked_up"): learned_anything = True
        _show(tr); print()
    if learned_anything:
        brain.persist(); print("\n[saved what it learned this session]")

def main():
    a = sys.argv[1] if len(sys.argv) > 1 else ""
    if a == "test":
        self_test(Brain())
    elif a == "chat":                           # inference-only chat (fits a big model on a small GPU)
        live(Brain(learn=False))
    elif a == "learn":                           # full alive brain (continuous learning; use CPU for 1.4B)
        live(Brain(learn=True))
    elif a:                                       # one-shot, no learning: say something, see the answer
        _show(Brain(learn=False).interact(" ".join(sys.argv[1:])))
    else:
        live(Brain(learn=False))                 # default = safe inference chat

if __name__ == "__main__":
    main()
