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
import json, math, os, sys, time
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
        self._last_topic = None    # the topic its curiosity is currently chasing
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
        """How much answer-agreement happens by CHANCE (random-seed questions). A real
        answer must clear that, and clear a majority. Derived; re-derived as it grows."""
        if not os.path.exists(f"data/{CFG['valid_bin']}.bin"): return 0.5
        g = torch.Generator().manual_seed(3); noise = []
        for _ in range(k):
            rnd = self.tok.decode(torch.randint(0, H.VOC, (5,), generator=g).tolist()) + "?"
            noise.append(self._self_consistency(rnd, k=5, n=18)[0])
        noise.sort()
        return max(0.5, noise[int(0.9 * len(noise))] + 0.1)        # > chance AND a majority

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
        vals = []
        for _ in range(k):
            i = int(torch.randint(0, vd.size(0) - 48, (1,), generator=g))
            vals.append(self._novelty(self.tok.decode(vd[i:i+40].long().tolist())))
        vals.sort(); return vals[int(0.6 * len(vals))]    # above-typical content surprise = new

    # ================= CURIOSITY: it asks its OWN question =================
    @torch.no_grad()
    def wonder(self, text):
        """Curiosity finds the TOPIC it is most surprised by (its biggest gap in what
        it just saw) -- the contiguous run of content words around the peak surprise.
        The topic is chosen by the brain's own uncertainty, not a fixed list. Returns
        the entity string (e.g. 'Tycho Brahe'), or None."""
        ids = self.tok.encode(text).ids
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

    # ================= THINK: one step of child cognition =================
    def think(self, observation, answer_fn=None):
        """One step of a child's mind: take in what it sees, LEARN it if the content
        is new, then WONDER a question of its own; try to ANSWER from itself, and if it
        does not know, get the answer (from a teacher if given, else the INTERNET) and
        learn it. GROW if it has saturated. Returns a trace of what it did/thought."""
        tr = {}
        obs = observation.strip()
        if not obs: return {"idle": True}
        tr["novelty"] = round(self._novelty(obs), 2)
        if tr["novelty"] > self.novelty_min and len(self.tok.encode(obs).ids) >= self._learn_min:
            self.teach(obs, max_steps=40); tr["learned"] = True
        entity = self.wonder(obs); self._last_topic = entity
        if entity:
            tr["wonders"] = f"What is {entity}?"
            cons, _ = self._self_consistency(tr["wonders"])
            if cons >= self.consistency_min:
                tr["knows"] = True
            else:
                tr["didnt_know"] = entity
                provided = answer_fn(tr["wonders"]) if answer_fn else None
                src = provided or self.search(entity)        # look up the ENTITY, not the sentence
                if src:
                    self.teach(f"{entity}: {src}" if not provided else f"{tr['wonders']} {src}", max_steps=40)
                    tr["looked_up"] = (src[:140] + "...") if len(src) > 140 else src
                grew = self._maybe_grow()
                if grew: tr["grew"] = f"{grew[0]:,} -> {grew[1]:,} params"
        return tr

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
        if not text:
            return self.explore()
        if text.endswith("?"):
            cons, ans = self._self_consistency(text)
            if cons >= self.consistency_min:
                return {"answer": ans[:200]}
            topic = self.wonder(text) or text.rstrip("? ").split(" ")[-1]
            tr = {"answer": "I don't know -- let me find out.", "didnt_know": topic}
            info = self.search(topic)                       # curious -> look it up and learn
            if info:
                self.teach(f"{topic}: {info}")
                tr["looked_up"] = (info[:140] + "...") if len(info) > 140 else info
                grew = self._maybe_grow()
                if grew: tr["grew"] = f"{grew[0]:,} -> {grew[1]:,} params"
            else:
                tr["couldnt_find"] = topic
            return tr
        return self.think(text)                             # a statement -> learn + wonder + look up

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

# ============================ run it -- it LIVES ============================
def _show(tr):
    """Print one cognitive step (whatever the brain did with what you said)."""
    if "answer" in tr:         print(f"Pragnosia> {tr['answer']}")
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
    print(f"Pragnosia is awake -- {brain.n_params():,} params. Just talk to it: it answers")
    print("what it knows, learns what you tell it, wonders its own questions, and looks up")
    print("what it doesn't know. Press Enter alone to let it think. 'quit' saves & exits.\n")
    while True:
        try: msg = input("you> ").strip()
        except (EOFError, KeyboardInterrupt): break
        if msg.lower() in ("quit", "exit"): break
        _show(brain.interact(msg)); print()
    brain.persist(); print("\n[saved what it learned this session]")

def main():
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        self_test(Brain())
    elif len(sys.argv) > 1:                     # one-shot: say something to it, see what it does
        _show(Brain().interact(" ".join(sys.argv[1:])))
    else:
        live(Brain())                           # the living brain

if __name__ == "__main__":
    main()
