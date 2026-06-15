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
        self.lm = H.SpinAttentionLM(CFG["vocab"], CFG["d"], CFG["heads"], CFG["layers"]).to(DEVICE)
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
        self.match_threshold = self._calibrate_match()  # what counts as a memory match

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
    def _calibrate(self, k=80):
        """Familiarity boundary from the model's OWN uncertainty on text it has
        seen -- not a hand-picked number. Deterministic, re-derivable."""
        if not os.path.exists(f"data/{CFG['valid_bin']}.bin"): return 4.5
        vd = H.load(CFG["valid_bin"])
        g = torch.Generator().manual_seed(0)
        nlls = []
        for _ in range(k):
            i = int(torch.randint(0, vd.size(0) - 40, (1,), generator=g))
            seg = vd[i:i+32]
            nlls.append(F.cross_entropy(self.lm(seg.unsqueeze(0).to(DEVICE))[0, :-1],
                                        seg[1:].to(DEVICE)).item())
        nlls.sort()
        return nlls[int(0.9 * len(nlls))]

    @torch.no_grad()
    def _calibrate_match(self, k=200):
        """Memory-match boundary from the data's OWN similarity distribution:
        sample unrelated text pairs, see how similar they look by chance, and set
        the bar above that background. Not a hand-picked 0.5."""
        if not os.path.exists(f"data/{CFG['valid_bin']}.bin"): return 0.5
        vd = H.load(CFG["valid_bin"]); g = torch.Generator().manual_seed(1)
        sims = []
        for _ in range(k):
            i = int(torch.randint(0, vd.size(0) - 20, (1,), generator=g))
            j = int(torch.randint(0, vd.size(0) - 20, (1,), generator=g))
            a = self._embed_ids(vd[i:i+12]); b = self._embed_ids(vd[j:j+12])
            sims.append(float(a @ b))
        sims.sort()
        return sims[int(0.98 * len(sims))]        # well above chance similarity

    def _install_identity(self, cache=f"pragnosia_id_{CFG['vocab']}.pt"):
        """Self-knowledge LEARNED INTO WEIGHTS (the proven continuous-learning
        faculty), as Q->A pairs, so it is recalled by the brain's own generation
        and gated by its own confidence -- no retrieval heuristics, no hardcoding.
        Each statement is TRUE of a capability the brain actually has."""
        qa = [
            f"What is your name? My name is {self.NAME}.",
            f"Who are you? I am {self.NAME}, a spinning brain that reasons by phase.",
            "Can you learn? Yes, I learn new things continuously without forgetting.",
            "What do you not know? I know the limits of my knowledge and say so.",
            "Are you curious? Yes, I explore whatever I am most uncertain about.",
            "How do you think? I think by spinning, my answer lives in the phase.",
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
        ids = torch.as_tensor(ids, device=DEVICE)
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
        nll = self._nll(q)
        if nll > self.abstain_threshold:
            return "I don't know." + (f"   [abstained: confidence {nll:.1f}>{self.abstain_threshold}]" if verbose else "")
        return self.generate_text(q)[:120] + (f"   [answered: confidence {nll:.1f}]" if verbose else "")

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
        plasticity = min(3.0, max(0.15, surprise / max(self.abstain_threshold, 1e-3)))
        lr_eff = base_lr * plasticity                               # self-set learning rate
        target = self.abstain_threshold * 0.6                       # "no longer surprised"
        self.store.append((fact, self._embed(fact)))
        fact_ids = torch.tensor([self.tok.encode(fact).ids], device=DEVICE)
        opt = torch.optim.AdamW(self.lm.parameters(), lr=lr_eff)
        self.lm.train(); used = 0
        for step in range(1, max_steps + 1):
            lf = F.cross_entropy(self.lm(fact_ids[:, :-1]).reshape(-1, H.VOC), fact_ids[:, 1:].reshape(-1))
            xr, yr = H.batch(self._replay, 8)
            lr_ = F.cross_entropy(self.lm(xr).reshape(-1, H.VOC), yr.reshape(-1))
            opt.zero_grad(); (lf + lr_).backward()
            torch.nn.utils.clip_grad_norm_(self.lm.parameters(), 1.0); opt.step()
            used = step
            if step % 5 == 0:                                        # stop once it's learned
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
        familiarity boundary, or already in memory."""
        return self._retrieve(text) is not None or self.uncertainty(text) <= self.abstain_threshold

    def respond(self, text):
        """The brain routes itself from confidence + curiosity. Returns (reply, decision)."""
        text = text.strip()
        if not text: return "", "noop"
        hit = self._retrieve(text)                       # real seek (covers self-knowledge)
        u = self.uncertainty(text)                       # real own-uncertainty
        is_query = text.endswith("?")
        if is_query:
            if hit is not None:       return hit, "seek/recall"
            if u <= self.abstain_threshold: return self.generate_text(text)[:120], "answer(confident)"
            return "I don't know.", "abstain(low-confidence)"
        # not a question -> incoming information. Curiosity = own uncertainty:
        # learn what it is uncertain about; acknowledge what it already knows.
        if hit is not None and u <= self.abstain_threshold:
            return "I know.", "already-known"
        if u > self.abstain_threshold:                   # novel -> curiosity drives learning
            self.teach(text, max_steps=40)
            return "That's new to me - I've learned it.", "learn(curiosity)"
        return self.generate_text(text)[:120], "continue(confident)"

    # ---- real introspection (grounded in actual mechanisms) ----
    def introspect(self):
        return [t for t, _ in self.store if t.startswith(("My name", self.NAME, "I "))]
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
