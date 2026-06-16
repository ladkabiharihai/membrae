"""
converse.py -- a long autonomous conversation with the wired Brain (Pragnosia),
fully logged. Tests, in particular, whether a HUGE never-before-seen passage is
actually LEARNED (surprise drop + recall) and whether learning it causes any
forgetting. Everything (my narration + every prompt + every answer + the brain's
own decision/confidence) is written to conversation_log.txt and .json.
"""
import os, json, time, datetime
import torch
import brain as B

LOG_TXT = "conversation_log.txt"
LOG_JSON = "conversation_log.json"
_fh = open(LOG_TXT, "w")
_records = []

def log(msg=""):
    print(msg)
    _fh.write(msg + "\n"); _fh.flush()

def narrate(msg):
    log(f"\n# ----- {msg} -----")

def turn(brain, text, kind="user"):
    """One conversational turn through the autonomous controller; log all signals."""
    t0 = time.time()
    u = brain.uncertainty(text)
    reply, decision = brain.respond(text)
    dt = time.time() - t0
    log(f"you> {text}")
    log(f"bot> {reply}")
    log(f"     [decided: {decision} | confidence(nll)={u:.2f} | boundary={brain.abstain_threshold:.2f} | {dt:.1f}s]")
    _records.append(dict(role=kind, prompt=text, reply=reply, decision=decision,
                         confidence=round(u, 3), boundary=round(brain.abstain_threshold, 3)))
    return reply, decision

def raw(brain, prompt, n=40):
    """Raw generation (bypass controller) -- to see what the weights produce."""
    g = brain.generate_text(prompt, n=n)
    log(f"   (raw generate) {prompt!r} -> {g[:160]!r}")
    return g

# ============================================================ build the brain
log("=" * 72)
log(f"PRAGNOSIA AUTONOMOUS CONVERSATION LOG   {datetime.datetime.now():%Y-%m-%d %H:%M}")
log("=" * 72)
narrate("STEP 1: waking the brain (loads new 176M model, re-installs identity, "
        "recalibrates abstention boundary from the corrected valid set)")
t0 = time.time()
brain = B.Brain()
log(f"brain up: {brain.n_params():,} params | abstention boundary = {brain.abstain_threshold:.2f} "
    f"| match boundary = {brain.match_threshold:.3f} | {time.time()-t0:.0f}s")

# ============================================================ Phase 1: identity / self-knowledge
narrate("STEP 2: ask it who it is and what it knows about itself (self-knowledge)")
for q in ["What is your name?", "Who are you?", "Can you learn?",
          "What do you not know?", "Are you curious?", "How do you think?"]:
    turn(brain, q)

# ============================================================ Phase 2: general knowledge + honesty
narrate("STEP 3: general knowledge + honesty (it should answer familiar, abstain on the unknown)")
for q in ["Once upon a time", "Water is made of", "Question: What is 5 plus 7?\nAnswer:",
          "def add(a, b):", "who is the president of india",
          "The quantum entanglement equation is", "What is 12 plus 13?"]:
    turn(brain, q)

# ============================================================ Phase 3: THE huge novel passage
narrate("STEP 4: feed a LARGE, entirely fictional passage it has never seen. As a "
        "statement (not a question), the controller's curiosity should route it to "
        "LEARN. I measure its surprise (nll) on the passage BEFORE and AFTER.")
NOVEL = (
"In the highland city of Quorimbar there is an old craft called sondering. "
"A sonder is a person who reads the rings of frostglass to predict the coming winter. "
"The chief sonder of Quorimbar is named Aldreth Vane. Frostglass forms only inside the Hollow Vale, "
"and only during the cold month the Quorimbari call Sextus. Each ring inside a piece of frostglass "
"marks exactly one stormy night that has passed. Sonders measure these rings with a slender brass tool "
"called a vimmer, and every vimmer has exactly nine notches along its edge. When a sonder counts more "
"than forty rings in a single shard, the people prepare for what they call a Deepwinter. The festival "
"that celebrates the first frost of the year is named Brumalia, and it is held once every seventh year "
"on the steps of the Pale Observatory. The first sonder in recorded history was a woman named Orin Telk, "
"who carved the very first vimmer from the antler of a snow elk. To this day, an apprentice sonder is "
"not allowed to touch frostglass with bare hands until they have memorized all nine notches of the vimmer "
"and recited the Telk Oath before the chief sonder."
)
log(f"[passage length: {len(NOVEL)} chars, {len(brain.tok.encode(NOVEL).ids)} tokens]")
nll_before = brain._nll(NOVEL)
log(f"surprise BEFORE seeing it : nll = {nll_before:.2f}  (boundary {brain.abstain_threshold:.2f}; "
    f"{'NOVEL -> above boundary' if nll_before > brain.abstain_threshold else 'familiar'})")
reply, decision = turn(brain, NOVEL)
nll_after = brain._nll(NOVEL)
log(f"surprise AFTER learning it : nll = {nll_after:.2f}   "
    f"(drop of {nll_before - nll_after:.2f} = {100*(nll_before-nll_after)/max(nll_before,1e-9):.0f}% -> "
    f"{'LEARNED' if nll_after < nll_before*0.8 else 'little change'})")
_records.append(dict(role="metric", novel_nll_before=round(nll_before,3), novel_nll_after=round(nll_after,3)))

# ============================================================ Phase 4: did it learn? recall the novel facts
narrate("STEP 5: quiz it on the fictional facts it just learned (recall via seek AND raw weights)")
for q in ["What is a sonder?", "Who is the chief sonder of Quorimbar?",
          "How many notches does a vimmer have?", "What is the festival of first frost called?",
          "Who carved the first vimmer?", "What is frostglass?"]:
    turn(brain, q)
    raw(brain, q.rstrip("?").replace("What is", "").replace("Who is", "").strip() + " is", n=24)

# ============================================================ Phase 5: forgetting check
narrate("STEP 6: did learning the new passage damage what it already knew? (forgetting check)")
for q in ["Question: What is 5 plus 7?\nAnswer:", "Once upon a time", "def add(a, b):"]:
    turn(brain, q)

# ============================================================ Phase 6: introspection + curiosity
narrate("STEP 7: introspection (what it believes about itself) and curiosity (it should "
        "pick the option it is MOST uncertain about)")
log(f"introspect -> {brain.introspect()}")
opts = ["the capital of France", "zxqwv flobble granfalloon", "two plus two"]
pick = brain.curiosity_pick(opts)
log(f"curiosity_pick({opts}) -> {pick!r}  (most uncertain = most curious about)")
for o in opts:
    log(f"     uncertainty({o!r}) = {brain.uncertainty(o):.2f}")

# ============================================================ wrap up
narrate("DONE. Saving structured log. (Not persisting weights to pragnosia.pt -- this was a test session.)")
json.dump(_records, open(LOG_JSON, "w"), indent=2)
log(f"\nwrote {LOG_TXT} and {LOG_JSON}")
_fh.close()
