"""
converse_physics.py -- re-run the autonomous conversation AFTER the controller fix,
on LONG physics passages with equations (quantum mechanics, black holes, particle
physics, astrophysics). Tests that the length-aware boundary now routes correctly:
long novel passages -> LEARN, short familiar prompts -> answer/continue (not
mis-learned), and seek recall actually fires. Everything logged.
"""
import os, json, time, datetime
import torch
import brain as B

LOG_TXT, LOG_JSON = "physics_conversation_log.txt", "physics_conversation_log.json"
_fh = open(LOG_TXT, "w"); _records = []
def log(m=""):
    print(m); _fh.write(m + "\n"); _fh.flush()
def narrate(m): log(f"\n# ===== {m} =====")

def turn(brain, text):
    t0 = time.time(); u = brain.uncertainty(text)
    n = len(brain.tok.encode(text).ids); bound = brain.boundary_for(n)
    reply, decision = brain.respond(text); dt = time.time() - t0
    short = text if len(text) < 90 else text[:87] + "..."
    log(f"you> {short}")
    log(f"bot> {reply}")
    log(f"     [decided: {decision} | nll={u:.2f} vs boundary({n}tok)={bound:.2f} | {dt:.1f}s]")
    _records.append(dict(prompt=text, reply=reply, decision=decision,
                         nll=round(u,3), n_tok=n, boundary=round(bound,3)))
    return reply, decision

def raw(brain, prompt, n=30):
    g = brain.generate_text(prompt, n=n)
    log(f"   (raw) {prompt!r} -> {g[:170]!r}")
    return g

# ---------------------------------------------------------------- the long passages
PASSAGES = {
"quantum mechanics":
"Quantum mechanics describes nature at the scale of atoms and subatomic particles. The state of a "
"system is a wavefunction psi, and its evolution obeys the time-dependent Schrodinger equation: "
"i*hbar * d(psi)/dt = H psi, where H is the Hamiltonian operator and hbar is the reduced Planck "
"constant. For stationary states this reduces to the eigenvalue problem H psi = E psi, giving discrete "
"energy levels E. A central limit is the Heisenberg uncertainty principle, delta x * delta p >= hbar/2, "
"which forbids simultaneous exact knowledge of position and momentum. Matter also behaves as waves, "
"with the de Broglie wavelength lambda = h / p. For the hydrogen atom the bound-state energies are "
"E_n = -13.6 eV / n^2, where n is the principal quantum number. The probability of finding a particle "
"is the squared magnitude of the wavefunction, |psi|^2.",
"black holes":
"A black hole is a region of spacetime where gravity is so strong that nothing, not even light, can "
"escape. Its boundary is the event horizon. For a non-rotating mass M the horizon lies at the "
"Schwarzschild radius r_s = 2 G M / c^2, where G is the gravitational constant and c the speed of light. "
"Stephen Hawking showed that black holes radiate with a temperature T = hbar c^3 / (8 pi G M k_B), so "
"smaller black holes are hotter and evaporate faster. Their entropy is proportional to the horizon area "
"A: S = k_B c^3 A / (4 G hbar), one quarter of the area in Planck units. At the centre lies a singularity "
"of formally infinite density. Seen from far away, time appears to freeze at the horizon because of "
"gravitational time dilation.",
"particle physics":
"Particle physics studies the elementary constituents of matter and their interactions, summarized by "
"the Standard Model. Matter is built from six quarks and six leptons, and the forces are carried by "
"gauge bosons: the photon, the W and Z bosons, and the gluons. The relativistic energy-momentum "
"relation is E^2 = (p c)^2 + (m c^2)^2, which for a particle at rest gives E = m c^2. Relativistic "
"spin-one-half particles obey the Dirac equation, (i gamma^mu d_mu - m) psi = 0. Particles acquire mass "
"through the Higgs mechanism, confirmed by the discovery of the Higgs boson in 2012 at the Large Hadron "
"Collider. The strength of the electromagnetic interaction is set by the fine-structure constant "
"alpha = e^2 / (4 pi epsilon_0 hbar c), approximately 1/137.",
"astrophysics":
"Astrophysics applies physics to stars, galaxies, and the universe as a whole. The expansion of the "
"cosmos follows the Friedmann equation, (a_dot / a)^2 = 8 pi G rho / 3 - k c^2 / a^2, relating the "
"expansion rate to the energy density rho. Nearby galaxies recede according to Hubble's law, v = H_0 d, "
"where H_0 is the Hubble constant and d the distance. A star shines with luminosity "
"L = 4 pi R^2 sigma T^4 by the Stefan-Boltzmann law. A white dwarf cannot exceed the Chandrasekhar "
"limit of about 1.4 solar masses before collapsing into a neutron star or a black hole. Light from "
"receding objects is stretched to longer wavelengths, an effect called redshift, denoted z.",
}
QUIZ = ["What is the Schrodinger equation?", "What is the Schwarzschild radius?",
        "What is the Hawking temperature of a black hole?", "What is the Chandrasekhar limit?",
        "What is the fine-structure constant?", "What is Hubble's law?"]
RAW_PROBES = ["The Schwarzschild radius is", "The Chandrasekhar limit is about",
              "Hubble's law states that", "The Heisenberg uncertainty principle says",
              "The energy levels of the hydrogen atom are"]

# ---------------------------------------------------------------- run
log("=" * 74)
log(f"PRAGNOSIA PHYSICS CONVERSATION (post controller-fix)   {datetime.datetime.now():%Y-%m-%d %H:%M}")
log("=" * 74)
narrate("STEP 1: wake the brain (length-aware boundary curve + separating seek-match)")
t0 = time.time(); brain = B.Brain()
log(f"brain up: {brain.n_params():,} params | {time.time()-t0:.0f}s")
log(f"calibrated length->boundary curve: " +
    "  ".join(f"{L}tok:{b:.2f}" for L, b in brain._bcurve))
log(f"seek match boundary = {brain.match_threshold:.3f}  (was 0.881, too strict before)")

narrate("STEP 2: baseline sanity -- short familiar prompts should now ANSWER/CONTINUE, "
        "not be mis-routed to 'learn'")
for q in ["Once upon a time", "Question: What is 5 plus 7?\nAnswer:", "def add(a, b):",
          "Water is made of"]:
    turn(brain, q)

narrate("STEP 3: feed the LONG physics passages (with equations). Each is novel -> should "
        "route to LEARN; I record surprise BEFORE and AFTER.")
for topic, text in PASSAGES.items():
    n = len(brain.tok.encode(text).ids)
    before = brain._nll(text)
    log(f"\n[{topic}] {n} tokens | surprise BEFORE = {before:.2f} "
        f"(boundary@{n}tok = {brain.boundary_for(n):.2f})")
    reply, decision = turn(brain, text)
    after = brain._nll(text)
    log(f"[{topic}] surprise AFTER = {after:.2f}  -> drop {before-after:.2f} "
        f"({100*(before-after)/max(before,1e-9):.0f}%, {'LEARNED' if after < before*0.8 else 'little change'})")
    _records.append(dict(topic=topic, n_tok=n, nll_before=round(before,3), nll_after=round(after,3)))

narrate("STEP 4: quiz it on the physics it just learned (controller route + raw weights)")
for q in QUIZ:
    turn(brain, q)
for p in RAW_PROBES:
    raw(brain, p)

narrate("STEP 5: honesty -- things OUTSIDE what it learned should ABSTAIN")
for q in ["who is the president of india", "What is the capital of Australia?",
          "zxqwv flobble granfalloon the"]:
    turn(brain, q)

narrate("STEP 6: forgetting check -- did learning four physics passages damage prior skills?")
for q in ["Question: What is 5 plus 7?\nAnswer:", "def add(a, b):", "Once upon a time"]:
    turn(brain, q)

narrate("STEP 7: introspection + curiosity")
log(f"introspect -> {brain.introspect()}")
opts = ["the Schwarzschild radius", "blorptle fnord wibgly", "the speed of light"]
log(f"curiosity_pick({opts}) -> {brain.curiosity_pick(opts)!r}")
for o in opts: log(f"     uncertainty({o!r}) = {brain.uncertainty(o):.2f}")

narrate("DONE (weights NOT persisted -- test session). saving logs.")
json.dump(_records, open(LOG_JSON, "w"), indent=2)
log(f"\nwrote {LOG_TXT} and {LOG_JSON}")
_fh.close()
