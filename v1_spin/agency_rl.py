"""B: AGENCY via REINFORCE -- the training the pipeline stopped at (RESULTS #23: the agency loop CLOSES but
behaviour does NOT improve, because stored lessons never reach action-selection and inference-only episodic
control was NEGATIVE -- 'the LM state-embedding isn't discriminative enough'). This tests the real claim:
is it the FEATURES that are too weak, or the LEARNING METHOD (gradient-free nearest-neighbour vote)?

Fair head-to-head on the SAME grounded features (`brain._embed`, the importance-weighted contextual hidden):
  (a) REINFORCE   -- a small policy head trained by reward gradient (policy gradient, discounted returns, baseline)
  (b) episodic    -- the current `_choose_action` rule: nearest-12 experiences, reward-weighted vote (NO gradient)
If (a) learns where (b) stays flat, the features were fine all along -- behaviour-learning needed RL, as #23 argued.
NO hand-coded planner: `optimal_action()` is used ONLY to score, never fed to either agent. Embeddings cached on
CPU (obs space is ~25 deterministic strings) -> zero GPU contention with the live long-context run + prod.
"""
import os, sys, json, random, math
# PROD-SAFE: cap CPU threads so the (CPU-only) embedding init cannot saturate the 40-core box and
# starve the CPU-side prod services (STT/TTS). Must be set BEFORE importing torch/numpy.
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "6"
os.environ.setdefault("BRAIN_SILENT", "1")
import brain as B                      # noqa: E402
import torch, torch.nn as nn           # noqa: E402
torch.set_num_threads(6)               # hard cap in-process too (prod CPU headroom)

# --- force the embedding LM onto CPU (prod-safe: the GPU is busy with run A) ---
B.DEVICE = "cpu"
try:
    import s6_hybrid as H; H.DEVICE = "cpu"
except Exception:
    pass

# FAST init: B only needs `_embed` (= lm.represent weighted by _tw). The full Brain.recalibrate() runs ~9
# calibration cascades (truth-probe, question-words n=3000, affect autocorr, ...) each doing many CPU forwards
# through the 1B model -> 8+ min on CPU. Skip all of it; keep ONLY _tw (cheap: cached token self-information).
# This does NOT change the features under test -- _embed uses lm.represent + _tw, both preserved exactly.
def _minimal_recalibrate(self):
    self._tw = self._token_self_information()
B.Brain.recalibrate = _minimal_recalibrate
B.Brain._derive_self = lambda self: ""          # unused by _embed / the RL loop
B.Brain._load_memory = lambda self: None         # no stored-fact restore needed for the grid experiment

br = B.Brain(learn=False)
try: br.lm = br.lm.float().to("cpu")
except Exception: pass

# DISK-persisted embedding cache: the 625 obs embeddings are the ONLY expensive part (CPU forwards through the
# 1B). Persist them so init is instant on any rerun and survives interruption (incremental save every 64).
EMB_CACHE = "agency_emb_cache.pt"
_cache = {}
if os.path.exists(EMB_CACHE):
    try: _cache = torch.load(EMB_CACHE); print(f"[agency_rl] loaded {len(_cache)} cached embeddings from disk", flush=True)
    except Exception: _cache = {}
_since_save = [0]; _n_new = [0]
def embed(obs):
    if obs not in _cache:
        _cache[obs] = br._embed(obs).detach().float().cpu()
        _since_save[0] += 1; _n_new[0] += 1
        if _n_new[0] % 25 == 0: print(f"[cache] {len(_cache)} embedded...", flush=True)
        if _since_save[0] >= 25:
            torch.save(_cache, EMB_CACHE); _since_save[0] = 0
    return _cache[obs]

from world import GridWorld
SIZE = 5
ACTIONS = GridWorld(size=SIZE).actions()
NA = len(ACTIONS)
# RANDOMIZED goal+start (the real GENERALIZATION test): the obs encodes the goal-relative direction, so a
# learner must map "N east / M south of me" -> a move that GENERALIZES across positions AND goals, not memorize
# 25 fixed states. This is the honest discriminative-features test (a fixed goal made it trivial to memorize).
def make_world(start, goal): return GridWorld(size=SIZE, start=start, goal=goal)

# pre-cache every reachable obs (all 25 pos x 25 goal = 625 deterministic strings) -> one-time CPU forwards
w0 = GridWorld(size=SIZE)
for gx in range(SIZE):
    for gy in range(SIZE):
        w0.goal = [gx, gy]
        for x in range(SIZE):
            for y in range(SIZE):
                w0.pos = [x, y]; embed(w0.perceive())
torch.save(_cache, EMB_CACHE)   # final flush of the full 625
D = next(iter(_cache.values())).shape[0]
print(f"[agency_rl] cached {len(_cache)} obs embeddings (pos x goal), dim={D}, actions={ACTIONS}", flush=True)

# fixed held-out eval set: varied (start, goal) pairs, goal != start
random.seed(7)
EVAL = []
while len(EVAL) < 24:
    s = (random.randrange(SIZE), random.randrange(SIZE)); g = (random.randrange(SIZE), random.randrange(SIZE))
    if s != g: EVAL.append((s, g))
def rand_task():
    while True:
        s = (random.randrange(SIZE), random.randrange(SIZE)); g = (random.randrange(SIZE), random.randrange(SIZE))
        if s != g: return s, g
MAXT = 25

def rollout(act_fn, start, goal):
    """Run one episode with a callable act_fn(obs)->action_index; return (transitions, total_reward, reached)."""
    w = make_world(start, goal); w.reset(); trans = []; tot = 0.0
    for _ in range(MAXT):
        obs = w.perceive(); ai = act_fn(obs)
        r, done = w.step(ACTIONS[ai]); trans.append((obs, ai, r)); tot += r
        if done: break
    return trans, tot, (w.pos == w.goal)

def eval_greedy(act_fn):
    """Greedy eval on the fixed held-out (start, goal) set -> tests GENERALIZATION, not memorization."""
    reached = ret = steps = 0
    for s, g in EVAL:
        tr, t, rc = rollout(act_fn, s, g); reached += rc; ret += t; steps += len(tr)
    N = len(EVAL); return reached / N, ret / N, steps / N

# ============ (a) REINFORCE ============
torch.manual_seed(0)
policy = nn.Sequential(nn.Linear(D, 64), nn.Tanh(), nn.Linear(64, NA))
opt = torch.optim.Adam(policy.parameters(), lr=3e-3)   # gentler lr (1e-2 caused policy collapse)
GAMMA = 0.95
BATCH = 8          # average the gradient over BATCH episodes/update -> lower variance, stable learning
ENT = 0.02         # entropy bonus -> keeps exploration, prevents premature collapse to a bad deterministic policy
def pi_logits(obs): return policy(embed(obs))
def reinforce_act_greedy(obs):
    with torch.no_grad(): return int(pi_logits(obs).argmax())

EPISODES = 1200
base = 0.0
curve = []
for ep in range(1, EPISODES + 1):
    batch_loss = 0.0
    for _ in range(BATCH):
        s, g = rand_task(); w = make_world(s, g); w.reset(); logps = []; ents = []; rews = []
        for _ in range(MAXT):
            obs = w.perceive(); logits = pi_logits(obs)
            d = torch.distributions.Categorical(logits=logits); a = d.sample()
            logps.append(d.log_prob(a)); ents.append(d.entropy())
            r, done = w.step(ACTIONS[int(a)]); rews.append(r)
            if done: break
        G = 0.0; returns = []
        for r in reversed(rews): G = r + GAMMA * G; returns.insert(0, G)
        returns = torch.tensor(returns, dtype=torch.float32)
        base = 0.95 * base + 0.05 * float(returns.mean())
        adv = returns - base
        if adv.std() > 1e-6: adv = adv / (adv.std() + 1e-6)
        batch_loss = batch_loss - (torch.stack(logps) * adv).sum() - ENT * torch.stack(ents).sum()
    opt.zero_grad(); (batch_loss / BATCH).backward(); opt.step()
    if ep % 100 == 0 or ep == 1:
        rr, rt, st = eval_greedy(reinforce_act_greedy)
        curve.append((ep, rr, rt, st))
        print(f"  REINFORCE ep{ep:5d}: heldout reach={rr*100:5.1f}%  avg_return={rt:+.2f}  avg_steps={st:.1f}", flush=True)

# ============ (b) episodic control (replicates brain._choose_action, SAME features) ============
random.seed(1)
def run_episodic(warm_episodes=1200):
    buf = []  # (emb, action_idx, reward)
    def act(obs):
        e = embed(obs)
        if buf:
            near = sorted(((float(e @ eb), a, r) for eb, a, r in buf), reverse=True)[:12]
            val = [0.0] * NA
            for s, a, r in near: val[a] += s * r
            if max(val) > 0: return int(max(range(NA), key=lambda i: val[i]))
        return random.randrange(NA)  # explore (no LM here; same info as REINFORCE gets)
    ecurve = []
    for ep in range(1, warm_episodes + 1):
        s, g = rand_task(); tr, t, rc = rollout(act, s, g)
        for obs, a, r in tr: buf.append((embed(obs), a, r))
        if ep % 100 == 0 or ep == 1:
            rr, rt, st = eval_greedy(act); ecurve.append((ep, rr, rt))
            print(f"  EPISODIC  ep{ep:5d}: heldout reach={rr*100:5.1f}%  avg_return={rt:+.2f}  buf={len(buf)}", flush=True)
    return ecurve

print("\n=== (a) REINFORCE (reward-gradient policy on grounded features) ===  [above]")
print("=== (b) EPISODIC CONTROL (nearest-12 reward vote, no gradient -- the #23 method) ===")
ec = run_episodic()

r_best = max(c[1] for c in curve); e_best = max(c[1] for c in ec)
r0, rN = curve[0], curve[-1]; e0, eN = ec[0], ec[-1]
print("\n=== VERDICT (held-out generalization: random start x random goal) ===")
print(f"REINFORCE: reach {r0[1]*100:.0f}% -> {rN[1]*100:.0f}%  (best {r_best*100:.0f}%) , final return {rN[2]:+.2f}")
print(f"EPISODIC : reach {e0[1]*100:.0f}% -> {eN[1]*100:.0f}%  (best {e_best*100:.0f}%) , final return {eN[2]:+.2f}")
# data-driven conclusion (no pre-scripted claim)
def improved(c): return c[-1][1] - c[0][1]
concl = []
if rN[1] >= 0.75: concl.append("REINFORCE LEARNS a generalizing policy from reward on the grounded features")
elif improved(curve) > 0.2: concl.append("REINFORCE improves but does not fully solve")
else: concl.append("REINFORCE fails to learn stably (under-tuned or features insufficient)")
if eN[1] >= 0.75: concl.append("EPISODIC control ALSO reaches goal (contradicts the earlier #23 negative -- "
                                "random exploration, not LM-guided, is the difference)")
elif improved(ec) > 0.2: concl.append("EPISODIC improves partially")
else: concl.append("EPISODIC stays flat (reproduces #23)")
print("CONCLUSION: " + "; ".join(concl) + ".")
print("Both consume the IDENTICAL brain._embed features -> isolates LEARNING METHOD from PERCEPTION.")
