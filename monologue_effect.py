"""Measure the internal monologue's honesty gate: does it actually stop self-poisoning? Drive the real
self-generated thought chain (deliberate -> wonder -> next topic) for K steps, and for each thought compare
the OLD gate (novelty only -> the model teaches ANY novel thought) with the NEW gate (novel AND passes the
truth probe). The number that matters: of the thoughts the old gate would have consolidated, how many does
the honesty gate REJECT as confabulation -> that is the self-poisoning prevented. Also reports the truth-probe
separation between kept and rejected thoughts (kept should look more 'known').

Run (GPU, Unreal closed):  python3 monologue_effect.py [steps]
"""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import brain

STEPS = int(sys.argv[1]) if len(sys.argv) > 1 else 20
SEEDS = ["the world", "language", "memory", "numbers", "the sky", "learning", "myself", "time"]


def main():
    b = brain.Brain(learn=False)
    floor = getattr(b, "_truth_floor", 0.5)
    rows = []
    topic = SEEDS[0]; si = 0
    for k in range(STEPS):
        if not topic:
            si = (si + 1) % len(SEEDS); topic = SEEDS[si]
        thought = b._deliberate(f"Briefly, {topic}:")[:120]
        nov = b._novelty(thought)
        knows = b.truth_probe(thought) if getattr(b, "_truth_probe", None) is not None else 1.0
        ok, why, _, _ = b._worth_consolidating(thought)
        old_gate = nov > b.novelty_min                         # OLD behaviour: consolidate on novelty alone
        rows.append({"topic": topic, "thought": thought[:80], "novelty": round(nov, 3),
                     "knows": round(knows, 3), "old_consolidate": bool(old_gate), "new_consolidate": bool(ok),
                     "reason": why})
        nxt = b.wonder(thought)
        topic = nxt if nxt and nxt.lower() != topic.lower() else None

    old_yes = [r for r in rows if r["old_consolidate"]]
    new_yes = [r for r in rows if r["new_consolidate"]]
    # thoughts the OLD gate would consolidate but the honesty gate REJECTS (self-poisoning prevented)
    poison_blocked = [r for r in old_yes if not r["new_consolidate"]]
    kept_knows = [r["knows"] for r in new_yes]
    blocked_knows = [r["knows"] for r in poison_blocked]
    mean = lambda xs: round(sum(xs) / len(xs), 3) if xs else None

    # honesty-gate discrimination on ALL thoughts (decoupled from the novelty gate, which may block first)
    pass_h = [r for r in rows if r["knows"] > floor]
    fail_h = [r for r in rows if r["knows"] <= floor]
    summary = {
        "steps": STEPS, "truth_floor": floor,
        "honesty_pass": len(pass_h), "honesty_fail": len(fail_h),
        "honesty_block_rate": round(len(fail_h) / max(1, len(rows)), 3),   # fraction rejected as confabulation
        "mean_knows_pass": mean([r["knows"] for r in pass_h]),             # want HIGH (kept look known)
        "mean_knows_fail": mean([r["knows"] for r in fail_h]),             # want LOW (blocked look confabulated)
        "old_gate_consolidated": len(old_yes),                 # novelty-only (may be 0 if novelty_min blocks all)
        "new_gate_consolidated": len(new_yes),                 # novelty + honesty
        "self_poison_blocked": len(poison_blocked),            # novel-but-confabulated, now rejected
    }
    os.makedirs("eval_registry", exist_ok=True)
    json.dump({"summary": summary, "rows": rows}, open("eval_registry/monologue_effect.json", "w"), indent=2)
    print(json.dumps(summary, indent=2), flush=True)
    print("\nblocked (novel but failed honesty -> would have self-poisoned):", flush=True)
    for r in poison_blocked[:6]:
        print(f"  knows={r['knows']:.2f}  {r['thought']!r}", flush=True)


if __name__ == "__main__":
    main()
