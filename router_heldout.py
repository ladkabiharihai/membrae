"""C2: router generalization on a HELD-OUT set. The prior 14/14 was on cases the _OTHER_SEED was tuned
around -- that is fitting the eval, not passing it. Here every probe is a fresh paraphrase or a new
negative that appears in NEITHER _INTENTS nor _OTHER_SEED, so this measures whether the nearest-example
router actually generalizes. Reports per-class accuracy + a confusion list of every miss.

Run (GPU, Unreal closed):  python3 router_heldout.py
"""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import brain

# (input, expected_route)  --  None = should go to the LM (persona/greeting/world), else the controller intent.
# NONE of these strings are seeds or _OTHER_SEED entries; they are held-out paraphrases / new negatives.
HELDOUT = [
    # --- controller intents: paraphrases distinct from the seed wording ---
    ("in what areas do you struggle", "self.cant"),
    ("where do you tend to fail", "self.cant"),
    ("do you have subjective experience", "self.conscious"),
    ("is there something it is like to be you", "self.conscious"),
    ("what are you trying to achieve", "self.goals"),
    ("what is your objective", "self.goals"),
    ("what is your emotional state", "self.feel"),
    ("are you in a good mood", "self.feel"),
    ("recap our chat so far", "memory.talk"),
    ("summarize what we have covered", "memory.talk"),
    ("what facts did I give you", "memory.told"),
    ("what have I informed you of", "memory.told"),
    ("what questions did I pose", "memory.asked"),
    ("list what I have asked", "memory.asked"),
    ("how did you come to know that", "provenance"),
    ("cite where that came from", "provenance"),
    ("what is occupying your attention", "workspace"),
    ("what idea are you dwelling on", "workspace"),
    # --- negatives that must go to the LM (persona / greeting / world facts), all held-out ---
    ("what is your name", None),
    ("are you a robot", None),
    ("hey there", None),
    ("good evening", None),
    ("how is it going", None),
    ("what causes rain", None),
    ("who painted starry night", None),
    ("explain photosynthesis", None),
    ("what is the capital of Brazil", None),
    ("how do airplanes fly", None),
    ("tell me a joke", None),
    ("what is the meaning of life", None),
]


def main():
    b = brain.Brain(learn=False)
    b._fit_intent_router()
    rows, hits, misses = [], 0, []
    pos_hit = pos_tot = neg_hit = neg_tot = 0
    for text, exp in HELDOUT:
        got = b._route_intent(text)
        ok = (got == exp)
        hits += ok
        if exp is None:
            neg_tot += 1; neg_hit += ok
        else:
            pos_tot += 1; pos_hit += ok
        if not ok:
            misses.append({"text": text, "expected": exp, "got": got})
        rows.append({"text": text, "expected": exp, "got": got, "ok": ok})

    out = {
        "n": len(HELDOUT),
        "accuracy": round(hits / len(HELDOUT), 3),
        "controller_recall": f"{pos_hit}/{pos_tot}",       # did real controller intents route correctly
        "negative_specificity": f"{neg_hit}/{neg_tot}",    # did LM-bound inputs correctly go to None
        "misses": misses,
        "rows": rows,
    }
    os.makedirs("eval_registry", exist_ok=True)
    json.dump(out, open("eval_registry/router_heldout.json", "w"), indent=2)
    print(json.dumps({k: out[k] for k in ("n", "accuracy", "controller_recall", "negative_specificity", "misses")},
                     indent=2), flush=True)


if __name__ == "__main__":
    main()
