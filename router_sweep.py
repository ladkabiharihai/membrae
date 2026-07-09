"""C2: does the embedding geometry support a good router operating point? For each held-out probe compute
bc = best controller-intent seed similarity, bo = best _other seed similarity. A margin rule routes to the
controller only if bc > bo + m (else -> LM/None). Sweep m and report accuracy / controller-recall /
negative-specificity, so we pick an operating point from DATA, not by curating the test.

Run (GPU, Unreal closed):  python3 router_sweep.py
"""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import brain
from router_heldout import HELDOUT


def main():
    b = brain.Brain(learn=False)
    b._fit_intent_router()
    seeds = b._intent_seeds
    other = seeds["_other"]
    ctrl = {k: v for k, v in seeds.items() if k != "_other"}

    # precompute bc, bo, and winning controller intent per probe
    probes = []
    for text, exp in HELDOUT:
        q = b._embed(text)
        bo = max(float(q @ e) for e in other)
        bc, bc_int = -1.0, None
        for intent, embs in ctrl.items():
            s = max(float(q @ e) for e in embs)
            if s > bc: bc, bc_int = s, intent
        probes.append({"text": text, "exp": exp, "bc": bc, "bo": bo, "bc_int": bc_int, "gap": bc - bo})

    # gap distribution: positives (real controller intents) should have HIGH bc-bo; negatives LOW/negative
    pos_gaps = sorted(p["gap"] for p in probes if p["exp"] is not None)
    neg_gaps = sorted(p["gap"] for p in probes if p["exp"] is None)

    def eval_margin(m):
        hits = pos_hit = pos_tot = neg_hit = neg_tot = 0
        for p in probes:
            got = p["bc_int"] if p["gap"] > m else None
            ok = (got == p["exp"])
            hits += ok
            if p["exp"] is None:
                neg_tot += 1; neg_hit += ok
            else:
                pos_tot += 1; pos_hit += ok
        return {"margin": round(m, 3), "acc": round(hits / len(probes), 3),
                "ctrl_recall": f"{pos_hit}/{pos_tot}", "neg_specificity": f"{neg_hit}/{neg_tot}",
                "neg_spec_rate": round(neg_hit / neg_tot, 3), "ctrl_recall_rate": round(pos_hit / pos_tot, 3)}

    sweep = [eval_margin(m) for m in [0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.10, 0.12, 0.15]]
    # best operating point: maximize (neg_spec + ctrl_recall), tie-break toward higher specificity
    best = max(sweep, key=lambda r: (r["neg_spec_rate"] + r["ctrl_recall_rate"], r["neg_spec_rate"]))

    out = {"pos_gap_quartiles": [round(x, 3) for x in pos_gaps],
           "neg_gap_quartiles": [round(x, 3) for x in neg_gaps],
           "sweep": sweep, "best": best}
    os.makedirs("eval_registry", exist_ok=True)
    json.dump(out, open("eval_registry/router_sweep.json", "w"), indent=2)
    print(json.dumps(out, indent=2), flush=True)


if __name__ == "__main__":
    main()
