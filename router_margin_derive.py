"""C2: pick the routing margin by a PRINCIPLED derivation, not by tuning on the eval. Compute several
candidate derivations of the embedding 'noise floor' from fit-time data only (seeds + mined corpus), then
report each candidate's value AND its held-out metrics. We commit to the DERIVATION RULE, and report the
held-out number it yields honestly (it is not chosen to maximize that number).

Run (GPU, Unreal closed):  python3 router_margin_derive.py
"""
import sys, os, json, statistics as st
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import brain
from router_heldout import HELDOUT


def main():
    b = brain.Brain(learn=False)
    b._fit_intent_router()
    seeds = b._intent_seeds
    others = seeds["_other"]
    ctrl = {k: v for k, v in seeds.items() if k != "_other"}

    # candidate margins, all from fit data (no peeking at HELDOUT):
    # 1) pairwise-sim std among _other corpus questions = the embedding noise floor for unrelated text
    pair = [float(others[i] @ others[j]) for i in range(len(others)) for j in range(i + 1, len(others))]
    m_paired_std = st.pstdev(pair) if len(pair) > 1 else 0.0
    # 2) 1 std above the mean cross-similarity, minus a within-controller-class cohesion reference
    #    within-class cohesion: mean nearest-mate sim inside each controller intent
    coh = []
    for embs in ctrl.values():
        for i, e in enumerate(embs):
            if len(embs) > 1:
                coh.append(max(float(e @ o) for j, o in enumerate(embs) if j != i))
    m_cohesion_gap = max(0.0, (st.mean(coh) - st.mean(pair))) if coh and pair else 0.0
    # 3) half the cohesion gap (a real controller match should be at least halfway from noise to in-class)
    m_half_cohesion = m_cohesion_gap / 2

    cands = {"paired_std": m_paired_std, "cohesion_gap": m_cohesion_gap, "half_cohesion": m_half_cohesion}

    # precompute held-out gaps once
    probes = []
    for text, exp in HELDOUT:
        q = b._embed(text)
        bo = max(float(q @ e) for e in others)
        bc, bi = -1.0, None
        for intent, embs in ctrl.items():
            s = max(float(q @ e) for e in embs)
            if s > bc: bc, bi = s, intent
        probes.append((exp, bc - bo, bi))

    def metrics(m):
        h = ph = pt = nh = nt = 0
        for exp, gap, bi in probes:
            got = bi if gap > m else None
            ok = got == exp; h += ok
            if exp is None: nt += 1; nh += ok
            else: pt += 1; ph += ok
        return {"acc": round(h / len(probes), 3), "ctrl_recall": f"{ph}/{pt}",
                "neg_spec": f"{nh}/{nt}", "spec_rate": round(nh / nt, 3), "recall_rate": round(ph / pt, 3)}

    report = {name: {"margin": round(v, 4), **metrics(v)} for name, v in cands.items()}
    print(json.dumps(report, indent=2), flush=True)
    json.dump(report, open("eval_registry/router_margin_derive.json", "w"), indent=2)


if __name__ == "__main__":
    main()
