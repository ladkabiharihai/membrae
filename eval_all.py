"""Canonical eval harness (T1.5) -- ONE source of truth for every checkpoint metric, so paper/site numbers
never drift (no more scattered 24.54 / 20.1 / 17.8 / 18.56). Given the active model, compute perplexity curve,
graded multi-hop (T1.7), and calibration separation (T1.1), and write a versioned JSON to eval_registry/.

Run (GPU, Unreal closed):  python3 eval_all.py [label]
"""
import sys, os, json, contextlib
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
import brain, s6_hybrid as H, train_pragnosia as TP

# ---- graded multi-hop probes (T1.7): the answer is the substring we look for in a greedy completion ----
MULTIHOP = {
    "1hop": [("The capital of France is", "paris"), ("The chemical symbol for oxygen is", "o"),
             ("The largest planet in the solar system is", "jupiter"), ("Water is made of hydrogen and", "oxygen")],
    "2hop": [("The capital of the country where the Eiffel Tower stands is", "paris"),
             ("The color of the sky on a clear day, mixed with yellow, gives", "green"),
             ("The planet closest to the star at the center of our solar system is", "mercury")],
    "3hop": [("The capital of the country whose flag is red white and blue and borders Spain is", "paris"),
             ("The first letter of the name of the largest planet is", "j")],
}


def main():
    label = sys.argv[1] if len(sys.argv) > 1 else "current"
    b = brain.Brain(learn=False)
    cfg = b.cfg
    H.VOC, H.L, TP.VOC = cfg["vocab"], cfg["ctx"], cfg["vocab"]     # the eval path reads module-level VOC
    nc = contextlib.nullcontext()

    # 1) perplexity curve (std + long-context carry)
    vd = H.load(cfg["valid_bin"])
    ppl = {}
    with torch.no_grad():
        for W in (1, 8, 32):
            ppl[f"W{W}"] = round(TP._val_long(b.lm, vd, W, cfg["ctx"], 2, nc, iters=6), 2)

    # 2) graded multi-hop (T1.7): does accuracy fall off with hop-count? (flat-with-tokens later = falsifies 'just undertrained')
    def ask(q):
        return b.generate_text(f"<user> {q}? <assistant>", n=16).lower()
    mh = {lvl: f"{sum(ans in ask(q) for q, ans in probes)}/{len(probes)}" for lvl, probes in MULTIHOP.items()}

    # 3) calibration separation (T1.1): known should be LOW semantic entropy, nonsense HIGH
    known = ["What is the capital of France", "What is 2 plus 2", "Who wrote Hamlet"]
    nonsense = ["What is the flarn of a quix", "What is the zorbal index of Mars"]
    se = lambda q: b._semantic_entropy(f"<user> {q}? <assistant>", k=5, n=16)[0]
    cal = {"known_sem_entropy": round(sum(map(se, known)) / len(known), 3),
           "nonsense_sem_entropy": round(sum(map(se, nonsense)) / len(nonsense), 3)}
    cal["separation"] = round(cal["nonsense_sem_entropy"] - cal["known_sem_entropy"], 3)   # >0 = the signal works

    out = {"label": label, "params_M": round(b.n_params() / 1e6),
           "abstain_threshold": round(b.abstain_threshold, 2), "match_threshold": round(b.match_threshold, 2),
           "ppl": ppl, "multihop": mh, "calibration": cal}
    os.makedirs("eval_registry", exist_ok=True)
    json.dump(out, open(f"eval_registry/{label}.json", "w"), indent=2)
    print(json.dumps(out, indent=2), flush=True)


if __name__ == "__main__":
    main()
