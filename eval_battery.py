"""Expanded eval battery (C1: kill the anecdotes) -- larger n on every axis so numbers carry statistical
weight, plus the T1.3 activation-probe calibration (the signal that WORKS) reported as a discrimination
metric, not a single scalar. Also serves C3: measures the honesty gate's false-abstain rate on facts the
model plausibly knows vs true-abstain on nonsense, so we can see over-abstention and tune the floor.

Run (GPU, Unreal closed):  python3 eval_battery.py [label]
Writes eval_registry/<label>_battery.json.
"""
import sys, os, json, contextlib
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
import brain, s6_hybrid as H, train_pragnosia as TP

# ---- graded multi-hop, expanded (answer = substring we look for in a greedy completion) ----
MULTIHOP = {
    "1hop": [("The capital of France is", "paris"), ("The chemical symbol for oxygen is", "o"),
             ("The largest planet in the solar system is", "jupiter"), ("Water is made of hydrogen and", "oxygen"),
             ("The capital of Japan is", "tokyo"), ("The capital of Italy is", "rome"),
             ("The capital of Russia is", "moscow"), ("The sun rises in the", "east"),
             ("The chemical symbol for gold is", "au"), ("The first president of the United States was",
             "washington"), ("The freezing point of water in Celsius is", "0"),
             ("The planet we live on is called", "earth"), ("The opposite of hot is", "cold"),
             ("The color of a clear daytime sky is", "blue")],
    "2hop": [("The capital of the country where the Eiffel Tower stands is", "paris"),
             ("The capital of the country where the Great Wall was built is", "beijing"),
             ("The planet closest to the star at the center of our solar system is", "mercury"),
             ("The language mainly spoken in the country whose capital is Tokyo is", "japanese"),
             ("The currency of the country whose capital is London is", "pound"),
             ("The color you get by mixing blue and yellow is", "green"),
             ("The capital of the country directly south of the United States is", "mexico"),
             ("The ocean between Europe and North America is the", "atlantic")],
    "3hop": [("The first letter of the name of the largest planet is", "j"),
             ("The capital of the country whose flag is blue white and red and borders Spain is", "paris"),
             ("The number of letters in the capital city of France is", "five"),
             ("The first letter of the chemical element whose symbol is O is", "o"),
             ("The capital of the continent-country whose largest city is Sydney is", "canberra")],
}

# ---- calibration: T1.3 activation probe on KNOWN (should score high), NONSENSE (low), HALF-KNOWN (the
#      over-abstention risk -- real facts the small model may or may not hold) ----
KNOWN = ["What is the capital of France", "What is 2 plus 2", "What is the capital of Japan",
         "What color is the sky", "What is the chemical symbol for oxygen", "What is the largest planet",
         "How many days are in a week", "What is the opposite of hot", "What is the capital of Italy",
         "What is water made of", "Where does the sun rise", "What is the capital of Russia"]
NONSENSE = ["What is the flarn of a quix", "What is the zorbal index of Mars", "What is the glorb of a snee",
            "What is the plimth rating of a wodget", "What is the quexal mass of a florn",
            "What is the drindle of a plonk", "What is the yotch coefficient of a squib",
            "What is the vorpal depth of a snicker", "What is the blang of a fentle",
            "What is the crumple factor of a whibble", "What is the naxil of a grommet",
            "What is the sprocket index of a florb"]
HALF = ["Who wrote Hamlet", "Who painted the Mona Lisa", "What year did World War Two end",
        "What is the boiling point of water in Celsius", "Who discovered gravity",
        "What is the speed of light", "What is the capital of Australia", "Who wrote Romeo and Juliet",
        "What is the tallest mountain on Earth", "What is the smallest prime number",
        "How many continents are there", "What is the chemical formula for salt"]


def main():
    label = sys.argv[1] if len(sys.argv) > 1 else "current"
    b = brain.Brain(learn=False)
    cfg = b.cfg
    H.VOC, H.L, TP.VOC = cfg["vocab"], cfg["ctx"], cfg["vocab"]
    nc = contextlib.nullcontext()

    # 1) perplexity curve (std + long-context carry) -- more iters for a stabler estimate
    vd = H.load(cfg["valid_bin"])
    ppl = {}
    with torch.no_grad():
        for W in (1, 8, 32):
            ppl[f"W{W}"] = round(TP._val_long(b.lm, vd, W, cfg["ctx"], 2, nc, iters=20), 2)

    # 2) graded multi-hop (report as fraction AND raw counts so n is explicit)
    def ask(q):
        return b.generate_text(f"<user> {q}? <assistant>", n=16).lower()
    mh = {}
    for lvl, probes in MULTIHOP.items():
        hit = sum(ans in ask(q) for q, ans in probes)
        mh[lvl] = {"acc": round(hit / len(probes), 3), "hit": hit, "n": len(probes)}

    # 3) T1.3 activation-probe calibration as a DISCRIMINATION metric (not one scalar).
    #    floor = b._truth_floor (0.5 by construction). Report per-set mean + the rate each set is
    #    on the "knows" side of the floor -> false-abstain (known below floor) and true-abstain (nonsense below).
    floor = getattr(b, "_truth_floor", 0.5)
    def probe_set(qs):
        vals = [b.truth_probe(f"<user> {q}? <assistant>") for q in qs]
        above = sum(v > floor for v in vals)
        return {"mean": round(sum(vals) / len(vals), 3), "above_floor": above, "n": len(vals),
                "vals": [round(v, 3) for v in vals]}
    known_p, nonsense_p, half_p = probe_set(KNOWN), probe_set(NONSENSE), probe_set(HALF)
    cal = {
        "floor": floor,
        "known": known_p, "nonsense": nonsense_p, "half_known": half_p,
        "known_answer_rate": round(known_p["above_floor"] / known_p["n"], 3),       # want HIGH
        "known_false_abstain_rate": round(1 - known_p["above_floor"] / known_p["n"], 3),  # want LOW
        "nonsense_true_abstain_rate": round(1 - nonsense_p["above_floor"] / nonsense_p["n"], 3),  # want HIGH
        "separation": round(known_p["mean"] - nonsense_p["mean"], 3),               # want >0
    }

    out = {"label": label, "params_M": round(b.n_params() / 1e6),
           "ppl": ppl, "multihop": mh, "calibration_activation_probe": cal}
    os.makedirs("eval_registry", exist_ok=True)
    json.dump(out, open(f"eval_registry/{label}_battery.json", "w"), indent=2)
    print(json.dumps(out, indent=2), flush=True)


if __name__ == "__main__":
    main()
