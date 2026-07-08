"""Faculty-value ablation (T1.6) -- measure what each faculty actually BUYS, ON vs OFF, on a small battery.
Turns 'we built faculties' into 'tools = +N on arithmetic, semantic-entropy = +M separation'. The missing
science the critic flagged. Run (GPU, Unreal closed): python3 faculty_ablate.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import brain

ARITH = [("23 + 45", "68"), ("9 * 7", "63"), ("100 - 37", "63"), ("144 / 12", "12"),
         ("15 + 27", "42"), ("8 * 12", "96"), ("250 - 175", "75")]
KNOWN = ["What is the capital of France", "What is 2 plus 2", "Who wrote Hamlet"]
NONSENSE = ["What is the flarn of a quix", "What is the zorbal index of Mars", "What is the glorb of a snee"]


def main():
    b = brain.Brain(learn=False)
    R = {}

    # --- TOOLS (calc) ON vs OFF: exact-arithmetic accuracy ---
    on = sum(str(b._tool_calc(q)) == a for q, a in ARITH)
    off = sum(a in b.generate_text(f"<user> What is {q}? <assistant>", n=10) for q, a in ARITH)
    R["tools_arithmetic"] = {"faculty_on": f"{on}/{len(ARITH)}", "faculty_off": f"{off}/{len(ARITH)}",
                             "delta": on - off}

    # --- CALIBRATION: semantic-entropy vs old token-consistency, separation on known vs nonsense ---
    se = lambda q: b._semantic_entropy(f"<user> {q}? <assistant>", k=5, n=16)[0]
    tc = lambda q: b._self_consistency(f"<user> {q}? <assistant>", k=5, n=16)[0]
    k_se, n_se = sum(map(se, KNOWN)) / len(KNOWN), sum(map(se, NONSENSE)) / len(NONSENSE)
    k_tc, n_tc = sum(map(tc, KNOWN)) / len(KNOWN), sum(map(tc, NONSENSE)) / len(NONSENSE)
    R["calibration"] = {
        "semantic_entropy": {"known": round(k_se, 3), "nonsense": round(n_se, 3), "separation": round(n_se - k_se, 3)},
        "token_consistency": {"known": round(k_tc, 3), "nonsense": round(n_tc, 3), "separation": round(k_tc - n_tc, 3)},
        "note": "semantic-entropy separation should be LARGER (nonsense higher); token-consistency often fails to separate",
    }

    # --- DELIBERATION (latent + step-by-step) ON vs OFF on a multi-step word problem ---
    WORD = [("If Tom has 5 apples, buys 3 more, and eats 2, how many apples does he have", "6"),
            ("A box has 4 rows of 3 balls each, how many balls in total", "12")]
    delib_on = sum(a in b._deliberate(q) for q, a in WORD)
    delib_off = sum(a in b.generate_text(f"<user> {q}? <assistant>", n=12) for q, a in WORD)
    R["deliberation_multistep"] = {"faculty_on": f"{delib_on}/{len(WORD)}", "faculty_off": f"{delib_off}/{len(WORD)}",
                                   "delta": delib_on - delib_off}

    import json
    os.makedirs("eval_registry", exist_ok=True)
    json.dump(R, open("eval_registry/faculty_ablation.json", "w"), indent=2)
    print(json.dumps(R, indent=2), flush=True)


if __name__ == "__main__":
    main()
