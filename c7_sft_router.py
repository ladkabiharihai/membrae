"""C7: is the SFT adapter shippable now that the router is precision-first? The prior blocker was that
merging the SFT LoRA shifted the embedding geometry and broke the (then argmax) router. The router is now
margin-gated with the margin RE-DERIVED at fit time, so it should re-adapt to the merged geometry. Test:
measure fluency + held-out routing BEFORE, apply+merge the adapter, RE-FIT the router, measure AFTER.
Ship only if specificity holds.

Run (GPU, Unreal closed):  python3 c7_sft_router.py
"""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch, brain, lora
from router_heldout import HELDOUT

FLUENCY = ["Hi", "Who are you?", "What is the capital of France?", "Who wrote Hamlet?",
           "What is photosynthesis?", "What are you bad at?"]


def route_metrics(b):
    b._fit_intent_router()
    ph = pt = nh = nt = 0
    for text, exp in HELDOUT:
        got = b._route_intent(text)
        if exp is None: nt += 1; nh += (got == exp)
        else: pt += 1; ph += (got == exp)
    return {"controller_recall": f"{ph}/{pt}", "neg_specificity": f"{nh}/{nt}",
            "spec_rate": round(nh / nt, 3), "recall_rate": round(ph / pt, 3)}


def fluency(b):
    out = {}
    for q in FLUENCY:
        a = b.generate_text(f"<user> {q} <assistant>", n=28).strip()
        out[q] = {"answer": a[:70], "words": len(a.split())}
    return out


def main():
    b = brain.Brain(learn=False)
    before = {"route": route_metrics(b), "fluency": fluency(b),
              "route_margin": round(getattr(b, "_route_margin", 0.0), 4)}

    # apply the SFT adapter (trained r=16) and merge it into the base weights
    adapter_path = "pragnosia_sft_lora.pt"
    if not os.path.exists(adapter_path):
        print("no adapter; skipping"); return
    n = lora.inject_lora(b.lm, r=16)
    sd = torch.load(adapter_path, map_location="cpu", weights_only=True)
    sd = {k: v.to(next(b.lm.parameters()).device).to(next(b.lm.parameters()).dtype) for k, v in sd.items()}
    missing, unexpected = b.lm.load_state_dict(sd, strict=False)
    loaded = sum(1 for k in sd)                                   # A/B tensors filled
    lora.merge_and_unload(b.lm)                                   # fold adapters into base -> plain Linears
    b._truth_probe = None                                         # force re-calibration on merged weights
    b.calibrate_truth_probe()

    after = {"route": route_metrics(b), "fluency": fluency(b),
             "route_margin": round(getattr(b, "_route_margin", 0.0), 4),
             "adapter_layers": n, "adapter_tensors_loaded": loaded}

    # ship criterion: specificity must not regress below the pre-merge value (recall may move)
    ship = after["route"]["spec_rate"] >= before["route"]["spec_rate"]
    words_before = sum(v["words"] for v in before["fluency"].values())
    words_after = sum(v["words"] for v in after["fluency"].values())

    R = {"before": before, "after": after,
         "verdict": {"specificity_holds": ship,
                     "spec_before": before["route"]["spec_rate"], "spec_after": after["route"]["spec_rate"],
                     "recall_before": before["route"]["recall_rate"], "recall_after": after["route"]["recall_rate"],
                     "total_words_before": words_before, "total_words_after": words_after,
                     "shippable": ship}}
    os.makedirs("eval_registry", exist_ok=True)
    json.dump(R, open("eval_registry/c7_sft_router.json", "w"), indent=2)
    print(json.dumps(R["verdict"], indent=2), flush=True)
    print("\n-- fluency AFTER --", flush=True)
    for q, v in after["fluency"].items():
        print(f"  {q!r:28} ({v['words']:2}w) -> {v['answer']!r}", flush=True)


if __name__ == "__main__":
    main()
