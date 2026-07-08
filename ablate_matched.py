"""Matched-fraction + dose-response ablation (T3.1) -- fixes the confound the critic flagged: the headline
carrier-ablation (all 22 spin layers) vs attention-ablation (7 layers) isn't apples-to-apples. Here BOTH
mixers are ablated identically (zero the mixer sublayer, KEEP the MLP), and we report:
  - a DOSE-RESPONSE curve (ablate k spin carriers, k = 1..22)
  - MATCHED 7-spin vs 7-attention (same layer count) -> is the gap intrinsic or just layer count?

Run (GPU): python3 ablate_matched.py
"""
import json, os, random
import torch
import s6_hybrid as H

c = json.load(open("pragnosia.json")); H.VOC, H.L = c["vocab"], c["ctx"]
DEV = "cuda"


def main():
    m = H.SpinAttentionLM(c["vocab"], c["d"], c["heads"], c["layers"],
                          mlp_mult=c["mlp_mult"], carrier=c["carrier"])          # build on CPU
    m.load_state_dict(torch.load(c["ckpt"], map_location="cpu", weights_only=True))
    m = m.bfloat16().to(DEV).eval()                                             # -> bf16 -> GPU (one copy, ~2GB)
    vd = H.load(c["valid_bin"])
    spin = [i for i, b in enumerate(m.blocks) if isinstance(b, H.SpinBlock)]
    attn = [i for i, b in enumerate(m.blocks) if isinstance(b, H.Block)]
    base = float(H.val_ppl(m, vd, iters=60))

    def ablate_spin(idxs):                                # zero the carrier gate (keep the block's MLP)
        sv = {}
        for i in idxs:
            sv[i] = float(m.blocks[i].carrier.gate.item())
            with torch.no_grad(): m.blocks[i].carrier.gate.fill_(-30.0)
        return sv

    def restore_spin(sv):
        for i, v in sv.items():
            with torch.no_grad(): m.blocks[i].carrier.gate.fill_(v)

    def ablate_attn(idxs):                                # zero the attention sublayer (keep the MLP)
        sv = {}
        for i in idxs:
            sv[i] = m.blocks[i].attn.forward
            m.blocks[i].attn.forward = (lambda x: torch.zeros_like(x))
        return sv

    def restore_attn(sv):
        for i, f in sv.items():
            m.blocks[i].attn.forward = f

    def xworse(idxs, kind):
        sv = ablate_spin(idxs) if kind == "spin" else ablate_attn(idxs)
        p = float(H.val_ppl(m, vd, iters=60))
        (restore_spin if kind == "spin" else restore_attn)(sv)
        return round(p / base, 1)

    R = {"base_ppl": round(base, 2), "n_spin": len(spin), "n_attn": len(attn)}
    random.seed(0)
    # dose-response: k evenly-spaced spin carriers
    dose = {}
    for k in [1, 2, 4, 7, 12, len(spin)]:
        picks = [spin[int(j * (len(spin) - 1) / max(1, k - 1))] for j in range(k)] if k > 1 else [spin[len(spin) // 2]]
        dose[k] = xworse(sorted(set(picks)), "spin")
    R["spin_dose_response_xworse"] = dose
    R["ablate_7_spin_matched"] = xworse(spin[:7], "spin")      # matched to attention's 7 layers
    R["ablate_7_attn"] = xworse(attn, "attn")
    R["ablate_all_spin"] = xworse(spin, "spin")
    os.makedirs("eval_registry", exist_ok=True)
    json.dump(R, open("eval_registry/ablate_matched.json", "w"), indent=2)
    print(json.dumps(R, indent=2), flush=True)


if __name__ == "__main__":
    main()
