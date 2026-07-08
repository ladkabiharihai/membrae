"""Mechanistic probes (T1.8 + T3.2).

T1.8 -- does the model COMPUTE the 2-hop intermediate? Logit-lens each layer's last-token hidden: if the
intermediate entity (e.g. 'France' for 'capital of the country where the Eiffel Tower stands') ever ranks high
at a mid layer, the model computes it; if it never surfaces, multi-hop fails at the COMPUTE step (depth/arch),
not the chaining -- diagnostic, not a shield.

T3.2 -- activation patching (ROME-style): run a clean prompt and a corrupted one, patch each block's clean
output into the corrupted run, and measure how much it restores the correct answer logit. Localizes WHICH
layers causally carry the computation (spin vs attention).

Run (GPU): python3 mech_probe.py
"""
import json, os
import torch
import s6_hybrid as H
from tokenizers import Tokenizer

c = json.load(open("pragnosia.json")); H.VOC, H.L = c["vocab"], c["ctx"]
DEV = "cuda"
tok = Tokenizer.from_file(c["tokenizer"])


def tid(word):
    ids = tok.encode(" " + word.strip()).ids
    return ids[0] if ids else 0


def load():
    m = H.SpinAttentionLM(c["vocab"], c["d"], c["heads"], c["layers"], mlp_mult=c["mlp_mult"], carrier=c["carrier"])
    m.load_state_dict(torch.load(c["ckpt"], map_location="cpu", weights_only=True))
    return m.bfloat16().to(DEV).eval()


@torch.no_grad()
def capture(m, ids):
    acts = []
    hs = [b.register_forward_hook(lambda mod, i, o: acts.append((o[0] if isinstance(o, tuple) else o).detach()))
          for b in m.blocks]
    m(torch.tensor([ids], device=DEV))
    for h in hs: h.remove()
    return acts                                              # list of [1,T,d] per block


def rank_of(logits, t):
    return int((logits.argsort(descending=True) == t).nonzero()[0])


def main():
    m = load()
    kind = ["spin" if isinstance(b, H.SpinBlock) else "attn" for b in m.blocks]
    R = {"layer_kind": kind}

    # ---- T1.8: intermediate-representation logit lens ----
    PROBES = [("The capital of the country where the Eiffel Tower stands is", "France", "Paris"),
              ("The largest planet, whose first letter is", "Jupiter", "J"),
              ("The language spoken in the country whose capital is Tokyo is", "Japan", "Japanese")]
    t18 = []
    for prompt, inter, ans in PROBES:
        acts = capture(m, tok.encode(prompt).ids)
        it, at = tid(inter), tid(ans)
        inter_best = min(rank_of(m.head(m.lnf(h[0, -1])), it) for h in acts)
        ans_best = min(rank_of(m.head(m.lnf(h[0, -1])), at) for h in acts)
        t18.append({"q": prompt[:40], "intermediate": inter, "best_inter_rank": inter_best, "best_answer_rank": ans_best,
                    "computes_intermediate": inter_best < 50})
    R["T1_8_intermediate_probe"] = t18

    # ---- T3.2: activation patching (clean 'France->Paris' vs corrupted 'Japan->Tokyo') ----
    clean, corrupt = "The capital of France is", "The capital of Japan is"
    ans = tid("Paris")
    with torch.no_grad():
        clean_logit = float(m(torch.tensor([tok.encode(clean).ids], device=DEV))[0, -1, ans])
        corrupt_logit = float(m(torch.tensor([tok.encode(corrupt).ids], device=DEV))[0, -1, ans])
    clean_acts = capture(m, tok.encode(clean).ids)
    restore = {}
    for li in range(len(m.blocks)):
        patched = {}

        def hook(mod, i, o, _li=li):
            out = o[0] if isinstance(o, tuple) else o
            out = out.clone(); out[0, -1] = clean_acts[_li][0, -1].to(out.dtype)   # patch the last-token activation
            return (out,) + tuple(o[1:]) if isinstance(o, tuple) else out
        hh = m.blocks[li].register_forward_hook(hook)
        with torch.no_grad():
            lg = float(m(torch.tensor([tok.encode(corrupt).ids], device=DEV))[0, -1, ans])
        hh.remove()
        restore[li] = round((lg - corrupt_logit) / (clean_logit - corrupt_logit + 1e-6), 2)   # 1.0 = fully restored
    # aggregate restoration by layer kind
    spin_r = [restore[i] for i in range(len(kind)) if kind[i] == "spin"]
    attn_r = [restore[i] for i in range(len(kind)) if kind[i] == "attn"]
    R["T3_2_patching"] = {"clean_logit": round(clean_logit, 2), "corrupt_logit": round(corrupt_logit, 2),
                          "mean_restore_spin": round(sum(spin_r) / max(1, len(spin_r)), 3),
                          "mean_restore_attn": round(sum(attn_r) / max(1, len(attn_r)), 3),
                          "top_layers": sorted(restore, key=restore.get, reverse=True)[:5]}
    os.makedirs("eval_registry", exist_ok=True)
    json.dump(R, open("eval_registry/mech_probe.json", "w"), indent=2)
    print(json.dumps(R, indent=2), flush=True)


if __name__ == "__main__":
    main()
