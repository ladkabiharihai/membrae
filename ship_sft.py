"""C7 ship: bake the SFT adapter into a NEW checkpoint (base untouched), now that C7 proved the precision-first
router survives the merge (specificity 0.833 held, fluency 113->49 words). Writes pragnosia_spin_sft.pt. To
activate: point pragnosia.json "ckpt" at it. Fully reversible -- the base pragnosia_spin.pt is never modified.

Run (GPU, Unreal closed):  python3 ship_sft.py
"""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch, s6_hybrid as H, lora

c = json.load(open("pragnosia.json")); H.VOC, H.L = c["vocab"], c["ctx"]
OUT = "pragnosia_spin_sft.pt"
ADAPTER = "pragnosia_sft_lora.pt"

m = H.SpinAttentionLM(c["vocab"], c["d"], c["heads"], c["layers"], mlp_mult=c["mlp_mult"], carrier=c["carrier"])
m.load_state_dict(torch.load(c["ckpt"], map_location="cpu", weights_only=True))     # base, on CPU
lora.inject_lora(m, r=16)
sd = torch.load(ADAPTER, map_location="cpu", weights_only=True)
missing, unexpected = m.load_state_dict(sd, strict=False)
loaded = sum(1 for _ in sd)
lora.merge_and_unload(m)                                                             # fold A/B into base -> plain Linears
torch.save(m.state_dict(), OUT)
print(f"merged {loaded} adapter tensors into base; wrote {OUT} "
      f"({os.path.getsize(OUT)/1e9:.2f}GB). base {c['ckpt']} untouched.", flush=True)
print(f"to activate: set pragnosia.json \"ckpt\" to \"{OUT}\" (reversible).", flush=True)
