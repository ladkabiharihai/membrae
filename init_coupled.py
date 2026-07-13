"""Warm-start the fast-slow COUPLED model (spin_dominant_coupled) from an existing spin_dominant checkpoint,
so the coupling fine-tune starts EQUAL to the base (coupling params init near-identity) and only diverges as
it learns -- cheap, controlled, no from-scratch cost. Writes CONFIG's ckpt. Then fine-tune with:
    CONFIG=pragnosia_coupled.json python3 train_pragnosia.py --resume --steps <N>

Run:  python3 init_coupled.py <base_ckpt>     # e.g. pragnosia_sft.pt (promoted) or pragnosia_spin.pt
CONFIG defaults to pragnosia_coupled.json.
"""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch, s6_hybrid as H

base_ckpt = sys.argv[1] if len(sys.argv) > 1 else "pragnosia_spin.pt"
cfg = json.load(open(os.environ.get("CONFIG", "pragnosia_coupled.json")))
H.VOC, H.L = cfg["vocab"], cfg["ctx"]
assert cfg["carrier"] == "spin_dominant_coupled", "CONFIG must be the coupled config"

m = H.SpinAttentionLM(cfg["vocab"], cfg["d"], cfg["heads"], cfg["layers"],
                      mlp_mult=cfg["mlp_mult"], carrier=cfg["carrier"])
base = torch.load(base_ckpt, map_location="cpu", weights_only=True)
missing, unexpected = m.load_state_dict(base, strict=False)     # coupling params (to_slow/slow/from_slow) stay at init
assert len(unexpected) == 0, f"name mismatch, unexpected keys: {unexpected[:5]}"
coupling_new = [k for k in missing if any(t in k for t in ("to_slow", "slow", "from_slow"))]
print(f"warm-started coupled from {base_ckpt}: {len(missing)} new params "
      f"({len(coupling_new)} coupling), 0 unexpected", flush=True)
torch.save(m.state_dict(), cfg["ckpt"])
print(f"wrote {cfg['ckpt']} ({os.path.getsize(cfg['ckpt'])/1e9:.2f}GB). now: "
      f"CONFIG={os.environ.get('CONFIG','pragnosia_coupled.json')} python3 train_pragnosia.py --resume --steps N", flush=True)
