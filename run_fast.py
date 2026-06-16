"""
Speed wrapper: runs train_pragnosia.py UNCHANGED but force-enables torch.compile
(which fuses the 256-step spin recurrence — the real bottleneck) and lets the
trainer autotune a big batch on the now-freed GPU. The training file is imported,
not edited: we only monkeypatch its autotune() return value.

  python3 run_fast.py --resume --no-grow            # compile + bf16 (try first)
  python3 run_fast.py --resume --no-grow --fp32      # fallback if bf16+compile bug
"""
import argparse
import train_pragnosia as T

pa = argparse.ArgumentParser()
pa.add_argument("--steps", type=int, default=150000)
pa.add_argument("--lr", type=float, default=6e-4)
pa.add_argument("--bs", type=int, default=0)
pa.add_argument("--resume", action="store_true")
pa.add_argument("--no-grow", action="store_true")
pa.add_argument("--fp32", action="store_true", help="compile with fp32 (dodges the bf16+tied-weights bug)")
a = pa.parse_args()

_orig = T.autotune
def patched():
    c = _orig()
    c["compile"] = True                      # fuse the sequential spin loop
    if a.fp32:
        c["bf16"] = False
    print(f"[run_fast] forcing compile=True bf16={c['bf16']} (was compile-off on big GPU)", flush=True)
    return c
T.autotune = patched

T.main(a.steps, a.lr, a.resume, a.bs, grow_enabled=not a.no_grow)
