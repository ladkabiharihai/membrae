"""Live ASCII view of the learning-curve experiment (learning_curves.json), refreshing in place.
  python3 lc_watch.py          # auto-refresh every 20s
  python3 lc_watch.py once     # single snapshot
S = spin-dominant, T = attention-only (transformer). Y = val ppl (log), X = tokens seen."""
import json, math, os, sys, time

W, H = 66, 20

def load():
    try: return json.load(open("learning_curves.json"))
    except Exception: return {}

def latest_log():
    try:
        lines = [l for l in open("lc_run.log") if "ppl" in l]
        return lines[-1].strip() if lines else ""
    except Exception: return ""

def render(d):
    series = [("S", "spin-dominant", d.get("spin-dominant", [])),
              ("T", "attention-only", d.get("attention-only", []))]
    pts = [(p["tokens"]/1e6, p["ppl"]) for _, _, c in series for p in c]
    if not pts:
        return "  waiting for the first validation point (~3 min in)...\n  " + latest_log()
    xs = [x for x, _ in pts]; ys = [y for _, y in pts]
    xmax = max(max(xs), 1.0); ymin, ymax = min(ys) * 0.96, max(ys) * 1.05
    lo, hi = math.log(ymin), math.log(ymax)
    yrow = lambda y: max(0, min(H-1, int((1 - (math.log(y)-lo)/(hi-lo+1e-9)) * (H-1))))
    xcol = lambda x: max(0, min(W-1, int(x/(xmax+1e-9) * (W-1))))
    grid = [[" "]*W for _ in range(H)]
    for mk, _, c in series:
        for p in c:
            grid[yrow(p["ppl"])][xcol(p["tokens"]/1e6)] = mk
    out = ["  Pragnosia learning curves  S=spin-dominant  T=transformer  (val ppl vs tokens)", ""]
    for r in range(H):
        ppl_at = math.exp(hi - (hi-lo) * r/(H-1))
        lbl = f"{ppl_at:6.0f}" if r % 3 == 0 else "      "
        out.append(f"  {lbl} |" + "".join(grid[r]))
    out.append("         +" + "-"*W)
    out.append("          0" + " "*(W-12) + f"{xmax:.0f}M tokens")
    # current standings
    for mk, name, c in series:
        if c: out.append(f"   {mk} {name:16} {c[-1]['ppl']:7.1f} ppl @ {c[-1]['tokens']/1e6:.0f}M tok ({len(c)} pts)")
    if "spin-dominant" in d and "attention-only" in d and d["spin-dominant"] and d["attention-only"]:
        s, t = d["spin-dominant"][-1]["ppl"], d["attention-only"][-1]["ppl"]
        out.append(f"   gap so far: {t/s:.2f}x (transformer/spin)")
    out.append("   " + latest_log())
    return "\n".join(out)

if __name__ == "__main__":
    once = len(sys.argv) > 1 and sys.argv[1] == "once"
    while True:
        os.system("clear")
        print(render(load()), flush=True)
        if once or not os.path.exists("lc_run.log"): break
        if "LC DONE" in open("lc_run.log").read(): print("\n  [experiment finished]"); break
        time.sleep(20)
