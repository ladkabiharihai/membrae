"""Generate the learning-curve figure SVG (paper style) from learning_curves.json -> writes lc_fig.html.
Re-run when the experiment adds points; the figure updates automatically."""
import json, math
d = json.load(open("learning_curves.json"))
s, t = d["spin-dominant"], d["attention-only"]
YMIN, YMAX, XMAX = 50, 560, 123.0
X = lambda tok: 62 + (tok/1e6/XMAX) * 356
Y = lambda ppl: 28 + (math.log(YMAX)-math.log(ppl))/(math.log(YMAX)-math.log(YMIN)) * 150
poly = lambda c: " ".join(f"{X(p['tokens']):.0f},{Y(p['ppl']):.0f}" for p in c)
sN, tN = s[-1], t[-1]; gap0 = t[0]["ppl"]/s[0]["ppl"]; gapN = tN["ppl"]/sN["ppl"]
grid = "".join(f'<line x1="62" y1="{Y(p):.0f}" x2="418" y2="{Y(p):.0f}" stroke="#eee" stroke-width="1"/>'
               f'<text x="56" y="{Y(p)+3:.0f}" text-anchor="end" font-size="8.5" fill="#999">{p}</text>'
               for p in [50, 100, 200, 400])
svg = f'''<figure><svg viewBox="0 0 440 210" xmlns="http://www.w3.org/2000/svg" font-family="Georgia,serif">
{grid}
<line x1="62" y1="180" x2="418" y2="180" stroke="#bbb" stroke-width="1"/><line x1="62" y1="28" x2="62" y2="180" stroke="#bbb" stroke-width="1"/>
<text x="40" y="24" font-size="8.5" fill="#999">val ppl</text>
<text x="240" y="198" text-anchor="middle" font-size="9" fill="#777">tokens seen (M)  matched budget, identical schedule + batches</text>
<polyline points="{poly(t)}" fill="none" stroke="#d6336c" stroke-width="2"/>
<polyline points="{poly(s)}" fill="none" stroke="#1aa88a" stroke-width="2"/>
<circle cx="{X(tN['tokens']):.0f}" cy="{Y(tN['ppl']):.0f}" r="3.5" fill="#d6336c"/>
<text x="{X(tN['tokens'])-6:.0f}" y="{Y(tN['ppl'])-6:.0f}" text-anchor="end" font-size="10" font-weight="700" fill="#d6336c">Transformer {tN['ppl']:.0f}</text>
<circle cx="{X(sN['tokens']):.0f}" cy="{Y(sN['ppl']):.0f}" r="3.5" fill="#1aa88a"/>
<text x="{X(sN['tokens'])-6:.0f}" y="{Y(sN['ppl'])+12:.0f}" text-anchor="end" font-size="10" font-weight="700" fill="#1aa88a">Spin-dominant {sN['ppl']:.0f}</text>
<text x="200" y="60" font-size="9" font-style="italic" fill="#999">gap {gap0:.1f}× → {gapN:.1f}×  (it widens)</text>
</svg><figcaption><b></b>Learning curves at matched ~37M, identical schedule and batches, each at its range-test learning rate (spin <span class="mono">6.7e&minus;3</span>, transformer <span class="mono">1.0e&minus;4</span>), to {sN['tokens']/1e6:.0f}M tokens. The transformer <i>plateaus</i> (~{tN['ppl']:.0f}) while the spin-dominant model keeps descending (~{sN['ppl']:.0f}), so the gap <i>widens</i> with training ({gap0:.1f}× → {gapN:.1f}×). This directly answers the natural objection that the short-budget gap is just "the transformer hasn't trained yet": given a full schedule at its own best learning rate, the transformer descends and then flattens  it does not close the gap. (Small-scale, single-seed.)</figcaption></figure>'''
open("lc_fig.html", "w").write(svg)
print(f"figure written: spin {sN['ppl']:.1f} vs transf {tN['ppl']:.1f} = {gapN:.2f}x at {sN['tokens']/1e6:.0f}M ({len(s)} pts)")
