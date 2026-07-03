"""Generate the dark-theme learning-curve SVG for the site (site palette: teal spin, pink transformer)."""
import json, math
d = json.load(open("learning_curves.json")); s, t = d["spin-dominant"], d["attention-only"]
YMIN, YMAX = 50, 560; XMAX = s[-1]["tokens"]/1e6
W0, W1, H0, H1 = 70, 680, 34, 250
X = lambda tok: W0 + (tok/1e6/XMAX) * (W1 - W0)
Y = lambda ppl: H0 + (math.log(YMAX)-math.log(ppl))/(math.log(YMAX)-math.log(YMIN)) * (H1 - H0)
poly = lambda c: " ".join(f"{X(p['tokens']):.0f},{Y(p['ppl']):.0f}" for p in c)
grid = "".join(f'<line x1="{W0}" y1="{Y(p):.0f}" x2="{W1}" y2="{Y(p):.0f}" stroke="rgba(255,255,255,.07)"/>'
               f'<text x="{W0-8}" y="{Y(p)+4:.0f}" text-anchor="end" font-size="12" fill="#6b76a0">{p}</text>' for p in [50,100,200,400])
sN, tN = s[-1], t[-1]; g0 = t[0]["ppl"]/s[0]["ppl"]; gN = tN["ppl"]/sN["ppl"]
svg = f'''<svg viewBox="0 0 700 288" style="width:100%;height:auto;margin-top:6px" font-family="Inter,sans-serif">
{grid}
<line x1="{W0}" y1="{H1}" x2="{W1}" y2="{H1}" stroke="rgba(255,255,255,.2)"/>
<text x="{W0}" y="274" font-size="12" fill="#6b76a0">0</text>
<text x="{W1}" y="274" text-anchor="end" font-size="12" fill="#6b76a0">{XMAX:.0f}M tokens</text>
<text x="30" y="26" font-size="12" fill="#6b76a0">val ppl</text>
<polyline points="{poly(t)}" fill="none" stroke="#ff7ac2" stroke-width="3"/>
<polyline points="{poly(s)}" fill="none" stroke="#23d5ab" stroke-width="3"/>
<circle cx="{X(tN['tokens']):.0f}" cy="{Y(tN['ppl']):.0f}" r="5" fill="#ff7ac2"/>
<text x="{X(tN['tokens'])-12:.0f}" y="{Y(tN['ppl'])-9:.0f}" text-anchor="end" font-size="14" font-weight="700" fill="#ff7ac2">Transformer {tN['ppl']:.0f}</text>
<circle cx="{X(sN['tokens']):.0f}" cy="{Y(sN['ppl']):.0f}" r="5" fill="#23d5ab"/>
<text x="{X(sN['tokens'])-12:.0f}" y="{Y(sN['ppl'])+20:.0f}" text-anchor="end" font-size="14" font-weight="700" fill="#23d5ab">Spin-dominant {sN['ppl']:.0f}</text>
<text x="{(W0+W1)/2:.0f}" y="{H0+16:.0f}" text-anchor="middle" font-size="12.5" font-style="italic" fill="#9aa6c8">gap {g0:.1f}× → {gN:.1f}×  it widens with training</text>
</svg>'''
open("site_lc.svg", "w").write(svg)
print(f"site SVG: spin {sN['ppl']:.0f} / transf {tN['ppl']:.0f} / {gN:.2f}x")
