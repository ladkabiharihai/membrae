"""Progression figure across grown sizes (284M/565M/706M): 3 mini-panels. Emits a light SVG (paper) and a
dark SVG (site). Data is the measured numbers from our tests."""
SIZES = ["284M", "565M", "706M", "~1B"]
PANELS = [("Validation ppl", [24.5, 22.0, 21.97, 20.1], True),   # lower better
          ("LAMBADA  %", [19.0, 21.7, 24.4, 25.3], False),        # higher better
          ("Multi-hop  /10", [2, 3, 4, 4], False)]

def svg(dark):
    ink   = "#e8ecf8" if dark else "#16171d"
    mut   = "#9aa6c8" if dark else "#666"
    grid  = "rgba(255,255,255,.08)" if dark else "#eee"
    line  = "#23d5ab" if dark else "#1aa88a"
    down  = "#ff7ac2" if dark else "#d6336c"
    W, H = 660, 210; pw = 200; py0, py1 = 40, 150
    out = [f'<svg viewBox="0 0 {W} {H}" xmlns="http://www.w3.org/2000/svg" font-family="{"Inter,sans-serif" if dark else "Georgia,serif"}">']
    for pi,(title, vals, lower) in enumerate(PANELS):
        px = 20 + pi*(pw+15); cx0, cx1 = px+34, px+pw-16
        lo, hi = min(vals)*0.96, max(vals)*1.04
        col = down if lower else line
        X = lambda i: cx0 + i*(cx1-cx0)/(len(vals)-1)
        Y = lambda v: py1 - (v-lo)/(hi-lo+1e-9)*(py1-py0)
        out.append(f'<text x="{px+pw/2:.0f}" y="26" text-anchor="middle" font-size="13" font-weight="700" fill="{ink}">{title}</text>')
        out.append(f'<text x="{px+pw/2:.0f}" y="38" text-anchor="middle" font-size="9" font-style="italic" fill="{mut}">{"lower is better" if lower else "higher is better"}</text>')
        out.append(f'<line x1="{cx0-6}" y1="{py1}" x2="{cx1+6}" y2="{py1}" stroke="{grid}"/>')
        out.append(f'<polyline points="{" ".join(f"{X(i):.0f},{Y(v):.0f}" for i,v in enumerate(vals))}" fill="none" stroke="{col}" stroke-width="2.5"/>')
        for i,v in enumerate(vals):
            out.append(f'<circle cx="{X(i):.0f}" cy="{Y(v):.0f}" r="4" fill="{col}"/>')
            out.append(f'<text x="{X(i):.0f}" y="{Y(v)-9:.0f}" text-anchor="middle" font-size="10.5" font-weight="700" fill="{ink}">{v}</text>')
            out.append(f'<text x="{X(i):.0f}" y="{py1+15:.0f}" text-anchor="middle" font-size="9.5" fill="{mut}">{SIZES[i]}</text>')
    out.append("</svg>")
    return "\n".join(out)

# paper (light) wrapped as a figure
fig = ('<figure>' + svg(False) + '\n'
       '<figcaption><b></b>Progression across the three grown sizes. Validation perplexity falls, and LAMBADA (the most fluency-dependent benchmark) and the greedy multi-hop battery both rise monotonically  the design improves consistently as it grows and trains, with the largest gains on the tokens-limited axes.</figcaption></figure>')
open("prog_fig.html","w",encoding="utf-8").write(fig)
open("prog_fig_site.svg","w",encoding="utf-8").write(svg(True).replace("<svg ", '<svg style="width:100%;height:auto;margin-top:6px" '))
print("wrote prog_fig.html (paper) + prog_fig_site.svg (site)")
