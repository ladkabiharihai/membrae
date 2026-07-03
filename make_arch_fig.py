"""Generate a large, detailed architecture figure (paper style) -> arch_fig.html.
Left: the macro spin-dominant stack. Right: one spin block expanded. Bottom: the carrier math + a phasor."""
def box(x, y, w, h, label, fill, stroke, sub=None, fs=9.5):
    t = f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="5" fill="{fill}" stroke="{stroke}" stroke-width="1.2"/>'
    ty = (y + h/2 - 2) if sub else (y + h/2 + 3.5)
    t += f'<text x="{x+w/2}" y="{ty:.0f}" text-anchor="middle" font-size="{fs}" fill="#16171d">{label}</text>'
    if sub: t += f'<text x="{x+w/2}" y="{y+h/2+9:.0f}" text-anchor="middle" font-size="7.3" fill="#666">{sub}</text>'
    return t
def arr(x, y1, y2):
    return f'<line x1="{x}" y1="{y1}" x2="{x}" y2="{y2-6}" stroke="#999" stroke-width="1.2" marker-end="url(#ar)"/>'

s = ['<figure><svg viewBox="0 0 440 468" xmlns="http://www.w3.org/2000/svg" font-family="Georgia,serif">',
     '<defs><marker id="ar" markerWidth="8" markerHeight="8" refX="5" refY="3" orient="auto"><path d="M0,0 L6,3 L0,6 Z" fill="#999"/></marker></defs>']
# ---- LEFT: macro stack (x center 96, width 150 -> 21..171) ----
LX, LW = 21, 150; cx = LX + LW/2
s.append(f'<text x="{cx}" y="14" text-anchor="middle" font-size="8" font-style="italic" fill="#888">spin-dominant stack</text>')
s.append(box(LX, 20, LW, 22, "Tokens → Embedding + Positions", "#eef0f8", "#9aa0c0"))
s.append(arr(cx, 42, 56))
# repeat group
s.append('<rect x="14" y="56" width="164" height="150" rx="7" fill="none" stroke="#c9c9d6" stroke-width="1.2" stroke-dasharray="4 3"/>')
s.append(f'<text x="186" y="133" font-size="8.5" font-weight="700" fill="#555" transform="rotate(90 186 133)">repeat × N/4</text>')
yy = 64
for i in range(3):
    s.append(box(LX, yy, LW, 26, "Spin block", "#ffe0bd", "#e8923a", "carrier mixes tokens + MLP")); yy += 32
s.append(box(LX, yy, LW, 26, "Attention block", "#ffc9de", "#d6336c", "every 4th layer (helper)"))
s.append(arr(cx, 206, 220))
s.append(box(LX, 220, LW, 20, "LayerNorm", "#eef0f8", "#9aa0c0", fs=9))
s.append(arr(cx, 240, 252))
s.append(box(LX, 252, LW, 22, "Head (tied) → logits", "#eef0f8", "#9aa0c0"))
# ---- RIGHT: one spin block expanded (x center 320, width 176 -> 232..408) ----
RX, RW = 232, 176; rcx = RX + RW/2
s.append(f'<text x="{rcx}" y="14" text-anchor="middle" font-size="8" font-style="italic" fill="#888">inside one spin block</text>')
s.append(f'<line x1="171" y1="88" x2="232" y2="70" stroke="#e8923a" stroke-width="1" stroke-dasharray="3 2"/>')
s.append(f'<line x1="171" y1="88" x2="232" y2="150" stroke="#e8923a" stroke-width="1" stroke-dasharray="3 2"/>')
s.append(box(RX, 22, RW, 18, "LayerNorm", "#f2f2f7", "#bbb", fs=8.5)); s.append(arr(rcx, 40, 52))
s.append(box(RX, 52, RW, 30, "SPIN CARRIER  (token mixer)", "#ffd9a8", "#e8923a", "diagonal-complex recurrence · gated residual", fs=9))
s.append(f'<text x="{RX+RW+2}" y="70" font-size="8" fill="#e8923a">⊕</text>'); s.append(arr(rcx, 82, 94))
s.append(box(RX, 94, RW, 18, "LayerNorm", "#f2f2f7", "#bbb", fs=8.5)); s.append(arr(rcx, 112, 124))
s.append(box(RX, 124, RW, 26, "MLP  (channel mixer)", "#eef0f8", "#9aa0c0", fs=9))
s.append(f'<text x="{RX+RW+2}" y="140" font-size="8" fill="#9aa0c0">⊕</text>')
# ---- BOTTOM: the carrier math + phasor ----
s.append('<line x1="21" y1="300" x2="419" y2="300" stroke="#e0e0e8" stroke-width="1"/>')
s.append('<text x="21" y="316" font-size="9.5" font-weight="700" fill="#16171d">The spin carrier</text>')
mono = 'font-family="Consolas,monospace" font-size="9"'
s.append(f'<text x="21" y="334" {mono} fill="#16171d">h_t = λ ⊙ h_(t−1) + b_t</text>')
s.append(f'<text x="21" y="352" {mono} fill="#16171d">λ_j = exp(−exp ν_j) · exp(i θ_j)</text>')
s.append('<text x="21" y="366" font-size="8" fill="#666">a damped rotation per channel: θ_j sets the turn, |λ_j|&lt;1 the decay</text>')
s.append('<text x="21" y="388" font-size="8.5" fill="#16171d">Run as a <tspan font-weight="700">log-depth parallel associative scan</tspan> — not a T-step loop.</text>')
s.append('<text x="21" y="402" font-size="8.5" fill="#16171d">Cost <tspan font-family="Consolas,monospace">O(T·d)</tspan> vs attention <tspan font-family="Consolas,monospace">O(T²·d)</tspan>; at inference recurrent, <tspan font-family="Consolas,monospace">O(1)</tspan>/token, no KV-cache.</text>')
# phasor (right of the math)
pcx, pcy, pr = 358, 372, 34
s.append(f'<circle cx="{pcx}" cy="{pcy}" r="{pr}" fill="none" stroke="#ddd" stroke-width="1"/>')
s.append(f'<line x1="{pcx-pr-4}" y1="{pcy}" x2="{pcx+pr+4}" y2="{pcy}" stroke="#eee"/><line x1="{pcx}" y1="{pcy-pr-4}" x2="{pcx}" y2="{pcy+pr+4}" stroke="#eee"/>')
# a decaying spiral
import math
pts = " ".join(f"{pcx+ (pr*0.92**(k/6))*math.cos(-k*0.6):.0f},{pcy+ (pr*0.92**(k/6))*math.sin(-k*0.6):.0f}" for k in range(22))
s.append(f'<polyline points="{pts}" fill="none" stroke="#e8923a" stroke-width="1.6"/>')
s.append(f'<line x1="{pcx}" y1="{pcy}" x2="{pcx+pr*0.86:.0f}" y2="{pcy-pr*0.4:.0f}" stroke="#e8923a" stroke-width="1.6" marker-end="url(#ar)"/>')
s.append(f'<text x="{pcx}" y="{pcy+pr+12}" text-anchor="middle" font-size="7.5" fill="#888">damped rotation (the "spin")</text>')
s.append('</svg><figcaption><b></b>The spin-dominant architecture in full. <b>Left:</b> the macro stack  the diagonal-complex spin carrier is the token-mixer in three of every four blocks, with an attention block every fourth as a periodic helper; the head is tied to the embedding. <b>Right:</b> one spin block  LayerNorm, the spin carrier as a gated-residual token-mixer, then LayerNorm and an MLP. <b>Bottom:</b> the carrier itself is a per-channel damped complex recurrence <span class="mono">λ_j = exp(−exp ν_j)·exp(iθ_j)</span> (a rotation of angle θ that decays by |λ|), evaluated by a log-depth parallel associative scan  <span class="mono">O(T·d)</span> and matmul-free, recurrent and <span class="mono">O(1)</span>/token at inference.</figcaption></figure>')
open("arch_fig.html", "w", encoding="utf-8").write("\n".join(s))
print("arch figure written:", len("\n".join(s)), "bytes")
