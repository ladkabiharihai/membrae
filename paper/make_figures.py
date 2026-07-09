"""Generate publication-quality figures for the Pragnosia paper (real data)."""
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import os
os.makedirs("paper/figs", exist_ok=True)
plt.rcParams.update({"font.family":"serif","font.size":11,"axes.linewidth":.8,
  "figure.dpi":200,"savefig.bbox":"tight","axes.spines.top":False,"axes.spines.right":False})
INK="#1a1a2e"; C1="#5b4bd6"; C2="#0f9b82"; C3="#d6336c"; C4="#c98a00"; GREY="#8a8a9a"

# ---- Fig 1: spinning vs settling phase space ----
fig,ax=plt.subplots(1,2,figsize=(8,3.4))
t=np.linspace(0,40,2000)
# settling: inward spiral
r=2.0*np.exp(-t/12); xa=r*np.cos(t*1.4); ya=r*np.sin(t*1.4)
ax[0].plot(xa,ya,color=C3,lw=1.4); ax[0].plot(0,0,'o',color=C3,ms=9)
ax[0].set_title("Conventional: settles",fontsize=12); ax[0].text(0,-2.5,"answer = a fixed resting state\n(easy to merely memorize)",ha="center",fontsize=9,color=GREY)
# spinning: closed orbit
xb=1.6*np.cos(t*1.4)+0.12*np.cos(t*5); yb=1.6*np.sin(t*1.4)+0.12*np.sin(t*5)
ax[1].plot(xb,yb,color=C2,lw=1.2,alpha=.85)
th=t[-1]*1.4; ax[1].annotate("",xy=(1.6*np.cos(th),1.6*np.sin(th)),xytext=(0,0),arrowprops=dict(arrowstyle="->",color=C2,lw=2))
ax[1].plot(1.6*np.cos(th),1.6*np.sin(th),'o',color=C2,ms=8)
ax[1].set_title("Pragnosia: spins",fontsize=12); ax[1].text(0,-2.5,"answer = the phase of an\nongoing, non-settling orbit",ha="center",fontsize=9,color=GREY)
for a in ax: a.set_xlim(-2.6,2.6);a.set_ylim(-2.9,2.6);a.set_aspect("equal");a.set_xticks([]);a.set_yticks([]);a.spines['left'].set_visible(False);a.spines['bottom'].set_visible(False)
plt.tight_layout(); plt.savefig("paper/figs/fig_spin.png"); plt.close()

# ---- Fig 2: architecture diagram ----
fig,ax=plt.subplots(figsize=(8,4.2)); ax.set_xlim(0,10); ax.set_ylim(0,6); ax.axis("off")
def box(x,y,w,h,t,c,fc="#ffffff",fs=9):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle="round,pad=0.05,rounding_size=0.12",lw=1.3,ec=c,fc=fc))
    ax.text(x+w/2,y+h/2,t,ha="center",va="center",fontsize=fs,color=INK)
def arr(x1,y1,x2,y2,c=GREY):
    ax.add_patch(FancyArrowPatch((x1,y1),(x2,y2),arrowstyle="-|>",mutation_scale=12,lw=1.1,color=c))
box(0.3,2.6,1.7,1.0,"Input\n(symbols · vision ·\naudio · text)",GREY,fs=8)
box(2.6,4.0,2.5,1.4,"SPINNING CORE\nW = -ρQQ$^T$ + S\nrotational dynamics",C1,"#f3f1ff",9)
box(2.6,0.5,2.5,1.4,"HYBRID LANGUAGE\nattention + gated\nspin carrier",C2,"#eefaf6",9)
box(5.8,4.0,1.9,1.4,"Faculties:\nreason · abstain\nseek · count · alive",C1,"#f3f1ff",8)
box(5.8,0.5,1.9,1.4,"generate ·\nknowledge ·\nteach (online)",C2,"#eefaf6",8)
box(8.1,2.4,1.6,1.4,"PRAGNOSIA\nautonomous\ncontroller",C3,"#fdeef4",9)
arr(2.0,3.1,2.6,4.4); arr(2.0,3.1,2.6,1.2)
arr(5.1,4.7,5.8,4.7); arr(5.1,1.2,5.8,1.2)
arr(7.7,4.7,8.4,3.7); arr(7.7,1.2,8.4,3.0)
ax.text(5,5.7,"One model. Spinning where it reasons; attention where it speaks; everything self-calibrated.",ha="center",fontsize=9,style="italic",color=GREY)
plt.savefig("paper/figs/fig_arch.png"); plt.close()

# ---- Fig 3: brain-swap causal test ----
fig,ax=plt.subplots(figsize=(5.2,3.2))
labels=["follows\nswapped-in state","follows\noriginal input","random-state\ncontrol"]
vals=[0.59,0.10,0.21]; cols=[C2,GREY,C3]
b=ax.bar(labels,vals,color=cols,width=.6,edgecolor="white")
ax.axhline(0.20,ls="--",color=INK,lw=.9); ax.text(2.4,0.225,"chance",fontsize=8,color=INK)
for bar,v in zip(b,vals): ax.text(bar.get_x()+bar.get_width()/2,v+0.02,f"{v:.2f}",ha="center",fontsize=10,fontweight="bold")
ax.set_ylim(0,0.72); ax.set_ylabel("fraction of trials"); ax.set_title("Brain-swap: the answer lives in the spinning state",fontsize=11)
plt.savefig("paper/figs/fig_brainswap.png"); plt.close()

# ---- Fig 4: perplexity, pure-spin vs hybrid (real trajectory) ----
fig,ax=plt.subplots(figsize=(5.6,3.4))
steps=[500,1000,1500,2000,2500,3000]
spin=[502,340,280,250,232,215]        # pure spin (s5, measured)
hyb=[226,142,99,76,60,50]             # hybrid (s6, measured, after the init+warmup fix)
ax.plot(steps,spin,"o-",color=C3,lw=1.6,ms=4,label="pure spinning LM")
ax.plot(steps,hyb,"s-",color=C2,lw=1.6,ms=4,label="hybrid (attention + spin carrier)")
ax.set_xlabel("training step"); ax.set_ylabel("validation perplexity"); ax.set_yscale("log")
ax.legend(frameon=False,fontsize=9); ax.set_title("The hybrid: ~3× lower perplexity",fontsize=11)
ax.grid(True,which="both",ls=":",alpha=.3)
plt.savefig("paper/figs/fig_ppl.png"); plt.close()

# ---- Fig 5: surprise-modulated plasticity ----
fig,ax=plt.subplots(figsize=(5.2,3.2))
s=np.linspace(0,3.2,200); pl=np.clip(s,0.15,3.0)
ax.plot(s,pl,color=C1,lw=2)
ax.axhline(1,ls=":",color=GREY); ax.axvline(1,ls=":",color=GREY)
ax.plot(1.31,1.31,'o',color=C3,ms=8); ax.text(1.4,1.5,"surprising fact\n(plasticity 1.31)",fontsize=8,color=C3)
ax.plot(0.65,0.65,'o',color=C2,ms=8); ax.text(0.7,0.25,"familiar input\n(plasticity 0.65)",fontsize=8,color=C2)
ax.set_xlabel("surprise ÷ familiarity boundary (the brain's own signal)")
ax.set_ylabel("learning rate multiplier"); ax.set_title("Self-modulated plasticity (neuromodulation-like)",fontsize=11)
plt.savefig("paper/figs/fig_plasticity.png"); plt.close()

# ---- Fig 6: the 14 faculties ----
fig,ax=plt.subplots(figsize=(6.2,4.4))
fac=["reason: parity-3","reason: sum-3","reason: max","P5 confidence gap","P6 abstain (known)↓",
     "P6 abstain (unknowable)","language parse","language generate","seek: query","seek: answer",
     "seek: random-ctrl↓","exact accumulate L=16","alive: final","alive: retention gap↓"]
val=[0.92,0.90,0.93,0.34,0.19,1.00,1.00,1.00,1.00,1.00,0.13,1.00,1.00,0.00]
# normalize "lower is better" ones for color only
good=[True]*14
y=np.arange(len(fac))[::-1]
ax.barh(y,val,color=C2,height=.62,edgecolor="white")
for yi,v in zip(y,val): ax.text(v+0.01,yi,f"{v:.2f}",va="center",fontsize=8)
ax.set_yticks(y); ax.set_yticklabels(fac,fontsize=8); ax.set_xlim(0,1.15)
ax.set_xlabel("measured score   (down = lower is better)"); ax.set_title("Retired 302K toy self-test (NOT the live 1B; see RESULTS_MEASURED.md)",fontsize=9)
plt.savefig("paper/figs/fig_faculties.png"); plt.close()
print("figures written to paper/figs/")
