"""
paper11_figure1.py -- Figure 1 for Paper 11

Three-panel figure, one panel per experiment:
  Panel A: FIELD_DECAY sweep -- sg4 vs nu for 4 FD values.
           FD=0.999 matches nu_cryst prediction (nu* shifts up).
           FD=0.995/0.99 don't: nu* stays low despite high nu_cryst.
  Panel B: SS sweep -- sg4 vs nu for 4 SS values.
           Upper boundary (nu_max) correctly shifts nu* for SS=5->10.
           Window SHOULD collapse at SS=20/40 but copy-forward prevents it.
  Panel C: FIELD_ALPHA sweep -- sg4 vs nu for 4 FA values.
           Clean upward shift: nu* tracks nu_max proportionally to FA.

Output: paper11_figure1.pdf, paper11_figure1.png
"""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DIR = os.path.dirname(__file__)

r1 = json.load(open(os.path.join(DIR, "results", "paper11_exp1_results.json")))
r2 = json.load(open(os.path.join(DIR, "results", "paper11_exp2_results.json")))
r3 = json.load(open(os.path.join(DIR, "results", "paper11_exp3_results.json")))

DEATH_PS = [0.0001, 0.0002, 0.0003, 0.0005, 0.001, 0.002,
            0.003, 0.005, 0.010, 0.020, 0.040, 0.080]

FIELD_DECAY_VALS  = [0.9997, 0.999, 0.995, 0.99]
SS_VALS           = [5, 10, 20, 40]
FIELD_ALPHA_VALS  = [0.04, 0.08, 0.16, 0.32]

COLORS4 = ["#2077b4", "#27a027", "#d62728", "#9467bd"]

import math
nu_cryst_ref = abs(math.log(0.9997)) / math.log(2)   # 0.000433

def get_sg4(results, key_fmt, param, nu):
    key = key_fmt.format(param, nu)
    if key in results:
        return float(np.mean([v["sg4"] for v in results[key]]))
    return np.nan

fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))

# --- Panel A: FIELD_DECAY sweep ---
ax = axes[0]
FD_LABELS = [
    r"$D=0.9997$ (ref, $\nu_c=4\times10^{-4}$)",
    r"$D=0.999$ ($\nu_c=1.4\times10^{-3}$)",
    r"$D=0.995$ ($\nu_c=7.2\times10^{-3}$)",
    r"$D=0.990$ ($\nu_c=1.5\times10^{-2}$)",
]
for fi, fd in enumerate(FIELD_DECAY_VALS):
    y = [get_sg4(r1, "{},{}", fd, nu) for nu in DEATH_PS]
    lw = 2.2 if fi == 0 else 1.5
    ls = "-" if fi == 0 else "--"
    ax.plot(DEATH_PS, y, color=COLORS4[fi], lw=lw, ls=ls,
            marker="o", ms=3.5, label=FD_LABELS[fi])
    # Mark nu_cryst
    nu_c = abs(math.log(fd)) / math.log(2)
    if nu_c <= 0.08:
        ax.axvline(x=nu_c, color=COLORS4[fi], lw=0.8, ls=":", alpha=0.6)
ax.set_xscale("log")
ax.set_xlabel(r"Turnover rate $\nu$", fontsize=10)
ax.set_ylabel(r"sg4", fontsize=10)
ax.set_title(r"\textbf{A.} FIELD\_DECAY sweep", fontsize=10)
ax.legend(fontsize=6.5, loc="lower left", framealpha=0.85)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# --- Panel B: SS sweep ---
ax = axes[1]
SS_LABELS = [
    r"SS=5 ($\nu_{\rm max}=4.3\times10^{-3}$)",
    r"SS=10 ref ($\nu_{\rm max}=1.8\times10^{-3}$)",
    r"SS=20 ($\nu_{\rm max}=3.2\times10^{-4}$, pred. collapse)",
    r"SS=40 ($\nu_{\rm max}\approx10^{-5}$, pred. collapse)",
]
COLORS_SS = ["#1f77b4", "#d62728", "#ff7f0e", "#9467bd"]
for si, ss in enumerate(SS_VALS):
    y = [get_sg4(r2, "{},{}", ss, nu) for nu in DEATH_PS]
    lw = 2.2 if si == 1 else 1.5
    ls = "-" if si == 1 else "--"
    ax.plot(DEATH_PS, y, color=COLORS_SS[si], lw=lw, ls=ls,
            marker="o", ms=3.5, label=SS_LABELS[si])
# Mark nu_cryst (fixed)
ax.axvline(x=nu_cryst_ref, color="gray", lw=1.0, ls=":", alpha=0.7,
           label=r"$\nu_{\rm cryst}=4.3\times10^{-4}$")
ax.set_xscale("log")
ax.set_xlabel(r"Turnover rate $\nu$", fontsize=10)
ax.set_ylabel(r"sg4", fontsize=10)
ax.set_title(r"\textbf{B.} SS (consolidation gate) sweep", fontsize=10)
ax.legend(fontsize=6.5, loc="lower right", framealpha=0.85)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# --- Panel C: FIELD_ALPHA sweep ---
ax = axes[2]
FA_LABELS = [
    r"$\alpha_F=0.04$ ($\nu_{\rm max}=4.5\times10^{-4}$)",
    r"$\alpha_F=0.08$ ($\nu_{\rm max}=9.0\times10^{-4}$)",
    r"$\alpha_F=0.16$ (ref, $\nu_{\rm max}=1.8\times10^{-3}$)",
    r"$\alpha_F=0.32$ ($\nu_{\rm max}=3.6\times10^{-3}$)",
]
for ai, fa in enumerate(FIELD_ALPHA_VALS):
    y = [get_sg4(r3, "{},{}", fa, nu) for nu in DEATH_PS]
    lw = 2.2 if ai == 2 else 1.5
    ls = "-" if ai == 2 else "--"
    ax.plot(DEATH_PS, y, color=COLORS4[ai], lw=lw, ls=ls,
            marker="o", ms=3.5, label=FA_LABELS[ai])
ax.axvline(x=nu_cryst_ref, color="gray", lw=1.0, ls=":", alpha=0.7,
           label=r"$\nu_{\rm cryst}=4.3\times10^{-4}$")
ax.set_xscale("log")
ax.set_xlabel(r"Turnover rate $\nu$", fontsize=10)
ax.set_ylabel(r"sg4", fontsize=10)
ax.set_title(r"\textbf{C.} FIELD\_ALPHA (consolidation rate) sweep", fontsize=10)
ax.legend(fontsize=6.5, loc="lower right", framealpha=0.85)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

fig.suptitle(
    r"Substrate parameters shape the adaptive window; copy-forward prevents predicted collapse at extreme SS",
    fontsize=9.5, y=1.01
)
fig.tight_layout()

OUT = os.path.join(DIR, "paper11_figure1")
fig.savefig(OUT + ".pdf", bbox_inches="tight")
fig.savefig(OUT + ".png", dpi=150, bbox_inches="tight")
print(f"Saved {OUT}.pdf and {OUT}.png")
