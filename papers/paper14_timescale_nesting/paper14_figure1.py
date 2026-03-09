"""
paper14_figure1.py -- Figure 1 for Paper 14

Three-panel figure showing the iso-ratio invariance test result:
  Panel A: sg4 vs R_A for both routes at T2 target (R_A=2.0, R_B=2.0).
           If ratio hypothesis holds: curves should overlap.
           Reality: R2 > R1 at low R_A, R1 > R2 at high R_A.
  Panel B: sg4 ratio R2/R1 across R_A values for all three targets.
           Systematic pattern: not random noise, but route-dependent.
  Panel C: Explanation diagram -- FA enters sg4 through two independent routes:
           (1) via R_B (which ratio parameterization captures)
           (2) via differentiation speed per event (which it misses).
           Show: at fixed R_B, higher FA -> higher sg4 amplitude.

Output: paper14_figure1.pdf, paper14_figure1.png
"""
import json, os
import numpy as np
import math
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DIR = os.path.dirname(__file__)
r1 = json.load(open(os.path.join(DIR, "results", "paper14_exp1_results.json")))

FD_R1 = 0.9997; FD_R2 = 0.9994
NU_CRYST_R1 = abs(math.log(FD_R1)) / math.log(2)
NU_CRYST_R2 = abs(math.log(FD_R2)) / math.log(2)
P_CALM = (1 - 4.8 / (2 * 15)) ** (10 + 15 - 1)

TARGETS = {
    "T1_1.5_1.5": (1.5, 1.5),
    "T2_2.0_2.0": (2.0, 2.0),
    "T3_1.5_3.0": (1.5, 3.0),
}
RA_SCAN = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0, 4.5]

def get_sg4(tname, route, ra):
    key = f"{tname},{route},{ra:.4f}"
    if key in r1:
        return float(np.mean([v["sg4"] for v in r1[key]]))
    return np.nan

fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))

# Colors for routes
C_R1 = "#1f77b4"   # blue -- Route R1 (FD=0.9997)
C_R2 = "#d62728"   # red  -- Route R2 (FD=0.9994)
TARGET_COLORS = {"T1_1.5_1.5": "#2ca02c", "T2_2.0_2.0": "#9467bd", "T3_1.5_3.0": "#ff7f0e"}

# --- Panel A: sg4 vs R_A for T2 target, both routes ---
ax = axes[0]
for tname, (ra_tgt, rb_tgt) in TARGETS.items():
    y1 = [get_sg4(tname, "R1", ra) for ra in RA_SCAN]
    y2 = [get_sg4(tname, "R2", ra) for ra in RA_SCAN]
    tc = TARGET_COLORS[tname]
    lab = f"R_A={ra_tgt}, R_B={rb_tgt}"
    ax.plot(RA_SCAN, y1, color=tc, lw=2.0, ls="-", marker="o", ms=4,
            label=f"{lab} (R1: FD=0.9997)")
    ax.plot(RA_SCAN, y2, color=tc, lw=1.5, ls="--", marker="s", ms=4,
            label=f"{lab} (R2: FD=0.9994)")

ax.axvline(x=1.0, color="gray", lw=1.0, ls=":", alpha=0.7,
           label=r"$R_A=1$ ($\nu=\nu_{\rm cryst}$)")
ax.set_xlabel(r"$R_A = \nu/\nu_{\rm cryst}$", fontsize=10)
ax.set_ylabel(r"sg4", fontsize=10)
ax.set_title(r"\textbf{A.} sg4 vs $R_A$ for both routes", fontsize=10)
ax.legend(fontsize=5.5, loc="lower right", framealpha=0.85)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.text(0.05, 0.95, "Solid=R1 (FD=0.9997)\nDashed=R2 (FD=0.9994)\nSame target (R_A,R_B) -> should overlap",
        transform=ax.transAxes, fontsize=6, va="top",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8))

# --- Panel B: ratio R2/R1 for all targets ---
ax = axes[1]
for tname, (ra_tgt, rb_tgt) in TARGETS.items():
    ratios = []
    valid_ra = []
    for ra in RA_SCAN:
        sg4_r1 = get_sg4(tname, "R1", ra)
        sg4_r2 = get_sg4(tname, "R2", ra)
        if sg4_r1 > 10 and sg4_r2 > 10:
            ratios.append(sg4_r2 / sg4_r1)
            valid_ra.append(ra)
    tc = TARGET_COLORS[tname]
    ax.plot(valid_ra, ratios, color=tc, lw=1.8, marker="o", ms=5,
            label=f"R_A={ra_tgt}, R_B={rb_tgt}")

ax.axhline(y=1.0, color="k", lw=1.0, ls="--", label="Perfect invariance (=1)")
ax.axhline(y=2.0, color="gray", lw=0.8, ls=":", alpha=0.6)
ax.axhline(y=0.5, color="gray", lw=0.8, ls=":", alpha=0.6)
ax.axvline(x=1.0, color="gray", lw=1.0, ls=":", alpha=0.7)
ax.set_xlabel(r"$R_A = \nu/\nu_{\rm cryst}$", fontsize=10)
ax.set_ylabel(r"sg4$_{\rm R2}$ / sg4$_{\rm R1}$", fontsize=10)
ax.set_title(r"\textbf{B.} Ratio R2/R1 (should be 1 if invariant)", fontsize=10)
ax.legend(fontsize=7, framealpha=0.85)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.text(0.05, 0.95, "Systematic: R2>R1 at low R_A\nR1>R2 at high R_A\n=> NOT noise",
        transform=ax.transAxes, fontsize=7, va="top",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.8))

# --- Panel C: FA's dual role -- at same R_B, higher FA -> higher sg4 ---
ax = axes[2]
# Show sg4 vs FA at fixed R_B~2, for Routes R1 and R2
# At R_A=2.0: R1(FA=0.16)->343, R2(FA=0.32)->476 for T2
# At R_A=2.0: R1(FA=0.10)->237, R2(FA=0.20)->389 for T1
# At R_A=2.0: R1(FA=0.20)->399, R2(FA=0.41)->517 for T3

fa_vals = []
sg4_vals = []
rb_vals = []
for tname, (ra_tgt, rb_tgt) in TARGETS.items():
    fa_r1 = rb_tgt * ra_tgt * NU_CRYST_R1 / (P_CALM * (0.5 ** (1.0/ra_tgt)))
    fa_r2 = rb_tgt * ra_tgt * NU_CRYST_R2 / (P_CALM * (0.5 ** (1.0/ra_tgt)))
    sg4_r1 = get_sg4(tname, "R1", 2.0)
    sg4_r2 = get_sg4(tname, "R2", 2.0)
    fa_vals.extend([fa_r1, fa_r2])
    sg4_vals.extend([sg4_r1, sg4_r2])
    rb_vals.extend([rb_tgt, rb_tgt])
    ax.scatter(fa_r1, sg4_r1, marker="o", s=70, color=TARGET_COLORS[tname],
               label=f"R_B={rb_tgt}", zorder=3)
    ax.scatter(fa_r2, sg4_r2, marker="s", s=70, color=TARGET_COLORS[tname], zorder=3)
    ax.plot([fa_r1, fa_r2], [sg4_r1, sg4_r2], color=TARGET_COLORS[tname],
            lw=1.2, ls="--", alpha=0.6)

ax.set_xlabel(r"FIELD\_ALPHA (FA)", fontsize=10)
ax.set_ylabel(r"sg4 at $R_A=2.0$", fontsize=10)
ax.set_title(r"\textbf{C.} FA's dual role at fixed $R_A=2$", fontsize=10)
ax.legend(fontsize=7, framealpha=0.85)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.text(0.05, 0.95,
        "Circles=R1 (low FA), Squares=R2 (2x FA)\n"
        "Same R_B but different FA -> different sg4\n"
        "=> FA has an independent role beyond R_B",
        transform=ax.transAxes, fontsize=6.5, va="top",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcyan", alpha=0.8))

fig.suptitle(
    r"Timescale ratio invariance fails: FA independently amplifies sg4 beyond its role in setting $\nu_{\rm max}$",
    fontsize=9.5, y=1.01
)
fig.tight_layout()

OUT = os.path.join(DIR, "paper14_figure1")
fig.savefig(OUT + ".pdf", bbox_inches="tight")
fig.savefig(OUT + ".png", dpi=150, bbox_inches="tight")
print(f"Saved {OUT}.pdf and {OUT}.png")
