"""
paper15_figure1.py -- Figure 1 for Paper 15

Three-panel figure showing the internal amplitude law:
  Panel A: sg4 vs FA (Exp 1, log-log) with power law fit.
           Main result: sg4 ~ FA^0.43 (approximately sqrt(FA)) inside window.
           Outside-window points shown unfilled for contrast.
  Panel B: sg4 growth trajectories at 4 checkpoints for selected FA values.
           Shows Phase 2 structural transition: dip at t=1500 then recovery.
           Higher FA -> steeper final climb, higher asymptote.
  Panel C: sg4 heatmap on (nu, FA) grid from Exp 2.
           Contour lines at constant FA/nu (straight lines in log-log).
           Structure: contours NOT horizontal -> FA/nu law is wrong.
           Actual pattern: saturation at high FA, non-monotone in nu.

Output: paper15_figure1.pdf, paper15_figure1.png
"""
import json, os
import numpy as np
import math
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

DIR = os.path.dirname(__file__)
r = json.load(open(os.path.join(DIR, "results", "paper15_results.json")))

FD = 0.9997
NU_CRYST = abs(math.log(FD)) / math.log(2)
WR = 4.8; WAVE_DUR = 15; SS = 10
P_CALM = (1 - WR / (2 * WAVE_DUR)) ** (SS + WAVE_DUR - 1)

EXP1_NU = 0.001
EXP1_FA_VALS = [0.05, 0.08, 0.10, 0.12, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]
EXP2_NU_VALS = [0.0005, 0.001, 0.002]
EXP2_FA_VALS = [0.10, 0.15, 0.20, 0.30, 0.50]
CHECKPOINTS = [1250, 1500, 1750, 2000]
N_SEEDS = 5

def get_exp1(fa, t):
    vals = [r[f"exp1,{fa:.4f},{s}"][f"sg4_{t}"]
            for s in range(N_SEEDS) if f"exp1,{fa:.4f},{s}" in r]
    return float(np.mean(vals)) if vals else float("nan"), \
           float(np.std(vals) / math.sqrt(len(vals))) if len(vals) > 1 else 0.0

def get_exp2(nu, fa):
    vals = [r[f"exp2,{nu:.5f},{fa:.4f},{s}"][f"sg4_2000"]
            for s in range(N_SEEDS) if f"exp2,{nu:.5f},{fa:.4f},{s}" in r]
    return float(np.mean(vals)) if vals else float("nan")

def rb(nu, fa):
    fd_term = FD ** (1.0 / nu)
    return P_CALM * fa * fd_term / nu

fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))

# ============================================================
# Panel A: sg4 vs FA (log-log), power law fit
# ============================================================
ax = axes[0]

fa_inside = []; sg4_inside = []; err_inside = []
fa_outside = []; sg4_outside = []; err_outside = []

for fa in EXP1_FA_VALS:
    fd_term = FD ** (1.0 / EXP1_NU)
    r_b = P_CALM * fa * fd_term / EXP1_NU
    mean, se = get_exp1(fa, 2000)
    if r_b >= 1.0:
        fa_inside.append(fa); sg4_inside.append(mean); err_inside.append(se)
    else:
        fa_outside.append(fa); sg4_outside.append(mean); err_outside.append(se)

# Power law fit on inside points
log_fa_in = [math.log(f) for f in fa_inside]
log_sg4_in = [math.log(s) for s in sg4_inside]
coeffs = np.polyfit(log_fa_in, log_sg4_in, 1)
alpha_exp = coeffs[0]
log_C = coeffs[1]
C = math.exp(log_C)

fa_fit = np.linspace(0.08, 0.55, 100)
sg4_fit = C * fa_fit ** alpha_exp

ax.errorbar(fa_inside, sg4_inside, yerr=err_inside,
            fmt="o", color="#1f77b4", ms=6, capsize=3, lw=1.5,
            label=r"Inside window ($R_B \geq 1$)", zorder=3)
ax.errorbar(fa_outside, sg4_outside, yerr=err_outside,
            fmt="o", color="#1f77b4", ms=6, capsize=3, lw=1.5,
            mfc="none", mec="#1f77b4", zorder=3,
            label=r"Outside window ($R_B < 1$)")
ax.plot(fa_fit, sg4_fit, "r-", lw=2.0, label=fr"Fit: sg4 $\propto$ FA$^{{{alpha_exp:.3f}}}$")

# Sqrt reference
sg4_sqrt = C * (np.array(fa_fit) / fa_fit[50]) ** 0.5 * sg4_fit[50]
ax.plot(fa_fit, sg4_sqrt, "k--", lw=1.0, alpha=0.5, label=r"Reference $\propto\sqrt{\rm FA}$")

ax.set_xlabel(r"FIELD\_ALPHA (FA)", fontsize=10)
ax.set_ylabel(r"sg4 at $t=2000$", fontsize=10)
ax.set_title(r"\textbf{A.} Amplitude law: sg4 vs FA", fontsize=10)
ax.legend(fontsize=6.5, framealpha=0.85)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.text(0.05, 0.95,
        fr"Power law exponent: {alpha_exp:.3f}" + "\n" +
        r"($\approx\sqrt{\rm FA}$ inside window)",
        transform=ax.transAxes, fontsize=7.5, va="top",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8))

# ============================================================
# Panel B: Growth trajectories for selected FA values
# ============================================================
ax = axes[1]
FA_SELECT = [0.10, 0.15, 0.20, 0.30, 0.50]
COLORS_B = ["#1f77b4", "#2ca02c", "#ff7f0e", "#9467bd", "#d62728"]

for fa, col in zip(FA_SELECT, COLORS_B):
    traj = []
    errs = []
    for t in CHECKPOINTS:
        mean, se = get_exp1(fa, t)
        traj.append(mean)
        errs.append(se)
    fd_term = FD ** (1.0 / EXP1_NU)
    r_b = P_CALM * fa * fd_term / EXP1_NU
    label = f"FA={fa:.2f} (R_B={r_b:.2f})"
    ax.errorbar(CHECKPOINTS, traj, yerr=errs,
                color=col, fmt="-o", ms=5, lw=1.8, capsize=3,
                label=label)

ax.axvline(x=1500, color="gray", lw=1.0, ls=":", alpha=0.7)
ax.text(1505, ax.get_ylim()[0] if ax.get_ylim()[0] > 0 else 150,
        "Phase 2 reorganization\ndip", fontsize=6.5, color="gray")
ax.set_xlabel(r"Simulation step", fontsize=10)
ax.set_ylabel(r"sg4", fontsize=10)
ax.set_title(r"\textbf{B.} Growth trajectories in Phase 2", fontsize=10)
ax.legend(fontsize=6.5, framealpha=0.85, loc="upper left")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.text(0.55, 0.05,
        r"Dip at $t=1500$: Phase 1$\to$Phase 2" + "\n" +
        "transition disrupts structure\nbefore rebuild",
        transform=ax.transAxes, fontsize=6.5, va="bottom",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcyan", alpha=0.8))

# ============================================================
# Panel C: sg4 heatmap on (nu, FA) grid with FA/nu contours
# ============================================================
ax = axes[2]

# Build heatmap data
# Rows = FA, Cols = nu
hmap = np.full((len(EXP2_FA_VALS), len(EXP2_NU_VALS)), np.nan)
for fi, fa in enumerate(EXP2_FA_VALS):
    for ni, nu in enumerate(EXP2_NU_VALS):
        hmap[fi, ni] = get_exp2(nu, fa)

im = ax.imshow(hmap, aspect="auto", origin="lower", cmap="hot_r",
               interpolation="nearest")
plt.colorbar(im, ax=ax, label="sg4 at t=2000")

ax.set_xticks(range(len(EXP2_NU_VALS)))
ax.set_xticklabels([f"{nu:.4f}" for nu in EXP2_NU_VALS], fontsize=8)
ax.set_yticks(range(len(EXP2_FA_VALS)))
ax.set_yticklabels([f"{fa:.2f}" for fa in EXP2_FA_VALS], fontsize=8)
ax.set_xlabel(r"Turnover rate $\nu$", fontsize=10)
ax.set_ylabel(r"FIELD\_ALPHA (FA)", fontsize=10)
ax.set_title(r"\textbf{C.} sg4 on $(\nu, {\rm FA})$ grid", fontsize=10)

# Mark outside-window cells
for fi, fa in enumerate(EXP2_FA_VALS):
    for ni, nu in enumerate(EXP2_NU_VALS):
        if rb(nu, fa) < 1.0:
            ax.add_patch(plt.Rectangle((ni-0.5, fi-0.5), 1, 1,
                                       fill=False, edgecolor="cyan",
                                       lw=2.5, ls="--"))

ax.text(0.05, 0.95,
        "Cyan border = outside window\n" +
        r"($R_B < 1$, $\nu > \nu_{\rm max}$)" + "\n" +
        "FA/nu contours would be diagonal\n-- structure does not follow them",
        transform=ax.transAxes, fontsize=6.5, va="top",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.8))

fig.suptitle(
    fr"Internal amplitude law: sg4 $\propto$ FA$^{{{alpha_exp:.3f}}}$ inside the adaptive window; "
    r"rate-lifetime hypothesis (sg4 $\propto$ FA/$\nu$) rejected",
    fontsize=9.5, y=1.01
)
fig.tight_layout()

OUT = os.path.join(DIR, "paper15_figure1")
fig.savefig(OUT + ".pdf", bbox_inches="tight")
fig.savefig(OUT + ".png", dpi=150, bbox_inches="tight")
print(f"Saved {OUT}.pdf and {OUT}.png")
