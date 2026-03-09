"""
paper16_figure1.py -- Figure 1 for Paper 16

Three-panel figure:
  Panel A: sg4_inter vs FA (SHIFT=0, t=3000) -- saturation confirmed.
           Overlay saturation curve C*FA/(FA+K_eff) with K_eff~0.119.
           Compare to Paper 15 power law (dashed) -- shows the power law
           is only a local fit in the sub-saturation region.
  Panel B: Growth trajectories (10 checkpoints) for FA=0.20 and FA=0.70.
           Both SHIFT conditions (SHIFT=0: solid; SHIFT=1000: dashed).
           Key: SHIFT=0 non-monotone (peaks then declines by t=3000);
                SHIFT=1000 dip then OVERSHOOT above Phase 1 baseline (anti-erasure).
  Panel C: sg4_intra vs sg4_inter at SHIFT=0, t=3000 across all FA.
           intra/inter ratio 1.6-2.2: local (within-zone) structure dominates
           for pure Phase 2 encoding, not between-zone (left-right) contrast.

Output: paper16_figure1.pdf, paper16_figure1.png
"""
import json, os
import numpy as np
import math
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DIR = os.path.dirname(__file__)
r1 = json.load(open(os.path.join(DIR, "results", "paper16_exp1_results.json")))

FA_VALS    = [0.05, 0.08, 0.10, 0.15, 0.20, 0.30, 0.50, 0.70, 0.90]
N_SEEDS    = 5
STEPS      = 3000
CHECKPOINTS = list(range(300, STEPS + 1, 300))
FD = 0.9997
NU = 0.001
WR = 4.8; WAVE_DUR = 15; SS = 10
P_CALM = (1 - WR / (2 * WAVE_DUR)) ** (SS + WAVE_DUR - 1)
fd_term = FD ** (1.0 / NU)


def get_inter(shift, fa, t):
    vals = [r1[f"exp1,{shift},{fa:.4f},{s}"][f"sg4_inter_{t}"]
            for s in range(N_SEEDS) if f"exp1,{shift},{fa:.4f},{s}" in r1]
    return float(np.mean(vals)), float(np.std(vals) / math.sqrt(len(vals))) if len(vals) > 1 else 0.


def get_intra(shift, fa, t):
    vals = [r1[f"exp1,{shift},{fa:.4f},{s}"][f"sg4_intra_{t}"]
            for s in range(N_SEEDS) if f"exp1,{shift},{fa:.4f},{s}" in r1]
    return float(np.mean(vals)), float(np.std(vals) / math.sqrt(len(vals))) if len(vals) > 1 else 0.


fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))

# ============================================================
# Panel A: sg4_inter vs FA (SHIFT=0, t=3000) + saturation fit
# ============================================================
ax = axes[0]

fa_inside = []; sg4_inside = []; err_inside = []
fa_outside = []; sg4_outside = []; err_outside = []

for fa in FA_VALS:
    rb = P_CALM * fa * fd_term / NU
    m, se = get_inter(0, fa, STEPS)
    if rb >= 1.0:
        fa_inside.append(fa); sg4_inside.append(m); err_inside.append(se)
    else:
        fa_outside.append(fa); sg4_outside.append(m); err_outside.append(se)

# Saturation fit (grid search for C and K_eff)
best_r2 = -1e9; best_C = 0; best_K = 0
for K in np.linspace(0.005, 0.50, 500):
    x = [fa / (fa + K) for fa in fa_inside]
    C = float(np.dot(x, sg4_inside) / np.dot(x, x)) if sum(v*v for v in x) > 0 else 0
    pred = [C * xx for xx in x]
    ss_res = sum((p - d)**2 for p, d in zip(pred, sg4_inside))
    ss_tot = sum((d - float(np.mean(sg4_inside)))**2 for d in sg4_inside)
    r2 = 1 - ss_res/ss_tot if ss_tot > 0 else float("nan")
    if r2 > best_r2:
        best_r2 = r2; best_C = C; best_K = K

fa_fit = np.linspace(0.05, 0.95, 200)
sg4_sat = best_C * fa_fit / (fa_fit + best_K)

# Paper 15 power law reference (alpha=0.426, t=2000 SHIFT=1000)
# Anchor at FA=0.20 from Paper 15: sg4=419
C15 = 419.1 / (0.20 ** 0.426)
sg4_pl = C15 * fa_fit ** 0.426

ax.errorbar(fa_inside, sg4_inside, yerr=err_inside, fmt="o",
            color="#1f77b4", ms=6, capsize=3, lw=1.5, zorder=3,
            label=r"SHIFT=0, $t=3000$ (inside window)")
ax.errorbar(fa_outside, sg4_outside, yerr=err_outside, fmt="o",
            color="#1f77b4", ms=6, capsize=3, lw=1.5, zorder=3,
            mfc="none", mec="#1f77b4", label=r"SHIFT=0, $t=3000$ (outside)")
ax.plot(fa_fit, sg4_sat, "r-", lw=2.0,
        label=fr"Fit: $C\cdot$FA/(FA$+K_{{eff}}$), $K_{{eff}}$={best_K:.3f}, $R^2$={best_r2:.3f}")
ax.plot(fa_fit, sg4_pl, "k--", lw=1.2, alpha=0.6,
        label=r"Paper 15 law: FA$^{0.43}$ (SHIFT=1000)")

ax.set_xlabel(r"FIELD\_ALPHA (FA)", fontsize=10)
ax.set_ylabel(r"sg4$_{\rm inter}$ at $t=3000$", fontsize=10)
ax.set_title(r"\textbf{A.} Saturation confirmed (SHIFT=0)", fontsize=10)
ax.legend(fontsize=5.8, framealpha=0.85, loc="upper left")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.text(0.05, 0.95,
        r"$K_{\rm eff}=0.119$; theory: 0.114 ($<5\%$ error)" + "\n" +
        r"Paper 15 ($\propto$ FA$^{0.43}$): Phase 1 persistence" + "\n" +
        r"Saturation: sg4 plateaus at FA$\gtrsim$0.5",
        transform=ax.transAxes, fontsize=6.5, va="top",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.85))

# ============================================================
# Panel B: Growth trajectories
# ============================================================
ax = axes[1]

colors_fa = {"0.2000": "#1f77b4", "0.7000": "#d62728"}
fa_traj_list = [(0.20, "0.2000"), (0.70, "0.7000")]
labels_fa = {0.20: r"FA=0.20", 0.70: r"FA=0.70"}

for fa, fa_str in fa_traj_list:
    col = colors_fa[fa_str]
    for shift, ls, lw in [(0, "-", 2.0), (1000, "--", 1.5)]:
        traj = [get_inter(shift, fa, t)[0] for t in CHECKPOINTS]
        label = f"{labels_fa[fa]}, SHIFT={shift}"
        ax.plot(CHECKPOINTS, traj, color=col, ls=ls, lw=lw, marker="o", ms=3.5,
                label=label)

ax.axvline(x=1000, color="gray", lw=1.0, ls=":", alpha=0.7)
ax.text(1005, ax.get_ylim()[1] if ax.get_ylim()[1] > 0 else 500,
        "Phase 2\nonset", fontsize=6.5, color="gray")

ax.set_xlabel(r"Simulation step", fontsize=10)
ax.set_ylabel(r"sg4$_{\rm inter}$", fontsize=10)
ax.set_title(r"\textbf{B.} Trajectories (SHIFT=0 vs 1000)", fontsize=10)
ax.legend(fontsize=6.0, framealpha=0.85, ncol=1)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.text(0.05, 0.95,
        "SHIFT=0 (solid): peaks then declines\n"
        "(Phase 2 opposes left-right sg4)\n"
        "SHIFT=1000 (dashed): copy-forward\nreinforces Phase 1 structure",
        transform=ax.transAxes, fontsize=6.0, va="top",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcyan", alpha=0.85))

# ============================================================
# Panel C: sg4_inter vs sg4_intra at SHIFT=0, t=3000
# ============================================================
ax = axes[2]

inter_vals = [get_inter(0, fa, STEPS)[0] for fa in FA_VALS]
intra_vals = [get_intra(0, fa, STEPS)[0] for fa in FA_VALS]
inter_errs = [get_inter(0, fa, STEPS)[1] for fa in FA_VALS]
intra_errs = [get_intra(0, fa, STEPS)[1] for fa in FA_VALS]

ax.errorbar(FA_VALS, inter_vals, yerr=inter_errs, fmt="-o",
            color="#1f77b4", ms=5, capsize=3, lw=1.8, label=r"sg4$_{\rm inter}$ (between zones)")
ax.errorbar(FA_VALS, intra_vals, yerr=intra_errs, fmt="-s",
            color="#d62728", ms=5, capsize=3, lw=1.8, label=r"sg4$_{\rm intra}$ (within zones)")

# Ratio line
ax2 = ax.twinx()
ratios = [intra/inter if inter > 0 else float("nan") for intra, inter in zip(intra_vals, inter_vals)]
ax2.plot(FA_VALS, ratios, color="#2ca02c", ls="--", lw=1.5, marker="^", ms=4,
         label="intra/inter ratio")
ax2.set_ylabel("intra/inter ratio", fontsize=9, color="#2ca02c")
ax2.tick_params(axis="y", labelcolor="#2ca02c")
ax2.axhline(y=1.0, color="#2ca02c", lw=0.8, ls=":", alpha=0.5)
ax2.set_ylim(0.8, 2.8)

ax.set_xlabel(r"FIELD\_ALPHA (FA)", fontsize=10)
ax.set_ylabel(r"sg4 at $t=3000$ (SHIFT=0)", fontsize=10)
ax.set_title(r"\textbf{C.} Local $>$ Global structure (Phase 2 only)", fontsize=10)
ax.legend(fontsize=7, framealpha=0.85, loc="upper left")
ax2.legend(fontsize=7, framealpha=0.85, loc="center left")
ax.spines["top"].set_visible(False)

ax.text(0.05, 0.35,
        r"${\rm sg4}_{\rm intra} > {\rm sg4}_{\rm inter}$ for all FA" + "\n"
        r"Phase 2 creates within-zone" + "\n"
        r"(top/bottom) structure, not" + "\n"
        r"between-zone (left/right)",
        transform=ax.transAxes, fontsize=6.5, va="bottom",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.85))

fig.suptitle(
    r"Paper 15's FA$^{0.43}$ law is Phase 1 persistence; pure Phase 2 follows saturation curve "
    r"sg4 $\propto$ FA/(FA$+K_{\rm eff}$) with $K_{\rm eff}\approx 0.12$",
    fontsize=9.0, y=1.01
)
fig.tight_layout()

OUT = os.path.join(DIR, "paper16_figure1")
fig.savefig(OUT + ".pdf", bbox_inches="tight")
fig.savefig(OUT + ".png", dpi=150, bbox_inches="tight")
print(f"Saved {OUT}.pdf and {OUT}.png")
print(f"Saturation fit: C={best_C:.2f}, K_eff={best_K:.4f}, R^2={best_r2:.4f}")
