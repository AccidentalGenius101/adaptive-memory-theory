"""
paper7_figure1_regime_map.py -- Figure 1 for Paper 7

Dual-metric empirical regime map.

Panel A: sg4 heatmap from Exp3 (zone differentiation)
Panel B: classifier accuracy heatmap from Exp5b (orthogonal metric)

Axes: nu (turnover) on x-axis, kappa (propagation) on y-axis.
The two panels should show the same adaptive window (mid nu, mid kappa),
demonstrating that regime boundaries are metric-independent.

Output: paper7_figure1.pdf (vector, for LaTeX inclusion)
Dependencies: numpy, matplotlib.
"""
import numpy as np
import json
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

SCRIPT_DIR = os.path.dirname(__file__)

NU_VALS    = [0.0003, 0.001, 0.005, 0.015, 0.050]
KAPPA_VALS = [0.000, 0.005, 0.020, 0.060, 0.150]

NU_LABELS    = [r"$3{\times}10^{-4}$", r"$10^{-3}$", r"$5{\times}10^{-3}$",
                r"$1.5{\times}10^{-2}$", r"$5{\times}10^{-2}$"]
KAPPA_LABELS = ["0", "0.005", "0.020", "0.060", "0.150"]

# ── Load Exp3 (sg4) ─────────────────────────────────────────────────────────
exp3 = json.load(open(os.path.join(SCRIPT_DIR, "results", "paper7_exp3_results.json")))
sg4_grid = np.zeros((len(KAPPA_VALS), len(NU_VALS)))
for i, kp in enumerate(KAPPA_VALS):
    for j, nu in enumerate(NU_VALS):
        key = f"{nu},{kp}"
        vals = exp3[key]
        sg4_grid[i, j] = float(np.mean([v["sg4"] for v in vals]))

# ── Load Exp5b (classifier accuracy) ────────────────────────────────────────
exp5b = json.load(open(os.path.join(SCRIPT_DIR, "results", "paper7_exp5b_results.json")))
acc_grid = np.zeros((len(KAPPA_VALS), len(NU_VALS)))
for i, kp in enumerate(KAPPA_VALS):
    for j, nu in enumerate(NU_VALS):
        key = f"{nu},{kp}"
        vals = exp5b[key]
        acc_grid[i, j] = float(np.mean([v["accuracy"] for v in vals]))

# ── Figure ───────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.6))
fig.subplots_adjust(wspace=0.38, left=0.10, right=0.95, top=0.88, bottom=0.18)

CMAP_SG4 = "YlOrRd"
CMAP_ACC = "YlGnBu"

def draw_panel(ax, data, cmap, title, cbar_label, vmin=None, vmax=None,
               chance_line=None):
    if vmin is None: vmin = data.min()
    if vmax is None: vmax = data.max()
    im = ax.imshow(data, origin="lower", aspect="auto",
                   cmap=cmap, vmin=vmin, vmax=vmax,
                   interpolation="nearest")
    ax.set_xticks(range(len(NU_VALS)))
    ax.set_xticklabels(NU_LABELS, fontsize=7, rotation=30, ha="right")
    ax.set_yticks(range(len(KAPPA_VALS)))
    ax.set_yticklabels(KAPPA_LABELS, fontsize=8)
    ax.set_xlabel(r"Turnover rate $\nu$", fontsize=9, labelpad=4)
    ax.set_ylabel(r"Propagation $\kappa$", fontsize=9, labelpad=4)
    ax.set_title(title, fontsize=10, pad=6)

    # Annotate cells with values
    for i in range(len(KAPPA_VALS)):
        for j in range(len(NU_VALS)):
            val = data[i, j]
            norm_val = (val - vmin) / (vmax - vmin + 1e-9)
            txt_color = "white" if norm_val > 0.55 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=6.5, color=txt_color)

    # Colorbar
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label(cbar_label, fontsize=8)
    cb.ax.tick_params(labelsize=7)

    # Outline the adaptive region (mid nu × mid kappa)
    # nu=0.001-0.005 (cols 1-2), kappa=0.005-0.060 (rows 1-3)
    rect = plt.Rectangle((0.5, 0.5), 2.0, 3.0,
                          linewidth=2, edgecolor="black",
                          facecolor="none", linestyle="--")
    ax.add_patch(rect)

draw_panel(axes[0], sg4_grid,
           CMAP_SG4, "(A)  Zone differentiation (sg4)",
           "sg4")

draw_panel(axes[1], acc_grid,
           CMAP_ACC, "(B)  Classifier accuracy",
           "accuracy",
           vmin=0.25, vmax=1.0)

# Dashed rectangle label
for ax in axes:
    ax.text(2.5, 3.65, "adaptive\nwindow", fontsize=6.5,
            ha="center", va="bottom", color="black",
            style="italic")

fig.suptitle("Empirical regime map: independent metrics identify the same adaptive window",
             fontsize=9.5, y=0.98)

OUT = os.path.join(SCRIPT_DIR, "paper7_figure1.pdf")
fig.savefig(OUT, bbox_inches="tight")
print(f"Saved {OUT}")

OUT_PNG = os.path.join(SCRIPT_DIR, "paper7_figure1.png")
fig.savefig(OUT_PNG, dpi=150, bbox_inches="tight")
print(f"Saved {OUT_PNG}")
