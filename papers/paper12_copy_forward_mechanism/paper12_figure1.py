"""
paper12_figure1.py -- Figure 1 for Paper 12

Three-panel figure:
  Panel A: SEED_BETA sweep -- sg4 vs nu for 5 SB values.
           SB=0.00 should show collapse (no relay).
           SB=0.05-0.50 should show progressive adaptive peak recovery.
  Panel B: sg4 heatmap -- nu x SEED_BETA 2D grid (Exp2).
           If product hypothesis holds, iso-product diagonals should be similar.
  Panel C: Product (nu*SB) vs sg4 scatter -- tests if nu*SB collapses the curve.
           Colored by nu value. If collapse: single line. If not: separated curves.

Output: paper12_figure1.pdf, paper12_figure1.png
"""
import json, os
import numpy as np
import math
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DIR = os.path.dirname(__file__)

r1 = json.load(open(os.path.join(DIR, "results", "paper12_exp1_results.json")))
r2 = json.load(open(os.path.join(DIR, "results", "paper12_exp2_results.json")))

DEATH_PS = [0.0001, 0.0002, 0.0003, 0.0005, 0.001, 0.002,
            0.003, 0.005, 0.010, 0.020, 0.040, 0.080]
SEED_BETA_VALS = [0.00, 0.05, 0.10, 0.25, 0.50]
NU_VALS = [0.001, 0.002, 0.005, 0.010]
SB_VALS = [0.05, 0.10, 0.25, 0.50]

COLORS5 = ["#d62728", "#ff7f0e", "#2ca02c", "#1f77b4", "#9467bd"]
FIELD_DECAY = 0.9997
nu_cryst = abs(math.log(FIELD_DECAY)) / math.log(2)

fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))

# --- Panel A: SEED_BETA sweep ---
ax = axes[0]
SB_LABELS = [
    "SB=0.00 (ablated)",
    "SB=0.05 (weak relay)",
    "SB=0.10 (moderate relay)",
    "SB=0.25 (reference)",
    "SB=0.50 (strong relay)",
]
for si, sb in enumerate(SEED_BETA_VALS):
    y = [float(np.mean([v["sg4"] for v in r1[f"{sb},{dp}"]]))
         if f"{sb},{dp}" in r1 else np.nan
         for dp in DEATH_PS]
    lw = 2.2 if si == 3 else 1.5
    ls = "-" if si == 3 else ("--" if si > 0 else ":")
    ax.plot(DEATH_PS, y, color=COLORS5[si], lw=lw, ls=ls,
            marker="o", ms=3.5, label=SB_LABELS[si])
ax.axvline(x=nu_cryst, color="gray", lw=1.0, ls=":", alpha=0.7,
           label=r"$\nu_{\rm cryst}=4.3\times10^{-4}$")
ax.set_xscale("log")
ax.set_xlabel(r"Turnover rate $\nu$", fontsize=10)
ax.set_ylabel(r"sg4", fontsize=10)
ax.set_title(r"\textbf{A.} SEED\_BETA sweep", fontsize=10)
ax.legend(fontsize=6.5, loc="upper right", framealpha=0.85)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# --- Panel B: sg4 heatmap (nu x SB) ---
ax = axes[1]
heatmap = np.zeros((len(NU_VALS), len(SB_VALS)))
for i, nu in enumerate(NU_VALS):
    for j, sb in enumerate(SB_VALS):
        key = f"{sb},{nu}"
        if key in r2:
            heatmap[i, j] = float(np.mean([v["sg4"] for v in r2[key]]))
        else:
            heatmap[i, j] = np.nan

im = ax.imshow(heatmap, aspect="auto", origin="lower",
               cmap="viridis", interpolation="nearest")
ax.set_xticks(range(len(SB_VALS)))
ax.set_xticklabels([f"{sb:.2f}" for sb in SB_VALS], fontsize=8)
ax.set_yticks(range(len(NU_VALS)))
ax.set_yticklabels([f"{nu:.3f}" for nu in NU_VALS], fontsize=8)
ax.set_xlabel("SEED_BETA", fontsize=10)
ax.set_ylabel(r"Turnover rate $\nu$", fontsize=10)
ax.set_title(r"\textbf{B.} sg4 heatmap ($\nu$ x SEED\_BETA)", fontsize=10)
plt.colorbar(im, ax=ax, label="sg4")
# Overlay iso-product contours
for i, nu in enumerate(NU_VALS):
    for j, sb in enumerate(SB_VALS):
        prod = nu * sb
        ax.text(j, i, f"{prod:.0e}", ha="center", va="center",
                fontsize=5, color="white", alpha=0.8)

# --- Panel C: Product collapse scatter ---
ax = axes[2]
NU_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
for i, nu in enumerate(NU_VALS):
    products = []
    sg4_vals = []
    for j, sb in enumerate(SB_VALS):
        key = f"{sb},{nu}"
        if key in r2:
            products.append(nu * sb)
            sg4_vals.append(float(np.mean([v["sg4"] for v in r2[key]])))
    ax.scatter(products, sg4_vals, color=NU_COLORS[i], s=50, zorder=3,
               label=r"$\nu$=" + f"{nu:.3f}")
    ax.plot(products, sg4_vals, color=NU_COLORS[i], lw=1.0, ls="--", alpha=0.5)
ax.set_xscale("log")
ax.set_xlabel(r"Product $\nu \times$ SEED\_BETA", fontsize=10)
ax.set_ylabel(r"sg4", fontsize=10)
ax.set_title(r"\textbf{C.} Product collapse ($\nu \times$ SEED\_BETA)", fontsize=10)
ax.legend(fontsize=7, framealpha=0.85)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.text(0.05, 0.95, "Collapse: curves overlap\nNo collapse: separated",
        transform=ax.transAxes, fontsize=7, va="top", alpha=0.7,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.5))

fig.suptitle(
    r"Copy-forward relay: SEED\_BETA governs structural maintenance independent of consolidation",
    fontsize=9.5, y=1.01
)
fig.tight_layout()

OUT = os.path.join(DIR, "paper12_figure1")
fig.savefig(OUT + ".pdf", bbox_inches="tight")
fig.savefig(OUT + ".png", dpi=150, bbox_inches="tight")
print(f"Saved {OUT}.pdf and {OUT}.png")
