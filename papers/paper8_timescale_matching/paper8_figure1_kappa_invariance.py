"""
paper8_figure1_kappa_invariance.py -- Figure 1 for Paper 8

Visual proof that nu_opt is invariant to propagation strength (kappa).
Data source: Paper 7 Exp3 (kappa x nu grid, 5x5, 5 seeds each).

Panel A: sg4 vs nu curves, one line per kappa value.
         All five lines peak at the same nu = 0.001.
         This is the core result of Paper 8.

Panel B: nu_opt vs kappa (horizontal line at nu=0.001).
         Explicit summary of the invariance.

Output: paper8_figure1.pdf, paper8_figure1.png
"""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# --- Data ---
RESULTS_FILE = os.path.join(
    os.path.dirname(__file__),
    "..", "paper7_regime_structure", "results", "paper7_exp3_results.json"
)
results = json.load(open(RESULTS_FILE))

NU_VALS    = [0.0003, 0.001, 0.005, 0.015, 0.050]
KAPPA_VALS = [0.000,  0.005, 0.020, 0.060, 0.150]

# Build sg4 matrix [kappa x nu]
sg4 = np.zeros((len(KAPPA_VALS), len(NU_VALS)))
for ki, kp in enumerate(KAPPA_VALS):
    for ni, nu in enumerate(NU_VALS):
        key = f"{nu},{kp}"
        sg4[ki, ni] = float(np.mean([v["sg4"] for v in results[key]]))

# nu_opt per kappa
nu_opt = [NU_VALS[int(np.argmax(sg4[ki]))] for ki in range(len(KAPPA_VALS))]

# --- Plot ---
COLORS = ["#555555", "#2077b4", "#27a027", "#d62728", "#9467bd"]
LABELS = [r"$\kappa = 0.000$ (no diffusion)",
          r"$\kappa = 0.005$",
          r"$\kappa = 0.020$ (baseline)",
          r"$\kappa = 0.060$",
          r"$\kappa = 0.150$"]

fig, axes = plt.subplots(1, 2, figsize=(9, 3.8))

# ----- Panel A: sg4 vs nu curves -----
ax = axes[0]
for ki in range(len(KAPPA_VALS)):
    lw  = 2.2 if ki == 2 else 1.5
    ls  = "-" if ki == 2 else "--"
    ax.plot(range(len(NU_VALS)), sg4[ki], color=COLORS[ki],
            lw=lw, ls=ls, marker="o", ms=4, label=LABELS[ki])

ax.axvline(x=1, color="gray", lw=1.0, ls=":", alpha=0.7)
ax.text(1.05, ax.get_ylim()[1] * 0.97 if ax.get_ylim()[1] > 0 else 400,
        r"$\nu^* = 0.001$", color="gray", fontsize=8, va="top")

ax.set_xticks(range(len(NU_VALS)))
ax.set_xticklabels([r"$3{\times}10^{-4}$", r"$10^{-3}$", r"$5{\times}10^{-3}$",
                    r"$1.5{\times}10^{-2}$", r"$5{\times}10^{-2}$"], fontsize=8)
ax.set_xlabel(r"Turnover rate $\nu$", fontsize=10)
ax.set_ylabel(r"Zone differentiation $\mathrm{sg4}$", fontsize=10)
ax.set_title(r"\textbf{A.} $\mathrm{sg4}$ vs.\ $\nu$ at each $\kappa$", fontsize=10)
ax.legend(fontsize=7.5, loc="upper right", framealpha=0.85)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# ----- Panel B: nu_opt vs kappa (horizontal line) -----
ax = axes[1]
ax.scatter(KAPPA_VALS, nu_opt, color=[COLORS[ki] for ki in range(len(KAPPA_VALS))],
           zorder=5, s=60, edgecolors="k", linewidths=0.6)
ax.axhline(y=0.001, color="gray", lw=1.2, ls="--", label=r"$\nu^* = 0.001$", zorder=1)

ax.set_xlim(-0.01, 0.165)
ax.set_ylim(0.0, 0.0035)
ax.set_xticks(KAPPA_VALS)
ax.set_xticklabels([str(k) for k in KAPPA_VALS], fontsize=8)
ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.4f"))
ax.set_xlabel(r"Propagation strength $\kappa$", fontsize=10)
ax.set_ylabel(r"Optimal turnover rate $\nu^*$", fontsize=10)
ax.set_title(r"\textbf{B.} $\nu^*$ is independent of $\kappa$", fontsize=10)
ax.legend(fontsize=9, loc="upper right", framealpha=0.85)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

fig.suptitle(
    r"Turnover optimum $\nu^* \approx 0.001$ is invariant to propagation strength $\kappa$",
    fontsize=11, y=1.01
)
fig.tight_layout()

OUT = os.path.join(os.path.dirname(__file__), "paper8_figure1")
fig.savefig(OUT + ".pdf", bbox_inches="tight")
fig.savefig(OUT + ".png", dpi=150, bbox_inches="tight")
print(f"Saved {OUT}.pdf and {OUT}.png")
