"""
paper9_figure1_xi_collapse.py -- Figure 1 for Paper 9

Shows whether sg4 curves collapse onto the dimensionless ratio
Xi = nu * (N_act/4) * 2 * WAVE_DUR / WR.

Panel A: sg4 vs Xi for Exp1 (WAVE_DUR sweep, WR=4.8 fixed).
          If Xi governs the regime, all curves should peak at the same Xi*.

Panel B: nu* vs WAVE_DUR with prediction line from balance equation.
          Shows how closely the empirical nu* follows nu*_pred = WR/(2*WD*N_act/4).

Output: paper9_figure1.pdf, paper9_figure1.png
"""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# --- Data ---
EXP1_FILE = os.path.join(os.path.dirname(__file__), "results", "paper9_exp1_results.json")
results1 = json.load(open(EXP1_FILE))

WR = 4.8
N_ACT = 400
DEATH_PS = [0.0001, 0.0002, 0.0003, 0.0005, 0.001, 0.002,
            0.003, 0.005, 0.010, 0.020, 0.040, 0.080]
WAVE_DUR_VALS = [7, 15, 30, 60]

# sg4 table [WAVE_DUR x nu]
sg4_e1 = np.zeros((len(WAVE_DUR_VALS), len(DEATH_PS)))
for wi, wd in enumerate(WAVE_DUR_VALS):
    for ni, dp in enumerate(DEATH_PS):
        key = f"{wd},{dp}"
        sg4_e1[wi, ni] = float(np.mean([v["sg4"] for v in results1[key]]))

# Xi = nu * (N_act/4) * 2 * WAVE_DUR / WR
def xi(nu, wd):
    return nu * (N_ACT / 4) * 2 * wd / WR

# nu_opt per WD
nu_opt = [DEATH_PS[int(np.argmax(sg4_e1[wi]))] for wi in range(len(WAVE_DUR_VALS))]
xi_opt = [xi(nu_opt[wi], WAVE_DUR_VALS[wi]) for wi in range(len(WAVE_DUR_VALS))]
nu_pred = [WR / (2 * wd * (N_ACT / 4)) for wd in WAVE_DUR_VALS]

# --- Plot ---
COLORS = ["#d62728", "#2077b4", "#27a027", "#9467bd"]
LABELS = [r"$\mathtt{WD}=7$, $R=0.69$",
          r"$\mathtt{WD}=15$, $R=0.32$ (reference)",
          r"$\mathtt{WD}=30$, $R=0.16$",
          r"$\mathtt{WD}=60$, $R=0.08$"]

fig, axes = plt.subplots(1, 2, figsize=(10, 4.0))

# ----- Panel A: sg4 vs Xi -----
ax = axes[0]
for wi, wd in enumerate(WAVE_DUR_VALS):
    xi_vals = [xi(dp, wd) for dp in DEATH_PS]
    # only show up to Xi=5 for readability
    mask = [x <= 5.0 for x in xi_vals]
    x_plot = [xi_vals[i] for i in range(len(mask)) if mask[i]]
    y_plot = [sg4_e1[wi, i] for i in range(len(mask)) if mask[i]]
    lw = 2.2 if wi == 1 else 1.5
    ls = "-" if wi == 1 else "--"
    ax.plot(x_plot, y_plot, color=COLORS[wi], lw=lw, ls=ls,
            marker="o", ms=3.5, label=LABELS[wi])

# Mark Xi* for each condition
for wi in range(len(WAVE_DUR_VALS)):
    if xi_opt[wi] <= 5.0:
        peak_sg4 = sg4_e1[wi, int(np.argmax(sg4_e1[wi]))]
        ax.axvline(x=xi_opt[wi], color=COLORS[wi], lw=0.8, ls=":", alpha=0.7)

ax.set_xlabel(r"Dimensionless ratio $\Xi = \nu \cdot \frac{N}{4} \cdot \frac{2\,\mathtt{WD}}{\mathtt{WR}}$",
              fontsize=9)
ax.set_ylabel(r"Zone differentiation $\mathrm{sg4}$", fontsize=10)
ax.set_title(r"\textbf{A.} $\mathrm{sg4}$ vs.\ $\Xi$ (Exp 1, $\mathtt{WR}=4.8$)", fontsize=10)
ax.legend(fontsize=7.5, loc="upper right", framealpha=0.85)
ax.set_xlim(-0.05, 3.5)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# ----- Panel B: nu_opt vs WAVE_DUR with prediction -----
ax = axes[1]
wd_fine = np.linspace(5, 65, 200)
nu_pred_line = [WR / (2 * wd * (N_ACT / 4)) for wd in wd_fine]
ax.plot(wd_fine, nu_pred_line, color="gray", lw=1.5, ls="--",
        label=r"$\nu^*_{\rm pred} = \mathtt{WR}/(2\mathtt{WD}\cdot N/4)$")
ax.scatter(WAVE_DUR_VALS, nu_opt, color=COLORS, zorder=5, s=70,
           edgecolors="k", linewidths=0.7, label=r"Empirical $\nu^*$")
for wi, (wd, nu) in enumerate(zip(WAVE_DUR_VALS, nu_opt)):
    ax.annotate(f"$\\Xi^*={xi_opt[wi]:.2f}$",
                xy=(wd, nu), xytext=(wd + 2, nu * 1.4),
                fontsize=7, color=COLORS[wi],
                arrowprops=dict(arrowstyle="->", color=COLORS[wi], lw=0.7))

ax.set_yscale("log")
ax.set_xlabel(r"$\mathtt{WAVE\_DUR}$", fontsize=10)
ax.set_ylabel(r"Optimal turnover rate $\nu^*$", fontsize=10)
ax.set_title(r"\textbf{B.} $\nu^*$ vs.\ $\mathtt{WAVE\_DUR}$: prediction vs.\ observed", fontsize=10)
ax.legend(fontsize=8, loc="upper right", framealpha=0.85)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

fig.suptitle(
    r"$\Xi$ collapse test: partial confirmation. $\Xi^*$ varies across conditions.",
    fontsize=10, y=1.01
)
fig.tight_layout()

OUT = os.path.join(os.path.dirname(__file__), "paper9_figure1")
fig.savefig(OUT + ".pdf", bbox_inches="tight")
fig.savefig(OUT + ".png", dpi=150, bbox_inches="tight")
print(f"Saved {OUT}.pdf and {OUT}.png")
print(f"\nXi* values: " + ", ".join(f"WD={wd}: {xi:.3f}" for wd, xi in zip(WAVE_DUR_VALS, xi_opt)))
print(f"nu* values: " + ", ".join(f"WD={wd}: {nu:.4f}" for wd, nu in zip(WAVE_DUR_VALS, nu_opt)))
print(f"nu*_pred:   " + ", ".join(f"WD={wd}: {nu:.4f}" for wd, nu in zip(WAVE_DUR_VALS, nu_pred)))
