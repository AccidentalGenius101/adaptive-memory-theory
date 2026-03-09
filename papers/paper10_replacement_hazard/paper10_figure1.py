"""
paper10_figure1.py -- Figure 1 for Paper 10

Three-panel figure:
  Panel A: sg4 vs nu for all WR conditions (WD=15 fixed).
           Shows sg4 amplitude increasing with WR, but peak location (nu*) stable.
  Panel B: sg4 vs Psi_approx across all (nu, WR) pairs.
           Shows the failure to collapse: wide spread at any Psi value.
  Panel C: nu* vs WR with two-boundary model.
           Upper boundary: nu_max = P_calm * FIELD_ALPHA * FD^(1/nu) (Psi=1).
           Lower boundary: nu_cryst (constant, governed by FIELD_DECAY).
           Observed nu* sits between them, shifts slowly.

Output: paper10_figure1.pdf, paper10_figure1.png
"""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

EXP1_FILE = os.path.join(os.path.dirname(__file__), "results", "paper10_exp1_results.json")
results = json.load(open(EXP1_FILE))

WR_VALS  = [0.8, 1.6, 2.4, 4.8, 7.2]
DEATH_PS = [0.0001, 0.0002, 0.0003, 0.0005, 0.001, 0.002,
            0.003, 0.005, 0.010, 0.020, 0.040, 0.080]
WAVE_DUR = 15; SS = 10; FIELD_ALPHA = 0.16; FIELD_DECAY = 0.9997

COLORS = ["#9467bd", "#2077b4", "#27a027", "#d62728", "#ff7f0e"]
LABELS = [r"$\mathtt{WR}=0.8$", r"$\mathtt{WR}=1.6$",
          r"$\mathtt{WR}=2.4$", r"$\mathtt{WR}=4.8$ (ref)",
          r"$\mathtt{WR}=7.2$"]

def p_calm_approx(wr):
    R = wr / WAVE_DUR
    return max((1 - R/2) ** (SS + WAVE_DUR - 1), 1e-10)

def psi(nu, wr):
    pc = p_calm_approx(wr)
    survival = FIELD_DECAY ** (1.0 / nu) if nu > 0 else 0.0
    lam_c = pc * FIELD_ALPHA * survival
    return nu / lam_c if lam_c > 0 else np.inf

# Build data arrays
sg4_table = {}    # (wr, nu) -> mean sg4
for wr in WR_VALS:
    for nu in DEATH_PS:
        key = f"{wr},{nu}"
        if key in results:
            sg4_table[(wr, nu)] = float(np.mean([v["sg4"] for v in results[key]]))

# nu* per WR (argmax of sg4)
nu_star_obs = {}
for wi, wr in enumerate(WR_VALS):
    vals = [sg4_table.get((wr, nu), 0) for nu in DEATH_PS]
    best = int(np.argmax(vals))
    nu_star_obs[wr] = DEATH_PS[best]

# Two-boundary model
# nu_max(WR): Psi = 1 solved iteratively
def nu_max_pred(wr, psi_target=1.0):
    """Solve nu: nu / (P_calm * FIELD_ALPHA * FD^(1/nu)) = psi_target"""
    pc = p_calm_approx(wr)
    # Iterate: start from approx ignoring FD survival
    nu = psi_target * pc * FIELD_ALPHA
    for _ in range(30):
        survival = FIELD_DECAY ** (1.0 / max(nu, 1e-8))
        nu = psi_target * pc * FIELD_ALPHA * survival
    return nu

nu_max_vals = [nu_max_pred(wr) for wr in WR_VALS]
# nu_cryst: approximate -- structures survive when FIELD_DECAY^(1/nu) > 0.5
# 0.9997^(1/nu) = 0.5 => 1/nu = ln(0.5)/ln(0.9997) => nu = ln(0.9997)/ln(0.5)
nu_cryst = abs(np.log(FIELD_DECAY) / np.log(0.5))   # ≈ 0.000231

# --- Figure ---
fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))

# ----- Panel A: sg4 vs nu -----
ax = axes[0]
for wi, wr in enumerate(WR_VALS):
    x = DEATH_PS
    y = [sg4_table.get((wr, nu), np.nan) for nu in DEATH_PS]
    lw = 2.2 if wi == 3 else 1.5
    ls = "-" if wi == 3 else "--"
    ax.plot(x, y, color=COLORS[wi], lw=lw, ls=ls,
            marker="o", ms=3.5, label=LABELS[wi])
ax.set_xscale("log")
ax.set_xlabel(r"Turnover rate $\nu$", fontsize=10)
ax.set_ylabel(r"Zone differentiation $\mathrm{sg4}$", fontsize=10)
ax.set_title(r"\textbf{A.} sg4 vs.\ $\nu$ ($\mathtt{WD}=15$)", fontsize=10)
ax.legend(fontsize=7.5, loc="lower left", framealpha=0.85)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
# Shade nu* band
ax.axvline(x=0.0005, color="gray", lw=0.8, ls=":", alpha=0.6)
ax.axvline(x=0.001,  color="gray", lw=0.8, ls=":", alpha=0.6)
ax.text(0.00075, ax.get_ylim()[1]*0.05 if ax.get_ylim()[1] > 0 else 100,
        r"$\nu^*$ band", ha="center", fontsize=7, color="gray")

# ----- Panel B: sg4 vs Psi -----
ax = axes[1]
for wi, wr in enumerate(WR_VALS):
    x_psi = [psi(nu, wr) for nu in DEATH_PS]
    y_sg4 = [sg4_table.get((wr, nu), np.nan) for nu in DEATH_PS]
    # clip to Psi <= 10 for readability
    mask = [p <= 10.0 for p in x_psi]
    xp = [x_psi[i] for i in range(len(mask)) if mask[i]]
    yp = [y_sg4[i] for i in range(len(mask)) if mask[i]]
    lw = 2.2 if wi == 3 else 1.5
    ls = "-" if wi == 3 else "--"
    ax.plot(xp, yp, color=COLORS[wi], lw=lw, ls=ls,
            marker="o", ms=3.5, label=LABELS[wi])
ax.set_xlabel(r"$\Psi = \nu\,/\,(P_{\rm calm} \cdot \alpha_F \cdot D^{1/\nu})$", fontsize=9)
ax.set_ylabel(r"Zone differentiation $\mathrm{sg4}$", fontsize=10)
ax.set_title(r"\textbf{B.} sg4 vs.\ $\Psi$: no collapse", fontsize=10)
ax.legend(fontsize=7.5, loc="upper right", framealpha=0.85)
ax.set_xlim(-0.1, 5.0)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# ----- Panel C: nu* vs WR with two-boundary model -----
ax = axes[2]
wr_fine = np.linspace(0.4, 7.5, 200)
nu_max_line = [nu_max_pred(wr) for wr in wr_fine]
nu_cryst_line = [nu_cryst] * len(wr_fine)
geom_mean_line = [np.sqrt(nu_max_pred(wr) * nu_cryst) for wr in wr_fine]

ax.semilogy(wr_fine, nu_max_line, color="salmon", lw=1.5, ls="--",
            label=r"$\nu_{\rm max}$ ($\Psi=1$)")
ax.semilogy(wr_fine, nu_cryst_line, color="steelblue", lw=1.5, ls=":",
            label=r"$\nu_{\rm cryst}$ (FIELD\_DECAY)")
ax.semilogy(wr_fine, geom_mean_line, color="gray", lw=1.5, ls="-",
            label=r"Geometric mean ($\sqrt{\nu_{\rm max}\,\nu_{\rm cryst}}$)")

# Observed nu*
obs_wr  = [wr for wr in WR_VALS if wr < 7.0]  # exclude noisy WR=7.2
obs_nu  = [nu_star_obs[wr] for wr in obs_wr]
ax.scatter(obs_wr, obs_nu, color=COLORS[:len(obs_wr)], zorder=5, s=70,
           edgecolors="k", linewidths=0.7, label=r"Empirical $\nu^*$")

ax.set_xlabel(r"Wave rate $\mathtt{WR}$", fontsize=10)
ax.set_ylabel(r"Optimal turnover $\nu^*$", fontsize=10)
ax.set_title(r"\textbf{C.} Two-boundary model vs.\ observed $\nu^*$", fontsize=10)
ax.legend(fontsize=7.5, loc="upper right", framealpha=0.85)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

fig.suptitle(
    r"$\nu^*$ is approximately invariant to WR: Ψ framework fails to predict the adaptive peak",
    fontsize=10, y=1.01
)
fig.tight_layout()

OUT = os.path.join(os.path.dirname(__file__), "paper10_figure1")
fig.savefig(OUT + ".pdf", bbox_inches="tight")
fig.savefig(OUT + ".png", dpi=150, bbox_inches="tight")
print(f"Saved {OUT}.pdf and {OUT}.png")

# Summary stats
print(f"\nnu* summary (WD={WAVE_DUR}):")
for wr in WR_VALS:
    print(f"  WR={wr:.1f}: nu*={nu_star_obs[wr]:.4f}  nu_max_pred={nu_max_pred(wr):.6f}  nu_cryst={nu_cryst:.6f}")
print(f"\nnu_cryst = {nu_cryst:.6f} (FIELD_DECAY^(1/nu) = 0.5 threshold)")
