"""
paper28_figure1.py  --  Paper 28 main figure

4 panels:
  A: Column profiles at different kappa (Exp A, one representative seed per kappa)
     Shows whether profile is step-like or smooth; how kappa shapes spatial structure
  B: sg4 vs kappa on log-log (Exp A)
     Key finding: sg4 weakly decreases with kappa (slope ~ -0.11, not Allen-Cahn -0.5)
  C: sg4(t) and xi_diff1(t) over time (Exp B)
     Key finding: CV_xi ~ 1, interface is a dynamically fluctuating object
  D: sg4 along isothermal direction (Exp C, kappa/FA = 0.05 fixed)
     Key finding: sg4 increases with absolute FA (FA is the primary driver)
"""
import json, os, math, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.optimize import curve_fit
from scipy.stats import linregress

# ── Load parameters from experiments module ───────────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))
from paper28_experiments import (
    KAPPA_VALS_A, N_SEEDS_A, T_END_A, CPS_A,
    N_SEEDS_B, CPS_B,
    ISOTHERMAL_PAIRS, N_SEEDS_C, CPS_C,
    KAPPA_STD, FA_STD, HALF, H, HS, N_ZONES, ZONE_W,
    key_a, key_b, key_c,
    _get_xi, _get_A, fit_tanh_boundary,
)

RESULTS_FILE = os.path.join(os.path.dirname(__file__),
                            "results", "paper28_results.json")
with open(RESULTS_FILE) as f:
    R = json.load(f)

T_LAST_A = CPS_A[-1]   # 4000
T_LAST_C = CPS_C[-1]   # 4000

# ── Helpers ───────────────────────────────────────────────────────────────────
def best_seed_profile(kappa):
    """
    Return col_profile full[:,0] for the seed with the largest
    |zone0_mean - zone1_mean|, sign-aligned so zone 0 >= zone 1.
    """
    best_contrast = -1
    best_prof     = None
    cols_per_zone = ZONE_W
    for s in range(N_SEEDS_A):
        k  = key_a(kappa, s)
        cp = R.get(k, {}).get(f"col_profile_{T_LAST_A}")
        if cp is None:
            continue
        full0 = np.array(cp["full"])[:, 0]
        z0    = full0[:cols_per_zone].mean()
        z1    = full0[cols_per_zone:2*cols_per_zone].mean()
        contrast = abs(z0 - z1)
        if contrast > best_contrast:
            best_contrast = contrast
            best_prof = full0 * (1 if z0 >= z1 else -1)
    return best_prof

def sg4_kappa(kappa, t):
    vals = [R.get(key_a(kappa, s), {}).get(f"sg4_{t}", float("nan"))
            for s in range(N_SEEDS_A)]
    return [v for v in vals if not math.isnan(v)]

# ── Figure setup ──────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(13, 9))
gs  = GridSpec(2, 2, figure=fig, hspace=0.40, wspace=0.35)
ax  = {k: fig.add_subplot(gs[r, c]) for k, (r, c) in
       zip("ABCD", [(0,0),(0,1),(1,0),(1,1)])}

kappa_colors = plt.cm.viridis(np.linspace(0.10, 0.90, len(KAPPA_VALS_A)))
x_cols       = np.arange(HALF)
boundaries   = [ZONE_W * (z + 1) - 0.5 for z in range(N_ZONES - 1)]

# ─────────────────────────────────────────────────────────────────────────────
# Panel A: Column profiles (best seed per kappa)
# ─────────────────────────────────────────────────────────────────────────────
for kappa, col in zip(KAPPA_VALS_A, kappa_colors):
    prof = best_seed_profile(kappa)
    if prof is None:
        continue
    ax["A"].plot(x_cols, prof, color=col,
                 label=rf"$\kappa={kappa}$", lw=1.4)
for xb in boundaries:
    ax["A"].axvline(xb, color="gray", lw=0.8, ls="--", alpha=0.6)
ax["A"].set_xlabel("Column index", fontsize=10)
ax["A"].set_ylabel(r"Field $F_{col}$ (HS[0], row-avg)", fontsize=10)
ax["A"].set_title(r"(A) Column profiles at $T=4000$", fontsize=11, fontweight="bold")
ax["A"].legend(fontsize=7, ncol=2, loc="upper right")
ax["A"].set_xticks([0, 5, 10, 15, 19])
ax["A"].set_xticklabels(["0", "5\n(z0|z1)", "10\n(z1|z2)", "15\n(z2|z3)", "19"])
ax["A"].tick_params(labelsize=8)

# ─────────────────────────────────────────────────────────────────────────────
# Panel B: sg4 vs kappa (log-log)
# ─────────────────────────────────────────────────────────────────────────────
kappa_arr = np.array(KAPPA_VALS_A)
sg4_means, sg4_stds = [], []
for kappa in KAPPA_VALS_A:
    vals = sg4_kappa(kappa, T_LAST_A)
    sg4_means.append(np.mean(vals) if vals else float("nan"))
    sg4_stds.append(np.std(vals)   if len(vals) > 1 else 0.0)
sg4_means = np.array(sg4_means)
sg4_stds  = np.array(sg4_stds)

valid = ~np.isnan(sg4_means)
ax["B"].errorbar(kappa_arr[valid], sg4_means[valid], yerr=sg4_stds[valid],
                 fmt="o-", color="steelblue", lw=1.8, ms=6, capsize=3,
                 label="Measured sg4")

# Log-log fit
lk  = np.log(kappa_arr[valid])
lsg = np.log(sg4_means[valid])
slope, intercept, r, *_ = linregress(lk, lsg)
k_fit = np.logspace(np.log10(kappa_arr.min()), np.log10(kappa_arr.max()), 50)
ax["B"].plot(k_fit, np.exp(intercept) * k_fit ** slope, "k--", lw=1.2,
             label=rf"fit: slope={slope:.2f}")

# Allen-Cahn reference (slope = -0.5)
k_ref = np.array([kappa_arr.min(), kappa_arr.max()])
ref_level = sg4_means[valid].mean() * (k_ref / kappa_arr[valid].mean()) ** (-0.5)
ax["B"].plot(k_ref, ref_level, "r:", lw=1.2, label="A-C slope = -0.5")

ax["B"].set_xscale("log"); ax["B"].set_yscale("log")
ax["B"].set_xlabel(r"$\kappa$", fontsize=10)
ax["B"].set_ylabel("sg4 (zone differentiation)", fontsize=10)
ax["B"].set_title(r"(B) sg4 vs $\kappa$ at $T=4000$", fontsize=11, fontweight="bold")
ax["B"].legend(fontsize=8)
ax["B"].tick_params(labelsize=8)
ax["B"].annotate(rf"$R^2={r**2:.2f}$", xy=(0.05, 0.12),
                 xycoords="axes fraction", fontsize=8)

# ─────────────────────────────────────────────────────────────────────────────
# Panel C: Temporal dynamics (Exp B) — sg4(t) and xi_diff1(t)
# ─────────────────────────────────────────────────────────────────────────────
t_arr = np.array(CPS_B)
sg4_B, sg4_B_std = [], []
xi_B,  xi_B_std  = [], []
for t in CPS_B:
    s4_t = [R.get(key_b(s), {}).get(f"sg4_{t}", float("nan"))
            for s in range(N_SEEDS_B)]
    s4_t = [v for v in s4_t if not math.isnan(v)]
    sg4_B.append(np.mean(s4_t) if s4_t else float("nan"))
    sg4_B_std.append(np.std(s4_t) if len(s4_t) > 1 else 0.0)

    xi_t = []
    for s in range(N_SEEDS_B):
        cp = R.get(key_b(s), {}).get(f"col_profile_{t}")
        if cp is not None:
            xi_s = _get_xi(cp, "full0")
            if not math.isnan(xi_s) and xi_s < 15:   # filter spurious fits
                xi_t.append(xi_s)
    xi_B.append(np.mean(xi_t) if xi_t else float("nan"))
    xi_B_std.append(np.std(xi_t) if len(xi_t) > 1 else 0.0)

sg4_B     = np.array(sg4_B)
sg4_B_std = np.array(sg4_B_std)
xi_B      = np.array(xi_B)
xi_B_std  = np.array(xi_B_std)

ax2 = ax["C"].twinx()

valid_sg = ~np.isnan(sg4_B)
ax["C"].fill_between(t_arr[valid_sg],
                     (sg4_B - sg4_B_std)[valid_sg],
                     (sg4_B + sg4_B_std)[valid_sg],
                     alpha=0.2, color="steelblue")
ax["C"].plot(t_arr[valid_sg], sg4_B[valid_sg],
             color="steelblue", lw=1.8, label="sg4 (left)")

valid_xi = ~np.isnan(xi_B)
if valid_xi.any():
    ax2.fill_between(t_arr[valid_xi],
                     (xi_B - xi_B_std)[valid_xi],
                     (xi_B + xi_B_std)[valid_xi],
                     alpha=0.15, color="coral")
    ax2.plot(t_arr[valid_xi], xi_B[valid_xi],
             color="coral", lw=1.8, ls="--", label=r"$\xi$ fit (right)")
    # Compute and annotate CV
    valid_xi_vals = xi_B[valid_xi]
    cv = float(np.std(valid_xi_vals) / np.mean(valid_xi_vals)) if np.mean(valid_xi_vals) > 0 else float("nan")
    ax2.set_ylabel(r"Interface width $\xi$ (tanh fit, cols)", fontsize=9, color="coral")
    ax2.tick_params(axis="y", colors="coral", labelsize=8)
    ax2.annotate(rf"CV$_\xi$={cv:.2f}", xy=(0.55, 0.88),
                 xycoords="axes fraction", fontsize=9, color="coral")

ax["C"].set_xlabel("Time step", fontsize=10)
ax["C"].set_ylabel("sg4", fontsize=10, color="steelblue")
ax["C"].tick_params(axis="y", colors="steelblue", labelsize=8)
ax["C"].tick_params(axis="x", labelsize=8)
ax["C"].set_title(r"(C) Interface temporal dynamics ($\kappa=0.020$, $T=6000$)",
                  fontsize=11, fontweight="bold")
lines1, labs1 = ax["C"].get_legend_handles_labels()
lines2, labs2 = ax2.get_legend_handles_labels()
ax["C"].legend(lines1 + lines2, labs1 + labs2, fontsize=8, loc="lower right")

# ─────────────────────────────────────────────────────────────────────────────
# Panel D: Isothermal test — sg4 vs FA at fixed kappa/FA = 0.05
# ─────────────────────────────────────────────────────────────────────────────
iso_FA    = [fa for _, fa in ISOTHERMAL_PAIRS]
iso_kappa = [kappa for kappa, _ in ISOTHERMAL_PAIRS]
sg4_C_mean, sg4_C_std = [], []
xi_C_mean = []
for kappa, fa in ISOTHERMAL_PAIRS:
    vals = [R.get(key_c(kappa, fa, s), {}).get(f"sg4_{T_LAST_C}", float("nan"))
            for s in range(N_SEEDS_C)]
    vals = [v for v in vals if not math.isnan(v)]
    sg4_C_mean.append(np.mean(vals) if vals else float("nan"))
    sg4_C_std.append(np.std(vals)   if len(vals) > 1 else 0.0)

    xis = []
    for s in range(N_SEEDS_C):
        cp = R.get(key_c(kappa, fa, s), {}).get(f"col_profile_{T_LAST_C}")
        if cp is not None:
            xi_s = _get_xi(cp, "full0")
            if not math.isnan(xi_s) and xi_s < 15:
                xis.append(xi_s)
    xi_C_mean.append(np.mean(xis) if xis else float("nan"))

sg4_C_mean = np.array(sg4_C_mean)
sg4_C_std  = np.array(sg4_C_std)
xi_C_mean  = np.array(xi_C_mean)

# Plot sg4 vs FA (primary) with kappa labeled on points
ax["D"].errorbar(iso_FA, sg4_C_mean, yerr=sg4_C_std,
                 fmt="o-", color="steelblue", lw=1.8, ms=7, capsize=3,
                 label=r"sg4 ($\kappa/FA=0.05$ fixed)")
for i, (kappa, fa) in enumerate(ISOTHERMAL_PAIRS):
    if not math.isnan(sg4_C_mean[i]):
        ax["D"].annotate(rf"$\kappa={kappa}$",
                         (fa, sg4_C_mean[i]),
                         textcoords="offset points",
                         xytext=(4, -12), fontsize=7)

# Allen-Cahn prediction: sg4 should be constant at fixed kappa/FA
ymin = max(0, sg4_C_mean[~np.isnan(sg4_C_mean)].min() * 0.9)
ymax = sg4_C_mean[~np.isnan(sg4_C_mean)].max() * 1.05
ax["D"].axhline(sg4_C_mean[~np.isnan(sg4_C_mean)].mean(),
                color="red", ls=":", lw=1.4,
                label="A-C prediction (constant)")

ax["D"].set_xlabel("FA (consolidation rate)", fontsize=10)
ax["D"].set_ylabel("sg4 (zone differentiation)", fontsize=10)
ax["D"].set_title(r"(D) Isothermal test: $\kappa/FA = 0.05$ fixed, $T=4000$",
                  fontsize=11, fontweight="bold")
ax["D"].legend(fontsize=8)
ax["D"].tick_params(labelsize=8)
ax["D"].set_xticks(iso_FA)
ax["D"].set_xticklabels([str(fa) for fa in iso_FA])

# ── Title and save ────────────────────────────────────────────────────────────
fig.suptitle(
    "Paper 28: The Zone Boundary as a Dynamical Extended Object\n"
    r"Not an Allen--Cahn kink; driven by copy-forward stochasticity and FA",
    fontsize=12, fontweight="bold", y=0.99
)
out_base = os.path.join(os.path.dirname(__file__), "paper28_figure1")
fig.savefig(out_base + ".pdf", bbox_inches="tight", dpi=150)
fig.savefig(out_base + ".png", bbox_inches="tight", dpi=150)
print(f"Saved: {out_base}.pdf / .png")
plt.close(fig)
