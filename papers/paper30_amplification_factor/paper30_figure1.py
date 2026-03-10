"""
paper30_figure1.py -- The Spatial Amplification Factor Gamma

4 panels:
  A: K_eff heatmap over (kappa, ZONE_W) grid
     Colour = K_eff; lower is more amplified.

  B: K_eff vs kappa (log-log) at ZONE_W=5
     Fit slope vs prediction +0.5. Reference line slope=0.5.

  C: K_eff vs ZONE_W (log-log) at kappa=0.020
     Fit slope vs prediction -1. Reference line slope=-1.

  D: K_eff vs nu (Exp B, kappa=0.020, ZONE_W=5)
     Shows non-monotone relationship; K_eff rises sharply at nu=0.003.
"""
import json, os, math, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.optimize import curve_fit
from scipy.stats import linregress

sys.path.insert(0, os.path.dirname(__file__))
from paper30_experiments import (
    FA_VALS, N_SEEDS, T_END, CPS, K_EFF_I,
    KAPPA_VALS, ZONE_W_VALS, NU_VALS_B,
    KAPPA_STD, ZONE_W_STD, NU_STD,
    key_a, key_b, fit_keff
)

RESULTS_FILE = os.path.join(os.path.dirname(__file__),
                            "results", "paper30_results.json")
with open(RESULTS_FILE) as f:
    R = json.load(f)

T_LAST = CPS[-1]   # 4000

# ── Helpers ───────────────────────────────────────────────────────────────────
def get_sg4_a(kappa, zone_w, fa):
    vals = [R.get(key_a(kappa, zone_w, fa, s), {}).get(f"sg4_{T_LAST}", float("nan"))
            for s in range(N_SEEDS)]
    return [v for v in vals if not math.isnan(v)]

def get_sg4_b(nu, fa):
    vals = [R.get(key_b(nu, fa, s), {}).get(f"sg4_{T_LAST}", float("nan"))
            for s in range(N_SEEDS)]
    return [v for v in vals if not math.isnan(v)]

def keff_a(kappa, zone_w):
    sg4_means = [float(np.mean(get_sg4_a(kappa, zone_w, fa)))
                 if get_sg4_a(kappa, zone_w, fa) else float("nan")
                 for fa in FA_VALS]
    _, K = fit_keff(FA_VALS, sg4_means)
    return K

def keff_b(nu):
    if nu == NU_STD:
        sg4_means = [float(np.mean(get_sg4_a(KAPPA_STD, ZONE_W_STD, fa)))
                     if get_sg4_a(KAPPA_STD, ZONE_W_STD, fa) else float("nan")
                     for fa in FA_VALS]
    else:
        sg4_means = [float(np.mean(get_sg4_b(nu, fa)))
                     if get_sg4_b(nu, fa) else float("nan")
                     for fa in FA_VALS]
    _, K = fit_keff(FA_VALS, sg4_means)
    return K

# ── Assemble grids ─────────────────────────────────────────────────────────────
keff_grid = np.full((len(KAPPA_VALS), len(ZONE_W_VALS)), float("nan"))
for i, kappa in enumerate(KAPPA_VALS):
    for j, zone_w in enumerate(ZONE_W_VALS):
        keff_grid[i, j] = keff_a(kappa, zone_w) or float("nan")

gamma_grid = K_EFF_I / keff_grid

# Column at ZONE_W=5 (index 1): K_eff vs kappa
kv_log  = [math.log(k) for k in KAPPA_VALS]
keffv_5 = [math.log(keff_grid[i, 1]) for i in range(len(KAPPA_VALS))
           if keff_grid[i, 1] > 0 and not math.isnan(keff_grid[i, 1])]
kv_5    = [kv_log[i] for i in range(len(KAPPA_VALS))
           if keff_grid[i, 1] > 0 and not math.isnan(keff_grid[i, 1])]
slope_k, int_k, *_ = linregress(kv_5, keffv_5) if len(kv_5) >= 2 else (float("nan"),)*5

# Row at kappa=0.020 (index 1): K_eff vs zone_w
zv_log   = [math.log(z) for z in ZONE_W_VALS]
keffv_20 = [math.log(keff_grid[1, j]) for j in range(len(ZONE_W_VALS))
            if keff_grid[1, j] > 0 and not math.isnan(keff_grid[1, j])]
zv_20    = [zv_log[j] for j in range(len(ZONE_W_VALS))
            if keff_grid[1, j] > 0 and not math.isnan(keff_grid[1, j])]
slope_z, int_z, *_ = linregress(zv_20, keffv_20) if len(zv_20) >= 2 else (float("nan"),)*5

# Exp B: K_eff vs nu
all_nu   = sorted([NU_STD] + list(NU_VALS_B))
nu_keff  = [(nu, keff_b(nu)) for nu in all_nu]
nu_keff  = [(nu, k) for nu, k in nu_keff if k and not math.isnan(k)]

# ── Figure ─────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(13, 9))
gs  = GridSpec(2, 2, figure=fig, hspace=0.42, wspace=0.38)
ax  = {k: fig.add_subplot(gs[r, c]) for k, (r, c) in
       zip("ABCD", [(0,0),(0,1),(1,0),(1,1)])}

# ─────────────────────────────────────────────────────────────────────────────
# Panel A: K_eff heatmap (kappa x ZONE_W)
# ─────────────────────────────────────────────────────────────────────────────
im = ax["A"].imshow(keff_grid, aspect="auto", origin="upper",
                    cmap="YlOrRd", vmin=0.0, vmax=0.15)
ax["A"].set_xticks(range(len(ZONE_W_VALS)))
ax["A"].set_xticklabels([str(z) for z in ZONE_W_VALS], fontsize=9)
ax["A"].set_yticks(range(len(KAPPA_VALS)))
ax["A"].set_yticklabels([f"{k:.3f}" for k in KAPPA_VALS], fontsize=9)
ax["A"].set_xlabel("ZONE_W", fontsize=10)
ax["A"].set_ylabel(r"$\kappa$", fontsize=11)
ax["A"].set_title(r"(A) $K_{\rm eff}$ heatmap ($\kappa$ $\times$ ZONE\_W)",
                  fontsize=11, fontweight="bold")

# Annotate cells with K_eff and Gamma
for i in range(len(KAPPA_VALS)):
    for j in range(len(ZONE_W_VALS)):
        k  = keff_grid[i, j]
        gm = gamma_grid[i, j]
        if not math.isnan(k):
            ax["A"].text(j, i, f"K={k:.4f}\n\u0393={gm:.1f}",
                         ha="center", va="center", fontsize=7.0,
                         color="black" if k < 0.08 else "white")

cbar = plt.colorbar(im, ax=ax["A"], fraction=0.046, pad=0.04)
cbar.set_label(r"$K_{\rm eff}$", fontsize=9)
cbar.ax.tick_params(labelsize=8)

# ─────────────────────────────────────────────────────────────────────────────
# Panel B: K_eff vs kappa (log-log) at ZONE_W=5
# ─────────────────────────────────────────────────────────────────────────────
kappa_arr = np.array(KAPPA_VALS)
keff_5    = keff_grid[:, 1]
valid_k   = ~np.isnan(keff_5) & (keff_5 > 0)

ax["B"].scatter(kappa_arr[valid_k], keff_5[valid_k],
                color="steelblue", s=70, zorder=5,
                label=r"Measured $K_{\rm eff}$ (ZONE\_W=5)")

if not math.isnan(slope_k):
    k_fine = np.logspace(np.log10(KAPPA_VALS[0]), np.log10(KAPPA_VALS[-1]), 100)
    ax["B"].plot(k_fine, np.exp(int_k) * k_fine**slope_k, "k-", lw=2.0,
                 label=rf"Fit: slope = {slope_k:.3f}")
    # Reference slope=+0.5 anchored at middle point
    mid_i = np.where(valid_k)[0][len(np.where(valid_k)[0])//2]
    ref_anch = keff_5[mid_i]
    ref_kapp = kappa_arr[mid_i]
    ax["B"].plot(k_fine, ref_anch * (k_fine / ref_kapp)**0.5, "r--", lw=1.4,
                 label="Prediction: slope = +0.5")

ax["B"].set_xscale("log"); ax["B"].set_yscale("log")
ax["B"].set_xlabel(r"$\kappa$ (diffusion)", fontsize=10)
ax["B"].set_ylabel(r"$K_{\rm eff}$", fontsize=10)
ax["B"].set_title(r"(B) $K_{\rm eff}$ vs $\kappa$ (ZONE\_W$=$5): log-log",
                  fontsize=11, fontweight="bold")
ax["B"].legend(fontsize=8, loc="upper left")
ax["B"].tick_params(labelsize=8)
ax["B"].annotate(rf"Measured slope: {slope_k:.3f}"
                 "\nPredicted: +0.50"
                 rf"\nRatio: {slope_k/0.5:.2f}$\times$",
                 xy=(0.60, 0.12), xycoords="axes fraction", fontsize=8.5,
                 bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow", ec="goldenrod"))

# ─────────────────────────────────────────────────────────────────────────────
# Panel C: K_eff vs ZONE_W (log-log) at kappa=0.020
# ─────────────────────────────────────────────────────────────────────────────
zone_arr = np.array(ZONE_W_VALS)
keff_20  = keff_grid[1, :]
valid_z  = ~np.isnan(keff_20) & (keff_20 > 0)

ax["C"].scatter(zone_arr[valid_z], keff_20[valid_z],
                color="darkorange", s=70, zorder=5,
                label=r"Measured $K_{\rm eff}$ ($\kappa$=0.020)")

if not math.isnan(slope_z):
    z_fine = np.linspace(ZONE_W_VALS[0] * 0.8, ZONE_W_VALS[-1] * 1.2, 100)
    ax["C"].plot(z_fine, np.exp(int_z) * z_fine**slope_z, "k-", lw=2.0,
                 label=rf"Fit: slope = {slope_z:.3f}")
    mid_j    = np.where(valid_z)[0][len(np.where(valid_z)[0])//2]
    ref_anch2 = keff_20[mid_j]
    ref_zone  = zone_arr[mid_j]
    ax["C"].plot(z_fine, ref_anch2 * (z_fine / ref_zone)**(-1.0), "r--", lw=1.4,
                 label="Prediction: slope = -1")

ax["C"].set_xscale("log"); ax["C"].set_yscale("log")
ax["C"].set_xlabel("ZONE_W", fontsize=10)
ax["C"].set_ylabel(r"$K_{\rm eff}$", fontsize=10)
ax["C"].set_title(r"(C) $K_{\rm eff}$ vs ZONE\_W ($\kappa$$=$0.020): log-log",
                  fontsize=11, fontweight="bold")
ax["C"].legend(fontsize=8, loc="upper right")
ax["C"].tick_params(labelsize=8)
ax["C"].annotate(rf"Measured slope: {slope_z:.3f}"
                 "\nPredicted: -1.00"
                 rf"\nRatio: {slope_z/(-1.0):.2f}$\times$",
                 xy=(0.04, 0.12), xycoords="axes fraction", fontsize=8.5,
                 bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow", ec="goldenrod"))

# ─────────────────────────────────────────────────────────────────────────────
# Panel D: K_eff vs nu (Exp B), non-monotone pattern
# ─────────────────────────────────────────────────────────────────────────────
nu_arr  = np.array([x[0] for x in nu_keff])
k_arr   = np.array([x[1] for x in nu_keff])
gamma_b = K_EFF_I / k_arr

ax_d2 = ax["D"].twinx()
ax["D"].plot(nu_arr, k_arr, "o-", color="steelblue", lw=2.0, ms=8,
             label=r"$K_{\rm eff}$ (left)")
ax_d2.plot(nu_arr, gamma_b, "s--", color="coral", lw=1.6, ms=7,
           label=r"$\Gamma$ (right)")

ax["D"].axvline(NU_STD, color="gray", ls=":", lw=1.0, alpha=0.7)
ax["D"].text(NU_STD * 1.08, ax["D"].get_ylim()[1] * 0.95 if not math.isnan(k_arr.max()) else 0.15,
             r"$\nu_{\rm std}$", fontsize=8, color="gray")

ax["D"].set_xscale("log")
ax["D"].set_xlabel(r"$\nu$ (collapse rate)", fontsize=10)
ax["D"].set_ylabel(r"$K_{\rm eff}$", fontsize=10, color="steelblue")
ax_d2.set_ylabel(r"$\Gamma = K_{\rm eff}^{\rm I}/K_{\rm eff}$", fontsize=10, color="coral")
ax["D"].tick_params(axis="y", labelcolor="steelblue", labelsize=8)
ax_d2.tick_params(axis="y", labelcolor="coral", labelsize=8)
ax["D"].tick_params(axis="x", labelsize=8)
ax["D"].set_title(r"(D) $K_{\rm eff}$ vs $\nu$: non-monotone amplification",
                  fontsize=11, fontweight="bold")

# Combined legend
lines1, labels1 = ax["D"].get_legend_handles_labels()
lines2, labels2 = ax_d2.get_legend_handles_labels()
ax["D"].legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc="upper center")

# ── Title and save ─────────────────────────────────────────────────────────────
fig.suptitle(
    r"Paper 30: The Spatial Amplification Factor $\Gamma$ --- "
    r"Partial Confirmation of $\Gamma \propto$ ZONE\_W$/\sqrt{\kappa}$"
    "\n"
    r"Slopes measured: $K_{\rm eff}\propto\kappa^{+0.26}$ (pred +0.50), "
    r"$K_{\rm eff}\propto$ZONE\_W$^{-0.31}$ (pred $-1.00$); "
    r"$\nu$ dependence is non-monotone",
    fontsize=10.5, fontweight="bold", y=0.998
)
out_base = os.path.join(os.path.dirname(__file__), "paper30_figure1")
fig.savefig(out_base + ".pdf", bbox_inches="tight", dpi=150)
fig.savefig(out_base + ".png", bbox_inches="tight", dpi=150)
print(f"Saved: {out_base}.pdf / .png")
plt.close(fig)
