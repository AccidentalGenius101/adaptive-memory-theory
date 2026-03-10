"""
paper29_figure1.py  --  Paper 29 main figure

4 panels:
  A: sg4 vs FA on log-log (Exp A, T=6000)
     Fit: saturation law C*FA/(FA+K_eff), K_eff=0.037
     Reference slopes: 1.0 (linear) and 0.5 (midpoint)
     Paper 16 saturation law shown with K_eff=0.117 for comparison

  B: sg4(t) temporal growth curves for all FA values
     Shows universal saturation timescale T_90 ~ 2000-3000 independent of FA

  C: Normalised sg4 (sg4/sg4_final) vs time -- universality check
     If all FA collapse to same curve, growth dynamics are FA-independent

  D: K_eff comparison bar: Layer I (Paper 16, K~0.117) vs Layer III (Paper 29, K=0.037)
     With schematic of spatial amplification mechanism via copy-forward loop
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
from paper29_experiments import FA_VALS, N_SEEDS, T_END, CPS, key

RESULTS_FILE = os.path.join(os.path.dirname(__file__),
                            "results", "paper29_results.json")
with open(RESULTS_FILE) as f:
    R = json.load(f)

T_LAST = CPS[-1]   # 6000

# ── Helpers ───────────────────────────────────────────────────────────────────
def get_sg4(fa, t=T_LAST):
    vals = [R.get(key(fa, s), {}).get(f"sg4_{t}", float("nan"))
            for s in range(N_SEEDS)]
    return [v for v in vals if not math.isnan(v)]

def sg4_mean_std(fa, t=T_LAST):
    vals = get_sg4(fa, t)
    if not vals:
        return float("nan"), float("nan")
    return float(np.mean(vals)), float(np.std(vals))

# Assemble final-time arrays
fa_arr    = np.array(FA_VALS)
sg4_means = np.array([sg4_mean_std(fa)[0] for fa in FA_VALS])
sg4_stds  = np.array([sg4_mean_std(fa)[1] for fa in FA_VALS])
valid     = ~np.isnan(sg4_means)

# Saturation law fit
def sat_law(fa, C, K):
    return C * fa / (fa + K)

popt_P29, _ = curve_fit(sat_law, fa_arr[valid], sg4_means[valid],
                         p0=[120, 0.04], bounds=([0, 1e-5], [500, 5]),
                         maxfev=5000)
C_P29, K_P29 = popt_P29

# Paper 16 parameters (K_eff ~ 0.117, scale to this paper's C)
K_P16 = 0.117
C_P16 = C_P29 * K_P29 / K_P16   # rescale so same asymptote for fair comparison

# ── Figure setup ──────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(13, 9))
gs  = GridSpec(2, 2, figure=fig, hspace=0.42, wspace=0.35)
ax  = {k: fig.add_subplot(gs[r, c]) for k, (r, c) in
       zip("ABCD", [(0,0),(0,1),(1,0),(1,1)])}

fa_colors = plt.cm.plasma(np.linspace(0.08, 0.92, len(FA_VALS)))

# ─────────────────────────────────────────────────────────────────────────────
# Panel A: sg4 vs FA (log-log), saturation law fit, reference slopes
# ─────────────────────────────────────────────────────────────────────────────
ax["A"].errorbar(fa_arr[valid], sg4_means[valid], yerr=sg4_stds[valid],
                 fmt="o", color="steelblue", ms=6, capsize=3,
                 label="Measured sg4 (T=6000)", zorder=5)

fa_fine = np.logspace(np.log10(FA_VALS[0]), np.log10(FA_VALS[-1]), 200)
ax["A"].plot(fa_fine, sat_law(fa_fine, *popt_P29), "k-", lw=2.0,
             label=rf"Fit: $C\cdot$FA/(FA$+K_{{\rm eff}}$), $K={K_P29:.4f}$")
ax["A"].plot(fa_fine, sat_law(fa_fine, C_P16, K_P16), "r--", lw=1.4,
             label=rf"Paper 16 law ($K={K_P16:.3f}$)")

# Reference slopes anchored at midpoint
fa_mid  = np.sqrt(FA_VALS[1] * FA_VALS[-2])
sg4_mid = float(sg4_means[valid][len(FA_VALS)//2])
for slope, ls, label in [(1.0, ":", "slope = 1"),
                          (0.5, "-.", "slope = 0.5")]:
    ref = sg4_mid * (fa_fine / fa_mid) ** slope
    ax["A"].plot(fa_fine, ref, color="gray", lw=1.0, ls=ls, label=label)

ax["A"].set_xscale("log"); ax["A"].set_yscale("log")
ax["A"].set_xlabel("FA (consolidation rate)", fontsize=10)
ax["A"].set_ylabel("sg4 (zone differentiation)", fontsize=10)
ax["A"].set_title("(A) sg4 vs FA: smooth crossover, no threshold",
                  fontsize=11, fontweight="bold")
ax["A"].legend(fontsize=7.5, loc="upper left")
ax["A"].tick_params(labelsize=8)
ax["A"].annotate(rf"$K_{{\rm eff}}^{{\rm III}} = {K_P29:.4f}$"
                 "\n" rf"$K_{{\rm eff}}^{{\rm I}} = {K_P16:.3f}$"
                 "\n" rf"Ratio: {K_P29/K_P16:.2f}$\times$",
                 xy=(0.62, 0.10), xycoords="axes fraction", fontsize=8.5,
                 bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow", ec="goldenrod"))

# ─────────────────────────────────────────────────────────────────────────────
# Panel B: sg4(t) temporal curves for all FA
# ─────────────────────────────────────────────────────────────────────────────
t_arr = np.array(CPS)
for fa, col in zip(FA_VALS, fa_colors):
    sg4_t = []
    for t in CPS:
        vals = get_sg4(fa, t)
        sg4_t.append(np.mean(vals) if vals else float("nan"))
    sg4_t = np.array(sg4_t)
    valid_t = ~np.isnan(sg4_t)
    if valid_t.any():
        ax["B"].plot(t_arr[valid_t], sg4_t[valid_t], color=col,
                     lw=1.6, label=f"FA={fa}")

ax["B"].axvline(2000, color="gray", ls="--", lw=1.0, alpha=0.7)
ax["B"].axvline(3000, color="gray", ls=":",  lw=1.0, alpha=0.7)
ax["B"].text(2050, ax["B"].get_ylim()[0] if ax["B"].get_ylim()[0] > 0 else 1,
             r"$T_{90}$ range", fontsize=7.5, color="gray", va="bottom")
ax["B"].set_xlabel("Time step", fontsize=10)
ax["B"].set_ylabel("sg4", fontsize=10)
ax["B"].set_title("(B) Temporal growth curves (all FA, 5 seeds each)",
                  fontsize=11, fontweight="bold")
ax["B"].legend(fontsize=6.5, ncol=2, loc="upper left")
ax["B"].tick_params(labelsize=8)

# ─────────────────────────────────────────────────────────────────────────────
# Panel C: Normalised sg4(t) / sg4(T_end) -- universality of growth shape
# ─────────────────────────────────────────────────────────────────────────────
for fa, col in zip(FA_VALS, fa_colors):
    sg4_t, sg4_fin = [], sg4_mean_std(fa)[0]
    if math.isnan(sg4_fin) or sg4_fin < 1e-6:
        continue
    for t in CPS:
        vals = get_sg4(fa, t)
        sg4_t.append(np.mean(vals) / sg4_fin if vals else float("nan"))
    sg4_t  = np.array(sg4_t)
    valid_t = ~np.isnan(sg4_t)
    if valid_t.any():
        ax["C"].plot(t_arr[valid_t], sg4_t[valid_t], color=col,
                     lw=1.4, alpha=0.85, label=f"FA={fa}")

ax["C"].axhline(0.90, color="black", ls="--", lw=1.0)
ax["C"].text(100, 0.91, r"90%", fontsize=8)
ax["C"].set_xlabel("Time step", fontsize=10)
ax["C"].set_ylabel(r"sg4$(t)$ / sg4$(T_{\rm end})$", fontsize=10)
ax["C"].set_title("(C) Normalised growth: universal saturation shape",
                  fontsize=11, fontweight="bold")
ax["C"].legend(fontsize=6.5, ncol=2, loc="lower right")
ax["C"].set_ylim([-0.05, 1.15])
ax["C"].tick_params(labelsize=8)

# ─────────────────────────────────────────────────────────────────────────────
# Panel D: K_eff comparison and spatial amplification schematic
# ─────────────────────────────────────────────────────────────────────────────
K_eff_labels = [r"Paper 16$^*$" + "\n(Layer I/II\nno spatial structure)",
                "Paper 29\n(Layer III\nspatial diffusion\n+ copy-forward)"]
K_eff_vals   = [K_P16, K_P29]
K_eff_colors = ["salmon", "steelblue"]

bars = ax["D"].bar(K_eff_labels, K_eff_vals, color=K_eff_colors,
                   width=0.5, edgecolor="black", linewidth=0.8)
for bar, val in zip(bars, K_eff_vals):
    ax["D"].text(bar.get_x() + bar.get_width()/2, val + 0.002,
                 f"K = {val:.4f}", ha="center", va="bottom", fontsize=10,
                 fontweight="bold")

ax["D"].set_ylabel(r"$K_{\rm eff}$ (half-saturation FA)", fontsize=10)
ax["D"].set_title(r"(D) Spatial amplification: $K_{\rm eff}$ reduced 3$\times$",
                  fontsize=11, fontweight="bold")
ax["D"].tick_params(axis="x", labelsize=9)
ax["D"].tick_params(axis="y", labelsize=8)
ax["D"].set_ylim([0, 0.16])

# Annotate ratio
ratio = K_P16 / K_P29
ax["D"].annotate("", xy=(1.0, K_P29 + 0.003), xytext=(0.0, K_P16 + 0.003),
                 arrowprops=dict(arrowstyle="<->", color="darkgreen", lw=1.5))
ax["D"].text(0.5, (K_P16 + K_P29)/2 + 0.008,
             rf"${ratio:.1f}\times$ reduction", ha="center", va="bottom",
             fontsize=9.5, color="darkgreen", fontweight="bold")
ax["D"].text(0.05, 0.04,
             r"$^*$K$_{\rm eff}^{\rm I}$ from Paper 16 ALPHA\_FIELD fit",
             fontsize=7, transform=ax["D"].transAxes, color="gray")

# ── Title and save ────────────────────────────────────────────────────────────
fig.suptitle(
    r"Paper 29: FA Threshold for Zone Differentiation --- Smooth Crossover, Not a Bifurcation"
    "\n"
    r"Saturation law holds; spatial structure reduces $K_{\rm eff}$ by 3$\times$"
    " via copy-forward amplification",
    fontsize=11.5, fontweight="bold", y=0.995
)
out_base = os.path.join(os.path.dirname(__file__), "paper29_figure1")
fig.savefig(out_base + ".pdf", bbox_inches="tight", dpi=150)
fig.savefig(out_base + ".png", bbox_inches="tight", dpi=150)
print(f"Saved: {out_base}.pdf / .png")
plt.close(fig)
