"""
paper31_figure1.py -- Zone Field Structure in VCML

4 panels:
  A: Within-zone autocorr C(lag) for each kappa (Exp A)
     Shows C[1] transitions from negative (low kappa, uniform zones)
     to positive (high kappa, diffusion-driven gradient zones)

  B: sg4(T=4000) vs kappa from Exp A (FA=0.200)
     Confirms high kappa hurts zone differentiation

  C: sg4(t) for nu=0.001 vs nu=0.003 at FA=0.200 (Exp C)
     Low-amplitude saturation: nu=0.003 forms structure but at ~40% amplitude

  D: Within-zone autocorr C(lag) for each nu (Exp B)
     No nu-dependence of spatial coherence -- uniform at all nu
"""
import json, os, math, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

sys.path.insert(0, os.path.dirname(__file__))
from paper31_experiments import (
    KAPPA_VALS_A, NU_VALS_B, NU_VALS_C, FA_VALS_C,
    N_SEEDS, CPS_AB, CPS_C, ZONE_W, N_ZONES,
    KAPPA_STD, NU_STD, FA_HIGH,
    key_a, key_b, key_c, fit_corr_length
)

RESULTS_FILE = os.path.join(os.path.dirname(__file__),
                            "results", "paper31_results.json")
with open(RESULTS_FILE) as f:
    R = json.load(f)

T_LAST_AB = CPS_AB[-1]   # 4000
T_LAST_C  = CPS_C[-1]    # 6000

# ── Helpers ───────────────────────────────────────────────────────────────────
def get_autocorr_a(kappa, t=T_LAST_AB):
    vals = [R.get(key_a(kappa, s), {}).get(f"wz_autocorr_{t}")
            for s in range(N_SEEDS)]
    return [v for v in vals if v is not None]

def get_autocorr_b(nu, t=T_LAST_AB):
    vals = [R.get(key_b(nu, s), {}).get(f"wz_autocorr_{t}")
            for s in range(N_SEEDS)]
    return [v for v in vals if v is not None]

def mean_corr(corr_lists):
    if not corr_lists:
        return None
    return np.array(corr_lists).mean(axis=0)

def get_sg4_a(kappa, t=T_LAST_AB):
    vals = [R.get(key_a(kappa, s), {}).get(f"sg4_{t}", float("nan"))
            for s in range(N_SEEDS)]
    return [v for v in vals if not math.isnan(v)]

def get_sg4_c(nu, fa, t):
    vals = [R.get(key_c(nu, fa, s), {}).get(f"sg4_{t}", float("nan"))
            for s in range(N_SEEDS)]
    return [v for v in vals if not math.isnan(v)]

# ── Figure ─────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(13, 9))
gs  = GridSpec(2, 2, figure=fig, hspace=0.42, wspace=0.38)
ax  = {k: fig.add_subplot(gs[r, c]) for k, (r, c) in
       zip("ABCD", [(0,0),(0,1),(1,0),(1,1)])}

kappa_colors = plt.cm.plasma(np.linspace(0.05, 0.90, len(KAPPA_VALS_A)))
nu_colors    = plt.cm.viridis(np.linspace(0.05, 0.90, len(NU_VALS_B)))
lags         = np.arange(ZONE_W)

# ─────────────────────────────────────────────────────────────────────────────
# Panel A: Within-zone autocorr C(lag) for each kappa
# ─────────────────────────────────────────────────────────────────────────────
c1_vals = []
for kappa, col in zip(KAPPA_VALS_A, kappa_colors):
    mc = mean_corr(get_autocorr_a(kappa))
    if mc is None:
        continue
    ax["A"].plot(lags, mc, "o-", color=col, lw=1.8, ms=6,
                 label=rf"$\kappa$={kappa}")
    c1_vals.append((kappa, float(mc[1]) if len(mc) > 1 else float("nan")))

ax["A"].axhline(0, color="black", lw=0.7, ls="--", alpha=0.5)
ax["A"].axhline(-1/(ZONE_W - 1), color="gray", lw=0.8, ls=":",
                label=rf"Finite-N floor ($-1/{ZONE_W-1}$)")
ax["A"].set_xlabel("Spatial lag (columns)", fontsize=10)
ax["A"].set_ylabel(r"$C(\Delta x)$ within zone", fontsize=10)
ax["A"].set_title(r"(A) Within-zone autocorr vs $\kappa$ (lag-1 transitions)",
                  fontsize=11, fontweight="bold")
ax["A"].legend(fontsize=7.5, loc="lower right")
ax["A"].tick_params(labelsize=8)
ax["A"].set_xticks(lags)
ax["A"].annotate("Uniform zones\n(copy-forward)", xy=(1, -0.20), fontsize=8,
                 color="steelblue", ha="center")
ax["A"].annotate("Gradient zones\n(diffusion)", xy=(1, 0.25), fontsize=8,
                 color="coral", ha="center")

# ─────────────────────────────────────────────────────────────────────────────
# Panel B: sg4(T=4000) vs kappa from Exp A
# ─────────────────────────────────────────────────────────────────────────────
kappa_arr = np.array(KAPPA_VALS_A)
sg4_arr   = np.array([float(np.mean(get_sg4_a(k))) if get_sg4_a(k) else float("nan")
                      for k in KAPPA_VALS_A])
sg4_std   = np.array([float(np.std(get_sg4_a(k)))  if len(get_sg4_a(k)) > 1 else 0.0
                      for k in KAPPA_VALS_A])

valid = ~np.isnan(sg4_arr) & (sg4_arr > 0)
ax["B"].errorbar(kappa_arr[valid], sg4_arr[valid], yerr=sg4_std[valid],
                 fmt="o-", color="steelblue", ms=7, capsize=3, lw=2.0,
                 label=r"sg4 (FA=0.200, T=4000)")

# Annotate C[1] transition
for kappa, c1 in c1_vals:
    style = "coral" if c1 > 0 else "steelblue"
    c1_str = f"C[1]={c1:+.3f}"
    # find index in kappa_arr
    idx = list(kappa_arr).index(kappa) if kappa in kappa_arr else None
    if idx is not None and not math.isnan(sg4_arr[idx]):
        ax["B"].annotate(c1_str,
                         xy=(kappa_arr[idx], sg4_arr[idx]),
                         xytext=(0, 8), textcoords="offset points",
                         fontsize=7, ha="center", color=style)

ax["B"].set_xscale("log")
ax["B"].set_xlabel(r"$\kappa$ (diffusion)", fontsize=10)
ax["B"].set_ylabel(r"sg4 (zone differentiation)", fontsize=10)
ax["B"].set_title(r"(B) sg4 vs $\kappa$ (FA=0.200): annotated with C[1]",
                  fontsize=11, fontweight="bold")
ax["B"].legend(fontsize=8, loc="upper right")
ax["B"].tick_params(labelsize=8)
ax["B"].annotate("Gradient regime\n" + r"(C[1]>0, high $\kappa$)" + "\nhurts sg4",
                 xy=(0.65, 0.50), xycoords="axes fraction", fontsize=8,
                 bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow", ec="goldenrod"))

# ─────────────────────────────────────────────────────────────────────────────
# Panel C: sg4(t) for nu=0.001 vs nu=0.003 at FA=0.200 (Exp C)
# ─────────────────────────────────────────────────────────────────────────────
t_arr = np.array(CPS_C)
nu_plot_colors = {"0.001": "steelblue", "0.003": "coral"}
for nu in NU_VALS_C:
    sg4_t = []
    for t in CPS_C:
        v = get_sg4_c(nu, FA_HIGH, t)
        sg4_t.append(float(np.mean(v)) if v else float("nan"))
    sg4_t  = np.array(sg4_t)
    sg4_std_t = []
    for t in CPS_C:
        v = get_sg4_c(nu, FA_HIGH, t)
        sg4_std_t.append(float(np.std(v)) if len(v) > 1 else 0.0)
    sg4_std_t = np.array(sg4_std_t)
    valid_t = ~np.isnan(sg4_t)
    col = nu_plot_colors.get(f"{nu}", "gray")
    ax["C"].plot(t_arr[valid_t], sg4_t[valid_t], "-", color=col, lw=2.2,
                 label=rf"$\nu$={nu}")
    ax["C"].fill_between(t_arr[valid_t],
                          sg4_t[valid_t] - sg4_std_t[valid_t],
                          sg4_t[valid_t] + sg4_std_t[valid_t],
                          alpha=0.15, color=col)

ax["C"].set_xlabel("Time step", fontsize=10)
ax["C"].set_ylabel("sg4 (zone differentiation)", fontsize=10)
ax["C"].set_title(r"(C) sg4$(t)$ at $\nu$=0.001 vs 0.003 (FA=0.200, Exp C)",
                  fontsize=11, fontweight="bold")
ax["C"].legend(fontsize=9)
ax["C"].tick_params(labelsize=8)
ax["C"].annotate(r"$\nu$=0.003: low-amplitude saturation" + "\n~40% of standard level\n(forms but does not grow)",
                 xy=(0.38, 0.35), xycoords="axes fraction", fontsize=8,
                 bbox=dict(boxstyle="round,pad=0.3", fc="mistyrose", ec="coral"))

# ─────────────────────────────────────────────────────────────────────────────
# Panel D: Within-zone autocorr C(lag) for each nu (Exp B)
# ─────────────────────────────────────────────────────────────────────────────
for nu, col in zip(NU_VALS_B, nu_colors):
    mc = mean_corr(get_autocorr_b(nu))
    if mc is None:
        continue
    ax["D"].plot(lags, mc, "s-", color=col, lw=1.8, ms=6,
                 label=rf"$\nu$={nu}")

ax["D"].axhline(0, color="black", lw=0.7, ls="--", alpha=0.5)
ax["D"].axhline(-1/(ZONE_W - 1), color="gray", lw=0.8, ls=":",
                label=rf"Finite-N floor ($-1/{ZONE_W-1}$)")
ax["D"].set_xlabel("Spatial lag (columns)", fontsize=10)
ax["D"].set_ylabel(r"$C(\Delta x)$ within zone", fontsize=10)
ax["D"].set_title(r"(D) Within-zone autocorr vs $\nu$: no $\nu$-dependence",
                  fontsize=11, fontweight="bold")
ax["D"].legend(fontsize=7.5, loc="lower right")
ax["D"].tick_params(labelsize=8)
ax["D"].set_xticks(lags)
ax["D"].annotate("All nu: C[1] near zero\n(finite-N floor)"
                 "\nNo detectable spatial gradient",
                 xy=(0.35, 0.15), xycoords="axes fraction", fontsize=8.5,
                 bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow", ec="goldenrod"))

# ── Title and save ─────────────────────────────────────────────────────────────
fig.suptitle(
    "Paper 31: Zone Field Structure in VCML --- Flat Attractors at Low "
    r"$\kappa$, Gradient Zones at High $\kappa$"
    "\n"
    r"C[1] transitions from negative (uniform, copy-forward) to positive (gradient, diffusion); "
    r"$\nu$=0.003 gives low-amplitude saturation",
    fontsize=10.5, fontweight="bold", y=0.998
)
out_base = os.path.join(os.path.dirname(__file__), "paper31_figure1")
fig.savefig(out_base + ".pdf", bbox_inches="tight", dpi=150)
fig.savefig(out_base + ".png", bbox_inches="tight", dpi=150)
print(f"Saved: {out_base}.pdf / .png")
plt.close(fig)
