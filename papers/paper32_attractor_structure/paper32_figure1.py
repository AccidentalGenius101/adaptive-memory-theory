"""
paper32_figure1.py -- The Zone Attractor Structure

4 panels:
  A: sg4(t) spaghetti plot T=3000-8000 for all 20 seeds (Exp A)
     Reveals diffuse attractor: high temporal variance, zone-swap events

  B: K_eff vs ZONE_W -- non-monotone with optimal at ZONE_W=5-7 (Exp C)
     Shows vote-count model breaks down at large zone widths
     Overlay: Paper 30 monotone prediction for comparison

  C: sg4(T=6000) vs FA at nu=0.003 (Exp B)
     FA recovery: FA=2.0 restores sg4 to standard level

  D: sg4 distribution across 20 seeds at T=5000, 6000, 7000, 8000 (Exp A)
     Quantifies attractor spread; horizontal axis = sg4, vertical = time
"""
import json, os, math, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

sys.path.insert(0, os.path.dirname(__file__))
from paper32_experiments import (
    N_SEEDS_A, N_SEEDS_B, N_SEEDS_C,
    CPS_A, CPS_B, CPS_C,
    FA_VALS_B, FA_VALS_C, ZONE_W_VALS_C,
    KAPPA_STD, NU_STD, FA_STD,
    key_a, key_b, key_c, fit_keff
)

RESULTS_FILE = os.path.join(os.path.dirname(__file__),
                            "results", "paper32_results.json")
with open(RESULTS_FILE) as f:
    R = json.load(f)

T_LAST_B = CPS_B[-1]   # 6000
T_LAST_C = CPS_C[-1]   # 4000
K_EFF_I  = 0.117

# ── Helpers ───────────────────────────────────────────────────────────────────
def get_sg4_a(seed, t):
    return R.get(key_a(seed), {}).get(f"sg4_{t}", float("nan"))

def get_sg4_b(fa, seed):
    return R.get(key_b(fa, seed), {}).get(f"sg4_{T_LAST_B}", float("nan"))

def get_keff_c(zone_w):
    sg4_means = []
    for fa in FA_VALS_C:
        vs = [R.get(key_c(zone_w, fa, s), {}).get(f"sg4_{T_LAST_C}", float("nan"))
              for s in range(N_SEEDS_C)]
        vs = [v for v in vs if not math.isnan(v)]
        sg4_means.append(float(np.mean(vs)) if vs else float("nan"))
    _, K = fit_keff(FA_VALS_C, sg4_means)
    return K

# ── Figure ─────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(13, 9))
gs  = GridSpec(2, 2, figure=fig, hspace=0.42, wspace=0.38)
ax  = {k: fig.add_subplot(gs[r, c]) for k, (r, c) in
       zip("ABCD", [(0,0),(0,1),(1,0),(1,1)])}

seed_colors = plt.cm.tab20(np.linspace(0, 1, N_SEEDS_A))
t_arr = np.array(CPS_A)

# ─────────────────────────────────────────────────────────────────────────────
# Panel A: sg4(t) spaghetti T=3000-8000 (Exp A)
# ─────────────────────────────────────────────────────────────────────────────
all_sg4_series = []
for s, col in enumerate(seed_colors):
    sg4_t = [get_sg4_a(s, t) for t in CPS_A]
    valid  = [not math.isnan(v) for v in sg4_t]
    t_plot = t_arr[[i for i, v in enumerate(valid) if v]]
    v_plot = [sg4_t[i] for i, v in enumerate(valid) if v]
    if v_plot:
        ax["A"].plot(t_plot, v_plot, color=col, lw=0.9, alpha=0.6)
        all_sg4_series.extend(v_plot)

# Mean and std band
sg4_by_t = []
for t in CPS_A:
    vs = [get_sg4_a(s, t) for s in range(N_SEEDS_A)]
    vs = [v for v in vs if not math.isnan(v)]
    sg4_by_t.append((float(np.mean(vs)) if vs else float("nan"),
                     float(np.std(vs))  if vs else float("nan")))
mu_arr  = np.array([x[0] for x in sg4_by_t])
std_arr = np.array([x[1] for x in sg4_by_t])
valid_t = ~np.isnan(mu_arr)
ax["A"].plot(t_arr[valid_t], mu_arr[valid_t], "k-", lw=2.2, label="Mean (20 seeds)", zorder=5)
ax["A"].fill_between(t_arr[valid_t],
                      mu_arr[valid_t] - std_arr[valid_t],
                      mu_arr[valid_t] + std_arr[valid_t],
                      alpha=0.15, color="black", label=r"$\pm$1 std")

overall_mean = float(np.nanmean(all_sg4_series))
overall_std  = float(np.nanstd(all_sg4_series))
cv = overall_std / overall_mean if overall_mean > 0 else float("nan")
ax["A"].set_xlabel("Time step", fontsize=10)
ax["A"].set_ylabel("sg4", fontsize=10)
ax["A"].set_title("(A) sg4(t) steady-state spaghetti (20 seeds, T=3000-8000)",
                  fontsize=11, fontweight="bold")
ax["A"].legend(fontsize=8)
ax["A"].tick_params(labelsize=8)
ax["A"].annotate(f"Overall CV = {cv:.3f}\n(diffuse attractor)",
                 xy=(0.04, 0.08), xycoords="axes fraction", fontsize=9,
                 bbox=dict(boxstyle="round,pad=0.3", fc="mistyrose", ec="coral"))

# ─────────────────────────────────────────────────────────────────────────────
# Panel B: K_eff vs ZONE_W -- non-monotone (Exp C)
# ─────────────────────────────────────────────────────────────────────────────
zw_arr   = np.array(ZONE_W_VALS_C)
keff_arr = np.array([get_keff_c(zw) or float("nan") for zw in ZONE_W_VALS_C])
gamma_arr = K_EFF_I / keff_arr

valid_z = ~np.isnan(keff_arr) & (keff_arr > 0)
ax["B"].plot(zw_arr[valid_z], keff_arr[valid_z], "o-", color="steelblue",
             lw=2.0, ms=8, label=r"Measured $K_{\rm eff}$")

# Annotate minimum
if valid_z.any():
    min_idx = int(np.nanargmin(keff_arr))
    ax["B"].annotate(f"Optimum\nZONE_W={ZONE_W_VALS_C[min_idx]}\nK={keff_arr[min_idx]:.4f}",
                     xy=(zw_arr[min_idx], keff_arr[min_idx]),
                     xytext=(zw_arr[min_idx] + 1.5, keff_arr[min_idx] + 0.010),
                     fontsize=8.5, color="darkgreen",
                     arrowprops=dict(arrowstyle="->", color="darkgreen", lw=1.2))

# Paper 30 power law reference (slope -0.314, anchored at ZONE_W=5)
zw_fine  = np.linspace(1.5, 16, 100)
anch_idx = list(ZONE_W_VALS_C).index(5) if 5 in ZONE_W_VALS_C else 0
anch_k   = keff_arr[anch_idx] if not math.isnan(keff_arr[anch_idx]) else 0.014
ax["B"].plot(zw_fine, anch_k * (zw_fine / 5)**(-0.314), "r--", lw=1.3,
             label=r"Paper 30 power law (slope $-0.31$)")

ax["B"].set_xlabel("ZONE_W", fontsize=10)
ax["B"].set_ylabel(r"$K_{\rm eff}$", fontsize=10)
ax["B"].set_title(r"(B) $K_{\rm eff}$ vs ZONE\_W: non-monotone, optimal at 5-7",
                  fontsize=11, fontweight="bold")
ax["B"].legend(fontsize=8)
ax["B"].tick_params(labelsize=8)
ax["B"].annotate("Vote-count model\nbreaks at wide zones\n(coherence range exceeded)",
                 xy=(0.55, 0.60), xycoords="axes fraction", fontsize=8.5,
                 bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow", ec="goldenrod"))

# ─────────────────────────────────────────────────────────────────────────────
# Panel C: sg4(T=6000) vs FA at nu=0.003 (Exp B)
# ─────────────────────────────────────────────────────────────────────────────
sg4_b_mean = []
sg4_b_std  = []
fa_valid   = []
for fa in FA_VALS_B:
    vs = [get_sg4_b(fa, s) for s in range(N_SEEDS_B)]
    vs = [v for v in vs if not math.isnan(v) and not np.isinf(v)]
    if vs:
        sg4_b_mean.append(float(np.mean(vs)))
        sg4_b_std.append(float(np.std(vs)) if len(vs) > 1 else 0.0)
        fa_valid.append(fa)

fa_arr_b   = np.array(fa_valid)
sg4_arr_b  = np.array(sg4_b_mean)
sg4_std_b  = np.array(sg4_b_std)

ax["C"].errorbar(fa_arr_b, sg4_arr_b, yerr=sg4_std_b,
                 fmt="o-", color="coral", lw=2.0, ms=8, capsize=3,
                 label=r"$\nu$=0.003 (Exp B)")

# Reference line: standard nu=0.001 at FA=0.200 from Exp A
ax["C"].axhline(overall_mean, color="steelblue", ls="--", lw=1.5,
                label=rf"Standard $\nu$=0.001 (mean={overall_mean:.0f})")

ax["C"].set_xscale("log")
ax["C"].set_xlabel("FA (consolidation rate)", fontsize=10)
ax["C"].set_ylabel("sg4 (T=6000)", fontsize=10)
ax["C"].set_title(r"(C) FA recovery at $\nu$=0.003: FA$\approx$2 restores standard level",
                  fontsize=11, fontweight="bold")
ax["C"].legend(fontsize=8)
ax["C"].tick_params(labelsize=8)
ax["C"].annotate("Q21 confirmed:\nFA=2.0 recovers sg4\n(~10x standard FA)",
                 xy=(0.55, 0.15), xycoords="axes fraction", fontsize=8.5,
                 bbox=dict(boxstyle="round,pad=0.3", fc="lightgreen", ec="green"))

# ─────────────────────────────────────────────────────────────────────────────
# Panel D: sg4 distribution across 20 seeds at 4 late time slices (Exp A)
# ─────────────────────────────────────────────────────────────────────────────
late_times = [5000, 6000, 7000, 8000]
offsets    = [0, 1, 2, 3]
box_data   = []
for t in late_times:
    vs = [get_sg4_a(s, t) for s in range(N_SEEDS_A)]
    box_data.append([v for v in vs if not math.isnan(v)])

bp = ax["D"].boxplot(box_data, positions=offsets, widths=0.6,
                     patch_artist=True,
                     medianprops=dict(color="black", lw=2),
                     boxprops=dict(facecolor="steelblue", alpha=0.6))

ax["D"].set_xticks(offsets)
ax["D"].set_xticklabels([str(t) for t in late_times], fontsize=9)
ax["D"].set_xlabel("Time step", fontsize=10)
ax["D"].set_ylabel("sg4", fontsize=10)
ax["D"].set_title("(D) sg4 spread across 20 seeds at late times",
                  fontsize=11, fontweight="bold")
ax["D"].tick_params(labelsize=8)

# Annotate with across-time CV
for i, (t, bx) in enumerate(zip(late_times, box_data)):
    if bx:
        mu_t  = float(np.mean(bx))
        std_t = float(np.std(bx))
        ax["D"].text(offsets[i], max(bx) + 5,
                     f"CV={std_t/mu_t:.2f}", ha="center", fontsize=7.5, color="coral")
ax["D"].annotate("Persistent spread: attractor\nis diffuse across seeds AND time",
                 xy=(0.02, 0.08), xycoords="axes fraction", fontsize=8.5,
                 bbox=dict(boxstyle="round,pad=0.3", fc="mistyrose", ec="coral"))

# ── Title and save ─────────────────────────────────────────────────────────────
fig.suptitle(
    "Paper 32: The Zone Attractor Structure --- Diffuse Noisy Attractor (CV=0.44), "
    "Non-Monotone ZONE_W Optimum, FA Recovery at "r"$\nu$=0.003"
    "\n"
    "Zones are stochastic, not fixed-point; optimal zone width ~5-7; "
    r"$\nu$=0.003 rescued by FA$\approx$2.0 (10$\times$ standard)",
    fontsize=10.0, fontweight="bold", y=0.998
)
out_base = os.path.join(os.path.dirname(__file__), "paper32_figure1")
fig.savefig(out_base + ".pdf", bbox_inches="tight", dpi=150)
fig.savefig(out_base + ".png", bbox_inches="tight", dpi=150)
print(f"Saved: {out_base}.pdf / .png")
plt.close(fig)
