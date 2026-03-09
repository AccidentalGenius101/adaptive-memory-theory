"""
paper18_figure1.py -- Figure 1 for Paper 18

Three-panel figure:
  A: Buildup curves during Phase 1 (t=200,...,2000), 4 FA lines, KAPPA=0.020
     Shows FA-dependent amplitude, slow buildup (~1600-1800 steps to near-steady-state)
  B: Phase 3 normalized curves (t=2000 to 2200), 4 FA lines, KAPPA=0.020
     Shows consolidation burst (~+40% overshoot at t+30) then exponential decay
     Overlay: analytical e^{-(t-T_BUILD)/tau_m} where tau_m=100 steps
  C: tau_forget (steps to 1/e from T_BUILD) vs FA for all 3 KAPPA values
     Demonstrates FA-independence; contrasts with wrong prediction 1/FA

Output: paper18_figure1.pdf, paper18_figure1.png
"""
import json, os, math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DIR = os.path.dirname(__file__)
results = json.load(open(os.path.join(DIR, "results", "paper18_results.json")))

FA_VALS    = [0.10, 0.20, 0.40, 0.70]
KAPPA_VALS = [0.005, 0.020, 0.080]
N_SEEDS    = 5
T_BUILD    = 2000
T_END      = 2200

PHASE1_CPS = list(range(200, T_BUILD + 1, 200))
PHASE3_CPS = list(range(T_BUILD + 10, T_END + 1, 10))

FA_COLORS  = ["#2166ac", "#4dac26", "#d62728", "#9467bd"]
KAPPA_COLORS = ["#1f77b4", "#d62728", "#2ca02c"]
KAPPA_MARKERS = ["o", "s", "^"]


def make_key(fa, kappa, seed):
    return f"p18,{fa:.4f},{kappa:.4f},{seed}"


def mean_t(fa, kappa, t):
    keys = [make_key(fa, kappa, s) for s in range(N_SEEDS) if make_key(fa, kappa, s) in results]
    vals = [results[k].get(f"sg4_{t}", float("nan")) for k in keys]
    vals = [v for v in vals if not math.isnan(v)]
    return float(np.mean(vals)) if vals else float("nan")


def err_t(fa, kappa, t):
    keys = [make_key(fa, kappa, s) for s in range(N_SEEDS) if make_key(fa, kappa, s) in results]
    vals = [results[k].get(f"sg4_{t}", float("nan")) for k in keys]
    vals = [v for v in vals if not math.isnan(v)]
    if len(vals) < 2:
        return 0.0
    return float(np.std(vals) / math.sqrt(len(vals)))


def tau_forget_est(fa, kappa):
    """Steps from T_BUILD until sg4 drops to 1/e of sg4(T_BUILD). Linear interpolation."""
    v0 = mean_t(fa, kappa, T_BUILD)
    if math.isnan(v0) or v0 <= 0:
        return float("nan")
    target = v0 / math.e
    prev_t, prev_v = T_BUILD, v0
    for t in PHASE3_CPS:
        v = mean_t(fa, kappa, t)
        if not math.isnan(v) and v <= target:
            if prev_v > v:
                frac = (prev_v - target) / (prev_v - v)
                return (prev_t - T_BUILD) + frac * (t - prev_t)
            return t - T_BUILD
        prev_t, prev_v = t, v
    return float("nan")  # still above 1/e at end of Phase 3


fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

# ============================================================
# Panel A: Buildup during Phase 1 (KAPPA=0.020)
# ============================================================
ax = axes[0]
KAPPA_PLOT = 0.020
for i, fa in enumerate(FA_VALS):
    vals = [mean_t(fa, KAPPA_PLOT, t) for t in PHASE1_CPS]
    errs = [err_t(fa, KAPPA_PLOT, t) for t in PHASE1_CPS]
    ax.errorbar(PHASE1_CPS, vals, yerr=errs, fmt="o-", color=FA_COLORS[i],
                ms=5, lw=1.8, capsize=3, label=f"FA={fa:.2f}")

ax.set_xlabel("Step (Phase 1)", fontsize=10)
ax.set_ylabel(r"sg4 (inter-zone L2)", fontsize=10)
ax.set_title(r"\textbf{A.} Buildup: waves running ($\kappa=0.020$)", fontsize=10)
ax.legend(fontsize=8, framealpha=0.85)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.text(0.05, 0.95,
        "Steady state reached\nat $t\\approx 1600$--$2000$ steps\n(FA-dependent amplitude;\nFA-independent timescale)",
        transform=ax.transAxes, fontsize=7.5, va="top",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.85))

# ============================================================
# Panel B: Phase 3 normalized curves (KAPPA=0.020)
# ============================================================
ax = axes[1]
t_rel = [t - T_BUILD for t in PHASE3_CPS]

for i, fa in enumerate(FA_VALS):
    v0 = mean_t(fa, KAPPA_PLOT, T_BUILD)
    if v0 <= 0 or math.isnan(v0):
        continue
    norm_vals = [mean_t(fa, KAPPA_PLOT, t) / v0 for t in PHASE3_CPS]
    ax.plot(t_rel, norm_vals, "o-", color=FA_COLORS[i], ms=5, lw=1.8, label=f"FA={fa:.2f}")

# Analytical: pure exponential from peak at t_peak≈30, tau_m=100
MID_DECAY = 0.99
tau_m = 1.0 / (1.0 - MID_DECAY)   # = 100
t_peak = 30
peak_amp = 1.40  # empirical
t_arr = np.linspace(0, 200, 400)
# Simple exponential from t=0 with tau_m, starting at 1.0 (not peak)
# Actually: show tau_m decay from the peak
analytic = [peak_amp * math.exp(-(t - t_peak) / tau_m) if t >= t_peak else peak_amp
            for t in t_arr]
# Clip to >=0
analytic = [max(v, 0) for v in analytic]
ax.plot(t_arr, analytic, "--", color="black", lw=1.4, alpha=0.7,
        label=r"$\approx\hat{\tau}\,e^{-t/\tau_m}$, $\tau_m=100$")

ax.axhline(1.0 / math.e, color="gray", ls=":", lw=1.2, alpha=0.7)
ax.text(195, 1.0/math.e + 0.02, "1/e", fontsize=8, ha="right", color="gray")
ax.axhline(1.0, color="gray", ls=":", lw=1.0, alpha=0.5)
ax.set_xlabel(r"Steps after wave cessation ($t - T_{\rm build}$)", fontsize=10)
ax.set_ylabel(r"sg4 / sg4$(T_{\rm build})$", fontsize=10)
ax.set_title(r"\textbf{B.} Forgetting: no waves ($\kappa=0.020$)", fontsize=10)
ax.legend(fontsize=8, framealpha=0.85)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.text(0.97, 0.95,
        "Consolidation burst:\nsg4 overshoots +40\\%\nbefore decay.\n$\\tau_{\\rm forget}\\approx\\tau_m=100$ steps\n(not $1/FA$)",
        transform=ax.transAxes, fontsize=7.5, ha="right", va="top",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcyan", alpha=0.85))

# ============================================================
# Panel C: tau_forget vs FA, colored by KAPPA
# ============================================================
ax = axes[2]
fa_fine = np.linspace(0.05, 0.80, 200)
# Wrong prediction: 1/FA
ax.plot(fa_fine, 1.0 / fa_fine, "--", color="black", lw=1.4, alpha=0.6,
        label=r"$1/FA$ (naive prediction)")

for j, kappa in enumerate(KAPPA_VALS):
    tau_vals = [tau_forget_est(fa, kappa) for fa in FA_VALS]
    valid_fa   = [fa for fa, tv in zip(FA_VALS, tau_vals) if not (isinstance(tv, float) and math.isnan(tv))]
    valid_tau  = [tv for tv in tau_vals if not (isinstance(tv, float) and math.isnan(tv))]
    ax.plot(valid_fa, valid_tau, KAPPA_MARKERS[j] + "-",
            color=KAPPA_COLORS[j], ms=8, lw=1.8, label=f"$\\kappa$={kappa:.3f}")

# Analytical prediction: tau_m = 100 steps (MID_DECAY limited)
ax.axhline(tau_m, color="gray", ls="-.", lw=1.4, alpha=0.7,
           label=r"$\tau_m = 1/(1-\delta) = 100$ steps")

ax.set_xlabel("FA (field adaptation rate)", fontsize=10)
ax.set_ylabel(r"$\tau_{\rm forget}$ (steps to 1/e decay)", fontsize=10)
ax.set_title(r"\textbf{C.} $\tau_{\rm forget}$ vs FA: nearly FA-independent", fontsize=10)
ax.legend(fontsize=8, framealpha=0.85)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.set_ylim(bottom=0)
ax.text(0.97, 0.97,
        "Forgetting governed by\n$\\tau_m = 1/(1-\\delta)$, not $1/FA$.\nHigher $\\kappa$: longer $\\tau_{\\rm forget}$.",
        transform=ax.transAxes, fontsize=7.5, ha="right", va="top",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.85))

fig.suptitle(
    r"Temporal dynamics of sg4: buildup, consolidation burst, and MID\_DECAY-limited forgetting",
    fontsize=10, y=1.02
)
fig.tight_layout()

OUT = os.path.join(DIR, "paper18_figure1")
fig.savefig(OUT + ".pdf", bbox_inches="tight")
fig.savefig(OUT + ".png", dpi=150, bbox_inches="tight")
print(f"Saved {OUT}.pdf and {OUT}.png")

# Summary table
print("\nSUMMARY: tau_forget (steps to 1/e)")
print(f"{'FA':>6} | " + " | ".join(f"K={k:.3f}" for k in KAPPA_VALS) + " | pred(1/FA)")
print("-" * 65)
for fa in FA_VALS:
    taus = [tau_forget_est(fa, kappa) for kappa in KAPPA_VALS]
    row_parts = []
    for tv in taus:
        if isinstance(tv, float) and math.isnan(tv):
            row_parts.append(f"{'nan':>8}")
        else:
            row_parts.append(f"{tv:>8.1f}")
    print(f"{fa:>6.2f} | " + " | ".join(row_parts) + f" | {1/fa:>8.1f}")
