"""
paper21_figure1.py -- Figure 1 for Paper 21

2x2 figure showing the C-factor (tau_buildup / tau_base) as a function of
each structural parameter, with power-law fits where significant.

Panel A: C vs NU        -- primary driver, C ~ nu^(-0.4)
Panel B: C vs SEED_BETA -- counterintuitive: high inheritance slows buildup
Panel C: C vs KAPPA     -- near-flat: field diffusion is secondary
Panel D: C vs zone_width -- noisy/null: finite-size confound

Output: paper21_figure1.pdf, paper21_figure1.png
"""
import json, os, math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import linregress

DIR     = os.path.dirname(__file__)
results = json.load(open(os.path.join(DIR, "results", "paper21_results.json")))

# ── Parameters (mirror experiment file) ──────────────────────────────────────
H         = 20
N_ZONES   = 4
BASE_BETA = 0.005
T_END     = 3000
TAU_BASE  = 1.0 / BASE_BETA   # 200
N_SEEDS   = 5
CPS       = list(range(200, T_END + 1, 200))

NU_VALS        = [0.0005, 0.001, 0.002, 0.004, 0.008]
SEED_BETA_VALS = [0.10, 0.20, 0.25, 0.40, 0.80]
KAPPA_VALS     = [0.005, 0.010, 0.020, 0.040, 0.080]
HALF_VALS      = [8, 12, 20, 32, 40]

NU_STD   = 0.001
HALF_STD = 20


def make_key(sweep, val, seed):
    return f"p21,{sweep},{val:.8g},{seed}"


def tau_buildup_all(sweep, val):
    """Return list of per-seed tau_buildup values."""
    taus = []
    for seed in range(N_SEEDS):
        key = make_key(sweep, val, seed)
        if key not in results:
            continue
        r = results[key]
        sg4_final = r.get(f"sg4_{T_END}", float("nan"))
        if math.isnan(sg4_final) or sg4_final <= 0:
            continue
        target     = 0.63 * sg4_final
        prev_t, prev_v = 0, 0.0
        tau        = float(T_END)
        for t in CPS:
            v = r.get(f"sg4_{t}", float("nan"))
            if math.isnan(v):
                break
            if v >= target:
                frac = (target - prev_v) / max(v - prev_v, 1e-12)
                tau  = prev_t + frac * (t - prev_t)
                break
            prev_t, prev_v = t, v
        taus.append(tau)
    return taus


def c_stats(sweep, val):
    taus = tau_buildup_all(sweep, val)
    if not taus:
        return float("nan"), float("nan")
    c_vals = [t / TAU_BASE for t in taus]
    return float(np.mean(c_vals)), float(np.std(c_vals) / math.sqrt(len(c_vals)))


def power_law_fit(x_vals, c_means):
    """Fit log(C) = alpha*log(x) + const via linear regression."""
    mask = [not math.isnan(c) and c > 0 for c in c_means]
    lx   = np.log([x_vals[i] for i in range(len(x_vals)) if mask[i]])
    lc   = np.log([c_means[i] for i in range(len(c_means)) if mask[i]])
    if len(lx) < 2:
        return float("nan"), float("nan"), float("nan")
    slope, intercept, r, _, _ = linregress(lx, lc)
    return slope, intercept, r**2


# ── Collect all C-factor values ───────────────────────────────────────────────
nu_c   = [c_stats("nu",    v) for v in NU_VALS]
sb_c   = [c_stats("sb",    v) for v in SEED_BETA_VALS]
kap_c  = [c_stats("kappa", v) for v in KAPPA_VALS]
half_c = [c_stats("half",  float(v)) for v in HALF_VALS]
zone_w = [v // N_ZONES for v in HALF_VALS]

nu_means  = [x[0] for x in nu_c]
sb_means  = [x[0] for x in sb_c]
kap_means = [x[0] for x in kap_c]
half_means= [x[0] for x in half_c]

# Power-law fits
nu_alpha,  nu_b,  nu_r2  = power_law_fit(NU_VALS, nu_means)
kap_alpha, kap_b, kap_r2 = power_law_fit(KAPPA_VALS, kap_means)

# ── Figure ────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(12, 9))
axes = axes.flatten()

REF_COLOR  = "#d62728"
FIT_COLOR  = "#1f77b4"
DATA_COLOR = "#2166ac"

# ── Panel A: C vs NU ─────────────────────────────────────────────────────────
ax = axes[0]
ax.errorbar(NU_VALS, [x[0] for x in nu_c], yerr=[x[1] for x in nu_c],
            fmt="o", color=DATA_COLOR, ms=8, lw=1.5, capsize=4,
            label="Simulation", zorder=5)

# Power-law fit overlay
nu_fine = np.logspace(np.log10(min(NU_VALS)*0.8), np.log10(max(NU_VALS)*1.2), 100)
ax.plot(nu_fine, np.exp(nu_b) * nu_fine**nu_alpha, "--",
        color=FIT_COLOR, lw=2.0,
        label=rf"$C \propto \nu^{{{nu_alpha:.2f}}}$ ($R^2={nu_r2:.3f}$)")

# Reference point
ax.axvline(NU_STD, color="gray", ls=":", lw=1.0, alpha=0.7)
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel(r"$\nu$ (turnover rate)", fontsize=11)
ax.set_ylabel(r"$C = \tau_{\rm buildup}/\tau_{\rm base}$", fontsize=11)
ax.set_title(r"\textbf{A.} $C$ vs turnover rate $\nu$ (primary driver)", fontsize=10)
ax.legend(fontsize=9, framealpha=0.9)
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
ax.text(0.97, 0.97,
        f"Slope: $\\alpha = {nu_alpha:.2f}$\n"
        f"Sub-linear: birth-mediated\npropagation is efficient\n"
        f"(faster than random walk\n$\\alpha=-1$)",
        transform=ax.transAxes, fontsize=8, ha="right", va="top",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.9))

# ── Panel B: C vs SEED_BETA ──────────────────────────────────────────────────
ax = axes[1]
ax.errorbar(SEED_BETA_VALS, [x[0] for x in sb_c], yerr=[x[1] for x in sb_c],
            fmt="s", color="#d62728", ms=8, lw=1.5, capsize=4, zorder=5)

ax.axvline(0.25, color="gray", ls=":", lw=1.0, alpha=0.7, label="Standard SEED_BETA=0.25")
ax.set_xlabel(r"SEED\_BETA (inheritance fraction)", fontsize=11)
ax.set_ylabel(r"$C = \tau_{\rm buildup}/\tau_{\rm base}$", fontsize=11)
ax.set_title(r"\textbf{B.} $C$ vs inheritance strength", fontsize=10)
ax.legend(fontsize=9, framealpha=0.9)
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
ax.text(0.05, 0.97,
        "Counterintuitive:\nhigh SEED\\_BETA $\\uparrow$ C.\n"
        "Early-run noise inherited\nstrongly; consolidation\nmust overcome it.\n"
        "Effect is weak for\n$\\mathrm{SEED\\_BETA} \\leq 0.4$.",
        transform=ax.transAxes, fontsize=8, ha="left", va="top",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcyan", alpha=0.9))

# ── Panel C: C vs KAPPA ──────────────────────────────────────────────────────
ax = axes[2]
ax.errorbar(KAPPA_VALS, [x[0] for x in kap_c], yerr=[x[1] for x in kap_c],
            fmt="^", color="#2ca02c", ms=8, lw=1.5, capsize=4, zorder=5)

kap_fine = np.logspace(np.log10(min(KAPPA_VALS)*0.8), np.log10(max(KAPPA_VALS)*1.2), 100)
ax.plot(kap_fine, np.exp(kap_b) * kap_fine**kap_alpha, "--",
        color="#aec7e8", lw=2.0,
        label=rf"$C \propto \kappa^{{{kap_alpha:.2f}}}$ ($R^2={kap_r2:.3f}$)")

ax.axvline(0.020, color="gray", ls=":", lw=1.0, alpha=0.7, label=r"Standard $\kappa=0.02$")
ax.set_xscale("log")
ax.set_xlabel(r"$\kappa$ (field diffusion rate)", fontsize=11)
ax.set_ylabel(r"$C = \tau_{\rm buildup}/\tau_{\rm base}$", fontsize=11)
ax.set_title(r"\textbf{C.} $C$ vs field diffusion $\kappa$ (near-flat)", fontsize=10)
ax.legend(fontsize=9, framealpha=0.9)
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
ax.text(0.97, 0.97,
        f"$\\kappa$ slope: $\\approx{kap_alpha:.2f}$\n"
        "Field diffusion is\n\\textbf{not} the primary\nspatial propagation\n"
        "mechanism. $D_{\\rm copy} \\gg \\kappa$.",
        transform=ax.transAxes, fontsize=8, ha="right", va="top",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.9))

# ── Panel D: C vs zone width ─────────────────────────────────────────────────
ax = axes[3]
ax.errorbar(zone_w, [x[0] for x in half_c], yerr=[x[1] for x in half_c],
            fmt="D", color="#9467bd", ms=8, lw=1.5, capsize=4, zorder=5,
            label="Simulation")

ax.axvline(HALF_STD // N_ZONES, color="gray", ls=":", lw=1.0, alpha=0.7,
           label="Standard zone\\_w=5")
ax.set_xlabel(r"Zone width (HALF $\div$ N\_ZONES, sites)", fontsize=11)
ax.set_ylabel(r"$C = \tau_{\rm buildup}/\tau_{\rm base}$", fontsize=11)
ax.set_title(r"\textbf{D.} $C$ vs zone width (noisy, non-monotone)", fontsize=10)
ax.legend(fontsize=9, framealpha=0.9)
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
ax.text(0.05, 0.97,
        "No clean scaling.\nFinite-size effects:\nsmall HALF $\\to$ large SEM,\n"
        "sg4 metric changes\ncharacter at small zones.\n"
        "Zone width is not an\nindependent predictor.",
        transform=ax.transAxes, fontsize=8, ha="left", va="top",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="mistyrose", alpha=0.9))

fig.suptitle(
    r"The spatial formation factor $C$: turnover rate $\nu$ is the primary driver; "
    r"field diffusion $\kappa$ is secondary",
    fontsize=11, y=1.01
)
fig.tight_layout()

OUT = os.path.join(DIR, "paper21_figure1")
fig.savefig(OUT + ".pdf", bbox_inches="tight")
fig.savefig(OUT + ".png", dpi=150, bbox_inches="tight")
print(f"Saved {OUT}.pdf and {OUT}.png")

# ── Summary ──────────────────────────────────────────────────────────────────
print(f"\nPower-law fits:")
print(f"  C ~ nu^({nu_alpha:.3f})     R2={nu_r2:.4f}   (nu is primary driver)")
print(f"  C ~ kappa^({kap_alpha:.3f}) R2={kap_r2:.4f}  (kappa near-flat)")
print(f"\nStandard params (nu=0.001, sb=0.25, kappa=0.020, half=20):")
c_std_mean, c_std_sem = c_stats("nu", NU_STD)
print(f"  C = {c_std_mean:.2f} +/- {c_std_sem:.2f}")
print(f"\nD_eff implication:")
print(f"  D_copy (from nu) >> KAPPA (field diffusion)")
print(f"  D_copy ~ nu^0.4 * const, independent of KAPPA")
print(f"  PDE: dF/dt = FA*(m-F) + D_eff*Laplacian(F), D_eff dominated by D_copy")
