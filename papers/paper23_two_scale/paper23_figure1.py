"""
paper23_figure1.py -- Figure 1 for Paper 23

Two-Scale Reynolds Decomposition of VCML.

Panel A: xi_inf vs kappa (main result -- direction correct, kappa^+0.28)
Panel B: xi_inf vs FA   (unexpected positive slope: inheritance mechanism)
Panel C: sg4(T_END) vs kappa (shows kappa controls zone differentiation amplitude)
Panel D: xi_inf vs nu   (flat, from Paper 22 -- confirms D_eff_within != f(nu))
"""
import json, os, math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import linregress

DIR = os.path.dirname(__file__)
results   = json.load(open(os.path.join(DIR, "results", "paper23_results.json")))
# Load Paper 22 results for the nu-sweep panel
p22_file  = os.path.join(DIR, "..", "paper22_spatial_correlation",
                         "results", "paper22_results.json")
results22 = json.load(open(p22_file))

# ── Parameters ────────────────────────────────────────────────────────────────
T_END      = 3000
CPS        = list(range(200, T_END + 1, 200))
FA_VALS    = [0.10, 0.20, 0.30, 0.40, 0.60, 0.80]
KAPPA_VALS = [0.005, 0.010, 0.020, 0.040, 0.080]
NU_VALS_22 = [0.0005, 0.001, 0.002, 0.004, 0.008]
N_SEEDS    = 5
ZONE_W     = 5
MAX_LAG    = 10
XI_CAP     = 30.0
P_CONSOL   = 0.175
FA_STD     = 0.40
KAPPA_STD  = 0.020


def fit_xi(corr):
    r_vals, log_c = [], []
    for r in range(1, min(ZONE_W + 1, MAX_LAG + 1)):
        if r >= len(corr):
            break
        c = corr[r]
        if not math.isnan(c) and 0.05 < c <= 1.0:
            r_vals.append(r)
            log_c.append(math.log(c))
    if len(r_vals) < 2:
        return float("nan")
    slope, _, _, _, _ = linregress(r_vals, log_c)
    if slope >= 0:
        return float("nan")
    return -1.0 / slope


def collect_xi(res, key_fn, vals, sweep_tag, late_ts=None):
    """Collect (means, sems) of xi_final across all vals."""
    if late_ts is None:
        late_ts = [t for t in CPS if t >= 2400]
    means, sems = [], []
    for val in vals:
        xis = []
        for seed in range(N_SEEDS):
            key = key_fn(sweep_tag, val, seed)
            if key not in res:
                continue
            for t in late_ts:
                corr = res[key].get(f"corr_{t}")
                if corr is not None:
                    xi = fit_xi(corr)
                    if not math.isnan(xi) and xi <= XI_CAP:
                        xis.append(xi)
        means.append(float(np.mean(xis)) if xis else float("nan"))
        sems.append(float(np.std(xis) / math.sqrt(len(xis))) if len(xis) > 1 else 0.0)
    return means, sems


def collect_sg4(res, key_fn, vals, sweep_tag):
    """Collect mean sg4(T_END) across seeds."""
    means, sems = [], []
    for val in vals:
        sg4s = []
        for seed in range(N_SEEDS):
            key = key_fn(sweep_tag, val, seed)
            if key not in res:
                continue
            sg4 = res[key].get(f"sg4_{T_END}")
            if sg4 is not None:
                sg4s.append(sg4)
        means.append(float(np.mean(sg4s)) if sg4s else float("nan"))
        sems.append(float(np.std(sg4s) / math.sqrt(len(sg4s))) if len(sg4s) > 1 else 0.0)
    return means, sems


def p23_key(sweep, val, seed):
    if sweep == "kappa" and abs(val - KAPPA_STD) < 1e-9:
        return f"p23,fa,{FA_STD:.8g},{seed}"
    return f"p23,{sweep},{val:.8g},{seed}"


def p22_key(_, val, seed):   # sweep arg ignored
    return f"p22,{val:.8g},{seed}"


# ── Collect data ──────────────────────────────────────────────────────────────
kap_xi, kap_xi_sem = collect_xi(results, p23_key, KAPPA_VALS, "kappa")
fa_xi,  fa_xi_sem  = collect_xi(results, p23_key, FA_VALS,    "fa")
kap_sg4, kap_sg4_sem = collect_sg4(results, p23_key, KAPPA_VALS, "kappa")
nu_xi,  nu_xi_sem  = collect_xi(results22, p22_key, NU_VALS_22, "nu")

# Power-law fits
def pwrfit(x_vals, y_means, label=""):
    valid = [(x, y) for x, y in zip(x_vals, y_means)
             if not math.isnan(y) and y > 0]
    if len(valid) < 3:
        return float("nan"), float("nan"), float("nan")
    lx = np.log([v[0] for v in valid])
    ly = np.log([v[1] for v in valid])
    slope, intercept, r, _, _ = linregress(lx, ly)
    return slope, intercept, r ** 2


kap_slope, kap_icpt, kap_r2 = pwrfit(KAPPA_VALS, kap_xi)
fa_slope,  fa_icpt,  fa_r2  = pwrfit(FA_VALS,    fa_xi)
nu_slope,  nu_icpt,  nu_r2  = pwrfit(NU_VALS_22, nu_xi)

# ── Figure ────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(12, 9))
axes = axes.flatten()
DATA_COLOR = "#2166ac"
FIT_COLOR  = "#aec7e8"
PRED_COLOR = "#d62728"

# ── Panel A: xi_inf vs kappa ──────────────────────────────────────────────────
ax = axes[0]
ax.errorbar(KAPPA_VALS, kap_xi, yerr=kap_xi_sem,
            fmt="o", color=DATA_COLOR, ms=8, lw=1.5, capsize=4,
            label="Simulation", zorder=5)

kap_fine = np.logspace(np.log10(min(KAPPA_VALS) * 0.8),
                        np.log10(max(KAPPA_VALS) * 1.2), 100)
ax.plot(kap_fine, np.exp(kap_icpt) * kap_fine ** kap_slope, "--",
        color=FIT_COLOR, lw=2.0,
        label=rf"$\xi_\infty \propto \kappa^{{{kap_slope:.2f}}}$ ($R^2={kap_r2:.3f}$)")

# Predicted slope +0.5 reference line
pred_slope = 0.5
pred_icpt = np.log(np.mean([y for y in kap_xi if not math.isnan(y)])) \
            - pred_slope * np.log(np.exp(np.mean(np.log(KAPPA_VALS))))
ax.plot(kap_fine, np.exp(pred_icpt) * kap_fine ** pred_slope, ":",
        color=PRED_COLOR, lw=2.0, alpha=0.7,
        label=r"Predicted slope $+0.5$")

ax.set_xscale("log"); ax.set_yscale("log")
ax.axvline(KAPPA_STD, color="gray", ls=":", lw=1.0, alpha=0.7)
ax.set_xlabel(r"$\kappa$ (site-level diffusion)", fontsize=11)
ax.set_ylabel(r"$\xi_\infty$ (equilibrium correlation length, sites)", fontsize=11)
ax.set_title(r"\textbf{A.} $\xi_\infty$ vs $\kappa$: direction confirmed"
             "\n" r"($\kappa$ sets within-zone diffusivity; not $D_{\rm copy}$)",
             fontsize=10)
ax.legend(fontsize=8, framealpha=0.9)
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
ax.text(0.05, 0.97,
        f"Slope: $+{kap_slope:.2f}$ (predicted $+0.5$)\n"
        r"$D_{\rm eff,within} \approx \kappa$" "\n"
        r"(not $D_{\rm copy}$, which would give" "\n"
        r"$\xi_\infty \propto \nu^{+0.20}$, not flat)",
        transform=ax.transAxes, fontsize=8, ha="left", va="top",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.9))

# ── Panel B: xi_inf vs FA ─────────────────────────────────────────────────────
ax = axes[1]
ax.errorbar(FA_VALS, fa_xi, yerr=fa_xi_sem,
            fmt="s", color="#d62728", ms=8, lw=1.5, capsize=4,
            label="Simulation", zorder=5)

fa_fine = np.logspace(np.log10(min(FA_VALS) * 0.8),
                       np.log10(max(FA_VALS) * 1.2), 100)
ax.plot(fa_fine, np.exp(fa_icpt) * fa_fine ** fa_slope, "--",
        color="#f4a582", lw=2.0,
        label=rf"$\xi_\infty \propto FA^{{{fa_slope:.2f}}}$ ($R^2={fa_r2:.3f}$)")

# Predicted slope -0.5 reference
pred_icpt_fa = np.log(np.mean([y for y in fa_xi if not math.isnan(y)])) \
               - (-0.5) * np.log(np.exp(np.mean(np.log(FA_VALS))))
ax.plot(fa_fine, np.exp(pred_icpt_fa) * fa_fine ** (-0.5), ":",
        color="navy", lw=2.0, alpha=0.7, label=r"Predicted slope $-0.5$")

ax.set_xscale("log"); ax.set_yscale("log")
ax.axvline(FA_STD, color="gray", ls=":", lw=1.0, alpha=0.7)
ax.set_xlabel(r"FA (consolidation rate)", fontsize=11)
ax.set_ylabel(r"$\xi_\infty$ (sites)", fontsize=11)
ax.set_title(r"\textbf{B.} $\xi_\infty$ vs FA: unexpected positive slope"
             "\n(inheritance mechanism competes with interface formula)",
             fontsize=10)
ax.legend(fontsize=8, framealpha=0.9)
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
ax.text(0.05, 0.97,
        f"Slope: $+{fa_slope:.2f}$ (predicted $-0.5$)\n"
        "Inheritance mechanism: newborns\n"
        r"copy $F_j$ from neighbors; high FA" "\n"
        r"$\Rightarrow$ larger $F$ amplitude" "\n"
        r"$\Rightarrow$ more inherited roughness" "\n"
        r"$\Rightarrow$ larger $\xi$",
        transform=ax.transAxes, fontsize=8, ha="left", va="top",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcyan", alpha=0.9))

# ── Panel C: sg4(T_END) vs kappa ─────────────────────────────────────────────
ax = axes[2]
ax.errorbar(KAPPA_VALS, kap_sg4, yerr=kap_sg4_sem,
            fmt="^", color="#2ca02c", ms=8, lw=1.5, capsize=4, zorder=5)
ax.axvline(KAPPA_STD, color="gray", ls=":", lw=1.0, alpha=0.7)
ax.set_xscale("log")
ax.set_xlabel(r"$\kappa$ (site-level diffusion)", fontsize=11)
ax.set_ylabel("sg4 (zone separation)", fontsize=11)
ax.set_title(r"\textbf{C.} Zone differentiation vs $\kappa$"
             r" (higher $\kappa$ blurs zones: sg4 decreases)", fontsize=10)
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
ax.text(0.97, 0.97,
        r"$\kappa$ acts at TWO scales:" "\n"
        r"(1) Site-level: sets $\xi_\infty$" "\n"
        r"(2) Zone-level: blurs zone boundaries" "\n"
        r"$\Rightarrow$ sg4 $\downarrow$ as $\kappa$ $\uparrow$" "\n"
        r"(Paper 21: $C \propto \kappa^{-0.07}$, weak)",
        transform=ax.transAxes, fontsize=8, ha="right", va="top",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.9))

# ── Panel D: xi_inf vs nu (Paper 22 reference) ───────────────────────────────
ax = axes[3]
ax.errorbar(NU_VALS_22, nu_xi, yerr=nu_xi_sem,
            fmt="D", color="#9467bd", ms=8, lw=1.5, capsize=4,
            label="Paper 22 data", zorder=5)

if not math.isnan(nu_slope):
    nu_fine = np.logspace(np.log10(min(NU_VALS_22) * 0.8),
                           np.log10(max(NU_VALS_22) * 1.2), 100)
    ax.plot(nu_fine, np.exp(nu_icpt) * nu_fine ** nu_slope, "--",
            color="#c5b0d5", lw=2.0,
            label=rf"$\xi_\infty \propto \nu^{{{nu_slope:.2f}}}$ ($R^2={nu_r2:.3f}$)")

ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel(r"$\nu$ (turnover rate)", fontsize=11)
ax.set_ylabel(r"$\xi_\infty$ (sites)", fontsize=11)
ax.set_title(r"\textbf{D.} $\xi_\infty$ vs $\nu$ (Paper~22): flat"
             "\n" r"($D_{\rm copy}(\nu)$ does not set within-zone diffusivity)",
             fontsize=10)
ax.legend(fontsize=9, framealpha=0.9)
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
ax.text(0.05, 0.97,
        rf"Slope $\approx {nu_slope:.2f}$ $\approx 0$" "\n"
        r"$D_{\rm eff,within} \neq D_{\rm copy}(\nu)$" "\n"
        r"Consistent with $D_{\rm eff,within} = \kappa$ (Panel A)",
        transform=ax.transAxes, fontsize=8, ha="left", va="top",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="mistyrose", alpha=0.9))

fig.suptitle(
    r"Two-scale decomposition: $\xi_\infty$ depends on $\kappa$ (confirmed) "
    r"but FA reveals competing inheritance roughness mechanism",
    fontsize=11, y=1.01
)
fig.tight_layout()

OUT = os.path.join(DIR, "paper23_figure1")
fig.savefig(OUT + ".pdf", bbox_inches="tight")
fig.savefig(OUT + ".png", dpi=150, bbox_inches="tight")
print(f"Saved {OUT}.pdf and {OUT}.png")

print(f"\nKey results:")
print(f"  xi_inf ~ kappa^{kap_slope:.3f}  R2={kap_r2:.4f}  (predicted: +0.5)")
print(f"  xi_inf ~ FA^{fa_slope:.3f}       R2={fa_r2:.4f}  (predicted: -0.5, REVERSED)")
print(f"  xi_inf ~ nu^{nu_slope:.3f}        R2={nu_r2:.4f}  (predicted: ~0, from Paper 22)")
print(f"\n  Geometric prefactor C_geo (kappa sweep): {np.mean([y/math.sqrt(k/(P_CONSOL*FA_STD)) for k,y in zip(KAPPA_VALS, kap_xi) if not math.isnan(y)]):.2f}")
