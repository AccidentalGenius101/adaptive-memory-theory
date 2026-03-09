"""
paper20_figure1.py -- Figure 1 for Paper 20

Three-panel figure:
  A: Steady-state law -- analytical FA/(FA+K_eff) vs Paper 17 simulation (normalised)
  B: Phase-3 forgetting -- Paper 18 simulation with tau_m analytical decay overlay
  C: Buildup gap -- ODE vs Paper 18 simulation (normalised), showing ~9x discrepancy

Output: paper20_figure1.pdf, paper20_figure1.png
"""
import json, os, math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DIR = os.path.dirname(__file__)
P17 = json.load(open(os.path.join(DIR, "..", "paper17_two_factor_decomposition",
                                  "results", "paper17_results.json")))
P18 = json.load(open(os.path.join(DIR, "..", "paper18_temporal_dynamics",
                                  "results", "paper18_results.json")))

# ── Constants ─────────────────────────────────────────────────────────────────
BETA       = 0.005
MID_DECAY  = 0.99
KAPPA      = 0.020
WR         = 4.8
WAVE_DUR   = 15
SS         = 10
FA_VALS    = [0.10, 0.20, 0.40, 0.70]
N_SEEDS    = 5
T_BUILD    = 2000
T_END      = 2200

P_CALM   = 1.0 - WR / (2.0 * WAVE_DUR)
P_CONSOL = P_CALM ** SS
K_EFF    = KAPPA / P_CONSOL
TAU_BASE = 1.0 / BETA
TAU_M    = 1.0 / (1.0 - MID_DECAY)

PHASE1_CPS = list(range(200, T_BUILD + 1, 200))
PHASE3_CPS = list(range(T_BUILD + 10, T_END + 1, 10))


# ── Helpers ───────────────────────────────────────────────────────────────────
def p18_mean(fa, kappa, t):
    keys = [f"p18,{fa:.4f},{kappa:.4f},{s}" for s in range(N_SEEDS)
            if f"p18,{fa:.4f},{kappa:.4f},{s}" in P18]
    vals = [P18[k].get(f"sg4_{t}", float("nan")) for k in keys]
    vals = [v for v in vals if not math.isnan(v)]
    return float(np.mean(vals)) if vals else float("nan")


def p18_err(fa, kappa, t):
    keys = [f"p18,{fa:.4f},{kappa:.4f},{s}" for s in range(N_SEEDS)
            if f"p18,{fa:.4f},{kappa:.4f},{s}" in P18]
    vals = [P18[k].get(f"sg4_{t}", float("nan")) for k in keys]
    vals = [v for v in vals if not math.isnan(v)]
    return float(np.std(vals) / math.sqrt(len(vals))) if len(vals) > 1 else 0.0


def p17_mean(fa, ss_val=10):
    keys = [f"ss,{ss_val},{fa:.4f},{s}" for s in range(N_SEEDS)
            if f"ss,{ss_val},{fa:.4f},{s}" in P17]
    vals = [P17[k]["sg4_2000"] for k in keys]
    return float(np.mean(vals)) if vals else float("nan")


def solve_ode_buildup(fa, steps=T_BUILD):
    """Simple Euler ODE for Phase 1 buildup. Returns F at each PHASE1_CPS."""
    a = 0.0; m = 0.0; F = 0.0
    result = {}
    for t in range(1, steps + 1):
        s   = 1.0
        p_c = P_CONSOL
        da  = (s - a) / TAU_BASE
        dm  = (a - m) / TAU_M
        dF  = p_c * fa * m - (p_c * fa + KAPPA) * F
        a += da;  m += dm;  F += dF
        if t in set(PHASE1_CPS):
            result[t] = F
    return result


FA_COLORS = ["#2166ac", "#4dac26", "#d62728", "#9467bd"]

fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

# ═══════════════════════════════════════════════════════════════════════════════
# Panel A: Steady-state law -- analytical vs simulation (normalised)
# ═══════════════════════════════════════════════════════════════════════════════
ax = axes[0]

fa_fine = np.linspace(0.05, 0.95, 300)
analytic_norm = np.array([fa / (fa + K_EFF) for fa in fa_fine])
analytic_norm /= analytic_norm.max()

sim_vals = np.array([p17_mean(fa) for fa in FA_VALS])
sim_norm = sim_vals / np.nanmax(sim_vals)

ax.plot(fa_fine, analytic_norm, "-", color="#1f77b4", lw=2.0,
        label=r"$FA/(FA+K_{\rm eff})$, $K_{\rm eff}=%g$" % round(K_EFF, 3))
ax.scatter(FA_VALS, sim_norm, color="#d62728", s=70, zorder=5,
           label="Paper 17 simulation")

ax.set_xlabel("FA (field adaptation rate)", fontsize=10)
ax.set_ylabel("Normalised sg4 (max=1)", fontsize=10)
ax.set_title(r"\textbf{A.} Steady-state law: ODE recovers shape", fontsize=9.5)
ax.legend(fontsize=8, framealpha=0.9)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.text(0.05, 0.97,
        r"$F_\infty = m_\infty \cdot \frac{FA}{FA + K_{\rm eff}}$" + "\n"
        r"$K_{\rm eff} = \kappa/P_{\rm consol}$" + f"\n$= {KAPPA}/{P_CONSOL:.4f} = {K_EFF:.4f}$",
        transform=ax.transAxes, fontsize=8, va="top",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.9))

# ═══════════════════════════════════════════════════════════════════════════════
# Panel B: Phase-3 forgetting -- simulation + tau_m analytical overlay
# ═══════════════════════════════════════════════════════════════════════════════
ax = axes[1]
t_rel = [t - T_BUILD for t in PHASE3_CPS]

for i, fa in enumerate(FA_VALS):
    v0 = p18_mean(fa, KAPPA, T_BUILD)
    if v0 <= 0 or math.isnan(v0):
        continue
    norm_vals = [p18_mean(fa, KAPPA, t) / v0 for t in PHASE3_CPS]
    errs      = [p18_err(fa, KAPPA, t) / v0 for t in PHASE3_CPS]
    ax.errorbar(t_rel, norm_vals, yerr=errs,
                fmt="o-", color=FA_COLORS[i], ms=3, lw=1.5, capsize=2,
                alpha=0.75, label=f"sim FA={fa:.2f}")

# Analytical: after burst peak at t+30, decay with tau_m=100
# Curve: A × exp(-(t - t_peak)/tau_m), anchored at empirical burst peak ≈ 1.42
T_PEAK  = 30
A_PEAK  = 1.42
t_analytic = np.linspace(T_PEAK, 200, 200)
analytic_decay = A_PEAK * np.exp(-(t_analytic - T_PEAK) / TAU_M)
ax.plot(t_analytic, analytic_decay, "--", color="black", lw=2.0,
        label=r"$\tau_m = 100$ steps (ODE)")

ax.axhline(1.0 / math.e, color="gray", ls=":", lw=1.0, alpha=0.7)
ax.text(195, 1.0/math.e + 0.02, "1/e", fontsize=8, ha="right", color="gray")
ax.axhline(1.0, color="gray", ls=":", lw=0.8, alpha=0.4)

ax.set_xlabel(r"Steps after wave cessation ($t - T_{\rm build}$)", fontsize=10)
ax.set_ylabel(r"sg4 / sg4$(T_{\rm build})$", fontsize=10)
ax.set_title(r"\textbf{B.} Phase-3 forgetting: ODE predicts $\tau_m=100$ steps",
             fontsize=9.5)
ax.legend(fontsize=7.5, framealpha=0.85)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.text(0.97, 0.50,
        "After burst peak,\ndecay follows $\\tau_m$.\nODE prediction:\n"
        r"$F(t) \propto e^{-(t-t_{\rm peak})/\tau_m}$",
        transform=ax.transAxes, fontsize=7.5, ha="right", va="top",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcyan", alpha=0.9))

# ═══════════════════════════════════════════════════════════════════════════════
# Panel C: Buildup gap -- ODE vs simulation (both normalised to T_BUILD)
# ═══════════════════════════════════════════════════════════════════════════════
ax = axes[2]

for i, fa in enumerate([0.20, 0.40]):
    # Simulation
    v0_sim = p18_mean(fa, KAPPA, T_BUILD)
    sim_norm_vals = [p18_mean(fa, KAPPA, t) / v0_sim
                     if v0_sim > 0 else float("nan")
                     for t in PHASE1_CPS]
    ax.plot(PHASE1_CPS, sim_norm_vals, "o-", color=FA_COLORS[i+1],
            ms=6, lw=1.8, label=f"Sim FA={fa:.2f}")

    # ODE
    ode_res = solve_ode_buildup(fa)
    F_T_ode = ode_res.get(T_BUILD, 1.0)
    ode_norm_vals = [ode_res.get(t, float("nan")) / F_T_ode
                     if F_T_ode > 0 else float("nan")
                     for t in PHASE1_CPS]
    ax.plot(PHASE1_CPS, ode_norm_vals, "--", color=FA_COLORS[i+1],
            lw=1.8, alpha=0.7, label=f"ODE FA={fa:.2f}")

ax.set_xlabel("Step (Phase 1)", fontsize=10)
ax.set_ylabel(r"sg4 / sg4$(T_{\rm build})$  [normalised]", fontsize=10)
ax.set_title(r"\textbf{C.} Buildup gap: ODE $\sim$9$\times$ faster than simulation",
             fontsize=9.5)
ax.legend(fontsize=8, framealpha=0.85)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.text(0.05, 0.97,
        "ODE (dashed): reaches 63\\%\nin $\\tau_{\\rm base}=200$ steps.\n"
        "Simulation (solid):\n63\\% at $\\sim$1800 steps.\n"
        r"$\rightarrow$ Spatial formation" + "\n   unmodeled (Paper 21)",
        transform=ax.transAxes, fontsize=7.5, va="top",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.9))

fig.suptitle(
    r"The coupled ODE recovers the saturation law and Phase-3 forgetting;"
    r" buildup gap identifies the spatial correlation mechanism (Paper 21)",
    fontsize=10, y=1.02
)
fig.tight_layout()

OUT = os.path.join(DIR, "paper20_figure1")
fig.savefig(OUT + ".pdf", bbox_inches="tight")
fig.savefig(OUT + ".png", dpi=150, bbox_inches="tight")
print(f"Saved {OUT}.pdf and {OUT}.png")

# Summary
print(f"\nKey ODE parameters:")
print(f"  tau_base = {TAU_BASE:.0f}  tau_m = {TAU_M:.0f}")
print(f"  K_eff    = KAPPA/P_consol = {KAPPA}/{P_CONSOL:.4f} = {K_EFF:.4f}")
print(f"\nSteady-state law (normalised):")
print(f"  {'FA':>5} | {'ODE':>8} | {'Sim':>8} | {'diff':>8}")
for fa, sn in zip(FA_VALS, sim_norm):
    ode_n = (fa/(fa+K_EFF)) / ((max(FA_VALS))/(max(FA_VALS)+K_EFF))
    print(f"  {fa:>5.2f} | {ode_n:>8.4f} | {sn:>8.4f} | {ode_n-sn:>+8.4f}")

print(f"\nBuildup gap (ODE/sim at each checkpoint, FA=0.40):")
ode_bu = solve_ode_buildup(0.40)
F_T    = ode_bu[T_BUILD]
v0_s   = p18_mean(0.40, KAPPA, T_BUILD)
print(f"  {'t':>6} | {'ODE/SS':>8} | {'Sim/SS':>8} | {'ratio':>8}")
for t in PHASE1_CPS:
    on = ode_bu.get(t, float("nan"))/F_T
    sn = p18_mean(0.40, KAPPA, t)/v0_s
    print(f"  {t:>6} | {on:>8.4f} | {sn:>8.4f} | {on/sn if sn>0 else float('nan'):>8.1f}x")
