"""
paper35_figure1.py -- Figure 1 for Paper 35: Relay Coupling

Two panels:
  (a) G vs WR  -- peaked gain curve, wave-flux threshold Phi=1 marked
  (b) G vs N_zones -- G vs zone count at WR=4.8, N_crit = HALF/(4*xi) marked

Physical framing:
  Phi = WR * r_wave / WAVE_DUR  (dimensionless wave flux)
  zeta = zone_width / xi        (zone width in units of correlation length)
  Optimal relay when Phi < 1 AND zeta > 4
"""
import json, math, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from collections import defaultdict

RES = "results/paper35_results.json"
with open(RES) as f:
    data = json.load(f)

by = defaultdict(list)
for r in data:
    by[(r['WR'], r['n_zones'], r['coupling'])].append(r)

def mn(lst):
    v = [x for x in lst if not math.isnan(x)]
    return float(np.mean(v)) if v else float('nan')

def std(lst):
    v = [x for x in lst if not math.isnan(x)]
    return float(np.std(v, ddof=1)) / math.sqrt(len(v)) if len(v) > 1 else float('nan')

WR_SWEEP = [0.6, 1.2, 2.4, 4.8, 9.6, 19.2]
N_FIXED  = 4
N_SWEEP  = [2, 4, 5, 8]
WR_FIXED = 4.8

xi      = 2.5    # correlation length (sites), Paper 3
r_wave  = 2.0    # wave footprint radius (Manhattan sites)
WAVE_DUR = 15.0
HALF    = 40

# ── Panel A data: G vs WR ─────────────────────────────────────────────────────
Gs_a, Gs_a_err, Phi_a = [], [], []
for WR in WR_SWEEP:
    geo_sn  = [r.get('l2_sg4n', float('nan')) for r in by[(WR, N_FIXED, 'geo')]]
    ctrl_sn = [r.get('l2_sg4n', float('nan')) for r in by[(WR, N_FIXED, 'ctrl')]]
    sg = mn(geo_sn); sc = mn(ctrl_sn)
    G  = sg / sc if sc > 1e-6 else float('nan')
    # seed-level G for error bars
    seed_Gs = [g/c for g,c in zip(geo_sn, ctrl_sn)
               if not math.isnan(g) and not math.isnan(c) and c > 1e-6]
    se = float(np.std(seed_Gs, ddof=1) / math.sqrt(len(seed_Gs))) if len(seed_Gs)>1 else 0
    Gs_a.append(G); Gs_a_err.append(se)
    Phi_a.append(WR * r_wave / WAVE_DUR)

WR_arr = np.array(WR_SWEEP)
G_arr  = np.array(Gs_a)

# Peaked Gaussian in log-WR space: G(WR) = 1 + A*exp(-0.5*((log WR - log WR_opt)/sigma)^2)
def peaked(WR, A, WR_opt, sigma):
    return 1 + A * np.exp(-0.5 * ((np.log(WR) - np.log(WR_opt))**2) / sigma**2)

try:
    popt, _ = curve_fit(peaked, WR_arr, G_arr, p0=[2.0, 5.0, 0.8], maxfev=5000)
    A_fit, WR_opt_fit, sigma_fit = popt
    WR_fine = np.logspace(np.log10(0.4), np.log10(25), 300)
    G_fit_curve = peaked(WR_fine, *popt)
except Exception as e:
    print(f"WR fit failed: {e}")
    A_fit, WR_opt_fit, sigma_fit = None, None, None
    WR_fine = G_fit_curve = None

# Wave-flux threshold Phi=1 -> WR_thresh
WR_thresh = WAVE_DUR / r_wave   # Phi=1 -> WR = WAVE_DUR/r_wave = 7.5

# ── Panel B data: G vs N ──────────────────────────────────────────────────────
Gs_b, Gs_b_err, zeta_b, N_b = [], [], [], []
for N in N_SWEEP:
    geo_sn  = [r.get('l2_sg4n', float('nan')) for r in by[(WR_FIXED, N, 'geo')]]
    ctrl_sn = [r.get('l2_sg4n', float('nan')) for r in by[(WR_FIXED, N, 'ctrl')]]
    sg = mn(geo_sn); sc = mn(ctrl_sn)
    G  = sg / sc if sc > 1e-6 else float('nan')
    seed_Gs = [g/c for g,c in zip(geo_sn, ctrl_sn)
               if not math.isnan(g) and not math.isnan(c) and c > 1e-6]
    se = float(np.std(seed_Gs, ddof=1) / math.sqrt(len(seed_Gs))) if len(seed_Gs)>1 else 0
    Gs_b.append(G); Gs_b_err.append(se)
    zeta_b.append((HALF // N) / xi)
    N_b.append(N)

# Critical zone width: zw_crit ~ 4*xi  -> N_crit = HALF/(4*xi)
N_crit = HALF / (4 * xi)   # = 4.0

# ── Figure ────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
BLUE   = "#1f77b4"
RED    = "#d62728"
ORANGE = "#ff7f0e"
GRAY   = "#888888"

# ─── Panel A ──────────────────────────────────────────────────────────────────
ax = axes[0]
ax.axhline(1.0, color=GRAY, lw=0.8, ls='--', zorder=0, label='G=1 (no gain)')
ax.axvline(WR_thresh, color=ORANGE, lw=1.2, ls=':', zorder=0,
           label=r'$\Phi=1$ ($\mathrm{WR}=7.5$)')
if WR_fine is not None:
    ax.plot(WR_fine, G_fit_curve, '-', color=BLUE, lw=1.6, alpha=0.7, zorder=1,
            label=rf'Peaked fit ($\mathrm{{WR}}_\star={WR_opt_fit:.1f}$)')
ax.errorbar(WR_arr, G_arr, yerr=Gs_a_err, fmt='o', color=BLUE,
            ms=6, capsize=3, zorder=2, label='Data (5 seeds)')

# annotate Phi values
for i, (WR, G, Phi) in enumerate(zip(WR_SWEEP, Gs_a, Phi_a)):
    if not math.isnan(G):
        va = 'bottom' if G > 1.0 else 'top'
        off = 0.08 if G > 1.0 else -0.08
        ax.annotate(rf'$\Phi={Phi:.2f}$', xy=(WR, G), xytext=(WR, G+off),
                    fontsize=7, ha='center', va=va, color=GRAY)

ax.set_xscale('log')
ax.set_xlabel('Wave rate WR (launches per $T_{\\mathrm{dur}}$)', fontsize=11)
ax.set_ylabel(r'Relay gain $G = \mathrm{sg4\_norm}_\mathrm{geo} / \mathrm{sg4\_norm}_\mathrm{ctrl}$', fontsize=10)
ax.set_title(r'\textbf{(a)}\ WR sweep ($N_\mathrm{zones}=4$)', fontsize=11)
ax.legend(fontsize=8, loc='upper right')
ax.set_ylim(bottom=0)

# ─── Panel B ──────────────────────────────────────────────────────────────────
ax = axes[1]
ax.axhline(1.0, color=GRAY, lw=0.8, ls='--', zorder=0, label='G=1 (no gain)')
ax.axvline(N_crit, color=ORANGE, lw=1.2, ls=':', zorder=0,
           label=rf'$N_\mathrm{{crit}}={N_crit:.0f}$ ($z_w=4\xi$)')
ax.errorbar(N_b, Gs_b, yerr=Gs_b_err, fmt='s', color=RED,
            ms=6, capsize=3, zorder=2, label='Data (5 seeds)')

# annotate zeta values
for N, G, zeta in zip(N_b, Gs_b, zeta_b):
    if not math.isnan(G):
        va = 'bottom' if G > 1.0 else 'top'
        off = 0.15 if G > 1.0 else -0.15
        ax.annotate(rf'$z_w={zeta:.1f}\xi$', xy=(N, G), xytext=(N+0.1, G+off),
                    fontsize=7, ha='left', va=va, color=GRAY)

ax.set_xlabel(r'Number of zones $N_\mathrm{zones}$', fontsize=11)
ax.set_ylabel(r'Relay gain $G$', fontsize=10)
ax.set_title(r'\textbf{(b)}\ $N_\mathrm{zones}$ sweep ($\mathrm{WR}=4.8$)', fontsize=11)
ax.legend(fontsize=8, loc='upper right')
ax.set_xticks(N_b)
ax.set_ylim(bottom=0)

plt.tight_layout()
fig.savefig('paper35_figure1.png', dpi=150, bbox_inches='tight')
fig.savefig('paper35_figure1.pdf', bbox_inches='tight')
print("Saved paper35_figure1.png / .pdf")

# ── Print summary table ───────────────────────────────────────────────────────
print("\n=== Panel A: WR sweep (N=4) ===")
print(f"{'WR':>6} {'Phi=WR*2/15':>12} {'G':>7} {'SE':>6}")
for WR, G, se, Phi in zip(WR_SWEEP, Gs_a, Gs_a_err, Phi_a):
    print(f"{WR:6.1f} {Phi:12.3f} {G:7.3f} {se:6.3f}")
if WR_opt_fit:
    print(f"Peaked fit: WR_opt={WR_opt_fit:.2f}, A={A_fit:.3f}, sigma={sigma_fit:.3f}")

print("\n=== Panel B: N sweep (WR=4.8) ===")
print(f"{'N':>4} {'zw':>6} {'zeta=zw/xi':>12} {'G':>7} {'SE':>6}")
for N, G, se, zeta in zip(N_b, Gs_b, Gs_b_err, zeta_b):
    zw = HALF // N
    print(f"{N:4d} {zw:6d} {zeta:12.2f} {G:7.3f} {se:6.3f}")
print(f"N_crit = HALF/(4*xi) = {N_crit:.1f}")
