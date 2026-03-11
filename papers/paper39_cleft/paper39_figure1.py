"""
paper39_figure1.py -- Figure 1 for Paper 39: Synaptic Cleft Geometry

Three panels:
  (a) G vs D_cleft: deterministic shift -- stays high at all offsets
  (b) G vs sigma: stochastic noise -- also stays high, slower degradation
  (c) G_det vs G_stoch at matched magnitudes -- prediction reversal: det >= stoch
"""
import json, math, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import defaultdict
from scipy.stats import norm

RES = "results/paper39_results.json"
with open(RES) as f: data = json.load(f)

ZW = 10  # zone width for N=4
D_CLEFT_SWEEP = [0, 2, 5, 8, 10, 12, 15, 20]
SIGMA_SWEEP   = [2, 5, 8, 10, 12, 15, 20]

def mn(lst):
    v=[x for x in lst if x is not None and not math.isnan(x)]
    return float(np.mean(v)) if v else float('nan')
def se(lst):
    v=[x for x in lst if x is not None and not math.isnan(x)]
    return float(np.std(v,ddof=1)/math.sqrt(len(v))) if len(v)>1 else 0.

by = defaultdict(list)
for r in data: by[(r['mode'], r['shift_param'])].append(r)

ctrl_sg4n = mn([r['l2_sg4n'] for r in by[('ctrl',0)]])
G_det0 = mn([r['l2_sg4n'] for r in by[('det',0)]]) / ctrl_sg4n

# Build G arrays
G_det  = [mn([r['l2_sg4n'] for r in by[('det',d)]]) / ctrl_sg4n for d in D_CLEFT_SWEEP]
SE_det = [se([r['l2_sg4n']/ctrl_sg4n for r in by[('det',d)]]) for d in D_CLEFT_SWEEP]

G_stoch  = [G_det0] + [mn([r['l2_sg4n'] for r in by[('stoch',s)]]) / ctrl_sg4n for s in SIGMA_SWEEP]
SE_stoch = [SE_det[0]] + [se([r['l2_sg4n']/ctrl_sg4n for r in by[('stoch',s)]]) for s in SIGMA_SWEEP]
sigma_all = [0] + list(SIGMA_SWEEP)

# Analytical misclassification rate (deterministic)
def misclass_det(d, zw=ZW):
    """Fraction of waves that land in a different zone after shift d."""
    return min(1.0, (d % zw) / zw) if d % zw != 0 else (0.0 if d==0 else 0.0)

# Analytical misclassification rate (stochastic Gaussian)
def misclass_stoch(sigma, zw=ZW):
    """Average probability a wave shifts outside its launch zone.
    Assumes launch position uniform in [0, zw), nearest boundary at zw/2 on average."""
    if sigma < 1e-9: return 0.0
    # Integrate P(|N(0,sigma^2)| > dist_to_boundary) over uniform launch positions
    # dist_to_boundary = min(cx_within, zw - cx_within) for cx_within in [0, zw)
    total = 0.0
    N = 1000
    for i in range(N):
        cx = (i + 0.5) / N * zw
        d_left  = cx
        d_right = zw - cx
        d_min   = min(d_left, d_right)
        total  += 2 * norm.cdf(-d_min / sigma)  # P(|noise| > d_min)
    return total / N

misclass_d_arr = [misclass_det(d) for d in D_CLEFT_SWEEP]
misclass_s_arr = [misclass_stoch(s) for s in sigma_all]

# ── Figure ──────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
GRAY='#888888'; BLUE='#1f77b4'; RED='#e31a1c'; GREEN='#2ca02c'; ORANGE='#ff7f0e'

# ─── Panel A: G vs D_cleft ──────────────────────────────────────────────
ax = axes[0]
ax.axhline(1.0, color=GRAY, lw=0.8, ls='--', alpha=0.6)
ax.axvline(ZW, color=ORANGE, lw=1.0, ls=':', alpha=0.7,
           label=f'$D_{{cleft}}=z_w={ZW}$ (100% misclass)')
ax.errorbar(D_CLEFT_SWEEP, G_det, yerr=SE_det, fmt='o-', color=RED,
            ms=7, capsize=3, lw=1.5, label='Deterministic $D_{{cleft}}$')
# Shade 100% misclassification region
ax.axvspan(ZW, max(D_CLEFT_SWEEP)+1, alpha=0.07, color=ORANGE,
           label='100% misclassification zone')
ax.set_xlabel(r'Cleft offset $D_\mathrm{cleft}$ (sites)', fontsize=11)
ax.set_ylabel(r'Relay gain $G$', fontsize=10)
ax.set_title(r'\textbf{(a)}\ $G$ vs deterministic offset $D_\mathrm{cleft}$', fontsize=11)
ax.legend(fontsize=8)
ax.set_ylim(bottom=0)
ax.set_xticks(D_CLEFT_SWEEP)

# ─── Panel B: G vs sigma ─────────────────────────────────────────────────
ax = axes[1]
ax.axhline(1.0, color=GRAY, lw=0.8, ls='--', alpha=0.6)
ax.axvline(ZW, color=ORANGE, lw=1.0, ls=':', alpha=0.7,
           label=f'$\\sigma=z_w={ZW}$')
ax.errorbar(sigma_all, G_stoch, yerr=SE_stoch, fmt='s-', color=BLUE,
            ms=7, capsize=3, lw=1.5, label='Stochastic $\\sigma$')
ax.set_xlabel(r'Cleft noise $\sigma$ (sites)', fontsize=11)
ax.set_ylabel(r'Relay gain $G$', fontsize=10)
ax.set_title(r'\textbf{(b)}\ $G$ vs stochastic noise $\sigma$', fontsize=11)
ax.legend(fontsize=8)
ax.set_ylim(bottom=0)

# ─── Panel C: G_det vs G_stoch at matched magnitudes ─────────────────────
ax = axes[2]
matched_m = [2, 5, 8, 10, 12, 15, 20]
Gs_det_m  = [mn([r['l2_sg4n'] for r in by[('det',m)]]) / ctrl_sg4n for m in matched_m]
Gs_stoch_m= [mn([r['l2_sg4n'] for r in by[('stoch',m)]]) / ctrl_sg4n for m in matched_m]

ax.axhline(1.0, color=GRAY, lw=0.8, ls='--', alpha=0.4, label='$G_\mathrm{det}/G_\mathrm{stoch}=1$')
ax.plot(matched_m, [d/s for d,s in zip(Gs_det_m,Gs_stoch_m)],
        'D-', color=GREEN, ms=7, lw=1.5, label=r'$G_\mathrm{det}/G_\mathrm{stoch}$')

# Also plot both G curves on twin axis
ax2=ax.twinx()
ax2.plot(matched_m, Gs_det_m,  'o:', color=RED,  ms=5, lw=1.0, alpha=0.6, label='$G_\mathrm{det}$')
ax2.plot(matched_m, Gs_stoch_m,'s:', color=BLUE, ms=5, lw=1.0, alpha=0.6, label='$G_\mathrm{stoch}$')
ax2.set_ylabel(r'$G$ (individual)', fontsize=9, color='gray')
ax2.tick_params(colors='gray')
ax2.set_ylim(bottom=0)
lines2,labs2=ax2.get_legend_handles_labels()

ax.set_xlabel(r'Shift magnitude $m$ (sites)', fontsize=11)
ax.set_ylabel(r'$G_\mathrm{det} / G_\mathrm{stoch}$', fontsize=10)
ax.set_title(r'\textbf{(c)}\ Ratio at matched magnitudes', fontsize=11)
lines1,labs1=ax.get_legend_handles_labels()
ax.legend(lines1+lines2, labs1+labs2, fontsize=8, loc='upper left')
ax.set_ylim(bottom=0)

plt.tight_layout()
fig.savefig('paper39_figure1.png', dpi=150, bbox_inches='tight')
fig.savefig('paper39_figure1.pdf', bbox_inches='tight')
print("Saved paper39_figure1.png / .pdf")

# ── Print misclassification analysis ────────────────────────────────────────
print("\n=== Analytical misclassification vs G ===")
print(f"  ctrl baseline sg4n = {ctrl_sg4n:.4f}")
print(f"\n  Deterministic:")
print(f"  {'D_cleft':>8} {'misclass%':>10} {'G':>8}")
for d, G, mc in zip(D_CLEFT_SWEEP, G_det, misclass_d_arr):
    print(f"  {d:8d} {mc*100:10.0f}% {G:8.3f}")

print(f"\n  Stochastic:")
print(f"  {'sigma':>7} {'misclass%':>10} {'G':>8}")
for s, G, mc in zip(sigma_all, G_stoch, misclass_s_arr):
    print(f"  {s:7d} {mc*100:10.1f}% {G:8.3f}")

print(f"\n  Ratio G_det/G_stoch at matched magnitudes:")
print(f"  {'m':>4} {'G_det':>8} {'G_stoch':>9} {'ratio':>7}")
for m, Gd, Gs in zip(matched_m, Gs_det_m, Gs_stoch_m):
    print(f"  {m:4d} {Gd:8.3f} {Gs:9.3f} {Gd/Gs:7.3f}")
