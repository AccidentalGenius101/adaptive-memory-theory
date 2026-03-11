"""Paper 68 figure: Order parameter exponent, functional-form discrimination, Manna diagnostic."""
import json, math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

ANALYSIS = Path(__file__).parent / 'results' / 'paper68_analysis.json'
with open(ANALYSIS) as f:
    d = json.load(f)

agg_A = d['A']
agg_B = d['B']
agg_C = d.get('C', {})

PC_A  = [0.001, 0.002, 0.003, 0.005, 0.007, 0.010, 0.015, 0.020, 0.030, 0.050]
PC_B  = [0.003, 0.005, 0.007, 0.010, 0.020]
L_B   = [80, 100, 120, 160]
COLORS = {80:'C0', 100:'C1', 120:'C2', 160:'C3'}

fig, axes = plt.subplots(2, 3, figsize=(15, 9))
fig.suptitle('Paper 68: Order Parameter Exponent at $p_c=0^+$ — '
             'Power-law vs Essential Singularity', fontsize=13, fontweight='bold')

# ── (a) log|M| vs log(P) — functional form test ──────────────────────────────
ax = axes[0, 0]
pc_vals = []; aM_vals = []; aM_se = []
for pc in PC_A:
    k = f'pc{pc:.4f}'
    if k in agg_A and agg_A[k].get('absM', float('nan')) > 1e-9:
        pc_vals.append(agg_A[k]['pc'])
        aM_vals.append(agg_A[k]['absM'])
        aM_se.append(agg_A[k].get('absM_se', 0.0))

log_pc = np.log(pc_vals); log_aM = np.log(aM_vals)
ax.errorbar(pc_vals, aM_vals,
            yerr=aM_se, fmt='o', color='C0', ms=7, capsize=3, label='Data (L=160)')

# Power-law fit
beta_fit, log_c = np.polyfit(log_pc, log_aM, 1)
p_fit = np.logspace(np.log10(min(pc_vals))*1.1, np.log10(max(pc_vals))*0.9, 100)
aM_pl = np.exp(log_c) * p_fit ** beta_fit
ax.plot(p_fit, aM_pl, '--', color='C3',
        label=fr'Power law fit $\beta={beta_fit:.3f}$')

# Manna reference line
aM_manna = np.exp(log_c) * p_fit ** 0.639
ax.plot(p_fit, aM_manna, ':', color='C1', label=r'Manna $\beta=0.639$')
# Paper63 reference line
aM_p63 = np.exp(log_c) * p_fit ** 0.65
ax.plot(p_fit, aM_p63, '-.', color='C2', label=r'Paper 63 $\beta=0.650$')

ax.set_xscale('log'); ax.set_yscale('log')
ax.set_xlabel('$P_{causal}$'); ax.set_ylabel('$|M|$')
ax.set_title('(a) $|M|$ vs $P$ — functional form (L=160)')
ax.legend(fontsize=8)
ax.set_xlim(5e-4, 0.1)

# ── (b) U4 vs P at L=160 ─────────────────────────────────────────────────────
ax = axes[0, 1]
u4_vals = [agg_A.get(f'pc{pc:.4f}', {}).get('U4', float('nan')) for pc in PC_A]
u4_se   = [agg_A.get(f'pc{pc:.4f}', {}).get('U4_se', 0.0) for pc in PC_A]
pc_arr = np.array(PC_A)
u4_arr = np.array(u4_vals); u4_se_arr = np.array(u4_se)
ok = ~np.isnan(u4_arr)
ax.errorbar(pc_arr[ok], u4_arr[ok], yerr=u4_se_arr[ok],
            fmt='s-', color='C0', ms=7, capsize=3)
ax.axhline(2/3, color='C3', linestyle='--', label='Ordered limit 2/3')
ax.axhline(0.0, color='gray', linestyle=':', alpha=0.5, label='Disordered limit 0')
ax.set_xscale('log')
ax.set_xlabel('$P_{causal}$'); ax.set_ylabel('$U_4$')
ax.set_title('(b) Binder cumulant vs P (L=160)')
ax.legend(fontsize=8)
ax.set_ylim(-0.05, 0.75)

# ── (c) R² comparison — which functional form fits best ──────────────────────
ax = axes[0, 2]
inv_sqrtP = np.array([1.0/math.sqrt(pc) for pc in pc_vals])
inv_P     = np.array([1.0/pc for pc in pc_vals])

# Compute R²
def r2_fit(x, y):
    _, resid = np.polyfit(x, y, 1, full=False)[:2], None
    c = np.polyfit(x, y, 1)
    y_pred = np.polyval(c, x)
    ss_res = np.sum((y - y_pred)**2)
    ss_tot = np.sum((y - np.mean(y))**2)
    return 1 - ss_res / ss_tot if ss_tot > 0 else float('nan')

log_aM_arr = np.array(log_aM)
r2_pl  = r2_fit(log_pc, log_aM_arr)
r2_bkt = r2_fit(inv_sqrtP, log_aM_arr)
r2_ord = r2_fit(inv_P, log_aM_arr)

bars = ax.bar(['Power law\n($|M|\\sim P^\\beta$)',
               'BKT ess. sing.\n($e^{-A/\\sqrt{P}}$)',
               'Ord. ess. sing.\n($e^{-A/P}$)'],
              [r2_pl, r2_bkt, r2_ord],
              color=['C0', 'C1', 'C2'], alpha=0.8, edgecolor='k')
for bar, val in zip(bars, [r2_pl, r2_bkt, r2_ord]):
    ax.text(bar.get_x() + bar.get_width()/2, val + 0.01,
            f'{val:.3f}', ha='center', fontsize=10, fontweight='bold')
ax.set_ylabel('$R^2$')
ax.set_title('(c) Goodness-of-fit: 3 functional forms')
ax.set_ylim(0, 1.0)
ax.axhline(0.95, color='C3', linestyle='--', alpha=0.5, label='R²=0.95 threshold')
ax.legend(fontsize=8)

# ── (d) FSS: |M| vs L at selected P values ───────────────────────────────────
ax = axes[1, 0]
for pc in [0.005, 0.010, 0.020]:
    L_v = []; aM_v = []
    for L in L_B:
        k = f'L{L}_pc{pc:.4f}'
        if k in agg_B:
            v = agg_B[k]
            if not math.isnan(v.get('absM', float('nan'))):
                L_v.append(L); aM_v.append(v['absM'])
    if L_v:
        ax.plot(L_v, aM_v, 'o-', label=f'P={pc:.3f}', ms=6)

# Reference slopes
L_ref = np.array([80, 160])
# Disordered: |M| ~ L^{-1}
ax.plot(L_ref, 0.001 * (L_ref/80.)**(-1.0), 'k--', alpha=0.4, label='$L^{-1}$ (disorder)')
# Critical: |M| ~ L^{-beta/nu} = L^{-0.670}
ax.plot(L_ref, 0.001 * (L_ref/80.)**(-0.670), 'k:', alpha=0.4, label='$L^{-0.670}$ (critical)')
ax.set_xscale('log'); ax.set_yscale('log')
ax.set_xlabel('L'); ax.set_ylabel('$|M|$')
ax.set_title('(d) FSS: $|M|$ vs L')
ax.legend(fontsize=7)
ax.set_xticks([80, 100, 120, 160]); ax.get_xaxis().set_major_formatter(
    matplotlib.ticker.ScalarFormatter())

# ── (e) FSS slope d(log|M|)/d(logL) vs P ─────────────────────────────────────
ax = axes[1, 1]
slopes = []
for pc in PC_B:
    L_v = []; aM_v = []
    for L in L_B:
        k = f'L{L}_pc{pc:.4f}'
        if k in agg_B and not math.isnan(agg_B[k].get('absM', float('nan'))):
            L_v.append(L); aM_v.append(agg_B[k]['absM'])
    if len(L_v) >= 3:
        slope, _ = np.polyfit(np.log(L_v), np.log(aM_v), 1)
        slopes.append((pc, slope))

if slopes:
    ps, ss = zip(*slopes)
    ax.plot(ps, ss, 'o-', color='C0', ms=8, label='Measured slope')
    ax.axhline(-0.670, color='C3', linestyle='--', label=r'$-\beta/\nu=-0.670$ (Paper 63)')
    ax.axhline(-1.0, color='C2', linestyle=':', label=r'$-1.0$ (disordered)')
    ax.axhline(0.0, color='gray', linestyle=':', alpha=0.5)
ax.set_xscale('log')
ax.set_xlabel('$P_{causal}$'); ax.set_ylabel(r'$d\ln|M|/d\ln L$')
ax.set_title('(e) FSS slope: disorder→order crossover')
ax.legend(fontsize=7)
ax.set_ylim(-1.5, 0.3)

# ── (f) Manna conservation diagnostic ────────────────────────────────────────
ax = axes[1, 2]
fc = agg_C.get('flux_create', float('nan'))
fd = agg_C.get('flux_decay', float('nan'))
fr = agg_C.get('flux_ratio', float('nan'))

if not math.isnan(fc):
    bars2 = ax.bar(['$\phi$ creation\n(consolidation)', '$\phi$ decay\n(FIELD\_DECAY)'],
                   [fc, fd], color=['C0', 'C3'], alpha=0.8, edgecolor='k')
    for bar, val in zip(bars2, [fc, fd]):
        ax.text(bar.get_x() + bar.get_width()/2, val * 1.02,
                f'{val:.4f}', ha='center', fontsize=10, fontweight='bold')
    ax.set_ylabel('Mean flux per step')
    ax.set_title(f'(f) Manna flux balance (L=120, P=0.010)\nRatio = {fr:.3f} '
                 r'(Manna requires $\approx 1.0$)')
    ax.set_ylim(0, max(fc, fd) * 1.3)
    ax.axhline(min(fc, fd), color='k', linestyle='--', alpha=0.3)
    # Annotate ratio
    ax.text(0.5, 0.85, f'create/decay = {fr:.2f}\n≠ 1.0 → NOT Manna class',
            transform=ax.transAxes, ha='center', fontsize=10,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
else:
    ax.text(0.5, 0.5, 'Phase C data unavailable', transform=ax.transAxes,
            ha='center', va='center', fontsize=12)
    ax.set_title('(f) Manna flux balance')

plt.tight_layout()
out = Path(__file__).parent / 'paper68_figure1.pdf'
plt.savefig(out, bbox_inches='tight', dpi=150)
print(f'Saved -> {out}  ({out.stat().st_size} bytes)')
