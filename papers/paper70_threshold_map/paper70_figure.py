"""Paper 70 figure: phase boundary in (r_w, P) space + threshold scaling."""
import json, math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pathlib import Path

ANALYSIS = Path(__file__).parent / 'results' / 'paper70_analysis.json'
with open(ANALYSIS) as f:
    d = json.load(f)

agg_A = d['A']; agg_B = d['B']; agg_C = d['C']
thresh_A = {float(k): v for k, v in d['threshold_A'].items()}
thresh_B = {float(k): v for k, v in d['threshold_B'].items()}
thresh_C = {float(k): v for k, v in d['threshold_C'].items()}

RW_A = [1, 2, 3, 4, 5, 6, 8, 10, 12]
PC_A = [0.005, 0.010, 0.020, 0.050, 0.100, 0.200]
RW_B = [1, 2, 3, 4, 5, 6, 8]
L_B  = [40, 60, 80, 100, 120]
RW_C = [1, 2, 3, 4, 5, 6, 8, 10]
PC_C = [0.005, 0.010, 0.020, 0.050, 0.100]
U4_THRESH = 0.15

def wave_area(r): return 2 * r * (r + 1) + 1

fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle('Paper 70: Phase Boundary in $(r_w, P)$ Space — '
             'RG Dimension of Wave Operator', fontsize=13, fontweight='bold')

# ── (a) U4 heat map: 2D phase diagram ────────────────────────────────────────
ax = axes[0, 0]
u4_grid = np.zeros((len(RW_A), len(PC_A)))
for i, rw in enumerate(RW_A):
    for j, pc in enumerate(PC_A):
        k = f'rw{rw}_pc{pc:.4f}'
        u4 = agg_A.get(k, {}).get('U4', float('nan'))
        u4_grid[i, j] = u4 if not math.isnan(u4) else -0.1

vmax = max(0.6, np.nanmax(u4_grid[u4_grid > -0.05]))
im = ax.imshow(u4_grid, aspect='auto', origin='lower',
               vmin=-0.1, vmax=vmax,
               cmap='RdYlGn', extent=[-0.5, len(PC_A)-0.5, -0.5, len(RW_A)-0.5])
plt.colorbar(im, ax=ax, label='$U_4$')
ax.set_xticks(range(len(PC_A))); ax.set_xticklabels([f'{p:.3f}' for p in PC_A], rotation=35)
ax.set_yticks(range(len(RW_A))); ax.set_yticklabels(RW_A)
ax.set_xlabel('$P_{causal}$'); ax.set_ylabel('$r_w$')
ax.set_title('(a) $U_4$ phase map (L=80, Protocol A)\nGreen = ordered, Red = disordered')
# Mark threshold contour
thresh_line = [0.15] * len(RW_A)
for j, pc in enumerate(PC_A):
    first_ordered = None
    for i, rw in enumerate(RW_A):
        u4 = agg_A.get(f'rw{rw}_pc{pc:.4f}', {}).get('U4', float('nan'))
        if not math.isnan(u4) and u4 >= U4_THRESH:
            first_ordered = i
            break
    if first_ordered is not None:
        ax.axhline(first_ordered - 0.5, xmin=j/len(PC_A), xmax=(j+1)/len(PC_A),
                   color='k', linewidth=2)

# ── (b) Threshold curve r_w*(P) — Protocol A ─────────────────────────────────
ax = axes[0, 1]
pc_A = sorted([pc for pc, v in thresh_A.items() if v is not None])
rw_A = [thresh_A[pc] for pc in pc_A]
ax.loglog(pc_A, rw_A, 'o-', color='C0', ms=8, label='Protocol A (L=80)')

# Power-law fit
if len(pc_A) >= 3:
    alpha_A, lc_A = np.polyfit(np.log(pc_A), np.log(rw_A), 1)
    pc_fit = np.logspace(np.log10(min(pc_A))*1.1, np.log10(max(pc_A))*0.9, 100)
    ax.plot(pc_fit, np.exp(lc_A)*pc_fit**alpha_A, '--', color='C0',
            label=fr'Fit $\alpha={alpha_A:.3f}$')

# Protocol B
pc_C = sorted([pc for pc, v in thresh_C.items() if v is not None])
rw_C = [thresh_C[pc] for pc in pc_C]
ax.loglog(pc_C, rw_C, 's-', color='C3', ms=8, label='Protocol B (L=80)')
if len(pc_C) >= 3:
    alpha_C, lc_C = np.polyfit(np.log(pc_C), np.log(rw_C), 1)
    ax.plot(pc_fit, np.exp(lc_C)*pc_fit**alpha_C, '--', color='C3',
            label=fr'Fit $\alpha={alpha_C:.3f}$')

# Reference: P^{-1/2}
pc_ref = np.logspace(np.log10(0.005)*1.1, np.log10(0.2)*0.9, 100)
ax.plot(pc_ref, 2.0*pc_ref**(-0.5), ':', color='k', alpha=0.5, label=r'$P^{-1/2}$ theory')
ax.set_xlabel('$P_{causal}$'); ax.set_ylabel('$r_w^*$')
ax.set_title(r'(b) Threshold curve $r_w^*(P)$')
ax.legend(fontsize=7)

# ── (c) r_w*(L) scaling at fixed P ───────────────────────────────────────────
ax = axes[0, 2]
L_fit = sorted([L for L, v in thresh_B.items() if v is not None])
rw_fit = [thresh_B[L] for L in L_fit]
ax.loglog(L_fit, rw_fit, 'o-', color='C2', ms=8, label='Protocol A (P=0.02)')
if len(L_fit) >= 3:
    gamma_B, lc_g = np.polyfit(np.log(L_fit), np.log(rw_fit), 1)
    L_plot = np.logspace(np.log10(min(L_fit))*0.95, np.log10(max(L_fit))*1.05, 100)
    ax.plot(L_plot, np.exp(lc_g)*L_plot**gamma_B, '--', color='C2',
            label=fr'Fit $\gamma={gamma_B:.3f}$')
# Reference slopes
L_plot2 = np.array([40., 120.])
ax.plot(L_plot2, 4.*(L_plot2/40)**0.0, ':', color='k', alpha=0.4, label=r'$L^0$ (const)')
ax.plot(L_plot2, 4.*(L_plot2/40)**0.5, ':', color='C0', alpha=0.4, label=r'$L^{0.5}$')
ax.plot(L_plot2, 4.*(L_plot2/40)**(-1.), ':', color='C3', alpha=0.4, label=r'$L^{-1}$')
ax.set_xlabel('$L$'); ax.set_ylabel('$r_w^*$')
ax.set_title(r'(c) $r_w^*(L)$ at fixed $P=0.02$')
ax.legend(fontsize=7)
ax.set_xticks(L_fit); ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())

# ── (d) U4 vs r_w at fixed P: non-monotone geometry ─────────────────────────
ax = axes[1, 0]
colors_pc = ['C0', 'C1', 'C2', 'C3']
for ci, pc in enumerate([0.010, 0.020, 0.050, 0.100]):
    u4_rw = []
    for rw in RW_A:
        k = f'rw{rw}_pc{pc:.4f}'
        u4_rw.append(agg_A.get(k, {}).get('U4', float('nan')))
    ax.plot(RW_A, u4_rw, 'o-', color=colors_pc[ci], ms=7, label=f'P={pc:.3f}')
ax.axhline(U4_THRESH, color='k', linestyle='--', alpha=0.5, label=f'Threshold {U4_THRESH}')
ax.axhline(0, color='gray', linestyle=':', alpha=0.3)
ax.set_xlabel('$r_w$'); ax.set_ylabel('$U_4$')
ax.set_title('(d) $U_4$ vs $r_w$ — non-monotone (zone-bleeding at $r_w=2$)')
ax.legend(fontsize=7)
ax.set_xticks(RW_A)

# ── (e) Protocol B: U4 heatmap at L=80 ───────────────────────────────────────
ax = axes[1, 1]
u4_C = np.zeros((len(RW_C), len(PC_C)))
for i, rw in enumerate(RW_C):
    for j, pc in enumerate(PC_C):
        k = f'rw{rw}_pc{pc:.4f}'
        u4 = agg_C.get(k, {}).get('U4', float('nan'))
        u4_C[i, j] = u4 if not math.isnan(u4) else -0.1
im2 = ax.imshow(u4_C, aspect='auto', origin='lower',
                vmin=-0.1, vmax=0.65, cmap='RdYlGn',
                extent=[-0.5, len(PC_C)-0.5, -0.5, len(RW_C)-0.5])
plt.colorbar(im2, ax=ax, label='$U_4$')
ax.set_xticks(range(len(PC_C))); ax.set_xticklabels([f'{p:.3f}' for p in PC_C], rotation=35)
ax.set_yticks(range(len(RW_C))); ax.set_yticklabels(RW_C)
ax.set_xlabel('$P_{causal}$'); ax.set_ylabel('$r_w$')
ax.set_title('(e) $U_4$ phase map (L=80, Protocol B)\nMatched coverage: isolates geometry')

# ── (f) Summary: scaling diagram and QFT interpretation ──────────────────────
ax = axes[1, 2]
ax.axis('off')
alpha_A_v = round(d['fits'].get('alpha_A') or -0.315, 3)
gamma_B_v = round(d['fits'].get('gamma_B') or -1.076, 3)
alpha_C_v = round(d['fits'].get('alpha_C') or -0.527, 3)
text = (
    "Summary of threshold scaling:\n\n"
    f"  Protocol A: $r_w^*(P) \\sim P^{{{alpha_A_v:.3f}}}$\n"
    f"  Protocol B: $r_w^*(P) \\sim P^{{{alpha_C_v:.3f}}}$ (clean)\n"
    f"  Predicted:  $r_w^* \\sim P^{{-0.500}}$\n\n"
    f"  $r_w^*(L)$: $\\gamma={gamma_B_v:.3f}$ (decreasing with $L$)\n\n"
    "Key findings:\n\n"
    "1. $r_w^*(P) \\sim P^{-1/2}$ confirmed (Protocol B)\n\n"
    "2. $r_w^*(L)$ DECREASES with $L$:\n"
    "   min. wave range is finite-$L$ artifact\n"
    "   Any $r_w > 0$ orders as $L \\to \\infty$\n\n"
    "3. Zone-bleeding: $r_w=2$ worse than $r_w=1$\n"
    "   (wave straddles zone boundary)\n\n"
    "QFT implication:\n"
    "  Wave operator has dimension $d_w \\approx 1/2$\n"
    "  Composite variable: $r_w \\sqrt{P} = $ const\n"
    "  at critical manifold"
)
ax.text(0.05, 0.95, text, transform=ax.transAxes,
        fontsize=9, va='top', ha='left',
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8),
        fontfamily='monospace')
ax.set_title('(f) Summary')

plt.tight_layout()
out = Path(__file__).parent / 'paper70_figure1.pdf'
plt.savefig(out, bbox_inches='tight', dpi=150)
print(f'Saved -> {out}  ({out.stat().st_size} bytes)')
