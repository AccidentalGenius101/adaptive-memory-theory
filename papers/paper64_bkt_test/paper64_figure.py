"""Paper 64 figure: BKT test, large-L Binder, and spatial correlator."""
import json, numpy as np, math, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path

with open(Path(__file__).parent / 'results/paper64_analysis.json') as f:
    d = json.load(f)

L_A   = [40, 60, 80, 100, 120]
PC_A  = [0.000, 0.002, 0.004, 0.006, 0.008, 0.010, 0.015, 0.020, 0.030]
L_B   = [60, 80, 100]
PC_B  = [0.000, 0.005, 0.010, 0.020, 0.050]

COLORS_L = plt.cm.viridis(np.linspace(0.05, 0.95, len(L_A)))

agg_A = d['A']
p_c   = d['p_c']
pc_ests = d.get('pc_estimates', [])

fig = plt.figure(figsize=(17, 5.2))
gs  = gridspec.GridSpec(1, 3, figure=fig, wspace=0.42)

# ── Panel (a): U4 vs P for all L (Phase A) ──────────────────────────────────
ax_a = fig.add_subplot(gs[0, 0])

for i, L in enumerate(L_A):
    pcs = []; u4s = []
    for pc in PC_A:
        k = f'L{L}_pc{pc:.3f}'
        if k in agg_A:
            pcs.append(pc)
            u4s.append(agg_A[k]['U4'])
    ax_a.plot(pcs, u4s, 'o-', color=COLORS_L[i], lw=1.8, ms=5,
              label=f'$L={L}$')

ax_a.axhline(2/3, color='gray', ls=':', lw=1, alpha=0.7, label='2/3 (ordered)')
ax_a.axhline(0.0, color='gray', ls='--', lw=0.8, alpha=0.5)
if p_c > 0:
    ax_a.axvline(p_c, color='red', ls='--', lw=1.3, alpha=0.8,
                 label=f'$p_c \\approx {p_c:.4f}$')
else:
    ax_a.axvline(0, color='red', ls='--', lw=1.3, alpha=0.8,
                 label='$p_c \\approx 0$')

ax_a.set_xlabel('$P_{\\rm causal}$', fontsize=11)
ax_a.set_ylabel('Binder cumulant $U_4$', fontsize=10)
ax_a.set_title(f'(a) Large-$L$ Binder: Phase A\n'
               f'$U_4(L=120) > U_4(L=40)$ for all $P \\geq 0.002$.\n'
               f'Consistent with $p_c = 0^+$ (no disorder threshold).',
               fontsize=8.5, fontweight='bold')
ax_a.legend(fontsize=7.5, loc='upper left')
ax_a.set_ylim(-0.05, 0.55)
ax_a.set_xlim(-0.001, 0.032)

# ── Panel (b): U4(L) at fixed P values — show convergence ───────────────────
ax_b = fig.add_subplot(gs[0, 1])

PC_PLOT = [0.000, 0.002, 0.010, 0.020, 0.030]
COLORS_P = plt.cm.plasma(np.linspace(0.1, 0.9, len(PC_PLOT)))

for j, pc in enumerate(PC_PLOT):
    Ls_f = []; u4s_f = []
    for L in L_A:
        k = f'L{L}_pc{pc:.3f}'
        if k in agg_A and not math.isnan(agg_A[k]['U4']):
            Ls_f.append(L)
            u4s_f.append(agg_A[k]['U4'])
    if len(Ls_f) >= 2:
        style = '-' if pc > 0 else '--'
        ax_b.plot(Ls_f, u4s_f, 'o'+style, color=COLORS_P[j], lw=1.6, ms=5,
                  label=f'$P={pc:.3f}$')

ax_b.axhline(0, color='gray', ls='--', lw=0.8, alpha=0.5)
ax_b.set_xlabel('System size $L$', fontsize=11)
ax_b.set_ylabel('Binder cumulant $U_4$', fontsize=10)
ax_b.set_title('(b) $U_4$ vs $L$ at fixed $P_{\\rm causal}$\n'
               'For all $P > 0$: $U_4$ rises with $L$ (ordered-phase sign).\n'
               'At $P=0$: $U_4$ flat/noisy (disordered).',
               fontsize=8.5, fontweight='bold')
ax_b.legend(fontsize=7.5, loc='upper left')

# ── Panel (c): Binder crossing summary + p_c estimates ───────────────────────
ax_c = fig.add_subplot(gs[0, 2])

# Show U4 at P=0 vs P=0.030 for each L side-by-side
xs_disordered = []; ys_disordered = []
xs_ordered    = []; ys_ordered    = []
for L in L_A:
    k0 = f'L{L}_pc0.000'; k3 = f'L{L}_pc0.030'
    if k0 in agg_A: xs_disordered.append(L); ys_disordered.append(agg_A[k0]['U4'])
    if k3 in agg_A: xs_ordered.append(L);    ys_ordered.append(agg_A[k3]['U4'])

ax_c.plot(xs_disordered, ys_disordered, 's--', color='steelblue', lw=1.8, ms=7,
          label='$P=0.000$ (noise only)')
ax_c.plot(xs_ordered,    ys_ordered,    'o-',  color='darkorange', lw=1.8, ms=7,
          label='$P=0.030$ (ordered)')

# Mark p_c estimates
if pc_ests:
    for est in pc_ests:
        ax_c.axhline(0, color='gray', ls=':', lw=0.5)
    # Show mean p_c as text
    ax_c.text(0.05, 0.92,
              f'$p_c$ estimates from crossings:\n' +
              '\n'.join([f'  {e:.4f}' for e in pc_ests]) +
              f'\nMean: {np.mean(pc_ests):.4f} $\\pm$ {np.std(pc_ests):.4f}',
              transform=ax_c.transAxes, fontsize=7.5,
              va='top', bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

ax_c.axhline(0, color='gray', ls='--', lw=0.8, alpha=0.5)
ax_c.set_xlabel('System size $L$', fontsize=11)
ax_c.set_ylabel('Binder cumulant $U_4$', fontsize=10)
ax_c.set_title('(c) $U_4$ at $P=0$ vs $P=0.030$\n'
               '$P=0$: flat/noisy (disorder). $P=0.030$: rising (order).\n'
               r'$p_c$ crossing estimates all $< 0.010 \to p_c \approx 0^+$.',
               fontsize=8.5, fontweight='bold')
ax_c.legend(fontsize=7.5, loc='lower right')

fig.suptitle(
    f'Paper 64: BKT Test -- Is $p_c = 0$?\n'
    f'Phase A (large-$L$ Binder, $L \\in {{40,..,120}}$, 12 seeds, 12000 steps):\n'
    f'$U_4(L=120) > U_4(L=40)$ for ALL $P \\geq 0.002$. '
    f'Crossing estimates: mean $p_c = {np.mean(pc_ests):.4f} \\pm {np.std(pc_ests):.4f}$. '
    r'Evidence: $p_c = 0^+$ (BKT-like, no disorder threshold). '
    r'Phase B: spatial correlator inconclusive (low $\phi$ amplitude).',
    fontsize=8.5, fontweight='bold', y=1.04
)

out = Path(__file__).parent / 'paper64_figure1.pdf'
plt.savefig(str(out), bbox_inches='tight')
print(f'Saved: {out}')
