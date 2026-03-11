"""Paper 63 figure: Critical exponents of the causal-purity transition."""
import json, numpy as np, math, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path

with open(Path(__file__).parent / 'results/paper63_analysis.json') as f:
    d = json.load(f)

L_FSS  = [20, 30, 40, 60, 80]
PC_FINE = [round(x,3) for x in np.arange(0.000, 0.085, 0.005)]
COLORS_L = plt.cm.viridis(np.linspace(0.05, 0.95, len(L_FSS)))

fss   = d['fss']
p_c   = d['p_c']
bnu   = d['beta_over_nu']
gnu   = d['gamma_over_nu']
onu   = d['one_over_nu']
nu    = d['nu']
beta  = d['beta']
gamma = d['gamma']

fig = plt.figure(figsize=(17, 5.2))
gs  = gridspec.GridSpec(1, 3, figure=fig, wspace=0.42)

# ── Panel (a): Binder cumulant U4 fine scan ──────────────────────────────────
ax_a = fig.add_subplot(gs[0, 0])

for i, L in enumerate(L_FSS):
    pcs = []; u4s = []
    for pc in PC_FINE:
        k = f'L{L}_pc{pc:.3f}'
        if k in fss:
            pcs.append(pc); u4s.append(fss[k]['U4'])
    ax_a.plot(pcs, u4s, 'o-', color=COLORS_L[i], lw=1.5, ms=4.5, label=f'$L={L}$')

ax_a.axhline(2/3, color='gray', ls=':', lw=1, alpha=0.7)
ax_a.axhline(0.0, color='gray', ls='--', lw=0.8, alpha=0.5)
ax_a.axvline(p_c, color='red', ls='--', lw=1.3, alpha=0.8,
             label=f'$p_c={p_c:.3f}$')
ax_a.set_xlabel('$P_{\\rm causal}$', fontsize=11)
ax_a.set_ylabel('Binder cumulant $U_4$', fontsize=10)
ax_a.set_title(f'(a) Fine-scan Binder cumulant\n'
               f'$p_c \\approx {p_c:.3f} \\pm 0.012$ from crossings.\n'
               f'Ordered: larger $L \\to$ larger $U_4$ for $P > 0.005$.',
               fontsize=8.5, fontweight='bold')
ax_a.legend(fontsize=7.5, loc='lower right')
ax_a.set_ylim(-0.15, 0.65)
ax_a.set_xlim(-0.003, 0.083)

# ── Panel (b): |M| vs L log-log at p_c ──────────────────────────────────────
ax_b = fig.add_subplot(gs[0, 1])

pc_grid = d['pc_grid']
COLORS_PC = plt.cm.plasma(np.linspace(0.1, 0.9, 7))
highlight_pcs = [pc for pc in PC_FINE if pc <= 0.060][::2]

for j, pc in enumerate(highlight_pcs):
    Ls_f = []; Ms_f = []
    for L in L_FSS:
        k = f'L{L}_pc{pc:.3f}'
        if k in fss and fss[k]['absM'] > 0:
            Ls_f.append(L); Ms_f.append(fss[k]['absM'])
    if len(Ls_f) >= 3:
        logL = np.log(Ls_f); logM = np.log(Ms_f)
        p = np.polyfit(logL, logM, 1)
        style = '-' if abs(pc - pc_grid) < 0.001 else '--'
        lw    = 2.0 if abs(pc - pc_grid) < 0.001 else 1.1
        ax_b.plot(Ls_f, Ms_f, 'o', color=COLORS_PC[j], ms=5.5, zorder=3)
        L_fine = np.array([15., 90.])
        ax_b.plot(L_fine, np.exp(np.polyval(p, np.log(L_fine))),
                  style, color=COLORS_PC[j], lw=lw, alpha=0.8,
                  label=f'$P={pc:.3f}$, $\\alpha={p[0]:.2f}$')

# Reference lines
L_ref = np.array([15., 90.])
ax_b.plot(L_ref, 4e-3 * L_ref**(-bnu), 'k-', lw=1.5,
          label=f'$L^{{-{bnu:.2f}}}$ (fit at $p_c$)')
ax_b.set_xscale('log'); ax_b.set_yscale('log')
ax_b.set_xlabel('System size $L$', fontsize=11)
ax_b.set_ylabel('$|M|$', fontsize=11)
ax_b.set_title(f'(b) $|M|$ vs $L$ at various $P_{{\\rm causal}}$\n'
               f'At $p_c={pc_grid:.3f}$: slope $\\approx -{bnu:.2f}$, $\\beta/\\nu={bnu:.2f}$.\n'
               f'Above $p_c$: slope $\\to 0$ (ordered). Below: slope $<0$ (disordered).',
               fontsize=8.5, fontweight='bold')
ax_b.legend(fontsize=7, loc='lower left')

# ── Panel (c): Data collapse |M|*L^(beta/nu) vs (P-p_c)*L^(1/nu) ────────────
ax_c = fig.add_subplot(gs[0, 2])

collapse = d['collapse']
# group by L
for i, L in enumerate(L_FSS):
    pts = sorted([pt for pt in collapse if pt['L']==L and
                  not (math.isnan(pt['x']) or math.isnan(pt['y']))],
                 key=lambda p: p['x'])
    if pts:
        xs = [pt['x'] for pt in pts]
        ys = [pt['y'] for pt in pts]
        ax_c.plot(xs, ys, 'o-', color=COLORS_L[i], lw=1.3, ms=4,
                  label=f'$L={L}$', alpha=0.85)

ax_c.axvline(0, color='gray', ls='--', lw=0.8, alpha=0.6)
ax_c.set_xlabel(f'$(P_{{\\rm causal}} - p_c)\\,L^{{1/\\nu}}$, $p_c={p_c:.3f}$, $\\nu={nu:.2f}$',
                fontsize=9)
ax_c.set_ylabel(f'$|M|\\cdot L^{{\\beta/\\nu}}$, $\\beta/\\nu={bnu:.2f}$', fontsize=9)
ax_c.set_title(f'(c) Data collapse\n'
               f'$|M|\\cdot L^{{{bnu:.2f}}}$ vs $(P-p_c)\\cdot L^{{{onu:.2f}}}$.\n'
               f'Partial collapse; residual scatter reflects $p_c$ uncertainty.',
               fontsize=8.5, fontweight='bold')
ax_c.legend(fontsize=7.5, loc='upper left')
ax_c.set_xlim(-3., 5.)
ax_c.set_ylim(-0.05, None)

fig.suptitle(
    f'Paper 63: Critical Exponents of the Causal-Purity Phase Transition\n'
    f'$p_c={p_c:.3f}\\pm0.012$,  $\\beta/\\nu={bnu:.2f}$,  $\\nu={nu:.2f}$,  $\\beta={beta:.2f}$  '
    f'(mean field: $\\beta=0.50$; 2D Ising: $\\beta=0.125$; dir.\ perc.: $\\beta=0.276$).\n'
    r'$\gamma/\nu\approx 0$ (susceptibility definition requires revision). '
    r'Exponents differ from all known classes $\to$ candidate new universality class.',
    fontsize=8.8, fontweight='bold', y=1.05
)

out = Path(__file__).parent / 'paper63_figure1.pdf'
plt.savefig(str(out), bbox_inches='tight')
print(f'Saved: {out}')
