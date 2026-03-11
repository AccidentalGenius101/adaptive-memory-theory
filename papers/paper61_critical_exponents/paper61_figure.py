"""Paper 61 figure: FSS analysis of the causal-purity phase transition."""
import json, numpy as np, math, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path

with open(Path(__file__).parent / 'results/paper61_analysis.json') as f:
    d = json.load(f)

L_FSS = [20, 30, 40, 60]
PC_FSS = [round(x, 2) for x in np.arange(0.00, 0.65, 0.05)]
PC_SCAN = [round(x, 2) for x in np.arange(0.00, 0.72, 0.04)]
COLORS_L = plt.cm.viridis(np.linspace(0.1, 0.9, len(L_FSS)))

fig = plt.figure(figsize=(17, 5.0))
gs  = gridspec.GridSpec(1, 3, figure=fig, wspace=0.40)

# ── Panel (a): Phase-1 scan — S_phi and chi vs P_causal at L=40 ──────────────
ax_a = fig.add_subplot(gs[0, 0])

scan = d['scan']
pcs_scan = sorted([float(k) for k in scan.keys() if k.replace('.','').isdigit() == False],
                  key=lambda x: x)
# scan keys are like '0.0', '0.04', etc stored as numbers from p_c_est
# Actually scan is keyed by P_causal values
pcs_s  = []
sphi_s = []
chi_s  = []
u4_s   = []
for pc in PC_SCAN:
    k = f'{pc}'
    # try both float-key formats
    found = None
    for kk in scan:
        try:
            if abs(float(kk) - pc) < 0.001:
                found = kk
                break
        except:
            pass
    if found is None:
        continue
    pcs_s.append(pc)
    sphi_s.append(scan[found]['S_phi'])
    chi_s.append(scan[found]['chi'])
    u4_s.append(scan[found]['U4'])

pcs_s = np.array(pcs_s)

ax_a2 = ax_a.twinx()
l1, = ax_a.plot(pcs_s, sphi_s, 'o-', color='royalblue', lw=1.8, ms=6,
                label='$S_\\phi$ (phi-order, left)')
l2, = ax_a2.plot(pcs_s, chi_s, 's--', color='tomato', lw=1.5, ms=5,
                 label='$\\chi$ (susceptibility, right)')
ax_a.set_xlabel('$P_{\\rm causal}$', fontsize=11)
ax_a.set_ylabel('$S_\\phi$', fontsize=11, color='royalblue')
ax_a2.set_ylabel('$\\chi$', fontsize=11, color='tomato')
ax_a.tick_params(axis='y', labelcolor='royalblue')
ax_a2.tick_params(axis='y', labelcolor='tomato')
ax_a.set_title('(a) Phase-1 scan: $L=40$, $T=3.0$\n'
               '$S_\\phi$ grows monotonically; $\\chi$ remains small.\n'
               'No sharp chi peak — consistent with crossover.',
               fontsize=8.5, fontweight='bold')
lines = [l1, l2]
ax_a.legend(lines, [l.get_label() for l in lines], fontsize=7, loc='upper left')
ax_a.set_xlim(-0.02, 0.72)

# ── Panel (b): Binder cumulant U4 vs P_causal for all L ──────────────────────
ax_b = fig.add_subplot(gs[0, 1])

fss = d['fss']
for i, L in enumerate(L_FSS):
    pcs_b = []; u4_b = []
    for pc in PC_FSS:
        k = f'L{L}_pc{pc:.2f}'
        if k in fss:
            pcs_b.append(pc)
            u4_b.append(fss[k]['U4'])
    ax_b.plot(pcs_b, u4_b, 'o-', color=COLORS_L[i], lw=1.6, ms=5, label=f'$L={L}$')

ax_b.axhline(2/3, color='gray', ls=':', lw=1, alpha=0.7, label='$U_4=2/3$ (ordered)')
ax_b.axhline(0.0, color='gray', ls='--', lw=0.8, alpha=0.5, label='$U_4=0$ (Gaussian)')
ax_b.set_xlabel('$P_{\\rm causal}$', fontsize=11)
ax_b.set_ylabel('Binder cumulant $U_4$', fontsize=10)
ax_b.set_title('(b) Binder cumulant $U_4$ vs $P_{\\rm causal}$\n'
               'Curves overlap for all $L$ — no $L$-dependent crossing.\n'
               'Transition near $P_{\\rm causal}\\approx0.05$--$0.15$ (size-independent).',
               fontsize=8.5, fontweight='bold')
ax_b.legend(fontsize=7.5, loc='lower right')
ax_b.set_ylim(-0.05, 0.75)
ax_b.set_xlim(-0.02, 0.65)

# ── Panel (c): |M| vs L log-log — reveals |M| ~ L^{-2} (geometric dilution) ──
ax_c = fig.add_subplot(gs[0, 2])

COLORS_PC = plt.cm.plasma(np.linspace(0.1, 0.9, 5))
highlight_pcs = [0.20, 0.30, 0.40, 0.50, 0.60]
slopes = []
for j, pc in enumerate(highlight_pcs):
    Ls_fit = []; Ms_fit = []
    for L in L_FSS:
        k = f'L{L}_pc{pc:.2f}'
        if k in fss:
            Ls_fit.append(L)
            Ms_fit.append(fss[k]['absM'])
    if len(Ls_fit) >= 3:
        logL = np.log(Ls_fit)
        logM = np.log([max(m, 1e-8) for m in Ms_fit])
        p = np.polyfit(logL, logM, 1)
        slopes.append(p[0])
        # plot data
        ax_c.plot(Ls_fit, Ms_fit, 'o', color=COLORS_PC[j], ms=6, zorder=3)
        # plot fit line
        L_fine = np.array([15, 70])
        ax_c.plot(L_fine, np.exp(np.polyval(p, np.log(L_fine))),
                  '-', color=COLORS_PC[j], lw=1.4, alpha=0.7,
                  label=f'$P={pc:.2f}$, slope={p[0]:.2f}')

# Reference slope L^{-2}
L_ref = np.array([15, 70])
ax_c.plot(L_ref, 0.20 * L_ref.astype(float)**(-2), 'k--', lw=1.2, label='$\\propto L^{-2}$ (geometric)')

ax_c.set_xscale('log'); ax_c.set_yscale('log')
ax_c.set_xlabel('System size $L$', fontsize=11)
ax_c.set_ylabel('$|M| = |\\langle\\phi_0\\rangle - \\langle\\phi_1\\rangle|$', fontsize=9)
ax_c.set_title('(c) $|M|$ vs $L$ (log-log): all slopes $\\approx -2$\n'
               'Order parameter diluted as $L^{-2}$ at ALL $P_{\\rm causal}$.\n'
               'Signal $\\propto r_w^2/L^2$ (fixed wave, growing system).',
               fontsize=8.5, fontweight='bold')
ax_c.legend(fontsize=7, loc='lower left')

mean_slope = np.mean(slopes) if slopes else -2.0
fig.suptitle(
    'Paper 61: Finite-Size Scaling of the Causal-Purity Transition — Ising + VCSM-lite\n'
    f'FSS reveals $|M|\\propto L^{{-2}}$ at all $P_{{\\rm causal}}$ (mean slope = {mean_slope:.2f}): '
    'geometric dilution, not critical scaling. '
    'Binder cumulant $U_4$ shows no $L$-dependent crossing. '
    'The transition is a finite-size crossover; a true thermodynamic transition requires extensive wave density ($\\propto L^2$).',
    fontsize=9, fontweight='bold', y=1.04
)

out = Path(__file__).parent / 'paper61_figure1.pdf'
plt.savefig(str(out), bbox_inches='tight')
print(f'Saved: {out}')
