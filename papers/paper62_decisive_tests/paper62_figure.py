"""Paper 62 figure: Three decisive tests."""
import json, numpy as np, math, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path

with open(Path(__file__).parent / 'results/paper62_analysis.json') as f:
    d = json.load(f)

L_FSS  = [20, 30, 40, 60]
PC_FSS = [round(x, 2) for x in np.arange(0.00, 0.65, 0.05)]
COLORS_L  = plt.cm.viridis(np.linspace(0.1, 0.9, len(L_FSS)))
COLORS_K  = plt.cm.plasma(np.linspace(0.1, 0.9, 4))

fss_A = d['A']

fig = plt.figure(figsize=(17, 5.0))
gs  = gridspec.GridSpec(1, 3, figure=fig, wspace=0.40)

# ── Panel (a): Binder cumulant U4 — Phase A (extensive-drive) ─────────────────
ax_a = fig.add_subplot(gs[0, 0])

for i, L in enumerate(L_FSS):
    pcs = []; u4s = []
    for pc in PC_FSS:
        k = f'L{L}_pc{pc:.2f}'
        if k in fss_A:
            pcs.append(pc); u4s.append(fss_A[k]['U4'])
    ax_a.plot(pcs, u4s, 'o-', color=COLORS_L[i], lw=1.6, ms=5, label=f'$L={L}$')

ax_a.axhline(2/3, color='gray', ls=':', lw=1, alpha=0.7, label='$U_4=2/3$ (ordered)')
ax_a.axhline(0.0, color='gray', ls='--', lw=0.8, alpha=0.5)
# Annotate p_c region
ax_a.axvspan(-0.01, 0.08, alpha=0.12, color='red', label='$p_c \\approx 0.02$–$0.05$')
ax_a.set_xlabel('$P_{\\rm causal}$', fontsize=11)
ax_a.set_ylabel('Binder cumulant $U_4$', fontsize=10)
ax_a.set_title('(a) Phase A: Extensive-drive FSS\n'
               '$U_4$ ordered by $L$ for $P>0.05$: larger $L\\to$ larger $U_4$.\n'
               '$p_c\\approx0.02$–$0.05$. True thermodynamic transition.',
               fontsize=8.5, fontweight='bold')
ax_a.legend(fontsize=7, loc='lower right')
ax_a.set_ylim(-0.05, 0.75); ax_a.set_xlim(-0.02, 0.65)

# ── Panel (b): |M| vs L log-log — now SIZE-INDEPENDENT (Phase A) ─────────────
ax_b = fig.add_subplot(gs[0, 1])

COLORS_PC = plt.cm.plasma(np.linspace(0.1, 0.9, 6))
highlight_pcs = [0.00, 0.05, 0.10, 0.20, 0.40, 0.60]

for j, pc in enumerate(highlight_pcs):
    Ls_fit = []; Ms_fit = []
    for L in L_FSS:
        k = f'L{L}_pc{pc:.2f}'
        if k in fss_A:
            Ls_fit.append(L); Ms_fit.append(fss_A[k]['absM'])
    if len(Ls_fit) >= 3:
        logL = np.log(Ls_fit)
        logM = np.log([max(m, 1e-8) for m in Ms_fit])
        p = np.polyfit(logL, logM, 1)
        ax_b.plot(Ls_fit, Ms_fit, 'o', color=COLORS_PC[j], ms=6, zorder=3)
        L_fine = np.array([15., 70.])
        ax_b.plot(L_fine, np.exp(np.polyval(p, np.log(L_fine))),
                  '-', color=COLORS_PC[j], lw=1.4, alpha=0.7,
                  label=f'$P={pc:.2f}$, $\\alpha={p[0]:.2f}$')

ax_b.set_xscale('log'); ax_b.set_yscale('log')
ax_b.set_xlabel('System size $L$', fontsize=11)
ax_b.set_ylabel('$|M|$', fontsize=11)
ax_b.set_title('(b) Phase A: $|M|$ vs $L$ (log-log)\n'
               'Slope $\\approx 0$ for $P\\geq 0.10$: size-independent $|M|$.\n'
               'True long-range order. Compare Paper 61: slope $\\approx -2$.',
               fontsize=8.5, fontweight='bold')
ax_b.legend(fontsize=7, loc='lower left')

# ── Panel (c): Phase B relay — S_phi_01 and S_phi_far vs D_phi ───────────────
ax_c = fig.add_subplot(gs[0, 2])

relay_B = d['B']
D_PHI   = [0.0, 0.02, 0.10]
K_ZONES = [2, 4, 6, 8]
pc_show = 0.60

for i, K in enumerate(K_ZONES):
    s01s = []; sfar_s = []; ds = []
    for D in D_PHI:
        k = f'K{K}_D{D:.2f}_pc{pc_show:.2f}'
        if k in relay_B:
            s01s.append(relay_B[k]['S_phi_01'])
            sfar_s.append(relay_B[k]['S_phi_far'])
            ds.append(D)
    if s01s:
        ax_c.plot(ds, s01s, 'o-', color=COLORS_K[i], lw=1.5, ms=6,
                  label=f'$K={K}$, $S_{{01}}$ (source pair)')
        ax_c.plot(ds, sfar_s, 's--', color=COLORS_K[i], lw=1.2, ms=5, alpha=0.6)

ax_c.set_xlabel('Phi-diffusion $D_\\phi$', fontsize=11)
ax_c.set_ylabel('$S_\\phi$', fontsize=11)
ax_c.set_title('(c) Phase B: Multi-zone relay ($P=0.60$)\n'
               'Solid=source pair ($S_{01}$), dashed=far pair ($S_{\\rm far}$).\n'
               'Diffusion boosts local order; far relay limited by zone driving.',
               fontsize=8.5, fontweight='bold')
ax_c.legend(fontsize=7.5, loc='upper left')
ax_c.set_xlim(-0.01, 0.12)

# Summary of Phase A slopes
slopes_A = {}
for pc in PC_FSS:
    vals = [(L, fss_A[f'L{L}_pc{pc:.2f}']['absM'])
            for L in L_FSS if f'L{L}_pc{pc:.2f}' in fss_A]
    if len(vals) >= 3:
        lL = np.log([v[0] for v in vals])
        lM = np.log([max(v[1], 1e-8) for v in vals])
        p = np.polyfit(lL, lM, 1)
        slopes_A[pc] = p[0]
slope_ordered = np.mean([slopes_A[pc] for pc in [0.20, 0.30, 0.40, 0.50, 0.60]])

fig.suptitle(
    'Paper 62: Three Decisive Tests — Extensive-Drive FSS, Multi-Zone Relay, VCML Ablation\n'
    f'Phase A key result: $|M|\\propto L^{{{slope_ordered:.2f}}}$ for $P>0.10$ (slope$\\approx 0$): '
    'true long-range order confirmed. '
    '$p_c\\approx0.02$–$0.05$ (much lower than Paper 60 appearance). '
    'Binder cumulant correctly ordered by $L$ for $P>0.05$. '
    'Causal-purity ordered phase is a genuine thermodynamic phase.',
    fontsize=9, fontweight='bold', y=1.04
)

out = Path(__file__).parent / 'paper62_figure1.pdf'
plt.savefig(str(out), bbox_inches='tight')
print(f'Saved: {out}')
