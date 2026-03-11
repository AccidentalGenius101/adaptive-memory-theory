"""Paper 65 figure: spin correlator, extended Binder, VCSM ablation."""
import json, numpy as np, math, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path

with open(Path(__file__).parent / 'results/paper65_analysis.json') as f:
    d = json.load(f)

agg_A1 = d['A1']
agg_A2 = d['A2']
agg_C  = d['C']

L_A1  = [60, 80, 100]
PC_A1 = [0.000, 0.005, 0.010, 0.020, 0.050]
L_A2  = [120, 160]
PC_A2 = [0.0000, 0.0005, 0.0010, 0.0020, 0.0050]
CONDS = ['Ref', 'NoGate', 'NoBaseline', 'Crystallized', 'NoCausal']

COLORS_L  = plt.cm.viridis(np.linspace(0.1, 0.9, max(len(L_A1), len(L_A2))))
COLORS_C  = plt.cm.tab10(np.linspace(0, 0.6, len(CONDS)))

fig = plt.figure(figsize=(17, 10.5))
gs  = gridspec.GridSpec(2, 3, figure=fig, wspace=0.40, hspace=0.52)

# ── Panel (a): U4 vs P for Phase A1 (spin-side) ──────────────────────────────
ax_a = fig.add_subplot(gs[0, 0])
for i, L in enumerate(L_A1):
    pcs = []; u4s = []
    for pc in PC_A1:
        k = f'L{L}_pc{pc:.4f}'
        if k in agg_A1 and not math.isnan(agg_A1[k]['U4']):
            pcs.append(pc); u4s.append(agg_A1[k]['U4'])
    ax_a.plot(pcs, u4s, 'o-', color=COLORS_L[i], lw=1.8, ms=5, label=f'$L={L}$')

ax_a.axhline(0, color='gray', ls='--', lw=0.8, alpha=0.5)
ax_a.set_xlabel('$P_{\\rm causal}$', fontsize=11)
ax_a.set_ylabel('Binder cumulant $U_4$', fontsize=10)
ax_a.set_title('(a) Phase A1: $U_4$ vs $P_{\\rm causal}$\n'
               '$L \\in \\{60,80,100\\}$. Ordering signal present but\n'
               'spin-spin correlator $G_s(r)$ is P-independent.',
               fontsize=8.5, fontweight='bold')
ax_a.legend(fontsize=8)

# ── Panel (b): spin G_norm(r) for L=80, several P ────────────────────────────
ax_b = fig.add_subplot(gs[0, 1])
COLORS_P = plt.cm.plasma(np.linspace(0.1, 0.9, len(PC_A1)))
r_arr = np.arange(1, 21)
plotted_any = False
for j, pc in enumerate(PC_A1):
    k = f'L80_pc{pc:.4f}'
    if k not in agg_A1: continue
    G = agg_A1[k].get('G_norm', [])
    if not G or len(G) < 5: continue
    G_arr = np.array(G[:20])
    r_plot = r_arr[:len(G_arr)]
    ax_b.semilogy(r_plot, np.maximum(G_arr, 1e-6), 'o-', color=COLORS_P[j],
                  lw=1.5, ms=4, label=f'$P={pc:.3f}$', alpha=0.85)
    plotted_any = True

# Overlay fits
k80 = 'L80_pc0.0200'
if k80 in agg_A1:
    xi    = agg_A1[k80].get('xi', float('nan'))
    eta   = agg_A1[k80].get('eta', float('nan'))
    r_fit = np.arange(1, 21)
    if not math.isnan(xi) and xi > 0:
        ax_b.semilogy(r_fit, np.exp(-r_fit / xi), 'k--', lw=1.5,
                      label=f'Exp fit $\\xi={xi:.2f}$', alpha=0.7)
    if not math.isnan(eta):
        ax_b.semilogy(r_fit, r_fit ** (-eta), 'k:', lw=1.5,
                      label=f'Power law $\\eta={eta:.2f}$', alpha=0.7)

ax_b.set_xlabel('$r$ (lattice spacings)', fontsize=11)
ax_b.set_ylabel('$G_s(r) / G_s(0)$', fontsize=10)
ax_b.set_title('(b) Spin correlator $G_s(r)$, $L=80$\n'
               'Exponential fit wins ($R^2=0.996$) over power law ($0.936$).\n'
               'Ising paramagnetic decay, $\\xi \\approx 1.1$ -- P-independent.',
               fontsize=8.5, fontweight='bold')
if plotted_any:
    ax_b.legend(fontsize=7, loc='upper right')

# ── Panel (c): Extended Binder Phase A2 (L=120 vs L=160) ─────────────────────
ax_c = fig.add_subplot(gs[0, 2])
L2_colors = ['royalblue', 'firebrick']
for i, L in enumerate(L_A2):
    pcs = []; u4s = []
    for pc in PC_A2:
        k = f'L{L}_pc{pc:.4f}'
        if k in agg_A2 and not math.isnan(agg_A2[k]['U4']):
            pcs.append(pc); u4s.append(agg_A2[k]['U4'])
    ax_c.plot(pcs, u4s, 'o-', color=L2_colors[i], lw=2, ms=6, label=f'$L={L}$')

ax_c.axhline(0, color='gray', ls='--', lw=0.8, alpha=0.5)
ax_c.set_xlabel('$P_{\\rm causal}$', fontsize=11)
ax_c.set_ylabel('Binder cumulant $U_4$', fontsize=10)
ax_c.set_title('(c) Phase A2: Extended Binder $L \\in \\{120,160\\}$\n'
               '$U_4(160) < U_4(120)$ for all $P \\leq 0.005$.\n'
               'Reversal: $p_c = 0^+$ not confirmed at $L=160$.',
               fontsize=8.5, fontweight='bold')
ax_c.legend(fontsize=9)

# ── Panel (d): K_phi vs L (stiffness) ────────────────────────────────────────
ax_d = fig.add_subplot(gs[1, 0])
PC_K = [0.000, 0.010, 0.020, 0.050]
CK   = plt.cm.viridis(np.linspace(0.1, 0.9, len(PC_K)))
for j, pc in enumerate(PC_K):
    Ls_f = []; Ks = []
    for L in L_A1:
        k = f'L{L}_pc{pc:.4f}'
        if k in agg_A1 and not math.isnan(agg_A1[k]['K_phi']):
            Ls_f.append(L); Ks.append(agg_A1[k]['K_phi'])
    if len(Ls_f) >= 2:
        ax_d.plot(Ls_f, Ks, 'o-', color=CK[j], lw=1.6, ms=5, label=f'$P={pc:.3f}$')

# Fit K ~ L^alpha
all_Ls = []; all_Ks = []
for L in L_A1:
    k = f'L{L}_pc0.0200'
    if k in agg_A1 and not math.isnan(agg_A1[k]['K_phi']):
        all_Ls.append(L); all_Ks.append(agg_A1[k]['K_phi'])
if len(all_Ls) >= 2:
    p = np.polyfit(np.log(all_Ls), np.log(all_Ks), 1)
    L_fine = np.array([55., 110.])
    ax_d.plot(L_fine, np.exp(np.polyval(p, np.log(L_fine))), 'k--',
              lw=1.3, alpha=0.7, label=f'$\\propto L^{{{p[0]:.2f}}}$')

ax_d.set_xlabel('System size $L$', fontsize=11)
ax_d.set_ylabel('$K_\\phi = \\mathrm{Var}(\\phi_{\\rm row}) \\cdot L^2 / \\mathrm{Var}(\\phi)$', fontsize=9)
ax_d.set_title('(d) $\\phi$-field stiffness $K_\\phi$ vs $L$\n'
               '$K_\\phi \\propto L^1$ (wave geometry, not BKT).\n'
               'P-independent: not a phase-sensitive observable.',
               fontsize=8.5, fontweight='bold')
ax_d.legend(fontsize=7)

# ── Panel (e): Phase C ablation U4 bar chart ─────────────────────────────────
ax_e = fig.add_subplot(gs[1, 1])
cond_labels = {'Ref': 'Ref\n(full)', 'NoGate': 'NoGate\n(SS=0)',
               'NoBaseline': 'NoBase\n(β=0)', 'Crystallized': 'Crystal\n(FD=1)',
               'NoCausal': 'NoP\n(P=0)'}
xs = np.arange(len(CONDS)); bar_vals = []; bar_errs = []
for cn in CONDS:
    if cn in agg_C:
        bar_vals.append(agg_C[cn]['U4'])
        bar_errs.append(agg_C[cn].get('U4_se', 0.))
    else:
        bar_vals.append(0.); bar_errs.append(0.)

bars = ax_e.bar(xs, bar_vals, color=COLORS_C, alpha=0.85,
                yerr=bar_errs, capsize=4, error_kw={'lw': 1.5})
ax_e.axhline(bar_vals[0], color='black', ls='--', lw=1, alpha=0.5)  # Ref line
ax_e.set_xticks(xs)
ax_e.set_xticklabels([cond_labels.get(c, c) for c in CONDS], fontsize=8)
ax_e.set_ylabel('Binder cumulant $U_4$', fontsize=10)
ax_e.set_title('(e) Phase C: VCSM-lite Ablation ($L=80$, $P=0.020$)\n'
               'NoGate (SS=0): kills ordering ($U_4 = 0.09$). Gate is load-bearing.\n'
               'Crystallized (FD=1): amplifies $U_4 = 0.26$ -- cockroach artifact.',
               fontsize=8.5, fontweight='bold')

# ── Panel (f): Ablation |M| bar chart ────────────────────────────────────────
ax_f = fig.add_subplot(gs[1, 2])
absM_vals = []
for cn in CONDS:
    absM_vals.append(agg_C[cn]['absM'] if cn in agg_C else 0.)

ax_f.bar(xs, absM_vals, color=COLORS_C, alpha=0.85)
ax_f.axhline(absM_vals[0], color='black', ls='--', lw=1, alpha=0.5)
ax_f.set_xticks(xs)
ax_f.set_xticklabels([cond_labels.get(c, c) for c in CONDS], fontsize=8)
ax_f.set_ylabel('$|\\bar{M}|$ (zone order parameter)', fontsize=10)
ax_f.set_title('(f) Ablation: $|\\bar{M}|$ across conditions\n'
               'NoGate has highest $|\\bar{M}|$ but lowest $U_4$ -- noisy average.\n'
               'Crystallized: large $|\\bar{M}|$ from uncontrolled accumulation.',
               fontsize=8.5, fontweight='bold')

fig.suptitle(
    'Paper 65: Algebraic Order, Spin Correlator, and VCSM Ablation\n'
    'Phase A1: Ising spin correlator $G_s(r)$ is P-independent (paramagnetic Ising, $\\xi \\approx 1.1$); '
    'exponential fit wins ($R^2=0.996$ vs $0.936$). '
    r'$\phi$-field ordering does not couple back to spins. '
    'Phase A2: $U_4(L=160) < U_4(L=120)$ for all $P \\leq 0.005$ -- '
    '$p_c = 0^+$ not confirmed at $L=160$ (finite $p_c$ or equilibration artifact).\n'
    'Phase C: viability gate (SS) is load-bearing; FIELD\\_DECAY=1 (cockroach) amplifies but does not create real ordering.',
    fontsize=8.5, fontweight='bold', y=1.02
)

out = Path(__file__).parent / 'paper65_figure1.pdf'
plt.savefig(str(out), bbox_inches='tight')
print(f'Saved: {out}')
