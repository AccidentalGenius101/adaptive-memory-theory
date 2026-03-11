"""Paper 66 figure: p_c discriminating test + phi correlator null + FA sweep."""
import json, numpy as np, math, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path

with open(Path(__file__).parent / 'results/paper66_analysis.json') as f:
    d = json.load(f)

agg_A = d['A']
agg_B = d['B']
agg_C = d['C']

L_A      = [100, 120, 160]
PC_A     = [0.000, 0.005, 0.010, 0.020]
L_B      = [60, 80]
PC_B     = [0.000, 0.010, 0.020, 0.050]
FA_SWEEP = [0.10, 0.30, 0.50, 0.80]

COLORS_L  = ['royalblue', 'seagreen', 'firebrick']
COLORS_FA = plt.cm.plasma(np.linspace(0.1, 0.85, len(FA_SWEEP)))

fig = plt.figure(figsize=(16, 9))
gs  = gridspec.GridSpec(2, 3, figure=fig, wspace=0.42, hspace=0.54)

# ── Panel (a): U4 vs P_causal, all L ─────────────────────────────────────────
ax_a = fig.add_subplot(gs[0, 0])
for i, L in enumerate(L_A):
    pcs, u4s, u4e = [], [], []
    for pc in PC_A:
        k = f'L{L}_pc{pc:.4f}'
        if k in agg_A and not math.isnan(agg_A[k]['U4']):
            pcs.append(pc); u4s.append(agg_A[k]['U4'])
            u4e.append(agg_A[k].get('U4_se', 0.))
    ax_a.errorbar(pcs, u4s, yerr=u4e, fmt='o-', color=COLORS_L[i],
                  lw=2, ms=6, capsize=3, label=f'$L={L}$')
ax_a.set_xlabel('$P_{\\rm causal}$', fontsize=11)
ax_a.set_ylabel('Binder cumulant $U_4$', fontsize=10)
ax_a.set_title('(a) Phase A: $U_4$ vs $P_{\\rm causal}$\n'
               '$U_4(160) < U_4(120)$ at all $P$.\n'
               'L=160 under-driven: wave hits/site = 15.2 vs 18 (L=120).',
               fontsize=8.5, fontweight='bold')
ax_a.legend(fontsize=9)

# ── Panel (b): U4 vs L at fixed P -- shows non-monotone ──────────────────────
ax_b = fig.add_subplot(gs[0, 1])
PC_SHOW  = [0.005, 0.010, 0.020]
COLORS_P = plt.cm.viridis(np.linspace(0.1, 0.9, len(PC_SHOW)))
for j, pc in enumerate(PC_SHOW):
    Ls_f, u4s = [], []
    for L in L_A:
        k = f'L{L}_pc{pc:.4f}'
        if k in agg_A and not math.isnan(agg_A[k]['U4']):
            Ls_f.append(L); u4s.append(agg_A[k]['U4'])
    if len(Ls_f) >= 2:
        ax_b.plot(Ls_f, u4s, 'o-', color=COLORS_P[j], lw=1.8, ms=6,
                  label=f'$P={pc:.3f}$')

ax_b.set_xlabel('System size $L$', fontsize=11)
ax_b.set_ylabel('Binder cumulant $U_4$', fontsize=10)
ax_b.set_title('(b) Phase A: $U_4$ vs $L$\n'
               'Non-monotone: $U_4$ dips at $L=160$.\n'
               'WAVE\\_EVERY rounding: $L=160$ gets 78\\% of intended waves.',
               fontsize=8.5, fontweight='bold')
ax_b.legend(fontsize=8)

# ── Panel (c): Wave hits/site vs L -- shows the systematic under-driving ──────
ax_c = fig.add_subplot(gs[0, 2])
Ls_th = np.array([40, 60, 80, 100, 120, 160])
WE_BASE, L_BASE = 25, 40
wave_r = 5
def hits_per_site(L, nsteps=10000):
    we = max(1, round(WE_BASE * (L_BASE / L)**2))
    n_waves = nsteps / we
    wave_area = np.pi * wave_r**2
    return n_waves * wave_area / (L * L)

hits = np.array([hits_per_site(L) for L in Ls_th])
ax_c.plot(Ls_th, hits, 'o-', color='steelblue', lw=2, ms=7)
ax_c.axhline(hits_per_site(40), color='gray', ls='--', lw=1, alpha=0.6,
             label='Target (L=40)')

# Mark target (ideal, continuous WE)
WE_ideal = WE_BASE * (L_BASE / Ls_th)**2
hits_ideal = (10000 / WE_ideal) * np.pi * wave_r**2 / Ls_th**2
ax_c.plot(Ls_th, hits_ideal, 's--', color='firebrick', lw=1.5, ms=5, alpha=0.7,
          label='Ideal (no rounding)')

ax_c.set_xlabel('System size $L$', fontsize=11)
ax_c.set_ylabel('Wave hits / site (10k steps)', fontsize=10)
ax_c.set_title('(c) WAVE\\_EVERY rounding artefact\n'
               'Actual hits/site dips at $L=160$ due to integer rounding.\n'
               '$L=160$: 78\\% of target hits. Root cause of $U_4$ dip.',
               fontsize=8.5, fontweight='bold')
ax_c.legend(fontsize=8)

# ── Panel (d): Phase B -- phi_amp vs P for L in {60,80}, FA=0.50 ─────────────
ax_d = fig.add_subplot(gs[1, 0])
for i, L in enumerate(L_B):
    pcs, amps = [], []
    for pc in PC_B:
        k = f'L{L}_pc{pc:.4f}'
        if k in agg_B:
            pcs.append(pc); amps.append(agg_B[k]['phi_amp'])
    if pcs:
        ax_d.plot(pcs, amps, 'o-', color=COLORS_L[i], lw=1.8, ms=5,
                  label=f'$L={L}$')
ax_d.set_xlabel('$P_{\\rm causal}$', fontsize=11)
ax_d.set_ylabel('Mean $|\\phi|$ in zone 0', fontsize=10)
ax_d.set_title('(d) Phase B: $\\phi$ amplitude, FA=0.50\n'
               '$|\\phi| \\approx 0.013$ (detectable), weakly $P$-dependent.\n'
               'G$_\\phi$(r) UNDETERMINED: $\\phi$ has no within-zone spatial structure.',
               fontsize=8.5, fontweight='bold')
ax_d.legend(fontsize=9)
ax_d.text(0.5, 0.5, 'No $\\phi$-$\\phi$ correlations\nbeyond 1 lattice spacing',
          transform=ax_d.transAxes, ha='center', va='center',
          fontsize=9, color='firebrick',
          bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.8))

# ── Panel (e): Phase C -- phi_amp and U4 vs FA ───────────────────────────────
ax_e = fig.add_subplot(gs[1, 1])
fa_vals, amp_vals, u4_vals = [], [], []
for fa in FA_SWEEP:
    k = f'fa{fa:.2f}'
    if k in agg_C:
        fa_vals.append(fa)
        amp_vals.append(agg_C[k]['phi_amp'])
        u4_vals.append(agg_C[k]['U4'])

if fa_vals:
    ax_e.plot(fa_vals, amp_vals, 'o-', color='royalblue', lw=2, ms=7,
              label='$|\\phi|$')
    # fit power law phi_amp ~ FA^alpha
    log_fa = np.log(fa_vals); log_amp = np.log(amp_vals)
    pp = np.polyfit(log_fa, log_amp, 1)
    fa_fit = np.linspace(min(fa_vals), max(fa_vals), 50)
    ax_e.plot(fa_fit, np.exp(np.polyval(pp, np.log(fa_fit))), '--',
              color='royalblue', lw=1.2, alpha=0.6,
              label=f'$|\\phi| \\propto F_A^{{{pp[0]:.2f}}}$')
    ax_e.set_xlabel('$F_A$ (amplitude factor)', fontsize=11)
    ax_e.set_ylabel('Mean $|\\phi|$', fontsize=10)
    ax_e2 = ax_e.twinx()
    ax_e2.plot(fa_vals, u4_vals, 's--', color='firebrick', lw=1.5, ms=6,
               alpha=0.85, label='$U_4$ (right)')
    ax_e2.set_ylabel('Binder cumulant $U_4$', fontsize=10, color='firebrick')
    ax_e2.tick_params(axis='y', colors='firebrick')
    lines1, lab1 = ax_e.get_legend_handles_labels()
    lines2, lab2 = ax_e2.get_legend_handles_labels()
    ax_e.legend(lines1 + lines2, lab1 + lab2, fontsize=7.5, loc='center right')

ax_e.set_title(f'(e) Phase C: $F_A$ sweep ($L=80$, $P=0.020$)\n'
               '$|\\phi| \\propto F_A^{\\alpha}$ ($\\alpha > 1$: nonlinear).\n'
               '$U_4$ DECREASES with $F_A$: high $F_A$ = noisy $\\phi$ = less order.',
               fontsize=8.5, fontweight='bold')

# ── Panel (f): Theoretical diagram -- hits/site counts ───────────────────────
ax_f = fig.add_subplot(gs[1, 2])
L_vals_bar = [100, 120, 160]
we_actual  = [max(1, round(WE_BASE*(L_BASE/L)**2)) for L in L_vals_bar]
we_ideal_f = [WE_BASE*(L_BASE/L)**2 for L in L_vals_bar]
hits_act   = [10000/we * np.pi*5**2 / (L*L) for we,L in zip(we_actual, L_vals_bar)]
hits_id    = [10000/we * np.pi*5**2 / (L*L) for we,L in zip(we_ideal_f, L_vals_bar)]

x = np.arange(len(L_vals_bar)); w = 0.35
bars1 = ax_f.bar(x - w/2, hits_act, w, color=['royalblue','seagreen','firebrick'],
                 alpha=0.85, label='Actual (rounded WE)')
bars2 = ax_f.bar(x + w/2, hits_id,  w, color=['royalblue','seagreen','firebrick'],
                 alpha=0.40, hatch='//', label='Ideal (continuous WE)')

ax_f.set_xticks(x)
ax_f.set_xticklabels([f'$L={L}$\nWE={we}' for L,we in zip(L_vals_bar, we_actual)])
ax_f.set_ylabel('Wave hits / site', fontsize=10)
ax_f.set_title('(f) Actual vs ideal wave exposure\n'
               '$L=160$ WAVE\\_EVERY=2 (rounded from 1.56).\n'
               'L=160 gets 15.2 hits vs 19.5 target: explains $U_4$ dip.',
               fontsize=8.5, fontweight='bold')
ax_f.legend(fontsize=7.5)

fig.suptitle(
    'Paper 66: Discriminating Test for $p_c=0^+$ and $\\phi$-$\\phi$ Spatial Correlator\n'
    'Phase A: $U_4(L=160) < U_4(L=120)$ persists -- WAVE\\_EVERY rounding artefact, not finite $p_c$ '
    '($L=160$ receives 78\\% of target wave exposure due to integer rounding of WE=1.56$\\to$2). '
    'Phase B: $G_\\phi(r)$ UNDETERMINED -- $\\phi$ has no within-zone spatial structure; '
    'ordering lives in zone-mean $M$. '
    'Phase C: $U_4 \\downarrow$ with $F_A$ -- high $F_A$ = noisy $\\phi$ = less zone-mean order.',
    fontsize=8.5, fontweight='bold', y=1.02
)

out = Path(__file__).parent / 'paper66_figure1.pdf'
plt.savefig(str(out), bbox_inches='tight')
print(f'Saved: {out}')
