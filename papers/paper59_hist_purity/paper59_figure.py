"""
Paper 59 figure: Field Equation of VCML and the Sufficiency of Two-Dimensional Purity.

Panel (a): VCML component -> Model A field equation mapping (typeset table + equation)
Panel (b): P_hist vs P_sp empirical comparison: they are nearly equal.
           Geometric P_hist prediction is severely wrong (same launch-asymmetry
           correction as P57). P_hist is NOT a distinct third dimension.
Panel (c): sigma_w/(1-P_sp) and sigma_w/(1-P_hist) for C_perturb:
           neither ratio is constant at r=8, confirming a structural anomaly
           not captured by any causal purity dimension.
"""
import json, numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path

with open(Path(__file__).parent / 'results/paper59_analysis.json') as f:
    d = json.load(f)

RWAVE = [2, 4, 8, 10]
ZONE_WIDTH = 20

def get(mode, rw, key):
    k = f'{mode}_rw{rw}'
    return d[k][key] if k in d else float('nan')

def geom_phist(r, W=ZONE_WIDTH):
    xs = np.arange(1, W)
    p = W / (W + 2*np.minimum(np.minimum(xs, W - xs), r))
    return float(np.mean(p))

psp_cp   = [get('CP',  r, 'final_p_causal') for r in RWAVE]
psp_ref  = [get('ref', r, 'final_p_causal') for r in RWAVE]
ph_cp    = [get('CP',  r, 'final_p_hist')   for r in RWAVE]
ph_ref   = [get('ref', r, 'final_p_hist')   for r in RWAVE]
ph_geom  = [geom_phist(r) for r in RWAVE]
sw_cp    = [get('CP',  r, 'final_sigma_w')  for r in RWAVE]
na_cp    = [get('CP',  r, 'final_na')       for r in RWAVE]

fig = plt.figure(figsize=(17, 5.2))
gs  = gridspec.GridSpec(1, 3, figure=fig, wspace=0.42)

rw_arr = np.array(RWAVE, float)

# ── Panel (a): VCML -> Model A mapping ────────────────────────────────────────
ax_a = fig.add_subplot(gs[0, 0])
ax_a.axis('off')

# Draw the mapping table
table_data = [
    ['VCML component', 'Model A field eq.'],
    ['GRU hidden state h', 'local field variable'],
    ['CML diffusion', r'$D\nabla^2\phi$ term'],
    ['Wave perturbations', 'stochastic forcing $\eta$'],
    ['fieldM $\phi(x)$', 'slow order parameter'],
    ['mid_mem $m(x)$', 'drive term $s(x)$'],
    ['$C_{\\rm order}$', r'signal/noise functional'],
]
col_colors = [['#d0e4f7', '#d0e4f7']] + [['#f5f5f5', '#f5f5f5']] * (len(table_data)-1)
t = ax_a.table(cellText=table_data[1:], colLabels=table_data[0],
               cellColours=col_colors[1:], colColours=col_colors[0],
               loc='upper center', cellLoc='left')
t.auto_set_font_size(False); t.set_fontsize(8.5)
t.scale(1, 1.55)

# Add the effective field equation below
eq_text = (
    r'Effective PDE (Model A):' + '\n'
    r'$\partial_t\phi = -\kappa\,\phi + D\nabla^2\phi + r_w FA\,s(x) + \eta$'
    + '\n'
    r'$C_{\rm order} \propto \bar{s}/\sigma_\phi$   (order param / fluctuation)'
)
ax_a.text(0.5, 0.04, eq_text, ha='center', va='bottom', transform=ax_a.transAxes,
          fontsize=8.5, style='normal',
          bbox=dict(boxstyle='round,pad=0.4', fc='#fffbe6', ec='#c8a900', lw=1.1))

ax_a.set_title('(a) VCML -> Model A field theory\n'
               'Three noise channels: spatial, temporal, historical purity.',
               fontsize=8.5, fontweight='bold')

# ── Panel (b): P_hist vs P_sp empirical + geometric ──────────────────────────
ax_b = fig.add_subplot(gs[0, 1])

ax_b.plot(rw_arr, psp_cp,  'o-',  color='#d62728', lw=1.8, ms=8, label='$P_{sp}$ C_perturb')
ax_b.plot(rw_arr, psp_ref, 'o--', color='#d62728', lw=1.2, ms=6, alpha=0.55, label='$P_{sp}$ C_ref')
ax_b.plot(rw_arr, ph_cp,   's-',  color='#2ca02c', lw=1.8, ms=8, label='$P_{hist}$ C_perturb (emp.)')
ax_b.plot(rw_arr, ph_ref,  's--', color='#2ca02c', lw=1.2, ms=6, alpha=0.55, label='$P_{hist}$ C_ref (emp.)')
ax_b.plot(rw_arr, ph_geom, 'k--', lw=1.3, label='$P_{hist}$ geometric pred.', alpha=0.7)

# annotate r=8 failure
for r, na in zip(RWAVE, na_cp):
    m = '*' if na > 1 else 'x'
    ax_b.annotate(f'{m}', (r, psp_cp[RWAVE.index(r)]-0.005),
                  ha='center', fontsize=11, color='#d62728')

ax_b.set_xlabel('$r_{\\rm wave}$', fontsize=10)
ax_b.set_ylabel('Purity', fontsize=9)
ax_b.set_title('(b) $P_{hist} \\approx P_{sp}$ empirically.\n'
               'Geometric $P_{hist}$ severely underestimates actual.\n'
               'Historical purity is NOT a new dimension.\n'
               '* = na>1 (identity), x = failure',
               fontsize=8.5, fontweight='bold')
ax_b.legend(fontsize=7.0, loc='lower left')
ax_b.set_ylim(0.60, 1.02); ax_b.set_xlim(1, 11)

# ── Panel (c): sigma_w / (1-p) ratios for C_perturb ──────────────────────────
ax_c = fig.add_subplot(gs[0, 2])

ratio_sp   = [sw/(1-ps) if (1-ps)>1e-4 else float('nan')
              for sw, ps in zip(sw_cp, psp_cp)]
ratio_hist = [sw/(1-ph) if (1-ph)>1e-4 else float('nan')
              for sw, ph in zip(sw_cp, ph_cp)]

ax_c.plot(rw_arr, ratio_sp,   'o-', color='#d62728', lw=1.8, ms=8,
          label=r'$\sigma_w/(1-P_{sp})$  $\approx 0.84$ trend')
ax_c.plot(rw_arr, ratio_hist, 's-', color='#2ca02c', lw=1.8, ms=8,
          label=r'$\sigma_w/(1-P_{hist})$')

ax_c.axhline(0.84, color='#d62728', lw=0.9, ls=':', alpha=0.65)
ax_c.axhline(1.0,  color='grey', lw=0.8, ls='--', alpha=0.5)

# annotate r=8
ax_c.annotate('r=8 anomaly\n(na=0.943, fail)', (8, ratio_sp[2]),
              textcoords='offset points', xytext=(-55, 12),
              fontsize=7.5, color='#d62728',
              arrowprops=dict(arrowstyle='->', color='#d62728', lw=0.9))

for i, r in enumerate(RWAVE):
    ax_c.annotate(f'r={r}', (r, ratio_sp[i]),
                  textcoords='offset points', xytext=(4, 5), fontsize=7, color='#d62728')

ax_c.set_xlabel('$r_{\\rm wave}$', fontsize=10)
ax_c.set_ylabel(r'$\sigma_w / (1 - P)$ ratio', fontsize=9)
ax_c.set_title('(c) C_perturb: neither $1-P_{sp}$ nor $1-P_{hist}$\n'
               'gives a constant ratio for $\\sigma_w$ at $r=8$.\n'
               'r=8 is a structural anomaly, not a purity failure.',
               fontsize=8.5, fontweight='bold')
ax_c.legend(fontsize=7.5); ax_c.set_ylim(0, 2.2); ax_c.set_xlim(1, 11)

fig.suptitle(
    'Paper 59: Field Equation Limit of VCML and the Sufficiency of Two-Dimensional Purity\n'
    'P_hist is NOT a distinct third dimension: empirical P_hist ≈ P_sp (geometric underestimate corrected by launch asymmetry). '
    'The r=8 anomaly is a structural (resonance) failure in the field equation, not a causal purity failure.',
    fontsize=9, fontweight='bold', y=1.03
)

out = Path(__file__).parent / 'paper59_figure1.pdf'
plt.savefig(str(out), bbox_inches='tight')
print(f'Saved: {out}')
