"""Paper 52 figure: boundary-coupled distinction."""
import json, numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

with open('results/paper52_analysis.json') as f:
    d = json.load(f)

CONDITIONS = ['C_ref','C_near','C_near_low','C_near_high',
              'C_perturb','C_rand','C_abs','C_raw','C_far']
LABELS = {
    'C_ref':       'C_ref\n(standard)',
    'C_near':      'C_near\n(near bdry)',
    'C_near_low':  'C_near_low\n(low bdry)',
    'C_near_high': 'C_near_high\n(high bdry)',
    'C_perturb':   'C_perturb\n(in wave)',
    'C_rand':      'C_rand\n(matched rate)',
    'C_abs':       'C_abs\n(|mid|)',
    'C_raw':       'C_raw\n(raw hid)',
    'C_far':       'C_far\n(far bdry)',
}
COLORS = {
    'C_ref':       '#888888',
    'C_near':      '#2ca02c',
    'C_near_low':  '#17becf',
    'C_near_high': '#9edae5',
    'C_perturb':   '#ff7f0e',
    'C_rand':      '#d6b4fc',
    'C_abs':       '#d62728',
    'C_raw':       '#8c564b',
    'C_far':       '#7f7f7f',
}

fig = plt.figure(figsize=(15, 5))
gs  = GridSpec(1, 3, figure=fig, wspace=0.40)
x   = np.arange(len(CONDITIONS))

# ── Panel (a): na_ratio at formation ─────────────────────────────────────────
ax_a = fig.add_subplot(gs[0, 0])
na_vals = [d[c]['na_form'] for c in CONDITIONS]
bars = ax_a.bar(x, na_vals, color=[COLORS[c] for c in CONDITIONS],
                width=0.60, edgecolor='k', linewidth=0.8)
ax_a.axhline(1.0, ls='--', color='k', linewidth=1.2, label='identity threshold')
for b, v in zip(bars, na_vals):
    enc = 'IDENT' if v > 1 else 'PARITY'
    fw = 'bold' if (v > 1 or v < 0.95) else 'normal'
    ax_a.text(b.get_x()+b.get_width()/2, b.get_height()+0.012,
              f'{v:.3f}\n[{enc}]', ha='center', va='bottom', fontsize=7, fontweight=fw)
ax_a.set_xticks(x); ax_a.set_xticklabels([LABELS[c] for c in CONDITIONS], fontsize=7)
ax_a.set_ylabel('na_ratio (formation)', fontsize=10)
ax_a.set_title('(a) Zone encoding type after formation\n'
               'C_abs = only parity failure; C_raw = highest identity',
               fontsize=9, fontweight='bold')
ax_a.legend(fontsize=8)
ax_a.set_ylim(0, max(na_vals)*1.30)

# ── Panel (b): cos_maint — original structure preserved under adversarial ────
ax_b = fig.add_subplot(gs[0, 1])
cos_vals = [d[c]['cos_maint'] for c in CONDITIONS]
bars_b = ax_b.bar(x, cos_vals, color=[COLORS[c] for c in CONDITIONS],
                  width=0.60, edgecolor='k', linewidth=0.8)
ax_b.axhline(0.0, ls='--', color='k', linewidth=1.0, label='no correlation (re-encoded)')
for b, v in zip(bars_b, cos_vals):
    ax_b.text(b.get_x()+b.get_width()/2,
              (b.get_height()+0.02 if v>=0 else b.get_height()-0.06),
              f'{v:.3f}', ha='center', va='bottom', fontsize=7)
ax_b.set_xticks(x); ax_b.set_xticklabels([LABELS[c] for c in CONDITIONS], fontsize=7)
ax_b.set_ylabel('cosine similarity (pre vs post adversarial)', fontsize=10)
ax_b.set_title('(b) Adversarial resistance\n'
               'C_ref & C_far flip (negative); C_perturb best maintenance',
               fontsize=9, fontweight='bold')
ax_b.legend(fontsize=8)
ax_b.set_ylim(-0.45, 1.15)

# ── Panel (c): writes_per_site vs cos_maint (write efficiency) ───────────────
ax_c = fig.add_subplot(gs[0, 2])
wps  = [d[c]['writes_per_site'] for c in CONDITIONS]
cos  = [d[c]['cos_maint']       for c in CONDITIONS]
for c, w, co in zip(CONDITIONS, wps, cos):
    ax_c.scatter(w, co, color=COLORS[c], s=80, zorder=3, edgecolors='k', linewidths=0.7)
    ax_c.annotate(c, (w, co), textcoords='offset points', xytext=(5,3), fontsize=7)
ax_c.axhline(0, ls='--', color='gray', linewidth=0.8)
ax_c.set_xlabel('writes per site per step', fontsize=10)
ax_c.set_ylabel('cosine maintenance (adversarial)', fontsize=10)
ax_c.set_title('(c) Write efficiency\n'
               'C_perturb: near-zero writes, best maintenance',
               fontsize=9, fontweight='bold')

fig.suptitle('Paper 52: Boundary-Coupled Distinction — Sign Is Load-Bearing\n'
             'C_abs (|mid|) = only parity failure; confirms sign of contrast is necessary',
             fontsize=11, fontweight='bold', y=1.02)
plt.savefig('paper52_figure1.pdf', bbox_inches='tight')
print('Saved: paper52_figure1.pdf')
