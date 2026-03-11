"""Paper 50 figure: zone parity vs identity in geographic relay."""
import json, numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

with open('results/paper50_analysis.json') as f:
    d = json.load(f)

CONDITIONS = ['ctrl_rnd','ctrl_std','geo_copy','geo_4class','geo_own_relay']
LABELS = {
    'ctrl_rnd':     'ctrl_rnd\n(random waves)',
    'ctrl_std':     'ctrl_std\n(own WaveEnvStd)',
    'geo_copy':     'geo_copy\n(amplitude copy)',
    'geo_4class':   'geo_4class\n(4-level amp)',
    'geo_own_relay':'geo_own_relay\n(own env + birth)',
}
COLORS = {
    'ctrl_rnd':     '#888888',
    'ctrl_std':     '#2ca02c',
    'geo_copy':     '#d62728',
    'geo_4class':   '#ff7f0e',
    'geo_own_relay':'#9467bd',
}

fig = plt.figure(figsize=(12, 5))
gs  = GridSpec(1, 2, figure=fig, wspace=0.38)

x = np.arange(len(CONDITIONS))
na_vals  = [d[c]['na_l2']   for c in CONDITIONS]
sg4_vals = [d[c]['sg4_l2']  for c in CONDITIONS]
dn_vals  = [d[c]['dnorm_l2']for c in CONDITIONS]

# ── Panel (a): na_ratio — the main finding ────────────────────────────────────
ax_a = fig.add_subplot(gs[0, 0])
bars = ax_a.bar(x, na_vals, color=[COLORS[c] for c in CONDITIONS],
                width=0.55, edgecolor='k', linewidth=0.8)
ax_a.axhline(1.0, ls='--', color='k', linewidth=1.2, label='identity threshold (na=1)')
ax_a.axhline(d['ctrl_rnd']['na_l1'], ls=':', color='#333333', alpha=0.5,
             label=f'L1 na={d["ctrl_rnd"]["na_l1"]:.3f}')
for b, v in zip(bars, na_vals):
    enc = 'IDENT' if v > 1 else 'parity'
    ax_a.text(b.get_x()+b.get_width()/2, b.get_height()+0.008,
              f'{v:.3f}\n[{enc}]', ha='center', va='bottom', fontsize=8,
              fontweight='bold' if v > 1 else 'normal')
ax_a.set_xticks(x)
ax_a.set_xticklabels([LABELS[c] for c in CONDITIONS], fontsize=8)
ax_a.set_ylabel('na_ratio (nonadj/adj)', fontsize=11)
ax_a.set_title('(a) Zone encoding type: parity vs identity\n'
               'na>1 = identity encoding; na<1 = parity encoding',
               fontsize=10, fontweight='bold')
ax_a.legend(fontsize=8, loc='upper right')
ax_a.set_ylim(0, max(na_vals)*1.25)

# ── Panel (b): sg4 and decode_norm comparison ─────────────────────────────────
ax_b = fig.add_subplot(gs[0, 1])
w = 0.35
bars_sg4 = ax_b.bar(x - w/2, sg4_vals, width=w,
                    color=[COLORS[c] for c in CONDITIONS],
                    edgecolor='k', linewidth=0.8, label='sg4 (L2)')
ax_b2 = ax_b.twinx()
ax_b2.plot(x, dn_vals, 'D--', color='#333333', markersize=8,
           label='decode_norm', linewidth=1.5)
ax_b2.set_ylabel('decode_norm', fontsize=10, color='#333333')
ax_b2.tick_params(axis='y', labelcolor='#333333')
ax_b.set_xticks(x)
ax_b.set_xticklabels([LABELS[c] for c in CONDITIONS], fontsize=8)
ax_b.set_ylabel('sg4 (L2)', fontsize=11)
ax_b.set_title('(b) sg4 and decode_norm across conditions\n'
               'ctrl_std: highest identity but lower sg4',
               fontsize=10, fontweight='bold')
lines1, labs1 = ax_b.get_legend_handles_labels()
lines2, labs2 = ax_b2.get_legend_handles_labels()
ax_b.legend(lines1+lines2, labs1+labs2, fontsize=8, loc='upper right')

# annotate ctrl_rnd as gain=1.00 baseline
ctrl_sg4 = d['ctrl_rnd']['sg4_l2']
for b, v in zip(bars_sg4, sg4_vals):
    g = v/ctrl_sg4
    ax_b.text(b.get_x()+b.get_width()/2, b.get_height()+0.0003,
              f'{g:.2f}x', ha='center', va='bottom', fontsize=8)

fig.suptitle('Paper 50: Zone Identity vs Parity — Geographic Relay Audit',
             fontsize=12, fontweight='bold', y=1.01)
plt.savefig('paper50_figure1.pdf', bbox_inches='tight')
print('Saved: paper50_figure1.pdf')
