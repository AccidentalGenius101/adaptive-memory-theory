"""Paper 54 figure: order parameter, correlation length, cross-axis surface."""
import json, numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from pathlib import Path

with open(Path(__file__).parent / 'results/paper54_analysis.json') as f:
    d = json.load(f)

AMP_LABELS  = ['A1_deep_sub','A2_near_thresh','A3_optimal','A4_over','A5_strong_over','A6_extreme']
AMP_LABELS_C = ['sub_thresh','optimal','over_pert']
RWAVE_LEVELS = [1, 2, 4, 8, 16]
WRITE_MODES  = ['C_ref','C_perturb']

fig = plt.figure(figsize=(16, 5))
gs  = GridSpec(1, 3, figure=fig, wspace=0.42)

# ── Panel (a): Exp A — sg_C test (snr vs coll/site) ─────────────────────────
ax_a = fig.add_subplot(gs[0, 0])
coll_a = [d[f'A|{l}']['coll_site'] for l in AMP_LABELS]
sg4_a  = [d[f'A|{l}']['sg4']  for l in AMP_LABELS]
snr_a  = [d[f'A|{l}']['snr']  for l in AMP_LABELS]
na_a   = [d[f'A|{l}']['na']   for l in AMP_LABELS]

color_na = ['#2ca02c' if v > 1 else '#d62728' for v in na_a]
ax_a.plot(coll_a, sg4_a, 'o-', color='#1f77b4', lw=1.8, ms=7, label='sg4')
ax_a2 = ax_a.twinx()
ax_a2.plot(coll_a, snr_a, 's--', color='#ff7f0e', lw=1.6, ms=6, label='snr = sg_C')
ax_a2.set_ylabel('snr (sg_C = sg4/sigma_w)', fontsize=9, color='#ff7f0e')
ax_a2.tick_params(axis='y', labelcolor='#ff7f0e')

# na_ratio annotations
for cx, na in zip(coll_a, na_a):
    enc = 'IDENT' if na > 1 else 'PAR'
    col = '#2ca02c' if na > 1 else '#d62728'
    ax_a.text(cx, 0.001, f'{na:.2f}\n[{enc}]', ha='center', va='bottom',
              fontsize=6.5, color=col, fontweight='bold')

ax_a.set_xlabel('coll / site / step', fontsize=10)
ax_a.set_ylabel('sg4', fontsize=10)
ax_a.set_title('(a) Exp A: sg_C as order parameter\n'
               'snr roughly flat in identity regime; na_ratio is cleaner regime indicator',
               fontsize=8.5, fontweight='bold')
lines1, labels1 = ax_a.get_legend_handles_labels()
lines2, labels2 = ax_a2.get_legend_handles_labels()
ax_a.legend(lines1+lines2, labels1+labels2, fontsize=8, loc='upper left')
ax_a.set_xlim(min(coll_a)*0.7, max(coll_a)*1.1)
ax_a.set_ylim(0.01, 0.030)
ax_a2.set_ylim(0.06, 0.14)

# ── Panel (b): Exp B — correlation length (r_wave sweep) ────────────────────
ax_b = fig.add_subplot(gs[0, 1])
rw   = np.array(RWAVE_LEVELS)
sg4_b = np.array([d[f'B|{r}']['sg4'] for r in RWAVE_LEVELS])
na_b  = np.array([d[f'B|{r}']['na']  for r in RWAVE_LEVELS])
snr_b = np.array([d[f'B|{r}']['snr'] for r in RWAVE_LEVELS])

ax_b.plot(rw, sg4_b, 'o-', color='#1f77b4', lw=1.8, ms=7, label='sg4')
ax_b.axhline(0, ls='--', color='gray', lw=0.7)
ax_b2 = ax_b.twinx()
ax_b2.plot(rw, na_b, 's:', color='#2ca02c', lw=1.6, ms=7, label='na_ratio')
ax_b2.axhline(1.0, ls='--', color='#2ca02c', lw=1.0, alpha=0.6)
ax_b2.set_ylabel('na_ratio', fontsize=9, color='#2ca02c')
ax_b2.tick_params(axis='y', labelcolor='#2ca02c')

# label delta at each r_wave
zone_w = 20  # S2 zone width
for r, s, n in zip(rw, sg4_b, na_b):
    delta = zone_w / r
    enc = 'IDENT' if n > 1 else 'PAR'
    ax_b.annotate(f'd={delta:.0f}\n{enc}', (r, s),
                  textcoords='offset points', xytext=(0, 8), ha='center',
                  fontsize=6.5, color='#2ca02c' if n > 1 else '#d62728')

# shade regimes
ax_b.fill_betweenx([0, 0.016], 0, 1.5,  alpha=0.10, color='red',   label='corr-length failure')
ax_b.fill_betweenx([0, 0.016], 10, 18,  alpha=0.08, color='orange', label='wave-bleed start')

ax_b.set_xlabel('r_wave (wave radius, S2 zone_width=20)', fontsize=10)
ax_b.set_ylabel('sg4', fontsize=10)
ax_b.set_title('(b) Exp B: Correlation length law\n'
               'r_wave=1 (delta=20) fails from corr-len not wave-bleed. '
               'na peaks at r=8 (delta=2.5).',
               fontsize=8.5, fontweight='bold')
ax_b.set_xlim(0, 18)
ax_b.set_ylim(0, 0.016)
ax_b2.set_ylim(0.6, 1.5)
lines1, labels1 = ax_b.get_legend_handles_labels()
lines2, labels2 = ax_b2.get_legend_handles_labels()
ax_b.legend(lines1+lines2, labels1+labels2, fontsize=7, loc='upper center')

# ── Panel (c): Exp C — cross-axis surface ───────────────────────────────────
ax_c = fig.add_subplot(gs[0, 2])
x     = np.arange(len(AMP_LABELS_C))
w     = 0.32
amp_xlabels = ['sub-threshold\n(supp=0.06)', 'optimal\n(supp=0.25)', 'over-pert\n(supp=0.80)']
cols  = {'C_ref': '#888888', 'C_perturb': '#ff7f0e'}

for i, wm in enumerate(WRITE_MODES):
    na_vals = [d[f'C|{al}|{wm}']['na'] for al in AMP_LABELS_C]
    offset  = (i - 0.5) * w
    bars = ax_c.bar(x + offset, na_vals, width=w, color=cols[wm],
                    edgecolor='k', linewidth=0.8, label=wm)
    for b, v in zip(bars, na_vals):
        enc = 'IDENT' if v > 1 else 'PAR'
        fw  = 'bold' if abs(v-1) > 0.05 else 'normal'
        ax_c.text(b.get_x()+b.get_width()/2, b.get_height()+0.02,
                  f'{v:.3f}\n[{enc}]', ha='center', va='bottom',
                  fontsize=6.5, fontweight=fw)

ax_c.axhline(1.0, ls='--', color='k', lw=1.2, label='identity threshold')
ax_c.set_xticks(x); ax_c.set_xticklabels(amp_xlabels, fontsize=8)
ax_c.set_ylabel('na_ratio at formation', fontsize=10)
ax_c.set_title('(c) Exp C: Write trigger x Amplitude (cross-axis)\n'
               'C_perturb: identity at ALL amplitudes. C_ref: parity at extremes.\n'
               'Prediction inverted — C_perturb stronger at sub-threshold.',
               fontsize=8.5, fontweight='bold')
ax_c.legend(fontsize=8)
ax_c.set_ylim(0.5, 1.9)

fig.suptitle('Paper 54: Order Parameter, Correlation Length Law, Cross-Axis Surface\n'
             'sg_C not unified OP; corr-len failure confirmed (r_wave=1, delta=20 fails); '
             'C_perturb noise-robust across all amplitudes',
             fontsize=10, fontweight='bold', y=1.02)
plt.savefig('paper54_figure1.pdf', bbox_inches='tight')
print('Saved: paper54_figure1.pdf')
