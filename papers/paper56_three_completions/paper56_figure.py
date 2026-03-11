"""Paper 56 figure: Three Completions for the VCML Regime Diagram.

Panel (a): Exp A -- SS sweep, na_ratio(t) trajectories + summary bar
Panel (b): Exp B -- r_wave sweep, na_ratio(t) + sigma_w(t) dual-axis
Panel (c): Exp C -- write_ready vs na_ratio scatter across all conditions
"""
import json, math, numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path

with open(Path(__file__).parent / 'results/paper56_analysis.json') as f:
    d = json.load(f)

SS_LEVELS   = [10, 20, 30, 50, 100, 200]
RWAVE_LEVELS = [1, 2, 4, 8]

# Colour maps
SS_COLORS = {
    10:  '#1f77b4',
    20:  '#2ca02c',
    30:  '#d62728',  # marginal identity peak
    50:  '#9467bd',
    100: '#8c564b',
    200: '#e377c2',
}
RW_COLORS = {
    1: '#9467bd',
    2: '#d62728',   # optimal
    4: '#2ca02c',
    8: '#1f77b4',
}

fig = plt.figure(figsize=(17, 5.2))
gs  = gridspec.GridSpec(1, 3, figure=fig, wspace=0.40)

# ── Panel (a): Exp A  na_ratio(t) by SS ────────────────────────────────────
ax_a = fig.add_subplot(gs[0, 0])
ax_a.axhline(1.0, color='k', lw=0.9, ls='--', alpha=0.55, zorder=0)

for ss in SS_LEVELS:
    k = f'SS{ss}'
    if k not in d['A']:
        continue
    r  = d['A'][k]
    st = np.array(r['steps'])
    na = np.array(r['na_mean'])
    se = np.array(r['na_se'])
    ok = ~np.isnan(na)
    lw = 2.2 if ss == 30 else 1.4
    ax_a.plot(st[ok], na[ok],
              color=SS_COLORS[ss], lw=lw, label=f'SS={ss}', zorder=4 if ss==30 else 2)
    ax_a.fill_between(st[ok], na[ok]-se[ok], na[ok]+se[ok],
                      color=SS_COLORS[ss], alpha=0.10)

ax_a.set_xlabel('Step', fontsize=10)
ax_a.set_ylabel('na_ratio = D_nonadj / D_adj', fontsize=9)
ax_a.set_title('(a) Exp A: SS threshold sweep (C_ref, sub-threshold)\n'
               'SS=30 marginal peak; high SS hurts via signal decay (MID_DECAY^SS)',
               fontsize=8.5, fontweight='bold')
ax_a.legend(fontsize=7.5, loc='upper right', ncol=2)
ax_a.set_xlim(0, 3000)
ax_a.set_ylim(0.5, 1.5)
ax_a.annotate('SS=30\n(marginal)', xy=(2800, 1.014), xytext=(2100, 1.22),
              fontsize=7, color=SS_COLORS[30],
              arrowprops=dict(arrowstyle='->', color=SS_COLORS[30], lw=1.1))

# ── Panel (b): Exp B  na_ratio(t) + sigma_w(t) for r_wave sweep ────────────
ax_b  = fig.add_subplot(gs[0, 1])
ax_b2 = ax_b.twinx()

ax_b.axhline(1.0, color='k', lw=0.9, ls='--', alpha=0.55, zorder=0)

for rw in RWAVE_LEVELS:
    k = f'rw{rw}'
    if k not in d['B']:
        continue
    r  = d['B'][k]
    st = np.array(r['steps'])
    na = np.array(r['na_mean'])
    sw = np.array(r['sigma_w_mean'])
    ok_na = ~np.isnan(na)
    ok_sw = ~np.isnan(sw)
    lw = 2.2 if rw == 2 else 1.4
    ax_b.plot(st[ok_na], na[ok_na],
              color=RW_COLORS[rw], lw=lw, label=f'r={rw}', zorder=4 if rw==2 else 2)
    ax_b2.plot(st[ok_sw], sw[ok_sw],
               color=RW_COLORS[rw], lw=lw, ls=':', alpha=0.6)

ax_b.set_xlabel('Step', fontsize=10)
ax_b.set_ylabel('na_ratio', fontsize=9)
ax_b2.set_ylabel('sigma_w (dotted)', fontsize=8, color='grey')
ax_b.set_title('(b) Exp B: r_wave sweep (C_perturb, S2, sub-threshold)\n'
               'Solid=na_ratio, dotted=sigma_w. r=2 optimal; r=8 fails (sigma_w too large)',
               fontsize=8.5, fontweight='bold')
ax_b.legend(fontsize=7.5, loc='upper left')
ax_b.set_xlim(0, 3000)
ax_b.set_ylim(0.5, 1.7)
ax_b2.set_ylim(0, 0.20)

# Annotate sigma_w values at step 3000
rw_sw_final = {}
for rw in RWAVE_LEVELS:
    k = f'rw{rw}'
    if k in d['B']:
        sw_m = d['B'][k]['sigma_w_mean']
        rw_sw_final[rw] = sw_m[-1] if sw_m else float('nan')

ax_b2.annotate(f"r=8: sw={rw_sw_final.get(8, float('nan')):.3f}",
               xy=(2900, rw_sw_final.get(8, 0.09)),
               fontsize=6.5, color=RW_COLORS[8], ha='right')
ax_b2.annotate(f"r=2: sw={rw_sw_final.get(2, float('nan')):.4f}",
               xy=(2900, min(rw_sw_final.get(2, 0.011)+0.015, 0.18)),
               fontsize=6.5, color=RW_COLORS[2], ha='right')

# ── Panel (c): Exp C -- write_ready vs na_ratio scatter ───────────────────
ax_c = fig.add_subplot(gs[0, 2])
ax_c.axhline(1.0, color='k', lw=0.9, ls='--', alpha=0.55)
ax_c.axvline(0.5, color='grey', lw=0.6, ls=':', alpha=0.5)

rows = d['C']
xs_A, ys_A, lbls_A = [], [], []
xs_B, ys_B, lbls_B = [], [], []

for row in rows:
    wr = row.get('mean_write_ready')
    na = row.get('final_na')
    if wr is None or na is None or math.isnan(wr) or math.isnan(na):
        continue
    if row.get('ss') is not None:
        xs_A.append(wr); ys_A.append(na); lbls_A.append(row['label'])
    else:
        xs_B.append(wr); ys_B.append(na); lbls_B.append(row['label'])

ax_c.scatter(xs_A, ys_A, marker='o', s=70, color='#1f77b4', label='Exp A (C_ref)', zorder=4)
ax_c.scatter(xs_B, ys_B, marker='^', s=80, color='#d62728', label='Exp B (C_perturb)', zorder=4)

for x, y, lbl in zip(xs_A, ys_A, lbls_A):
    ax_c.annotate(lbl, (x, y), textcoords='offset points', xytext=(4, 3),
                  fontsize=6.5, color='#1f77b4')
for x, y, lbl in zip(xs_B, ys_B, lbls_B):
    ax_c.annotate(lbl, (x, y), textcoords='offset points', xytext=(4, -8),
                  fontsize=6.5, color='#d62728')

ax_c.set_xlabel('mean write_ready_frac', fontsize=9)
ax_c.set_ylabel('final na_ratio', fontsize=9)
ax_c.set_title('(c) Exp C: write_ready vs identity (all conditions)\n'
               'No monotone relationship -- na_ratio is write-frequency-invariant',
               fontsize=8.5, fontweight='bold')
ax_c.legend(fontsize=7.5, loc='lower right')
ax_c.set_xlim(0.3, 1.05)
ax_c.set_ylim(0.6, 1.5)

fig.suptitle(
    'Paper 56: Three Completions for the VCML Regime Diagram\n'
    '(a) SS threshold does not substitute for C_perturb noise-gating. '
    '(b) C_perturb shifts effective correlation-length threshold (r=1 achieves identity at S2). '
    '(c) write_ready_frac is not the identity driver -- na_ratio - 1 is write-frequency-invariant.',
    fontsize=9, fontweight='bold', y=1.03
)

out = Path(__file__).parent / 'paper56_figure1.pdf'
plt.savefig(str(out), bbox_inches='tight')
print(f'Saved: {out}')
