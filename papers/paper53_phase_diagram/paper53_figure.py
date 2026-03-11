"""Paper 53 figure: VCML behavioral phase diagram — five axes of regime variation."""
import json, numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from pathlib import Path

# ── Embedded data from prior experiment series ────────────────────────────────

# V93: WR sweep at S3 (N=25,600) — Axis I (perturbation intensity)
V93 = {
    'wps':     [1,  2,   5,    10,    20   ],   # WR_MULT
    'coll':    [0.00195, 0.00441, 0.00990, 0.01889, 0.03480],
    'sg4_ref': [0.0071,  0.0383,  0.0112,  0.0048,  0.0062],
    'sg4_sta': [0.0042,  0.0041,  0.0061,  0.0067,  0.0072],
}

# V94: Scale sweep (2x WR) — Axis V (scale / correlation length)
V94 = {
    'N':          [1600,   6400,   25600,  102400 ],
    'sg4_norm_r': [0.4270, 0.4593, 0.3647, 0.1874 ],
    'sg4_norm_s': [0.1415, 0.0789, 0.0351, 0.0187 ],
    'na_ratio':   [0.839,  0.942,  0.915,  0.732  ],
}

# P47: delta sweep — Axis II (spatial resolution)
# na_ratio vs delta = W_zone / r_wave. Values reconstructed from paper text.
P47 = {
    'delta':    [1.25, 2.5,  5.0,  10.0, 20.0 ],
    'na_ratio': [0.88, 0.97, 1.24, 1.19, 1.18 ],
}

# P52: write policy 2D space — Axis IV
P52_PATH = Path(__file__).parent / '../paper52_boundary_distinction/results/paper52_analysis.json'

CONDITIONS = ['C_ref','C_near','C_near_low','C_near_high',
              'C_perturb','C_rand','C_abs','C_raw','C_far']
COLORS52 = {
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
# content axis encoding: raw_hid=2, signed_mid=1, abs_mid=0
CONTENT_LEVEL = {
    'C_raw': 2, 'C_ref': 1, 'C_near': 1, 'C_near_low': 1,
    'C_near_high': 1, 'C_perturb': 1, 'C_rand': 1, 'C_abs': 0, 'C_far': 1,
}
CONTENT_LABEL = {0: '|mid|', 1: 'signed mid', 2: 'raw hid'}

# ── Figure layout ─────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(17, 12))
gs = GridSpec(2, 3, figure=fig, hspace=0.42, wspace=0.38)

# ── Panel (a): Axis I — perturbation intensity (coll/site) ───────────────────
ax_a = fig.add_subplot(gs[0, 0])
cx = np.array(V93['coll'])
sg_r = np.array(V93['sg4_ref'])
sg_s = np.array(V93['sg4_sta'])
ax_a.plot(cx, sg_r, 'o-', color='#1f77b4', lw=1.8, ms=7, label='dynamic (C_ref)')
ax_a.plot(cx, sg_s, 's--', color='#aec7e8', lw=1.4, ms=6, label='static')
ax_a.fill_between([0,       0.0030], 0, 0.045, alpha=0.10, color='blue',
                  label='sub-optimal')
ax_a.fill_between([0.0030,  0.0060], 0, 0.045, alpha=0.14, color='green',
                  label='adaptive peak')
ax_a.fill_between([0.0060,  0.040 ], 0, 0.045, alpha=0.10, color='red',
                  label='over-perturbed')
ax_a.set_xlabel('coll / site / step', fontsize=10)
ax_a.set_ylabel('sg4', fontsize=10)
ax_a.set_xlim(0, 0.037)
ax_a.set_ylim(0, 0.045)
ax_a.set_title('(a)  Axis I: Perturbation intensity\n'
               'Non-monotone peak at coll/site ~ 0.004', fontsize=9, fontweight='bold')
ax_a.legend(fontsize=7, loc='upper right')
ax_a.axvline(0.00441, ls=':', color='green', lw=1.2)
ax_a.text(0.00441, 0.040, ' peak', fontsize=7, color='green')

# ── Panel (b): Axis II — spatial resolution (delta = W_zone / r_wave) ────────
ax_b = fig.add_subplot(gs[0, 1])
dx = np.array(P47['delta'])
na = np.array(P47['na_ratio'])
ax_b.plot(dx, na, 'D-', color='#9467bd', lw=1.8, ms=7)
ax_b.axhline(1.0, ls='--', color='k', lw=1.0, label='identity threshold (na=1)')
ax_b.fill_between([0, 2.5], 0.8, 1.35, alpha=0.12, color='red',   label='wave-bleed (parity)')
ax_b.fill_between([2.5, 22], 0.8, 1.35, alpha=0.10, color='green', label='identity encoding')
ax_b.set_xlabel('delta  =  W_zone / r_wave', fontsize=10)
ax_b.set_ylabel('na_ratio  (nonadj / adj)', fontsize=10)
ax_b.set_title('(b)  Axis II: Spatial resolution\n'
               'Identity encoding requires delta >= ~2.5', fontsize=9, fontweight='bold')
ax_b.set_ylim(0.80, 1.35)
ax_b.legend(fontsize=7)

# ── Panel (c): Axis V — scale (N_active, finite correlation length) ───────────
ax_c = fig.add_subplot(gs[0, 2])
Nx = np.array(V94['N'])
nr = np.array(V94['sg4_norm_r'])
ns = np.array(V94['sg4_norm_s'])
na_sc = np.array(V94['na_ratio'])
ax_c.semilogx(Nx, nr, 'o-', color='#1f77b4', lw=1.8, ms=7, label='dynamic sg4_norm')
ax_c.semilogx(Nx, ns, 's--', color='#aec7e8', lw=1.4, ms=6, label='static sg4_norm')
ax_c.semilogx(Nx, na_sc, '^:', color='#ff7f0e', lw=1.4, ms=6, label='na_ratio (right scale)')
ax_c.axhline(1.0, ls='--', color='k', lw=0.8)
ax_c.axvline(25600, ls=':', color='gray', lw=1.0)
ax_c.text(25600*1.1, 0.05, 'patch\nformation\n>S3', fontsize=7, color='gray')
ax_c.set_xlabel('N_active (log scale)', fontsize=10)
ax_c.set_ylabel('sg4_norm  /  na_ratio', fontsize=10)
ax_c.set_title('(c)  Axis V: Scale\n'
               'Finite correlation length; patch formation at S4', fontsize=9, fontweight='bold')
ax_c.legend(fontsize=7)

# ── Panel (d): Axis IV — write policy 2D space (Paper 52) ────────────────────
ax_d = fig.add_subplot(gs[1, 0:2])
with open(P52_PATH) as f:
    d52 = json.load(f)

# place conditions in 2D write policy space
# x-axis: log10(wps+1e-5) — trigger selectivity (right=promiscuous, left=selective)
# y-axis: na_form — identity encoding strength
for c in CONDITIONS:
    wps  = d52[c]['writes_per_site']
    na_f = d52[c]['na_form']
    cos  = d52[c]['cos_maint']
    cl   = CONTENT_LEVEL[c]
    marker = {0: 'X', 1: 'o', 2: 's'}[cl]
    ax_d.scatter(np.log10(wps + 1e-5), na_f,
                 color=COLORS52[c], s=140, marker=marker,
                 zorder=4, edgecolors='k', linewidths=0.8)
    ax_d.annotate(c, (np.log10(wps + 1e-5), na_f),
                  textcoords='offset points', xytext=(6, 3), fontsize=7)
ax_d.axhline(1.0, ls='--', color='k', lw=1.0, label='identity threshold')
ax_d.set_xlabel('log10(writes per site)   <-- selective       promiscuous -->', fontsize=10)
ax_d.set_ylabel('na_ratio at formation  (identity encoding strength)', fontsize=10)
ax_d.set_title('(d)  Axis IV: Write policy 2D space (Paper 52)\n'
               'Content axis (marker shape) x Trigger selectivity (x-axis). '
               'C_abs only parity failure.', fontsize=9, fontweight='bold')
# legend for content type
p_raw  = mpatches.Patch(facecolor='#8c564b', label='raw hid (s)')
p_mid  = mpatches.Patch(facecolor='#888888', label='signed mid (o)')
p_abs  = mpatches.Patch(facecolor='#d62728', label='|mid| (X)')
ax_d.legend(handles=[p_raw, p_mid, p_abs], fontsize=7, loc='lower right')

# shade regimes
ax_d.fill_between([-5.5, -2.5], 0.7, 2.0, alpha=0.10, color='green',
                  label='selective-trigger zone')
ax_d.fill_between([-0.5, 0.2],  0.7, 2.0, alpha=0.08, color='red',
                  label='promiscuous-trigger zone (re-encoding risk)')
ax_d.set_xlim(-5.5, 0.5)
ax_d.set_ylim(0.7, 2.0)

# ── Panel (e): Regime summary table ──────────────────────────────────────────
ax_e = fig.add_subplot(gs[1, 2])
ax_e.axis('off')
table_data = [
    ['Axis', 'Control param', 'Phase boundary', 'Regimes'],
    ['I', 'coll/site', '~0.003 / ~0.010', 'sub-opt | PEAK | over-pert'],
    ['II', 'delta=W/r', '~2.5', 'wave-bleed | identity'],
    ['III', 'Omega=WR*phi', 'context-dep', 'formation flux'],
    ['IV', '2D: trigger x content', 'sign(mid)=0', 'resistant | responsive'],
    ['V', 'N_active', '~25k sites/zone', 'gradient | patch'],
]
tbl = ax_e.table(cellText=table_data[1:], colLabels=table_data[0],
                 loc='center', cellLoc='left')
tbl.auto_set_font_size(False)
tbl.set_fontsize(7.5)
tbl.scale(1.0, 1.6)
for (r, c), cell in tbl.get_celld().items():
    if r == 0:
        cell.set_facecolor('#dddddd')
        cell.set_text_props(fontweight='bold')
    cell.set_edgecolor('#aaaaaa')
ax_e.set_title('(e)  Regime summary', fontsize=9, fontweight='bold', pad=14)

fig.suptitle('Paper 53: VCML Behavioral Phase Diagram\n'
             'Five control axes; each produces distinct regime boundaries. '
             'System is generative (not descriptive) at each axis.',
             fontsize=11, fontweight='bold', y=1.01)

plt.savefig('paper53_figure1.pdf', bbox_inches='tight')
print('Saved: paper53_figure1.pdf')
