"""Paper 55 figure: Lyapunov drift test across behavioral regimes."""
import json, math, numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from pathlib import Path

with open(Path(__file__).parent / 'results/paper55_analysis.json') as f:
    d = json.load(f)

REGIMES = {
    'ident_cp':    dict(label='Identity (C_perturb)',          color='#2ca02c', ls='-',  lw=2.2, zorder=5),
    'ident_opt':   dict(label='Identity (C_ref, optimal amp)', color='#1f77b4', ls='--', lw=1.6, zorder=4),
    'parity_sub':  dict(label='Parity (sub-threshold amp)',    color='#d62728', ls=':',  lw=1.4, zorder=3),
    'parity_over': dict(label='Parity (over-perturbed)',       color='#ff7f0e', ls='-.',  lw=1.4, zorder=2),
    'patch':       dict(label='Patch (corr-len failure)',      color='#9467bd', ls='--', lw=1.4, zorder=1),
}

fig = plt.figure(figsize=(16, 5))
gs  = GridSpec(1, 3, figure=fig, wspace=0.40)

# ── Panel (a): C_order(t) trajectories ────────────────────────────────────────
ax_a = fig.add_subplot(gs[0, 0])
ax_a.axhline(0, color='k', lw=0.8, ls='--', alpha=0.6, zorder=0)
ax_a.axvspan(1000, 2000, alpha=0.06, color='green', label='_nolegend_')
ax_a.text(1500, 0.26, 'formation\nepoch', ha='center', va='top',
          fontsize=7, color='green', style='italic')

for rname, cfg in REGIMES.items():
    r   = d[rname]
    st  = np.array(r['steps'])
    co  = np.array(r['c_order_mean'])
    se  = np.array(r['c_order_se'])
    ok  = ~np.isnan(co)
    ax_a.plot(st[ok], co[ok], color=cfg['color'], ls=cfg['ls'],
              lw=cfg['lw'], label=cfg['label'], zorder=cfg['zorder'])
    ax_a.fill_between(st[ok], co[ok]-se[ok], co[ok]+se[ok],
                      color=cfg['color'], alpha=0.12)

ax_a.set_xlabel('Step', fontsize=10)
ax_a.set_ylabel('C_order = (D_nonadj - D_adj) / σ_w', fontsize=9)
ax_a.set_title('(a) C_order(t) by regime\n'
               'Positive drift only in C_perturb (noise-gated) identity.',
               fontsize=8.5, fontweight='bold')
ax_a.legend(fontsize=7.5, loc='upper left')
ax_a.set_xlim(0, 3000); ax_a.set_ylim(-0.12, 0.32)

# ── Panel (b): na_ratio(t) trajectories ───────────────────────────────────────
ax_b = fig.add_subplot(gs[0, 1])
ax_b.axhline(1.0, color='k', lw=0.8, ls='--', alpha=0.6)
ax_b.axvspan(1000, 2000, alpha=0.06, color='green')

for rname, cfg in REGIMES.items():
    r  = d[rname]
    st = np.array(r['steps'])
    na = np.array(r['na_mean'])
    ok = ~np.isnan(na)
    ax_b.plot(st[ok], na[ok], color=cfg['color'], ls=cfg['ls'],
              lw=cfg['lw'], label=cfg['label'], zorder=cfg['zorder'])

ax_b.set_xlabel('Step', fontsize=10)
ax_b.set_ylabel('na_ratio = D_nonadj / D_adj', fontsize=9)
ax_b.set_title('(b) na_ratio(t) by regime\n'
               'Both ident regimes reach na>1; C_ref transient only.',
               fontsize=8.5, fontweight='bold')
ax_b.legend(fontsize=7.5, loc='upper left')
ax_b.set_xlim(0, 3000); ax_b.set_ylim(0.3, 1.9)

# ── Panel (c): sigma_w(t) trajectories ────────────────────────────────────────
ax_c = fig.add_subplot(gs[0, 2])

for rname, cfg in REGIMES.items():
    r  = d[rname]
    st = np.array(r['steps'])
    sw = np.array(r['sigma_w_mean'])
    ok = ~np.isnan(sw)
    ax_c.plot(st[ok], sw[ok], color=cfg['color'], ls=cfg['ls'],
              lw=cfg['lw'], label=cfg['label'], zorder=cfg['zorder'])

# annotate the separation between C_perturb and C_ref
ax_c.annotate('C_perturb:\nσ_w controlled\n(~0.01–0.03)',
              xy=(2000, 0.012), xytext=(1700, 0.07),
              fontsize=7, color='#2ca02c',
              arrowprops=dict(arrowstyle='->', color='#2ca02c', lw=1.2))
ax_c.annotate('C_ref:\nσ_w large\n(~0.15–0.33)',
              xy=(800, 0.32), xytext=(400, 0.22),
              fontsize=7, color='#1f77b4',
              arrowprops=dict(arrowstyle='->', color='#1f77b4', lw=1.2))

ax_c.set_xlabel('Step', fontsize=10)
ax_c.set_ylabel('σ_w (within-zone fieldM std)', fontsize=9)
ax_c.set_title('(c) σ_w(t) by regime\n'
               'Mechanism: C_perturb controls noise floor; C_ref accumulates noise.',
               fontsize=8.5, fontweight='bold')
ax_c.legend(fontsize=7.5, loc='upper right')
ax_c.set_xlim(0, 3000); ax_c.set_ylim(0, 0.38)

fig.suptitle(
    'Paper 55: Lyapunov Drift Test\n'
    'C_order has monotone positive drift ONLY in C_perturb (noise-gated) regime. '
    'C_ref identity is transient: na crosses 1 at step ~2000 but C_order stays negative. '
    'Mechanism: σ_w must be controlled for Lyapunov property.',
    fontsize=9.5, fontweight='bold', y=1.02
)
plt.savefig('paper55_figure1.pdf', bbox_inches='tight')
print('Saved: paper55_figure1.pdf')
