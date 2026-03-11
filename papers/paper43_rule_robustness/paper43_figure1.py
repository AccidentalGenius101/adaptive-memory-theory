"""paper43_figure1.py -- 2-panel figure for Paper 43"""
import json, math, numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

DATA    = 'results/paper43_results.json'
OUT_PDF = 'paper43_figure1.pdf'
OUT_PNG = 'paper43_figure1.png'

FIELD_DECAY = 0.9997
T_WAVES = 1500   # wave stop time for Exp B

GRAY   = '#888888'
RED    = '#d6604d'
BLUE   = '#2166ac'
GREEN  = '#4dac26'
ORANGE = '#e6a817'
PURPLE = '#762a83'

ABLATION_CONDS = {'no_birth_seed', 'no_gating', 'no_both'}
VARIANT_CONDS  = {'ss_5', 'ss_20', 'mid_095', 'mid_0999', 'fsb_005', 'fsb_050'}

COND_ORDER_A = ['ref',
                'no_birth_seed', 'no_gating', 'no_both',
                'ss_5', 'ss_20', 'mid_095', 'mid_0999', 'fsb_005', 'fsb_050']

COND_LABELS = {
    'ref':           'C$_\\mathrm{ref}$',
    'no_birth_seed': 'no birth seed\n(SB=0)',
    'no_gating':     'no gating\n(streak ignored)',
    'no_both':       'no seed\n+ no gate',
    'ss_5':          'SS=5\n(fast consol.)',
    'ss_20':         'SS=20\n(slow consol.)',
    'mid_095':       'MD=0.95\n(fast decay)',
    'mid_0999':      'MD=0.9999\n(slow decay)',
    'fsb_005':       'SB=0.05\n(weak inherit.)',
    'fsb_050':       'SB=0.50\n(strong inherit.)',
}

PERSIST_ORDER = ['ref', 'no_birth_seed', 'no_gating', 'no_both']
PERSIST_STYLES = {
    'ref':           (GRAY,   '-',  'o', 'C$_\\mathrm{ref}$ (all mechanisms)'),
    'no_birth_seed': (BLUE,   '--', 's', 'No birth seed (SB=0)'),
    'no_gating':     (RED,    ':',  '^', 'No viability gate'),
    'no_both':       (PURPLE, '-.', 'D', 'No seed + no gate'),
}

def mn(lst):
    v=[x for x in lst if x is not None and not math.isnan(x)]
    return (float(np.mean(v)), float(np.std(v)/np.sqrt(len(v)))) if v else (float('nan'), float('nan'))

with open(DATA) as f: data = json.load(f)
for r in data:
    if 'exp' not in r: r['exp'] = 'A'

fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
fig.subplots_adjust(left=0.07, right=0.97, bottom=0.20, top=0.88, wspace=0.35)

# ── Panel (a): Exp A — Bar chart of sg4n at T=3000 ────────────────────────────
ax = axes[0]

means_A = {}; sems_A = {}
for cond in COND_ORDER_A:
    runs = [r for r in data if r['exp']=='A' and r['cond']==cond]
    if runs:
        m, se = mn([r['sg4ns'][-1] for r in runs])
        means_A[cond] = m; sems_A[cond] = se

x = np.arange(len(COND_ORDER_A))
colors = []
for cond in COND_ORDER_A:
    if cond == 'ref':              colors.append(GRAY)
    elif cond in ABLATION_CONDS:   colors.append(RED)
    else:                          colors.append(BLUE)

bars = ax.bar(x, [means_A.get(c, 0) for c in COND_ORDER_A],
              color=colors, alpha=0.85, width=0.6, zorder=3)
ax.errorbar(x, [means_A.get(c, 0) for c in COND_ORDER_A],
            yerr=[sems_A.get(c, 0) for c in COND_ORDER_A],
            fmt='none', color='k', capsize=4, lw=1.2, zorder=4)

# Reference line at ref level
ref_level = means_A.get('ref', 0)
ax.axhline(ref_level, color=GRAY, lw=1.2, ls='--', alpha=0.6, zorder=2,
           label=f'C$_\\mathrm{{ref}}$ level ({ref_level:.3f})')

ax.set_xticks(x)
ax.set_xticklabels([COND_LABELS[c] for c in COND_ORDER_A],
                   fontsize=7.5, rotation=0, ha='center')
ax.set_ylabel('sg4n at T=3000 (continuous waves)', fontsize=9)
ax.set_title('(a) Formation: structure robust to all rule variants\n'
             '(ablations in red, variants in blue)',
             fontsize=9, fontweight='bold')

legend_handles = [
    Patch(color=GRAY,  label='Reference'),
    Patch(color=RED,   label='Mechanism ablation'),
    Patch(color=BLUE,  label='Rule variant (+/-50%)'),
]
ax.legend(handles=legend_handles, fontsize=7.5, loc='upper right')
ax.set_ylim(bottom=0)
ax.tick_params(labelsize=8)

# Separator between ablations and variants
ax.axvline(3.5, color='k', lw=0.7, ls='--', alpha=0.3)
ax.text(3.5, ax.get_ylim()[1]*0.95, 'ablations | variants',
        ha='center', va='top', fontsize=6.5, color='k', alpha=0.5)

# ── Panel (b): Exp B — Persistence time series ────────────────────────────────
ax = axes[1]

for cond in PERSIST_ORDER:
    runs = [r for r in data if r['exp']=='B' and r['cond']==cond]
    if not runs: continue
    col, ls, mk, lbl = PERSIST_STYLES[cond]
    ts = np.array(runs[0]['ts'])
    n  = len(ts)
    vals = [mn([r['sg4ns'][i] for r in runs]) for i in range(n)]
    means = [v[0] for v in vals]; sems = [v[1] for v in vals]
    ax.plot(ts, means, color=col, ls=ls, lw=1.8, marker=mk, ms=4,
            markevery=2, label=lbl, zorder=3)
    ax.fill_between(ts, np.array(means)-np.array(sems),
                    np.array(means)+np.array(sems), color=col, alpha=0.12)

# Field-decay reference line from T=1500
runs_ref_b = [r for r in data if r['exp']=='B' and r['cond']=='ref']
if runs_ref_b:
    stops = [r['sg4n_at_stop'] for r in runs_ref_b if r.get('sg4n_at_stop') is not None]
    v_stop = mn(stops)[0]
    ts_b = np.array(runs_ref_b[0]['ts'])
    t_after = ts_b[ts_b > T_WAVES]
    fd_line = v_stop * FIELD_DECAY**(t_after - T_WAVES)
    ax.plot(t_after, fd_line, color=ORANGE, ls='-.', lw=1.2, alpha=0.8,
            label='Field decay only\n(from T=1500)')

ax.axvline(T_WAVES, color='k', lw=0.8, ls='--', alpha=0.5)
ax.text(T_WAVES+60, ax.get_ylim()[1] if ax.get_ylim()[1] > 0 else 0.8,
        'waves\nstop', fontsize=7.5, color='k', alpha=0.7, va='top')

# Annotate ratios
for cond in PERSIST_ORDER:
    runs = [r for r in data if r['exp']=='B' and r['cond']==cond]
    if not runs: continue
    col = PERSIST_STYLES[cond][0]
    s3, _ = mn([r['sg4ns'][-1] for r in runs])
    stops = [r['sg4n_at_stop'] for r in runs if r.get('sg4n_at_stop') is not None]
    sstop, _ = mn(stops)
    ratio = s3/sstop if sstop > 1e-6 else float('nan')
    if not math.isnan(ratio):
        ts_b = np.array(runs[0]['ts'])
        ax.annotate(f'{ratio:.2f}x', xy=(ts_b[-1], s3),
                    xytext=(ts_b[-1]+30, s3),
                    fontsize=7, color=col, va='center',
                    arrowprops=dict(arrowstyle='->', color=col, lw=0.5))

ax.set_xlabel('Time (steps)', fontsize=9)
ax.set_ylabel('sg4n', fontsize=9)
ax.set_title('(b) Maintenance: viability gating is the load-bearing primitive\n'
             '(ratio after wave stop; field-decay-only = 0.64x)',
             fontsize=9, fontweight='bold')
ax.legend(fontsize=7.5, loc='upper left')
ax.set_xlim(0, 3200); ax.set_ylim(bottom=0)
ax.tick_params(labelsize=8)

plt.suptitle('Paper 43: Rule-Family Robustness --- '
             'Formation is Robust; Maintenance Requires Copy-Forward + Viability Gate',
             fontsize=10, fontweight='bold', y=0.97)

plt.savefig(OUT_PDF, dpi=150, bbox_inches='tight')
plt.savefig(OUT_PNG, dpi=120, bbox_inches='tight')
print(f"Saved {OUT_PDF} / {OUT_PNG}")

# Key numbers
print("\n--- Key ratios (Exp B: sg4n@T3000 / sg4n@wave-stop) ---")
from math import isnan
fd_pred = FIELD_DECAY ** (3000 - T_WAVES)
print(f"Field-decay only: {fd_pred:.4f}")
for cond in PERSIST_ORDER:
    runs = [r for r in data if r['exp']=='B' and r['cond']==cond]
    if not runs: continue
    s3, se3 = mn([r['sg4ns'][-1] for r in runs])
    stops = [r['sg4n_at_stop'] for r in runs if r.get('sg4n_at_stop') is not None]
    sstop, _ = mn(stops)
    ratio = s3/sstop if sstop > 1e-6 else float('nan')
    print(f"  {cond:<16}: sg4n@stop={sstop:.4f}  sg4n@3000={s3:.4f}+-{se3:.4f}  ratio={ratio:.4f}")

print("\n--- Exp A: sg4n at T=3000 (continuous waves) ---")
fd_ref = means_A.get('ref', 0)
for cond in COND_ORDER_A:
    m = means_A.get(cond, float('nan'))
    se = sems_A.get(cond, float('nan'))
    t = 'ablation' if cond in ABLATION_CONDS else ('variant' if cond in VARIANT_CONDS else 'ref')
    print(f"  {cond:<16}: {m:.4f}+-{se:.4f}  ({t})")
