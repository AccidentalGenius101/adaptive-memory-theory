"""paper42_figure1.py -- 3-panel figure for Paper 42"""
import json, math, numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

DATA    = 'results/paper42_results.json'
OUT_PDF = 'paper42_figure1.pdf'
OUT_PNG = 'paper42_figure1.png'

FIELD_DECAY = 0.9997
T_COMMIT    = 2000    # wave stop time for C_wavestop
T_SHIFT_B   = 1500   # zone shift for C_half

BLUE   = '#2166ac'
RED    = '#d6604d'
GREEN  = '#4dac26'
GRAY   = '#888888'
ORANGE = '#e6a817'
PURPLE = '#762a83'

def mn(lst):
    v = [x for x in lst if x is not None and not math.isnan(x)]
    return (float(np.mean(v)), float(np.std(v)/np.sqrt(len(v)))) if v else (float('nan'), float('nan'))

with open(DATA) as f: data = json.load(f)

fig, axes = plt.subplots(1, 3, figsize=(13, 4))
fig.subplots_adjust(left=0.07, right=0.97, bottom=0.14, top=0.88, wspace=0.38)

# ── Panel (a): Exp A -- Wave ablation ─────────────────────────────────────────
ax = axes[0]

cond_styles = {
    'ref':      (GRAY,  '-',  'o', 'C_ref (waves throughout)'),
    'wavestop': (BLUE,  '--', 's', 'C_wavestop (stop at T=2000)'),
    'nowaves':  (RED,   ':',  '^', 'C_nowaves (WR=0 always)'),
}

for cond, (col, ls, mk, lbl) in cond_styles.items():
    runs = [r for r in data if r['exp']=='A' and r['cond']==cond]
    if not runs: continue
    ts   = np.array(runs[0]['ts']); n = len(ts)
    vals = [mn([r['sg4ns'][i] for r in runs]) for i in range(n)]
    means = [v[0] for v in vals]; sems = [v[1] for v in vals]
    ax.plot(ts, means, color=col, ls=ls, lw=1.8, marker=mk, ms=4, markevery=4,
            label=lbl, zorder=3)
    ax.fill_between(ts, np.array(means)-np.array(sems),
                    np.array(means)+np.array(sems), color=col, alpha=0.15)

# Field-decay reference line from T=2000
runs_ref_a = [r for r in data if r['exp']=='A' and r['cond']=='ref']
ts_a = np.array(runs_ref_a[0]['ts'])
idx2000 = list(ts_a).index(2000) if 2000 in ts_a else 9  # 10th checkpoint = T=2000
v2000 = mn([r['sg4ns'][idx2000] for r in runs_ref_a])[0]
t_ref = ts_a[ts_a >= 2000]
fd_line = v2000 * FIELD_DECAY**(t_ref - 2000)
ax.plot(t_ref, fd_line, color=ORANGE, ls='-.', lw=1.2, alpha=0.8,
        label='Field decay only\n(from T=2000 peak)')

ax.axvline(T_COMMIT, color='k', lw=0.8, ls='--', alpha=0.5)
ax.text(T_COMMIT+60, 0.36, 'waves\nstop', fontsize=7, color='k', alpha=0.7)
ax.set_xlabel('Time (steps)', fontsize=9)
ax.set_ylabel('sg4n', fontsize=9)
ax.set_title('(a) Anomaly 1: Spontaneous copy-forward\nmaintains structure after wave removal',
             fontsize=9, fontweight='bold')
ax.legend(fontsize=7, loc='upper right')
ax.set_xlim(0, 4000); ax.set_ylim(bottom=0)
ax.tick_params(labelsize=8)

# ── Panel (b): Exp B -- C_half surge anatomy ──────────────────────────────────
ax = axes[1]

b_styles = {
    'ref':        (GRAY, '-',  'o', 'C_ref (delta=0)'),
    'half_shift': (RED,  '-',  's', 'C_half (delta=5 at T=1500)'),
}

for cond, (col, ls, mk, lbl) in b_styles.items():
    runs = [r for r in data if r['exp']=='B' and r['cond']==cond]
    if not runs: continue
    ts   = np.array(runs[0]['ts']); n = len(ts)
    vals = [mn([r['sg4ns_new'][i] for r in runs]) for i in range(n)]
    means = [v[0] for v in vals]; sems = [v[1] for v in vals]
    ax.plot(ts, means, color=col, ls=ls, lw=1.8, marker=mk, ms=4, markevery=3,
            label=lbl, zorder=3)
    ax.fill_between(ts, np.array(means)-np.array(sems),
                    np.array(means)+np.array(sems), color=col, alpha=0.15)

# Add collapse rate on right y-axis
ax2b = ax.twinx()
for cond, (col, ls, mk, lbl) in b_styles.items():
    runs = [r for r in data if r['exp']=='B' and r['cond']==cond]
    if not runs: continue
    ts = np.array(runs[0]['ts']); n = len(ts)
    # Collapse rate per step (incremental cc over each 300-step window)
    N_CELLS = 40 * 40  # HALF * H
    cc_rates = []
    for i in range(n):
        if i == 0:
            rate = mn([r['cc_totals'][0] / ts[0] / N_CELLS for r in runs])
        else:
            dt = ts[i] - ts[i-1]
            rate = mn([(r['cc_totals'][i]-r['cc_totals'][i-1]) / dt / N_CELLS
                       for r in runs])
        cc_rates.append(rate[0])
    ax2b.plot(ts, cc_rates, color=col, ls=':', lw=1.0, alpha=0.5, zorder=1)

ax2b.set_ylabel('Collapse rate (per cell per step)', fontsize=7, color=GRAY)
ax2b.tick_params(labelsize=7, colors=GRAY)
ax2b.set_ylim(bottom=0)

ax.axvline(T_SHIFT_B, color='k', lw=0.8, ls='--', alpha=0.5)
ax.text(T_SHIFT_B+100, 0.38, 'zone\nshift', fontsize=7, color='k', alpha=0.7)

# Mark Paper 40's T=4000 result for reference
ax.axhline(0.798, color=RED, lw=0.6, ls=':', alpha=0.4)
ax.text(300, 0.82, 'Paper 40 peak (T=4000)', fontsize=6.5, color=RED, alpha=0.6)

ax.set_xlabel('Time (steps)', fontsize=9)
ax.set_ylabel('sg4n (new zone labels)', fontsize=9)
ax.set_title('(b) Anomaly 2: C_half surge sustained to T=6000\n(same copy-forward mechanism as Paper 41)',
             fontsize=9, fontweight='bold')
ax.legend(fontsize=7.5, loc='upper right')
ax.set_xlim(0, 6000); ax.set_ylim(bottom=0)
ax.tick_params(labelsize=8)

# ── Panel (c): Exp C -- Phase aliasing ────────────────────────────────────────
ax = axes[2]

TSW_C = [50, 100]
x_positions = np.array([0.5, 1.5, 3.0, 4.0])
bar_w = 0.35

# Compute biased CPS measurements (last 3 checkpoints)
biased_A = {}; biased_B = {}
epoch_A  = {}; epoch_B  = {}

for tsw in TSW_C:
    runs = [r for r in data if r['exp']=='C' and r['t_switch']==tsw]
    if not runs: continue
    biased_A[tsw] = mn([np.mean(r['sg4ns_A_cp'][-3:]) for r in runs])
    biased_B[tsw] = mn([np.mean(r['sg4ns_B_cp'][-3:]) for r in runs])
    def tail20(lst, k=20):
        tail = lst[-k:] if len(lst) >= k else lst
        return [float(x) for x in tail if not math.isnan(x)]
    epoch_A[tsw] = mn([np.mean(tail20(r['epoch_end_A'])) for r in runs if r['epoch_end_A']])
    epoch_B[tsw] = mn([np.mean(tail20(r['epoch_end_B'])) for r in runs if r['epoch_end_B']])

# x layout: [biased_50, epoch_50] gap [biased_100, epoch_100]
x = np.array([0.0, 1.0, 2.5, 3.5])
labels = ['Biased\n(T_sw=50)', 'Corrected\n(T_sw=50)', 'Biased\n(T_sw=100)', 'Corrected\n(T_sw=100)']

for i, (tsw, kind) in enumerate([(50,'biased'), (50,'epoch'), (100,'biased'), (100,'epoch')]):
    if kind == 'biased':
        vA = biased_A[tsw]; vB = biased_B[tsw]
    else:
        vA = epoch_A[tsw];  vB = epoch_B[tsw]
    ax.bar(x[i]-bar_w/2, vA[0], bar_w, color=BLUE, alpha=0.85, zorder=3)
    ax.bar(x[i]+bar_w/2, vB[0], bar_w, color=RED,  alpha=0.85, zorder=3)
    ax.errorbar(x[i]-bar_w/2, vA[0], yerr=vA[1], fmt='none', color='k', capsize=3, lw=1.0)
    ax.errorbar(x[i]+bar_w/2, vB[0], yerr=vB[1], fmt='none', color='k', capsize=3, lw=1.0)
    # Annotate which task wins
    winner_str = 'A>B' if vA[0] >= vB[0] else 'B>A(!)'
    color_ann  = 'k' if vA[0] >= vB[0] else RED
    ax.text(x[i], max(vA[0], vB[0]) + 0.04, winner_str,
            ha='center', va='bottom', fontsize=8, color=color_ann, fontweight='bold')

ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=8)
ax.set_ylabel('Mean sg4n (steady state)', fontsize=9)
ax.set_title('(c) Anomaly 3: Phase aliasing artifact\nEpoch-end measurement restores A>B',
             fontsize=9, fontweight='bold')

from matplotlib.patches import Patch
ax.legend(handles=[Patch(color=BLUE, label='Task A (delta=0)'),
                   Patch(color=RED,  label='Task B (delta=5)')],
          fontsize=7.5, loc='upper right')
ax.set_xlim(-0.5, 4.2); ax.set_ylim(bottom=0, top=0.85)
ax.tick_params(labelsize=8)

# Add phase aliasing annotation
ax.annotate('300/50=6 (even):\nall CPS in task B',
            xy=(x[0], 0.05), xytext=(x[0]-0.2, 0.60),
            fontsize=6.5, color=GRAY,
            arrowprops=dict(arrowstyle='->', color=GRAY, lw=0.7))
ax.annotate('300/100=3 (odd):\nCPS alternate A/B',
            xy=(x[2], 0.05), xytext=(x[2]-0.4, 0.70),
            fontsize=6.5, color=GRAY,
            arrowprops=dict(arrowstyle='->', color=GRAY, lw=0.7))

# Vertical separator between T_sw=50 and T_sw=100 groups
ax.axvline(1.75, color=GRAY, lw=0.5, ls='--', alpha=0.4)
ax.text(1.75, 0.78, 'T_sw=50 | T_sw=100', ha='center', fontsize=6.5, color=GRAY, alpha=0.7)

plt.suptitle('Paper 42: Three Anomalies Closed --- '
             'Copy-Forward Maintenance, C_half Surge, Phase Aliasing',
             fontsize=10, fontweight='bold', y=0.97)

plt.savefig(OUT_PDF, dpi=150, bbox_inches='tight')
plt.savefig(OUT_PNG, dpi=120, bbox_inches='tight')
print(f"Saved {OUT_PDF} / {OUT_PNG}")

# Key numbers
print("\n--- Key results ---")
runs_ws = [r for r in data if r['exp']=='A' and r['cond']=='wavestop']
runs_rf = [r for r in data if r['exp']=='A' and r['cond']=='ref']
ts_a = list(runs_ws[0]['ts'])
if 2000 in ts_a:
    i2 = ts_a.index(2000); i4 = ts_a.index(4000)
    v2_ws = mn([r['sg4ns'][i2] for r in runs_ws]); v4_ws = mn([r['sg4ns'][i4] for r in runs_ws])
    v2_rf = mn([r['sg4ns'][i2] for r in runs_rf]); v4_rf = mn([r['sg4ns'][i4] for r in runs_rf])
    fd_pred = FIELD_DECAY**2000
    print(f"Exp A: C_ref ratio T4/T2 = {v4_rf[0]/v2_rf[0]:.3f} "
          f"| C_wavestop ratio = {v4_ws[0]/v2_ws[0]:.3f} "
          f"| Field-decay only = {fd_pred:.3f}")

runs_bh = [r for r in data if r['exp']=='B' and r['cond']=='half_shift']
runs_br = [r for r in data if r['exp']=='B' and r['cond']=='ref']
if runs_bh and runs_br:
    sg6_h = mn([r['sg4ns_new'][-1] for r in runs_bh])
    sg6_r = mn([r['sg4ns_new'][-1] for r in runs_br])
    print(f"Exp B: C_ref sg4n@T=6000 = {sg6_r[0]:.4f} "
          f"| C_half sg4n@T=6000 = {sg6_h[0]:.4f} "
          f"| ratio = {sg6_h[0]/sg6_r[0]:.2f}x")

for tsw in [50, 100]:
    print(f"Exp C T_sw={tsw}: "
          f"Biased A={biased_A.get(tsw,[(0,0)])[0]:.3f} B={biased_B.get(tsw,[(0,0)])[0]:.3f} "
          f"| Epoch-end A={epoch_A.get(tsw,[(0,0)])[0]:.3f} B={epoch_B.get(tsw,[(0,0)])[0]:.3f}")
