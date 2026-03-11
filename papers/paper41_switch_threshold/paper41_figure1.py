"""paper41_figure1.py -- 3-panel figure for Paper 41"""
import json, math, numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

DATA    = 'results/paper41_results.json'
OUT_PDF = 'paper41_figure1.pdf'
OUT_PNG = 'paper41_figure1.png'

T_SWITCH_SWEEP = [50, 100, 200, 500, 800, 1000, 1500, 2000, 3000]
T_END = 6000
T_HALF_PRED = 800

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
fig.subplots_adjust(left=0.07, right=0.97, bottom=0.14, top=0.88, wspace=0.35)

# ── Panel (a): Temporal trajectories — C_ref vs interleaved ──────────────────
ax = axes[0]

# C_ref
refs = [r for r in data if r['exp']=='ref']
ts_r = np.array(refs[0]['ts'])
n_r  = len(ts_r)
ref_A = [mn([r['sg4ns_A'][i] for r in refs]) for i in range(n_r)]
ref_means = [v[0] for v in ref_A]; ref_sems = [v[1] for v in ref_A]
ax.plot(ts_r, ref_means, color=GRAY, lw=2.0, ls='-', label='C_ref (single task)',
        zorder=5)
ax.fill_between(ts_r, np.array(ref_means)-np.array(ref_sems),
                np.array(ref_means)+np.array(ref_sems), color=GRAY, alpha=0.2)

# T_switch = 500 (best condition)
for tsw, col, lbl in [(500, BLUE, 'T_sw=500 (task A)'),
                       (500, RED,  'T_sw=500 (task B)')]:
    runs = [r for r in data if r['exp']=='sweep' and r['t_switch']==tsw]
    ts_s = np.array(runs[0]['ts']); n_s = len(ts_s)
    key = 'sg4ns_A' if col==BLUE else 'sg4ns_B'
    vals = [mn([r[key][i] for r in runs]) for i in range(n_s)]
    means = [v[0] for v in vals]; sems = [v[1] for v in vals]
    ax.plot(ts_s, means, color=col, lw=1.6, ls='-', marker='o', ms=3,
            markevery=3, label=lbl, zorder=3)
    ax.fill_between(ts_s, np.array(means)-np.array(sems),
                    np.array(means)+np.array(sems), color=col, alpha=0.12)

# T_switch = 1500 for contrast
runs_15 = [r for r in data if r['exp']=='sweep' and r['t_switch']==1500]
ts_15 = np.array(runs_15[0]['ts']); n_15 = len(ts_15)
vals_A15 = [mn([r['sg4ns_A'][i] for r in runs_15]) for i in range(n_15)]
ax.plot(ts_15, [v[0] for v in vals_A15], color=GREEN, lw=1.2, ls='--',
        marker='s', ms=2.5, markevery=4, label='T_sw=1500 (task A)', alpha=0.8)

ax.set_xlabel('Time (steps)', fontsize=9)
ax.set_ylabel('sg4n', fontsize=9)
ax.set_title('(a) Single task decays;\ninterleaving maintains structure', fontsize=9, fontweight='bold')
ax.legend(fontsize=7.5, loc='upper right')
ax.set_xlim(0, T_END); ax.set_ylim(bottom=0)
ax.tick_params(labelsize=8)
# Mark metastability half-life
ax.axvline(T_HALF_PRED, color=GRAY, lw=0.7, ls=':', alpha=0.6)
ax.text(T_HALF_PRED+80, 0.62, r'$\tau_{1/2}$', fontsize=8, color=GRAY, alpha=0.8)

# ── Panel (b): Sweep summary — final sg4n vs T_switch ─────────────────────
ax = axes[1]
ref_final = mn([r['sg4ns_A'][-1] for r in refs])

sw_vals = T_SWITCH_SWEEP
vA_final = []; vB_final = []; seA = []; seB = []
for tsw in sw_vals:
    runs = [r for r in data if r['exp']=='sweep' and r['t_switch']==tsw]
    vA = mn([np.mean(r['sg4ns_A'][-3:]) for r in runs])
    vB = mn([np.mean(r['sg4ns_B'][-3:]) for r in runs])
    vA_final.append(vA[0]); seA.append(vA[1])
    vB_final.append(vB[0]); seB.append(vB[1])

sw_arr = np.array(sw_vals)
ax.errorbar(sw_arr, vA_final, yerr=seA, color=BLUE, lw=1.8, marker='o',
            ms=6, capsize=3, label='Task A (delta=0)', zorder=3)
ax.errorbar(sw_arr, vB_final, yerr=seB, color=RED,  lw=1.8, marker='s',
            ms=6, capsize=3, label='Task B (delta=5)', zorder=3)
ax.axhline(ref_final[0], color=GRAY, lw=1.5, ls='--',
           label=f'C_ref = {ref_final[0]:.3f}')
ax.fill_between([sw_arr[0], sw_arr[-1]], [ref_final[0]-ref_final[1]]*2,
                [ref_final[0]+ref_final[1]]*2, color=GRAY, alpha=0.15)

# Mark T_half prediction
ax.axvline(T_HALF_PRED, color=ORANGE, lw=1.2, ls='--', alpha=0.8,
           label=f'T_half ~ {T_HALF_PRED}')

ax.set_xscale('log')
ax.set_xlabel('T_switch (log scale)', fontsize=9)
ax.set_ylabel('Mean sg4n (final 3 checkpoints)', fontsize=9)
ax.set_title('(b) All interleaved >> single task;\noptimal T_switch ~ 500', fontsize=9, fontweight='bold')
ax.legend(fontsize=7.5, loc='upper right')
ax.set_xlim(40, 4000); ax.set_ylim(bottom=0)
ax.tick_params(labelsize=8)

# ── Panel (c): Sawtooth pattern for T_switch=1000 ─────────────────────────
ax = axes[2]
runs_10 = [r for r in data if r['exp']=='sweep' and r['t_switch']==1000]
ts_10   = np.array(runs_10[0]['ts']); n_10 = len(ts_10)
vA_10   = [mn([r['sg4ns_A'][i] for r in runs_10]) for i in range(n_10)]
vB_10   = [mn([r['sg4ns_B'][i] for r in runs_10]) for i in range(n_10)]
mA = [v[0] for v in vA_10]; sA = [v[1] for v in vA_10]
mB = [v[0] for v in vB_10]; sB = [v[1] for v in vB_10]

ax.plot(ts_10, mA, color=BLUE, lw=1.6, marker='o', ms=3, markevery=2,
        label='Task A (delta=0)')
ax.plot(ts_10, mB, color=RED,  lw=1.6, marker='s', ms=3, markevery=2,
        label='Task B (delta=5)')
ax.fill_between(ts_10, np.array(mA)-np.array(sA),
                np.array(mA)+np.array(sA), color=BLUE, alpha=0.12)
ax.fill_between(ts_10, np.array(mB)-np.array(sB),
                np.array(mB)+np.array(sB), color=RED,  alpha=0.12)

# shade A-active and B-active epochs
for k in range(6):
    t0 = k * 1000; t1 = t0 + 1000
    col = BLUE if k%2==0 else RED
    ax.axvspan(t0, t1, color=col, alpha=0.04)

ax.axhline(ref_final[0], color=GRAY, lw=1.0, ls='--', alpha=0.7,
           label=f'C_ref ({ref_final[0]:.3f})')
ax.set_xlabel('Time (steps)', fontsize=9)
ax.set_ylabel('sg4n', fontsize=9)
ax.set_title('(c) T_switch=1000: sawtooth alternation;\nboth tasks stay above single-task baseline',
             fontsize=9, fontweight='bold')
ax.legend(fontsize=7.5, loc='upper right')
ax.set_xlim(0, T_END); ax.set_ylim(bottom=0)
ax.tick_params(labelsize=8)

plt.suptitle('Paper 41: Interleaving Maintains Zone Structure via Copy-Forward Refreshing',
             fontsize=10, fontweight='bold', y=0.97)

plt.savefig(OUT_PDF, dpi=150, bbox_inches='tight')
plt.savefig(OUT_PNG, dpi=120, bbox_inches='tight')
print(f"Saved {OUT_PDF} / {OUT_PNG}")

# Print key numbers
print(f"\nC_ref final:  {ref_final[0]:.4f} +- {ref_final[1]:.4f}")
print(f"Best interleaved (T_sw=500): A={vA_final[T_SWITCH_SWEEP.index(500)]:.4f}  B={vB_final[T_SWITCH_SWEEP.index(500)]:.4f}")
print(f"Amplification factor (best/ref): {vA_final[T_SWITCH_SWEEP.index(500)]/ref_final[0]:.1f}x")
