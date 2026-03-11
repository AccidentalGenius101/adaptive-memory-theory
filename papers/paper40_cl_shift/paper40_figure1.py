"""paper40_figure1.py -- 3-panel figure for Paper 40"""
import json, math, numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

DATA = 'results/paper40_results.json'
OUT_PDF = 'paper40_figure1.pdf'
OUT_PNG = 'paper40_figure1.png'

T_SHIFT_A  = 1500
T_COMMIT_C = 2000
ZW = 10

BLUE  = '#2166ac'
RED   = '#d6604d'
GREEN = '#4dac26'
GRAY  = '#888888'
ORANGE= '#f4a582'
PURPLE= '#762a83'

def mn(lst):
    v = [x for x in lst if x is not None and not math.isnan(x)]
    return (float(np.mean(v)), float(np.std(v)/np.sqrt(len(v)))) if v else (float('nan'), float('nan'))

def load():
    with open(DATA) as f: return json.load(f)

data = load()

fig, axes = plt.subplots(1, 3, figsize=(13, 4))
fig.subplots_adjust(left=0.07, right=0.97, bottom=0.14, top=0.88, wspace=0.35)

# ── Panel (a): Exp A — new zone structure building ────────────────────────────
ax = axes[0]
CPS_A = list(range(200, 4001, 200))
label_map = {
    0:  ('C_ref  (no shift)',        GRAY,   '-',  'o'),
    5:  ('C_half (delta=5,  50%)',   BLUE,   '-',  's'),
    10: ('C_full (delta=10, 100%)', RED,    '-',  '^'),
}
for delta, (lbl, col, ls, mk) in label_map.items():
    runs = [r for r in data if r['exp']=='A' and r['delta_after']==delta]
    ts   = runs[0]['ts']
    n    = len(ts)
    vals = [mn([r['sg4ns_new'][i] for r in runs]) for i in range(n)]
    means = [v[0] for v in vals]; sems = [v[1] for v in vals]
    ts_a = np.array(ts)
    ax.plot(ts_a, means, color=col, ls=ls, lw=1.6, marker=mk, ms=4,
            markevery=3, label=lbl, zorder=3)
    ax.fill_between(ts_a, np.array(means)-np.array(sems),
                    np.array(means)+np.array(sems), color=col, alpha=0.15)

ax.axvline(T_SHIFT_A, color='k', lw=1.0, ls='--', alpha=0.5)
ax.text(T_SHIFT_A+50, 0.02, 'T_shift', fontsize=7, color='k', alpha=0.7)
ax.set_xlabel('Time (steps)', fontsize=9)
ax.set_ylabel('sg4n (new zone labels)', fontsize=9)
ax.set_title('(a) New structure — zone boundary shift', fontsize=9, fontweight='bold')
ax.legend(fontsize=7, loc='upper left')
ax.set_xlim(0, 4000); ax.set_ylim(bottom=0)
ax.tick_params(labelsize=8)

# ── Panel (b): Exp A — old structure forgetting (C_half most interesting) ─────
ax = axes[1]
for delta, (lbl, col, ls, mk) in label_map.items():
    runs = [r for r in data if r['exp']=='A' and r['delta_after']==delta]
    ts   = runs[0]['ts']; n = len(ts)
    # old = sg4n_old for shifted conds, = sg4n_new for ref (they're the same)
    vals = [mn([r['sg4ns_old'][i] for r in runs]) for i in range(n)]
    means = [v[0] for v in vals]; sems = [v[1] for v in vals]
    ts_a = np.array(ts)
    ax.plot(ts_a, means, color=col, ls=ls, lw=1.6, marker=mk, ms=4,
            markevery=3, label=lbl, zorder=3)
    ax.fill_between(ts_a, np.array(means)-np.array(sems),
                    np.array(means)+np.array(sems), color=col, alpha=0.15)

ax.axvline(T_SHIFT_A, color='k', lw=1.0, ls='--', alpha=0.5)
ax.text(T_SHIFT_A+50, 0.02, 'T_shift', fontsize=7, color='k', alpha=0.7)
ax.set_xlabel('Time (steps)', fontsize=9)
ax.set_ylabel('sg4n (old zone labels)', fontsize=9)
ax.set_title('(b) Old structure — forgetting after shift', fontsize=9, fontweight='bold')
ax.legend(fontsize=7, loc='upper left')
ax.set_xlim(0, 4000); ax.set_ylim(bottom=0)
ax.tick_params(labelsize=8)

# ── Panel (c): Exp C — post-commitment interleaved switching ──────────────────
ax = axes[2]
CPS_C = list(range(200, 5001, 200))
runs_B = [r for r in data if r['exp']=='B']
runs_C = [r for r in data if r['exp']=='C']

# Exp B (no commitment): sg4n_A, sg4n_B
if runs_B:
    ts_b = np.array(runs_B[0]['ts'])
    n_b  = len(ts_b)
    vA = [mn([r['sg4ns_A'][i] for r in runs_B]) for i in range(n_b)]
    vB = [mn([r['sg4ns_B'][i] for r in runs_B]) for i in range(n_b)]
    ax.plot(ts_b, [v[0] for v in vA], color=BLUE,   ls='--', lw=1.2,
            label='Exp B: task A (no commit)', alpha=0.7)
    ax.plot(ts_b, [v[0] for v in vB], color=ORANGE, ls='--', lw=1.2,
            label='Exp B: task B (no commit)', alpha=0.7)

# Exp C (post-commitment): sg4n_A, sg4n_B
if runs_C:
    ts_c = np.array(runs_C[0]['ts'])
    n_c  = len(ts_c)
    vA = [mn([r['sg4ns_A'][i] for r in runs_C]) for i in range(n_c)]
    vB = [mn([r['sg4ns_B'][i] for r in runs_C]) for i in range(n_c)]
    mA = [v[0] for v in vA]; sA = [v[1] for v in vA]
    mB = [v[0] for v in vB]; sB = [v[1] for v in vB]
    ax.plot(ts_c, mA, color=BLUE,   ls='-', lw=1.6, marker='o', ms=3,
            markevery=3, label='Exp C: task A (commit+switch)')
    ax.plot(ts_c, mB, color=RED,    ls='-', lw=1.6, marker='s', ms=3,
            markevery=3, label='Exp C: task B (commit+switch)')
    ax.fill_between(ts_c, np.array(mA)-np.array(sA),
                    np.array(mA)+np.array(sA), color=BLUE, alpha=0.12)
    ax.fill_between(ts_c, np.array(mB)-np.array(sB),
                    np.array(mB)+np.array(sB), color=RED, alpha=0.12)

ax.axvline(T_COMMIT_C, color='k', lw=1.0, ls='--', alpha=0.5)
ax.text(T_COMMIT_C+60, 0.02, 'switch\nstarts', fontsize=7, color='k', alpha=0.7)

ax.set_xlabel('Time (steps)', fontsize=9)
ax.set_ylabel('sg4n', fontsize=9)
ax.set_title('(c) Two-task interleaving\n(Exp B: no commit; Exp C: commit first)',
             fontsize=9, fontweight='bold')
ax.legend(fontsize=7, loc='upper right')
ax.set_xlim(0, 5000); ax.set_ylim(bottom=0)
ax.tick_params(labelsize=8)

plt.suptitle('Paper 40: Real Continual Learning — Zone Boundary Shift',
             fontsize=10, fontweight='bold', y=0.97)

plt.savefig(OUT_PDF, dpi=150, bbox_inches='tight')
plt.savefig(OUT_PNG, dpi=120, bbox_inches='tight')
print(f"Saved {OUT_PDF} / {OUT_PNG}")
