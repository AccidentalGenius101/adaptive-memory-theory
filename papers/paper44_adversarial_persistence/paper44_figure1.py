"""paper44_figure1.py -- Adversarial Persistence Test Figure

Two panels:
  (a) Fidelity trajectories over T_ADV for all three conditions (mean ± SE)
  (b) Late-phase fidelity (T_ADV=2000) bar chart with individual seed points
      to show: adaptive > rigid > passive at long timescale

Key interpretive annotations:
  - adaptive: maintains near-original structure (fidelity oscillates near zero but
    stays positive at T=2000) -- copy-forward resistance
  - passive: gradually encodes adversarial pattern (fidelity goes negative) -- imprinting
  - rigid: frozen at original structure (fidelity positive throughout) -- no plasticity
"""
import json, numpy as np, os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

RESULTS_FILE = os.path.join(os.path.dirname(__file__), "results", "paper44_results.json")
OUT_PDF = os.path.join(os.path.dirname(__file__), "paper44_figure1.pdf")
OUT_PNG = os.path.join(os.path.dirname(__file__), "paper44_figure1.png")

COND_ORDER  = ['adaptive', 'passive', 'rigid']
COND_LABELS = {'adaptive': 'Adaptive (copy-forward)',
               'passive':  'Passive (field-decay only)',
               'rigid':    'Rigid (no turnover)'}
COND_COLORS = {'adaptive': '#2196F3',   # blue
               'passive':  '#F44336',   # red
               'rigid':    '#4CAF50'}   # green

with open(RESULTS_FILE) as f:
    all_results = json.load(f)

# Organise by condition
by_cond = {c: [r for r in all_results if r['cond']==c] for c in COND_ORDER}

# ── Compute mean ± SE trajectories ────────────────────────────────────────────
def traj_stats(results):
    ts  = results[0]['ts_adv']
    mat = np.array([r['fidelity'] for r in results])  # seeds x timesteps
    return ts, mat.mean(0), mat.std(0)/np.sqrt(len(results))

# ── Figure ────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.subplots_adjust(wspace=0.35)

# ── Panel (a): Fidelity over time ─────────────────────────────────────────────
ax = axes[0]
ax.axhline(0, color='k', lw=0.8, ls='--', alpha=0.4)
ax.axhline(1, color='gray', lw=0.6, ls=':', alpha=0.4)

for cond in COND_ORDER:
    rs = by_cond[cond]
    ts, mean, se = traj_stats(rs)
    c = COND_COLORS[cond]
    ax.plot(ts, mean, lw=2.0, color=c, label=COND_LABELS[cond])
    ax.fill_between(ts, mean-se, mean+se, alpha=0.15, color=c)

# Annotations
ax.annotate("Copy-forward maintains\noriginal structure",
            xy=(800, by_cond['adaptive'][0]['fidelity']
                [by_cond['adaptive'][0]['ts_adv'].index(800)]),
            xytext=(1000, 0.35), fontsize=8,
            color=COND_COLORS['adaptive'],
            arrowprops=dict(arrowstyle='->', color=COND_COLORS['adaptive'], lw=1))

ax.annotate("Passive encodes\nadversarial pattern",
            xy=(1500, np.mean([r['fidelity'][r['ts_adv'].index(1500)]
                               for r in by_cond['passive']])),
            xytext=(700, -0.55), fontsize=8,
            color=COND_COLORS['passive'],
            arrowprops=dict(arrowstyle='->', color=COND_COLORS['passive'], lw=1))

ax.set_xlabel("Adversarial input duration (steps after zone-flip)", fontsize=11)
ax.set_ylabel("Fidelity (cosine similarity to original encoding)", fontsize=10)
ax.set_title("(a) Structure retention under adversarial input", fontsize=11)
ax.legend(fontsize=9, loc='lower right')
ax.set_xlim(0, 2000)
ax.set_ylim(-0.85, 0.75)
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

# ── Panel (b): Late-phase fidelity bar chart ──────────────────────────────────
ax2 = axes[1]
T_LATE = 2000

bar_x = [0, 1, 2]
bar_means = []
bar_ses   = []
all_seeds = []
for cond in COND_ORDER:
    rs = by_cond[cond]
    vals = [r['fidelity'][r['ts_adv'].index(T_LATE)] for r in rs
            if T_LATE in r['ts_adv']]
    bar_means.append(float(np.mean(vals)))
    bar_ses.append(float(np.std(vals)/np.sqrt(len(vals))))
    all_seeds.append(vals)

bars = ax2.bar(bar_x, bar_means,
               color=[COND_COLORS[c] for c in COND_ORDER],
               yerr=bar_ses, capsize=5, alpha=0.8, width=0.55)

for xi, seeds in zip(bar_x, all_seeds):
    jitter = np.linspace(-0.15, 0.15, len(seeds))
    for xj, sv in zip(jitter, seeds):
        ax2.scatter(xi+xj, sv, color='k', s=18, zorder=5, alpha=0.7)

ax2.axhline(0, color='k', lw=0.8, ls='--', alpha=0.4)

# Significance annotations
def bracket(ax, x1, x2, y, h, text):
    ax.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.0, color='k')
    ax.text((x1+x2)/2, y+h, text, ha='center', va='bottom', fontsize=9)

ymax = max(bar_means) + max(bar_ses) + 0.05
bracket(ax2, 0, 1, ymax, 0.04, f'Adv. vs. Passive')

ax2.set_xticks(bar_x)
ax2.set_xticklabels(['Adaptive', 'Passive', 'Rigid'], fontsize=10)
ax2.set_ylabel("Fidelity at T_ADV=2000 steps", fontsize=10)
ax2.set_title("(b) Late-phase structure retention", fontsize=11)
ax2.spines['top'].set_visible(False); ax2.spines['right'].set_visible(False)

# Table inset below the bars
from matplotlib.table import Table
cell_text = []
headers = ['Cond.', 'Fid@200', 'Fid@800', 'Fid@2000', 'Interpretation']
interp  = {'adaptive': 'Memory: resists + recovers',
           'passive':  'Imprinting: encodes new',
           'rigid':    'Rigid: frozen, no plasticity'}

def fid_mean_at(cond, t_tgt):
    vals = [r['fidelity'][r['ts_adv'].index(t_tgt)]
            for r in by_cond[cond] if t_tgt in r['ts_adv']]
    return float(np.mean(vals)) if vals else float('nan')

table_data = []
for cond in COND_ORDER:
    table_data.append([cond[:8], f"{fid_mean_at(cond,200):.3f}",
                       f"{fid_mean_at(cond,800):.3f}",
                       f"{fid_mean_at(cond,2000):.3f}",
                       interp[cond]])

print("\nFidelity table:")
print(f"{'Cond':<10} {'@200':>8} {'@800':>8} {'@2000':>8}  Interpretation")
for row in table_data:
    print(f"{row[0]:<10} {row[1]:>8} {row[2]:>8} {row[3]:>8}  {row[4]}")

fig.suptitle(
    "Paper 44: Adversarial Persistence Test\n"
    r"Adaptive mechanism (copy-forward) maintains original zone structure" "\n"
    r"under adversarial input; passive mechanism encodes new statistics.",
    fontsize=10, y=1.01)

plt.tight_layout()
plt.savefig(OUT_PDF, bbox_inches='tight')
plt.savefig(OUT_PNG, bbox_inches='tight', dpi=150)
print(f"\nSaved: {OUT_PDF}")
print(f"Saved: {OUT_PNG}")
