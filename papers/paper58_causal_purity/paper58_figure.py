"""Paper 58 figure: Empirical Causal Purity and the Two-Dimensional Identity Condition.

Panel (a): P_causal(empirical) vs r_wave for C_perturb and C_ref — nearly identical
Panel (b): sigma_w vs (1-P_causal): linear for C_perturb, breaks for C_ref
Panel (c): sigma_w comparison C_perturb vs C_ref at same r_wave — write timing gap
"""
import json, math, numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path

with open(Path(__file__).parent / 'results/paper58_analysis.json') as f:
    d = json.load(f)

RWAVE = [2, 4, 8, 10]

def get(mode, rw, key):
    k = f'{mode}_rw{rw}'
    return d[k][key] if k in d else float('nan')

pc_cp  = [get('CP',  r, 'final_p_causal') for r in RWAVE]
pc_ref = [get('ref', r, 'final_p_causal') for r in RWAVE]
sw_cp  = [get('CP',  r, 'final_sigma_w')  for r in RWAVE]
sw_ref = [get('ref', r, 'final_sigma_w')  for r in RWAVE]
na_cp  = [get('CP',  r, 'final_na')       for r in RWAVE]
na_ref = [get('ref', r, 'final_na')       for r in RWAVE]

imp_cp  = [1-p for p in pc_cp]
imp_ref = [1-p for p in pc_ref]

fig = plt.figure(figsize=(17, 5.0))
gs  = gridspec.GridSpec(1, 3, figure=fig, wspace=0.38)

# ── Panel (a): P_causal vs r_wave ──────────────────────────────────────────────
ax_a = fig.add_subplot(gs[0, 0])
rw_arr = np.array(RWAVE, float)

ax_a.plot(rw_arr, pc_cp,  'o-', color='#d62728', lw=1.8, ms=8, label='C_perturb')
ax_a.plot(rw_arr, pc_ref, 's--', color='#1f77b4', lw=1.8, ms=8, label='C_ref')

# annotate: identity (open circle) vs failure (x)
for rw, na, pc in zip(RWAVE, na_cp, pc_cp):
    m = '*' if na > 1 else 'x'
    ax_a.annotate(f'na={na:.2f}{m}', (rw, pc), textcoords='offset points',
                  xytext=(3, 6), fontsize=7, color='#d62728')
for rw, na, pc in zip(RWAVE, na_ref, pc_ref):
    m = '*' if na > 1 else 'x'
    ax_a.annotate(f'na={na:.2f}{m}', (rw, pc), textcoords='offset points',
                  xytext=(3, -14), fontsize=7, color='#1f77b4')

ax_a.set_xlabel('r_wave', fontsize=10)
ax_a.set_ylabel('P_causal (empirical)', fontsize=9)
ax_a.set_title('(a) Spatial P_causal nearly identical\n'
               'between C_perturb and C_ref at same r_wave.\n'
               '* = identity (na>1), x = failure',
               fontsize=8.5, fontweight='bold')
ax_a.legend(fontsize=8); ax_a.set_ylim(0.85, 1.01); ax_a.set_xlim(1, 11)

# ── Panel (b): sigma_w vs (1-P_causal) ────────────────────────────────────────
ax_b = fig.add_subplot(gs[0, 1])

# C_perturb: linear fit
m_cp, b_cp = np.polyfit(imp_cp, sw_cp, 1)
x_fit = np.linspace(0, 0.15, 100)
ax_b.plot(x_fit, m_cp*x_fit + b_cp, '--', color='#d62728', lw=1.2, alpha=0.7,
          label=f'CP linear (slope={m_cp:.2f})')

ax_b.scatter(imp_cp, sw_cp, marker='o', s=90, color='#d62728', label='C_perturb', zorder=5)
ax_b.scatter(imp_ref, sw_ref, marker='s', s=90, color='#1f77b4', label='C_ref', zorder=5)

for rw, x, y in zip(RWAVE, imp_cp, sw_cp):
    ax_b.annotate(f'r={rw}', (x, y), textcoords='offset points',
                  xytext=(4, 4), fontsize=7, color='#d62728')
for rw, x, y in zip(RWAVE, imp_ref, sw_ref):
    ax_b.annotate(f'r={rw}', (x, y), textcoords='offset points',
                  xytext=(4, -12), fontsize=7, color='#1f77b4')

ax_b.set_xlabel('1 - P_causal (spatial impurity)', fontsize=9)
ax_b.set_ylabel('sigma_w', fontsize=9)
ax_b.set_title('(b) sigma_w ~ A*(1-P_causal) for C_perturb.\n'
               'C_ref: 3-10x above linear prediction.\n'
               'Temporal causal purity is the extra dimension.',
               fontsize=8.5, fontweight='bold')
ax_b.legend(fontsize=7.5); ax_b.set_xlim(-0.005, 0.135); ax_b.set_ylim(0, 0.27)

# ── Panel (c): sigma_w side-by-side at same r_wave ────────────────────────────
ax_c = fig.add_subplot(gs[0, 2])
x = np.arange(len(RWAVE)); w = 0.35
bars_cp  = ax_c.bar(x - w/2, sw_cp,  w, color='#d62728', alpha=0.8, label='C_perturb')
bars_ref = ax_c.bar(x + w/2, sw_ref, w, color='#1f77b4', alpha=0.8, label='C_ref')
ax_c.axhline(0.03, color='k', lw=0.9, ls=':', alpha=0.6)
ax_c.text(3.6, 0.031, 'sigma_w < 0.03\nidentity zone', fontsize=6.5, color='k')

# Mark identity outcomes
for i, (na, sw) in enumerate(zip(na_cp, sw_cp)):
    if na > 1.0: ax_c.annotate('*', (x[i]-w/2, sw+0.004), ha='center', fontsize=10, color='#d62728')
for i, (na, sw) in enumerate(zip(na_ref, sw_ref)):
    if na > 1.0: ax_c.annotate('*', (x[i]+w/2, sw+0.004), ha='center', fontsize=10, color='#1f77b4')

# Ratio annotations
for i, (s1, s2) in enumerate(zip(sw_cp, sw_ref)):
    ratio = s2/s1 if s1 > 1e-5 else float('nan')
    ax_c.annotate(f'{ratio:.1f}x', (x[i], max(s1,s2)+0.012), ha='center',
                  fontsize=7.5, color='grey')

ax_c.set_xticks(x); ax_c.set_xticklabels([f'r={r}' for r in RWAVE])
ax_c.set_ylabel('sigma_w', fontsize=9)
ax_c.set_title('(c) sigma_w ratio C_ref/C_perturb at same r_wave.\n'
               '* = identity (na>1). Same spatial P_causal,\n'
               'different temporal purity -> different sigma_w.',
               fontsize=8.5, fontweight='bold')
ax_c.legend(fontsize=8); ax_c.set_ylim(0, 0.29)

fig.suptitle(
    'Paper 58: Empirical Causal Purity and the Two-Dimensional Identity Condition\n'
    'Spatial P_causal (zone matching) is equal across write policies at same r_wave. '
    'Temporal causal purity (write timing during vs after wave) is the missing dimension: '
    'it explains the 3-10x sigma_w gap between C_perturb and C_ref.',
    fontsize=9, fontweight='bold', y=1.03
)

out = Path(__file__).parent / 'paper58_figure1.pdf'
plt.savefig(str(out), bbox_inches='tight')
print(f'Saved: {out}')
