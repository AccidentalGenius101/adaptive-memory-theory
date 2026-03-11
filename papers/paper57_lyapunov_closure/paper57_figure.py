"""Paper 57 figure: Analytic Closure of the Lyapunov Condition.

Panel (a): sigma_w vs r_wave with geometric prediction overlay
Panel (b): na_ratio and C_order vs r_wave, colour-coded by Lyapunov status
Panel (c): sigma_w vs (1 - f_int) scatter — test of linear noise model
"""
import json, math, numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path

W_ZONE = 20

def f_interior(rw, wz=W_ZONE):
    return max(0.0, 1.0 - 2*rw/wz)

def delta(rw, wz=W_ZONE):
    return wz/rw

with open(Path(__file__).parent / 'results/paper57_analysis.json') as f:
    d = json.load(f)

RWAVE = [2, 3, 4, 5, 6, 7, 8, 10]

rw_arr  = np.array(RWAVE, float)
del_arr = W_ZONE / rw_arr
fi_arr  = np.array([f_interior(r) for r in RWAVE])
sw_arr  = np.array([d[f'rw{r}']['final_sigma_w'] for r in RWAVE])
na_arr  = np.array([d[f'rw{r}']['final_na']      for r in RWAVE])
co_arr  = np.array([d[f'rw{r}']['final_c_order'] for r in RWAVE])

# sigma_w se
sw_se = np.array([d[f'rw{r}']['sigma_w_se'][-1] if 'sigma_w_se' in d[f'rw{r}'] else 0.0 for r in RWAVE])
na_se = np.array([d[f'rw{r}']['na_se'][-1]      if 'na_se'      in d[f'rw{r}'] else 0.0 for r in RWAVE])

# colours: identity (C_order>0 and na>1) = green, marginal = orange, failure = red
def colour(co, na):
    if co > 0.005 and na > 1.0: return '#2ca02c'
    if na > 1.0:                 return '#ff7f0e'
    return '#d62728'

cols = [colour(co, na) for co, na in zip(co_arr, na_arr)]

fig = plt.figure(figsize=(17, 5.0))
gs  = gridspec.GridSpec(1, 3, figure=fig, wspace=0.38)

# ── Panel (a): sigma_w vs r_wave ───────────────────────────────────────────────
ax_a = fig.add_subplot(gs[0, 0])

# Geometric prediction: sigma_w ~ C * (2/delta) = C * r_wave/W_zone
# fit C from the upper end where f_int ~ 0
C_fit = float(np.mean(sw_arr[4:] / (2*rw_arr[4:]/W_ZONE)))
rw_fine = np.linspace(2, 11, 200)
sw_pred = C_fit * (2*rw_fine / W_ZONE)
ax_a.plot(rw_fine, sw_pred, 'k--', lw=1.2, alpha=0.6, label=f'geometric: C*2/delta (C={C_fit:.3f})')

ax_a.axvline(W_ZONE/2, color='grey', lw=0.8, ls=':', alpha=0.7)
ax_a.text(W_ZONE/2 + 0.1, 0.09, 'delta_c=2\n(W_zone/2)', fontsize=7.5, color='grey')

for rw, sw, c in zip(rw_arr, sw_arr, cols):
    ax_a.scatter([rw], [sw], color=c, s=80, zorder=5)
ax_a.plot(rw_arr, sw_arr, color='#1f77b4', lw=1.4, zorder=3)

ax_a.set_xlabel('r_wave', fontsize=10)
ax_a.set_ylabel('sigma_w (within-zone noise floor)', fontsize=9)
ax_a.set_title('(a) sigma_w rises with r_wave\n'
               'Dashed = geometric prediction sigma_w ~ 2r/W_zone\n'
               'Deviation at small r shows interior-cell protection',
               fontsize=8.5, fontweight='bold')
ax_a.legend(fontsize=7.5)
ax_a.set_xlim(1.5, 10.5); ax_a.set_ylim(0, 0.13)

# ── Panel (b): na_ratio and C_order vs r_wave ──────────────────────────────────
ax_b  = fig.add_subplot(gs[0, 1])
ax_b2 = ax_b.twinx()

ax_b.axhline(1.0, color='k', lw=0.9, ls='--', alpha=0.5)
ax_b2.axhline(0.0, color='grey', lw=0.7, ls=':', alpha=0.5)

ax_b.plot(rw_arr, na_arr, color='#1f77b4', lw=1.8, marker='o', ms=8, label='na_ratio', zorder=5)
ax_b2.plot(rw_arr, co_arr, color='#d62728', lw=1.5, marker='^', ms=7, ls='--',
           label='C_order', zorder=4)

ax_b.set_xlabel('r_wave', fontsize=10)
ax_b.set_ylabel('na_ratio', fontsize=9, color='#1f77b4')
ax_b2.set_ylabel('C_order', fontsize=9, color='#d62728')
ax_b.tick_params(axis='y', labelcolor='#1f77b4')
ax_b2.tick_params(axis='y', labelcolor='#d62728')

ax_b.set_title('(b) na_ratio and C_order vs r_wave\n'
               'Both degrade past r=4; r=8 alone fails na<1.\n'
               'r=10 (delta=2) partially recovers -- geometric model incomplete',
               fontsize=8.5, fontweight='bold')
ax_b.set_xlim(1.5, 10.5)
ax_b.set_ylim(0.8, 1.45)
ax_b2.set_ylim(-0.06, 0.08)

lines1, labs1 = ax_b.get_legend_handles_labels()
lines2, labs2 = ax_b2.get_legend_handles_labels()
ax_b.legend(lines1+lines2, labs1+labs2, fontsize=7.5, loc='lower left')

# Annotate delta_c region
ax_b.axvspan(9.5, 10.5, alpha=0.08, color='red')
ax_b.text(9.6, 1.38, 'delta=2', fontsize=7, color='red')

# ── Panel (c): sigma_w vs geometric boundary fraction (1-f_int) ───────────────
ax_c = fig.add_subplot(gs[0, 2])
bnd_arr = 1.0 - fi_arr  # = min(1, 2/delta)

ax_c.scatter(bnd_arr, sw_arr, c=cols, s=90, zorder=5)
for rw, bnd, sw in zip(RWAVE, bnd_arr, sw_arr):
    ax_c.annotate(f'r={rw}', (bnd, sw), textcoords='offset points',
                  xytext=(5, 3), fontsize=7.5)

# linear fit
m, b_off = np.polyfit(bnd_arr, sw_arr, 1)
x_fit = np.linspace(0, 1, 100)
ax_c.plot(x_fit, m*x_fit + b_off, 'k--', lw=1.2, alpha=0.6,
          label=f'linear fit: slope={m:.3f}')

ax_c.set_xlabel('boundary fraction (1 - f_interior) = 2/delta', fontsize=9)
ax_c.set_ylabel('sigma_w', fontsize=9)
ax_c.set_title('(c) sigma_w vs boundary exposure\n'
               'Broadly linear but not perfect -- causal purity\n'
               'is the correct generalisation (see text)',
               fontsize=8.5, fontweight='bold')
ax_c.legend(fontsize=7.5)
ax_c.set_xlim(-0.02, 1.02); ax_c.set_ylim(0, 0.13)

fig.suptitle(
    'Paper 57: Analytic Closure of the VCML Lyapunov Condition\n'
    'sigma_w scales with boundary exposure (geometric approximation). '
    'Full closure requires causal purity p_same, not just distance to edge.',
    fontsize=9, fontweight='bold', y=1.03
)

out = Path(__file__).parent / 'paper57_figure1.pdf'
plt.savefig(str(out), bbox_inches='tight')
print(f'Saved: {out}')
