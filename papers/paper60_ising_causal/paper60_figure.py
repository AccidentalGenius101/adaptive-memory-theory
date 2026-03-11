"""Paper 60 figure: Causal Purity as an Independent Control Parameter in Ising."""
import json, numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize
from pathlib import Path

with open(Path(__file__).parent / 'results/paper60_analysis.json') as f:
    d = json.load(f)

T_SWEEP       = [1.0, 1.5, 2.0, 2.27, 3.0, 4.0]
PCAUSAL_SWEEP = [0.50, 0.60, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.00]
TC            = 2.269

def get(T, pc, key):
    k = f'T{T:.2f}_pc{pc:.2f}'
    return d[k][key] if k in d else float('nan')

fig = plt.figure(figsize=(17, 5.0))
gs  = gridspec.GridSpec(1, 3, figure=fig, wspace=0.38)

pc_arr = np.array(PCAUSAL_SWEEP)
T_arr  = np.array(T_SWEEP)

colors_T = plt.cm.plasma(np.linspace(0.1, 0.9, len(T_SWEEP)))

# ── Panel (a): S_phi vs P_causal for three key temperatures ───────────────────
ax_a = fig.add_subplot(gs[0, 0])

highlight_T = [1.0, 2.27, 3.0, 4.0]
lws         = [1.2, 1.2, 2.2, 1.2]
alphas      = [0.5, 0.7, 1.0, 0.5]

for T, lw, alpha in zip(highlight_T, lws, alphas):
    S = [get(T, pc, 'final_S_phi') for pc in PCAUSAL_SWEEP]
    idx = T_SWEEP.index(T)
    label = f'T={T:.2f}' + (' (disordered, m=0.07)' if T == 3.0 else
                             ' (T$_c$=2.27)'         if T == 2.27 else
                             ' (ordered, m=1.00)'     if T == 1.0 else
                             ' (very disordered)')
    ax_a.plot(pc_arr, S, 'o-', color=colors_T[idx], lw=lw, ms=6, alpha=alpha,
              label=label)

ax_a.axvline(0.5, color='gray', ls='--', lw=0.9, alpha=0.5, label='P_causal=0.5 (50% noise)')
ax_a.set_xlabel('$P_{\\rm causal}$', fontsize=11)
ax_a.set_ylabel('$S_{\\phi}$ (phi-order)', fontsize=10)
ax_a.set_title('(a) $S_\\phi$ increases with $P_{\\rm causal}$\n'
               'at ALL temperatures, including $T > T_c$.\n'
               'Phi-order exists even when Ising spins are disordered.',
               fontsize=8.5, fontweight='bold')
ax_a.legend(fontsize=7, loc='upper left')
ax_a.set_ylim(0, 1.05); ax_a.set_xlim(0.48, 1.02)

# ── Panel (b): m vs T (T-controlled, P_causal-independent) ───────────────────
ax_b = fig.add_subplot(gs[0, 1])

for i, pc in enumerate([0.50, 0.75, 1.00]):
    m_vals = [get(T, pc, 'final_m') for T in T_SWEEP]
    ax_b.plot(T_arr, m_vals, 'o-', lw=1.8, ms=6,
              label=f'$P_{{\\rm causal}}$={pc:.2f}')

ax_b.axvline(TC, color='red', ls='--', lw=1.1, label=f'$T_c$={TC}', alpha=0.7)
ax_b.set_xlabel('Temperature $T$', fontsize=11)
ax_b.set_ylabel('Magnetization $m$', fontsize=10)
ax_b.set_title('(b) Magnetization $m$ controlled by $T$ only.\n'
               'Three P_causal values give identical $m$.\n'
               'Standard Ising transition at $T_c \\approx 2.27$.',
               fontsize=8.5, fontweight='bold')
ax_b.legend(fontsize=8, loc='upper right')
ax_b.set_ylim(-0.05, 1.10)

# ── Panel (c): Phase diagram heatmap S_phi(T, P_causal) ──────────────────────
ax_c = fig.add_subplot(gs[0, 2])

S_grid = np.array([[get(T, pc, 'final_S_phi') for pc in PCAUSAL_SWEEP]
                    for T in T_SWEEP])
m_grid = np.array([[get(T, pc, 'final_m') for pc in PCAUSAL_SWEEP]
                    for T in T_SWEEP])

im = ax_c.imshow(S_grid, aspect='auto', origin='lower',
                  extent=[pc_arr[0]-0.025, pc_arr[-1]+0.025,
                           T_arr[0]-0.25,  T_arr[-1]+0.25],
                  cmap='YlOrRd', vmin=0, vmax=1)
plt.colorbar(im, ax=ax_c, label='$S_\\phi$ (phi-order)')

# Overlay contour for m=0.5 (Ising transition)
ax_c.axhline(TC, color='cyan', lw=2, ls='--', label='$T_c$ (Ising transition)')

# Mark cells where m < 0.15 (disordered) with text
for i, T in enumerate(T_SWEEP):
    for j, pc in enumerate(PCAUSAL_SWEEP):
        m = m_grid[i, j]
        if m < 0.15:
            ax_c.text(pc, T, f'{S_grid[i,j]:.2f}', ha='center', va='center',
                      fontsize=6, color='white', fontweight='bold')

ax_c.set_xlabel('$P_{\\rm causal}$', fontsize=11)
ax_c.set_ylabel('Temperature $T$', fontsize=10)
ax_c.set_title('(c) Phase diagram: $S_\\phi(T, P_{\\rm causal})$.\n'
               'Above $T_c$ (dashed cyan): $m \\approx 0$, yet $S_\\phi > 0$.\n'
               'White numbers = $S_\\phi$ in the disordered spin region.\n'
               'Two independent order params, two independent control params.',
               fontsize=8.5, fontweight='bold')
ax_c.legend(fontsize=7.5, loc='upper left')

fig.suptitle(
    'Paper 60: Causal Purity as an Independent Control Parameter — Ising + VCSM-lite\n'
    'Magnetization $m$ is controlled by temperature $T$ (standard physics). '
    'Phi-order $S_\\phi$ is controlled by causal purity $P_{\\rm causal}$ (new). '
    'At $T > T_c$: $m \\approx 0$ (spins disordered) but $S_\\phi > 0$ (phi ordered). '
    'Cross-substrate confirmation of the causal-purity universality class.',
    fontsize=9, fontweight='bold', y=1.04
)

out = Path(__file__).parent / 'paper60_figure1.pdf'
plt.savefig(str(out), bbox_inches='tight')
print(f'Saved: {out}')
