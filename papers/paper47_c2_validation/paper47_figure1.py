"""
Paper 47: Figure 1 (4 panels)

Panel (a): Exp A -- sg4 vs delta (bar chart, secondary axis na_ratio).
Panel (b): Exp A -- na_ratio vs delta, with delta=1 reference line.
Panel (c): Exp B -- fidelity matrix (adv_amp x gate condition).
Panel (d): Exp B -- fidelity trajectories for adv_amp=0.25, all gates.
"""
import json, sys, math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

ANALYSIS_FILE = "results/paper47_analysis.json"
RESULTS_FILE  = "results/paper47_results.json"
FIGURE_FILE   = "paper47_figure1.pdf"

def load():
    for f in [ANALYSIS_FILE, RESULTS_FILE]:
        if not __import__('os').path.exists(f):
            print(f"Run paper47_experiments.py first to generate {f}")
            sys.exit(1)
    with open(ANALYSIS_FILE) as f:
        analysis = json.load(f)
    with open(RESULTS_FILE) as f:
        results = json.load(f)
    return analysis, results

N_ZONES_SWEEP   = [2, 4, 5, 8, 10]
SEEDS           = list(range(5))
ADV_AMPS        = [0.0, 0.125, 0.25, 0.50]
GATE_CONDITIONS = ['ss0','ss5','ss10','ss15','ss20','rand_p60']
GATE_LABELS     = ['SS=0','SS=5','SS=10','SS=15','SS=20','rand\np=0.60']
R_WAVE_STD      = 2
HALF            = 40

def plot(analysis, results):
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    fig.suptitle(
        "Paper 47 -- C2 Coordinate Validation\n"
        "Exp A: delta-sweep | Exp B: gate x adversarial pressure",
        fontsize=11, fontweight='bold'
    )

    # ── Panel (a): Exp A -- sg4 vs delta ─────────────────────────────────────
    ax = axes[0, 0]
    exp_a = analysis['exp_a']
    deltas    = []; sg4_means = []; sg4_ses = []; na_ratios = []
    for nz in N_ZONES_SWEEP:
        zw = HALF // nz
        d  = zw / R_WAVE_STD
        key = str(nz)
        if key not in exp_a: continue
        deltas.append(d)
        sg4_means.append(exp_a[key]['sg4_mean'])
        sg4_ses.append(exp_a[key]['sg4_se'])
        na_ratios.append(exp_a[key]['na_ratio'])

    x = np.arange(len(deltas))
    colors = ['#5C6BC0','#26A69A','#66BB6A','#FFA726','#EF5350']
    bars = ax.bar(x, sg4_means, yerr=sg4_ses, capsize=4,
                  color=colors[:len(deltas)], alpha=0.85, edgecolor='k', linewidth=0.7)
    ax2 = ax.twinx()
    ax2.plot(x, na_ratios, 'D--', color='#555', markersize=6, linewidth=1.4,
             label='nonadj/adj', alpha=0.85)
    ax2.axhline(1.0, color='#888', linewidth=1.0, linestyle=':')
    ax2.set_ylabel('nonadj/adj ratio', fontsize=9, color='#555')
    ax2.tick_params(axis='y', labelcolor='#555')
    ax.set_xticks(x)
    ax.set_xticklabels([f'delta={d:.1f}\n(NZ={nz})' for d, nz in
                         zip(deltas, N_ZONES_SWEEP[:len(deltas)])], fontsize=8)
    ax.set_ylabel('sg4 (zone separation)', fontsize=10)
    ax.set_title('(a) delta-sweep: sg4 vs zone width / wave radius\n'
                 'Prediction: sg4 peaks at delta~4-5, na_ratio > 1 for proper encoding',
                 fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')
    note = ("r_wave=2 fixed; Omega=62.4 fixed\n"
            "na_ratio < 1: wave-level position, not zone-level\n"
            "na_ratio > 1: monotonic zone gradient (correct)")
    ax.text(0.97, 0.97, note, transform=ax.transAxes, fontsize=7,
            va='top', ha='right', color='#444',
            bbox=dict(boxstyle='round,pad=0.3', fc='lightyellow', alpha=0.8))

    # ── Panel (b): Exp A -- na_ratio vs delta ────────────────────────────────
    ax3 = axes[0, 1]
    valid = [(d, r) for d, r in zip(deltas, na_ratios) if not math.isnan(r)]
    if valid:
        d_vals, r_vals = zip(*valid)
        ax3.plot(d_vals, r_vals, 'o-', color='#1f77b4', linewidth=2, markersize=8,
                 label='na_ratio(delta)')
        ax3.fill_between(d_vals, 1.0, r_vals, alpha=0.15, color='#1f77b4',
                         where=[r > 1.0 for r in r_vals], label='na_ratio > 1 (zone encoding)')
        ax3.fill_between(d_vals, 1.0, r_vals, alpha=0.15, color='#E53935',
                         where=[r < 1.0 for r in r_vals], label='na_ratio < 1 (sub-zone)')
    ax3.axhline(1.0, color='k', linewidth=1.5, linestyle='--', label='random baseline')
    ax3.axvspan(0, 3, alpha=0.08, color='#E53935', label='bleed regime (delta < r_wave)')
    ax3.set_xlabel('delta = W_zone / r_wave', fontsize=10)
    ax3.set_ylabel('nonadj/adj zone distance ratio', fontsize=10)
    ax3.set_title('(b) Zone encoding quality vs delta\n'
                  'delta > 1: wave fits in zone; delta > 5: clean zone gradient',
                  fontsize=9)
    ax3.legend(fontsize=7, loc='lower right')
    ax3.grid(True, alpha=0.3)

    # ── Panel (c): Exp B -- fidelity matrix ──────────────────────────────────
    ax4 = axes[1, 0]
    exp_b = analysis['exp_b']
    matrix = np.full((len(ADV_AMPS), len(GATE_CONDITIONS)), float('nan'))
    for i, adv in enumerate(ADV_AMPS):
        for j, cond in enumerate(GATE_CONDITIONS):
            k = f"{adv:.3f}_{cond}"
            if k in exp_b:
                matrix[i, j] = exp_b[k]['fidelity_mean']
    im = ax4.imshow(matrix, aspect='auto', cmap='RdYlGn', vmin=-0.3, vmax=0.3)
    ax4.set_xticks(range(len(GATE_CONDITIONS)))
    ax4.set_xticklabels(GATE_LABELS, fontsize=8)
    ax4.set_yticks(range(len(ADV_AMPS)))
    ax4.set_yticklabels([f'adv={a:.3f}' for a in ADV_AMPS], fontsize=8)
    ax4.set_title('(c) Final fidelity: gate x adversarial pressure\n'
                  'Green=preserved, Red=imprinted adversarial; T_ENCODE=2000',
                  fontsize=9)
    for i in range(len(ADV_AMPS)):
        for j in range(len(GATE_CONDITIONS)):
            if not math.isnan(matrix[i, j]):
                ax4.text(j, i, f'{matrix[i,j]:.3f}', ha='center', va='center',
                         fontsize=7.5, color='black', fontweight='bold')
    plt.colorbar(im, ax=ax4, shrink=0.85, label='fidelity')
    note4 = ("T_ENCODE=2000 is pre-commitment (t*~2000).\n"
             "Fidelity measures commitment rate,\n"
             "not adversarial resistance.")
    ax4.text(1.25, 0.05, note4, transform=ax4.transAxes, fontsize=7.5,
             va='bottom', ha='left', color='#444',
             bbox=dict(boxstyle='round,pad=0.3', fc='lightcyan', alpha=0.8))

    # ── Panel (d): Exp B -- fidelity trajectories for adv_amp=0.25 ──────────
    ax5 = axes[1, 1]
    colors_b = ['#FF5252','#FF9800','#1f77b4','#2196F3','#0D47A1','#E53935']
    T_ADV = 1000
    cps   = list(range(100, T_ADV+1, 100))
    for j, (cond, lab, col) in enumerate(zip(GATE_CONDITIONS, GATE_LABELS, colors_b)):
        trajs = []
        for s in SEEDS:
            k = f"B,amp0.250,{cond},{s}"
            if k in results and 'fidelity_traj' in results[k]:
                trajs.append(results[k]['fidelity_traj'])
        if not trajs: continue
        arr = np.array(trajs)
        m = arr.mean(0); se = arr.std(0) / np.sqrt(len(arr))
        t_axis = [cps[i] for i in range(len(m))]
        ax5.plot(t_axis, m, '-', color=col, linewidth=1.8,
                 label=lab.replace('\n', ' '))
        ax5.fill_between(t_axis, m-se, m+se, alpha=0.12, color=col)
    ax5.axhline(0.0, color='k', linewidth=1.0, linestyle='--')
    ax5.axhline(0.5, color='gray', linewidth=0.8, linestyle=':', alpha=0.6)
    ax5.set_xlabel('Adversarial phase step', fontsize=10)
    ax5.set_ylabel('Fidelity (cosine sim vs T=2000)', fontsize=10)
    ax5.set_title('(d) Fidelity trajectories: adv_amp=0.25\n'
                  'All gates converge to near-zero; commitment epoch visible',
                  fontsize=9)
    ax5.legend(fontsize=7.5, loc='upper right', ncol=2)
    ax5.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(FIGURE_FILE, dpi=150, bbox_inches='tight')
    print(f"Saved: {FIGURE_FILE}")


if __name__ == "__main__":
    analysis, results = load()
    plot(analysis, results)
