"""
Paper 46: Figure 1 (2 panels)

Panel (a): Exp A — sg4 vs gate condition.
  Bar chart: ss0, ss5, ss10, ss15, ss20, rand_p60.
  Horizontal line: rand_p60 sg4 (timing-destroyed baseline).
  Markers: write_rate as secondary axis.
  Annotation: t-test result for ss10 vs rand_p60.

Panel (b): Exp B — sg4 vs r_wave for omega_const vs wr_const.
  Two lines: omega-constant (sg4 should be flat) vs WR-constant (varies).
  Secondary x-axis: phi_w values.
  Annotation: CV of omega_const line.
"""
import json, sys, math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

ANALYSIS_FILE = "results/paper46_analysis.json"
FIGURE_FILE   = "paper46_figure1.pdf"

def load():
    try:
        with open(ANALYSIS_FILE) as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Run paper46_experiments.py first to generate {ANALYSIS_FILE}")
        sys.exit(1)

def plot(a):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    fig.suptitle(
        "Paper 46 -- Temporal Gate Invariant and Causal Coordinate Validation\n"
        "Exp A: Timing vs quantity | Exp B: Omega-constant sweep",
        fontsize=11, fontweight='bold'
    )

    # ── Panel (a): Exp A ─────────────────────────────────────────────────────
    ax = axes[0]
    exp_a = a['exp_a']
    cond_order = ['ss0','ss5','ss10','ss15','ss20','rand_p60']
    cond_labels = ['SS=0\n(no gate)', 'SS=5', 'SS=10\n(std)', 'SS=15', 'SS=20',
                   'rand\np=0.60']
    colors = ['#FF9800','#FFC107','#1f77b4','#2196F3','#0D47A1','#E53935']

    x = np.arange(len(cond_order))
    sg4_means = []
    sg4_ses   = []
    wr_means  = []
    for c in cond_order:
        if c in exp_a and isinstance(exp_a[c], dict):
            sg4_means.append(exp_a[c]['sg4_mean'])
            sg4_ses.append(exp_a[c]['sg4_se'])
            wr_means.append(exp_a[c]['wr_mean'])
        else:
            sg4_means.append(0.); sg4_ses.append(0.); wr_means.append(0.)

    bars = ax.bar(x, sg4_means, yerr=sg4_ses, capsize=4,
                  color=colors, alpha=0.85, edgecolor='k', linewidth=0.7)

    # Horizontal reference: rand_p60 mean
    rand_idx = cond_order.index('rand_p60')
    rand_mean = sg4_means[rand_idx]
    ax.axhline(rand_mean, color='#E53935', linewidth=1.5, linestyle='--',
               label=f'rand_p60 baseline ({rand_mean:.4f})')

    # Annotate t-test
    if 'ss10_vs_rand_t' in exp_a:
        t = exp_a['ss10_vs_rand_t']
        p = exp_a['ss10_vs_rand_p']
        ss10_idx = cond_order.index('ss10')
        top = sg4_means[ss10_idx] + sg4_ses[ss10_idx] + 0.001
        stars = '***' if p < 0.001 else ('**' if p < 0.01 else ('*' if p < 0.05 else 'ns'))
        ax.annotate(f't={t:.2f}\n{stars}', xy=(ss10_idx, top),
                    ha='center', va='bottom', fontsize=8, color='#1f77b4',
                    fontweight='bold')
        # bracket ss10 to rand_p60
        y_br = max(sg4_means[ss10_idx], sg4_means[rand_idx]) + sg4_ses[ss10_idx] + 0.003
        ax.annotate('', xy=(rand_idx, y_br), xytext=(ss10_idx, y_br),
                    arrowprops=dict(arrowstyle='-', color='gray'))

    # Secondary axis: write rate
    ax2 = ax.twinx()
    ax2.plot(x, wr_means, 'D--', color='#555555', markersize=5, linewidth=1.2,
             label='write rate/step', alpha=0.7)
    ax2.set_ylabel('Write rate (events/cell/step)', fontsize=9, color='#555555')
    ax2.tick_params(axis='y', labelcolor='#555555')

    ax.set_xticks(x)
    ax.set_xticklabels(cond_labels, fontsize=9)
    ax.set_ylabel('sg4 (zone separation)', fontsize=10)
    ax.set_title('(a) Temporal gate: timing vs quantity\n'
                 'Prediction: ss10 > rand_p60 (same rate, different timing)',
                 fontsize=9)
    ax.legend(fontsize=8, loc='upper left')
    ax.grid(True, alpha=0.3, axis='y')

    note = ("rand_p60: random writes at p=0.60\n"
            "(matched rate, no quiescence requirement)\n"
            "If ss10 >> rand_p60: timing is operative")
    ax.text(0.97, 0.97, note, transform=ax.transAxes, fontsize=7,
            va='top', ha='right', color='#444444',
            bbox=dict(boxstyle='round,pad=0.3', fc='lightyellow', alpha=0.8))

    # ── Panel (b): Exp B ─────────────────────────────────────────────────────
    ax3 = axes[1]
    exp_b = a['exp_b']

    r_vals = sorted(set(v['r_wave'] for v in exp_b.values()))
    phi_vals = [2*r*r + 2*r + 1 for r in r_vals]

    sg4_omega = []
    sg4_wr    = []
    se_omega  = []
    se_wr     = []
    for r in r_vals:
        ko = f'r{r}_omega_const'; kw = f'r{r}_wr_const'
        sg4_omega.append(exp_b[ko]['sg4_mean'] if ko in exp_b else float('nan'))
        sg4_wr.append(   exp_b[kw]['sg4_mean'] if kw in exp_b else float('nan'))
        se_omega.append( exp_b[ko]['sg4_se']   if ko in exp_b else 0.)
        se_wr.append(    exp_b[kw]['sg4_se']   if kw in exp_b else 0.)

    ax3.errorbar(r_vals, sg4_omega, yerr=se_omega, fmt='o-', color='#1f77b4',
                 linewidth=2, markersize=7, capsize=4, label=r'$\Omega$-constant (WR$\times\phi_w$ fixed)')
    ax3.errorbar(r_vals, sg4_wr,    yerr=se_wr,    fmt='s--', color='#FF7043',
                 linewidth=1.8, markersize=7, capsize=4, label='WR-constant (WR=4.8 fixed)')

    # Flat reference line at omega_const mean
    valid_omega = [v for v in sg4_omega if not math.isnan(v)]
    if valid_omega:
        omega_mean = np.mean(valid_omega)
        cv = np.std(valid_omega) / omega_mean if omega_mean > 0 else float('nan')
        ax3.axhline(omega_mean, color='#1f77b4', linewidth=1, linestyle=':',
                    alpha=0.6, label=f'mean={omega_mean:.4f}')
        ax3.text(0.97, 0.05, f'CV($\\Omega$-const)={cv:.3f}\n'
                 f'{"FLAT: $\\Omega$ is operative" if cv < 0.15 else "Not flat"}',
                 transform=ax3.transAxes, fontsize=8, va='bottom', ha='right',
                 color='#1f77b4',
                 bbox=dict(boxstyle='round,pad=0.3', fc='lightblue', alpha=0.7))

    # Annotate phi_w values
    for r, phi in zip(r_vals, phi_vals):
        ax3.annotate(f'$\\phi_w$={phi}', (r, max(sg4_omega[r_vals.index(r)],
                                                   sg4_wr[r_vals.index(r)]) + 0.001),
                     ha='center', va='bottom', fontsize=8, color='gray')

    ax3.set_xticks(r_vals)
    ax3.set_xticklabels([f'r={r}' for r in r_vals])
    ax3.set_xlabel('Wave radius $r_w$', fontsize=10)
    ax3.set_ylabel('sg4 (zone separation)', fontsize=10)
    ax3.set_title('(b) $\\phi_w = 2r^2+2r+1$ causal coordinate\n'
                  'Prediction: $\\Omega$-const line is flat, WR-const varies',
                  fontsize=9)
    ax3.legend(fontsize=8, loc='upper right')
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(FIGURE_FILE, dpi=150, bbox_inches='tight')
    print(f"Saved: {FIGURE_FILE}")


if __name__ == "__main__":
    analysis = load()
    plot(analysis)
