"""
Paper 48: Figure 1 (4 panels)

Panel (a): Exp A -- sg4 vs gate condition (replication of Paper 46 finding).
Panel (b): Exp A -- SNR (sg4/sigma_w) and decode_norm vs gate condition.
           Shows gate FLIP on SNR even though no flip on decode_acc.
Panel (c): Exp B -- sg4 vs delta (goes wrong direction).
Panel (d): Exp B -- sg_C (decode_norm) vs delta (goes right direction).
           Together (c)+(d) show the dissociation.
"""
import json, sys, math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ANALYSIS_FILE = "results/paper48_analysis.json"
FIGURE_FILE   = "paper48_figure1.pdf"

def load():
    if not __import__('os').path.exists(ANALYSIS_FILE):
        print(f"Run paper48_experiments.py first"); sys.exit(1)
    with open(ANALYSIS_FILE) as f:
        return json.load(f)

GATE_ORDER  = ['ss0','ss5','ss10','ss15','ss20','rand_p60']
GATE_LABELS = ['SS=0','SS=5','SS=10\n(std)','SS=15','SS=20','rand\np=0.60']
COLORS_GATE = ['#FF9800','#FFC107','#1f77b4','#2196F3','#0D47A1','#E53935']
NZ_SWEEP    = [2,4,5,8,10]
HALF        = 40
R_WAVE_STD  = 2

def plot(a):
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    fig.suptitle(
        "Paper 48 -- sg_C: The Causal Structure Metric\n"
        "VCML encodes zone structure as a population code; "
        "gate optimizes SNR, not amplitude",
        fontsize=11, fontweight='bold'
    )

    exp_a = a['exp_a']
    exp_b = a['exp_b']

    # ── Panel (a): sg4 vs gate condition ─────────────────────────────────────
    ax = axes[0, 0]
    x = np.arange(len(GATE_ORDER))
    sg4_m  = [exp_a[c]['sg4_mean']  for c in GATE_ORDER if c in exp_a]
    sg4_se = [exp_a[c]['sg4_se']    for c in GATE_ORDER if c in exp_a]
    ax.bar(x, sg4_m, yerr=sg4_se, capsize=4, color=COLORS_GATE, alpha=0.85,
           edgecolor='k', linewidth=0.7)
    ax.axhline(exp_a.get('ss10',{}).get('sg4_mean',0), color='#1f77b4',
               linewidth=1.2, linestyle='--', alpha=0.6, label='ss10 reference')
    ax.set_xticks(x); ax.set_xticklabels(GATE_LABELS, fontsize=8)
    ax.set_ylabel('sg4 (between-zone amplitude)', fontsize=10)
    ax.set_title('(a) sg4: rand_p60 beats ss10 by 1.69x\n'
                 '(Paper 46 replication)', fontsize=9)
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3, axis='y')
    ax.text(0.97, 0.97,
            'sg4 = between-zone distance\n(spatial amplitude, population-level)',
            transform=ax.transAxes, fontsize=7.5, va='top', ha='right',
            bbox=dict(boxstyle='round,pad=0.3', fc='lightyellow', alpha=0.8))

    # ── Panel (b): SNR and decode_norm vs gate condition ─────────────────────
    ax2 = axes[0, 1]
    snr_m    = [exp_a[c]['snr']         for c in GATE_ORDER if c in exp_a]
    dnorm_m  = [exp_a[c]['decode_norm'] for c in GATE_ORDER if c in exp_a]
    sigma_m  = [exp_a[c]['sigma_within']for c in GATE_ORDER if c in exp_a]

    ax2b = ax2.twinx()
    bars2 = ax2.bar(x, snr_m, color=COLORS_GATE, alpha=0.7,
                    edgecolor='k', linewidth=0.7, label='SNR = sg4/sigma_w')
    ax2b.plot(x, dnorm_m, 'D--', color='#333', markersize=6,
              linewidth=1.4, label='decode_norm (sg_C)', alpha=0.85)
    ax2b.axhline(0, color='#888', linewidth=0.8, linestyle=':')

    ax2.set_xticks(x); ax2.set_xticklabels(GATE_LABELS, fontsize=8)
    ax2.set_ylabel('SNR = sg4 / sigma_within', fontsize=10, color='#333')
    ax2b.set_ylabel('decode_norm (above chance)', fontsize=9, color='#666')
    ax2.set_title('(b) SNR: gate DOES flip (ss20>ss10>rand>ss0)\n'
                  'decode_norm near zero -- population code, not single-cell',
                  fontsize=9)
    lines1, labs1 = ax2.get_legend_handles_labels()
    lines2, labs2 = ax2b.get_legend_handles_labels()
    ax2.legend(lines1+lines2, labs1+labs2, fontsize=7.5, loc='upper left')
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.text(0.97, 0.03,
             'SNR = signal-to-noise ratio\n'
             'Gate reduces sigma_w without\n'
             'proportionally reducing sg4',
             transform=ax2.transAxes, fontsize=7.5, va='bottom', ha='right',
             bbox=dict(boxstyle='round,pad=0.3', fc='lightcyan', alpha=0.8))

    # ── Panel (c): sg4 vs delta ───────────────────────────────────────────────
    ax3 = axes[1, 0]
    deltas = []; sg4_b = []; sg4_se_b = []
    for nz in NZ_SWEEP:
        k = str(nz)
        if k not in exp_b: continue
        deltas.append(exp_b[k]['delta'])
        sg4_b.append(exp_b[k]['sg4_mean'])
        sg4_se_b.append(exp_b[k]['sg4_se'])
    x_b = np.arange(len(deltas))
    ax3.bar(x_b, sg4_b, yerr=sg4_se_b, capsize=4,
            color='#5C6BC0', alpha=0.8, edgecolor='k', linewidth=0.7)
    ax3.set_xticks(x_b)
    ax3.set_xticklabels([f'delta={d:.1f}\n(NZ={nz})'
                         for d,nz in zip(deltas,NZ_SWEEP[:len(deltas)])], fontsize=8)
    ax3.set_ylabel('sg4 (between-zone amplitude)', fontsize=10)
    ax3.set_title('(c) sg4 increases at small delta (wrong direction)\n'
                  'Spatial metric: more zones = more pairs = higher raw distance',
                  fontsize=9)
    ax3.grid(True, alpha=0.3, axis='y')
    ax3.text(0.97, 0.97,
             'sg4 artifact: N_ZONES=10 has\n'
             '45 pairs vs 6 pairs for N_ZONES=4\n'
             'More pairs -> higher average distance',
             transform=ax3.transAxes, fontsize=7.5, va='top', ha='right',
             bbox=dict(boxstyle='round,pad=0.3', fc='#ffdddd', alpha=0.8))

    # ── Panel (d): sg_C (decode_norm) vs delta ────────────────────────────────
    ax4 = axes[1, 1]
    dec_b = []; na_b = []
    for nz in NZ_SWEEP:
        k = str(nz)
        if k not in exp_b: continue
        dec_b.append(exp_b[k]['decode_norm'])
        na_b.append(exp_b[k].get('na_ratio', float('nan')))

    ax4.bar(x_b, dec_b, color='#26A69A', alpha=0.8,
            edgecolor='k', linewidth=0.7, label='decode_norm (sg_C)')
    ax4.axhline(0, color='k', linewidth=1.0, linestyle='--')

    ax4b = ax4.twinx()
    valid_na = [(d, r) for d, r in zip(deltas, na_b) if not math.isnan(r)]
    if valid_na:
        d_na, r_na = zip(*valid_na)
        x_na = [deltas.index(d) for d in d_na]
        ax4b.plot(x_na, r_na, 'o--', color='#FF7043', markersize=6,
                  linewidth=1.4, label='na_ratio (Paper 47)')
        ax4b.axhline(1.0, color='#FF7043', linewidth=0.8, linestyle=':', alpha=0.6)
    ax4b.set_ylabel('na_ratio (nonadj/adj)', fontsize=9, color='#FF7043')
    ax4b.tick_params(axis='y', labelcolor='#FF7043')

    ax4.set_xticks(x_b)
    ax4.set_xticklabels([f'delta={d:.1f}\n(NZ={nz})'
                         for d,nz in zip(deltas,NZ_SWEEP[:len(deltas)])], fontsize=8)
    ax4.set_ylabel('decode_norm (sg_C, above chance)', fontsize=10)
    ax4.set_title('(d) sg_C decreases at small delta (correct direction)\n'
                  'Causal metric: less per-cell zone information as delta shrinks',
                  fontsize=9)
    lines1, labs1 = ax4.get_legend_handles_labels()
    lines2, labs2 = ax4b.get_legend_handles_labels()
    ax4.legend(lines1+lines2, labs1+labs2, fontsize=7.5, loc='upper right')
    ax4.grid(True, alpha=0.3, axis='y')
    ax4.text(0.97, 0.03,
             'sg_C and na_ratio agree:\n'
             'causal degradation at delta<5\n'
             'sg4 missed this (panel c)',
             transform=ax4.transAxes, fontsize=7.5, va='bottom', ha='right',
             bbox=dict(boxstyle='round,pad=0.3', fc='lightgreen', alpha=0.7))

    plt.tight_layout()
    plt.savefig(FIGURE_FILE, dpi=150, bbox_inches='tight')
    print(f"Saved: {FIGURE_FILE}")


if __name__ == "__main__":
    plot(load())
