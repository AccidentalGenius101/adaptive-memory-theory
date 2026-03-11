"""Paper 49 figure: 4 panels (updated with full metric suite)."""
import json, numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

with open('results/paper49_analysis.json') as f:
    d = json.load(f)
with open('results/paper49_results.json') as f:
    raw = json.load(f)

A = [r for r in raw if r['exp']=='A']
B = [r for r in raw if r['exp']=='B']
C = [r for r in raw if r['exp']=='C']

SEED_BETAS    = [0.0, 0.1, 0.25, 0.5, 0.75, 1.0]
T_ENCODES     = [1000, 2000, 3000]
ADV_AMPS      = [0.0, 0.25, 0.5]
GATES         = ['ss0', 'ss10', 'ss20', 'rand_p60']
GATE_LABELS   = {'ss0':'SS=0','ss10':'SS=10','ss20':'SS=20','rand_p60':'rand'}
GATE_COLORS   = {'ss0':'#d62728','ss10':'#1f77b4','ss20':'#2ca02c','rand_p60':'#ff7f0e'}
COUPLINGS     = ['ctrl','geo','mean_relay']
COUP_LABELS   = {'ctrl':'ctrl (random)','geo':'geo (wave)','mean_relay':'mean_relay (birth)'}
COUP_COLORS   = {'ctrl':'#888888','geo':'#1f77b4','mean_relay':'#2ca02c'}

fig = plt.figure(figsize=(12, 9))
gs  = GridSpec(2, 2, figure=fig, hspace=0.48, wspace=0.38)

# ── Panel (a): Exp A — beta_s flatness ────────────────────────────────────────
ax_a = fig.add_subplot(gs[0, 0])
sg4_vals = [d['A'][str(sb)]['sg4'] for sb in SEED_BETAS]
se_vals  = [d['A'][str(sb)]['sg4_se'] for sb in SEED_BETAS]
snr_vals = [d['A'][str(sb)]['snr'] for sb in SEED_BETAS]
dnorm_vals=[d['A'][str(sb)]['dnorm'] for sb in SEED_BETAS]

ax_a.errorbar(SEED_BETAS, sg4_vals, yerr=se_vals, fmt='o-',
              color='#1f77b4', capsize=4, label='sg4', zorder=3)
ax_a2 = ax_a.twinx()
ax_a2.plot(SEED_BETAS, snr_vals, 's--', color='#ff7f0e', label='SNR', alpha=0.8)
ax_a2.plot(SEED_BETAS, dnorm_vals, '^:', color='#9467bd', label='dnorm', alpha=0.8)
ax_a.set_xlabel(r'$\beta_s$ (seeding scale)', fontsize=11)
ax_a.set_ylabel('sg4', color='#1f77b4', fontsize=11)
ax_a2.set_ylabel('SNR / decode\_norm', color='#555555', fontsize=10)
ax_a.set_title('(a) Exp A: $\\beta_s$ sweep (NULL)\nsg4, SNR, decode_norm all flat',
               fontsize=10, fontweight='bold')
ax_a.set_xticks(SEED_BETAS)
ax_a.tick_params(axis='y', labelcolor='#1f77b4')
ax_a.axhline(np.mean(sg4_vals), ls=':', color='#1f77b4', alpha=0.4)
lines1, labs1 = ax_a.get_legend_handles_labels()
lines2, labs2 = ax_a2.get_legend_handles_labels()
ax_a.legend(lines1+lines2, labs1+labs2, fontsize=8, loc='lower right')

# ── Panel (b): Exp B adv=0 and adv=0.5 — two-condition overlay ────────────────
ax_b = fig.add_subplot(gs[0, 1])
for gn in GATES:
    snr_0   = [np.nanmean([r['fin_snr'] for r in B
                if r['gate']==gn and r['t_encode']==te and abs(r['adv_amp'])<1e-6])
               for te in T_ENCODES]
    snr_05  = [np.nanmean([r['fin_snr'] for r in B
                if r['gate']==gn and r['t_encode']==te and abs(r['adv_amp']-0.5)<1e-6])
               for te in T_ENCODES]
    c = GATE_COLORS[gn]; lbl = GATE_LABELS[gn]
    ax_b.plot(T_ENCODES, snr_0,  'o-',  color=c, label=f'{lbl} (adv=0)')
    ax_b.plot(T_ENCODES, snr_05, 's--', color=c, alpha=0.55)

ax_b.set_xlabel('$T_{\\mathrm{encode}}$ (steps)', fontsize=11)
ax_b.set_ylabel('Final SNR', fontsize=11)
ax_b.set_title('(b) Exp B: SNR vs $T_{\\mathrm{encode}}$\n'
               'solid=adv=0, dashed=adv=0.5', fontsize=10, fontweight='bold')
ax_b.set_xticks(T_ENCODES)
ax_b.legend(fontsize=7, ncol=2)

# ── Panel (c): Exp B — gate winner per (T_encode, adv_amp) ───────────────────
ax_c = fig.add_subplot(gs[1, 0])
ax_c.axis('off')
rows_text = [['', 'adv=0.0', 'adv=0.25', 'adv=0.5']]
winner_color = {'ss0':'#ffcccc','ss10':'#cce5ff','ss20':'#ccffcc','rand_p60':'#ffe5cc'}
cell_colors  = [['#f0f0f0','#f0f0f0','#f0f0f0','#f0f0f0']]
for te in T_ENCODES:
    row = [f'T={te}']
    crow = ['#f0f0f0']
    for aa in ADV_AMPS:
        best_gate=None; best_snr=-1
        for gn in GATES:
            rows_b=[r for r in B if r['gate']==gn and r['t_encode']==te
                    and abs(r['adv_amp']-aa)<1e-6]
            s=np.nanmean([r['fin_snr'] for r in rows_b]) if rows_b else float('nan')
            if not np.isnan(s) and s>best_snr: best_snr=s; best_gate=gn
        snr_str=f'{best_snr:.3f}' if best_gate else '?'
        row.append(f'{GATE_LABELS.get(best_gate,"?")} ({snr_str})')
        crow.append(winner_color.get(best_gate,'#ffffff'))
    rows_text.append(row)
    cell_colors.append(crow)

tbl = ax_c.table(cellText=rows_text[1:], colLabels=rows_text[0],
                 cellLoc='center', loc='center',
                 cellColours=cell_colors[1:],
                 colColours=cell_colors[0])
tbl.auto_set_font_size(False); tbl.set_fontsize(9)
tbl.scale(1.0, 2.0)
ax_c.set_title('(c) Exp B: Best gate per cell (SNR winner)\n'
               'red=SS0  blue=SS10  green=SS20  orange=rand',
               fontsize=10, fontweight='bold')

# ── Panel (d): Exp C — sg4 bar chart + na_ratio overlay ──────────────────────
ax_d = fig.add_subplot(gs[1, 1])
ctrl_sg4  = d['C']['ctrl']['sg4_l2']
coup_order = ['geo','mean_relay','ctrl']
sg4_vals_c = [d['C'][co]['sg4_l2'] for co in coup_order]
na_vals_c  = [d['C'][co]['na_l2']  for co in coup_order]
gains      = [v/ctrl_sg4 for v in sg4_vals_c]

x = np.arange(len(coup_order))
bars=ax_d.bar(x, sg4_vals_c, color=[COUP_COLORS[c] for c in coup_order],
              width=0.45, edgecolor='k', linewidth=0.8)
for i,(b,g) in enumerate(zip(bars,gains)):
    ax_d.text(b.get_x()+b.get_width()/2, b.get_height()+0.0002,
              f'{g:.2f}x', ha='center', va='bottom', fontsize=9, fontweight='bold')

# na_ratio overlay (right axis)
ax_d2 = ax_d.twinx()
ax_d2.plot(x, na_vals_c, 'D--', color='#8c564b', markersize=8, linewidth=1.5,
           label='na_ratio', zorder=5)
ax_d2.axhline(1.0, ls=':', color='#8c564b', alpha=0.4, linewidth=1)
ax_d2.set_ylabel('na_ratio (nonadj/adj)', color='#8c564b', fontsize=10)
ax_d2.tick_params(axis='y', labelcolor='#8c564b')

ax_d.set_xticks(x)
ax_d.set_xticklabels(['geo\n(wave)', 'mean_relay\n(birth)', 'ctrl\n(random)'], fontsize=9)
ax_d.set_ylabel('sg4 (L2)', fontsize=11)
ax_d.set_title('(d) Exp C: Relay modes vs ctrl\nctrl wins; geo encodes zone parity (na<1)',
               fontsize=10, fontweight='bold')
ax_d.axhline(ctrl_sg4, ls='--', color='#888888', alpha=0.6, label='ctrl baseline')
lines1, labs1 = ax_d.get_legend_handles_labels()
lines2, labs2 = ax_d2.get_legend_handles_labels()
ax_d.legend(lines1+lines2, labs1+labs2, fontsize=8, loc='upper right')

fig.suptitle('Paper 49: C2 Completion, Gate Optimum, Population-Code Relay',
             fontsize=12, fontweight='bold', y=1.01)
plt.savefig('paper49_figure1.pdf', bbox_inches='tight')
print('Saved: paper49_figure1.pdf')
