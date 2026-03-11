"""
paper38_figure1.py -- Figure 1 for Paper 38

Three panels:
  (a) Exp A: sg4n(t) for all conditions, showing C_cryst spike + polarity-blindness
  (b) Exp B: sg4n distribution geo vs ctrl -- bimodal attractor + basin fraction
  (c) Exp C: long-run sg4n(t) -- metastability, not monotone saturation
"""
import json, math, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import defaultdict

RES = "results/paper38_results.json"
with open(RES) as f: data = json.load(f)

FA_STD = 0.16; FA_CRYST = 0.01; T_SHIFT = 2000

def mn(lst):
    v = [x for x in lst if x is not None and not math.isnan(x)]
    return float(np.mean(v)) if v else float('nan')

def se(lst):
    v = [x for x in lst if x is not None and not math.isnan(x)]
    return float(np.std(v, ddof=1)/math.sqrt(len(v))) if len(v)>1 else 0.

# ── Exp A ──────────────────────────────────────────────────────────────────
A_runs = defaultdict(list)
for r in data:
    if r['exp']=='A':
        A_runs[(r['fa_phase2'], r['do_shift'])].append(r)

ts_A = A_runs[(FA_STD, True)][0]['ts']

A_conds = [
    (FA_STD,   True,  r'$C_\mathrm{ref}$ (FA=0.16, shift)',    '#e31a1c', '-',  'o'),
    (FA_CRYST, True,  r'$C_\mathrm{cryst}$ (FA=0.01, shift)',  '#1f77b4', '--', 's'),
    (FA_STD,   False, r'$C_\mathrm{null}$ (FA=0.16, no shift)','#2ca02c', ':',  '^'),
]
sg4n_A = {}; se_A = {}
for fa2, do_shift, label, col, ls, mk in A_conds:
    runs = A_runs[(fa2, do_shift)]
    n = len(ts_A)
    sg4n_A[(fa2,do_shift)] = [mn([r['sg4ns'][i] for r in runs]) for i in range(n)]
    se_A[(fa2,do_shift)]   = [se([r['sg4ns'][i] for r in runs]) for i in range(n)]

# ── Exp B ──────────────────────────────────────────────────────────────────
B_geo  = [r for r in data if r['exp']=='B' and r['coupling']=='geo']
B_ctrl = [r for r in data if r['exp']=='B' and r['coupling']=='ctrl']
geo_vals  = [r['sg4n'] for r in B_geo  if not math.isnan(r['sg4n'])]
ctrl_vals = [r['sg4n'] for r in B_ctrl if not math.isnan(r['sg4n'])]
threshold = float(np.percentile(ctrl_vals, 75))
P_hi_geo  = float(np.mean([v > threshold for v in geo_vals]))
P_hi_ctrl = float(np.mean([v > threshold for v in ctrl_vals]))
G_basin = P_hi_geo / P_hi_ctrl if P_hi_ctrl > 0 else float('nan')

# ── Exp C ──────────────────────────────────────────────────────────────────
C_runs = [r for r in data if r['exp']=='C']
ts_C   = C_runs[0]['ts']
sg4n_C = [mn([r['sg4ns'][i] for r in C_runs]) for i in range(len(ts_C))]
se_C   = [se([r['sg4ns'][i] for r in C_runs]) for i in range(len(ts_C))]
pk_i   = int(np.nanargmax(sg4n_C))

# ── Figure ─────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
GRAY = '#888888'; ORANGE = '#ff7f0e'; GREEN = '#2ca02c'

# ─── Panel A: Exp A sg4n(t) ───────────────────────────────────────────────
ax = axes[0]
ax.axvline(T_SHIFT, color=GRAY, lw=1.0, ls='--', alpha=0.6, zorder=0,
           label=f'Shift at t={T_SHIFT}')
ax.axhline(1.0, color=GRAY, lw=0.5, ls=':', alpha=0.3, zorder=0)
for fa2, do_shift, label, col, ls, mk in A_conds:
    ys = sg4n_A[(fa2,do_shift)]
    ses = se_A[(fa2,do_shift)]
    ax.errorbar(ts_A, ys, yerr=ses, fmt=mk+ls, color=col, ms=4, capsize=2,
                lw=1.5, label=label, zorder=2)

# Annotate C_cryst spike
cryst_vals = sg4n_A[(FA_CRYST, True)]
pk_t = ts_A[int(np.nanargmax(cryst_vals))]
pk_v = max([v for v in cryst_vals if not math.isnan(v)])
ax.annotate(f'C_cryst spike\n(rigid fieldM\nreinforcement)',
            xy=(pk_t, pk_v), xytext=(pk_t+300, pk_v+0.05),
            fontsize=8, ha='left', color='#1f77b4',
            arrowprops=dict(arrowstyle='->', color='#1f77b4', lw=0.8))

ax.set_xlabel(r'Step $t$', fontsize=11)
ax.set_ylabel(r'Zone structure $\mathrm{sg4n}$', fontsize=10)
ax.set_title(r'\textbf{(a)}\ Continual learning: wave polarity shift', fontsize=11)
ax.legend(fontsize=8, loc='upper left')
ax.set_ylim(bottom=0)

# ─── Panel B: Exp B basin fraction distributions ──────────────────────────
ax = axes[1]
bins = np.linspace(0, max(max(geo_vals), max(ctrl_vals)) * 1.05, 25)
ax.hist(ctrl_vals, bins=bins, alpha=0.55, color='#1f77b4',
        label=f'ctrl: mean={mn(ctrl_vals):.3f}')
ax.hist(geo_vals,  bins=bins, alpha=0.55, color='#e31a1c',
        label=f'geo:  mean={mn(geo_vals):.3f}')
ax.axvline(threshold, color='k', lw=1.2, ls='--',
           label=f'75th pct ctrl = {threshold:.3f}')

# Basin-fraction annotation
ymax = max(ax.get_ylim()[1], 5)
ax.text(threshold * 1.08, ymax * 0.85,
        f'P(high | geo)  = {P_hi_geo:.2f}\n'
        f'P(high | ctrl) = {P_hi_ctrl:.2f}\n'
        f'G$_{{basin}}$ = {G_basin:.2f}',
        fontsize=9, va='top', color='k',
        bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.8))

ax.set_xlabel(r'sg4n at $T=3000$', fontsize=11)
ax.set_ylabel('Count', fontsize=10)
ax.set_title(r'\textbf{(b)}\ Basin fraction (30 seeds, $N=4$)', fontsize=11)
ax.legend(fontsize=8)

# ─── Panel C: Exp C long-run metastability ────────────────────────────────
ax = axes[2]
ax.errorbar(ts_C, sg4n_C, yerr=se_C, fmt='o-', color='#1f77b4',
            ms=5, capsize=2, lw=1.5, label=f'geo (n={len(C_runs)} seeds)')
ax.axvline(3000, color=ORANGE, lw=1.0, ls=':', alpha=0.8,
           label='T=3000 (standard endpoint)')
ax.axvline(ts_C[pk_i], color='#e31a1c', lw=1.2, ls='-.', alpha=0.8,
           label=f'Peak: t={ts_C[pk_i]}, sg4n={sg4n_C[pk_i]:.3f}')
# Half-peak line
half = sg4n_C[pk_i] / 2
ax.axhline(half, color='#e31a1c', lw=0.8, ls=':', alpha=0.5,
           label=f'Half-peak = {half:.3f}')

ax.set_xlabel(r'Step $t$', fontsize=11)
ax.set_ylabel(r'Mean sg4n', fontsize=10)
ax.set_title(r'\textbf{(c)}\ Long-run saturation ($T=8000$)', fontsize=11)
ax.legend(fontsize=8)
ax.set_ylim(bottom=0)

plt.tight_layout()
fig.savefig('paper38_figure1.png', dpi=150, bbox_inches='tight')
fig.savefig('paper38_figure1.pdf', bbox_inches='tight')
print("Saved paper38_figure1.png / .pdf")

print(f"\n=== Exp A: sg4n at key timepoints ===")
for fa2, do_shift, label, col, ls, mk in A_conds:
    ys = sg4n_A[(fa2,do_shift)]
    t2200_i = ts_A.index(2200)
    t4000_i = ts_A.index(4000)
    print(f"  {label}: @T_SHIFT={ys[ts_A.index(T_SHIFT)]:.3f}, "
          f"@2200={ys[t2200_i]:.3f}, @4000={ys[t4000_i]:.3f}, "
          f"peak={max(v for v in ys if not math.isnan(v)):.3f}")

print(f"\n=== Exp B ===")
print(f"  geo : n={len(geo_vals)}, mean={mn(geo_vals):.4f}, P_high={P_hi_geo:.3f}")
print(f"  ctrl: n={len(ctrl_vals)}, mean={mn(ctrl_vals):.4f}, P_high={P_hi_ctrl:.3f}")
print(f"  G_basin = {G_basin:.2f}, mean ratio = {mn(geo_vals)/mn(ctrl_vals):.2f}")

print(f"\n=== Exp C ===")
print(f"  Peak: t={ts_C[pk_i]}, sg4n={sg4n_C[pk_i]:.4f}")
print(f"  sg4n at T=3000: {sg4n_C[ts_C.index(3200)]:.4f} (t=3200 nearest)")
print(f"  sg4n at T=8000: {sg4n_C[-1]:.4f}")
for i in range(pk_i, len(ts_C)):
    if not math.isnan(sg4n_C[i]) and sg4n_C[i] < half:
        print(f"  Half-peak (sg4n<{half:.3f}): t~{ts_C[i]}")
        break
