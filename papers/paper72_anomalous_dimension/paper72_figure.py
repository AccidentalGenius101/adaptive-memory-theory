"""Paper 72 - Figure generation.
Six-panel figure:
  (a) G(r) vs r log-log at criticality, multiple L (Phase A) -- eta from slope
  (b) G(r)/G(3) normalized collapse at criticality (Phase A)
  (c) eta vs L (check for L-independence)
  (d) eta vs r_w (Levy-DP test, Phase B)
  (e) G(r) vs r for Phase C P values (off-critical decay)
  (f) xi_corr vs |P - P_c| log-log, nu from slope (Phase C)
"""
import json, math, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

sys.stdout = __import__('io').TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

ANALYSIS_FILE = Path(__file__).parent / 'results' / 'paper72_analysis.json'
OUT_PDF       = Path(__file__).parent / 'paper72_figure1.pdf'

with open(ANALYSIS_FILE) as f:
    data = json.load(f)

eta_A   = data['eta_A']
eta_B   = data['eta_B']
eta_best = data.get('eta_best') or float('nan')
xi_corr = data['xi_corr']
nu      = data['nu']
L_A     = data['L_a']
RW_B    = data['rw_b']
PC_C    = data['pc_c']
PC_A    = data['pc_a']
corr_A  = data['corr_A']
corr_B  = data['corr_B']
corr_C  = data['corr_C']

L_colours  = {80: '#a6cee3', 100: '#1f78b4', 120: '#33a02c', 160: '#e31a1c'}
rw_colours = {3: '#ff7f00', 5: '#377eb8', 8: '#984ea3'}
P_colours  = ['#a6cee3','#1f78b4','#33a02c','#fb9a99','#e31a1c','#6a3d9a']

fig, axes = plt.subplots(2, 3, figsize=(13, 8))
(ax_a, ax_b, ax_c), (ax_d, ax_e, ax_f) = axes

# ── Panel A: G(r) log-log at criticality, multiple L ─────────────────────────
ax_a.set_title('(a) G(r) at criticality, multiple L', fontsize=10)
ax_a.set_xlabel('r', fontsize=9)
ax_a.set_ylabel('G(r)', fontsize=9)
ax_a.set_xscale('log'); ax_a.set_yscale('log')

for L in L_A:
    cd = corr_A.get(str(L), {})
    rv = cd.get('r', []); Cv = cd.get('C', [])
    if not rv:
        continue
    r_arr = np.array(rv); C_arr = np.array(Cv)
    mask  = C_arr > 0
    ax_a.plot(r_arr[mask], C_arr[mask], 'o-', ms=4, lw=1.2,
              color=L_colours.get(L, 'k'), label=f'L={L}')
    # Power law guide from eta_best
    if not math.isnan(eta_best):
        r_fit = r_arr[mask & (r_arr >= 3) & (r_arr <= L//4)]
        if len(r_fit) >= 2:
            C_fit = C_arr[mask & (r_arr >= 3) & (r_arr <= L//4)]
            amp   = float(np.exp(np.mean(np.log(C_fit) + eta_best * np.log(r_fit))))
            ax_a.plot(r_fit, amp * r_fit**(-eta_best), '--',
                      color=L_colours.get(L, 'k'), lw=0.8, alpha=0.6)

if not math.isnan(eta_best):
    ax_a.text(0.05, 0.15, f'eta_direct={eta_best:.2f}',
              transform=ax_a.transAxes, fontsize=8,
              bbox=dict(fc='white', alpha=0.7))
ax_a.legend(fontsize=7)

# ── Panel B: normalized G(r)/G(r=3) collapse ──────────────────────────────────
ax_b.set_title('(b) G(r)/G(3) collapse', fontsize=10)
ax_b.set_xlabel('r', fontsize=9)
ax_b.set_ylabel('G(r) / G(3)', fontsize=9)
ax_b.set_xscale('log'); ax_b.set_yscale('log')

for L in L_A:
    cd = corr_A.get(str(L), {})
    rv = cd.get('r', []); Cv = cd.get('C', [])
    if not rv or 3 not in rv:
        continue
    r_arr = np.array(rv); C_arr = np.array(Cv)
    g3 = C_arr[rv.index(3)]
    if g3 <= 0:
        continue
    mask = C_arr > 0
    ax_b.plot(r_arr[mask], C_arr[mask] / g3, 'o-', ms=4, lw=1.2,
              color=L_colours.get(L, 'k'), label=f'L={L}')

if not math.isnan(eta_best):
    r_ref = np.logspace(math.log10(3), math.log10(60), 40)
    ax_b.plot(r_ref, (r_ref / 3)**(-eta_best), 'k--', lw=1.2,
              label=f'r^{{-{eta_best:.2f}}}')
ax_b.axhline(1, color='grey', lw=0.6, ls=':')
ax_b.legend(fontsize=7)

# ── Panel C: eta vs L (L-independence check) ──────────────────────────────────
ax_c.set_title('(c) eta vs L at criticality', fontsize=10)
ax_c.set_xlabel('L', fontsize=9)
ax_c.set_ylabel('eta', fontsize=9)

Ls = [L for L in L_A if str(L) in eta_A and not math.isnan(float(eta_A[str(L)]))]
etas = [float(eta_A[str(L)]) for L in Ls]
if Ls:
    ax_c.plot(Ls, etas, 'ko-', ms=8, lw=1.5)
    if not math.isnan(eta_best):
        ax_c.axhline(eta_best, color='blue', lw=1., ls='--',
                     label=f'mean eta={eta_best:.2f}')
    ax_c.axhline(1.28, color='red', lw=0.9, ls=':',
                 label='indirect eta=1.28 (P71)')
ax_c.legend(fontsize=8)
ax_c.set_ylim(bottom=0)

# ── Panel D: eta vs r_w (Levy-DP test) ────────────────────────────────────────
ax_d.set_title('(d) Levy-DP test: eta vs r_w', fontsize=10)
ax_d.set_xlabel('r_w', fontsize=9)
ax_d.set_ylabel('eta', fontsize=9)

rws = [r for r in RW_B if str(r) in eta_B and not math.isnan(float(eta_B[str(r)]))]
eta_b_vals = [float(eta_B[str(r)]) for r in rws]
if rws:
    ax_d.plot(rws, eta_b_vals, 'ko-', ms=8, lw=1.5)
    for rw, e in zip(rws, eta_b_vals):
        ax_d.annotate(f'  {e:.2f}', (rw, e), fontsize=8)
ax_d.axhline(1.28, color='red', lw=0.9, ls=':',
             label='indirect eta=1.28 (P71)')
ax_d.legend(fontsize=8)
ax_d.set_xticks(RW_B)

# ── Panel E: G(r) for Phase C P values ────────────────────────────────────────
ax_e.set_title('(e) G(r) off criticality (Phase C)', fontsize=10)
ax_e.set_xlabel('r', fontsize=9)
ax_e.set_ylabel('G(r)', fontsize=9)
ax_e.set_xscale('log'); ax_e.set_yscale('log')

for i, pc in enumerate(PC_C):
    cd  = corr_C.get(str(pc), {})
    rv  = cd.get('r', []); Cv = cd.get('C', [])
    if not rv:
        continue
    r_arr = np.array(rv); C_arr = np.array(Cv)
    mask  = C_arr > 0
    xi    = 5 * math.sqrt(pc)
    ax_e.plot(r_arr[mask], C_arr[mask], 'o-', ms=3, lw=1.0,
              color=P_colours[i % len(P_colours)],
              label=f'P={pc:.3f} xi={xi:.3f}')
ax_e.legend(fontsize=6, ncol=2)

# ── Panel F: xi_corr vs |P - P_c|, nu fit ────────────────────────────────────
ax_f.set_title('(f) xi_corr vs |P - P_c|  (nu fit)', fontsize=10)
ax_f.set_xlabel('|P - P_c|', fontsize=9)
ax_f.set_ylabel('xi_corr', fontsize=9)
ax_f.set_xscale('log'); ax_f.set_yscale('log')

P_c = PC_A
xc_pairs = [(abs(float(pc) - P_c), float(xi_corr[str(pc)]))
            for pc in PC_C
            if str(pc) in xi_corr
            and not math.isnan(float(xi_corr.get(str(pc), float('nan'))))
            and float(xi_corr.get(str(pc), 0)) > 0]

if xc_pairs:
    dP_arr = np.array([p[0] for p in xc_pairs])
    xc_arr = np.array([p[1] for p in xc_pairs])
    ax_f.plot(dP_arr, xc_arr, 'ko', ms=7, label='data')
    if not math.isnan(nu):
        # Power-law fit line
        dP_fit = np.logspace(math.log10(min(dP_arr) * 0.8),
                             math.log10(max(dP_arr) * 1.2), 40)
        # Amplitude from mean
        amps = [xc * dp**nu for dp, xc in zip(dP_arr, xc_arr)]
        amp  = float(np.median(amps))
        ax_f.plot(dP_fit, amp * dP_fit**(-nu), 'r--', lw=1.4,
                  label=f'nu={nu:.2f}')
    ax_f.legend(fontsize=8)
    ax_f.set_xlabel('|P - 0.010|', fontsize=9)

fig.suptitle('Paper 72: Direct eta from G(r) ~ r^{-eta} Two-Point Correlator',
             fontsize=11, fontweight='bold')
plt.tight_layout()
plt.savefig(OUT_PDF, bbox_inches='tight')
print(f'Saved {OUT_PDF}  ({OUT_PDF.stat().st_size} bytes)')
