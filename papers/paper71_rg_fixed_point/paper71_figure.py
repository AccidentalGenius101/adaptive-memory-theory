"""Paper 71 – Figure generation.
Six-panel figure:
  (a) Phase A: iso-xi universality -- U4 vs xi, coloured by r_w
  (b) Phase A: |M| vs xi, coloured by r_w (universality check)
  (c) Phase B: |M| vs xi for r_w=5 and r_w=8 with power-law fit
  (d) Phase B: log-log |M| vs (xi - xi*) showing beta
  (e) Phase C: U4 vs L for five P values (FSS ordering)
  (f) Phase C: FSS slopes d(ln|M|)/d(lnL) vs xi
"""
import json, math, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

sys.stdout = __import__('io').TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

ANALYSIS_FILE = Path(__file__).parent / 'results' / 'paper71_analysis.json'
OUT_PDF       = Path(__file__).parent / 'paper71_figure1.pdf'

def safe_linfit(xs, ys):
    valid = [(x, y) for x, y in zip(xs, ys)
             if x > 0 and y > 0 and not math.isnan(y) and not math.isnan(x)]
    if len(valid) < 3:
        return float('nan'), float('nan'), float('nan')
    lx = [math.log(x) for x, y in valid]
    ly = [math.log(y) for x, y in valid]
    n = len(lx)
    mx, my = sum(lx)/n, sum(ly)/n
    ssxx = sum((x-mx)**2 for x in lx)
    ssxy = sum((x-mx)*(y-my) for x, y in zip(lx, ly))
    if ssxx < 1e-20:
        return float('nan'), float('nan'), float('nan')
    slope = ssxy / ssxx
    intercept = my - slope*mx
    yhat = [my + slope*(x-mx) for x in lx]
    sstot = sum((y-my)**2 for y in ly)
    ssres = sum((y-yh)**2 for y, yh in zip(ly, yhat))
    r2 = 1 - ssres/sstot if sstot > 1e-20 else float('nan')
    return slope, math.exp(intercept), r2

with open(ANALYSIS_FILE) as f:
    data = json.load(f)

agg_A  = data['phase_A']
agg_B  = data['phase_B']
agg_C  = data['phase_C']
XI_A   = data['xi_A']
PAIRS_A = {float(k): v for k, v in data['pairs_A'].items()}
PC_C   = data['pc_c']
L_C    = data['L_c']
RW_C   = data['rw_c']
RW_B5  = data['rw_b5_pc']
RW_B8  = data['rw_b8_pc']

# ── Colours ────────────────────────────────────────────────────────────────────
rw_colours = {2:'#e41a1c', 3:'#ff7f00', 4:'#4daf4a', 5:'#377eb8', 8:'#984ea3'}
L_colours  = {40:'#a6cee3', 60:'#1f78b4', 80:'#33a02c', 100:'#fb9a99', 120:'#e31a1c'}

fig, axes = plt.subplots(2, 3, figsize=(12, 7))
(ax_a, ax_b, ax_c), (ax_d, ax_e, ax_f) = axes

# ── Panel A: U4 vs xi, iso-xi groups ──────────────────────────────────────────
ax_a.set_title('(a) Iso-xi: U4 vs xi', fontsize=10)
ax_a.set_xlabel('xi = r_w * sqrt(P)', fontsize=9)
ax_a.set_ylabel('U4 (Binder)', fontsize=9)
ax_a.axhline(2/3, color='grey', lw=0.8, ls='--', label='ordered limit 2/3')
for xi in XI_A:
    for r_w, pc in PAIRS_A[xi]:
        key = f'xi{xi:.2f}_rw{r_w}_pc{pc:.4f}'
        if key in agg_A:
            d = agg_A[key]
            u = d['U4']; e = d['U4_se']
            if not math.isnan(u):
                ax_a.errorbar(xi, u, yerr=e, fmt='o',
                              color=rw_colours.get(r_w, 'black'),
                              ms=6, capsize=3,
                              label=f'r_w={r_w}' if (xi == XI_A[0]) else '')
ax_a.set_ylim(-0.1, 0.80)
handles = [plt.Line2D([0],[0], marker='o', color=c, lw=0, ms=6, label=f'r_w={rw}')
           for rw, c in rw_colours.items()]
ax_a.legend(handles=handles, fontsize=7, ncol=2)

# ── Panel B: |M| vs xi ────────────────────────────────────────────────────────
ax_b.set_title('(b) Iso-xi: |M| vs xi', fontsize=10)
ax_b.set_xlabel('xi', fontsize=9)
ax_b.set_ylabel('|M|', fontsize=9)
for xi in XI_A:
    for r_w, pc in PAIRS_A[xi]:
        key = f'xi{xi:.2f}_rw{r_w}_pc{pc:.4f}'
        if key in agg_A:
            d = agg_A[key]
            m = d['absM']; e = d['absM_se']
            if not math.isnan(m):
                ax_b.errorbar(xi, m, yerr=e, fmt='s',
                              color=rw_colours.get(r_w, 'black'),
                              ms=6, capsize=3)
ax_b.legend(handles=handles, fontsize=7, ncol=2)

# ── Panel C: |M| vs xi for r_w=5 and r_w=8 (Phase B) ────────────────────────
ax_c.set_title('(c) |M| vs xi, two r_w (Phase B)', fontsize=10)
ax_c.set_xlabel('xi', fontsize=9)
ax_c.set_ylabel('|M|', fontsize=9)
for rw_b, pc_list, colour, marker in [
        (5, RW_B5, '#377eb8', 'o'), (8, RW_B8, '#984ea3', 's')]:
    xi_v, M_v = [], []
    for pc in pc_list:
        key = f'rw{rw_b}_pc{pc:.4f}'
        if key in agg_B:
            d = agg_B[key]
            xi = rw_b * math.sqrt(pc)
            m  = d['absM']
            if not math.isnan(m) and m > 0:
                ax_c.errorbar(xi, m, yerr=d['absM_se'], fmt=marker,
                              color=colour, ms=6, capsize=3,
                              label=f'r_w={rw_b}' if not xi_v else '')
                xi_v.append(xi); M_v.append(m)
ax_c.legend(fontsize=8)

# ── Panel D: log-log |M| vs (xi - xi*), beta fit ──────────────────────────────
XI_STAR = 0.55   # working estimate; will be updated from Phase A results
ax_d.set_title(f'(d) beta from xi-scan (xi*={XI_STAR:.2f})', fontsize=10)
ax_d.set_xlabel('xi - xi*', fontsize=9)
ax_d.set_ylabel('|M|', fontsize=9)
ax_d.set_xscale('log'); ax_d.set_yscale('log')
for rw_b, pc_list, colour, marker in [
        (5, RW_B5, '#377eb8', 'o'), (8, RW_B8, '#984ea3', 's')]:
    xi_v, M_v = [], []
    for pc in pc_list:
        key = f'rw{rw_b}_pc{pc:.4f}'
        if key in agg_B:
            d = agg_B[key]
            xi = rw_b * math.sqrt(pc)
            dxi = xi - XI_STAR
            m = d['absM']
            if dxi > 0 and not math.isnan(m) and m > 0:
                ax_d.errorbar(dxi, m, yerr=d['absM_se'], fmt=marker,
                              color=colour, ms=6, capsize=3,
                              label=f'r_w={rw_b}' if not xi_v else '')
                xi_v.append(dxi); M_v.append(m)
    if len(xi_v) >= 3:
        slope, amp, r2 = safe_linfit(xi_v, M_v)
        x_fit = np.logspace(math.log10(min(xi_v)), math.log10(max(xi_v)), 40)
        ax_d.plot(x_fit, amp * x_fit**slope, '--', color=colour, lw=1.2,
                  label=f'r_w={rw_b} beta={slope:.2f} R2={r2:.2f}')
ax_d.legend(fontsize=7)

# ── Panel E: U4 vs L for Phase C ──────────────────────────────────────────────
ax_e.set_title('(e) FSS: U4 vs L (Phase C)', fontsize=10)
ax_e.set_xlabel('L', fontsize=9)
ax_e.set_ylabel('U4', fontsize=9)
for pc in PC_C:
    xi = RW_C * math.sqrt(pc)
    u4s = []; errs = []
    for L in L_C:
        key = f'pc{pc:.4f}_L{L}'
        if key in agg_C:
            d = agg_C[key]
            u4s.append(d['U4']); errs.append(d['U4_se'])
        else:
            u4s.append(float('nan')); errs.append(0.0)
    valid = [(L, u, e) for L, u, e in zip(L_C, u4s, errs) if not math.isnan(u)]
    if valid:
        Ls = [v[0] for v in valid]; us = [v[1] for v in valid]; es = [v[2] for v in valid]
        ax_e.errorbar(Ls, us, yerr=es, marker='o', capsize=3, lw=1.2,
                      label=f'xi={xi:.3f}')
ax_e.axhline(2/3, color='grey', lw=0.8, ls='--')
ax_e.legend(fontsize=7)

# ── Panel F: FSS slopes d(ln|M|)/d(lnL) vs xi ────────────────────────────────
ax_f.set_title('(f) FSS slopes vs xi (Phase C)', fontsize=10)
ax_f.set_xlabel('xi = 5*sqrt(P)', fontsize=9)
ax_f.set_ylabel('d(ln|M|)/d(lnL)', fontsize=9)
fss_slopes = data.get('fss_slopes', {})
xi_s, slope_s = [], []
for pc in PC_C:
    xi = RW_C * math.sqrt(pc)
    s  = fss_slopes.get(str(pc), float('nan'))
    if not math.isnan(float(s)):
        xi_s.append(xi); slope_s.append(float(s))
if xi_s:
    ax_f.plot(xi_s, slope_s, 'ko-', ms=7, lw=1.5)
ax_f.axhline(-0.67, color='blue', lw=0.9, ls='--', label='-beta/nu=-0.67 (Paper 63)')
ax_f.axhline(-1.0,  color='red',  lw=0.9, ls=':',  label='disorder floor -1.0')
ax_f.axhline(0.0,   color='grey', lw=0.8, ls='-')
ax_f.legend(fontsize=7)

fig.suptitle('Paper 71: RG Fixed Point via Composite Variable xi = r_w * sqrt(P)',
             fontsize=11, fontweight='bold')
plt.tight_layout()
plt.savefig(OUT_PDF, bbox_inches='tight')
print(f'Saved {OUT_PDF}  ({OUT_PDF.stat().st_size} bytes)')
