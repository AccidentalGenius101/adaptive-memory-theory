"""Paper 69 figure: wave-range dependence of ordering and exponents."""
import json, math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

ANALYSIS = Path(__file__).parent / 'results' / 'paper69_analysis.json'
with open(ANALYSIS) as f:
    d = json.load(f)

agg_A = d['A']
agg_B = d['B']
agg_C = d['C']

RW_A = [1, 2, 3, 4, 5, 6, 8]
RW_B = [1, 5]
L_B  = [40, 60, 80, 100, 120]
PC_B = [0.005, 0.010, 0.020, 0.050]
PC_C = [0.001, 0.002, 0.003, 0.005, 0.007, 0.010, 0.015, 0.020, 0.030, 0.050]

def wave_area(r): return 2 * r * (r + 1) + 1
WE_BASE = 25; L_BASE = 40
def _wef(L): return WE_BASE * (L_BASE / L) ** 2
A_REF = wave_area(5)
def _prob_B(L, r_w): return min(1.0, (A_REF / wave_area(r_w)) / _wef(L))

fig, axes = plt.subplots(2, 3, figsize=(15, 9))
fig.suptitle('Paper 69: Wave-Range Dependence of Ordering and Critical Exponents',
             fontsize=13, fontweight='bold')

# ── (a) U4 vs r_w (Phase A, constant rate) ────────────────────────────────────
ax = axes[0, 0]
rw_arr = np.array(RW_A)
u4_arr = np.array([agg_A.get(str(r), {}).get('U4', float('nan')) for r in RW_A])
u4_se  = np.array([agg_A.get(str(r), {}).get('U4_se', 0.) for r in RW_A])
areas  = np.array([wave_area(r) for r in RW_A])
ax.errorbar(rw_arr, u4_arr, yerr=u4_se, fmt='o-', color='C0', ms=8, capsize=4)
ax.axhline(2/3, color='C3', linestyle='--', alpha=0.5, label='Ordered limit 2/3')
ax.set_xlabel('Wave radius $r_w$'); ax.set_ylabel('$U_4$')
ax.set_title(f'(a) Binder $U_4$ vs $r_w$ (L=80, P=0.02, const rate)')
ax.legend(fontsize=8)

# ── (b) absM vs wave area (Phase A) ──────────────────────────────────────────
ax = axes[0, 1]
aM_arr = np.array([agg_A.get(str(r), {}).get('absM', float('nan')) for r in RW_A])
aM_se  = np.array([agg_A.get(str(r), {}).get('absM_se', 0.) for r in RW_A])
ok = ~np.isnan(aM_arr)
ax.errorbar(areas[ok], aM_arr[ok], yerr=aM_se[ok], fmt='o', color='C0', ms=8, capsize=4)
if ok.sum() >= 3:
    sl, lc = np.polyfit(np.log(areas[ok]), np.log(aM_arr[ok]), 1)
    A_fit = np.linspace(areas[ok].min()*0.9, areas[ok].max()*1.1, 100)
    ax.plot(A_fit, np.exp(lc)*A_fit**sl, '--', color='C1',
            label=fr'Power law $\alpha={sl:.2f}$')
ax.set_xscale('log'); ax.set_yscale('log')
ax.set_xlabel('Wave area $A(r_w)$'); ax.set_ylabel('$|M|$')
ax.set_title('(b) $|M|$ vs wave area (Phase A)')
ax.legend(fontsize=8)

# ── (c) FSS slopes vs P for r_w=1 and r_w=5 ─────────────────────────────────
ax = axes[0, 2]
colors = {1: 'C3', 5: 'C0'}
markers = {1: 's', 5: 'o'}
for rw in RW_B:
    slopes = []
    for pc in PC_B:
        Lv = []; aMv = []
        for L in L_B:
            k = f'rw{rw}_L{L}_pc{pc:.4f}'
            if k in agg_B:
                v = agg_B[k]
                if not math.isnan(v.get('absM', float('nan'))):
                    Lv.append(L); aMv.append(v['absM'])
        if len(Lv) >= 3:
            sl, _ = np.polyfit(np.log(Lv), np.log(aMv), 1)
            slopes.append((pc, sl))
    if slopes:
        ps, ss = zip(*slopes)
        ax.plot(ps, ss, f'{markers[rw]}-', color=colors[rw], ms=8,
                label=f'$r_w={rw}$ (area={wave_area(rw)})')

ax.axhline(-0.670, color='k', linestyle='--', alpha=0.5, label=r'$-\beta/\nu=-0.670$')
ax.axhline(-1.0,   color='gray', linestyle=':',  alpha=0.5, label=r'Disorder $-1$')
ax.axhline(0.0,    color='gray', linestyle=':',  alpha=0.3)
ax.set_xlabel('$P_{causal}$'); ax.set_ylabel(r'$d\ln|M|/d\ln L$')
ax.set_title('(c) FSS slopes: $r_w=1$ vs $r_w=5$')
ax.legend(fontsize=7)
ax.set_ylim(-2.2, 0.3)

# ── (d) |M| vs P: r_w=1 (Phase C) vs r_w=5 (Paper 68) ───────────────────────
ax = axes[1, 0]
p68_absM = {0.001: 1.3e-4, 0.002: 1.3e-4, 0.003: 1.3e-4, 0.005: 1.4e-4,
            0.007: 1.4e-4, 0.010: 1.6e-4, 0.015: 2.1e-4, 0.020: 2.8e-4,
            0.030: 4.1e-4, 0.050: 7.0e-4}
pc_c  = [agg_C[f'pc{pc:.4f}']['pc']   for pc in PC_C if f'pc{pc:.4f}' in agg_C]
aM_c  = [agg_C[f'pc{pc:.4f}']['absM'] for pc in PC_C if f'pc{pc:.4f}' in agg_C]
pc_68 = sorted(p68_absM.keys())
aM_68 = [p68_absM[pc] for pc in pc_68]

ax.plot(pc_c, aM_c, 'o-', color='C3', ms=7, label=f'$r_w=1$ (L=160, matched)')
ax.plot(pc_68, aM_68, 's-', color='C0', ms=7, label='$r_w=5$ (Paper 68, L=160)')
ax.set_xscale('log'); ax.set_yscale('log')
ax.set_xlabel('$P_{causal}$'); ax.set_ylabel('$|M|$')
ax.set_title('(d) $|M|$ vs $P$: r_w=1 vs r_w=5 at L=160')
ax.legend(fontsize=8)

# ── (e) U4 vs P at r_w=1 (Phase C) ──────────────────────────────────────────
ax = axes[1, 1]
u4_c  = [agg_C[f'pc{pc:.4f}']['U4']    for pc in PC_C if f'pc{pc:.4f}' in agg_C]
u4_se_c = [agg_C[f'pc{pc:.4f}']['U4_se'] for pc in PC_C if f'pc{pc:.4f}' in agg_C]
ax.errorbar(pc_c, u4_c, yerr=u4_se_c, fmt='o-', color='C3', ms=7, capsize=3,
            label=f'$r_w=1$ (Phase C)')
ax.axhline(2/3, color='C3', linestyle='--', alpha=0.4, label='Ordered limit')
p68_u4 = {0.001: 0.064, 0.005: 0.038, 0.010: 0.083, 0.020: 0.321, 0.050: 0.584}
ax.plot(list(p68_u4.keys()), list(p68_u4.values()), 's--', color='C0', ms=7,
        label='$r_w=5$ (Paper 68)')
ax.set_xscale('log')
ax.set_xlabel('$P_{causal}$'); ax.set_ylabel('$U_4$')
ax.set_title('(e) Binder $U_4$ vs $P$: $r_w=1$ vs $r_w=5$')
ax.legend(fontsize=7)

# ── (f) Effective coverage vs L for r_w=1 and r_w=5 (protocol breakdown) ─────
ax = axes[1, 2]
L_range = np.array([40, 60, 80, 100, 120, 160])
for rw in [1, 5]:
    cov = []
    for L in L_range:
        p_wave = _prob_B(L, rw)
        # hits/site/step
        hits_per_site = p_wave * wave_area(rw) / (L * L)
        cov.append(hits_per_site)
    ax.plot(L_range, cov, f'{"s" if rw==1 else "o"}-',
            color='C3' if rw==1 else 'C0', ms=7, label=f'$r_w={rw}$')

# Ideal (should be flat)
ideal = wave_area(5) / (40**2) * (1/_wef(40))  # at L=40, r_w=5
# Actually compute ideal coverage at each L for r_w=5
ideal_cov = [_prob_B(L, 5) * wave_area(5) / (L*L) for L in L_range]
ax.plot(L_range, ideal_cov, '--', color='C0', alpha=0.5, label='$r_w=5$ ideal (no cap)')
ax.set_xlabel('L'); ax.set_ylabel('Hits / site / step')
ax.set_title('(f) Protocol breakdown: actual coverage vs L\n'
             '($r_w=1$ caps at $p_{wave}=1$ for all $L$)')
ax.legend(fontsize=7)
ax.set_ylim(0, None)

plt.tight_layout()
out = Path(__file__).parent / 'paper69_figure1.pdf'
plt.savefig(out, bbox_inches='tight', dpi=150)
print(f'Saved -> {out}  ({out.stat().st_size} bytes)')
