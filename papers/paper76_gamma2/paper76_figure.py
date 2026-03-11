"""
Paper 76 - Figure generator
Panels:
  (a) chi_int vs P -- Phase A (L=80) + Phase C (L=120): FLAT (noise floor)
  (b) chi_int vs L -- Phase B (FSS at P=0.010): FLAT (no L divergence)
  (c) U4 vs P -- showing ordering only at high P
  (d) absM vs L -- geometric dilution confirms disordered/near-critical state
"""

import json, math, numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

RESULTS_A = Path(__file__).parent / 'results' / 'paper76_analysis.json'
RESULTS_R = Path(__file__).parent / 'results' / 'paper76_results.json'
OUT       = Path(__file__).parent / 'results' / 'paper76_figure1.pdf'

with open(RESULTS_A) as f:
    da = json.load(f)
with open(RESULTS_R) as f:
    dr = json.load(f)

# Unpack phase A (L=80, P scan)
A = dr['phase_A']
PA  = [e['P'] for e in A]
chiA = [e['chi_int'] for e in A]
U4A  = [e['U4'] for e in A]
absMa = [e['absM'] for e in A]

# Phase B (P=0.010, L scan)
B = dr['phase_B']
LB   = [e['L'] for e in B]
chiB = [e['chi_int'] for e in B]
U4B  = [e['U4'] for e in B]
absMb = [e['absM'] for e in B]

# Phase C (L=120, P scan)
C = dr['phase_C']
PC  = [e['P'] for e in C]
chiC = [e['chi_int'] for e in C]
U4C  = [e['U4'] for e in C]
absMc = [e['absM'] for e in C]

# ── theoretical predictions ──────────────────────────────────────────
nu = 0.98; beta = 0.628; eta = 1.28
gamma_Fisher = nu * (2.0 - eta)   # = 0.706
# If chi_int ~ P^{-gamma} and we're in ordered phase (P>>P_c=0+):
#   chi_int SHOULD DECREASE as P increases (ordered side)
# Geometric noise floor ~ var(phi) * (L/2)^2 * (1/(L/2)) = const
noise_floor = np.mean(chiA)

# ── Layout ──────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(11, 8))
ax_chiP, ax_chiL, ax_U4P, ax_MvsL = axes.flat

RED   = '#d62728'; BLUE = '#1f77b4'; ORG = '#ff7f0e'; GREY = '#999999'
GREEN = '#2ca02c'

# ── (a) chi_int vs P ─────────────────────────────────────────────────
ax = ax_chiP
ax.scatter(PA,  chiA, color=RED,  s=70, zorder=5, label=r'Phase A: $L=80$')
ax.scatter(PC,  chiC, color=BLUE, s=70, zorder=5, label=r'Phase C: $L=120$')
ax.axhline(noise_floor, color=GREY, ls='--', lw=1.5,
           label=rf'Noise floor $\approx${noise_floor:.4f}')
# What chi_int SHOULD look like if gamma=0.71
P_ref = 0.050; chi_ref = noise_floor
P_line = np.logspace(-2.5, -1.3, 60)
chi_theory = chi_ref * (P_line / P_ref) ** (-gamma_Fisher)
ax.plot(P_line, chi_theory, 'k:', lw=1.5,
        label=rf'Fisher $\gamma={gamma_Fisher:.2f}$ (expected shape)')
ax.set_xscale('log'); ax.set_yscale('log')
ax.set_xlabel(r'$P_\mathrm{causal}$', fontsize=11)
ax.set_ylabel(r'$\chi_\mathrm{int} = L^2\,\mathrm{var}(M)$', fontsize=11)
ax.set_title(r'(a) $\chi_\mathrm{int}$ vs $P$ — NULL (noise floor)', fontsize=11)
ax.legend(fontsize=8)
ax.grid(True, which='both', alpha=0.25)
ax.annotate('ALL points on\nnoise floor', xy=(0.010, noise_floor),
            xytext=(0.020, noise_floor*1.5),
            arrowprops=dict(arrowstyle='->', color='black', lw=1.2), fontsize=8)

# ── (b) chi_int vs L ─────────────────────────────────────────────────
ax = ax_chiL
ax.scatter(LB, chiB, color=GREEN, s=70, zorder=5, label=r'Phase B: $P=0.010$')
ax.axhline(np.mean(chiB), color=GREY, ls='--', lw=1.5,
           label=rf'Mean = {np.mean(chiB):.4f} (flat)')
# Expected: chi_int ~ L^{gamma/nu} in ordered FSS
L_plot = np.linspace(35, 175, 100)
L_ref = 80; chi_ref_B = np.interp(L_ref, LB, chiB)
chi_fss = chi_ref_B * (L_plot / L_ref) ** (gamma_Fisher / nu)
ax.plot(L_plot, chi_fss, 'k:', lw=1.5,
        label=rf'FSS: $\chi \sim L^{{\gamma/\nu={gamma_Fisher/nu:.2f}}}$ (expected)')
ax.set_xscale('log'); ax.set_yscale('log')
ax.set_xlabel(r'$L$', fontsize=11)
ax.set_ylabel(r'$\chi_\mathrm{int} = L^2\,\mathrm{var}(M)$', fontsize=11)
ax.set_title(r'(b) $\chi_\mathrm{int}$ vs $L$ — NULL (no FSS divergence)', fontsize=11)
ax.legend(fontsize=8)
ax.grid(True, which='both', alpha=0.25)

# ── (c) U4 vs P ──────────────────────────────────────────────────────
ax = ax_U4P
ax.scatter(PA, U4A, color=RED,  s=70, zorder=5, label=r'Phase A: $L=80$')
ax.scatter(PC, U4C, color=BLUE, s=70, zorder=5, label=r'Phase C: $L=120$')
ax.axhline(0,   color='k',    ls='-',  lw=0.8, alpha=0.5)
ax.axhline(2/3, color=ORG,  ls='--', lw=1.2,
           label=r'Ordered limit $U_4 \to 2/3$')
ax.axhline(0.1, color=GREY, ls=':',  lw=1.0, label=r'$U_4=0.1$ threshold')
ax.set_xscale('log')
ax.set_xlabel(r'$P_\mathrm{causal}$', fontsize=11)
ax.set_ylabel(r'$U_4$ (Binder cumulant)', fontsize=11)
ax.set_title(r'(c) Binder $U_4$ vs $P$: barely ordered at $P\leq 0.030$', fontsize=11)
ax.legend(fontsize=8)
ax.grid(True, which='both', alpha=0.25)
ax.annotate(r'Disordered ($U_4\approx 0$)', xy=(0.010, 0.002),
            xytext=(0.004, 0.08),
            arrowprops=dict(arrowstyle='->', color='black', lw=1.2), fontsize=8)

# ── (d) absM vs L ────────────────────────────────────────────────────
ax = ax_MvsL
ax.scatter(LB, absMb, color=GREEN, s=70, zorder=5,
           label=r'Phase B: $P=0.010$')
# Fit power law
logL = np.log(np.array(LB, float))
logM = np.log(np.array(absMb, float))
slope, intercept = np.polyfit(logL, logM, 1)
L_plot2 = np.linspace(35, 175, 100)
ax.plot(L_plot2, np.exp(intercept) * (L_plot2 ** slope),
        'k--', lw=1.5, label=rf'$|M|\sim L^{{{slope:.2f}}}$')
# Expected for ordered: |M| ~ L^{-beta/nu}
ax.plot(L_plot2, np.exp(intercept) * (L_plot2 / L_ref) ** (-beta/nu) * absMb[2],
        ':', color=ORG, lw=1.2,
        label=rf'Ordered: $L^{{-\beta/\nu={beta/nu:.2f}}}$')
# Expected for disordered: |M| ~ L^{-1}
ax.plot(L_plot2, absMb[2] * (L_ref / L_plot2),
        ':', color=GREY, lw=1.2, label=r'Disordered: $L^{-1}$')
ax.set_xscale('log'); ax.set_yscale('log')
ax.set_xlabel(r'$L$', fontsize=11)
ax.set_ylabel(r'$|\bar{M}|$', fontsize=11)
ax.set_title(r'(d) $|M|$ vs $L$ at $P=0.010$ — near $L^{-1}$ (disordered)', fontsize=11)
ax.legend(fontsize=8)
ax.grid(True, which='both', alpha=0.25)

fig.suptitle(
    r'Paper 76: $\gamma$ Measurement via $\chi_\mathrm{int}=L^2\,\mathrm{var}(M)$ — NULL Result'
    '\n'
    r'$\chi_\mathrm{int}\approx 0.0024$ (noise floor) everywhere; $\gamma_\mathrm{direct}$ UNDEFINED; '
    r'$\gamma_{\rm Fisher}=\nu(2-\eta)=' + f'{gamma_Fisher:.2f}' + r'$ (indirect)',
    fontsize=10
)

plt.tight_layout()
fig.savefig(str(OUT), dpi=180, bbox_inches='tight')
print(f'Figure saved: {OUT}')
