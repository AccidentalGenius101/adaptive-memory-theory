"""
Paper 75 - Figure generator
Panels:
  (a) tau vs P -- Phase A (L=160) + Phase B (L=80) with tau-floor and FSS boundary
  (b) tau vs L -- Phase C FSS
  (c) chi vs P -- Phase B (flat chi at noise floor)
  (d) chi vs L -- Phase C (chi ~ L^{-2} geometric dilution)
"""

import json, math, numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

RESULTS = Path(__file__).parent / 'results' / 'paper75_analysis.json'
OUT     = Path(__file__).parent / 'results' / 'paper75_figure1.pdf'

with open(RESULTS) as f:
    d = json.load(f)

# Unpack data
tau_A = d['tau_A']   # {P: tau}
tau_B = d['tau_B']   # {P: tau}
tau_C = d['tau_C']   # {L: tau}
chi_A = d['chi_A']   # {P: chi}
chi_B = d['chi_B']   # {P: chi}
chi_C = d['chi_C']   # {L: chi}

pc_A_arr = sorted(tau_A.keys(), key=float)
pc_B_arr = sorted(tau_B.keys(), key=float)
lc_arr   = sorted(tau_C.keys(), key=int)

PA = [float(p) for p in pc_A_arr];  tauA = [tau_A[p] for p in pc_A_arr]
PB = [float(p) for p in pc_B_arr];  tauB = [tau_B[p] for p in pc_B_arr]
LC = [int(l)   for l in lc_arr];    tauCv = [tau_C[l] for l in lc_arr]

chiA = [chi_A[p] for p in pc_A_arr]
chiB = [chi_B[p] for p in pc_B_arr]
chiCv= [chi_C[l] for l in lc_arr]

# Fit references from Paper 74
z_P74 = 0.48; nu_P74 = 0.98
# tau ~ P^{-z*nu}, normalised to P74 Phase A midpoint
P_ref = 1e-3; tau_ref_P74 = 3000  # approximate
def tau_p74(P): return tau_ref_P74 * (P / P_ref) ** (-z_P74 * nu_P74)

# FSS scaling tau ~ L^z
z_C  = d['z_C'];   r2_C = d['r2_zC']
# fit line
LC_arr = np.array(LC); logL = np.log(LC_arr)
logT   = np.log(np.array(tauCv))
zC_fit = np.polyfit(logL, logT, 1)

# chi ~ L^{-2} reference
chiC_arr = np.array(chiCv)
chiC_ref = chiC_arr[0] * (LC[0]/np.array(LC, float))**2

# ── Layout ──────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(11, 8))
ax_tau_P, ax_tau_L, ax_chi_P, ax_chi_L = axes.flat

GREY  = '#999999'; RED = '#d62728'; BLUE = '#1f77b4'; ORG = '#ff7f0e'
GREEN = '#2ca02c'

# ── (a) tau vs P ─────────────────────────────────────────────────────────────
ax = ax_tau_P
Pall  = PA + PB
tall  = tauA + tauB
col   = [RED]*len(PA) + [BLUE]*len(PB)
for P,t,c in zip(Pall,tall,col):
    ax.scatter(P, t, color=c, s=60, zorder=5)

# Paper 74 reference line
P_line = np.logspace(-5, -2, 100)
ax.plot(P_line, [tau_p74(p) for p in P_line], 'k--', lw=1.5,
        label=r'P74: $\tau \sim P^{-z\nu}$, $z=0.48$')
# tau_VCSM floor
ax.axhline(200,  color=GREY, ls=':', lw=1.2, label=r'$\tau_\mathrm{VCSM}=200$')
# tau_floor phi
ax.axhline(1000, color=ORG,  ls=':', lw=1.2, label=r'$\tau_\phi\approx 1000$')
# FSS ceiling L=80: once xi(P)=L the tau saturates
P_fss = (1/80)**(1/nu_P74)  # xi ~ P^{-nu} = L => P = L^{-1/nu}
ax.axvline(P_fss, color=GREEN, ls='-.', lw=1.2,
           label=rf'FSS boundary $L=80$: $P\approx{P_fss:.1e}$')

# dummy scatters for legend
ax.scatter([], [], color=RED,  s=60, label=r'Phase A: $L=160$')
ax.scatter([], [], color=BLUE, s=60, label=r'Phase B: $L=80$')
ax.set_xscale('log'); ax.set_yscale('log')
ax.set_xlabel(r'$P_\mathrm{causal}$', fontsize=11)
ax.set_ylabel(r'$\tau_\mathrm{corr}$', fontsize=11)
ax.set_title(r'(a) Correlation time vs $P$', fontsize=11)
ax.legend(fontsize=7.5, loc='upper right')
ax.grid(True, which='both', alpha=0.25)

# ── (b) tau vs L ─────────────────────────────────────────────────────────────
ax = ax_tau_L
ax.scatter(LC, tauCv, color=BLUE, s=70, zorder=5, label=r'Phase C ($P=0.0005$)')
# fit line
L_plot = np.linspace(35, 175, 100)
ax.plot(L_plot, np.exp(np.polyval(zC_fit, np.log(L_plot))),
        'k--', lw=1.5,
        label=rf'slope $z_C={z_C:.3f}$ ($R^2={r2_C:.2f}$)')
# Model A reference
ref_norm = np.exp(zC_fit[1])
ax.plot(L_plot, ref_norm * (L_plot/L_plot[50])**2.0 * tauCv[2],
        ls=':', color=GREY, lw=1.2, label=r'Model A: $z=2$')
ax.set_xscale('log'); ax.set_yscale('log')
ax.set_xlabel(r'$L$', fontsize=11)
ax.set_ylabel(r'$\tau_\mathrm{corr}$', fontsize=11)
ax.set_title(r'(b) FSS relaxation time $\tau_L$ vs $L$', fontsize=11)
ax.legend(fontsize=8)
ax.grid(True, which='both', alpha=0.25)

# ── (c) chi vs P ─────────────────────────────────────────────────────────────
ax = ax_chi_P
for P, c in zip(PA, chiA):
    ax.scatter(P, c, color=RED, s=60, zorder=5)
for P, c in zip(PB, chiB):
    ax.scatter(P, c, color=BLUE, s=60, zorder=5)
# flat reference
ax.axhline(np.mean(chiB), color=BLUE, ls='--', lw=1.2, alpha=0.6,
           label=r'Phase B mean (flat: $\gamma_B\approx 0$)')
ax.scatter([], [], color=RED, s=60, label=r'Phase A: $L=160$')
ax.scatter([], [], color=BLUE, s=60, label=r'Phase B: $L=80$')
ax.set_xscale('log'); ax.set_yscale('log')
ax.set_xlabel(r'$P_\mathrm{causal}$', fontsize=11)
ax.set_ylabel(r'$\chi = \mathrm{var}(M)$', fontsize=11)
ax.set_title(r'(c) Susceptibility proxy $\chi$ vs $P$ — NULL', fontsize=11)
ax.legend(fontsize=8)
ax.grid(True, which='both', alpha=0.25)
ax.annotate('Noise floor\n(disordered phase)', xy=(1e-4, np.mean(chiB)),
            xytext=(3e-4, np.mean(chiB)*3),
            arrowprops=dict(arrowstyle='->', color='black', lw=1.2),
            fontsize=8)

# ── (d) chi vs L ─────────────────────────────────────────────────────────────
ax = ax_chi_L
ax.scatter(LC, chiCv, color=BLUE, s=70, zorder=5, label=r'Phase C ($P=0.0005$)')
ax.plot(LC, chiC_ref, 'k--', lw=1.5, label=r'$\chi \sim L^{-2}$ (geometric dilution)')
ax.set_xscale('log'); ax.set_yscale('log')
ax.set_xlabel(r'$L$', fontsize=11)
ax.set_ylabel(r'$\chi = \mathrm{var}(M)$', fontsize=11)
ax.set_title(r'(d) $\chi$ vs $L$ — geometric dilution $\chi\sim L^{-2}$', fontsize=11)
ax.legend(fontsize=8)
ax.grid(True, which='both', alpha=0.25)
ax.annotate(r'$\chi \sim L^{-2}$: fluctuation noise,' + '\nnot physical susceptibility',
            xy=(100, chiCv[3]), xytext=(50, chiCv[0]*0.3),
            arrowprops=dict(arrowstyle='->', color='black', lw=1.2),
            fontsize=8)

fig.suptitle(
    r'Paper 75: $z$ Probe at Deep $P$; $\gamma$ Null; FSS Blocking'
    '\n'
    r'$z_\mathrm{P74}=0.48$ stands; $\chi$ is noise-floor (disordered phase); '
    r'$\xi(P) > L$ blocks $\tau(P)$ divergence below $P_\mathrm{FSS}(L{=}80)$',
    fontsize=10
)

plt.tight_layout()
fig.savefig(str(OUT), dpi=180, bbox_inches='tight')
print(f'Figure saved: {OUT}')
