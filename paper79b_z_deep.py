"""
paper79b_z_deep.py  --  Deep z rerun: tau ~ P^(-z*nu) in true critical regime

Motivation: Paper 74 measured z=0.48 (R2=0.71, non-monotone) using
P in [1e-4, 1e-3]. FSS blocking (xi >> L) contaminated results at P<1e-3.
This paper pushes deeper while controlling for FSS blocking.

Strategy
--------
Use tau ~ P^{-znu} with nu=0.98. For the fit to be clean:
  - Need xi(P) = P^{-nu} << L, i.e. P >> L^{-1/nu} ~ L^{-1.02}
  - At L=80: P >> 80^{-1.02} ~ 0.012  (FSS blocking for P < 0.012!)
  - At L=40: P >> 40^{-1.02} ~ 0.025
  - At L=20: P >> 20^{-1.02} ~ 0.051

So the ONLY accessible regime free from FSS blocking is P in [0.012, 0.050]
at L=80. This is exactly where tau_VCSM~200 dominates (Paper 73).

Resolution: measure tau at MULTIPLE L and decompose:
  tau_obs(L,P) = tau_crit(P) + tau_VCSM  (approximately)
  tau_crit(P) = tau_obs(L->inf, P) - tau_VCSM

Two-pronged approach:
  Phase A: tau(P) at L=80, P in {0.010, 0.020, 0.030, 0.050} (safe FSS window)
           Expect tau_obs ~ tau_VCSM ~ 200 (flat) -- documents the floor
  Phase B: tau(L) at P=0.020, L in {40,60,80,120,160}
           If tau independent of L: tau = tau_VCSM (intrinsic, not critical)
           If tau ~ L^z: FSS regime -> extract z directly
  Phase C: tau(P) at L=40, P in {0.020,0.030,0.050,0.100}
           Smaller L -> FSS blocking pushed to higher P
           At L=40, FSS-free for P > 0.025
           Compare to L=80 results to isolate tau_crit from tau_VCSM
"""

import os, sys, json, time
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
os.chdir(Path(__file__).parent)

from vcml_cluster import run_batch_cluster

OUT = Path('results') / 'paper79b'
OUT.mkdir(parents=True, exist_ok=True)

SEEDS  = list(range(8))
R_W    = 5
NU     = 0.98
Z_V    = 0.48   # target

def get_tau(res_list):
    """Extract mean tau_corr and sem from result list."""
    taus = [r['tau_corr'] for r in res_list if 'tau_corr' in r and r['tau_corr'] > 0]
    if not taus:
        return float('nan'), float('nan')
    return float(np.mean(taus)), float(np.std(taus)/np.sqrt(len(taus)))

# ── Phase A: tau(P) at L=80 -- document tau_VCSM floor ───────────────
print('=' * 60)
print('Phase A: tau(P) at L=80, FSS-safe window')
print(f'  FSS-free for P > L^(-1/nu) = 80^(-1.02) = {80**(-1/NU):.3f}')
print('=' * 60)

PA_P  = [0.010, 0.020, 0.030, 0.050]
PA_L  = 80
PA_N  = 300_000

PA_data = {}
for P in PA_P:
    t0  = time.time()
    res = run_batch_cluster(PA_L, [P], SEEDS, PA_N, R_W, verbose=False)
    dt  = time.time() - t0
    tau, tau_e = get_tau(res)
    absM = float(np.mean([r['absM'] for r in res]))
    PA_data[P] = {'tau': tau, 'tau_e': tau_e, 'absM': absM}
    print(f'  P={P:.3f}  tau={tau:.0f}±{tau_e:.0f}  |M|={absM:.4f}  ({dt:.0f}s)')

P_arr  = np.array(PA_P)
tau_arr = np.array([PA_data[P]['tau'] for P in PA_P])
finite  = np.isfinite(tau_arr)
if finite.sum() >= 3:
    coeffs = np.polyfit(np.log(P_arr[finite]), np.log(tau_arr[finite]), 1)
    znu_A  = float(-coeffs[0])
    print(f'\n  tau ~ P^{coeffs[0]:.3f}  =>  z*nu = {znu_A:.3f}')
    print(f'  z = {znu_A/NU:.3f}  (if nu={NU})')
    print(f'  Expected tau_VCSM ~ 200 (flat) if in intrinsic floor regime')
else:
    znu_A = float('nan')

with open(OUT / 'phaseA.json', 'w') as f:
    json.dump({'L': PA_L, 'Ps': PA_P, 'nsteps': PA_N,
               'data': {str(P): PA_data[P] for P in PA_P},
               'znu_fit': znu_A}, f, indent=2)

# ── Phase B: tau(L) at P=0.020 -- FSS z directly ─────────────────────
print('\n' + '=' * 60)
print('Phase B: tau(L) at P=0.020 -- direct z from FSS')
print('  tau ~ L^z if in FSS regime (xi >> L)')
print(f'  xi(P=0.020) ~ {0.020**(-NU):.1f}  vs L range {[40,60,80,120,160]}')
print('=' * 60)

PB_Ls = [40, 60, 80, 120, 160]
PB_P  = 0.020
PB_N  = 300_000

PB_data = {}
for L in PB_Ls:
    t0  = time.time()
    res = run_batch_cluster(L, [PB_P], SEEDS, PB_N, R_W, verbose=False)
    dt  = time.time() - t0
    tau, tau_e = get_tau(res)
    absM = float(np.mean([r['absM'] for r in res]))
    PB_data[L] = {'tau': tau, 'tau_e': tau_e, 'absM': absM}
    print(f'  L={L:3d}  tau={tau:.0f}±{tau_e:.0f}  |M|={absM:.4f}  ({dt:.0f}s)')

L_arr   = np.array(PB_Ls)
tau_B   = np.array([PB_data[L]['tau'] for L in PB_Ls])
finite  = np.isfinite(tau_B) & (tau_B > 0)
if finite.sum() >= 3:
    coeffs = np.polyfit(np.log(L_arr[finite]), np.log(tau_B[finite]), 1)
    z_B    = float(coeffs[0])
    print(f'\n  tau ~ L^{z_B:.3f}  => z_FSS = {z_B:.3f}')
    print(f'  VCML Paper 74 value: z={Z_V}')
    if abs(z_B - Z_V) < 0.05:
        print('  => CONFIRMS z=0.48')
    elif abs(z_B) < 0.05:
        print('  => tau FLAT with L: intrinsic tau_VCSM dominates, not critical z')
    else:
        print(f'  => z_FSS={z_B:.3f} (new estimate from FSS)')
else:
    z_B = float('nan')

with open(OUT / 'phaseB.json', 'w') as f:
    json.dump({'P': PB_P, 'Ls': PB_Ls, 'nsteps': PB_N,
               'xi_P': float(PB_P**(-NU)),
               'data': {str(L): PB_data[L] for L in PB_Ls},
               'z_FSS': z_B}, f, indent=2)

# ── Phase C: tau(P) at L=40 -- compare to Phase A ────────────────────
print('\n' + '=' * 60)
print('Phase C: tau(P) at L=40 -- smaller L, different tau_VCSM/tau_crit mix')
print(f'  FSS-free for P > L^(-1/nu) = 40^(-1.02) = {40**(-1/NU):.3f}')
print('=' * 60)

PC_P  = [0.020, 0.030, 0.050, 0.100]
PC_L  = 40
PC_N  = 300_000

PC_data = {}
for P in PC_P:
    t0  = time.time()
    res = run_batch_cluster(PC_L, [P], SEEDS, PC_N, R_W, verbose=False)
    dt  = time.time() - t0
    tau, tau_e = get_tau(res)
    absM = float(np.mean([r['absM'] for r in res]))
    PC_data[P] = {'tau': tau, 'tau_e': tau_e, 'absM': absM}
    print(f'  P={P:.3f}  tau={tau:.0f}±{tau_e:.0f}  |M|={absM:.4f}  ({dt:.0f}s)')

# Compare L=40 vs L=80 at same P
print('\n  tau comparison L=40 vs L=80:')
common_P = [P for P in PC_P if P in PA_data]
for P in common_P:
    t40 = PC_data[P]['tau']; t80 = PA_data[P]['tau']
    ratio = t40/t80 if t80 > 0 else float('nan')
    print(f'  P={P:.3f}  tau(L=40)={t40:.0f}  tau(L=80)={t80:.0f}  '
          f'ratio={ratio:.3f}  (expected L^z={40**Z_V/80**Z_V:.3f} if z={Z_V})')

with open(OUT / 'phaseC.json', 'w') as f:
    json.dump({'L': PC_L, 'Ps': PC_P, 'nsteps': PC_N,
               'data': {str(P): PC_data[P] for P in PC_P}}, f, indent=2)

# ── Summary ────────────────────────────────────────────────────────────
print('\n' + '=' * 60)
print('PAPER 79b SUMMARY: z rerun')
print('=' * 60)
print(f'  Phase A (tau vs P, L=80):  z*nu = {znu_A:.3f}  => z = {znu_A/NU:.3f}')
print(f'  Phase B (tau vs L, P=0.020): z_FSS = {z_B:.3f}')
print(f'  Paper 74 reference:          z = {Z_V}  (R2=0.71)')
print(f'  Results in: {OUT}')
