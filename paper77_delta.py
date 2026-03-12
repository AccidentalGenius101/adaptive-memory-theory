"""
paper77_delta.py  --  Equation-of-state measurement of delta exponent
                      M ~ h^{1/delta} as h -> 0 at P -> P_c = 0+

Strategy
--------
At criticality (P -> 0+), an external conjugate field h breaks Z2 symmetry:
    M(h) ~ A * h^{1/delta}

The field h is added to phi directly in the CUDA kernel:
    phi <- phi * FIELD_DECAY + h * z0f    (zone-0 gets +h, zone-1 gets -h)

At equilibrium (no other dynamics): phi_eq = h / (1 - FIELD_DECAY) = 1000*h
So effective field seen by M is ~1000x amplified by the slow phi decay.
We therefore use h values in [1e-6, 1e-3], giving effective phi bias ~[1e-3, 1].

Phases
------
Phase A: L=80, P=0.001, h in [1e-6..1e-3], 8 seeds, 100k steps
         -> Primary delta fit: M ~ h^{1/delta}

Phase B: Same h scan at P=0.003 and P=0.010
         -> Check delta stability; Widom scaling collapse

Phase C: L=80, P=0.001, h=0 (baseline); L in {40,80,120} at h=1e-4
         -> Finite-size check: delta should be L-independent

Predicted (indirect): delta = 1 + gamma/beta = 1 + 0.706/0.628 ≈ 2.12
"""

import os, sys, json, time
import numpy as np
from scipy.optimize import curve_fit
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
os.chdir(Path(__file__).parent)

from vcml_gpu_v4 import run_batch_gpu_v4

OUT = Path('results') / 'paper77'
OUT.mkdir(parents=True, exist_ok=True)

SEEDS   = list(range(8))
R_W     = 5

# External field values: geometric sequence 1e-6 -> 1e-3
H_VALS = [0.0, 1e-6, 3e-6, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3]

# ── helpers ───────────────────────────────────────────────────────────
def mean_absM(results):
    """Mean |M| across seeds for a list of result dicts."""
    return np.mean([r['absM'] for r in results])

def sem_absM(results):
    return np.std([r['absM'] for r in results]) / np.sqrt(len(results))

def power_law(h, A, inv_delta):
    return A * h ** inv_delta

def fit_delta(h_vals, M_vals, M_errs=None):
    """Fit M = A * h^{1/delta}, return (delta, delta_err, A, R2)."""
    mask = np.array(h_vals) > 0
    h = np.array(h_vals)[mask]
    M = np.array(M_vals)[mask]
    e = np.array(M_errs)[mask] if M_errs is not None else None
    try:
        popt, pcov = curve_fit(power_law, h, M, p0=[1.0, 0.5],
                               sigma=e, absolute_sigma=(e is not None),
                               maxfev=5000)
        A, inv_d = popt
        A_err    = np.sqrt(pcov[0, 0])
        id_err   = np.sqrt(pcov[1, 1])
        delta    = 1.0 / inv_d
        delta_err= id_err / inv_d**2
        # R2
        M_pred = power_law(h, *popt)
        ss_res = np.sum((M - M_pred)**2)
        ss_tot = np.sum((M - M.mean())**2)
        R2     = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
        return delta, delta_err, A, R2
    except Exception as ex:
        print(f'  fit failed: {ex}')
        return None, None, None, None

# ── Phase A: primary h scan, P=0.001, L=80 ───────────────────────────
print('=' * 60)
print('Phase A: L=80, P=0.001, h scan (100k steps, 8 seeds)')
print('=' * 60)

PA_P   = 0.001
PA_L   = 80
PA_N   = 100_000

PA_results = {}   # h -> [result_dicts]
for h in H_VALS:
    t0 = time.time()
    res = run_batch_gpu_v4(PA_L, [PA_P], SEEDS, PA_N, R_W, h_field=h)
    dt  = time.time() - t0
    M   = mean_absM(res)
    e   = sem_absM(res)
    PA_results[h] = res
    print(f'  h={h:.1e}  |M|={M:.4f} ± {e:.4f}  ({dt:.0f}s)')

# fit delta
h_list = list(PA_results.keys())
M_list = [mean_absM(PA_results[h]) for h in h_list]
e_list = [sem_absM(PA_results[h])  for h in h_list]
delta_A, d_err_A, A_A, R2_A = fit_delta(h_list, M_list, e_list)
print(f'\nPhase A fit: delta={delta_A:.3f} ± {d_err_A:.3f}  A={A_A:.4f}  R²={R2_A:.3f}')

# save
with open(OUT / 'phaseA.json', 'w') as f:
    json.dump({'P': PA_P, 'L': PA_L, 'nsteps': PA_N,
               'h_vals': h_list,
               'M_mean': M_list, 'M_sem': e_list,
               'delta': delta_A, 'delta_err': d_err_A,
               'A': A_A, 'R2': R2_A}, f, indent=2)

# ── Phase B: delta vs P (3 P values) ─────────────────────────────────
print('\n' + '=' * 60)
print('Phase B: delta stability across P values (50k steps, 8 seeds)')
print('=' * 60)

PB_Ps    = [0.001, 0.003, 0.010]
PB_L     = 80
PB_N     = 50_000
PB_H     = [h for h in H_VALS if h > 0]   # skip h=0

PB_data = {}
for P in PB_Ps:
    PB_data[P] = {}
    for h in PB_H:
        res = run_batch_gpu_v4(PB_L, [P], SEEDS, PB_N, R_W, h_field=h)
        PB_data[P][h] = mean_absM(res)

    M_list = [PB_data[P][h] for h in PB_H]
    delta_B, d_err_B, A_B, R2_B = fit_delta(PB_H, M_list)
    print(f'  P={P:.3f}  delta={delta_B:.3f} ± {d_err_B:.3f}  R²={R2_B:.3f}')

with open(OUT / 'phaseB.json', 'w') as f:
    json.dump({'Ps': PB_Ps, 'L': PB_L, 'nsteps': PB_N,
               'h_vals': PB_H,
               'data': {str(P): {str(h): v for h, v in PB_data[P].items()}
                        for P in PB_Ps}}, f, indent=2)

# ── Phase C: FSS at fixed h, check L independence ─────────────────────
print('\n' + '=' * 60)
print('Phase C: FSS at h=1e-4, P=0.001 (50k steps, 8 seeds)')
print('=' * 60)

PC_Ls = [40, 80, 120]
PC_H  = 1e-4
PC_P  = 0.001
PC_N  = 50_000

PC_data = {}
for L in PC_Ls:
    res = run_batch_gpu_v4(L, [PC_P], SEEDS, PC_N, R_W, h_field=PC_H)
    M   = mean_absM(res)
    e   = sem_absM(res)
    PC_data[L] = {'M': M, 'e': e}
    print(f'  L={L:3d}  |M|={M:.4f} ± {e:.4f}')

with open(OUT / 'phaseC.json', 'w') as f:
    json.dump({'Ls': PC_Ls, 'h': PC_H, 'P': PC_P, 'nsteps': PC_N,
               'data': {str(L): PC_data[L] for L in PC_Ls}}, f, indent=2)

# ── Summary ───────────────────────────────────────────────────────────
print('\n' + '=' * 60)
print('PAPER 77 SUMMARY')
print('=' * 60)
print(f'delta (Phase A, P=0.001): {delta_A:.3f} ± {d_err_A:.3f}  R²={R2_A:.3f}')
print(f'Predicted (Widom, indirect): ~2.12')
print(f'Mean field: 3.0,  2D Ising: 15.0')
print(f'Results in: {OUT}')
