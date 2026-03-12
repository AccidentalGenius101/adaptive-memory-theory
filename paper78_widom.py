"""
paper78_widom.py  --  Widom scaling collapse of the equation of state
                      M(P, h) = P^beta * Phi(h / P^{beta*delta})

Strategy
--------
The scaling hypothesis at P_c = 0+ predicts:

    M(P, h) = P^beta * Phi(x),   x = h / P^{beta*delta}

Limits:
    x -> 0 (h << P):  M -> C * P^beta          (zero-field ordered phase)
    x -> inf (h >> P): M -> A * h^{1/delta}     (field-dominated, Paper 77)

If (beta, delta) are correct, plotting M/P^beta vs h/P^{beta*delta}
collapses all (P, h) data onto a single curve Phi(x).

Phases
------
Phase A: 2D (P, h) scan at L=80, 8 seeds, 100k steps [CLUSTER]
         P in {0.001, 0.003, 0.010, 0.030}
         h in {0, 1e-6, 3e-6, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3}
         -> Primary collapse plot

Phase B: L-dependence of collapse at fixed P=0.001, h=1e-4 [CLUSTER]
         L in {40, 60, 80, 120}
         -> Verify collapse is L-independent (genuine scaling)

Phase C: Scaling function fit
         Phi(x) fit to Widom-Griffiths form and power laws
         -> Extract amplitude ratio A/C

Exponents used: beta=0.628, delta=1.623 (Papers 71+77)
Predicted: beta*delta = 1.019 ~ 1 (close to linear)
"""

import os, sys, json, time
import numpy as np
from scipy.optimize import curve_fit
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
os.chdir(Path(__file__).parent)

from vcml_cluster import run_batch_cluster

OUT = Path('results') / 'paper78'
OUT.mkdir(parents=True, exist_ok=True)

SEEDS   = list(range(8))
R_W     = 5
BETA    = 0.628
DELTA   = 1.623
BD      = BETA * DELTA   # = 1.019

# ── Phase A: 2D (P, h) scan ───────────────────────────────────────────
print('=' * 60)
print('Phase A: 2D (P,h) scan, L=80, 8 seeds, 100k steps [CLUSTER]')
print(f'         beta={BETA}, delta={DELTA}, beta*delta={BD:.3f}')
print('=' * 60)

PA_P    = [0.001, 0.003, 0.010, 0.030]
PA_H    = [0.0, 1e-6, 3e-6, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3]
PA_L    = 80
PA_N    = 100_000

PA_data = {}   # (P, h) -> {'M': float, 'e': float}

for P in PA_P:
    PA_data[P] = {}
    for h in PA_H:
        t0  = time.time()
        res = run_batch_cluster(PA_L, [P], SEEDS, PA_N, R_W,
                                h_field=h, verbose=False)
        dt  = time.time() - t0
        Ms  = [r['absM'] for r in res]
        M   = float(np.mean(Ms))
        e   = float(np.std(Ms) / np.sqrt(len(Ms)))
        PA_data[P][h] = {'M': M, 'e': e}
        print(f'  P={P:.3f}  h={h:.1e}  |M|={M:.4f}±{e:.4f}  ({dt:.0f}s)')

# ── Widom collapse ─────────────────────────────────────────────────────
print('\n--- Widom collapse check ---')
print(f'  x = h / P^{BD:.3f},   y = M / P^{BETA}')

x_all, y_all, ye_all = [], [], []
for P in PA_P:
    for h, d in PA_data[P].items():
        if h == 0.0:
            continue
        x = h / P ** BD
        y = d['M'] / P ** BETA
        ye = d['e'] / P ** BETA
        x_all.append(x); y_all.append(y); ye_all.append(ye)
        print(f'    P={P:.3f}  h={h:.1e}  x={x:.3e}  y={y:.4f}')

x_all  = np.array(x_all)
y_all  = np.array(y_all)
ye_all = np.array(ye_all)

# Sort by x for plotting
idx = np.argsort(x_all)
x_sorted  = x_all[idx]
y_sorted  = y_all[idx]
ye_sorted = ye_all[idx]

# Check collapse quality: fit y = A * x^{1/delta} at large x
mask_large = x_sorted > 0.1
if mask_large.sum() >= 3:
    def field_limit(x, A):
        return A * x ** (1.0 / DELTA)
    try:
        popt, _ = curve_fit(field_limit, x_sorted[mask_large],
                            y_sorted[mask_large],
                            sigma=ye_sorted[mask_large],
                            absolute_sigma=True, p0=[1.0])
        A_fit = float(popt[0])
        # R2
        yp = field_limit(x_sorted[mask_large], A_fit)
        ss_res = np.sum((y_sorted[mask_large] - yp)**2)
        ss_tot = np.sum((y_sorted[mask_large] - y_sorted[mask_large].mean())**2)
        R2 = 1 - ss_res/ss_tot if ss_tot > 0 else 0.0
        print(f'\n  Large-x limit: Phi(x) ~ {A_fit:.4f} * x^(1/{DELTA:.3f})')
        print(f'  R2={R2:.3f}  (should be ~1 if collapse valid)')
    except Exception as ex:
        A_fit = None; R2 = None
        print(f'  Large-x fit failed: {ex}')
else:
    A_fit = None; R2 = None

# Zero-field extrapolation: at h=0, M = C * P^beta
h0_P = [P for P in PA_P if PA_data[P][0.0]['M'] > 0.01]
h0_M = [PA_data[P][0.0]['M'] for P in h0_P]
if len(h0_P) >= 2:
    def zf_law(P, C):
        return C * np.array(P) ** BETA
    try:
        popt2, _ = curve_fit(zf_law, h0_P, h0_M, p0=[1.0])
        C_fit = float(popt2[0])
        print(f'  Zero-field:  M ~ {C_fit:.4f} * P^{BETA}')
        if A_fit:
            print(f'  Amplitude ratio A/C = {A_fit/C_fit:.3f}')
    except Exception as ex:
        C_fit = None
        print(f'  Zero-field fit failed: {ex}')
else:
    C_fit = None

with open(OUT / 'phaseA.json', 'w') as f:
    json.dump({
        'Ps': PA_P, 'hs': PA_H, 'L': PA_L, 'nsteps': PA_N,
        'beta': BETA, 'delta': DELTA, 'beta_delta': BD,
        'data': {str(P): {str(h): v for h, v in PA_data[P].items()}
                 for P in PA_P},
        'collapse': {
            'x': x_sorted.tolist(), 'y': y_sorted.tolist(),
            'ye': ye_sorted.tolist(),
            'A_large_x': A_fit, 'R2_large_x': R2,
            'C_zero_field': C_fit,
        }
    }, f, indent=2)

# ── Phase B: L-dependence of collapse ────────────────────────────────
print('\n' + '=' * 60)
print('Phase B: FSS of collapse at P=0.001, h=1e-4 [CLUSTER]')
print('=' * 60)

PB_Ls = [40, 60, 80, 120]
PB_P  = 0.001
PB_H  = 1e-4
PB_N  = 100_000

PB_data = {}
for L in PB_Ls:
    t0  = time.time()
    res = run_batch_cluster(L, [PB_P], SEEDS, PB_N, R_W,
                            h_field=PB_H, verbose=False)
    dt  = time.time() - t0
    Ms  = [r['absM'] for r in res]
    M   = float(np.mean(Ms))
    e   = float(np.std(Ms) / np.sqrt(len(Ms)))
    PB_data[L] = {'M': M, 'e': e}
    x_L = PB_H / PB_P ** BD
    y_L = M / PB_P ** BETA
    print(f'  L={L:3d}  |M|={M:.4f}±{e:.4f}  y={y_L:.4f}  ({dt:.0f}s)')

print(f'  Fixed scaling variable x = {PB_H/PB_P**BD:.4f}')
print(f'  y should be L-independent if collapse is genuine')

y_vals = [PB_data[L]['M'] / PB_P**BETA for L in PB_Ls]
y_spread = max(y_vals) - min(y_vals)
y_mean   = np.mean(y_vals)
print(f'  y spread / y_mean = {y_spread/y_mean:.3f}  (< 0.1 = good collapse)')

with open(OUT / 'phaseB.json', 'w') as f:
    json.dump({'Ls': PB_Ls, 'P': PB_P, 'h': PB_H, 'nsteps': PB_N,
               'x_scaling': float(PB_H / PB_P**BD),
               'data': {str(L): PB_data[L] for L in PB_Ls}}, f, indent=2)

# ── Summary ───────────────────────────────────────────────────────────
print('\n' + '=' * 60)
print('PAPER 78 SUMMARY')
print('=' * 60)
print(f'Scaling variable: x = h / P^{BD:.3f}')
print(f'Scaled M:         y = M / P^{BETA}')
if A_fit and R2:
    print(f'Large-x fit:  Phi(x) ~ {A_fit:.4f} * x^(1/{DELTA:.3f})  R2={R2:.3f}')
if C_fit:
    print(f'Zero-field:   Phi(0) ~ {C_fit:.4f}')
y_spread_str = f'{y_spread/y_mean:.3f}' if y_mean > 0 else 'N/A'
print(f'FSS spread/mean: {y_spread_str}  (Phase B)')
print(f'Results in: {OUT}')
