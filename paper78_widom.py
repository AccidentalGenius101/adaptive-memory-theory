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

# ── Phase D: Scaling function Phi(x) fit -- 0+1d vs QFT ──────────────
print('\n' + '=' * 60)
print('Phase D: Phi(x) functional form -- 0+1d Langevin vs QFT')
print('=' * 60)
print('Candidate forms (all use measured delta=1.623):')
print('  F1 (mean-field null): Phi^3 - Phi = x  [delta_MF=3, expect bad fit]')
print('  F2 (additive crossover): Phi = (C^d + x)^(1/d)  [1-param, power potential]')
print('  F3 (algebraic crossover): Phi = C*(1 + x/x_c)^(1/d)  [2-param, two-scale]')
print('  F4 (power crossover): Phi = Phi0*(1+(x/x0)^a)^(1/(a*d))  [3-param, general]')

from scipy.optimize import brentq, curve_fit as _cf

def _r2(y_obs, y_pred):
    ss_res = np.sum((y_obs - y_pred)**2)
    ss_tot = np.sum((y_obs - y_obs.mean())**2)
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

def _chi2r(y_obs, y_pred, ye, npar):
    dof = len(y_obs) - npar
    return np.sum(((y_obs - y_pred) / ye)**2) / dof if dof > 0 else np.nan

# Use Phase A collapsed data (exclude h=0 already done above)
x_D  = x_sorted
y_D  = y_sorted
ye_D = ye_sorted

PD = {}

# F1: mean-field crossover -- rescaled (u^3 - u = rhs, Phi = C*u)
def _mf_phi(x_arr, C_mf, x_mf):
    out = []
    for xi in x_arr:
        rhs = xi / x_mf
        def _eqn(u):
            return u**3 - u - rhs
        try:
            u_lo = max(1.0, abs(rhs)**(1.0/3))
            u_hi = u_lo + max(10.0, abs(rhs))
            while _eqn(u_hi) < 0:
                u_hi *= 2
            u_sol = brentq(_eqn, u_lo, u_hi)
        except Exception:
            u_sol = abs(rhs)**(1.0/3) + 1.0
        out.append(C_mf * u_sol)
    return np.array(out)

print('\n--- F1: mean-field crossover ---')
try:
    x0_mf = float(x_D[len(x_D)//2]) if len(x_D) > 0 else 0.1
    popt, _ = _cf(_mf_phi, x_D, y_D, sigma=ye_D, absolute_sigma=True,
                  p0=[float(y_D[0]), x0_mf], bounds=([0.01, 1e-6], [20, 1e4]))
    yp = _mf_phi(x_D, *popt)
    R2_f1 = _r2(y_D, yp); chi_f1 = _chi2r(y_D, yp, ye_D, 2)
    print(f'  C_mf={popt[0]:.4f}  x_mf={popt[1]:.4e}')
    print(f'  R2={R2_f1:.4f}  chi2/dof={chi_f1:.2f}  (delta_MF=3 forced)')
    PD['F1_MF'] = {'C_mf': float(popt[0]), 'x_mf': float(popt[1]),
                   'R2': R2_f1, 'chi2dof': float(chi_f1), 'y_pred': yp.tolist()}
except Exception as ex:
    print(f'  FAILED: {ex}')
    PD['F1_MF'] = None

# F2: additive crossover Phi = (C^delta + x)^(1/delta)
def _phi_add(x, C):
    return (np.maximum(C**DELTA + x, 0))**(1.0 / DELTA)

print('\n--- F2: additive crossover ---')
try:
    popt, _ = _cf(_phi_add, x_D, y_D, sigma=ye_D, absolute_sigma=True,
                  p0=[float(y_D[0])], bounds=([1e-6], [20.0]))
    yp = _phi_add(x_D, *popt)
    R2_f2 = _r2(y_D, yp); chi_f2 = _chi2r(y_D, yp, ye_D, 1)
    print(f'  C={popt[0]:.4f}  [V_eff~|M|^{DELTA+1:.3f}, 0+1d power potential]')
    print(f'  R2={R2_f2:.4f}  chi2/dof={chi_f2:.2f}')
    PD['F2_additive'] = {'C': float(popt[0]), 'delta_used': DELTA,
                          'R2': R2_f2, 'chi2dof': float(chi_f2), 'y_pred': yp.tolist()}
except Exception as ex:
    print(f'  FAILED: {ex}')
    PD['F2_additive'] = None

# F3: algebraic crossover Phi = C*(1 + x/x_c)^(1/delta)
def _phi_alg(x, C, x_c):
    return C * (1.0 + x / x_c)**(1.0 / DELTA)

print('\n--- F3: algebraic crossover ---')
try:
    x0_alg = float(np.median(x_D)) if len(x_D) > 0 else 0.1
    popt, _ = _cf(_phi_alg, x_D, y_D, sigma=ye_D, absolute_sigma=True,
                  p0=[float(y_D[0]), x0_alg], bounds=([1e-6, 1e-8], [20, 1e5]))
    yp = _phi_alg(x_D, *popt)
    R2_f3 = _r2(y_D, yp); chi_f3 = _chi2r(y_D, yp, ye_D, 2)
    print(f'  C={popt[0]:.4f}  x_c={popt[1]:.4e}')
    print(f'  R2={R2_f3:.4f}  chi2/dof={chi_f3:.2f}')
    PD['F3_algebraic'] = {'C': float(popt[0]), 'x_c': float(popt[1]),
                           'delta_used': DELTA, 'R2': R2_f3, 'chi2dof': float(chi_f3),
                           'y_pred': yp.tolist()}
except Exception as ex:
    print(f'  FAILED: {ex}')
    PD['F3_algebraic'] = None

# F4: power crossover Phi = Phi0*(1+(x/x0)^alpha)^(1/(alpha*delta))
def _phi_pc(x, Phi0, x0, alpha):
    return Phi0 * (1.0 + (x / x0)**alpha)**(1.0 / (alpha * DELTA))

print('\n--- F4: power crossover ---')
try:
    x0_pc = float(np.median(x_D)) if len(x_D) > 0 else 0.1
    popt, _ = _cf(_phi_pc, x_D, y_D, sigma=ye_D, absolute_sigma=True,
                  p0=[float(y_D[0]), x0_pc, 1.0],
                  bounds=([1e-6, 1e-8, 0.1], [20, 1e5, 10]))
    yp = _phi_pc(x_D, *popt)
    R2_f4 = _r2(y_D, yp); chi_f4 = _chi2r(y_D, yp, ye_D, 3)
    print(f'  Phi0={popt[0]:.4f}  x0={popt[1]:.4e}  alpha={popt[2]:.3f}')
    print(f'  R2={R2_f4:.4f}  chi2/dof={chi_f4:.2f}')
    PD['F4_power_crossover'] = {
        'Phi0': float(popt[0]), 'x0': float(popt[1]), 'alpha': float(popt[2]),
        'delta_used': DELTA, 'R2': R2_f4, 'chi2dof': float(chi_f4),
        'y_pred': yp.tolist()
    }
except Exception as ex:
    print(f'  FAILED: {ex}')
    PD['F4_power_crossover'] = None

print('\n--- Phase D ranking ---')
valid = {k: v for k, v in PD.items() if v is not None}
ranked = sorted(valid.items(), key=lambda kv: kv[1].get('R2', -1), reverse=True)
for rank, (name, res) in enumerate(ranked, 1):
    print(f'  #{rank} {name:<25s}  R2={res["R2"]:.4f}  chi2/dof={res.get("chi2dof", float("nan")):.2f}')

best_name = ranked[0][0] if ranked else 'none'
best_res  = ranked[0][1] if ranked else {}
print(f'\n  Best form: {best_name}  R2={best_res.get("R2", 0):.4f}')
if 'F1' in best_name:
    print('  INTERPRETATION: Mean-field wins => delta_eff~3, system is d>=4 or MF.')
    print('  (Surprising -- contradicts measured delta=1.623. Check data quality.)')
elif 'F2' in best_name:
    C_v = best_res.get('C', 0)
    print(f'  INTERPRETATION: Additive crossover => single-scale power potential.')
    print(f'  V_eff(M) ~ |M|^{DELTA+1:.3f}  [0+1d Langevin, d_eff=0 consistent]')
    print(f'  Phi(0) = C = {C_v:.4f}  (zero-field amplitude)')
elif 'F3' in best_name:
    xc_v = best_res.get('x_c', 0)
    print(f'  INTERPRETATION: Algebraic crossover => two-scale structure.')
    print(f'  Crossover at x_c = h*/P^{BD:.3f} = {xc_v:.4e}')
    print(f'  0+1d Langevin with two relevant couplings [upstream of QFT].')
elif 'F4' in best_name:
    a_v = best_res.get('alpha', 1)
    print(f'  INTERPRETATION: Power crossover, alpha={a_v:.3f}.')
    if abs(a_v - 1.0) < 0.15:
        print('  alpha~1 => reduces to algebraic F3.')
    else:
        print(f'  alpha!= 1 => non-trivial crossover exponent. Novel 0+1d universality.')

with open(OUT / 'phaseD.json', 'w') as f:
    json.dump({
        'description': 'Phi(x) scaling function fits -- 0+1d vs QFT',
        'x_data': x_D.tolist(), 'y_data': y_D.tolist(), 'ye_data': ye_D.tolist(),
        'beta': BETA, 'delta': DELTA, 'beta_delta': BD,
        'fits': PD, 'best_form': best_name,
    }, f, indent=2)

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
if best_name != 'none':
    print(f'Best Phi(x) form: {best_name}  R2={best_res.get("R2", 0):.4f}  (Phase D)')
print(f'Results in: {OUT}')
