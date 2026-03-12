"""
paper78_phaseD.py  --  Phase D standalone: fit Phi(x) scaling function
Loads results/paper78/phaseA.json, filters to h >= 3e-4 (field-dominated),
fits candidate functional forms, identifies best fit.

Clean filter motivation: at h < 3e-4, M_field << sigma_M (zero-field noise),
so the response signal is buried. Only h in {3e-4, 1e-3} are field-dominated
across all P values used.
"""

import json, sys
import numpy as np
from pathlib import Path
from scipy.optimize import curve_fit, brentq

sys.path.insert(0, str(Path(__file__).parent))

BETA  = 0.628
DELTA = 1.623
BD    = BETA * DELTA   # 1.019

OUT = Path('results') / 'paper78'
OUT.mkdir(parents=True, exist_ok=True)

# ── Load Phase A data ──────────────────────────────────────────────────
with open(OUT / 'phaseA.json') as f:
    phaseA = json.load(f)

Ps   = phaseA['Ps']
data = phaseA['data']

# ── Build clean dataset (h >= 3e-4 only) ──────────────────────────────
H_CLEAN = [3e-4, 1e-3]

x_c, y_c, ye_c = [], [], []
for P in Ps:
    for h in H_CLEAN:
        key = str(float(h))
        # try exact and nearby keys
        d = data[str(P)].get(key)
        if d is None:
            # search for close key
            for k, v in data[str(P)].items():
                if abs(float(k) - h) / h < 0.01:
                    d = v; break
        if d is None:
            print(f'  WARNING: P={P} h={h:.0e} not found in data')
            continue
        x = h / P**BD
        y = d['M'] / P**BETA
        ye = d['e'] / P**BETA
        x_c.append(x); y_c.append(y); ye_c.append(ye)

x_c  = np.array(x_c);  idx = np.argsort(x_c)
x_c  = x_c[idx];  y_c  = np.array(y_c)[idx];  ye_c = np.array(ye_c)[idx]

print(f'Clean data: {len(x_c)} points (h in {{3e-4, 1e-3}}, field-dominated)')
print(f'  x range: [{x_c.min():.4f}, {x_c.max():.4f}]')
print(f'  y range: [{y_c.min():.4f}, {y_c.max():.4f}]')
for xi, yi, yei in zip(x_c, y_c, ye_c):
    print(f'  x={xi:.4f}  Phi={yi:.4f}±{yei:.4f}')

# ── Fit helpers ────────────────────────────────────────────────────────
def _r2(yo, yp):
    ss_res = np.sum((yo - yp)**2)
    ss_tot = np.sum((yo - yo.mean())**2)
    return 1.0 - ss_res/ss_tot if ss_tot > 0 else 0.0

def _chi2r(yo, yp, ye, npar):
    dof = len(yo) - npar
    return float(np.sum(((yo-yp)/ye)**2)/dof) if dof > 0 else np.nan

# ── F2: additive crossover Phi = (C^delta + x)^(1/delta) ──────────────
# Large-x:  Phi ~ x^(1/delta)  [consistent with delta measurement]
# Small-x:  Phi -> C            [zero-field amplitude]
# Implies V_eff(M) = lambda/(delta+1) * |M|^(delta+1)  [power potential]
def phi_add(x, C):
    return (C**DELTA + x)**(1.0 / DELTA)

# ── F3: algebraic crossover Phi = C*(1 + x/x_c)^(1/delta) ─────────────
# Large-x:  Phi ~ (C/x_c^{1/delta}) * x^{1/delta}
# Small-x:  Phi -> C
# Two-scale structure: amplitude C and crossover scale x_c
def phi_alg(x, C, x_c):
    return C * (1.0 + x/x_c)**(1.0 / DELTA)

# ── F4: power crossover Phi = Phi0*(1+(x/x0)^alpha)^(1/(alpha*delta)) ─
# General 3-param interpolator.
# alpha=1 -> reduces to F3. alpha!=1 -> non-trivial crossover exponent.
def phi_pc(x, Phi0, x0, alpha):
    return Phi0 * (1.0 + (x/x0)**alpha)**(1.0 / (alpha * DELTA))

# ── F1: mean-field crossover (u^3 - u = rhs, Phi=C*u) ─────────────────
# delta_MF=3 forced. Expect poor fit since measured delta=1.623 != 3.
# Included as null hypothesis only.
def phi_mf(x_arr, C_mf, x_mf):
    out = []
    for xi in x_arr:
        rhs = xi / x_mf
        def eqn(u): return u**3 - u - rhs
        try:
            u_lo = max(1.0, abs(rhs)**(1/3))
            u_hi = u_lo + max(10.0, abs(rhs))
            while eqn(u_hi) < 0: u_hi *= 2
            u_sol = brentq(eqn, u_lo, u_hi)
        except Exception:
            u_sol = abs(rhs)**(1/3) + 1.0
        out.append(C_mf * u_sol)
    return np.array(out)

# ── Run fits ───────────────────────────────────────────────────────────
print('\n' + '='*60)
print('Phase D: Phi(x) scaling function fits')
print(f'  Using measured delta={DELTA}, beta={BETA}')
print(f'  Null hypothesis: mean-field delta_MF=3')
print('='*60)

results = {}

print('\n--- F1: mean-field null (delta=3 forced) ---')
try:
    x0g = float(np.median(x_c))
    p0  = [float(y_c[0]), x0g]
    po, _ = curve_fit(phi_mf, x_c, y_c, sigma=ye_c, absolute_sigma=True,
                      p0=p0, bounds=([0.01, 1e-6], [100, 1e5]), maxfev=5000)
    yp = phi_mf(x_c, *po)
    R2 = _r2(y_c, yp); chi = _chi2r(y_c, yp, ye_c, 2)
    print(f'  C_mf={po[0]:.4f}  x_mf={po[1]:.4e}')
    print(f'  R2={R2:.4f}  chi2/dof={chi:.2f}  [delta_MF=3 forced]')
    results['F1_MF'] = {'C_mf': float(po[0]), 'x_mf': float(po[1]),
                         'R2': R2, 'chi2dof': chi, 'y_pred': yp.tolist(),
                         'npar': 2}
except Exception as ex:
    print(f'  FAILED: {ex}'); results['F1_MF'] = None

print('\n--- F2: additive crossover Phi=(C^delta+x)^(1/delta) ---')
try:
    po, _ = curve_fit(phi_add, x_c, y_c, sigma=ye_c, absolute_sigma=True,
                      p0=[float(y_c[0])], bounds=([1e-4], [100.0]))
    yp = phi_add(x_c, *po)
    R2 = _r2(y_c, yp); chi = _chi2r(y_c, yp, ye_c, 1)
    print(f'  C={po[0]:.4f}')
    print(f'  V_eff(M) ~ |M|^{DELTA+1:.3f} / {DELTA+1:.3f}  [0+1d power potential]')
    print(f'  R2={R2:.4f}  chi2/dof={chi:.2f}')
    results['F2_additive'] = {'C': float(po[0]), 'delta_used': DELTA,
                               'R2': R2, 'chi2dof': chi, 'y_pred': yp.tolist(),
                               'npar': 1}
except Exception as ex:
    print(f'  FAILED: {ex}'); results['F2_additive'] = None

print('\n--- F3: algebraic crossover Phi=C*(1+x/x_c)^(1/delta) ---')
try:
    x0g = float(np.median(x_c))
    po, _ = curve_fit(phi_alg, x_c, y_c, sigma=ye_c, absolute_sigma=True,
                      p0=[float(y_c[0]), x0g], bounds=([1e-4, 1e-6], [100, 1e6]))
    yp = phi_alg(x_c, *po)
    R2 = _r2(y_c, yp); chi = _chi2r(y_c, yp, ye_c, 2)
    print(f'  C={po[0]:.4f}  x_c={po[1]:.4e}')
    print(f'  Crossover at h* = x_c * P^{BD:.3f}')
    print(f'  R2={R2:.4f}  chi2/dof={chi:.2f}')
    results['F3_algebraic'] = {'C': float(po[0]), 'x_c': float(po[1]),
                                'delta_used': DELTA, 'R2': R2, 'chi2dof': chi,
                                'y_pred': yp.tolist(), 'npar': 2}
except Exception as ex:
    print(f'  FAILED: {ex}'); results['F3_algebraic'] = None

print('\n--- F4: power crossover Phi=Phi0*(1+(x/x0)^alpha)^(1/(alpha*delta)) ---')
try:
    x0g = float(np.median(x_c))
    po, _ = curve_fit(phi_pc, x_c, y_c, sigma=ye_c, absolute_sigma=True,
                      p0=[float(y_c[0]), x0g, 1.0],
                      bounds=([1e-4, 1e-6, 0.05], [100, 1e6, 10]), maxfev=5000)
    yp = phi_pc(x_c, *po)
    R2 = _r2(y_c, yp); chi = _chi2r(y_c, yp, ye_c, 3)
    print(f'  Phi0={po[0]:.4f}  x0={po[1]:.4e}  alpha={po[2]:.3f}')
    if abs(po[2]-1.0) < 0.15:
        print(f'  alpha~1: reduces to algebraic F3')
    else:
        print(f'  alpha={po[2]:.3f} != 1: non-trivial crossover exponent')
    print(f'  R2={R2:.4f}  chi2/dof={chi:.2f}')
    results['F4_power'] = {'Phi0': float(po[0]), 'x0': float(po[1]),
                            'alpha': float(po[2]), 'delta_used': DELTA,
                            'R2': R2, 'chi2dof': chi, 'y_pred': yp.tolist(),
                            'npar': 3}
except Exception as ex:
    print(f'  FAILED: {ex}'); results['F4_power'] = None

# ── Ranking ────────────────────────────────────────────────────────────
print('\n--- Phase D ranking ---')
valid = {k: v for k, v in results.items() if v is not None}
ranked = sorted(valid.items(), key=lambda kv: kv[1]['R2'], reverse=True)

for rank, (name, res) in enumerate(ranked, 1):
    print(f'  #{rank}  {name:<20s}  R2={res["R2"]:.4f}  '
          f'chi2/dof={res["chi2dof"]:.2f}  npar={res["npar"]}')

best_name = ranked[0][0] if ranked else 'none'
best_res  = ranked[0][1] if ranked else {}
print(f'\n  Best form: {best_name}  R2={best_res.get("R2",0):.4f}')

if 'F1' in best_name:
    print('  => MEAN-FIELD wins: delta_eff=3. Contradicts direct measurement delta=1.623.')
    print('     Check data quality -- something is wrong.')
elif 'F2' in best_name:
    C = best_res['C']
    print(f'  => ADDITIVE crossover: single-scale, power potential.')
    print(f'     V_eff(M) = lambda/{DELTA+1:.3f} * |M|^{DELTA+1:.3f}')
    print(f'     Phi(0) = C = {C:.4f}  (zero-field scaled amplitude)')
    print(f'     Consistent with 0+1d Langevin at d_eff=0.')
elif 'F3' in best_name:
    C = best_res['C']; xc = best_res['x_c']
    print(f'  => ALGEBRAIC crossover: two-scale structure.')
    print(f'     C={C:.4f}, x_c={xc:.4e}')
    print(f'     Two relevant couplings in effective action.')
elif 'F4' in best_name:
    a = best_res['alpha']
    print(f'  => POWER crossover: alpha={a:.3f}')
    print(f'     Non-trivial crossover structure in effective action.')

# AIC comparison (penalizes extra params)
print('\n--- AIC comparison (lower=better) ---')
n = len(x_c)
for name, res in ranked:
    if res is None: continue
    k = res['npar']
    ss = np.sum((np.array(y_c) - np.array(res['y_pred']))**2)
    aic = 2*k + n * np.log(ss/n) if ss > 0 else -np.inf
    print(f'  {name:<20s}  AIC={aic:.2f}  npar={k}')

# ── Save ───────────────────────────────────────────────────────────────
with open(OUT / 'phaseD.json', 'w') as f:
    json.dump({
        'description': 'Phi(x) scaling function fits, clean filter h>=3e-4',
        'x_data': x_c.tolist(), 'y_data': y_c.tolist(), 'ye_data': ye_c.tolist(),
        'beta': BETA, 'delta': DELTA, 'BD': BD,
        'h_filter': '>=3e-4',
        'fits': results,
        'best_form': best_name,
        'best_R2': float(best_res.get('R2', 0)),
    }, f, indent=2)

print(f'\nSaved: {OUT}/phaseD.json')
