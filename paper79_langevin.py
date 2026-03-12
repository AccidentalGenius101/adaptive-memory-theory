"""
paper79_langevin.py  --  Direct Langevin comparison to VCML fixed point

Tests whether the effective 0+1d stochastic equation
    dM/dt = -lambda*|M|^delta*sign(M) + r*M + h + sqrt(2D)*eta(t)
reproduces VCML exponents (beta=0.628, delta=1.623, z=0.48).

Strategy
--------
The equation has three sectors:
  - Power-law restoring force: -lambda*|M|^delta*sign(M)
  - Linear ordering drive:     +r*M   (r plays role of P in VCML)
  - External field:            +h
  - Noise:                     sqrt(2D)*eta(t),  D=fixed

Measure:
  beta_r:  <|M|> ~ r^beta_r  at h=0, vary r
  delta_m: <|M|> ~ h^(1/delta_m) at fixed r, vary h
  z_L:     tau ~ r^(-z*nu_L) from autocorrelation

Compare to VCML: beta=0.628, delta=1.623, z=0.48.

Key prediction (analytic saddle point):
  At h=0: V'(M*)=0 -> lambda*|M*|^delta = r*M* -> M* ~ (r/lambda)^{1/(delta-1)}
  So beta_saddle = 1/(delta-1) = 1/0.623 = 1.605  (NOT 0.628)
  If simulation gives beta_r ~ 0.628: fluctuations correct the saddle point.
  If simulation gives beta_r ~ 1.605: simple Langevin != VCML -> genuinely novel.

GPU: pure PyTorch, no custom CUDA. Batch over trajectories.
"""

import os, sys, json, time
import numpy as np
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
os.chdir(Path(__file__).parent)

OUT = Path('results') / 'paper79'
OUT.mkdir(parents=True, exist_ok=True)

# ── Config ─────────────────────────────────────────────────────────────
DELTA  = 1.623        # from VCML Paper 77
BETA_V = 0.628        # VCML measured
NU_V   = 0.98         # VCML measured
Z_V    = 0.48         # VCML measured
LAMBDA = 1.0          # amplitude (non-universal)
D      = 0.01         # noise amplitude (fixed)
DT     = 5e-4         # Euler-Maruyama step
N_TRAJ = 8192         # parallel trajectories (GPU batch)
T_BURN = 200_000      # equilibration steps
T_MEAS = 100_000      # measurement steps

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}')
if device.type == 'cuda':
    print(f'  {torch.cuda.get_device_name(0)}')

sqrt2D_dt = float(np.sqrt(2 * D * DT))

def langevin_batch(r_val, h_val, n_traj, t_burn, t_meas):
    """
    Run N_TRAJ independent Langevin trajectories.
    Returns: absM_mean, absM_std, tau_corr
    """
    # Initialize near steady-state guess
    if r_val > 0:
        M0 = float((r_val / LAMBDA) ** (1.0 / (DELTA - 1)))
    else:
        M0 = 0.1
    M = torch.full((n_traj,), M0, device=device, dtype=torch.float32)
    M *= (2 * torch.randint(0, 2, (n_traj,), device=device).float() - 1)

    # Euler-Maruyama integration
    def step(M):
        drift  = -LAMBDA * M.abs()**DELTA * M.sign() + r_val * M + h_val
        noise  = sqrt2D_dt * torch.randn_like(M)
        return M + drift * DT + noise

    # Burn-in
    for _ in range(t_burn):
        M = step(M)

    # Measurement
    M_hist = torch.zeros(t_meas, n_traj, device=device)
    for t in range(t_meas):
        M = step(M)
        M_hist[t] = M

    absM = M_hist.abs()
    mean_absM = float(absM.mean())
    std_absM  = float(absM.std(dim=0).mean())  # across trajectories

    # Autocorrelation time from M (not |M|) -- use first traj
    m_series = M_hist[:, 0].cpu().float()
    m_series = m_series - m_series.mean()
    var = float(m_series.var())
    if var > 1e-12:
        lags = min(5000, t_meas // 4)
        acf = torch.zeros(lags)
        for lag in range(lags):
            if lag < t_meas - 1:
                acf[lag] = float((m_series[:t_meas-lag] * m_series[lag:]).mean()) / var
        # Integrate until first zero crossing
        tau = 0.5
        for lag in range(1, lags):
            if acf[lag] <= 0:
                break
            tau += float(acf[lag])
    else:
        tau = float('nan')

    return mean_absM, std_absM, tau

# ── Phase A: beta_r  <|M|> ~ r^beta_r at h=0 ─────────────────────────
print('=' * 60)
print('Phase A: beta_r  <|M|> ~ r^beta_r  at h=0')
print(f'  Saddle-point prediction: beta_saddle = 1/(delta-1) = {1/(DELTA-1):.3f}')
print(f'  VCML measured:           beta_V      = {BETA_V}')
print('=' * 60)

R_VALS = [0.001, 0.003, 0.010, 0.030, 0.100]
PA = {}
for r in R_VALS:
    t0 = time.time()
    m, s, tau = langevin_batch(r, 0.0, N_TRAJ, T_BURN, T_MEAS)
    dt = time.time() - t0
    PA[r] = {'absM': m, 'std': s, 'tau': tau}
    print(f'  r={r:.3f}  <|M|>={m:.5f}±{s:.5f}  tau={tau:.1f}  ({dt:.0f}s)')

# Fit beta_r
r_arr = np.array(R_VALS)
m_arr = np.array([PA[r]['absM'] for r in R_VALS])
mask  = m_arr > 1e-4
if mask.sum() >= 3:
    coeffs = np.polyfit(np.log(r_arr[mask]), np.log(m_arr[mask]), 1)
    beta_r = float(coeffs[0])
    print(f'\n  Fit: <|M|> ~ r^{beta_r:.3f}')
    print(f'  Saddle prediction: {1/(DELTA-1):.3f}')
    print(f'  VCML value:        {BETA_V}')
    if abs(beta_r - BETA_V) < 0.05:
        print('  => MATCHES VCML: Langevin in same class')
    elif abs(beta_r - 1/(DELTA-1)) < 0.05:
        print('  => MATCHES SADDLE: simple Langevin, VCML is richer')
    else:
        print(f'  => NOVEL: beta_r={beta_r:.3f} matches neither prediction')
else:
    beta_r = float('nan')
    print('  Not enough ordered points to fit beta_r')

# ── Phase B: delta_m  <|M|> ~ h^(1/delta_m) at fixed r ───────────────
print('\n' + '=' * 60)
print('Phase B: delta_m  <|M|> ~ h^(1/delta_m)  at fixed r=0.010')
print(f'  VCML measured: delta_V = {DELTA}')
print('=' * 60)

R_FIXED = 0.010
H_VALS  = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2]
PB = {}
for h in H_VALS:
    t0 = time.time()
    m, s, _ = langevin_batch(R_FIXED, h, N_TRAJ, T_BURN, T_MEAS)
    dt = time.time() - t0
    PB[h] = {'absM': m, 'std': s}
    print(f'  h={h:.0e}  <|M|>={m:.5f}±{s:.5f}  ({dt:.0f}s)')

h_arr = np.array(H_VALS)
mh_arr = np.array([PB[h]['absM'] for h in H_VALS])
mask  = mh_arr > 1e-4
if mask.sum() >= 3:
    coeffs = np.polyfit(np.log(h_arr[mask]), np.log(mh_arr[mask]), 1)
    inv_delta_m = float(coeffs[0])
    delta_m = 1.0 / inv_delta_m if abs(inv_delta_m) > 1e-6 else float('nan')
    print(f'\n  Fit: <|M|> ~ h^{inv_delta_m:.3f}  =>  delta_m = {delta_m:.3f}')
    print(f'  VCML value: delta_V = {DELTA}')
    if abs(delta_m - DELTA) < 0.1:
        print('  => delta MATCHES VCML')
    else:
        print(f'  => delta MISMATCH: Langevin gives {delta_m:.3f}, VCML gives {DELTA}')
else:
    delta_m = float('nan')
    print('  Not enough data to fit delta_m')

# ── Phase C: z from tau ~ r^(-z*nu) ───────────────────────────────────
print('\n' + '=' * 60)
print('Phase C: z from tau ~ r^(-z*nu)')
print(f'  VCML: z={Z_V}, nu={NU_V}, z*nu={Z_V*NU_V:.3f}')
print('=' * 60)

tau_arr = np.array([PA[r]['tau'] for r in R_VALS])
mask_tau = np.isfinite(tau_arr) & (tau_arr > 0) & (r_arr > 0)
if mask_tau.sum() >= 3:
    coeffs = np.polyfit(np.log(r_arr[mask_tau]), np.log(tau_arr[mask_tau]), 1)
    znu_L = float(-coeffs[0])
    print(f'  tau ~ r^{coeffs[0]:.3f}  =>  z*nu = {znu_L:.3f}')
    print(f'  VCML: z*nu = {Z_V*NU_V:.3f}')
    if abs(znu_L - Z_V*NU_V) < 0.05:
        print('  => z*nu MATCHES VCML')
    else:
        print(f'  => z*nu MISMATCH: Langevin {znu_L:.3f} vs VCML {Z_V*NU_V:.3f}')
    for r, tau in zip(r_arr[mask_tau], tau_arr[mask_tau]):
        print(f'    r={r:.3f}  tau={tau:.1f}')
else:
    znu_L = float('nan')
    print('  Insufficient finite tau values')

# ── Summary ────────────────────────────────────────────────────────────
print('\n' + '=' * 60)
print('PAPER 79 LANGEVIN SUMMARY')
print('=' * 60)
print(f'  Equation: dM/dt = -|M|^{DELTA}*sign(M) + r*M + h + sqrt(2D)*eta')
print(f'  D={D},  dt={DT},  N_traj={N_TRAJ}')
print()
print(f'  Exponent     Langevin      VCML      Saddle-point')
print(f'  beta_r       {beta_r:<10.3f}    {BETA_V:<8}  {1/(DELTA-1):.3f}')
print(f'  delta        {delta_m:<10.3f}    {DELTA:<8}  {DELTA} (imposed)')
print(f'  z*nu         {znu_L:<10.3f}    {Z_V*NU_V:<8.3f}  --')

verdict = []
if abs(beta_r - BETA_V) < 0.05:
    verdict.append('beta: MATCH')
elif abs(beta_r - 1/(DELTA-1)) < 0.1:
    verdict.append('beta: SADDLE (Langevin != VCML)')
else:
    verdict.append(f'beta: NOVEL ({beta_r:.3f})')
if abs(delta_m - DELTA) < 0.15:
    verdict.append('delta: MATCH')
else:
    verdict.append(f'delta: MISMATCH ({delta_m:.3f})')
if np.isfinite(znu_L):
    if abs(znu_L - Z_V*NU_V) < 0.05:
        verdict.append('z: MATCH')
    else:
        verdict.append(f'z: MISMATCH ({znu_L:.3f})')

print('\n  VERDICT:')
for v in verdict:
    print(f'    {v}')

if all('MATCH' in v for v in verdict):
    print('\n  => VCML IS the 0+1d Langevin class. Architecture sufficient.')
elif 'SADDLE' in ' '.join(verdict):
    print('\n  => Simple Langevin gives saddle-point exponents.')
    print('     VCML beta != beta_saddle: causal mechanism generates')
    print('     non-trivial fluctuation corrections. Genuinely novel.')
else:
    print('\n  => Mixed: some exponents match, some do not. Investigate.')

with open(OUT / 'langevin.json', 'w') as f:
    json.dump({
        'delta': DELTA, 'D': D, 'dt': DT, 'n_traj': N_TRAJ,
        'phaseA': {str(r): PA[r] for r in R_VALS},
        'phaseB': {str(h): PB[h] for h in H_VALS},
        'beta_r': beta_r, 'beta_saddle': 1/(DELTA-1), 'beta_vcml': BETA_V,
        'delta_m': delta_m, 'delta_vcml': DELTA,
        'znu_langevin': znu_L, 'znu_vcml': Z_V * NU_V,
        'verdict': verdict,
    }, f, indent=2)

print(f'\nResults saved: {OUT}/langevin.json')
