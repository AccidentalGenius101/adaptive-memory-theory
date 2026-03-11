"""
Paper 76 - Susceptibility Exponent gamma — Ordered-Phase Protocol
GPU-accelerated via vcml_gpu_v3.py (manual CUDAGraph, 229k steps/min at L=80)

Paper 75 LESSON: chi = var(M) was at the noise floor because the system was
in the DISORDERED phase (M≈0). The fix: measure chi_int = L^2 * var(M) in
the ORDERED phase (U4 > 0), fixing L and scanning P from ordered (large P)
toward critical (small P near P_c=0+).

Phase A: L=80,  P in {0.003,0.005,0.010,0.020,0.030,0.050}, 8 seeds, 300k steps
         chi_int = L^2 * var(M).  Fit chi_int ~ P^{-gamma}.
         The system is ORDERED at L=80 for all these P values (U4>0 from Papers 63,64).

Phase B: FSS at P=0.010, L in {40,60,80,100,120,160}, 8 seeds, 200k steps
         chi_int(L) ~ L^{gamma/nu}  (independent gamma/nu estimate)

Phase C: L=120, P in {0.002,0.003,0.005,0.010,0.020,0.030}, 8 seeds, 300k steps
         Extend P range toward smaller P at larger L (P_cross(L=120) < P_cross(L=80))

Analysis:
  gamma   from Phase A: fit log(chi_int) = -gamma*log(P) + const
  gamma/nu from Phase B: fit log(chi_int) = (gamma/nu)*log(L) + const
  gamma_B = (gamma/nu) * nu   [consistency check]
  Scaling relations: Rushbrooke (alpha+2beta+gamma=2), Fisher (gamma=nu*(2-eta)),
                     Widom (gamma=beta*(delta-1))
"""

import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import os
os.environ.setdefault('TRITON_CACHE_DIR',        'C:/tc')
os.environ.setdefault('TORCHINDUCTOR_CACHE_DIR', 'C:/ind')

import numpy as np
import json, math
import torch
from pathlib import Path

# Import v3 GPU core
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from vcml_gpu_v3 import run_batch_gpu_v3, fit_power_law

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {DEVICE}')
if DEVICE.type == 'cuda':
    print(f'  GPU: {torch.cuda.get_device_name(0)}')
    print(f'  VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB')

# Known exponents from prior papers
NU   = 0.98
BETA = 0.628

# ── Phases ─────────────────────────────────────────────────────────────────────
PC_A   = [0.003, 0.005, 0.010, 0.020, 0.030, 0.050]
L_A    = 80;   SEEDS_A = list(range(8)); NSTEPS_A = 300_000

PC_B   = 0.010
LC_B   = [40, 60, 80, 100, 120, 160]; SEEDS_B = list(range(8)); NSTEPS_B = 200_000

PC_C   = [0.002, 0.003, 0.005, 0.010, 0.020, 0.030]
L_C    = 120;  SEEDS_C = list(range(8)); NSTEPS_C = 300_000

RESULTS_FILE  = Path(__file__).parent / 'results' / 'paper76_results.json'
ANALYSIS_FILE = Path(__file__).parent / 'results' / 'paper76_analysis.json'


# ── Helpers ────────────────────────────────────────────────────────────────────

def run_phase(label, L, P_list, seeds, nsteps):
    """Run one (L, P) combination, return aggregated stats over seeds."""
    results_raw = run_batch_gpu_v3(L, [P_list] if isinstance(P_list, float) else P_list,
                                   seeds, nsteps, verbose=True)
    # Aggregate over seeds
    absM = float(np.mean([r['absM'] for r in results_raw]))
    M1   = float(np.mean([r['M1']   for r in results_raw]))
    M2   = float(np.mean([r['M2']   for r in results_raw]))
    M4   = float(np.mean([r['M4']   for r in results_raw]))
    chi  = float(M2 - M1**2)    # var(M) = <M^2> - <M>^2
    chi_int = L**2 * chi        # intensive susceptibility
    U4   = float(1. - M4/(3.*M2**2)) if M2 > 1e-12 else float('nan')
    return dict(L=L, P=P_list if not isinstance(P_list, list) else None,
                absM=absM, M1=M1, M2=M2, M4=M4, chi=chi, chi_int=chi_int, U4=U4)


# ── Phase A: chi_int vs P at fixed L=80 ───────────────────────────────────────
print('\n' + '='*60)
print('PHASE A: chi_int vs P, L=80 (ordered-phase gamma fit)')
print('='*60)

results_A = []
for P in PC_A:
    print(f'\nPhase A: P={P:.3f}, L={L_A}, {len(SEEDS_A)} seeds, {NSTEPS_A} steps ...')
    raw = run_batch_gpu_v3(L_A, [P]*len(SEEDS_A), SEEDS_A, NSTEPS_A, verbose=True)
    M1s = [r['M1'] for r in raw]; M2s = [r['M2'] for r in raw]
    M4s = [r['M4'] for r in raw]; absMs = [r['absM'] for r in raw]
    M1   = float(np.mean(M1s))
    M2   = float(np.mean(M2s))
    M4   = float(np.mean(M4s))
    chi  = float(M2 - M1**2)
    chi_int = L_A**2 * chi
    U4   = float(1. - M4/(3.*M2**2)) if M2 > 1e-12 else float('nan')
    absM = float(np.mean(absMs))
    results_A.append(dict(P=P, L=L_A, absM=absM, M1=M1, M2=M2, M4=M4,
                          chi=chi, chi_int=chi_int, U4=U4))
    print(f'  P={P:.3f}  U4={U4:.3f}  |M|={absM:.5f}  chi={chi:.3e}  chi_int={chi_int:.3e}')
    print(f'  Phase A P={P:.3f} done.')

# ── Phase B: chi_int vs L at fixed P=0.010 ────────────────────────────────────
print('\n' + '='*60)
print(f'PHASE B: chi_int vs L, P={PC_B} (FSS gamma/nu)')
print('='*60)

results_B = []
for L in LC_B:
    print(f'\nPhase B: L={L}, P={PC_B}, {len(SEEDS_B)} seeds, {NSTEPS_B} steps ...')
    raw = run_batch_gpu_v3(L, [PC_B]*len(SEEDS_B), SEEDS_B, NSTEPS_B, verbose=True)
    M1s = [r['M1'] for r in raw]; M2s = [r['M2'] for r in raw]
    M4s = [r['M4'] for r in raw]; absMs = [r['absM'] for r in raw]
    M1   = float(np.mean(M1s))
    M2   = float(np.mean(M2s))
    M4   = float(np.mean(M4s))
    chi  = float(M2 - M1**2)
    chi_int = L**2 * chi
    U4   = float(1. - M4/(3.*M2**2)) if M2 > 1e-12 else float('nan')
    absM = float(np.mean(absMs))
    results_B.append(dict(P=PC_B, L=L, absM=absM, M1=M1, M2=M2, M4=M4,
                          chi=chi, chi_int=chi_int, U4=U4))
    print(f'  L={L}  U4={U4:.3f}  |M|={absM:.5f}  chi={chi:.3e}  chi_int={chi_int:.3e}')
    print(f'  Phase B L={L} done.')

# ── Phase C: chi_int vs P at L=120 ────────────────────────────────────────────
print('\n' + '='*60)
print(f'PHASE C: chi_int vs P, L={L_C} (extend range)')
print('='*60)

results_C = []
for P in PC_C:
    print(f'\nPhase C: P={P:.3f}, L={L_C}, {len(SEEDS_C)} seeds, {NSTEPS_C} steps ...')
    raw = run_batch_gpu_v3(L_C, [P]*len(SEEDS_C), SEEDS_C, NSTEPS_C, verbose=True)
    M1s = [r['M1'] for r in raw]; M2s = [r['M2'] for r in raw]
    M4s = [r['M4'] for r in raw]; absMs = [r['absM'] for r in raw]
    M1   = float(np.mean(M1s))
    M2   = float(np.mean(M2s))
    M4   = float(np.mean(M4s))
    chi  = float(M2 - M1**2)
    chi_int = L_C**2 * chi
    U4   = float(1. - M4/(3.*M2**2)) if M2 > 1e-12 else float('nan')
    absM = float(np.mean(absMs))
    results_C.append(dict(P=P, L=L_C, absM=absM, M1=M1, M2=M2, M4=M4,
                          chi=chi, chi_int=chi_int, U4=U4))
    print(f'  P={P:.3f}  U4={U4:.3f}  |M|={absM:.5f}  chi={chi:.3e}  chi_int={chi_int:.3e}')
    print(f'  Phase C P={P:.3f} done.')

# ── Save raw results ────────────────────────────────────────────────────────────
RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
with open(RESULTS_FILE, 'w') as f:
    json.dump(dict(phase_A=results_A, phase_B=results_B, phase_C=results_C), f, indent=2)
print(f'\nRaw results -> {RESULTS_FILE}')

# ── Analysis ───────────────────────────────────────────────────────────────────
print('\n' + '='*60)
print('ANALYSIS')
print('='*60)

# Phase A: gamma from chi_int ~ P^{-gamma}
PA_P  = [r['P'] for r in results_A]
PA_ci = [r['chi_int'] for r in results_A]
gamma_A, r2_gA = fit_power_law(PA_P, PA_ci)
gamma_A = -gamma_A   # chi ~ P^{-gamma} => slope = -gamma
print(f'\n=== PHASE A (L={L_A}, chi_int vs P) ===')
print(f'  {"P":>8}  {"U4":>6}  {"absM":>8}  {"chi":>10}  {"chi_int":>12}')
for r in results_A:
    print(f'  {r["P"]:8.3f}  {r["U4"]:6.3f}  {r["absM"]:8.5f}  {r["chi"]:10.3e}  {r["chi_int"]:12.3e}')
print(f'  gamma_A = {gamma_A:.3f}  (R2={r2_gA:.3f})')

# Phase B: gamma/nu from chi_int ~ L^{gamma/nu}
PB_L  = [r['L']       for r in results_B]
PB_ci = [r['chi_int'] for r in results_B]
gonu_B, r2_gonuB = fit_power_law(PB_L, PB_ci)
gamma_B_indirect = gonu_B * NU
print(f'\n=== PHASE B (P={PC_B}, chi_int vs L) ===')
print(f'  {"L":>5}  {"U4":>6}  {"absM":>8}  {"chi":>10}  {"chi_int":>12}')
for r in results_B:
    print(f'  {r["L"]:5d}  {r["U4"]:6.3f}  {r["absM"]:8.5f}  {r["chi"]:10.3e}  {r["chi_int"]:12.3e}')
print(f'  gamma/nu (FSS) = {gonu_B:.3f}  (R2={r2_gonuB:.3f})')
print(f'  gamma_B (=gonu*nu) = {gamma_B_indirect:.3f}')

# Phase C: gamma from chi_int at L=120
PC_P  = [r['P'] for r in results_C]
PC_ci = [r['chi_int'] for r in results_C]
gamma_C, r2_gC = fit_power_law(PC_P, PC_ci)
gamma_C = -gamma_C
print(f'\n=== PHASE C (L={L_C}, chi_int vs P) ===')
print(f'  {"P":>8}  {"U4":>6}  {"absM":>8}  {"chi":>10}  {"chi_int":>12}')
for r in results_C:
    print(f'  {r["P"]:8.3f}  {r["U4"]:6.3f}  {r["absM"]:8.5f}  {r["chi"]:10.3e}  {r["chi_int"]:12.3e}')
print(f'  gamma_C = {gamma_C:.3f}  (R2={r2_gC:.3f})')

# Global gamma (combine A+C at same L values if available; use all chi_int vs P)
all_P  = PA_P + PC_P
all_ci = PA_ci + PC_ci
gamma_all, r2_gall = fit_power_law(all_P, all_ci)
gamma_all = -gamma_all
print(f'\n=== GLOBAL gamma (Phases A+C combined) ===')
print(f'  gamma_all = {gamma_all:.3f}  (R2={r2_gall:.3f})')

# Scaling relations
# Fisher:    gamma = nu*(2-eta)  =>  eta_eff = 2 - gamma/nu
# Widom:     gamma = beta*(delta-1)  =>  delta = 1 + gamma/beta
# Rushbrooke: alpha = 2 - 2*beta - gamma
# Josephson: nu*d = 2 - alpha  (d=0 => alpha=2, hyperscaling breaks)
g_use = gamma_A if not math.isnan(gamma_A) else gamma_all
eta_eff  = 2. - g_use / NU
delta    = 1. + g_use / BETA
alpha_RB = 2. - 2.*BETA - g_use
print(f'\n=== SCALING RELATIONS (gamma={g_use:.3f}, beta={BETA}, nu={NU}) ===')
print(f'  Fisher:     eta_eff  = 2 - gamma/nu        = {eta_eff:.3f}')
print(f'  Widom:      delta    = 1 + gamma/beta       = {delta:.3f}')
print(f'  Rushbrooke: alpha    = 2 - 2*beta - gamma   = {alpha_RB:.3f}')
print(f'  Josephson d=0: alpha_hyp = 2 (hyperscaling BREAKS for d_eff=0)')
print(f'  Paper71 eta (indirect 2*beta/nu): {2*BETA/NU:.3f}')
print(f'  Paper75 eta_eff (cross-phase, unreliable): 1.61')

print('\nDone.')

# ── Save analysis ──────────────────────────────────────────────────────────────
analysis = dict(
    # Phase A
    chi_int_A={str(r['P']): r['chi_int'] for r in results_A},
    U4_A={str(r['P']): r['U4'] for r in results_A},
    absM_A={str(r['P']): r['absM'] for r in results_A},
    gamma_A=gamma_A, r2_gA=r2_gA,
    # Phase B
    chi_int_B={str(r['L']): r['chi_int'] for r in results_B},
    U4_B={str(r['L']): r['U4'] for r in results_B},
    absM_B={str(r['L']): r['absM'] for r in results_B},
    gonu_B=gonu_B, r2_gonuB=r2_gonuB,
    gamma_B_indirect=gamma_B_indirect,
    # Phase C
    chi_int_C={str(r['P']): r['chi_int'] for r in results_C},
    U4_C={str(r['P']): r['U4'] for r in results_C},
    absM_C={str(r['P']): r['absM'] for r in results_C},
    gamma_C=gamma_C, r2_gC=r2_gC,
    # Global
    gamma_all=gamma_all, r2_gall=r2_gall,
    # Scaling
    eta_eff=eta_eff, delta=delta, alpha_RB=alpha_RB,
    # Params
    nu=NU, beta=BETA,
    L_A=L_A, L_C=L_C, PC_B=PC_B,
)
with open(ANALYSIS_FILE, 'w') as f:
    json.dump(analysis, f, indent=2)
print(f'Analysis -> {ANALYSIS_FILE}')
