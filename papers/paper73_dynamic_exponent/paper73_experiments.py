"""
Paper 73 - Dynamic Exponent z from Temporal Autocorrelator C(t) = <|M(tau)||M(tau+t)|>

Background:
  Paper 72 established that VCML ordering is a ZONE-MEAN transition: M(t) is a
  0-dimensional stochastic variable with no spatial correlations. The exponents
  beta~0.63 and nu~0.98 describe its TEMPORAL critical behaviour. The anomalous
  dimension eta is undefined.

  The remaining open question: what is the dynamic exponent z?
  At criticality (P_c=0+, approached from the ordered side), the temporal
  correlation function should obey:
    C(t) = <|M|(0)|M|(t)> - <|M|>^2 / Var(|M|)
  with:
    - Power-law decay C(t) ~ t^{-lambda}  at criticality (t << tau_L)
    - Exponential decay C(t) ~ exp(-t/tau) in ordered phase (t >> correlation time)
    - tau_L ~ L^z  (finite-size scaling at criticality)
    - tau_corr ~ P^{-z*nu} (divergence approaching P_c=0+)
    - lambda = 2*beta / (z*nu)  (scaling relation)

  Three phases:
    A: tau_L(L) at P=0.010 -- finite-size scaling gives z directly
    B: tau_corr(P) at L=80 -- divergence gives z*nu (cross-check)
    C: C(t) power-law test at large L -- extract lambda, cross-check z = 2*beta/(lambda*nu)

  Model A (mean-field dynamics) predicts z=2, lambda=2*beta/(z*nu)=0.64.
  A non-trivial z would be a new dynamical universality class.
"""

import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import numpy as np
import json
import math
import multiprocessing as mp
from pathlib import Path

# -- Physical parameters (identical to Papers 68-72) ---------------------------
J           = 1.0
MID_DECAY   = 0.97
BETA_BASE   = 0.005
SS          = 8
WAVE_DUR    = 5
EXT_FIELD   = 1.5
T_FIXED     = 3.0
FIELD_DECAY = 0.999
SS_FRAC     = 0.40
FA          = 0.30
WE_BASE     = 25
L_BASE      = 40

def wave_area(r): return 2 * r * (r + 1) + 1

def _wef(L): return WE_BASE * (L_BASE / L) ** 2
def _prob_A(L): return min(1.0, 1.0 / _wef(L))

# -- Phase A: tau_L(L) -- FSS of relaxation time at P=0.010 --------------------
RW_A     = 5
PC_A     = 0.010
L_A      = [40, 60, 80, 100, 120]
SEEDS_A  = list(range(8))
NSTEPS_A = 30_000   # 18k data points per seed

# -- Phase B: tau_corr(P) at fixed L -- critical divergence --------------------
RW_B     = 5
PC_B     = [0.007, 0.010, 0.015, 0.020, 0.030, 0.050]
L_B      = 80
SEEDS_B  = list(range(8))
NSTEPS_B = 40_000   # 24k data points per seed

# -- Phase C: C(t) power law test at large L, long run -------------------------
RW_C     = 5
PC_C     = 0.010
L_C      = 120
SEEDS_C  = list(range(8))
NSTEPS_C = 60_000   # 36k data points per seed

RESULTS_FILE  = Path(__file__).parent / 'results' / 'paper73_results.json'
ANALYSIS_FILE = Path(__file__).parent / 'results' / 'paper73_analysis.json'


# -- Lattice helpers (verbatim from Papers 68-72) ------------------------------
def build_nb(L):
    N = L * L
    row = np.arange(N) // L; col = np.arange(N) % L
    return np.stack([((row-1) % L)*L + col, ((row+1) % L)*L + col,
                     row*L + (col-1) % L,   row*L + (col+1) % L], axis=1)

def metro(s, nb, T, rng, h_ext=None, hit=None):
    N = len(s)
    row = np.arange(N) // int(round(math.sqrt(N)))
    col = np.arange(N) %  int(round(math.sqrt(N)))
    h = np.zeros(N)
    if h_ext is not None and hit is not None:
        h[hit] = h_ext
    for sub in [0, 1]:
        idx = np.where((row + col) % 2 == sub)[0]
        ns  = s[nb[idx]].sum(1)
        dE  = 2.*J*s[idx]*ns - 2.*h[idx]*s[idx]
        acc = (dE <= 0) | (rng.random(len(idx)) < np.exp(-np.clip(dE/T, -20, 20)))
        s[idx[acc]] *= -1
    return s

def wsites(cx, cy, r, L):
    N = L * L
    row = np.arange(N) // L; col = np.arange(N) % L
    dx = np.minimum(np.abs(col - cx), L - np.abs(col - cx))
    dy = np.minimum(np.abs(row - cy), L - np.abs(row - cy))
    return np.where(dx + dy <= r)[0]

def ms(vals):
    v = [x for x in vals if x is not None and not math.isnan(x)]
    return float(np.mean(v)) if v else float('nan')

def ses(vals):
    v = [x for x in vals if x is not None and not math.isnan(x)]
    return float(np.std(v) / math.sqrt(len(v))) if len(v) > 1 else 0.


# -- Temporal autocorrelation --------------------------------------------------
def compute_autocorr_absM(Mseries, max_lag=None):
    """Compute normalized autocorrelation of |M(t)|.
    C(t) = [<|M|(s)*|M|(s+t)>  - <|M|>^2] / Var(|M|)
    Returns (lag_list, acf_list) with C(0)=1.
    Uses FFT for efficiency.
    """
    M    = np.array(Mseries, dtype=float)
    absM = np.abs(M)
    n    = len(absM)
    if n < 20:
        return [], []
    mean_a = float(np.mean(absM))
    a_c    = absM - mean_a
    var_a  = float(np.mean(a_c ** 2))
    if var_a < 1e-20:
        return [], []

    if max_lag is None:
        max_lag = min(n // 4, 10000)

    # FFT-based autocorrelation (zero-padded to avoid circular artifacts)
    npad = 1
    while npad < 2 * n:
        npad *= 2
    F   = np.fft.rfft(a_c, n=npad)
    raw = np.real(np.fft.irfft(np.abs(F) ** 2))[:n]
    # Unbiased: raw[t] / (n-t) / var_a
    lags   = np.arange(max_lag + 1)
    counts = n - lags
    acf    = raw[:max_lag + 1] / (counts * var_a)
    return lags.tolist(), acf.tolist()


# -- Fit helpers ---------------------------------------------------------------
def fit_tau_exp(lags, acf, t_min=50, t_max=None):
    """Fit C(t) ~ exp(-t/tau) by linear regression of log(C) vs t.
    Returns (tau, r2).
    """
    if t_max is None:
        t_max = max(lags) // 2 if lags else 100
    pairs = [(t, C) for t, C in zip(lags, acf)
             if t_min <= t <= t_max and C > 0]
    if len(pairs) < 5:
        return float('nan'), float('nan')
    xs = [t for t, C in pairs]
    ys = [math.log(C) for t, C in pairs]
    n  = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    ssxx = sum((x - mx) ** 2 for x in xs)
    ssxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    if ssxx < 1e-10:
        return float('nan'), float('nan')
    slope = ssxy / ssxx
    yhat  = [my + slope * (x - mx) for x in xs]
    sstot = sum((y - my) ** 2 for y in ys)
    ssres = sum((y - yh) ** 2 for y, yh in zip(ys, yhat))
    r2    = 1. - ssres / sstot if sstot > 1e-10 else float('nan')
    if slope >= 0:
        return float('nan'), float('nan')
    return -1.0 / slope, r2


def fit_lambda_pow(lags, acf, t_min=100, t_max=None):
    """Fit C(t) ~ t^{-lambda} by log-log linear regression.
    Returns (lambda_val, r2).
    """
    if t_max is None:
        t_max = max(lags) // 2 if lags else 100
    pairs = [(t, C) for t, C in zip(lags, acf)
             if t_min <= t <= t_max and C > 0 and t > 0]
    if len(pairs) < 5:
        return float('nan'), float('nan')
    lx = [math.log(t) for t, C in pairs]
    ly = [math.log(C) for t, C in pairs]
    n  = len(lx)
    mx, my = sum(lx) / n, sum(ly) / n
    ssxx = sum((x - mx) ** 2 for x in lx)
    ssxy = sum((x - mx) * (y - my) for x, y in zip(lx, ly))
    if ssxx < 1e-20:
        return float('nan'), float('nan')
    slope = ssxy / ssxx
    yhat  = [my + slope * (x - mx) for x in lx]
    sstot = sum((y - my) ** 2 for y in ly)
    ssres = sum((y - yh) ** 2 for y, yh in zip(ly, yhat))
    r2    = 1. - ssres / sstot if sstot > 1e-10 else float('nan')
    return -slope, r2   # lambda = -slope (C ~ t^{-lambda} -> slope = -lambda)


def fit_z_from_tau(L_list, tau_list):
    """Fit tau_L ~ L^z by log-log regression. Returns (z, r2)."""
    pairs = [(L, tau) for L, tau in zip(L_list, tau_list)
             if not math.isnan(tau) and tau > 0 and L > 0]
    if len(pairs) < 3:
        return float('nan'), float('nan')
    lx = [math.log(L) for L, tau in pairs]
    ly = [math.log(tau) for L, tau in pairs]
    n  = len(lx)
    mx, my = sum(lx) / n, sum(ly) / n
    ssxx = sum((x - mx) ** 2 for x in lx)
    ssxy = sum((x - mx) * (y - my) for x, y in zip(lx, ly))
    if ssxx < 1e-20:
        return float('nan'), float('nan')
    slope = ssxy / ssxx
    yhat  = [my + slope * (x - mx) for x in lx]
    sstot = sum((y - my) ** 2 for y in ly)
    ssres = sum((y - yh) ** 2 for y, yh in zip(ly, yhat))
    r2    = 1. - ssres / sstot if sstot > 1e-10 else float('nan')
    return slope, r2   # z = slope (tau ~ L^z)


def fit_z_from_P(P_list, tau_list, nu=0.98):
    """Fit tau ~ P^{-z*nu} by log-log regression.
    P_c = 0+, so |P - P_c| = P.
    Returns (z, r2).
    """
    pairs = [(P, tau) for P, tau in zip(P_list, tau_list)
             if not math.isnan(tau) and tau > 0 and P > 0]
    if len(pairs) < 3:
        return float('nan'), float('nan')
    lx = [math.log(P) for P, tau in pairs]
    ly = [math.log(tau) for P, tau in pairs]
    n  = len(lx)
    mx, my = sum(lx) / n, sum(ly) / n
    ssxx = sum((x - mx) ** 2 for x in lx)
    ssxy = sum((x - mx) * (y - my) for x, y in zip(lx, ly))
    if ssxx < 1e-20:
        return float('nan'), float('nan')
    slope = ssxy / ssxx
    yhat  = [my + slope * (x - mx) for x in lx]
    sstot = sum((y - my) ** 2 for y in ly)
    ssres = sum((y - yh) ** 2 for y, yh in zip(ly, yhat))
    r2    = 1. - ssres / sstot if sstot > 1e-10 else float('nan')
    z_nu  = -slope   # tau ~ P^{-z*nu} -> slope = -z*nu
    z     = z_nu / nu
    return z, r2


def avg_acf(results_list):
    """Average autocorrelation functions over seeds."""
    valid = [r for r in results_list
             if r and 'acf' in r and len(r.get('acf', [])) > 0]
    if not valid:
        return [], []
    lags_ref = valid[0]['lags']
    acf_acc  = np.zeros(len(lags_ref))
    cnt      = 0
    for r in valid:
        if r.get('lags') == lags_ref:
            acf_arr = np.array(r['acf'])
            if len(acf_arr) == len(lags_ref):
                acf_acc += acf_arr
                cnt     += 1
    return lags_ref, (acf_acc / cnt).tolist() if cnt > 0 else []


# -- Core simulation with temporal autocorrelator ------------------------------
def run_one_temporal(L, P_causal, seed, nsteps, wave_prob, r_w, max_lag=None):
    """Run VCML and return standard observables + temporal ACF of |M(t)|."""
    rng    = np.random.RandomState(seed)
    nb     = build_nb(L)
    N      = L * L
    col_a  = np.arange(N) % L
    zone   = (col_a >= L // 2).astype(int)
    s      = rng.choice([-1, 1], N).astype(float)
    base   = s * 0.1
    mid    = np.zeros(N)
    phi    = np.zeros(N)
    streak = np.zeros(N, int)
    wave_z = 0
    ss0    = int(nsteps * SS_FRAC)
    Mseries = []

    for t in range(nsteps):
        s = metro(s, nb, T_FIXED, rng)
        base  += BETA_BASE * (s - base)
        same   = (s > 0) == (base > 0)
        streak = np.where(same, streak + 1, 0)
        mid   *= MID_DECAY

        if rng.random() < wave_prob:
            cx = rng.randint(0, L // 2) if wave_z == 0 else rng.randint(L // 2, L)
            cy = rng.randint(L)
            hit = wsites(cx, cy, r_w, L)
            if len(hit) > 0:
                he = EXT_FIELD if wave_z == 0 else -EXT_FIELD
                for _ in range(WAVE_DUR):
                    s = metro(s, nb, T_FIXED, rng, h_ext=he, hit=hit)
                dev = s[hit] - base[hit]
                zm  = (zone[hit] == wave_z)
                sig = np.where(zm, dev, -dev)
                cau = rng.random(len(hit)) < P_causal
                nse = rng.normal(0, float(np.std(dev)) + 0.5, len(hit))
                mid[hit] += FA * np.where(cau, sig, nse)
            wave_z = 1 - wave_z

        gate = streak >= SS
        if gate.any():
            wi = np.where(gate)[0]
            phi[wi] += FA * (mid[wi] - phi[wi])
            streak[wi] = 0
        phi *= FIELD_DECAY

        if t >= ss0:
            M = float(np.mean(phi[zone == 0])) - float(np.mean(phi[zone == 1]))
            Mseries.append(M)

    # Standard observables
    M_arr = np.array(Mseries)
    M2    = float(np.mean(M_arr ** 2))
    M4    = float(np.mean(M_arr ** 4))
    absM  = float(np.mean(np.abs(M_arr)))
    U4    = float(1. - M4 / (3. * M2 ** 2)) if M2 > 1e-12 else float('nan')

    # Temporal autocorrelation of |M(t)|
    if max_lag is None:
        max_lag = min(len(Mseries) // 4, 8000)
    lags, acf = compute_autocorr_absM(Mseries, max_lag=max_lag)

    return dict(absM=absM, M2=M2, U4=U4, lags=lags, acf=acf,
                n_data=len(Mseries))


# -- Workers -------------------------------------------------------------------
def _wA(args):
    L, seed = args
    ml = min(int((NSTEPS_A * (1 - SS_FRAC)) // 4), 5000)
    return ('A', L, seed,
            run_one_temporal(L, PC_A, seed, NSTEPS_A, _prob_A(L), RW_A,
                             max_lag=ml))

def _wB(args):
    pc, seed = args
    ml = min(int((NSTEPS_B * (1 - SS_FRAC)) // 4), 6000)
    return ('B', pc, seed,
            run_one_temporal(L_B, pc, seed, NSTEPS_B, _prob_A(L_B), RW_B,
                             max_lag=ml))

def _wC(args):
    seed = args
    ml = min(int((NSTEPS_C * (1 - SS_FRAC)) // 4), 9000)
    return ('C', seed,
            run_one_temporal(L_C, PC_C, seed, NSTEPS_C, _prob_A(L_C), RW_C,
                             max_lag=ml))


# -- MAIN ----------------------------------------------------------------------
if __name__ == '__main__':
    mp.freeze_support()
    RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)

    raw = {}
    if RESULTS_FILE.exists():
        with open(RESULTS_FILE) as f:
            raw = json.load(f)

    ncpu = mp.cpu_count()

    # -- Phase A: tau_L(L) -------------------------------------------------------
    if 'A' not in raw:
        args_A = [(L, seed) for L in L_A for seed in SEEDS_A]
        print(f'Phase A: {len(args_A)} runs -- tau_L FSS, r_w={RW_A}, '
              f'P={PC_A}, L in {L_A} ...')
        with mp.Pool(processes=min(len(args_A), ncpu)) as pool:
            res = pool.map(_wA, args_A)
        raw['A'] = {}
        for _, L, seed, r in res:
            raw['A'].setdefault(f'L{L}', {})[str(seed)] = r
        with open(RESULTS_FILE, 'w') as f:
            json.dump(raw, f)
        print('Phase A done.')

    # -- Phase B: tau_corr(P) ----------------------------------------------------
    if 'B' not in raw:
        args_B = [(pc, seed) for pc in PC_B for seed in SEEDS_B]
        print(f'\nPhase B: {len(args_B)} runs -- tau_corr(P), r_w={RW_B}, '
              f'L={L_B} ...')
        with mp.Pool(processes=min(len(args_B), ncpu)) as pool:
            res = pool.map(_wB, args_B)
        raw['B'] = {}
        for _, pc, seed, r in res:
            raw['B'].setdefault(f'pc{pc:.4f}', {})[str(seed)] = r
        with open(RESULTS_FILE, 'w') as f:
            json.dump(raw, f)
        print('Phase B done.')

    # -- Phase C: C(t) power law test -------------------------------------------
    if 'C' not in raw:
        args_C = list(SEEDS_C)
        print(f'\nPhase C: {len(args_C)} runs -- C(t) power law, '
              f'r_w={RW_C}, P={PC_C}, L={L_C} ...')
        with mp.Pool(processes=min(len(args_C), ncpu)) as pool:
            res = pool.map(_wC, args_C)
        raw['C'] = {}
        for _, seed, r in res:
            raw['C'][str(seed)] = r
        with open(RESULTS_FILE, 'w') as f:
            json.dump(raw, f)
        print('Phase C done.')

    # -- Analysis ---------------------------------------------------------------

    # Phase A: tau_L at each L
    print(f'\n=== PHASE A: FSS of tau_L (r_w={RW_A}, P={PC_A}) ===')
    print(f'{"L":>5}  {"U4":>7}  {"absM":>9}  {"tau_exp":>10}  '
          f'{"R2_exp":>7}  {"tau_pow":>10}  {"R2_pow":>7}')
    print('-' * 62)

    tau_A = {}
    for L in L_A:
        key   = f'L{L}'
        seeds = raw.get('A', {}).get(key, {})
        u4s   = [v['U4']   for v in seeds.values()
                 if not math.isnan(v.get('U4', float('nan')))]
        Ms    = [v['absM'] for v in seeds.values()
                 if not math.isnan(v.get('absM', float('nan')))]
        U4    = ms(u4s);  absM = ms(Ms)
        lags, acf = avg_acf(list(seeds.values()))
        t_max_fit = max(lags) // 2 if lags else 100
        tau_e, r2_e = fit_tau_exp(lags, acf, t_min=50,  t_max=t_max_fit)
        tau_p, r2_p = fit_lambda_pow(lags, acf, t_min=100, t_max=t_max_fit)
        tau_A[L]  = tau_e
        te_str = f'{tau_e:10.1f}' if not math.isnan(tau_e) else '      n/a'
        tp_str = f'{tau_p:10.3f}' if not math.isnan(tau_p) else '      n/a'
        print(f'{L:5d}  {U4:7.3f}  {absM:9.5f}  {te_str}  '
              f'{r2_e:7.3f}  {tp_str}  {r2_p:7.3f}')

    z_A, r2_zA = fit_z_from_tau(
        [L for L in L_A if not math.isnan(tau_A.get(L, float('nan')))],
        [tau_A[L] for L in L_A if not math.isnan(tau_A.get(L, float('nan')))])
    print(f'\n  z from tau_L ~ L^z:  z={z_A:.3f}  (R2={r2_zA:.3f})')
    print(f'  Model A (mean-field) prediction: z=2.00')

    # Phase B: tau_corr(P)
    print(f'\n=== PHASE B: tau_corr(P) divergence (r_w={RW_B}, L={L_B}) ===')
    print(f'{"P":>7}  {"xi":>6}  {"U4":>7}  {"absM":>9}  '
          f'{"tau_exp":>10}  {"R2":>7}')
    print('-' * 55)

    tau_B   = {}
    P_B_fit = []
    tau_B_fit = []
    for pc in PC_B:
        key   = f'pc{pc:.4f}'
        xi    = RW_B * math.sqrt(pc)
        seeds = raw.get('B', {}).get(key, {})
        u4s   = [v['U4']   for v in seeds.values()
                 if not math.isnan(v.get('U4', float('nan')))]
        Ms    = [v['absM'] for v in seeds.values()
                 if not math.isnan(v.get('absM', float('nan')))]
        U4    = ms(u4s);  absM = ms(Ms)
        lags, acf = avg_acf(list(seeds.values()))
        t_max_fit = max(lags) // 2 if lags else 100
        tau_e, r2_e = fit_tau_exp(lags, acf, t_min=50, t_max=t_max_fit)
        tau_B[pc] = tau_e
        if not math.isnan(tau_e) and tau_e > 0:
            P_B_fit.append(pc)
            tau_B_fit.append(tau_e)
        te_str = f'{tau_e:10.1f}' if not math.isnan(tau_e) else '       n/a'
        r2_str = f'{r2_e:7.3f}'   if not math.isnan(r2_e)  else '    n/a'
        print(f'{pc:7.4f}  {xi:6.3f}  {U4:7.3f}  {absM:9.5f}  {te_str}  {r2_str}')

    z_B, r2_zB = fit_z_from_P(P_B_fit, tau_B_fit, nu=0.98)
    print(f'\n  Fit tau ~ P^{{-z*nu}} (P_c=0+):')
    print(f'  z={z_B:.3f}  (R2={r2_zB:.3f},  nu=0.98 fixed)')
    print(f'  Model A: z=2.00 -> tau ~ P^{{-1.96}}')

    # Phase C: C(t) shape at criticality
    print(f'\n=== PHASE C: C(t) shape at criticality (r_w={RW_C}, '
          f'P={PC_C}, L={L_C}) ===')
    seeds_C = raw.get('C', {})
    u4s_C   = [v['U4']   for v in seeds_C.values()
               if not math.isnan(v.get('U4', float('nan')))]
    Ms_C    = [v['absM'] for v in seeds_C.values()
               if not math.isnan(v.get('absM', float('nan')))]
    print(f'  U4={ms(u4s_C):.3f}  absM={ms(Ms_C):.5f}')

    lags_C, acf_C = avg_acf(list(seeds_C.values()))
    t_max_C = max(lags_C) // 2 if lags_C else 100

    tau_C_e, r2_C_e = fit_tau_exp(lags_C, acf_C,  t_min=50,  t_max=t_max_C)
    lam_C,   r2_C_p = fit_lambda_pow(lags_C, acf_C, t_min=100, t_max=t_max_C)

    print(f'  Exp fit:  tau = {tau_C_e:.1f}  (R2={r2_C_e:.3f})'
          f'  t_range=[50, {t_max_C}]')
    print(f'  Pow fit:  lambda = {lam_C:.3f}  (R2={r2_C_p:.3f})'
          f'  t_range=[100, {t_max_C}]')

    better = 'EXP' if (not math.isnan(r2_C_e) and
                       (math.isnan(r2_C_p) or r2_C_e >= r2_C_p)) else 'POW'
    print(f'  Better fit: {better}')
    if not math.isnan(lam_C):
        z_C = 2. * 0.628 / (lam_C * 0.98)
        print(f'  If power law: z_C = 2*beta/(lambda*nu) = '
              f'2*0.628/({lam_C:.3f}*0.98) = {z_C:.3f}')
    else:
        z_C = float('nan')

    # Summary
    print(f'\n=== DYNAMIC EXPONENT SUMMARY ===')
    print(f'  z (Phase A, FSS tau_L ~ L^z):         z={z_A:.3f}  '
          f'(R2={r2_zA:.3f})')
    print(f'  z (Phase B, tau ~ P^{{-z*nu}}):          z={z_B:.3f}  '
          f'(R2={r2_zB:.3f})')
    if not math.isnan(z_C):
        print(f'  z (Phase C, 2*beta/(lambda*nu)):      z={z_C:.3f}  '
              f'(lambda={lam_C:.3f})')
    z_vals = [v for v in [z_A, z_B, z_C]
              if not math.isnan(v) and v > 0]
    if z_vals:
        z_mean = sum(z_vals) / len(z_vals)
        z_std  = math.sqrt(sum((v - z_mean) ** 2 for v in z_vals) / len(z_vals))
        print(f'\n  Best estimate: z = {z_mean:.3f} +- {z_std:.3f}')
        diff_mf = abs(z_mean - 2.0)
        verdict = 'CONSISTENT with Model A (z=2)' if diff_mf < 0.3 else \
                  'ANOMALOUS (z != 2, new dynamic universality class)'
        print(f'  Model A (z=2): difference = {diff_mf:.3f}  -> {verdict}')
        # lambda from z
        lam_pred = 2. * 0.628 / (z_mean * 0.98)
        print(f'  Predicted lambda = 2*beta/(z*nu) = '
              f'2*0.628/({z_mean:.3f}*0.98) = {lam_pred:.3f}')

    # Save analysis
    analysis = dict(
        tau_A={str(L): tau_A.get(L, float('nan')) for L in L_A},
        tau_B={str(pc): tau_B.get(pc, float('nan')) for pc in PC_B},
        z_A=z_A, r2_zA=r2_zA,
        z_B=z_B, r2_zB=r2_zB,
        z_C=z_C if not math.isnan(z_C) else None,
        lambda_C=lam_C if not math.isnan(lam_C) else None,
        tau_C_exp=tau_C_e if not math.isnan(tau_C_e) else None,
        rw_a=RW_A, pc_a=PC_A, L_a=L_A,
        rw_b=RW_B, pc_b=PC_B, L_b=L_B,
        rw_c=RW_C, pc_c=PC_C, L_c=L_C,
        # Store averaged ACFs for figures
        acf_A={},
        acf_B={},
        acf_C={'lags': lags_C, 'acf': acf_C},
    )
    for L in L_A:
        key = f'L{L}'
        seeds = raw.get('A', {}).get(key, {})
        lags, acf = avg_acf(list(seeds.values()))
        analysis['acf_A'][str(L)] = {'lags': lags, 'acf': acf}
    for pc in PC_B:
        key = f'pc{pc:.4f}'
        seeds = raw.get('B', {}).get(key, {})
        lags, acf = avg_acf(list(seeds.values()))
        analysis['acf_B'][str(pc)] = {'lags': lags, 'acf': acf}

    with open(ANALYSIS_FILE, 'w') as f:
        json.dump(analysis, f, indent=2)
    print(f'\nAnalysis saved to {ANALYSIS_FILE}')
