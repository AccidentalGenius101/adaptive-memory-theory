"""
Paper 72 - Direct Anomalous Dimension eta from Two-Point fieldM Correlator

Background:
  Paper 71 found indirect eta ~ 1.28 from hyperscaling: eta = 2*beta/nu ~ 2*0.628/0.98.
  This is only valid if hyperscaling holds exactly in d=2.
  Paper 72 measures eta DIRECTLY from the two-point correlator:
    G(r) ~ r^{-(d-2+eta)} = r^{-eta}  (in d=2)
  at criticality (xi = xi* ~ 0.50, r_w=5, P = P_c = 0.010).

  Three phases:
    A: Direct eta at criticality (r_w=5, P=0.010), L in {80,100,120,160}
       G(r) time-averaged over snapshots; fit log-log slope in [r_min, L/4].
    B: eta vs r_w (Levy-DP test): r_w in {3,5,8} at each r_w's P_c, L=120.
       If eta shifts with r_w -> Levy manifold. If constant -> universal.
    C: Correlation length xi_corr(P) off criticality (r_w=5, 6 P values, L=120).
       Fit G(r) ~ r^{-eta} * exp(-r/xi_corr); then xi_corr ~ |P-P_c|^{-nu}.
       Cross-check nu with Paper 63 (nu=0.97) and Paper 71 (nu=0.98).
"""

import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import numpy as np
import json
import math
import multiprocessing as mp
from pathlib import Path

# -- Physical parameters (identical to Papers 68-71) ---------------------------
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

# -- Phase A: eta at criticality -----------------------------------------------
RW_A     = 5
PC_A     = 0.010    # P_c for r_w=5 (xi* = 5*sqrt(0.010) = 0.500)
L_A      = [80, 100, 120, 160]
SEEDS_A  = list(range(8))
NSTEPS_A = 20_000
SNAP_A   = 50       # snapshot every 50 steps after equilibration

# -- Phase B: Levy-DP test (eta vs r_w) ----------------------------------------
# Use xi*=0.50 as common critical manifold (Protocol B assumption)
# P_c = (0.50 / r_w)^2
RW_B     = [3, 5, 8]
PC_B     = {3: 0.0278, 5: 0.0100, 8: 0.0039}   # (0.50/r_w)^2 rounded
L_B      = 120
SEEDS_B  = list(range(8))
NSTEPS_B = 20_000
SNAP_B   = 50

# -- Phase C: correlation length off-criticality --------------------------------
RW_C     = 5
PC_C     = [0.005, 0.007, 0.010, 0.013, 0.017, 0.022]
L_C      = 120
SEEDS_C  = list(range(8))
NSTEPS_C = 15_000
SNAP_C   = 50

RESULTS_FILE  = Path(__file__).parent / 'results' / 'paper72_results.json'
ANALYSIS_FILE = Path(__file__).parent / 'results' / 'paper72_analysis.json'


# -- Lattice helpers (verbatim from Papers 68-71) ------------------------------
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


# -- Radial average of 2D periodic correlator ----------------------------------
def radial_average(corr_2d, L):
    """Radially average a 2D periodic correlator (L x L).
    Returns (r_list, c_list) for integer r in [1, L//2]."""
    max_r = L // 2
    bins = {}
    for iy in range(L):
        dy = iy if iy <= L // 2 else L - iy
        for ix in range(L):
            dx = ix if ix <= L // 2 else L - ix
            r = int(round(math.sqrt(dx * dx + dy * dy)))
            if 1 <= r <= max_r:
                bins.setdefault(r, []).append(corr_2d[iy, ix])
    r_list = sorted(bins.keys())
    c_list = [float(np.mean(bins[r])) for r in r_list]
    return r_list, c_list


# -- Core simulation with correlator -------------------------------------------
def run_one_corr(L, P_causal, seed, nsteps, wave_prob, r_w, snap_every=50):
    """Run VCML simulation and compute time-averaged two-point correlator of phi.

    The correlator is of the signed phi field:
      phi_signed[i] = +phi[i] if zone[i]==0, -phi[i] if zone[i]==1
    Connected: G(r) = <phi_s(x) phi_s(x+r)> - <phi_s>^2
    Time-averaged over snapshots after equilibration.
    """
    rng    = np.random.RandomState(seed)
    nb     = build_nb(L)
    N      = L * L
    col_a  = np.arange(N) % L
    zone   = (col_a >= L // 2).astype(int)
    zone_sign = np.where(zone == 0, 1.0, -1.0)

    s      = rng.choice([-1, 1], N).astype(float)
    base   = s * 0.1
    mid    = np.zeros(N)
    phi    = np.zeros(N)
    streak = np.zeros(N, int)
    wave_z = 0
    ss0    = int(nsteps * SS_FRAC)

    Mseries = []
    # Accumulate power spectrum for time-averaged correlator (rfft2 shape)
    corr_accum = np.zeros((L, L // 2 + 1))
    corr_count = 0

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

            # Snapshot for correlator
            if (t - ss0) % snap_every == 0:
                phi_s  = phi * zone_sign
                phi_c  = phi_s - float(np.mean(phi_s))
                phi_2d = phi_c.reshape(L, L)
                F = np.fft.rfft2(phi_2d)
                corr_accum += np.abs(F) ** 2
                corr_count += 1

    # Convert accumulated power to real-space correlator
    if corr_count > 0:
        avg_pow  = corr_accum / corr_count
        corr_2d  = np.real(np.fft.irfft2(avg_pow)) / N
        r_list, c_list = radial_average(corr_2d, L)
    else:
        r_list, c_list = [], []

    M_arr = np.array(Mseries)
    M2    = float(np.mean(M_arr ** 2))
    M4    = float(np.mean(M_arr ** 4))
    absM  = float(np.mean(np.abs(M_arr)))
    U4    = float(1. - M4 / (3. * M2 ** 2)) if M2 > 1e-12 else float('nan')

    return dict(absM=absM, M2=M2, U4=U4,
                corr_r=r_list, corr_C=c_list,
                n_snaps=corr_count)


# -- Fit helpers ---------------------------------------------------------------
def fit_eta(r_list, C_list, r_min=3, r_max_frac=0.25, L=None):
    """Fit G(r) ~ r^{-eta} by log-log linear regression.
    Returns (eta, r2, n_pts)."""
    r_max = int((L or (max(r_list) if r_list else 10)) * r_max_frac)
    pairs = [(r, C) for r, C in zip(r_list, C_list)
             if r_min <= r <= r_max and C > 0]
    if len(pairs) < 4:
        return float('nan'), float('nan'), 0
    lx = [math.log(r) for r, C in pairs]
    ly = [math.log(C) for r, C in pairs]
    n = len(lx)
    mx, my = sum(lx) / n, sum(ly) / n
    ssxx = sum((x - mx) ** 2 for x in lx)
    ssxy = sum((x - mx) * (y - my) for x, y in zip(lx, ly))
    if ssxx < 1e-20:
        return float('nan'), float('nan'), n
    slope = ssxy / ssxx
    yhat  = [my + slope * (x - mx) for x in lx]
    sstot = sum((y - my) ** 2 for y in ly)
    ssres = sum((y - yh) ** 2 for y, yh in zip(ly, yhat))
    r2    = 1. - ssres / sstot if sstot > 1e-20 else float('nan')
    return -slope, r2, n   # eta = -slope since G ~ r^{-eta}


def fit_xi_corr(r_list, C_list, eta, r_min=3, r_max_frac=0.45, L=None):
    """Fit G(r) ~ A * r^{-eta} * exp(-r/xi_corr) with fixed eta.
    Linearise: log(G) + eta*log(r) = log(A) - r/xi_corr
    Linear regression of y = log(C) + eta*log(r) vs x = r.
    Returns (xi_corr, r2)."""
    r_max = int((L or (max(r_list) if r_list else 10)) * r_max_frac)
    pairs = [(r, C) for r, C in zip(r_list, C_list)
             if r_min <= r <= r_max and C > 0]
    if len(pairs) < 4 or math.isnan(eta):
        return float('nan'), float('nan')
    xs = [r for r, C in pairs]
    ys = [math.log(C) + eta * math.log(r) for r, C in pairs]
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    ssxx = sum((x - mx) ** 2 for x in xs)
    ssxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    if ssxx < 1e-20:
        return float('nan'), float('nan')
    slope = ssxy / ssxx
    yhat  = [my + slope * (x - mx) for x in xs]
    sstot = sum((y - my) ** 2 for y in ys)
    ssres = sum((y - yh) ** 2 for y, yh in zip(ys, yhat))
    r2    = 1. - ssres / sstot if sstot > 1e-20 else float('nan')
    if slope >= 0:
        return float('nan'), float('nan')
    return -1.0 / slope, r2


def fit_nu_from_xi(P_list, xi_list, P_c):
    """Fit xi_corr ~ |P - P_c|^{-nu} by log-log regression.
    Returns (nu, r2)."""
    pairs = [(abs(P - P_c), xi) for P, xi in zip(P_list, xi_list)
             if not math.isnan(xi) and xi > 0 and abs(P - P_c) > 1e-8]
    if len(pairs) < 3:
        return float('nan'), float('nan')
    lx = [math.log(dp) for dp, xi in pairs]
    ly = [math.log(xi) for dp, xi in pairs]
    n = len(lx)
    mx, my = sum(lx) / n, sum(ly) / n
    ssxx = sum((x - mx) ** 2 for x in lx)
    ssxy = sum((x - mx) * (y - my) for x, y in zip(lx, ly))
    if ssxx < 1e-20:
        return float('nan'), float('nan')
    slope = ssxy / ssxx
    yhat  = [my + slope * (x - mx) for x in lx]
    sstot = sum((y - my) ** 2 for y in ly)
    ssres = sum((y - yh) ** 2 for y, yh in zip(ly, yhat))
    r2    = 1. - ssres / sstot if sstot > 1e-20 else float('nan')
    return -slope, r2   # nu = -slope since xi ~ |dP|^{-nu}


def avg_corr(results_list):
    """Average correlators over a list of run_one_corr results."""
    valid = [r for r in results_list
             if r and 'corr_r' in r and len(r.get('corr_r', [])) > 0]
    if not valid:
        return [], []
    r_ref = valid[0]['corr_r']
    C_acc = np.zeros(len(r_ref))
    cnt   = 0
    for r in valid:
        if r['corr_r'] == r_ref:
            C_acc += np.array(r['corr_C'])
            cnt   += 1
    return r_ref, (C_acc / cnt).tolist() if cnt > 0 else []


# -- Workers -------------------------------------------------------------------
def _wA(args):
    L, seed = args
    return ('A', L, seed,
            run_one_corr(L, PC_A, seed, NSTEPS_A, _prob_A(L), RW_A, SNAP_A))

def _wB(args):
    r_w, seed = args
    return ('B', r_w, seed,
            run_one_corr(L_B, PC_B[r_w], seed, NSTEPS_B, _prob_A(L_B), r_w, SNAP_B))

def _wC(args):
    pc, seed = args
    return ('C', pc, seed,
            run_one_corr(L_C, pc, seed, NSTEPS_C, _prob_A(L_C), RW_C, SNAP_C))


# -- MAIN ----------------------------------------------------------------------
if __name__ == '__main__':
    mp.freeze_support()
    RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)

    raw = {}
    if RESULTS_FILE.exists():
        with open(RESULTS_FILE) as f:
            raw = json.load(f)

    ncpu = mp.cpu_count()

    # -- Phase A: eta at criticality -------------------------------------------
    if 'A' not in raw:
        args_A = [(L, seed) for L in L_A for seed in SEEDS_A]
        print(f'Phase A: {len(args_A)} runs -- direct eta, r_w={RW_A}, P={PC_A}, '
              f'L in {L_A} ...')
        with mp.Pool(processes=min(len(args_A), ncpu)) as pool:
            res = pool.map(_wA, args_A)
        raw['A'] = {}
        for _, L, seed, r in res:
            raw['A'].setdefault(f'L{L}', {})[str(seed)] = r
        with open(RESULTS_FILE, 'w') as f:
            json.dump(raw, f)
        print('Phase A done.')

    # -- Phase B: eta vs r_w ---------------------------------------------------
    if 'B' not in raw:
        args_B = [(r_w, seed) for r_w in RW_B for seed in SEEDS_B]
        print(f'\nPhase B: {len(args_B)} runs -- Levy test, r_w in {RW_B}, L={L_B} ...')
        with mp.Pool(processes=min(len(args_B), ncpu)) as pool:
            res = pool.map(_wB, args_B)
        raw['B'] = {}
        for _, r_w, seed, r in res:
            raw['B'].setdefault(f'rw{r_w}', {})[str(seed)] = r
        with open(RESULTS_FILE, 'w') as f:
            json.dump(raw, f)
        print('Phase B done.')

    # -- Phase C: xi_corr(P) off criticality -----------------------------------
    if 'C' not in raw:
        args_C = [(pc, seed) for pc in PC_C for seed in SEEDS_C]
        print(f'\nPhase C: {len(args_C)} runs -- xi_corr scan, r_w={RW_C}, '
              f'P in {PC_C} ...')
        with mp.Pool(processes=min(len(args_C), ncpu)) as pool:
            res = pool.map(_wC, args_C)
        raw['C'] = {}
        for _, pc, seed, r in res:
            raw['C'].setdefault(f'pc{pc:.4f}', {})[str(seed)] = r
        with open(RESULTS_FILE, 'w') as f:
            json.dump(raw, f)
        print('Phase C done.')

    # -- Analysis --------------------------------------------------------------

    # Phase A: eta at each L
    print(f'\n=== PHASE A: Direct eta from G(r) ~ r^{{-eta}} at criticality ===')
    print(f'  r_w={RW_A}, P={PC_A} (xi={RW_A*math.sqrt(PC_A):.3f})')
    print(f'  {"L":>5}  {"U4":>7}  {"absM":>9}  {"eta":>7}  {"R2":>6}  '
          f'{"n_pts":>6}  {"G(r=3)":>10}')
    print('-' * 65)

    eta_A = {}
    for L in L_A:
        key = f'L{L}'
        seeds = raw.get('A', {}).get(key, {})
        # Standard observables
        u4s  = [v['U4']   for v in seeds.values() if not math.isnan(v.get('U4', float('nan')))]
        Ms   = [v['absM'] for v in seeds.values() if not math.isnan(v.get('absM', float('nan')))]
        U4   = ms(u4s)
        absM = ms(Ms)
        # Averaged correlator
        r_avg, C_avg = avg_corr(list(seeds.values()))
        eta, r2, npts = fit_eta(r_avg, C_avg, r_min=3, r_max_frac=0.25, L=L)
        eta_A[L] = eta
        # G at r=3
        g3 = C_avg[r_avg.index(3)] if 3 in r_avg else float('nan')
        print(f'  {L:5d}  {U4:7.3f}  {absM:9.5f}  {eta:7.3f}  {r2:6.3f}  '
              f'{npts:6d}  {g3:10.3e}')

    eta_vals = [v for v in eta_A.values() if not math.isnan(v)]
    if eta_vals:
        eta_mean = sum(eta_vals) / len(eta_vals)
        eta_std  = math.sqrt(sum((v - eta_mean)**2 for v in eta_vals) / len(eta_vals))
        print(f'\n  Mean eta_direct = {eta_mean:.3f} +- {eta_std:.3f}')
        eta_indirect = 2 * 0.628 / 0.98   # from Paper 71
        print(f'  Indirect eta (2*beta/nu=2*0.628/0.98) = {eta_indirect:.3f}')
        diff = abs(eta_mean - eta_indirect)
        verdict = 'CONSISTENT' if diff < 0.15 else 'INCONSISTENT'
        print(f'  Difference = {diff:.3f}  -> {verdict}')
        eta_best = eta_mean
    else:
        eta_best = float('nan')
        print('  No valid eta measurements.')

    # Phase B: eta vs r_w (Levy-DP test)
    print(f'\n=== PHASE B: Levy-DP Test (eta vs r_w) ===')
    print(f'  {"r_w":>4}  {"P_c":>7}  {"xi_c":>6}  {"U4":>7}  {"eta":>7}  '
          f'{"R2":>6}')
    print('-' * 50)

    eta_B = {}
    for r_w in RW_B:
        key = f'rw{r_w}'
        pc  = PC_B[r_w]
        xi  = r_w * math.sqrt(pc)
        seeds = raw.get('B', {}).get(key, {})
        u4s = [v['U4']   for v in seeds.values() if not math.isnan(v.get('U4', float('nan')))]
        U4  = ms(u4s)
        r_avg, C_avg = avg_corr(list(seeds.values()))
        eta, r2, _ = fit_eta(r_avg, C_avg, r_min=3, r_max_frac=0.25, L=L_B)
        eta_B[r_w] = eta
        print(f'  {r_w:4d}  {pc:7.4f}  {xi:6.3f}  {U4:7.3f}  {eta:7.3f}  {r2:6.3f}')

    eta_B_vals = [v for v in eta_B.values() if not math.isnan(v)]
    if len(eta_B_vals) >= 2:
        spread = max(eta_B_vals) - min(eta_B_vals)
        verdict = 'UNIVERSAL' if spread < 0.15 else 'NON-UNIVERSAL (Levy shift)'
        print(f'\n  eta spread across r_w = {spread:.3f}  -> {verdict}')

    # Phase C: xi_corr(P) and nu
    print(f'\n=== PHASE C: Correlation Length xi_corr(P) ===')
    print(f'  r_w={RW_C}, L={L_C}, eta_fixed={eta_best:.3f}')
    print(f'  {"P":>7}  {"xi":>6}  {"U4":>7}  {"xi_corr":>9}  {"R2":>6}')
    print('-' * 45)

    xi_corr_list = []
    for pc in PC_C:
        key   = f'pc{pc:.4f}'
        xi    = RW_C * math.sqrt(pc)
        seeds = raw.get('C', {}).get(key, {})
        u4s   = [v['U4'] for v in seeds.values() if not math.isnan(v.get('U4', float('nan')))]
        U4    = ms(u4s)
        r_avg, C_avg = avg_corr(list(seeds.values()))
        if not math.isnan(eta_best):
            xi_c, r2_c = fit_xi_corr(r_avg, C_avg, eta_best, r_min=3,
                                     r_max_frac=0.45, L=L_C)
        else:
            xi_c, r2_c = float('nan'), float('nan')
        xi_corr_list.append(xi_c)
        xi_str = f'{xi_c:9.2f}' if not math.isnan(xi_c) else '      inf'
        print(f'  {pc:7.4f}  {xi:6.3f}  {U4:7.3f}  {xi_str}  {r2_c:6.3f}')

    # Fit nu from xi_corr ~ |P - P_c|^{-nu}
    P_c_fit = PC_A   # 0.010
    nu, r2_nu = fit_nu_from_xi(PC_C, xi_corr_list, P_c_fit)
    print(f'\n  nu fit (xi_corr ~ |P-{P_c_fit}|^{{-nu}}):  nu = {nu:.3f}  '
          f'(R2={r2_nu:.3f})')
    print(f'  Paper 63 nu=0.97; Paper 71 nu=0.98')

    # Scaling relation check
    print(f'\n=== SCALING RELATION CHECK ===')
    if not math.isnan(eta_best):
        beta_check = 0.628; nu_check = 0.98
        lhs = 2 * beta_check / nu_check   # should equal eta
        print(f'  2*beta/nu = 2*{beta_check}/{nu_check} = {lhs:.3f}')
        print(f'  eta_direct = {eta_mean:.3f} (from Phase A)')
        diff = abs(eta_mean - lhs)
        print(f'  Difference = {diff:.3f}  '
              f'-> {"SCALING HOLDS" if diff < 0.15 else "SCALING VIOLATED"}')
    if not math.isnan(nu):
        print(f'  nu_direct (Phase C) = {nu:.3f}  '
              f'vs Papers 63/71 = 0.97-0.98')

    # -- Save analysis ---------------------------------------------------------
    analysis = dict(
        eta_A={str(L): eta_A.get(L, float('nan')) for L in L_A},
        eta_B={str(r_w): eta_B.get(r_w, float('nan')) for r_w in RW_B},
        eta_best=eta_best if not math.isnan(eta_best) else None,
        xi_corr={str(pc): xi_corr_list[i] for i, pc in enumerate(PC_C)},
        nu=nu, nu_r2=r2_nu,
        rw_a=RW_A, pc_a=PC_A, L_a=L_A,
        rw_b=RW_B, pc_b=PC_B,
        rw_c=RW_C, pc_c=PC_C, L_c=L_C,
        # Store averaged correlators for figure
        corr_A={},
        corr_B={},
        corr_C={},
    )
    for L in L_A:
        key = f'L{L}'
        seeds = raw.get('A', {}).get(key, {})
        r_avg, C_avg = avg_corr(list(seeds.values()))
        analysis['corr_A'][str(L)] = {'r': r_avg, 'C': C_avg}
    for r_w in RW_B:
        key = f'rw{r_w}'
        seeds = raw.get('B', {}).get(key, {})
        r_avg, C_avg = avg_corr(list(seeds.values()))
        analysis['corr_B'][str(r_w)] = {'r': r_avg, 'C': C_avg}
    for pc in PC_C:
        key = f'pc{pc:.4f}'
        seeds = raw.get('C', {}).get(key, {})
        r_avg, C_avg = avg_corr(list(seeds.values()))
        analysis['corr_C'][str(pc)] = {'r': r_avg, 'C': C_avg}

    with open(ANALYSIS_FILE, 'w') as f:
        json.dump(analysis, f, indent=2)
    print(f'\nAnalysis saved to {ANALYSIS_FILE}')
