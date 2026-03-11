"""
Paper 71 - RG Fixed Point Characterisation via Composite Variable xi = r_w * sqrtP

Background:
  Paper 70 showed that the ordering threshold follows r_w*(P) ~ P^{-1/2} under
  matched-coverage (Protocol B), establishing the composite variable
  xi = r_w * sqrt(P) as scale-invariant at threshold.  The wave operator has
  RG dimension d_wave = 1/2; the threshold is the critical manifold xi = xi*.

  The open questions are:
    1. Does U4 (and |M|) depend ONLY on xi, or also on (r_w, P) separately?
       (iso-xi universality test)
    2. What is the order parameter exponent beta extracted from |M| ~ (xi - xi*)^beta?
       Paper 63 gave beta ~ 0.65 via a P-scan, which may be contaminated.
       A xi-scan is cleaner: xi is the actual scaling field at the RG fixed point.
    3. What is the correlation-length exponent nu?
       FSS analysis: |M| ~ L^{-beta/nu} f(L^{1/nu}(xi-xi*)).

Phase A: Iso-xi universality test.
  Fix xi  in  {0.40, 0.55, 0.70, 0.85}; for each xi use 3 different (r_w, P) pairs.
  L=80, Protocol A, 8 seeds, 10k steps.
  Test: U4 and |M| should depend only on xi, not on the specific (r_w, P).

Phase B: beta extraction from xi-scan.
  Fix r_w=5, scan P in {0.004..0.080} -> xi = 5*sqrt(P) in [0.32, 1.41], L=80.
  Also fix r_w=8, scan P in {0.002..0.030} -> xi = 8*sqrt(P) in [0.36, 1.39].
  Protocol A, 8 seeds, 12k steps.
  Fit |M| ~ (xi - xi*)^beta for xi > xi* using both r_w values; check consistency.

Phase C: FSS collapse and nu extraction.
  Fix r_w=5, P  in  {0.010, 0.015, 0.020, 0.030, 0.050} (xi  in  {0.50..1.12}).
  L  in  {40, 60, 80, 100, 120}.
  Protocol A, 8 seeds, 12k steps.
  Extract beta/nu from FSS slopes d(ln|M|)/d(lnL) vs xi.
  Attempt FSS collapse onto F(L^{1/nu}*(xi-xi*)) to determine nu.
"""

import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import numpy as np
import json
import math
import multiprocessing as mp
from pathlib import Path

# -- Physical parameters (identical to Papers 68-70) ---------------------------
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
A_REF = wave_area(5)   # 61

def _wef(L): return WE_BASE * (L_BASE / L) ** 2
def _prob_A(L): return min(1.0, 1.0 / _wef(L))

# -- Phase A: iso-xi pairs -------------------------------------------------------
# For each target xi: pairs (r_w, P) satisfying r_w * sqrt(P) = xi exactly.
# P = (xi / r_w)^2, rounded to 4 decimal places.
def _xi_pairs(xi_target, rw_list):
    pairs = []
    for r in rw_list:
        P = round((xi_target / r) ** 2, 4)
        if 0.001 <= P <= 0.25:
            pairs.append((r, P))
    return pairs

XI_A      = [0.40, 0.55, 0.70, 0.85]
RW_A_POOL = [2, 3, 4, 5, 8]
PAIRS_A   = {xi: _xi_pairs(xi, RW_A_POOL) for xi in XI_A}
L_A       = 80
SEEDS_A   = list(range(8))
NSTEPS_A  = 10_000

# -- Phase B: beta extraction ------------------------------------------------------
RW_B5_PC  = [0.004, 0.006, 0.008, 0.010, 0.015, 0.020, 0.030, 0.050, 0.080]
RW_B8_PC  = [0.002, 0.003, 0.004, 0.006, 0.010, 0.015, 0.020, 0.030]
L_B       = 80
SEEDS_B   = list(range(8))
NSTEPS_B  = 12_000

# -- Phase C: FSS --------------------------------------------------------------
RW_C     = 5
PC_C     = [0.010, 0.015, 0.020, 0.030, 0.050]   # xi = 5*sqrt(P)
L_C      = [40, 60, 80, 100, 120]
SEEDS_C  = list(range(8))
NSTEPS_C = 12_000

RESULTS_FILE  = Path(__file__).parent / 'results' / 'paper71_results.json'
ANALYSIS_FILE = Path(__file__).parent / 'results' / 'paper71_analysis.json'


# -- Lattice helpers (verbatim from Papers 68-70) ------------------------------
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


# -- Core simulation (verbatim from Papers 68-70) -----------------------------
def run_one(L, P_causal, seed, nsteps, wave_prob, r_w):
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

    M_arr = np.array(Mseries)
    M2    = float(np.mean(M_arr**2)); M4 = float(np.mean(M_arr**4))
    absM  = float(np.mean(np.abs(M_arr)))
    U4    = float(1. - M4 / (3. * M2**2)) if M2 > 1e-12 else float('nan')
    return dict(absM=absM, M2=M2, U4=U4)


# -- Workers --------------------------------------------------------------------
def _wA(args):
    r_w, pc, seed, xi = args
    return ('A', xi, r_w, pc, seed,
            run_one(L_A, pc, seed, NSTEPS_A, _prob_A(L_A), r_w))

def _wB(args):
    r_w, pc, seed = args
    return ('B', r_w, pc, seed,
            run_one(L_B, pc, seed, NSTEPS_B, _prob_A(L_B), r_w))

def _wC(args):
    pc, L, seed = args
    return ('C', pc, L, seed,
            run_one(L, pc, seed, NSTEPS_C, _prob_A(L), RW_C))


# -- Aggregation ----------------------------------------------------------------
def agg_group(d):
    vals = list(d.values())
    def _ms(k): return ms([v.get(k, float('nan')) for v in vals])
    def _se(k): return ses([v.get(k, float('nan')) for v in vals])
    return dict(U4=_ms('U4'), U4_se=_se('U4'),
                absM=_ms('absM'), absM_se=_se('absM'))


# -- Analysis helpers -----------------------------------------------------------
def safe_linfit(xs, ys):
    """Linear fit log(y) ~ slope*log(x) + const. Returns (slope, r2)."""
    valid = [(x, y) for x, y in zip(xs, ys)
             if x > 0 and y > 0 and not math.isnan(y) and not math.isnan(x)]
    if len(valid) < 3:
        return float('nan'), float('nan')
    lx = [math.log(x) for x, y in valid]
    ly = [math.log(y) for x, y in valid]
    n = len(lx)
    mx, my = sum(lx)/n, sum(ly)/n
    ssxx = sum((x-mx)**2 for x in lx)
    ssxy = sum((x-mx)*(y-my) for x, y in zip(lx, ly))
    if ssxx < 1e-20:
        return float('nan'), float('nan')
    slope = ssxy / ssxx
    yhat = [my + slope*(x-mx) for x in lx]
    sstot = sum((y-my)**2 for y in ly)
    ssres = sum((y-yh)**2 for y, yh in zip(ly, yhat))
    r2 = 1 - ssres/sstot if sstot > 1e-20 else float('nan')
    return slope, r2


def fit_beta(xi_list, absM_list, xi_star):
    """Fit |M| ~ (xi - xi*)^beta for xi > xi*."""
    pairs = [(xi - xi_star, M) for xi, M in zip(xi_list, absM_list)
             if xi > xi_star and M > 0 and not math.isnan(M)]
    if len(pairs) < 3:
        return float('nan'), float('nan')
    dxi = [p[0] for p in pairs]
    M   = [p[1] for p in pairs]
    return safe_linfit(dxi, M)


# -- MAIN -----------------------------------------------------------------------
if __name__ == '__main__':
    mp.freeze_support()
    RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)

    raw = {}
    if RESULTS_FILE.exists():
        with open(RESULTS_FILE) as f:
            raw = json.load(f)

    ncpu = mp.cpu_count()

    # -- Phase A: iso-xi universality --------------------------------------------
    if 'A' not in raw:
        args_A = [(r_w, pc, seed, xi)
                  for xi in XI_A
                  for r_w, pc in PAIRS_A[xi]
                  for seed in SEEDS_A]
        print(f'Phase A: {len(args_A)} runs -- iso-xi universality test, L={L_A} ...')
        with mp.Pool(processes=min(len(args_A), ncpu)) as pool:
            res = pool.map(_wA, args_A)
        raw['A'] = {}
        for _, xi, r_w, pc, seed, r in res:
            raw['A'].setdefault(f'xi{xi:.2f}_rw{r_w}_pc{pc:.4f}', {})[str(seed)] = r
        with open(RESULTS_FILE, 'w') as f:
            json.dump(raw, f)
        print('Phase A done.')

    # -- Phase B: beta extraction --------------------------------------------------
    if 'B' not in raw:
        args_B5 = [(5, pc, seed) for pc in RW_B5_PC for seed in SEEDS_B]
        args_B8 = [(8, pc, seed) for pc in RW_B8_PC for seed in SEEDS_B]
        args_B  = args_B5 + args_B8
        print(f'\nPhase B: {len(args_B)} runs -- beta extraction, r_w in {{5,8}}, L={L_B} ...')
        with mp.Pool(processes=min(len(args_B), ncpu)) as pool:
            res = pool.map(_wB, args_B)
        raw['B'] = {}
        for _, r_w, pc, seed, r in res:
            raw['B'].setdefault(f'rw{r_w}_pc{pc:.4f}', {})[str(seed)] = r
        with open(RESULTS_FILE, 'w') as f:
            json.dump(raw, f)
        print('Phase B done.')

    # -- Phase C: FSS ----------------------------------------------------------
    if 'C' not in raw:
        args_C = [(pc, L, seed) for pc in PC_C for L in L_C for seed in SEEDS_C]
        print(f'\nPhase C: {len(args_C)} runs -- FSS, r_w={RW_C}, L in {L_C} ...')
        with mp.Pool(processes=min(len(args_C), ncpu)) as pool:
            res = pool.map(_wC, args_C)
        raw['C'] = {}
        for _, pc, L, seed, r in res:
            raw['C'].setdefault(f'pc{pc:.4f}_L{L}', {})[str(seed)] = r
        with open(RESULTS_FILE, 'w') as f:
            json.dump(raw, f)
        print('Phase C done.')

    # -- Aggregate --------------------------------------------------------------
    agg_A = {k: agg_group(v) for k, v in raw.get('A', {}).items()}
    agg_B = {k: agg_group(v) for k, v in raw.get('B', {}).items()}
    agg_C = {k: agg_group(v) for k, v in raw.get('C', {}).items()}

    # -- Phase A: print iso-xi table ---------------------------------------------
    print(f'\n=== PHASE A: Iso-xi Universality Test (L={L_A}) ===')
    print(f'{"xi":>6} {"r_w":>5} {"P":>8}  {"U4":>7} {"±":>6}  {"|M|":>9} {"±":>7}')
    print('-' * 58)
    for xi in XI_A:
        for r_w, pc in PAIRS_A[xi]:
            key = f'xi{xi:.2f}_rw{r_w}_pc{pc:.4f}'
            if key in agg_A:
                d = agg_A[key]
                print(f'{xi:6.2f} {r_w:5d} {pc:8.4f}  '
                      f'{d["U4"]:7.3f} {d["U4_se"]:6.3f}  '
                      f'{d["absM"]:9.4f} {d["absM_se"]:7.4f}')

    # Test universality: for each xi, check U4 variance across (r_w, P) pairs
    print(f'\n--- Universality test: U4 spread at each xi ---')
    for xi in XI_A:
        u4s = []
        for r_w, pc in PAIRS_A[xi]:
            key = f'xi{xi:.2f}_rw{r_w}_pc{pc:.4f}'
            if key in agg_A:
                u = agg_A[key]['U4']
                if not math.isnan(u): u4s.append(u)
        if len(u4s) >= 2:
            spread = max(u4s) - min(u4s)
            print(f'  xi={xi:.2f}: U4 in [{min(u4s):.3f}, {max(u4s):.3f}]  '
                  f'spread={spread:.3f}  {"UNIVERSAL" if spread < 0.10 else "NON-UNIVERSAL"}')

    # -- Phase B: beta extraction --------------------------------------------------
    print(f'\n=== PHASE B: beta Extraction from xi-Scan (L={L_B}) ===')
    for rw_b, pc_list in [(5, RW_B5_PC), (8, RW_B8_PC)]:
        print(f'\n  r_w={rw_b}:')
        print(f'  {"P":>8}  {"xi":>6}  {"|M|":>9}  {"U4":>7}')
        xi_vals, M_vals = [], []
        for pc in pc_list:
            key = f'rw{rw_b}_pc{pc:.4f}'
            if key in agg_B:
                d = agg_B[key]
                xi = rw_b * math.sqrt(pc)
                print(f'  {pc:8.4f}  {xi:6.3f}  {d["absM"]:9.5f}  {d["U4"]:7.3f}')
                if not math.isnan(d['absM']) and d['absM'] > 0:
                    xi_vals.append(xi)
                    M_vals.append(d['absM'])
        # Estimate xi* from Phase A: use xi where U4 ~ 0.30
        # Use all values; try xi*=0.60, 0.65, 0.70
        print(f'\n  beta fits (r_w={rw_b}) at various xi* estimates:')
        for xi_star in [0.45, 0.50, 0.55, 0.60, 0.65, 0.70]:
            beta, r2 = fit_beta(xi_vals, M_vals, xi_star)
            print(f'    xi*={xi_star:.2f}: beta={beta:.3f}  R²={r2:.3f}')

    # -- Phase C: FSS slopes ----------------------------------------------------
    print(f'\n=== PHASE C: FSS Slopes (r_w={RW_C}) ===')
    print(f'{"P":>7}  {"xi":>6}  ', end='')
    for L in L_C: print(f'  L={L:3d}', end='')
    print()
    print('-' * (15 + 8*len(L_C)))

    fss_slopes = {}
    for pc in PC_C:
        xi = RW_C * math.sqrt(pc)
        print(f'{pc:7.4f}  {xi:6.3f}  ', end='')
        u4_L = []
        M_L  = []
        for L in L_C:
            key = f'pc{pc:.4f}_L{L}'
            if key in agg_C:
                d = agg_C[key]
                print(f'  {d["U4"]:6.3f}', end='')
                u4_L.append(d['U4'])
                M_L.append(d['absM'])
            else:
                print(f'  {"?":>6}', end='')
                u4_L.append(float('nan'))
                M_L.append(float('nan'))
        print()
        # FSS slope: d(ln|M|)/d(lnL)
        valid_M = [(L, M) for L, M in zip(L_C, M_L) if M > 0 and not math.isnan(M)]
        if len(valid_M) >= 3:
            slope, r2 = safe_linfit([x[0] for x in valid_M], [x[1] for x in valid_M])
            fss_slopes[pc] = slope
            print(f'         slope={slope:.3f} (R²={r2:.3f})')
        else:
            fss_slopes[pc] = float('nan')

    print(f'\n--- FSS slope vs xi (should pass through -beta/nu at xi=xi*) ---')
    for pc in PC_C:
        xi = RW_C * math.sqrt(pc)
        s  = fss_slopes.get(pc, float('nan'))
        print(f'  xi={xi:.3f}  slope={s:.3f}')

    # Estimate beta/nu from slope at near-critical xi
    # beta/nu ~ -slope at xi closest to xi* (where slope is flattest / sign change)
    slopes_list = [(RW_C*math.sqrt(pc), fss_slopes.get(pc, float('nan'))) for pc in PC_C]
    slopes_list = [(xi, s) for xi, s in slopes_list if not math.isnan(s)]
    if slopes_list:
        # beta/nu from slope closest to 0 (least negative = most ordered side)
        min_abs = min(slopes_list, key=lambda x: abs(x[1]))
        print(f'\n  Most ordered slope: xi={min_abs[0]:.3f}, slope={min_abs[1]:.3f}')
        print(f'  Estimate beta/nu = {-min_abs[1]:.3f}')

    # -- Save analysis ---------------------------------------------------------
    analysis = dict(
        phase_A=agg_A,
        phase_B=agg_B,
        phase_C=agg_C,
        fss_slopes={str(k): v for k, v in fss_slopes.items()},
        xi_A=XI_A,
        pairs_A={str(k): v for k, v in PAIRS_A.items()},
        rw_b5_pc=RW_B5_PC,
        rw_b8_pc=RW_B8_PC,
        rw_c=RW_C, pc_c=PC_C, L_c=L_C,
    )
    with open(ANALYSIS_FILE, 'w') as f:
        json.dump(analysis, f, indent=2)
    print(f'\nAnalysis saved to {ANALYSIS_FILE}')
