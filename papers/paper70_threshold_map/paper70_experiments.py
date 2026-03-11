"""
Paper 70: Phase Boundary Map in (r_w, P) Space.
Scaling of the Ordering Threshold and RG Dimension of the Wave Operator.

Background:
  Paper 69: r_w=1 does not produce ordered phase at P<=0.050. Ordering requires
  minimum wave range r_w*(P). The failure mode identified a NEW critical manifold
  in (r_w, P) space: the threshold curve r_w*(P) below which the wave operator
  is irrelevant (flows to zero under RG coarse-graining).

  Prediction from signal-to-noise analysis:
    Zone-mean signal per wave ~ P * r_w^2 / L^2
    Zone-mean noise ~ 1/L
    With extensive drive (n_waves ~ L^2): signal accumulates as sqrt(n_waves) * signal/wave
    Condition for ordering: P * r_w^2 * sqrt(L^2) / L^2 > threshold
                            P * r_w^2 / L > threshold
    -> r_w*(P, L) ~ sqrt(L / P) ??? Or does the L cancel?

  Under Protocol A (constant wave count): n_waves ~ const (L-independent at fixed L).
  Signal-per-step ~ P * r_w^2 / L^2. Noise ~ 1/L.
  After nsteps: signal ~ P * r_w^2 * nsteps / L^2. Noise ~ 1/L.
  Condition: P * r_w^2 * nsteps / L^2 > 1/L -> r_w^2 > L / (P * nsteps)
  -> r_w*(P, L) ~ sqrt(L / P) (for constant nsteps, Protocol A).

  Under Protocol B (extensive drive, n_waves ~ L^2 * const):
  Signal per step scales as P * r_w^2 / L^2.
  After nsteps: signal ~ P * r_w^2 * n_waves / L^2 ~ P * r_w^2 (L-independent).
  Condition: P * r_w^2 > threshold -> r_w*(P) ~ 1/sqrt(P) (L-INDEPENDENT!).

  KEY PREDICTION:
    Protocol A: r_w*(P, L) ~ sqrt(L) / sqrt(P)  (grows with L -- ordering harder at large L)
    Protocol B: r_w*(P) ~ 1/sqrt(P)               (L-independent -- ordering is thermodynamic)

  Testing both protocols tells us which scaling is correct and gives d_wave.

Phase A: 2D (r_w, P) phase map at fixed L=80.
  r_w in {1, 2, 3, 4, 5, 6, 8, 10, 12}
  P in {0.005, 0.010, 0.020, 0.050, 0.100, 0.200}
  Protocol A (constant rate). 6 seeds, 8k steps.
  Measure U4. Find threshold curve r_w*(P) at U4=0.15 (onset of ordering).

Phase B: L-dependence of threshold at fixed P=0.020.
  r_w in {1, 2, 3, 4, 5, 6, 8}, L in {40, 60, 80, 100, 120}.
  Protocol A. 6 seeds, 8k steps.
  Extract r_w*(L) at fixed P: does it scale as L^0, L^{1/2}, or L^1?
  This directly measures d_wave (RG dimension of wave operator).

Phase C: r_w*(P) scaling under Protocol B (extensive drive).
  L=80, r_w in {1, 2, 3, 4, 5, 6, 8, 10}.
  P in {0.005, 0.010, 0.020, 0.050, 0.100}.
  Protocol B (matched coverage). 6 seeds, 8k steps.
  Under Protocol B: prediction r_w*(P) ~ P^{-1/2} if L-independent.
  Compare with Phase A to disentangle geometry from coverage.
"""

import numpy as np, json, math, multiprocessing as mp
from pathlib import Path

# ── Physical parameters ───────────────────────────────────────────────────────
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
A_REF = wave_area(5)  # 61

def _wef(L): return WE_BASE * (L_BASE / L) ** 2
def _prob_A(L): return min(1.0, 1.0 / _wef(L))
def _prob_B(L, r_w): return min(1.0, (A_REF / wave_area(r_w)) / _wef(L))

# ── Phase A parameters ────────────────────────────────────────────────────────
L_A    = 80
RW_A   = [1, 2, 3, 4, 5, 6, 8, 10, 12]
PC_A   = [round(x, 4) for x in [0.005, 0.010, 0.020, 0.050, 0.100, 0.200]]
SEEDS_A = list(range(6))
NSTEPS_A = 8000

# ── Phase B parameters ────────────────────────────────────────────────────────
PC_B_FIXED = 0.020
RW_B   = [1, 2, 3, 4, 5, 6, 8]
L_B    = [40, 60, 80, 100, 120]
SEEDS_B = list(range(6))
NSTEPS_B = 8000

# ── Phase C parameters ────────────────────────────────────────────────────────
L_C    = 80
RW_C   = [1, 2, 3, 4, 5, 6, 8, 10]
PC_C   = [round(x, 4) for x in [0.005, 0.010, 0.020, 0.050, 0.100]]
SEEDS_C = list(range(6))
NSTEPS_C = 8000

RESULTS_FILE  = Path(__file__).parent / 'results' / 'paper70_results.json'
ANALYSIS_FILE = Path(__file__).parent / 'results' / 'paper70_analysis.json'

U4_ORDER_THRESH = 0.15  # U4 above this = "ordered" for threshold extraction


# ── Lattice helpers ───────────────────────────────────────────────────────────
def build_nb(L):
    N = L * L
    row = np.arange(N) // L; col = np.arange(N) % L
    return np.stack([((row-1)%L)*L+col, ((row+1)%L)*L+col,
                     row*L+(col-1)%L,   row*L+(col+1)%L], axis=1)

def metro(s, nb, T, rng, h_ext=None, hit=None):
    N = len(s); L = int(round(math.sqrt(N)))
    row = np.arange(N) // L; col = np.arange(N) % L
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


# ── Core simulation ────────────────────────────────────────────────────────────
def run_one(L, P_causal, seed, nsteps, wave_prob, r_w):
    rng    = np.random.RandomState(seed)
    nb     = build_nb(L)
    N      = L * L
    N_zone = N // 2
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


# ── Workers ────────────────────────────────────────────────────────────────────
def _wA(args):
    r_w, pc, seed = args
    return ('A', r_w, pc, seed,
            run_one(L_A, pc, seed, NSTEPS_A, _prob_A(L_A), r_w))

def _wB(args):
    r_w, L, seed = args
    return ('B', r_w, L, seed,
            run_one(L, PC_B_FIXED, seed, NSTEPS_B, _prob_A(L), r_w))

def _wC(args):
    r_w, pc, seed = args
    return ('C', r_w, pc, seed,
            run_one(L_C, pc, seed, NSTEPS_C, _prob_B(L_C, r_w), r_w))


# ── Aggregation ────────────────────────────────────────────────────────────────
def agg_group(d):
    vals = list(d.values())
    def _ms(k): return ms([v.get(k, float('nan')) for v in vals])
    def _ses(k): return ses([v.get(k, float('nan')) for v in vals])
    return dict(U4=_ms('U4'), U4_se=_ses('U4'),
                absM=_ms('absM'), absM_se=_ses('absM'))


# ── Threshold extraction ───────────────────────────────────────────────────────
def find_threshold(rw_list, u4_list, thresh=U4_ORDER_THRESH):
    """
    Find minimum r_w such that U4 >= thresh.
    If none: return None (no ordering at any r_w).
    If all ordered: return rw_list[0] (threshold below min tested).
    Uses linear interpolation between the first pair that crosses thresh.
    """
    ordered = [u >= thresh and not math.isnan(u) for u in u4_list]
    if not any(ordered):
        return None
    if all(ordered):
        return float(rw_list[0])
    # Find first transition
    for i in range(len(ordered) - 1):
        if not ordered[i] and ordered[i+1]:
            # Linear interpolation
            u0 = u4_list[i]; u1 = u4_list[i+1]
            r0 = rw_list[i]; r1 = rw_list[i+1]
            if u1 > u0:
                frac = (thresh - u0) / (u1 - u0)
                return r0 + frac * (r1 - r0)
    return None


# ── MAIN ───────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    mp.freeze_support()
    RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)

    raw = {}
    if RESULTS_FILE.exists():
        with open(RESULTS_FILE) as f:
            raw = json.load(f)

    ncpu = mp.cpu_count()

    # ── Phase A ───────────────────────────────────────────────────────────────
    if 'A' not in raw:
        args_A = [(r_w, pc, seed) for r_w in RW_A for pc in PC_A for seed in SEEDS_A]
        print(f'Phase A: {len(args_A)} runs -- L={L_A}, '
              f'r_w in {RW_A}, P in {PC_A}, {NSTEPS_A} steps ...')
        with mp.Pool(processes=min(len(args_A), ncpu)) as pool:
            res = pool.map(_wA, args_A)
        raw['A'] = {}
        for _, r_w, pc, seed, r in res:
            raw['A'].setdefault(f'rw{r_w}_pc{pc:.4f}', {})[str(seed)] = r
        with open(RESULTS_FILE, 'w') as f:
            json.dump(raw, f)
        print('Phase A done.')

    # ── Phase B ───────────────────────────────────────────────────────────────
    if 'B' not in raw:
        args_B = [(r_w, L, seed) for r_w in RW_B for L in L_B for seed in SEEDS_B]
        print(f'\nPhase B: {len(args_B)} runs -- P={PC_B_FIXED}, '
              f'r_w in {RW_B}, L in {L_B}, {NSTEPS_B} steps ...')
        with mp.Pool(processes=min(len(args_B), ncpu)) as pool:
            res = pool.map(_wB, args_B)
        raw['B'] = {}
        for _, r_w, L, seed, r in res:
            raw['B'].setdefault(f'rw{r_w}_L{L}', {})[str(seed)] = r
        with open(RESULTS_FILE, 'w') as f:
            json.dump(raw, f)
        print('Phase B done.')

    # ── Phase C ───────────────────────────────────────────────────────────────
    if 'C' not in raw:
        args_C = [(r_w, pc, seed) for r_w in RW_C for pc in PC_C for seed in SEEDS_C]
        print(f'\nPhase C: {len(args_C)} runs -- L={L_C}, '
              f'r_w in {RW_C}, P in {PC_C}, Protocol B, {NSTEPS_C} steps ...')
        with mp.Pool(processes=min(len(args_C), ncpu)) as pool:
            res = pool.map(_wC, args_C)
        raw['C'] = {}
        for _, r_w, pc, seed, r in res:
            raw['C'].setdefault(f'rw{r_w}_pc{pc:.4f}', {})[str(seed)] = r
        with open(RESULTS_FILE, 'w') as f:
            json.dump(raw, f)
        print('Phase C done.')

    # ── Aggregate ─────────────────────────────────────────────────────────────
    agg_A = {k: agg_group(v) for k, v in raw.get('A', {}).items()}
    agg_B = {k: agg_group(v) for k, v in raw.get('B', {}).items()}
    agg_C = {k: agg_group(v) for k, v in raw.get('C', {}).items()}

    # ── Phase A: print heat map ────────────────────────────────────────────────
    print(f'\n=== PHASE A: U4 heat map (L={L_A}, Protocol A) ===')
    print(f'{"r_w":>5}', end='')
    for pc in PC_A:
        print(f'  P={pc:.3f}', end='')
    print()
    print('-' * (7 + 10 * len(PC_A)))
    for r_w in RW_A:
        print(f'{r_w:>5}', end='')
        for pc in PC_A:
            k = f'rw{r_w}_pc{pc:.4f}'
            u4 = agg_A.get(k, {}).get('U4', float('nan'))
            marker = '*' if (not math.isnan(u4) and u4 >= U4_ORDER_THRESH) else ' '
            print(f'  {u4:>5.3f}{marker}', end='')
        print()
    print(f'  (* = U4 >= {U4_ORDER_THRESH}, ordered)')

    # Threshold curve r_w*(P)
    print(f'\nThreshold curve r_w*(P) at U4 = {U4_ORDER_THRESH}:')
    threshold_A = {}
    for pc in PC_A:
        rw_list = RW_A
        u4_list = [agg_A.get(f'rw{rw}_pc{pc:.4f}', {}).get('U4', float('nan'))
                   for rw in rw_list]
        rw_star = find_threshold(rw_list, u4_list)
        threshold_A[pc] = rw_star
        print(f'  P={pc:.4f}: r_w* = {f"{rw_star:.2f}" if rw_star else "None (never orders)"}')

    # Power-law fit r_w* ~ P^alpha
    pc_fit = [pc for pc in PC_A if threshold_A.get(pc) is not None]
    rw_fit = [threshold_A[pc] for pc in pc_fit]
    if len(pc_fit) >= 3:
        alpha_A, lc = np.polyfit(np.log(pc_fit), np.log(rw_fit), 1)
        resid = np.log(rw_fit) - np.polyval([alpha_A, lc], np.log(pc_fit))
        r2_A = 1 - np.var(resid) / np.var(np.log(rw_fit))
        print(f'\nPower-law fit r_w*(P) ~ P^alpha (Protocol A, L={L_A}):')
        print(f'  alpha = {alpha_A:.4f},  R^2 = {r2_A:.4f}')
        print(f'  Prediction: alpha = -0.500 (signal/noise theory)')
        print(f'  Match: {"YES" if abs(alpha_A + 0.5) < 0.1 else "NO"} '
              f'(|alpha + 0.5| = {abs(alpha_A + 0.5):.4f})')

    # ── Phase B: r_w*(L) scaling ───────────────────────────────────────────────
    print(f'\n=== PHASE B: r_w threshold vs L (P={PC_B_FIXED}, Protocol A) ===')
    print(f'{"L":>5}', end='')
    for r_w in RW_B:
        print(f'  rw={r_w}', end='')
    print()
    threshold_B = {}
    for L in L_B:
        print(f'{L:>5}', end='')
        rw_list = RW_B
        u4_list = [agg_B.get(f'rw{rw}_L{L}', {}).get('U4', float('nan'))
                   for rw in rw_list]
        for u4 in u4_list:
            marker = '*' if (not math.isnan(u4) and u4 >= U4_ORDER_THRESH) else ' '
            print(f'  {u4:.3f}{marker}', end='')
        rw_star = find_threshold(rw_list, u4_list)
        threshold_B[L] = rw_star
        print(f'  -> r_w*={f"{rw_star:.2f}" if rw_star else "None"}')

    # Power-law fit r_w*(L) ~ L^gamma
    L_fit  = [L for L in L_B if threshold_B.get(L) is not None]
    rw_fit_B = [threshold_B[L] for L in L_fit]
    if len(L_fit) >= 3:
        gamma_B, lc_B = np.polyfit(np.log(L_fit), np.log(rw_fit_B), 1)
        resid_B = np.log(rw_fit_B) - np.polyval([gamma_B, lc_B], np.log(L_fit))
        r2_B = 1 - np.var(resid_B) / np.var(np.log(rw_fit_B))
        print(f'\nPower-law fit r_w*(L) ~ L^gamma (P={PC_B_FIXED}, Protocol A):')
        print(f'  gamma = {gamma_B:.4f},  R^2 = {r2_B:.4f}')
        print(f'  Predictions: gamma=0 (extensive drive), gamma=0.5 (Protocol A theory)')
        if abs(gamma_B) < 0.1:
            print('  -> GAMMA ~ 0: threshold is L-INDEPENDENT. Wave operator RELEVANT at any r_w > r_w*(P).')
            print('     d_wave = 0. Ordering is a true thermodynamic phase.')
        elif abs(gamma_B - 0.5) < 0.1:
            print('  -> GAMMA ~ 0.5: threshold grows as sqrt(L). Wave operator MARGINAL.')
            print('     d_wave = 1/2. Ordering disappears in thermodynamic limit for any fixed r_w.')
        elif abs(gamma_B - 1.0) < 0.15:
            print('  -> GAMMA ~ 1: threshold grows as L. Ordering requires r_w/L > const.')
            print('     d_wave = 1. r_w/L is the scaling variable; non-local operator is marginal.')
        else:
            print(f'  -> INTERMEDIATE: novel scaling gamma={gamma_B:.3f}.')

    # ── Phase C: Protocol B threshold ─────────────────────────────────────────
    print(f'\n=== PHASE C: r_w*(P) under Protocol B (L={L_C}, matched coverage) ===')
    threshold_C = {}
    for pc in PC_C:
        rw_list = RW_C
        u4_list = [agg_C.get(f'rw{rw}_pc{pc:.4f}', {}).get('U4', float('nan'))
                   for rw in rw_list]
        rw_star = find_threshold(rw_list, u4_list)
        threshold_C[pc] = rw_star
        u4_str = ', '.join(f'{u:.3f}' for u in u4_list)
        print(f'  P={pc:.4f}: [{u4_str}] -> r_w*={f"{rw_star:.2f}" if rw_star else "None"}')

    pc_fit_C  = [pc for pc in PC_C if threshold_C.get(pc) is not None]
    rw_fit_C  = [threshold_C[pc] for pc in pc_fit_C]
    if len(pc_fit_C) >= 3:
        alpha_C, lc_C = np.polyfit(np.log(pc_fit_C), np.log(rw_fit_C), 1)
        resid_C = np.log(rw_fit_C) - np.polyval([alpha_C, lc_C], np.log(pc_fit_C))
        r2_C = 1 - np.var(resid_C) / np.var(np.log(rw_fit_C))
        print(f'\nPower-law fit r_w*(P) ~ P^alpha (Protocol B, L={L_C}):')
        print(f'  alpha = {alpha_C:.4f},  R^2 = {r2_C:.4f}')
        print(f'  Prediction: alpha = -0.500 (signal/noise theory, L-independent)')
        print(f'  Protocol A fit gave: alpha_A = {alpha_A:.4f}')
        print(f'  Same exponent? {"YES" if abs(alpha_C - alpha_A) < 0.1 else "NO"} '
              f'(difference = {abs(alpha_C - alpha_A):.4f})')

    analysis = dict(A=agg_A, B=agg_B, C=agg_C,
                    threshold_A=threshold_A, threshold_B=threshold_B,
                    threshold_C=threshold_C,
                    fits=dict(alpha_A=float(alpha_A) if 'alpha_A' in dir() else None,
                              gamma_B=float(gamma_B) if 'gamma_B' in dir() else None,
                              alpha_C=float(alpha_C) if 'alpha_C' in dir() else None))
    with open(ANALYSIS_FILE, 'w') as f:
        json.dump(analysis, f, indent=2)
    print(f'\nSaved -> {ANALYSIS_FILE}')
