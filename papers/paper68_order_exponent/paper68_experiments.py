"""
Paper 68: Order Parameter Exponent at p_c = 0+.
Fine P scan near 0 — power-law vs essential singularity.

Background:
  Paper 67 confirmed p_c = 0+ with corrected (probabilistic) wave firing.
  U4(L=160) > U4(L=120) for P >= 0.010; P=0.005 ambiguous.
  Paper 63 exponents (FSS): beta/nu = 0.668, nu = 0.97, beta = 0.65.
  Manna/stochastic-sandpile class in 2D: beta ~ 0.639, nu ~ 0.799.
  The beta match is striking. nu discrepancy real (quenched disorder / conserved-mode?).

Goal: discriminate the functional form of |M(P)| as P -> 0+.
  H0 (power law):           |M| ~ P^beta   => log|M| vs log(P) linear, slope = beta
  H1 (BKT essential sing.): |M| ~ exp(-A/sqrt(P)) => log|M| vs 1/sqrt(P) linear
  H2 (ordinary ess. sing.): |M| ~ exp(-A/P)        => log|M| vs 1/P linear

  If H0 holds with beta = 0.639 +/- 0.01: strong evidence Manna universality class.
  If H0 holds with beta != 0.639: new universality class.
  If H1: BKT-like infinite-order transition (consistent with Papers 60-63 framing).

Conservation-law diagnostic (GPT comment):
  Manna class requires an approximately conserved activity. In VCSM, phi decays by
  FIELD_DECAY per step. True conservation would mean consolidation flux = decay flux.
  Phase C measures <|delta_phi|_create> and <|delta_phi|_decay> per step in steady
  state to test whether they approximately balance (emergent conservation).

Phase A: Fine P scan at fixed L=160 (probabilistic wave firing, Paper 67 protocol).
  P in {0.001, 0.002, 0.003, 0.005, 0.007, 0.010, 0.015, 0.020, 0.030, 0.050}
  6 seeds, 10k steps each. Measures: absM, U4, chi.

Phase B: FSS at selected P values to extract beta and nu.
  L in {80, 100, 120, 160}, P in {0.003, 0.005, 0.007, 0.010, 0.020}
  4 seeds, 10k steps.

Phase C: Conservation-law flux balance (Manna diagnostic).
  L=120, P=0.010, 4 seeds, 10k steps.
  Track per-step: flux_create (consolidation writes to phi),
                  flux_decay  (FIELD_DECAY removes from phi).
  If flux_create/flux_decay -> 1 in steady state: emergent conservation holds.
"""

import numpy as np, json, math, multiprocessing as mp
from pathlib import Path

# ── Physical parameters (identical to Papers 64-67) ──────────────────────────
J           = 1.0
MID_DECAY   = 0.97
BETA_BASE   = 0.005
SS          = 8
WAVE_RADIUS = 5
WAVE_DUR    = 5
EXT_FIELD   = 1.5
T_FIXED     = 3.0
FIELD_DECAY = 0.999
SS_FRAC     = 0.40
FA          = 0.30
WE_BASE     = 25
L_BASE      = 40

# ── Phase A parameters ────────────────────────────────────────────────────────
L_A       = 160
PC_A      = [round(x, 4) for x in
             [0.001, 0.002, 0.003, 0.005, 0.007, 0.010, 0.015, 0.020, 0.030, 0.050]]
SEEDS_A   = list(range(6))
NSTEPS_A  = 10000

# ── Phase B parameters ────────────────────────────────────────────────────────
L_B       = [80, 100, 120, 160]
PC_B      = [round(x, 4) for x in [0.003, 0.005, 0.007, 0.010, 0.020]]
SEEDS_B   = list(range(4))
NSTEPS_B  = 10000

# ── Phase C parameters (Manna flux balance) ───────────────────────────────────
L_C       = 120
PC_C      = 0.010
SEEDS_C   = list(range(4))
NSTEPS_C  = 10000

RESULTS_FILE  = Path(__file__).parent / 'results' / 'paper68_results.json'
ANALYSIS_FILE = Path(__file__).parent / 'results' / 'paper68_analysis.json'


# ── Wave density helpers ──────────────────────────────────────────────────────
def _wef(L):
    """Exact (non-integer) wave_every for size L."""
    return WE_BASE * (L_BASE / L) ** 2

def _prob(L):
    """Fire probability per step (probabilistic mode)."""
    return min(1.0, 1.0 / _wef(L))


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


# ── Statistics ─────────────────────────────────────────────────────────────────
def ms(vals):
    v = [x for x in vals if x is not None and not math.isnan(x)]
    return float(np.mean(v)) if v else float('nan')

def ses(vals):
    v = [x for x in vals if x is not None and not math.isnan(x)]
    return float(np.std(v) / math.sqrt(len(v))) if len(v) > 1 else 0.


# ── Core simulation ────────────────────────────────────────────────────────────
def run_one(L, P_causal, seed, nsteps, wave_prob, track_flux=False):
    """
    Single simulation run with probabilistic wave firing.
    If track_flux=True: also return phi creation/decay flux for Manna diagnostic.
    """
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

    # Flux tracking for Phase C
    flux_create_series = [] if track_flux else None
    flux_decay_series  = [] if track_flux else None

    for t in range(nsteps):
        s = metro(s, nb, T_FIXED, rng)
        base  += BETA_BASE * (s - base)
        same   = (s > 0) == (base > 0)
        streak = np.where(same, streak + 1, 0)
        mid   *= MID_DECAY

        if rng.random() < wave_prob:
            cx = rng.randint(0, L // 2) if wave_z == 0 else rng.randint(L // 2, L)
            cy = rng.randint(L)
            hit = wsites(cx, cy, WAVE_RADIUS, L)
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

        # Consolidation gate: measure flux before applying
        gate = streak >= SS
        fc = 0.0
        if gate.any():
            wi = np.where(gate)[0]
            delta = FA * (mid[wi] - phi[wi])
            phi[wi] += delta
            if track_flux and t >= ss0:
                fc = float(np.sum(np.abs(delta)))
            streak[wi] = 0

        # Decay: measure flux
        fd = 0.0
        if track_flux and t >= ss0:
            fd = float(np.sum(np.abs(phi) * (1.0 - FIELD_DECAY)))
        phi *= FIELD_DECAY

        if t >= ss0:
            M = float(np.mean(phi[zone == 0])) - float(np.mean(phi[zone == 1]))
            Mseries.append(M)
            if track_flux:
                flux_create_series.append(fc)
                flux_decay_series.append(fd)

    M_arr = np.array(Mseries)
    M2    = float(np.mean(M_arr**2)); M4 = float(np.mean(M_arr**4))
    absM  = float(np.mean(np.abs(M_arr)))
    U4    = float(1. - M4 / (3. * M2**2)) if M2 > 1e-12 else float('nan')
    chi   = float(N_zone * np.var(M_arr))

    result = dict(absM=absM, M2=M2, M4=M4, U4=U4, chi=chi)
    if track_flux:
        fc_arr = np.array(flux_create_series)
        fd_arr = np.array(flux_decay_series)
        result['flux_create'] = float(np.mean(fc_arr))
        result['flux_decay']  = float(np.mean(fd_arr))
        result['flux_ratio']  = (float(np.mean(fc_arr) / np.mean(fd_arr))
                                  if np.mean(fd_arr) > 1e-12 else float('nan'))
    return result


# ── Module-level workers (required for Windows multiprocessing spawn) ─────────
def _wA(args):
    pc, seed = args
    return ('A', L_A, pc, seed,
            run_one(L_A, pc, seed, NSTEPS_A, wave_prob=_prob(L_A)))

def _wB(args):
    L, pc, seed = args
    return ('B', L, pc, seed,
            run_one(L, pc, seed, NSTEPS_B, wave_prob=_prob(L)))

def _wC(args):
    seed = args[0]
    return ('C', seed,
            run_one(L_C, PC_C, seed, NSTEPS_C, wave_prob=_prob(L_C),
                    track_flux=True))


# ── Aggregation ────────────────────────────────────────────────────────────────
def agg_group(d_by_seed):
    vals = list(d_by_seed.values())
    def _ms(key):  return ms([v.get(key, float('nan')) for v in vals])
    def _ses(key): return ses([v.get(key, float('nan')) for v in vals])
    out = dict(absM=_ms('absM'), absM_se=_ses('absM'),
               U4=_ms('U4'), U4_se=_ses('U4'),
               chi=_ms('chi'))
    # Flux keys if present
    if 'flux_ratio' in vals[0]:
        out['flux_create'] = _ms('flux_create')
        out['flux_decay']  = _ms('flux_decay')
        out['flux_ratio']  = _ms('flux_ratio')
    return out


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
        args_A = [(pc, seed) for pc in PC_A for seed in SEEDS_A]
        print(f'Phase A: {len(args_A)} runs -- L={L_A}, '
              f'p_wave={_prob(L_A):.3f}, P in {PC_A}, {NSTEPS_A} steps ...')
        with mp.Pool(processes=min(len(args_A), ncpu)) as pool:
            res = pool.map(_wA, args_A)
        raw['A'] = {}
        for _, L, pc, seed, r in res:
            k = f'pc{pc:.4f}'
            raw['A'].setdefault(k, {})[str(seed)] = r
        with open(RESULTS_FILE, 'w') as f:
            json.dump(raw, f)
        print('Phase A done.')

    # ── Phase B ───────────────────────────────────────────────────────────────
    if 'B' not in raw:
        args_B = [(L, pc, seed) for L in L_B for pc in PC_B for seed in SEEDS_B]
        print(f'\nPhase B: {len(args_B)} runs -- L in {L_B}, '
              f'P in {PC_B}, {NSTEPS_B} steps ...')
        with mp.Pool(processes=min(len(args_B), ncpu)) as pool:
            res = pool.map(_wB, args_B)
        raw['B'] = {}
        for _, L, pc, seed, r in res:
            k = f'L{L}_pc{pc:.4f}'
            raw['B'].setdefault(k, {})[str(seed)] = r
        with open(RESULTS_FILE, 'w') as f:
            json.dump(raw, f)
        print('Phase B done.')

    # ── Phase C ───────────────────────────────────────────────────────────────
    if 'C' not in raw:
        args_C = [(seed,) for seed in SEEDS_C]
        print(f'\nPhase C (Manna flux balance): {len(args_C)} runs -- '
              f'L={L_C}, P={PC_C}, {NSTEPS_C} steps ...')
        with mp.Pool(processes=min(len(args_C), ncpu)) as pool:
            res = pool.map(_wC, args_C)
        raw['C'] = {}
        for _, seed, r in res:
            raw['C'][str(seed)] = r
        with open(RESULTS_FILE, 'w') as f:
            json.dump(raw, f)
        print('Phase C done.')

    # ── Aggregate ─────────────────────────────────────────────────────────────
    agg_A = {}
    for pc in PC_A:
        k = f'pc{pc:.4f}'
        if k in raw.get('A', {}):
            agg_A[k] = dict(pc=pc, **agg_group(raw['A'][k]))

    agg_B = {}
    for L in L_B:
        for pc in PC_B:
            k = f'L{L}_pc{pc:.4f}'
            if k in raw.get('B', {}):
                agg_B[k] = dict(L=L, pc=pc, **agg_group(raw['B'][k]))

    agg_C = agg_group(raw.get('C', {})) if 'C' in raw else {}

    # ── Functional form test ──────────────────────────────────────────────────
    print('\n=== PHASE A: Fine P scan at L=160 ===')
    print(f'{"P":>8}  {"absM":>8}  {"se":>7}  {"U4":>7}  '
          f'{"log|M|":>8}  {"1/sqrt(P)":>10}')
    print('-' * 60)
    for pc in PC_A:
        k = f'pc{pc:.4f}'
        if k not in agg_A:
            print(f'{pc:>8.4f}  MISSING')
            continue
        v = agg_A[k]
        aM = v['absM']; se = v['absM_se']
        u4 = v['U4']
        logM = math.log(aM) if aM > 1e-9 else float('nan')
        inv_sqrtP = 1.0 / math.sqrt(pc) if pc > 0 else float('nan')
        print(f'{pc:>8.4f}  {aM:>8.5f}  {se:>7.5f}  {u4:>7.4f}  '
              f'{logM:>8.3f}  {inv_sqrtP:>10.2f}')

    # Power-law fit: log|M| = beta*log(P) + c
    pc_vals = [agg_A[f'pc{pc:.4f}']['pc']   for pc in PC_A
               if f'pc{pc:.4f}' in agg_A and agg_A[f'pc{pc:.4f}']['absM'] > 1e-9]
    aM_vals = [agg_A[f'pc{pc:.4f}']['absM'] for pc in PC_A
               if f'pc{pc:.4f}' in agg_A and agg_A[f'pc{pc:.4f}']['absM'] > 1e-9]
    if len(pc_vals) >= 3:
        log_pc = np.log(pc_vals); log_aM = np.log(aM_vals)
        beta_fit, log_c = np.polyfit(log_pc, log_aM, 1)
        resid_pl = log_aM - np.polyval([beta_fit, log_c], log_pc)
        r2_pl = 1 - np.var(resid_pl) / np.var(log_aM)
        print(f'\nPower-law fit: |M| ~ P^beta')
        print(f'  beta = {beta_fit:.4f},  R^2 = {r2_pl:.4f}')
        print(f'  Manna prediction: beta=0.639  Paper63 measurement: beta=0.650')
        print(f'  Match: {"YES" if abs(beta_fit - 0.639) < 0.05 else "NO"} '
              f'(|beta_fit - 0.639| = {abs(beta_fit - 0.639):.4f})')

        # BKT essential singularity fit: log|M| = -A/sqrt(P) + c
        inv_sqrtP_vals = [1.0/math.sqrt(pc) for pc in pc_vals]
        A_bkt, c_bkt = np.polyfit(inv_sqrtP_vals, log_aM, 1)
        resid_bkt = log_aM - np.polyval([A_bkt, c_bkt], inv_sqrtP_vals)
        r2_bkt = 1 - np.var(resid_bkt) / np.var(log_aM)
        print(f'\nBKT essential singularity fit: |M| ~ exp(-A/sqrt(P))')
        print(f'  A = {-A_bkt:.4f},  R^2 = {r2_bkt:.4f}')

        # Ordinary essential singularity: log|M| = -A/P + c
        inv_P_vals = [1.0/pc for pc in pc_vals]
        A_ord, c_ord = np.polyfit(inv_P_vals, log_aM, 1)
        resid_ord = log_aM - np.polyval([A_ord, c_ord], inv_P_vals)
        r2_ord = 1 - np.var(resid_ord) / np.var(log_aM)
        print(f'\nOrdinary essential singularity fit: |M| ~ exp(-A/P)')
        print(f'  A = {-A_ord:.4f},  R^2 = {r2_ord:.4f}')

        print(f'\nBest fit: ', end='')
        fits = [('Power law', r2_pl, f'beta={beta_fit:.4f}'),
                ('BKT', r2_bkt, f'A={-A_bkt:.4f}'),
                ('Exp', r2_ord, f'A={-A_ord:.4f}')]
        fits.sort(key=lambda x: -x[1])
        for name, r2, params in fits:
            print(f'{name} (R^2={r2:.4f}, {params})', end='  ')
        print()

    # ── Phase B: beta and nu from FSS ────────────────────────────────────────
    print('\n=== PHASE B: FSS at fine P values ===')
    print(f'{"P":>7}', end='')
    for L in L_B:
        print(f'  absM(L={L})', end='')
    print()
    print('-' * 70)
    for pc in PC_B:
        print(f'{pc:>7.4f}', end='')
        for L in L_B:
            k = f'L{L}_pc{pc:.4f}'
            v = agg_B.get(k, {})
            aM = v.get('absM', float('nan'))
            print(f'  {aM:>10.5f}', end='')
        print()

    # Extract beta/nu from FSS: |M| ~ L^{-beta/nu} at fixed P (near p_c=0+)
    # Use largest P where U4 is well-ordered as the "critical" point
    print(f'\nbeta/nu from log-log |M| vs L:')
    for pc in PC_B:
        L_vals = []; aM_vals_fss = []
        for L in L_B:
            k = f'L{L}_pc{pc:.4f}'
            if k in agg_B:
                v = agg_B[k]
                if not math.isnan(v.get('absM', float('nan'))):
                    L_vals.append(L); aM_vals_fss.append(v['absM'])
        if len(L_vals) >= 3:
            slope, _ = np.polyfit(np.log(L_vals), np.log(aM_vals_fss), 1)
            print(f'  P={pc:.4f}: d(log|M|)/d(logL) = {slope:.4f} '
                  f'[expect -beta/nu = {-0.65/0.97:.4f} at p_c; '
                  f'positive = away from p_c]')

    # ── Phase C: Manna flux balance ───────────────────────────────────────────
    print(f'\n=== PHASE C: Manna Conservation-Law Diagnostic (L={L_C}, P={PC_C}) ===')
    if agg_C:
        fc = agg_C.get('flux_create', float('nan'))
        fd = agg_C.get('flux_decay', float('nan'))
        fr = agg_C.get('flux_ratio', float('nan'))
        print(f'Mean creation flux per step: {fc:.6f}')
        print(f'Mean decay flux per step:    {fd:.6f}')
        print(f'Ratio create/decay:          {fr:.4f}')
        if not math.isnan(fr):
            if abs(fr - 1.0) < 0.05:
                print('RESULT: Fluxes balance to within 5% -- emergent conservation holds.')
                print('        Supports Manna-class universality (conservation approximate).')
            elif fr < 0.95:
                print('RESULT: Creation < decay -- net phi loss. NOT conservative.')
                print('        System lives below conservation threshold.')
                print('        Manna mapping may require rescaling.')
            else:
                print('RESULT: Creation > decay -- net phi gain. Consolidation-dominant.')
    else:
        print('Phase C data missing.')

    analysis = dict(A=agg_A, B=agg_B, C=agg_C)
    with open(ANALYSIS_FILE, 'w') as f:
        json.dump(analysis, f, indent=2)
    print(f'\nSaved -> {ANALYSIS_FILE}')
