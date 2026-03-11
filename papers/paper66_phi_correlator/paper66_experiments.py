"""
Paper 66: Discriminating Test for p_c = 0+ and Direct phi-phi Spatial Correlator.

Three phases:

Phase A: L=160 definitive Binder test.
  Paper 65 Phase A2 found U4(L=160) < U4(L=120) -- reversal of expected ordering.
  Two hypotheses:
    (i)  Finite p_c: p_c is not 0+ but some positive threshold (e.g. 0.005--0.010).
    (ii) Equilibration artefact: WAVE_EVERY rounds to 2 for L=160 (target 1.56),
         giving ~35% fewer waves per site; 10000 steps insufficient to equilibrate.
  THIS PAPER: WAVE_EVERY=1 (forced, not from scaling formula), 50000 steps (5x A2).
  L in {120, 160}, P in {0, 0.001, 0.002, 0.005, 0.010, 0.020, 0.050}, 8 seeds.
  If U4(L=160) > U4(L=120) for all P > 0: p_c = 0+ confirmed (artefact hypothesis).
  If reversal persists: finite p_c; estimate critical P from L-ordering reversal.

Phase B: phi-field spatial correlator at boosted FA.
  Paper 64 attempted phi-phi correlator but phi amplitude ~1e-3 -- too small.
  Paper 65 used spin field -- P-independent (Ising paramagnetic, wrong proxy).
  THIS PAPER: FA=0.50 (boosted from 0.30), measure G_phi(r) = <phi(y)*phi(y+r)>_c
  within zone-0 columns via FFT autocorrelation.
  L in {60, 80, 100}, P in {0, 0.010, 0.020, 0.050}, 10 seeds, 20000 steps.
  Fit to power law r^{-eta} and exponential exp(-r/xi), compare R^2.
  Key question: does P_causal > 0 extend phi correlations beyond wave radius (xi~5)?
  If power law wins with eta varying continuously with P: BKT algebraic quasi-LRO.

Phase C: FA amplitude sweep.
  FA in {0.10, 0.20, 0.30, 0.50, 0.80} at L=80, P=0.020, 12 seeds, 15000 steps.
  Characterises phi amplitude and G_phi(r) quality as function of FA.
  Determines minimum FA for reliable phi-phi correlator measurement.
  Also measures U4(FA) to check whether FA itself modulates ordering strength.
"""

import numpy as np, json, math, multiprocessing as mp
from pathlib import Path

# ── Physical parameters (shared) ─────────────────────────────────────────────
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

# ── Phase A: L=160 discriminating test ───────────────────────────────────────
# Same step count as Paper 65 A2 (10k) but with more seeds for clearer statistics.
# Paper 65 A2 used 8 seeds and found reversal at P<=0.005.
# Here 6 seeds + added L=100 for monotonicity check.
# Single run at L=160: ~80 seconds. 32 runs / 12 cores -> ~4 min.
FA_A      = 0.30
L_A       = [100, 120, 160]
PC_A      = [round(x, 4) for x in [0.000, 0.005, 0.010, 0.020]]
SEEDS_A   = list(range(4))
NSTEPS_A  = 10000
WAVE_EV_A = None       # None = use standard scaling formula

# ── Phase B: phi-phi correlator at boosted FA ─────────────────────────────────
# Single run at L=80, 8k steps: ~16 seconds. 32 runs / 12 cores -> ~1 min.
FA_B      = 0.50       # boosted from 0.30
L_B       = [60, 80]
PC_B      = [0.000, 0.010, 0.020, 0.050]
SEEDS_B   = list(range(4))
NSTEPS_B  = 8000
WAVE_EV_B = {60: 11, 80: 6}

# ── Phase C: FA amplitude sweep ───────────────────────────────────────────────
L_C       = 80
PC_C      = 0.020
SEEDS_C   = list(range(4))
NSTEPS_C  = 8000
WAVE_EV_C = 6
FA_SWEEP  = [0.10, 0.30, 0.50, 0.80]

RESULTS_FILE  = Path(__file__).parent / 'results' / 'paper66_results.json'
ANALYSIS_FILE = Path(__file__).parent / 'results' / 'paper66_analysis.json'


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


# ── Statistics ────────────────────────────────────────────────────────────────
def ms(vals):
    v = [x for x in vals if x is not None and not math.isnan(x)]
    return float(np.mean(v)) if v else float('nan')

def ses(vals):
    v = [x for x in vals if x is not None and not math.isnan(x)]
    return float(np.std(v) / math.sqrt(len(v))) if len(v) > 1 else 0.


# ── Core simulation ───────────────────────────────────────────────────────────
def run_one(L, P_causal, FA, seed, nsteps, wave_every):
    """
    Single simulation run.
    Returns Binder cumulant, phi amplitude, and phi-phi spatial correlator stats.
    wave_every: fire one wave event every this many steps (WAVE_EV_A=1 for Phase A).
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

    # phi autocorrelator accumulator (along y, within zone 0 columns)
    PhiC_acc    = np.zeros(L, float)
    PhiC_cnt    = 0
    phi_amp_acc = 0.
    phi_amp_cnt = 0

    for t in range(nsteps):
        s = metro(s, nb, T_FIXED, rng)
        base  += BETA_BASE * (s - base)
        same   = (s > 0) == (base > 0)
        streak = np.where(same, streak + 1, 0)
        mid   *= MID_DECAY

        if t % wave_every == 0:
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

        gate = streak >= SS
        if gate.any():
            wi = np.where(gate)[0]
            phi[wi] += FA * (mid[wi] - phi[wi])
            streak[wi] = 0
        phi *= FIELD_DECAY

        if t >= ss0:
            M = float(np.mean(phi[zone == 0])) - float(np.mean(phi[zone == 1]))
            Mseries.append(M)

            # accumulate phi spatial autocorrelator every 50 steps
            if (t - ss0) % 50 == 0:
                phi_2d = phi.reshape(L, L)
                phi_z0 = phi_2d[:, :L // 2].astype(float)
                phi_amp_acc += float(np.mean(np.abs(phi_z0)))
                phi_amp_cnt += 1
                # FFT autocorrelation along y for each zone-0 column
                for x in range(L // 2):
                    col   = phi_z0[:, x]
                    col_d = col - col.mean()
                    f     = np.fft.rfft(col_d, n=L)
                    acorr = np.fft.irfft(f * np.conj(f), n=L).real
                    PhiC_acc += acorr
                PhiC_cnt += L // 2

    # ── Binder cumulant ───────────────────────────────────────────────────────
    M_arr = np.array(Mseries)
    M2    = float(np.mean(M_arr**2)); M4 = float(np.mean(M_arr**4))
    absM  = float(np.mean(np.abs(M_arr)))
    U4    = float(1. - M4 / (3. * M2**2)) if M2 > 1e-12 else float('nan')
    chi   = float(N_zone * np.var(M_arr))

    # ── phi amplitude ─────────────────────────────────────────────────────────
    phi_amp = phi_amp_acc / phi_amp_cnt if phi_amp_cnt > 0 else float('nan')

    # ── phi-phi spatial autocorrelator ────────────────────────────────────────
    eta = xi = r2_pow = r2_exp = float('nan')
    G_phi_norm = []

    if PhiC_cnt > 0:
        C_avg  = PhiC_acc / (PhiC_cnt * L)
        C0     = C_avg[0]
        r_max  = L // 2
        if C0 > 1e-12:
            C_norm = C_avg[:r_max] / C0
            r_arr  = np.arange(1, r_max)
            C_fit  = np.maximum(C_norm[1:], 1e-12)
            mask   = C_fit > 0.02   # fit where signal > 2% of peak
            if mask.sum() >= 4:
                logr   = np.log(r_arr[mask])
                logC   = np.log(C_fit[mask])
                ss_tot = float(((logC - logC.mean())**2).sum())
                if ss_tot > 1e-12:
                    try:
                        pp     = np.polyfit(logr, logC, 1)
                        eta    = float(-pp[0])
                        r2_pow = float(1. - ((logC - np.polyval(pp, logr))**2).sum() / ss_tot)
                    except Exception:
                        pass
                    try:
                        pe     = np.polyfit(r_arr[mask], logC, 1)
                        xi     = float(-1. / pe[0]) if pe[0] < -1e-6 else float(r_max)
                        r2_exp = float(1. - ((logC - np.polyval(pe, r_arr[mask]))**2).sum() / ss_tot)
                    except Exception:
                        pass
            G_phi_norm = C_norm[:20].tolist()

    return dict(absM=absM, M2=M2, M4=M4, U4=U4, chi=chi,
                phi_amp=phi_amp, eta=eta, xi=xi,
                r2_pow=r2_pow, r2_exp=r2_exp,
                G_phi_norm=G_phi_norm)


# ── Module-level workers (required for Windows multiprocessing spawn) ─────────
def _we(L):
    """Standard extensive-drive WAVE_EVERY scaling."""
    L_BASE = 40; WE_BASE = 25
    return max(1, round(WE_BASE * (L_BASE / L) ** 2))

def _wA(args):
    L, pc, seed = args
    we = _we(L) if WAVE_EV_A is None else WAVE_EV_A
    return ('A', L, pc, seed, run_one(L, pc, FA_A, seed, NSTEPS_A, we))

def _wB(args):
    L, pc, seed = args
    return ('B', L, pc, seed, run_one(L, pc, FA_B, seed, NSTEPS_B, WAVE_EV_B[L]))

def _wC(args):
    fa, seed = args
    return ('C', fa, seed, run_one(L_C, PC_C, fa, seed, NSTEPS_C, WAVE_EV_C))


# ── Aggregation ───────────────────────────────────────────────────────────────
def agg_group(d_by_seed):
    vals = list(d_by_seed.values())
    def _ms(key):
        return ms([v.get(key, float('nan')) for v in vals])
    def _ses(key):
        return ses([v.get(key, float('nan')) for v in vals])
    glist = [v['G_phi_norm'] for v in vals if v.get('G_phi_norm')]
    g_rep = glist[0] if glist else []
    return dict(absM=_ms('absM'), U4=_ms('U4'), U4_se=_ses('U4'),
                chi=_ms('chi'), phi_amp=_ms('phi_amp'),
                eta=_ms('eta'), xi=_ms('xi'),
                r2_pow=_ms('r2_pow'), r2_exp=_ms('r2_exp'),
                G_phi_norm=g_rep)


# ── MAIN ──────────────────────────────────────────────────────────────────────
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
        args_A = [(L, pc, seed) for L in L_A for pc in PC_A for seed in SEEDS_A]
        we_info = {L: (_we(L) if WAVE_EV_A is None else WAVE_EV_A) for L in L_A}
        print(f'Phase A: {len(args_A)} runs '
              f'-- L in {L_A}, WAVE_EVERY={we_info}, {NSTEPS_A} steps ...')
        with mp.Pool(processes=min(len(args_A), ncpu)) as pool:
            res = pool.map(_wA, args_A)
        raw['A'] = {}
        for _, L, pc, seed, r in res:
            k = f'L{L}_pc{pc:.4f}'
            raw['A'].setdefault(k, {})[str(seed)] = r
        with open(RESULTS_FILE, 'w') as f:
            json.dump(raw, f)
        print('Phase A done.')

    # ── Phase B ───────────────────────────────────────────────────────────────
    if 'B' not in raw:
        args_B = [(L, pc, seed) for L in L_B for pc in PC_B for seed in SEEDS_B]
        print(f'\nPhase B: {len(args_B)} runs '
              f'-- phi correlator, FA={FA_B}, {NSTEPS_B} steps ...')
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
        args_C = [(fa, seed) for fa in FA_SWEEP for seed in SEEDS_C]
        print(f'\nPhase C: {len(args_C)} runs '
              f'-- FA sweep {FA_SWEEP}, L={L_C}, P={PC_C} ...')
        with mp.Pool(processes=min(len(args_C), ncpu)) as pool:
            res = pool.map(_wC, args_C)
        raw['C'] = {}
        for _, fa, seed, r in res:
            k = f'fa{fa:.2f}'
            raw['C'].setdefault(k, {})[str(seed)] = r
        with open(RESULTS_FILE, 'w') as f:
            json.dump(raw, f)
        print('Phase C done.')

    # ── Aggregate ─────────────────────────────────────────────────────────────
    agg_A = {}
    for L in L_A:
        for pc in PC_A:
            k = f'L{L}_pc{pc:.4f}'
            if k in raw['A']:
                agg_A[k] = dict(L=L, pc=pc, **agg_group(raw['A'][k]))

    agg_B = {}
    for L in L_B:
        for pc in PC_B:
            k = f'L{L}_pc{pc:.4f}'
            if k in raw['B']:
                agg_B[k] = dict(L=L, pc=pc, **agg_group(raw['B'][k]))

    agg_C = {}
    for fa in FA_SWEEP:
        k = f'fa{fa:.2f}'
        if k in raw['C']:
            agg_C[k] = dict(FA=fa, **agg_group(raw['C'][k]))

    # ── Print Phase A ─────────────────────────────────────────────────────────
    print('\n=== PHASE A: L=160 Discriminating Binder Test '
          f'(standard WAVE_EVERY scaling, {NSTEPS_A} steps) ===')
    print(f'{"P":>8}', end='')
    for L in L_A:
        print(f'  U4(L={L})', end='')
    print('  ordering vs L=160')
    print('-' * 65)
    for pc in PC_A:
        print(f'{pc:>8.4f}', end='')
        u_vals = {}
        for L in L_A:
            k = f'L{L}_pc{pc:.4f}'
            u = agg_A[k]['U4'] if k in agg_A else float('nan')
            u_vals[L] = u
            print(f'  {u:>9.4f}', end='')
        # Check L ordering
        if pc == 0.000:
            print('  baseline')
        else:
            u120 = u_vals.get(120, float('nan'))
            u160 = u_vals.get(160, float('nan'))
            if math.isnan(u120) or math.isnan(u160):
                print('  MISSING')
            elif u160 > u120 + 0.01:
                print('  ORDERED -> p_c=0+')
            elif u160 < u120 - 0.01:
                print('  REVERSED -> finite p_c?')
            else:
                print('  AMBIGUOUS')

    print()
    n_ordered = sum(
        1 for pc in PC_A if pc > 0
        and f'L120_pc{pc:.4f}' in agg_A and f'L160_pc{pc:.4f}' in agg_A
        and agg_A[f'L160_pc{pc:.4f}']['U4'] > agg_A[f'L120_pc{pc:.4f}']['U4'] + 0.01
    )
    n_total = sum(1 for pc in PC_A if pc > 0)
    print(f'U4(160) > U4(120) for {n_ordered}/{n_total} P > 0 values.')
    if n_ordered == n_total:
        print('CONCLUSION: p_c = 0+ CONFIRMED (equilibration was the issue).')
    elif n_ordered == 0:
        print('CONCLUSION: Reversal persists -- finite p_c likely.')
    else:
        print('CONCLUSION: Mixed -- partial ordering, p_c estimate from crossover.')

    # ── Print Phase B ─────────────────────────────────────────────────────────
    print(f'\n=== PHASE B: phi-phi Correlator (FA={FA_B}) ===')
    print(f'{"L":>5} {"P":>7} {"phi_amp":>9} {"xi":>7} {"eta":>7} '
          f'{"R2_pow":>8} {"R2_exp":>8}  winner')
    print('-' * 68)
    for L in L_B:
        for pc in PC_B:
            k = f'L{L}_pc{pc:.4f}'
            if k not in agg_B:
                continue
            v = agg_B[k]
            rp = v['r2_pow']; re = v['r2_exp']
            if math.isnan(rp) or math.isnan(re):
                winner = 'UNDETERMINED'
            elif rp > re + 0.02:
                winner = 'POWER-LAW'
            elif re > rp + 0.02:
                winner = 'EXPONENTIAL'
            else:
                winner = 'AMBIGUOUS'
            print(f'{L:>5} {pc:>7.4f} {v["phi_amp"]:>9.5f} '
                  f'{v["xi"]:>7.2f} {v["eta"]:>7.3f} '
                  f'{rp:>8.4f} {re:>8.4f}  {winner}')

    # ── Print Phase C ─────────────────────────────────────────────────────────
    print(f'\n=== PHASE C: FA Amplitude Sweep (L={L_C}, P={PC_C}) ===')
    print(f'{"FA":>6} {"phi_amp":>9} {"U4":>8} {"xi":>7} '
          f'{"R2_pow":>8} {"R2_exp":>8}')
    print('-' * 52)
    for fa in FA_SWEEP:
        k = f'fa{fa:.2f}'
        if k not in agg_C:
            continue
        v = agg_C[k]
        print(f'{fa:>6.2f} {v["phi_amp"]:>9.5f} {v["U4"]:>8.4f} '
              f'{v["xi"]:>7.2f} {v["r2_pow"]:>8.4f} {v["r2_exp"]:>8.4f}')

    analysis = dict(A=agg_A, B=agg_B, C=agg_C)
    with open(ANALYSIS_FILE, 'w') as f:
        json.dump(analysis, f, indent=2)
    print(f'\nSaved -> {ANALYSIS_FILE}')
