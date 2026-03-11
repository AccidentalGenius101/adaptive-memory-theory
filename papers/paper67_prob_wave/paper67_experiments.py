"""
Paper 67: Probabilistic Wave Firing -- Correcting the WAVE_EVERY Rounding Artefact.

Background:
  Paper 66 Phase A: U4(L=160) < U4(L=120) for all tested P.
  Root cause identified: WAVE_EVERY(L=160) rounds from 1.5625 to 2 (integer),
  giving L=160 only ~78% of intended wave exposure (15.2 vs 19.5 hits/site).
  Paper 66 Conclusion: this is a numerical artefact, not evidence for finite p_c.
  Paper 67 Test: with the artefact REMOVED, does U4(L=160) > U4(L=120)?

Fix: Probabilistic wave firing.
  At each step: fire wave with prob = min(1, 1/wave_every_float),
  where wave_every_float = WE_BASE*(L_BASE/L)^2 (exact, no integer rounding).
  L=160: prob=1/1.5625=0.64; avg 6400 waves per 10k steps (vs 5000 with det WE=2).
  Speed: ~4.2 sweeps/step avg, vs 6 for det WE=1 (~30% faster than forced WE=1).

Phase A: Discriminating test with corrected wave density.
  L in {80, 100, 120, 160}, P in {0, 0.005, 0.010, 0.020}.
  Probabilistic firing. 3 seeds, 10k steps.
  Prediction (p_c=0+): U4 monotone increasing with L for all P > 0.
  Prediction (finite p_c): reversal persists at U4(160) < U4(120) for small P.

Phase B: Direct artefact quantification at L=160.
  L=160, P in {0.005, 0.020}, 4 seeds, 6k steps.
  Three conditions: (i) det WE=2 (old), (ii) det WE=1 (maximum density),
                    (iii) prob (target density = 1/1.5625).
  Direct measurement: does U4(prob) match U4(WE=1) better than U4(WE=2)?
  If yes: rounding artefact confirmed quantitatively.
"""

import numpy as np, json, math, multiprocessing as mp
from pathlib import Path

# ── Physical parameters (identical to Papers 64-66) ─────────────────────────
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

# ── Phase A parameters ───────────────────────────────────────────────────────
L_A       = [80, 100, 120, 160]
PC_A      = [round(x, 4) for x in [0.000, 0.005, 0.010, 0.020]]
SEEDS_A   = list(range(3))
NSTEPS_A  = 10000

# ── Phase B parameters ───────────────────────────────────────────────────────
L_B       = 160
PC_B      = [0.005, 0.020]
SEEDS_B   = list(range(4))
NSTEPS_B  = 6000
# (cond_name, wave_every_int_or_None) -- None => probabilistic
CONDS_B   = [('det2', 2), ('det1', 1), ('prob', None)]

RESULTS_FILE  = Path(__file__).parent / 'results' / 'paper67_results.json'
ANALYSIS_FILE = Path(__file__).parent / 'results' / 'paper67_analysis.json'


# ── Wave density helpers ─────────────────────────────────────────────────────
def _wef(L):
    """Exact (non-integer) wave_every for size L."""
    return WE_BASE * (L_BASE / L) ** 2

def _wei(L):
    """Integer (rounded) wave_every for size L."""
    return max(1, round(_wef(L)))

def _prob(L):
    """Fire probability per step (probabilistic mode)."""
    return min(1.0, 1.0 / _wef(L))


# ── Lattice helpers ──────────────────────────────────────────────────────────
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
def run_one(L, P_causal, seed, nsteps, wave_every_int=None, wave_prob=None):
    """
    Single simulation run.
    wave_every_int: fire every this many steps (deterministic).
    wave_prob:      fire with this probability each step (probabilistic).
    Exactly one of these should be set (not None).
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

    for t in range(nsteps):
        s = metro(s, nb, T_FIXED, rng)
        base  += BETA_BASE * (s - base)
        same   = (s > 0) == (base > 0)
        streak = np.where(same, streak + 1, 0)
        mid   *= MID_DECAY

        do_wave = ((wave_every_int is not None and t % wave_every_int == 0) or
                   (wave_prob is not None and rng.random() < wave_prob))

        if do_wave:
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

    M_arr = np.array(Mseries)
    M2    = float(np.mean(M_arr**2)); M4 = float(np.mean(M_arr**4))
    absM  = float(np.mean(np.abs(M_arr)))
    U4    = float(1. - M4 / (3. * M2**2)) if M2 > 1e-12 else float('nan')
    chi   = float(N_zone * np.var(M_arr))
    return dict(absM=absM, M2=M2, M4=M4, U4=U4, chi=chi)


# ── Module-level workers (required for Windows multiprocessing spawn) ─────────
def _wA(args):
    L, pc, seed = args
    return ('A', L, pc, seed, run_one(L, pc, seed, NSTEPS_A,
                                      wave_every_int=None,
                                      wave_prob=_prob(L)))

def _wB(args):
    pc, seed, cond_name, we_int = args
    if we_int is not None:
        r = run_one(L_B, pc, seed, NSTEPS_B,
                    wave_every_int=we_int, wave_prob=None)
    else:
        r = run_one(L_B, pc, seed, NSTEPS_B,
                    wave_every_int=None, wave_prob=_prob(L_B))
    return ('B', cond_name, pc, seed, r)


# ── Aggregation ────────────────────────────────────────────────────────────────
def agg_group(d_by_seed):
    vals = list(d_by_seed.values())
    def _ms(key):  return ms([v.get(key, float('nan')) for v in vals])
    def _ses(key): return ses([v.get(key, float('nan')) for v in vals])
    return dict(absM=_ms('absM'), U4=_ms('U4'), U4_se=_ses('U4'), chi=_ms('chi'))


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
        prob_info = {L: f'{_prob(L):.3f}' for L in L_A}
        print(f'Phase A: {len(args_A)} runs -- L in {L_A}, '
              f'prob per step = {prob_info}, {NSTEPS_A} steps ...')
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
        args_B = [(pc, seed, cond_name, we_int)
                  for pc in PC_B for seed in SEEDS_B
                  for cond_name, we_int in CONDS_B]
        print(f'\nPhase B: {len(args_B)} runs -- L={L_B}, '
              f'conditions {[c for c,_ in CONDS_B]}, {NSTEPS_B} steps ...')
        with mp.Pool(processes=min(len(args_B), ncpu)) as pool:
            res = pool.map(_wB, args_B)
        raw['B'] = {}
        for _, cond_name, pc, seed, r in res:
            k = f'{cond_name}_pc{pc:.4f}'
            raw['B'].setdefault(k, {})[str(seed)] = r
        with open(RESULTS_FILE, 'w') as f:
            json.dump(raw, f)
        print('Phase B done.')

    # ── Aggregate ─────────────────────────────────────────────────────────────
    agg_A = {}
    for L in L_A:
        for pc in PC_A:
            k = f'L{L}_pc{pc:.4f}'
            if k in raw.get('A', {}):
                agg_A[k] = dict(L=L, pc=pc, **agg_group(raw['A'][k]))

    agg_B = {}
    for cond_name, _ in CONDS_B:
        for pc in PC_B:
            k = f'{cond_name}_pc{pc:.4f}'
            if k in raw.get('B', {}):
                agg_B[k] = dict(cond=cond_name, pc=pc, **agg_group(raw['B'][k]))

    # ── Print Phase A ─────────────────────────────────────────────────────────
    print('\n=== PHASE A: Discriminating Test with Probabilistic Wave Firing ===')
    print(f'{"P":>8}', end='')
    for L in L_A:
        print(f'  U4(L={L:3d})', end='')
    print('  verdict')
    print('-' * 80)
    for pc in PC_A:
        print(f'{pc:>8.4f}', end='')
        u_vals = {}
        for L in L_A:
            k = f'L{L}_pc{pc:.4f}'
            u = agg_A[k]['U4'] if k in agg_A else float('nan')
            u_vals[L] = u
            print(f'  {u:>9.4f}', end='')
        if pc == 0.000:
            print('  baseline')
        else:
            u160 = u_vals.get(160, float('nan'))
            u120 = u_vals.get(120, float('nan'))
            if math.isnan(u120) or math.isnan(u160):
                print('  MISSING')
            elif u160 > u120 + 0.01:
                print('  U4(160)>U4(120) -> p_c=0+')
            elif u160 < u120 - 0.01:
                print('  REVERSED -> finite p_c?')
            else:
                print('  AMBIGUOUS')

    n_ordered = sum(
        1 for pc in PC_A if pc > 0
        and f'L120_pc{pc:.4f}' in agg_A and f'L160_pc{pc:.4f}' in agg_A
        and not math.isnan(agg_A[f'L160_pc{pc:.4f}']['U4'])
        and not math.isnan(agg_A[f'L120_pc{pc:.4f}']['U4'])
        and agg_A[f'L160_pc{pc:.4f}']['U4'] > agg_A[f'L120_pc{pc:.4f}']['U4'] + 0.01
    )
    n_total = sum(1 for pc in PC_A if pc > 0)
    print(f'\nU4(160) > U4(120)+0.01 for {n_ordered}/{n_total} P > 0 values.')
    if n_ordered == n_total:
        print('CONCLUSION: p_c = 0+ CONFIRMED. Rounding artefact was the cause.')
    elif n_ordered == 0:
        print('CONCLUSION: Reversal persists with correct wave density -- finite p_c.')
    else:
        print('CONCLUSION: Mixed result.')

    # ── Print Phase B ─────────────────────────────────────────────────────────
    print(f'\n=== PHASE B: Wave-firing Policy Comparison (L={L_B}) ===')
    print(f'{"Cond":>6} {"P":>7}  U4 +/- se')
    print('-' * 40)
    for cond_name, _ in CONDS_B:
        for pc in PC_B:
            k = f'{cond_name}_pc{pc:.4f}'
            if k in agg_B:
                v = agg_B[k]
                print(f'{cond_name:>6} {pc:>7.4f}  {v["U4"]:.4f} +/- {v["U4_se"]:.4f}')

    analysis = dict(A=agg_A, B=agg_B)
    with open(ANALYSIS_FILE, 'w') as f:
        json.dump(analysis, f, indent=2)
    print(f'\nSaved -> {ANALYSIS_FILE}')
