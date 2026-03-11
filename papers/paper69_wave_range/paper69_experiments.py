"""
Paper 69: Wave-Range Dependence of Critical Exponents.

Background:
  Paper 68 falsified the Manna universality class (conservation law violated).
  Paper 63 FSS: beta=0.65, nu=0.97 -- match no known class.
  Conjecture: the long-range wave coupling (r_w=5) displaces VCSM from
  Directed Percolation (DP). DP exponents in 2+1D: beta~0.584, nu_perp~0.734.

  IF r_w is the key: setting r_w=1 (nearest-neighbour) should recover DP exponents.
  IF r_w is irrelevant: exponents stay near beta=0.65 across all r_w.

Wave coverage note:
  Manhattan ball of radius r_w has area A(r_w) = 2*r_w*(r_w+1) + 1.
  r_w=1: A=5 sites.  r_w=5: A=61 sites.  r_w=8: A=145 sites.
  With fixed wave firing rate (prob mode), r_w=1 covers 5/61 ~ 8% as many
  sites per wave as r_w=5.  Two protocols:
  (A) Constant wave-RATE (same fires/step): easier to implement.
  (B) Constant wave-COVERAGE (same hits/site/step): requires rescaling rate
      by A(r_w=5)/A(r_w) -- fires up to 12x more often at r_w=1.

  Phase A uses protocol (A) for speed: sweeps r_w at fixed rate, measuring
  how ordering depends on wave geometry alone.
  Phase B uses protocol (B) at a fixed hits/site for the FSS, ensuring
  that the exponent comparison is clean (same total drive, different geometry).
  Phase C: fine P scan at r_w=1 (protocol B, matched coverage) to extract beta
  directly from |M(P)| and compare with Paper 68 Phase A (r_w=5, same protocol).

Phase A: r_w sweep at fixed conditions.
  L=80, P=0.020, r_w in {1,2,3,4,5,6,8}, 6 seeds, 10k steps.
  Protocol A (constant rate). Measures U4, absM.

Phase B: FSS comparison r_w in {1, 5}.
  L in {40, 60, 80, 100, 120}, P in {0.005, 0.010, 0.020, 0.050}.
  6 seeds, 10k steps. Protocol B (matched coverage).
  Extracts beta/nu from absM vs L. Compares exponents.

Phase C: Fine P scan at r_w=1, L=160, matched coverage.
  P in {0.001, 0.002, 0.003, 0.005, 0.007, 0.010, 0.015, 0.020, 0.030, 0.050}.
  6 seeds, 10k steps. Protocol B.
  Direct comparison with Paper 68 Phase A (r_w=5 same protocol).
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

# ── Wave area (Manhattan ball) ────────────────────────────────────────────────
def wave_area(r):
    """Number of sites in Manhattan ball of radius r."""
    return 2 * r * (r + 1) + 1

R_W_REF = 5  # reference radius (Papers 63-68)
A_REF   = wave_area(R_W_REF)  # 61

# ── Phase A parameters ────────────────────────────────────────────────────────
L_A     = 80
PC_A    = 0.020
RW_A    = [1, 2, 3, 4, 5, 6, 8]
SEEDS_A = list(range(6))
NSTEPS_A = 10000

# ── Phase B parameters (FSS comparison) ──────────────────────────────────────
RW_B    = [1, 5]
L_B     = [40, 60, 80, 100, 120]
PC_B    = [round(x, 4) for x in [0.005, 0.010, 0.020, 0.050]]
SEEDS_B = list(range(6))
NSTEPS_B = 10000

# ── Phase C parameters (fine P scan at r_w=1) ────────────────────────────────
RW_C    = 1
L_C     = 160
PC_C    = [round(x, 4) for x in
           [0.001, 0.002, 0.003, 0.005, 0.007, 0.010, 0.015, 0.020, 0.030, 0.050]]
SEEDS_C = list(range(6))
NSTEPS_C = 10000

RESULTS_FILE  = Path(__file__).parent / 'results' / 'paper69_results.json'
ANALYSIS_FILE = Path(__file__).parent / 'results' / 'paper69_analysis.json'


# ── Wave helpers ──────────────────────────────────────────────────────────────
def _wef(L):
    """Exact wave_every float (reference, r_w=5)."""
    return WE_BASE * (L_BASE / L) ** 2

def _prob_A(L):
    """Protocol A: constant rate (same as Papers 67-68)."""
    return min(1.0, 1.0 / _wef(L))

def _prob_B(L, r_w):
    """Protocol B: matched coverage -- scale rate by A_REF/A(r_w)."""
    scale = A_REF / wave_area(r_w)
    return min(1.0, scale / _wef(L))


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
    chi   = float(N_zone * np.var(M_arr))
    return dict(absM=absM, M2=M2, M4=M4, U4=U4, chi=chi)


# ── Module-level workers ──────────────────────────────────────────────────────
def _wA(args):
    r_w, seed = args
    return ('A', r_w, seed,
            run_one(L_A, PC_A, seed, NSTEPS_A,
                    wave_prob=_prob_A(L_A), r_w=r_w))

def _wB(args):
    r_w, L, pc, seed = args
    return ('B', r_w, L, pc, seed,
            run_one(L, pc, seed, NSTEPS_B,
                    wave_prob=_prob_B(L, r_w), r_w=r_w))

def _wC(args):
    pc, seed = args
    return ('C', pc, seed,
            run_one(L_C, pc, seed, NSTEPS_C,
                    wave_prob=_prob_B(L_C, RW_C), r_w=RW_C))


# ── Aggregation ────────────────────────────────────────────────────────────────
def agg_group(d_by_seed):
    vals = list(d_by_seed.values())
    def _ms(key):  return ms([v.get(key, float('nan')) for v in vals])
    def _ses(key): return ses([v.get(key, float('nan')) for v in vals])
    return dict(absM=_ms('absM'), absM_se=_ses('absM'),
                U4=_ms('U4'), U4_se=_ses('U4'),
                chi=_ms('chi'))


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
        args_A = [(r_w, seed) for r_w in RW_A for seed in SEEDS_A]
        areas  = {r: wave_area(r) for r in RW_A}
        print(f'Phase A: {len(args_A)} runs -- L={L_A}, P={PC_A}, '
              f'r_w in {RW_A}, areas={areas} ...')
        with mp.Pool(processes=min(len(args_A), ncpu)) as pool:
            res = pool.map(_wA, args_A)
        raw['A'] = {}
        for _, r_w, seed, r in res:
            raw['A'].setdefault(str(r_w), {})[str(seed)] = r
        with open(RESULTS_FILE, 'w') as f:
            json.dump(raw, f)
        print('Phase A done.')

    # ── Phase B ───────────────────────────────────────────────────────────────
    if 'B' not in raw:
        args_B = [(r_w, L, pc, seed)
                  for r_w in RW_B for L in L_B for pc in PC_B for seed in SEEDS_B]
        print(f'\nPhase B: {len(args_B)} runs -- r_w in {RW_B}, '
              f'L in {L_B}, P in {PC_B}, {NSTEPS_B} steps (matched coverage) ...')
        with mp.Pool(processes=min(len(args_B), ncpu)) as pool:
            res = pool.map(_wB, args_B)
        raw['B'] = {}
        for _, r_w, L, pc, seed, r in res:
            k = f'rw{r_w}_L{L}_pc{pc:.4f}'
            raw['B'].setdefault(k, {})[str(seed)] = r
        with open(RESULTS_FILE, 'w') as f:
            json.dump(raw, f)
        print('Phase B done.')

    # ── Phase C ───────────────────────────────────────────────────────────────
    if 'C' not in raw:
        args_C = [(pc, seed) for pc in PC_C for seed in SEEDS_C]
        p_wave = _prob_B(L_C, RW_C)
        print(f'\nPhase C: {len(args_C)} runs -- r_w={RW_C}, L={L_C}, '
              f'p_wave={p_wave:.4f} (matched coverage), {NSTEPS_C} steps ...')
        with mp.Pool(processes=min(len(args_C), ncpu)) as pool:
            res = pool.map(_wC, args_C)
        raw['C'] = {}
        for _, pc, seed, r in res:
            raw['C'].setdefault(f'pc{pc:.4f}', {})[str(seed)] = r
        with open(RESULTS_FILE, 'w') as f:
            json.dump(raw, f)
        print('Phase C done.')

    # ── Aggregate ─────────────────────────────────────────────────────────────
    agg_A = {str(r_w): agg_group(raw['A'][str(r_w)])
             for r_w in RW_A if str(r_w) in raw.get('A', {})}

    agg_B = {}
    for r_w in RW_B:
        for L in L_B:
            for pc in PC_B:
                k = f'rw{r_w}_L{L}_pc{pc:.4f}'
                if k in raw.get('B', {}):
                    agg_B[k] = dict(r_w=r_w, L=L, pc=pc, **agg_group(raw['B'][k]))

    agg_C = {}
    for pc in PC_C:
        k = f'pc{pc:.4f}'
        if k in raw.get('C', {}):
            agg_C[k] = dict(pc=pc, **agg_group(raw['C'][k]))

    # ── Print Phase A ─────────────────────────────────────────────────────────
    print(f'\n=== PHASE A: r_w sweep (L={L_A}, P={PC_A}, protocol-A constant rate) ===')
    print(f'{"r_w":>5}  {"area":>6}  {"U4":>7}  {"se":>6}  {"absM":>9}  {"p_wave":>8}')
    print('-' * 55)
    for r_w in RW_A:
        v = agg_A.get(str(r_w), {})
        a = wave_area(r_w)
        pw = _prob_A(L_A)
        print(f'{r_w:>5}  {a:>6}  {v.get("U4", float("nan")):>7.4f}  '
              f'{v.get("U4_se", 0.):>6.4f}  {v.get("absM", float("nan")):>9.5f}  '
              f'{pw:>8.4f}')

    # ── Print Phase B: beta/nu comparison ─────────────────────────────────────
    print(f'\n=== PHASE B: FSS exponent comparison (r_w=1 vs r_w=5, matched coverage) ===')
    for r_w in RW_B:
        print(f'\n--- r_w = {r_w} (wave area = {wave_area(r_w)}) ---')
        print(f'  {"P":>7}', end='')
        for L in L_B:
            print(f'  absM(L={L})', end='')
        print()
        for pc in PC_B:
            print(f'  {pc:>7.4f}', end='')
            for L in L_B:
                k = f'rw{r_w}_L{L}_pc{pc:.4f}'
                aM = agg_B.get(k, {}).get('absM', float('nan'))
                print(f'  {aM:>10.5f}', end='')
            print()

        # beta/nu slopes
        print(f'  {"P":>7}  {"slope":>8}  {"note"}')
        for pc in PC_B:
            Lv = []; aMv = []
            for L in L_B:
                k = f'rw{r_w}_L{L}_pc{pc:.4f}'
                if k in agg_B and not math.isnan(agg_B[k]['absM']):
                    Lv.append(L); aMv.append(agg_B[k]['absM'])
            if len(Lv) >= 3:
                sl, _ = np.polyfit(np.log(Lv), np.log(aMv), 1)
                note = 'disorder' if sl < -0.8 else ('ordered' if sl > -0.4 else 'critical')
                print(f'  {pc:>7.4f}  {sl:>8.4f}  {note}')

    # ── Print Phase C: r_w=1 fine P scan ─────────────────────────────────────
    print(f'\n=== PHASE C: Fine P scan r_w={RW_C}, L={L_C} (matched coverage) ===')
    print(f'{"P":>8}  {"absM":>9}  {"U4":>7}  comparison with Paper68 (r_w=5)')
    print('-' * 55)
    p68_absM = {  # Paper 68 Phase A reference values at r_w=5
        0.001: 1.3e-4, 0.002: 1.3e-4, 0.003: 1.3e-4, 0.005: 1.4e-4,
        0.007: 1.4e-4, 0.010: 1.6e-4, 0.015: 2.1e-4, 0.020: 2.8e-4,
        0.030: 4.1e-4, 0.050: 7.0e-4
    }
    for pc in PC_C:
        k = f'pc{pc:.4f}'
        v = agg_C.get(k, {})
        aM = v.get('absM', float('nan'))
        u4 = v.get('U4', float('nan'))
        ref = p68_absM.get(round(pc, 3), float('nan'))
        ratio = aM / ref if not math.isnan(aM) and ref > 0 else float('nan')
        print(f'{pc:>8.4f}  {aM:>9.5f}  {u4:>7.4f}  '
              f'P68={ref:.1e}  ratio={ratio:.2f}')

    # ── Power-law fits for Phase C ────────────────────────────────────────────
    pc_fit = [agg_C[f'pc{pc:.4f}']['pc'] for pc in PC_C
              if f'pc{pc:.4f}' in agg_C and agg_C[f'pc{pc:.4f}']['absM'] > 1e-9]
    aM_fit = [agg_C[f'pc{pc:.4f}']['absM'] for pc in PC_C
              if f'pc{pc:.4f}' in agg_C and agg_C[f'pc{pc:.4f}']['absM'] > 1e-9]
    if len(pc_fit) >= 4:
        beta_c, logc = np.polyfit(np.log(pc_fit), np.log(aM_fit), 1)
        resid = np.log(aM_fit) - np.polyval([beta_c, logc], np.log(pc_fit))
        r2 = 1 - np.var(resid) / np.var(np.log(aM_fit))
        print(f'\nPower-law fit r_w={RW_C}: beta={beta_c:.4f}, R^2={r2:.4f}')
        print(f'  cf. Paper63 r_w=5: beta=0.650, Paper68 r_w=5 direct: beta=0.406')
        print(f'  DP prediction: beta=0.584')
        if abs(beta_c - 0.584) < 0.05:
            print('  -> CONSISTENT WITH DP: wave range IS the key differentiator.')
        elif abs(beta_c - 0.650) < 0.05:
            print('  -> CONSISTENT WITH PAPER63 EXPONENTS: wave range is IRRELEVANT.')
        else:
            print('  -> NEITHER DP NOR PAPER63: novel class persists at r_w=1.')

    analysis = dict(A=agg_A, B=agg_B, C=agg_C)
    with open(ANALYSIS_FILE, 'w') as f:
        json.dump(analysis, f, indent=2)
    print(f'\nSaved -> {ANALYSIS_FILE}')
