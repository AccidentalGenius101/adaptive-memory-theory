"""
Paper 61: Critical Exponents of the Causal-Purity Phase Transition.

Finite-size scaling (FSS) analysis of the purity-controlled transition in Ising + VCSM-lite.

Phase 1: Dense P_causal sweep at L=40, T=3.0, range [0.0, 0.70] to locate p_c precisely.
Phase 2: FSS at L=[20,30,40,60], dense sweep near p_c to extract critical exponents.

Measurements per run (time-series in steady state):
  M(t) = phi_zone0(t) - phi_zone1(t)    (raw order parameter)
  <|M|>, <M^2>, <M^4>                   (moments for Binder cumulant + FSS)
  chi = (N_zone) * (<M^2> - <M>^2)      (susceptibility)
  U4  = 1 - <M^4>/(3*<M^2>^2)           (Binder cumulant; p_c-independent crossing)

Expected:
  U4 crossing at p_c (system-size-independent at the critical point)
  |M| ~ L^(-beta/nu) at p_c
  chi ~ L^(gamma/nu) at p_c
  Data collapse: |M|*L^(beta/nu) = f((P_causal - p_c)*L^(1/nu))

Comparison targets:
  Mean field (Landau): beta=1/2, nu=1/2, gamma=1
  2D Ising: beta=1/8, nu=1, gamma=7/4
  If neither fits -> new universality class
"""
import numpy as np, json, os, math, multiprocessing as mp, scipy.optimize as opt

J           = 1.0
BETA_BASE   = 0.005
FA          = 0.30
FIELD_DECAY = 0.999
MID_DECAY   = 0.97
SS          = 8
WAVE_EVERY  = 25
WAVE_RADIUS = 5
WAVE_DUR    = 5
EXT_FIELD   = 1.5
T_FIXED     = 3.0          # above T_c; isolates purity transition

# Phase 1: find p_c
L_SCAN        = 40
PC_SCAN       = [round(x, 2) for x in np.arange(0.00, 0.72, 0.04)]
SEEDS_SCAN    = list(range(5))
NSTEPS_SCAN   = 8000

# Phase 2: FSS
L_FSS         = [20, 30, 40, 60]
PC_FSS        = [round(x, 2) for x in np.arange(0.00, 0.65, 0.05)]
SEEDS_FSS     = list(range(5))
NSTEPS_FSS    = 8000
SS_START_FRAC = 0.40       # use last 60% for statistics

RESULTS_FILE  = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             'results', 'paper61_results.json')
ANALYSIS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             'results', 'paper61_analysis.json')


# ── Ising + VCSM-lite ──────────────────────────────────────────────────────────
def build_neighbors(L):
    N = L * L
    row = np.arange(N) // L; col = np.arange(N) % L
    return np.stack([((row-1)%L)*L+col, ((row+1)%L)*L+col,
                     row*L+(col-1)%L,   row*L+(col+1)%L], axis=1)


def metropolis_step(s, nb, T, rng, h_ext=None, hit=None):
    N = len(s); L = int(round(math.sqrt(N)))
    row = np.arange(N) // L; col = np.arange(N) % L
    h = np.zeros(N)
    if h_ext is not None and hit is not None:
        h[hit] = h_ext
    for sub in [0, 1]:
        idx = np.where((row + col) % 2 == sub)[0]
        nb_sum = s[nb[idx]].sum(1)
        dE = 2.0 * J * s[idx] * nb_sum - 2.0 * h[idx] * s[idx]
        acc = (dE <= 0) | (rng.random(len(idx)) < np.exp(-np.clip(dE / T, -20, 20)))
        s[idx[acc]] *= -1
    return s


def wave_sites_L(cx, cy, r, L):
    N = L * L
    row = np.arange(N) // L; col = np.arange(N) % L
    dx = np.minimum(np.abs(col - cx), L - np.abs(col - cx))
    dy = np.minimum(np.abs(row - cy), L - np.abs(row - cy))
    return np.where(dx + dy <= r)[0]


def run_one(L, P_causal, seed, N_STEPS):
    rng = np.random.RandomState(seed)
    nb  = build_neighbors(L)
    N   = L * L
    col_arr = np.arange(N) % L
    zone    = (col_arr >= L // 2).astype(int)
    N_zone  = N // 2

    s      = rng.choice([-1, 1], N).astype(float)
    base   = s * 0.1
    mid    = np.zeros(N)
    phi    = np.zeros(N)
    streak = np.zeros(N, int)
    wave_z = 0

    # Collect time series of M in steady state
    ss_start = int(N_STEPS * SS_START_FRAC)
    M_series = []

    for t in range(N_STEPS):
        s = metropolis_step(s, nb, T_FIXED, rng)
        base  += BETA_BASE * (s - base)
        same   = (s > 0) == (base > 0)
        streak = np.where(same, streak + 1, 0)
        mid   *= MID_DECAY

        if t % WAVE_EVERY == 0:
            cx = rng.randint(0, L//2) if wave_z == 0 else rng.randint(L//2, L)
            cy = rng.randint(L)
            hit = wave_sites_L(cx, cy, WAVE_RADIUS, L)
            if len(hit) > 0:
                h_ext = EXT_FIELD if wave_z == 0 else -EXT_FIELD
                s_bef = s[hit].copy()
                for _ in range(WAVE_DUR):
                    s = metropolis_step(s, nb, T_FIXED, rng, h_ext=h_ext, hit=hit)
                dev_hit = s[hit] - base[hit]
                zone_match  = (zone[hit] == wave_z)
                true_signal = np.where(zone_match, dev_hit, -dev_hit)
                is_causal   = rng.random(len(hit)) < P_causal
                std_dev     = float(np.std(dev_hit)) + 0.5
                noise       = rng.normal(0, std_dev, len(hit))
                mid[hit]   += FA * np.where(is_causal, true_signal, noise)
            wave_z = 1 - wave_z

        gate = streak >= SS
        if gate.any():
            wi = np.where(gate)[0]
            phi[wi]   += FA * (mid[wi] - phi[wi])
            streak[wi] = 0
        phi *= FIELD_DECAY

        # Collect order parameter in steady state
        if t >= ss_start:
            M = float(np.mean(phi[zone == 0])) - float(np.mean(phi[zone == 1]))
            M_series.append(M)

    # Compute moments
    M_arr  = np.array(M_series)
    M2     = float(np.mean(M_arr**2))
    M4     = float(np.mean(M_arr**4))
    absM   = float(np.mean(np.abs(M_arr)))
    varM   = float(np.var(M_arr))
    U4     = float(1.0 - M4 / (3.0 * M2**2)) if M2 > 1e-12 else float('nan')
    chi    = float(N_zone * varM)
    sigma_phi = float(np.std(phi)) + 1e-8
    S_phi  = abs(float(np.mean(phi[zone==0])) - float(np.mean(phi[zone==1]))) / sigma_phi

    return dict(absM=absM, M2=M2, M4=M4, U4=U4, chi=chi, S_phi=S_phi, varM=varM)


def _worker(args):
    L, pc, seed, nsteps = args
    return (L, pc, seed, run_one(L, pc, seed, nsteps))


def mean_s(vals):
    v = [x for x in vals if x is not None and not math.isnan(float(x))]
    return float(np.mean(v)) if v else float('nan')

def se_s(vals):
    v = [x for x in vals if x is not None and not math.isnan(float(x))]
    return float(np.std(v) / math.sqrt(len(v))) if len(v) > 1 else 0.


if __name__ == '__main__':
    mp.freeze_support()
    os.makedirs(os.path.dirname(RESULTS_FILE), exist_ok=True)

    # ── Phase 1: scan to find p_c ─────────────────────────────────────────────
    scan_key = 'scan'
    raw = {}
    if os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE) as f: raw = json.load(f)

    if scan_key not in raw:
        scan_args = [(L_SCAN, pc, seed, NSTEPS_SCAN)
                     for pc in PC_SCAN for seed in SEEDS_SCAN]
        print(f'Phase 1: {len(scan_args)} scan runs...')
        with mp.Pool(processes=min(len(scan_args), mp.cpu_count())) as pool:
            res = pool.map(_worker, scan_args)
        raw[scan_key] = {}
        for L, pc, seed, r in res:
            k = f'pc{pc:.2f}'
            if k not in raw[scan_key]: raw[scan_key][k] = {}
            raw[scan_key][k][str(seed)] = r
        with open(RESULTS_FILE, 'w') as f: json.dump(raw, f)
        print('Phase 1 done.')

    # Aggregate scan
    print('\nPhase 1 scan: S_phi and U4 vs P_causal (L=40, T=3.0)')
    print(f'{"P_causal":>10} {"S_phi":>8} {"|M|":>8} {"U4":>8} {"chi":>10}')
    scan_summary = {}
    for pc in PC_SCAN:
        k = f'pc{pc:.2f}'
        if k not in raw[scan_key]: continue
        vals = list(raw[scan_key][k].values())
        sp = mean_s([v['S_phi'] for v in vals])
        am = mean_s([v['absM']  for v in vals])
        u4 = mean_s([v['U4']    for v in vals])
        ch = mean_s([v['chi']   for v in vals])
        scan_summary[pc] = dict(S_phi=sp, absM=am, U4=u4, chi=ch)
        print(f'{pc:>10.2f} {sp:>8.3f} {am:>8.4f} {u4:>8.3f} {ch:>10.3f}')

    # Estimate p_c from where S_phi crosses 1 (or where chi peaks)
    pcs  = sorted(scan_summary.keys())
    sps  = [scan_summary[pc]['S_phi'] for pc in pcs]
    chis = [scan_summary[pc]['chi']   for pc in pcs]
    chi_peak_idx = int(np.argmax(chis))
    p_c_est = pcs[chi_peak_idx]
    print(f'\nEstimated p_c from chi peak: {p_c_est:.2f}')
    print(f'S_phi at p_c: {scan_summary[p_c_est]["S_phi"]:.3f}')

    # ── Phase 2: FSS ──────────────────────────────────────────────────────────
    fss_key = 'fss'
    if fss_key not in raw:
        fss_args = [(L, pc, seed, NSTEPS_FSS)
                    for L in L_FSS for pc in PC_FSS for seed in SEEDS_FSS]
        print(f'\nPhase 2: {len(fss_args)} FSS runs...')
        with mp.Pool(processes=min(len(fss_args), mp.cpu_count())) as pool:
            res = pool.map(_worker, fss_args)
        raw[fss_key] = {}
        for L, pc, seed, r in res:
            k = f'L{L}_pc{pc:.2f}'
            if k not in raw[fss_key]: raw[fss_key][k] = {}
            raw[fss_key][k][str(seed)] = r
        with open(RESULTS_FILE, 'w') as f: json.dump(raw, f)
        print('Phase 2 done.')

    # Aggregate FSS
    fss_summary = {}
    print('\nFSS results: |M| and chi by L and P_causal')
    for L in L_FSS:
        print(f'\n  L={L}:')
        print(f'  {"P_causal":>10} {"S_phi":>8} {"|M|":>8} {"U4":>8} {"chi":>10}')
        for pc in PC_FSS:
            k = f'L{L}_pc{pc:.2f}'
            if k not in raw[fss_key]: continue
            vals = list(raw[fss_key][k].values())
            sp = mean_s([v['S_phi'] for v in vals])
            am = mean_s([v['absM']  for v in vals])
            u4 = mean_s([v['U4']    for v in vals])
            ch = mean_s([v['chi']   for v in vals])
            sp_se = se_s([v['S_phi'] for v in vals])
            am_se = se_s([v['absM']  for v in vals])
            fss_summary[f'L{L}_pc{pc:.2f}'] = dict(S_phi=sp, S_phi_se=sp_se,
                                                     absM=am, absM_se=am_se,
                                                     U4=u4, chi=ch, L=L, pc=pc)
            print(f'  {pc:>10.2f} {sp:>8.3f} {am:>8.4f} {u4:>8.3f} {ch:>10.3f}')

    # Binder cumulant crossing
    print('\nBinder cumulant U4 crossing analysis:')
    print(f'{"P_causal":>10}', end='')
    for L in L_FSS:
        print(f'  L={L:2d}:U4', end='')
    print()
    for pc in PC_FSS:
        print(f'{pc:>10.2f}', end='')
        for L in L_FSS:
            k = f'L{L}_pc{pc:.2f}'
            u4 = fss_summary[k]['U4'] if k in fss_summary else float('nan')
            print(f'  {u4:>8.3f}', end='')
        print()

    # Estimate beta from |M| ~ L^(-beta/nu) at p_c
    # At the critical point: |M|(L) ~ L^(-beta/nu)
    # Log fit: log(|M|) = -beta/nu * log(L) + const
    pcs_fss = [p_c_est] if p_c_est in [round(x,2) for x in PC_FSS] else [PC_FSS[len(PC_FSS)//2]]
    for pc_fit in pcs_fss[:1]:
        Ls_fit = []; aMs_fit = []
        for L in L_FSS:
            k = f'L{L}_pc{round(pc_fit,2):.2f}'
            if k in fss_summary and not math.isnan(fss_summary[k]['absM']):
                Ls_fit.append(L); aMs_fit.append(fss_summary[k]['absM'])
        if len(Ls_fit) >= 3:
            logL = np.log(Ls_fit); logM = np.log([max(x,1e-6) for x in aMs_fit])
            p = np.polyfit(logL, logM, 1)
            print(f'\nAt P_causal={pc_fit:.2f}: log|M| = {p[0]:.3f}*log(L) + {p[1]:.3f}')
            print(f'  -> -beta/nu = {p[0]:.3f}  (mean field: -0.5, 2D Ising: -0.125)')
            beta_over_nu = -p[0]
            print(f'  -> beta/nu = {beta_over_nu:.3f}')

    analysis = dict(scan=scan_summary,
                    fss=fss_summary,
                    p_c_est=p_c_est)
    with open(ANALYSIS_FILE, 'w') as f: json.dump(analysis, f, indent=2)
    print(f'\nSaved analysis -> {ANALYSIS_FILE}')
