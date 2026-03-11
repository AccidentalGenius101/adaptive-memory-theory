"""
Paper 63: Critical Exponents of the Causal-Purity Phase Transition.

Paper 62 (Phase A) confirmed true long-range order under extensive-drive FSS
and located p_c ~ 0.02-0.05.

This paper does the precision work:
  - Fine scan P_causal in [0.000, 0.080], step 0.005
  - L in {20, 30, 40, 60, 80}, WAVE_EVERY scaled as (40/L)^2 * 25
  - 8 seeds, 12000 steps
  - Extract: p_c (Binder crossing), beta/nu (|M| vs L at p_c),
             1/nu (dU4/dP vs L at p_c), gamma/nu (chi vs L at p_c)
  - Data collapse: |M| * L^(beta/nu) vs (P - p_c) * L^(1/nu)
  - Compare exponents to mean field, 2D Ising, directed percolation
"""
import numpy as np, json, os, math, multiprocessing as mp
import scipy.optimize as opt
from pathlib import Path

# ── Physical parameters (same as Papers 61-62) ─────────────────────────────
J           = 1.0
BETA_BASE   = 0.005
FA          = 0.30
FIELD_DECAY = 0.999
MID_DECAY   = 0.97
SS          = 8
WAVE_RADIUS = 5
WAVE_DUR    = 5
EXT_FIELD   = 1.5
T_FIXED     = 3.0
L_BASE      = 40
WAVE_EVERY_BASE = 25

# ── Scan parameters ─────────────────────────────────────────────────────────
L_FSS         = [20, 30, 40, 60, 80]
PC_FINE       = [round(x, 3) for x in np.arange(0.000, 0.085, 0.005)]
SEEDS         = list(range(8))
NSTEPS        = 12000
SS_START_FRAC = 0.40

RESULTS_FILE  = Path(__file__).parent / 'results' / 'paper63_results.json'
ANALYSIS_FILE = Path(__file__).parent / 'results' / 'paper63_analysis.json'


def wave_every(L):
    return max(1, round(WAVE_EVERY_BASE * (L_BASE / L) ** 2))


# ── Ising helpers ────────────────────────────────────────────────────────────
def build_neighbors(L):
    N = L * L
    row = np.arange(N) // L; col = np.arange(N) % L
    return np.stack([((row-1)%L)*L+col, ((row+1)%L)*L+col,
                     row*L+(col-1)%L,   row*L+(col+1)%L], axis=1)

def metro(s, nb, T, rng, h_ext=None, hit=None):
    N = len(s); L = int(round(math.sqrt(N)))
    row = np.arange(N)//L; col = np.arange(N)%L
    h = np.zeros(N)
    if h_ext is not None and hit is not None: h[hit] = h_ext
    for sub in [0, 1]:
        idx = np.where((row+col)%2==sub)[0]
        nb_sum = s[nb[idx]].sum(1)
        dE = 2.*J*s[idx]*nb_sum - 2.*h[idx]*s[idx]
        acc = (dE<=0)|(rng.random(len(idx)) < np.exp(-np.clip(dE/T,-20,20)))
        s[idx[acc]] *= -1
    return s

def wave_sites(cx, cy, r, L):
    N = L*L
    row = np.arange(N)//L; col = np.arange(N)%L
    dx = np.minimum(np.abs(col-cx), L-np.abs(col-cx))
    dy = np.minimum(np.abs(row-cy), L-np.abs(row-cy))
    return np.where(dx+dy<=r)[0]

def mean_s(vals):
    v = [x for x in vals if x is not None and not (isinstance(x,float) and math.isnan(x))]
    return float(np.mean(v)) if v else float('nan')

def se_s(vals):
    v = [x for x in vals if x is not None and not (isinstance(x,float) and math.isnan(x))]
    return float(np.std(v)/math.sqrt(len(v))) if len(v)>1 else 0.


# ── Single run ───────────────────────────────────────────────────────────────
def run_one(L, P_causal, seed):
    rng    = np.random.RandomState(seed)
    nb     = build_neighbors(L)
    N      = L*L; N_zone = N//2
    col_a  = np.arange(N)%L
    zone   = (col_a >= L//2).astype(int)
    we     = wave_every(L)

    s      = rng.choice([-1,1], N).astype(float)
    base   = s * 0.1
    mid    = np.zeros(N)
    phi    = np.zeros(N)
    streak = np.zeros(N, int)
    wave_z = 0
    ss_start = int(NSTEPS * SS_START_FRAC)
    M_series = []

    for t in range(NSTEPS):
        s = metro(s, nb, T_FIXED, rng)
        base  += BETA_BASE*(s-base)
        same   = (s>0)==(base>0)
        streak = np.where(same, streak+1, 0)
        mid   *= MID_DECAY

        if t % we == 0:
            cx = rng.randint(0, L//2) if wave_z==0 else rng.randint(L//2, L)
            cy = rng.randint(L)
            hit = wave_sites(cx, cy, WAVE_RADIUS, L)
            if len(hit) > 0:
                h_ext = EXT_FIELD if wave_z==0 else -EXT_FIELD
                for _ in range(WAVE_DUR):
                    s = metro(s, nb, T_FIXED, rng, h_ext=h_ext, hit=hit)
                dev     = s[hit] - base[hit]
                zm      = (zone[hit]==wave_z)
                sig     = np.where(zm, dev, -dev)
                causal  = rng.random(len(hit)) < P_causal
                noise   = rng.normal(0, float(np.std(dev))+0.5, len(hit))
                mid[hit]+= FA * np.where(causal, sig, noise)
            wave_z = 1-wave_z

        gate = streak >= SS
        if gate.any():
            wi = np.where(gate)[0]
            phi[wi] += FA*(mid[wi]-phi[wi]); streak[wi] = 0
        phi *= FIELD_DECAY

        if t >= ss_start:
            M = float(np.mean(phi[zone==0])) - float(np.mean(phi[zone==1]))
            M_series.append(M)

    M_arr = np.array(M_series)
    M2    = float(np.mean(M_arr**2))
    M4    = float(np.mean(M_arr**4))
    absM  = float(np.mean(np.abs(M_arr)))
    varM  = float(np.var(M_arr))
    U4    = float(1. - M4/(3.*M2**2)) if M2>1e-12 else float('nan')
    chi   = float(N_zone * varM)
    return dict(absM=absM, M2=M2, M4=M4, U4=U4, chi=chi, varM=varM)

def _worker(args):
    L, pc, seed = args
    return (L, pc, seed, run_one(L, pc, seed))


# ── FSS analysis helpers ─────────────────────────────────────────────────────
def binder_crossing(pcs, u4_La, u4_Lb):
    """Find p_c by linear interpolation of the crossing of U4 for two L values."""
    crossings = []
    diff = np.array(u4_Lb) - np.array(u4_La)
    for i in range(len(diff)-1):
        if not (math.isnan(diff[i]) or math.isnan(diff[i+1])):
            if diff[i]*diff[i+1] <= 0:  # sign change
                t = -diff[i]/(diff[i+1]-diff[i])
                crossings.append(pcs[i] + t*(pcs[i+1]-pcs[i]))
    return crossings

def power_fit(x_arr, y_arr):
    """Fit y = a * x^b on log-log scale. Returns (b, a)."""
    x = np.array([xi for xi,yi in zip(x_arr,y_arr) if yi>0 and not math.isnan(yi)])
    y = np.array([yi for xi,yi in zip(x_arr,y_arr) if yi>0 and not math.isnan(yi)])
    if len(x) < 3: return float('nan'), float('nan')
    logx = np.log(x); logy = np.log(y)
    p = np.polyfit(logx, logy, 1)
    return p[0], float(np.exp(p[1]))


# ── MAIN ─────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    mp.freeze_support()
    RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)

    raw = {}
    if RESULTS_FILE.exists():
        with open(RESULTS_FILE) as f: raw = json.load(f)

    # ── Run simulations ──────────────────────────────────────────────────────
    if 'fss' not in raw:
        all_args = [(L, pc, seed)
                    for L in L_FSS for pc in PC_FINE for seed in SEEDS]
        print(f'Running {len(all_args)} FSS runs on {mp.cpu_count()} cores...')
        print('WAVE_EVERY per L:',
              {L: wave_every(L) for L in L_FSS})
        with mp.Pool(processes=min(len(all_args), mp.cpu_count())) as pool:
            results = pool.map(_worker, all_args)
        raw['fss'] = {}
        for L, pc, seed, r in results:
            k = f'L{L}_pc{pc:.3f}'
            if k not in raw['fss']: raw['fss'][k] = {}
            raw['fss'][k][str(seed)] = r
        with open(RESULTS_FILE, 'w') as f: json.dump(raw, f)
        print('Done.')

    # ── Aggregate ────────────────────────────────────────────────────────────
    fss = raw['fss']
    agg = {}
    for L in L_FSS:
        for pc in PC_FINE:
            k = f'L{L}_pc{pc:.3f}'
            if k not in fss: continue
            vals = list(fss[k].values())
            agg[k] = dict(
                L=L, pc=pc,
                absM  = mean_s([v['absM'] for v in vals]),
                absM_se = se_s([v['absM'] for v in vals]),
                U4    = mean_s([v['U4'] for v in vals]),
                U4_se = se_s([v['U4'] for v in vals]),
                chi   = mean_s([v['chi'] for v in vals]),
                chi_se = se_s([v['chi'] for v in vals]),
                M2    = mean_s([v['M2'] for v in vals]),
            )

    # ── Print raw table ──────────────────────────────────────────────────────
    print(f'\nFine FSS table:')
    print(f'{"P_causal":>10}', end='')
    for L in L_FSS: print(f'  L{L}:U4  ', end='')
    print()
    for pc in PC_FINE:
        print(f'{pc:>10.3f}', end='')
        for L in L_FSS:
            k = f'L{L}_pc{pc:.3f}'
            u4 = agg[k]['U4'] if k in agg else float('nan')
            print(f'  {u4:>7.4f}  ', end='')
        print()

    # ── Find p_c from Binder crossings ───────────────────────────────────────
    print('\nBinder crossings (all L pairs):')
    pc_estimates = []
    L_pairs = [(L_FSS[i], L_FSS[j]) for i in range(len(L_FSS))
               for j in range(i+1, len(L_FSS))]
    for La, Lb in L_pairs:
        u4_a = [agg[f'L{La}_pc{pc:.3f}']['U4']
                if f'L{La}_pc{pc:.3f}' in agg else float('nan')
                for pc in PC_FINE]
        u4_b = [agg[f'L{Lb}_pc{pc:.3f}']['U4']
                if f'L{Lb}_pc{pc:.3f}' in agg else float('nan')
                for pc in PC_FINE]
        crossings = binder_crossing(PC_FINE, u4_a, u4_b)
        for c in crossings:
            pc_estimates.append(c)
            print(f'  L={La} vs L={Lb}: p_c = {c:.4f}')
    p_c = float(np.mean(pc_estimates)) if pc_estimates else 0.03
    p_c_se = float(np.std(pc_estimates)) if len(pc_estimates)>1 else 0.
    print(f'\nEstimated p_c = {p_c:.4f} +/- {p_c_se:.4f}')

    # ── At p_c: fit |M| ~ L^(-beta/nu) ──────────────────────────────────────
    # Find nearest pc in grid
    pc_grid = min(PC_FINE, key=lambda x: abs(x-p_c))
    print(f'\nUsing p_c grid point = {pc_grid:.3f} for exponent fits')

    Ls_at_pc = []; Ms_at_pc = []; chis_at_pc = []
    for L in L_FSS:
        k = f'L{L}_pc{pc_grid:.3f}'
        if k in agg:
            Ls_at_pc.append(L)
            Ms_at_pc.append(agg[k]['absM'])
            chis_at_pc.append(agg[k]['chi'])

    beta_over_nu, _ = power_fit(Ls_at_pc, Ms_at_pc)
    gamma_over_nu, _ = power_fit(Ls_at_pc, chis_at_pc)
    print(f'At p_c: |M| ~ L^({beta_over_nu:.3f})  ->  beta/nu = {-beta_over_nu:.3f}')
    print(f'At p_c: chi ~ L^({gamma_over_nu:.3f})  ->  gamma/nu = {gamma_over_nu:.3f}')

    # ── dU4/dP at p_c ~ L^(1/nu) ─────────────────────────────────────────────
    print('\nEstimating 1/nu from dU4/dP at p_c:')
    # numerical derivative using central differences around pc_grid
    pc_idx = PC_FINE.index(pc_grid)
    dU4_dP = {}
    for L in L_FSS:
        pc_lo = PC_FINE[max(0, pc_idx-2)]
        pc_hi = PC_FINE[min(len(PC_FINE)-1, pc_idx+2)]
        k_lo = f'L{L}_pc{pc_lo:.3f}'; k_hi = f'L{L}_pc{pc_hi:.3f}'
        if k_lo in agg and k_hi in agg and pc_hi > pc_lo:
            dU4_dP[L] = (agg[k_hi]['U4'] - agg[k_lo]['U4']) / (pc_hi - pc_lo)
        else:
            dU4_dP[L] = float('nan')
        print(f'  L={L}: dU4/dP = {dU4_dP[L]:.3f}')

    Ls_nu = [L for L in L_FSS if not math.isnan(dU4_dP.get(L, float('nan')))]
    dU4s  = [dU4_dP[L] for L in Ls_nu]
    one_over_nu, _ = power_fit(Ls_nu, [abs(x) for x in dU4s])
    nu_est = 1./one_over_nu if one_over_nu > 0 else float('nan')
    print(f'dU4/dP ~ L^({one_over_nu:.3f})  ->  1/nu = {one_over_nu:.3f},  nu = {nu_est:.3f}')

    # ── Derive beta and gamma ─────────────────────────────────────────────────
    beta_est  = -beta_over_nu * nu_est  if not math.isnan(nu_est) else float('nan')
    gamma_est =  gamma_over_nu * nu_est if not math.isnan(nu_est) else float('nan')
    print(f'\nCritical exponents:')
    print(f'  p_c        = {p_c:.4f}')
    print(f'  beta/nu    = {-beta_over_nu:.3f}')
    print(f'  gamma/nu   = {gamma_over_nu:.3f}')
    print(f'  1/nu       = {one_over_nu:.3f}')
    print(f'  nu         = {nu_est:.3f}')
    print(f'  beta       = {beta_est:.3f}')
    print(f'  gamma      = {gamma_est:.3f}')
    print(f'\nComparison:')
    print(f'  Mean field:        beta=0.500, nu=0.500, gamma=1.000')
    print(f'  2D Ising:          beta=0.125, nu=1.000, gamma=1.750')
    print(f'  Directed perc.:    beta=0.276, nu_perp=0.735, gamma=0.000 (absorbing)')
    print(f'  This system:       beta={beta_est:.3f}, nu={nu_est:.3f}, gamma={gamma_est:.3f}')

    # ── Data collapse quality ─────────────────────────────────────────────────
    # Build collapsed data: x = (P - p_c)*L^(1/nu), y = |M|*L^(beta/nu)
    collapse_data = []
    for L in L_FSS:
        for pc in PC_FINE:
            k = f'L{L}_pc{pc:.3f}'
            if k not in agg: continue
            if math.isnan(agg[k]['absM']): continue
            x = (pc - p_c) * L**(one_over_nu) if not math.isnan(one_over_nu) else float('nan')
            y = agg[k]['absM'] * L**(-beta_over_nu) if not math.isnan(beta_over_nu) else float('nan')
            collapse_data.append(dict(L=L, pc=pc, x=x, y=y,
                                      U4=agg[k]['U4'], absM=agg[k]['absM']))

    # Save everything
    analysis = dict(
        p_c=p_c, p_c_se=p_c_se,
        beta_over_nu=-beta_over_nu,
        gamma_over_nu=gamma_over_nu,
        one_over_nu=one_over_nu,
        nu=nu_est, beta=beta_est, gamma=gamma_est,
        pc_grid=pc_grid,
        fss=agg,
        collapse=collapse_data,
        pc_estimates=pc_estimates,
    )
    with open(ANALYSIS_FILE, 'w') as f: json.dump(analysis, f, indent=2)
    print(f'\nSaved -> {ANALYSIS_FILE}')
