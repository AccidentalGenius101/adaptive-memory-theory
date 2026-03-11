"""
Paper 64: Is p_c = 0? BKT Test, Spatial Correlator, and Response Susceptibility.

Paper 63 raised two open questions:
  (1) Is p_c = 0+? (Binder ordering at ALL P > 0, even P=0.002)
  (2) What is gamma? (chi = N*Var(M) self-averages; need spatial correlator)

This paper performs:

Phase A: Large-L Binder crossing at very small P
  L in {40, 60, 80, 100, 120}, P in [0.000, 0.030], 12 seeds, 12000 steps.
  If U4(L=120) > U4(L=40) for ALL P > 0 -> p_c = 0+ (no disorder phase).
  If crossings converge to a finite p_c -> standard second-order transition.

Phase B: Spatial phi-phi correlator within zone 0
  L in {60, 80, 100}, P in [0.000, 0.005, 0.010, 0.020, 0.050}, 12 seeds.
  Compute C(Dy) = <phi(y) phi(y+Dy)>_c along y-axis, fit to exp(-r/xi).
  If xi > L for all P > 0 -> algebraic (BKT quasi-LRO).
  If xi saturates at finite value below p_c -> exponential (disordered phase exists).
  chi_true = sum_r C(r) is the correct susceptibility.

Both phases use extensive-drive scaling (WAVE_EVERY = 25*(40/L)^2).
"""
import numpy as np, json, os, math, multiprocessing as mp
from pathlib import Path

# ── Physical parameters ──────────────────────────────────────────────────────
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
WE_BASE     = 25

# ── Phase A: large-L Binder ──────────────────────────────────────────────────
L_A    = [40, 60, 80, 100, 120]
PC_A   = [round(x, 3) for x in
          [0.000, 0.002, 0.004, 0.006, 0.008, 0.010, 0.015, 0.020, 0.030]]
SEEDS_A  = list(range(12))
NSTEPS_A = 12000
SS_FRAC  = 0.40

# ── Phase B: spatial correlator ──────────────────────────────────────────────
L_B    = [60, 80, 100]
PC_B   = [round(x, 3) for x in [0.000, 0.005, 0.010, 0.020, 0.050]]
SEEDS_B  = list(range(12))
NSTEPS_B = 12000

RESULTS_FILE = Path(__file__).parent / 'results' / 'paper64_results.json'
ANALYSIS_FILE= Path(__file__).parent / 'results' / 'paper64_analysis.json'


def we(L):
    return max(1, round(WE_BASE * (L_BASE / L)**2))


# ── Ising + VCSM-lite kernel ─────────────────────────────────────────────────
def build_nb(L):
    N = L*L; row = np.arange(N)//L; col = np.arange(N)%L
    return np.stack([((row-1)%L)*L+col, ((row+1)%L)*L+col,
                     row*L+(col-1)%L,   row*L+(col+1)%L], axis=1)

def metro(s, nb, T, rng, h_ext=None, hit=None):
    N = len(s); L = int(round(math.sqrt(N)))
    row = np.arange(N)//L; col = np.arange(N)%L
    h = np.zeros(N)
    if h_ext is not None and hit is not None: h[hit] = h_ext
    for sub in [0, 1]:
        idx = np.where((row+col)%2==sub)[0]
        ns = s[nb[idx]].sum(1)
        dE = 2.*J*s[idx]*ns - 2.*h[idx]*s[idx]
        acc = (dE<=0)|(rng.random(len(idx)) < np.exp(-np.clip(dE/T,-20,20)))
        s[idx[acc]] *= -1
    return s

def wsites(cx, cy, r, L):
    N = L*L; row = np.arange(N)//L; col = np.arange(N)%L
    dx = np.minimum(np.abs(col-cx), L-np.abs(col-cx))
    dy = np.minimum(np.abs(row-cy), L-np.abs(row-cy))
    return np.where(dx+dy<=r)[0]

def ms(vals):
    v = [x for x in vals if x is not None and not (isinstance(x,float) and math.isnan(x))]
    return float(np.mean(v)) if v else float('nan')

def ses(vals):
    v = [x for x in vals if x is not None and not (isinstance(x,float) and math.isnan(x))]
    return float(np.std(v)/math.sqrt(len(v))) if len(v)>1 else 0.


# ── Spatial correlator (Phase B) ─────────────────────────────────────────────
def corr_length_from_phi(phi, L):
    """
    Compute phi-phi correlation along y within zone 0 (left half, col < L//2).
    Returns xi (exponential fit) and chi_true (sum of C(r)).
    """
    phi_2d = phi.reshape(L, L)          # phi_2d[row, col]
    phi_z0 = phi_2d[:, :L//2]          # shape (L, L//2) -- zone 0 columns

    C_total = np.zeros(L)
    for x in range(L//2):
        col_v  = phi_z0[:, x]           # L values along y
        col_d  = col_v - col_v.mean()
        f      = np.fft.rfft(col_d, n=L)
        acorr  = np.fft.irfft(f * np.conj(f), n=L).real
        C_total += acorr
    C_total /= (L * L//2)              # normalise: avg over columns, then by L
    # C_total[r] = (1/L) * sum_{y} cov(phi(y), phi(y+r)) averaged over columns

    # Take r = 0 .. L//2 (non-redundant half of periodic correlator)
    r_max  = L // 2
    C_half = C_total[:r_max]
    C0     = C_half[0]

    if C0 <= 1e-12:
        return float('nan'), float('nan'), C_half.tolist()

    C_norm = C_half / C0              # normalised: C_norm[0] = 1

    # chi_true = C0 * sum_r C_norm[r] * (1 for r=0, 2 for r>0 by symmetry, periodic)
    chi_true = float(C0 * (C_norm[0] + 2.0 * C_norm[1:].sum()))

    # Fit C_norm(r) = exp(-r / xi) for r >= 1
    r_arr  = np.arange(1, r_max)
    C_fit  = np.maximum(C_norm[1:], 1e-12)
    mask   = C_fit > 0.01
    xi     = float('nan')
    if mask.sum() >= 3:
        try:
            p = np.polyfit(r_arr[mask], np.log(C_fit[mask]), 1)
            xi = float(-1.0 / p[0]) if p[0] < -1e-6 else float(r_max)
        except Exception:
            pass

    return xi, chi_true, C_half.tolist()


# ── Single run (Phase A) ─────────────────────────────────────────────────────
def run_A(L, P_causal, seed):
    rng    = np.random.RandomState(seed)
    nb     = build_nb(L)
    N      = L*L; N_zone = N//2
    col_a  = np.arange(N)%L
    zone   = (col_a >= L//2).astype(int)
    wave_e = we(L)

    s      = rng.choice([-1,1], N).astype(float)
    base   = s * 0.1
    mid    = np.zeros(N)
    phi    = np.zeros(N)
    streak = np.zeros(N, int)
    wave_z = 0
    ss0    = int(NSTEPS_A * SS_FRAC)
    Mseries= []

    for t in range(NSTEPS_A):
        s = metro(s, nb, T_FIXED, rng)
        base  += BETA_BASE*(s-base)
        same   = (s>0)==(base>0)
        streak = np.where(same, streak+1, 0)
        mid   *= MID_DECAY

        if t % wave_e == 0:
            cx = rng.randint(0, L//2) if wave_z==0 else rng.randint(L//2, L)
            cy = rng.randint(L)
            hit = wsites(cx, cy, WAVE_RADIUS, L)
            if len(hit)>0:
                he = EXT_FIELD if wave_z==0 else -EXT_FIELD
                for _ in range(WAVE_DUR):
                    s = metro(s, nb, T_FIXED, rng, h_ext=he, hit=hit)
                dev = s[hit]-base[hit]
                zm  = (zone[hit]==wave_z)
                sig = np.where(zm, dev, -dev)
                cau = rng.random(len(hit)) < P_causal
                nse = rng.normal(0, float(np.std(dev))+0.5, len(hit))
                mid[hit] += FA * np.where(cau, sig, nse)
            wave_z = 1-wave_z

        gate = streak >= SS
        if gate.any():
            wi = np.where(gate)[0]
            phi[wi] += FA*(mid[wi]-phi[wi]); streak[wi] = 0
        phi *= FIELD_DECAY

        if t >= ss0:
            M = float(np.mean(phi[zone==0])) - float(np.mean(phi[zone==1]))
            Mseries.append(M)

    M_arr = np.array(Mseries)
    M2 = float(np.mean(M_arr**2)); M4 = float(np.mean(M_arr**4))
    absM = float(np.mean(np.abs(M_arr)))
    U4   = float(1.-M4/(3.*M2**2)) if M2>1e-12 else float('nan')
    chi  = float(N_zone * np.var(M_arr))
    return dict(absM=absM, M2=M2, M4=M4, U4=U4, chi=chi)

def _wA(args):
    L, pc, seed = args
    return ('A', L, pc, seed, run_A(L, pc, seed))


# ── Single run (Phase B) — same as A but also computes spatial correlator ───
def run_B(L, P_causal, seed):
    rng    = np.random.RandomState(seed)
    nb     = build_nb(L)
    N      = L*L; N_zone = N//2
    col_a  = np.arange(N)%L
    zone   = (col_a >= L//2).astype(int)
    wave_e = we(L)

    s      = rng.choice([-1,1], N).astype(float)
    base   = s * 0.1
    mid    = np.zeros(N)
    phi    = np.zeros(N)
    streak = np.zeros(N, int)
    wave_z = 0
    ss0    = int(NSTEPS_B * SS_FRAC)
    Mseries= []

    # accumulators for time-averaged spatial correlator
    C_acc  = np.zeros(L//2)
    C_cnt  = 0

    for t in range(NSTEPS_B):
        s = metro(s, nb, T_FIXED, rng)
        base  += BETA_BASE*(s-base)
        same   = (s>0)==(base>0)
        streak = np.where(same, streak+1, 0)
        mid   *= MID_DECAY

        if t % wave_e == 0:
            cx = rng.randint(0, L//2) if wave_z==0 else rng.randint(L//2, L)
            cy = rng.randint(L)
            hit = wsites(cx, cy, WAVE_RADIUS, L)
            if len(hit)>0:
                he = EXT_FIELD if wave_z==0 else -EXT_FIELD
                for _ in range(WAVE_DUR):
                    s = metro(s, nb, T_FIXED, rng, h_ext=he, hit=hit)
                dev = s[hit]-base[hit]
                zm  = (zone[hit]==wave_z)
                sig = np.where(zm, dev, -dev)
                cau = rng.random(len(hit)) < P_causal
                nse = rng.normal(0, float(np.std(dev))+0.5, len(hit))
                mid[hit] += FA * np.where(cau, sig, nse)
            wave_z = 1-wave_z

        gate = streak >= SS
        if gate.any():
            wi = np.where(gate)[0]
            phi[wi] += FA*(mid[wi]-phi[wi]); streak[wi] = 0
        phi *= FIELD_DECAY

        if t >= ss0:
            M = float(np.mean(phi[zone==0])) - float(np.mean(phi[zone==1]))
            Mseries.append(M)
            # Accumulate spatial correlator every 50 steps to save time
            if (t - ss0) % 50 == 0:
                phi_2d = phi.reshape(L, L)
                phi_z0 = phi_2d[:, :L//2]
                for x in range(L//2):
                    col_d  = phi_z0[:, x] - phi_z0[:, x].mean()
                    f      = np.fft.rfft(col_d, n=L)
                    acorr  = np.fft.irfft(f * np.conj(f), n=L).real
                    C_acc += acorr[:L//2]
                C_cnt += L//2

    # Finalize correlator
    if C_cnt > 0:
        C_avg = C_acc / (C_cnt * L)   # normalize
        C0 = C_avg[0]
        if C0 > 1e-12:
            C_norm = C_avg / C0
            chi_true = float(C0 * (C_norm[0] + 2.*C_norm[1:].sum()))
            r_arr = np.arange(1, L//2)
            C_fit = np.maximum(C_norm[1:], 1e-12)
            mask  = C_fit > 0.01
            xi    = float('nan')
            if mask.sum() >= 3:
                try:
                    p = np.polyfit(r_arr[mask], np.log(C_fit[mask]), 1)
                    xi = float(-1.0/p[0]) if p[0] < -1e-6 else float(L//2)
                except Exception:
                    pass
        else:
            C_norm = C_avg.tolist(); chi_true = float('nan'); xi = float('nan')
        C_norm_list = C_norm.tolist() if hasattr(C_norm,'tolist') else list(C_norm)
    else:
        xi = float('nan'); chi_true = float('nan'); C_norm_list = []

    M_arr = np.array(Mseries)
    M2 = float(np.mean(M_arr**2)); M4 = float(np.mean(M_arr**4))
    absM = float(np.mean(np.abs(M_arr)))
    U4   = float(1.-M4/(3.*M2**2)) if M2>1e-12 else float('nan')
    chi_zone = float(N_zone * np.var(M_arr))
    return dict(absM=absM, M2=M2, M4=M4, U4=U4,
                chi_zone=chi_zone, chi_true=chi_true,
                xi=xi, C_norm=C_norm_list[:20])   # store first 20 lags

def _wB(args):
    L, pc, seed = args
    return ('B', L, pc, seed, run_B(L, pc, seed))


# ── MAIN ─────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    mp.freeze_support()
    RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)

    raw = {}
    if RESULTS_FILE.exists():
        with open(RESULTS_FILE) as f: raw = json.load(f)

    # Phase A
    if 'A' not in raw:
        args_A = [(L, pc, seed) for L in L_A for pc in PC_A for seed in SEEDS_A]
        print(f'Phase A: {len(args_A)} runs (large-L Binder)...')
        print('WAVE_EVERY:', {L: we(L) for L in L_A})
        with mp.Pool(processes=min(len(args_A), mp.cpu_count())) as pool:
            res = pool.map(_wA, args_A)
        raw['A'] = {}
        for _, L, pc, seed, r in res:
            k = f'L{L}_pc{pc:.3f}'
            raw['A'].setdefault(k, {})[str(seed)] = r
        with open(RESULTS_FILE, 'w') as f: json.dump(raw, f)
        print('Phase A done.')

    # Phase B
    if 'B' not in raw:
        args_B = [(L, pc, seed) for L in L_B for pc in PC_B for seed in SEEDS_B]
        print(f'\nPhase B: {len(args_B)} runs (spatial correlator)...')
        with mp.Pool(processes=min(len(args_B), mp.cpu_count())) as pool:
            res = pool.map(_wB, args_B)
        raw['B'] = {}
        for _, L, pc, seed, r in res:
            k = f'L{L}_pc{pc:.3f}'
            raw['B'].setdefault(k, {})[str(seed)] = r
        with open(RESULTS_FILE, 'w') as f: json.dump(raw, f)
        print('Phase B done.')

    # ── Aggregate Phase A ────────────────────────────────────────────────────
    agg_A = {}
    for L in L_A:
        for pc in PC_A:
            k = f'L{L}_pc{pc:.3f}'
            if k not in raw['A']: continue
            vals = list(raw['A'][k].values())
            agg_A[k] = dict(L=L, pc=pc,
                absM = ms([v['absM'] for v in vals]),
                U4   = ms([v['U4']   for v in vals]),
                U4_se= ses([v['U4']  for v in vals]),
                chi  = ms([v['chi']  for v in vals]))

    print('\n=== PHASE A: Large-L Binder ===')
    print(f'{"P":>8}', end='')
    for L in L_A: print(f'  L{L:3d}:U4 ', end='')
    print()
    for pc in PC_A:
        print(f'{pc:>8.3f}', end='')
        for L in L_A:
            k = f'L{L}_pc{pc:.3f}'
            u4 = agg_A[k]['U4'] if k in agg_A else float('nan')
            print(f'  {u4:>8.4f} ', end='')
        print()

    # Check ordering: is U4(L_large) > U4(L_small) for all P > 0?
    print('\nU4 ordering test (is L=120 > L=40 for ALL P >= 0.002?):')
    for pc in PC_A:
        if pc == 0.000: continue
        k40  = f'L40_pc{pc:.3f}'; k120 = f'L120_pc{pc:.3f}'
        if k40 in agg_A and k120 in agg_A:
            u40 = agg_A[k40]['U4']; u120 = agg_A[k120]['U4']
            tag = 'ORDERED' if u120 > u40 else 'DISORDERED' if u40 > u120 else 'EQUAL'
            print(f'  P={pc:.3f}: U4(40)={u40:.4f}, U4(120)={u120:.4f}  -> {tag}')

    # Binder crossings
    from itertools import combinations
    print('\nBinder crossings:')
    pc_ests = []
    for La, Lb in combinations(L_A, 2):
        u4_a = [agg_A.get(f'L{La}_pc{pc:.3f}', {}).get('U4', float('nan')) for pc in PC_A]
        u4_b = [agg_A.get(f'L{Lb}_pc{pc:.3f}', {}).get('U4', float('nan')) for pc in PC_A]
        diff = [b-a for a,b in zip(u4_a,u4_b)]
        for i in range(len(diff)-1):
            if not (math.isnan(diff[i]) or math.isnan(diff[i+1])):
                if diff[i]*diff[i+1] < 0:
                    t = -diff[i]/(diff[i+1]-diff[i])
                    pc_cross = PC_A[i] + t*(PC_A[i+1]-PC_A[i])
                    pc_ests.append(pc_cross)
                    print(f'  L={La} vs L={Lb}: p_c={pc_cross:.4f}')
    if pc_ests:
        print(f'  -> Mean p_c = {np.mean(pc_ests):.4f} +/- {np.std(pc_ests):.4f}')
        p_c_final = float(np.mean(pc_ests))
    else:
        print('  -> No crossing found! p_c likely = 0')
        p_c_final = 0.0

    # ── Aggregate Phase B ────────────────────────────────────────────────────
    agg_B = {}
    for L in L_B:
        for pc in PC_B:
            k = f'L{L}_pc{pc:.3f}'
            if k not in raw['B']: continue
            vals = list(raw['B'][k].values())
            xi_vals = [v['xi'] for v in vals if not math.isnan(v['xi'])]
            ct_vals = [v['chi_true'] for v in vals if not math.isnan(v.get('chi_true', float('nan')))]
            agg_B[k] = dict(L=L, pc=pc,
                absM    = ms([v['absM'] for v in vals]),
                U4      = ms([v['U4']   for v in vals]),
                xi      = ms(xi_vals) if xi_vals else float('nan'),
                xi_se   = ses(xi_vals) if xi_vals else 0.,
                chi_true= ms(ct_vals) if ct_vals else float('nan'))

    print('\n=== PHASE B: Spatial Correlator ===')
    print(f'{"L":>5} {"P":>7} {"xi":>8} {"xi_se":>7} {"chi_true":>10} {"|M|":>8} {"U4":>8}')
    for L in L_B:
        for pc in PC_B:
            k = f'L{L}_pc{pc:.3f}'
            if k not in agg_B: continue
            v = agg_B[k]
            print(f'{L:>5} {pc:>7.3f} {v["xi"]:>8.2f} {v["xi_se"]:>7.2f} '
                  f'{v["chi_true"]:>10.5f} {v["absM"]:>8.5f} {v["U4"]:>8.4f}')

    # Test: does xi scale with L (QLRO) or saturate (finite xi)?
    print('\nxi / L test (> 0.5 suggests xi > L = quasi-LRO / no disorder phase):')
    for L in L_B:
        for pc in PC_B:
            k = f'L{L}_pc{pc:.3f}'
            if k not in agg_B: continue
            xi = agg_B[k]['xi']
            ratio = xi / L if not math.isnan(xi) else float('nan')
            print(f'  L={L}, P={pc:.3f}: xi={xi:.2f}, xi/L={ratio:.3f}')

    # chi_true vs L at fixed P (should diverge if transition is second-order)
    print('\nchi_true vs L at P=0.020:')
    for L in L_B:
        k = f'L{L}_pc0.020'
        if k in agg_B:
            print(f'  L={L}: chi_true={agg_B[k]["chi_true"]:.5f}')

    analysis = dict(A=agg_A, B=agg_B, p_c=p_c_final,
                    pc_estimates=pc_ests)
    with open(ANALYSIS_FILE, 'w') as f: json.dump(analysis, f, indent=2)
    print(f'\nSaved -> {ANALYSIS_FILE}')
