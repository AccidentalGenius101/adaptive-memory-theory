"""
Paper 65: Algebraic Order, Spin Correlator, and VCSM Ablation.

Three phases:

Phase A: Spin-field spatial correlator + anomalous dimension eta.
  The phi-field correlator failed in Paper 64 (low amplitude).
  Here we use the Ising spin field s (values +/-1, full SNR) to compute:
    G_s(r) = <s(y)*s(y+r)>_c   within zone-0 columns (y-axis autocorr)
  Fit to BOTH:
    Power law:   G_s(r) = A * r^{-eta}       [BKT / algebraic quasi-LRO]
    Exponential: G_s(r) = B * exp(-r / xi)   [disordered / exponential]
  Compare R^2 to determine which fits better at each (L, P_causal).
  Also compute chi_spin = C0 * sum_r G_norm(r) (true susceptibility).
  Also compute K_phi = Var(phi_row_mean) * L^2 / Var(phi) as phi stiffness proxy.
  L in {60, 80, 100}, P in {0, 0.005, 0.010, 0.020, 0.050}, 12 seeds, 12000 steps.

  Extended Binder (Phase A2): L in {120, 160},
  P in {0.000, 0.0005, 0.001, 0.002, 0.005}, 8 seeds, 10000 steps.
  Tests whether p_c = 0+ holds at even smaller P and larger L.

Phase B: phi-field spatial coherence (stiffness proxy).
  Same runs as Phase A. Computed simultaneously.
  K_phi(L, P) = Var(phi_row_mean_z0) * L^2 / Var(phi_z0)
  In algebraic (BKT): K_phi grows with L (long-range coherence).
  In disordered: K_phi ~ O(1) or decreases.

Phase C: VCSM-lite ablation study within Ising system.
  Five conditions at L=80, P_causal=0.020 (above p_c), 12 seeds, 8000 steps:
    Ref:          Full system (SS=8, BETA_BASE=0.005, FIELD_DECAY=0.999, P=0.020)
    NoGate:       SS=0   (viability gate disabled -- all writes go through)
    NoBaseline:   BETA_BASE=0 (no contrastive tracking -- mid_mem = s directly)
    Crystallized: FIELD_DECAY=1.0 (phi never decays -- cockroach mode)
    NoCausal:     P_causal=0.000 (pure noise -- disorder reference)
  Measures: U4, |M|, G_s(r=1) (nearest-neighbor spin correlation proxy).
  Tests which VCSM component is load-bearing for ordering.
"""

import numpy as np, json, os, math, multiprocessing as mp
from pathlib import Path

# ── Physical parameters (shared) ─────────────────────────────────────────────
J           = 1.0
FA          = 0.30
FIELD_DECAY = 0.999
MID_DECAY   = 0.97
SS          = 8
BETA_BASE   = 0.005
WAVE_RADIUS = 5
WAVE_DUR    = 5
EXT_FIELD   = 1.5
T_FIXED     = 3.0
L_BASE      = 40
WE_BASE     = 25

# ── Phase A1: spin correlator ─────────────────────────────────────────────────
L_A1     = [60, 80, 100]
PC_A1    = [round(x, 4) for x in [0.000, 0.005, 0.010, 0.020, 0.050]]
SEEDS_A1 = list(range(12))
NSTEPS_A1= 12000
SS_FRAC  = 0.40

# ── Phase A2: extended Binder (tiny P, large L) ───────────────────────────────
L_A2     = [120, 160]
PC_A2    = [round(x, 4) for x in [0.000, 0.0005, 0.001, 0.002, 0.005]]
SEEDS_A2 = list(range(8))
NSTEPS_A2= 10000

# ── Phase C: ablation ─────────────────────────────────────────────────────────
L_C      = 80
SEEDS_C  = list(range(12))
NSTEPS_C = 8000
CONDITIONS_C = {
    'Ref':          dict(ss=8,  beta=0.005, fd=0.999, pc=0.020),
    'NoGate':       dict(ss=0,  beta=0.005, fd=0.999, pc=0.020),
    'NoBaseline':   dict(ss=8,  beta=0.000, fd=0.999, pc=0.020),
    'Crystallized': dict(ss=8,  beta=0.005, fd=1.000, pc=0.020),
    'NoCausal':     dict(ss=8,  beta=0.005, fd=0.999, pc=0.000),
}

RESULTS_FILE  = Path(__file__).parent / 'results' / 'paper65_results.json'
ANALYSIS_FILE = Path(__file__).parent / 'results' / 'paper65_analysis.json'


def we(L):
    return max(1, round(WE_BASE * (L_BASE / L) ** 2))


# ── Ising + VCSM-lite kernel ──────────────────────────────────────────────────
def build_nb(L):
    N = L * L
    row = np.arange(N) // L
    col = np.arange(N) % L
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


# ── Spin autocorrelator ───────────────────────────────────────────────────────
def spin_correlator(s_2d, L):
    """
    Compute spin-spin autocorrelation G(r) along y within zone 0 (left half).
    Returns G_norm (normalized), eta, xi, chi_spin, r2_powerlaw, r2_exp.
    """
    s_z0 = s_2d[:, :L // 2]      # shape (L, L//2)
    C_total = np.zeros(L)
    for x in range(L // 2):
        col   = s_z0[:, x].astype(float)
        col_d = col - col.mean()
        f     = np.fft.rfft(col_d, n=L)
        acorr = np.fft.irfft(f * np.conj(f), n=L).real
        C_total += acorr
    C_total /= (L * (L // 2))

    r_max  = L // 2
    C_half = C_total[:r_max]
    C0     = C_half[0]

    if C0 <= 1e-12:
        return dict(G_norm=[0.]*r_max, eta=float('nan'), xi=float('nan'),
                    chi_spin=float('nan'), r2_pow=float('nan'), r2_exp=float('nan'))

    C_norm = C_half / C0
    chi_spin = float(C0 * (C_norm[0] + 2. * C_norm[1:].sum()))

    r_arr  = np.arange(1, r_max)
    C_fit  = np.maximum(C_norm[1:], 1e-12)
    mask   = C_fit > 0.02    # only fit where signal is >2% of peak

    eta = xi = r2_pow = r2_exp = float('nan')

    if mask.sum() >= 4:
        logr = np.log(r_arr[mask])
        logC = np.log(C_fit[mask])

        # Power-law fit: log G = log A - eta * log r
        try:
            pp   = np.polyfit(logr, logC, 1)
            eta  = float(-pp[0])
            resid_pow = logC - np.polyval(pp, logr)
            ss_res_pow = float((resid_pow**2).sum())
            ss_tot     = float(((logC - logC.mean())**2).sum())
            r2_pow = float(1. - ss_res_pow / ss_tot) if ss_tot > 0 else float('nan')
        except Exception:
            pass

        # Exponential fit: log G = log B - r/xi
        try:
            pe  = np.polyfit(r_arr[mask], logC, 1)
            xi  = float(-1. / pe[0]) if pe[0] < -1e-6 else float(r_max)
            resid_exp = logC - np.polyval(pe, r_arr[mask])
            ss_res_exp = float((resid_exp**2).sum())
            r2_exp = float(1. - ss_res_exp / ss_tot) if ss_tot > 0 else float('nan')
        except Exception:
            pass

    return dict(G_norm=C_norm[:20].tolist(), eta=eta, xi=xi,
                chi_spin=chi_spin, r2_pow=r2_pow, r2_exp=r2_exp)


# ── phi stiffness proxy ───────────────────────────────────────────────────────
def phi_stiffness(phi_2d, L):
    """
    K_phi = Var(phi_row_mean_z0) * L^2 / (Var(phi_z0) + eps)
    In BKT ordered phase: K_phi grows (or stays large) with L.
    In disordered: K_phi ~ O(1).
    """
    phi_z0      = phi_2d[:, :L // 2]   # (L, L//2)
    row_means   = phi_z0.mean(axis=1)  # (L,)
    var_row     = float(np.var(row_means))
    var_phi     = float(np.var(phi_z0))
    K_phi       = (var_row * L * L) / (var_phi + 1e-12)
    return float(K_phi)


# ── Single run (Phase A1) ─────────────────────────────────────────────────────
def run_A(L, P_causal, seed, nsteps=None):
    if nsteps is None:
        nsteps = NSTEPS_A1
    rng    = np.random.RandomState(seed)
    nb     = build_nb(L)
    N      = L * L; N_zone = N // 2
    col_a  = np.arange(N) % L
    zone   = (col_a >= L // 2).astype(int)
    wave_e = we(L)

    s      = rng.choice([-1, 1], N).astype(float)
    base   = s * 0.1
    mid    = np.zeros(N)
    phi    = np.zeros(N)
    streak = np.zeros(N, int)
    wave_z = 0
    ss0    = int(nsteps * SS_FRAC)
    Mseries= []

    # Accumulators for spin correlator and phi stiffness
    G_acc  = np.zeros(L)
    G_cnt  = 0
    K_acc  = 0.
    K_cnt  = 0

    for t in range(nsteps):
        s = metro(s, nb, T_FIXED, rng)
        base  += BETA_BASE * (s - base)
        same   = (s > 0) == (base > 0)
        streak = np.where(same, streak + 1, 0)
        mid   *= MID_DECAY

        if t % wave_e == 0:
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

            # Accumulate spin correlator every 50 steps
            if (t - ss0) % 50 == 0:
                s_2d = s.reshape(L, L)
                s_z0 = s_2d[:, :L // 2].astype(float)
                for x in range(L // 2):
                    col_d = s_z0[:, x] - s_z0[:, x].mean()
                    f     = np.fft.rfft(col_d, n=L)
                    acorr = np.fft.irfft(f * np.conj(f), n=L).real
                    G_acc += acorr
                G_cnt += L // 2

                # phi stiffness
                phi_2d = phi.reshape(L, L)
                K_acc += phi_stiffness(phi_2d, L)
                K_cnt += 1

    # Finalise spin correlator
    if G_cnt > 0:
        C_avg  = G_acc / (G_cnt * L)
        corr   = spin_correlator.__func__ if hasattr(spin_correlator, '__func__') else spin_correlator
        # direct computation from C_avg
        C0     = C_avg[0]
        r_max  = L // 2
        C_half = C_avg[:r_max]
        if C0 > 1e-12:
            C_norm   = C_half / C0
            chi_spin = float(C0 * (C_norm[0] + 2. * C_norm[1:].sum()))
            r_arr    = np.arange(1, r_max)
            C_fit    = np.maximum(C_norm[1:], 1e-12)
            mask     = C_fit > 0.02
            eta = xi = r2_pow = r2_exp = float('nan')
            ss_tot   = float(((np.log(C_fit[mask]) - np.log(C_fit[mask]).mean())**2).sum()) if mask.sum() >= 4 else 0.
            if mask.sum() >= 4:
                logr = np.log(r_arr[mask]); logC = np.log(C_fit[mask])
                try:
                    pp  = np.polyfit(logr, logC, 1); eta = float(-pp[0])
                    r2_pow = float(1. - ((logC - np.polyval(pp, logr))**2).sum() / ss_tot) if ss_tot > 0 else float('nan')
                except Exception: pass
                try:
                    pe  = np.polyfit(r_arr[mask], logC, 1)
                    xi  = float(-1./pe[0]) if pe[0] < -1e-6 else float(r_max)
                    r2_exp = float(1. - ((logC - np.polyval(pe, r_arr[mask]))**2).sum() / ss_tot) if ss_tot > 0 else float('nan')
                except Exception: pass
            G_norm_list = C_norm[:20].tolist()
        else:
            chi_spin = eta = xi = r2_pow = r2_exp = float('nan')
            G_norm_list = []
    else:
        chi_spin = eta = xi = r2_pow = r2_exp = float('nan')
        G_norm_list = []

    K_phi = K_acc / K_cnt if K_cnt > 0 else float('nan')

    M_arr = np.array(Mseries)
    M2    = float(np.mean(M_arr ** 2)); M4 = float(np.mean(M_arr ** 4))
    absM  = float(np.mean(np.abs(M_arr)))
    U4    = float(1. - M4 / (3. * M2 ** 2)) if M2 > 1e-12 else float('nan')
    chi_zone = float(N_zone * np.var(M_arr))

    return dict(absM=absM, M2=M2, M4=M4, U4=U4, chi_zone=chi_zone,
                chi_spin=chi_spin, eta=eta, xi=xi,
                r2_pow=r2_pow, r2_exp=r2_exp,
                K_phi=K_phi, G_norm=G_norm_list)

def _wA1(args):
    L, pc, seed = args
    return ('A1', L, pc, seed, run_A(L, pc, seed, NSTEPS_A1))

def _wA2(args):
    L, pc, seed = args
    return ('A2', L, pc, seed, run_A(L, pc, seed, NSTEPS_A2))


# ── Single run (Phase C: ablation) ───────────────────────────────────────────
def run_C(L, seed, cond_name, cond):
    rng    = np.random.RandomState(seed)
    nb     = build_nb(L)
    N      = L * L; N_zone = N // 2
    col_a  = np.arange(N) % L
    zone   = (col_a >= L // 2).astype(int)
    wave_e = we(L)

    ss_val    = cond['ss']
    beta_val  = cond['beta']
    fd_val    = cond['fd']
    pc_val    = cond['pc']

    s      = rng.choice([-1, 1], N).astype(float)
    base   = s * 0.1
    mid    = np.zeros(N)
    phi    = np.zeros(N)
    streak = np.zeros(N, int)
    wave_z = 0
    ss0    = int(NSTEPS_C * SS_FRAC)
    Mseries= []

    # spin nn-correlator accumulator
    G1_acc = 0.; G1_cnt = 0

    for t in range(NSTEPS_C):
        s = metro(s, nb, T_FIXED, rng)
        if beta_val > 0:
            base += beta_val * (s - base)
        same   = (s > 0) == (base > 0)
        streak = np.where(same, streak + 1, 0)
        mid   *= MID_DECAY

        if t % wave_e == 0:
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
                cau = rng.random(len(hit)) < pc_val
                nse = rng.normal(0, float(np.std(dev)) + 0.5, len(hit))
                mid[hit] += FA * np.where(cau, sig, nse)
            wave_z = 1 - wave_z

        # Gate: ss_val=0 means gate always open
        if ss_val == 0:
            gate = np.ones(N, bool)
        else:
            gate = streak >= ss_val
        if gate.any():
            wi = np.where(gate)[0]
            phi[wi] += FA * (mid[wi] - phi[wi])
            if ss_val > 0:
                streak[wi] = 0
        phi *= fd_val

        if t >= ss0:
            M = float(np.mean(phi[zone == 0])) - float(np.mean(phi[zone == 1]))
            Mseries.append(M)

            # Nearest-neighbour spin correlation within zone 0 as BKT proxy
            if (t - ss0) % 50 == 0:
                s_2d  = s.reshape(L, L)
                s_z0  = s_2d[:, :L // 2].astype(float)
                # G(r=1) = mean over all (y,x) of s(y,x)*s(y+1,x) - <s>^2
                mean_s = s_z0.mean()
                G1     = float((s_z0 * np.roll(s_z0, -1, axis=0)).mean()) - mean_s ** 2
                G1_acc += G1; G1_cnt += 1

    M_arr = np.array(Mseries)
    M2    = float(np.mean(M_arr ** 2)); M4 = float(np.mean(M_arr ** 4))
    absM  = float(np.mean(np.abs(M_arr)))
    U4    = float(1. - M4 / (3. * M2 ** 2)) if M2 > 1e-12 else float('nan')
    G1    = G1_acc / G1_cnt if G1_cnt > 0 else float('nan')

    return dict(absM=absM, M2=M2, M4=M4, U4=U4, G1=G1, cond=cond_name)

def _wC(args):
    seed, cond_name = args
    return ('C', L_C, cond_name, seed, run_C(L_C, seed, cond_name, CONDITIONS_C[cond_name]))


# ── MAIN ──────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    mp.freeze_support()
    RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)

    raw = {}
    if RESULTS_FILE.exists():
        with open(RESULTS_FILE) as f:
            raw = json.load(f)

    # ── Phase A1 ──────────────────────────────────────────────────────────────
    if 'A1' not in raw:
        args_A1 = [(L, pc, seed) for L in L_A1 for pc in PC_A1 for seed in SEEDS_A1]
        print(f'Phase A1: {len(args_A1)} runs (spin correlator + phi stiffness)...')
        print('WAVE_EVERY:', {L: we(L) for L in L_A1})
        with mp.Pool(processes=min(len(args_A1), mp.cpu_count())) as pool:
            res = pool.map(_wA1, args_A1)
        raw['A1'] = {}
        for _, L, pc, seed, r in res:
            k = f'L{L}_pc{pc:.4f}'
            raw['A1'].setdefault(k, {})[str(seed)] = r
        with open(RESULTS_FILE, 'w') as f:
            json.dump(raw, f)
        print('Phase A1 done.')

    # ── Phase A2 ──────────────────────────────────────────────────────────────
    if 'A2' not in raw:
        args_A2 = [(L, pc, seed) for L in L_A2 for pc in PC_A2 for seed in SEEDS_A2]
        print(f'\nPhase A2: {len(args_A2)} runs (extended Binder, large L)...')
        print('WAVE_EVERY:', {L: we(L) for L in L_A2})
        with mp.Pool(processes=min(len(args_A2), mp.cpu_count())) as pool:
            res = pool.map(_wA2, args_A2)
        raw['A2'] = {}
        for _, L, pc, seed, r in res:
            k = f'L{L}_pc{pc:.4f}'
            raw['A2'].setdefault(k, {})[str(seed)] = r
        with open(RESULTS_FILE, 'w') as f:
            json.dump(raw, f)
        print('Phase A2 done.')

    # ── Phase C ───────────────────────────────────────────────────────────────
    if 'C' not in raw:
        args_C = [(seed, cn) for cn in CONDITIONS_C for seed in SEEDS_C]
        print(f'\nPhase C: {len(args_C)} runs (VCSM ablation)...')
        with mp.Pool(processes=min(len(args_C), mp.cpu_count())) as pool:
            res = pool.map(_wC, args_C)
        raw['C'] = {}
        for _, L, cond_name, seed, r in res:
            raw['C'].setdefault(cond_name, {})[str(seed)] = r
        with open(RESULTS_FILE, 'w') as f:
            json.dump(raw, f)
        print('Phase C done.')

    # ── Aggregate Phase A1 ────────────────────────────────────────────────────
    agg_A1 = {}
    for L in L_A1:
        for pc in PC_A1:
            k = f'L{L}_pc{pc:.4f}'
            if k not in raw['A1']:
                continue
            vals = list(raw['A1'][k].values())
            eta_v = [v['eta']      for v in vals if not math.isnan(v.get('eta', float('nan')))]
            xi_v  = [v['xi']       for v in vals if not math.isnan(v.get('xi',  float('nan')))]
            cs_v  = [v['chi_spin'] for v in vals if not math.isnan(v.get('chi_spin', float('nan')))]
            rp_v  = [v['r2_pow']   for v in vals if not math.isnan(v.get('r2_pow',  float('nan')))]
            re_v  = [v['r2_exp']   for v in vals if not math.isnan(v.get('r2_exp',  float('nan')))]
            kp_v  = [v['K_phi']    for v in vals if not math.isnan(v.get('K_phi',   float('nan')))]
            agg_A1[k] = dict(L=L, pc=pc,
                absM     = ms([v['absM'] for v in vals]),
                U4       = ms([v['U4']   for v in vals]),
                U4_se    = ses([v['U4']  for v in vals]),
                eta      = ms(eta_v), xi  = ms(xi_v),
                chi_spin = ms(cs_v),
                r2_pow   = ms(rp_v),  r2_exp = ms(re_v),
                K_phi    = ms(kp_v))

    print('\n=== PHASE A1: Spin Correlator ===')
    print(f'{"L":>5} {"P":>7} {"eta":>7} {"xi":>7} {"R2_pow":>8} {"R2_exp":>8} '
          f'{"chi_s":>9} {"K_phi":>8} {"U4":>7}')
    for L in L_A1:
        for pc in PC_A1:
            k = f'L{L}_pc{pc:.4f}'
            if k not in agg_A1:
                continue
            v = agg_A1[k]
            print(f'{L:>5} {pc:>7.4f} {v["eta"]:>7.3f} {v["xi"]:>7.2f} '
                  f'{v["r2_pow"]:>8.4f} {v["r2_exp"]:>8.4f} '
                  f'{v["chi_spin"]:>9.5f} {v["K_phi"]:>8.3f} {v["U4"]:>7.4f}')

    # Which fit wins?
    print('\nFit comparison (power law vs exponential -- which R^2 is higher?):')
    for L in L_A1:
        for pc in PC_A1:
            if pc == 0.000:
                continue
            k = f'L{L}_pc{pc:.4f}'
            if k not in agg_A1:
                continue
            v = agg_A1[k]
            rp = v['r2_pow']; re = v['r2_exp']
            if math.isnan(rp) or math.isnan(re):
                winner = 'UNDETERMINED'
            elif rp > re + 0.02:
                winner = 'POWER-LAW (BKT)'
            elif re > rp + 0.02:
                winner = 'EXPONENTIAL (disorder)'
            else:
                winner = 'AMBIGUOUS'
            print(f'  L={L}, P={pc:.4f}: R2_pow={rp:.4f}, R2_exp={re:.4f}  -> {winner}')

    # K_phi vs L at fixed P
    print('\nK_phi vs L (stiffness: grows -> BKT, flat -> disorder):')
    for pc in [0.010, 0.020, 0.050]:
        row = []
        for L in L_A1:
            k = f'L{L}_pc{pc:.4f}'
            if k in agg_A1:
                row.append(f'L{L}:{agg_A1[k]["K_phi"]:.2f}')
        print(f'  P={pc:.3f}: ' + '  '.join(row))

    # ── Aggregate Phase A2 ────────────────────────────────────────────────────
    agg_A2 = {}
    for L in L_A2:
        for pc in PC_A2:
            k = f'L{L}_pc{pc:.4f}'
            if k not in raw['A2']:
                continue
            vals = list(raw['A2'][k].values())
            agg_A2[k] = dict(L=L, pc=pc,
                absM  = ms([v['absM'] for v in vals]),
                U4    = ms([v['U4']   for v in vals]),
                U4_se = ses([v['U4']  for v in vals]))

    print('\n=== PHASE A2: Extended Binder (Tiny P, Large L) ===')
    print(f'{"P":>8}', end='')
    for L in L_A2:
        print(f'  L{L}:U4 ', end='')
    print()
    for pc in PC_A2:
        print(f'{pc:>8.4f}', end='')
        for L in L_A2:
            k = f'L{L}_pc{pc:.4f}'
            u4 = agg_A2[k]['U4'] if k in agg_A2 else float('nan')
            print(f'  {u4:>8.4f} ', end='')
        print()

    # Check if U4(L=160) > U4(L=120) for ALL tiny P > 0
    print('\nU4 ordering (L=160 vs L=120):')
    for pc in PC_A2:
        if pc == 0.000:
            continue
        k120 = f'L120_pc{pc:.4f}'; k160 = f'L160_pc{pc:.4f}'
        if k120 in agg_A2 and k160 in agg_A2:
            u120 = agg_A2[k120]['U4']; u160 = agg_A2[k160]['U4']
            tag  = 'ORDERED' if u160 > u120 else 'DISORDERED'
            print(f'  P={pc:.4f}: U4(120)={u120:.4f}, U4(160)={u160:.4f}  -> {tag}')

    # ── Aggregate Phase C ─────────────────────────────────────────────────────
    agg_C = {}
    for cn in CONDITIONS_C:
        if cn not in raw['C']:
            continue
        vals = list(raw['C'][cn].values())
        agg_C[cn] = dict(
            absM  = ms([v['absM'] for v in vals]),
            U4    = ms([v['U4']   for v in vals]),
            U4_se = ses([v['U4']  for v in vals]),
            G1    = ms([v['G1']   for v in vals if not math.isnan(v.get('G1', float('nan')))]))

    print('\n=== PHASE C: VCSM-lite Ablation ===')
    print(f'{"Condition":>15} {"U4":>8} {"|M|":>8} {"G(r=1)":>10}')
    print('-' * 45)
    for cn in ['Ref', 'NoGate', 'NoBaseline', 'Crystallized', 'NoCausal']:
        if cn not in agg_C:
            continue
        v = agg_C[cn]
        print(f'{cn:>15} {v["U4"]:>8.4f} {v["absM"]:>8.5f} {v["G1"]:>10.5f}')

    analysis = dict(A1=agg_A1, A2=agg_A2, C=agg_C)
    with open(ANALYSIS_FILE, 'w') as f:
        json.dump(analysis, f, indent=2)
    print(f'\nSaved -> {ANALYSIS_FILE}')
