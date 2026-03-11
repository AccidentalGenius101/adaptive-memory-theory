"""
Paper 62: Three Decisive Tests Beyond Paper 61.

Phase A: Extensive-Drive FSS
  Fix the thermodynamic limit identified in Paper 61.
  Scale WAVE_EVERY as (L_base/L)^2 so wave density per site is constant.
  Expected: |M| size-independent in ordered phase; Binder crossing at p_c;
  extractable critical exponents beta/nu, gamma/nu, 1/nu.

Phase B: Multi-Zone Relay Coupling
  L=40, K zones (K in {2,4,6,8}). Phi-diffusion D_phi added to allow
  zone-to-zone field communication (relay mechanism).
  Test: does phi-order propagate to far zones? Does coherence length
  grow with K and D_phi?
  Expected: with relay, S_phi(zone k) decays slower with zone distance k.

Phase C: VCML Birth-Seeding Ablation (minimal inline CML)
  Minimal VCML substrate (2D, HS=1, VCSM rule, mandatory turnover).
  Sweep FIELD_SEED_BETA in {0, 0.001, 0.003, 0.005, 0.01}.
  FIELD_SEED_BETA=0: new births get random hid (no copy-forward).
  FIELD_SEED_BETA>0: new births partially inherit neighbor fieldM.
  Test: does sg4 collapse to zero at FIELD_SEED_BETA=0?
  Expected: yes — reconstructive layer requires birth seeding.
  Confirms two-layer coexistence: attractor (local) survives; zone structure dies.
"""
import numpy as np, json, os, math, multiprocessing as mp, scipy.optimize as opt
from pathlib import Path

# ── Shared Ising + VCSM-lite kernel ───────────────────────────────────────────
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

# Phase A
L_FSS         = [20, 30, 40, 60]
L_BASE        = 40
WAVE_EVERY_BASE = 25
PC_FSS        = [round(x, 2) for x in np.arange(0.00, 0.65, 0.05)]
SEEDS_FSS     = list(range(5))
NSTEPS_FSS    = 8000
SS_START_FRAC = 0.40

# Phase B
L_B      = 40
K_ZONES  = [2, 4, 6, 8]
D_PHI    = [0.0, 0.02, 0.10]      # phi-diffusion coefficient
PC_B     = [0.20, 0.40, 0.60, 0.80, 1.00]
SEEDS_B  = list(range(5))
NSTEPS_B = 8000
WAVE_EVERY_B = 25

# Phase C — minimal VCML ablation
W_C          = 40
H_C          = 20
K_ZONES_C    = 4
NSTEPS_C     = 4000
SEEDS_C      = list(range(5))
BETA_C       = 0.005
MID_DECAY_C  = 0.99
FIELD_DECAY_C= 0.9997
FA_C         = 0.16
SS_C         = 10
WAVE_EVERY_C = 20
WAVE_RAD_C   = 4
WAVE_AMP_C   = 0.50
TURNOVER_RATE= 0.005           # fraction of sites that reset per step
FSB_VALS     = [0.0, 0.001, 0.003, 0.005, 0.01]   # FIELD_SEED_BETA sweep

RESULTS_FILE  = Path(__file__).parent / 'results' / 'paper62_results.json'
ANALYSIS_FILE = Path(__file__).parent / 'results' / 'paper62_analysis.json'


# ── Ising helpers ──────────────────────────────────────────────────────────────
def build_neighbors_ising(L):
    N = L * L
    row = np.arange(N) // L; col = np.arange(N) % L
    return np.stack([((row-1)%L)*L+col, ((row+1)%L)*L+col,
                     row*L+(col-1)%L,   row*L+(col+1)%L], axis=1)

def metro_step(s, nb, T, rng, h_ext=None, hit=None):
    N = len(s); L = int(round(math.sqrt(N)))
    row = np.arange(N)//L; col = np.arange(N)%L
    h = np.zeros(N)
    if h_ext is not None and hit is not None: h[hit] = h_ext
    for sub in [0,1]:
        idx = np.where((row+col)%2==sub)[0]
        nb_sum = s[nb[idx]].sum(1)
        dE = 2.*J*s[idx]*nb_sum - 2.*h[idx]*s[idx]
        acc = (dE<=0)|(rng.random(len(idx)) < np.exp(-np.clip(dE/T,-20,20)))
        s[idx[acc]] *= -1
    return s

def wave_sites_2d(cx, cy, r, L):
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


# ══════════════════════════════════════════════════════════════════════════════
# PHASE A: Extensive-Drive FSS
# ══════════════════════════════════════════════════════════════════════════════
def wave_every_for_L(L):
    """Scale WAVE_EVERY so wave density per site is constant across L."""
    return max(1, round(WAVE_EVERY_BASE * (L_BASE / L)**2))


def run_A(L, P_causal, seed, n_steps):
    """Ising+VCSM-lite with extensive-drive wave scaling."""
    rng    = np.random.RandomState(seed)
    nb     = build_neighbors_ising(L)
    N      = L*L
    col_a  = np.arange(N) % L
    zone   = (col_a >= L//2).astype(int)
    N_zone = N//2
    wave_every = wave_every_for_L(L)

    s      = rng.choice([-1,1], N).astype(float)
    base   = s * 0.1
    mid    = np.zeros(N)
    phi    = np.zeros(N)
    streak = np.zeros(N, int)
    wave_z = 0
    ss_start = int(n_steps * SS_START_FRAC)
    M_series = []

    for t in range(n_steps):
        s = metro_step(s, nb, T_FIXED, rng)
        base  += BETA_BASE*(s-base)
        same   = (s>0)==(base>0)
        streak = np.where(same, streak+1, 0)
        mid   *= MID_DECAY

        if t % wave_every == 0:
            cx = rng.randint(0, L//2) if wave_z==0 else rng.randint(L//2, L)
            cy = rng.randint(L)
            hit = wave_sites_2d(cx, cy, WAVE_RADIUS, L)
            if len(hit)>0:
                h_ext = EXT_FIELD if wave_z==0 else -EXT_FIELD
                for _ in range(WAVE_DUR):
                    s = metro_step(s, nb, T_FIXED, rng, h_ext=h_ext, hit=hit)
                dev_hit     = s[hit] - base[hit]
                zone_match  = (zone[hit]==wave_z)
                true_signal = np.where(zone_match, dev_hit, -dev_hit)
                is_causal   = rng.random(len(hit)) < P_causal
                std_dev     = float(np.std(dev_hit)) + 0.5
                noise       = rng.normal(0, std_dev, len(hit))
                mid[hit]   += FA * np.where(is_causal, true_signal, noise)
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
    U4    = float(1.-M4/(3.*M2**2)) if M2>1e-12 else float('nan')
    chi   = float(N_zone*varM)
    sigma_phi = float(np.std(phi))+1e-8
    S_phi = abs(float(np.mean(phi[zone==0]))-float(np.mean(phi[zone==1])))/sigma_phi
    return dict(absM=absM, M2=M2, M4=M4, U4=U4, chi=chi, S_phi=S_phi,
                wave_every=wave_every)

def _worker_A(args):
    L, pc, seed, n_steps = args
    return ('A', L, pc, seed, run_A(L, pc, seed, n_steps))


# ══════════════════════════════════════════════════════════════════════════════
# PHASE B: Multi-Zone Relay Coupling
# ══════════════════════════════════════════════════════════════════════════════
def run_B(K, D_phi, P_causal, seed, n_steps):
    """Ising+VCSM-lite with K zones and phi-diffusion relay coupling."""
    L   = L_B
    rng = np.random.RandomState(seed)
    nb  = build_neighbors_ising(L)
    N   = L*L
    col_a = np.arange(N) % L
    # zones: 0..K-1, each of width L//K
    W_zone = L // K
    zone   = col_a // W_zone
    zone   = np.clip(zone, 0, K-1)

    # phi-diffusion neighbor indices (4-connected)
    row_a = np.arange(N) // L
    nb_phi = np.stack([((row_a-1)%L)*L+col_a, ((row_a+1)%L)*L+col_a,
                       row_a*L+(col_a-1)%L,   row_a*L+(col_a+1)%L], axis=1)

    s      = rng.choice([-1,1], N).astype(float)
    base   = s * 0.1
    mid    = np.zeros(N)
    phi    = np.zeros(N)
    streak = np.zeros(N, int)
    wave_z = 0
    ss_start = int(n_steps * SS_START_FRAC)

    # per-zone time series
    zone_phi_series = [[] for _ in range(K)]

    for t in range(n_steps):
        s = metro_step(s, nb, T_FIXED, rng)
        base  += BETA_BASE*(s-base)
        same   = (s>0)==(base>0)
        streak = np.where(same, streak+1, 0)
        mid   *= MID_DECAY

        if t % WAVE_EVERY_B == 0:
            # drive zone wave_z with signed field
            cx = rng.randint(wave_z*W_zone, (wave_z+1)*W_zone)
            cy = rng.randint(L)
            hit = wave_sites_2d(cx, cy, min(WAVE_RADIUS, W_zone//2+1), L)
            if len(hit)>0:
                h_ext = EXT_FIELD if wave_z%2==0 else -EXT_FIELD
                for _ in range(WAVE_DUR):
                    s = metro_step(s, nb, T_FIXED, rng, h_ext=h_ext, hit=hit)
                dev_hit     = s[hit] - base[hit]
                zone_match  = (zone[hit]==wave_z)
                true_signal = np.where(zone_match, dev_hit, -dev_hit)
                is_causal   = rng.random(len(hit)) < P_causal
                std_dev     = float(np.std(dev_hit)) + 0.5
                noise       = rng.normal(0, std_dev, len(hit))
                mid[hit]   += FA * np.where(is_causal, true_signal, noise)
            wave_z = (wave_z + 1) % K

        gate = streak >= SS
        if gate.any():
            wi = np.where(gate)[0]
            phi[wi] += FA*(mid[wi]-phi[wi]); streak[wi] = 0
        phi *= FIELD_DECAY

        # phi-diffusion relay
        if D_phi > 0:
            phi_nb_mean = phi[nb_phi].mean(axis=1)
            phi += D_phi * (phi_nb_mean - phi)

        if t >= ss_start:
            for k in range(K):
                zm = zone==k
                if zm.any():
                    zone_phi_series[k].append(float(np.mean(phi[zm])))

    # Compute zone-mean phi and inter-zone distances
    zone_means = []
    for k in range(K):
        arr = np.array(zone_phi_series[k])
        zone_means.append(float(np.mean(arr)) if len(arr)>0 else 0.0)

    # S_phi: signed difference zone0 - zone1 (reference pair, always exists)
    sigma_phi = float(np.std(phi)) + 1e-8
    S_phi_01 = abs(zone_means[0] - zone_means[1]) / sigma_phi if K>=2 else 0.
    # S_phi far: zone 0 vs zone K//2 (the furthest relay hop)
    S_phi_far = abs(zone_means[0] - zone_means[K//2]) / sigma_phi if K>=2 else 0.

    # Relay decay: fit zone_mean vs zone_index
    zm_arr = np.array([abs(zone_means[0] - zone_means[k]) for k in range(K)])
    relay_slope = float('nan')
    if K >= 4:
        x = np.arange(1, K)
        y = zm_arr[1:]
        if np.any(y > 1e-8):
            logy = np.log(np.maximum(y, 1e-8))
            p = np.polyfit(x, logy, 1)
            relay_slope = float(p[0])   # negative = decaying, ~0 = sustained relay

    return dict(S_phi_01=S_phi_01, S_phi_far=S_phi_far,
                zone_means=zone_means, relay_slope=relay_slope,
                sigma_phi=sigma_phi)

def _worker_B(args):
    K, D_phi, pc, seed, n_steps = args
    return ('B', K, D_phi, pc, seed, run_B(K, D_phi, pc, seed, n_steps))


# ══════════════════════════════════════════════════════════════════════════════
# PHASE C: VCML Birth-Seeding Ablation (minimal inline CML)
# ══════════════════════════════════════════════════════════════════════════════
def build_neighbors_cml(W, H):
    N = W * H
    row = np.arange(N) // W; col = np.arange(N) % W
    return np.stack([((row-1)%H)*W+col, ((row+1)%H)*W+col,
                     row*W+(col-1)%W,   row*W+(col+1)%W], axis=1)

def wave_sites_cml(cx, cy, r, W, H):
    N = W*H
    row = np.arange(N)//W; col = np.arange(N)%W
    dx = np.minimum(np.abs(col-cx), W-np.abs(col-cx))
    dy = np.minimum(np.abs(row-cy), H-np.abs(row-cy))
    return np.where(dx+dy<=r)[0]

def run_C(fsb, seed, n_steps):
    """Minimal VCML ablation: FIELD_SEED_BETA sweep."""
    rng   = np.random.RandomState(seed)
    W, H  = W_C, H_C
    N     = W * H
    nb    = build_neighbors_cml(W, H)
    col_a = np.arange(N) % W
    W_zone = W // K_ZONES_C
    zone   = np.clip(col_a // W_zone, 0, K_ZONES_C-1)

    # VCSM fields (HS=1, scalar per site)
    hid    = rng.normal(0, 0.1, N)
    base   = hid * 0.1
    mid    = np.zeros(N)
    fieldM = np.zeros(N)
    streak = np.zeros(N, int)
    wave_z = 0
    ss_start = int(n_steps * 0.40)

    sg4_series = []

    for t in range(n_steps):
        # 1. Non-linear map update (chaotic carrier, logistic-like)
        hid = np.tanh(hid + 0.05 * rng.normal(0, 1, N))

        # 2. VCSM baseline + streak
        base  += BETA_C * (hid - base)
        same   = (hid > 0) == (base > 0)
        streak = np.where(same, streak+1, 0)
        mid   *= MID_DECAY_C

        # 3. Wave perturbation (zone-specific)
        if t % WAVE_EVERY_C == 0:
            cx  = rng.randint(wave_z*W_zone, (wave_z+1)*W_zone)
            cy  = rng.randint(H)
            hit = wave_sites_cml(cx, cy, WAVE_RAD_C, W, H)
            if len(hit) > 0:
                sign     = 1.0 if wave_z % 2 == 0 else -1.0
                dev_hit  = sign * WAVE_AMP_C * np.ones(len(hit))
                hid[hit] += dev_hit
                hid[hit]  = np.clip(hid[hit], -2., 2.)
                zone_match  = (zone[hit] == wave_z)
                true_signal = np.where(zone_match, dev_hit, -dev_hit)
                mid[hit]   += FA_C * true_signal
            wave_z = (wave_z + 1) % K_ZONES_C

        # 4. Viability gate: mid -> fieldM when streak >= SS
        gate = streak >= SS_C
        if gate.any():
            wi = np.where(gate)[0]
            fieldM[wi] += FA_C * (mid[wi] - fieldM[wi])
            streak[wi]  = 0
        fieldM *= FIELD_DECAY_C

        # 5. Turnover (mandatory reset)
        n_die = max(1, int(TURNOVER_RATE * N))
        die   = rng.choice(N, n_die, replace=False)
        if fsb > 0:
            # Birth seeding: new hid partially from neighbor fieldM
            nb_fieldM = fieldM[nb[die]].mean(axis=1)
            hid[die]  = (1.0 - fsb) * rng.normal(0, 0.1, n_die) + fsb * nb_fieldM
        else:
            # Ablation: pure random rebirth, no copy-forward
            hid[die]  = rng.normal(0, 0.1, n_die)
        base[die]   = hid[die] * 0.1
        mid[die]    = 0.
        streak[die] = 0

        # 6. Metrics
        if t >= ss_start:
            zone_fM = [float(np.mean(fieldM[zone==k])) for k in range(K_ZONES_C)]
            pairs = [(i,j) for i in range(K_ZONES_C) for j in range(i+1,K_ZONES_C)]
            sg4 = float(np.mean([abs(zone_fM[i]-zone_fM[j]) for i,j in pairs]))
            sg4_series.append(sg4)

    # Zone-mean fieldM at end
    zone_fM  = [float(np.mean(fieldM[zone==k])) for k in range(K_ZONES_C)]
    pairs    = [(i,j) for i in range(K_ZONES_C) for j in range(i+1,K_ZONES_C)]
    adj_p    = [(0,1),(1,2),(2,3)]
    nadj_p   = [(i,j) for i,j in pairs if (i,j) not in adj_p]
    dists    = {(i,j): abs(zone_fM[i]-zone_fM[j]) for i,j in pairs}
    adj_d    = float(np.mean([dists[p] for p in adj_p]))
    nadj_d   = float(np.mean([dists[p] for p in nadj_p]))
    na_ratio = nadj_d/adj_d if adj_d>1e-8 else float('nan')

    sg4_mean = float(np.mean(sg4_series)) if sg4_series else float('nan')
    sg4_se   = float(np.std(sg4_series)/math.sqrt(len(sg4_series))) if len(sg4_series)>1 else 0.

    return dict(sg4=sg4_mean, sg4_se=sg4_se, na_ratio=na_ratio,
                zone_fM=zone_fM)

def _worker_C(args):
    fsb, seed, n_steps = args
    return ('C', fsb, seed, run_C(fsb, seed, n_steps))


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    mp.freeze_support()
    RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)

    raw = {}
    if RESULTS_FILE.exists():
        with open(RESULTS_FILE) as f: raw = json.load(f)

    # ── Phase A ──────────────────────────────────────────────────────────────
    if 'A' not in raw:
        args_A = [(L, pc, seed, NSTEPS_FSS)
                  for L in L_FSS for pc in PC_FSS for seed in SEEDS_FSS]
        print(f'Phase A: {len(args_A)} runs (extensive-drive FSS)...')
        for L in L_FSS:
            print(f'  L={L}: WAVE_EVERY={wave_every_for_L(L)} (base={WAVE_EVERY_BASE})')
        with mp.Pool(processes=min(len(args_A), mp.cpu_count())) as pool:
            res_A = pool.map(_worker_A, args_A)
        raw['A'] = {}
        for _, L, pc, seed, r in res_A:
            k = f'L{L}_pc{pc:.2f}'
            if k not in raw['A']: raw['A'][k] = {}
            raw['A'][k][str(seed)] = r
        with open(RESULTS_FILE, 'w') as f: json.dump(raw, f)
        print('Phase A done.')

    # ── Phase B ──────────────────────────────────────────────────────────────
    if 'B' not in raw:
        args_B = [(K, D, pc, seed, NSTEPS_B)
                  for K in K_ZONES for D in D_PHI for pc in PC_B for seed in SEEDS_B]
        print(f'\nPhase B: {len(args_B)} runs (multi-zone relay)...')
        with mp.Pool(processes=min(len(args_B), mp.cpu_count())) as pool:
            res_B = pool.map(_worker_B, args_B)
        raw['B'] = {}
        for _, K, D, pc, seed, r in res_B:
            k = f'K{K}_D{D:.2f}_pc{pc:.2f}'
            if k not in raw['B']: raw['B'][k] = {}
            raw['B'][k][str(seed)] = r
        with open(RESULTS_FILE, 'w') as f: json.dump(raw, f)
        print('Phase B done.')

    # ── Phase C ──────────────────────────────────────────────────────────────
    if 'C' not in raw:
        args_C = [(fsb, seed, NSTEPS_C) for fsb in FSB_VALS for seed in SEEDS_C]
        print(f'\nPhase C: {len(args_C)} runs (VCML ablation)...')
        with mp.Pool(processes=min(len(args_C), mp.cpu_count())) as pool:
            res_C = pool.map(_worker_C, args_C)
        raw['C'] = {}
        for _, fsb, seed, r in res_C:
            k = f'fsb{fsb:.4f}'
            if k not in raw['C']: raw['C'][k] = {}
            raw['C'][k][str(seed)] = r
        with open(RESULTS_FILE, 'w') as f: json.dump(raw, f)
        print('Phase C done.')

    # ── Aggregate Phase A ─────────────────────────────────────────────────────
    print('\n=== PHASE A: Extensive-Drive FSS ===')
    print(f'{"L":>4} {"WAVE_EVERY":>10} {"P_causal":>10} {"|M|":>8} {"U4":>8} {"chi":>10}')
    fss_A = {}
    for L in L_FSS:
        we = wave_every_for_L(L)
        for pc in PC_FSS:
            k = f'L{L}_pc{pc:.2f}'
            if k not in raw['A']: continue
            vals = list(raw['A'][k].values())
            am = mean_s([v['absM'] for v in vals])
            u4 = mean_s([v['U4']  for v in vals])
            ch = mean_s([v['chi'] for v in vals])
            m2 = mean_s([v['M2']  for v in vals])
            m4 = mean_s([v['M4']  for v in vals])
            fss_A[k] = dict(L=L, pc=pc, absM=am, U4=u4, chi=ch, M2=m2, M4=m4,
                             wave_every=we)
            print(f'{L:>4} {we:>10} {pc:>10.2f} {am:>8.5f} {u4:>8.3f} {ch:>10.4f}')

    # Binder crossing
    print(f'\nBinder cumulant U4 (extensive-drive):')
    print(f'{"P_causal":>10}', end='')
    for L in L_FSS: print(f'  L={L:2d}:U4', end='')
    print()
    for pc in PC_FSS:
        print(f'{pc:>10.2f}', end='')
        for L in L_FSS:
            k = f'L{L}_pc{pc:.2f}'
            u4 = fss_A[k]['U4'] if k in fss_A else float('nan')
            print(f'  {u4:>8.3f}', end='')
        print()

    # |M| scaling slopes
    print(f'\n|M| ~ L^alpha slopes (extensive-drive):')
    for pc in PC_FSS:
        vals = [(L, fss_A[f'L{L}_pc{pc:.2f}']['absM'])
                for L in L_FSS if f'L{L}_pc{pc:.2f}' in fss_A]
        if len(vals) >= 3:
            logL = np.log([v[0] for v in vals])
            logM = np.log([max(v[1], 1e-8) for v in vals])
            p = np.polyfit(logL, logM, 1)
            print(f'  P={pc:.2f}: slope={p[0]:.3f}')

    # ── Aggregate Phase B ─────────────────────────────────────────────────────
    print('\n=== PHASE B: Multi-Zone Relay ===')
    relay_B = {}
    for K in K_ZONES:
        for D in D_PHI:
            for pc in PC_B:
                k = f'K{K}_D{D:.2f}_pc{pc:.2f}'
                if k not in raw['B']: continue
                vals = list(raw['B'][k].values())
                s01  = mean_s([v['S_phi_01']  for v in vals])
                sfar = mean_s([v['S_phi_far'] for v in vals])
                rslp = mean_s([v['relay_slope'] for v in vals
                               if not math.isnan(v['relay_slope'])])
                relay_B[k] = dict(K=K, D=D, pc=pc, S_phi_01=s01, S_phi_far=sfar,
                                   relay_slope=rslp)

    print(f'{"K":>4} {"D_phi":>7} {"P_c":>6} {"S_01":>8} {"S_far":>8} {"relay_slope":>12}')
    for K in K_ZONES:
        for D in D_PHI:
            for pc in [0.60]:   # show high-purity result
                k = f'K{K}_D{D:.2f}_pc{pc:.2f}'
                if k not in relay_B: continue
                v = relay_B[k]
                print(f'{K:>4} {D:>7.2f} {pc:>6.2f} {v["S_phi_01"]:>8.3f} '
                      f'{v["S_phi_far"]:>8.3f} {v["relay_slope"]:>12.4f}')

    # ── Aggregate Phase C ─────────────────────────────────────────────────────
    print('\n=== PHASE C: VCML Birth-Seeding Ablation ===')
    print(f'{"FIELD_SEED_BETA":>16} {"sg4":>8} {"sg4_se":>8} {"na_ratio":>10}')
    abl_C = {}
    for fsb in FSB_VALS:
        k = f'fsb{fsb:.4f}'
        if k not in raw['C']: continue
        vals = list(raw['C'][k].values())
        sg4  = mean_s([v['sg4'] for v in vals])
        sg4e = float(np.mean([v['sg4_se'] for v in vals]))
        nar  = mean_s([v['na_ratio'] for v in vals
                       if not math.isnan(v['na_ratio'])])
        abl_C[k] = dict(fsb=fsb, sg4=sg4, sg4_se=sg4e, na_ratio=nar)
        print(f'{fsb:>16.4f} {sg4:>8.4f} {sg4e:>8.4f} {nar:>10.3f}')

    analysis = dict(A=fss_A, B=relay_B, C=abl_C)
    with open(ANALYSIS_FILE, 'w') as f: json.dump(analysis, f, indent=2)
    print(f'\nSaved -> {ANALYSIS_FILE}')
