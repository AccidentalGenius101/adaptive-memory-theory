"""
Paper 60: Causal Purity as an Independent Control Parameter in an Ising Spin System.

Standard Ising: phase transition at T_c ≈ 2.27, controlled by T.
New: VCSM-lite slow field phi on top of Ising, updated by causal gating.
Claim: phi-order (S_phi) has its OWN phase transition in P_causal, ORTHOGONAL to T.

Wave mechanism (corrected):
  Zone 0 waves apply h_ext = +EXT_FIELD to sites in the wave radius (bias spins -> +1)
  Zone 1 waves apply h_ext = -EXT_FIELD (bias spins -> -1)
  This creates REAL zone-specific perturbations.
  dev = s - base captures the zone-specific signal.
  P_causal: fraction of writes correctly attributed to zone -> mid.
  phi accumulates zone-specific signal via write gate (streak >= SS).
  S_phi = |phi_zone0 - phi_zone1| / sigma_phi = phi-order parameter.

Expected result:
  m = f(T only) -- Ising transition, P_causal-independent
  S_phi = f(P_causal > p_c) -- purity-controlled transition, T-independent above T_c
  Both exist simultaneously at T>T_c: disordered spins + ordered phi
"""
import numpy as np, json, os, math, multiprocessing as mp

L           = 40
N           = L * L
J           = 1.0
BETA_BASE   = 0.005
FA          = 0.30
FIELD_DECAY = 0.999
MID_DECAY   = 0.97
SS          = 8
WAVE_EVERY  = 25
WAVE_RADIUS = 5
WAVE_DUR    = 5         # Metropolis steps of external field per wave event
EXT_FIELD   = 1.5      # external field amplitude during wave

T_SWEEP       = [1.0, 1.5, 2.0, 2.27, 3.0, 4.0]
PCAUSAL_SWEEP = [0.50, 0.60, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.00]
SEEDS         = list(range(3))
N_STEPS       = 8000
TRACK_EVERY   = 100

RESULTS_FILE  = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             'results', 'paper60_results.json')
ANALYSIS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             'results', 'paper60_analysis.json')


def build_neighbors(L):
    N = L * L
    row = np.arange(N) // L
    col = np.arange(N) % L
    return np.stack([
        ((row - 1) % L) * L + col,
        ((row + 1) % L) * L + col,
        row * L + (col - 1) % L,
        row * L + (col + 1) % L,
    ], axis=1)


def metropolis_step(s, nb, T, rng, h_ext=None, hit=None):
    """One checkerboard Metropolis sweep. Optionally applies external field at sites hit."""
    L = int(round(math.sqrt(len(s))))
    row = np.arange(len(s)) // L
    col = np.arange(len(s)) % L
    h = np.zeros(len(s))
    if h_ext is not None and hit is not None:
        h[hit] = h_ext
    for sub in [0, 1]:
        idx = np.where((row + col) % 2 == sub)[0]
        nb_sum = s[nb[idx]].sum(1)
        # dE = 2*J*s*nb_sum - 2*h*s (energy change on flip including external field)
        dE = 2.0 * J * s[idx] * nb_sum - 2.0 * h[idx] * s[idx]
        acc = (dE <= 0) | (rng.random(len(idx)) < np.exp(-np.clip(dE / T, -20, 20)))
        s[idx[acc]] *= -1
    return s


def wave_sites(cx, cy, r, L):
    N = L * L
    row = np.arange(N) // L
    col = np.arange(N) % L
    dx = np.minimum(np.abs(col - cx), L - np.abs(col - cx))
    dy = np.minimum(np.abs(row - cy), L - np.abs(row - cy))
    return np.where(dx + dy <= r)[0]


def run_one(T, P_causal, seed):
    rng  = np.random.RandomState(seed)
    nb   = build_neighbors(L)
    col_arr = np.arange(N) % L
    zone = (col_arr >= L // 2).astype(int)   # left=0, right=1

    s      = rng.choice([-1, 1], N).astype(float)
    base   = s * 0.1
    mid    = np.zeros(N)
    phi    = np.zeros(N)
    streak = np.zeros(N, int)

    traj   = []
    wave_z = 0

    for t in range(N_STEPS):
        # 1. Standard Metropolis
        s = metropolis_step(s, nb, T, rng)

        # 2. Baseline and streak
        base  += BETA_BASE * (s - base)
        same   = (s > 0) == (base > 0)
        streak = np.where(same, streak + 1, 0)

        # 3. Mid decay
        mid *= MID_DECAY

        # 4. Wave event: apply zone-specific external field + write to mid
        if t % WAVE_EVERY == 0:
            if wave_z == 0:
                cx = rng.randint(0, L // 2)
            else:
                cx = rng.randint(L // 2, L)
            cy  = rng.randint(L)
            hit = wave_sites(cx, cy, WAVE_RADIUS, L)

            if len(hit) > 0:
                # External field: zone 0 -> +EXT_FIELD (bias toward +1)
                #                 zone 1 -> -EXT_FIELD (bias toward -1)
                h_ext = EXT_FIELD if wave_z == 0 else -EXT_FIELD

                # Apply perturbing field for WAVE_DUR steps
                s_before = s[hit].copy()
                for _ in range(WAVE_DUR):
                    s = metropolis_step(s, nb, T, rng, h_ext=h_ext, hit=hit)

                # Capture deviation (s after perturbation vs baseline)
                dev_hit = s[hit] - base[hit]

                # True signal: zone-consistent signed deviation
                # Zone 0 wave & zone 0 site -> dev should be positive (both biased +)
                # Zone 1 wave & zone 1 site -> dev should be negative (both biased -)
                # Zone 0 wave & zone 1 site -> dev was biased WRONG direction
                zone_match  = (zone[hit] == wave_z)
                true_signal = np.where(zone_match, dev_hit, -dev_hit)
                # true_signal > 0 if the perturbation confirms zone identity

                # Causal gate
                is_causal    = rng.random(len(hit)) < P_causal
                std_dev      = float(np.std(dev_hit)) + 0.5
                noise        = rng.normal(0, std_dev, len(hit))
                contribution = np.where(is_causal, true_signal, noise)
                mid[hit]    += FA * contribution

            wave_z = 1 - wave_z

        # 5. Write gate: mid -> phi when streak >= SS
        gate = streak >= SS
        if gate.any():
            wi = np.where(gate)[0]
            phi[wi]   += FA * (mid[wi] - phi[wi])
            streak[wi] = 0

        phi *= FIELD_DECAY

        # 6. Metrics
        if (t + 1) % TRACK_EVERY == 0:
            m       = float(abs(np.mean(s)))
            phi0    = float(np.mean(phi[zone == 0]))
            phi1    = float(np.mean(phi[zone == 1]))
            sig_phi = float(np.std(phi)) + 1e-8
            S_phi   = abs(phi0 - phi1) / sig_phi
            traj.append(dict(step=t+1, m=m, S_phi=float(S_phi),
                             phi0=phi0, phi1=phi1, sig_phi=sig_phi))
    return traj


def _worker(args):
    T, pc, seed = args
    return (T, pc, seed, run_one(T, pc, seed))


def mean_s(vals):
    v = [x for x in vals if x is not None and not math.isnan(float(x))]
    return float(np.mean(v)) if v else float('nan')


def aggregate(trajs):
    n_tp = min(len(t) for t in trajs)
    out  = {'steps': [trajs[0][i]['step'] for i in range(n_tp)]}
    for k in ['m', 'S_phi']:
        out[f'{k}_mean'] = [mean_s([t[i][k] for t in trajs]) for i in range(n_tp)]
    n_ss = max(1, n_tp // 5)   # last 20% = steady state
    out['final_m']     = mean_s([np.mean([t[i]['m']     for i in range(n_tp-n_ss, n_tp)])
                                  for t in trajs])
    out['final_S_phi'] = mean_s([np.mean([t[i]['S_phi'] for i in range(n_tp-n_ss, n_tp)])
                                  for t in trajs])
    return out


if __name__ == '__main__':
    mp.freeze_support()
    os.makedirs(os.path.dirname(RESULTS_FILE), exist_ok=True)

    # Delete old results to rerun
    if os.path.exists(RESULTS_FILE):
        os.remove(RESULTS_FILE)

    all_args = [(T, pc, seed)
                for T   in T_SWEEP
                for pc  in PCAUSAL_SWEEP
                for seed in SEEDS]
    print(f'Running {len(all_args)} experiments on {mp.cpu_count()} cores...')
    with mp.Pool(processes=min(len(all_args), mp.cpu_count())) as pool:
        results = pool.map(_worker, all_args)

    raw = {}
    for T, pc, seed, traj in results:
        k = f'T{T:.2f}_pc{pc:.2f}'
        if k not in raw: raw[k] = {}
        raw[k][str(seed)] = traj
    with open(RESULTS_FILE, 'w') as f:
        json.dump(raw, f)
    print(f'Saved -> {RESULTS_FILE}')

    summary = {}
    for T in T_SWEEP:
        for pc in PCAUSAL_SWEEP:
            k = f'T{T:.2f}_pc{pc:.2f}'
            if k not in raw: continue
            summary[k] = aggregate(list(raw[k].values()))

    with open(ANALYSIS_FILE, 'w') as f:
        json.dump(summary, f, indent=2)

    # Print phase diagrams
    print(f'\nS_phi phase diagram (phi-order, purity-controlled):')
    print('    T \\ P_c  ' + ''.join(f'{pc:6.2f}' for pc in PCAUSAL_SWEEP))
    for T in T_SWEEP:
        row = f'T={T:.2f}   '
        for pc in PCAUSAL_SWEEP:
            k = f'T{T:.2f}_pc{pc:.2f}'
            v = summary[k]['final_S_phi'] if k in summary else float('nan')
            row += f'{v:6.2f}'
        print(row)

    print(f'\nMagnetization (spin-order, temperature-controlled):')
    print('    T \\ P_c  ' + ''.join(f'{pc:6.2f}' for pc in PCAUSAL_SWEEP))
    for T in T_SWEEP:
        row = f'T={T:.2f}   '
        for pc in PCAUSAL_SWEEP:
            k = f'T{T:.2f}_pc{pc:.2f}'
            v = summary[k]['final_m'] if k in summary else float('nan')
            row += f'{v:6.2f}'
        print(row)
    print(f'\nSaved analysis -> {ANALYSIS_FILE}')
