"""
paper21_spatial_formation.py -- Experiment for Paper 21

The Spatial Formation Factor: measuring how C = tau_buildup / tau_base
depends on the four structural parameters that govern copy-forward propagation.

Four one-parameter sweeps (all others fixed at standard values):
  1. NU        in [0.0005, 0.001, 0.002, 0.004, 0.008]  (turnover rate)
  2. SEED_BETA in [0.10, 0.20, 0.25, 0.40, 0.80]        (inheritance fraction)
  3. KAPPA     in [0.005, 0.010, 0.020, 0.040, 0.080]    (field diffusion)
  4. HALF      in [8, 12, 20, 32, 40]                    (zone_width = HALF//4)

Standard: NU=0.001, SEED_BETA=0.25, KAPPA=0.020, HALF=20
Protocol: Phase 1 only (waves continuous for full T_END=3000 steps).
Checkpoints every 200 steps (15 checkpoints per run).
5 seeds per condition -> 100 runs total.

Key metric: C-factor = tau_buildup / tau_base
  tau_buildup = time for sg4 to reach 63% of sg4(T_END)
  tau_base    = 1 / BASE_BETA = 200 steps
"""
import numpy as np, json, os, multiprocessing as mp, math

# ── Fixed simulation parameters ───────────────────────────────────────────────
H         = 20
N_ZONES   = 4
BASE_BETA = 0.005      # tau_base = 200
ALPHA_MID = 0.15
MID_DECAY = 0.99
FD        = 0.9997
FA        = 0.40
SS        = 10
WR        = 4.8
WAVE_DUR  = 15
HS        = 2

T_END   = 3000
CPS_SET = set(range(200, T_END + 1, 200))
CPS     = sorted(CPS_SET)

TAU_BASE = 1.0 / BASE_BETA   # 200

# ── Standard defaults ─────────────────────────────────────────────────────────
NU_STD        = 0.001
SEED_BETA_STD = 0.25
KAPPA_STD     = 0.020
HALF_STD      = 20

N_SEEDS = 5

# ── Sweep values ──────────────────────────────────────────────────────────────
NU_VALS        = [0.0005, 0.001, 0.002, 0.004, 0.008]
SEED_BETA_VALS = [0.10, 0.20, 0.25, 0.40, 0.80]
KAPPA_VALS     = [0.005, 0.010, 0.020, 0.040, 0.080]
HALF_VALS      = [8, 12, 20, 32, 40]


# ── Simulation ────────────────────────────────────────────────────────────────
def sg4_fn(F, zone_id):
    means = [F[zone_id == z].mean(0) for z in range(N_ZONES)]
    pairs = [(i, j) for i in range(N_ZONES) for j in range(i + 1, N_ZONES)]
    return float(np.mean([np.linalg.norm(means[i] - means[j]) for i, j in pairs]))


def run(seed, nu=NU_STD, seed_beta=SEED_BETA_STD, kappa=KAPPA_STD, half=HALF_STD):
    rng   = np.random.default_rng(seed)
    N_ACT = half * H
    zone_w = max(1, half // N_ZONES)

    # Geometry (computed per run to support variable half)
    _col     = np.arange(N_ACT) % half
    _row     = np.arange(N_ACT) // half
    zone_id  = _col // zone_w
    top_mask = _row < H // 2
    bot_mask = _row >= H // 2
    d_B      = np.array([0.0, 1.0])

    NB = []
    for i in range(N_ACT):
        c, r = int(_col[i]), int(_row[i])
        nb = []
        for dc, dr in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nc, nr = c + dc, r + dr
            if 0 <= nc < half and 0 <= nr < H:
                nb.append(nr * half + nc)
        NB.append(np.array(nb, dtype=int))

    h      = rng.normal(0, 0.1, (N_ACT, HS))
    F      = np.zeros((N_ACT, HS))
    m      = np.zeros((N_ACT, HS))
    base   = h.copy()
    streak = np.zeros(N_ACT, int)
    ttl    = rng.geometric(p=max(nu, 1e-6), size=N_ACT).astype(float)
    waves  = []
    result = {}

    for step in range(1, T_END + 1):

        # Launch waves (Phase 1 throughout -- no Phase 3 in this experiment)
        n = int(WR / WAVE_DUR)
        n += int(rng.random() < (WR / WAVE_DUR - n))
        for _ in range(n):
            top  = rng.random() < 0.5
            sign = 1.0 if top else -1.0
            waves.append([0 if top else 1, WAVE_DUR, sign * d_B.copy(), False])

        pert = np.zeros(N_ACT, bool)
        for w in waves:
            pert |= (top_mask if w[0] == 0 else bot_mask)

        base += BASE_BETA * (h - base)
        for w in waves:
            mask = top_mask if w[0] == 0 else bot_mask
            h[mask] += 0.3 * w[2]

        m[pert] += ALPHA_MID * (h - base)[pert]
        m       *= MID_DECAY

        streak[pert]  = 0
        streak[~pert] += 1
        ok = streak >= SS
        F[ok] += FA * (m[ok] - F[ok])
        F     *= FD

        dF = np.zeros_like(F)
        for i in range(N_ACT):
            if len(NB[i]):
                dF[i] = kappa * (F[NB[i]].mean(0) - F[i])
        F += dF

        ttl -= 1.0
        ttl[pert] -= 1.0
        dead = np.where(ttl <= 0)[0]
        for i in dead:
            j = NB[i][rng.integers(len(NB[i]))] if len(NB[i]) else i
            h[i]      = (1 - seed_beta) * rng.normal(0, 0.1, HS) + seed_beta * F[j]
            m[i]      = 0
            streak[i] = 0
            ttl[i]    = float(rng.geometric(p=nu))

        waves = [[w[0], w[1] - 1, w[2], w[3]] for w in waves if w[1] - 1 > 0]

        if step in CPS_SET:
            result[f"sg4_{step}"] = sg4_fn(F, zone_id)

    return result


def make_key(sweep, val, seed):
    return f"p21,{sweep},{val:.8g},{seed}"


def _worker(args):
    sweep, val, seed = args
    if sweep == "nu":
        return run(seed, nu=val)
    elif sweep == "sb":
        return run(seed, seed_beta=val)
    elif sweep == "kappa":
        return run(seed, kappa=val)
    elif sweep == "half":
        return run(seed, half=int(val))
    else:
        raise ValueError(f"Unknown sweep: {sweep}")


RESULTS_FILE = os.path.join(os.path.dirname(__file__), "results", "paper21_results.json")


# ── Analysis helpers ──────────────────────────────────────────────────────────
def tau_buildup_seeds(results, sweep, val):
    """Return (mean_tau, sem_tau) across seeds for a given (sweep, val)."""
    taus = []
    for seed in range(N_SEEDS):
        key = make_key(sweep, val, seed)
        if key not in results:
            continue
        r = results[key]
        sg4_final = r.get(f"sg4_{T_END}", float("nan"))
        if math.isnan(sg4_final) or sg4_final <= 0:
            continue
        target   = 0.63 * sg4_final
        prev_t, prev_v = 0, 0.0
        tau      = float(T_END)   # upper bound if 63% never reached
        for t in CPS:
            v = r.get(f"sg4_{t}", float("nan"))
            if math.isnan(v):
                break
            if v >= target:
                frac = (target - prev_v) / max(v - prev_v, 1e-12)
                tau  = prev_t + frac * (t - prev_t)
                break
            prev_t, prev_v = t, v
        taus.append(tau)
    if not taus:
        return float("nan"), float("nan")
    m = float(np.mean(taus))
    s = float(np.std(taus) / math.sqrt(len(taus))) if len(taus) > 1 else 0.0
    return m, s


if __name__ == "__main__":
    mp.freeze_support()

    # Build full condition list
    all_conditions = []
    for nu   in NU_VALS:        all_conditions += [("nu",   nu,          s) for s in range(N_SEEDS)]
    for sb   in SEED_BETA_VALS: all_conditions += [("sb",   sb,          s) for s in range(N_SEEDS)]
    for kap  in KAPPA_VALS:     all_conditions += [("kappa",kap,         s) for s in range(N_SEEDS)]
    for half in HALF_VALS:      all_conditions += [("half", float(half), s) for s in range(N_SEEDS)]

    # Load cached or run
    if os.path.exists(RESULTS_FILE):
        results = json.load(open(RESULTS_FILE))
        print(f"Loaded {len(results)} cached results.")
    else:
        results = {}

    todo = [args for args in all_conditions if make_key(*args) not in results]

    if todo:
        print(f"Running {len(todo)} / {len(all_conditions)} simulations (Paper 21)...")
        with mp.Pool(processes=min(len(todo), mp.cpu_count())) as pool:
            raw = pool.map(_worker, todo)
        for i, args in enumerate(todo):
            results[make_key(*args)] = raw[i]
        os.makedirs(os.path.dirname(RESULTS_FILE), exist_ok=True)
        json.dump(results, open(RESULTS_FILE, "w"), indent=2)
        print("Saved results.")
    else:
        print("All results cached -- skipping simulation.")

    # ── Print analysis ─────────────────────────────────────────────────────
    print()
    print("=" * 65)
    print("C-FACTOR ANALYSIS  (C = tau_buildup / tau_base,  tau_base=200)")
    print("=" * 65)

    sweeps = [
        ("nu",    NU_VALS,                  "NU (turnover rate)",          "nu"),
        ("sb",    SEED_BETA_VALS,            "SEED_BETA (inheritance)",     "seed_beta"),
        ("kappa", KAPPA_VALS,               "KAPPA (field diffusion)",     "kappa"),
        ("half",  [float(h) for h in HALF_VALS], "HALF -> zone_w=HALF//4", "zone_w"),
    ]

    for sweep_tag, vals, label, plabel in sweeps:
        print(f"\n  Sweep: {label}")
        print(f"  {'param':>12} | {'tau_buildup':>12} | {'SEM':>8} | {'C':>8}")
        print("  " + "-" * 48)
        for val in vals:
            tau, sem = tau_buildup_seeds(results, sweep_tag, val)
            C = tau / TAU_BASE if not math.isnan(tau) else float("nan")
            extra = ""
            if sweep_tag == "half":
                extra = f"  [zone_w={int(val)//N_ZONES}]"
            print(f"  {val:>12.6g} | {tau:>12.1f} | {sem:>8.1f} | {C:>8.2f}{extra}")
