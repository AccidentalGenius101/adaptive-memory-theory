"""
paper23_two_scale.py -- Experiment for Paper 23

Two-Scale Reynolds Decomposition of VCML: Validating the Interface Formula

The within-zone PDE (PDE 2 from the two-scale decomposition) predicts:

    xi_inf = sqrt(kappa / (P_consol * FA))

where kappa is the explicit site-level diffusion.  We test this with two sweeps:

  1. FA sweep  (nu=0.001, kappa=0.020 fixed):
       FA in {0.10, 0.20, 0.30, 0.40, 0.60, 0.80}
       Predicts: xi_inf ~ FA^(-0.5)

  2. KAPPA sweep (nu=0.001, FA=0.40 fixed):
       KAPPA in {0.005, 0.010, 0.020, 0.040, 0.080}
       Predicts: xi_inf ~ kappa^(+0.5)

Standard condition (nu=0.001, FA=0.40, kappa=0.020) is shared -- 5 seeds,
counted once.  Total unique conditions: 10 x 5 seeds = 50 runs.

Per checkpoint: record sg4 AND corr[0..MAX_LAG] (same as Paper 22).
"""
import numpy as np, json, os, multiprocessing as mp, math
from scipy.stats import linregress

# ── Fixed parameters ──────────────────────────────────────────────────────────
H         = 20
N_ZONES   = 4
HALF      = 20
BASE_BETA = 0.005
ALPHA_MID = 0.15
MID_DECAY = 0.99
FD        = 0.9997
SS        = 10
WR        = 4.8
WAVE_DUR  = 15
HS        = 2
SEED_BETA = 0.25

T_END    = 3000
CPS_SET  = set(range(200, T_END + 1, 200))
CPS      = sorted(CPS_SET)
N_ACT    = HALF * H
ZONE_W   = HALF // N_ZONES   # 5 sites per zone
MAX_LAG  = HALF // 2         # 10
TAU_BASE = 1.0 / BASE_BETA   # 200

# ── Standard defaults ─────────────────────────────────────────────────────────
NU_STD    = 0.001
FA_STD    = 0.40
KAPPA_STD = 0.020

# ── Sweep values ──────────────────────────────────────────────────────────────
FA_VALS    = [0.10, 0.20, 0.30, 0.40, 0.60, 0.80]
KAPPA_VALS = [0.005, 0.010, 0.020, 0.040, 0.080]
N_SEEDS    = 5

# ── Geometry (fixed HALF=20) ──────────────────────────────────────────────────
_col     = np.arange(N_ACT) % HALF
_row     = np.arange(N_ACT) // HALF
zone_id  = _col // ZONE_W
top_mask = _row < H // 2
bot_mask = _row >= H // 2
d_B      = np.array([0.0, 1.0])

NB = []
for _i in range(N_ACT):
    c, r = int(_col[_i]), int(_row[_i])
    nb = []
    for dc, dr in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        nc, nr = c + dc, r + dr
        if 0 <= nc < HALF and 0 <= nr < H:
            nb.append(nr * HALF + nc)
    NB.append(np.array(nb, dtype=int))


# ── Measurement functions ─────────────────────────────────────────────────────
def sg4_fn(F):
    means = [F[zone_id == z].mean(0) for z in range(N_ZONES)]
    pairs = [(i, j) for i in range(N_ZONES) for j in range(i + 1, N_ZONES)]
    return float(np.mean([np.linalg.norm(means[i] - means[j]) for i, j in pairs]))


def spatial_autocorr(F):
    """Column-direction autocorrelation of zone-mean-subtracted F."""
    F_res = F.copy()
    for z in range(N_ZONES):
        mask = zone_id == z
        if mask.any():
            F_res[mask] -= F[mask].mean(0)

    F_grid = F_res.reshape(H, HALF, HS)
    var_0  = float(np.mean(np.sum(F_grid ** 2, axis=2)))

    if var_0 < 1e-12:
        return [float("nan")] * (MAX_LAG + 1)

    result = [1.0]
    for r in range(1, MAX_LAG + 1):
        sub1 = F_grid[:, :HALF - r, :]
        sub2 = F_grid[:, r:, :]
        dot  = np.sum(sub1 * sub2, axis=2)
        result.append(float(dot.mean() / var_0))

    return result


# ── Simulation ────────────────────────────────────────────────────────────────
def run(seed, nu=NU_STD, fa=FA_STD, kappa=KAPPA_STD):
    rng    = np.random.default_rng(seed)
    h      = rng.normal(0, 0.1, (N_ACT, HS))
    F      = np.zeros((N_ACT, HS))
    m      = np.zeros((N_ACT, HS))
    base   = h.copy()
    streak = np.zeros(N_ACT, int)
    ttl    = rng.geometric(p=max(nu, 1e-6), size=N_ACT).astype(float)
    waves  = []
    result = {}

    for step in range(1, T_END + 1):

        # Wave launches (Phase 1 throughout)
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
        F[ok] += fa * (m[ok] - F[ok])
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
            h[i]      = (1 - SEED_BETA) * rng.normal(0, 0.1, HS) + SEED_BETA * F[j]
            m[i]      = 0
            streak[i] = 0
            ttl[i]    = float(rng.geometric(p=nu))

        waves = [[w[0], w[1] - 1, w[2], w[3]] for w in waves if w[1] - 1 > 0]

        if step in CPS_SET:
            result[f"sg4_{step}"]  = sg4_fn(F)
            result[f"corr_{step}"] = spatial_autocorr(F)

    return result


def make_key(sweep, val, seed):
    return f"p23,{sweep},{val:.8g},{seed}"


def _worker(args):
    sweep, val, seed = args
    if sweep == "fa":
        return run(seed, fa=val)
    elif sweep == "kappa":
        return run(seed, kappa=val)
    else:
        raise ValueError(f"Unknown sweep: {sweep}")


RESULTS_FILE = os.path.join(os.path.dirname(__file__), "results", "paper23_results.json")


# ── Analysis helpers ──────────────────────────────────────────────────────────
XI_CAP = 30.0   # cap spurious early-noise xi values


def fit_xi(corr):
    """Log-linear fit C(r)=exp(-r/xi), r=1..ZONE_W. Returns xi or nan."""
    r_vals, log_c = [], []
    for r in range(1, min(ZONE_W + 1, MAX_LAG + 1)):
        if r >= len(corr):
            break
        c = corr[r]
        if not math.isnan(c) and 0.05 < c <= 1.0:
            r_vals.append(r)
            log_c.append(math.log(c))
    if len(r_vals) < 2:
        return float("nan")
    slope, _, _, _, _ = linregress(r_vals, log_c)
    if slope >= 0:
        return float("nan")
    return -1.0 / slope


def xi_final(results, sweep, val, late_ts=None):
    """Mean xi over last few checkpoints, across seeds."""
    if late_ts is None:
        late_ts = [t for t in CPS if t >= 2400]
    xis = []
    for seed in range(N_SEEDS):
        key = make_key(sweep, val, seed)
        if key not in results:
            continue
        for t in late_ts:
            corr = results[key].get(f"corr_{t}")
            if corr is not None:
                xi = fit_xi(corr)
                if not math.isnan(xi) and xi <= XI_CAP:
                    xis.append(xi)
    if not xis:
        return float("nan"), float("nan")
    return float(np.mean(xis)), float(np.std(xis) / math.sqrt(len(xis)))


if __name__ == "__main__":
    mp.freeze_support()

    # Build full condition list
    all_conditions = []
    for fa    in FA_VALS:    all_conditions += [("fa",    fa,    s) for s in range(N_SEEDS)]
    for kappa in KAPPA_VALS: all_conditions += [("kappa", kappa, s) for s in range(N_SEEDS)]

    # Dedup: standard condition (FA=0.40, KAPPA=0.020) appears in both sweeps
    # Keep key: p23,fa,0.4,s  (from FA sweep) -- these will be used for the kappa
    # standard reference as well (both share nu=0.001, fa=0.40, kappa=0.020)
    seen = set()
    unique_conditions = []
    for args in all_conditions:
        sweep, val, seed = args
        # Map kappa standard to fa standard (same simulation)
        if sweep == "kappa" and abs(val - KAPPA_STD) < 1e-9:
            proxy_key = make_key("fa", FA_STD, seed)
            if proxy_key not in seen:
                unique_conditions.append(("fa", FA_STD, seed))
                seen.add(proxy_key)
        else:
            key = make_key(sweep, val, seed)
            if key not in seen:
                unique_conditions.append(args)
                seen.add(key)

    if os.path.exists(RESULTS_FILE):
        results = json.load(open(RESULTS_FILE))
        print(f"Loaded {len(results)} cached results.")
    else:
        results = {}

    todo = [args for args in unique_conditions if make_key(*args) not in results]

    if todo:
        print(f"Running {len(todo)} / {len(unique_conditions)} simulations (Paper 23)...")
        with mp.Pool(processes=min(len(todo), mp.cpu_count())) as pool:
            raw = pool.map(_worker, todo)
        for i, args in enumerate(todo):
            results[make_key(*args)] = raw[i]
        os.makedirs(os.path.dirname(RESULTS_FILE), exist_ok=True)
        json.dump(results, open(RESULTS_FILE, "w"), indent=2)
        print("Saved results.")
    else:
        print("All results cached.")

    # Mirror kappa-standard keys to fa-standard keys in lookup
    # (they're the same simulation; we use the fa key as canonical)
    def get_key(sweep, val, seed):
        if sweep == "kappa" and abs(val - KAPPA_STD) < 1e-9:
            return make_key("fa", FA_STD, seed)
        return make_key(sweep, val, seed)

    # ── Analysis ──────────────────────────────────────────────────────────────
    print()
    print("=" * 65)
    print("TWO-SCALE DECOMPOSITION  (Paper 23)")
    print(f"Predicted: xi_inf = sqrt(kappa / (P_consol * FA))")
    print("=" * 65)

    # FA sweep
    print(f"\n  FA sweep (kappa={KAPPA_STD}, nu={NU_STD})")
    print(f"  {'FA':>6} | {'xi_inf':>8} | {'SEM':>6} | {'pred':>8} | {'ratio':>6}")
    print("  " + "-" * 45)
    P_CONSOL = 0.175   # measured in earlier papers

    fa_xi_means, fa_xi_sems = [], []
    for fa in FA_VALS:
        # Use get_key for the standard
        xis = []
        late_ts = [t for t in CPS if t >= 2400]
        for seed in range(N_SEEDS):
            key = get_key("fa", fa, seed)
            if key not in results:
                continue
            for t in late_ts:
                corr = results[key].get(f"corr_{t}")
                if corr is not None:
                    xi = fit_xi(corr)
                    if not math.isnan(xi) and xi <= XI_CAP:
                        xis.append(xi)
        m  = float(np.mean(xis)) if xis else float("nan")
        s  = float(np.std(xis) / math.sqrt(len(xis))) if len(xis) > 1 else 0.0
        fa_xi_means.append(m)
        fa_xi_sems.append(s)
        pred = math.sqrt(KAPPA_STD / (P_CONSOL * fa)) if fa > 0 else float("nan")
        ratio = m / pred if not math.isnan(m) and pred > 0 else float("nan")
        print(f"  {fa:>6.2f} | {m:>8.3f} | {s:>6.3f} | {pred:>8.3f} | {ratio:>6.2f}")

    valid_fa = [(fa, x) for fa, x in zip(FA_VALS, fa_xi_means)
                if not math.isnan(x) and x > 0]
    if len(valid_fa) >= 3:
        log_fa = np.log([v[0] for v in valid_fa])
        log_xi = np.log([v[1] for v in valid_fa])
        slope_fa, _, r_fa, _, _ = linregress(log_fa, log_xi)
        print(f"\n  xi_inf ~ FA^{slope_fa:.3f}  (R2={r_fa**2:.4f}),  predicted: -0.5")

    # KAPPA sweep
    print(f"\n  KAPPA sweep (FA={FA_STD}, nu={NU_STD})")
    print(f"  {'kappa':>7} | {'xi_inf':>8} | {'SEM':>6} | {'pred':>8} | {'ratio':>6}")
    print("  " + "-" * 47)

    kap_xi_means, kap_xi_sems = [], []
    for kappa in KAPPA_VALS:
        xis = []
        for seed in range(N_SEEDS):
            key = get_key("kappa", kappa, seed)
            if key not in results:
                continue
            for t in late_ts:
                corr = results[key].get(f"corr_{t}")
                if corr is not None:
                    xi = fit_xi(corr)
                    if not math.isnan(xi) and xi <= XI_CAP:
                        xis.append(xi)
        m  = float(np.mean(xis)) if xis else float("nan")
        s  = float(np.std(xis) / math.sqrt(len(xis))) if len(xis) > 1 else 0.0
        kap_xi_means.append(m)
        kap_xi_sems.append(s)
        pred = math.sqrt(kappa / (P_CONSOL * FA_STD)) if kappa > 0 else float("nan")
        ratio = m / pred if not math.isnan(m) and pred > 0 else float("nan")
        print(f"  {kappa:>7.3f} | {m:>8.3f} | {s:>6.3f} | {pred:>8.3f} | {ratio:>6.2f}")

    valid_kap = [(k, x) for k, x in zip(KAPPA_VALS, kap_xi_means)
                 if not math.isnan(x) and x > 0]
    if len(valid_kap) >= 3:
        log_kap = np.log([v[0] for v in valid_kap])
        log_xi  = np.log([v[1] for v in valid_kap])
        slope_kap, _, r_kap, _, _ = linregress(log_kap, log_xi)
        print(f"\n  xi_inf ~ kappa^{slope_kap:.3f}  (R2={r_kap**2:.4f}),  predicted: +0.5")
