"""
paper24_birth_bias.py -- Experiment for Paper 24

The Birth Bias Axis: Testing the Model H Transition from Allen-Cahn to Burgers

In current VCML, a newborn cell copies fieldM from a UNIFORMLY RANDOM neighbour.
This creates symmetric diffusion -> Allen-Cahn phase separation (xi decreasing, Paper 22).

Adding a birth bias alpha in [0,1] makes the newborn preferentially copy from the
HIGHEST-|F| neighbour (fitness-proportionate selection).  This breaks symmetry and
injects a directed advection term u*dF/dx into the effective PDE -> Burgers/Fisher-KPP
regime (xi should increase over time, beta > 0).

The prediction (Model H framework):
  alpha=0  : Allen-Cahn  -- xi(t) DECREASING  (confirmed, Paper 22)
  alpha=0.5: KPZ boundary -- xi(t) FLAT or slow growth (beta~1/3)
  alpha=1  : Burgers/Fisher-KPP -- xi(t) INCREASING  (beta > 0)

Implementation:
  With probability alpha  -> select j = argmax_{nb} |F[nb]|^2  (fitness-proportionate)
  With probability 1-alpha-> select j = uniform random neighbour (current VCML)

Sweep: alpha in {0.0, 0.2, 0.4, 0.6, 0.8, 1.0}
       5 seeds each -> 30 runs
       Standard params: nu=0.001, FA=0.40, kappa=0.020, T_END=3000

Per checkpoint: sg4 AND full C(r) autocorrelation (same as Papers 22-23).
Also record C(r) WITHOUT zone-mean subtraction to capture zone-level correlation.
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
FA        = 0.40
SS        = 10
WR        = 4.8
WAVE_DUR  = 15
HS        = 2
SEED_BETA = 0.25
KAPPA     = 0.020
NU        = 0.001

T_END    = 3000
CPS_SET  = set(range(200, T_END + 1, 200))
CPS      = sorted(CPS_SET)
N_ACT    = HALF * H
ZONE_W   = HALF // N_ZONES   # 5
MAX_LAG  = HALF // 2         # 10
TAU_BASE = 1.0 / BASE_BETA   # 200

ALPHA_VALS = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
N_SEEDS    = 5

# ── Geometry ──────────────────────────────────────────────────────────────────
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


# ── Measurement ───────────────────────────────────────────────────────────────
def sg4_fn(F):
    means = [F[zone_id == z].mean(0) for z in range(N_ZONES)]
    pairs = [(i, j) for i in range(N_ZONES) for j in range(i + 1, N_ZONES)]
    return float(np.mean([np.linalg.norm(means[i] - means[j]) for i, j in pairs]))


def autocorr(F, subtract_zone_mean=True):
    """Column-direction autocorrelation, optionally after zone-mean subtraction."""
    Fc = F.copy()
    if subtract_zone_mean:
        for z in range(N_ZONES):
            mask = zone_id == z
            if mask.any():
                Fc[mask] -= F[mask].mean(0)
    F_grid = Fc.reshape(H, HALF, HS)
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
def run(seed, birth_alpha=0.0):
    """
    birth_alpha in [0,1]:
      0.0 -> uniform random neighbour (current VCML, Allen-Cahn)
      1.0 -> always pick highest-|F| neighbour (fitness-proportionate, Burgers)
    """
    rng    = np.random.default_rng(seed)
    h      = rng.normal(0, 0.1, (N_ACT, HS))
    F      = np.zeros((N_ACT, HS))
    m      = np.zeros((N_ACT, HS))
    base   = h.copy()
    streak = np.zeros(N_ACT, int)
    ttl    = rng.geometric(p=max(NU, 1e-6), size=N_ACT).astype(float)
    waves  = []
    result = {}

    for step in range(1, T_END + 1):

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
                dF[i] = KAPPA * (F[NB[i]].mean(0) - F[i])
        F += dF

        ttl -= 1.0
        ttl[pert] -= 1.0
        dead = np.where(ttl <= 0)[0]
        for i in dead:
            nbs = NB[i]
            if len(nbs) == 0:
                j = i
            elif birth_alpha <= 0.0 or rng.random() > birth_alpha:
                # Uniform random (Allen-Cahn regime)
                j = nbs[rng.integers(len(nbs))]
            else:
                # Fitness-proportionate: pick highest |F| neighbour
                f_mags = np.sum(F[nbs] ** 2, axis=1)
                total  = f_mags.sum()
                if total < 1e-12:
                    j = nbs[rng.integers(len(nbs))]
                else:
                    probs = f_mags / total
                    j     = nbs[rng.choice(len(nbs), p=probs)]

            h[i]      = (1 - SEED_BETA) * rng.normal(0, 0.1, HS) + SEED_BETA * F[j]
            m[i]      = 0
            streak[i] = 0
            ttl[i]    = float(rng.geometric(p=NU))

        waves = [[w[0], w[1] - 1, w[2], w[3]] for w in waves if w[1] - 1 > 0]

        if step in CPS_SET:
            result[f"sg4_{step}"]       = sg4_fn(F)
            result[f"corr_{step}"]      = autocorr(F, subtract_zone_mean=True)
            result[f"corr_full_{step}"] = autocorr(F, subtract_zone_mean=False)

    return result


def make_key(alpha, seed):
    return f"p24,{alpha:.8g},{seed}"


def _worker(args):
    alpha, seed = args
    return run(seed, birth_alpha=alpha)


RESULTS_FILE = os.path.join(os.path.dirname(__file__), "results", "paper24_results.json")


# ── Analysis helpers ──────────────────────────────────────────────────────────
XI_CAP = 30.0


def fit_xi(corr):
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


def xi_trajectory(results, alpha):
    """Return (times, means) of xi, capped at XI_CAP."""
    by_t = {t: [] for t in CPS}
    for seed in range(N_SEEDS):
        key = make_key(alpha, seed)
        if key not in results:
            continue
        for t in CPS:
            corr = results[key].get(f"corr_{t}")
            if corr is not None:
                xi = fit_xi(corr)
                if not math.isnan(xi) and xi <= XI_CAP:
                    by_t[t].append(xi)
    times, means, sems = [], [], []
    for t in CPS:
        v = by_t[t]
        if v:
            times.append(t)
            means.append(float(np.mean(v)))
            sems.append(float(np.std(v) / math.sqrt(len(v))) if len(v) > 1 else 0.0)
    return times, means, sems


def fit_late_beta(times, means, t_min=1400):
    """Log-log fit of xi ~ t^beta using only t >= t_min."""
    valid = [(t, x) for t, x in zip(times, means) if t >= t_min and x > 0.05]
    if len(valid) < 3:
        return float("nan"), float("nan")
    lt = np.log([v[0] for v in valid])
    lx = np.log([v[1] for v in valid])
    slope, _, r, _, _ = linregress(lt, lx)
    return slope, r ** 2


if __name__ == "__main__":
    mp.freeze_support()

    all_conditions = [(alpha, seed)
                      for alpha in ALPHA_VALS
                      for seed in range(N_SEEDS)]

    if os.path.exists(RESULTS_FILE):
        results = json.load(open(RESULTS_FILE))
        print(f"Loaded {len(results)} cached results.")
    else:
        results = {}

    todo = [args for args in all_conditions if make_key(*args) not in results]

    if todo:
        print(f"Running {len(todo)} / {len(all_conditions)} simulations (Paper 24)...")
        with mp.Pool(processes=min(len(todo), mp.cpu_count())) as pool:
            raw = pool.map(_worker, todo)
        for i, args in enumerate(todo):
            results[make_key(*args)] = raw[i]
        os.makedirs(os.path.dirname(RESULTS_FILE), exist_ok=True)
        json.dump(results, open(RESULTS_FILE, "w"), indent=2)
        print("Saved results.")
    else:
        print("All results cached.")

    # ── Analysis ──────────────────────────────────────────────────────────────
    print()
    print("=" * 65)
    print("BIRTH BIAS AXIS  (Paper 24)")
    print("alpha=0: Allen-Cahn (uniform)  alpha=1: Burgers (fitness)")
    print("=" * 65)

    print(f"\n  {'alpha':>6} | {'xi_early':>8} | {'xi_late':>8} | "
          f"{'delta_xi':>9} | {'beta':>6} | {'R2':>6} | {'sg4_end':>8}")
    print("  " + "-" * 65)

    for alpha in ALPHA_VALS:
        times, means, _ = xi_trajectory(results, alpha)

        # Early xi (first available)
        xi_early = means[0] if means else float("nan")

        # Late xi (last few points)
        late = [(t, x) for t, x in zip(times, means) if t >= 2400]
        xi_late = float(np.mean([v[1] for v in late])) if late else float("nan")

        delta_xi = xi_late - xi_early if not math.isnan(xi_early) and \
                   not math.isnan(xi_late) else float("nan")

        beta, r2 = fit_late_beta(times, means)

        # sg4 at T_END
        sg4s = [results[make_key(alpha, s)].get(f"sg4_{T_END}", 0)
                for s in range(N_SEEDS) if make_key(alpha, s) in results]
        sg4_end = float(np.mean(sg4s)) if sg4s else float("nan")

        sign = "+" if not math.isnan(delta_xi) and delta_xi > 0 else ""
        print(f"  {alpha:>6.1f} | {xi_early:>8.2f} | {xi_late:>8.2f} | "
              f"{sign}{delta_xi:>8.2f} | {beta:>6.3f} | {r2:>6.3f} | "
              f"{sg4_end:>8.1f}")

    print()
    print("  Prediction: delta_xi goes from negative (Allen-Cahn, alpha=0)")
    print("  to positive (Burgers, alpha=1). Sign flip = universality class transition.")
