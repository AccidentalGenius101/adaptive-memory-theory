"""
paper22_spatial_correlation.py -- Experiment for Paper 22

Spatial Correlation Function of fieldM in VCML.

Directly measures how spatial coherence builds up during Phase 1 by computing
the column-direction autocorrelation of zone-mean-subtracted fieldM at each
checkpoint.  From C(r, t) = <F(x) . F(x+r)> / <|F(x)|^2>, we extract:

  xi(t) -- correlation length (sites) via fit C(r) = exp(-r/xi)
  beta   -- growth law exponent: xi(t) ~ t^beta
             beta=0.5 -> diffusion  (xi ~ sqrt(D_eff * t))
             beta=1.0 -> directed   (xi ~ v_copy * t)

Nu sweep (same five values as Paper 21):
  NU in [0.0005, 0.001, 0.002, 0.004, 0.008]
  5 seeds each -> 25 runs
  T_END=3000, checkpoints every 200 steps (15 per run)

Per checkpoint: record sg4 AND corr[0..MAX_LAG] (column autocorrelation).

Key prediction from Paper 21 (C ~ nu^-0.40):
  If beta=1 (directed):  v_copy ~ nu^0.40
  If beta=0.5 (diffusive): D_eff ~ nu^0.80
Measuring beta resolves the propagation mechanism.
"""
import numpy as np, json, os, multiprocessing as mp, math
from scipy.stats import linregress

# ── Fixed parameters ──────────────────────────────────────────────────────────
H         = 20
N_ZONES   = 4
HALF      = 20
BASE_BETA = 0.005      # tau_base = 200
ALPHA_MID = 0.15
MID_DECAY = 0.99
FD        = 0.9997
FA        = 0.40
SS        = 10
WR        = 4.8
WAVE_DUR  = 15
HS        = 2
KAPPA     = 0.020
SEED_BETA = 0.25

T_END    = 3000
CPS_SET  = set(range(200, T_END + 1, 200))
CPS      = sorted(CPS_SET)
N_ACT    = HALF * H
ZONE_W   = HALF // N_ZONES   # 5 sites per zone
MAX_LAG  = HALF // 2         # 10 (avoid wrapping artefacts)
TAU_BASE = 1.0 / BASE_BETA   # 200

NU_VALS = [0.0005, 0.001, 0.002, 0.004, 0.008]
N_SEEDS = 5

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
    """
    Column-direction spatial autocorrelation of zone-mean-subtracted F.

    Steps:
      1. Subtract zone mean from each site -> F_res
      2. Reshape to (H, HALF, HS) grid
      3. C(r) = mean_dot(F_res[:, c], F_res[:, c+r]) / variance  for r=0..MAX_LAG

    Returns list of MAX_LAG+1 floats (C(0)=1 always).
    nan entries if variance is too small.
    """
    F_res = F.copy()
    for z in range(N_ZONES):
        mask = zone_id == z
        if mask.any():
            F_res[mask] -= F[mask].mean(0)

    F_grid = F_res.reshape(H, HALF, HS)   # (H, HALF, HS)
    var_0  = float(np.mean(np.sum(F_grid ** 2, axis=2)))

    if var_0 < 1e-12:
        return [float("nan")] * (MAX_LAG + 1)

    result = [1.0]   # C(0) = 1 by definition
    for r in range(1, MAX_LAG + 1):
        sub1 = F_grid[:, :HALF - r, :]   # (H, HALF-r, HS)
        sub2 = F_grid[:, r:, :]           # (H, HALF-r, HS)
        dot  = np.sum(sub1 * sub2, axis=2)  # (H, HALF-r)
        result.append(float(dot.mean() / var_0))

    return result


# ── Simulation ────────────────────────────────────────────────────────────────
def run(seed, nu):
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

        # Phase 1 only -- waves run throughout
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


def make_key(nu, seed):
    return f"p22,{nu:.8g},{seed}"


def _worker(args):
    nu, seed = args
    return run(seed, nu)


RESULTS_FILE = os.path.join(os.path.dirname(__file__), "results", "paper22_results.json")


# ── Analysis helpers ──────────────────────────────────────────────────────────
def fit_xi(corr):
    """
    Extract correlation length xi from C(r) = exp(-r/xi).
    Log-linear regression on r=1..min(ZONE_W, 4), positive C only.
    Returns xi in sites, or nan.
    """
    r_vals, log_c = [], []
    for r in range(1, min(ZONE_W + 1, MAX_LAG + 1)):
        if r >= len(corr):
            break
        c = corr[r]
        if not math.isnan(c) and c > 0.05:
            r_vals.append(r)
            log_c.append(math.log(c))
    if len(r_vals) < 2:
        return float("nan")
    slope, _, _, _, _ = linregress(r_vals, log_c)
    if slope >= 0:
        return float("nan")
    return -1.0 / slope


def xi_trajectory(results, nu):
    """Return (times, mean_xi, sem_xi) across seeds for a given nu."""
    xi_by_t = {t: [] for t in CPS}
    for seed in range(N_SEEDS):
        key = make_key(nu, seed)
        if key not in results:
            continue
        r = results[key]
        for t in CPS:
            corr = r.get(f"corr_{t}")
            if corr is not None:
                xi = fit_xi(corr)
                if not math.isnan(xi):
                    xi_by_t[t].append(xi)

    times, means, sems = [], [], []
    for t in CPS:
        v = xi_by_t[t]
        if v:
            times.append(t)
            means.append(float(np.mean(v)))
            sems.append(float(np.std(v) / math.sqrt(len(v))) if len(v) > 1 else 0.0)
    return times, means, sems


def fit_growth_law(times, xi_means):
    """Fit xi ~ t^beta via log-log regression. Returns (beta, A, R2)."""
    valid = [(t, x) for t, x in zip(times, xi_means) if x > 0.05]
    if len(valid) < 3:
        return float("nan"), float("nan"), float("nan")
    log_t  = np.log([v[0] for v in valid])
    log_xi = np.log([v[1] for v in valid])
    slope, intercept, r, _, _ = linregress(log_t, log_xi)
    return slope, math.exp(intercept), r ** 2


if __name__ == "__main__":
    mp.freeze_support()

    all_conditions = [(nu, seed) for nu in NU_VALS for seed in range(N_SEEDS)]

    if os.path.exists(RESULTS_FILE):
        results = json.load(open(RESULTS_FILE))
        print(f"Loaded {len(results)} cached results.")
    else:
        results = {}

    todo = [args for args in all_conditions if make_key(*args) not in results]

    if todo:
        print(f"Running {len(todo)} / {len(all_conditions)} simulations (Paper 22)...")
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
    print(f"SPATIAL CORRELATION ANALYSIS  (zone_width={ZONE_W} sites, "
          f"tau_base={TAU_BASE:.0f})")
    print("=" * 65)

    # Growth law table
    print(f"\n  {'nu':>8} | {'xi(T_END)':>10} | {'beta':>6} | {'A':>8} | "
          f"{'R2':>6} | {'v_copy':>8} | {'D_eff':>8}")
    print("  " + "-" * 65)

    betas, v_copies, D_effs = [], [], []
    for nu in NU_VALS:
        times, means, _ = xi_trajectory(results, nu)
        beta, A, r2 = fit_growth_law(times, means)
        xi_final = means[-1] if means else float("nan")
        # Late-time estimates
        t_late = [t for t in times if t >= 1000]
        xi_late = [means[times.index(t)] for t in t_late]
        v_copy = float(np.mean([xi / t for xi, t in zip(xi_late, t_late)])) if xi_late else float("nan")
        D_eff  = float(np.mean([xi**2 / (2*t) for xi, t in zip(xi_late, t_late)])) if xi_late else float("nan")

        betas.append(beta)
        v_copies.append(v_copy)
        D_effs.append(D_eff)

        print(f"  {nu:>8.4f} | {xi_final:>10.3f} | {beta:>6.3f} | {A:>8.4f} | "
              f"{r2:>6.3f} | {v_copy:>8.5f} | {D_eff:>8.5f}")

    # Power-law fit of v_copy vs nu
    valid_nu = [(nu, v) for nu, v in zip(NU_VALS, v_copies) if not math.isnan(v)]
    if len(valid_nu) >= 3:
        log_nu   = np.log([x[0] for x in valid_nu])
        log_v    = np.log([x[1] for x in valid_nu])
        slope_v, _, r_v, _, _ = linregress(log_nu, log_v)
        print(f"\n  v_copy ~ nu^{slope_v:.3f}  (R2={r_v**2:.4f})")
        print(f"  Paper 21 predicted C ~ nu^(-0.40) -> v_copy ~ nu^0.40 (if beta=1)")

    valid_nu_d = [(nu, d) for nu, d in zip(NU_VALS, D_effs) if not math.isnan(d)]
    if len(valid_nu_d) >= 3:
        log_nu  = np.log([x[0] for x in valid_nu_d])
        log_d   = np.log([x[1] for x in valid_nu_d])
        slope_d, _, r_d, _, _ = linregress(log_nu, log_d)
        print(f"  D_eff  ~ nu^{slope_d:.3f}  (R2={r_d**2:.4f})")
        print(f"  Paper 21 predicted C ~ nu^(-0.40) -> D_eff ~ nu^0.80 (if beta=0.5)")

    print(f"\n  Mean beta = {float(np.nanmean(betas)):.3f}")
    print(f"  beta=0.5 -> diffusion,  beta=1.0 -> directed propagation")
