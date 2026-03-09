"""
paper26_experiments.py -- Experiments for Paper 26

Four-part culmination of the spatial dynamics arc (Papers 22-25):

  Part A (FREE -- Paper 25 data):
    Verify unified stochastic PDE noise amplitude scaling.
    xi_inf^2 vs beta_s^2 from Paper 25 Exp3A: test self-consistency equation
      P_c*FA*xi^2 - nu*beta_s*|F|*xi - kappa = 0  (quadratic in xi)
    No new simulations required.

  Part B (NEW -- 50 runs):
    Cross the Burgers threshold.
    kappa in {0.0002, 0.0003, 0.0005, 0.001, 0.002}, alpha in {0.0, 1.0}
    nu=0.001 fixed -> nu/kappa = {5.0, 3.3, 2.0, 1.0, 0.5}
    Prediction: sign flip in delta_xi at nu/kappa ~ 3.5 when alpha=1.

  Part C (NEW -- 25 runs):
    Zone width vs. correlation length.
    N_ZONES in {2, 4, 5, 10, 20} with HALF=20 -> ZONE_W in {10, 5, 4, 2, 1}
    Standard params. Measure sg4 and xi_inf vs ZONE_W.
    Prediction: inverted U-shape peaking near ZONE_W ~ 3*xi_inf ~ 7-12 sites.

  Part D (NEW -- 5 runs, stores full F field):
    Temporal correlations and aging dynamics.
    C(t1, t2) = <F_res(x,t1).F_res(x,t2)> / sqrt(var(t1)*var(t2))
    Measures: does the system "age"? Older epochs (t_ref large) should
    decorrelate slower because zones are more established.
"""
import numpy as np, json, os, multiprocessing as mp, math
from scipy.stats import linregress

# ── Fixed parameters ──────────────────────────────────────────────────────────
H             = 20
HALF          = 20
BASE_BETA     = 0.005
ALPHA_MID     = 0.15
MID_DECAY     = 0.99
FD            = 0.9997
SS            = 10
WR            = 4.8
WAVE_DUR      = 15
HS            = 2
SEED_BETA_STD = 0.25
KAPPA_STD     = 0.020
FA_STD        = 0.40
NU            = 0.001
N_ZONES_STD   = 4

T_END   = 3000
CPS_SET = set(range(200, T_END + 1, 200))
CPS     = sorted(CPS_SET)
N_ACT   = HALF * H
MAX_LAG = HALF // 2   # 10

# ── Experiment grids ──────────────────────────────────────────────────────────
N_SEEDS       = 5
KAPPA_VALS_B  = [0.0002, 0.0003, 0.0005, 0.001, 0.002]   # Part B
ALPHA_B       = [0.0, 1.0]                                 # Part B
N_ZONES_C     = [2, 4, 5, 10, 20]                         # Part C (divides 20)

# ── Standard geometry (Parts A, B, D) ─────────────────────────────────────────
_col_std     = np.arange(N_ACT) % HALF
_row_std     = np.arange(N_ACT) // HALF
_zone_std    = _col_std // (HALF // N_ZONES_STD)
_top_std     = _row_std < H // 2
_bot_std     = _row_std >= H // 2
d_B          = np.array([0.0, 1.0])

NB_STD = []
for _i in range(N_ACT):
    c, r = int(_col_std[_i]), int(_row_std[_i])
    nb = []
    for dc, dr in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        nc, nr = c + dc, r + dr
        if 0 <= nc < HALF and 0 <= nr < H:
            nb.append(nr * HALF + nc)
    NB_STD.append(np.array(nb, dtype=int))


# ── Geometry builder for Part C (variable N_ZONES) ────────────────────────────
def make_zone_id(n_zones):
    """Column-direction zone assignment for HALF=20 grid."""
    zone_w = HALF // n_zones
    return (_col_std // zone_w).clip(0, n_zones - 1)


# ── Measurements ──────────────────────────────────────────────────────────────
def sg4_fn(F, zone_id, n_zones):
    means = [F[zone_id == z].mean(0) for z in range(n_zones)
             if (zone_id == z).any()]
    if len(means) < 2:
        return 0.0
    pairs = [(i, j) for i in range(len(means))
             for j in range(i + 1, len(means))]
    return float(np.mean([np.linalg.norm(means[i] - means[j])
                          for i, j in pairs]))


def autocorr(F, zone_id, n_zones, subtract_zone_mean=True):
    Fc = F.copy()
    if subtract_zone_mean:
        for z in range(n_zones):
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
        result.append(float(np.sum(sub1 * sub2, axis=2).mean() / var_0))
    return result


# ── Core simulation ───────────────────────────────────────────────────────────
def run(seed, birth_alpha=0.0, kappa=KAPPA_STD,
        seed_beta=SEED_BETA_STD, fa=FA_STD,
        n_zones=N_ZONES_STD, store_F=False):

    zone_id = _zone_std if n_zones == N_ZONES_STD else make_zone_id(n_zones)
    NB      = NB_STD   # NB only depends on H, HALF (fixed for all experiments)
    top_mask, bot_mask = _top_std, _bot_std

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
            nbs = NB[i]
            if len(nbs) == 0:
                j = i
            elif birth_alpha <= 0.0 or rng.random() > birth_alpha:
                j = nbs[rng.integers(len(nbs))]
            else:
                f_mags = np.sum(F[nbs] ** 2, axis=1)
                total  = f_mags.sum()
                if total < 1e-12:
                    j = nbs[rng.integers(len(nbs))]
                else:
                    j = nbs[rng.choice(len(nbs), p=f_mags / total)]

            h[i]      = ((1 - seed_beta) * rng.normal(0, 0.1, HS)
                         + seed_beta * F[j])
            m[i]      = 0
            streak[i] = 0
            ttl[i]    = float(rng.geometric(p=NU))

        waves = [[w[0], w[1]-1, w[2], w[3]] for w in waves if w[1]-1 > 0]

        if step in CPS_SET:
            result[f"sg4_{step}"]  = sg4_fn(F, zone_id, n_zones)
            result[f"corr_{step}"] = autocorr(F, zone_id, n_zones,
                                              subtract_zone_mean=True)
            if store_F:
                result[f"F_{step}"] = F.tolist()   # full field for temporal corr

    return result


# ── Key factories ─────────────────────────────────────────────────────────────
def key_b(alpha, kappa, seed):
    return f"p26b,{alpha:.8g},{kappa:.8g},{seed}"

def key_c(n_zones, seed):
    return f"p26c,{n_zones},{seed}"

def key_d(seed):
    return f"p26d,{seed}"


def _worker(args):
    tag, params, seed = args
    if tag == "b":
        alpha, kappa = params
        return run(seed, birth_alpha=alpha, kappa=kappa)
    elif tag == "c":
        nz, = params
        return run(seed, n_zones=nz)
    else:  # "d"
        return run(seed, store_F=True)


RESULTS_FILE = os.path.join(os.path.dirname(__file__),
                            "results", "paper26_results.json")


if __name__ == "__main__":
    mp.freeze_support()

    all_conditions = []
    # Part B
    for alpha in ALPHA_B:
        for kappa in KAPPA_VALS_B:
            for seed in range(N_SEEDS):
                all_conditions.append(("b", (alpha, kappa), seed,
                                       key_b(alpha, kappa, seed)))
    # Part C
    for nz in N_ZONES_C:
        for seed in range(N_SEEDS):
            all_conditions.append(("c", (nz,), seed, key_c(nz, seed)))
    # Part D
    for seed in range(N_SEEDS):
        all_conditions.append(("d", (), seed, key_d(seed)))

    # Load cache
    if os.path.exists(RESULTS_FILE):
        results = json.load(open(RESULTS_FILE))
        print(f"Loaded {len(results)} cached results.")
    else:
        results = {}

    todo      = [(tag, params, seed)
                 for tag, params, seed, key in all_conditions
                 if key not in results]
    todo_keys = [key for tag, params, seed, key in all_conditions
                 if key not in results]

    if todo:
        print(f"Running {len(todo)} / {len(all_conditions)} simulations (Paper 26)...")
        with mp.Pool(processes=min(len(todo), mp.cpu_count())) as pool:
            raw = pool.map(_worker, todo)
        for key, res in zip(todo_keys, raw):
            results[key] = res
        os.makedirs(os.path.dirname(RESULTS_FILE), exist_ok=True)
        json.dump(results, open(RESULTS_FILE, "w"), indent=2)
        print("Saved results.")
    else:
        print("All results cached.")

    # ── Quick analysis ────────────────────────────────────────────────────────
    XI_CAP = 30.0

    def fit_xi(corr, zone_w=5):
        r_vals, log_c = [], []
        for r in range(1, min(zone_w + 1, MAX_LAG + 1)):
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
        return min(-1.0 / slope, XI_CAP)

    def xi_late(key_fn, t_min=2400, zone_w=5):
        vals = []
        for seed in range(N_SEEDS):
            k = key_fn(seed)
            if k not in results:
                continue
            for t in CPS:
                if t < t_min:
                    continue
                corr = results[k].get(f"corr_{t}")
                if corr:
                    xi = fit_xi(corr, zone_w)
                    if not math.isnan(xi):
                        vals.append(xi)
        return float(np.mean(vals)) if vals else float("nan")

    print()
    print("=" * 65)
    print("PART B: Burgers threshold crossing")
    print(f"  {'kappa':>8} | {'nu/kappa':>9} | {'xi(a=0)':>9} | "
          f"{'xi(a=1)':>9} | {'ratio':>7} | {'Dxi(a=1)':>10}")
    print("  " + "-" * 65)

    def delta_xi(key_fn, zone_w=5):
        early_vals, late_vals = [], []
        for seed in range(N_SEEDS):
            k = key_fn(seed)
            if k not in results:
                continue
            for t in CPS:
                corr = results[k].get(f"corr_{t}")
                if corr:
                    xi = fit_xi(corr, zone_w)
                    if not math.isnan(xi):
                        if t <= 1200:
                            early_vals.append(xi)
                        elif t >= 2400:
                            late_vals.append(xi)
        em = float(np.mean(early_vals)) if early_vals else float("nan")
        lm = float(np.mean(late_vals)) if late_vals else float("nan")
        return lm - em if not (math.isnan(em) or math.isnan(lm)) else float("nan")

    for kappa in KAPPA_VALS_B:
        xi0 = xi_late(lambda s, k=kappa: key_b(0.0, k, s))
        xi1 = xi_late(lambda s, k=kappa: key_b(1.0, k, s))
        dxi = delta_xi(lambda s, k=kappa: key_b(1.0, k, s))
        ratio = xi1/xi0 if not math.isnan(xi0) and xi0 > 0.01 else float("nan")
        sign = "+" if not math.isnan(dxi) and dxi > 0 else ""
        print(f"  {kappa:>8.4f} | {NU/kappa:>9.3f} | {xi0:>9.2f} | {xi1:>9.2f} | "
              f"{ratio:>7.3f} | {sign}{dxi:>9.2f}")

    print()
    print("=" * 65)
    print("PART C: Zone width sweep")
    print(f"  {'n_zones':>8} | {'zone_w':>7} | {'xi_inf':>8} | {'sg4_end':>9}")
    print("  " + "-" * 42)
    for nz in N_ZONES_C:
        zw = HALF // nz
        xi = xi_late(lambda s, n=nz: key_c(n, s), zone_w=zw)
        sg4s = []
        for seed in range(N_SEEDS):
            k = key_c(nz, seed)
            if k in results:
                v = results[k].get(f"sg4_{T_END}")
                if v is not None:
                    sg4s.append(v)
        sg4m = float(np.mean(sg4s)) if sg4s else float("nan")
        print(f"  {nz:>8} | {zw:>7} | {xi:>8.2f} | {sg4m:>9.1f}")

    print()
    print("=" * 65)
    print("PART D: Temporal correlation snapshot")
    # Print C(t, t=3000) for a few reference times
    zone_id_arr = _zone_std
    print(f"  Correlation with t=3000 for different t_ref:")
    for t_ref in [400, 800, 1200, 1600, 2000, 2400, 2800, 3000]:
        vals = []
        for seed in range(N_SEEDS):
            k = key_d(seed)
            if k not in results:
                continue
            F1 = results[k].get(f"F_{t_ref}")
            F2 = results[k].get(f"F_{T_END}")
            if F1 is None or F2 is None:
                continue
            F1 = np.array(F1); F2 = np.array(F2)
            for z in range(N_ZONES_STD):
                mask = zone_id_arr == z
                if mask.any():
                    F1[mask] -= F1[mask].mean(0)
                    F2[mask] -= F2[mask].mean(0)
            n1 = np.linalg.norm(F1); n2 = np.linalg.norm(F2)
            if n1 > 1e-10 and n2 > 1e-10:
                vals.append(float(np.sum(F1 * F2) / (n1 * n2)))
        print(f"    C({t_ref:4d}, 3000) = {np.mean(vals):.3f}" if vals else
              f"    C({t_ref:4d}, 3000) = nan")
