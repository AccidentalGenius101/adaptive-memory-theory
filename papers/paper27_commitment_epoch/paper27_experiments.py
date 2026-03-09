"""
paper27_experiments.py -- Experiments for Paper 27

The Zone Formation Commitment Epoch:
  Exp A (NEW, 80 runs): FA x WR sweep -- commitment timescale t*
    FA in {0.10, 0.20, 0.40, 0.80} x WR in {1.2, 2.4, 4.8, 9.6}
    T_END=5000, 5 seeds. store_F=True for temporal correlations.
    Tests: t* ~ 1/(P_c * FA * WR), sg4 saturation.

  Exp B (NEW, 15 runs): Zone-level trajectory -- pitchfork vs spinodal
    Standard params (FA=0.40, WR=4.8), 15 seeds, T_END=6000.
    Stores zone means at each checkpoint only (tiny footprint).
    Tests: within-seed t*_z spread vs between-seed spread.

Total: 95 new runs.
"""
import numpy as np, json, os, multiprocessing as mp, math
from scipy.stats import linregress

# ── Fixed parameters (identical to Papers 22-26) ─────────────────────────────
H         = 20
N_ZONES   = 4
HALF      = 20
BASE_BETA = 0.005
ALPHA_MID = 0.15
MID_DECAY = 0.99
FD        = 0.9997
SS        = 10
KAPPA     = 0.020
WAVE_DUR  = 15
HS        = 2
SEED_BETA = 0.25
NU        = 0.001

# Standard defaults
WR_STD = 4.8
FA_STD = 0.40
ZONE_W = HALF // N_ZONES   # 5
MAX_LAG = HALF // 2        # 10

# ── Experiment grids ──────────────────────────────────────────────────────────
N_SEEDS_A = 5
N_SEEDS_B = 15
FA_VALS_A = [0.10, 0.20, 0.40, 0.80]
WR_VALS_A = [1.2, 2.4, 4.8, 9.6]
T_END_A   = 5000
CPS_A     = list(range(400, T_END_A + 1, 400))   # 12 checkpoints
T_END_B   = 6000
CPS_B     = list(range(200, T_END_B + 1, 200))   # 30 checkpoints

# ── Geometry ──────────────────────────────────────────────────────────────────
N_ACT    = HALF * H
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


def zone_means_fn(F):
    """Zone-mean field vectors, one per zone."""
    return [F[zone_id == z].mean(0).tolist() for z in range(N_ZONES)]


def autocorr_fn(F):
    Fc = F.copy()
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


def fit_xi(corr, zone_w=ZONE_W):
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
    slope, *_ = linregress(r_vals, log_c)
    return float("nan") if slope >= 0 else -1.0 / slope


# ── Core simulation ───────────────────────────────────────────────────────────
def run(seed, fa=FA_STD, wr=WR_STD, t_end=T_END_A, cps=None, store_F=True):
    if cps is None:
        cps = CPS_A
    cps_set = set(cps)

    rng    = np.random.default_rng(seed)
    h      = rng.normal(0, 0.1, (N_ACT, HS))
    F      = np.zeros((N_ACT, HS))
    m      = np.zeros((N_ACT, HS))
    base   = h.copy()
    streak = np.zeros(N_ACT, int)
    ttl    = rng.geometric(p=max(NU, 1e-6), size=N_ACT).astype(float)
    waves  = []
    result = {}

    for step in range(1, t_end + 1):

        # Wave launch (Poisson-like, parametrised by wr)
        n = int(wr / WAVE_DUR)
        n += int(rng.random() < (wr / WAVE_DUR - n))
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
                dF[i] = KAPPA * (F[NB[i]].mean(0) - F[i])
        F += dF

        ttl -= 1.0
        ttl[pert] -= 1.0
        dead = np.where(ttl <= 0)[0]
        for i in dead:
            nbs = NB[i]
            if len(nbs) == 0:
                j = i
            else:
                j = nbs[rng.integers(len(nbs))]
            h[i]      = (1 - SEED_BETA) * rng.normal(0, 0.1, HS) + SEED_BETA * F[j]
            m[i]      = 0
            streak[i] = 0
            ttl[i]    = float(rng.geometric(p=NU))

        waves = [[w[0], w[1] - 1, w[2], w[3]] for w in waves if w[1] - 1 > 0]

        if step in cps_set:
            result[f"sg4_{step}"]      = sg4_fn(F)
            result[f"autocorr_{step}"] = autocorr_fn(F)
            result[f"zmeans_{step}"]   = zone_means_fn(F)
            if store_F:
                result[f"F_{step}"]    = F.tolist()

    return result


# ── Key factories ─────────────────────────────────────────────────────────────
def key_a(fa, wr, seed):
    return f"p27a,{fa:.8g},{wr:.8g},{seed}"


def key_b(seed):
    return f"p27b,{seed}"


# ── Worker (module-level for Windows spawn) ───────────────────────────────────
def _worker(args):
    tag, params, seed = args
    if tag == "a":
        fa, wr = params
        return run(seed, fa=fa, wr=wr, t_end=T_END_A, cps=CPS_A, store_F=True)
    else:  # tag == "b"
        return run(seed, fa=FA_STD, wr=WR_STD, t_end=T_END_B, cps=CPS_B,
                   store_F=False)


RESULTS_FILE = os.path.join(os.path.dirname(__file__),
                            "results", "paper27_results.json")


if __name__ == "__main__":
    mp.freeze_support()

    # Build full condition list
    all_conditions = []
    # Exp A
    for fa in FA_VALS_A:
        for wr in WR_VALS_A:
            for seed in range(N_SEEDS_A):
                all_conditions.append(("a", (fa, wr), seed, key_a(fa, wr, seed)))
    # Exp B
    for seed in range(N_SEEDS_B):
        all_conditions.append(("b", (), seed, key_b(seed)))

    # Load cache
    if os.path.exists(RESULTS_FILE):
        results = json.load(open(RESULTS_FILE))
        print(f"Loaded {len(results)} cached results.")
    else:
        results = {}

    todo = [(tag, params, seed)
            for tag, params, seed, key in all_conditions
            if key not in results]
    todo_keys = [key for tag, params, seed, key in all_conditions
                 if key not in results]

    if todo:
        print(f"Running {len(todo)} / {len(all_conditions)} simulations (Paper 27)...")
        with mp.Pool(processes=min(len(todo), mp.cpu_count())) as pool:
            raw = pool.map(_worker, todo)
        for key, res in zip(todo_keys, raw):
            results[key] = res
        os.makedirs(os.path.dirname(RESULTS_FILE), exist_ok=True)
        json.dump(results, open(RESULTS_FILE, "w"), indent=2)
        print("Saved results.")
    else:
        print("All results cached.")

    # ── Helpers ───────────────────────────────────────────────────────────────
    P_C = 0.175   # estimated P_consol

    def compute_temporal_C(fa, wr, seed, t_ref):
        """Cosine similarity between zone-mean-subtracted F(t_ref) and F(T_END_A)."""
        k = key_a(fa, wr, seed)
        if k not in results:
            return float("nan")
        F_ref = results[k].get(f"F_{t_ref}")
        F_end = results[k].get(f"F_{T_END_A}")
        if F_ref is None or F_end is None:
            return float("nan")
        F_ref = np.array(F_ref, dtype=float)
        F_end = np.array(F_end, dtype=float)
        for z in range(N_ZONES):
            mask = zone_id == z
            if mask.any():
                F_ref[mask] -= F_ref[mask].mean(0)
                F_end[mask] -= F_end[mask].mean(0)
        n1, n2 = np.linalg.norm(F_ref), np.linalg.norm(F_end)
        if n1 < 1e-10 or n2 < 1e-10:
            return float("nan")
        return float(np.sum(F_ref * F_end) / (n1 * n2))

    def find_t_star_a(fa, wr):
        """Zero-crossing of mean C(t_ref, T_END_A) over seeds."""
        prev_c, prev_t = None, None
        for t in CPS_A:
            c_vals = [compute_temporal_C(fa, wr, s, t)
                      for s in range(N_SEEDS_A)]
            c_vals = [v for v in c_vals if not math.isnan(v)]
            if not c_vals:
                continue
            c = float(np.mean(c_vals))
            if prev_c is not None and prev_c < 0 <= c:
                # Linear interpolation
                return prev_t + (t - prev_t) * (-prev_c) / (c - prev_c)
            prev_c, prev_t = c, t
        return float("nan")

    def find_t_sat(fa, wr):
        """First t where sg4 changes < 2% over the next checkpoint."""
        sg4_prev = None
        for t in CPS_A:
            sg4s = [results[key_a(fa, wr, s)].get(f"sg4_{t}", float("nan"))
                    for s in range(N_SEEDS_A) if key_a(fa, wr, s) in results]
            sg4s = [v for v in sg4s if not math.isnan(v)]
            if not sg4s:
                continue
            sg4 = float(np.mean(sg4s))
            if sg4_prev is not None and sg4_prev > 1e-4:
                if abs(sg4 - sg4_prev) / sg4_prev < 0.02:
                    return t
            sg4_prev = sg4
        return float("nan")

    # ── Analysis printout: Exp A ───────────────────────────────────────────────
    print()
    print("=" * 72)
    print("EXP A: Commitment epoch t* and saturation")
    print("=" * 72)
    print(f"  {'FA':>5} {'WR':>5} | {'t*':>8} {'P_c*FA*WR':>12} {'t**Pc*FA*WR':>13} | {'T_sat':>7}")
    print("  " + "-" * 60)
    for fa in FA_VALS_A:
        for wr in WR_VALS_A:
            t_star = find_t_star_a(fa, wr)
            t_sat  = find_t_sat(fa, wr)
            rate   = P_C * fa * wr
            product = t_star * rate if not math.isnan(t_star) else float("nan")
            print(f"  {fa:>5.2f} {wr:>5.1f} | {t_star:>8.0f} {rate:>12.4f} {product:>13.2f} | {t_sat:>7.0f}")

    print()
    print("=" * 72)
    print("EXP B: Zone-level commitment -- pitchfork vs spinodal")
    print("=" * 72)

    def compute_zone_C(seed, zone_z, t_ref):
        """Cosine similarity of zone z mean vector between t_ref and T_END_B."""
        k = key_b(seed)
        if k not in results:
            return float("nan")
        zm_ref = results[k].get(f"zmeans_{t_ref}")
        zm_end = results[k].get(f"zmeans_{T_END_B}")
        if zm_ref is None or zm_end is None:
            return float("nan")
        v1 = np.array(zm_ref[zone_z]); v2 = np.array(zm_end[zone_z])
        n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
        if n1 < 1e-10 or n2 < 1e-10:
            return float("nan")
        return float(np.dot(v1, v2) / (n1 * n2))

    def find_t_star_zone(seed, zone_z):
        prev_c, prev_t = None, None
        for t in CPS_B:
            c = compute_zone_C(seed, zone_z, t)
            if math.isnan(c):
                continue
            if prev_c is not None and prev_c < 0 <= c:
                return prev_t + (t - prev_t) * (-prev_c) / (c - prev_c)
            prev_c, prev_t = c, t
        return float("nan")

    # Collect t*_z for each (seed, zone)
    t_star_matrix = []   # shape: [N_SEEDS_B, N_ZONES]
    for seed in range(N_SEEDS_B):
        row = [find_t_star_zone(seed, z) for z in range(N_ZONES)]
        t_star_matrix.append(row)

    t_star_matrix = np.array(t_star_matrix, dtype=float)  # (15, 4)

    # Within-seed spread: std over zones for each seed
    within_seed_stds = []
    for seed_row in t_star_matrix:
        valid = seed_row[~np.isnan(seed_row)]
        if len(valid) > 1:
            within_seed_stds.append(float(np.std(valid)))

    # Between-seed spread: std of per-seed mean t*
    seed_means = np.nanmean(t_star_matrix, axis=1)
    valid_means = seed_means[~np.isnan(seed_means)]
    between_seed_std = float(np.std(valid_means)) if len(valid_means) > 1 else float("nan")

    print(f"  Per-zone mean t* (across seeds):")
    for z in range(N_ZONES):
        col = t_star_matrix[:, z]
        valid = col[~np.isnan(col)]
        print(f"    Zone {z}: mean={np.mean(valid):.0f} +/- {np.std(valid):.0f} (n={len(valid)})")

    sigma_in  = float(np.mean(within_seed_stds)) if within_seed_stds else float("nan")
    print(f"\n  sigma_in  (within-seed zone spread):  {sigma_in:.1f} steps")
    print(f"  sigma_btw (between-seed t* spread):   {between_seed_std:.1f} steps")
    if not math.isnan(sigma_in) and not math.isnan(between_seed_std):
        ratio = sigma_in / between_seed_std
        verdict = "PITCHFORK (zones flip together)" if ratio < 0.5 else \
                  "SPINODAL (zones flip independently)" if ratio > 1.5 else "MIXED"
        print(f"  sigma_in / sigma_btw = {ratio:.2f}  -->  {verdict}")
