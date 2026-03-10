"""
paper30_experiments.py -- The Spatial Amplification Factor Gamma

Paper 29 found K_eff^III = 0.037, three times smaller than K_eff^I = 0.117
(Paper 16). The ratio Gamma = K_eff^I / K_eff^III ~ 3.2 is a new coupling
constant. Paper 30 measures how Gamma depends on:
  - kappa (diffusion): prediction Gamma ~ ZONE_W / sqrt(kappa)
  - ZONE_W (zone width): prediction Gamma ~ ZONE_W / sqrt(kappa)
  - nu (collapse rate): prediction unclear (more collapses -> more copy-forward
    events -> more amplification, but also faster field erasure)

Exp A (kappa x ZONE_W grid, 135 runs):
  kappa in {0.005, 0.020, 0.080} x ZONE_W in {3, 5, 10}
  FA in {0.005, 0.020, 0.080, 0.200, 0.800} (5 values per condition)
  3 seeds, T_END=4000
  At each (kappa, ZONE_W): fit saturation law -> K_eff(kappa, ZONE_W)
  Test: K_eff ~ sqrt(kappa) / ZONE_W (i.e. Gamma ~ ZONE_W / sqrt(kappa))

Exp B (nu sweep, 45 runs):
  nu in {0.0003, 0.003, 0.010} (3 values; nu=0.001 already in Exp A standard)
  kappa=0.020, ZONE_W=5 (standard)
  FA in {0.005, 0.020, 0.080, 0.200, 0.800}
  3 seeds, T_END=4000

Total: 135 + 45 = 180 runs.
"""
import numpy as np, json, os, multiprocessing as mp, math
from scipy.optimize import curve_fit
from scipy.stats import linregress

# ── Fixed parameters ───────────────────────────────────────────────────────────
H         = 20
N_ZONES   = 4
BASE_BETA = 0.005
ALPHA_MID = 0.15
MID_DECAY = 0.99
FD        = 0.9997
SS        = 10
WAVE_DUR  = 15
HS        = 2
SEED_BETA = 0.25
WR_STD    = 2.4

# Standard defaults
KAPPA_STD  = 0.020
NU_STD     = 0.001
ZONE_W_STD = 5

# Paper 16 K_eff^I reference
K_EFF_I = 0.117

# ── Experiment grids ───────────────────────────────────────────────────────────
KAPPA_VALS  = [0.005, 0.020, 0.080]
ZONE_W_VALS = [3, 5, 10]
NU_VALS_B   = [0.0003, 0.003, 0.010]
FA_VALS     = [0.005, 0.020, 0.080, 0.200, 0.800]
N_SEEDS     = 3
T_END       = 4000
CPS         = [1000, 2000, 3000, 4000]

d_B = np.array([0.0, 1.0])


# ── Dynamic geometry builder ───────────────────────────────────────────────────
def make_geometry(zone_w, n_zones=N_ZONES, h=H):
    half    = n_zones * zone_w
    n_act   = half * h
    col     = np.arange(n_act) % half
    row     = np.arange(n_act) // half
    zone_id = col // zone_w
    top_mask = row < h // 2
    bot_mask = row >= h // 2
    NB = []
    for i in range(n_act):
        c, r = int(col[i]), int(row[i])
        nb = []
        for dc, dr in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nc, nr = c + dc, r + dr
            if 0 <= nc < half and 0 <= nr < h:
                nb.append(nr * half + nc)
        NB.append(np.array(nb, dtype=int))
    return n_act, half, zone_id, top_mask, bot_mask, NB


# ── Measurement ────────────────────────────────────────────────────────────────
def sg4_fn(F, zone_id, n_zones=N_ZONES):
    means = [F[zone_id == z].mean(0) for z in range(n_zones)]
    pairs = [(i, j) for i in range(n_zones) for j in range(i + 1, n_zones)]
    return float(np.mean([np.linalg.norm(means[i] - means[j]) for i, j in pairs]))


# ── Core simulation ────────────────────────────────────────────────────────────
def run(seed, kappa=KAPPA_STD, zone_w=ZONE_W_STD, nu=NU_STD, fa=0.10,
        wr=WR_STD, t_end=T_END, cps=None):
    if cps is None:
        cps = CPS
    cps_set = set(cps)

    n_act, half, zone_id, top_mask, bot_mask, NB = make_geometry(zone_w)

    rng    = np.random.default_rng(seed)
    h      = rng.normal(0, 0.1, (n_act, HS))
    F      = np.zeros((n_act, HS))
    m      = np.zeros((n_act, HS))
    base   = h.copy()
    streak = np.zeros(n_act, int)
    ttl    = rng.geometric(p=max(nu, 1e-6), size=n_act).astype(float)
    waves  = []
    result = {}

    for step in range(1, t_end + 1):

        n_launch = int(wr / WAVE_DUR)
        n_launch += int(rng.random() < (wr / WAVE_DUR - n_launch))
        for _ in range(n_launch):
            top  = rng.random() < 0.5
            sign = 1.0 if top else -1.0
            waves.append([0 if top else 1, WAVE_DUR, sign * d_B.copy(), False])

        pert = np.zeros(n_act, bool)
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
        for i in range(n_act):
            if len(NB[i]):
                dF[i] = kappa * (F[NB[i]].mean(0) - F[i])
        F += dF

        ttl -= 1.0
        ttl[pert] -= 1.0
        dead = np.where(ttl <= 0)[0]
        for i in dead:
            nbs = NB[i]
            j   = nbs[rng.integers(len(nbs))] if len(nbs) else i
            h[i]      = (1 - SEED_BETA) * rng.normal(0, 0.1, HS) + SEED_BETA * F[j]
            m[i]      = 0
            streak[i] = 0
            ttl[i]    = float(rng.geometric(p=nu))

        waves = [[w[0], w[1]-1, w[2], w[3]] for w in waves if w[1]-1 > 0]

        if step in cps_set:
            result[f"sg4_{step}"] = sg4_fn(F, zone_id)

    return result


# ── Key factories ──────────────────────────────────────────────────────────────
def key_a(kappa, zone_w, fa, seed):
    return f"p30a,{kappa:.8g},{zone_w},{fa:.8g},{seed}"

def key_b(nu, fa, seed):
    return f"p30b,{nu:.8g},{fa:.8g},{seed}"


# ── Worker ─────────────────────────────────────────────────────────────────────
def _worker(args):
    tag, params, seed = args
    if tag == "a":
        kappa, zone_w, fa = params
        return run(seed, kappa=kappa, zone_w=zone_w, nu=NU_STD, fa=fa)
    else:  # "b"
        nu, fa = params
        return run(seed, kappa=KAPPA_STD, zone_w=ZONE_W_STD, nu=nu, fa=fa)


RESULTS_FILE = os.path.join(os.path.dirname(__file__),
                            "results", "paper30_results.json")


# ── Saturation law fit helper ──────────────────────────────────────────────────
def fit_keff(fa_vals, sg4_means):
    """Fit sg4 = C * FA / (FA + K) and return (C, K_eff) or None."""
    fa_arr  = np.array(fa_vals)
    sg4_arr = np.array(sg4_means)
    valid   = ~np.isnan(sg4_arr) & (sg4_arr > 0)
    if valid.sum() < 3:
        return None, None

    def sat_law(fa, C, K):
        return C * fa / (fa + K)

    try:
        popt, _ = curve_fit(sat_law, fa_arr[valid], sg4_arr[valid],
                            p0=[sg4_arr[valid].max() * 2, 0.05],
                            bounds=([0, 1e-6], [1e5, 10]),
                            maxfev=5000)
        return float(popt[0]), float(popt[1])
    except Exception:
        return None, None


if __name__ == "__main__":
    mp.freeze_support()

    # Build all conditions
    all_conditions = []
    for kappa in KAPPA_VALS:
        for zone_w in ZONE_W_VALS:
            for fa in FA_VALS:
                for seed in range(N_SEEDS):
                    all_conditions.append(
                        ("a", (kappa, zone_w, fa), seed,
                         key_a(kappa, zone_w, fa, seed)))
    for nu in NU_VALS_B:
        for fa in FA_VALS:
            for seed in range(N_SEEDS):
                all_conditions.append(
                    ("b", (nu, fa), seed,
                     key_b(nu, fa, seed)))

    # Load cache
    if os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE) as f:
            results = json.load(f)
        print(f"Loaded {len(results)} cached results.")
    else:
        results = {}

    todo      = [(tag, params, seed)
                 for tag, params, seed, k in all_conditions
                 if k not in results]
    todo_keys = [k for tag, params, seed, k in all_conditions
                 if k not in results]

    if todo:
        print(f"Running {len(todo)} / {len(all_conditions)} simulations (Paper 30)...")
        with mp.Pool(processes=min(len(todo), mp.cpu_count())) as pool:
            raw = pool.map(_worker, todo)
        for k, res in zip(todo_keys, raw):
            results[k] = res
        os.makedirs(os.path.dirname(RESULTS_FILE), exist_ok=True)
        with open(RESULTS_FILE, "w") as f:
            json.dump(results, f)
        print("Saved.")
    else:
        print("All results cached.")

    # ── Analysis ───────────────────────────────────────────────────────────────
    T_LAST = CPS[-1]   # 4000

    def get_sg4_a(kappa, zone_w, fa):
        vals = [results.get(key_a(kappa, zone_w, fa, s), {}).get(f"sg4_{T_LAST}",
                            float("nan"))
                for s in range(N_SEEDS)]
        return [v for v in vals if not math.isnan(v)]

    def get_sg4_b(nu, fa):
        vals = [results.get(key_b(nu, fa, s), {}).get(f"sg4_{T_LAST}",
                            float("nan"))
                for s in range(N_SEEDS)]
        return [v for v in vals if not math.isnan(v)]

    # Exp A: K_eff for each (kappa, zone_w) condition
    print()
    print("=" * 72)
    print("EXP A: K_eff(kappa, ZONE_W) -- testing Gamma ~ ZONE_W / sqrt(kappa)")
    print("=" * 72)
    print(f"  {'kappa':>8} {'ZONE_W':>8} | {'K_eff':>8} {'Gamma':>8} | "
          f"{'pred_Gamma':>10}")
    print("  " + "-" * 55)

    keff_grid = {}
    # A normalisation constant from Paper 29 standard condition
    # kappa=0.020, zone_w=5: K_eff=0.037, Gamma=3.17
    # Predict: Gamma = A * ZONE_W / sqrt(kappa)
    # A = Gamma * sqrt(kappa) / ZONE_W = 3.17 * sqrt(0.020) / 5 = 0.0897
    A_PRED = 3.17 * math.sqrt(0.020) / 5.0

    for kappa in KAPPA_VALS:
        for zone_w in ZONE_W_VALS:
            sg4_means_a = [float(np.mean(get_sg4_a(kappa, zone_w, fa))) if get_sg4_a(kappa, zone_w, fa)
                           else float("nan")
                           for fa in FA_VALS]
            C_fit, K_fit = fit_keff(FA_VALS, sg4_means_a)
            gamma    = K_EFF_I / K_fit if K_fit else float("nan")
            pred_g   = A_PRED * zone_w / math.sqrt(kappa)
            keff_grid[(kappa, zone_w)] = K_fit
            print(f"  {kappa:>8.4f} {zone_w:>8d} | {(K_fit or float('nan')):>8.4f} "
                  f"{gamma:>8.2f} | {pred_g:>10.2f}")

    # Log-log slopes: K_eff vs kappa (at zone_w=5)
    print()
    print("  Log-log slope K_eff vs kappa (ZONE_W=5):")
    kv, keffv = [], []
    for kappa in KAPPA_VALS:
        k = keff_grid.get((kappa, ZONE_W_STD))
        if k and k > 0:
            kv.append(math.log(kappa)); keffv.append(math.log(k))
    if len(kv) >= 2:
        sl, *_ = linregress(kv, keffv)
        print(f"    slope = {sl:.3f}  (prediction: +0.5 if Gamma ~ 1/sqrt(kappa))")

    # Log-log slopes: K_eff vs zone_w (at kappa=0.020)
    print("  Log-log slope K_eff vs ZONE_W (kappa=0.020):")
    zv, keffv2 = [], []
    for zone_w in ZONE_W_VALS:
        k = keff_grid.get((KAPPA_STD, zone_w))
        if k and k > 0:
            zv.append(math.log(zone_w)); keffv2.append(math.log(k))
    if len(zv) >= 2:
        sl2, *_ = linregress(zv, keffv2)
        print(f"    slope = {sl2:.3f}  (prediction: -1 if Gamma ~ ZONE_W)")

    # Exp B: K_eff vs nu
    print()
    print("=" * 72)
    print("EXP B: K_eff vs nu (kappa=0.020, ZONE_W=5)")
    print("=" * 72)
    print(f"  {'nu':>10} | {'K_eff':>8} {'Gamma':>8}")
    print("  " + "-" * 32)
    all_nu = [NU_STD] + list(NU_VALS_B)
    for nu in sorted(all_nu):
        if nu == NU_STD:
            # Use Exp A standard condition
            sg4_means_b = [float(np.mean(get_sg4_a(KAPPA_STD, ZONE_W_STD, fa)))
                           if get_sg4_a(KAPPA_STD, ZONE_W_STD, fa)
                           else float("nan")
                           for fa in FA_VALS]
        else:
            sg4_means_b = [float(np.mean(get_sg4_b(nu, fa))) if get_sg4_b(nu, fa)
                           else float("nan")
                           for fa in FA_VALS]
        C_fit, K_fit = fit_keff(FA_VALS, sg4_means_b)
        gamma = K_EFF_I / K_fit if K_fit else float("nan")
        print(f"  {nu:>10.4f} | {(K_fit or float('nan')):>8.4f} {gamma:>8.2f}")
