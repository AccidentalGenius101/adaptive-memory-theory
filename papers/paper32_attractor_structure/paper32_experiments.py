"""
paper32_experiments.py -- The Zone Attractor Structure

Paper 31 showed that VCML zones are spatially uniform flat attractors (no within-zone
gradient). Paper 32 asks: what IS the attractor mathematically?
  - Fixed point (zone means lock to specific values)?
  - Noisy fixed point (means fluctuate around a stable centre)?
  - Multi-state attractor (zones occasionally swap identities -- Kramers escape)?

Also answers Paper 31's open questions:
  Q20: Extended ZONE_W sweep to refine the vote-count exponent
  Q21: Does stronger FA rescue the nu=0.003 low-amplitude saturation?

Exp A (attractor characterisation, 20 runs):
  Standard condition: kappa=0.020, nu=0.001, FA=0.200, ZONE_W=5
  20 seeds, T_END=8000
  CPS = range(3000, 8001, 200) -- 26 late checkpoints once system is settled
  Store: sg4_{t}, zmeans_{t} (4 zones x 2 HS each)
  Measures:
    - CV(sg4) over last 26 checkpoints per seed (attractor noise)
    - Zone-swap events: sudden drops in sg4 (Kramers escape)
    - Decorrelation time tau: autocorr of sg4(t) series per seed

Exp B (FA recovery at nu=0.003, Q21, 25 runs):
  nu=0.003, FA in {0.050, 0.200, 0.800, 2.000, 5.000}
  kappa=0.020, ZONE_W=5, 5 seeds, T_END=6000
  CPS = range(500, 6001, 500) -- 12 checkpoints
  Q: does stronger FA compensate for high-turnover disruption?
  Test: does sg4(T=6000) recover to Paper 29 standard (~80-100)?

Exp C (extended ZONE_W sweep, Q20, 90 runs):
  ZONE_W in {2, 3, 5, 7, 10, 15}  (H=20 fixed, N_ZONES=4)
  FA in {0.005, 0.020, 0.080, 0.200, 0.800}
  kappa=0.020, nu=0.001, 3 seeds, T_END=4000
  Fit saturation law -> K_eff(ZONE_W); log-log slope vs -1 prediction (Paper 30: -0.31)

Total: 20 + 25 + 90 = 135 runs.
"""
import numpy as np, json, os, multiprocessing as mp, math
from scipy.optimize import curve_fit
from scipy.stats import linregress

# ── Fixed parameters ───────────────────────────────────────────────────────────
H         = 20
N_ZONES   = 4
HALF_STD  = 20
ZONE_W_STD= 5
BASE_BETA = 0.005
ALPHA_MID = 0.15
MID_DECAY = 0.99
FD        = 0.9997
SS        = 10
WAVE_DUR  = 15
HS        = 2
SEED_BETA = 0.25
WR_STD    = 2.4
KAPPA_STD = 0.020
NU_STD    = 0.001
FA_STD    = 0.200

# ── Experiment grids ───────────────────────────────────────────────────────────
N_SEEDS_A  = 20
N_SEEDS_B  = 5
N_SEEDS_C  = 3
T_END_A    = 8000
T_END_BC   = 6000
T_END_C    = 4000
CPS_A      = list(range(3000, T_END_A + 1, 200))   # 26 late checkpoints
CPS_B      = list(range(500,  T_END_BC + 1, 500))   # 12 checkpoints
CPS_C      = [1000, 2000, 3000, 4000]

FA_VALS_B  = [0.050, 0.200, 0.800, 2.000, 5.000]
ZONE_W_VALS_C = [2, 3, 5, 7, 10, 15]
FA_VALS_C  = [0.005, 0.020, 0.080, 0.200, 0.800]

d_B = np.array([0.0, 1.0])


# ── Dynamic geometry builder (reused from Paper 30) ────────────────────────────
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


# ── Measurements ───────────────────────────────────────────────────────────────
def sg4_fn(F, zone_id, n_zones=N_ZONES):
    means = [F[zone_id == z].mean(0) for z in range(n_zones)]
    pairs = [(i, j) for i in range(n_zones) for j in range(i + 1, n_zones)]
    return float(np.mean([np.linalg.norm(means[i] - means[j]) for i, j in pairs]))


def zmeans_fn(F, zone_id, n_zones=N_ZONES):
    return [F[zone_id == z].mean(0).tolist() for z in range(n_zones)]


# ── Saturation law fit helper ──────────────────────────────────────────────────
def fit_keff(fa_vals, sg4_means):
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


# ── Core simulation ────────────────────────────────────────────────────────────
def run(seed, kappa=KAPPA_STD, zone_w=ZONE_W_STD, nu=NU_STD, fa=FA_STD,
        wr=WR_STD, t_end=T_END_A, cps=None, store_zmeans=True):
    if cps is None:
        cps = CPS_A
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
            if store_zmeans:
                result[f"zmeans_{step}"] = zmeans_fn(F, zone_id)

    return result


# ── Key factories ──────────────────────────────────────────────────────────────
def key_a(seed):
    return f"p32a,{seed}"

def key_b(fa, seed):
    return f"p32b,{fa:.8g},{seed}"

def key_c(zone_w, fa, seed):
    return f"p32c,{zone_w},{fa:.8g},{seed}"


# ── Worker ─────────────────────────────────────────────────────────────────────
def _worker(args):
    tag, params, seed = args
    if tag == "a":
        return run(seed, kappa=KAPPA_STD, zone_w=ZONE_W_STD, nu=NU_STD,
                   fa=FA_STD, t_end=T_END_A, cps=CPS_A, store_zmeans=True)
    elif tag == "b":
        fa, = params
        return run(seed, kappa=KAPPA_STD, zone_w=ZONE_W_STD, nu=0.003,
                   fa=fa, t_end=T_END_BC, cps=CPS_B, store_zmeans=False)
    else:  # "c"
        zone_w, fa = params
        return run(seed, kappa=KAPPA_STD, zone_w=zone_w, nu=NU_STD,
                   fa=fa, t_end=T_END_C, cps=CPS_C, store_zmeans=False)


RESULTS_FILE = os.path.join(os.path.dirname(__file__),
                            "results", "paper32_results.json")


if __name__ == "__main__":
    mp.freeze_support()

    all_conditions = []
    for seed in range(N_SEEDS_A):
        all_conditions.append(("a", (), seed, key_a(seed)))
    for fa in FA_VALS_B:
        for seed in range(N_SEEDS_B):
            all_conditions.append(("b", (fa,), seed, key_b(fa, seed)))
    for zone_w in ZONE_W_VALS_C:
        for fa in FA_VALS_C:
            for seed in range(N_SEEDS_C):
                all_conditions.append(("c", (zone_w, fa), seed, key_c(zone_w, fa, seed)))

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
        print(f"Running {len(todo)} / {len(all_conditions)} simulations (Paper 32)...")
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

    # Exp A: attractor characterisation
    print()
    print("=" * 72)
    print("EXP A: Attractor characterisation (standard condition, 20 seeds)")
    print("=" * 72)
    sg4_all = []
    cv_all  = []
    swap_events = 0
    for s in range(N_SEEDS_A):
        sg4_series = [results.get(key_a(s), {}).get(f"sg4_{t}", float("nan"))
                      for t in CPS_A]
        sg4_series = [v for v in sg4_series if not math.isnan(v)]
        if not sg4_series:
            continue
        mu  = float(np.mean(sg4_series))
        std = float(np.std(sg4_series))
        cv  = std / mu if mu > 0 else float("nan")
        sg4_all.append(mu)
        cv_all.append(cv)
        # Zone-swap detection: drop > 50% of mean in a single step
        for i in range(1, len(sg4_series)):
            if sg4_series[i] < 0.5 * mu and sg4_series[i-1] >= 0.5 * mu:
                swap_events += 1

    if sg4_all:
        print(f"  sg4 at steady state (T=3000-8000):")
        print(f"    Mean across seeds: {np.mean(sg4_all):.3f}")
        print(f"    Std  across seeds: {np.std(sg4_all):.3f}")
        print(f"    Mean CV (within-seed): {np.mean(cv_all):.4f}")
        print(f"    Max  CV (within-seed): {np.max(cv_all):.4f}")
        print(f"  Zone-swap events (sg4 drops >50% of mean): {swap_events}")
        cv_mean = float(np.mean(cv_all))
        if cv_mean < 0.05:
            verdict = "TIGHT FIXED POINT (CV < 5%)"
        elif cv_mean < 0.15:
            verdict = "NOISY FIXED POINT (CV 5-15%)"
        else:
            verdict = "DIFFUSE ATTRACTOR (CV > 15%)"
        print(f"  Attractor verdict: {verdict}")

    # Exp B: FA recovery at nu=0.003
    T_LAST_B = CPS_B[-1]
    print()
    print("=" * 72)
    print("EXP B: FA recovery at nu=0.003 (kappa=0.020, ZONE_W=5)")
    print("=" * 72)
    print(f"  {'FA':>8} | {'sg4_mean':>10} {'sg4_std':>10}")
    print("  " + "-" * 36)
    for fa in FA_VALS_B:
        vals = [results.get(key_b(fa, s), {}).get(f"sg4_{T_LAST_B}", float("nan"))
                for s in range(N_SEEDS_B)]
        vals = [v for v in vals if not math.isnan(v)]
        if vals:
            mu  = float(np.mean(vals))
            std = float(np.std(vals)) if len(vals) > 1 else 0.0
            print(f"  {fa:>8.3f} | {mu:>10.3f} {std:>10.3f}")

    # Exp C: K_eff vs ZONE_W (extended sweep)
    T_LAST_C = CPS_C[-1]
    print()
    print("=" * 72)
    print("EXP C: K_eff vs ZONE_W (extended, kappa=0.020, nu=0.001)")
    print("=" * 72)
    print(f"  {'ZONE_W':>8} | {'K_eff':>8} {'Gamma':>8} | log(ZW) log(K)")
    print("  " + "-" * 50)
    zw_vals, keff_vals = [], []
    K_EFF_I = 0.117
    for zone_w in ZONE_W_VALS_C:
        sg4_means = []
        for fa in FA_VALS_C:
            vs = [results.get(key_c(zone_w, fa, s), {}).get(f"sg4_{T_LAST_C}", float("nan"))
                  for s in range(N_SEEDS_C)]
            vs = [v for v in vs if not math.isnan(v)]
            sg4_means.append(float(np.mean(vs)) if vs else float("nan"))
        _, K = fit_keff(FA_VALS_C, sg4_means)
        gamma = K_EFF_I / K if K else float("nan")
        log_zw = math.log(zone_w) if zone_w > 0 else float("nan")
        log_k  = math.log(K) if K and K > 0 else float("nan")
        print(f"  {zone_w:>8} | {(K or float('nan')):>8.4f} {gamma:>8.2f} | "
              f"{log_zw:>6.3f} {log_k:>7.3f}")
        if K and K > 0:
            zw_vals.append(log_zw); keff_vals.append(log_k)

    if len(zw_vals) >= 2:
        sl, intercept, r, *_ = linregress(zw_vals, keff_vals)
        print(f"\n  Log-log slope K_eff vs ZONE_W: {sl:.3f}  "
              f"(Paper 30: -0.314; prediction: -1.0)  R2={r**2:.3f}")
