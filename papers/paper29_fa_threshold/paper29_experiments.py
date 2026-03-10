"""
paper29_experiments.py -- The FA Threshold for Zone Differentiation

Paper 28 showed that FA (consolidation rate) is the primary driver of zone
differentiation -- not the kappa/FA ratio predicted by Allen-Cahn.
Paper 29 asks: what is the FA threshold FA* below which zones fail to
differentiate? Is the transition smooth (like Paper 16's saturation law)
or sharp (a genuine bifurcation in the spatial regime)?

Exp A (FA sweep, 50 runs): Dense FA sweep across ~4 decades
  FA in {0.001, 0.002, 0.005, 0.010, 0.020, 0.050, 0.100, 0.200, 0.400, 0.800}
  kappa=0.020, WR=2.4, T_END=6000, N_SEEDS=5
  CPS: range(500, 6001, 500) -- 12 checkpoints
  Store: sg4_{t}, zmeans_{t} at each checkpoint
  Tests: functional form of sg4(FA); smooth crossover vs sharp threshold;
         does the Paper 16 saturation law hold in the Layer III spatial regime?

Total: 50 runs.
"""
import numpy as np, json, os, multiprocessing as mp, math
from scipy.optimize import curve_fit
from scipy.stats import linregress

# ── Fixed parameters (identical to Papers 22-28) ──────────────────────────────
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
WR_STD    = 2.4
ZONE_W    = HALF // N_ZONES   # 5

# ── Experiment grid ────────────────────────────────────────────────────────────
FA_VALS  = [0.001, 0.002, 0.005, 0.010, 0.020, 0.050, 0.100, 0.200, 0.400, 0.800]
N_SEEDS  = 5
T_END    = 6000
CPS      = list(range(500, T_END + 1, 500))   # 12 checkpoints: 500..6000

# ── Geometry ───────────────────────────────────────────────────────────────────
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

# ── Measurements ───────────────────────────────────────────────────────────────
def sg4_fn(F):
    means = [F[zone_id == z].mean(0) for z in range(N_ZONES)]
    pairs = [(i, j) for i in range(N_ZONES) for j in range(i + 1, N_ZONES)]
    return float(np.mean([np.linalg.norm(means[i] - means[j]) for i, j in pairs]))


def zone_means_fn(F):
    return [F[zone_id == z].mean(0).tolist() for z in range(N_ZONES)]


# ── Core simulation ────────────────────────────────────────────────────────────
def run(seed, fa, wr=WR_STD, t_end=T_END, cps=None):
    if cps is None:
        cps = CPS
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
            j   = nbs[rng.integers(len(nbs))] if len(nbs) else i
            h[i]      = (1 - SEED_BETA) * rng.normal(0, 0.1, HS) + SEED_BETA * F[j]
            m[i]      = 0
            streak[i] = 0
            ttl[i]    = float(rng.geometric(p=NU))

        waves = [[w[0], w[1]-1, w[2], w[3]] for w in waves if w[1]-1 > 0]

        if step in cps_set:
            result[f"sg4_{step}"]    = sg4_fn(F)
            result[f"zmeans_{step}"] = zone_means_fn(F)

    return result


# ── Keys ───────────────────────────────────────────────────────────────────────
def key(fa, seed):
    return f"p29,{fa:.8g},{seed}"


def _worker(args):
    fa, seed = args
    return run(seed, fa=fa)


RESULTS_FILE = os.path.join(os.path.dirname(__file__),
                            "results", "paper29_results.json")


if __name__ == "__main__":
    mp.freeze_support()

    all_conditions = [(fa, seed)
                      for fa in FA_VALS
                      for seed in range(N_SEEDS)]

    if os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE) as f:
            results = json.load(f)
        print(f"Loaded {len(results)} cached results.")
    else:
        results = {}

    todo      = [(fa, seed) for fa, seed in all_conditions
                 if key(fa, seed) not in results]
    todo_keys = [key(fa, seed) for fa, seed in all_conditions
                 if key(fa, seed) not in results]

    if todo:
        print(f"Running {len(todo)} / {len(all_conditions)} simulations (Paper 29)...")
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
    T_LAST = CPS[-1]   # 6000

    def get_sg4(fa, t=T_LAST):
        vals = [results.get(key(fa, s), {}).get(f"sg4_{t}", float("nan"))
                for s in range(N_SEEDS)]
        return [v for v in vals if not math.isnan(v)]

    print()
    print("=" * 72)
    print("EXP A: sg4 vs FA at T=6000")
    print("=" * 72)
    print(f"  {'FA':>8} | {'sg4_mean':>10} {'sg4_std':>10} | {'CV':>8}")
    print("  " + "-" * 48)

    fa_arr, sg4_arr, sg4_std_arr = [], [], []
    for fa in FA_VALS:
        vals = get_sg4(fa)
        if not vals:
            continue
        mu  = float(np.mean(vals))
        std = float(np.std(vals)) if len(vals) > 1 else 0.0
        cv  = std / mu if mu > 1e-10 else float("nan")
        print(f"  {fa:>8.4f} | {mu:>10.3f} {std:>10.3f} | {cv:>8.3f}")
        fa_arr.append(fa)
        sg4_arr.append(mu)
        sg4_std_arr.append(std)

    fa_arr    = np.array(fa_arr)
    sg4_arr   = np.array(sg4_arr)

    # Log-log slope
    valid = (fa_arr > 0) & (sg4_arr > 1e-6)
    if valid.sum() >= 3:
        slope, intercept, r, *_ = linregress(np.log(fa_arr[valid]),
                                             np.log(sg4_arr[valid]))
        print(f"\n  Log-log slope: {slope:.3f}  (linear: 1.0 | saturating: 0)  R2={r**2:.3f}")

    # Fit Paper 16 saturation law: sg4 = C * FA / (FA + K)
    def sat_law(fa, C, K):
        return C * fa / (fa + K)

    try:
        popt, _ = curve_fit(sat_law, fa_arr[valid], sg4_arr[valid],
                            p0=[sg4_arr.max() * 2, 0.05],
                            bounds=([0, 1e-6], [np.inf, 10]),
                            maxfev=5000)
        C_fit, K_fit = popt
        sg4_pred = sat_law(fa_arr[valid], *popt)
        resid = float(np.std(sg4_arr[valid] - sg4_pred))
        print(f"\n  Saturation law fit: sg4 = {C_fit:.2f} * FA / (FA + {K_fit:.4f})")
        print(f"  Residual std: {resid:.3f}")
        print(f"  FA_half-max (K_eff) = {K_fit:.4f}")
        print(f"  Paper 16 predicted K_eff ~ 0.114-0.119")
    except Exception as e:
        print(f"\n  Saturation law fit failed: {e}")

    # Saturation time: first t where sg4 > 90% of T_LAST value
    print()
    print("=" * 72)
    print("Saturation time T_90 (time to reach 90% of T=6000 sg4)")
    print("=" * 72)
    print(f"  {'FA':>8} | {'sg4_final':>10} | {'T_90':>8}")
    print("  " + "-" * 35)
    for fa in FA_VALS:
        final_vals = get_sg4(fa, T_LAST)
        if not final_vals:
            continue
        sg4_final = float(np.mean(final_vals))
        thresh    = 0.90 * sg4_final
        t90 = float("nan")
        for t in CPS:
            sg_t = get_sg4(fa, t)
            if sg_t and float(np.mean(sg_t)) >= thresh:
                t90 = float(t)
                break
        print(f"  {fa:>8.4f} | {sg4_final:>10.2f} | {t90:>8.0f}")
