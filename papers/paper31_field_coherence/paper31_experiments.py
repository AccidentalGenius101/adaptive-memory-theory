"""
paper31_experiments.py -- The Field Coherence Length

Paper 30 found K_eff ~ (sqrt(kappa)/ZONE_W)^0.6, suggesting a finite field
coherence length ell mediates the spatial amplification factor Gamma. The
copy-forward loop propagates field information locally; ell is how far that
information travels before being overwritten by noise (new births, diffusion).

Prediction: ell ~ sqrt(kappa / nu)
  - ell vs kappa: log-log slope +0.5
  - ell vs nu:    log-log slope -0.5

Exp A (kappa sweep, 25 runs):
  kappa in {0.001, 0.005, 0.020, 0.080, 0.200}
  nu=0.001, ZONE_W=5, FA=0.200 (high FA for strong, stable field)
  5 seeds, T_END=4000, CPS=[1000, 2000, 3000, 4000]
  Store: sg4_{t}, wz_autocorr_{t} (within-zone autocorr, length ZONE_W=5)
  Fit: ell(kappa) at T=4000; test log-log slope.

Exp B (nu sweep, 25 runs):
  nu in {0.0003, 0.001, 0.003, 0.010, 0.030}
  kappa=0.020, ZONE_W=5, FA=0.200
  5 seeds, T_END=4000, CPS=[1000, 2000, 3000, 4000]
  nu=0.001 runs cached separately from Exp A (5 extra simulations).
  Fit: ell(nu) at T=4000; test log-log slope.

Exp C (nu=0.003 failure mode temporal dynamics, 50 runs):
  nu in {0.001, 0.003}
  FA in {0.005, 0.020, 0.080, 0.200, 0.800}
  kappa=0.020, ZONE_W=5
  5 seeds, T_END=6000, CPS=range(500, 6001, 500) -- 12 checkpoints
  Q18: does sg4 fail to form, or form then collapse at nu=0.003?

Total: ~100 runs.
"""
import numpy as np, json, os, multiprocessing as mp, math
from scipy.optimize import curve_fit
from scipy.stats import linregress

# ── Fixed parameters ───────────────────────────────────────────────────────────
H         = 20
N_ZONES   = 4
HALF      = 20
ZONE_W    = HALF // N_ZONES   # 5
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
KAPPA_STD = 0.020
NU_STD    = 0.001
FA_HIGH   = 0.200    # used in Exps A & B for strong signal

# ── Experiment grids ───────────────────────────────────────────────────────────
KAPPA_VALS_A = [0.001, 0.005, 0.020, 0.080, 0.200]
NU_VALS_B    = [0.0003, 0.001, 0.003, 0.010, 0.030]
NU_VALS_C    = [0.001, 0.003]
FA_VALS_C    = [0.005, 0.020, 0.080, 0.200, 0.800]
N_SEEDS      = 5
T_END_AB     = 4000
T_END_C      = 6000
CPS_AB       = [1000, 2000, 3000, 4000]
CPS_C        = list(range(500, T_END_C + 1, 500))   # 12 checkpoints

d_B = np.array([0.0, 1.0])

# ── Geometry (fixed ZONE_W=5) ──────────────────────────────────────────────────
N_ACT    = HALF * H
_col     = np.arange(N_ACT) % HALF
_row     = np.arange(N_ACT) // HALF
zone_id  = _col // ZONE_W
top_mask = _row < H // 2
bot_mask = _row >= H // 2

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


def within_zone_autocorr_fn(F, hs_idx=0):
    """Within-zone column spatial autocorrelation, averaged over zones.
    Returns list of length ZONE_W (lags 0..ZONE_W-1).
    Each zone's column profile is demeaned before computing correlation."""
    F_col = F.reshape(H, HALF, HS).mean(axis=0)  # (HALF, HS)
    corr_sum = np.zeros(ZONE_W)
    count    = 0
    for z in range(N_ZONES):
        f = F_col[z * ZONE_W:(z + 1) * ZONE_W, hs_idx].copy()
        f -= f.mean()
        c = np.correlate(f, f, mode='full')
        c = c[ZONE_W - 1:]    # lags 0..ZONE_W-1
        if c[0] > 1e-10:
            corr_sum += c / c[0]
            count += 1
    if count == 0:
        return [1.0] + [0.0] * (ZONE_W - 1)
    return (corr_sum / count).tolist()


# ── Coherence-length fit ────────────────────────────────────────────────────────
def fit_corr_length(corr_list):
    """Fit C(lag) = exp(-lag / ell) to within-zone autocorr. Return ell."""
    c    = np.array(corr_list, dtype=float)
    lags = np.arange(len(c), dtype=float)
    # Use lags 1..ZONE_W-1 (lag 0 = 1 by construction)
    valid = (lags >= 1) & (c > 0.0)
    if valid.sum() < 2:
        return float("nan")
    try:
        popt, _ = curve_fit(lambda x, l: np.exp(-x / l), lags[valid], c[valid],
                            p0=[1.5], bounds=([0.05], [100.0]),
                            maxfev=2000)
        return float(popt[0])
    except Exception:
        return float("nan")


# ── Core simulation ────────────────────────────────────────────────────────────
def run(seed, kappa=KAPPA_STD, nu=NU_STD, fa=FA_HIGH,
        wr=WR_STD, t_end=T_END_AB, cps=None, store_autocorr=True):
    if cps is None:
        cps = CPS_AB
    cps_set = set(cps)

    rng    = np.random.default_rng(seed)
    h      = rng.normal(0, 0.1, (N_ACT, HS))
    F      = np.zeros((N_ACT, HS))
    m      = np.zeros((N_ACT, HS))
    base   = h.copy()
    streak = np.zeros(N_ACT, int)
    ttl    = rng.geometric(p=max(nu, 1e-6), size=N_ACT).astype(float)
    waves  = []
    result = {}

    for step in range(1, t_end + 1):

        n_launch = int(wr / WAVE_DUR)
        n_launch += int(rng.random() < (wr / WAVE_DUR - n_launch))
        for _ in range(n_launch):
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
            j   = nbs[rng.integers(len(nbs))] if len(nbs) else i
            h[i]      = (1 - SEED_BETA) * rng.normal(0, 0.1, HS) + SEED_BETA * F[j]
            m[i]      = 0
            streak[i] = 0
            ttl[i]    = float(rng.geometric(p=nu))

        waves = [[w[0], w[1]-1, w[2], w[3]] for w in waves if w[1]-1 > 0]

        if step in cps_set:
            result[f"sg4_{step}"] = sg4_fn(F)
            if store_autocorr:
                result[f"wz_autocorr_{step}"] = within_zone_autocorr_fn(F)

    return result


# ── Key factories ──────────────────────────────────────────────────────────────
def key_a(kappa, seed):
    return f"p31a,{kappa:.8g},{seed}"

def key_b(nu, seed):
    return f"p31b,{nu:.8g},{seed}"

def key_c(nu, fa, seed):
    return f"p31c,{nu:.8g},{fa:.8g},{seed}"


# ── Worker ─────────────────────────────────────────────────────────────────────
def _worker(args):
    tag, params, seed = args
    if tag == "a":
        kappa, = params
        return run(seed, kappa=kappa, nu=NU_STD, fa=FA_HIGH,
                   t_end=T_END_AB, cps=CPS_AB, store_autocorr=True)
    elif tag == "b":
        nu, = params
        return run(seed, kappa=KAPPA_STD, nu=nu, fa=FA_HIGH,
                   t_end=T_END_AB, cps=CPS_AB, store_autocorr=True)
    else:  # "c"
        nu, fa = params
        return run(seed, kappa=KAPPA_STD, nu=nu, fa=fa,
                   t_end=T_END_C, cps=CPS_C, store_autocorr=False)


RESULTS_FILE = os.path.join(os.path.dirname(__file__),
                            "results", "paper31_results.json")


if __name__ == "__main__":
    mp.freeze_support()

    # Build all conditions
    all_conditions = []
    for kappa in KAPPA_VALS_A:
        for seed in range(N_SEEDS):
            all_conditions.append(("a", (kappa,), seed, key_a(kappa, seed)))
    for nu in NU_VALS_B:
        for seed in range(N_SEEDS):
            all_conditions.append(("b", (nu,), seed, key_b(nu, seed)))
    for nu in NU_VALS_C:
        for fa in FA_VALS_C:
            for seed in range(N_SEEDS):
                all_conditions.append(("c", (nu, fa), seed, key_c(nu, fa, seed)))

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
        print(f"Running {len(todo)} / {len(all_conditions)} simulations (Paper 31)...")
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
    T_LAST_AB = CPS_AB[-1]   # 4000
    T_LAST_C  = CPS_C[-1]    # 6000

    def get_autocorr_a(kappa, t=T_LAST_AB):
        vals = [results.get(key_a(kappa, s), {}).get(f"wz_autocorr_{t}")
                for s in range(N_SEEDS)]
        return [v for v in vals if v is not None]

    def get_autocorr_b(nu, t=T_LAST_AB):
        vals = [results.get(key_b(nu, s), {}).get(f"wz_autocorr_{t}")
                for s in range(N_SEEDS)]
        return [v for v in vals if v is not None]

    def mean_corr(corr_list_of_lists):
        if not corr_list_of_lists:
            return None
        arr = np.array(corr_list_of_lists)
        return arr.mean(axis=0)

    def get_ell_a(kappa):
        corrs = get_autocorr_a(kappa)
        mc = mean_corr(corrs)
        return fit_corr_length(mc) if mc is not None else float("nan")

    def get_ell_b(nu):
        corrs = get_autocorr_b(nu)
        mc = mean_corr(corrs)
        return fit_corr_length(mc) if mc is not None else float("nan")

    def get_sg4_c(nu, fa, t=T_LAST_C):
        vals = [results.get(key_c(nu, fa, s), {}).get(f"sg4_{t}", float("nan"))
                for s in range(N_SEEDS)]
        return [v for v in vals if not math.isnan(v)]

    # Exp A: ell vs kappa
    print()
    print("=" * 60)
    print("EXP A: Coherence length ell vs kappa (nu=0.001, ZONE_W=5)")
    print("=" * 60)
    print(f"  {'kappa':>8} | {'ell':>8} | {'C[1]':>8} {'C[2]':>8}")
    print("  " + "-" * 40)
    kv, ellv = [], []
    for kappa in KAPPA_VALS_A:
        ell  = get_ell_a(kappa)
        corrs = get_autocorr_a(kappa)
        mc = mean_corr(corrs)
        c1 = mc[1] if mc is not None and len(mc) > 1 else float("nan")
        c2 = mc[2] if mc is not None and len(mc) > 2 else float("nan")
        print(f"  {kappa:>8.4f} | {ell:>8.4f} | {c1:>8.4f} {c2:>8.4f}")
        if not math.isnan(ell) and ell > 0:
            kv.append(math.log(kappa)); ellv.append(math.log(ell))

    if len(kv) >= 2:
        sl, *_ = linregress(kv, ellv)
        print(f"\n  Log-log slope ell vs kappa: {sl:.3f}  (prediction: +0.5)")

    # Exp B: ell vs nu
    print()
    print("=" * 60)
    print("EXP B: Coherence length ell vs nu (kappa=0.020, ZONE_W=5)")
    print("=" * 60)
    print(f"  {'nu':>8} | {'ell':>8} | {'C[1]':>8} {'C[2]':>8}")
    print("  " + "-" * 40)
    nuv, ellv2 = [], []
    for nu in NU_VALS_B:
        ell  = get_ell_b(nu)
        corrs = get_autocorr_b(nu)
        mc = mean_corr(corrs)
        c1 = mc[1] if mc is not None and len(mc) > 1 else float("nan")
        c2 = mc[2] if mc is not None and len(mc) > 2 else float("nan")
        print(f"  {nu:>8.4f} | {ell:>8.4f} | {c1:>8.4f} {c2:>8.4f}")
        if not math.isnan(ell) and ell > 0:
            nuv.append(math.log(nu)); ellv2.append(math.log(ell))

    if len(nuv) >= 2:
        sl2, *_ = linregress(nuv, ellv2)
        print(f"\n  Log-log slope ell vs nu:    {sl2:.3f}  (prediction: -0.5)")

    # Exp C: sg4(t) at nu=0.001 vs nu=0.003
    print()
    print("=" * 60)
    print("EXP C: sg4(t) -- nu=0.001 vs nu=0.003 (FA=0.200)")
    print("=" * 60)
    print(f"  {'t':>6} | {'nu=0.001':>10} {'nu=0.003':>10}")
    print("  " + "-" * 32)
    for t in CPS_C:
        v1 = get_sg4_c(0.001, 0.200, t)
        v3 = get_sg4_c(0.003, 0.200, t)
        m1 = float(np.mean(v1)) if v1 else float("nan")
        m3 = float(np.mean(v3)) if v3 else float("nan")
        print(f"  {t:>6} | {m1:>10.3f} {m3:>10.3f}")
