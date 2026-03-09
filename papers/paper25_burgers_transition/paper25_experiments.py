"""
paper25_experiments.py -- Experiments for Paper 25

Three-Part Completion of the Model H Picture:
  Exp 1 (FREE): Zone-level C_full autocorrelation from Paper 24 data.
  Exp 2 (NEW) : kappa sweep at alpha=0 and alpha=1 -- tracks the
                Burgers ratio xi_late(1)/xi_late(0) as nu/kappa increases.
  Exp 3 (NEW) : Inheritance roughness isolation:
    Exp 3A: SEED_BETA sweep at FA=0.40 -> xi_inf increases with SEED_BETA
    Exp 3B: FA sweep at SEED_BETA=0   -> xi_inf ∝ FA^{-0.5} (predicted slope,
            reversed to +0.21 by inheritance when SEED_BETA=0.25, Paper 23)

Exp 2: alpha in {0.0, 1.0} x kappa in {0.005, 0.008, 0.012, 0.016, 0.020}
       5 seeds -> 50 new runs
Exp 3: SEED_BETA in {0.0, 0.05, 0.10, 0.25, 0.50} x FA=0.40 -> 25 runs
       FA in {0.10, 0.20, 0.40, 0.60, 0.80} x SEED_BETA=0.0  -> 20 runs
                                               (FA=0.40 already covered above)
Total: 50 + 25 + 20 = 95 new runs.
"""
import numpy as np, json, os, multiprocessing as mp, math
from scipy.stats import linregress

# ── Fixed parameters (identical to Paper 24) ─────────────────────────────────
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
SEED_BETA_STD = 0.25
KAPPA_STD = 0.020
FA_STD    = 0.40
NU        = 0.001

T_END   = 3000
CPS_SET = set(range(200, T_END + 1, 200))
CPS     = sorted(CPS_SET)
N_ACT   = HALF * H
ZONE_W  = HALF // N_ZONES   # 5
MAX_LAG = HALF // 2         # 10

# ── Experiment parameter grids ────────────────────────────────────────────────
N_SEEDS      = 5
KAPPA_VALS_B = [0.005, 0.008, 0.012, 0.016, 0.020]   # Exp 2
ALPHA_B      = [0.0, 1.0]                              # Exp 2
SEED_BETA_VALS = [0.0, 0.05, 0.10, 0.25, 0.50]        # Exp 3A
FA_VALS_C    = [0.10, 0.20, 0.40, 0.60, 0.80]         # Exp 3B (SEED_BETA=0)

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
    return float(np.mean([np.linalg.norm(means[i] - means[j])
                          for i, j in pairs]))


def autocorr(F, subtract_zone_mean=True):
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


# ── Core simulation (generalised: kappa, seed_beta, fa, birth_alpha) ─────────
def run(seed, birth_alpha=0.0, kappa=KAPPA_STD,
        seed_beta=SEED_BETA_STD, fa=FA_STD):
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
        F[ok] += fa * (m[ok] - F[ok])     # parameterised FA
        F     *= FD

        dF = np.zeros_like(F)
        for i in range(N_ACT):
            if len(NB[i]):
                dF[i] = kappa * (F[NB[i]].mean(0) - F[i])   # parameterised kappa
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
                    probs = f_mags / total
                    j     = nbs[rng.choice(len(nbs), p=probs)]

            h[i]      = ((1 - seed_beta) * rng.normal(0, 0.1, HS)
                         + seed_beta * F[j])    # parameterised seed_beta
            m[i]      = 0
            streak[i] = 0
            ttl[i]    = float(rng.geometric(p=NU))

        waves = [[w[0], w[1] - 1, w[2], w[3]] for w in waves if w[1] - 1 > 0]

        if step in CPS_SET:
            result[f"sg4_{step}"]       = sg4_fn(F)
            result[f"corr_{step}"]      = autocorr(F, subtract_zone_mean=True)

    return result


# ── Key factories ─────────────────────────────────────────────────────────────
def key_b(alpha, kappa, seed):
    """Exp 2: kappa sweep."""
    return f"p25b,{alpha:.8g},{kappa:.8g},{seed}"


def key_c(seed_beta, fa, seed):
    """Exp 3: seed_beta × FA grid."""
    return f"p25c,{seed_beta:.8g},{fa:.8g},{seed}"


def _worker(args):
    exp, params, seed = args
    if exp == "b":
        alpha, kappa = params
        return run(seed, birth_alpha=alpha, kappa=kappa)
    else:  # exp == "c"
        sb, fa = params
        return run(seed, birth_alpha=0.0, kappa=KAPPA_STD,
                   seed_beta=sb, fa=fa)


RESULTS_FILE = os.path.join(os.path.dirname(__file__),
                            "results", "paper25_results.json")


if __name__ == "__main__":
    mp.freeze_support()

    # Build full condition list
    all_conditions = []
    # Exp 2
    for alpha in ALPHA_B:
        for kappa in KAPPA_VALS_B:
            for seed in range(N_SEEDS):
                all_conditions.append(("b", (alpha, kappa), seed,
                                       key_b(alpha, kappa, seed)))
    # Exp 3A: SEED_BETA sweep at FA=0.40
    for sb in SEED_BETA_VALS:
        for seed in range(N_SEEDS):
            all_conditions.append(("c", (sb, FA_STD), seed,
                                   key_c(sb, FA_STD, seed)))
    # Exp 3B: FA sweep at SEED_BETA=0 (skip FA=0.40 already in 3A)
    for fa in FA_VALS_C:
        if abs(fa - FA_STD) < 1e-9:
            continue   # already covered by Exp 3A
        for seed in range(N_SEEDS):
            all_conditions.append(("c", (0.0, fa), seed,
                                   key_c(0.0, fa, seed)))

    # Load cache
    if os.path.exists(RESULTS_FILE):
        results = json.load(open(RESULTS_FILE))
        print(f"Loaded {len(results)} cached results.")
    else:
        results = {}

    todo = [(exp, params, seed)
            for exp, params, seed, key in all_conditions
            if key not in results]
    todo_keys = [key for exp, params, seed, key in all_conditions
                 if key not in results]

    if todo:
        print(f"Running {len(todo)} / {len(all_conditions)} simulations (Paper 25)...")
        with mp.Pool(processes=min(len(todo), mp.cpu_count())) as pool:
            raw = pool.map(_worker, todo)
        for key, res in zip(todo_keys, raw):
            results[key] = res
        os.makedirs(os.path.dirname(RESULTS_FILE), exist_ok=True)
        json.dump(results, open(RESULTS_FILE, "w"), indent=2)
        print("Saved results.")
    else:
        print("All results cached.")

    # ── Quick analysis printout ───────────────────────────────────────────────
    XI_CAP = 30.0
    ZONE_W_A = 5

    def fit_xi(corr):
        r_vals, log_c = [], []
        for r in range(1, min(ZONE_W_A + 1, MAX_LAG + 1)):
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

    def xi_late(key_fn, *args):
        vals = []
        for seed in range(N_SEEDS):
            k = key_fn(*args, seed)
            if k not in results:
                continue
            for t in CPS:
                if t < 2400:
                    continue
                corr = results[k].get(f"corr_{t}")
                if corr:
                    xi = fit_xi(corr)
                    if not math.isnan(xi) and xi <= XI_CAP:
                        vals.append(xi)
        return float(np.mean(vals)) if vals else float("nan")

    print()
    print("=" * 65)
    print("EXP 2: Burgers ratio xi_late(alpha=1) / xi_late(alpha=0)")
    print("=" * 65)
    print(f"  {'kappa':>8} | {'nu/kappa':>9} | {'xi(a=0)':>9} | "
          f"{'xi(a=1)':>9} | {'ratio':>7}")
    print("  " + "-" * 55)
    for kappa in KAPPA_VALS_B:
        xi0 = xi_late(key_b, 0.0, kappa)
        xi1 = xi_late(key_b, 1.0, kappa)
        ratio = xi1 / xi0 if not (math.isnan(xi0) or math.isnan(xi1) or xi0 == 0) \
                else float("nan")
        print(f"  {kappa:>8.4f} | {NU/kappa:>9.3f} | {xi0:>9.2f} | "
              f"{xi1:>9.2f} | {ratio:>7.3f}")

    print()
    print("=" * 65)
    print("EXP 3A: xi_inf vs SEED_BETA  (FA=0.40)")
    print("=" * 65)
    print(f"  {'sb':>6} | {'xi_inf':>9}")
    print("  " + "-" * 20)
    for sb in SEED_BETA_VALS:
        xi = xi_late(key_c, sb, FA_STD)
        print(f"  {sb:>6.2f} | {xi:>9.2f}")

    print()
    print("=" * 65)
    print("EXP 3B: xi_inf vs FA  (SEED_BETA=0)")
    print("=" * 65)
    print(f"  {'FA':>6} | {'xi_inf(sb=0)':>13} | {'xi_inf(Paper23)':>16}")
    print("  " + "-" * 42)
    # Paper 23 FA values for comparison
    p23_file = os.path.join(os.path.dirname(__file__), "..",
                            "paper23_two_scale", "results",
                            "paper23_results.json")
    r23 = json.load(open(p23_file)) if os.path.exists(p23_file) else {}
    def p23_xi_late(fa):
        vals = []
        for seed in range(N_SEEDS):
            k = f"p23,fa,{fa:.8g},{seed}"
            if k not in r23:
                continue
            for t in CPS:
                if t < 2400:
                    continue
                corr = r23[k].get(f"corr_{t}")
                if corr:
                    xi = fit_xi(corr)
                    if not math.isnan(xi) and xi <= XI_CAP:
                        vals.append(xi)
        return float(np.mean(vals)) if vals else float("nan")

    for fa in FA_VALS_C:
        xi_sb0 = xi_late(key_c, 0.0, fa)
        xi_p23 = p23_xi_late(fa)
        print(f"  {fa:>6.2f} | {xi_sb0:>13.2f} | {xi_p23:>16.2f}")
