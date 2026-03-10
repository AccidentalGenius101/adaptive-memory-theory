"""
paper34_experiments.py -- The (nu, kappa) Phase Diagram of VCML

Paper 33 proved VCML is formally non-gradient (curl = -FA != 0). Paper 34 maps
the full (nu, kappa) phase space to characterise:

  Q4: Critical exponents of the Allen-Cahn -> Burgers crossover
      Paper 26: transition at nu/kappa ~ 1. Is this a sharp phase transition?
      What is the order parameter exponent? Is there a diverging length scale?

  Q5: Are C1 and C2 truly independent manifolds?
      C1 = (nu, FD, FA, betas) -- viability volume
      C2 = (kappa, nu/kappa, betas, W_zone) -- structural manifold
      Test: hold nu/kappa constant, vary absolute scale (nu, kappa) together.
      If C1 and C2 independent: sg4 should vary with nu even at fixed nu/kappa.
      If only nu/kappa matters: sg4 should be constant at fixed nu/kappa.

Exp A: (nu, kappa) phase diagram (125 runs)
  nu    in {0.001, 0.002, 0.005, 0.010, 0.020}   -- 5 values
  kappa in {0.005, 0.010, 0.020, 0.040, 0.080}   -- 5 values
  5 seeds each = 5 x 5 x 5 = 125 runs, T_END=4000
  CPS = [1000, 2000, 3000, 4000]
  Store: sg4_{t}, within_std_{t} (within-zone field std, proxy for noise)
  Key: "p34a,{nu:.8g},{kappa:.8g},{seed}"

  ν/κ spans: 0.001/0.080=0.0125 to 0.020/0.005=4.0   [straddles transition at ~1]
  Measures:
    - sg4 at T=4000 (differentiation)
    - within_std at T=4000 (within-zone noise)
    - SNR = sg4 / within_std (signal-to-noise ratio -- quality of differentiation)
    - Phase assignment: Allen-Cahn (nu/kappa < 1) vs Burgers (nu/kappa > 1)

Exp B: C1/C2 independence test (45 runs)
  Hold nu/kappa = r_ratio, vary absolute scale:
    r=0.05: (nu=0.001, kappa=0.020), (nu=0.002, kappa=0.040), (nu=0.005, kappa=0.100)
    r=0.50: (nu=0.001, kappa=0.002), (nu=0.002, kappa=0.004), (nu=0.005, kappa=0.010)
    r=5.00: (nu=0.005, kappa=0.001), (nu=0.010, kappa=0.002), (nu=0.020, kappa=0.004)
  5 seeds each = 9 conditions x 5 = 45 runs, T_END=4000
  Key: "p34b,{nu:.8g},{kappa:.8g},{seed}"

  If C1 and C2 are INDEPENDENT:
    - At fixed r=0.05 (Allen-Cahn): sg4 varies with nu (viability volume changes)
    - The ratio nu/kappa sets PDE class, but nu alone sets consolidation rate
  If ONLY nu/kappa matters:
    - sg4 constant within each r group (manifolds merged into 1D)

Total: 125 + 45 = 170 runs.
"""
import numpy as np, json, os, multiprocessing as mp, math
from scipy.stats import linregress

# ── Fixed parameters ────────────────────────────────────────────────────────
H         = 20
N_ZONES   = 4
ZONE_W    = 5
BASE_BETA = 0.005
ALPHA_MID = 0.15
MID_DECAY = 0.99
FD        = 0.9997
SS        = 10
WAVE_DUR  = 15
HS        = 2
SEED_BETA = 0.25
WR_STD    = 2.4
FA_STD    = 0.200
T_END     = 4000
CPS       = [1000, 2000, 3000, 4000]

HALF = N_ZONES * ZONE_W   # = 20
N_ACT = HALF * H          # = 400

# ── Experiment grids ─────────────────────────────────────────────────────────
N_SEEDS = 5

NU_VALS_A    = [0.001, 0.002, 0.005, 0.010, 0.020]
KAPPA_VALS_A = [0.005, 0.010, 0.020, 0.040, 0.080]

# Exp B: (nu, kappa) pairs grouped by nu/kappa ratio
EXP_B_GROUPS = {
    0.05: [(0.001, 0.020), (0.002, 0.040), (0.005, 0.100)],
    0.50: [(0.001, 0.002), (0.002, 0.004), (0.005, 0.010)],
    5.00: [(0.005, 0.001), (0.010, 0.002), (0.020, 0.004)],
}


# ── Geometry (fixed ZONE_W=5) ────────────────────────────────────────────────
col     = np.arange(N_ACT) % HALF
row     = np.arange(N_ACT) // HALF
zone_id = col // ZONE_W
top_mask = row < H // 2
bot_mask = row >= H // 2

NB = []
for i in range(N_ACT):
    c, r = int(col[i]), int(row[i])
    nb = []
    for dc, dr in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        nc, nr = c + dc, r + dr
        if 0 <= nc < HALF and 0 <= nr < H:
            nb.append(nr * HALF + nc)
    NB.append(np.array(nb, dtype=int))


# ── Measurements ─────────────────────────────────────────────────────────────
def sg4_fn(F):
    means = [F[zone_id == z].mean(0) for z in range(N_ZONES)]
    pairs = [(i, j) for i in range(N_ZONES) for j in range(i+1, N_ZONES)]
    return float(np.mean([np.linalg.norm(means[i] - means[j]) for i, j in pairs]))


def within_std_fn(F):
    """Mean within-zone standard deviation (noise level inside zones)."""
    stds = []
    for z in range(N_ZONES):
        Fz = F[zone_id == z]
        if len(Fz) > 1:
            # std of the HS-norm of each site's field vector
            norms = np.linalg.norm(Fz, axis=1)
            stds.append(float(np.std(norms)))
    return float(np.mean(stds)) if stds else 0.0


# ── Core simulation ──────────────────────────────────────────────────────────
d_B = np.array([0.0, 1.0])

def run(seed, nu, kappa, fa=FA_STD, wr=WR_STD, t_end=T_END, cps=None):
    if cps is None:
        cps = CPS
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

        # Wave launching (Bernoulli, WR << WAVE_DUR so at most 1 per step here)
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

        # Diffusion
        dF = np.zeros_like(F)
        for i in range(N_ACT):
            if len(NB[i]):
                dF[i] = kappa * (F[NB[i]].mean(0) - F[i])
        F += dF

        # Turnover
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
            result[f"sg4_{step}"]       = sg4_fn(F)
            result[f"within_std_{step}"] = within_std_fn(F)

    return result


# ── Key factories ────────────────────────────────────────────────────────────
def key_a(nu, kappa, seed):
    return f"p34a,{nu:.8g},{kappa:.8g},{seed}"

def key_b(nu, kappa, seed):
    return f"p34b,{nu:.8g},{kappa:.8g},{seed}"


# ── Worker ───────────────────────────────────────────────────────────────────
def _worker(args):
    tag, nu, kappa, seed = args
    return run(seed, nu=nu, kappa=kappa)


RESULTS_FILE = os.path.join(os.path.dirname(__file__),
                            "results", "paper34_results.json")


if __name__ == "__main__":
    mp.freeze_support()

    # Build all conditions
    all_conditions = []
    for nu in NU_VALS_A:
        for kappa in KAPPA_VALS_A:
            for seed in range(N_SEEDS):
                all_conditions.append(("a", nu, kappa, seed, key_a(nu, kappa, seed)))

    for r_ratio, pairs in EXP_B_GROUPS.items():
        for (nu, kappa) in pairs:
            for seed in range(N_SEEDS):
                all_conditions.append(("b", nu, kappa, seed, key_b(nu, kappa, seed)))

    # Load cache
    if os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE) as f:
            results = json.load(f)
        print(f"Loaded {len(results)} cached results.")
    else:
        results = {}

    todo      = [(tag, nu, kappa, seed)
                 for tag, nu, kappa, seed, k in all_conditions if k not in results]
    todo_keys = [k for tag, nu, kappa, seed, k in all_conditions if k not in results]

    if todo:
        print(f"Running {len(todo)} / {len(all_conditions)} simulations (Paper 34)...")
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

    T_LAST = CPS[-1]

    # ── Exp A Analysis: (nu, kappa) phase diagram ─────────────────────────────
    print()
    print("=" * 72)
    print("EXP A: (nu, kappa) Phase Diagram")
    print("=" * 72)
    print(f"  {'nu':>8}  {'kappa':>8}  {'nu/kappa':>10}  {'sg4':>8}  "
          f"{'w_std':>8}  {'SNR':>8}  {'regime':>12}")
    print("  " + "-" * 72)

    sg4_grid    = np.full((len(NU_VALS_A), len(KAPPA_VALS_A)), float("nan"))
    snr_grid    = np.full((len(NU_VALS_A), len(KAPPA_VALS_A)), float("nan"))

    ac_sg4, bu_sg4 = [], []
    ac_snr, bu_snr = [], []

    for i, nu in enumerate(NU_VALS_A):
        for j, kappa in enumerate(KAPPA_VALS_A):
            sg4_vals = [results.get(key_a(nu, kappa, s), {}).get(f"sg4_{T_LAST}", float("nan"))
                        for s in range(N_SEEDS)]
            ws_vals  = [results.get(key_a(nu, kappa, s), {}).get(f"within_std_{T_LAST}", float("nan"))
                        for s in range(N_SEEDS)]
            sg4_vals = [v for v in sg4_vals if not math.isnan(v)]
            ws_vals  = [v for v in ws_vals  if not math.isnan(v)]

            if not sg4_vals:
                continue

            sg4_mu = float(np.mean(sg4_vals))
            ws_mu  = float(np.mean(ws_vals)) if ws_vals else float("nan")
            snr    = sg4_mu / ws_mu if ws_mu and ws_mu > 0 else float("nan")
            ratio  = nu / kappa
            regime = "Allen-Cahn" if ratio < 1.0 else "Burgers"

            sg4_grid[i, j] = sg4_mu
            snr_grid[i, j] = snr

            print(f"  {nu:>8.4f}  {kappa:>8.4f}  {ratio:>10.3f}  {sg4_mu:>8.4f}  "
                  f"{ws_mu:>8.4f}  {snr:>8.2f}  {regime:>12}")

            if ratio < 1.0:
                ac_sg4.append(sg4_mu)
                if not math.isnan(snr): ac_snr.append(snr)
            else:
                bu_sg4.append(sg4_mu)
                if not math.isnan(snr): bu_snr.append(snr)

    print()
    if ac_sg4 and bu_sg4:
        print(f"  Allen-Cahn regime (nu/kappa < 1): mean sg4 = {np.mean(ac_sg4):.4f}")
        print(f"  Burgers   regime (nu/kappa > 1): mean sg4 = {np.mean(bu_sg4):.4f}")
    if ac_snr and bu_snr:
        print(f"  Allen-Cahn SNR = {np.mean(ac_snr):.2f}  |  Burgers SNR = {np.mean(bu_snr):.2f}")

    # Log-log: sg4 vs nu/kappa (all points)
    ratios_all, sg4_all = [], []
    for i, nu in enumerate(NU_VALS_A):
        for j, kappa in enumerate(KAPPA_VALS_A):
            v = sg4_grid[i, j]
            if not math.isnan(v) and v > 0:
                ratios_all.append(nu / kappa)
                sg4_all.append(v)

    if len(ratios_all) >= 4:
        log_r = np.log(ratios_all)
        log_s = np.log(sg4_all)
        sl, intercept, r_val, *_ = linregress(log_r, log_s)
        print(f"\n  Log-log: sg4 vs nu/kappa: slope={sl:.3f}  R2={r_val**2:.3f}")
        print(f"  Interpretation: sg4 ~ (nu/kappa)^{sl:.3f}")

    # ── Exp B Analysis: C1/C2 independence ───────────────────────────────────
    print()
    print("=" * 72)
    print("EXP B: C1/C2 Independence Test (fixed nu/kappa, varying absolute scale)")
    print("=" * 72)
    print(f"  {'r=nu/kappa':>12}  {'nu':>8}  {'kappa':>8}  {'sg4':>8}  verdict")
    print("  " + "-" * 60)

    for r_ratio in sorted(EXP_B_GROUPS.keys()):
        pairs = EXP_B_GROUPS[r_ratio]
        sg4_at_r = []
        for (nu, kappa) in pairs:
            sg4_vals = [results.get(key_b(nu, kappa, s), {}).get(f"sg4_{T_LAST}", float("nan"))
                        for s in range(N_SEEDS)]
            sg4_vals = [v for v in sg4_vals if not math.isnan(v)]
            mu = float(np.mean(sg4_vals)) if sg4_vals else float("nan")
            sg4_at_r.append(mu)
            print(f"  {r_ratio:>12.2f}  {nu:>8.4f}  {kappa:>8.4f}  {mu:>8.4f}")

        valid = [v for v in sg4_at_r if not math.isnan(v)]
        if len(valid) >= 2:
            cv = float(np.std(valid) / np.mean(valid)) if np.mean(valid) > 0 else float("nan")
            verdict = "VARIES (C1 x C2 independent)" if cv > 0.15 else "FLAT (only nu/kappa matters)"
            print(f"  {'':>12}  CV at r={r_ratio:.2f}: {cv:.3f} -> {verdict}")
        print()

    print()
    print("All done.")
