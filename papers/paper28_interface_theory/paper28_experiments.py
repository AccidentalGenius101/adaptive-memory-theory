"""
paper28_experiments.py -- The Zone Boundary as an Extended Object

Exp A (kappa sweep, 30 runs): Interface profile vs kappa
  kappa in {0.002, 0.005, 0.010, 0.020, 0.040, 0.080}
  FA=0.40, WR=2.4, T_END=4000, N_SEEDS=5
  Store: col_profile_{t}, sg4_{t} at 4 checkpoints
  Tests: xi_boundary scales as sqrt(kappa)? Profile is tanh-shaped?

Exp B (temporal dynamics, 5 runs): Fine-grained interface tracking
  kappa=0.020, FA=0.40, WR=2.4, T_END=6000, N_SEEDS=5
  Store: col_profile_{t}, sg4_{t} at 30 checkpoints
  Tests: xi(t) and A(t) as internal DOF of the boundary object

Exp C (isothermal, 20 runs): Fixed kappa/FA, vary both
  4 pairs: (0.005, 0.10), (0.010, 0.20), (0.020, 0.40), (0.040, 0.80)
  WR=2.4, T_END=4000, N_SEEDS=5
  Tests: profile depends on kappa/FA ratio alone, or separately?

Total: 55 runs.
"""
import numpy as np, json, os, multiprocessing as mp, math
from scipy.optimize import curve_fit
from scipy.stats import linregress

# ── Fixed parameters (identical to Papers 22-27) ──────────────────────────────
H         = 20
N_ZONES   = 4
HALF      = 20
BASE_BETA = 0.005
ALPHA_MID = 0.15
MID_DECAY = 0.99
FD        = 0.9997
SS        = 10
WAVE_DUR  = 15
HS        = 2
SEED_BETA = 0.25
NU        = 0.001

# Standard defaults
KAPPA_STD = 0.020
WR_STD    = 2.4
FA_STD    = 0.40
ZONE_W    = HALF // N_ZONES   # 5 columns per zone

# ── Experiment grids ──────────────────────────────────────────────────────────
# Exp A: kappa sweep
KAPPA_VALS_A = [0.002, 0.005, 0.010, 0.020, 0.040, 0.080]
N_SEEDS_A    = 5
T_END_A      = 4000
CPS_A        = [1000, 2000, 3000, 4000]

# Exp B: temporal dynamics (fine checkpoints)
N_SEEDS_B = 5
T_END_B   = 6000
CPS_B     = list(range(200, T_END_B + 1, 200))   # 30 checkpoints

# Exp C: isothermal (kappa/FA = KAPPA_STD/FA_STD = 0.05 = const)
ISOTHERMAL_PAIRS = [(0.005, 0.10), (0.010, 0.20), (0.020, 0.40), (0.040, 0.80)]
N_SEEDS_C        = 5
T_END_C          = 4000
CPS_C            = [1000, 2000, 3000, 4000]

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

# ── Measurements ──────────────────────────────────────────────────────────────
def sg4_fn(F):
    means = [F[zone_id == z].mean(0) for z in range(N_ZONES)]
    pairs = [(i, j) for i in range(N_ZONES) for j in range(i + 1, N_ZONES)]
    return float(np.mean([np.linalg.norm(means[i] - means[j]) for i, j in pairs]))


def col_profile_fn(F):
    """
    Column-averaged field separated by top/bottom half.
    Returns dict with 'top' (H/2 rows averaged), 'bot', 'diff' = top - bot.
    Each is a (HALF, HS) list.
    The 'diff' channel isolates the perturbation-direction signal (HS[1]).
    The 'full' channel (all rows averaged) captures HS[0] zone differentiation.
    """
    F_grid  = F.reshape(H, HALF, HS)
    full    = F_grid.mean(axis=0)
    top     = F_grid[:H // 2, :, :].mean(axis=0)
    bot     = F_grid[H // 2:, :, :].mean(axis=0)
    return {
        "full": full.tolist(),     # (HALF, HS): both-row average
        "top":  top.tolist(),      # (HALF, HS): top-half average
        "bot":  bot.tolist(),      # (HALF, HS): bottom-half average
        "diff": (top - bot).tolist()  # (HALF, HS): top minus bottom contrast
    }


def fit_tanh_boundary(profile_1d, boundary_b):
    """
    Fit A*tanh((x - x0)/xi) + offset to a 1D column profile at boundary b.
    boundary_b: 0,1,2 for zone-boundaries 0/1, 1/2, 2/3.
    Uses all columns in the two adjacent zones (2*ZONE_W = 10 data points).
    Returns dict(A, x0, xi, off, resid) or None.
    """
    y   = np.asarray(profile_1d, dtype=float)   # length HALF
    x   = np.arange(HALF, dtype=float)
    b   = boundary_b
    # Columns spanning zones b and b+1
    left  = b * ZONE_W
    right = (b + 2) * ZONE_W   # exclusive
    right = min(right, HALF)
    cols  = np.arange(left, right)
    if len(cols) < 4:
        return None
    x_fit = x[cols]
    y_fit = y[cols]
    # Expected boundary position (between zones b and b+1)
    x0_mid = (b + 1) * ZONE_W - 0.5
    A_g    = (y_fit[-1] - y_fit[0]) / 2.0
    if abs(A_g) < 1e-12:
        A_g = 1e-6

    def tanh_fn(xx, A, x0, xi, off):
        return A * np.tanh((xx - x0) / max(abs(xi), 0.01)) + off

    try:
        popt, _ = curve_fit(
            tanh_fn, x_fit, y_fit,
            p0=[A_g, x0_mid, 1.5, float(y_fit.mean())],
            bounds=([-np.inf, float(left),   0.01, -np.inf],
                    [ np.inf, float(right),  30.0,  np.inf]),
            maxfev=3000
        )
        A, x0, xi, off = popt
        resid = float(np.std(y_fit - tanh_fn(x_fit, *popt)))
        return {"A": float(A), "x0": float(x0), "xi": float(xi),
                "off": float(off), "resid": resid}
    except Exception:
        return None


def interface_stats_fn(F):
    """
    Fit all 3 zone boundaries in BOTH 'full[:,0]' and 'diff[:,1]' channels.
    Returns list of 3 dicts (one per boundary), each with fits from both channels.
    """
    cp     = col_profile_fn(F)
    full0  = np.array(cp["full"])[:, 0]   # HS[0] of full-row average
    diff1  = np.array(cp["diff"])[:, 1]   # HS[1] of top-bot contrast
    stats  = []
    for b in range(N_ZONES - 1):
        f0 = fit_tanh_boundary(full0, b)
        f1 = fit_tanh_boundary(diff1, b)
        stats.append({"full0": f0, "diff1": f1})
    return stats


# ── Core simulation ───────────────────────────────────────────────────────────
def run(seed, kappa=KAPPA_STD, fa=FA_STD, wr=WR_STD,
        t_end=T_END_A, cps=None):
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

        # Wave launch (Poisson-like)
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

        # Spatial diffusion with kappa as parameter
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
            ttl[i]    = float(rng.geometric(p=NU))

        waves = [[w[0], w[1] - 1, w[2], w[3]] for w in waves if w[1] - 1 > 0]

        if step in cps_set:
            result[f"sg4_{step}"]        = sg4_fn(F)
            result[f"col_profile_{step}"] = col_profile_fn(F)

    return result


# ── Key factories ─────────────────────────────────────────────────────────────
def key_a(kappa, seed):         return f"p28a,{kappa:.8g},{seed}"
def key_b(seed):                return f"p28b,{seed}"
def key_c(kappa, fa, seed):     return f"p28c,{kappa:.8g},{fa:.8g},{seed}"


# ── Worker (module-level for Windows spawn) ───────────────────────────────────
def _worker(args):
    tag, params, seed = args
    if tag == "a":
        kappa, = params
        return run(seed, kappa=kappa, fa=FA_STD, wr=WR_STD,
                   t_end=T_END_A, cps=CPS_A)
    elif tag == "b":
        return run(seed, kappa=KAPPA_STD, fa=FA_STD, wr=WR_STD,
                   t_end=T_END_B, cps=CPS_B)
    else:  # tag == "c"
        kappa, fa = params
        return run(seed, kappa=kappa, fa=fa, wr=WR_STD,
                   t_end=T_END_C, cps=CPS_C)


RESULTS_FILE = os.path.join(os.path.dirname(__file__),
                            "results", "paper28_results.json")


# ── Helpers (shared with figure script) ───────────────────────────────────────
def _get_xi(col_profile_dict, channel="full0"):
    """
    Extract mean xi over 3 boundaries from a stored col_profile dict.
    channel: 'full0' uses full[:,0]; 'diff1' uses diff[:,1].
    """
    if channel == "full0":
        arr = np.array(col_profile_dict["full"])[:, 0]
    else:
        arr = np.array(col_profile_dict["diff"])[:, 1]
    xis = []
    for b in range(N_ZONES - 1):
        fit = fit_tanh_boundary(arr, b)
        if fit is not None:
            xis.append(fit["xi"])
    return float(np.mean(xis)) if xis else float("nan")


def _get_A(col_profile_dict, channel="full0"):
    """Extract mean |A| over 3 boundaries."""
    if channel == "full0":
        arr = np.array(col_profile_dict["full"])[:, 0]
    else:
        arr = np.array(col_profile_dict["diff"])[:, 1]
    As = []
    for b in range(N_ZONES - 1):
        fit = fit_tanh_boundary(arr, b)
        if fit is not None:
            As.append(abs(fit["A"]))
    return float(np.mean(As)) if As else float("nan")


if __name__ == "__main__":
    mp.freeze_support()

    # Build full condition list
    all_conditions = []
    for kappa in KAPPA_VALS_A:
        for seed in range(N_SEEDS_A):
            all_conditions.append(("a", (kappa,), seed, key_a(kappa, seed)))
    for seed in range(N_SEEDS_B):
        all_conditions.append(("b", (), seed, key_b(seed)))
    for kappa, fa in ISOTHERMAL_PAIRS:
        for seed in range(N_SEEDS_C):
            all_conditions.append(("c", (kappa, fa), seed, key_c(kappa, fa, seed)))

    # Load cache
    if os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE) as f:
            results = json.load(f)
        print(f"Loaded {len(results)} cached results.")
    else:
        results = {}

    todo      = [(tag, params, seed)
                 for tag, params, seed, key in all_conditions
                 if key not in results]
    todo_keys = [key for tag, params, seed, key in all_conditions
                 if key not in results]

    if todo:
        print(f"Running {len(todo)} / {len(all_conditions)} simulations (Paper 28)...")
        with mp.Pool(processes=min(len(todo), mp.cpu_count())) as pool:
            raw = pool.map(_worker, todo)
        for key, res in zip(todo_keys, raw):
            results[key] = res
        os.makedirs(os.path.dirname(RESULTS_FILE), exist_ok=True)
        with open(RESULTS_FILE, "w") as f:
            json.dump(results, f)
        print("Saved results.")
    else:
        print("All results cached.")

    # ── Analysis: Exp A ───────────────────────────────────────────────────────
    T_LAST_A = CPS_A[-1]   # 4000

    print()
    print("=" * 72)
    print("EXP A: Interface width xi vs kappa (t=4000)")
    print("       Both channels: full[:,0] (HS[0]) and diff[:,1] (HS[1])")
    print("=" * 72)
    print(f"  {'kappa':>8} | {'xi_full0':>10} {'xi_diff1':>10} | {'sg4':>8}")
    print("  " + "-" * 50)

    xi_a_full0 = {}
    xi_a_diff1 = {}
    sg4_a      = {}
    for kappa in KAPPA_VALS_A:
        xf, xd, sg = [], [], []
        for s in range(N_SEEDS_A):
            k  = key_a(kappa, s)
            if k not in results:
                continue
            cp  = results[k].get(f"col_profile_{T_LAST_A}")
            sg4 = results[k].get(f"sg4_{T_LAST_A}", float("nan"))
            if cp is not None:
                xf.append(_get_xi(cp, "full0"))
                xd.append(_get_xi(cp, "diff1"))
            sg.append(sg4)
        xi_a_full0[kappa] = xf
        xi_a_diff1[kappa] = xd
        sg4_a[kappa]      = sg
        xf_m = float(np.nanmean(xf)) if xf else float("nan")
        xd_m = float(np.nanmean(xd)) if xd else float("nan")
        sg_m = float(np.nanmean(sg)) if sg else float("nan")
        print(f"  {kappa:>8.4f} | {xf_m:>10.3f} {xd_m:>10.3f} | {sg_m:>8.4f}")

    # Log-log slope test for best channel
    for ch_name, xi_dict in [("full0", xi_a_full0), ("diff1", xi_a_diff1)]:
        kv, xv = [], []
        for kappa in KAPPA_VALS_A:
            xis = [v for v in xi_dict.get(kappa, []) if not math.isnan(v)]
            if xis and np.nanmean(xis) > 0.01:
                kv.append(math.log(kappa))
                xv.append(math.log(np.nanmean(xis)))
        if len(kv) >= 3:
            slope, _, r, *_ = linregress(kv, xv)
            print(f"\n  [{ch_name}] log-log slope = {slope:.3f}  "
                  f"(Allen-Cahn prediction: 0.50)  R2={r**2:.3f}")

    # ── Analysis: Exp B ───────────────────────────────────────────────────────
    print()
    print("=" * 72)
    print("EXP B: Interface internal dynamics (kappa=0.020, T_END=6000)")
    print("=" * 72)

    xi_B, A_B, sg4_B = [], [], []
    for t in CPS_B:
        xis_t, As_t, sg_t = [], [], []
        for s in range(N_SEEDS_B):
            k = key_b(s)
            if k not in results:
                continue
            cp  = results[k].get(f"col_profile_{t}")
            sg  = results[k].get(f"sg4_{t}", float("nan"))
            if cp is not None:
                xi_s = _get_xi(cp, "full0")
                A_s  = _get_A(cp, "full0")
                if not math.isnan(xi_s):
                    xis_t.append(xi_s)
                if not math.isnan(A_s):
                    As_t.append(A_s)
            sg_t.append(sg)
        xi_B.append(float(np.nanmean(xis_t)) if xis_t else float("nan"))
        A_B.append( float(np.nanmean(As_t))  if As_t  else float("nan"))
        sg4_B.append(float(np.nanmean(sg_t)) if sg_t  else float("nan"))

    valid_xi = [v for v in xi_B if not math.isnan(v)]
    valid_A  = [v for v in A_B  if not math.isnan(v)]
    if valid_xi:
        cv_xi = np.std(valid_xi) / np.mean(valid_xi)
        print(f"  xi: mean={np.mean(valid_xi):.3f}, std={np.std(valid_xi):.3f}, "
              f"CV={cv_xi:.3f}")
        verdict = "SIGNIFICANT internal dynamics" if cv_xi > 0.10 else "STABLE interface"
        print(f"  --> {verdict} (threshold CV > 0.10)")
    if valid_A:
        cv_A = np.std(valid_A) / np.mean(valid_A)
        print(f"  |A|: mean={np.mean(valid_A):.4f}, std={np.std(valid_A):.4f}, "
              f"CV={cv_A:.3f}")

    # ── Analysis: Exp C ───────────────────────────────────────────────────────
    print()
    print("=" * 72)
    print("EXP C: Isothermal test (kappa/FA = 0.05 fixed)")
    print("=" * 72)
    print(f"  {'kappa':>8} {'FA':>6} | {'xi_full0':>10} {'xi_diff1':>10} | {'sg4':>8}")
    print("  " + "-" * 55)
    T_LAST_C = CPS_C[-1]   # 4000

    xi_C_full0, xi_C_diff1 = [], []
    for kappa, fa in ISOTHERMAL_PAIRS:
        xf, xd, sg = [], [], []
        for s in range(N_SEEDS_C):
            k  = key_c(kappa, fa, s)
            if k not in results:
                continue
            cp  = results[k].get(f"col_profile_{T_LAST_C}")
            sg4 = results[k].get(f"sg4_{T_LAST_C}", float("nan"))
            if cp is not None:
                xf.append(_get_xi(cp, "full0"))
                xd.append(_get_xi(cp, "diff1"))
            sg.append(sg4)
        xi_C_full0.append(float(np.nanmean(xf)) if xf else float("nan"))
        xi_C_diff1.append(float(np.nanmean(xd)) if xd else float("nan"))
        xf_m = float(np.nanmean(xf)) if xf else float("nan")
        xd_m = float(np.nanmean(xd)) if xd else float("nan")
        sg_m = float(np.nanmean(sg)) if sg else float("nan")
        print(f"  {kappa:>8.4f} {fa:>6.2f} | {xf_m:>10.3f} {xd_m:>10.3f} | {sg_m:>8.4f}")

    valid_C0 = [v for v in xi_C_full0 if not math.isnan(v)]
    valid_C1 = [v for v in xi_C_diff1 if not math.isnan(v)]
    if valid_C0:
        cv_c = np.std(valid_C0) / np.mean(valid_C0) if np.mean(valid_C0) > 0 else float("nan")
        print(f"\n  [full0] xi CV across isothermal pairs = {cv_c:.3f}")
        verdict = "SAME width (kappa/FA is sufficient)" if cv_c < 0.15 \
                  else "DIFFERENT widths (kappa and FA matter separately)"
        print(f"  --> {verdict}")
