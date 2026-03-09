"""
paper18_temporal_dynamics.py -- Experiment for Paper 18

Temporal Dynamics of the Saturation Law:
  Phase 1: waves running (t=1 to T_BUILD=2000) -- buildup
  Phase 3: no waves (t=2001 to T_END=2200)    -- forgetting

Sweeps:
  FA    in {0.10, 0.20, 0.40, 0.70}
  KAPPA in {0.005, 0.020, 0.080}
  5 seeds each -> 60 runs total

Checkpoints:
  Phase 1: every 200 steps (t=200,400,...,2000) -- 10 points
  Phase 3: every 10 steps  (t=2010,2020,...,2200) -- 20 points

Key: "p18,{fa:.4f},{kp:.4f},{seed}" -> {"sg4_{t}": float for all checkpoint t}

Analytical predictions:
  tau_build ~ 1/BASE_BETA = 200 steps (baseline-limited, FA-independent)
  tau_forget ~ 1/FA steps (consolidation-driven decay in Phase 3)
"""
import numpy as np, json, os, multiprocessing as mp, math

W, H = 40, 20; HALF = W // 2
HS = 2; N_ZONES = 4; ZONE_W = HALF // N_ZONES
N_ACT = HALF * H

BASE_BETA  = 0.005
ALPHA_MID  = 0.15
MID_DECAY  = 0.99
SS         = 10
FD         = 0.9997
NU         = 0.001
WR         = 4.8
WAVE_DUR   = 15
SEED_BETA  = 0.25

T_BUILD = 2000    # last step of Phase 1
T_END   = 2200    # last step of Phase 3

PHASE1_CPS = list(range(200, T_BUILD + 1, 200))           # [200,400,...,2000]
PHASE3_CPS = list(range(T_BUILD + 10, T_END + 1, 10))     # [2010,2020,...,2200]
ALL_CPS_SET = set(PHASE1_CPS + PHASE3_CPS)

FA_VALS    = [0.10, 0.20, 0.40, 0.70]
KAPPA_VALS = [0.005, 0.020, 0.080]
N_SEEDS    = 5

# Geometry
_col = np.arange(N_ACT) % HALF
_row = np.arange(N_ACT) // HALF
zone_id = _col // ZONE_W

NB = []
for i in range(N_ACT):
    c, r = _col[i], _row[i]
    nb = []
    for dc, dr in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        nc, nr = c + dc, r + dr
        if 0 <= nc < HALF and 0 <= nr < H:
            nb.append(nr * HALF + nc)
    NB.append(np.array(nb, dtype=int))

top_mask = _row < H // 2
bot_mask = _row >= H // 2
d_B = np.array([0.0, 1.0])


def sg4_inter_fn(F):
    means = [F[zone_id == z].mean(0) for z in range(N_ZONES)]
    pairs = [(i, j) for i in range(N_ZONES) for j in range(i + 1, N_ZONES)]
    return float(np.mean([np.linalg.norm(means[i] - means[j]) for i, j in pairs]))


def run(seed, fa, kappa):
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

        # Phase 1 only: launch new waves
        if step <= T_BUILD:
            n = int(WR / WAVE_DUR)
            n += int(rng.random() < (WR / WAVE_DUR - n))
            for _ in range(n):
                top  = rng.random() < 0.5
                sign = 1.0 if top else -1.0
                waves.append([0 if top else 1, WAVE_DUR, sign * d_B.copy(), False])

        # Which sites are perturbed this step
        pert = np.zeros(N_ACT, bool)
        for w in waves:
            pert |= (top_mask if w[0] == 0 else bot_mask)

        # Baseline tracking (all sites, always)
        base += BASE_BETA * (h - base)

        # Apply wave perturbations to h
        for w in waves:
            mask = top_mask if w[0] == 0 else bot_mask
            h[mask] += 0.3 * w[2]

        # mid_mem update
        m[pert] += ALPHA_MID * (h - base)[pert]
        m *= MID_DECAY

        # Streak and consolidation gate
        streak[pert]  = 0
        streak[~pert] += 1
        ok = streak >= SS
        F[ok] += fa * (m[ok] - F[ok])
        F *= FD

        # Spatial diffusion
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
            j = NB[i][rng.integers(len(NB[i]))] if len(NB[i]) else i
            h[i]      = (1 - SEED_BETA) * rng.normal(0, 0.1, HS) + SEED_BETA * F[j]
            m[i]      = 0
            streak[i] = 0
            ttl[i]    = float(rng.geometric(p=NU))

        # Advance wave timers
        waves = [[w[0], w[1] - 1, w[2], w[3]] for w in waves if w[1] - 1 > 0]

        # Record checkpoint
        if step in ALL_CPS_SET:
            result[f"sg4_{step}"] = sg4_inter_fn(F)

    return result


def make_key(fa, kappa, seed):
    return f"p18,{fa:.4f},{kappa:.4f},{seed}"


def _worker(args):
    return run(*args)


RESULTS_FILE = os.path.join(os.path.dirname(__file__), "results", "paper18_results.json")


if __name__ == "__main__":
    mp.freeze_support()

    if os.path.exists(RESULTS_FILE):
        results = json.load(open(RESULTS_FILE))
        print("Loaded cached results.")
    else:
        all_conditions = [
            (seed, fa, kappa)
            for fa    in FA_VALS
            for kappa in KAPPA_VALS
            for seed  in range(N_SEEDS)
        ]
        print(f"Running {len(all_conditions)} simulations (Paper 18)...")
        with mp.Pool(processes=min(len(all_conditions), mp.cpu_count())) as pool:
            raw = pool.map(_worker, all_conditions)

        results = {}
        for i, (seed, fa, kappa) in enumerate(all_conditions):
            results[make_key(fa, kappa, seed)] = raw[i]

        os.makedirs(os.path.dirname(RESULTS_FILE), exist_ok=True)
        json.dump(results, open(RESULTS_FILE, "w"), indent=2)
        print("Saved results.")

    # ---- Analysis ----
    P_CALM  = 1.0 - WR / (2.0 * WAVE_DUR)
    P_CONSOL = P_CALM ** SS
    print(f"\np_calm = {P_CALM:.4f},  P_consol = {P_CONSOL:.4f}")
    print(f"tau_build predicted: max(1/BASE_BETA, 1/(1-MID_DECAY)) = "
          f"max({1/BASE_BETA:.0f}, {1/(1-MID_DECAY):.0f}) = {max(1/BASE_BETA, 1/(1-MID_DECAY)):.0f} steps")
    print()

    def mean_t(keys, t):
        vals = [results[k].get(f"sg4_{t}", float("nan")) for k in keys if k in results]
        vals = [v for v in vals if not math.isnan(v)]
        return float(np.mean(vals)) if vals else float("nan")

    def tau_forget_est(keys):
        """Steps from T_BUILD until sg4 drops to 1/e of sg4(T_BUILD)."""
        v0 = mean_t(keys, T_BUILD)
        if math.isnan(v0) or v0 <= 0:
            return float("nan")
        target = v0 / math.e
        prev_t, prev_v = T_BUILD, v0
        for t in PHASE3_CPS:
            v = mean_t(keys, t)
            if not math.isnan(v) and v <= target:
                # Linear interpolation
                frac = (prev_v - target) / max(prev_v - v, 1e-12)
                return prev_t - T_BUILD + frac * (t - prev_t)
            prev_t, prev_v = t, v
        return float(">200")   # still above 1/e at end of Phase 3

    print("=" * 70)
    print("BUILDUP: sg4 at Phase 1 checkpoints (KAPPA=0.020)")
    print(f"  {'FA':>5} | " + " | ".join(f"t={t:4d}" for t in PHASE1_CPS))
    print("  " + "-" * 80)
    for fa in FA_VALS:
        keys = [make_key(fa, 0.020, s) for s in range(N_SEEDS)]
        vals = [f"{mean_t(keys, t):.4f}" for t in PHASE1_CPS]
        print(f"  {fa:.2f}  | " + " | ".join(vals))

    print()
    print("=" * 70)
    print("FORGETTING: normalized sg4 in Phase 3 (KAPPA=0.020)")
    print(f"  {'FA':>5} | " + " | ".join(f"t+{t-T_BUILD:3d}" for t in PHASE3_CPS[:10]))
    print("  " + "-" * 80)
    for fa in FA_VALS:
        keys = [make_key(fa, 0.020, s) for s in range(N_SEEDS)]
        v0 = mean_t(keys, T_BUILD)
        if v0 > 0:
            vals = [f"{mean_t(keys, t)/v0:.3f}" for t in PHASE3_CPS[:10]]
        else:
            vals = ["nan"] * 10
        print(f"  {fa:.2f}  | " + " | ".join(vals))

    print()
    print("=" * 70)
    print("FORGETTING TIMESCALE tau_forget (steps to 1/e decay)")
    print(f"  {'FA':>5} | " + " | ".join(f"K={k:.3f}" for k in KAPPA_VALS)
          + " | " + " | ".join(f"pred(K={k:.3f})" for k in KAPPA_VALS))
    print("  " + "-" * 80)
    for fa in FA_VALS:
        tau_vals = []
        for kappa in KAPPA_VALS:
            keys = [make_key(fa, kappa, s) for s in range(N_SEEDS)]
            tau_vals.append(tau_forget_est(keys))
        pred_1_fa = 1.0 / fa
        row = f"  {fa:.2f}  | " + " | ".join(f"{v:>10}" if isinstance(v, str)
                                               else f"{v:>10.1f}" for v in tau_vals)
        row += f" |   pred~{pred_1_fa:.1f}"
        print(row)
