"""
paper14_exp1_iso_ratio.py -- Experiment 1 for Paper 14

Timescale Nesting and Iso-Ratio Invariance.

THE CLAIM (from Paper 13 discussion):
  Spatial structure in VCML emerges when three process rates are properly ordered:
      lambda_decay  <  nu  <  lambda_consolidate

  The system can therefore be described by two dimensionless ratios:
      R_A = nu / nu_cryst          (replacement rate / crystallization rate)
      R_B = nu_max^consol / nu     (consolidation throughput / replacement rate)

  PREDICTION: sg4 depends only on (R_A, R_B), not on the absolute parameter values
  that achieve them. Two conditions with the same (R_A, R_B) but different
  {FD, FA, nu} should give the same sg4.

THE KEY IDENTITY:
  FIELD_DECAY^(1/nu) = FIELD_DECAY^(1 / (R_A * nu_cryst))
                     = (FIELD_DECAY^(1/nu_cryst))^(1/R_A)
                     = (1/2)^(1/R_A)           [by definition of nu_cryst]

  So FIELD_DECAY^(1/nu) depends ONLY on R_A, not on the absolute FD or nu.
  This means nu_max^consol = P_calm * FA * (1/2)^(1/R_A).
  And R_B = nu_max / nu = P_calm * FA * 0.5^(1/R_A) / (R_A * nu_cryst).

  For two routes to have the same (R_A, R_B):
      FA_route2 / FA_route1 = nu_cryst_route2 / nu_cryst_route1

  So doubling nu_cryst (e.g., by choosing FD_new = exp(2*ln(FD_ref))) requires
  doubling FA to maintain the same R_B. Absolute nu doubles, but R_A is unchanged.

EXPERIMENTAL DESIGN:
  Two routes to each (R_A, R_B) target:
    Route R1: FD = 0.9997 -> nu_cryst = 4.33e-4  (reference)
    Route R2: FD = 0.9994 -> nu_cryst = 8.66e-4  (double nu_cryst)

  FA_R2 = 2 * FA_R1  (to maintain same R_B)
  nu_R2 = 2 * nu_R1  (to maintain same R_A)

  Three targets:
    T1: (R_A=1.5, R_B=1.5) -- narrow adaptive window
    T2: (R_A=2.0, R_B=2.0) -- moderate window (reference-like)
    T3: (R_A=1.5, R_B=3.0) -- wide upper margin

  For each target, scan R_A values from 0.5 to 6.0 while holding (FA, FD) fixed
  and computing nu = R_A * nu_cryst. Plot sg4 vs R_A for both routes.
  If ratio hypothesis holds: Routes R1 and R2 should give IDENTICAL sg4(R_A) curves.

  Control T0: (R_A=0.5, outside lower bound) -- both routes should show low sg4.

DERIVED CONDITIONS:
  P_calm(WR=4.8, WD=15, SS=10) = (1 - 4.8/(2*15))^(10+15-1) = 0.84^24 = 0.01523

  T1 (R_A=1.5, R_B=1.5):  0.5^(1/1.5) = 0.6299
    FA_R1 = 1.5 * 1.5 * nu_cryst_R1 / (0.01523 * 0.6299) = 2.25 * 4.33e-4 / 9.59e-3 = 0.1016
    FA_R2 = 2 * FA_R1 = 0.2032

  T2 (R_A=2.0, R_B=2.0):  0.5^(1/2.0) = 0.7071
    FA_R1 = 2.0 * 2.0 * 4.33e-4 / (0.01523 * 0.7071) = 4 * 4.33e-4 / 0.01077 = 0.1609
    FA_R2 = 2 * FA_R1 = 0.3218

  T3 (R_A=1.5, R_B=3.0):  0.5^(1/1.5) = 0.6299
    FA_R1 = 3.0 * 1.5 * 4.33e-4 / (0.01523 * 0.6299) = 4.5 * 4.33e-4 / 9.59e-3 = 0.2031
    FA_R2 = 2 * FA_R1 = 0.4062

Grid: 8 R_A scan points, 5 seeds.
Per target: 2 routes * 8 RA * 5 seeds = 80 runs.
3 targets: 240 runs total. Runtime ~10 min.
"""
import numpy as np, json, os, multiprocessing as mp
import math

W, H = 40, 20; HALF = W // 2
HS = 2; N_ZONES = 4; ZONE_W = HALF // N_ZONES
N_ACT = HALF * H; STEPS = 2000; SHIFT = 1000; N_SEEDS = 5

MID_DECAY = 0.99; BASE_BETA = 0.005
ALPHA_MID = 0.15; SEED_BETA = 0.25
SS = 10; KAPPA = 0.020; WR = 4.8; WAVE_DUR = 15

# Two routes: different FD (and therefore different nu_cryst)
FD_R1 = 0.9997   # nu_cryst_R1 = 4.33e-4
FD_R2 = 0.9994   # nu_cryst_R2 = 8.66e-4  (double)

def nu_cryst_fn(fd):
    return abs(math.log(fd)) / math.log(2)

NU_CRYST_R1 = nu_cryst_fn(FD_R1)
NU_CRYST_R2 = nu_cryst_fn(FD_R2)

P_CALM = (1 - WR / (2 * WAVE_DUR)) ** (SS + WAVE_DUR - 1)

def compute_fa(ra_target, rb_target, nu_cryst):
    """Compute FA needed to achieve target (R_A, R_B) at given nu_cryst."""
    fd_term = 0.5 ** (1.0 / ra_target)   # FD^(1/nu) = 0.5^(1/R_A)
    return rb_target * ra_target * nu_cryst / (P_CALM * fd_term)

# Target (R_A, R_B) pairs
TARGETS = {
    "T1_1.5_1.5": (1.5, 1.5),
    "T2_2.0_2.0": (2.0, 2.0),
    "T3_1.5_3.0": (1.5, 3.0),
}

# R_A scan: same R_A values for both routes (x-axis is R_A, not absolute nu)
RA_SCAN = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0, 4.5]

_col = np.arange(N_ACT) % HALF
_row = np.arange(N_ACT) // HALF
zone_id = _col // ZONE_W
NB = []
for i in range(N_ACT):
    c, r = _col[i], _row[i]
    nb = []
    for dc, dr in [(-1,0),(1,0),(0,-1),(0,1)]:
        nc, nr = c+dc, r+dr
        if 0 <= nc < HALF and 0 <= nr < H:
            nb.append(nr*HALF+nc)
    NB.append(np.array(nb, dtype=int))

left_mask  = zone_id <= 1
right_mask = zone_id >= 2
top_mask   = _row < H // 2
bot_mask   = _row >= H // 2
d_A = np.array([1.0, 0.0])
d_B = np.array([0.0, 1.0])


def sg4_fn(F):
    means = [F[zone_id == z].mean(0) for z in range(N_ZONES)]
    pairs = [(i,j) for i in range(N_ZONES) for j in range(i+1,N_ZONES)]
    return float(np.mean([np.linalg.norm(means[i]-means[j]) for i,j in pairs]))


def run(seed, death_p, field_decay, field_alpha):
    rng = np.random.default_rng(seed)
    h   = rng.normal(0, 0.1, (N_ACT, HS))
    F   = np.zeros((N_ACT, HS))
    m   = np.zeros((N_ACT, HS))
    base = h.copy()
    streak = np.zeros(N_ACT, int)
    ttl = rng.geometric(p=max(death_p, 1e-6), size=N_ACT).astype(float)
    waves = []; total_collapses = 0

    for step in range(STEPS):
        in_phase2 = step >= SHIFT
        n = int(WR / WAVE_DUR)
        n += int(rng.random() < (WR / WAVE_DUR - n))
        for _ in range(n):
            if not in_phase2:
                z = int(rng.integers(N_ZONES))
                sign = 1.0 if z <= 1 else -1.0
                waves.append([z, WAVE_DUR, sign * d_A, True])
            else:
                top = rng.random() < 0.5
                sign = 1.0 if top else -1.0
                waves.append([0 if top else 1, WAVE_DUR, sign * d_B, False])

        pert = np.zeros(N_ACT, bool)
        for w in waves:
            if w[3]: pert |= (left_mask if w[0] <= 1 else right_mask)
            else:    pert |= (top_mask if w[0] == 0 else bot_mask)

        base += BASE_BETA * (h - base)
        for w in waves:
            mask = (left_mask if w[0] <= 1 else right_mask) if w[3] \
                   else (top_mask if w[0]==0 else bot_mask)
            h[mask] += 0.3 * np.array(w[2])

        m[pert] += ALPHA_MID * (h - base)[pert]
        m *= MID_DECAY
        streak[pert] = 0; streak[~pert] += 1
        ok = streak >= SS
        F[ok] += field_alpha * (m[ok] - F[ok])
        F *= field_decay

        dF = np.zeros_like(F)
        for i in range(N_ACT):
            if len(NB[i]):
                dF[i] = KAPPA * (F[NB[i]].mean(0) - F[i])
        F += dF

        if death_p > 0:
            ttl -= 1.0; ttl[pert] -= 1.0
            dead = np.where(ttl <= 0)[0]
            total_collapses += len(dead)
            for i in dead:
                j = NB[i][rng.integers(len(NB[i]))] if len(NB[i]) else i
                h[i] = (1-SEED_BETA)*rng.normal(0,0.1,HS) + SEED_BETA*F[j]
                m[i] = 0; streak[i] = 0
                ttl[i] = float(rng.geometric(p=death_p))

        waves = [[w[0],w[1]-1,w[2],w[3]] for w in waves if w[1]-1 > 0]

    return {"sg4": sg4_fn(F), "coll_rate": total_collapses / (N_ACT * STEPS)}


def _worker(args):
    return run(*args)


RESULTS_FILE = os.path.join(os.path.dirname(__file__), "results", "paper14_exp1_results.json")

if __name__ == "__main__":
    mp.freeze_support()

    if os.path.exists(RESULTS_FILE):
        results = json.load(open(RESULTS_FILE))
        print("Loaded cached results.")
    else:
        # Build all (seed, death_p, fd, fa) tuples
        # Key: "target_name,route,ra_scan"
        all_conditions = []   # (target_name, route_name, ra_val, fd, fa, nu)
        for tname, (ra_tgt, rb_tgt) in TARGETS.items():
            for route_name, fd, nu_cryst in [("R1", FD_R1, NU_CRYST_R1),
                                              ("R2", FD_R2, NU_CRYST_R2)]:
                fa = compute_fa(ra_tgt, rb_tgt, nu_cryst)
                fa = min(fa, 0.99)   # cap at 1 for safety
                for ra_val in RA_SCAN:
                    nu = ra_val * nu_cryst
                    nu = max(nu, 1e-5)
                    all_conditions.append((tname, route_name, ra_val, fd, fa, nu))

        all_args = [(seed, cond[5], cond[3], cond[4])
                    for cond in all_conditions
                    for seed in range(N_SEEDS)]

        print(f"Running {len(all_args)} simulations...")
        with mp.Pool(processes=min(len(all_args), mp.cpu_count())) as pool:
            raw = pool.map(_worker, all_args)

        results = {}
        idx = 0
        for cond in all_conditions:
            tname, route_name, ra_val, fd, fa, nu = cond
            key = f"{tname},{route_name},{ra_val:.4f}"
            results[key] = []
            for seed in range(N_SEEDS):
                results[key].append(raw[idx]); idx += 1

        os.makedirs(os.path.dirname(RESULTS_FILE), exist_ok=True)
        json.dump(results, open(RESULTS_FILE, "w"), indent=2)
        print("Saved results.")

    print(f"\nTimescale Nesting -- Iso-Ratio Invariance Test")
    print(f"P_calm = {P_CALM:.5f}")
    print(f"nu_cryst R1 = {NU_CRYST_R1:.6f}  (FD={FD_R1})")
    print(f"nu_cryst R2 = {NU_CRYST_R2:.6f}  (FD={FD_R2}, 2x R1)")
    print()

    for tname, (ra_tgt, rb_tgt) in TARGETS.items():
        fa_r1 = compute_fa(ra_tgt, rb_tgt, NU_CRYST_R1)
        fa_r2 = compute_fa(ra_tgt, rb_tgt, NU_CRYST_R2)
        print(f"\n{tname} (R_A={ra_tgt}, R_B={rb_tgt}):")
        print(f"  Route R1: FD={FD_R1}, FA={fa_r1:.4f}")
        print(f"  Route R2: FD={FD_R2}, FA={fa_r2:.4f}")
        print(f"  {'R_A':>5} | {'sg4 R1':>8} | {'sg4 R2':>8} | {'ratio R2/R1':>11}")
        print("  " + "-" * 40)
        for ra_val in RA_SCAN:
            k1 = f"{tname},R1,{ra_val:.4f}"
            k2 = f"{tname},R2,{ra_val:.4f}"
            if k1 in results and k2 in results:
                sg4_r1 = float(np.mean([v["sg4"] for v in results[k1]]))
                sg4_r2 = float(np.mean([v["sg4"] for v in results[k2]]))
                ratio = sg4_r2 / sg4_r1 if sg4_r1 > 0 else float('nan')
                print(f"  {ra_val:>5.2f} | {sg4_r1:>8.1f} | {sg4_r2:>8.1f} | {ratio:>11.3f}")

    # Summary: max deviation across all iso-ratio comparisons
    print("\nISO-RATIO INVARIANCE SUMMARY:")
    print(f"{'Target':>20} | {'R_A':>5} | {'sg4_R1':>8} | {'sg4_R2':>8} | {'|log ratio|':>11}")
    for tname, (ra_tgt, rb_tgt) in TARGETS.items():
        for ra_val in RA_SCAN:
            k1 = f"{tname},R1,{ra_val:.4f}"
            k2 = f"{tname},R2,{ra_val:.4f}"
            if k1 in results and k2 in results:
                sg4_r1 = float(np.mean([v["sg4"] for v in results[k1]]))
                sg4_r2 = float(np.mean([v["sg4"] for v in results[k2]]))
                if sg4_r1 > 5 and sg4_r2 > 5:
                    log_ratio = abs(math.log(sg4_r2 / sg4_r1))
                    print(f"  {tname:>18} | {ra_val:>5.2f} | {sg4_r1:>8.1f} | {sg4_r2:>8.1f} | {log_ratio:>11.3f}")
