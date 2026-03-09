"""
paper16_exp1_trajectory.py -- Experiment 1 for Paper 16

Growth Trajectory and Saturation of sg4 vs FA

THE QUESTION:
  Paper 15 found sg4 ~ FA^0.43 in FA range [0.10, 0.50]. Two gaps:
  (1) Does the exponent represent true power-law or a local fit to a saturation curve?
  (2) What is the full growth trajectory? Is it sigmoidal? Is the dip timescale FA-independent?

ANALYTICAL PREDICTION:
  Consolidation-diffusion balance at zone boundaries gives:
      sg4(FA) approx C * FA / (FA + KAPPA_eff)
  where KAPPA_eff = KAPPA / P(streak >= SS) approx 0.020 / 0.175 approx 0.114.
  Empirical fit from Paper 15 gives KAPPA_eff approx 0.16.

  log-log local slope at FA = KAPPA_eff: 0.50  (matches Paper 15's 0.43 in center of range)
  Prediction: extending FA to 0.70-0.90 should show saturation (slope -> 0)

EXPERIMENTAL DESIGN:
  FA in {0.05, 0.08, 0.10, 0.15, 0.20, 0.30, 0.50, 0.70, 0.90}  (extends Paper 15 to 0.90)
  nu = 0.001, FD = 0.9997, KAPPA = 0.020 (all fixed)

  Two SHIFT conditions:
    SHIFT=0:    Pure Phase 2 from step 0. No Phase 1 disruption. Clean sigmoidal growth.
    SHIFT=1000: Standard (matches Paper 15). Phase 1 -> 2 transition creates dip at t=1500.

  STEPS=3000, record at 10 checkpoints: t=300, 600, ..., 3000
  Two metrics per checkpoint:
    sg4_inter: between-zone contrast (standard sg4 from Paper 15)
    sg4_intra: within-zone spatial variance (local structure)
              = mean over zones of mean |F[i] - zone_mean_F|
  If local structure forms before global: sg4_intra should rise before sg4_inter in Phase 2.

  9 FA * 2 shifts * 5 seeds = 90 runs. Runtime ~4-5 min.

Result key: "exp1,{shift},{fa:.4f},{seed}"
Value: dict with keys "sg4_inter_{t}" and "sg4_intra_{t}" for each t in CHECKPOINTS
"""
import numpy as np, json, os, multiprocessing as mp
import math

W, H = 40, 20; HALF = W // 2
HS = 2; N_ZONES = 4; ZONE_W = HALF // N_ZONES
N_ACT = HALF * H; N_SEEDS = 5

MID_DECAY = 0.99; BASE_BETA = 0.005
ALPHA_MID = 0.15; SEED_BETA = 0.25
SS = 10; KAPPA = 0.020; WR = 4.8; WAVE_DUR = 15

FD = 0.9997
NU = 0.001
NU_CRYST = abs(math.log(FD)) / math.log(2)
P_CALM = (1 - WR / (2 * WAVE_DUR)) ** (SS + WAVE_DUR - 1)

FA_VALS = [0.05, 0.08, 0.10, 0.15, 0.20, 0.30, 0.50, 0.70, 0.90]
SHIFT_VALS = [0, 1000]
STEPS = 3000
CHECKPOINTS = list(range(300, STEPS + 1, 300))  # 300, 600, ..., 3000 (10 points)

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

left_mask  = zone_id <= 1
right_mask = zone_id >= 2
top_mask   = _row < H // 2
bot_mask   = _row >= H // 2
d_A = np.array([1.0, 0.0])
d_B = np.array([0.0, 1.0])


def sg4_inter_fn(F):
    """Between-zone mean pairwise L2 distance (standard sg4)."""
    means = [F[zone_id == z].mean(0) for z in range(N_ZONES)]
    pairs = [(i, j) for i in range(N_ZONES) for j in range(i + 1, N_ZONES)]
    return float(np.mean([np.linalg.norm(means[i] - means[j]) for i, j in pairs]))


def sg4_intra_fn(F):
    """Within-zone mean absolute deviation -- local structure metric."""
    intra = []
    for z in range(N_ZONES):
        zf = F[zone_id == z]
        zm = zf.mean(0)
        intra.append(float(np.mean(np.linalg.norm(zf - zm, axis=1))))
    return float(np.mean(intra))


def run(seed, death_p, field_decay, field_alpha, shift):
    """Run simulation with given SHIFT, record at all CHECKPOINTS."""
    rng = np.random.default_rng(seed)
    h   = rng.normal(0, 0.1, (N_ACT, HS))
    F   = np.zeros((N_ACT, HS))
    m   = np.zeros((N_ACT, HS))
    base = h.copy()
    streak = np.zeros(N_ACT, int)
    ttl = rng.geometric(p=max(death_p, 1e-6), size=N_ACT).astype(float)
    waves = []
    cp_set = set(CHECKPOINTS)
    results = {}

    for step in range(STEPS):
        in_phase2 = step >= shift
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
                   else (top_mask if w[0] == 0 else bot_mask)
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
            for i in dead:
                j = NB[i][rng.integers(len(NB[i]))] if len(NB[i]) else i
                h[i] = (1 - SEED_BETA) * rng.normal(0, 0.1, HS) + SEED_BETA * F[j]
                m[i] = 0; streak[i] = 0
                ttl[i] = float(rng.geometric(p=death_p))

        waves = [[w[0], w[1] - 1, w[2], w[3]] for w in waves if w[1] - 1 > 0]

        if (step + 1) in cp_set:
            t = step + 1
            results[f"sg4_inter_{t}"] = sg4_inter_fn(F)
            results[f"sg4_intra_{t}"] = sg4_intra_fn(F)

    return results


def _worker(args):
    return run(*args)


RESULTS_FILE = os.path.join(os.path.dirname(__file__), "results", "paper16_exp1_results.json")


if __name__ == "__main__":
    mp.freeze_support()

    if os.path.exists(RESULTS_FILE):
        results = json.load(open(RESULTS_FILE))
        print("Loaded cached results.")
    else:
        all_conditions = []
        for shift in SHIFT_VALS:
            for fa in FA_VALS:
                for seed in range(N_SEEDS):
                    all_conditions.append((shift, fa, seed))

        all_args = [(cond[2], NU, FD, cond[1], cond[0]) for cond in all_conditions]
        print(f"Running {len(all_args)} simulations (Paper 16 Exp1)...")
        with mp.Pool(processes=min(len(all_args), mp.cpu_count())) as pool:
            raw = pool.map(_worker, all_args)

        results = {}
        for i, cond in enumerate(all_conditions):
            shift, fa, seed = cond
            key = f"exp1,{shift},{fa:.4f},{seed}"
            results[key] = raw[i]

        os.makedirs(os.path.dirname(RESULTS_FILE), exist_ok=True)
        json.dump(results, open(RESULTS_FILE, "w"), indent=2)
        print("Saved results.")

    print(f"\nP_calm = {P_CALM:.5f}")
    print(f"nu_cryst = {NU_CRYST:.6f}  (FD={FD})")
    print(f"nu = {NU}, KAPPA = {KAPPA}")
    print(f"KAPPA_eff (predicted) = {KAPPA / 0.175:.4f}")
    print()

    # Summary: sg4_inter at t=3000 (final) for both SHIFT conditions
    print(f"{'SHIFT':>7} | {'FA':>6} | {'R_B':>5} | {'sg4_inter@3000':>15} | {'sg4_intra@3000':>15}")
    print("-" * 60)
    fd_term = FD ** (1.0 / NU)
    for shift in SHIFT_VALS:
        for fa in FA_VALS:
            rb = P_CALM * fa * fd_term / NU
            inter_vals = [results[f"exp1,{shift},{fa:.4f},{s}"].get(f"sg4_inter_{STEPS}", float("nan"))
                          for s in range(N_SEEDS) if f"exp1,{shift},{fa:.4f},{s}" in results]
            intra_vals = [results[f"exp1,{shift},{fa:.4f},{s}"].get(f"sg4_intra_{STEPS}", float("nan"))
                          for s in range(N_SEEDS) if f"exp1,{shift},{fa:.4f},{s}" in results]
            mean_inter = float(np.mean(inter_vals)) if inter_vals else float("nan")
            mean_intra = float(np.mean(intra_vals)) if intra_vals else float("nan")
            flag = "  [OUT]" if rb < 1.0 else ""
            print(f"  {shift:>5} | {fa:>6.3f} | {rb:>5.2f} | {mean_inter:>15.2f} | {mean_intra:>15.4f}{flag}")
        print()

    # Power law fit for SHIFT=0 at t=3000 (inside-window, FA >= 0.10)
    print("POWER LAW FIT (SHIFT=0, t=3000, R_B >= 1):")
    log_fa = []; log_sg4 = []
    for fa in FA_VALS:
        rb = P_CALM * fa * fd_term / NU
        if rb < 1.0:
            continue
        vals = [results[f"exp1,0,{fa:.4f},{s}"].get(f"sg4_inter_{STEPS}", float("nan"))
                for s in range(N_SEEDS) if f"exp1,0,{fa:.4f},{s}" in results]
        if vals:
            log_fa.append(math.log(fa)); log_sg4.append(math.log(float(np.mean(vals))))
    if len(log_fa) >= 3:
        coeffs = np.polyfit(log_fa, log_sg4, 1)
        log_sg4_pred = np.polyval(coeffs, np.array(log_fa))
        ss_res = sum((a - b) ** 2 for a, b in zip(log_sg4, log_sg4_pred.tolist()))
        ss_tot = sum((a - float(np.mean(log_sg4))) ** 2 for a in log_sg4)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
        print(f"  sg4 ~ FA^{coeffs[0]:.4f}  (R^2={r2:.4f})")

    # Saturation fit: sg4 = C * FA / (FA + K_eff)  (nonlinear)
    print("\nSATURATION FIT (SHIFT=0, t=3000): sg4 = C * FA / (FA + K_eff)")
    fa_data = []; sg4_data = []
    for fa in FA_VALS:
        rb = P_CALM * fa * fd_term / NU
        if rb < 1.0:
            continue
        vals = [results[f"exp1,0,{fa:.4f},{s}"].get(f"sg4_inter_{STEPS}", float("nan"))
                for s in range(N_SEEDS) if f"exp1,0,{fa:.4f},{s}" in results]
        if vals:
            fa_data.append(fa); sg4_data.append(float(np.mean(vals)))
    # Grid search for best (C, K_eff)
    best_r2 = -1e9; best_C = 0; best_K = 0
    for K in np.linspace(0.01, 0.50, 100):
        # Linear regression for C at fixed K
        x = [fa / (fa + K) for fa in fa_data]
        if not x:
            continue
        C = float(np.dot(x, sg4_data) / np.dot(x, x))
        pred = [C * xx for xx in x]
        ss_res = sum((p - d) ** 2 for p, d in zip(pred, sg4_data))
        ss_tot = sum((d - float(np.mean(sg4_data))) ** 2 for d in sg4_data)
        r2 = 1 - ss_res / ss_tot
        if r2 > best_r2:
            best_r2 = r2; best_C = C; best_K = K
    print(f"  Best fit: C={best_C:.2f}, K_eff={best_K:.4f}, R^2={best_r2:.4f}")
    print(f"  Predicted K_eff from theory: {KAPPA / 0.175:.4f}")

    # Dip timescale for SHIFT=1000 across FA values
    print("\nDIP TIMESCALE (SHIFT=1000): time of minimum sg4_inter in Phase 2")
    phase2_checkpoints = [t for t in CHECKPOINTS if t >= 1000]
    for fa in FA_VALS:
        rb = P_CALM * fa * fd_term / NU
        if rb < 1.0:
            continue
        traj = []
        for t in phase2_checkpoints:
            vals = [results[f"exp1,1000,{fa:.4f},{s}"].get(f"sg4_inter_{t}", float("nan"))
                    for s in range(N_SEEDS) if f"exp1,1000,{fa:.4f},{s}" in results]
            traj.append(float(np.mean(vals)) if vals else float("nan"))
        if not any(math.isnan(v) for v in traj):
            t_min = phase2_checkpoints[int(np.argmin(traj))]
            dip_depth = min(traj) / traj[0] if traj[0] > 0 else float("nan")
            print(f"  FA={fa:.3f}: dip at t={t_min}, depth={dip_depth:.3f}")
