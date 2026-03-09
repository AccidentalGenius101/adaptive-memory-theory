"""
paper15_amplitude_law.py -- Experiments 1-2 for Paper 15

Internal Amplitude Law: What Determines sg4 Inside the Adaptive Regime?

THE QUESTION:
  Papers 11-14 characterize when the adaptive phase exists (boundary geometry).
  This paper asks: once inside the viable region, what governs sg4 AMPLITUDE?

THE HYPOTHESIS (from Paper 14 residuals):
  FIELD_ALPHA has two roles:
    (1) Sets the upper consolidation boundary nu_max [captured by R_B]
    (2) Controls differentiation amplitude per consolidation event [NOT captured]
  Role (2) suggests an internal amplitude law of the form:
      sg4 ~ FA^alpha   (power law in FA)
  And a rate-lifetime scaling:
      sg4 ~ FA / nu   (differentiation speed x lifetime)

EXPERIMENTAL DESIGN:

  Exp 1: FA sweep at fixed nu, FD
    - nu = 0.001 (R_A=2.31, R_B=1.13 to 5.64 across FA sweep)
    - FD = 0.9997, nu_cryst = 4.33e-4
    - FA in {0.05, 0.08, 0.10, 0.12, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50}
    - Records sg4 at 4 checkpoints in Phase 2: t=1250, 1500, 1750, 2000
    - Measures: sg4 scaling with FA, growth trajectory

  Exp 2: Joint nu x FA grid at fixed FD=0.9997
    - nu in {0.0005, 0.001, 0.002}
    - FA in {0.10, 0.15, 0.20, 0.30, 0.50}
    - 15 conditions (some outside window -- flagged in analysis)
    - Tests: sg4 ~ FA / nu (rate-lifetime hypothesis)

  Grid: W=40, H=20, N_ACT=400, HS=2, STEPS=2000, SHIFT=1000
  5 seeds per condition.
  Total: 10*5 + 15*5 = 125 runs. Runtime ~5 min.

Key boundary: R_B = nu_max / nu = P_calm * FA * FD^(1/nu) / nu
  At nu=0.001: FD^(1/nu) = 0.9997^1000 = 0.7408
    nu_max = 0.01523 * FA * 0.7408 = 0.01129 * FA
    R_B = 11.29 * FA
  So FA=0.10 -> R_B=1.13 (inside), FA=0.05 -> R_B=0.56 (outside)

Result key format:
  Exp1: "exp1,{fa:.4f},{seed}"  -> {sg4_1250, sg4_1500, sg4_1750, sg4_2000}
  Exp2: "exp2,{nu:.5f},{fa:.4f},{seed}" -> {sg4_1250, sg4_1500, sg4_1750, sg4_2000}
"""
import numpy as np, json, os, multiprocessing as mp
import math

W, H = 40, 20; HALF = W // 2
HS = 2; N_ZONES = 4; ZONE_W = HALF // N_ZONES
N_ACT = HALF * H; STEPS = 2000; SHIFT = 1000; N_SEEDS = 5

MID_DECAY = 0.99; BASE_BETA = 0.005
ALPHA_MID = 0.15; SEED_BETA = 0.25
SS = 10; KAPPA = 0.020; WR = 4.8; WAVE_DUR = 15

FD = 0.9997
NU_CRYST = abs(math.log(FD)) / math.log(2)
P_CALM = (1 - WR / (2 * WAVE_DUR)) ** (SS + WAVE_DUR - 1)

# Exp 1: FA sweep at fixed nu
EXP1_NU = 0.001
EXP1_FA_VALS = [0.05, 0.08, 0.10, 0.12, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]

# Exp 2: joint nu x FA grid
EXP2_NU_VALS = [0.0005, 0.001, 0.002]
EXP2_FA_VALS = [0.10, 0.15, 0.20, 0.30, 0.50]

# Checkpoints in Phase 2 (after SHIFT) to track growth rate
CHECKPOINTS = [SHIFT + 250, SHIFT + 500, SHIFT + 750, STEPS]  # 1250, 1500, 1750, 2000

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
    """Run simulation, recording sg4 at each checkpoint in Phase 2."""
    rng = np.random.default_rng(seed)
    h   = rng.normal(0, 0.1, (N_ACT, HS))
    F   = np.zeros((N_ACT, HS))
    m   = np.zeros((N_ACT, HS))
    base = h.copy()
    streak = np.zeros(N_ACT, int)
    ttl = rng.geometric(p=max(death_p, 1e-6), size=N_ACT).astype(float)
    waves = []

    checkpoint_sg4 = {}
    cp_set = set(CHECKPOINTS)

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
            for i in dead:
                j = NB[i][rng.integers(len(NB[i]))] if len(NB[i]) else i
                h[i] = (1-SEED_BETA)*rng.normal(0,0.1,HS) + SEED_BETA*F[j]
                m[i] = 0; streak[i] = 0
                ttl[i] = float(rng.geometric(p=death_p))

        waves = [[w[0],w[1]-1,w[2],w[3]] for w in waves if w[1]-1 > 0]

        # Record at checkpoints
        if (step + 1) in cp_set:
            checkpoint_sg4[step + 1] = sg4_fn(F)

    return {f"sg4_{t}": checkpoint_sg4.get(t, float("nan")) for t in CHECKPOINTS}


def _worker(args):
    return run(*args)


RESULTS_FILE = os.path.join(os.path.dirname(__file__), "results", "paper15_results.json")


if __name__ == "__main__":
    mp.freeze_support()

    if os.path.exists(RESULTS_FILE):
        results = json.load(open(RESULTS_FILE))
        print("Loaded cached results.")
    else:
        all_conditions = []

        # Exp 1: FA sweep at nu=0.001
        for fa in EXP1_FA_VALS:
            for seed in range(N_SEEDS):
                all_conditions.append(("exp1", EXP1_NU, fa, seed))

        # Exp 2: nu x FA grid
        for nu in EXP2_NU_VALS:
            for fa in EXP2_FA_VALS:
                for seed in range(N_SEEDS):
                    all_conditions.append(("exp2", nu, fa, seed))

        all_args = [(cond[3], cond[1], FD, cond[2]) for cond in all_conditions]

        print(f"Running {len(all_args)} simulations (Paper 15)...")
        with mp.Pool(processes=min(len(all_args), mp.cpu_count())) as pool:
            raw = pool.map(_worker, all_args)

        results = {}
        for i, cond in enumerate(all_conditions):
            exp_name, nu, fa, seed = cond
            if exp_name == "exp1":
                key = f"exp1,{fa:.4f},{seed}"
            else:
                key = f"exp2,{nu:.5f},{fa:.4f},{seed}"
            results[key] = raw[i]

        os.makedirs(os.path.dirname(RESULTS_FILE), exist_ok=True)
        json.dump(results, open(RESULTS_FILE, "w"), indent=2)
        print("Saved results.")

    print(f"\nP_calm = {P_CALM:.5f}")
    print(f"nu_cryst = {NU_CRYST:.6f}  (FD={FD})")
    print()

    # --- Exp 1 Summary ---
    print(f"EXP 1: FA sweep at nu={EXP1_NU:.4f} (R_A={EXP1_NU/NU_CRYST:.2f})")
    fd_term = FD ** (1.0 / EXP1_NU)
    print(f"  FD^(1/nu) = {fd_term:.5f}")
    header = f"{'FA':>6} | {'R_B':>5} | {'sg4@1250':>10} | {'sg4@1500':>10} | {'sg4@1750':>10} | {'sg4@2000':>10}"
    print("  " + header)
    print("  " + "-"*len(header))
    for fa in EXP1_FA_VALS:
        nu_max = P_CALM * fa * fd_term
        rb = nu_max / EXP1_NU
        vals = {t: [] for t in CHECKPOINTS}
        for seed in range(N_SEEDS):
            rec = results.get(f"exp1,{fa:.4f},{seed}", {})
            for t in CHECKPOINTS:
                v = rec.get(f"sg4_{t}", float("nan"))
                if not math.isnan(v):
                    vals[t].append(v)
        means = [float(np.mean(vals[t])) if vals[t] else float("nan") for t in CHECKPOINTS]
        flag = "  [OUTSIDE]" if rb < 1.0 else ""
        print(f"  {fa:>6.3f} | {rb:>5.2f} | {means[0]:>10.4f} | {means[1]:>10.4f} | {means[2]:>10.4f} | {means[3]:>10.4f}{flag}")

    print()

    # --- Exp 2 Summary ---
    print("EXP 2: nu x FA grid (FD=0.9997)")
    print(f"  {'nu':>7} | {'FA':>5} | {'R_A':>5} | {'R_B':>5} | {'sg4_final':>10}")
    print("  " + "-"*45)
    for nu in EXP2_NU_VALS:
        fd_term = FD ** (1.0 / nu)
        ra = nu / NU_CRYST
        for fa in EXP2_FA_VALS:
            nu_max = P_CALM * fa * fd_term
            rb = nu_max / nu
            vals = []
            for seed in range(N_SEEDS):
                rec = results.get(f"exp2,{nu:.5f},{fa:.4f},{seed}", {})
                v = rec.get(f"sg4_{STEPS}", float("nan"))
                if not math.isnan(v):
                    vals.append(v)
            mean_sg4 = float(np.mean(vals)) if vals else float("nan")
            flag = " [OUT]" if rb < 1.0 else ""
            print(f"  {nu:>7.4f} | {fa:>5.2f} | {ra:>5.2f} | {rb:>5.2f} | {mean_sg4:>10.4f}{flag}")

    # --- Power law fit for Exp 1 (inside-window only) ---
    print("\nPOWER LAW FIT (Exp 1, inside-window points R_B >= 1):")
    log_fa = []
    log_sg4 = []
    for fa in EXP1_FA_VALS:
        fd_term = FD ** (1.0 / EXP1_NU)
        rb = P_CALM * fa * fd_term / EXP1_NU
        if rb < 1.0:
            continue
        vals = []
        for seed in range(N_SEEDS):
            rec = results.get(f"exp1,{fa:.4f},{seed}", {})
            v = rec.get(f"sg4_{STEPS}", float("nan"))
            if not math.isnan(v) and v > 0:
                vals.append(v)
        if vals:
            log_fa.append(math.log(fa))
            log_sg4.append(math.log(float(np.mean(vals))))

    if len(log_fa) >= 3:
        # Linear fit in log-log
        coeffs = np.polyfit(log_fa, log_sg4, 1)
        alpha_exp = coeffs[0]
        print(f"  sg4 ~ FA^{alpha_exp:.3f}  (R^2 from {len(log_fa)} points)")

    # --- Rate-lifetime test for Exp 2 ---
    print("\nRATE-LIFETIME TEST (Exp 2): does sg4 ~ FA / nu?")
    print(f"  {'nu':>7} | {'FA':>5} | {'FA/nu':>8} | {'sg4':>10} | {'sg4/(FA/nu)':>12}")
    print("  " + "-"*55)
    for nu in EXP2_NU_VALS:
        fd_term = FD ** (1.0 / nu)
        for fa in EXP2_FA_VALS:
            rb = P_CALM * fa * fd_term / nu
            if rb < 1.0:
                continue
            vals = []
            for seed in range(N_SEEDS):
                rec = results.get(f"exp2,{nu:.5f},{fa:.4f},{seed}", {})
                v = rec.get(f"sg4_{STEPS}", float("nan"))
                if not math.isnan(v):
                    vals.append(v)
            if vals:
                mean_sg4 = float(np.mean(vals))
                ratio_var = fa / nu
                normalized = mean_sg4 / ratio_var if ratio_var > 0 else float("nan")
                print(f"  {nu:>7.4f} | {fa:>5.2f} | {ratio_var:>8.2f} | {mean_sg4:>10.4f} | {normalized:>12.6f}")
