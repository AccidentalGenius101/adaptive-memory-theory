"""
paper16_exp2_kappa.py -- Experiment 2 for Paper 16

KAPPA Sweep: Consolidation-Diffusion Crossover

THE PREDICTION:
  sg4(FA) approx C * FA / (FA + KAPPA_eff)
  where KAPPA_eff = KAPPA / P(streak >= SS)

  As KAPPA increases, KAPPA_eff increases, and the crossover FA shifts right.
  In a fixed FA range [0.10, 0.50], the effective power-law exponent alpha(KAPPA)
  should increase monotonically with KAPPA.

  Analytical prediction for the local log-log slope at FA_mid in the tested range:
    alpha_pred(KAPPA) = KAPPA_eff / (FA_mid + KAPPA_eff)
    where KAPPA_eff = KAPPA / 0.175 and FA_mid is the geometric mean of the tested range.

DESIGN:
  KAPPA in {0.005, 0.010, 0.020, 0.040}  (0.020 = Paper 15 reference)
  FA in {0.05, 0.10, 0.20, 0.40, 0.80}  (wide range to span both linear and saturation regimes)
  nu = 0.001, FD = 0.9997, SHIFT = 0, STEPS = 2000
  5 seeds per condition.
  4 KAPPA * 5 FA * 5 seeds = 100 runs. Runtime ~4 min.

Result key: "exp2,{kappa:.4f},{fa:.4f},{seed}"
Value: {"sg4_inter_2000": float}
"""
import numpy as np, json, os, multiprocessing as mp
import math

W, H = 40, 20; HALF = W // 2
HS = 2; N_ZONES = 4; ZONE_W = HALF // N_ZONES
N_ACT = HALF * H; STEPS = 2000; SHIFT = 0; N_SEEDS = 5

MID_DECAY = 0.99; BASE_BETA = 0.005
ALPHA_MID = 0.15; SEED_BETA = 0.25
SS = 10; WR = 4.8; WAVE_DUR = 15

FD = 0.9997
NU = 0.001
NU_CRYST = abs(math.log(FD)) / math.log(2)
P_CALM = (1 - WR / (2 * WAVE_DUR)) ** (SS + WAVE_DUR - 1)
P_CONSOL_EST = 0.84 ** 10  # P(not pert for 10 steps) ~ P(streak >= SS)

KAPPA_VALS = [0.005, 0.010, 0.020, 0.040]
FA_VALS     = [0.05, 0.10, 0.20, 0.40, 0.80]

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
    means = [F[zone_id == z].mean(0) for z in range(N_ZONES)]
    pairs = [(i, j) for i in range(N_ZONES) for j in range(i + 1, N_ZONES)]
    return float(np.mean([np.linalg.norm(means[i] - means[j]) for i, j in pairs]))


def run(seed, death_p, field_decay, field_alpha, kappa):
    """Run with variable KAPPA, SHIFT=0, pure Phase 2 from start."""
    rng = np.random.default_rng(seed)
    h   = rng.normal(0, 0.1, (N_ACT, HS))
    F   = np.zeros((N_ACT, HS))
    m   = np.zeros((N_ACT, HS))
    base = h.copy()
    streak = np.zeros(N_ACT, int)
    ttl = rng.geometric(p=max(death_p, 1e-6), size=N_ACT).astype(float)
    waves = []

    for step in range(STEPS):
        # SHIFT=0: Phase 2 from step 0
        n = int(WR / WAVE_DUR)
        n += int(rng.random() < (WR / WAVE_DUR - n))
        for _ in range(n):
            top = rng.random() < 0.5
            sign = 1.0 if top else -1.0
            waves.append([0 if top else 1, WAVE_DUR, sign * d_B, False])

        pert = np.zeros(N_ACT, bool)
        for w in waves:
            pert |= (top_mask if w[0] == 0 else bot_mask)

        base += BASE_BETA * (h - base)
        for w in waves:
            mask = top_mask if w[0] == 0 else bot_mask
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
                dF[i] = kappa * (F[NB[i]].mean(0) - F[i])
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

    return {"sg4_inter_2000": sg4_inter_fn(F)}


def _worker(args):
    return run(*args)


RESULTS_FILE = os.path.join(os.path.dirname(__file__), "results", "paper16_exp2_results.json")


if __name__ == "__main__":
    mp.freeze_support()

    if os.path.exists(RESULTS_FILE):
        results = json.load(open(RESULTS_FILE))
        print("Loaded cached results.")
    else:
        all_conditions = []
        for kappa in KAPPA_VALS:
            for fa in FA_VALS:
                for seed in range(N_SEEDS):
                    all_conditions.append((kappa, fa, seed))

        all_args = [(cond[2], NU, FD, cond[1], cond[0]) for cond in all_conditions]
        print(f"Running {len(all_args)} simulations (Paper 16 Exp2)...")
        with mp.Pool(processes=min(len(all_args), mp.cpu_count())) as pool:
            raw = pool.map(_worker, all_args)

        results = {}
        for i, cond in enumerate(all_conditions):
            kappa, fa, seed = cond
            key = f"exp2,{kappa:.4f},{fa:.4f},{seed}"
            results[key] = raw[i]

        os.makedirs(os.path.dirname(RESULTS_FILE), exist_ok=True)
        json.dump(results, open(RESULTS_FILE, "w"), indent=2)
        print("Saved results.")

    print(f"\nP_consol_est = {P_CONSOL_EST:.4f}")
    print(f"Predicted KAPPA_eff values: {[round(k / P_CONSOL_EST, 4) for k in KAPPA_VALS]}")
    print()

    fd_term = FD ** (1.0 / NU)

    # For each KAPPA: print sg4 vs FA table and fit power law + saturation formula
    for kappa in KAPPA_VALS:
        kappa_eff_pred = kappa / P_CONSOL_EST
        print(f"KAPPA={kappa:.3f} (predicted KAPPA_eff={kappa_eff_pred:.4f}):")
        print(f"  {'FA':>5} | {'R_B':>5} | {'sg4_final':>10}")
        print("  " + "-" * 27)
        fa_data = []; sg4_data = []
        for fa in FA_VALS:
            rb = P_CALM * fa * fd_term / NU
            vals = [results[f"exp2,{kappa:.4f},{fa:.4f},{s}"]["sg4_inter_2000"]
                    for s in range(N_SEEDS) if f"exp2,{kappa:.4f},{fa:.4f},{s}" in results]
            mean_sg4 = float(np.mean(vals)) if vals else float("nan")
            flag = " [OUT]" if rb < 1.0 else ""
            print(f"  {fa:>5.2f} | {rb:>5.2f} | {mean_sg4:>10.2f}{flag}")
            if rb >= 1.0 and not math.isnan(mean_sg4) and mean_sg4 > 0:
                fa_data.append(fa); sg4_data.append(mean_sg4)

        if len(fa_data) >= 3:
            # Power law fit
            log_fa = [math.log(f) for f in fa_data]
            log_sg4 = [math.log(s) for s in sg4_data]
            coeffs = np.polyfit(log_fa, log_sg4, 1)
            log_pred = np.polyval(coeffs, np.array(log_fa))
            ss_res = sum((a - b) ** 2 for a, b in zip(log_sg4, log_pred.tolist()))
            ss_tot = sum((a - float(np.mean(log_sg4))) ** 2 for a in log_sg4)
            r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
            print(f"  Power law:  sg4 ~ FA^{coeffs[0]:.4f}  (R^2={r2:.4f})")

            # Saturation fit
            best_r2 = -1e9; best_C = 0; best_K = 0
            for K in np.linspace(0.005, 1.0, 200):
                x = [fa / (fa + K) for fa in fa_data]
                C = float(np.dot(x, sg4_data) / np.dot(x, x)) if sum(v*v for v in x) > 0 else 0
                pred = [C * xx for xx in x]
                ss_res_s = sum((p - d) ** 2 for p, d in zip(pred, sg4_data))
                ss_tot_s = sum((d - float(np.mean(sg4_data))) ** 2 for d in sg4_data)
                r2_s = 1 - ss_res_s / ss_tot_s if ss_tot_s > 0 else float("nan")
                if r2_s > best_r2:
                    best_r2 = r2_s; best_C = C; best_K = K
            print(f"  Saturation: sg4 = {best_C:.1f} * FA / (FA + {best_K:.4f}), R^2={best_r2:.4f}")
            print(f"  Predicted KAPPA_eff = {kappa_eff_pred:.4f},  Fitted KAPPA_eff = {best_K:.4f}")
        print()

    # Summary table: alpha vs KAPPA
    print("EXPONENT SUMMARY (Power law alpha vs KAPPA):")
    print(f"  {'KAPPA':>7} | {'KAPPA_eff_pred':>14} | {'alpha':>7} | {'R^2':>6}")
    print("  " + "-" * 45)
    for kappa in KAPPA_VALS:
        kappa_eff_pred = kappa / P_CONSOL_EST
        fa_data = []; sg4_data = []
        for fa in FA_VALS:
            rb = P_CALM * fa * fd_term / NU
            vals = [results[f"exp2,{kappa:.4f},{fa:.4f},{s}"]["sg4_inter_2000"]
                    for s in range(N_SEEDS) if f"exp2,{kappa:.4f},{fa:.4f},{s}" in results]
            if rb >= 1.0 and vals:
                mean_sg4 = float(np.mean(vals))
                if mean_sg4 > 0:
                    fa_data.append(fa); sg4_data.append(mean_sg4)
        if len(fa_data) >= 2:
            log_fa = [math.log(f) for f in fa_data]
            log_sg4 = [math.log(s) for s in sg4_data]
            coeffs = np.polyfit(log_fa, log_sg4, 1)
            log_pred = np.polyval(coeffs, np.array(log_fa))
            ss_res = sum((a - b) ** 2 for a, b in zip(log_sg4, log_pred.tolist()))
            ss_tot = sum((a - float(np.mean(log_sg4))) ** 2 for a in log_sg4)
            r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
            print(f"  {kappa:>7.4f} | {kappa_eff_pred:>14.4f} | {coeffs[0]:>7.4f} | {r2:>6.4f}")
