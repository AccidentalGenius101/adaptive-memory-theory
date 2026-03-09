"""
paper8_exp3_wr_sweep.py -- Experiment 3 for Paper 8

Wave rate sweep: how does the optimal turnover rate shift when the
environmental update rate (WAVE_RATIO) changes?

Hypothesis:
  tau_input = WAVE_DUR / WAVE_RATIO  (average steps between consecutive
  wave events for a given site). Higher WR = more frequent perturbations.

  Copy-forward events only fire during rebirths triggered (partly) by
  wave perturbations. Higher WR -> more copy-forward events per step ->
  faster propagation of new field values.

  Prediction: nu* proportional to WR.
  (System can tolerate more turnover when input is more frequent,
  because each rebirth is more likely to receive an informative seeding.)

Protocol:
  For each WR in {1.2, 2.4, 4.8, 9.6}:
    Sweep DEATH_PS (9 values)
    5 seeds per (WR, nu) pair
  Find nu* = argmax sg4 for each WR.

Note: all WR values are < WAVE_DUR=15, so the Poisson launcher handles
them correctly (no saturation artifact). WR=9.6 launches 0-1 waves/step.

Base parameters: Paper 7 Exp1 values. Only WAVE_RATIO varies.
Grid: 40x20, N_ACT=400, HS=2, KAPPA=0.020, SS=10, FIELD_ALPHA=0.16.
Runtime: ~8 minutes.
Dependencies: numpy only.
"""
import numpy as np, json, os, multiprocessing as mp

W, H = 40, 20; HALF = W // 2
HS = 2; N_ZONES = 4; ZONE_W = HALF // N_ZONES
N_ACT = HALF * H; STEPS = 2000; SHIFT = 1000; N_SEEDS = 5

MID_DECAY = 0.99; FIELD_DECAY = 0.9997; BASE_BETA = 0.005
ALPHA_MID = 0.15; FIELD_ALPHA = 0.16; SEED_BETA = 0.25
SS = 10; KAPPA = 0.020; WAVE_DUR = 15

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

DEATH_PS = [0.0001, 0.0003, 0.001, 0.002, 0.005, 0.010, 0.020, 0.040, 0.080]
WR_VALS  = [1.2, 2.4, 4.8, 9.6]

def sg4(F):
    means = [F[zone_id == z].mean(0) for z in range(N_ZONES)]
    pairs = [(i,j) for i in range(N_ZONES) for j in range(i+1,N_ZONES)]
    return float(np.mean([np.linalg.norm(means[i]-means[j]) for i,j in pairs]))

def run(seed, death_p, wave_ratio):
    rng = np.random.default_rng(seed)
    h = rng.normal(0, 0.1, (N_ACT, HS))
    F = np.zeros((N_ACT, HS))
    m = np.zeros((N_ACT, HS))
    base = h.copy()
    streak = np.zeros(N_ACT, int)
    ttl = rng.geometric(p=max(death_p, 1e-6), size=N_ACT).astype(float)
    waves = []
    total_collapses = 0

    for step in range(STEPS):
        in_phase2 = step >= SHIFT

        # Poisson launcher (correct at all WR < WAVE_DUR)
        n = int(wave_ratio / WAVE_DUR)
        n += int(rng.random() < (wave_ratio / WAVE_DUR - n))
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
            mask = (left_mask if w[0] <= 1 else right_mask) if w[3] else (top_mask if w[0]==0 else bot_mask)
            h[mask] += 0.3 * np.array(w[2])
        m[pert] += ALPHA_MID * (h - base)[pert]
        m *= MID_DECAY
        streak[pert] = 0; streak[~pert] += 1
        ok = streak >= SS
        F[ok] += FIELD_ALPHA * (m[ok] - F[ok])
        F *= FIELD_DECAY

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

    return {
        "sg4":       sg4(F),
        "coll_rate": total_collapses / (N_ACT * STEPS),
    }


def _worker(args):
    return run(*args)


RESULTS_FILE = os.path.join(os.path.dirname(__file__), "results", "paper8_exp3_results.json")

if __name__ == "__main__":
    mp.freeze_support()

    if os.path.exists(RESULTS_FILE):
        results = json.load(open(RESULTS_FILE))
        print("Loaded cached results.")
    else:
        all_args = [(seed, dp, wr)
                    for wr in WR_VALS
                    for dp in DEATH_PS
                    for seed in range(N_SEEDS)]
        with mp.Pool(processes=min(len(all_args), mp.cpu_count())) as pool:
            raw = pool.map(_worker, all_args)
        results = {}
        idx = 0
        for wr in WR_VALS:
            for dp in DEATH_PS:
                key = f"{wr},{dp}"
                results[key] = []
                for seed in range(N_SEEDS):
                    results[key].append(raw[idx]); idx += 1
        os.makedirs(os.path.dirname(RESULTS_FILE), exist_ok=True)
        json.dump(results, open(RESULTS_FILE, "w"), indent=2)
        print("Saved results.")

    print(f"\nWave rate sweep: optimal nu shifts with WAVE_RATIO ({N_SEEDS} seeds)")
    print(f"Hypothesis: nu* proportional to WR\n")
    print(f"{'WR':>5} | {'nu*':>8} | {'sg4*':>8} | sg4 profile")
    print("-" * 70)
    for wr in WR_VALS:
        sg4_vals = []
        for dp in DEATH_PS:
            key = f"{wr},{dp}"
            sg4_vals.append(float(np.mean([v["sg4"] for v in results[key]])))
        best_idx = int(np.argmax(sg4_vals))
        nu_star = DEATH_PS[best_idx]
        sg4_star = sg4_vals[best_idx]
        profile = "  ".join(f"{v:5.0f}" for v in sg4_vals)
        print(f"{wr:>5.1f} | {nu_star:>8.4f} | {sg4_star:>8.1f} | {profile}")

    print()
    print("Prediction: nu* at WR=9.6 ~ 2x nu* at WR=4.8 (proportional shift)")
    print("  Higher WR -> more frequent perturbation -> system can sustain higher turnover")
