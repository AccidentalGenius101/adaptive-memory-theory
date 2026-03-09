"""
paper7_exp5b_classifier_grid.py -- Experiment 5b for Paper 7

Logistic regression zone-prediction accuracy across the full 2D regime map.
Matches the (nu, kappa) grid of Experiment 3 so the figure can overlay
classifier accuracy directly on the sg4 heatmap.

Grid (same as Exp3):
  NU_VALS    = [0.0003, 0.001, 0.005, 0.015, 0.050]   (turnover rate)
  KAPPA_VALS = [0.000, 0.005, 0.020, 0.060, 0.150]   (field diffusion)

For each (nu, kappa) cell: 5 seeds x 5-fold CV logistic regression on final
fieldM vectors. 4-class (zone 0-3), balanced, 400 samples. Chance = 0.25.

Key format matches Exp3: f"{nu},{kp}"
Save to results/paper7_exp5b_results.json
Runtime: ~12 minutes (25 cells x 5 seeds = 125 runs).
Dependencies: numpy, sklearn.
"""
import numpy as np, json, os, multiprocessing as mp

W, H = 40, 20; HALF = W // 2
HS = 2; N_ZONES = 4; ZONE_W = HALF // N_ZONES
N_ACT = HALF * H; STEPS = 2000; SHIFT = 1000; N_SEEDS = 5

MID_DECAY = 0.99; FIELD_DECAY = 0.9997; BASE_BETA = 0.005
ALPHA_MID = 0.15; FIELD_ALPHA = 0.16; SEED_BETA = 0.25
WAVE_RATIO = 4.8; WAVE_DUR = 15; SS = 10

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

NU_VALS    = [0.0003, 0.001, 0.005, 0.015, 0.050]
KAPPA_VALS = [0.000, 0.005, 0.020, 0.060, 0.150]


def run(seed, death_p, kappa):
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score

    rng = np.random.default_rng(seed)
    h = rng.normal(0, 0.1, (N_ACT, HS))
    F = np.zeros((N_ACT, HS))
    m = np.zeros((N_ACT, HS))
    base = h.copy()
    streak = np.zeros(N_ACT, int)
    ttl = rng.geometric(p=max(death_p, 1e-6), size=N_ACT).astype(float)
    waves = []

    for step in range(STEPS):
        in_phase2 = step >= SHIFT

        n = int(WAVE_RATIO / WAVE_DUR)
        n += int(rng.random() < (WAVE_RATIO / WAVE_DUR - n))
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

        if kappa > 0:
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
                h[i] = (1-SEED_BETA)*rng.normal(0,0.1,HS) + SEED_BETA*F[j]
                m[i] = 0; streak[i] = 0
                ttl[i] = float(rng.geometric(p=death_p))

        waves = [[w[0],w[1]-1,w[2],w[3]] for w in waves if w[1]-1 > 0]

    clf = LogisticRegression(max_iter=1000, random_state=0)
    scores = cross_val_score(clf, F, zone_id, cv=5)
    return {"accuracy": float(scores.mean())}


def _worker(args):
    return run(*args)


RESULTS_FILE = os.path.join(os.path.dirname(__file__), "results", "paper7_exp5b_results.json")

if __name__ == "__main__":
    mp.freeze_support()

    if os.path.exists(RESULTS_FILE):
        results = json.load(open(RESULTS_FILE))
        print("Loaded cached results.")
    else:
        all_args = [(seed, nu, kp)
                    for nu in NU_VALS
                    for kp in KAPPA_VALS
                    for seed in range(N_SEEDS)]
        with mp.Pool(processes=min(len(all_args), mp.cpu_count())) as pool:
            raw = pool.map(_worker, all_args)
        results = {}
        idx = 0
        for nu in NU_VALS:
            for kp in KAPPA_VALS:
                key = f"{nu},{kp}"
                results[key] = []
                for seed in range(N_SEEDS):
                    results[key].append(raw[idx]); idx += 1
        os.makedirs(os.path.dirname(RESULTS_FILE), exist_ok=True)
        json.dump(results, open(RESULTS_FILE, "w"), indent=2)
        print("Saved results.")

    import numpy as np_
    print(f"\nClassifier accuracy: 5x5 grid ({N_SEEDS} seeds, 5-fold CV). Chance = 0.25.")
    print(f"{'nu':>8} | " + " | ".join(f"k={kp:.3f}" for kp in KAPPA_VALS))
    print("-" * 72)
    for nu in NU_VALS:
        row = []
        for kp in KAPPA_VALS:
            key = f"{nu},{kp}"
            acc = float(np_.mean([v["accuracy"] for v in results[key]]))
            row.append(f"{acc:.3f}")
        print(f"{nu:>8.4f} | " + " | ".join(row))
