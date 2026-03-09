"""
paper7_exp5_classifier.py -- Experiment 5 for Paper 7

Orthogonal metric validation: logistic regression zone-prediction accuracy.

Addresses the reviewer concern that all claims rely on sg4 (a single metric).
We train a linear classifier to predict zone membership (0-3) from fieldM vectors.
If zone structure is real, accuracy should exceed chance (0.25) and track the
same regime boundaries identified by sg4.

Protocol: same turnover sweep as Exp1 (DEATH_PS x 5 seeds).
After each run, collect N_ACT site fieldM vectors with zone labels.
Fit LogisticRegression, report cross-validated accuracy.

Grid: 40x20, active half=20x20=400 sites, HS=2, 4 zones, 5 seeds.
Chance level: 0.25 (4 balanced classes of 100 sites each).
Dependencies: numpy, scikit-learn.
Runtime: ~5 minutes.
"""
import numpy as np, json, os, multiprocessing as mp

W, H = 40, 20; HALF = W // 2
HS = 2; N_ZONES = 4; ZONE_W = HALF // N_ZONES
N_ACT = HALF * H; STEPS = 2000; SHIFT = 1000; N_SEEDS = 5

MID_DECAY = 0.99; FIELD_DECAY = 0.9997; BASE_BETA = 0.005
ALPHA_MID = 0.15; FIELD_ALPHA = 0.16; SEED_BETA = 0.25
KAPPA = 0.020; WAVE_RATIO = 4.8; WAVE_DUR = 15; SS = 10

_col = np.arange(N_ACT) % HALF
_row = np.arange(N_ACT) // HALF
zone_id = _col // ZONE_W   # 0-3

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

def run(seed, death_p):
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
    total_collapses = 0

    for step in range(STEPS):
        in_phase2 = step >= SHIFT

        # Launch waves
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

        # Build perturbation mask
        pert = np.zeros(N_ACT, bool)
        for w in waves:
            if w[3]:
                pert |= (left_mask if w[0] <= 1 else right_mask)
            else:
                pert |= (top_mask if w[0] == 0 else bot_mask)

        # VCSM update
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

        # Field diffusion
        dF = np.zeros_like(F)
        for i in range(N_ACT):
            if len(NB[i]):
                dF[i] = KAPPA * (F[NB[i]].mean(0) - F[i])
        F += dF

        # Turnover
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

    # Logistic regression: predict zone from fieldM
    # 4 balanced classes (100 sites each) -> chance = 0.25
    clf = LogisticRegression(max_iter=1000, random_state=0)
    scores = cross_val_score(clf, F, zone_id, cv=5)
    accuracy = float(scores.mean())

    coll_rate = total_collapses / (N_ACT * STEPS)
    if coll_rate < 0.001:    regime = "frozen"
    elif coll_rate > 0.015:  regime = "turnover-dominated"
    else:                    regime = "adaptive"

    return {
        "accuracy":   accuracy,
        "coll_rate":  coll_rate,
        "regime":     regime,
    }


def _worker(args):
    return run(*args)

DEATH_PS = [0.0001, 0.0003, 0.001, 0.002, 0.005, 0.010, 0.020, 0.040, 0.080]
RESULTS_FILE = os.path.join(os.path.dirname(__file__), "results", "paper7_exp5_results.json")

if __name__ == "__main__":
    mp.freeze_support()

    if os.path.exists(RESULTS_FILE):
        results = json.load(open(RESULTS_FILE))
        print("Loaded cached results.")
    else:
        all_args = [(seed, dp) for dp in DEATH_PS for seed in range(N_SEEDS)]
        with mp.Pool(processes=min(len(all_args), mp.cpu_count())) as pool:
            raw = pool.map(_worker, all_args)
        results = {}
        idx = 0
        for dp in DEATH_PS:
            key = str(dp)
            results[key] = []
            for seed in range(N_SEEDS):
                results[key].append(raw[idx]); idx += 1
        os.makedirs(os.path.dirname(RESULTS_FILE), exist_ok=True)
        json.dump(results, open(RESULTS_FILE, "w"), indent=2)
        print("Saved results.")

    print(f"\nClassifier accuracy: zone prediction from fieldM ({N_SEEDS} seeds)")
    print(f"Chance level: 0.25 (4 balanced zones of 100 sites)")
    print(f"{'death_p':>9} | {'coll/site':>9} | {'accuracy':>9} | regime")
    print("-" * 52)
    for dp in DEATH_PS:
        key = str(dp)
        vals = results[key]
        cr  = float(np.mean([v["coll_rate"] for v in vals]))
        acc = float(np.mean([v["accuracy"]  for v in vals]))
        reg = vals[0]["regime"]
        print(f"{dp:>9.4f} | {cr:>9.5f} | {acc:>9.4f} | {reg}")

    print()
    print("Prediction: accuracy tracks regime boundaries identified by sg4.")
    print("  frozen/adaptive: high accuracy (zones linearly separable in fieldM)")
    print("  turnover-dominated: accuracy near chance (no coherent zone structure)")
