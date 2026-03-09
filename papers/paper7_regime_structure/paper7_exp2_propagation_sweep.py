"""
paper7_exp2_propagation_sweep.py -- Experiment 2 for Paper 7

Propagation sweep: vary KAPPA (field diffusion rate) from 0 to high.
KAPPA controls how strongly each site's fieldM is blended with its neighbors:
  dF[i] = KAPPA * (mean(F[neighbors]) - F[i])
Low KAPPA = no diffusion = purely local structure (fragmented).
High KAPPA = strong diffusion = zone-wide coherence (coherent).

Note: SEED_BETA (birth-seeding) also propagates structure, but KAPPA is the
dominant propagation mechanism for within-generation field coherence.

Metric: sg4 (zone differentiation) + nonadj/adj ratio (spatial encoding quality).
nonadj/adj > 1 indicates genuine location encoding.

Grid: 40x20, active=20x20=400 sites. HS=2. 4 zones. 5 seeds.
DEATH_P=0.005 fixed (adaptive regime center). SEED_BETA=0.25 fixed.
WR=4.8. 2000 steps.
Runtime: ~3 minutes.
Dependencies: numpy only.
"""
import numpy as np, json, os, multiprocessing as mp

W, H = 40, 20; HALF = W // 2
HS = 2; N_ZONES = 4; ZONE_W = HALF // N_ZONES
N_ACT = HALF * H; STEPS = 2000; N_SEEDS = 5

MID_DECAY = 0.99; FIELD_DECAY = 0.9997; BASE_BETA = 0.005
ALPHA_MID = 0.15; FIELD_ALPHA = 0.16; SEED_BETA = 0.25
WAVE_RATIO = 4.8; WAVE_DUR = 15; DEATH_P = 0.005; SS = 10

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

left_mask = zone_id <= 1; right_mask = zone_id >= 2
d_A = np.array([1.0, 0.0])

def sg4_and_ratio(F):
    means = [F[zone_id == z].mean(0) for z in range(N_ZONES)]
    adj    = [np.linalg.norm(means[i]-means[i+1]) for i in range(N_ZONES-1)]
    nonadj = [np.linalg.norm(means[i]-means[j])
               for i in range(N_ZONES) for j in range(i+2, N_ZONES)]
    sg = float(np.mean([np.linalg.norm(means[i]-means[j])
                         for i in range(N_ZONES) for j in range(i+1,N_ZONES)]))
    ratio = float(np.mean(nonadj) / (np.mean(adj) + 1e-9))
    return sg, ratio

def run(seed, kappa):
    rng = np.random.default_rng(seed)
    h = rng.normal(0, 0.1, (N_ACT, HS))
    F = np.zeros((N_ACT, HS)); m = np.zeros((N_ACT, HS))
    base = h.copy(); streak = np.zeros(N_ACT, int)
    ttl = rng.geometric(p=DEATH_P, size=N_ACT).astype(float)
    waves = []

    for step in range(STEPS):
        n = int(WAVE_RATIO / WAVE_DUR)
        n += int(rng.random() < (WAVE_RATIO / WAVE_DUR - n))
        for _ in range(n):
            z = int(rng.integers(N_ZONES))
            sign = 1.0 if z <= 1 else -1.0
            waves.append([z, WAVE_DUR, sign * d_A])

        pert = np.zeros(N_ACT, bool)
        for w in waves:
            pert |= (left_mask if w[0] <= 1 else right_mask)

        base += BASE_BETA * (h - base)
        for w in waves:
            mask = left_mask if w[0] <= 1 else right_mask
            h[mask] += 0.3 * np.array(w[2])
        m[pert] += ALPHA_MID * (h - base)[pert]
        m *= MID_DECAY; streak[pert] = 0; streak[~pert] += 1
        ok = streak >= SS
        F[ok] += FIELD_ALPHA * (m[ok] - F[ok]); F *= FIELD_DECAY

        # Field diffusion with variable KAPPA
        if kappa > 0:
            dF = np.zeros_like(F)
            for i in range(N_ACT):
                if len(NB[i]):
                    dF[i] = kappa * (F[NB[i]].mean(0) - F[i])
            F += dF

        ttl -= 1.0; ttl[pert] -= 1.0
        dead = np.where(ttl <= 0)[0]
        for i in dead:
            j = NB[i][rng.integers(len(NB[i]))] if len(NB[i]) else i
            h[i] = (1-SEED_BETA)*rng.normal(0,0.1,HS) + SEED_BETA*F[j]
            m[i] = 0; streak[i] = 0
            ttl[i] = float(rng.geometric(p=DEATH_P))

        waves = [[w[0],w[1]-1,w[2]] for w in waves if w[1]-1 > 0]

    s4, ratio = sg4_and_ratio(F)
    return {"sg4": s4, "nonadj_ratio": ratio}


def _worker(args):
    return run(*args)

KAPPAS = [0.000, 0.002, 0.005, 0.010, 0.020, 0.040, 0.080, 0.150]
RESULTS_FILE = os.path.join(os.path.dirname(__file__), "results", "paper7_exp2_results.json")

if __name__ == "__main__":
    mp.freeze_support()

    if os.path.exists(RESULTS_FILE):
        results = json.load(open(RESULTS_FILE))
        print("Loaded cached results.")
    else:
        all_args = [(seed, k) for k in KAPPAS for seed in range(N_SEEDS)]
        with mp.Pool(processes=min(len(all_args), mp.cpu_count())) as pool:
            raw = pool.map(_worker, all_args)
        results = {}
        idx = 0
        for k in KAPPAS:
            key = str(k)
            results[key] = []
            for _ in range(N_SEEDS):
                results[key].append(raw[idx]); idx += 1
        os.makedirs(os.path.dirname(RESULTS_FILE), exist_ok=True)
        json.dump(results, open(RESULTS_FILE, "w"), indent=2)
        print("Saved results.")

    print(f"\nPropagation sweep: KAPPA (field diffusion) vs structure ({N_SEEDS} seeds)")
    print(f"{'kappa':>8} | {'sg4':>8} | {'nonadj/adj':>10} | regime")
    print("-" * 42)
    for k in KAPPAS:
        key = str(k)
        vals = results[key]
        s4    = float(np.mean([v["sg4"]          for v in vals]))
        ratio = float(np.mean([v["nonadj_ratio"] for v in vals]))
        if k == 0.0:       regime = "fragmented"
        elif k <= 0.005:   regime = "weakly coherent"
        elif k >= 0.080:   regime = "over-diffused"
        else:              regime = "coherent"
        print(f"{k:>8.3f} | {s4:>8.2f} | {ratio:>10.3f} | {regime}")
    print()
    print("Prediction: sg4 and nonadj/adj rise with KAPPA then plateau or decline.")
    print("At KAPPA=0, fragmented: local structure only, nonadj/adj near 1.")
