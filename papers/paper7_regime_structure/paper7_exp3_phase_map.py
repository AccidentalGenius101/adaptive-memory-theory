"""
paper7_exp3_phase_map.py -- Experiment 3 for Paper 7

2D phase map: sweep turnover (nu = DEATH_P) x propagation (kappa = KAPPA field diffusion).
This is the central figure of Paper 7 -- the empirical phase diagram.

Axes:
  nu    = DEATH_P      (turnover rate; frozen=low, adaptive=mid, chaotic=high)
  kappa = KAPPA_field  (field diffusion rate; fragmented=low, coherent=high)

Metrics at each (nu, kappa) point:
  sg4   -- zone differentiation (structure metric)
  adapt -- cosine similarity of fieldM contrast with new pattern (in [-1,1])

Four predicted regimes:
  (low nu,  low kappa)  = frozen + fragmented
  (low nu,  high kappa) = frozen + coherent (crystallized)
  (high nu, low kappa)  = chaotic + fragmented
  (mid nu,  mid kappa)  = ADAPTIVE (feasible region)

Grid: 40x20, active=20x20=400 sites. HS=2. 5x5 parameter grid. 5 seeds.
Protocol: 2000 steps, pattern shift at 1000.
Runtime: ~10-15 minutes (parallel).
Dependencies: numpy only.
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

left_mask  = zone_id <= 1; right_mask = zone_id >= 2
top_mask   = _row < H // 2; bot_mask  = _row >= H // 2
d_A = np.array([1.0, 0.0]); d_B = np.array([0.0, 1.0])

def sg4(F):
    means = [F[zone_id == z].mean(0) for z in range(N_ZONES)]
    pairs = [(i,j) for i in range(N_ZONES) for j in range(i+1,N_ZONES)]
    return float(np.mean([np.linalg.norm(means[i]-means[j]) for i,j in pairs]))

def align(F, d, pos, neg):
    d = d / np.linalg.norm(d)
    contrast = F[pos].mean(0) - F[neg].mean(0)
    mag = float(np.linalg.norm(contrast))
    if mag < 1e-9: return 0.0
    return float(np.dot(contrast / mag, d))

def run(seed, death_p, kappa):
    rng = np.random.default_rng(seed)
    h = rng.normal(0, 0.1, (N_ACT, HS))
    F = np.zeros((N_ACT, HS)); m = np.zeros((N_ACT, HS))
    base = h.copy(); streak = np.zeros(N_ACT, int)
    ttl = rng.geometric(p=max(death_p, 1e-6), size=N_ACT).astype(float)
    waves = []; adapt_early = 0.0

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
            if w[3]: mask = left_mask if w[0] <= 1 else right_mask
            else:    mask = top_mask if w[0] == 0 else bot_mask
            h[mask] += 0.3 * np.array(w[2])
        m[pert] += ALPHA_MID * (h - base)[pert]
        m *= MID_DECAY; streak[pert] = 0; streak[~pert] += 1
        ok = streak >= SS
        F[ok] += FIELD_ALPHA * (m[ok] - F[ok]); F *= FIELD_DECAY

        if kappa > 0:
            dF = np.zeros_like(F)
            for i in range(N_ACT):
                if len(NB[i]): dF[i] = kappa * (F[NB[i]].mean(0) - F[i])
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

        if step == SHIFT + 200:
            adapt_early = align(F, d_B, top_mask, bot_mask)

    return {"sg4": sg4(F), "adapt": adapt_early}


def _worker(args):
    return run(*args)

NU_VALS    = [0.0003, 0.001, 0.005, 0.015, 0.050]
KAPPA_VALS = [0.000, 0.005, 0.020, 0.060, 0.150]
RESULTS_FILE = os.path.join(os.path.dirname(__file__), "results", "paper7_exp3_results.json")

if __name__ == "__main__":
    mp.freeze_support()

    if os.path.exists(RESULTS_FILE):
        results = json.load(open(RESULTS_FILE))
        print("Loaded cached results.")
    else:
        all_args = [(seed, nu, kp)
                    for nu in NU_VALS for kp in KAPPA_VALS
                    for seed in range(N_SEEDS)]
        with mp.Pool(processes=min(len(all_args), mp.cpu_count())) as pool:
            raw = pool.map(_worker, all_args)
        results = {}
        idx = 0
        for nu in NU_VALS:
            for kp in KAPPA_VALS:
                key = f"{nu},{kp}"
                results[key] = []
                for _ in range(N_SEEDS):
                    results[key].append(raw[idx]); idx += 1
        os.makedirs(os.path.dirname(RESULTS_FILE), exist_ok=True)
        json.dump(results, open(RESULTS_FILE, "w"), indent=2)
        print("Saved results.")

    print(f"\n2D Phase Map: nu (rows) x kappa (cols) -- sg4 / adapt [{N_SEEDS} seeds]")
    kh = "  ".join(f"k={kp:.3f}" for kp in KAPPA_VALS)
    print(f"\n{'nu':>8} | {kh}")
    print("-" * 75)
    for nu in NU_VALS:
        row_sg4 = []; row_ad = []
        for kp in KAPPA_VALS:
            key = f"{nu},{kp}"
            vals = results[key]
            row_sg4.append(float(np.mean([v["sg4"]   for v in vals])))
            row_ad.append( float(np.mean([v["adapt"] for v in vals])))
        sg4_str = "  ".join(f"{v:6.2f}" for v in row_sg4)
        ad_str  = "  ".join(f"{v:+5.2f}" for v in row_ad)
        print(f"nu={nu:.4f} | sg4:   {sg4_str}")
        print(f"         | adapt: {ad_str}")
        print()
    print("adapt in [-1,1]: +1=full adaptation, 0=no encoding, -1=reversed.")
    print("Prediction: peak sg4 and adapt>0 at mid nu, mid-high kappa (adaptive regime).")
