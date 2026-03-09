"""
paper7_exp1_turnover_sweep.py -- Experiment 1 for Paper 7

Turnover sweep: vary collapse rate (death probability) from frozen to chaotic.
Identifies three regimes: frozen (low nu), adaptive (mid nu), chaotic (high nu).

Metrics per condition:
  sg4        -- mean pairwise L2 between zone-mean fieldM (zone differentiation)
  coll_rate  -- collapses per site per step
  adapt      -- alignment score after orthogonal pattern shift (adaptation quality)

Protocol (2000 steps, shift at 1000):
  Phase 1: Left/Right pattern. Waves perturb zone 0-1 with d_A, zones 2-3 with -d_A.
  Phase 2: Top/Bottom pattern (orthogonal). Measure alignment with d_B at end.

Grid: 40x20, active half=20x20=400 sites, HS=2, 4 zones, 5 seeds.
Turnover controlled by DEATH_P (Bernoulli TTL death) + WR=4.8 fixed.
SEED_BETA=0.25 fixed.
Runtime: ~3 minutes.
Dependencies: numpy only.
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

def sg4(F):
    means = [F[zone_id == z].mean(0) for z in range(N_ZONES)]
    pairs = [(i,j) for i in range(N_ZONES) for j in range(i+1,N_ZONES)]
    return float(np.mean([np.linalg.norm(means[i]-means[j]) for i,j in pairs]))

def align(F, d, pos, neg):
    """Cosine similarity of zone contrast with target direction. Scale-invariant."""
    d = d / np.linalg.norm(d)
    contrast = F[pos].mean(0) - F[neg].mean(0)
    mag = float(np.linalg.norm(contrast))
    if mag < 1e-9: return 0.0
    return float(np.dot(contrast / mag, d))  # in [-1, 1]

def run(seed, death_p):
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

        # Record early Phase 2 alignment (adaptation speed)
        if step == SHIFT + 200:
            adapt_early = align(F, d_B, top_mask, bot_mask)

    return {
        "sg4":        sg4(F),
        "adapt":      align(F, d_B, top_mask, bot_mask),
        "adapt_early": adapt_early,
        "coll_rate":  total_collapses / (N_ACT * STEPS),
    }


def _worker(args):
    return run(*args)

DEATH_PS = [0.0001, 0.0003, 0.001, 0.002, 0.005, 0.010, 0.020, 0.040, 0.080]
RESULTS_FILE = os.path.join(os.path.dirname(__file__), "results", "paper7_exp1_results.json")

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

    print(f"\nTurnover sweep: DEATH_P vs regime metrics ({N_SEEDS} seeds)")
    print(f"{'death_p':>9} | {'coll/site':>9} | {'sg4':>8} | {'adapt@1200':>11} | regime")
    print("-" * 55)
    for dp in DEATH_PS:
        key = str(dp)
        vals = results[key]
        cr   = float(np.mean([v["coll_rate"] for v in vals]))
        s4   = float(np.mean([v["sg4"]       for v in vals]))
        ad   = float(np.mean([v["adapt"]     for v in vals]))
        ae   = float(np.mean([v["adapt_early"] for v in vals]))
        if cr < 0.001:    regime = "frozen"
        elif cr > 0.015:  regime = "chaotic"
        else:             regime = "adaptive"
        print(f"{dp:>9.4f} | {cr:>9.5f} | {s4:>8.4f} | {ae:>8.3f} | {regime}")
    print()
    print("Prediction: sg4 and adapt peak in adaptive regime, low in frozen and chaotic.")
