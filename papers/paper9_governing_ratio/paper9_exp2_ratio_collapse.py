"""
paper9_exp2_ratio_collapse.py -- Experiment 2 for Paper 9

Ratio collapse: if R = WR/WAVE_DUR governs nu*, then (WR, WAVE_DUR) pairs
with the same ratio should produce the same nu*, regardless of the individual
values of WR and WAVE_DUR.

Hypothesis: the adaptive window location depends only on R = WR/WAVE_DUR,
not WR or WAVE_DUR individually.

Protocol:
  Test 3 (WR, WAVE_DUR) pairs, all with R = WR/WAVE_DUR = 0.32 (reference value):
    Pair A: WR=1.6, WAVE_DUR=5   (R=0.320)
    Pair B: WR=4.8, WAVE_DUR=15  (R=0.320) -- reference condition
    Pair C: WR=9.6, WAVE_DUR=30  (R=0.320)
  Extended DEATH_PS, 5 seeds per (pair, nu)

Prediction: all three pairs show nu* ~ 0.001 (same peak location).
  Amplitude (sg4*) may differ because WAVE_DUR affects P_calm and perturbation pattern.

Runtime: ~10 minutes.
"""
import numpy as np, json, os, multiprocessing as mp

W, H = 40, 20; HALF = W // 2
HS = 2; N_ZONES = 4; ZONE_W = HALF // N_ZONES
N_ACT = HALF * H; STEPS = 2000; SHIFT = 1000; N_SEEDS = 5

MID_DECAY = 0.99; FIELD_DECAY = 0.9997; BASE_BETA = 0.005
ALPHA_MID = 0.15; FIELD_ALPHA = 0.16; SEED_BETA = 0.25
SS = 10; KAPPA = 0.020

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

# Extended DEATH_PS grid
DEATH_PS = [0.0001, 0.0002, 0.0003, 0.0005, 0.001, 0.002,
            0.003, 0.005, 0.010, 0.020, 0.040, 0.080]

# (WR, WAVE_DUR) pairs, all with R = WR/WAVE_DUR = 0.32
PAIRS = [(1.6, 5), (4.8, 15), (9.6, 30)]

def sg4(F):
    means = [F[zone_id == z].mean(0) for z in range(N_ZONES)]
    pairs = [(i,j) for i in range(N_ZONES) for j in range(i+1,N_ZONES)]
    return float(np.mean([np.linalg.norm(means[i]-means[j]) for i,j in pairs]))

def run(seed, death_p, wr, wave_dur):
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

        n = int(wr / wave_dur)
        n += int(rng.random() < (wr / wave_dur - n))
        for _ in range(n):
            if not in_phase2:
                z = int(rng.integers(N_ZONES))
                sign = 1.0 if z <= 1 else -1.0
                waves.append([z, wave_dur, sign * d_A, True])
            else:
                top = rng.random() < 0.5
                sign = 1.0 if top else -1.0
                waves.append([0 if top else 1, wave_dur, sign * d_B, False])

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

    return {"sg4": sg4(F), "coll_rate": total_collapses / (N_ACT * STEPS)}


def _worker(args):
    return run(*args)


RESULTS_FILE = os.path.join(os.path.dirname(__file__), "results", "paper9_exp2_results.json")

if __name__ == "__main__":
    mp.freeze_support()

    if os.path.exists(RESULTS_FILE):
        results = json.load(open(RESULTS_FILE))
        print("Loaded cached results.")
    else:
        all_args = [(seed, dp, wr, wd)
                    for (wr, wd) in PAIRS
                    for dp in DEATH_PS
                    for seed in range(N_SEEDS)]
        with mp.Pool(processes=min(len(all_args), mp.cpu_count())) as pool:
            raw = pool.map(_worker, all_args)
        results = {}
        idx = 0
        for (wr, wd) in PAIRS:
            for dp in DEATH_PS:
                key = f"{wr},{wd},{dp}"
                results[key] = []
                for seed in range(N_SEEDS):
                    results[key].append(raw[idx]); idx += 1
        os.makedirs(os.path.dirname(RESULTS_FILE), exist_ok=True)
        json.dump(results, open(RESULTS_FILE, "w"), indent=2)
        print("Saved results.")

    N_act_over_4 = N_ACT / 4
    print(f"\nRatio collapse: same R=WR/WAVE_DUR -> same nu*? ({N_SEEDS} seeds)")
    print(f"All pairs have R = WR/WAVE_DUR = 0.32\n")
    print(f"{'WR':>5} {'WD':>4} {'R':>6} | {'nu*':>8} | {'sg4*':>8} | {'Xi*':>6} | sg4 profile")
    print("-" * 95)
    for (wr, wd) in PAIRS:
        sg4_vals = []
        for dp in DEATH_PS:
            key = f"{wr},{wd},{dp}"
            sg4_vals.append(float(np.mean([v["sg4"] for v in results[key]])))
        best_idx = int(np.argmax(sg4_vals))
        nu_star = DEATH_PS[best_idx]
        sg4_star = sg4_vals[best_idx]
        xi_star = nu_star * N_act_over_4 * 2 * wd / wr
        R = wr / wd
        profile = "  ".join(f"{v:6.0f}" for v in sg4_vals)
        print(f"{wr:>5.1f} {wd:>4} {R:>6.3f} | {nu_star:>8.4f} | {sg4_star:>8.1f} | {xi_star:>6.3f} | {profile}")

    print()
    print("Xi* column: should be equal across all rows if R alone governs nu*.")
