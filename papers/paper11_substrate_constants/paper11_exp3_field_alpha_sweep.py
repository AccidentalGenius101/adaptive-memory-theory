"""
paper11_exp3_field_alpha_sweep.py -- Experiment 3 for Paper 11

FIELD_ALPHA sweep. Tests whether nu_max scales linearly with FIELD_ALPHA as
the two-boundary model predicts:
  nu_max ~ P_calm * FIELD_ALPHA * FD^(1/nu)

If FIELD_ALPHA doubles, nu_max should roughly double, shifting nu* upward.
If FIELD_ALPHA halves, nu_max halves, shifting nu* downward.

Prediction (at WR=4.8, WD=15, SS=10):
  P_calm ~ 0.015, FD^(1/nu) ~ 0.74 at nu=0.001
  FIELD_ALPHA  nu_max_pred   expected nu* shift
  0.04         0.000445      downward toward nu_cryst (window narrow)
  0.08         0.000890      near nu_cryst (window very narrow)
  0.16         0.001780      reference
  0.32         0.003560      upward (nu* ~ 0.002-0.005)

nu_cryst = 0.000433 (fixed, FIELD_DECAY-governed).
At FIELD_ALPHA=0.04: nu_max ~ 0.000445 barely above nu_cryst -> very narrow window.
At FIELD_ALPHA=0.08: nu_max ~ 0.000890 -> still narrow.

Fixed: WR=4.8, WAVE_DUR=15, SS=10, FIELD_DECAY=0.9997, KAPPA=0.020.
Varied: FIELD_ALPHA in [0.04, 0.08, 0.16, 0.32].
Grid: 12 nu values, 5 seeds. 240 runs. Runtime ~10 min.
"""
import numpy as np, json, os, multiprocessing as mp

W, H = 40, 20; HALF = W // 2
HS = 2; N_ZONES = 4; ZONE_W = HALF // N_ZONES
N_ACT = HALF * H; STEPS = 2000; SHIFT = 1000; N_SEEDS = 5

MID_DECAY = 0.99; FIELD_DECAY = 0.9997; BASE_BETA = 0.005
ALPHA_MID = 0.15; SEED_BETA = 0.25
SS = 10; KAPPA = 0.020; WR = 4.8; WAVE_DUR = 15

FIELD_ALPHA_VALS = [0.04, 0.08, 0.16, 0.32]
DEATH_PS = [0.0001, 0.0002, 0.0003, 0.0005, 0.001, 0.002,
            0.003, 0.005, 0.010, 0.020, 0.040, 0.080]

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


def run(seed, death_p, field_alpha):
    rng = np.random.default_rng(seed)
    h   = rng.normal(0, 0.1, (N_ACT, HS))
    F   = np.zeros((N_ACT, HS))
    m   = np.zeros((N_ACT, HS))
    base = h.copy()
    streak = np.zeros(N_ACT, int)
    ttl = rng.geometric(p=max(death_p, 1e-6), size=N_ACT).astype(float)
    waves = []; total_collapses = 0

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
        F[ok] += field_alpha * (m[ok] - F[ok])   # <-- varied parameter
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

    return {"sg4": sg4_fn(F), "coll_rate": total_collapses / (N_ACT * STEPS)}


def _worker(args):
    return run(*args)


RESULTS_FILE = os.path.join(os.path.dirname(__file__), "results", "paper11_exp3_results.json")

if __name__ == "__main__":
    mp.freeze_support()

    if os.path.exists(RESULTS_FILE):
        results = json.load(open(RESULTS_FILE))
        print("Loaded cached results.")
    else:
        all_args = [(seed, dp, fa)
                    for fa in FIELD_ALPHA_VALS
                    for dp in DEATH_PS
                    for seed in range(N_SEEDS)]
        with mp.Pool(processes=min(len(all_args), mp.cpu_count())) as pool:
            raw = pool.map(_worker, all_args)
        results = {}
        idx = 0
        for fa in FIELD_ALPHA_VALS:
            for dp in DEATH_PS:
                key = f"{fa},{dp}"
                results[key] = []
                for seed in range(N_SEEDS):
                    results[key].append(raw[idx]); idx += 1
        os.makedirs(os.path.dirname(RESULTS_FILE), exist_ok=True)
        json.dump(results, open(RESULTS_FILE, "w"), indent=2)
        print("Saved results.")

    nu_cryst = abs(np.log(FIELD_DECAY)) / np.log(2)
    print(f"\nFIELD_ALPHA sweep (WR={WR}, WD={WAVE_DUR}, SS={SS}, {N_SEEDS} seeds)")
    print(f"nu_cryst = {nu_cryst:.6f} (fixed)")
    print(f"{'FA':>6} | {'nu_max_pred':>11} | {'window?':>12} | {'nu*':>8} | {'sg4*':>8}")
    print("-" * 65)
    pc = (1 - WR/(2*WAVE_DUR))**(SS + WAVE_DUR - 1)
    for fa in FIELD_ALPHA_VALS:
        nu_max = pc * fa * (FIELD_DECAY**(1.0/0.001))
        window_pred = "YES" if nu_max > nu_cryst else "NO (collapse)"
        sg4_arr = []
        for dp in DEATH_PS:
            key = f"{fa},{dp}"
            sg4_arr.append(float(np.mean([v["sg4"] for v in results[key]])))
        best_idx = int(np.argmax(sg4_arr))
        nu_star = DEATH_PS[best_idx]
        sg4_star = sg4_arr[best_idx]
        print(f"{fa:>6.2f} | {nu_max:>11.6f} | {window_pred:>12} | {nu_star:>8.4f} | {sg4_star:>8.1f}")
    print("\nsg4 profiles:")
    for fa in FIELD_ALPHA_VALS:
        sg4_arr = [float(np.mean([v["sg4"] for v in results[f"{fa},{dp}"]]))
                   for dp in DEATH_PS]
        profile = "  ".join(f"{v:6.0f}" for v in sg4_arr)
        print(f"  FA={fa:.2f}: {profile}")
