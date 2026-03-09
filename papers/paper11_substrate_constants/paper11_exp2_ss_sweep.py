"""
paper11_exp2_ss_sweep.py -- Experiment 2 for Paper 11

SS sweep. Tests whether changing the consolidation gate (SS) shifts the UPPER
boundary of the adaptive window as predicted by:
  nu_max ~ P_calm(SS) * FIELD_ALPHA * FD^(1/nu)
  P_calm(SS) = (1 - WR/(2*WD))^(SS + WD - 1)

At WR=4.8, WD=15:
  SS=5:  P_calm = 0.84^19 ~ 0.040  -> nu_max ~ 0.0047
  SS=10: P_calm = 0.84^24 ~ 0.015  -> nu_max ~ 0.0018  (reference)
  SS=20: P_calm = 0.84^34 ~ 0.0021 -> nu_max ~ 0.00025
  SS=40: P_calm = 0.84^54 ~ 8.2e-5 -> nu_max ~ 9.7e-6  < nu_cryst!

CRITICAL PREDICTION for SS=40:
  nu_max(SS=40) < nu_cryst ~ 4.3e-4
  -> The adaptive window COLLAPSES: no nu satisfies both constraints.
  -> sg4 should remain low for ALL tested nu values.

Fixed: WR=4.8, WAVE_DUR=15, FIELD_DECAY=0.9997, FIELD_ALPHA=0.16, KAPPA=0.020.
Varied: SS in [5, 10, 20, 40].
Grid: 12 nu values, 5 seeds. 240 runs. Runtime ~10 min.
"""
import numpy as np, json, os, multiprocessing as mp

W, H = 40, 20; HALF = W // 2
HS = 2; N_ZONES = 4; ZONE_W = HALF // N_ZONES
N_ACT = HALF * H; STEPS = 2000; SHIFT = 1000; N_SEEDS = 5

MID_DECAY = 0.99; FIELD_DECAY = 0.9997; BASE_BETA = 0.005
ALPHA_MID = 0.15; FIELD_ALPHA = 0.16; SEED_BETA = 0.25
KAPPA = 0.020; WR = 4.8; WAVE_DUR = 15

SS_VALS  = [5, 10, 20, 40]
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


def run(seed, death_p, ss):
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
        ok = streak >= ss    # <-- varied parameter
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

    return {"sg4": sg4_fn(F), "coll_rate": total_collapses / (N_ACT * STEPS)}


def _worker(args):
    return run(*args)


RESULTS_FILE = os.path.join(os.path.dirname(__file__), "results", "paper11_exp2_results.json")

if __name__ == "__main__":
    mp.freeze_support()

    if os.path.exists(RESULTS_FILE):
        results = json.load(open(RESULTS_FILE))
        print("Loaded cached results.")
    else:
        all_args = [(seed, dp, ss)
                    for ss in SS_VALS
                    for dp in DEATH_PS
                    for seed in range(N_SEEDS)]
        with mp.Pool(processes=min(len(all_args), mp.cpu_count())) as pool:
            raw = pool.map(_worker, all_args)
        results = {}
        idx = 0
        for ss in SS_VALS:
            for dp in DEATH_PS:
                key = f"{ss},{dp}"
                results[key] = []
                for seed in range(N_SEEDS):
                    results[key].append(raw[idx]); idx += 1
        os.makedirs(os.path.dirname(RESULTS_FILE), exist_ok=True)
        json.dump(results, open(RESULTS_FILE, "w"), indent=2)
        print("Saved results.")

    print(f"\nSS sweep (WR={WR}, WD={WAVE_DUR}, FD={FIELD_DECAY}, {N_SEEDS} seeds)")
    print(f"{'SS':>4} | {'P_calm':>8} | {'nu_max':>8} | {'nu*':>8} | {'sg4*':>8} | {'window?':>8}")
    print("-" * 80)
    for ss in SS_VALS:
        pc = (1 - WR/(2*WAVE_DUR))**(ss + WAVE_DUR - 1)
        nu_max = pc * FIELD_ALPHA * (FIELD_DECAY**(1.0/0.001))   # approx at nu=0.001
        nu_cryst = abs(np.log(FIELD_DECAY)) / np.log(2)
        sg4_arr = []
        for dp in DEATH_PS:
            key = f"{ss},{dp}"
            sg4_arr.append(float(np.mean([v["sg4"] for v in results[key]])))
        best_idx = int(np.argmax(sg4_arr))
        nu_star = DEATH_PS[best_idx]
        sg4_star = sg4_arr[best_idx]
        window_pred = "YES" if nu_max > nu_cryst else "NO (collapse)"
        print(f"{ss:>4} | {pc:>8.6f} | {nu_max:>8.6f} | {nu_star:>8.4f} | {sg4_star:>8.1f} | {window_pred}")
    print("\nsg4 profiles:")
    for ss in SS_VALS:
        sg4_arr = [float(np.mean([v["sg4"] for v in results[f"{ss},{dp}"]]))
                   for dp in DEATH_PS]
        profile = "  ".join(f"{v:6.0f}" for v in sg4_arr)
        print(f"  SS={ss:>2}: {profile}")
