"""
paper13_exp1_geometric_mean.py -- Experiment 1 for Paper 13

Geometric mean law test. The two-boundary model (Paper 10) predicts:
    nu* ~ sqrt(nu_cryst * nu_max^consol)

If this holds, then independently varying nu_cryst (via FIELD_DECAY) and
nu_max^consol (via FIELD_ALPHA) should produce nu* that tracks their
geometric mean.

Cross design: 2 x 3 x 2 = 12 conditions
  FIELD_DECAY in {0.9997, 0.999}  -> nu_cryst in {4.3e-4, 1.44e-3}
  FIELD_ALPHA in {0.08, 0.16, 0.32}  -> nu_max^consol roughly doubles
  SEED_BETA   in {0.00, 0.25}        -> ablated vs reference copy-forward

Predicted nu_max^consol (approximate, at nu~0.001, WR=4.8, WD=15, SS=10):
  P_calm = (1 - WR/(2*WD))^(SS+WD-1) ~ 0.015
  nu_max^consol ~ P_calm * FA * FD^(1/0.001) ~ 0.015 * FA * 0.74
    FA=0.08: nu_max ~ 0.00089
    FA=0.16: nu_max ~ 0.00178  (reference)
    FA=0.32: nu_max ~ 0.00356

Predicted nu* = geometric mean of nu_cryst and nu_max^consol:
                    FD=0.9997           FD=0.999
    FA=0.08:   sqrt(4.3e-4 * 8.9e-4)  sqrt(1.44e-3 * 8.9e-4)
             = sqrt(3.83e-7) = 0.00062  = sqrt(1.28e-6) = 0.00113
    FA=0.16:   sqrt(4.3e-4 * 1.78e-3)  sqrt(1.44e-3 * 1.78e-3)
             = sqrt(7.65e-7) = 0.00088  = sqrt(2.56e-6) = 0.00160
    FA=0.32:   sqrt(4.3e-4 * 3.56e-3)  sqrt(1.44e-3 * 3.56e-3)
             = sqrt(1.53e-6) = 0.00124  = sqrt(5.12e-6) = 0.00226

If the geometric mean law holds:
  * nu* should track these predictions (to within ~2x).
  * Adding copy-forward (SB=0.25 vs SB=0) should shift nu* rightward,
    but should NOT destroy the geometric mean structure.
  * The residual (obs/pred) should be close to 1.0 across all 12 conditions.

Fixed: WR=4.8, WAVE_DUR=15, SS=10, KAPPA=0.020.
Grid: 12 nu values, 5 seeds. 720 runs total. Runtime ~28 min.
"""
import numpy as np, json, os, multiprocessing as mp
import math

W, H = 40, 20; HALF = W // 2
HS = 2; N_ZONES = 4; ZONE_W = HALF // N_ZONES
N_ACT = HALF * H; STEPS = 2000; SHIFT = 1000; N_SEEDS = 5

MID_DECAY = 0.99; BASE_BETA = 0.005
ALPHA_MID = 0.15; KAPPA = 0.020; WR = 4.8; WAVE_DUR = 15
SS = 10

FIELD_DECAY_VALS = [0.9997, 0.999]
FIELD_ALPHA_VALS = [0.08, 0.16, 0.32]
SEED_BETA_VALS   = [0.00, 0.25]

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


def run(seed, death_p, field_decay, field_alpha, seed_beta):
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
        F[ok] += field_alpha * (m[ok] - F[ok])   # varied
        F *= field_decay                           # varied

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
                h[i] = (1-seed_beta)*rng.normal(0,0.1,HS) + seed_beta*F[j]  # varied
                m[i] = 0; streak[i] = 0
                ttl[i] = float(rng.geometric(p=death_p))

        waves = [[w[0],w[1]-1,w[2],w[3]] for w in waves if w[1]-1 > 0]

    return {"sg4": sg4_fn(F), "coll_rate": total_collapses / (N_ACT * STEPS)}


def _worker(args):
    return run(*args)


RESULTS_FILE = os.path.join(os.path.dirname(__file__), "results", "paper13_exp1_results.json")

if __name__ == "__main__":
    mp.freeze_support()

    if os.path.exists(RESULTS_FILE):
        results = json.load(open(RESULTS_FILE))
        print("Loaded cached results.")
    else:
        all_args = [(seed, dp, fd, fa, sb)
                    for fd in FIELD_DECAY_VALS
                    for fa in FIELD_ALPHA_VALS
                    for sb in SEED_BETA_VALS
                    for dp in DEATH_PS
                    for seed in range(N_SEEDS)]
        with mp.Pool(processes=min(len(all_args), mp.cpu_count())) as pool:
            raw = pool.map(_worker, all_args)
        results = {}
        idx = 0
        for fd in FIELD_DECAY_VALS:
            for fa in FIELD_ALPHA_VALS:
                for sb in SEED_BETA_VALS:
                    for dp in DEATH_PS:
                        key = f"{fd},{fa},{sb},{dp}"
                        results[key] = []
                        for seed in range(N_SEEDS):
                            results[key].append(raw[idx]); idx += 1
        os.makedirs(os.path.dirname(RESULTS_FILE), exist_ok=True)
        json.dump(results, open(RESULTS_FILE, "w"), indent=2)
        print("Saved results.")

    print(f"\nGeometric mean law test (WR={WR}, WD={WAVE_DUR}, SS={SS}, {N_SEEDS} seeds)")
    pc = (1 - WR / (2 * WAVE_DUR)) ** (SS + WAVE_DUR - 1)
    print(f"P_calm ~ {pc:.5f}")
    print()
    print(f"{'FD':>7} | {'FA':>5} | {'SB':>5} | {'nu_cryst':>9} | {'nu_max_pred':>11} | {'nu*_pred':>9} | {'nu*_obs':>8} | {'ratio':>6} | sg4*")
    print("-" * 100)

    for fd in FIELD_DECAY_VALS:
        nu_cryst = abs(math.log(fd)) / math.log(2)
        for fa in FIELD_ALPHA_VALS:
            nu_max_pred = pc * fa * (fd ** (1.0 / 0.001))
            nu_star_pred_geom = math.sqrt(nu_cryst * nu_max_pred)
            for sb in SEED_BETA_VALS:
                sg4_arr = []
                for dp in DEATH_PS:
                    key = f"{fd},{fa},{sb},{dp}"
                    sg4_arr.append(float(np.mean([v["sg4"] for v in results[key]])))
                best_idx = int(np.argmax(sg4_arr))
                nu_star_obs = DEATH_PS[best_idx]
                sg4_star = sg4_arr[best_idx]
                ratio = nu_star_obs / nu_star_pred_geom
                print(f"{fd:>7.4f} | {fa:>5.2f} | {sb:>5.2f} | {nu_cryst:>9.6f} | "
                      f"{nu_max_pred:>11.6f} | {nu_star_pred_geom:>9.6f} | "
                      f"{nu_star_obs:>8.4f} | {ratio:>6.2f} | {sg4_star:.0f}")

    print("\nsg4 profiles (SB=0.25 only, rows=FD/FA combos):")
    for fd in FIELD_DECAY_VALS:
        for fa in FIELD_ALPHA_VALS:
            sg4_arr = [float(np.mean([v["sg4"] for v in results[f"{fd},{fa},0.25,{dp}"]]))
                       for dp in DEATH_PS]
            profile = "  ".join(f"{v:6.0f}" for v in sg4_arr)
            print(f"  FD={fd}, FA={fa}: {profile}")
