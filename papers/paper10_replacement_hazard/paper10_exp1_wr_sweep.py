"""
paper10_exp1_wr_sweep.py -- Experiment 1 for Paper 10

WR sweep at fixed WAVE_DUR=15: tests whether Psi = nu / (P_calm * FIELD_ALPHA * FD^(1/nu))
collapses sg4 across independently varied (nu, WR) conditions.

Critically: also measures P_calm EMPIRICALLY from simulation (fraction of site-steps
with calm_streak >= SS), enabling Psi_empirical vs Psi_approx comparison.

Fixed: WAVE_DUR=15, N_ACT=400, all other VCSM params at reference values.
Varied: WR in [0.8, 1.6, 2.4, 4.8, 7.2], DEATH_PS (12 values).

Prediction: if Psi governs, sg4 as a function of Psi should collapse onto a
single peaked curve across all WR conditions. P_calm_empirical should give
tighter collapse than P_calm_approx.

300 runs total. Runtime ~15 min.
"""
import numpy as np, json, os, multiprocessing as mp

# --- Grid geometry ---
W, H = 40, 20; HALF = W // 2
HS = 2; N_ZONES = 4; ZONE_W = HALF // N_ZONES
N_ACT = HALF * H; STEPS = 2000; SHIFT = 1000; N_SEEDS = 5

# --- VCSM params (reference) ---
MID_DECAY = 0.99; FIELD_DECAY = 0.9997; BASE_BETA = 0.005
ALPHA_MID = 0.15; FIELD_ALPHA = 0.16; SEED_BETA = 0.25
SS = 10; KAPPA = 0.020; WAVE_DUR = 15

# --- Sweep parameters ---
WR_VALS = [0.8, 1.6, 2.4, 4.8, 7.2]
DEATH_PS = [0.0001, 0.0002, 0.0003, 0.0005, 0.001, 0.002,
            0.003, 0.005, 0.010, 0.020, 0.040, 0.080]

# --- Site geometry ---
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


def run(seed, death_p, wr):
    rng = np.random.default_rng(seed)
    h   = rng.normal(0, 0.1, (N_ACT, HS))
    F   = np.zeros((N_ACT, HS))
    m   = np.zeros((N_ACT, HS))
    base = h.copy()
    streak = np.zeros(N_ACT, int)
    ttl = rng.geometric(p=max(death_p, 1e-6), size=N_ACT).astype(float)
    waves = []
    total_collapses = 0
    calm_access_steps = 0   # site-steps where streak >= SS
    total_site_steps = 0    # site-steps in SHIFT..STEPS range (Phase 2 only)

    for step in range(STEPS):
        in_phase2 = step >= SHIFT

        # Wave launches: Poisson(wr/wave_dur)
        n = int(wr / WAVE_DUR)
        n += int(rng.random() < (wr / WAVE_DUR - n))
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
        F[ok] += FIELD_ALPHA * (m[ok] - F[ok])
        F *= FIELD_DECAY

        dF = np.zeros_like(F)
        for i in range(N_ACT):
            if len(NB[i]):
                dF[i] = KAPPA * (F[NB[i]].mean(0) - F[i])
        F += dF

        # Measure empirical P_calm (Phase 2 only, after burnin)
        if in_phase2:
            calm_access_steps += int(ok.sum())
            total_site_steps  += N_ACT

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

    p_calm_empirical = calm_access_steps / max(total_site_steps, 1)
    return {
        "sg4":          sg4_fn(F),
        "coll_rate":    total_collapses / (N_ACT * STEPS),
        "p_calm_emp":   p_calm_empirical,
    }


def _worker(args):
    return run(*args)


RESULTS_FILE = os.path.join(os.path.dirname(__file__), "results", "paper10_exp1_results.json")


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

    # --- Psi functions ---
    def p_calm_approx(wr):
        R = wr / WAVE_DUR
        return (1 - R/2) ** (SS + WAVE_DUR - 1)

    def psi(nu, wr, p_calm):
        survival = FIELD_DECAY ** (1.0 / nu) if nu > 0 else 0.0
        lam_c = p_calm * FIELD_ALPHA * survival
        return nu / lam_c if lam_c > 0 else np.inf

    print(f"\nWR sweep at WAVE_DUR={WAVE_DUR}  ({N_SEEDS} seeds each)")
    print(f"{'WR':>5} | {'nu':>8} | {'sg4':>8} | {'Pcalm_a':>8} | {'Pcalm_e':>8} | {'Psi_a':>7} | {'Psi_e':>7}")
    print("-" * 75)

    for wr in WR_VALS:
        pc_a = p_calm_approx(wr)
        sg4_arr, pc_e_arr = [], []
        for dp in DEATH_PS:
            key = f"{wr},{dp}"
            vals = results[key]
            sg4_arr.append(float(np.mean([v["sg4"] for v in vals])))
            pc_e_arr.append(float(np.mean([v["p_calm_emp"] for v in vals])))
        best_idx = int(np.argmax(sg4_arr))
        print(f"WR={wr}:")
        for ni, dp in enumerate(DEATH_PS):
            pc_e = pc_e_arr[ni]
            psi_a = psi(dp, wr, pc_a)
            psi_e = psi(dp, wr, pc_e)
            marker = " <--" if ni == best_idx else ""
            print(f"  {wr:>5.1f} | {dp:>8.4f} | {sg4_arr[ni]:>8.1f} | {pc_a:>8.5f} | {pc_e:>8.5f} | {psi_a:>7.4f} | {psi_e:>7.4f}{marker}")
