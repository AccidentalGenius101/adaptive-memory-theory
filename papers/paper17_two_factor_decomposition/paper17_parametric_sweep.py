"""
paper17_parametric_sweep.py -- Experiment for Paper 17

Two-Factor Decomposition of the Saturation Law:
    sg4_steady = C * FA / (FA + K_eff)

Every free parameter maps onto C, K_eff, or both.
Four sweeps (one parameter varied, three fixed at baseline):

  SS sweep   : SS in {5, 7, 10, 15, 20}           -> K_eff moves, C flat
  MD sweep   : MID_DECAY in {0.97,0.98,0.99,0.995,0.999} -> C moves, K_eff flat
  SB sweep   : SEED_BETA in {0.00,0.10,0.25,0.50,0.75}   -> C moves, K_eff flat
  KP sweep   : KAPPA in {0.005,0.010,0.020,0.040,0.080}   -> BOTH move

Baselines: SS=10, MID_DECAY=0.99, SEED_BETA=0.25, KAPPA=0.020
           FD=0.9997, NU=0.001, SHIFT=0, STEPS=2000

FA in {0.10, 0.20, 0.40, 0.70, 0.90}  x  5 seeds  x  4*5 conditions = 500 runs

Key: "{sweep},{val},{fa:.4f},{seed}" -> {"sg4_2000": float}
"""
import numpy as np, json, os, multiprocessing as mp, math

W, H = 40, 20; HALF = W // 2
HS = 2; N_ZONES = 4; ZONE_W = HALF // N_ZONES
N_ACT = HALF * H; STEPS = 2000; N_SEEDS = 5

BASE_BETA = 0.005; ALPHA_MID = 0.15
WR = 4.8; WAVE_DUR = 15
FD = 0.9997; NU = 0.001

SS_BASE = 10; MD_BASE = 0.99; SB_BASE = 0.25; KP_BASE = 0.020

FA_VALS = [0.10, 0.20, 0.40, 0.70, 0.90]
SS_VALS = [5, 7, 10, 15, 20]
MD_VALS = [0.97, 0.98, 0.99, 0.995, 0.999]
SB_VALS = [0.00, 0.10, 0.25, 0.50, 0.75]
KP_VALS = [0.005, 0.010, 0.020, 0.040, 0.080]

# Analytical calibration (from Paper 16)
P_CALM_PER_STEP = 1.0 - WR / (2.0 * WAVE_DUR)   # = 0.84

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

top_mask = _row < H // 2
bot_mask = _row >= H // 2
d_B = np.array([0.0, 1.0])


def sg4_inter_fn(F):
    means = [F[zone_id == z].mean(0) for z in range(N_ZONES)]
    pairs = [(i,j) for i in range(N_ZONES) for j in range(i+1,N_ZONES)]
    return float(np.mean([np.linalg.norm(means[i]-means[j]) for i,j in pairs]))


def run(seed, fa, ss, mid_decay, seed_beta, kappa):
    rng = np.random.default_rng(seed)
    h    = rng.normal(0, 0.1, (N_ACT, HS))
    F    = np.zeros((N_ACT, HS))
    m    = np.zeros((N_ACT, HS))
    base = h.copy()
    streak = np.zeros(N_ACT, int)
    ttl = rng.geometric(p=max(NU, 1e-6), size=N_ACT).astype(float)
    waves = []

    for step in range(STEPS):
        n = int(WR / WAVE_DUR)
        n += int(rng.random() < (WR / WAVE_DUR - n))
        for _ in range(n):
            top = rng.random() < 0.5
            sign = 1.0 if top else -1.0
            waves.append([0 if top else 1, WAVE_DUR, sign * d_B, False])

        pert = np.zeros(N_ACT, bool)
        for w in waves:
            pert |= (top_mask if w[0] == 0 else bot_mask)

        base += BASE_BETA * (h - base)
        for w in waves:
            mask = top_mask if w[0] == 0 else bot_mask
            h[mask] += 0.3 * np.array(w[2])

        m[pert] += ALPHA_MID * (h - base)[pert]
        m *= mid_decay
        streak[pert] = 0; streak[~pert] += 1
        ok = streak >= ss
        F[ok] += fa * (m[ok] - F[ok])
        F *= FD

        dF = np.zeros_like(F)
        for i in range(N_ACT):
            if len(NB[i]):
                dF[i] = kappa * (F[NB[i]].mean(0) - F[i])
        F += dF

        ttl -= 1.0; ttl[pert] -= 1.0
        dead = np.where(ttl <= 0)[0]
        for i in dead:
            j = NB[i][rng.integers(len(NB[i]))] if len(NB[i]) else i
            h[i] = (1-seed_beta)*rng.normal(0,0.1,HS) + seed_beta*F[j]
            m[i] = 0; streak[i] = 0
            ttl[i] = float(rng.geometric(p=NU))

        waves = [[w[0],w[1]-1,w[2],w[3]] for w in waves if w[1]-1 > 0]

    return {"sg4_2000": sg4_inter_fn(F)}


def make_key(sweep, val, fa, seed):
    if sweep == "ss":
        return f"ss,{int(val)},{fa:.4f},{seed}"
    elif sweep == "md":
        return f"md,{val:.5f},{fa:.4f},{seed}"
    elif sweep == "sb":
        return f"sb,{val:.4f},{fa:.4f},{seed}"
    else:
        return f"kp,{val:.4f},{fa:.4f},{seed}"


def _worker(args):
    return run(*args)


RESULTS_FILE = os.path.join(os.path.dirname(__file__), "results", "paper17_results.json")


def sat_fit(fa_list, sg4_list):
    """Grid search for best (C, K_eff) in sg4 = C*FA/(FA+K_eff). Returns (C, K_eff, R2)."""
    if len(fa_list) < 2:
        return float("nan"), float("nan"), float("nan")
    best_r2 = -1e9; best_C = 0; best_K = 0
    for K in np.linspace(0.005, 1.5, 500):
        x = [fa/(fa+K) for fa in fa_list]
        denom = sum(v*v for v in x)
        if denom == 0: continue
        C = float(np.dot(x, sg4_list) / denom)
        pred = [C*xx for xx in x]
        ss_res = sum((p-d)**2 for p,d in zip(pred, sg4_list))
        ss_tot = sum((d-float(np.mean(sg4_list)))**2 for d in sg4_list)
        r2 = 1 - ss_res/ss_tot if ss_tot > 0 else float("nan")
        if r2 > best_r2:
            best_r2 = r2; best_C = C; best_K = K
    return best_C, best_K, best_r2


if __name__ == "__main__":
    mp.freeze_support()

    if os.path.exists(RESULTS_FILE):
        results = json.load(open(RESULTS_FILE))
        print("Loaded cached results.")
    else:
        sweeps = [
            ("ss", SS_VALS, lambda v: (int(v), MD_BASE, SB_BASE, KP_BASE)),
            ("md", MD_VALS, lambda v: (SS_BASE, float(v), SB_BASE, KP_BASE)),
            ("sb", SB_VALS, lambda v: (SS_BASE, MD_BASE, float(v), KP_BASE)),
            ("kp", KP_VALS, lambda v: (SS_BASE, MD_BASE, SB_BASE, float(v))),
        ]

        all_conditions = []
        for sweep_name, vals, param_fn in sweeps:
            for val in vals:
                ss, md, sb, kp = param_fn(val)
                for fa in FA_VALS:
                    for seed in range(N_SEEDS):
                        all_conditions.append((sweep_name, val, fa, seed, ss, md, sb, kp))

        all_args = [(c[3], c[2], c[4], c[5], c[6], c[7]) for c in all_conditions]
        print(f"Running {len(all_args)} simulations (Paper 17)...")
        with mp.Pool(processes=min(len(all_args), mp.cpu_count())) as pool:
            raw = pool.map(_worker, all_args)

        results = {}
        for i, cond in enumerate(all_conditions):
            sweep, val, fa, seed = cond[0], cond[1], cond[2], cond[3]
            results[make_key(sweep, val, fa, seed)] = raw[i]

        os.makedirs(os.path.dirname(RESULTS_FILE), exist_ok=True)
        json.dump(results, open(RESULTS_FILE, "w"), indent=2)
        print("Saved results.")

    # ---- Analysis ----
    p_calm = P_CALM_PER_STEP
    print(f"\np_calm_per_step = {p_calm:.4f}")
    print(f"P_consol(SS=10) = {p_calm**10:.4f}")
    print()

    for sweep_name, vals, label in [
        ("ss", SS_VALS, "SS"),
        ("md", MD_VALS, "MID_DECAY"),
        ("sb", SB_VALS, "SEED_BETA"),
        ("kp", KP_VALS, "KAPPA"),
    ]:
        print(f"=== {label} SWEEP ===")
        print(f"  {'Val':>8} | {'C':>8} | {'K_eff':>8} | {'K_eff_pred':>10} | {'R2':>6}")
        print("  " + "-"*50)
        for val in vals:
            fa_data, sg4_data = [], []
            for fa in FA_VALS:
                key_list = [make_key(sweep_name, val, fa, s) for s in range(N_SEEDS)
                            if make_key(sweep_name, val, fa, s) in results]
                if key_list:
                    vals_sg4 = [results[k]["sg4_2000"] for k in key_list]
                    fa_data.append(fa); sg4_data.append(float(np.mean(vals_sg4)))
            C, K, r2 = sat_fit(fa_data, sg4_data)
            if sweep_name == "ss":
                k_pred = KP_BASE / (p_calm ** int(val))
            elif sweep_name == "kp":
                k_pred = float(val) / (p_calm ** SS_BASE)
            else:
                k_pred = KP_BASE / (p_calm ** SS_BASE)
            print(f"  {val:>8} | {C:>8.2f} | {K:>8.4f} | {k_pred:>10.4f} | {r2:>6.4f}")
        print()
