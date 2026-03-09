"""
paper12_exp1_seed_beta_sweep.py -- Experiment 1 for Paper 12

SEED_BETA sweep. Tests how the copy-forward inheritance strength shapes both
the height and position of the adaptive peak.

At birth, a dead site's hidden state is seeded as:
    h_new = (1-SEED_BETA)*noise + SEED_BETA*F[neighbor]

SEED_BETA=0: pure-noise birth. Copy-forward pathway ablated. New sites
    have no structural prior; they must re-learn from scratch. Adaptive
    peak should drop dramatically (or disappear) at mid-nu.

SEED_BETA>0: proportional inheritance. Each collapse reinjects local
    field structure. At high nu, collapses become the PRIMARY driver of
    sg4 -- not wave-driven consolidation but structural relay.

Key predictions:
  SEED_BETA=0.00  -> adaptive peak collapses (no relay). sg4 low at all nu.
  SEED_BETA=0.05  -> weak relay. Partial recovery.
  SEED_BETA=0.10  -> moderate relay. Peak appears at mid-nu.
  SEED_BETA=0.25  -> reference. Standard VCML behavior.
  SEED_BETA=0.50  -> strong relay. Peak may shift to higher nu (relay more
                     dominant); but very high nu corrupts F so diminishing returns.

If Gamma_cf = nu * SEED_BETA * F_quality governs relay:
  - At fixed F_quality, nu* should shift right as SEED_BETA doubles.
  - sg4 peak height should increase monotonically with SEED_BETA.
  - At nu->0 (no deaths), SEED_BETA is irrelevant (no births).
  - At nu very high (nu=0.08), F_quality~0 (field always fresh noise)
    so SEED_BETA*0 = 0 regardless.

Fixed: WR=4.8, WAVE_DUR=15, SS=10, FIELD_DECAY=0.9997, FIELD_ALPHA=0.16, KAPPA=0.020.
Varied: SEED_BETA in [0.00, 0.05, 0.10, 0.25, 0.50].
Grid: 12 nu values, 5 seeds. 300 runs. Runtime ~12 min.
"""
import numpy as np, json, os, multiprocessing as mp

W, H = 40, 20; HALF = W // 2
HS = 2; N_ZONES = 4; ZONE_W = HALF // N_ZONES
N_ACT = HALF * H; STEPS = 2000; SHIFT = 1000; N_SEEDS = 5

MID_DECAY = 0.99; FIELD_DECAY = 0.9997; BASE_BETA = 0.005
ALPHA_MID = 0.15; FIELD_ALPHA = 0.16
SS = 10; KAPPA = 0.020; WR = 4.8; WAVE_DUR = 15

SEED_BETA_VALS = [0.00, 0.05, 0.10, 0.25, 0.50]
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


def run(seed, death_p, seed_beta):
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
                # seed_beta varied here -- this is the copy-forward strength
                h[i] = (1-seed_beta)*rng.normal(0,0.1,HS) + seed_beta*F[j]
                m[i] = 0; streak[i] = 0
                ttl[i] = float(rng.geometric(p=death_p))

        waves = [[w[0],w[1]-1,w[2],w[3]] for w in waves if w[1]-1 > 0]

    # Also record mean field norm as proxy for F_quality
    f_norm = float(np.mean(np.linalg.norm(F, axis=1)))
    return {"sg4": sg4_fn(F), "coll_rate": total_collapses / (N_ACT * STEPS),
            "f_norm": f_norm}


def _worker(args):
    return run(*args)


RESULTS_FILE = os.path.join(os.path.dirname(__file__), "results", "paper12_exp1_results.json")

if __name__ == "__main__":
    mp.freeze_support()

    if os.path.exists(RESULTS_FILE):
        results = json.load(open(RESULTS_FILE))
        print("Loaded cached results.")
    else:
        all_args = [(seed, dp, sb)
                    for sb in SEED_BETA_VALS
                    for dp in DEATH_PS
                    for seed in range(N_SEEDS)]
        with mp.Pool(processes=min(len(all_args), mp.cpu_count())) as pool:
            raw = pool.map(_worker, all_args)
        results = {}
        idx = 0
        for sb in SEED_BETA_VALS:
            for dp in DEATH_PS:
                key = f"{sb},{dp}"
                results[key] = []
                for seed in range(N_SEEDS):
                    results[key].append(raw[idx]); idx += 1
        os.makedirs(os.path.dirname(RESULTS_FILE), exist_ok=True)
        json.dump(results, open(RESULTS_FILE, "w"), indent=2)
        print("Saved results.")

    import math
    nu_cryst = abs(math.log(FIELD_DECAY)) / math.log(2)
    print(f"\nSEED_BETA sweep (WR={WR}, WD={WAVE_DUR}, SS={SS}, {N_SEEDS} seeds)")
    print(f"nu_cryst = {nu_cryst:.6f} (fixed)")
    print(f"{'SB':>5} | {'nu*':>8} | {'sg4*':>8} | {'sg4(SB=0)':>10} | f_norm")
    print("-" * 65)
    for sb in SEED_BETA_VALS:
        sg4_arr = []
        f_arr = []
        for dp in DEATH_PS:
            key = f"{sb},{dp}"
            sg4_arr.append(float(np.mean([v["sg4"] for v in results[key]])))
            f_arr.append(float(np.mean([v["f_norm"] for v in results[key]])))
        best_idx = int(np.argmax(sg4_arr))
        nu_star = DEATH_PS[best_idx]
        sg4_star = sg4_arr[best_idx]
        fn_star = f_arr[best_idx]
        print(f"{sb:>5.2f} | {nu_star:>8.4f} | {sg4_star:>8.1f} | {'---':>10} | {fn_star:.4f}")
    print("\nsg4 profiles:")
    for sb in SEED_BETA_VALS:
        sg4_arr = [float(np.mean([v["sg4"] for v in results[f"{sb},{dp}"]]))
                   for dp in DEATH_PS]
        profile = "  ".join(f"{v:6.0f}" for v in sg4_arr)
        print(f"  SB={sb:.2f}: {profile}")
