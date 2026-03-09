"""
paper12_exp2_product_collapse.py -- Experiment 2 for Paper 12

Product collapse test. If the copy-forward maintenance rate is:
    Gamma_cf = nu * SEED_BETA * F_quality

then for fixed F_quality, pairs (nu1, SB1) and (nu2, SB2) with
    nu1 * SB1 = nu2 * SB2
should yield similar sg4.

Design: 4x4 grid of (nu, SEED_BETA) pairs covering the adaptive regime.
nu_VALS    = [0.001, 0.002, 0.005, 0.010]
SB_VALS    = [0.05, 0.10, 0.25, 0.50]

Products (nu * SB):
         SB=0.05  SB=0.10  SB=0.25  SB=0.50
nu=0.001  5e-5    1e-4     2.5e-4   5e-4
nu=0.002  1e-4    2e-4     5e-4     1e-3
nu=0.005  2.5e-4  5e-4     1.25e-3  2.5e-3
nu=0.010  5e-4    1e-3     2.5e-3   5e-3

Iso-product pairs (predicted same sg4):
  P=1e-4:  (0.001, 0.10) <-> (0.002, 0.05)
  P=2.5e-4:(0.001, 0.25) <-> (0.005, 0.05) <-> ... (approx)
  P=5e-4:  (0.001, 0.50) <-> (0.002, 0.25) <-> (0.005, 0.10) <-> (0.010, 0.05)

If product hypothesis holds: iso-product rows in the heatmap should have
approximately equal sg4 values. If it fails, the 2D heatmap will show
independent nu and SB gradients.

Fixed: WR=4.8, WAVE_DUR=15, SS=10, FIELD_DECAY=0.9997, FIELD_ALPHA=0.16, KAPPA=0.020.
Varied: nu in [0.001, 0.002, 0.005, 0.010] x SEED_BETA in [0.05, 0.10, 0.25, 0.50].
5 seeds per cell = 80 runs. Runtime ~3 min.
"""
import numpy as np, json, os, multiprocessing as mp

W, H = 40, 20; HALF = W // 2
HS = 2; N_ZONES = 4; ZONE_W = HALF // N_ZONES
N_ACT = HALF * H; STEPS = 2000; SHIFT = 1000; N_SEEDS = 5

MID_DECAY = 0.99; FIELD_DECAY = 0.9997; BASE_BETA = 0.005
ALPHA_MID = 0.15; FIELD_ALPHA = 0.16
SS = 10; KAPPA = 0.020; WR = 4.8; WAVE_DUR = 15

NU_VALS = [0.001, 0.002, 0.005, 0.010]
SB_VALS = [0.05, 0.10, 0.25, 0.50]

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
                h[i] = (1-seed_beta)*rng.normal(0,0.1,HS) + seed_beta*F[j]
                m[i] = 0; streak[i] = 0
                ttl[i] = float(rng.geometric(p=death_p))

        waves = [[w[0],w[1]-1,w[2],w[3]] for w in waves if w[1]-1 > 0]

    f_norm = float(np.mean(np.linalg.norm(F, axis=1)))
    return {"sg4": sg4_fn(F), "coll_rate": total_collapses / (N_ACT * STEPS),
            "f_norm": f_norm}


def _worker(args):
    return run(*args)


RESULTS_FILE = os.path.join(os.path.dirname(__file__), "results", "paper12_exp2_results.json")

if __name__ == "__main__":
    mp.freeze_support()

    if os.path.exists(RESULTS_FILE):
        results = json.load(open(RESULTS_FILE))
        print("Loaded cached results.")
    else:
        all_args = [(seed, nu, sb)
                    for sb in SB_VALS
                    for nu in NU_VALS
                    for seed in range(N_SEEDS)]
        with mp.Pool(processes=min(len(all_args), mp.cpu_count())) as pool:
            raw = pool.map(_worker, all_args)
        results = {}
        idx = 0
        for sb in SB_VALS:
            for nu in NU_VALS:
                key = f"{sb},{nu}"
                results[key] = []
                for seed in range(N_SEEDS):
                    results[key].append(raw[idx]); idx += 1
        os.makedirs(os.path.dirname(RESULTS_FILE), exist_ok=True)
        json.dump(results, open(RESULTS_FILE, "w"), indent=2)
        print("Saved results.")

    print(f"\nProduct collapse test (WR={WR}, WD={WAVE_DUR}, SS={SS}, {N_SEEDS} seeds)")
    print(f"\nsg4 heatmap (rows=nu, cols=SEED_BETA):")
    header = f"{'nu':>7} | " + " | ".join(f"SB={sb:.2f}" for sb in SB_VALS)
    print(header)
    print("-" * len(header))
    for nu in NU_VALS:
        row = []
        for sb in SB_VALS:
            key = f"{sb},{nu}"
            sg4_val = float(np.mean([v["sg4"] for v in results[key]]))
            row.append(f"{sg4_val:7.1f}")
        print(f"{nu:>7.4f} | " + " | ".join(row))

    print(f"\nProduct (nu*SB) heatmap -- checking iso-product similarity:")
    print(f"{'nu':>7} | " + " | ".join(f"SB={sb:.2f}" for sb in SB_VALS))
    print("-" * len(header))
    for nu in NU_VALS:
        row = []
        for sb in SB_VALS:
            row.append(f"{nu*sb:.5f}")
        print(f"{nu:>7.4f} | " + " | ".join(row))

    # Check iso-product pairs explicitly
    print("\nIso-product pair comparison (product=5e-4):")
    iso_pairs = [(0.001, 0.50), (0.002, 0.25), (0.005, 0.10), (0.010, 0.05)]
    for nu, sb in iso_pairs:
        key = f"{sb},{nu}"
        if key in results:
            sg4_val = float(np.mean([v["sg4"] for v in results[key]]))
            print(f"  nu={nu:.3f}, SB={sb:.2f}, product={nu*sb:.5f} -> sg4={sg4_val:.1f}")

    print("\nf_norm heatmap (rows=nu, cols=SEED_BETA) -- proxy for F_quality:")
    for nu in NU_VALS:
        row = []
        for sb in SB_VALS:
            key = f"{sb},{nu}"
            fn_val = float(np.mean([v["f_norm"] for v in results[key]]))
            row.append(f"{fn_val:7.4f}")
        print(f"{nu:>7.4f} | " + " | ".join(row))
