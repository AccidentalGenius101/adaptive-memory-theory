"""
paper7_exp6_ablation.py -- Experiment 6 for Paper 7

Copy-forward ablation: causal validation of the intergenerational field-seeding mechanism.

The copy-forward loop is the claim that birth events propagate field structure forward
across generations. It is implemented via SEED_BETA: at rebirth, h_new blends debris
noise with the fieldM at the birth site (h_new = (1-sb)*noise + sb*F[neighbor]).

To test causality, we ablate the loop by setting SEED_BETA=0 -- births use pure noise,
so no field structure is inherited across generations. Field diffusion (KAPPA=0.020)
remains active in both conditions, isolating copy-forward from within-lifetime diffusion.

If the adaptive peak in sg4 (non-monotone with nu) disappears under ablation,
then copy-forward is causally necessary for the peak -- not just correlated with it.

Conditions:
  standard: SEED_BETA=0.25, KAPPA=0.020 (copy-forward active)
  ablated:  SEED_BETA=0.00, KAPPA=0.020 (births = pure noise, no field inheritance)

Protocol: same turnover sweep as Exp1 (DEATH_PS x 5 seeds x 2 conditions = 90 runs).
Grid: 40x20, active half=400 sites, HS=2, 4 zones, 5 seeds.
Dependencies: numpy only.
Runtime: ~6 minutes.
"""
import numpy as np, json, os, multiprocessing as mp

W, H = 40, 20; HALF = W // 2
HS = 2; N_ZONES = 4; ZONE_W = HALF // N_ZONES
N_ACT = HALF * H; STEPS = 2000; SHIFT = 1000; N_SEEDS = 5

MID_DECAY = 0.99; FIELD_DECAY = 0.9997; BASE_BETA = 0.005
ALPHA_MID = 0.15; FIELD_ALPHA = 0.16
KAPPA = 0.020; WAVE_RATIO = 4.8; WAVE_DUR = 15; SS = 10

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

def sg4(F):
    means = [F[zone_id == z].mean(0) for z in range(N_ZONES)]
    pairs = [(i,j) for i in range(N_ZONES) for j in range(i+1,N_ZONES)]
    return float(np.mean([np.linalg.norm(means[i]-means[j]) for i,j in pairs]))

def run(seed, death_p, seed_beta):
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

        # Field diffusion (active in both standard and ablated)
        dF = np.zeros_like(F)
        for i in range(N_ACT):
            if len(NB[i]):
                dF[i] = KAPPA * (F[NB[i]].mean(0) - F[i])
        F += dF

        # Turnover: seed_beta=0 in ablated condition (pure noise birth)
        if death_p > 0:
            ttl -= 1.0; ttl[pert] -= 1.0
            dead = np.where(ttl <= 0)[0]
            total_collapses += len(dead)
            for i in dead:
                j = NB[i][rng.integers(len(NB[i]))] if len(NB[i]) else i
                # seed_beta=0: h_new = noise (no copy-forward)
                # seed_beta=0.25: h_new = 0.75*noise + 0.25*F[j] (copy-forward active)
                h[i] = (1-seed_beta)*rng.normal(0,0.1,HS) + seed_beta*F[j]
                m[i] = 0; streak[i] = 0
                ttl[i] = float(rng.geometric(p=death_p))

        waves = [[w[0],w[1]-1,w[2],w[3]] for w in waves if w[1]-1 > 0]

    return {
        "sg4":       sg4(F),
        "coll_rate": total_collapses / (N_ACT * STEPS),
    }


def _worker(args):
    return run(*args)

DEATH_PS = [0.0001, 0.0003, 0.001, 0.002, 0.005, 0.010, 0.020, 0.040, 0.080]
CONDITIONS = {"standard": 0.25, "ablated": 0.00}
RESULTS_FILE = os.path.join(os.path.dirname(__file__), "results", "paper7_exp6_results.json")

if __name__ == "__main__":
    mp.freeze_support()

    if os.path.exists(RESULTS_FILE):
        results = json.load(open(RESULTS_FILE))
        print("Loaded cached results.")
    else:
        all_args = [(seed, dp, sb)
                    for cond, sb in CONDITIONS.items()
                    for dp in DEATH_PS
                    for seed in range(N_SEEDS)]
        with mp.Pool(processes=min(len(all_args), mp.cpu_count())) as pool:
            raw = pool.map(_worker, all_args)
        results = {}
        idx = 0
        for cond, sb in CONDITIONS.items():
            for dp in DEATH_PS:
                key = f"{cond},{dp}"
                results[key] = []
                for seed in range(N_SEEDS):
                    results[key].append(raw[idx]); idx += 1
        os.makedirs(os.path.dirname(RESULTS_FILE), exist_ok=True)
        json.dump(results, open(RESULTS_FILE, "w"), indent=2)
        print("Saved results.")

    print(f"\nCopy-forward ablation: sg4 with/without field seeding at birth ({N_SEEDS} seeds)")
    print(f"KAPPA=0.020 active in both conditions (only birth seeding differs)")
    print(f"\n{'death_p':>9} | {'sg4 standard':>13} | {'sg4 ablated':>12} | {'ratio std/abl':>13} | regime")
    print("-" * 68)
    for dp in DEATH_PS:
        std = results[f"standard,{dp}"]
        abl = results[f"ablated,{dp}"]
        s4_std = float(np.mean([v["sg4"] for v in std]))
        s4_abl = float(np.mean([v["sg4"] for v in abl]))
        cr     = float(np.mean([v["coll_rate"] for v in std]))
        ratio  = s4_std / (s4_abl + 1e-9)
        if cr < 0.001:   regime = "frozen"
        elif cr > 0.015: regime = "turnover-dominated"
        else:            regime = "adaptive"
        print(f"{dp:>9.4f} | {s4_std:>13.4f} | {s4_abl:>12.4f} | {ratio:>13.3f} | {regime}")

    print()
    print("Prediction:")
    print("  adaptive regime (mid nu): std >> abl (copy-forward amplifies zone structure)")
    print("  frozen/turnover-dominated: std ~ abl (copy-forward not the bottleneck)")
    print("  Peak in std at mid nu should weaken or disappear in abl.")
