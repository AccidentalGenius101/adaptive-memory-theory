"""
paper7_exp4_regime_fingerprints.py -- Experiment 4 for Paper 7

Regime fingerprints: time-series signatures of each regime.
For each of the four canonical regimes, track sg4(t), collapse_rate(t),
and nonadj/adj(t) across the run. Shows qualitatively distinct trajectories:

  Frozen:      sg4 rises and stays; collapse_rate ~0; no dip at shift.
  Adaptive:    sg4 dips at shift then recovers; collapse_rate ~0.01.
  Chaotic:     sg4 never rises; collapse_rate ~0.1; no coherent structure.
  Fragmented:  sg4 low; nonadj/adj near 1; structure is local, not zonal.

Fragmented: KAPPA=0 (no field diffusion) + SEED_BETA=0 (no birth seeding).
This eliminates both propagation mechanisms, giving pure local structure.

Protocol: 2000 steps, pattern shift at 1000. Seed=42. Snapshots every 50 steps.
Dependencies: numpy only.
"""
import numpy as np, json, os

W, H = 40, 20; HALF = W // 2
HS = 2; N_ZONES = 4; ZONE_W = HALF // N_ZONES
N_ACT = HALF * H; STEPS = 2000; SHIFT = 1000
SNAP_EVERY = 50

MID_DECAY = 0.99; FIELD_DECAY = 0.9997; BASE_BETA = 0.005
ALPHA_MID = 0.15; FIELD_ALPHA = 0.16
WAVE_RATIO = 4.8; WAVE_DUR = 15; SS = 10

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

left_mask = zone_id <= 1; right_mask = zone_id >= 2
top_mask  = _row < H // 2; bot_mask  = _row >= H // 2
d_A = np.array([1.0, 0.0]); d_B = np.array([0.0, 1.0])

REGIMES = {
    "frozen":     {"death_p": 0.0001, "kappa": 0.020, "seed_beta": 0.25},
    "adaptive":   {"death_p": 0.005,  "kappa": 0.020, "seed_beta": 0.25},
    "chaotic":    {"death_p": 0.060,  "kappa": 0.020, "seed_beta": 0.25},
    "fragmented": {"death_p": 0.005,  "kappa": 0.000, "seed_beta": 0.00},
}

def sg4(F):
    means = [F[zone_id == z].mean(0) for z in range(N_ZONES)]
    pairs = [(i,j) for i in range(N_ZONES) for j in range(i+1,N_ZONES)]
    return float(np.mean([np.linalg.norm(means[i]-means[j]) for i,j in pairs]))

def nonadj_ratio(F):
    means = [F[zone_id == z].mean(0) for z in range(N_ZONES)]
    adj    = [np.linalg.norm(means[i]-means[i+1]) for i in range(N_ZONES-1)]
    nonadj = [np.linalg.norm(means[i]-means[j])
               for i in range(N_ZONES) for j in range(i+2, N_ZONES)]
    return float(np.mean(nonadj) / (np.mean(adj) + 1e-9))

def align_cosine(F, d, pos, neg):
    d = d / np.linalg.norm(d)
    contrast = F[pos].mean(0) - F[neg].mean(0)
    mag = float(np.linalg.norm(contrast))
    if mag < 1e-9: return 0.0
    return float(np.dot(contrast / mag, d))

def run_regime(name, death_p, kappa, seed_beta, seed=42):
    rng = np.random.default_rng(seed)
    h = rng.normal(0, 0.1, (N_ACT, HS))
    F = np.zeros((N_ACT, HS)); m = np.zeros((N_ACT, HS))
    base = h.copy(); streak = np.zeros(N_ACT, int)
    ttl = rng.geometric(p=max(death_p, 1e-6), size=N_ACT).astype(float)
    waves = []; snapshots = []; window_collapses = 0

    for step in range(STEPS):
        in_phase2 = step >= SHIFT
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

        pert = np.zeros(N_ACT, bool)
        for w in waves:
            if w[3]: pert |= (left_mask if w[0] <= 1 else right_mask)
            else:    pert |= (top_mask if w[0] == 0 else bot_mask)

        base += BASE_BETA * (h - base)
        for w in waves:
            if w[3]: mask = left_mask if w[0] <= 1 else right_mask
            else:    mask = top_mask if w[0] == 0 else bot_mask
            h[mask] += 0.3 * np.array(w[2])
        m[pert] += ALPHA_MID * (h - base)[pert]
        m *= MID_DECAY; streak[pert] = 0; streak[~pert] += 1
        ok = streak >= SS
        F[ok] += FIELD_ALPHA * (m[ok] - F[ok]); F *= FIELD_DECAY

        if kappa > 0:
            dF = np.zeros_like(F)
            for i in range(N_ACT):
                if len(NB[i]): dF[i] = kappa * (F[NB[i]].mean(0) - F[i])
            F += dF

        if death_p > 0:
            ttl -= 1.0; ttl[pert] -= 1.0
            dead = np.where(ttl <= 0)[0]
            window_collapses += len(dead)
            for i in dead:
                j = NB[i][rng.integers(len(NB[i]))] if len(NB[i]) else i
                h[i] = (1-seed_beta)*rng.normal(0,0.1,HS) + seed_beta*F[j]
                m[i] = 0; streak[i] = 0
                ttl[i] = float(rng.geometric(p=death_p))

        waves = [[w[0],w[1]-1,w[2],w[3]] for w in waves if w[1]-1 > 0]

        if (step + 1) % SNAP_EVERY == 0:
            snapshots.append({
                "step":      step + 1,
                "sg4":       sg4(F),
                "coll_rate": window_collapses / (N_ACT * SNAP_EVERY),
                "ratio":     nonadj_ratio(F),
                "adapt_B":   align_cosine(F, d_B, top_mask, bot_mask),
            })
            window_collapses = 0

    return snapshots


RESULTS_FILE = os.path.join(os.path.dirname(__file__), "results", "paper7_exp4_results.json")

if __name__ == "__main__":
    if os.path.exists(RESULTS_FILE):
        results = json.load(open(RESULTS_FILE))
        print("Loaded cached results.")
    else:
        results = {}
        for name, params in REGIMES.items():
            print(f"Running {name}...")
            results[name] = run_regime(name, **params)
        os.makedirs(os.path.dirname(RESULTS_FILE), exist_ok=True)
        json.dump(results, open(RESULTS_FILE, "w"), indent=2)
        print("Saved.")

    print(f"\nRegime fingerprints: sg4, collapse rate, adapt over time")
    print(f"Pattern shift at step {SHIFT}. adapt in [-1,1].\n")
    for name, snaps in results.items():
        early      = next(s for s in snaps if s["step"] >= 500)
        pre_shift  = next(s for s in snaps if s["step"] >= 950)
        post_shift = next(s for s in snaps if s["step"] >= 1200)
        final      = snaps[-1]
        p = REGIMES[name]
        print(f"[{name.upper():12s}]  death_p={p['death_p']:.4f}  kappa={p['kappa']:.3f}  seed_beta={p['seed_beta']:.2f}")
        print(f"  t= 500: sg4={early['sg4']:6.2f}  coll={early['coll_rate']:.4f}  ratio={early['ratio']:.3f}  adapt_B={early['adapt_B']:+.3f}")
        print(f"  t=1000: sg4={pre_shift['sg4']:6.2f}  coll={pre_shift['coll_rate']:.4f}  ratio={pre_shift['ratio']:.3f}  adapt_B={pre_shift['adapt_B']:+.3f}")
        print(f"  t=1200: sg4={post_shift['sg4']:6.2f}  coll={post_shift['coll_rate']:.4f}  ratio={post_shift['ratio']:.3f}  adapt_B={post_shift['adapt_B']:+.3f}")
        print(f"  t=2000: sg4={final['sg4']:6.2f}  coll={final['coll_rate']:.4f}  ratio={final['ratio']:.3f}  adapt_B={final['adapt_B']:+.3f}")
        print()
    print("Signatures:")
    print("  adaptive:   dip in sg4 at t=1000, recovery at t=2000, adapt_B>0 at end")
    print("  frozen:     sg4 stays high, adapt_B~0 at end (locked in Phase 1 pattern)")
    print("  chaotic:    sg4 ~0 throughout, high coll_rate")
    print("  fragmented: sg4 low-moderate, ratio<1.2 (no zone structure)")
