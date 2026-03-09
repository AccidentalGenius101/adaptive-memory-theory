"""
paper6_exp2_scale_limit.py -- Experiment 2 for Paper 6

Flat substrate scale limit and hierarchy recovery.

Part A (scale limit): Run flat VCML at 3 grid sizes.
  Measure eta_c = sg4_norm (coherent structure fraction).
  Prediction: eta_c ~ (L/D)^2 -- declines as grid grows.

Part B (hierarchy recovery): At the largest grid, compare:
  - Flat substrate:    no relay between halves
  - Relay substrate:   two 1/2-size regions that copy their zone means to each other
  Prediction: relay recovers eta_c close to small-grid baseline.

Grid sizes (active sites):
  S1:  20x20 =   400  (HALF=20, 4 zones x 5 cols)
  S2:  40x20 =   800  (HALF=40, 4 zones x 10 cols)
  S3:  80x20 =  1600  (HALF=80, 4 zones x 20 cols)

WR scales with active sites (2x density from V93: optimal WR ~ 2*N/1600 * 4.8 base).
kappa fixed at 0.020 (near optimal from Exp 1).

Runtime: ~6-8 minutes.
Dependencies: numpy only.
"""
import numpy as np

# Fixed params
HS = 2; N_ZONES = 4
MID_DECAY = 0.99; FIELD_DECAY = 0.9997
BASE_BETA = 0.005; ALPHA_MID = 0.15; FIELD_ALPHA = 0.16
SEED_BETA = 0.25; WAVE_DUR = 15; DEATH_P = 0.005; SS = 10
KAPPA = 0.020
STEPS = 2000; N_SEEDS = 3

def make_grid(HALF, H):
    N_ACT = HALF * H
    ZONE_W = HALF // N_ZONES
    _col = np.arange(N_ACT) % HALF
    zone = _col // ZONE_W
    def _nb(i):
        c, r = _col[i], i // HALF
        out = []
        for dc, dr in [(-1,0),(1,0),(0,-1),(0,1)]:
            nc, nr = c+dc, r+dr
            if 0 <= nc < HALF and 0 <= nr < H:
                out.append(nr*HALF+nc)
        return np.array(out, dtype=int)
    NB = [_nb(i) for i in range(N_ACT)]
    return N_ACT, zone, NB

def sg4_norm(F, zone):
    zm = [F[zone==z].mean(0) for z in range(N_ZONES)]
    pairs = [(i,j) for i in range(N_ZONES) for j in range(i+1,N_ZONES)]
    sg = float(np.mean([np.linalg.norm(zm[i]-zm[j]) for i,j in pairs]))
    sw = float(np.mean([np.linalg.norm(F[zone==z] - zm[z], axis=1).mean()
                        for z in range(N_ZONES)]))
    return (sg/sw if sw > 1e-9 else 0.0), sg

def run_flat(seed, HALF, H, wave_ratio):
    rng = np.random.default_rng(seed)
    N_ACT, zone, NB = make_grid(HALF, H)
    h = rng.normal(0, 0.1, (N_ACT, HS))
    F = np.zeros((N_ACT, HS)); m = np.zeros((N_ACT, HS))
    base = h.copy(); streak = np.zeros(N_ACT, int)
    ttl = rng.geometric(p=DEATH_P, size=N_ACT).astype(float)
    waves = []; hist_norm = []; hist_sg = []

    for step in range(STEPS):
        n = int(wave_ratio / WAVE_DUR)
        n += int(rng.random() < (wave_ratio / WAVE_DUR - n))
        for _ in range(n):
            z = int(rng.integers(N_ZONES))
            d = rng.standard_normal(HS); d /= np.linalg.norm(d)
            waves.append([z, WAVE_DUR, d])

        pert = np.zeros(N_ACT, bool)
        for w in waves:
            pert |= (zone == w[0])

        base += BASE_BETA * (h - base); m *= MID_DECAY
        for w in waves:
            h[zone == w[0]] += 0.3 * w[2]
        m[pert] += ALPHA_MID * (h - base)[pert]

        streak[pert] = 0; streak[~pert] += 1
        ok = streak >= SS
        F[ok] += FIELD_ALPHA * (m[ok] - F[ok]); F *= FIELD_DECAY

        dF = np.zeros_like(F)
        for i in range(N_ACT):
            if len(NB[i]):
                dF[i] = KAPPA * (F[NB[i]].mean(0) - F[i])
        F += dF

        ttl -= 1.0; ttl[pert] -= 1.0
        dead = np.where(ttl <= 0)[0]
        for i in dead:
            j = NB[i][rng.integers(len(NB[i]))] if len(NB[i]) else i
            h[i] = (1-SEED_BETA)*rng.normal(0,0.1,HS) + SEED_BETA*F[j]
            m[i] = 0; streak[i] = 0; ttl[i] = float(rng.geometric(p=DEATH_P))
        waves = [w for w in waves if (w.__setitem__(1,w[1]-1) or True) and w[1]>0]

        if step % 25 == 24:
            n_val, s_val = sg4_norm(F, zone)
            hist_norm.append(n_val); hist_sg.append(s_val)

    n_mean = float(np.mean(hist_norm[-20:])) if hist_norm else 0.0
    s_mean = float(np.mean(hist_sg[-20:])) if hist_sg else 0.0
    return n_mean, s_mean

def run_relay(seed, HALF, H, wave_ratio):
    """
    Two half-width regions. Each runs its own substrate.
    Every 25 steps they exchange zone-mean fieldM (relay).
    relay_alpha=0.1: each region gently pulled toward the other's zone means.
    """
    rng = np.random.default_rng(seed)
    HALF_R = HALF // 2    # each region is half width
    N_ACT_R, zone_r, NB_r = make_grid(HALF_R, H)

    def init_region():
        h = rng.normal(0, 0.1, (N_ACT_R, HS))
        F = np.zeros((N_ACT_R, HS)); m = np.zeros((N_ACT_R, HS))
        base = h.copy(); streak = np.zeros(N_ACT_R, int)
        ttl = rng.geometric(p=DEATH_P, size=N_ACT_R).astype(float)
        return h, F, m, base, streak, ttl

    h0,F0,m0,b0,s0,t0 = init_region()
    h1,F1,m1,b1,s1,t1 = init_region()
    waves0 = []; waves1 = []
    RELAY_ALPHA = 0.10
    hist_norm = []

    for step in range(STEPS):
        # Same wave pattern for both (same zone structure)
        n = int(wave_ratio / WAVE_DUR)
        n += int(rng.random() < (wave_ratio / WAVE_DUR - n))
        for _ in range(n):
            z = int(rng.integers(N_ZONES))
            d = rng.standard_normal(HS); d /= np.linalg.norm(d)
            waves0.append([z, WAVE_DUR, d])
            waves1.append([z, WAVE_DUR, d])

        for (h, F, m, base, streak, ttl, waves) in [
            (h0,F0,m0,b0,s0,t0,waves0),
            (h1,F1,m1,b1,s1,t1,waves1)
        ]:
            pert = np.zeros(N_ACT_R, bool)
            for w in waves:
                pert |= (zone_r == w[0])
            base += BASE_BETA * (h - base); m *= MID_DECAY
            for w in waves:
                h[zone_r == w[0]] += 0.3 * w[2]
            m[pert] += ALPHA_MID * (h - base)[pert]
            streak[pert] = 0; streak[~pert] += 1
            ok = streak >= SS
            F[ok] += FIELD_ALPHA * (m[ok] - F[ok]); F *= FIELD_DECAY
            dF = np.zeros_like(F)
            for i in range(N_ACT_R):
                if len(NB_r[i]):
                    dF[i] = KAPPA * (F[NB_r[i]].mean(0) - F[i])
            F += dF
            ttl -= 1.0; ttl[pert] -= 1.0
            dead = np.where(ttl <= 0)[0]
            for i in dead:
                j = NB_r[i][rng.integers(len(NB_r[i]))] if len(NB_r[i]) else i
                h[i] = (1-SEED_BETA)*rng.normal(0,0.1,HS) + SEED_BETA*F[j]
                m[i] = 0; streak[i] = 0; ttl[i] = float(rng.geometric(p=DEATH_P))
            waves[:] = [w for w in waves if (w.__setitem__(1,w[1]-1) or True) and w[1]>0]

        # Relay: exchange zone means every 25 steps
        if step % 25 == 24:
            zm0 = np.array([F0[zone_r==z].mean(0) for z in range(N_ZONES)])
            zm1 = np.array([F1[zone_r==z].mean(0) for z in range(N_ZONES)])
            # Each site pulled toward other region's zone mean
            for z in range(N_ZONES):
                idx0 = zone_r == z; idx1 = zone_r == z
                F0[idx0] += RELAY_ALPHA * (zm1[z] - F0[idx0])
                F1[idx1] += RELAY_ALPHA * (zm0[z] - F1[idx1])

            # Measure on combined substrate (concatenate)
            F_all = np.vstack([F0, F1])
            zone_all = np.concatenate([zone_r, zone_r])
            n_val, _ = sg4_norm(F_all, zone_all)
            hist_norm.append(n_val)

    return float(np.mean(hist_norm[-20:])) if hist_norm else 0.0


# --- Part A: scale limit ---
GRIDS = [
    ("S1", 20, 20, 4.8),     # WR=4.8 ~ optimal at 400 sites (from V93 scaling)
    ("S2", 40, 20, 9.6),     # WR=9.6 ~ 2x N ratio
    ("S3", 80, 20, 19.2),    # WR=19.2 ~ 4x N ratio
]

print("Part A: Flat substrate scale limit")
print(f"kappa={KAPPA}, {STEPS} steps, {N_SEEDS} seeds")
print()
print(f"{'Grid':>5}  {'N_active':>8}  {'sg4_norm':>9}  {'sg4_raw':>8}")

s1_baseline = None
for gname, HALF, H, WR in GRIDS:
    N_ACT = HALF * H
    vals_n = []; vals_s = []
    for seed in range(N_SEEDS):
        n, s = run_flat(seed, HALF, H, WR)
        vals_n.append(n); vals_s.append(s)
    mn = np.mean(vals_n); ms = np.mean(vals_s)
    if s1_baseline is None:
        s1_baseline = mn
    print(f"{gname:>5}  {N_ACT:>8}  {mn:>9.3f}  {ms:>8.4f}")

print()
print("Prediction: sg4_norm declines with N (finite correlation length)")
print()

# --- Part B: hierarchy recovery at S3 ---
print("Part B: Hierarchy recovery at S3")
print(f"(relay_alpha=0.10, two {GRIDS[2][1]//2}x{GRIDS[2][2]} modules)")
print()

HALF_S3, H_S3, WR_S3 = GRIDS[2][1], GRIDS[2][2], GRIDS[2][3]
vals_flat = []; vals_relay = []
for seed in range(N_SEEDS):
    n_flat, _ = run_flat(seed, HALF_S3, H_S3, WR_S3)
    n_relay   = run_relay(seed, HALF_S3, H_S3, WR_S3 / 2)  # per-module WR
    vals_flat.append(n_flat); vals_relay.append(n_relay)

print(f"  S3 flat:  sg4_norm = {np.mean(vals_flat):.3f} +- {np.std(vals_flat):.3f}")
print(f"  S3 relay: sg4_norm = {np.mean(vals_relay):.3f} +- {np.std(vals_relay):.3f}")
if s1_baseline:
    print(f"  S1 baseline: {s1_baseline:.3f}")
    print(f"  Relay recovery ratio vs S1: {np.mean(vals_relay)/s1_baseline:.2f}")
print()
print("Prediction: relay sg4_norm > flat, approaching S1 baseline")
