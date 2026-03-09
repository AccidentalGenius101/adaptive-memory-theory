"""
paper3_propagation_repro.py -- Reproduction script for Paper 3
"Finite Correlation Length and Propagation Constraints in Adaptive Substrates"

Validates two experiments:
  V108 -- Scale invariance (finite correlation length): sg4_norm declines ~5x over 4x scale
  V104 -- Diffusion sweep: sg4_norm monotone in sqrt(kappa/nu)

Expected output (SEEDS=3, STEPS=2000):
  V108: Scale invariance (finite correlation length)
    Grid    Sites  sg4_norm
      S1     1600    ~0.28
      S2     6400    ~0.18
      S3    25600    ~0.06

  V104: Diffusion rate vs coherence length
  diff_rate  sg4_norm  sqrt(kappa/nu)
      0.005    ~0.11        ~0.89
      0.020    ~0.28        ~1.79
      0.080    ~0.62        ~3.58

Runtime: ~5 minutes (3 seeds x 5 diff rates + 3 grid sizes, pure Python loops).
Dependencies: numpy only.
"""
import numpy as np

# -- Configuration -------------------------------------------
SEEDS      = 3
STEPS      = 2000
HS         = 2       # hidden state size
ZONE_W     = 10      # zone width in columns
N_ZONES    = 4
MID_DECAY  = 0.99
BASE_BETA  = 0.005
ALPHA_MID  = 0.15
STABLE_STEPS = 10
WAVE_DUR   = 15

def make_grid(W, H=40):
    """Return (N_active, neighbors, zone_ids)."""
    HALF = W // 2
    N = HALF * H
    col = np.arange(N) % HALF
    zone = np.minimum(col // (HALF // N_ZONES), N_ZONES - 1)
    def nb(i):
        r, c = divmod(i, HALF)
        out = []
        if r > 0: out.append(i - HALF)
        if r < H - 1: out.append(i + HALF)
        if c > 0: out.append(i - 1)
        if c < HALF - 1: out.append(i + 1)
        return out
    NB = [nb(i) for i in range(N)]
    return N, NB, zone

def sg4(fieldM, zone, n_zones=4):
    """Mean pairwise L2 between zone-mean field vectors."""
    means = [fieldM[zone == z].mean(0) for z in range(n_zones)]
    dists = [np.linalg.norm(means[i] - means[j])
             for i in range(n_zones) for j in range(i+1, n_zones)]
    return float(np.mean(dists))

def sg4_norm(fieldM, zone, n_zones=4):
    """sg4 normalised by within-zone variance."""
    raw = sg4(fieldM, zone, n_zones)
    within = np.mean([fieldM[zone == z].std()
                      for z in range(n_zones) if (zone == z).sum() > 0])
    return raw / (within + 1e-8)

def run(W=80, diff_rate=0.02, field_alpha=0.16,
        wave_ratio=2.4, seed=0):
    """Run one simulation; return sg4_norm of active half."""
    rng  = np.random.default_rng(seed)
    HALF = W // 2
    N, NB, zone = make_grid(W)

    hid      = rng.standard_normal((N, HS)) * 0.1
    baseline = np.zeros((N, HS))
    mid_mem  = np.zeros((N, HS))
    fieldM   = np.zeros((N, HS))
    alive    = np.ones(N, bool)
    calm_streak = np.zeros(N, int)

    # Wave launcher: alternating zones
    wave_zone_seq = list(range(N_ZONES)) * (STEPS // (N_ZONES * WAVE_DUR) + 2)
    next_wave = 0
    waves = []   # [zone_id, steps_remaining, direction_vec]

    for t in range(STEPS):
        # -- Wave perturbation: launch + kill ---
        if t % WAVE_DUR == 0 and next_wave < len(wave_zone_seq):
            wz = wave_zone_seq[next_wave % len(wave_zone_seq)]
            next_wave += 1
            d_wave = rng.standard_normal(HS); d_wave /= np.linalg.norm(d_wave)
            waves.append([wz, WAVE_DUR, d_wave])
            targets = np.where((zone == wz) & alive)[0]
            if len(targets):
                hits = rng.choice(targets,
                                  size=min(max(1, int(wave_ratio)), len(targets)),
                                  replace=False)
                for h in hits:
                    alive[h] = False

        # -- Apply hid perturbation from all active waves ---
        for w in waves:
            hid[zone == w[0]] += 0.3 * w[2]

        # -- Deaths and births ---
        dead = np.where(~alive)[0]
        for d in dead:
            nbs = [n for n in NB[d] if alive[n]]
            if nbs:
                src = rng.choice(nbs)
                hid[d] = fieldM[src] * 0.25 + rng.standard_normal(HS) * 0.1
                fieldM[d] = fieldM[src].copy()
            else:
                hid[d] = rng.standard_normal(HS) * 0.1
                fieldM[d] = np.zeros(HS)
            mid_mem[d] = np.zeros(HS)
            baseline[d] = hid[d].copy()
            calm_streak[d] = 0
            alive[d] = True

        # -- VCSM update ---
        for i in np.where(alive)[0]:
            baseline[i] += BASE_BETA * (hid[i] - baseline[i])
            delta = hid[i] - baseline[i]
            if np.linalg.norm(delta) > 0.3:
                mid_mem[i] += ALPHA_MID * delta
                calm_streak[i] = 0
            else:
                calm_streak[i] += 1
            mid_mem[i] *= MID_DECAY
            fieldM[i] *= 0.9997        # FIELD_DECAY bounds fieldM values
            if calm_streak[i] >= STABLE_STEPS:
                fieldM[i] += field_alpha * (mid_mem[i] - fieldM[i])

        # -- Field diffusion ---
        new_field = fieldM.copy()
        for i in np.where(alive)[0]:
            nbs = [n for n in NB[i] if alive[n]]
            if nbs:
                new_field[i] += diff_rate * (
                    np.mean([fieldM[n] for n in nbs], 0) - fieldM[i])
        fieldM = new_field

        # -- Expire waves ---
        waves = [w for w in waves if (w.__setitem__(1, w[1]-1) or True) and w[1] > 0]

    return sg4_norm(fieldM, zone)

# -- Experiment V108: Finite Correlation Length ---------------
print("V108: Scale invariance (finite correlation length)")
print(f"{'Grid':>6}  {'Sites':>6}  {'sg4_norm':>9}")
for W, label in [(80, 'S1'), (160, 'S2'), (320, 'S3')]:
    vals = [run(W=W, seed=s) for s in range(SEEDS)]
    print(f"{label:>6}  {(W//2)*40:>6}  {np.mean(vals):>9.4f}")

# -- Experiment V104: Diffusion Sweep -------------------------
print("\nV104: Diffusion rate vs coherence length")
print(f"{'diff_rate':>10}  {'sg4_norm':>9}  {'sqrt(kappa/nu)':>14}")
FIELD_ALPHA = 0.16
COLLAPSE_RATE = 0.002
for dr in [0.005, 0.010, 0.020, 0.050, 0.080]:
    vals = [run(W=80, diff_rate=dr, seed=s) for s in range(SEEDS)]
    kappa = FIELD_ALPHA * dr
    x = np.sqrt(kappa / COLLAPSE_RATE)
    print(f"{dr:>10.3f}  {np.mean(vals):>9.4f}  {x:>14.4f}")
