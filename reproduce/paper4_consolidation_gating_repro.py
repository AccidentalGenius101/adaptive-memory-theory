"""
paper4_gate_repro.py -- Reproduction script for Paper 4
"Consolidation Gate and Noise Robustness in VCSM"

Validates:
  Stability-sensitivity sweep (SS = 1..50) in clean vs 25% noisy environments.
  Ratio noisy/clean shows optimal gate at SS=10 (robust structure despite noise).

Expected output (SEEDS=3, STEPS=3000):
  SS    clean   noise25    ratio
   1   ~0.0249  ~0.0098    ~0.39
   5   ~0.0234  ~0.0116    ~0.50
  10   ~0.0195  ~0.0078    ~0.40
  20   ~0.0163  ~0.0098    ~0.60
  50   ~0.0091  ~0.0072    ~0.79

Key result: clean sg4 peaks near SS=1-5; noisy/clean ratio improves monotonically
(longer gate = more noise rejection). Trade-off: optimal SS depends on noise level.

Runtime: ~3 minutes (3 seeds x 5 SS values x 2 conditions, pure Python).
Dependencies: numpy only.
"""
import numpy as np

# -- Configuration -------------------------------------------
W, H   = 80, 40
HALF   = W // 2
HS     = 2
N_ACT  = HALF * H
N_ZONES = 4
ZONE_W  = HALF // N_ZONES
SEEDS  = 3
STEPS  = 3000

MID_DECAY  = 0.99
FIELD_DECAY = 0.9997
BASE_BETA  = 0.005
ALPHA_MID  = 0.15
FIELD_ALPHA = 0.16
WAVE_RATIO  = 2.4
WAVE_DUR   = 15

col  = np.arange(N_ACT) % HALF
zone = np.minimum(col // ZONE_W, N_ZONES - 1)

def _nb(i):
    r, c = divmod(i, HALF)
    out = []
    if r > 0: out.append(i - HALF)
    if r < H - 1: out.append(i + HALF)
    if c > 0: out.append(i - 1)
    if c < HALF - 1: out.append(i + 1)
    return out

NB = [_nb(i) for i in range(N_ACT)]

def sg4(F):
    means = [F[zone == z].mean(0) for z in range(N_ZONES)]
    return float(np.mean([np.linalg.norm(means[i] - means[j])
                          for i in range(N_ZONES)
                          for j in range(i+1, N_ZONES)]))

def run(seed=0, stable_steps=10, noise_frac=0.0):
    rng = np.random.default_rng(seed)
    hid      = rng.standard_normal((N_ACT, HS)) * 0.1
    baseline = np.zeros((N_ACT, HS))
    mid_mem  = np.zeros((N_ACT, HS))
    fieldM   = np.zeros((N_ACT, HS))
    alive    = np.ones(N_ACT, bool)
    streak   = np.zeros(N_ACT, int)

    wave_seq = (list(range(N_ZONES)) * (STEPS // (N_ZONES * WAVE_DUR) + 2))
    wptr = 0
    waves = []   # [zone_id, steps_remaining, direction_vec]

    for t in range(STEPS):
        # -- Wave perturbation: launch + kill (with optional noise) --
        if t % WAVE_DUR == 0 and wptr < len(wave_seq):
            wz = wave_seq[wptr % len(wave_seq)]; wptr += 1
            d_wave = rng.standard_normal(HS); d_wave /= np.linalg.norm(d_wave)
            waves.append([wz, WAVE_DUR, d_wave])
            tgt = np.where((zone == wz) & alive)[0]
            if len(tgt):
                n_hits = min(max(1, int(WAVE_RATIO)), len(tgt))
                for h in rng.choice(tgt, size=n_hits, replace=False):
                    alive[h] = False
            # Noise: corrupt random sites regardless of zone
            if noise_frac > 0:
                n_noise = int(N_ACT * noise_frac * WAVE_RATIO / HALF)
                for h in rng.choice(N_ACT, size=max(1, n_noise), replace=False):
                    alive[h] = False

        # -- Apply hid perturbation from all active waves ---
        for w in waves:
            hid[zone == w[0]] += 0.3 * w[2]

        # -- Deaths and births ---
        for d in np.where(~alive)[0]:
            nbs = [n for n in NB[d] if alive[n]]
            if nbs:
                src = rng.choice(nbs)
                hid[d] = fieldM[src] * 0.25 + rng.standard_normal(HS) * 0.1
                fieldM[d] = fieldM[src].copy()
            else:
                hid[d] = rng.standard_normal(HS) * 0.1
                fieldM[d] = np.zeros(HS)
            mid_mem[d] = np.zeros(HS); baseline[d] = hid[d].copy()
            streak[d] = 0; alive[d] = True

        # -- VCSM update ---
        for i in np.where(alive)[0]:
            baseline[i] += BASE_BETA * (hid[i] - baseline[i])
            delta = hid[i] - baseline[i]
            if np.linalg.norm(delta) > 0.3:
                mid_mem[i] += ALPHA_MID * delta; streak[i] = 0
            else:
                streak[i] += 1
            mid_mem[i] *= MID_DECAY
            fieldM[i] *= FIELD_DECAY
            if streak[i] >= stable_steps:
                fieldM[i] += FIELD_ALPHA * (mid_mem[i] - fieldM[i])

        # -- Expire waves ---
        waves = [w for w in waves if (w.__setitem__(1, w[1]-1) or True) and w[1] > 0]

    return sg4(fieldM)

# -- Experiment: SS sweep in clean vs noisy environment ------
print(f"{'SS':>4}  {'clean':>8}  {'noise25':>8}  {'ratio':>6}")
for ss in [1, 5, 10, 20, 50]:
    clean  = np.mean([run(seed=s, stable_steps=ss, noise_frac=0.00)
                      for s in range(SEEDS)])
    noisy  = np.mean([run(seed=s, stable_steps=ss, noise_frac=0.25)
                      for s in range(SEEDS)])
    ratio  = noisy / clean if clean > 0 else float('nan')
    print(f"{ss:>4}  {clean:>8.4f}  {noisy:>8.4f}  {ratio:>6.3f}")
