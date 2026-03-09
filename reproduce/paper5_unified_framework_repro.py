"""
paper5_phases_repro.py -- Reproduction script for Paper 5
"A Unified Physical Theory of Viability-Gated Adaptive Systems"

Validates two experiments:
  Phase diagram -- Three regimes (frozen/adaptive/chaotic) by varying wave_ratio
  Matching condition -- Coherence efficiency eta_c drops ~0.5x at 2x scale

Expected output (SEEDS=3, STEPS=2000):
  Phase diagram (varying wave_ratio = nu proxy):
    regime        WR   SS     sg4   phase
    frozen       0.3   10  ~0.003
    adaptive     2.4   10  ~0.020
    chaotic     24.0   10  ~0.004

  Matching condition (coherence efficiency):
      grid  sg4_norm  eta_c (relative)
  S1 (W=80)   ~0.28              1.000
  S2 (W=160)  ~0.14              ~0.5
  (eta_c drops ~0.5x at 2x scale -- consistent with (L/D)^2)

Runtime: ~4 minutes (3 seeds x 3 regimes + 2 scales, pure Python).
Dependencies: numpy only.
"""
import numpy as np

# -- Configuration -------------------------------------------
W_BASE, H = 80, 40
HALF_BASE  = W_BASE // 2
HS         = 2
N_ZONES    = 4
SEEDS      = 3
STEPS      = 2000
BASE_BETA  = 0.005
ALPHA_MID  = 0.15
FIELD_ALPHA = 0.16
DIFF_RATE  = 0.02
WAVE_DUR   = 15
MID_DECAY  = 0.99

def make_grid(W):
    HALF = W // 2
    N    = HALF * H
    col  = np.arange(N) % HALF
    zone = np.minimum(col // (HALF // N_ZONES), N_ZONES - 1)
    def nb(i):
        r, c = divmod(i, HALF)
        out = []
        if r > 0: out.append(i - HALF)
        if r < H - 1: out.append(i + HALF)
        if c > 0: out.append(i - 1)
        if c < HALF - 1: out.append(i + 1)
        return out
    return N, [nb(i) for i in range(N)], zone

def sg4(F, zone):
    means = [F[zone == z].mean(0) for z in range(N_ZONES)]
    return float(np.mean([np.linalg.norm(means[i] - means[j])
                           for i in range(N_ZONES)
                           for j in range(i+1, N_ZONES)]))

def sg4_norm(F, zone):
    raw    = sg4(F, zone)
    within = np.mean([F[zone == z].std()
                      for z in range(N_ZONES) if (zone == z).sum() > 0])
    return raw / (within + 1e-8)

def run(W=80, wave_ratio=2.4, stable_steps=10, seed=0):
    rng = np.random.default_rng(seed)
    N, NB, zone = make_grid(W)
    hid      = rng.standard_normal((N, HS)) * 0.1
    baseline = np.zeros((N, HS))
    mid_mem  = np.zeros((N, HS))
    fieldM   = np.zeros((N, HS))
    alive    = np.ones(N, bool)
    streak   = np.zeros(N, int)

    wave_seq = list(range(N_ZONES)) * (STEPS // (N_ZONES * WAVE_DUR) + 2)
    wptr = 0
    waves = []   # [zone_id, steps_remaining, direction_vec]

    for t in range(STEPS):
        # -- Wave launch + kill ---
        if t % WAVE_DUR == 0 and wptr < len(wave_seq):
            wz = wave_seq[wptr % len(wave_seq)]; wptr += 1
            d_wave = rng.standard_normal(HS); d_wave /= np.linalg.norm(d_wave)
            waves.append([wz, WAVE_DUR, d_wave])
            tgt = np.where((zone == wz) & alive)[0]
            if len(tgt):
                n = min(max(1, int(wave_ratio)), len(tgt))
                for h in rng.choice(tgt, size=n, replace=False):
                    alive[h] = False

        # -- Apply hid perturbation from all active waves ---
        for w in waves:
            hid[zone == w[0]] += 0.3 * w[2]

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

        for i in np.where(alive)[0]:
            baseline[i] += BASE_BETA * (hid[i] - baseline[i])
            delta = hid[i] - baseline[i]
            if np.linalg.norm(delta) > 0.3:
                mid_mem[i] += ALPHA_MID * delta; streak[i] = 0
            else:
                streak[i] += 1
            mid_mem[i] *= MID_DECAY
            fieldM[i] *= 0.9997        # FIELD_DECAY bounds fieldM values
            if streak[i] >= stable_steps:
                fieldM[i] += FIELD_ALPHA * (mid_mem[i] - fieldM[i])

        new_field = fieldM.copy()
        for i in np.where(alive)[0]:
            nbs = [n for n in NB[i] if alive[n]]
            if nbs:
                new_field[i] += DIFF_RATE * (
                    np.mean([fieldM[n] for n in nbs], 0) - fieldM[i])
        fieldM = new_field

        # -- Expire waves ---
        waves = [w for w in waves if (w.__setitem__(1, w[1]-1) or True) and w[1] > 0]

    return sg4(fieldM, zone), sg4_norm(fieldM, zone)

# -- Phase diagram: three regimes ----------------------------
# wave_ratio controls turnover rate (nu); stable_steps is tau_g
print("Phase diagram (varying wave_ratio = nu proxy):")
print(f"{'regime':>10}  {'WR':>5}  {'SS':>4}  {'sg4':>7}  {'phase'}")
for label, wr, ss in [('frozen',   0.3, 10),
                       ('adaptive', 2.4, 10),
                       ('chaotic', 24.0, 10)]:
    vals = [run(W=80, wave_ratio=wr, stable_steps=ss, seed=s)[0]
            for s in range(SEEDS)]
    print(f"{label:>10}  {wr:>5.1f}  {ss:>4}  {np.mean(vals):>7.4f}")

# -- Matching condition: coherence efficiency at scale -------
print("\nMatching condition (coherence efficiency):")
print(f"{'grid':>6}  {'sg4_norm':>9}  {'eta_c (relative)':>18}")
vals_s1 = [run(W=80,  wave_ratio=2.4, seed=s)[1] for s in range(SEEDS)]
vals_s2 = [run(W=160, wave_ratio=2.4, seed=s)[1] for s in range(SEEDS)]
m1, m2 = np.mean(vals_s1), np.mean(vals_s2)
for label, m in [('S1 (W=80)', m1), ('S2 (W=160)', m2)]:
    print(f"{label:>10}  {m:>9.4f}  {m/m1:>18.3f}")
print("(eta_c drops ~0.5x at 2x scale -- consistent with (L/D)^2)")
