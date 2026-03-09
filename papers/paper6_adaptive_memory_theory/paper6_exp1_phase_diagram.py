"""
paper6_exp1_phase_diagram.py -- Experiment 1 for Paper 6

Phase diagram of adaptive memory: sweep over two effective axes.

Axis 1 (plasticity rate proxy): wave_ratio (WR) -- controls B/N = turnover/capacity
  - too low  -> rigid/frozen  (no updates)
  - moderate -> adaptive
  - too high -> chaotic       (over-perturbation erases structure)

Axis 2 (coherence scale proxy): kappa / nu proxy
  - Use fixed WR inside each row, vary kappa
  - At fixed medium WR, increasing kappa increases L ~ sqrt(kappa/nu)
  - At small grid, use LARGER range of kappa to see saturation

But the key insight from V87/V88/V93: the most important phase axis is WR alone
(B/N ratio), which produces a non-monotone (frozen -> adaptive -> chaotic) curve.
kappa adds coherence length but on a small grid the WR axis dominates.

Strategy:
  - Show the WR axis clearly at one fixed kappa: frozen(low) -> peak(mid) -> chaotic(high)
  - Show the kappa axis at one fixed WR:         low coherence -> plateau
  - Combined: 2 panels that actually show the regime transitions

Extended WR range: [0.1, 0.3, 0.8, 1.6, 3.2, 6.4, 12.8, 25.6]
  (V87 showed non-monotone sg4 vs wave_ratio at coll/site ~0.004 optimal)
Fixed kappa = 0.020.

Extended kappa range at fixed WR=3.2: [0.001, 0.003, 0.010, 0.020, 0.040, 0.080]
  (need large grid to see propagation failure: use 40x20 = 800 sites)

Grid: 40x20 = 800 sites (HALF=40, H=20), 4 zones x 10 cols.
Seeds: 3, steps: 1500.
Runtime: ~6-8 minutes.
Dependencies: numpy only.
"""
import numpy as np

# Grid -- larger to see propagation effects
HALF, H = 40, 20; HS = 2; N_ZONES = 4; ZONE_W = HALF // N_ZONES
N_ACT = HALF * H
STEPS = 1500; N_SEEDS = 3

MID_DECAY = 0.99; FIELD_DECAY = 0.9997
BASE_BETA = 0.005; ALPHA_MID = 0.15; FIELD_ALPHA = 0.16
SEED_BETA = 0.25; WAVE_DUR = 15; DEATH_P = 0.005; SS = 10

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

def sg4_norm(F):
    zm = [F[zone==z].mean(0) for z in range(N_ZONES)]
    pairs = [(i,j) for i in range(N_ZONES) for j in range(i+1,N_ZONES)]
    sg = float(np.mean([np.linalg.norm(zm[i]-zm[j]) for i,j in pairs]))
    sw = float(np.mean([np.linalg.norm(F[zone==z]-zm[z],axis=1).mean()
                        for z in range(N_ZONES)]))
    return sg/sw if sw>1e-9 else 0.0

def run(seed, kappa, wave_ratio):
    rng = np.random.default_rng(seed)
    h = rng.normal(0, 0.1, (N_ACT, HS))
    F = np.zeros((N_ACT, HS)); m = np.zeros((N_ACT, HS))
    base = h.copy(); streak = np.zeros(N_ACT, int)
    ttl = rng.geometric(p=DEATH_P, size=N_ACT).astype(float)
    waves = []; hist = []

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
                dF[i] = kappa * (F[NB[i]].mean(0) - F[i])
        F += dF

        ttl -= 1.0; ttl[pert] -= 1.0
        dead = np.where(ttl <= 0)[0]
        for i in dead:
            j = NB[i][rng.integers(len(NB[i]))] if len(NB[i]) else i
            h[i] = (1-SEED_BETA)*rng.normal(0,0.1,HS) + SEED_BETA*F[j]
            m[i] = 0; streak[i] = 0; ttl[i] = float(rng.geometric(p=DEATH_P))
        waves = [w for w in waves if (w.__setitem__(1,w[1]-1) or True) and w[1]>0]

        if step % 25 == 24:
            hist.append(sg4_norm(F))

    return float(np.mean(hist[-20:])) if hist else sg4_norm(F)


# === Panel A: WR sweep (frozen -> adaptive -> chaotic) ===
# Fixed kappa=0.020, vary WR across wide range
KAPPA_FIXED = 0.020
WR_VALS = [0.1, 0.3, 0.8, 1.6, 3.2, 6.4, 12.8, 25.6]

print("Panel A: WR sweep at fixed kappa=0.020")
print(f"Grid: {HALF}x{H} ({N_ACT} sites), {STEPS} steps, {N_SEEDS} seeds")
print(f"Expected: non-monotone sg4_norm (low at WR=0.1, peak near WR=3-6, low at WR=25)")
print()
print(f"{'WR':>6}  {'sg4_norm':>9}  regime")
wr_results = {}
for wr in WR_VALS:
    vals = [run(s, KAPPA_FIXED, wr) for s in range(N_SEEDS)]
    v = np.mean(vals)
    wr_results[wr] = v
    regime = "frozen" if wr <= 0.3 else ("chaotic" if wr >= 12.8 else "adaptive")
    print(f"{wr:>6.1f}  {v:>9.3f}  {regime}", flush=True)

peak_wr = max(wr_results, key=wr_results.get)
print(f"\nPeak sg4_norm={wr_results[peak_wr]:.3f} at WR={peak_wr}")

# === Panel B: kappa sweep (propagation failure at low kappa) ===
# Fixed WR=peak_wr (or WR=3.2 if peak is too high), vary kappa
WR_FIXED = min(peak_wr, 3.2)
KAPPA_VALS = [0.001, 0.003, 0.007, 0.015, 0.030, 0.060]

print()
print(f"Panel B: kappa sweep at fixed WR={WR_FIXED}")
print(f"Expected: sg4_norm increases with kappa (more coherent propagation)")
print()
print(f"{'kappa':>7}  {'sg4_norm':>9}  {'L_proxy':>8}")
for kappa in KAPPA_VALS:
    vals = [run(s, kappa, WR_FIXED) for s in range(N_SEEDS)]
    v = np.mean(vals)
    L_proxy = np.sqrt(kappa / DEATH_P)   # L ~ sqrt(kappa/nu)
    print(f"{kappa:>7.3f}  {v:>9.3f}  {L_proxy:>8.2f}", flush=True)

# === Combined summary ===
print()
print("Phase regime summary (Panel A):")
frozen_sg  = np.mean([wr_results[w] for w in WR_VALS if w <= 0.3])
chaotic_sg = np.mean([wr_results[w] for w in WR_VALS if w >= 12.8])
adapt_sg   = np.mean([wr_results[w] for w in WR_VALS if 0.8 <= w <= 6.4])
print(f"  frozen  (WR<=0.3):  sg4_norm = {frozen_sg:.3f}")
print(f"  adaptive (WR 0.8-6.4): sg4_norm = {adapt_sg:.3f}")
print(f"  chaotic (WR>=12.8): sg4_norm = {chaotic_sg:.3f}")
print(f"  adaptive / frozen  = {adapt_sg/max(frozen_sg,1e-6):.1f}x")
print(f"  adaptive / chaotic = {adapt_sg/max(chaotic_sg,1e-6):.1f}x")
