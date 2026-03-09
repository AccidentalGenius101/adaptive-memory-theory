"""
paper6_exp3_adaptive_vs_frozen.py -- Experiment 3 for Paper 6

Adaptive regime vs collapsed attractor (frozen weights) under distribution shift.

Protocol:
  Phase 1 (T=0..999):   Both systems learn a 2-zone spatial pattern (Left vs Right).
                         Waves perturb zone 0-1 with direction d_A, zones 2-3 with -d_A.
  Freeze point (T=1000): Frozen system: stop turnover, stop consolidation.
                         Adaptive system: continues normally.
  Phase 2 (T=1000..1999): New pattern: Top vs Bottom.
                           Waves perturb top half (rows 0..H/2-1) with d_B,
                           bottom half (rows H/2..H-1) with -d_B.
                           This is ORTHOGONAL to the Phase 1 pattern.

Metric: alignment score -- dot product between observed zone-mean field and
  the target direction for that zone under the current pattern.
  alignment > 0: field encodes current pattern.
  alignment near 0 or negative: field still encodes old (Phase 1) pattern.

Phase 1 alignment:  both should be high (both learn left/right)
Phase 2 alignment (end of run):
  Frozen:   still encodes left/right -> alignment with top/bottom pattern near zero
  Adaptive: has overwritten with top/bottom -> alignment with top/bottom pattern high

Grid: 20x20 = 400 sites. kappa=0.020, WR=4.8.
Seeds: 5, Steps: 2000.
Runtime: ~2-3 minutes.
Dependencies: numpy only.
"""
import numpy as np

W, H = 40, 20; HALF = W // 2
HS = 2; N_ZONES = 4; ZONE_W = HALF // N_ZONES
N_ACT = HALF * H; STEPS = 2000; N_SEEDS = 5
SHIFT_POINT = 1000

MID_DECAY = 0.99; FIELD_DECAY = 0.9997
BASE_BETA = 0.005; ALPHA_MID = 0.15; FIELD_ALPHA = 0.16
SEED_BETA = 0.25; WAVE_DUR = 15; DEATH_P = 0.005; SS = 10
KAPPA = 0.020; WAVE_RATIO = 4.8

_col = np.arange(N_ACT) % HALF
_row = np.arange(N_ACT) // HALF
zone_col = _col // ZONE_W          # 0-3 by column (left->right)
zone_row = (_row >= H // 2).astype(int) * 2   # 0 or 2 by row (top=0, bottom=2)
# We use zone_col for Phase 1, zone_row for Phase 2 assignment

def _nb(i):
    c, r = _col[i], i // HALF
    out = []
    for dc, dr in [(-1,0),(1,0),(0,-1),(0,1)]:
        nc, nr = c+dc, r+dr
        if 0 <= nc < HALF and 0 <= nr < H:
            out.append(nr*HALF+nc)
    return np.array(out, dtype=int)

NB = [_nb(i) for i in range(N_ACT)]

def alignment_score(F, target_dir, positive_mask, negative_mask):
    """
    Mean projection of zone-mean fieldM onto target direction.
    positive_mask sites should have field aligned with target_dir,
    negative_mask sites should have field anti-aligned.
    Returns mean |projection| (strength) with sign convention:
      +1 = perfectly aligned, 0 = no encoding, -1 = reversed.
    """
    d = np.array(target_dir, dtype=float)
    d /= np.linalg.norm(d)
    pos_mean = F[positive_mask].mean(0) if positive_mask.any() else np.zeros(HS)
    neg_mean = F[negative_mask].mean(0) if negative_mask.any() else np.zeros(HS)
    return float(np.dot(pos_mean, d) - np.dot(neg_mean, d))

def run(seed, frozen_after_shift):
    rng = np.random.default_rng(seed)
    h = rng.normal(0, 0.1, (N_ACT, HS))
    F = np.zeros((N_ACT, HS)); m = np.zeros((N_ACT, HS))
    base = h.copy(); streak = np.zeros(N_ACT, int)
    ttl = rng.geometric(p=DEATH_P, size=N_ACT).astype(float)
    waves = []

    # Phase 1 pattern: Left (zones 0-1) vs Right (zones 2-3), direction d_A
    d_A = np.array([1.0, 0.0])   # fixed for determinism
    left_mask  = (zone_col <= 1)
    right_mask = (zone_col >= 2)

    # Phase 2 pattern: Top (rows 0..H/2-1) vs Bottom (rows H/2..H-1), direction d_B
    d_B = np.array([0.0, 1.0])   # orthogonal to d_A
    top_mask    = (_row < H // 2)
    bottom_mask = (_row >= H // 2)

    align_phase1_end = None

    for step in range(STEPS):
        in_phase2 = step >= SHIFT_POINT
        do_consolidate = not (frozen_after_shift and in_phase2)
        dead_p = 0.0 if (frozen_after_shift and in_phase2) else DEATH_P

        # Wave launch -- direction encodes current pattern
        n = int(WAVE_RATIO / WAVE_DUR)
        n += int(rng.random() < (WAVE_RATIO / WAVE_DUR - n))
        for _ in range(n):
            if not in_phase2:
                # Phase 1: perturb by column zone
                z = int(rng.integers(N_ZONES))
                sign = 1.0 if z <= 1 else -1.0
                d = sign * d_A
            else:
                # Phase 2: perturb by row zone (top/bottom)
                top = rng.random() < 0.5
                sign = 1.0 if top else -1.0
                d = sign * d_B
                z = 0 if top else 2   # dummy zone id for wave list
            waves.append([z, WAVE_DUR, d, not in_phase2])  # store phase flag

        pert = np.zeros(N_ACT, bool)
        for w in waves:
            # Phase 1 waves perturb column zones, phase 2 waves perturb row zones
            if w[3]:  # phase 1 wave
                col_z = w[0]
                pert |= (zone_col == col_z) | (zone_col == (1-col_z if col_z<=1 else 5-col_z))
                # simpler: wave targets left or right half
                if w[0] <= 1:
                    pert |= left_mask
                else:
                    pert |= right_mask
            else:     # phase 2 wave
                if w[0] == 0:
                    pert |= top_mask
                else:
                    pert |= bottom_mask

        # Deduplicate pert (OR already handles it)
        base += BASE_BETA * (h - base); m *= MID_DECAY

        for w in waves:
            if w[3]:  # phase 1
                mask = left_mask if w[0] <= 1 else right_mask
            else:     # phase 2
                mask = top_mask if w[0] == 0 else bottom_mask
            h[mask] += 0.3 * np.array(w[2])

        m[pert] += ALPHA_MID * (h - base)[pert]
        streak[pert] = 0; streak[~pert] += 1

        if do_consolidate:
            ok = streak >= SS
            F[ok] += FIELD_ALPHA * (m[ok] - F[ok])
        F *= FIELD_DECAY

        dF = np.zeros_like(F)
        for i in range(N_ACT):
            if len(NB[i]):
                dF[i] = KAPPA * (F[NB[i]].mean(0) - F[i])
        F += dF

        if dead_p > 0:
            ttl -= 1.0; ttl[pert] -= 1.0
            dead = np.where(ttl <= 0)[0]
            for i in dead:
                j = NB[i][rng.integers(len(NB[i]))] if len(NB[i]) else i
                h[i] = (1-SEED_BETA)*rng.normal(0,0.1,HS) + SEED_BETA*F[j]
                m[i] = 0; streak[i] = 0; ttl[i] = float(rng.geometric(p=DEATH_P))

        waves = [w for w in waves if (w.__setitem__(1,w[1]-1) or True) and w[1]>0]

        if step == SHIFT_POINT - 1:
            align_phase1_end = alignment_score(F, d_A, left_mask, right_mask)

    # End of Phase 2: measure alignment with Phase 2 pattern (top/bottom)
    align_phase2_end = alignment_score(F, d_B, top_mask, bottom_mask)
    return align_phase1_end, align_phase2_end


print("Adaptive vs frozen under distribution shift (orthogonal pattern)")
print(f"Grid: {HALF}x{H} ({N_ACT} sites), shift at step {SHIFT_POINT}/{STEPS}, {N_SEEDS} seeds")
print(f"Phase 1: Left/Right encoded in direction d_A=[1,0]")
print(f"Phase 2: Top/Bottom encoded in direction d_B=[0,1] (orthogonal)")
print()

pre_adapt = []; post_adapt = []
pre_frozen = []; post_frozen = []

for seed in range(N_SEEDS):
    p1, p2 = run(seed, frozen_after_shift=False)
    pre_adapt.append(p1); post_adapt.append(p2)
    f1, f2 = run(seed, frozen_after_shift=True)
    pre_frozen.append(f1); post_frozen.append(f2)

print(f"Alignment score = projection of zone-mean fieldM onto current target direction")
print(f"(higher = field encodes current spatial pattern)")
print()
print(f"{'':16}  {'Phase1 align':>13}  {'Phase2 align':>13}")
print(f"{'Adaptive':16}  {np.mean(pre_adapt):>13.4f}  {np.mean(post_adapt):>13.4f}")
print(f"{'Frozen':16}  {np.mean(pre_frozen):>13.4f}  {np.mean(post_frozen):>13.4f}")
print()
print("Prediction:")
print("  Phase 1: Adaptive ~ Frozen  (both encode Left/Right pattern)")
print("  Phase 2: Adaptive >> Frozen (adaptive rewrites; frozen retains Left/Right -> near-zero Top/Bottom alignment)")
print()
a2 = np.mean(post_adapt); f2 = np.mean(post_frozen)
a1 = np.mean(pre_adapt);  f1 = np.mean(pre_frozen)
print(f"Phase 1 parity: adaptive={a1:.4f}, frozen={f1:.4f}  (should be similar)")
print(f"Phase 2 shift:  adaptive={a2:.4f}, frozen={f2:.4f}")
if a2 > 2 * max(f2, 1e-6):
    print(f"CONFIRMED: adaptive {a2:.4f} >> frozen {f2:.4f} ({a2/max(f2,1e-6):.1f}x)")
elif a2 > f2:
    print(f"Partial: adaptive {a2:.4f} > frozen {f2:.4f} ({a2/max(f2,1e-6):.1f}x)")
else:
    print(f"Not confirmed: adaptive={a2:.4f}, frozen={f2:.4f}")
