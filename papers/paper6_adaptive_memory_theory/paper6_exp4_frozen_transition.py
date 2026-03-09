"""
Experiment 4: Adaptive-to-Frozen Transition
Sweeps a freezing parameter F in [0, 1] that continuously interpolates
between the fully adaptive regime (F=0) and the collapsed attractor (F=1).

Parameterization: after the shift point, scale both key dynamics by (1-F):
  death_p_eff      = BASE_DEATH_P * (1 - F)   [turnover rate -> 0]
  field_alpha_eff  = FIELD_ALPHA   * (1 - F)   [consolidation rate -> 0]
  (calm_streak stays fixed at BASE_SS; only the magnitude fades)

This produces a smooth continuous transition:
  F=0.00 -> fully adaptive (normal VCSM)
  F=0.25 -> reduced turnover + consolidation
  F=0.50 -> half-speed turnover + consolidation
  F=0.75 -> near-frozen (quarter rate)
  F=1.00 -> frozen (no dynamics after shift)

Task: learn Phase 1 pattern (left/right, steps 0-999),
      measure adaptation to orthogonal Phase 2 pattern (top/bottom, steps 1000-1999).
Prediction: align_B degrades monotonically with F.
"""

import numpy as np
import json, os

W, H = 40, 20; HALF = W // 2
HS = 2; N_ACT = HALF * H; STEPS = 2000; N_SEEDS = 5
SHIFT_POINT = 1000

MID_DECAY = 0.99; FIELD_DECAY = 0.9997
BASE_BETA = 0.005; ALPHA_MID = 0.15; BASE_FIELD_ALPHA = 0.16
SEED_BETA = 0.25; WAVE_DUR = 15; BASE_DEATH_P = 0.005; BASE_SS = 10
KAPPA = 0.020; WAVE_RATIO = 4.8

_col = np.arange(N_ACT) % HALF
_row = np.arange(N_ACT) // HALF
left_mask   = (_col < HALF // 2)
right_mask  = (_col >= HALF // 2)
top_mask    = (_row < H // 2)
bottom_mask = (_row >= H // 2)

def _nb(i):
    c, r = _col[i], i // HALF
    out = []
    for dc, dr in [(-1,0),(1,0),(0,-1),(0,1)]:
        nc, nr = c+dc, r+dr
        if 0 <= nc < HALF and 0 <= nr < H:
            out.append(nr*HALF+nc)
    return np.array(out, dtype=int)

NB = [_nb(i) for i in range(N_ACT)]

def alignment_score(Fld, target_dir, positive_mask, negative_mask):
    d = np.array(target_dir, dtype=float); d /= np.linalg.norm(d)
    pos_mean = Fld[positive_mask].mean(0)
    neg_mean = Fld[negative_mask].mean(0)
    return float(np.dot(pos_mean, d) - np.dot(neg_mean, d))

def run(seed, freeze_factor):
    rng = np.random.default_rng(seed)
    h = rng.normal(0, 0.1, (N_ACT, HS))
    Fld = np.zeros((N_ACT, HS)); m = np.zeros((N_ACT, HS))
    base = h.copy(); streak = np.zeros(N_ACT, int)
    ttl = rng.geometric(p=BASE_DEATH_P, size=N_ACT).astype(float)
    waves = []

    d_A = np.array([1.0, 0.0])
    d_B = np.array([0.0, 1.0])
    align_phase1_end = None

    for step in range(STEPS):
        in_phase2 = step >= SHIFT_POINT

        # After shift, scale adaptive dynamics by (1 - freeze_factor)
        if in_phase2:
            this_death_p   = BASE_DEATH_P   * (1.0 - freeze_factor)
            this_fa        = BASE_FIELD_ALPHA * (1.0 - freeze_factor)
        else:
            this_death_p   = BASE_DEATH_P
            this_fa        = BASE_FIELD_ALPHA

        # Wave launch
        n = int(WAVE_RATIO / WAVE_DUR)
        n += int(rng.random() < (WAVE_RATIO / WAVE_DUR - n))
        for _ in range(n):
            if not in_phase2:
                top_wave = rng.random() < 0.5
                sign = 1.0 if top_wave else -1.0
                w_mask = left_mask if top_wave else right_mask
                waves.append([w_mask, WAVE_DUR, sign * d_A])
            else:
                top_wave = rng.random() < 0.5
                sign = 1.0 if top_wave else -1.0
                w_mask = top_mask if top_wave else bottom_mask
                waves.append([w_mask, WAVE_DUR, sign * d_B])

        pert = np.zeros(N_ACT, bool)
        for w in waves:
            h[w[0]] += 0.3 * w[2]
            pert[w[0]] = True

        # VCSM update
        base += BASE_BETA * (h - base)
        m *= MID_DECAY
        m[pert] += ALPHA_MID * (h - base)[pert]
        streak[pert] = 0; streak[~pert] += 1

        ok = streak >= BASE_SS
        if np.any(ok):
            Fld[ok] += this_fa * (m[ok] - Fld[ok])
        Fld *= FIELD_DECAY

        # Coupling
        dFld = np.zeros_like(Fld)
        for i in range(N_ACT):
            if len(NB[i]):
                dFld[i] = KAPPA * (Fld[NB[i]].mean(0) - Fld[i])
        Fld += dFld

        # Turnover (copy-forward)
        if this_death_p > 0:
            ttl -= 1.0; ttl[pert] -= 1.0
            dead = np.where(ttl <= 0)[0]
            for i in dead:
                j = NB[i][rng.integers(len(NB[i]))] if len(NB[i]) else i
                h[i] = (1-SEED_BETA)*rng.normal(0,0.1,HS) + SEED_BETA*Fld[j]
                m[i] = 0; streak[i] = 0
                ttl[i] = float(rng.geometric(p=BASE_DEATH_P))

        waves = [w for w in waves if (w.__setitem__(1, w[1]-1) or True) and w[1] > 0]

        if step == SHIFT_POINT - 1:
            align_phase1_end = alignment_score(Fld, d_A, left_mask, right_mask)

    align_phase2_end = alignment_score(Fld, d_B, top_mask, bottom_mask)
    return align_phase1_end, align_phase2_end


if __name__ == '__main__':
    FREEZE_FACTORS = [0.0, 0.25, 0.50, 0.75, 1.0]
    RESULTS_FILE = os.path.join(os.path.dirname(__file__),
                                'results', 'paper6_exp4_results.json')
    os.makedirs(os.path.dirname(RESULTS_FILE), exist_ok=True)

    if os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE) as f:
            all_results = json.load(f)
        print("Loaded cached results.")
    else:
        all_results = {}

    for ff in FREEZE_FACTORS:
        key = f"{ff:.2f}"
        if key in all_results:
            print(f"F={ff:.2f}: cached ({np.mean(all_results[key]['align_B']):.1f})")
            continue
        p1s, p2s = [], []
        for seed in range(N_SEEDS):
            p1, p2 = run(seed, ff)
            p1s.append(p1); p2s.append(p2)
            print(f"F={ff:.2f}, seed={seed}: A={p1:.1f}, B={p2:.1f}")
        all_results[key] = {'align_A': p1s, 'align_B': p2s}

    with open(RESULTS_FILE, 'w') as f:
        json.dump(all_results, f, indent=2)

    label = {0.0:'adaptive', 0.25:'reduced', 0.50:'half-speed',
             0.75:'near-frozen', 1.0:'frozen'}
    print("\n" + "="*72)
    print(f"{'F':>5} | {'death_p_eff':>11} | {'fa_eff':>7} | "
          f"{'align_A':>9} | {'align_B':>9} | regime")
    print("-"*72)
    for ff in FREEZE_FACTORS:
        key = f"{ff:.2f}"
        r = all_results[key]
        a = np.mean(r['align_A']); b = np.mean(r['align_B'])
        dp = BASE_DEATH_P * (1-ff); fa = BASE_FIELD_ALPHA * (1-ff)
        print(f"{ff:5.2f} | {dp:11.5f} | {fa:7.4f} | "
              f"{a:9.1f} | {b:9.1f} | {label[ff]}")
    print("="*72)
    print("\nPrediction: align_B decreases monotonically with F.")
    print("F=0 should match Exp3 adaptive; F=1 should match Exp3 frozen (~0).")
