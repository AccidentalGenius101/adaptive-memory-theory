"""
paper19_blocked_buffer.py -- Experiment for Paper 19

The Blocked-Site Buffer: Direct measurement of the third reservoir.

Three predictions from the blocked-site model:
  (1) N_blocked_measured  ~= 1 - P_consol = 1 - p_calm^SS
  (2) burst_timing        ~= WAVE_DUR + SS  (in-flight wave expiry + gate unlock)
  (3) burst_amplitude     ~ (1 - P_consol) / P_consol  (blocked fraction vs active fraction)

Also confirms: m_blocked_norm > m_unblocked_norm
(perturbed sites accumulate more mid_mem signal than calm sites)

Sweeps:
  SS   in {5, 10, 15, 20}
  WR   in {2.4, 4.8, 9.6}
  FA   = 0.40 (fixed)
  5 seeds each -> 60 runs

Protocol: same two-phase setup as Paper 18.
  Phase 1: t=1..2000 (waves running)
  Phase 3: t=2001..2200 (no waves, dense checkpoints every 10 steps)

Key:  "p19,ss{ss},wr{wr:.1f},{seed}"
Val:  {
        "sg4_{t}": float (Phase 1: t=400,800,...,2000; Phase 3: t=2010,...,2200)
        "n_blocked_2000": float  -- fraction(streak < SS) at T_BUILD
        "m_blocked_norm_2000": float  -- mean ||m|| for blocked sites
        "m_unblocked_norm_2000": float  -- mean ||m|| for unblocked sites
        "f_norm_2000": float  -- mean ||F|| at T_BUILD
      }
"""
import numpy as np, json, os, multiprocessing as mp, math

W, H = 40, 20; HALF = W // 2
HS = 2; N_ZONES = 4; ZONE_W = HALF // N_ZONES
N_ACT = HALF * H

BASE_BETA  = 0.005
ALPHA_MID  = 0.15
MID_DECAY  = 0.99
FD         = 0.9997
NU         = 0.001
WAVE_DUR   = 15
SEED_BETA  = 0.25
KAPPA      = 0.020
FA         = 0.40       # fixed

T_BUILD = 2000
T_END   = 2200

SS_VALS = [5, 10, 15, 20]
WR_VALS = [2.4, 4.8, 9.6]
N_SEEDS = 5

PHASE1_CPS  = list(range(400, T_BUILD + 1, 400))          # [400,800,...,2000]
PHASE3_CPS  = list(range(T_BUILD + 10, T_END + 1, 10))    # [2010,...,2200]
ALL_CPS_SET = set(PHASE1_CPS + PHASE3_CPS)

# Geometry
_col = np.arange(N_ACT) % HALF
_row = np.arange(N_ACT) // HALF
zone_id = _col // ZONE_W

NB = []
for i in range(N_ACT):
    c, r = _col[i], _row[i]
    nb = []
    for dc, dr in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        nc, nr = c + dc, r + dr
        if 0 <= nc < HALF and 0 <= nr < H:
            nb.append(nr * HALF + nc)
    NB.append(np.array(nb, dtype=int))

top_mask = _row < H // 2
bot_mask = _row >= H // 2
d_B = np.array([0.0, 1.0])


def sg4_inter_fn(F):
    means = [F[zone_id == z].mean(0) for z in range(N_ZONES)]
    pairs = [(i, j) for i in range(N_ZONES) for j in range(i + 1, N_ZONES)]
    return float(np.mean([np.linalg.norm(means[i] - means[j]) for i, j in pairs]))


def run(seed, ss, wr):
    rng    = np.random.default_rng(seed)
    h      = rng.normal(0, 0.1, (N_ACT, HS))
    F      = np.zeros((N_ACT, HS))
    m      = np.zeros((N_ACT, HS))
    base   = h.copy()
    streak = np.zeros(N_ACT, int)
    ttl    = rng.geometric(p=max(NU, 1e-6), size=N_ACT).astype(float)
    waves  = []
    result = {}

    for step in range(1, T_END + 1):

        # Phase 1 only: launch new waves
        if step <= T_BUILD:
            n = int(wr / WAVE_DUR)
            n += int(rng.random() < (wr / WAVE_DUR - n))
            for _ in range(n):
                top  = rng.random() < 0.5
                sign = 1.0 if top else -1.0
                waves.append([0 if top else 1, WAVE_DUR, sign * d_B.copy(), False])

        # Perturbed sites this step
        pert = np.zeros(N_ACT, bool)
        for w in waves:
            pert |= (top_mask if w[0] == 0 else bot_mask)

        base += BASE_BETA * (h - base)
        for w in waves:
            mask = top_mask if w[0] == 0 else bot_mask
            h[mask] += 0.3 * w[2]

        m[pert] += ALPHA_MID * (h - base)[pert]
        m *= MID_DECAY

        streak[pert]  = 0
        streak[~pert] += 1
        ok = streak >= ss
        F[ok] += FA * (m[ok] - F[ok])
        F *= FD

        dF = np.zeros_like(F)
        for i in range(N_ACT):
            if len(NB[i]):
                dF[i] = KAPPA * (F[NB[i]].mean(0) - F[i])
        F += dF

        ttl -= 1.0
        ttl[pert] -= 1.0
        dead = np.where(ttl <= 0)[0]
        for i in dead:
            j = NB[i][rng.integers(len(NB[i]))] if len(NB[i]) else i
            h[i]      = (1 - SEED_BETA) * rng.normal(0, 0.1, HS) + SEED_BETA * F[j]
            m[i]      = 0
            streak[i] = 0
            ttl[i]    = float(rng.geometric(p=NU))

        waves = [[w[0], w[1] - 1, w[2], w[3]] for w in waves if w[1] - 1 > 0]

        if step in ALL_CPS_SET:
            result[f"sg4_{step}"] = sg4_inter_fn(F)

        # At T_BUILD: record blocked-buffer state
        if step == T_BUILD:
            blocked_mask   = streak < ss
            unblocked_mask = streak >= ss

            result["n_blocked_2000"] = float(blocked_mask.mean())

            m_norms = np.linalg.norm(m, axis=1)
            if blocked_mask.any():
                result["m_blocked_norm_2000"]   = float(m_norms[blocked_mask].mean())
            else:
                result["m_blocked_norm_2000"]   = 0.0
            if unblocked_mask.any():
                result["m_unblocked_norm_2000"] = float(m_norms[unblocked_mask].mean())
            else:
                result["m_unblocked_norm_2000"] = 0.0

            result["f_norm_2000"] = float(np.linalg.norm(F, axis=1).mean())

    return result


def make_key(ss, wr, seed):
    return f"p19,ss{ss},wr{wr:.1f},{seed}"


def _worker(args):
    return run(*args)


RESULTS_FILE = os.path.join(os.path.dirname(__file__), "results", "paper19_results.json")


if __name__ == "__main__":
    mp.freeze_support()

    if os.path.exists(RESULTS_FILE):
        results = json.load(open(RESULTS_FILE))
        print("Loaded cached results.")
    else:
        all_conditions = [
            (seed, ss, wr)
            for ss   in SS_VALS
            for wr   in WR_VALS
            for seed in range(N_SEEDS)
        ]
        print(f"Running {len(all_conditions)} simulations (Paper 19)...")
        with mp.Pool(processes=min(len(all_conditions), mp.cpu_count())) as pool:
            raw = pool.map(_worker, all_conditions)

        results = {}
        for i, (seed, ss, wr) in enumerate(all_conditions):
            results[make_key(ss, wr, seed)] = raw[i]

        os.makedirs(os.path.dirname(RESULTS_FILE), exist_ok=True)
        json.dump(results, open(RESULTS_FILE, "w"), indent=2)
        print("Saved results.")

    # ---- Analysis ----
    def mean_field(ss, wr, field, default=float("nan")):
        keys = [make_key(ss, wr, s) for s in range(N_SEEDS) if make_key(ss, wr, s) in results]
        vals = [results[k].get(field, default) for k in keys]
        vals = [v for v in vals if not math.isnan(v)]
        return float(np.mean(vals)) if vals else float("nan")

    def burst_stats(ss, wr):
        """Returns (burst_amplitude, burst_timing_steps) averaged over seeds."""
        amps, timings = [], []
        for seed in range(N_SEEDS):
            key = make_key(ss, wr, seed)
            if key not in results:
                continue
            r     = results[key]
            v0    = r.get(f"sg4_{T_BUILD}", float("nan"))
            if math.isnan(v0) or v0 <= 0:
                continue
            phase3_sg4 = [(t, r.get(f"sg4_{t}", float("nan"))) for t in PHASE3_CPS]
            phase3_sg4 = [(t, v) for t, v in phase3_sg4 if not math.isnan(v)]
            if not phase3_sg4:
                continue
            t_peak, v_peak = max(phase3_sg4, key=lambda x: x[1])
            amps.append((v_peak - v0) / v0)
            timings.append(t_peak - T_BUILD)
        if not amps:
            return float("nan"), float("nan")
        return float(np.mean(amps)), float(np.mean(timings))

    p_calm_of = {wr: 1.0 - wr / (2.0 * WAVE_DUR) for wr in WR_VALS}

    print("\n" + "=" * 75)
    print("PREDICTION 1: N_blocked measured vs predicted (1 - P_consol)")
    print(f"  {'SS':>4} {'WR':>5} | {'p_calm':>7} {'P_consol':>9} {'N_blocked_pred':>15} "
          f"{'N_blocked_meas':>15} {'err':>6}")
    print("  " + "-" * 70)
    for ss in SS_VALS:
        for wr in WR_VALS:
            p_calm  = p_calm_of[wr]
            p_c     = p_calm ** ss
            pred    = 1.0 - p_c
            meas    = mean_field(ss, wr, "n_blocked_2000")
            err     = meas - pred if not math.isnan(meas) else float("nan")
            print(f"  {ss:>4} {wr:>5.1f} | {p_calm:>7.4f} {p_c:>9.4f} {pred:>15.4f} "
                  f"{meas:>15.4f} {err:>+6.4f}")

    print("\n" + "=" * 75)
    print("PREDICTION 2: Burst timing ~= WAVE_DUR + SS")
    print(f"  {'SS':>4} {'WR':>5} | {'pred_timing':>12} {'meas_timing':>12} {'burst_amp':>10}")
    print("  " + "-" * 50)
    for ss in SS_VALS:
        for wr in WR_VALS:
            pred_timing = WAVE_DUR + ss
            bamp, btim  = burst_stats(ss, wr)
            print(f"  {ss:>4} {wr:>5.1f} | {pred_timing:>12} {btim:>12.1f} {bamp:>+10.3f}")

    print("\n" + "=" * 75)
    print("PREDICTION 3: burst_amplitude ~ (1-P_consol)/P_consol")
    print(f"  {'SS':>4} {'WR':>5} | {'(1-P)/P':>10} {'burst_amp':>10} | "
          f"{'m_blocked':>10} {'m_unblocked':>12} {'f_norm':>8}")
    print("  " + "-" * 70)
    for ss in SS_VALS:
        for wr in WR_VALS:
            p_c    = p_calm_of[wr] ** ss
            ratio  = (1.0 - p_c) / p_c
            bamp, _ = burst_stats(ss, wr)
            mb  = mean_field(ss, wr, "m_blocked_norm_2000")
            mu  = mean_field(ss, wr, "m_unblocked_norm_2000")
            fn  = mean_field(ss, wr, "f_norm_2000")
            print(f"  {ss:>4} {wr:>5.1f} | {ratio:>10.3f} {bamp:>+10.3f} | "
                  f"{mb:>10.2f} {mu:>12.2f} {fn:>8.2f}")
