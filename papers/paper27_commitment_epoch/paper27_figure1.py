"""
paper27_figure1.py -- Figure for Paper 27: The Zone Formation Commitment Epoch

4 panels:
  A: C_zone(t_ref, T_end) vs t_ref for FA in {0.10,0.20,0.40,0.80}, WR=4.8
     Uses zone-mean stacked vector correlation (not full field -- zone means
     give cleaner signal; within-zone noise dilutes global correlation).
  B: log-log t* vs P_c*FA*WR for all 16 (FA,WR) conditions
     Test: slope ~ -1 (linear model) or -2/3 (signal/noise model)
  C: sg4(t) trajectories for FA in {0.10,0.20,0.40,0.80}, WR=4.8 (T=5000)
  D: Zone trajectories (Exp B) -- per-zone mean[1] vs t, one representative
     seed, with inset showing t*_z spread (pitchfork vs spinodal diagnosis)
"""
import numpy as np, json, os, math, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import linregress

# ── Load results ──────────────────────────────────────────────────────────────
DIR = os.path.dirname(__file__)
RESULTS_FILE = os.path.join(DIR, "results", "paper27_results.json")
r27 = json.load(open(RESULTS_FILE))
print(f"Loaded {len(r27)} results.")

# ── Constants (must match paper27_experiments.py) ─────────────────────────────
N_ZONES   = 4
HALF      = 20
H         = 20
ZONE_W    = 5
HS        = 2
N_ACT     = HALF * H
_col      = np.arange(N_ACT) % HALF
zone_id   = _col // ZONE_W

FA_VALS_A  = [0.10, 0.20, 0.40, 0.80]
WR_VALS_A  = [1.2, 2.4, 4.8, 9.6]
N_SEEDS_A  = 5
N_SEEDS_B  = 15
T_END_A    = 4800   # last actual checkpoint (range(400,5001,400) ends at 4800)
T_END_B    = 6000
CPS_A      = list(range(400, 4801, 400))   # 12 checkpoints: 400..4800
CPS_B      = list(range(200, T_END_B + 1, 200))
WR_STD     = 4.8
FA_STD     = 0.40
P_C        = 0.175    # estimated P_consol

def key_a(fa, wr, seed): return f"p27a,{fa:.8g},{wr:.8g},{seed}"
def key_b(seed):         return f"p27b,{seed}"

# ── Zone-mean stacked temporal correlation ────────────────────────────────────
def zmean_vec(key, t):
    """Stacked zone-mean vector for key at time t (8-dim for N_ZONES=4, HS=2)."""
    zm = r27.get(key, {}).get(f"zmeans_{t}")
    if zm is None:
        return None
    return np.array(zm).flatten()   # (N_ZONES * HS,)

def zone_C(key, t_ref, t_end):
    """Cosine similarity between zmean_vec at t_ref and t_end."""
    v1 = zmean_vec(key, t_ref)
    v2 = zmean_vec(key, t_end)
    if v1 is None or v2 is None:
        return float("nan")
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 < 1e-10 or n2 < 1e-10:
        return float("nan")
    return float(np.dot(v1, v2) / (n1 * n2))

def mean_C(fa, wr, t_ref, n_seeds=N_SEEDS_A, t_end=T_END_A):
    vals = [zone_C(key_a(fa, wr, s), t_ref, t_end)
            for s in range(n_seeds)]
    vals = [v for v in vals if not math.isnan(v)]
    return float(np.mean(vals)) if vals else float("nan")

def find_t_star(fa, wr, n_seeds=N_SEEDS_A, t_end=T_END_A, cps=CPS_A):
    prev_c, prev_t = None, None
    for t in cps:
        c = mean_C(fa, wr, t, n_seeds=n_seeds, t_end=t_end)
        if math.isnan(c):
            continue
        if prev_c is not None and prev_c < 0 <= c:
            return prev_t + (t - prev_t) * (-prev_c) / (c - prev_c)
        prev_c, prev_t = c, t
    return float("nan")

def find_t_sat(fa, wr):
    """First t where sg4 changes < 5% vs previous checkpoint."""
    sg4_prev = None
    for t in CPS_A:
        sg4s = [r27[key_a(fa, wr, s)].get(f"sg4_{t}", float("nan"))
                for s in range(N_SEEDS_A) if key_a(fa, wr, s) in r27]
        sg4s = [v for v in sg4s if not math.isnan(v)]
        if not sg4s:
            continue
        sg4 = float(np.mean(sg4s))
        if sg4_prev is not None and sg4_prev > 1e-4:
            if abs(sg4 - sg4_prev) / sg4_prev < 0.05:
                return t
        sg4_prev = sg4
    return float("nan")

# ── Exp A: t* table ───────────────────────────────────────────────────────────
print("\n=== Commitment epoch t* (zone-mean cosine similarity) ===")
print(f"  {'FA':>5} {'WR':>5} | {'t*':>8} {'Pc*FA*WR':>10} {'t**Pc*FA*WR':>13}")
print("  " + "-" * 50)
t_star_dict = {}
for fa in FA_VALS_A:
    for wr in WR_VALS_A:
        ts = find_t_star(fa, wr)
        t_star_dict[(fa, wr)] = ts
        rate = P_C * fa * wr
        prod = ts * rate if not math.isnan(ts) else float("nan")
        print(f"  {fa:>5.2f} {wr:>5.1f} | {ts:>8.0f} {rate:>10.4f} {prod:>13.2f}")

# ── Exp B: zone t* per zone per seed ─────────────────────────────────────────
def zone_C_b(seed, zone_z, t_ref):
    k = key_b(seed)
    zm_ref = r27.get(k, {}).get(f"zmeans_{t_ref}")
    zm_end = r27.get(k, {}).get(f"zmeans_{T_END_B}")
    if zm_ref is None or zm_end is None:
        return float("nan")
    v1 = np.array(zm_ref[zone_z]); v2 = np.array(zm_end[zone_z])
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 < 1e-10 or n2 < 1e-10:
        return float("nan")
    return float(np.dot(v1, v2) / (n1 * n2))

def find_t_star_zone(seed, zone_z):
    prev_c, prev_t = None, None
    for t in CPS_B:
        c = zone_C_b(seed, zone_z, t)
        if math.isnan(c):
            continue
        if prev_c is not None and prev_c < 0 <= c:
            return prev_t + (t - prev_t) * (-prev_c) / (c - prev_c)
        prev_c, prev_t = c, t
    return float("nan")

t_star_matrix = np.full((N_SEEDS_B, N_ZONES), float("nan"))
for seed in range(N_SEEDS_B):
    for z in range(N_ZONES):
        t_star_matrix[seed, z] = find_t_star_zone(seed, z)

# Within-seed spread (std over zones)
within_stds = []
for row in t_star_matrix:
    valid = row[~np.isnan(row)]
    if len(valid) > 1:
        within_stds.append(float(np.std(valid)))

# Between-seed spread
seed_means = np.nanmean(t_star_matrix, axis=1)
valid_means = seed_means[~np.isnan(seed_means)]
between_std = float(np.std(valid_means)) if len(valid_means) > 1 else float("nan")
sigma_in    = float(np.mean(within_stds)) if within_stds else float("nan")

print(f"\n=== Exp B pitchfork/spinodal ===")
print(f"  sigma_in  (within-seed zone spread): {sigma_in:.1f}")
print(f"  sigma_btw (between-seed spread):     {between_std:.1f}")
if not math.isnan(sigma_in) and not math.isnan(between_std) and between_std > 0:
    ratio = sigma_in / between_std
    verdict = ("PITCHFORK" if ratio < 0.5 else
               "SPINODAL"  if ratio > 1.5 else "MIXED")
    print(f"  ratio = {ratio:.2f}  ->  {verdict}")

# ── Figure layout ─────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(13, 9))
axes = axes.flatten()
COLORS_FA = ["#e41a1c", "#ff7f00", "#2166ac", "#4dac26"]   # 4 FA values

# ═════════════════════════════════════════════════════════════════════════════
# PANEL A: C_zone(t_ref, T_end) vs t_ref -- shows OSCILLATION
# WR=4.8 slice, one line per FA
# ═════════════════════════════════════════════════════════════════════════════
ax = axes[0]
wr_plot = WR_STD

for idx_fa, fa in enumerate(FA_VALS_A):
    c_vals = [mean_C(fa, wr_plot, t) for t in CPS_A]
    ax.plot(CPS_A, c_vals, "o-", color=COLORS_FA[idx_fa],
            ms=6, lw=2.0, label=f"FA={fa:.2f}")
    # Mark first t*
    ts = t_star_dict.get((fa, wr_plot), float("nan"))
    if not math.isnan(ts):
        ax.axvline(ts, color=COLORS_FA[idx_fa], ls="--", lw=1.2, alpha=0.7)

ax.axhline(0, color="black", lw=1.0, ls="-")
ax.fill_between([CPS_A[0] - 100, CPS_A[-1] + 100], -1, 0,
                color="steelblue", alpha=0.05)
ax.set_xlabel(r"$t_{\rm ref}$ (time)", fontsize=11)
ax.set_ylabel(r"$C_{\rm zone}(t_{\rm ref},\, T_{\rm end})$", fontsize=11)
ax.set_title(r"\textbf{A.} Zone-level temporal correlation: oscillatory structure"
             "\n"
             r"WR = 4.8; zones flip polarity repeatedly (not a one-time bifurcation)",
             fontsize=10)
ax.legend(fontsize=9, framealpha=0.9)
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
ax.set_xlim(CPS_A[0] - 50, CPS_A[-1] + 50)
ax.text(0.97, 0.97,
        "Multiple zero-crossings = repeated\npolarity flips. Zone structure\n"
        "is metastable, not permanent.",
        transform=ax.transAxes, fontsize=8, ha="right", va="top",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.9))

# ═════════════════════════════════════════════════════════════════════════════
# PANEL B: log-log t* vs P_c * FA * WR
# ═════════════════════════════════════════════════════════════════════════════
ax = axes[1]

# Collect valid (rate, t*) pairs
rates, t_stars = [], []
for fa in FA_VALS_A:
    for wr in WR_VALS_A:
        ts = t_star_dict.get((fa, wr), float("nan"))
        if not math.isnan(ts) and ts > 0:
            rates.append(P_C * fa * wr)
            t_stars.append(ts)

if len(rates) >= 2:
    log_r = np.log10(rates)
    log_t = np.log10(t_stars)
    slope, intercept, r, *_ = linregress(log_r, log_t)
    x_fit = np.linspace(min(log_r) - 0.2, max(log_r) + 0.2, 100)
    y_fit = slope * x_fit + intercept
    ax.plot(10**x_fit, 10**y_fit, "--", color="gray", lw=1.5,
            label=rf"Fit: slope = {slope:.2f} (R$^2$={r**2:.2f})")

# Reference slope = -1
if rates:
    x_ref = np.array([min(rates) * 0.8, max(rates) * 1.2])
    amp = np.mean(t_stars) * np.mean(rates)
    ax.plot(x_ref, amp / x_ref, ":", color="black", lw=1.5, label="Slope = -1 (linear)")

# Scatter colored by FA
for fa, col in zip(FA_VALS_A, COLORS_FA):
    for wr in WR_VALS_A:
        ts = t_star_dict.get((fa, wr), float("nan"))
        if not math.isnan(ts):
            ax.scatter(P_C * fa * wr, ts, color=col, s=80, zorder=5)

ax.scatter([], [], color="gray", s=60, label="Data points\n(colour = FA)")
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel(r"$P_c \cdot \mathrm{FA} \cdot \mathrm{WR}$ (commitment rate)", fontsize=11)
ax.set_ylabel(r"$t^*$ (commitment epoch, steps)", fontsize=11)
ax.set_title(r"\textbf{B.} Power-law scaling of $t^*$"
             "\n"
             r"Linear model predicts slope $= -1$ on log-log",
             fontsize=10)
ax.legend(fontsize=8, framealpha=0.9)
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

# If few t* values, fall back to showing flip count vs rate
if len(rates) < 4:
    ax.cla()
    # Flip count = number of zero-crossings in C(t, T_end) per run
    def count_flips(fa, wr):
        counts = []
        for seed in range(N_SEEDS_A):
            k = key_a(fa, wr, seed)
            if k not in r27: continue
            c_vals = []
            for t in CPS_A:
                zm_t = r27[k].get(f"zmeans_{t}")
                zm_e = r27[k].get(f"zmeans_{T_END_A}")
                if zm_t and zm_e:
                    v1, v2 = np.array(zm_t).flatten(), np.array(zm_e).flatten()
                    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
                    if n1 > 1e-10 and n2 > 1e-10:
                        c_vals.append(np.dot(v1, v2)/(n1*n2))
            flips = sum(1 for i in range(1, len(c_vals))
                        if c_vals[i-1] * c_vals[i] < 0)
            counts.append(flips)
        return float(np.mean(counts)) if counts else float("nan")

    all_rates_flips = []
    for fa in FA_VALS_A:
        for wr in WR_VALS_A:
            rate = P_C * fa * wr
            nf = count_flips(fa, wr)
            if not math.isnan(nf):
                all_rates_flips.append((rate, nf, fa))

    for rate, nf, fa in all_rates_flips:
        idx_fa = FA_VALS_A.index(fa)
        ax.scatter(rate, nf, color=COLORS_FA[idx_fa], s=80, zorder=5)

    ax.set_xlabel(r"$P_c \cdot \mathrm{FA} \cdot \mathrm{WR}$", fontsize=11)
    ax.set_ylabel("Number of polarity flips in [0, T_end]", fontsize=11)
    ax.set_title(r"\textbf{B.} Flip count vs commitment rate"
                 "\n"
                 r"Zone polarity oscillates; higher rate $\to$ more flips?",
                 fontsize=10)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

# ═════════════════════════════════════════════════════════════════════════════
# PANEL C: sg4(t) trajectories for FA sweep at WR=4.8
# ═════════════════════════════════════════════════════════════════════════════
ax = axes[2]

for idx_fa, fa in enumerate(FA_VALS_A):
    sg4_traj, sg4_err = [], []
    for t in CPS_A:
        sg4s = [r27[key_a(fa, wr_plot, s)].get(f"sg4_{t}", float("nan"))
                for s in range(N_SEEDS_A) if key_a(fa, wr_plot, s) in r27]
        sg4s = [v for v in sg4s if not math.isnan(v)]
        sg4_traj.append(float(np.mean(sg4s)) if sg4s else float("nan"))
        sg4_err.append(float(np.std(sg4s) / math.sqrt(len(sg4s))) if len(sg4s) > 1 else 0.0)

    sg4_traj = np.array(sg4_traj); sg4_err = np.array(sg4_err)
    valid = ~np.isnan(sg4_traj)
    ax.plot(np.array(CPS_A)[valid], sg4_traj[valid], "o-",
            color=COLORS_FA[idx_fa], ms=5, lw=2.0, label=f"FA={fa:.2f}")
    ax.fill_between(np.array(CPS_A)[valid],
                    (sg4_traj - sg4_err)[valid],
                    (sg4_traj + sg4_err)[valid],
                    color=COLORS_FA[idx_fa], alpha=0.15)
    # Mark t*
    ts = t_star_dict.get((fa, wr_plot), float("nan"))
    if not math.isnan(ts):
        ax.axvline(ts, color=COLORS_FA[idx_fa], ls="--", lw=1.0, alpha=0.6)

ax.set_xlabel(r"$t$ (time steps)", fontsize=11)
ax.set_ylabel("sg4 (zone differentiation)", fontsize=11)
ax.set_title(r"\textbf{C.} sg4$(t)$ trajectories: long-run saturation check"
             "\n"
             r"WR = 4.8, $T_{\rm end} = 5000$; dashed = $t^*$ per FA",
             fontsize=10)
ax.legend(fontsize=9, framealpha=0.9)
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

# ═════════════════════════════════════════════════════════════════════════════
# PANEL D: Zone trajectories (Exp B) -- pitchfork vs spinodal
# ═════════════════════════════════════════════════════════════════════════════
ax = axes[3]

# Pick seed with clearest t* spread (most representative)
valid_seed_mask = ~np.any(np.isnan(t_star_matrix), axis=1)
if valid_seed_mask.any():
    within_seed_range = np.ptp(t_star_matrix[valid_seed_mask], axis=1)
    idx_med = np.argsort(within_seed_range)[len(within_seed_range) // 2]
    rep_seed = int(np.where(valid_seed_mask)[0][idx_med])
else:
    rep_seed = 0

ZONE_COLORS = ["#e41a1c", "#ff7f00", "#2166ac", "#4dac26"]
k_rep = key_b(rep_seed)
for z in range(N_ZONES):
    zm_traj = []
    for t in CPS_B:
        zm = r27.get(k_rep, {}).get(f"zmeans_{t}")
        if zm is not None:
            zm_traj.append((t, zm[z][1]))   # perturbation component (index 1)
    if zm_traj:
        ts_plot = [x[0] for x in zm_traj]
        vs_plot = [x[1] for x in zm_traj]
        ax.plot(ts_plot, vs_plot, "-", color=ZONE_COLORS[z], lw=2.0,
                label=f"Zone {z}")
        # Mark t*_z
        t_star_z = t_star_matrix[rep_seed, z]
        if not math.isnan(t_star_z):
            ax.axvline(t_star_z, color=ZONE_COLORS[z], ls=":", lw=1.2, alpha=0.8)

ax.axhline(0, color="black", lw=0.8, ls="-")
ax.set_xlabel(r"$t$ (time steps)", fontsize=11)
ax.set_ylabel(r"Zone-mean $F_z[1]$ (perturbation component)", fontsize=11)
ax.set_title(r"\textbf{D.} Zone-level trajectories (Exp B, representative seed)"
             "\n"
             rf"Dotted lines = $t^*_z$ per zone; "
             rf"$\sigma_{{in}}/\sigma_{{btw}} = {sigma_in:.0f}/{between_std:.0f}$",
             fontsize=10)
ax.legend(fontsize=9, framealpha=0.9, loc="upper left")
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

# Inset: t*_z distribution by zone (all seeds)
ax_in = ax.inset_axes([0.55, 0.05, 0.42, 0.45])
for z in range(N_ZONES):
    col_data = t_star_matrix[:, z]
    col_valid = col_data[~np.isnan(col_data)]
    if len(col_valid) > 0:
        ax_in.scatter([z] * len(col_valid), col_valid,
                      color=ZONE_COLORS[z], alpha=0.6, s=25, zorder=3)
        ax_in.errorbar(z, np.mean(col_valid), yerr=np.std(col_valid),
                       fmt="s", color=ZONE_COLORS[z], ms=7, capsize=4, lw=2)
ax_in.set_xticks(range(N_ZONES)); ax_in.set_xticklabels([f"Z{z}" for z in range(N_ZONES)])
ax_in.set_ylabel(r"$t^*_z$", fontsize=8)
ax_in.set_title("Per-zone t*", fontsize=8)
ax_in.tick_params(labelsize=7)
verdict_str = ("PITCHFORK" if sigma_in / between_std < 0.5 else
               "SPINODAL"  if sigma_in / between_std > 1.5 else
               "MIXED") if not math.isnan(sigma_in) and between_std > 0 else "?"
ax_in.text(0.05, 0.97, verdict_str, transform=ax_in.transAxes,
           fontsize=8, va="top", fontweight="bold",
           color="#2166ac" if verdict_str == "PITCHFORK" else
                 "#d62728" if verdict_str == "SPINODAL" else "#8c6d31")

# ── Suptitle and save ─────────────────────────────────────────────────────────
fig.suptitle(
    r"Paper 27: Commitment epoch $t^*$ scaling, long-run saturation, "
    r"and zone-level bifurcation mechanism",
    fontsize=11, y=1.01
)
fig.tight_layout()

OUT = os.path.join(DIR, "paper27_figure1")
fig.savefig(OUT + ".pdf", bbox_inches="tight")
fig.savefig(OUT + ".png", dpi=150, bbox_inches="tight")
print(f"\nSaved {OUT}.pdf and {OUT}.png")

# Summary stats
print(f"\nPanel A: C at first checkpoint (t=400) for FA=0.40, WR=4.8: "
      f"{mean_C(0.40, 4.8, 400):.3f}")
valid_ts = [(fa, wr, t_star_dict[(fa,wr)])
            for fa in FA_VALS_A for wr in WR_VALS_A
            if not math.isnan(t_star_dict.get((fa,wr), float("nan")))]
print(f"Panel B: {len(valid_ts)} valid t* values out of {len(FA_VALS_A)*len(WR_VALS_A)}")
if valid_ts:
    for fa, wr, ts in valid_ts:
        print(f"  FA={fa:.2f}, WR={wr:.1f}: t*={ts:.0f}, Pc*FA*WR={P_C*fa*wr:.4f}")
print(f"Panel D: sigma_in={sigma_in:.1f}, sigma_btw={between_std:.1f}, "
      f"ratio={sigma_in/between_std:.2f}, verdict={verdict_str}")
