"""
paper19_figure1.py -- Figure 1 for Paper 19

Three-panel figure:
  A: Phase 3 sg4 trajectories for SS=10, three WR values
     Shows the two-regime split: moderate burst (WR<=4.8) vs explosive growth (WR=9.6)
     Annotates burst timing prediction WAVE_DUR+SS
  B: Burst timing (empirical) vs WAVE_DUR+SS (predicted)
     All 12 moderate conditions; expected 1:1 line
  C: Buffer contents (m_blocked_norm) vs field at T_BUILD (f_norm)
     Coloured by WR; shows the buffer-field tradeoff

Output: paper19_figure1.pdf, paper19_figure1.png
"""
import json, os, math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm

DIR = os.path.dirname(__file__)
results = json.load(open(os.path.join(DIR, "results", "paper19_results.json")))

SS_VALS   = [5, 10, 15, 20]
WR_VALS   = [2.4, 4.8, 9.6]
N_SEEDS   = 5
T_BUILD   = 2000
WAVE_DUR  = 15
PHASE3_CPS = list(range(T_BUILD + 10, 2201, 10))

WR_COLORS  = ["#2166ac", "#d62728", "#9467bd"]
WR_MARKERS = ["o", "s", "^"]
SS_COLORS  = ["#1f77b4", "#d62728", "#2ca02c", "#ff7f0e"]


def make_key(ss, wr, seed):
    return f"p19,ss{ss},wr{wr:.1f},{seed}"


def mean_t(ss, wr, t):
    keys = [make_key(ss, wr, s) for s in range(N_SEEDS) if make_key(ss, wr, s) in results]
    vals = [results[k].get(f"sg4_{t}", float("nan")) for k in keys]
    vals = [v for v in vals if not math.isnan(v)]
    return float(np.mean(vals)) if vals else float("nan")


def err_t(ss, wr, t):
    keys = [make_key(ss, wr, s) for s in range(N_SEEDS) if make_key(ss, wr, s) in results]
    vals = [results[k].get(f"sg4_{t}", float("nan")) for k in keys]
    vals = [v for v in vals if not math.isnan(v)]
    return float(np.std(vals) / math.sqrt(len(vals))) if len(vals) > 1 else 0.0


def mean_field(ss, wr, field):
    keys = [make_key(ss, wr, s) for s in range(N_SEEDS) if make_key(ss, wr, s) in results]
    vals = [results[k].get(field, float("nan")) for k in keys]
    vals = [v for v in vals if not math.isnan(v)]
    return float(np.mean(vals)) if vals else float("nan")


def burst_stats(ss, wr):
    """(burst_amplitude, burst_timing_steps) averaged over seeds."""
    amps, timings = [], []
    for seed in range(N_SEEDS):
        key = make_key(ss, wr, seed)
        if key not in results:
            continue
        r  = results[key]
        v0 = r.get(f"sg4_{T_BUILD}", float("nan"))
        if math.isnan(v0) or v0 <= 0:
            continue
        phase3 = [(t, r.get(f"sg4_{t}", float("nan"))) for t in PHASE3_CPS]
        phase3 = [(t, v) for t, v in phase3 if not math.isnan(v)]
        if not phase3:
            continue
        t_peak, v_peak = max(phase3, key=lambda x: x[1])
        amps.append((v_peak - v0) / v0)
        timings.append(t_peak - T_BUILD)
    if not amps:
        return float("nan"), float("nan")
    return float(np.mean(amps)), float(np.mean(timings))


fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

# ============================================================
# Panel A: Phase 3 trajectories (SS=10, three WR values)
# ============================================================
ax = axes[0]
SS_PLOT = 10
t_rel = [t - T_BUILD for t in PHASE3_CPS]

for j, wr in enumerate(WR_VALS):
    v0 = mean_t(SS_PLOT, wr, T_BUILD)
    if v0 <= 0 or math.isnan(v0):
        # WR=9.6 may have near-zero f at T_BUILD; show absolute sg4 instead
        sg4_vals = [mean_t(SS_PLOT, wr, t) for t in PHASE3_CPS]
        # Normalise to first non-zero value or just plot absolute
        sg4_vals_plot = sg4_vals
        errs = [err_t(SS_PLOT, wr, t) for t in PHASE3_CPS]
        ax.errorbar(t_rel, sg4_vals_plot, yerr=errs,
                    fmt=WR_MARKERS[j] + "-", color=WR_COLORS[j],
                    ms=4, lw=1.8, capsize=2,
                    label=f"WR={wr:.1f} (abs)")
    else:
        norm_vals = [mean_t(SS_PLOT, wr, t) / v0 for t in PHASE3_CPS]
        errs = [err_t(SS_PLOT, wr, t) / v0 for t in PHASE3_CPS]
        ax.errorbar(t_rel, norm_vals, yerr=errs,
                    fmt=WR_MARKERS[j] + "-", color=WR_COLORS[j],
                    ms=4, lw=1.8, capsize=2,
                    label=f"WR={wr:.1f}")

    # Mark predicted burst timing
    pred_timing = WAVE_DUR + SS_PLOT
    ax.axvline(pred_timing, color=WR_COLORS[j], ls=":", lw=1.0, alpha=0.6)

ax.axhline(1.0, color="gray", ls="--", lw=1.0, alpha=0.5)
ax.set_xlabel(r"Steps after wave cessation ($t - T_{\rm build}$)", fontsize=10)
ax.set_ylabel(r"sg4 / sg4$(T_{\rm build})$   [normalised]", fontsize=10)
ax.set_title(r"\textbf{A.} Two regimes: buffer release vs buffer dominance ($SS=10$)",
             fontsize=9.5)
ax.legend(fontsize=8, framealpha=0.85)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.text(0.97, 0.97,
        "Dotted lines: predicted\nburst timing $=$ WAVE\\_DUR $+$ SS\n"
        "WR=9.6: F$\\approx$0 at $T_{\\rm build}$;\nburst establishes structure",
        transform=ax.transAxes, fontsize=7.5, ha="right", va="top",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.9))

# ============================================================
# Panel B: Burst timing vs WAVE_DUR + SS (moderate WR only)
# ============================================================
ax = axes[1]

pred_all, meas_all = [], []
for i, ss in enumerate(SS_VALS):
    for j, wr in enumerate([2.4, 4.8]):   # moderate WR only
        _, btim = burst_stats(ss, wr)
        if math.isnan(btim):
            continue
        pred = WAVE_DUR + ss
        pred_all.append(pred)
        meas_all.append(btim)
        ax.scatter(pred, btim, color=SS_COLORS[i],
                   marker=WR_MARKERS[j], s=70, zorder=3,
                   label=f"SS={ss}, WR={wr:.1f}")

# 1:1 line
lo = min(pred_all + meas_all) - 2
hi = max(pred_all + meas_all) + 2
ax.plot([lo, hi], [lo, hi], "--", color="black", lw=1.4, alpha=0.7, label="1:1 line")

ax.set_xlabel(r"Predicted timing: WAVE\_DUR $+$ SS (steps)", fontsize=10)
ax.set_ylabel(r"Measured burst timing (steps)", fontsize=10)
ax.set_title(r"\textbf{B.} Burst timing prediction confirmed ($\rm WR \leq 4.8$)",
             fontsize=9.5)
ax.legend(fontsize=7, framealpha=0.85, ncol=2)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# ============================================================
# Panel C: Buffer (m_blocked_norm) vs Field (f_norm) at T_BUILD
# ============================================================
ax = axes[2]

for j, wr in enumerate(WR_VALS):
    mb_vals, fn_vals, ss_labels = [], [], []
    for ss in SS_VALS:
        mb = mean_field(ss, wr, "m_blocked_norm_2000")
        fn = mean_field(ss, wr, "f_norm_2000")
        if not (math.isnan(mb) or math.isnan(fn)):
            mb_vals.append(mb)
            fn_vals.append(fn)
            ss_labels.append(ss)
    ax.scatter(mb_vals, fn_vals, color=WR_COLORS[j], marker=WR_MARKERS[j],
               s=80, zorder=3, label=f"WR={wr:.1f}")
    # Annotate SS values
    for mb, fn, ss in zip(mb_vals, fn_vals, ss_labels):
        ax.annotate(f"SS={ss}", (mb, fn), textcoords="offset points",
                    xytext=(4, 4), fontsize=6.5, color=WR_COLORS[j])

ax.set_xlabel(r"Buffer contents: $\|m_{\rm blocked}\|$ mean at $T_{\rm build}$", fontsize=10)
ax.set_ylabel(r"Field at $T_{\rm build}$: $\|F\|$ mean", fontsize=10)
ax.set_title(r"\textbf{C.} Buffer fills as field empties (WR trade-off)", fontsize=9.5)
ax.legend(fontsize=8, framealpha=0.85)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.text(0.97, 0.97,
        "High WR: buffer full,\nfield empty at $T_{\\rm build}$\n"
        "Low WR: buffer partial,\nfield populated",
        transform=ax.transAxes, fontsize=7.5, ha="right", va="top",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcyan", alpha=0.9))

fig.suptitle(
    r"The blocked-site buffer: direct measurement of the third reservoir in VCML",
    fontsize=10.5, y=1.02
)
fig.tight_layout()

OUT = os.path.join(DIR, "paper19_figure1")
fig.savefig(OUT + ".pdf", bbox_inches="tight")
fig.savefig(OUT + ".png", dpi=150, bbox_inches="tight")
print(f"Saved {OUT}.pdf and {OUT}.png")

# Summary
print("\nBURST TIMING vs PREDICTION (moderate WR):")
print(f"  {'SS':>4} {'WR':>5} | pred | meas | diff")
for ss in SS_VALS:
    for wr in [2.4, 4.8]:
        _, btim = burst_stats(ss, wr)
        pred = WAVE_DUR + ss
        diff = btim - pred if not math.isnan(btim) else float("nan")
        print(f"  {ss:>4} {wr:>5.1f} | {pred:>4} | {btim:>4.0f} | {diff:>+5.0f}")
