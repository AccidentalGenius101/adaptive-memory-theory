"""
paper24_figure1.py -- Figure 1 for Paper 24

Birth Bias Axis: Testing the Allen-Cahn -> Burgers Transition.

Four panels:
  A: xi(t) trajectories for all alpha values (mean +/- SEM across seeds)
  B: xi_late vs alpha (equilibrium correlation length)
  C: sg4(T_END) vs alpha (zone differentiation amplitude)
  D: delta_xi (= xi_late - xi_early) vs alpha with zero reference line

Key result: No sign flip in delta_xi (all negative) but xi_late increases
monotonically with alpha, showing partial directional effect.  The full
Burgers transition is suppressed because nu/kappa = 0.001/0.020 = 0.05 << 1,
so the advection term is ~20x weaker than diffusion at standard parameters.
"""
import json, os, math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import linregress

DIR = os.path.dirname(__file__)
results = json.load(open(os.path.join(DIR, "results", "paper24_results.json")))

# ── Parameters (must match paper24_birth_bias.py) ─────────────────────────────
T_END      = 3000
CPS        = list(range(200, T_END + 1, 200))
ALPHA_VALS = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
N_SEEDS    = 5
ZONE_W     = 5
MAX_LAG    = 10
XI_CAP     = 30.0
NU         = 0.001
KAPPA      = 0.020


def make_key(alpha, seed):
    return f"p24,{alpha:.8g},{seed}"


def fit_xi(corr):
    r_vals, log_c = [], []
    for r in range(1, min(ZONE_W + 1, MAX_LAG + 1)):
        if r >= len(corr):
            break
        c = corr[r]
        if not math.isnan(c) and 0.05 < c <= 1.0:
            r_vals.append(r)
            log_c.append(math.log(c))
    if len(r_vals) < 2:
        return float("nan")
    slope, _, _, _, _ = linregress(r_vals, log_c)
    if slope >= 0:
        return float("nan")
    return -1.0 / slope


def xi_trajectory(alpha):
    """Return (times, means, sems) of xi(t) for a given alpha."""
    by_t = {t: [] for t in CPS}
    for seed in range(N_SEEDS):
        key = make_key(alpha, seed)
        if key not in results:
            continue
        for t in CPS:
            corr = results[key].get(f"corr_{t}")
            if corr is not None:
                xi = fit_xi(corr)
                if not math.isnan(xi) and xi <= XI_CAP:
                    by_t[t].append(xi)
    times, means, sems = [], [], []
    for t in CPS:
        v = by_t[t]
        if v:
            times.append(t)
            means.append(float(np.mean(v)))
            sems.append(float(np.std(v) / math.sqrt(len(v))) if len(v) > 1 else 0.0)
    return times, means, sems


def xi_late_mean(alpha, t_min=2400):
    """Mean xi_late (t >= t_min) across all seeds and checkpoints."""
    vals = []
    for seed in range(N_SEEDS):
        key = make_key(alpha, seed)
        if key not in results:
            continue
        for t in CPS:
            if t < t_min:
                continue
            corr = results[key].get(f"corr_{t}")
            if corr is not None:
                xi = fit_xi(corr)
                if not math.isnan(xi) and xi <= XI_CAP:
                    vals.append(xi)
    if not vals:
        return float("nan"), 0.0
    return float(np.mean(vals)), float(np.std(vals) / math.sqrt(len(vals)))


def xi_early_mean(alpha, t_max=1000):
    """Mean xi_early (t <= t_max, first reliable estimates) across seeds."""
    vals = []
    for seed in range(N_SEEDS):
        key = make_key(alpha, seed)
        if key not in results:
            continue
        for t in CPS:
            if t > t_max:
                break
            corr = results[key].get(f"corr_{t}")
            if corr is not None:
                xi = fit_xi(corr)
                if not math.isnan(xi) and xi <= XI_CAP:
                    vals.append(xi)
    if not vals:
        return float("nan"), 0.0
    return float(np.mean(vals)), float(np.std(vals) / math.sqrt(len(vals)))


def sg4_end_mean(alpha):
    vals = []
    for seed in range(N_SEEDS):
        key = make_key(alpha, seed)
        if key not in results:
            continue
        v = results[key].get(f"sg4_{T_END}")
        if v is not None:
            vals.append(v)
    if not vals:
        return float("nan"), 0.0
    return float(np.mean(vals)), float(np.std(vals) / math.sqrt(len(vals)))


# ── Collect data ───────────────────────────────────────────────────────────────
late_xi_m, late_xi_s    = [], []
early_xi_m, early_xi_s  = [], []
delta_xi_m, delta_xi_s  = [], []
sg4_m, sg4_s            = [], []

for alpha in ALPHA_VALS:
    lm, ls = xi_late_mean(alpha)
    em, es = xi_early_mean(alpha)
    gm, gs = sg4_end_mean(alpha)
    late_xi_m.append(lm);  late_xi_s.append(ls)
    early_xi_m.append(em); early_xi_s.append(es)
    sg4_m.append(gm);      sg4_s.append(gs)
    # delta_xi: propagate SEM in quadrature
    dm = lm - em if not (math.isnan(lm) or math.isnan(em)) else float("nan")
    ds = math.sqrt(ls**2 + es**2) if not math.isnan(dm) else 0.0
    delta_xi_m.append(dm)
    delta_xi_s.append(ds)

# ── Figure ────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(12, 9))
axes = axes.flatten()

CMAP   = plt.cm.plasma
colors = [CMAP(i / (len(ALPHA_VALS) - 1)) for i in range(len(ALPHA_VALS))]
ALPHA_LABELS = [rf"$\alpha={a}$" for a in ALPHA_VALS]

# ── Panel A: xi(t) trajectories ───────────────────────────────────────────────
ax = axes[0]
for i, alpha in enumerate(ALPHA_VALS):
    times, means, sems = xi_trajectory(alpha)
    ax.plot(times, means, "-o", color=colors[i], ms=4, lw=1.5,
            label=ALPHA_LABELS[i], zorder=3)
    ax.fill_between(times,
                    [m - s for m, s in zip(means, sems)],
                    [m + s for m, s in zip(means, sems)],
                    color=colors[i], alpha=0.18)
ax.set_xlabel("Time step", fontsize=11)
ax.set_ylabel(r"$\xi(t)$ (correlation length, sites)", fontsize=11)
ax.set_title(r"\textbf{A.} $\xi(t)$ trajectories for all $\alpha$" "\n"
             r"All curves decrease: Allen-Cahn regime persists at "
             r"$\nu/\kappa = 0.05$",
             fontsize=10)
ax.legend(fontsize=8, framealpha=0.9, ncol=2)
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
ax.text(0.97, 0.97,
        r"$\nu/\kappa = 0.001/0.020 = 0.05$" "\n"
        r"Advection term $\propto \nu$" "\n"
        r"Diffusion term $\propto \kappa$" "\n"
        r"$\Rightarrow$ diffusion dominates, Allen-Cahn",
        transform=ax.transAxes, fontsize=8, ha="right", va="top",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.9))

# ── Panel B: xi_late vs alpha ─────────────────────────────────────────────────
ax = axes[1]
ax.errorbar(ALPHA_VALS, late_xi_m, yerr=late_xi_s,
            fmt="o-", color="#2166ac", ms=8, lw=2.0, capsize=4,
            zorder=5, label=r"$\xi_{\rm late}$ (mean, $t \geq 2400$)")

# Linear fit to show trend
valid = [(a, m) for a, m in zip(ALPHA_VALS, late_xi_m) if not math.isnan(m)]
if len(valid) >= 3:
    ax_fit = np.array([v[0] for v in valid])
    ay_fit = np.array([v[1] for v in valid])
    slope, intercept, r_val, _, _ = linregress(ax_fit, ay_fit)
    a_line = np.linspace(0, 1, 100)
    ax.plot(a_line, slope * a_line + intercept, "--",
            color="#aec7e8", lw=1.5,
            label=rf"Linear fit: slope={slope:.2f} ($R^2={r_val**2:.3f}$)")

ax.set_xlabel(r"Birth bias $\alpha$", fontsize=11)
ax.set_ylabel(r"$\xi_{\rm late}$ (equilibrium corr. length, sites)", fontsize=11)
ax.set_title(r"\textbf{B.} Equilibrium $\xi_{\rm late}$ vs $\alpha$" "\n"
             r"Monotone increase: directional effect of fitness bias confirmed",
             fontsize=10)
ax.legend(fontsize=9, framealpha=0.9)
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
ax.text(0.05, 0.97,
        r"$\alpha=0$: uniform (Allen-Cahn)" "\n"
        r"$\alpha=1$: fitness-proportionate" "\n"
        r"$\xi_{\rm late}$ rises $2.4 \to 4.5$ sites" "\n"
        r"(partial Burgers signature)",
        transform=ax.transAxes, fontsize=8, ha="left", va="top",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcyan", alpha=0.9))

# ── Panel C: sg4(T_END) vs alpha ──────────────────────────────────────────────
ax = axes[2]
ax.errorbar(ALPHA_VALS, sg4_m, yerr=sg4_s,
            fmt="s-", color="#d62728", ms=8, lw=2.0, capsize=4, zorder=5)
ax.set_xlabel(r"Birth bias $\alpha$", fontsize=11)
ax.set_ylabel(r"sg4$(T_{\rm end})$ (zone differentiation)", fontsize=11)
ax.set_title(r"\textbf{C.} Zone differentiation sg4 vs $\alpha$" "\n"
             r"Fitness bias enhances zone structure at $\alpha=0.2$ and $\alpha=1.0$",
             fontsize=10)
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
ax.text(0.97, 0.97,
        r"Non-monotone: $\alpha=0.2$ and" "\n"
        r"$\alpha=1.0$ give highest sg4." "\n"
        r"$\alpha=0.4$ dip may reflect" "\n"
        r"competition between regimes.",
        transform=ax.transAxes, fontsize=8, ha="right", va="top",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.9))

# ── Panel D: delta_xi vs alpha ────────────────────────────────────────────────
ax = axes[3]
ax.axhline(0, color="black", lw=1.2, ls="--", alpha=0.6,
           label="Zero line (sign flip threshold)")
ax.errorbar(ALPHA_VALS, delta_xi_m, yerr=delta_xi_s,
            fmt="D-", color="#7b2d8b", ms=8, lw=2.0, capsize=4, zorder=5,
            label=r"$\Delta\xi = \xi_{\rm late} - \xi_{\rm early}$")
ax.set_xlabel(r"Birth bias $\alpha$", fontsize=11)
ax.set_ylabel(r"$\Delta\xi$ (late $-$ early correlation length)", fontsize=11)
ax.set_title(r"\textbf{D.} $\Delta\xi$ vs $\alpha$: no sign flip observed" "\n"
             r"Allen-Cahn regime persists; Burgers transition requires "
             r"$\nu \gtrsim \kappa$",
             fontsize=10)
ax.legend(fontsize=9, framealpha=0.9)
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
ax.text(0.05, 0.03,
        r"\textbf{Prediction}: sign flip at $\alpha^*$" "\n"
        r"\textbf{Observed}: all $\Delta\xi < 0$" "\n"
        r"\textbf{Reason}: $\nu/\kappa = 0.05 \ll 1$" "\n"
        r"Need $\nu \sim \kappa$ for Burgers" "\n"
        r"(i.e., $\nu \approx 0.02$, 20x standard)",
        transform=ax.transAxes, fontsize=8, ha="left", va="bottom",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="mistyrose", alpha=0.9))

fig.suptitle(
    r"Birth Bias Axis: Partial Burgers Signal at $\nu/\kappa = 0.05$ "
    r"($\xi_{\rm late}$ increases, sign flip absent)",
    fontsize=11, y=1.01
)
fig.tight_layout()

OUT = os.path.join(DIR, "paper24_figure1")
fig.savefig(OUT + ".pdf", bbox_inches="tight")
fig.savefig(OUT + ".png", dpi=150, bbox_inches="tight")
print(f"Saved {OUT}.pdf and {OUT}.png")

# ── Summary table ──────────────────────────────────────────────────────────────
print(f"\n{'alpha':>6} | {'xi_early':>9} | {'xi_late':>9} | "
      f"{'delta_xi':>10} | {'sg4_end':>9}")
print("  " + "-" * 60)
for i, alpha in enumerate(ALPHA_VALS):
    dxi_str = (f"{delta_xi_m[i]:>+10.2f}"
               if not math.isnan(delta_xi_m[i]) else f"{'nan':>10}")
    print(f"  {alpha:>5.1f} | {early_xi_m[i]:>9.2f} | {late_xi_m[i]:>9.2f} | "
          f"{dxi_str} | {sg4_m[i]:>9.1f}")
print()
print(f"  nu/kappa ratio = {NU}/{KAPPA} = {NU/KAPPA:.3f}  (Burgers requires ~1.0)")
print(f"  xi_late range: {min(late_xi_m):.2f} -> {max(late_xi_m):.2f} sites")
print(f"  (monotone directional effect despite no sign flip)")
