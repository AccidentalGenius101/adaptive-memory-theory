"""
paper13_figure1.py -- Figure 1 for Paper 13

Three-panel figure:
  Panel A: Predicted vs Observed nu* scatter -- tests geometric mean law.
           One point per condition (12 conditions x 2 SB values = 24 points).
           Perfect prediction = diagonal line. Systematic offset reveals bias.
  Panel B: Viability volume map -- sg4 heatmap over nu x (FD, FA) conditions
           at reference SB=0.25, with geometric mean marked.
  Panel C: Copy-forward effect -- delta_nu* (SB=0.25 minus SB=0.00) as bar chart
           for each condition. Shows where copy-forward extends the window.

Output: paper13_figure1.pdf, paper13_figure1.png
"""
import json, os
import numpy as np
import math
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

DIR = os.path.dirname(__file__)
r1 = json.load(open(os.path.join(DIR, "results", "paper13_exp1_results.json")))

DEATH_PS = [0.0001, 0.0002, 0.0003, 0.0005, 0.001, 0.002,
            0.003, 0.005, 0.010, 0.020, 0.040, 0.080]
FIELD_DECAY_VALS = [0.9997, 0.999]
FIELD_ALPHA_VALS = [0.08, 0.16, 0.32]
SEED_BETA_VALS   = [0.00, 0.25]

WR = 4.8; WAVE_DUR = 15; SS = 10
pc = (1 - WR / (2 * WAVE_DUR)) ** (SS + WAVE_DUR - 1)

def nu_cryst_fn(fd):
    return abs(math.log(fd)) / math.log(2)

def nu_max_fn(fd, fa):
    return pc * fa * (fd ** (1.0 / 0.001))

def get_sg4_profile(fd, fa, sb):
    return [float(np.mean([v["sg4"] for v in r1[f"{fd},{fa},{sb},{dp}"]]))
            for dp in DEATH_PS]

fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))

# --- Panel A: Predicted vs observed nu* ---
ax = axes[0]
FD_MARKERS = {0.9997: "o", 0.999: "s"}
FA_COLORS  = {0.08: "#1f77b4", 0.16: "#2ca02c", 0.32: "#d62728"}
SB_FILLS   = {0.00: "none", 0.25: None}   # hollow vs filled

pred_list = []
obs_list  = []
for fd in FIELD_DECAY_VALS:
    nc = nu_cryst_fn(fd)
    for fa in FIELD_ALPHA_VALS:
        nm = nu_max_fn(fd, fa)
        nu_pred = math.sqrt(nc * nm)
        for sb in SEED_BETA_VALS:
            profile = get_sg4_profile(fd, fa, sb)
            nu_obs = DEATH_PS[int(np.argmax(profile))]
            pred_list.append(nu_pred)
            obs_list.append(nu_obs)
            fc = FA_COLORS[fa] if sb == 0.25 else "none"
            ec = FA_COLORS[fa]
            ax.scatter(nu_pred, nu_obs, marker=FD_MARKERS[fd],
                       facecolors=fc, edgecolors=ec, s=60, linewidths=1.5,
                       zorder=3)

# Diagonal (perfect prediction)
lims = [5e-5, 5e-3]
ax.plot(lims, lims, "k--", lw=1.0, label="Perfect prediction", zorder=1)
ax.plot(lims, [x*0.5 for x in lims], "gray", lw=0.8, ls=":", zorder=1)
ax.plot(lims, [x*0.2 for x in lims], "gray", lw=0.8, ls=":", zorder=1)
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel(r"Predicted $\nu^*$ (geometric mean)", fontsize=10)
ax.set_ylabel(r"Observed $\nu^*$", fontsize=10)
ax.set_title(r"\textbf{A.} Geometric mean law test", fontsize=10)
ax.text(0.05, 0.05, "Points below diagonal:\ngeometric mean over-predicts",
        transform=ax.transAxes, fontsize=7, va="bottom",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.7))
# Legend patches
patches = [
    mpatches.Patch(facecolor=FA_COLORS[fa], label=f"FA={fa}") for fa in FIELD_ALPHA_VALS
]
patches += [
    mpatches.Patch(facecolor="w", edgecolor="k", label="SB=0 (hollow)"),
    mpatches.Patch(facecolor="gray", label="SB=0.25 (filled)"),
]
ax.legend(handles=patches, fontsize=6, loc="upper left", framealpha=0.85)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# --- Panel B: sg4 heatmap at SB=0.25 with geometric mean marked ---
ax = axes[1]
# 6 conditions (2 FD x 3 FA) x 12 nu values
heatmap = np.zeros((6, 12))
ylabels = []
for ci, (fd, fa) in enumerate([(fd, fa)
                                for fd in FIELD_DECAY_VALS
                                for fa in FIELD_ALPHA_VALS]):
    profile = get_sg4_profile(fd, fa, 0.25)
    heatmap[ci, :] = profile
    ylabels.append(f"FD={fd},FA={fa}")

im = ax.imshow(heatmap, aspect="auto", origin="upper", cmap="hot_r",
               interpolation="nearest")
ax.set_yticks(range(6))
ax.set_yticklabels(ylabels, fontsize=6)
ax.set_xticks(range(12))
ax.set_xticklabels([f"{dp:.0e}" for dp in DEATH_PS], fontsize=5.5, rotation=45)
ax.set_xlabel(r"Turnover rate $\nu$", fontsize=10)
ax.set_title(r"\textbf{B.} sg4 viability map (SB=0.25)", fontsize=10)
plt.colorbar(im, ax=ax, label="sg4")

# Mark geometric mean prediction for each condition
for ci, (fd, fa) in enumerate([(fd, fa)
                                for fd in FIELD_DECAY_VALS
                                for fa in FIELD_ALPHA_VALS]):
    nc = nu_cryst_fn(fd)
    nm = nu_max_fn(fd, fa)
    nu_pred = math.sqrt(nc * nm)
    # Find nearest nu index
    dists = [abs(math.log(nu_pred) - math.log(dp)) for dp in DEATH_PS]
    xi = dists.index(min(dists))
    ax.scatter(xi, ci, marker="|", color="cyan", s=80, linewidths=2, zorder=4)

# --- Panel C: Copy-forward shift delta_nu* ---
ax = axes[2]
cond_labels = []
delta_vals  = []
colors_bar  = []
FA_COLORS_LIST = [FA_COLORS[fa] for fa in FIELD_ALPHA_VALS] * 2

for ci, (fd, fa) in enumerate([(fd, fa)
                                for fd in FIELD_DECAY_VALS
                                for fa in FIELD_ALPHA_VALS]):
    profile_off = get_sg4_profile(fd, fa, 0.00)
    profile_on  = get_sg4_profile(fd, fa, 0.25)
    nu_off = DEATH_PS[int(np.argmax(profile_off))]
    nu_on  = DEATH_PS[int(np.argmax(profile_on))]
    delta = math.log10(nu_on / nu_off) if nu_on != nu_off else 0.0
    delta_vals.append(delta)
    cond_labels.append(f"FD={fd}\nFA={fa}")
    colors_bar.append(FA_COLORS_LIST[ci])

y_pos = range(len(cond_labels))
bars = ax.barh(list(y_pos), delta_vals, color=colors_bar, alpha=0.8, edgecolor="k", lw=0.5)
ax.axvline(x=0, color="k", lw=1.0)
ax.set_yticks(list(y_pos))
ax.set_yticklabels(cond_labels, fontsize=6.5)
ax.set_xlabel(r"$\log_{10}(\nu^*_{\rm SB=0.25}\,/\,\nu^*_{\rm SB=0})$", fontsize=9)
ax.set_title(r"\textbf{C.} Copy-forward shift in $\nu^*$", fontsize=10)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.text(0.05, 0.95, "Positive: SB=0.25\nshifts nu* right",
        transform=ax.transAxes, fontsize=7, va="top",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8))

fig.suptitle(
    r"The 4D viability volume: geometric mean law partially holds; copy-forward extends the upper boundary",
    fontsize=9.5, y=1.01
)
fig.tight_layout()

OUT = os.path.join(DIR, "paper13_figure1")
fig.savefig(OUT + ".pdf", bbox_inches="tight")
fig.savefig(OUT + ".png", dpi=150, bbox_inches="tight")
print(f"Saved {OUT}.pdf and {OUT}.png")
