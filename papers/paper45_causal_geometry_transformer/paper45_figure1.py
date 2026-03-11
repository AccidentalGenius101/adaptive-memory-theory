"""
Paper 45: Figure generation
Reads paper45_analysis.json produced by paper45_experiments.py.

Figure 1 (2 panels):
  (a) Layer profile — partial correlation of ablation influence with geometry,
      controlling for sequence distance. Includes rollout and random control.
  (b) Dependency-distance matched analysis at the representative layer.
      Dep vs non-dep pairwise geometry at matched sequence distance bands.

Run: py paper45_figure1.py
"""

import json, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

ANALYSIS_FILE = "paper45_analysis.json"
FIGURE_FILE   = "paper45_figure1.pdf"

def load():
    try:
        with open(ANALYSIS_FILE) as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Run paper45_experiments.py first to generate {ANALYSIS_FILE}")
        sys.exit(1)

def plot(analysis):
    n_layers    = analysis["n_layers"]
    layer_stats = analysis["layer_stats"]
    matched     = analysis["matched"]
    n_sent      = analysis["n_sentences"]
    n_pairs     = analysis["n_pairs"]

    layers      = [s["layer"] for s in layer_stats]
    pcorr_abl   = [s["pcorr_abl"]  for s in layer_stats]
    pcorr_roll  = [s["pcorr_roll"] for s in layer_stats]
    pcorr_rand  = [s["pcorr_rand"] for s in layer_stats]
    p_abl       = [s["p_abl"]      for s in layer_stats]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    fig.suptitle(
        f"Paper 45 — Causal Influence and Representation Geometry in GPT-2\n"
        f"({n_sent} UD sentences, {n_pairs} token pairs)",
        fontsize=11, fontweight="bold"
    )

    # ── Panel (a): Layer profile ──────────────────────────────────────────────
    ax = axes[0]

    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")

    ax.plot(layers, pcorr_abl,  "o-", color="#1f77b4", linewidth=2,
            markersize=6, label="ablation influence (mean embed)")
    ax.plot(layers, pcorr_roll, "s--", color="#ff7f0e", linewidth=1.5,
            markersize=5, label="attention rollout")
    ax.plot(layers, pcorr_rand, "^:", color="#7f7f7f", linewidth=1.2,
            markersize=4, label="random-token control")

    # Mark significant layers (p < 0.05) for ablation
    for l, r, p in zip(layers, pcorr_abl, p_abl):
        if p < 0.05:
            ax.plot(l, r, "o", color="#1f77b4", markersize=11,
                    markerfacecolor="none", markeredgewidth=1.5)

    ax.set_xlabel("GPT-2 Layer (0 = embedding)", fontsize=10)
    ax.set_ylabel("Partial correlation\n(controlling for seq distance)", fontsize=10)
    ax.set_title("(a) Layer-wise: geom ~ ablation inf | seq dist", fontsize=10)
    ax.legend(fontsize=8, loc="lower left")
    ax.set_xticks(range(0, n_layers, 2))
    ax.grid(True, alpha=0.3)

    note = ("Open circles: p < 0.05\n"
            "Negative = more influence -> closer geometry")
    ax.text(0.03, 0.97, note, transform=ax.transAxes,
            fontsize=7, va="top", color="#444444")

    # ── Panel (b): Dependency-distance matched ────────────────────────────────
    ax2 = axes[1]

    if matched:
        ranges    = [m["range"]     for m in matched]
        dep_m     = [m["dep_mean"]  for m in matched]
        ndep_m    = [m["ndep_mean"] for m in matched]
        p_vals    = [m["p"]         for m in matched]
        n_bands   = len(ranges)
        x         = np.arange(n_bands)
        w         = 0.35

        bars_dep  = ax2.bar(x - w/2, dep_m,  w, color="#2196F3", alpha=0.85,
                            label="syntactically dependent")
        bars_ndep = ax2.bar(x + w/2, ndep_m, w, color="#FF9800", alpha=0.85,
                            label="syntactically independent")

        # Significance markers
        for xi, (d, nd, p) in enumerate(zip(dep_m, ndep_m, p_vals)):
            top = max(d, nd) + 0.005
            if p < 0.001:
                marker = "***"
            elif p < 0.01:
                marker = "** "
            elif p < 0.05:
                marker = "*  "
            else:
                marker = "ns "
            ax2.text(xi, top, marker, ha="center", va="bottom", fontsize=9)

        ax2.set_xticks(x)
        ax2.set_xticklabels(ranges)
        ax2.set_xlabel("Sequence distance band", fontsize=10)
        ax2.set_ylabel("Mean geometric distance\n(1 - cosine similarity)", fontsize=10)
        ax2.set_title(
            f"(b) Geometry at matched distance (layer {n_layers//2})\n"
            "Dep pairs should be closer than non-dep at same seq dist",
            fontsize=9
        )
        ax2.legend(fontsize=8)
        ax2.grid(True, alpha=0.3, axis="y")
    else:
        ax2.text(0.5, 0.5, "No matched data\n(check dep pair count)",
                 ha="center", va="center", transform=ax2.transAxes)

    plt.tight_layout()
    plt.savefig(FIGURE_FILE, dpi=150, bbox_inches="tight")
    print(f"Saved: {FIGURE_FILE}")


if __name__ == "__main__":
    analysis = load()
    plot(analysis)
