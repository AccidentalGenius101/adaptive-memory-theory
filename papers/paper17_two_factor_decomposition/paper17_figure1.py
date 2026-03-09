"""
paper17_figure1.py -- Figure 1 for Paper 17

Four-panel figure (2x2). Each panel: one parameter sweep.
Dual y-axis: K_eff (left, blue circles) and C (right, orange squares).
Analytical K_eff prediction shown as dashed blue line (SS and KP sweeps).

Panels:
  A: SS sweep     -- K_eff tracks K=KAPPA/p_calm^SS; C flat
  B: MID_DECAY    -- C rises; K_eff flat
  C: SEED_BETA    -- C monotone; K_eff flat
  D: KAPPA sweep  -- both move; K_eff_pred = KAPPA/P_consol

Output: paper17_figure1.pdf, paper17_figure1.png
"""
import json, os, math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DIR = os.path.dirname(__file__)
results = json.load(open(os.path.join(DIR, "results", "paper17_results.json")))

FA_VALS = [0.10, 0.20, 0.40, 0.70, 0.90]
N_SEEDS = 5
SS_VALS  = [5, 7, 10, 15, 20]
MD_VALS  = [0.97, 0.98, 0.99, 0.995, 0.999]
SB_VALS  = [0.00, 0.10, 0.25, 0.50, 0.75]
KP_VALS  = [0.005, 0.010, 0.020, 0.040, 0.080]

SS_BASE = 10; MD_BASE = 0.99; SB_BASE = 0.25; KP_BASE = 0.020
WR = 4.8; WAVE_DUR = 15
P_CALM = 1.0 - WR / (2.0 * WAVE_DUR)   # 0.84


def make_key(sweep, val, fa, seed):
    if sweep == "ss":
        return f"ss,{int(val)},{fa:.4f},{seed}"
    elif sweep == "md":
        return f"md,{val:.5f},{fa:.4f},{seed}"
    elif sweep == "sb":
        return f"sb,{val:.4f},{fa:.4f},{seed}"
    else:
        return f"kp,{val:.4f},{fa:.4f},{seed}"


def get_mean(sweep, val, fa):
    keys = [make_key(sweep, val, fa, s) for s in range(N_SEEDS)
            if make_key(sweep, val, fa, s) in results]
    if not keys:
        return float("nan")
    return float(np.mean([results[k]["sg4_2000"] for k in keys]))


def sat_fit(fa_list, sg4_list):
    if len(fa_list) < 2:
        return float("nan"), float("nan"), float("nan")
    best_r2 = -1e9; best_C = 0; best_K = 0
    for K in np.linspace(0.005, 2.0, 800):
        x = [fa/(fa+K) for fa in fa_list]
        denom = sum(v*v for v in x)
        if denom == 0: continue
        C = float(np.dot(x, sg4_list) / denom)
        pred = [C*xx for xx in x]
        ss_res = sum((p-d)**2 for p,d in zip(pred, sg4_list))
        ss_tot = sum((d-float(np.mean(sg4_list)))**2 for d in sg4_list)
        r2 = 1 - ss_res/ss_tot if ss_tot > 0 else float("nan")
        if r2 > best_r2:
            best_r2 = r2; best_C = C; best_K = K
    return best_C, best_K, best_r2


def extract_CK(sweep, vals):
    Cs, Ks = [], []
    for val in vals:
        fa_data = []; sg4_data = []
        for fa in FA_VALS:
            m = get_mean(sweep, val, fa)
            if not math.isnan(m):
                fa_data.append(fa); sg4_data.append(m)
        C, K, _ = sat_fit(fa_data, sg4_data)
        Cs.append(C); Ks.append(K)
    return np.array(Cs), np.array(Ks)


fig, axes = plt.subplots(2, 2, figsize=(11, 8))
axes = axes.flatten()

BLUE = "#1f77b4"; ORANGE = "#d62728"; GREEN = "#2ca02c"

# ---- Panel A: SS sweep ----
ax = axes[0]
C_ss, K_ss = extract_CK("ss", SS_VALS)
K_pred_ss = [KP_BASE / (P_CALM ** ss) for ss in SS_VALS]

l1, = ax.plot(SS_VALS, K_ss, "o-", color=BLUE, ms=7, lw=2.0, label=r"$K_{\rm eff}$ (fitted)")
ax.plot(SS_VALS, K_pred_ss, "--", color=BLUE, lw=1.4, alpha=0.7,
        label=r"$K_{\rm pred}=\kappa/p_{\rm calm}^{SS}$")
ax2 = ax.twinx()
l3, = ax2.plot(SS_VALS, C_ss, "s-", color=ORANGE, ms=7, lw=2.0, label=r"$C$ (fitted)")
ax.set_xlabel("SS (consolidation threshold)", fontsize=9)
ax.set_ylabel(r"$K_{\rm eff}$", color=BLUE, fontsize=9)
ax2.set_ylabel(r"$C$", color=ORANGE, fontsize=9)
ax.tick_params(axis="y", labelcolor=BLUE)
ax2.tick_params(axis="y", labelcolor=ORANGE)
ax.set_title(r"\textbf{A.} SS sweep: $K_{\rm eff}$ shifts, $C$ flat", fontsize=9)
ax.spines["top"].set_visible(False)
lines = [l1, ax.lines[1], l3]
labs  = [l.get_label() for l in lines]
ax.legend(lines, labs, fontsize=6.5, framealpha=0.85)
ax.text(0.97, 0.55,
        r"$K_{\rm eff} = \kappa\,/\,p_{\rm calm}^{SS}$" + "\n"
        r"(no free parameters)",
        transform=ax.transAxes, fontsize=7, ha="right", va="top",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.85))

# ---- Panel B: MID_DECAY sweep ----
ax = axes[1]
C_md, K_md = extract_CK("md", MD_VALS)
K_flat_md = [KP_BASE / (P_CALM ** SS_BASE)] * len(MD_VALS)

ax.plot(MD_VALS, K_md, "o-", color=BLUE, ms=7, lw=2.0, label=r"$K_{\rm eff}$ (fitted)")
ax.plot(MD_VALS, K_flat_md, "--", color=BLUE, lw=1.4, alpha=0.7, label=r"$K_{\rm pred}$ (flat)")
ax2 = ax.twinx()
ax2.plot(MD_VALS, C_md, "s-", color=ORANGE, ms=7, lw=2.0, label=r"$C$ (fitted)")
ax.set_xlabel("MID\\_DECAY", fontsize=9)
ax.set_ylabel(r"$K_{\rm eff}$", color=BLUE, fontsize=9)
ax2.set_ylabel(r"$C$", color=ORANGE, fontsize=9)
ax.tick_params(axis="y", labelcolor=BLUE)
ax2.tick_params(axis="y", labelcolor=ORANGE)
ax.set_title(r"\textbf{B.} MID\_DECAY sweep: $C$ shifts, $K_{\rm eff}$ flat", fontsize=9)
ax.spines["top"].set_visible(False)
ax.text(0.05, 0.95,
        r"$C \propto 1/(1-\delta)$ approx" + "\n" + r"$K_{\rm eff}$ unchanged",
        transform=ax.transAxes, fontsize=7, va="top",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcyan", alpha=0.85))

# ---- Panel C: SEED_BETA sweep ----
ax = axes[2]
C_sb, K_sb = extract_CK("sb", SB_VALS)
K_flat_sb = [KP_BASE / (P_CALM ** SS_BASE)] * len(SB_VALS)

ax.plot(SB_VALS, K_sb, "o-", color=BLUE, ms=7, lw=2.0, label=r"$K_{\rm eff}$ (fitted)")
ax.plot(SB_VALS, K_flat_sb, "--", color=BLUE, lw=1.4, alpha=0.7, label=r"$K_{\rm pred}$ (flat)")
ax2 = ax.twinx()
ax2.plot(SB_VALS, C_sb, "s-", color=ORANGE, ms=7, lw=2.0, label=r"$C$ (fitted)")
ax.set_xlabel(r"SEED\_BETA (copy-forward strength)", fontsize=9)
ax.set_ylabel(r"$K_{\rm eff}$", color=BLUE, fontsize=9)
ax2.set_ylabel(r"$C$", color=ORANGE, fontsize=9)
ax.tick_params(axis="y", labelcolor=BLUE)
ax2.tick_params(axis="y", labelcolor=ORANGE)
ax.set_title(r"\textbf{C.} SEED\_BETA sweep: $C$ shifts, $K_{\rm eff}$ flat", fontsize=9)
ax.spines["top"].set_visible(False)
ax.text(0.05, 0.95,
        "Copy-forward strength controls\n"
        "plateau amplitude; crossover\n"
        "unchanged",
        transform=ax.transAxes, fontsize=7, va="top",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.85))

# ---- Panel D: KAPPA sweep ----
ax = axes[3]
C_kp, K_kp = extract_CK("kp", KP_VALS)
K_pred_kp = [kp / (P_CALM ** SS_BASE) for kp in KP_VALS]

l1, = ax.plot(KP_VALS, K_kp, "o-", color=BLUE, ms=7, lw=2.0, label=r"$K_{\rm eff}$ (fitted)")
ax.plot(KP_VALS, K_pred_kp, "--", color=BLUE, lw=1.4, alpha=0.7,
        label=r"$K_{\rm pred}=\kappa/P_{\rm consol}$")
ax2 = ax.twinx()
l3, = ax2.plot(KP_VALS, C_kp, "s-", color=ORANGE, ms=7, lw=2.0, label=r"$C$ (fitted)")
ax.set_xlabel(r"$\kappa$ (diffusion coefficient)", fontsize=9)
ax.set_ylabel(r"$K_{\rm eff}$", color=BLUE, fontsize=9)
ax2.set_ylabel(r"$C$", color=ORANGE, fontsize=9)
ax.tick_params(axis="y", labelcolor=BLUE)
ax2.tick_params(axis="y", labelcolor=ORANGE)
ax.set_title(r"\textbf{D.} $\kappa$ sweep: \textit{both} $K_{\rm eff}$ and $C$ shift", fontsize=9)
ax.spines["top"].set_visible(False)
lines = [l1, ax.lines[1], l3]
labs  = [l.get_label() for l in lines]
ax.legend(lines, labs, fontsize=6.5, framealpha=0.85, loc="upper left")
ax.text(0.97, 0.95,
        r"$K_{\rm eff}$ shifts right $+$" + "\n"
        r"$C$ decreases (gradient" + "\n"
        r"erasure by diffusion)",
        transform=ax.transAxes, fontsize=7, ha="right", va="top",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.7))

fig.suptitle(
    r"Saturation law $\mathrm{sg4}=C\cdot\mathrm{FA}/(\mathrm{FA}+K_{\rm eff})$ is complete: "
    r"every parameter maps onto $C$, $K_{\rm eff}$, or both",
    fontsize=9.5, y=1.01
)
fig.tight_layout()

OUT = os.path.join(DIR, "paper17_figure1")
fig.savefig(OUT + ".pdf", bbox_inches="tight")
fig.savefig(OUT + ".png", dpi=150, bbox_inches="tight")
print(f"Saved {OUT}.pdf and {OUT}.png")

# Print summary table
print("\nSUMMARY TABLE (C, K_eff per sweep):")
for sweep, vals, name in [("ss",SS_VALS,"SS"),("md",MD_VALS,"MID_DECAY"),
                           ("sb",SB_VALS,"SEED_BETA"),("kp",KP_VALS,"KAPPA")]:
    Cs, Ks = extract_CK(sweep, vals)
    print(f"\n{name}:")
    for v, c, k in zip(vals, Cs, Ks):
        print(f"  {v:>8} -> C={c:7.2f}  K_eff={k:.4f}")
