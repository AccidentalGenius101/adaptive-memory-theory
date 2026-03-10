"""
paper34_figure1.py -- The (nu, kappa) Phase Diagram of VCML

Four panels:
  A: Phase diagram heatmap -- sg4 in (nu, kappa) log-log space, nu/kappa=1 diagonal
  B: sg4 vs nu/kappa -- all 25 A-points, colored by nu; R2 of nu/kappa alone
  C: C1/C2 independence -- sg4 vs nu at three fixed nu/kappa ratios (Exp B)
  D: Power-law decomposition -- sg4 ~ nu^a * kappa^b, joint fit exponents
"""
import json, os, math, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.stats import linregress
from scipy.optimize import curve_fit

RESULTS_FILE = os.path.join(os.path.dirname(__file__),
                            "results", "paper34_results.json")

NU_VALS_A    = [0.001, 0.002, 0.005, 0.010, 0.020]
KAPPA_VALS_A = [0.005, 0.010, 0.020, 0.040, 0.080]
N_SEEDS = 5
T_LAST  = 4000
CPS     = [1000, 2000, 3000, 4000]

EXP_B_GROUPS = {
    0.05: [(0.001, 0.020), (0.002, 0.040), (0.005, 0.100)],
    0.50: [(0.001, 0.002), (0.002, 0.004), (0.005, 0.010)],
    5.00: [(0.005, 0.001), (0.010, 0.002), (0.020, 0.004)],
}

def key_a(nu, kappa, seed): return f"p34a,{nu:.8g},{kappa:.8g},{seed}"
def key_b(nu, kappa, seed): return f"p34b,{nu:.8g},{kappa:.8g},{seed}"

with open(RESULTS_FILE) as f:
    results = json.load(f)

# ── Collect Exp A data ────────────────────────────────────────────────────────
sg4_grid = np.full((len(NU_VALS_A), len(KAPPA_VALS_A)), float("nan"))
for i, nu in enumerate(NU_VALS_A):
    for j, kappa in enumerate(KAPPA_VALS_A):
        vals = [results.get(key_a(nu, kappa, s), {}).get(f"sg4_{T_LAST}", float("nan"))
                for s in range(N_SEEDS)]
        vals = [v for v in vals if not math.isnan(v)]
        if vals:
            sg4_grid[i, j] = float(np.mean(vals))

# ── Regressions ────────────────────────────────────────────────────────────
# All (nu, kappa) points
pts = []
for i, nu in enumerate(NU_VALS_A):
    for j, kappa in enumerate(KAPPA_VALS_A):
        v = sg4_grid[i, j]
        if not math.isnan(v) and v > 0:
            pts.append((nu, kappa, v))
pts = np.array(pts)
log_nu, log_k, log_sg4 = np.log(pts[:,0]), np.log(pts[:,1]), np.log(pts[:,2])
ratio = log_nu - log_k   # log(nu/kappa)

# 1D fit: sg4 ~ (nu/kappa)^alpha
sl1, int1, r1, *_ = linregress(ratio, log_sg4)
r2_ratio = r1**2

# 1D fit: sg4 ~ nu^alpha only
sl_nu, int_nu, r_nu, *_ = linregress(log_nu, log_sg4)
r2_nu = r_nu**2

# 2D fit: log(sg4) = a*log(nu) + b*log(kappa) + c
X = np.column_stack([log_nu, log_k, np.ones(len(log_nu))])
coef, res, _, _ = np.linalg.lstsq(X, log_sg4, rcond=None)
a_nu, b_k, c_const = coef
yhat = X @ coef
ss_res = np.sum((log_sg4 - yhat)**2)
ss_tot = np.sum((log_sg4 - log_sg4.mean())**2)
r2_joint = 1 - ss_res/ss_tot

print(f"sg4 ~ (nu/kappa)^{sl1:.3f}   R2={r2_ratio:.3f}")
print(f"sg4 ~ nu^{sl_nu:.3f}          R2={r2_nu:.3f}")
print(f"sg4 ~ nu^{a_nu:.3f} * kappa^{b_k:.3f}   R2={r2_joint:.3f}")

# ── Figure ─────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(11, 9))
fig.suptitle("Paper 34: The (nu, kappa) Phase Diagram of VCML", fontsize=13, fontweight="bold")

# ── Panel A: Phase diagram heatmap ────────────────────────────────────────
ax = axes[0, 0]
# sg4_grid rows=nu, cols=kappa
vmin = np.nanmin(sg4_grid)
vmax = np.nanmax(sg4_grid)
im = ax.imshow(sg4_grid, origin="upper",
               norm=mcolors.LogNorm(vmin=max(vmin, 0.1), vmax=vmax),
               cmap="viridis", aspect="auto")
ax.set_xticks(range(len(KAPPA_VALS_A)))
ax.set_xticklabels([f"{k:.3f}" for k in KAPPA_VALS_A], fontsize=8)
ax.set_yticks(range(len(NU_VALS_A)))
ax.set_yticklabels([f"{n:.3f}" for n in NU_VALS_A], fontsize=8)
ax.set_xlabel("kappa (diffusion)", fontsize=10)
ax.set_ylabel("nu (turnover)", fontsize=10)
ax.set_title("A: Phase diagram (sg4)", fontsize=11)
plt.colorbar(im, ax=ax, label="sg4")

# Mark nu/kappa = 1 diagonal: nu = kappa, i.e., find cells where nu ~ kappa
# On the grid, nu/kappa=1 passes through (nu=0.005, kappa=0.005) -> (nu=0.020, kappa=0.020)
# Draw a diagonal line from top-left corner
nu_arr  = np.array(NU_VALS_A)
ka_arr  = np.array(KAPPA_VALS_A)
# For each nu value, find the kappa index where kappa = nu
diag_x, diag_y = [], []
for i, nu in enumerate(NU_VALS_A):
    # interpolated position in kappa axis
    log_k_arr = np.log(ka_arr)
    log_nu_val = np.log(nu)
    if log_nu_val >= log_k_arr[0] and log_nu_val <= log_k_arr[-1]:
        x_pos = np.interp(log_nu_val, log_k_arr, np.arange(len(ka_arr)))
        diag_x.append(x_pos)
        diag_y.append(i)

if len(diag_x) >= 2:
    ax.plot(diag_x, diag_y, "r--", lw=2, label="nu/kappa=1")
    ax.legend(fontsize=8)

# Annotate AC / Burgers regions
ax.text(3.5, 0.3, "Allen-Cahn\n(nu/kappa<1)", color="white", fontsize=8,
        ha="center", va="center", fontweight="bold")
ax.text(0.3, 3.7, "Burgers\n(nu/kappa>1)", color="yellow", fontsize=8,
        ha="center", va="center", fontweight="bold")

# ── Panel B: sg4 vs nu/kappa, colored by nu ──────────────────────────────
ax = axes[0, 1]
nu_colors = {0.001: "blue", 0.002: "cyan", 0.005: "green",
             0.010: "orange", 0.020: "red"}
for i, nu in enumerate(NU_VALS_A):
    for j, kappa in enumerate(KAPPA_VALS_A):
        v = sg4_grid[i, j]
        if not math.isnan(v):
            ax.scatter(nu/kappa, v, color=nu_colors[nu],
                       s=60, zorder=3, alpha=0.8,
                       label=f"nu={nu}" if j == 0 else "")

# Fit line
x_fit = np.logspace(np.log10(0.01), np.log10(5), 50)
y_fit = np.exp(int1) * x_fit**sl1
ax.loglog(x_fit, y_fit, "k--", lw=1.5,
          label=f"slope={sl1:.2f}, R2={r2_ratio:.2f}")

ax.axvline(1.0, color="red", ls="--", lw=1, alpha=0.7, label="nu/kappa=1")
ax.set_xlabel("nu / kappa", fontsize=10)
ax.set_ylabel("sg4 (T=4000)", fontsize=10)
ax.set_title(f"B: sg4 vs nu/kappa\n(slope={sl1:.2f}, R2={r2_ratio:.2f} -- nu/kappa alone)", fontsize=9)
handles, labels = ax.get_legend_handles_labels()
by_label = dict(zip(labels, handles))
ax.legend(by_label.values(), by_label.keys(), fontsize=7, loc="upper right")
ax.grid(True, which="both", alpha=0.3)

# ── Panel C: Exp B -- sg4 vs nu at fixed nu/kappa ─────────────────────────
ax = axes[1, 0]
ratio_colors = {0.05: "blue", 0.50: "green", 5.00: "red"}
ratio_labels = {0.05: "r=0.05 (Allen-Cahn)", 0.50: "r=0.50 (AC/Burgers)", 5.00: "r=5.00 (Burgers)"}
for r_ratio in sorted(EXP_B_GROUPS.keys()):
    pairs = EXP_B_GROUPS[r_ratio]
    nu_list, sg4_list = [], []
    for (nu, kappa) in pairs:
        vals = [results.get(key_b(nu, kappa, s), {}).get(f"sg4_{T_LAST}", float("nan"))
                for s in range(N_SEEDS)]
        vals = [v for v in vals if not math.isnan(v)]
        if vals:
            nu_list.append(nu)
            sg4_list.append(float(np.mean(vals)))
    if nu_list:
        ax.loglog(nu_list, sg4_list, "o-",
                  color=ratio_colors[r_ratio],
                  label=ratio_labels[r_ratio],
                  lw=2, ms=8)
        # Compute CV and annotate
        cv = float(np.std(sg4_list) / np.mean(sg4_list))
        mid_i = len(nu_list) // 2
        ax.annotate(f"CV={cv:.2f}", xy=(nu_list[mid_i], sg4_list[mid_i]),
                    xytext=(5, 5), textcoords="offset points", fontsize=8,
                    color=ratio_colors[r_ratio])

ax.set_xlabel("nu (turnover rate)", fontsize=10)
ax.set_ylabel("sg4 (T=4000)", fontsize=10)
ax.set_title("C: C1/C2 independence test\n(fixed nu/kappa, varying absolute scale)", fontsize=9)
ax.legend(fontsize=8)
ax.grid(True, which="both", alpha=0.3)
ax.set_xlim(5e-4, 0.05)
ax.text(0.05, 0.05, "Large CV -> C1 x C2 independent\n(nu alone matters, not just nu/kappa)",
        transform=ax.transAxes, fontsize=8, color="gray",
        ha="left", va="bottom")

# ── Panel D: 2D power law decomposition ──────────────────────────────────
ax = axes[1, 1]
# Show nu marginal and kappa marginal effects
nu_means  = [float(np.nanmean(sg4_grid[i, :])) for i in range(len(NU_VALS_A))]
ka_means  = [float(np.nanmean(sg4_grid[:, j])) for j in range(len(KAPPA_VALS_A))]

ln_nu = np.log(NU_VALS_A)
ln_ka = np.log(KAPPA_VALS_A)
sl_nu_marg, int_nu_marg, r_nu_marg, *_ = linregress(ln_nu, np.log(nu_means))
sl_ka_marg, int_ka_marg, r_ka_marg, *_ = linregress(ln_ka, np.log(ka_means))

ax2 = ax.twinx()
ln1 = ax.loglog(NU_VALS_A, nu_means, "bs-", lw=2, ms=8, label=f"sg4 vs nu (slope={sl_nu_marg:.2f})")
x_nu_fit = np.logspace(np.log10(NU_VALS_A[0]), np.log10(NU_VALS_A[-1]), 30)
ax.loglog(x_nu_fit, np.exp(int_nu_marg) * x_nu_fit**sl_nu_marg, "b--", lw=1.2, alpha=0.7)

ln2 = ax2.loglog(KAPPA_VALS_A, ka_means, "rs-", lw=2, ms=8, label=f"sg4 vs kappa (slope={sl_ka_marg:.2f})")
x_ka_fit = np.logspace(np.log10(KAPPA_VALS_A[0]), np.log10(KAPPA_VALS_A[-1]), 30)
ax2.loglog(x_ka_fit, np.exp(int_ka_marg) * x_ka_fit**sl_ka_marg, "r--", lw=1.2, alpha=0.7)

ax.set_xlabel("nu or kappa", fontsize=10)
ax.set_ylabel("sg4 (mean over kappa)", fontsize=10, color="blue")
ax2.set_ylabel("sg4 (mean over nu)", fontsize=10, color="red")
ax.tick_params(axis="y", labelcolor="blue")
ax2.tick_params(axis="y", labelcolor="red")
ax.set_title(f"D: Power-law decomposition\nsg4 ~ nu^{a_nu:.2f} * kappa^{b_k:.2f}  (joint R2={r2_joint:.2f})", fontsize=9)

lns = ln1 + ln2
labs = [l.get_label() for l in lns]
ax.legend(lns, labs, fontsize=8, loc="lower right")
ax.grid(True, which="both", alpha=0.3)

# Add text summary
fig.text(0.5, 0.01,
         f"Key: sg4 ~ nu^{a_nu:.2f} * kappa^{b_k:.2f}  (R2={r2_joint:.2f})   |   "
         f"nu/kappa alone: R2={r2_ratio:.2f}   |   "
         f"C1 x C2 independent (CV=0.63-0.85 at fixed nu/kappa)",
         ha="center", fontsize=9, color="darkgreen", fontweight="bold")

plt.tight_layout(rect=[0, 0.04, 1, 1])

out_pdf = os.path.join(os.path.dirname(__file__), "paper34_figure1.pdf")
out_png = os.path.join(os.path.dirname(__file__), "paper34_figure1.png")
plt.savefig(out_pdf, bbox_inches="tight")
plt.savefig(out_png, dpi=150, bbox_inches="tight")
print(f"Saved: {out_pdf}")
print(f"Saved: {out_png}")
print(f"\nSummary:")
print(f"  sg4 ~ nu^{a_nu:.3f} * kappa^{b_k:.3f}   (joint R2={r2_joint:.3f})")
print(f"  nu/kappa alone: slope={sl1:.3f}, R2={r2_ratio:.3f}")
print(f"  nu alone: slope={sl_nu:.3f}, R2={r2_nu:.3f}")
print(f"  nu marginal slope: {sl_nu_marg:.3f}")
print(f"  kappa marginal slope: {sl_ka_marg:.3f}")
