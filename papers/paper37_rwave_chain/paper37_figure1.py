"""
paper37_figure1.py -- Figure 1 for Paper 37: r_wave sweep + k-stage relay chain

Three panels:
  (a) G vs zw/r_wave, colored by r_wave -- shows collapse onto common threshold
      except r_wave=4 which is suppressed by over-perturbation
  (b) N_crit vs r_wave (log-log): observed vs L/(4*r_wave) and L/(5*r_wave) predictions
  (c) G_k vs k for sigma in {0, 2, 5}: chain relay -- no compounding
"""
import json, math, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import defaultdict

RES = "results/paper37_results.json"
with open(RES) as f: data = json.load(f)

by = defaultdict(list)
for r in data:
    by[(r['exp'], r.get('r_wave'), r.get('n_zones'), r.get('coupling'),
        r.get('sigma'), r.get('k_stages'))].append(r)

def mn(lst):
    v = [x for x in lst if not math.isnan(x)]
    return float(np.mean(v)) if v else float('nan')

HALF = 40

# ── Exp A: G(N) per r_wave ──────────────────────────────────────────────────
R_WAVE_SWEEP = [1, 2, 3, 4]
N_PER_RWAVE = {1:[5,8,10,20], 2:[2,4,5,8], 3:[2,4,5], 4:[2,4,5]}

COLORS_RW = {1:'#e31a1c', 2:'#1f77b4', 3:'#2ca02c', 4:'#ff7f0e'}
LABELS_RW = {1:r'$r_w=1$', 2:r'$r_w=2$ (std)', 3:r'$r_w=3$', 4:r'$r_w=4$'}

gdata_A = {}   # {rw: [(zw_rw, G, se), ...]}
ncrit_obs = {}

for rw in R_WAVE_SWEEP:
    pts = []
    for N in N_PER_RWAVE[rw]:
        geo  = by[('A', rw, N, 'geo', None, None)]
        ctrl = by[('A', rw, N, 'ctrl', None, None)]
        if not geo or not ctrl: continue
        sg_list = [r.get('l2_sg4n', float('nan')) for r in geo]
        sc_list = [r.get('l2_sg4n', float('nan')) for r in ctrl]
        sg = mn(sg_list); sc = mn(sc_list)
        G  = sg/sc if sc > 1e-6 else float('nan')
        seed_Gs = [g/c for g, c in zip(sg_list, sc_list)
                   if not math.isnan(g) and not math.isnan(c) and c > 1e-6]
        se = float(np.std(seed_Gs, ddof=1)/math.sqrt(len(seed_Gs))) if len(seed_Gs)>1 else 0
        zw = HALF // N
        pts.append((zw/rw, G, se, N))
    pts.sort()
    gdata_A[rw] = pts
    nc_obs = max((N for _, G, _, N in pts if not math.isnan(G) and G > 1.5), default=float('nan'))
    ncrit_obs[rw] = nc_obs

# ── Exp B: k-stage chain ────────────────────────────────────────────────────
SIGMA_SWEEP = [0, 2, 5]
K_STAGES = [1, 2, 3, 4]
COLORS_SIG = {0:'#1f77b4', 2:'#ff7f0e', 5:'#2ca02c'}
LABELS_SIG = {0:r'$\sigma=0$ (no noise)', 2:r'$\sigma=2$', 5:r'$\sigma=5$'}

# Ctrl baseline: B_ctrl records have l1_sg4n and ctrl_sg4n
ctrl_runs = [r for r in data if r.get('exp') == 'B_ctrl']
ctrl_sg4n = mn([r.get('ctrl_sg4n', float('nan')) for r in ctrl_runs])
l1_sg4n_base = mn([r.get('l1_sg4n', float('nan')) for r in ctrl_runs])

# Exp B records have sg4ns: list of sg4n per stage (sg4ns[0]=L1, sg4ns[-1]=Lk)
b_runs = [r for r in data if r.get('exp') == 'B']

gdata_B = {}   # {sigma: [(k, G_k, ratio), ...]}
for sigma in SIGMA_SWEEP:
    pts = []
    for k in K_STAGES:
        runs = [r for r in b_runs if r.get('sigma')==sigma and r.get('k_stages')==k]
        if not runs: continue
        l1_vals = [r['sg4ns'][0] for r in runs if r.get('sg4ns')]
        lk_vals = [r['sg4ns'][-1] for r in runs if r.get('sg4ns')]
        l1 = mn(l1_vals); lk = mn(lk_vals)
        G_k = lk / ctrl_sg4n if ctrl_sg4n > 1e-6 else float('nan')
        ratio = lk / l1 if l1 > 1e-6 else float('nan')
        pts.append((k, G_k, ratio))
    gdata_B[sigma] = pts

# ── Figure ───────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
GRAY = '#888888'; ORANGE = '#ff7f0e'; GREEN = '#2ca02c'; RED = '#d62728'

# ─── Panel A: G vs zw/r_wave ─────────────────────────────────────────────
ax = axes[0]
ax.axhline(1.0, color=GRAY, lw=0.8, ls='--', zorder=0)
ax.axhline(1.5, color=ORANGE, lw=0.8, ls=':', zorder=0, label=r'$G=1.5$ (threshold)')
ax.axvline(5.0, color=GREEN, lw=1.0, ls='-.', zorder=0, alpha=0.7,
           label=r'$z_w/r_w=5$ (obs. threshold)')

for rw in R_WAVE_SWEEP:
    pts = gdata_A[rw]
    xs = [p[0] for p in pts]; Gs=[p[1] for p in pts]; ses=[p[2] for p in pts]
    ax.errorbar(xs, Gs, yerr=ses, fmt='o-', color=COLORS_RW[rw], ms=7, capsize=3,
                lw=1.5, label=LABELS_RW[rw], zorder=2)

ax.set_xlabel(r'Zone width / wave radius $z_w / r_w$', fontsize=11)
ax.set_ylabel(r'Relay gain $G$', fontsize=10)
ax.set_title(r'\textbf{(a)}\ $G$ vs $z_w/r_w$ by wave radius', fontsize=11)
ax.legend(fontsize=8, loc='upper left')
ax.set_ylim(bottom=0)
ax.set_xticks([2, 4, 5, 6, 8, 10])

# ─── Panel B: N_crit vs r_wave ───────────────────────────────────────────
ax = axes[1]
rw_fine = np.linspace(0.8, 4.5, 200)
# Prediction: N_crit = HALF/(4*r_wave)
nc_pred4 = HALF / (4 * rw_fine)
ax.plot(rw_fine, nc_pred4, '--', color=RED, lw=2, alpha=0.8,
        label=r'Theory: $N_\mathrm{crit}=L/(4r_w)$')
# Empirical fit: N_crit = HALF/(5*r_wave)
nc_pred5 = HALF / (5 * rw_fine)
ax.plot(rw_fine, nc_pred5, '-', color=GREEN, lw=2, alpha=0.8,
        label=r'Fit: $N_\mathrm{crit}=L/(5r_w)$')

# Observed
Rws_obs = [rw for rw in R_WAVE_SWEEP if not math.isnan(ncrit_obs[rw])]
Nc_obs  = [ncrit_obs[rw] for rw in Rws_obs]
ax.scatter(Rws_obs, Nc_obs, s=80, zorder=5, color='k', marker='D',
           label=r'Observed $N_\mathrm{crit}$')
for rw, nc in zip(Rws_obs, Nc_obs):
    ax.annotate(rf'$N={nc:.0f}$', xy=(rw, nc), xytext=(rw+0.1, nc+0.3), fontsize=9)

# r_wave=4 annotation (all G<1)
ax.annotate(r'$r_w=4$: all $G<1$', xy=(4, 0), xytext=(3.3, 1.5),
            fontsize=9, color='gray',
            arrowprops=dict(arrowstyle='->', color='gray', lw=0.8))

ax.set_xlabel(r'Wave radius $r_w$ (sites)', fontsize=11)
ax.set_ylabel(r'$N_\mathrm{crit}$', fontsize=10)
ax.set_title(r'\textbf{(b)}\ $N_\mathrm{crit}$ vs wave radius', fontsize=11)
ax.legend(fontsize=8, loc='upper right')
ax.set_ylim(0, 12)
ax.set_xlim(0.5, 4.8)
ax.set_xticks([1, 2, 3, 4])

# ─── Panel C: G_k vs k (chain) ───────────────────────────────────────────
ax = axes[2]
ax.axhline(1.0, color=GRAY, lw=0.8, ls='--', zorder=0)

# G_baseline from B_ctrl: l1_sg4n / ctrl_sg4n
G_base = l1_sg4n_base / ctrl_sg4n if ctrl_sg4n > 1e-6 else float('nan')
ax.axhline(G_base, color='gray', lw=0.7, ls=':', alpha=0.6, zorder=0,
           label=rf'$G_\mathrm{{base}}={G_base:.2f}$ (L1 / ctrl)')

for sigma in SIGMA_SWEEP:
    pts = gdata_B[sigma]
    ks = [p[0] for p in pts]; Gks = [p[1] for p in pts]
    ax.plot(ks, Gks, 'o-', color=COLORS_SIG[sigma], ms=7, lw=1.5,
            label=LABELS_SIG[sigma], zorder=2)

ax.set_xlabel(r'Chain stage $k$', fontsize=11)
ax.set_ylabel(r'Relay gain $G_k = \mathrm{sg4n}(L_k) / \mathrm{sg4n}(\mathrm{ctrl})$', fontsize=10)
ax.set_title(r'\textbf{(c)}\ $G_k$ vs chain stage', fontsize=11)
ax.legend(fontsize=8, loc='upper right')
ax.set_xticks([1, 2, 3, 4])
ax.set_ylim(bottom=0)

plt.tight_layout()
fig.savefig('paper37_figure1.png', dpi=150, bbox_inches='tight')
fig.savefig('paper37_figure1.pdf', bbox_inches='tight')
print("Saved paper37_figure1.png / .pdf")

# ── Print summary ──────────────────────────────────────────────────────────
print("\n=== Panel A: G(zw/rw) per r_wave ===")
for rw in R_WAVE_SWEEP:
    print(f"\n  r_wave={rw}  N_crit_pred_4={HALF/(4*rw):.1f}  N_crit_pred_5={HALF/(5*rw):.1f}  N_crit_obs={ncrit_obs[rw]}")
    print(f"  {'N':>4} {'zw':>4} {'zw/rw':>7} {'G':>8} {'SE':>6}")
    for zwr, G, se, N in gdata_A[rw]:
        zw = HALF // N
        print(f"  {N:4d} {zw:4d} {zwr:7.2f} {G:8.3f} {se:6.3f}")

print("\n=== Panel B: N_crit scaling ===")
print(f"  {'r_wave':>8} {'N_crit/pred4':>14} {'N_crit/pred5':>14} {'obs':>6}")
for rw in [1,2,3]:
    nc = ncrit_obs[rw]
    if not math.isnan(nc):
        print(f"  {rw:8d} {nc/(HALF/(4*rw)):14.2f} {nc/(HALF/(5*rw)):14.2f} {nc:6.0f}")

print("\n=== Panel C: chain G_k ===")
print(f"  ctrl_sg4n={ctrl_sg4n:.4f}")
for sigma in SIGMA_SWEEP:
    print(f"\n  sigma={sigma}")
    print(f"  {'k':>4} {'G_k':>8} {'ratio_Lk/L1':>12}")
    for k, Gk, ratio in gdata_B[sigma]:
        print(f"  {k:4d} {Gk:8.3f} {ratio:12.3f}")
