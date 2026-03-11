"""
paper36_figure1.py -- Figure 1 for Paper 36: xi_eff = wave radius, not diffusion length

Two panels:
  (a) G vs N_zones for each DIFFUSE -- showing that the N_crit transition
      is approximately invariant to DIFFUSE (curves overlay)
  (b) N_crit (predicted from sqrt-DIFFUSE law) vs observed (flat at ~5)
      -- shows the failure of the diffusion hypothesis and confirmation
         that ξ_eff = r_wave = 2 sites
"""
import json, math, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import defaultdict

RES = "results/paper36_results.json"
with open(RES) as f: data = json.load(f)

by = defaultdict(list)
for r in data: by[(r['diffuse'],r['n_zones'],r['coupling'],r['exp_type'])].append(r)

def mn(lst): v=[x for x in lst if not math.isnan(x)]; return float(np.mean(v)) if v else float('nan')

HALF=40; r_wave=2.0
DIFFUSE_TEST=[0.005, 0.02, 0.08]
DIFFUSE_STD=0.02; xi_std=2.5
N_ALL=[2,4,5,8,10]
COLORS={"0.005":"#e31a1c","0.02":"#1f77b4","0.08":"#2ca02c"}
LABELS={"0.005":r"$D=0.005$","0.02":r"$D=0.020$ (std)","0.08":r"$D=0.080$"}

# ── Compute G(N) for each DIFFUSE ─────────────────────────────────────────────
gdata = {}   # {D: [(N, G, G_se), ...]}
ncrit_obs = {}
for D in DIFFUSE_TEST:
    pts = []
    for N in N_ALL:
        if HALF % N != 0: continue
        geo  = by[(D,N,'geo','B')]; ctrl=by[(D,N,'ctrl','B')]
        if not geo or not ctrl: continue
        sg_list = [r.get('l2_sg4n', float('nan')) for r in geo]
        sc_list = [r.get('l2_sg4n', float('nan')) for r in ctrl]
        sg = mn(sg_list); sc = mn(sc_list)
        G  = sg/sc if sc>1e-6 else float('nan')
        # seed-level G for SE
        seed_Gs = [g/c for g,c in zip(sg_list,sc_list)
                   if not math.isnan(g) and not math.isnan(c) and c>1e-6]
        se = float(np.std(seed_Gs,ddof=1)/math.sqrt(len(seed_Gs))) if len(seed_Gs)>1 else 0
        pts.append((N, G, se))
    gdata[D] = pts
    nc_obs = max((N for N,G,_ in pts if not math.isnan(G) and G>1.5), default=float('nan'))
    ncrit_obs[D] = nc_obs

# ── Panel B: N_crit predicted vs observed ─────────────────────────────────────
Ds_all = np.array(DIFFUSE_TEST)
# Diffusion prediction: xi_diffusion = xi_std * sqrt(D/D_std)  ->  N_crit = HALF/(4*xi_diffusion)
xi_diffusion = xi_std * np.sqrt(Ds_all / DIFFUSE_STD)
ncrit_diffusion = HALF / (4 * xi_diffusion)
# Wave-radius prediction: xi_eff = r_wave = 2 -> N_crit = HALF/(4*r_wave) = 5
ncrit_wave = np.full_like(Ds_all, HALF / (4 * r_wave))

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
BLUE="#1f77b4"; GRAY="#888888"; ORANGE="#ff7f0e"; RED="#d62728"; GREEN="#2ca02c"

# ─── Panel A: G vs N ──────────────────────────────────────────────────────────
ax = axes[0]
ax.axhline(1.0, color=GRAY, lw=0.8, ls='--', zorder=0)
ax.axhline(1.5, color=ORANGE, lw=0.8, ls=':', zorder=0, label='G=1.5 (threshold)')
# N_crit from wave-radius prediction
nc_wave = HALF / (4 * r_wave)  # = 5
ax.axvline(nc_wave, color=GREEN, lw=1.3, ls='-.',
           label=rf'$N_\mathrm{{crit}}={nc_wave:.0f}$ ($z_w=4r_\mathrm{{wave}}$)', zorder=0)

for D in DIFFUSE_TEST:
    col = COLORS[str(D)]
    pts = gdata[D]
    Ns = [p[0] for p in pts]; Gs=[p[1] for p in pts]; ses=[p[2] for p in pts]
    ax.errorbar(Ns, Gs, yerr=ses, fmt='o-', color=col, ms=6, capsize=3,
                lw=1.5, label=LABELS[str(D)], zorder=2)

ax.set_xlabel(r'Number of zones $N_\mathrm{zones}$', fontsize=11)
ax.set_ylabel(r'Relay gain $G$', fontsize=10)
ax.set_title(r'\textbf{(a)}\ $G$ vs $N_\mathrm{zones}$ at each DIFFUSE', fontsize=11)
ax.legend(fontsize=8, loc='upper right')
ax.set_xticks(N_ALL)
ax.set_ylim(bottom=0)

# ─── Panel B: N_crit predicted vs observed ────────────────────────────────────
ax = axes[1]

# sqrt-DIFFUSE prediction
D_fine = np.logspace(np.log10(0.003), np.log10(0.12), 200)
xi_fine = xi_std * np.sqrt(D_fine / DIFFUSE_STD)
nc_fine = HALF / (4 * xi_fine)
ax.plot(D_fine, nc_fine, '-', color=RED, lw=2, alpha=0.8,
        label=r'Diffusion theory: $N_\mathrm{crit}=\frac{L}{4\xi_0}\sqrt{\frac{D_0}{D}}$')
ax.axhline(nc_wave, color=GREEN, lw=2, ls='-.', alpha=0.9,
           label=rf'Wave-radius theory: $N_\mathrm{{crit}}=L/(4r_\mathrm{{wave}})={nc_wave:.0f}$')

# Observed N_crit
Ds_obs = [D for D in DIFFUSE_TEST]
Nc_obs = [ncrit_obs[D] for D in DIFFUSE_TEST]
ax.scatter(Ds_obs, Nc_obs, s=80, zorder=5, color='k', marker='D',
           label='Observed $N_\\mathrm{crit}$ (G>1.5)')

# Annotate observed points
for D, nc in zip(Ds_obs, Nc_obs):
    if not math.isnan(nc):
        ax.annotate(rf'$N={nc:.0f}$', xy=(D, nc), xytext=(D*1.15, nc+0.4),
                    fontsize=9, ha='left')

ax.set_xscale('log')
ax.set_xlabel(r'DIFFUSE rate $D$', fontsize=11)
ax.set_ylabel(r'$N_\mathrm{crit}$ (largest $N$ with $G>1.5$)', fontsize=10)
ax.set_title(r'\textbf{(b)}\ Predicted vs observed $N_\mathrm{crit}$', fontsize=11)
ax.legend(fontsize=8, loc='upper right')
ax.set_ylim(0, 12)

plt.tight_layout()
fig.savefig('paper36_figure1.png', dpi=150, bbox_inches='tight')
fig.savefig('paper36_figure1.pdf', bbox_inches='tight')
print("Saved paper36_figure1.png / .pdf")

# ── Print summary ─────────────────────────────────────────────────────────────
print("\n=== Panel A: G(N) per DIFFUSE ===")
for D in DIFFUSE_TEST:
    xi_pred = xi_std * math.sqrt(D/DIFFUSE_STD)
    nc_pred_diff = HALF/(4*xi_pred)
    print(f"\n  DIFFUSE={D}: xi_pred(diffusion)={xi_pred:.2f}  N_crit_pred={nc_pred_diff:.1f}")
    print(f"  {'N':>4} {'zw':>5} {'G':>8} {'SE':>6}")
    for N,G,se in gdata[D]:
        zw=HALF//N
        print(f"  {N:4d} {zw:5d} {G:8.3f} {se:6.3f}")

print("\n=== Panel B: N_crit summary ===")
print(f"  Wave-radius theory: N_crit = HALF/(4*r_wave) = {HALF/(4*r_wave):.1f}  (r_wave={r_wave})")
print(f"  {'DIFFUSE':>10} {'xi_pred':>9} {'N_crit_diff':>12} {'N_crit_wave':>12} {'N_crit_obs':>12}")
for D in DIFFUSE_TEST:
    xi_pred = xi_std * math.sqrt(D/DIFFUSE_STD)
    nc_diff = HALF/(4*xi_pred); nc_wave_val = HALF/(4*r_wave)
    print(f"  {D:10.4g} {xi_pred:9.2f} {nc_diff:12.1f} {nc_wave_val:12.1f} {ncrit_obs[D]:12.1f}")
