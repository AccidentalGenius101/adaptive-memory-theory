"""Paper 74 - Figure generation.
Six-panel figure:
  (a) tau(P) full range log-log: Phases A+B combined, tau_VCSM=200 reference
  (b) C(t) for multiple P from Phase A (bridge region, semi-log)
  (c) C(t) for multiple P from Phase B (deep P, semi-log)
  (d) tau_L vs L (Phase C FSS, log-log) with z fit
  (e) U4 and |M| vs P across all phases
  (f) z summary: z_all, z_B, z_C vs Model A
"""
import json, math, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

sys.stdout = __import__('io').TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

ANALYSIS_FILE = Path(__file__).parent / 'results' / 'paper74_analysis.json'
OUT_PDF       = Path(__file__).parent / 'paper74_figure1.pdf'

with open(ANALYSIS_FILE) as f:
    data = json.load(f)

tau_A   = data['tau_A']
tau_B   = data['tau_B']
tau_C   = data['tau_C']
pc_A    = data['pc_A']
pc_B    = data['pc_B']
pc_C    = data['pc_C']
L_C     = data['L_C']
acf_A   = data['acf_A']
acf_B   = data['acf_B']
acf_C   = data['acf_C']
tau_VCS = data.get('tau_VCSM', 200.0)
nu      = data.get('nu', 0.98)

def safe_nan(v):
    if v is None: return float('nan')
    try: return float(v)
    except: return float('nan')

z_all = safe_nan(data.get('z_all'))
z_B   = safe_nan(data.get('z_B'))
z_C   = safe_nan(data.get('z_C'))
r2_all = safe_nan(data.get('r2_all'))
r2_B   = safe_nan(data.get('r2_B'))
r2_C   = safe_nan(data.get('r2_C'))

# Colors
C_A = ['#1b7837','#5aae61','#a6dba0','#d9f0d3','#762a83','#9970ab']
C_B = ['#b2182b','#d6604d','#f4a582']
C_C = {'40':'#a6cee3','60':'#1f78b4','80':'#33a02c','100':'#fb9a99','120':'#e31a1c'}

fig, axes = plt.subplots(2, 3, figsize=(14, 8))
(ax_a, ax_b, ax_c), (ax_d, ax_e, ax_f) = axes

# ── Panel A: tau(P) full range ─────────────────────────────────────────────────
ax_a.set_title('(a) tau(P): full P range (Phases A+B)', fontsize=10)
ax_a.set_xlabel('P', fontsize=9)
ax_a.set_ylabel('tau_corr (steps)', fontsize=9)
ax_a.set_xscale('log'); ax_a.set_yscale('log')

# Collect all tau data
all_P_plot = []
all_T_plot = []

for pc in pc_A:
    t = safe_nan(tau_A.get(str(pc)))
    if not math.isnan(t) and t > 0:
        all_P_plot.append(pc)
        all_T_plot.append(t)
for pc in pc_B:
    t = safe_nan(tau_B.get(str(pc)))
    if not math.isnan(t) and t > 0:
        all_P_plot.append(pc)
        all_T_plot.append(t)

if all_P_plot:
    # Sort by P
    pairs = sorted(zip(all_P_plot, all_T_plot))
    Ps = [p for p, t in pairs]
    Ts = [t for p, t in pairs]
    ax_a.plot(Ps, Ts, 'ko-', ms=7, lw=1.5, label='data')

    # tau_VCSM reference
    ax_a.axhline(tau_VCS, color='grey', lw=1.2, ls='--', label=f'tau_VCSM={tau_VCS:.0f}')

    # z fit
    if not math.isnan(z_all):
        P_fit = np.logspace(math.log10(min(Ps)*0.5), math.log10(max(Ps)*2), 30)
        znu = z_all * nu
        amps = [t * p**znu for p, t in zip(Ps, Ts)]
        amp = float(np.median(amps))
        ax_a.plot(P_fit, amp * P_fit**(-znu), 'r--', lw=1.5,
                  label=f'z={z_all:.2f} (all, R2={r2_all:.2f})')

    # Model A reference
    if Ps:
        amp2 = float(np.median([t * p**(2*nu) for p, t in zip(Ps, Ts)]))
        ax_a.plot(np.logspace(math.log10(min(Ps)*0.5), math.log10(max(Ps)*2), 30),
                  amp2 * np.logspace(math.log10(min(Ps)*0.5),
                                      math.log10(max(Ps)*2), 30)**(-2*nu),
                  'b:', lw=1.2, label='z=2 (Model A)')
ax_a.legend(fontsize=7)

# ── Panel B: C(t) Phase A (bridge P values) ────────────────────────────────────
ax_b.set_title('(b) C(t) for bridge P values (Phase A)', fontsize=10)
ax_b.set_xlabel('t (steps)', fontsize=9)
ax_b.set_ylabel('C(t)', fontsize=9)
ax_b.set_yscale('log')

for i, pc in enumerate(pc_A):
    cd = acf_A.get(str(pc), {})
    lg = cd.get('lags', []); ac = cd.get('acf', [])
    if not lg: continue
    t_arr = np.array(lg[1:], dtype=float)
    C_arr = np.array(ac[1:], dtype=float)
    mask  = (C_arr > 0) & (t_arr > 10)
    if mask.sum() < 2: continue
    ax_b.plot(t_arr[mask], C_arr[mask], '-', lw=1.0, alpha=0.9,
              color=C_A[i % len(C_A)], label=f'P={pc:.4f}')
ax_b.axhline(1/math.e, color='grey', lw=0.7, ls=':', alpha=0.6, label='1/e')
ax_b.legend(fontsize=6, ncol=2)

# ── Panel C: C(t) Phase B (deep P values) ─────────────────────────────────────
ax_c.set_title('(c) C(t) for deep P values (Phase B)', fontsize=10)
ax_c.set_xlabel('t (steps)', fontsize=9)
ax_c.set_ylabel('C(t)', fontsize=9)
ax_c.set_yscale('log')

for i, pc in enumerate(pc_B):
    cd = acf_B.get(str(pc), {})
    lg = cd.get('lags', []); ac = cd.get('acf', [])
    if not lg: continue
    t_arr = np.array(lg[1:], dtype=float)
    C_arr = np.array(ac[1:], dtype=float)
    mask  = (C_arr > 0) & (t_arr > 10)
    if mask.sum() < 2: continue
    ax_c.plot(t_arr[mask], C_arr[mask], '-', lw=1.2, alpha=0.9,
              color=C_B[i % len(C_B)], label=f'P={pc:.4f}')
ax_c.axhline(1/math.e, color='grey', lw=0.7, ls=':', alpha=0.6, label='1/e')
ax_c.legend(fontsize=8)

# ── Panel D: tau_L vs L (Phase C FSS) ─────────────────────────────────────────
ax_d.set_title(f'(d) tau_L vs L at P={pc_C} (Phase C FSS)', fontsize=10)
ax_d.set_xlabel('L', fontsize=9)
ax_d.set_ylabel('tau_L (steps)', fontsize=9)
ax_d.set_xscale('log'); ax_d.set_yscale('log')

tau_pairs = [(L, safe_nan(tau_C.get(str(L)))) for L in L_C
             if not math.isnan(safe_nan(tau_C.get(str(L))))]
if tau_pairs:
    Ls = np.array([p[0] for p in tau_pairs])
    ts = np.array([p[1] for p in tau_pairs])
    ax_d.plot(Ls, ts, 'ko-', ms=8, lw=1.5, label='data')
    if not math.isnan(z_C):
        L_fit = np.logspace(math.log10(min(Ls)*0.9), math.log10(max(Ls)*1.1), 30)
        amps  = [t / l**z_C for l, t in zip(Ls, ts)]
        amp   = float(np.median(amps))
        ax_d.plot(L_fit, amp * L_fit**z_C, 'r--', lw=1.4,
                  label=f'z={z_C:.2f} (R2={r2_C:.2f})')
    # Model A
    amp2 = float(np.median([t / l**2 for l, t in zip(Ls, ts)]))
    ax_d.plot(np.logspace(math.log10(min(Ls)*0.9), math.log10(max(Ls)*1.1), 30),
              amp2 * np.logspace(math.log10(min(Ls)*0.9),
                                  math.log10(max(Ls)*1.1), 30)**2,
              'b:', lw=1.2, label='z=2 (Model A)')
    ax_d.axhline(tau_VCS, color='grey', lw=0.8, ls='--', alpha=0.6,
                 label=f'tau_VCSM={tau_VCS:.0f}')
ax_d.legend(fontsize=7)

# ── Panel E: U4 and |M| vs P across all phases ────────────────────────────────
ax_e.set_title('(e) Order parameter vs P', fontsize=10)
ax_e.set_xlabel('P', fontsize=9)
ax_e.set_xscale('log')

# We need to read raw U4 and absM from the results file
try:
    results_path = Path(__file__).parent / 'results' / 'paper74_results.json'
    with open(results_path) as rf:
        raw = json.load(rf)

    def get_obs(phase, key_fmt, pc_list):
        u4s = []; Ms = []
        for pc in pc_list:
            key = key_fmt.format(pc)
            seeds = raw.get(phase, {}).get(key, {})
            u4v = [v['U4']   for v in seeds.values()
                   if not math.isnan(v.get('U4', float('nan')))]
            mv  = [v['absM'] for v in seeds.values()
                   if not math.isnan(v.get('absM', float('nan')))]
            u4s.append(float(np.mean(u4v)) if u4v else float('nan'))
            Ms.append(float(np.mean(mv))   if mv   else float('nan'))
        return u4s, Ms

    u4_A, M_A = get_obs('A', 'pc{:.4f}', pc_A)
    u4_B, M_B = get_obs('B', 'pc{:.4f}', pc_B)

    ax_e2 = ax_e.twinx()
    ax_e.set_ylabel('U4 (Binder)', fontsize=9, color='#1f78b4')
    ax_e2.set_ylabel('|M|', fontsize=9, color='#e31a1c')

    all_pc = pc_A + pc_B
    all_u4 = u4_A + u4_B
    all_M  = M_A  + M_B
    pairs_sorted = sorted(zip(all_pc, all_u4, all_M))
    Ps_e = [p for p, u, m in pairs_sorted]
    U4s  = [u for p, u, m in pairs_sorted]
    Ms_e = [m for p, u, m in pairs_sorted]

    ax_e.plot(Ps_e, U4s, 'o-', color='#1f78b4', ms=6, lw=1.2, label='U4')
    ax_e2.plot(Ps_e, Ms_e, 's-', color='#e31a1c', ms=6, lw=1.2, label='|M|')
    ax_e.axhline(0, color='grey', lw=0.5, ls=':')
    ax_e.legend(fontsize=8, loc='upper left')
    ax_e2.legend(fontsize=8, loc='upper right')
except Exception as ex:
    ax_e.text(0.5, 0.5, f'No data\n{ex}', ha='center', va='center',
              transform=ax_e.transAxes, fontsize=8)

# ── Panel F: z summary ────────────────────────────────────────────────────────
ax_f.set_title('(f) Dynamic exponent z summary', fontsize=10)
ax_f.set_ylabel('z', fontsize=9)
ax_f.set_xlim(-0.5, 3.5)

labels = ['All phases\n(tau~P^{-znu})', 'Phase B\n(deep P)', 'Phase C\n(FSS tau_L)']
z_vals = [z_all, z_B, z_C]
r2s    = [r2_all, r2_B, r2_C]
colors = ['#377eb8', '#e41a1c', '#4daf4a']
xs     = [0, 1, 2]
for x, label, z, r2, c in zip(xs, labels, z_vals, r2s, colors):
    if not math.isnan(z):
        ax_f.bar(x, z, color=c, alpha=0.75, width=0.6,
                 label=f'{label.split(chr(10))[0]}: z={z:.2f} (R2={r2:.2f})')
        ax_f.text(x, z + 0.05, f'{z:.2f}', ha='center', fontsize=9)
ax_f.axhline(2.0, color='grey', lw=1.2, ls='--', label='z=2 (Model A)')
ax_f.axhline(1.5, color='orange', lw=0.9, ls=':', label='z=3/2 (KPZ)')
ax_f.set_xticks(xs); ax_f.set_xticklabels(labels, fontsize=7)
ax_f.legend(fontsize=7)
valid_z = [v for v in z_vals if not math.isnan(v)]
ax_f.set_ylim(0, max(valid_z + [2.5]) * 1.4 if valid_z else 3.0)

fig.suptitle('Paper 74: Dynamic Exponent z — Approaching P_c=0+  '
             '(crossing tau_VCSM=200 floor)',
             fontsize=11, fontweight='bold')
plt.tight_layout()
plt.savefig(OUT_PDF, bbox_inches='tight')
print(f'Saved {OUT_PDF}  ({OUT_PDF.stat().st_size} bytes)')
