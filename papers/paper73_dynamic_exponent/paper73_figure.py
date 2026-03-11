"""Paper 73 - Figure generation.
Six-panel figure:
  (a) C(t) vs t log-log for multiple L at criticality (Phase A) -- power law / exp
  (b) tau_L vs L log-log with z fit (Phase A)
  (c) C(t) vs t for multiple P values (Phase B) -- exponential decay
  (d) tau_corr vs P log-log with z fit (Phase B)
  (e) C(t) at L=120 (Phase C) -- both exp and pow law fits shown
  (f) Summary: z from all three phases + Model A reference
"""
import json, math, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

sys.stdout = __import__('io').TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

ANALYSIS_FILE = Path(__file__).parent / 'results' / 'paper73_analysis.json'
OUT_PDF       = Path(__file__).parent / 'paper73_figure1.pdf'

with open(ANALYSIS_FILE) as f:
    data = json.load(f)

tau_A   = data['tau_A']
tau_B   = data['tau_B']
z_A     = data.get('z_A')  or float('nan')
z_B     = data.get('z_B')  or float('nan')
z_C     = data.get('z_C')  or float('nan')
lam_C   = data.get('lambda_C') or float('nan')
L_A     = data['L_a']
PC_B    = data['pc_b']
acf_A   = data['acf_A']
acf_B   = data['acf_B']
acf_C   = data['acf_C']

def safe_nan(v):
    if v is None: return float('nan')
    try: return float(v)
    except: return float('nan')

z_A = safe_nan(z_A); z_B = safe_nan(z_B); z_C = safe_nan(z_C)
lam_C = safe_nan(lam_C)

L_colours  = {40:'#a6cee3', 60:'#1f78b4', 80:'#33a02c', 100:'#fb9a99', 120:'#e31a1c'}
P_colours  = ['#a6cee3','#1f78b4','#33a02c','#fb9a99','#e31a1c','#6a3d9a']

fig, axes = plt.subplots(2, 3, figsize=(13, 8))
(ax_a, ax_b, ax_c), (ax_d, ax_e, ax_f) = axes

# ── Panel A: C(t) for multiple L ─────────────────────────────────────────────
ax_a.set_title('(a) C(t) at criticality, multiple L', fontsize=10)
ax_a.set_xlabel('t (steps)', fontsize=9)
ax_a.set_ylabel('C(t)', fontsize=9)
ax_a.set_xscale('log'); ax_a.set_yscale('log')

for L in L_A:
    cd = acf_A.get(str(L), {})
    lg = cd.get('lags', []); ac = cd.get('acf', [])
    if not lg: continue
    t_arr = np.array(lg[1:], dtype=float)  # skip lag=0
    C_arr = np.array(ac[1:], dtype=float)
    mask  = (C_arr > 0) & (t_arr > 10)
    if mask.sum() < 2: continue
    ax_a.plot(t_arr[mask], C_arr[mask], '-', lw=1.2, alpha=0.8,
              color=L_colours.get(L, 'k'), label=f'L={L}')
ax_a.legend(fontsize=7)

# ── Panel B: tau_L vs L ────────────────────────────────────────────────────────
ax_b.set_title('(b) tau_L vs L  (z from slope)', fontsize=10)
ax_b.set_xlabel('L', fontsize=9)
ax_b.set_ylabel('tau_L', fontsize=9)
ax_b.set_xscale('log'); ax_b.set_yscale('log')

tau_pairs = [(L, float(tau_A[str(L)])) for L in L_A
             if str(L) in tau_A and tau_A[str(L)] is not None
             and not math.isnan(float(tau_A[str(L)]))]
if tau_pairs:
    Ls = np.array([p[0] for p in tau_pairs])
    ts = np.array([p[1] for p in tau_pairs])
    ax_b.plot(Ls, ts, 'ko-', ms=8, lw=1.5, label='data')
    if not math.isnan(z_A):
        L_fit = np.logspace(math.log10(min(Ls)*0.9), math.log10(max(Ls)*1.1), 30)
        amps  = [t / l**z_A for l, t in zip(Ls, ts)]
        amp   = float(np.median(amps))
        ax_b.plot(L_fit, amp * L_fit**z_A, 'r--', lw=1.4,
                  label=f'z={z_A:.2f}')
    ax_b.plot([], [], 'b:', lw=1.2, label='z=2 (Model A)')
    if tau_pairs:
        amp2 = float(np.median([t / l**2 for l, t in zip(Ls, ts)]))
        L_fit2 = np.logspace(math.log10(min(Ls)*0.9), math.log10(max(Ls)*1.1), 30)
        ax_b.plot(L_fit2, amp2 * L_fit2**2, 'b:', lw=1.2)
ax_b.legend(fontsize=8)

# ── Panel C: C(t) for multiple P ──────────────────────────────────────────────
ax_c.set_title('(c) C(t) for multiple P (Phase B)', fontsize=10)
ax_c.set_xlabel('t (steps)', fontsize=9)
ax_c.set_ylabel('C(t)', fontsize=9)
ax_c.set_yscale('log')

for i, pc in enumerate(PC_B):
    cd = acf_B.get(str(pc), {})
    lg = cd.get('lags', []); ac = cd.get('acf', [])
    if not lg: continue
    t_arr = np.array(lg[1:], dtype=float)
    C_arr = np.array(ac[1:], dtype=float)
    mask  = (C_arr > 0) & (t_arr > 10)
    if mask.sum() < 2: continue
    xi = 5 * math.sqrt(pc)
    ax_c.plot(t_arr[mask], C_arr[mask], '-', lw=1.0, alpha=0.9,
              color=P_colours[i % len(P_colours)],
              label=f'P={pc:.3f}')
ax_c.legend(fontsize=6, ncol=2)

# ── Panel D: tau_corr vs P ─────────────────────────────────────────────────────
ax_d.set_title('(d) tau_corr vs P  (z from slope)', fontsize=10)
ax_d.set_xlabel('P', fontsize=9)
ax_d.set_ylabel('tau_corr', fontsize=9)
ax_d.set_xscale('log'); ax_d.set_yscale('log')

tau_B_pairs = [(float(pc), float(tau_B[str(pc)])) for pc in PC_B
               if str(pc) in tau_B and tau_B[str(pc)] is not None
               and not math.isnan(float(tau_B[str(pc)]))]
if tau_B_pairs:
    Ps = np.array([p[0] for p in tau_B_pairs])
    ts = np.array([p[1] for p in tau_B_pairs])
    ax_d.plot(Ps, ts, 'ko-', ms=7, lw=1.5, label='data')
    if not math.isnan(z_B):
        znu = z_B * 0.98
        P_fit = np.logspace(math.log10(min(Ps)*0.8), math.log10(max(Ps)*1.2), 30)
        amps  = [t * p**znu for p, t in zip(Ps, ts)]
        amp   = float(np.median(amps))
        ax_d.plot(P_fit, amp * P_fit**(-znu), 'r--', lw=1.4,
                  label=f'z={z_B:.2f} (z*nu={znu:.2f})')
ax_d.legend(fontsize=8)

# ── Panel E: C(t) Phase C with both fits ──────────────────────────────────────
ax_e.set_title(f'(e) C(t) at criticality (L={data["L_c"]}, Phase C)', fontsize=10)
ax_e.set_xlabel('t (steps)', fontsize=9)
ax_e.set_ylabel('C(t)', fontsize=9)
ax_e.set_xscale('log'); ax_e.set_yscale('log')

lg_C = acf_C.get('lags', []); ac_C = acf_C.get('acf', [])
if lg_C and ac_C:
    t_arr = np.array(lg_C[1:], dtype=float)
    C_arr = np.array(ac_C[1:], dtype=float)
    mask  = (C_arr > 0) & (t_arr > 10)
    if mask.sum() >= 2:
        ax_e.plot(t_arr[mask], C_arr[mask], 'ko', ms=3, lw=0,
                  alpha=0.7, label='data')
        t_max = t_arr[mask].max()
        t_fit = np.logspace(math.log10(50), math.log10(t_max / 2), 50)
        # Exponential fit
        tau_C_exp = safe_nan(data.get('tau_C_exp'))
        if not math.isnan(tau_C_exp):
            amp_e = float(np.median([C * np.exp(t / tau_C_exp)
                                     for t, C in zip(t_arr[mask], C_arr[mask])
                                     if 50 <= t <= t_max / 2]))
            ax_e.plot(t_fit, amp_e * np.exp(-t_fit / tau_C_exp), 'r--',
                      lw=1.4, label=f'exp: tau={tau_C_exp:.0f}')
        # Power law fit
        if not math.isnan(lam_C):
            amp_p = float(np.median([C * t**lam_C
                                     for t, C in zip(t_arr[mask], C_arr[mask])
                                     if 100 <= t <= t_max / 2]))
            ax_e.plot(t_fit, amp_p * t_fit**(-lam_C), 'b-.',
                      lw=1.4, label=f'pow: lambda={lam_C:.2f}')
        ax_e.legend(fontsize=8)

# ── Panel F: Summary z values ──────────────────────────────────────────────────
ax_f.set_title('(f) Dynamic exponent z summary', fontsize=10)
ax_f.set_ylabel('z', fontsize=9)
ax_f.set_xlim(-0.5, 3.5)

labels = ['Phase A\n(FSS tau_L)', 'Phase B\n(tau~P^{-znu})', 'Phase C\n(lambda)']
z_vals = [z_A, z_B, z_C]
colors  = ['#377eb8', '#e41a1c', '#4daf4a']
xs = [0, 1, 2]
for x, label, z, c in zip(xs, labels, z_vals, colors):
    if not math.isnan(z):
        ax_f.bar(x, z, color=c, alpha=0.7, width=0.6, label=f'{label}: z={z:.2f}')
        ax_f.text(x, z + 0.05, f'{z:.2f}', ha='center', fontsize=9)
ax_f.axhline(2.0, color='grey', lw=1.2, ls='--', label='z=2 (Model A)')
ax_f.axhline(1.5, color='orange', lw=0.9, ls=':', label='z=3/2 (KPZ)')
ax_f.set_xticks(xs); ax_f.set_xticklabels(labels, fontsize=7)
ax_f.legend(fontsize=7)
ax_f.set_ylim(0, max([v for v in z_vals if not math.isnan(v)] + [2.5]) * 1.3)

fig.suptitle('Paper 73: Dynamic Exponent z -- Temporal Autocorrelator C(t) of Zone-Mean M(t)',
             fontsize=11, fontweight='bold')
plt.tight_layout()
plt.savefig(OUT_PDF, bbox_inches='tight')
print(f'Saved {OUT_PDF}  ({OUT_PDF.stat().st_size} bytes)')
