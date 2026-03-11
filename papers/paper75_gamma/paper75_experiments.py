"""
Paper 75 - z Confirmation + Susceptibility Exponent gamma
GPU-accelerated (RTX 3060, torch.compile mode='default')

Phase A: L=160, P in {0.001,0.002,0.005}, 8 seeds, 200k steps
         Goal: U4>0 at L=160; tau(P) at large L; chi(P)
Phase B: L=80,  P in {0.00003,0.00005,0.0001,0.0002,0.0005}, 8 seeds, 300k steps
         Goal: monotone tau(P) -> clean z_B; fix P=0.0001 non-monotonicity
Phase C: P=0.0005, L in {40,60,80,100,120,160}, 8 seeds, 150k steps
         Goal: clean FSS tau_L~L^z at deeper P; z_C with R2>0.80
Phase D: chi=var(M) from all phases -> fit chi~P^{-gamma} -> extract gamma
"""

import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import numpy as np
import json, math
import torch
from pathlib import Path

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DTYPE  = torch.float32
print(f'Device: {DEVICE}')
if DEVICE.type == 'cuda':
    print(f'  GPU: {torch.cuda.get_device_name(0)}')
    print(f'  VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB')

# -- Physical constants --------------------------------------------------------
J=1.0; MID_DECAY=0.97; BETA_BASE=0.005; SS=8; WAVE_DUR=5
EXT_FIELD=1.5; T_FIXED=3.0; FIELD_DECAY=0.999; SS_FRAC=0.40; FA=0.30
WE_BASE=25; L_BASE=40; RW=5; NU=0.98; BETA_EXP=0.628

def _prob_A(L): return min(1.0, 1.0 / (WE_BASE * (L_BASE/L)**2))

# -- Phases --------------------------------------------------------------------
PC_A  = [0.001, 0.002, 0.005];  L_A  = 160; SEEDS_A = list(range(8)); NSTEPS_A = 200_000
PC_B  = [0.00003, 0.00005, 0.0001, 0.0002, 0.0005]
L_B   = 80;  SEEDS_B = list(range(8)); NSTEPS_B = 300_000
PC_C  = 0.0005; L_C = [40, 60, 80, 100, 120, 160]
SEEDS_C = list(range(8)); NSTEPS_C = 150_000

RESULTS_FILE  = Path(__file__).parent / 'results' / 'paper75_results.json'
ANALYSIS_FILE = Path(__file__).parent / 'results' / 'paper75_analysis.json'
CHUNK_RAND    = 5000


# == Compiled step function ====================================================

def build_step_fn(nb, cb0, cb1, col_g, row_g, z0f, z1f, n_z0, n_z1,
                  L, B, N, r_w):
    cb0_e = cb0.unsqueeze(0).expand(B, -1)
    cb1_e = cb1.unsqueeze(0).expand(B, -1)
    col_e = col_g.unsqueeze(0).expand(B, -1).contiguous()
    row_e = row_g.unsqueeze(0).expand(B, -1).contiguous()
    n_z0f = n_z0.to(DTYPE)
    n_z1f = n_z1.to(DTYPE)

    def _metro_sub(s, idx_sub, idx_sub_e, rnd):
        nb_idx = nb[idx_sub]
        ns     = s[:, nb_idx].sum(-1)
        dE     = 2.0 * s[:, idx_sub] * ns
        acc    = (dE <= 0) | (rnd < torch.exp(torch.clamp(-dE / T_FIXED, -20., 20.)))
        s_sub  = torch.where(acc, -s[:, idx_sub], s[:, idx_sub])
        return s.scatter(1, idx_sub_e, s_sub)

    @torch.compile(mode='default', dynamic=False)
    def step(s, base, mid, phi, streak, wave_z,
             rnd0, rnd1,
             fires, cx_z0, cx_z1, cy,
             rnd_w0, rnd_w1,
             cau_rnd, nse_rnd, P_ten):

        # Main Metropolis
        s = _metro_sub(s, cb0, cb0_e, rnd0)
        s = _metro_sub(s, cb1, cb1_e, rnd1)

        # Baseline + streak
        base   = base + BETA_BASE * (s - base)
        same   = (s > 0) == (base > 0)
        streak = torch.where(same, streak + 1, torch.zeros_like(streak))
        mid    = mid * MID_DECAY

        # Wave (always computed, masked by fires)
        cx  = torch.where(wave_z == 0, cx_z0, cx_z1)
        dx  = torch.minimum((col_e - cx.unsqueeze(1)).abs(),
                            L - (col_e - cx.unsqueeze(1)).abs())
        dy  = torch.minimum((row_e - cy.unsqueeze(1)).abs(),
                            L - (row_e - cy.unsqueeze(1)).abs())
        in_wave  = (dx + dy <= r_w)
        fires_e  = fires.unsqueeze(1).expand(-1, N)
        hit_f    = (in_wave & fires_e).to(DTYPE)

        he_sign  = torch.where(wave_z == 0, 1., -1.).to(DTYPE)
        h_ext    = hit_f * he_sign.unsqueeze(1) * EXT_FIELD

        s_w = s
        for d in range(WAVE_DUR):
            ns0w = s_w[:, nb[cb0]].sum(-1)
            h0   = h_ext[:, cb0]
            dE0w = 2.0*s_w[:, cb0]*ns0w - 2.0*h0*s_w[:, cb0]
            ac0w = (dE0w <= 0) | (rnd_w0[d*B:(d+1)*B] <
                                  torch.exp(torch.clamp(-dE0w/T_FIXED, -20., 20.)))
            s_w  = s_w.scatter(1, cb0_e, torch.where(ac0w, -s_w[:,cb0], s_w[:,cb0]))

            ns1w = s_w[:, nb[cb1]].sum(-1)
            h1   = h_ext[:, cb1]
            dE1w = 2.0*s_w[:,cb1]*ns1w - 2.0*h1*s_w[:,cb1]
            ac1w = (dE1w <= 0) | (rnd_w1[d*B:(d+1)*B] <
                                  torch.exp(torch.clamp(-dE1w/T_FIXED, -20., 20.)))
            s_w  = s_w.scatter(1, cb1_e, torch.where(ac1w, -s_w[:,cb1], s_w[:,cb1]))

        s = torch.where(fires_e, s_w, s)

        # mid update
        dev   = s - base
        wz_e  = (wave_z == 0).unsqueeze(1).expand(-1, N)
        z0_e  = z0f.bool().unsqueeze(0).expand(B, -1)
        sig   = torch.where(z0_e == wz_e, dev, -dev)
        cau   = cau_rnd < P_ten.unsqueeze(1)
        nse   = nse_rnd * (dev.std(dim=1, keepdim=True).clamp(min=0.01) + 0.5)
        mid   = mid + FA * torch.where(cau, sig, nse) * hit_f

        wave_z = torch.where(fires, 1 - wave_z, wave_z)

        # Gate + phi decay
        gate   = streak >= SS
        phi    = phi + FA * torch.where(gate, mid - phi, torch.zeros_like(phi))
        streak = torch.where(gate, torch.zeros_like(streak), streak)
        phi    = phi * FIELD_DECAY

        # Zone-mean M
        M = (phi * z0f).sum(1) / n_z0f - (phi * z1f).sum(1) / n_z1f

        return s, base, mid, phi, streak, wave_z, M

    return step


# == Main simulation ===========================================================

def run_batch_gpu(L, P_causal_list, seed_list, nsteps, r_w):
    B = len(seed_list); N = L*L; wp = _prob_A(L)

    idx   = torch.arange(N, device=DEVICE)
    row_g = idx // L; col_g = idx % L
    nb    = torch.stack([((row_g-1)%L)*L+col_g, ((row_g+1)%L)*L+col_g,
                         row_g*L+(col_g-1)%L,   row_g*L+(col_g+1)%L], dim=1)
    par   = (row_g + col_g) % 2
    cb0   = idx[par==0].contiguous()
    cb1   = idx[par==1].contiguous()
    z0f   = (col_g < L//2).to(DTYPE)
    z1f   = (col_g >= L//2).to(DTYPE)
    n_z0  = z0f.sum().long(); n_z1 = z1f.sum().long()

    step_fn = build_step_fn(nb, cb0, cb1, col_g, row_g, z0f, z1f,
                            n_z0, n_z1, L, B, N, r_w)

    P_ten = torch.tensor(P_causal_list, device=DEVICE, dtype=DTYPE)

    s = torch.zeros(B, N, device=DEVICE, dtype=DTYPE)
    for b, seed in enumerate(seed_list):
        rng_np = np.random.RandomState(seed)
        s[b]   = torch.from_numpy(
            rng_np.choice([-1.,1.], N).astype(np.float32)).to(DEVICE)
    base   = s * 0.1
    mid    = torch.zeros(B, N, device=DEVICE, dtype=DTYPE)
    phi    = torch.zeros(B, N, device=DEVICE, dtype=DTYPE)
    streak = torch.zeros(B, N, device=DEVICE, dtype=torch.int32)
    wave_z = torch.zeros(B,   device=DEVICE, dtype=torch.int32)

    ss0    = int(nsteps * SS_FRAC)
    n_data = nsteps - ss0
    M_buf  = torch.zeros(n_data, B, device=DEVICE, dtype=DTYPE)
    m_idx  = 0

    rnd0   = torch.empty(B, len(cb0), device=DEVICE, dtype=DTYPE)
    rnd1   = torch.empty(B, len(cb1), device=DEVICE, dtype=DTYPE)
    rnd_w0 = torch.empty(WAVE_DUR*B, len(cb0), device=DEVICE, dtype=DTYPE)
    rnd_w1 = torch.empty(WAVE_DUR*B, len(cb1), device=DEVICE, dtype=DTYPE)
    cau_r  = torch.empty(B, N, device=DEVICE, dtype=DTYPE)
    nse_r  = torch.empty(B, N, device=DEVICE, dtype=DTYPE)

    fires_c  = torch.empty(CHUNK_RAND, B, device=DEVICE, dtype=torch.bool)
    cx_z0_c  = torch.empty(CHUNK_RAND, B, device=DEVICE, dtype=torch.long)
    cx_z1_c  = torch.empty(CHUNK_RAND, B, device=DEVICE, dtype=torch.long)
    cy_c     = torch.empty(CHUNK_RAND, B, device=DEVICE, dtype=torch.long)
    cptr     = CHUNK_RAND

    print(f'    Warming up compiler...', flush=True)

    for t in range(nsteps):
        if cptr >= CHUNK_RAND:
            fires_c.copy_(torch.rand(CHUNK_RAND, B, device=DEVICE) < wp)
            torch.randint(0,    L//2, (CHUNK_RAND, B), device=DEVICE, out=cx_z0_c)
            torch.randint(L//2, L,   (CHUNK_RAND, B), device=DEVICE, out=cx_z1_c)
            torch.randint(0,    L,   (CHUNK_RAND, B), device=DEVICE, out=cy_c)
            cptr = 0

        rnd0.uniform_()
        rnd1.uniform_()
        rnd_w0.uniform_()
        rnd_w1.uniform_()
        cau_r.uniform_()
        nse_r.normal_()

        s, base, mid, phi, streak, wave_z, M = step_fn(
            s, base, mid, phi, streak, wave_z,
            rnd0, rnd1,
            fires_c[cptr], cx_z0_c[cptr], cx_z1_c[cptr], cy_c[cptr],
            rnd_w0, rnd_w1,
            cau_r, nse_r, P_ten)
        cptr += 1

        if t >= ss0 and m_idx < n_data:
            M_buf[m_idx] = M
            m_idx += 1

        if (t+1) % 10000 == 0:
            print(f'    step {t+1}/{nsteps}', flush=True)

    M_stack = M_buf[:m_idx].cpu().numpy()

    results = []
    for b in range(B):
        M_arr  = M_stack[:, b]
        absM   = float(np.mean(np.abs(M_arr)))
        M1     = float(np.mean(M_arr))          # <M>  (signed mean)
        M2     = float(np.mean(M_arr**2))        # <M^2>
        M4     = float(np.mean(M_arr**4))
        chi    = float(M2 - M1**2)              # var(M) = susceptibility
        U4     = float(1. - M4/(3.*M2**2)) if M2 > 1e-12 else float('nan')
        lags, acf = compute_autocorr_absM(M_arr.tolist())
        results.append(dict(absM=absM, M1=M1, M2=M2, M4=M4,
                            chi=chi, U4=U4,
                            lags=lags, acf=acf, n_data=len(M_arr)))
    return results


# == Helpers ===================================================================

def compute_autocorr_absM(Mseries, max_lag=None):
    M=np.array(Mseries,dtype=float); absM=np.abs(M); n=len(absM)
    if n<20: return [],[]
    a_c=absM-float(np.mean(absM)); var_a=float(np.mean(a_c**2))
    if var_a<1e-20: return [],[]
    if max_lag is None: max_lag=min(n//4,20000)
    npad=1
    while npad<2*n: npad*=2
    raw=np.real(np.fft.irfft(np.abs(np.fft.rfft(a_c,n=npad))**2))[:n]
    lags=np.arange(max_lag+1); acf=raw[:max_lag+1]/((n-lags)*var_a)
    return lags.tolist(), acf.tolist()

def fit_tau_exp(lags, acf, t_min=50, t_max=None):
    if t_max is None: t_max=max(lags)//2 if lags else 100
    pairs=[(t,C) for t,C in zip(lags,acf) if t_min<=t<=t_max and C>0]
    if len(pairs)<5: return float('nan'),float('nan')
    xs=[t for t,C in pairs]; ys=[math.log(C) for t,C in pairs]
    n=len(xs); mx,my=sum(xs)/n,sum(ys)/n
    ssxx=sum((x-mx)**2 for x in xs); ssxy=sum((x-mx)*(y-my) for x,y in zip(xs,ys))
    if ssxx<1e-10: return float('nan'),float('nan')
    slope=ssxy/ssxx
    if slope>=0: return float('nan'),float('nan')
    yhat=[my+slope*(x-mx) for x in xs]
    sstot=sum((y-my)**2 for y in ys); ssres=sum((y-yh)**2 for y,yh in zip(ys,yhat))
    r2=1.-ssres/sstot if sstot>1e-10 else float('nan')
    return -1./slope, r2

def fit_power(x_list, y_list, x_exp=-1.0):
    """Fit y ~ x^x_exp by OLS in log-log. Returns (exponent, r2).
    x_exp=-1 gives slope directly (general). Returns slope."""
    pairs=[(x,y) for x,y in zip(x_list,y_list)
           if not math.isnan(y) and y>0 and x>0]
    if len(pairs)<3: return float('nan'),float('nan')
    lx=[math.log(x) for x,y in pairs]; ly=[math.log(y) for x,y in pairs]
    n=len(lx); mx,my=sum(lx)/n,sum(ly)/n
    ssxx=sum((x-mx)**2 for x in lx); ssxy=sum((x-mx)*(y-my) for x,y in zip(lx,ly))
    if ssxx<1e-20: return float('nan'),float('nan')
    slope=ssxy/ssxx
    yhat=[my+slope*(x-mx) for x in lx]
    sstot=sum((y-my)**2 for y in ly); ssres=sum((y-yh)**2 for y,yh in zip(ly,yhat))
    r2=1.-ssres/sstot if sstot>1e-10 else float('nan')
    return slope, r2

def fit_z_from_P(P_list, tau_list, nu=0.98):
    slope, r2 = fit_power(P_list, tau_list)
    return (-slope/nu if not math.isnan(slope) else float('nan')), r2

def fit_gamma_from_P(P_list, chi_list):
    """chi ~ P^{-gamma} -> slope of log(chi) vs log(P) = -gamma."""
    slope, r2 = fit_power(P_list, chi_list)
    return (-slope if not math.isnan(slope) else float('nan')), r2

def fit_z_from_tau_L(L_list, tau_list):
    slope, r2 = fit_power(L_list, tau_list)
    return slope, r2

def avg_acf(results_list):
    valid=[r for r in results_list if r and len(r.get('acf',[]))>0]
    if not valid: return [],[]
    lags_ref=valid[0]['lags']; acc=np.zeros(len(lags_ref)); cnt=0
    for r in valid:
        if r.get('lags')==lags_ref:
            a=np.array(r['acf'])
            if len(a)==len(lags_ref): acc+=a; cnt+=1
    return lags_ref,(acc/cnt).tolist() if cnt>0 else []

def ms(vals):
    v=[x for x in vals if x is not None and not math.isnan(x)]
    return float(np.mean(v)) if v else float('nan')


# == MAIN ======================================================================
if __name__ == '__main__':
    RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    raw={}
    if RESULTS_FILE.exists():
        with open(RESULTS_FILE) as f: raw=json.load(f)

    # ── Phase A ────────────────────────────────────────────────────────────────
    if 'A' not in raw: raw['A']={}
    for pc in PC_A:
        key=f'pc{pc:.4f}'
        if key in raw['A']:
            print(f'Phase A P={pc:.4f}: already done, skipping.'); continue
        print(f'\nPhase A: P={pc:.4f}, L={L_A}, {len(SEEDS_A)} seeds, '
              f'{NSTEPS_A} steps (GPU+compile) ...')
        with torch.no_grad():
            res=run_batch_gpu(L_A, [pc]*len(SEEDS_A), SEEDS_A, NSTEPS_A, RW)
        raw['A'][key]={str(s):r for s,r in zip(SEEDS_A,res)}
        with open(RESULTS_FILE,'w') as f: json.dump(raw,f)
        print(f'  Phase A P={pc:.4f} done.')

    # ── Phase B ────────────────────────────────────────────────────────────────
    if 'B' not in raw: raw['B']={}
    for pc in PC_B:
        key=f'pc{pc:.5f}'
        if key in raw['B']:
            print(f'Phase B P={pc:.5f}: already done, skipping.'); continue
        print(f'\nPhase B: P={pc:.5f}, L={L_B}, {len(SEEDS_B)} seeds, '
              f'{NSTEPS_B} steps (GPU+compile) ...')
        with torch.no_grad():
            res=run_batch_gpu(L_B, [pc]*len(SEEDS_B), SEEDS_B, NSTEPS_B, RW)
        raw['B'][key]={str(s):r for s,r in zip(SEEDS_B,res)}
        with open(RESULTS_FILE,'w') as f: json.dump(raw,f)
        print(f'  Phase B P={pc:.5f} done.')

    # ── Phase C ────────────────────────────────────────────────────────────────
    if 'C' not in raw: raw['C']={}
    for L in L_C:
        key=f'L{L}'
        if key in raw['C']:
            print(f'Phase C L={L}: already done, skipping.'); continue
        print(f'\nPhase C: L={L}, P={PC_C}, {len(SEEDS_C)} seeds, '
              f'{NSTEPS_C} steps (GPU+compile) ...')
        with torch.no_grad():
            res=run_batch_gpu(L, [PC_C]*len(SEEDS_C), SEEDS_C, NSTEPS_C, RW)
        raw['C'][key]={str(s):r for s,r in zip(SEEDS_C,res)}
        with open(RESULTS_FILE,'w') as f: json.dump(raw,f)
        print(f'  Phase C L={L} done.')

    # ── Analysis ───────────────────────────────────────────────────────────────
    print(f'\n=== PHASE A (L={L_A}, large-L validation) ===')
    print(f'{"P":>8}  {"U4":>7}  {"absM":>9}  {"chi":>10}  {"tau":>10}  {"R2":>7}')
    tau_A={}; chi_A={}
    for pc in PC_A:
        seeds=raw.get('A',{}).get(f'pc{pc:.4f}',{})
        u4s=[v['U4']   for v in seeds.values() if not math.isnan(v.get('U4',float('nan')))]
        Ms =[v['absM'] for v in seeds.values() if not math.isnan(v.get('absM',float('nan')))]
        ch =[v['chi']  for v in seeds.values() if not math.isnan(v.get('chi',float('nan')))]
        lags,acf=avg_acf(list(seeds.values()))
        tau_e,r2_e=fit_tau_exp(lags,acf,t_min=50,t_max=max(lags)//2 if lags else 100)
        tau_A[pc]=tau_e; chi_A[pc]=ms(ch)
        print(f'{pc:8.4f}  {ms(u4s):7.3f}  {ms(Ms):9.5f}  {ms(ch):10.4f}  '
              f'{tau_e:10.1f}  {r2_e:7.3f}')

    print(f'\n=== PHASE B (L={L_B}, deep P) ===')
    print(f'{"P":>9}  {"U4":>7}  {"absM":>9}  {"chi":>10}  {"tau":>10}  {"R2":>7}')
    tau_B={}; chi_B={}
    for pc in PC_B:
        seeds=raw.get('B',{}).get(f'pc{pc:.5f}',{})
        u4s=[v['U4']   for v in seeds.values() if not math.isnan(v.get('U4',float('nan')))]
        Ms =[v['absM'] for v in seeds.values() if not math.isnan(v.get('absM',float('nan')))]
        ch =[v['chi']  for v in seeds.values() if not math.isnan(v.get('chi',float('nan')))]
        lags,acf=avg_acf(list(seeds.values()))
        tau_e,r2_e=fit_tau_exp(lags,acf,t_min=50,t_max=max(lags)//2 if lags else 100)
        tau_B[pc]=tau_e; chi_B[pc]=ms(ch)
        print(f'{pc:9.5f}  {ms(u4s):7.3f}  {ms(Ms):9.5f}  {ms(ch):10.4f}  '
              f'{tau_e:10.1f}  {r2_e:7.3f}')

    print(f'\n=== PHASE C (FSS P={PC_C}) ===')
    print(f'{"L":>5}  {"U4":>7}  {"absM":>9}  {"chi":>10}  {"tau":>10}  {"R2":>7}')
    tau_C={}; chi_C={}
    for L in L_C:
        seeds=raw.get('C',{}).get(f'L{L}',{})
        u4s=[v['U4']   for v in seeds.values() if not math.isnan(v.get('U4',float('nan')))]
        Ms =[v['absM'] for v in seeds.values() if not math.isnan(v.get('absM',float('nan')))]
        ch =[v['chi']  for v in seeds.values() if not math.isnan(v.get('chi',float('nan')))]
        lags,acf=avg_acf(list(seeds.values()))
        tau_e,r2_e=fit_tau_exp(lags,acf,t_min=50,t_max=max(lags)//2 if lags else 100)
        tau_C[L]=tau_e; chi_C[L]=ms(ch)
        print(f'{L:5d}  {ms(u4s):7.3f}  {ms(Ms):9.5f}  {ms(ch):10.4f}  '
              f'{tau_e:10.1f}  {r2_e:7.3f}')

    print(f'\n=== z FIT ===')
    all_P  = [pc for pc in PC_A+PC_B]
    all_T  = [{**tau_A,**tau_B}.get(pc,float('nan')) for pc in all_P]
    z_all, r2_zall = fit_z_from_P(all_P, all_T)
    z_B,   r2_zB   = fit_z_from_P(list(tau_B.keys()), list(tau_B.values()))
    z_C,   r2_zC   = fit_z_from_tau_L(list(tau_C.keys()), list(tau_C.values()))
    print(f'  z (all phases):  {z_all:.3f}  R2={r2_zall:.3f}')
    print(f'  z (Phase B):     {z_B:.3f}  R2={r2_zB:.3f}')
    print(f'  z (Phase C FSS): {z_C:.3f}  R2={r2_zC:.3f}')
    print(f'  Model A:   z=2.000  |  KPZ: z=1.500')

    print(f'\n=== gamma FIT (chi~P^{{-gamma}}) ===')
    all_chi_P = [pc for pc in PC_A+PC_B]
    all_chi   = [{**chi_A,**chi_B}.get(pc,float('nan')) for pc in all_chi_P]
    gamma_all, r2_gall = fit_gamma_from_P(all_chi_P, all_chi)
    gamma_B,   r2_gB   = fit_gamma_from_P(list(chi_B.keys()), list(chi_B.values()))
    print(f'  gamma (all):     {gamma_all:.3f}  R2={r2_gall:.3f}')
    print(f'  gamma (Phase B): {gamma_B:.3f}  R2={r2_gB:.3f}')
    # Scaling relation check: gamma = nu*(2-eta_eff)
    eta_eff = 2.0 - gamma_all/NU if not math.isnan(gamma_all) else float('nan')
    print(f'  Implied eta_eff (from Fisher gamma=nu*(2-eta)): {eta_eff:.3f}')
    # Widom: gamma = beta*(delta-1) -> delta = 1 + gamma/beta
    delta = 1.0 + gamma_all/BETA_EXP if not math.isnan(gamma_all) else float('nan')
    print(f'  Implied delta (from Widom gamma=beta*(delta-1)):  {delta:.3f}')
    # Rushbrooke: alpha + 2*beta + gamma = 2 -> alpha = 2 - 2*beta - gamma
    alpha = 2.0 - 2.0*BETA_EXP - gamma_all if not math.isnan(gamma_all) else float('nan')
    print(f'  Implied alpha (Rushbrooke 2-2beta-gamma):         {alpha:.3f}')

    # Build analysis dict
    analysis = dict(
        tau_A={str(pc): tau_A.get(pc, float('nan')) for pc in PC_A},
        tau_B={str(pc): tau_B.get(pc, float('nan')) for pc in PC_B},
        tau_C={str(L):  tau_C.get(L,  float('nan')) for L  in L_C},
        chi_A={str(pc): chi_A.get(pc, float('nan')) for pc in PC_A},
        chi_B={str(pc): chi_B.get(pc, float('nan')) for pc in PC_B},
        chi_C={str(L):  chi_C.get(L,  float('nan')) for L  in L_C},
        z_all=z_all   if not math.isnan(z_all)   else None,
        r2_zall=r2_zall if not math.isnan(r2_zall) else None,
        z_B=z_B       if not math.isnan(z_B)     else None,
        r2_zB=r2_zB   if not math.isnan(r2_zB)   else None,
        z_C=z_C       if not math.isnan(z_C)     else None,
        r2_zC=r2_zC   if not math.isnan(r2_zC)   else None,
        gamma_all=gamma_all if not math.isnan(gamma_all) else None,
        r2_gall=r2_gall     if not math.isnan(r2_gall)   else None,
        gamma_B=gamma_B     if not math.isnan(gamma_B)   else None,
        r2_gB=r2_gB         if not math.isnan(r2_gB)     else None,
        eta_eff=eta_eff if not math.isnan(eta_eff) else None,
        delta=delta     if not math.isnan(delta)   else None,
        alpha=alpha     if not math.isnan(alpha)   else None,
        pc_A=PC_A, L_A=L_A,
        pc_B=PC_B, L_B=L_B,
        pc_C=PC_C, L_C=L_C,
        nu=NU, beta=BETA_EXP, tau_VCSM=1000.0,
        acf_A={}, acf_B={}, acf_C={},
    )
    for pc in PC_A:
        lags,acf=avg_acf(list(raw.get('A',{}).get(f'pc{pc:.4f}',{}).values()))
        analysis['acf_A'][str(pc)]={'lags':lags,'acf':acf}
    for pc in PC_B:
        lags,acf=avg_acf(list(raw.get('B',{}).get(f'pc{pc:.5f}',{}).values()))
        analysis['acf_B'][str(pc)]={'lags':lags,'acf':acf}
    for L in L_C:
        lags,acf=avg_acf(list(raw.get('C',{}).get(f'L{L}',{}).values()))
        analysis['acf_C'][str(L)]={'lags':lags,'acf':acf}

    with open(ANALYSIS_FILE,'w') as f: json.dump(analysis, f, indent=2)
    print(f'\nDone. Analysis -> {ANALYSIS_FILE}')
