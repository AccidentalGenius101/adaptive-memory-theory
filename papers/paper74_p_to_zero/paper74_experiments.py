"""
Paper 74 - Dynamic Exponent z: Approaching P_c = 0+
GPU-accelerated via PyTorch (RTX 3060) -- v3

Key optimizations:
  1. No .item() sync barriers -- always compute wave with masking
  2. In-place uniform_() / normal_() on pre-allocated buffers (no allocation per step)
  3. Chunk-based pre-generation of fires/cx/cy (5000 steps at a time)
  4. @torch.compile(mode='reduce-overhead') on step function -- CUDA graph replay
  5. Pre-allocated M buffer, single .cpu() transfer at end

Phase A: loaded from saved results
Phase B: P=0.0001 loaded; resume P=0.0002 and P=0.0005
Phase C: FSS at P=0.001, L in {40,60,80,100,120}
"""

import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import numpy as np
import json
import math
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

# Phases
PC_A=[0.001,0.002,0.003,0.005,0.007,0.010]; L_A=80
PC_B=[0.0001,0.0002,0.0005]; L_B=80; SEEDS_B=list(range(8)); NSTEPS_B=150_000
PC_C=0.001; L_C=[40,60,80,100,120]; SEEDS_C=list(range(8)); NSTEPS_C=100_000

RESULTS_FILE  = Path(__file__).parent / 'results' / 'paper74_results.json'
ANALYSIS_FILE = Path(__file__).parent / 'results' / 'paper74_analysis.json'
CHUNK_RAND    = 5000   # refill fires/cx/cy every this many steps


# == Compiled step function ====================================================

def build_step_fn(nb, cb0, cb1, col_g, row_g, z0f, z1f, n_z0, n_z1,
                  L, B, N, r_w):
    """Build and compile the per-step function for fixed (L, B, N)."""
    cb0_e = cb0.unsqueeze(0).expand(B, -1)   # (B, len_cb0)
    cb1_e = cb1.unsqueeze(0).expand(B, -1)
    col_e = col_g.unsqueeze(0).expand(B, -1).contiguous()  # (B, N)
    row_e = row_g.unsqueeze(0).expand(B, -1).contiguous()

    # Scalar tensors for zone normalisation (avoids .item() in compile)
    n_z0f = n_z0.to(DTYPE)
    n_z1f = n_z1.to(DTYPE)

    def _metro_sub(s, idx_sub, idx_sub_e, ns_buf, rnd):
        """Checkerboard sub-lattice Metro update. Returns new s."""
        nb_idx = nb[idx_sub]                            # (len_sub, 4)
        ns     = s[:, nb_idx].sum(-1)                  # (B, len_sub)
        dE     = 2.0 * s[:, idx_sub] * ns
        acc    = (dE <= 0) | (rnd < torch.exp(torch.clamp(-dE / T_FIXED, -20., 20.)))
        s_sub  = torch.where(acc, -s[:, idx_sub], s[:, idx_sub])
        return s.scatter(1, idx_sub_e, s_sub)

    @torch.compile(mode='default', dynamic=False)
    def step(s, base, mid, phi, streak, wave_z,
             rnd0, rnd1,                   # (B, len_cb0/1)  main metro randoms
             fires,                        # (B,)   bool
             cx_z0, cx_z1, cy,            # (B,)   int64 wave centres
             rnd_w0, rnd_w1,              # (WAVE_DUR*B, len_cb0/1) wave metro
             cau_rnd, nse_rnd,            # (B, N)
             P_ten):                      # (B,)   P_causal per seed
        """One full VCML step -- no .item(), no Python branches on tensors."""

        # ---- Main Metropolis -------------------------------------------------
        s = _metro_sub(s, cb0, cb0_e, None, rnd0)
        s = _metro_sub(s, cb1, cb1_e, None, rnd1)

        # ---- Baseline + streak -----------------------------------------------
        base   = base + BETA_BASE * (s - base)
        same   = (s > 0) == (base > 0)
        streak = torch.where(same, streak + 1, torch.zeros_like(streak))
        mid    = mid * MID_DECAY

        # ---- Wave (always computed, masked by fires) -------------------------
        cx  = torch.where(wave_z == 0, cx_z0, cx_z1)     # (B,)
        dx  = torch.minimum((col_e - cx.unsqueeze(1)).abs(),
                            L - (col_e - cx.unsqueeze(1)).abs())
        dy  = torch.minimum((row_e - cy.unsqueeze(1)).abs(),
                            L - (row_e - cy.unsqueeze(1)).abs())
        in_wave  = (dx + dy <= r_w)                       # (B, N)
        fires_e  = fires.unsqueeze(1).expand(-1, N)
        hit_f    = (in_wave & fires_e).to(DTYPE)          # (B, N) float mask

        he_sign  = torch.where(wave_z == 0, 1., -1.).to(DTYPE)  # (B,)
        h_ext    = hit_f * he_sign.unsqueeze(1) * EXT_FIELD     # (B, N)

        # Inner Metro with field
        s_w = s
        for d in range(WAVE_DUR):
            s_w = _metro_sub(s_w, cb0, cb0_e, None,
                             rnd_w0[d*B:(d+1)*B] + h_ext[:, cb0] * 0.)  # pass rnd
            # Actually need to pass h_ext into metro -- redo inline:
            ns0w = s_w[:, nb[cb0]].sum(-1)
            h0   = h_ext[:, cb0]
            dE0w = 2.0*s_w[:, cb0]*ns0w - 2.0*h0*s_w[:, cb0]
            ac0w = (dE0w <= 0) | (rnd_w0[d*B:(d+1)*B] <
                                  torch.exp(torch.clamp(-dE0w/T_FIXED,-20.,20.)))
            s_w  = s_w.scatter(1, cb0_e, torch.where(ac0w,-s_w[:,cb0],s_w[:,cb0]))

            ns1w = s_w[:, nb[cb1]].sum(-1)
            h1   = h_ext[:, cb1]
            dE1w = 2.0*s_w[:,cb1]*ns1w - 2.0*h1*s_w[:,cb1]
            ac1w = (dE1w <= 0) | (rnd_w1[d*B:(d+1)*B] <
                                  torch.exp(torch.clamp(-dE1w/T_FIXED,-20.,20.)))
            s_w  = s_w.scatter(1, cb1_e, torch.where(ac1w,-s_w[:,cb1],s_w[:,cb1]))

        # Apply wave to firers only
        s = torch.where(fires_e, s_w, s)

        # mid update (causal or noise)
        dev   = s - base
        wz_e  = (wave_z == 0).unsqueeze(1).expand(-1, N)
        z0_e  = z0f.bool().unsqueeze(0).expand(B, -1)
        sig   = torch.where(z0_e == wz_e, dev, -dev)
        cau   = cau_rnd < P_ten.unsqueeze(1)
        nse   = nse_rnd * (dev.std(dim=1, keepdim=True).clamp(min=0.01) + 0.5)
        mid   = mid + FA * torch.where(cau, sig, nse) * hit_f

        # Flip wave_zone for firers
        wave_z = torch.where(fires, 1 - wave_z, wave_z)

        # ---- Gate + phi decay -----------------------------------------------
        gate   = streak >= SS
        phi    = phi + FA * torch.where(gate, mid - phi, torch.zeros_like(phi))
        streak = torch.where(gate, torch.zeros_like(streak), streak)
        phi    = phi * FIELD_DECAY

        # ---- Zone-mean M -----------------------------------------------------
        M = (phi * z0f).sum(1) / n_z0f - (phi * z1f).sum(1) / n_z1f  # (B,)

        return s, base, mid, phi, streak, wave_z, M

    return step


# == Main simulation ===========================================================

def run_batch_gpu(L, P_causal_list, seed_list, nsteps, r_w):
    B = len(seed_list); N = L*L; wp = _prob_A(L)

    # Geometry
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

    # Init state
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

    # Pre-allocated per-step random buffers (filled in-place via uniform_/normal_)
    rnd0   = torch.empty(B, len(cb0), device=DEVICE, dtype=DTYPE)
    rnd1   = torch.empty(B, len(cb1), device=DEVICE, dtype=DTYPE)
    rnd_w0 = torch.empty(WAVE_DUR*B, len(cb0), device=DEVICE, dtype=DTYPE)
    rnd_w1 = torch.empty(WAVE_DUR*B, len(cb1), device=DEVICE, dtype=DTYPE)
    cau_r  = torch.empty(B, N, device=DEVICE, dtype=DTYPE)
    nse_r  = torch.empty(B, N, device=DEVICE, dtype=DTYPE)

    # Chunk-pre-generated fires / wave centres
    fires_c  = torch.empty(CHUNK_RAND, B, device=DEVICE, dtype=torch.bool)
    cx_z0_c  = torch.empty(CHUNK_RAND, B, device=DEVICE, dtype=torch.long)
    cx_z1_c  = torch.empty(CHUNK_RAND, B, device=DEVICE, dtype=torch.long)
    cy_c     = torch.empty(CHUNK_RAND, B, device=DEVICE, dtype=torch.long)
    cptr     = CHUNK_RAND  # force refill on step 0

    print(f'    Warming up compiler...', flush=True)

    for t in range(nsteps):
        # Refill chunk cache
        if cptr >= CHUNK_RAND:
            fires_c.copy_(torch.rand(CHUNK_RAND, B, device=DEVICE) < wp)
            torch.randint(0,    L//2, (CHUNK_RAND, B), device=DEVICE, out=cx_z0_c)
            torch.randint(L//2, L,   (CHUNK_RAND, B), device=DEVICE, out=cx_z1_c)
            torch.randint(0,    L,   (CHUNK_RAND, B), device=DEVICE, out=cy_c)
            cptr = 0

        # Fill per-step buffers in-place (no allocation)
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
        M_arr = M_stack[:, b]
        absM  = float(np.mean(np.abs(M_arr)))
        M2    = float(np.mean(M_arr**2))
        M4    = float(np.mean(M_arr**4))
        U4    = float(1. - M4/(3.*M2**2)) if M2 > 1e-12 else float('nan')
        lags, acf = compute_autocorr_absM(M_arr.tolist())
        results.append(dict(absM=absM, M2=M2, U4=U4, lags=lags, acf=acf,
                            n_data=len(M_arr)))
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

def fit_tau_exp(lags,acf,t_min=50,t_max=None):
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

def fit_z_from_P(P_list,tau_list,nu=0.98):
    pairs=[(P,t) for P,t in zip(P_list,tau_list) if not math.isnan(t) and t>0 and P>0]
    if len(pairs)<3: return float('nan'),float('nan')
    lx=[math.log(P) for P,t in pairs]; ly=[math.log(t) for P,t in pairs]
    n=len(lx); mx,my=sum(lx)/n,sum(ly)/n
    ssxx=sum((x-mx)**2 for x in lx); ssxy=sum((x-mx)*(y-my) for x,y in zip(lx,ly))
    if ssxx<1e-20: return float('nan'),float('nan')
    slope=ssxy/ssxx
    yhat=[my+slope*(x-mx) for x in lx]
    sstot=sum((y-my)**2 for y in ly); ssres=sum((y-yh)**2 for y,yh in zip(ly,yhat))
    r2=1.-ssres/sstot if sstot>1e-10 else float('nan')
    return -slope/nu, r2

def fit_z_from_tau_L(L_list,tau_list):
    pairs=[(L,t) for L,t in zip(L_list,tau_list) if not math.isnan(t) and t>0 and L>0]
    if len(pairs)<3: return float('nan'),float('nan')
    lx=[math.log(L) for L,t in pairs]; ly=[math.log(t) for L,t in pairs]
    n=len(lx); mx,my=sum(lx)/n,sum(ly)/n
    ssxx=sum((x-mx)**2 for x in lx); ssxy=sum((x-mx)*(y-my) for x,y in zip(lx,ly))
    if ssxx<1e-20: return float('nan'),float('nan')
    slope=ssxy/ssxx
    yhat=[my+slope*(x-mx) for x in lx]
    sstot=sum((y-my)**2 for y in ly); ssres=sum((y-yh)**2 for y,yh in zip(ly,yhat))
    r2=1.-ssres/sstot if sstot>1e-10 else float('nan')
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

    print('Phase A: loaded.' if 'A' in raw else 'WARNING: Phase A missing.')

    if 'B' not in raw: raw['B']={}
    for pc in PC_B:
        key=f'pc{pc:.4f}'
        if key in raw['B']:
            print(f'Phase B P={pc:.4f}: already done, skipping.'); continue
        print(f'\nPhase B: P={pc:.4f}, L={L_B}, {len(SEEDS_B)} seeds, '
              f'{NSTEPS_B} steps (GPU+compile) ...')
        with torch.no_grad():
            res=run_batch_gpu(L_B,[pc]*len(SEEDS_B),SEEDS_B,NSTEPS_B,RW)
        raw['B'][key]={str(s):r for s,r in zip(SEEDS_B,res)}
        with open(RESULTS_FILE,'w') as f: json.dump(raw,f)
        print(f'  Phase B P={pc:.4f} done.')

    if 'C' not in raw: raw['C']={}
    for L in L_C:
        key=f'L{L}'
        if key in raw['C']:
            print(f'Phase C L={L}: already done, skipping.'); continue
        print(f'\nPhase C: L={L}, P={PC_C}, {len(SEEDS_C)} seeds, '
              f'{NSTEPS_C} steps (GPU+compile) ...')
        with torch.no_grad():
            res=run_batch_gpu(L,[PC_C]*len(SEEDS_C),SEEDS_C,NSTEPS_C,RW)
        raw['C'][key]={str(s):r for s,r in zip(SEEDS_C,res)}
        with open(RESULTS_FILE,'w') as f: json.dump(raw,f)
        print(f'  Phase C L={L} done.')

    # Analysis
    print(f'\n=== PHASE A (bridge) ===')
    print(f'{"P":>8}  {"xi":>6}  {"U4":>7}  {"absM":>9}  {"tau":>10}  {"R2":>7}')
    tau_A={}
    for pc in PC_A:
        seeds=raw.get('A',{}).get(f'pc{pc:.4f}',{})
        u4s=[v['U4'] for v in seeds.values() if not math.isnan(v.get('U4',float('nan')))]
        Ms =[v['absM'] for v in seeds.values() if not math.isnan(v.get('absM',float('nan')))]
        lags,acf=avg_acf(list(seeds.values()))
        tau_e,r2_e=fit_tau_exp(lags,acf,t_min=50,t_max=max(lags)//2 if lags else 100)
        tau_A[pc]=tau_e
        flag='***' if (not math.isnan(tau_e) and tau_e>260) else ''
        print(f'{pc:8.4f}  {5*math.sqrt(pc):6.3f}  {ms(u4s):7.3f}  {ms(Ms):9.5f}  '
              f'{tau_e:10.1f}  {r2_e:7.3f}  {flag}')

    print(f'\n=== PHASE B (deep P) ===')
    print(f'{"P":>8}  {"xi":>6}  {"U4":>7}  {"absM":>9}  {"tau":>10}  {"R2":>7}')
    tau_B={}
    for pc in PC_B:
        seeds=raw.get('B',{}).get(f'pc{pc:.4f}',{})
        u4s=[v['U4'] for v in seeds.values() if not math.isnan(v.get('U4',float('nan')))]
        Ms =[v['absM'] for v in seeds.values() if not math.isnan(v.get('absM',float('nan')))]
        lags,acf=avg_acf(list(seeds.values()))
        tau_e,r2_e=fit_tau_exp(lags,acf,t_min=50,t_max=max(lags)//2 if lags else 100)
        tau_B[pc]=tau_e
        flag='***' if (not math.isnan(tau_e) and tau_e>260) else ''
        print(f'{pc:8.4f}  {5*math.sqrt(pc):6.3f}  {ms(u4s):7.3f}  {ms(Ms):9.5f}  '
              f'{tau_e:10.1f}  {r2_e:7.3f}  {flag}')

    print(f'\n=== PHASE C (FSS P={PC_C}) ===')
    print(f'{"L":>5}  {"U4":>7}  {"absM":>9}  {"tau":>10}  {"R2":>7}')
    tau_C={}
    for L in L_C:
        seeds=raw.get('C',{}).get(f'L{L}',{})
        u4s=[v['U4'] for v in seeds.values() if not math.isnan(v.get('U4',float('nan')))]
        Ms =[v['absM'] for v in seeds.values() if not math.isnan(v.get('absM',float('nan')))]
        lags,acf=avg_acf(list(seeds.values()))
        tau_e,r2_e=fit_tau_exp(lags,acf,t_min=50,t_max=max(lags)//2 if lags else 100)
        tau_C[L]=tau_e
        print(f'{L:5d}  {ms(u4s):7.3f}  {ms(Ms):9.5f}  {tau_e:10.1f}  {r2_e:7.3f}')

    print(f'\n=== z FIT ===')
    all_P=[pc for pc in PC_A+PC_B if not math.isnan({**tau_A,**tau_B}.get(pc,float('nan')))]
    all_T=[{**tau_A,**tau_B}[pc] for pc in all_P]
    z_all,r2_all=fit_z_from_P(all_P,all_T)
    z_Bf,r2_Bf=fit_z_from_P([p for p in PC_B if not math.isnan(tau_B.get(p,float('nan')))],
                              [tau_B[p] for p in PC_B if not math.isnan(tau_B.get(p,float('nan')))])
    z_Cf,r2_Cf=fit_z_from_tau_L([L for L in L_C if not math.isnan(tau_C.get(L,float('nan')))],
                                  [tau_C[L] for L in L_C if not math.isnan(tau_C.get(L,float('nan')))])
    print(f'  z (all phases):  {z_all:.3f}  R2={r2_all:.3f}')
    print(f'  z (Phase B):     {z_Bf:.3f}  R2={r2_Bf:.3f}')
    print(f'  z (Phase C FSS): {z_Cf:.3f}  R2={r2_Cf:.3f}')
    print(f'  Model A:   z=2.000')

    analysis=dict(
        tau_A={str(pc):tau_A.get(pc,float('nan')) for pc in PC_A},
        tau_B={str(pc):tau_B.get(pc,float('nan')) for pc in PC_B},
        tau_C={str(L): tau_C.get(L, float('nan')) for L in L_C},
        z_all=z_all if not math.isnan(z_all) else None, r2_all=r2_all if not math.isnan(r2_all) else None,
        z_B=z_Bf if not math.isnan(z_Bf) else None, r2_B=r2_Bf if not math.isnan(r2_Bf) else None,
        z_C=z_Cf if not math.isnan(z_Cf) else None, r2_C=r2_Cf if not math.isnan(r2_Cf) else None,
        pc_A=PC_A,L_A=L_A, pc_B=PC_B,L_B=L_B, pc_C=PC_C,L_C=L_C,
        nu=NU,beta=BETA_EXP,tau_VCSM=200.0,
        acf_A={},acf_B={},acf_C={},
    )
    for pc in PC_A:
        lags,acf=avg_acf(list(raw.get('A',{}).get(f'pc{pc:.4f}',{}).values()))
        analysis['acf_A'][str(pc)]={'lags':lags,'acf':acf}
    for pc in PC_B:
        lags,acf=avg_acf(list(raw.get('B',{}).get(f'pc{pc:.4f}',{}).values()))
        analysis['acf_B'][str(pc)]={'lags':lags,'acf':acf}
    for L in L_C:
        lags,acf=avg_acf(list(raw.get('C',{}).get(f'L{L}',{}).values()))
        analysis['acf_C'][str(L)]={'lags':lags,'acf':acf}

    with open(ANALYSIS_FILE,'w') as f: json.dump(analysis,f,indent=2)
    print(f'\nDone. Analysis -> {ANALYSIS_FILE}')
