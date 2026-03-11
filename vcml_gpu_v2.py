"""
vcml_gpu_v2.py - Optimised VCML GPU core
==========================================
Changes over v1 (paper74/75):
  1. bfloat16  -- halves memory bandwidth; same CUDA-core compute throughput
                  on Ampere. Memory-bound ops (scatter, gather, element-wise)
                  run ~2x faster. No kernel-fusion loss.
  2. TRITON_CACHE_DIR = 'C:/tc'  -- avoids Windows 260-char MAX_PATH bug
                  (fused kernel names become very long; default temp path
                   pushes total path length over 260 chars on Windows)
  3. mode='default'  -- keep the v1 approach that is proven to work.
                  reduce-overhead was tested but silently degrades to eager
                  when in-place mutations are detected, losing all fusion.
  4. Out-of-place ops inside compiled fn  -- same as v1, enables full
                  Triton kernel fusion across the entire step.
  5. chi = var(M) stored per seed  -- free from existing M buffer;
                  enables gamma (susceptibility exponent) measurement.

Verdict on reduce-overhead / CUDA graphs:
  PyTorch inductor's CUDA graph capture requires no in-place mutations on
  any traced tensor (explicit arg OR closure variable). Our simulation is
  inherently stateful; every meaningful op mutates s, phi, base, etc.
  The only way to force CUDA graphs is torch.cuda.CUDAGraph with manual
  buffer management -- possible but not worth the complexity given that
  bfloat16 + mode='default' already gives ~2x over float32.

Usage:
    from vcml_gpu_v2 import run_batch_gpu_v2, compute_autocorr_absM, fit_tau_exp
    results = run_batch_gpu_v2(L=80, P_causal_list=[0.001]*8,
                               seed_list=list(range(8)), nsteps=100_000)
"""

import os
os.environ.setdefault('TRITON_CACHE_DIR',       'C:/tc')
os.environ.setdefault('TORCHINDUCTOR_CACHE_DIR', 'C:/ind')

import numpy as np
import math
import torch
from typing import List

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DTYPE  = torch.bfloat16      # v2: bfloat16 for ~2x memory-bandwidth speedup

# Physical constants (identical to papers 62-75)
J          = 1.0
MID_DECAY  = 0.97
BETA_BASE  = 0.005
SS         = 8
WAVE_DUR   = 5
EXT_FIELD  = 1.5
T_FIXED    = 3.0
FIELD_DECAY= 0.999
SS_FRAC    = 0.40
FA         = 0.30
WE_BASE    = 25
L_BASE     = 40
RW         = 5

def _wave_prob(L: int) -> float:
    return min(1.0, 1.0 / (WE_BASE * (L_BASE / L) ** 2))


# == Compiled step function (out-of-place, mode='default') =====================

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
                                  torch.exp(torch.clamp(-dE0w/T_FIXED,-20.,20.)))
            s_w  = s_w.scatter(1, cb0_e, torch.where(ac0w,-s_w[:,cb0],s_w[:,cb0]))

            ns1w = s_w[:, nb[cb1]].sum(-1)
            h1   = h_ext[:, cb1]
            dE1w = 2.0*s_w[:,cb1]*ns1w - 2.0*h1*s_w[:,cb1]
            ac1w = (dE1w <= 0) | (rnd_w1[d*B:(d+1)*B] <
                                  torch.exp(torch.clamp(-dE1w/T_FIXED,-20.,20.)))
            s_w  = s_w.scatter(1, cb1_e, torch.where(ac1w,-s_w[:,cb1],s_w[:,cb1]))

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

def run_batch_gpu_v2(
    L: int,
    P_causal_list: List[float],
    seed_list: List[int],
    nsteps: int,
    r_w: int = RW,
    verbose: bool = True,
) -> List[dict]:
    B = len(seed_list); N = L * L; wp = _wave_prob(L)

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
        rng = np.random.RandomState(seed)
        s[b] = torch.from_numpy(
            rng.choice([-1.,1.], N).astype(np.float32)).to(DTYPE).to(DEVICE)
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

    CHUNK    = 5000
    fires_c  = torch.empty(CHUNK, B, device=DEVICE, dtype=torch.bool)
    cx_z0_c  = torch.empty(CHUNK, B, device=DEVICE, dtype=torch.long)
    cx_z1_c  = torch.empty(CHUNK, B, device=DEVICE, dtype=torch.long)
    cy_c     = torch.empty(CHUNK, B, device=DEVICE, dtype=torch.long)
    cptr     = CHUNK

    if verbose:
        print(f'    Warming up compiler (bf16/default)...', flush=True)

    with torch.no_grad():
        for t in range(nsteps):
            if cptr >= CHUNK:
                fires_c.copy_(torch.rand(CHUNK, B, device=DEVICE) < wp)
                torch.randint(0,    L//2, (CHUNK, B), device=DEVICE, out=cx_z0_c)
                torch.randint(L//2, L,   (CHUNK, B), device=DEVICE, out=cx_z1_c)
                torch.randint(0,    L,   (CHUNK, B), device=DEVICE, out=cy_c)
                cptr = 0

            rnd0.uniform_(); rnd1.uniform_()
            rnd_w0.uniform_(); rnd_w1.uniform_()
            cau_r.uniform_(); nse_r.normal_()

            s, base, mid, phi, streak, wave_z, M = step_fn(
                s, base, mid, phi, streak, wave_z,
                rnd0, rnd1,
                fires_c[cptr], cx_z0_c[cptr], cx_z1_c[cptr], cy_c[cptr],
                rnd_w0, rnd_w1, cau_r, nse_r, P_ten)
            cptr += 1

            if t >= ss0 and m_idx < n_data:
                M_buf[m_idx] = M
                m_idx += 1

            if verbose and (t + 1) % 10_000 == 0:
                print(f'    step {t+1}/{nsteps}', flush=True)

    M_stack = M_buf[:m_idx].float().cpu().numpy()

    results = []
    for b in range(B):
        M_arr = M_stack[:, b]
        absM  = float(np.mean(np.abs(M_arr)))
        M1    = float(np.mean(M_arr))
        M2    = float(np.mean(M_arr**2))
        M4    = float(np.mean(M_arr**4))
        chi   = float(M2 - M1**2)
        U4    = float(1. - M4/(3.*M2**2)) if M2 > 1e-12 else float('nan')
        lags, acf = compute_autocorr_absM(M_arr.tolist())
        results.append(dict(absM=absM, M1=M1, M2=M2, M4=M4,
                            chi=chi, U4=U4, lags=lags, acf=acf,
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

def fit_tau_exp(lags, acf, t_min=50, t_max=None):
    if t_max is None: t_max=max(lags)//2 if lags else 100
    pairs=[(t,C) for t,C in zip(lags,acf) if t_min<=t<=t_max and C>0]
    if len(pairs)<5: return float('nan'),float('nan')
    xs=[t for t,C in pairs]; ys=[math.log(C) for t,C in pairs]
    n=len(xs); mx,my=sum(xs)/n,sum(ys)/n
    ssxx=sum((x-mx)**2 for x in xs); ssxy=sum((x-mx)*(y-my) for x,y in zip(xs,ys))
    if ssxx<1e-10 or ssxy/ssxx>=0: return float('nan'),float('nan')
    slope=ssxy/ssxx
    yhat=[my+slope*(x-mx) for x in xs]
    sstot=sum((y-my)**2 for y in ys); ssres=sum((y-yh)**2 for y,yh in zip(ys,yhat))
    r2=1.-ssres/sstot if sstot>1e-10 else float('nan')
    return -1./slope, r2

def fit_power_law(x_list, y_list):
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


# == Benchmark =================================================================

if __name__ == '__main__':
    import sys, io, time
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

    print(f'Device: {DEVICE}')
    if DEVICE.type == 'cuda':
        print(f'  GPU: {torch.cuda.get_device_name(0)}')
        print(f'  VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB')
    print(f'  dtype: {DTYPE}  (v2=bfloat16 vs v1=float32)')

    for L_bm, B_bm in [(80, 8), (80, 16), (160, 8), (160, 16)]:
        print(f'\nBenchmark: L={L_bm}, B={B_bm}', flush=True)
        P_list = [0.001] * B_bm
        s_list = list(range(B_bm))
        N_COMPILE = 3000
        N_BENCH   = 30_000

        print(f'  compiling...', flush=True)
        t0 = time.time()
        run_batch_gpu_v2(L_bm, P_list, s_list, N_COMPILE, verbose=False)
        torch.cuda.synchronize()
        print(f'  compile time: {time.time()-t0:.1f}s', flush=True)

        t0 = time.time()
        run_batch_gpu_v2(L_bm, P_list, s_list, N_BENCH, verbose=False)
        torch.cuda.synchronize()
        dt = time.time() - t0
        sps = N_BENCH / dt
        print(f'  throughput: {sps:.0f} steps/sec  ({sps*60/1000:.1f}k steps/min)',
              flush=True)

    print('\nBenchmark done.')
