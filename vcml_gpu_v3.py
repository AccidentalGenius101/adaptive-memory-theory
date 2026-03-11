"""
vcml_gpu_v3.py - VCML GPU core with manual CUDA graph replay
=============================================================
Strategy:
  1. torch.compile(mode='default') fuses kernels inside the step
  2. torch.cuda.CUDAGraph captures the compiled kernel sequence
  3. Main loop only does: fill_randoms() (async) + g.replay() (async)
     -> zero Python overhead in the inner loop

Why manual CUDAGraph works where reduce-overhead failed:
  torch.compile's reduce-overhead auto-detects in-place mutations and
  refuses to build a CUDA graph. Manual torch.cuda.CUDAGraph has no such
  check -- you capture and replay whatever kernel sequence you want.
  The copy_() ops at the end of each step are valid captured CUDA ops.

Design:
  - State tensors (s, base, mid, phi, streak, wave_z) are pre-allocated
    at fixed addresses. The captured step writes new values back via copy_().
  - Random buffers (rnd0, rnd1, ...) are filled in-place BEFORE each replay
    on the same CUDA stream -> captured kernels see fresh randoms.
  - M accumulates in a GPU buffer; single .cpu() transfer at end.

Expected speedup vs v1 (mode='default' only):
  v1: ~75k steps/min (Python dispatch: ~0.7ms/step overhead)
  v3: target 300-800k steps/min (graph replay: ~10-50μs Python overhead)
  -> 4-10x improvement
"""

import os
os.environ.setdefault('TRITON_CACHE_DIR',       'C:/tc')
os.environ.setdefault('TORCHINDUCTOR_CACHE_DIR', 'C:/ind')

import numpy as np
import math
import torch
from typing import List

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DTYPE  = torch.float32     # float32: bfloat16 gave no benefit (CUDA-core ops)

# Physical constants
J=1.0; MID_DECAY=0.97; BETA_BASE=0.005; SS=8; WAVE_DUR=5
EXT_FIELD=1.5; T_FIXED=3.0; FIELD_DECAY=0.999; SS_FRAC=0.40; FA=0.30
WE_BASE=25; L_BASE=40; RW=5

def _wave_prob(L): return min(1.0, 1.0/(WE_BASE*(L_BASE/L)**2))


# == Build compiled compute fn + graph-capture wrapper ========================

def build_step_for_graph(
    nb, cb0, cb1, col_g, row_g, z0f, z1f, n_z0, n_z1,
    L, B, N, r_w,
    # Fixed state buffers (modified in-place during replay)
    s_buf, base_buf, mid_buf, phi_buf, streak_buf, wave_z_buf,
    M_out_buf,          # shape (B,) -- updated every replay
    P_ten,              # shape (B,) -- constant across replays
):
    """
    Returns step_fn(rnd0,rnd1,fires,cx_z0,cx_z1,cy,rnd_w0,rnd_w1,cau_r,nse_r)
    that:
      1. Reads current state from *_buf
      2. Computes new state (torch.compiled, out-of-place)
      3. Copies new state back to *_buf  (captured in CUDAGraph)
      4. Writes M to M_out_buf          (captured in CUDAGraph)
    """
    cb0_e = cb0.unsqueeze(0).expand(B,-1)
    cb1_e = cb1.unsqueeze(0).expand(B,-1)
    col_e = col_g.unsqueeze(0).expand(B,-1).contiguous()
    row_e = row_g.unsqueeze(0).expand(B,-1).contiguous()
    n_z0f = n_z0.to(DTYPE); n_z1f = n_z1.to(DTYPE)

    # ── Compiled compute (out-of-place, full kernel fusion) ───────────────────
    @torch.compile(mode='default', dynamic=False)
    def _compute(s, base, mid, phi, streak, wave_z,
                 rnd0, rnd1, fires, cx_z0, cx_z1, cy,
                 rnd_w0, rnd_w1, cau_r, nse_r):

        # Main Metropolis
        nb0 = nb[cb0]; ns0 = s[:,nb0].sum(-1); dE0 = 2.*s[:,cb0]*ns0
        acc0 = (dE0<=0)|(rnd0<torch.exp(torch.clamp(-dE0/T_FIXED,-20.,20.)))
        s = s.scatter(1,cb0_e,torch.where(acc0,-s[:,cb0],s[:,cb0]))
        nb1 = nb[cb1]; ns1 = s[:,nb1].sum(-1); dE1 = 2.*s[:,cb1]*ns1
        acc1 = (dE1<=0)|(rnd1<torch.exp(torch.clamp(-dE1/T_FIXED,-20.,20.)))
        s = s.scatter(1,cb1_e,torch.where(acc1,-s[:,cb1],s[:,cb1]))

        # Baseline + streak
        base   = base + BETA_BASE*(s-base)
        same   = (s>0)==(base>0)
        streak = torch.where(same, streak+1, torch.zeros_like(streak))
        mid    = mid*MID_DECAY

        # Wave
        cx  = torch.where(wave_z==0, cx_z0, cx_z1)
        dx  = torch.minimum((col_e-cx.unsqueeze(1)).abs(),
                            L-(col_e-cx.unsqueeze(1)).abs())
        dy  = torch.minimum((row_e-cy.unsqueeze(1)).abs(),
                            L-(row_e-cy.unsqueeze(1)).abs())
        in_wave = (dx+dy<=r_w)
        fires_e = fires.unsqueeze(1).expand(-1,N)
        hit_f   = (in_wave&fires_e).to(DTYPE)
        he_sign = torch.where(wave_z==0,1.,-1.).to(DTYPE)
        h_ext   = hit_f*he_sign.unsqueeze(1)*EXT_FIELD

        s_w = s
        for d in range(WAVE_DUR):
            ns0w = s_w[:,nb[cb0]].sum(-1); h0=h_ext[:,cb0]
            dE0w = 2.*s_w[:,cb0]*ns0w-2.*h0*s_w[:,cb0]
            ac0w = (dE0w<=0)|(rnd_w0[d*B:(d+1)*B]<
                              torch.exp(torch.clamp(-dE0w/T_FIXED,-20.,20.)))
            s_w  = s_w.scatter(1,cb0_e,torch.where(ac0w,-s_w[:,cb0],s_w[:,cb0]))
            ns1w = s_w[:,nb[cb1]].sum(-1); h1=h_ext[:,cb1]
            dE1w = 2.*s_w[:,cb1]*ns1w-2.*h1*s_w[:,cb1]
            ac1w = (dE1w<=0)|(rnd_w1[d*B:(d+1)*B]<
                              torch.exp(torch.clamp(-dE1w/T_FIXED,-20.,20.)))
            s_w  = s_w.scatter(1,cb1_e,torch.where(ac1w,-s_w[:,cb1],s_w[:,cb1]))
        s = torch.where(fires_e, s_w, s)

        # mid
        dev   = s-base
        wz_e  = (wave_z==0).unsqueeze(1).expand(-1,N)
        z0_e  = z0f.bool().unsqueeze(0).expand(B,-1)
        sig   = torch.where(z0_e==wz_e, dev, -dev)
        cau   = cau_r<P_ten.unsqueeze(1)
        nse   = nse_r*(dev.std(dim=1,keepdim=True).clamp(min=0.01)+0.5)
        mid   = mid + FA*torch.where(cau,sig,nse)*hit_f
        wave_z = torch.where(fires, 1-wave_z, wave_z)

        # Gate + phi
        gate   = streak>=SS
        phi    = phi + FA*torch.where(gate, mid-phi, torch.zeros_like(phi))
        streak = torch.where(gate, torch.zeros_like(streak), streak)
        phi    = phi*FIELD_DECAY

        M = (phi*z0f).sum(1)/n_z0f - (phi*z1f).sum(1)/n_z1f
        return s, base, mid, phi, streak, wave_z, M

    # ── Wrapper that writes results back to fixed buffers ─────────────────────
    # This is what CUDAGraph captures: _compute + copy_ ops
    def step_capture(rnd0, rnd1, fires, cx_z0, cx_z1, cy,
                     rnd_w0, rnd_w1, cau_r, nse_r):
        s_n, b_n, m_n, p_n, st_n, wz_n, M = _compute(
            s_buf, base_buf, mid_buf, phi_buf, streak_buf, wave_z_buf,
            rnd0, rnd1, fires, cx_z0, cx_z1, cy,
            rnd_w0, rnd_w1, cau_r, nse_r)
        # In-place copy new state back to fixed buffers (captured in graph)
        s_buf.copy_(s_n)
        base_buf.copy_(b_n)
        mid_buf.copy_(m_n)
        phi_buf.copy_(p_n)
        streak_buf.copy_(st_n)
        wave_z_buf.copy_(wz_n)
        M_out_buf.copy_(M)

    return step_capture


# == Main simulation with CUDAGraph loop ======================================

def run_batch_gpu_v3(
    L: int,
    P_causal_list: List[float],
    seed_list: List[int],
    nsteps: int,
    r_w: int = RW,
    verbose: bool = True,
) -> List[dict]:
    B = len(seed_list); N = L*L; wp = _wave_prob(L)

    # Geometry
    idx   = torch.arange(N, device=DEVICE)
    row_g = idx//L; col_g = idx%L
    nb    = torch.stack([((row_g-1)%L)*L+col_g, ((row_g+1)%L)*L+col_g,
                         row_g*L+(col_g-1)%L,   row_g*L+(col_g+1)%L], dim=1)
    par   = (row_g+col_g)%2
    cb0   = idx[par==0].contiguous(); cb1 = idx[par==1].contiguous()
    z0f   = (col_g<L//2).to(DTYPE); z1f = (col_g>=L//2).to(DTYPE)
    n_z0  = z0f.sum().long(); n_z1 = z1f.sum().long()

    P_ten = torch.tensor(P_causal_list, device=DEVICE, dtype=DTYPE)

    # ── Fixed state buffers ────────────────────────────────────────────────────
    s_buf      = torch.zeros(B, N, device=DEVICE, dtype=DTYPE)
    for b, seed in enumerate(seed_list):
        rng = np.random.RandomState(seed)
        s_buf[b] = torch.from_numpy(
            rng.choice([-1.,1.],N).astype(np.float32)).to(DTYPE).to(DEVICE)
    base_buf   = s_buf.clone() * 0.1
    mid_buf    = torch.zeros(B, N, device=DEVICE, dtype=DTYPE)
    phi_buf    = torch.zeros(B, N, device=DEVICE, dtype=DTYPE)
    streak_buf = torch.zeros(B, N, device=DEVICE, dtype=torch.int32)
    wave_z_buf = torch.zeros(B,   device=DEVICE, dtype=torch.int32)
    M_out_buf  = torch.zeros(B,   device=DEVICE, dtype=DTYPE)

    # ── Random buffers (filled before each replay) ────────────────────────────
    rnd0_buf   = torch.empty(B, len(cb0), device=DEVICE, dtype=DTYPE)
    rnd1_buf   = torch.empty(B, len(cb1), device=DEVICE, dtype=DTYPE)
    rnd_w0_buf = torch.empty(WAVE_DUR*B, len(cb0), device=DEVICE, dtype=DTYPE)
    rnd_w1_buf = torch.empty(WAVE_DUR*B, len(cb1), device=DEVICE, dtype=DTYPE)
    cau_r_buf  = torch.empty(B, N, device=DEVICE, dtype=DTYPE)
    nse_r_buf  = torch.empty(B, N, device=DEVICE, dtype=DTYPE)

    CHUNK = 5000
    fires_c  = torch.empty(CHUNK, B, device=DEVICE, dtype=torch.bool)
    cx_z0_c  = torch.empty(CHUNK, B, device=DEVICE, dtype=torch.long)
    cx_z1_c  = torch.empty(CHUNK, B, device=DEVICE, dtype=torch.long)
    cy_c     = torch.empty(CHUNK, B, device=DEVICE, dtype=torch.long)
    cptr     = CHUNK

    def _refill_chunk():
        nonlocal cptr
        fires_c.copy_(torch.rand(CHUNK, B, device=DEVICE) < wp)
        torch.randint(0,   L//2, (CHUNK,B), device=DEVICE, out=cx_z0_c)
        torch.randint(L//2, L,  (CHUNK,B), device=DEVICE, out=cx_z1_c)
        torch.randint(0,   L,   (CHUNK,B), device=DEVICE, out=cy_c)
        cptr = 0

    def _fill_randoms_and_get_wave(step_t):
        nonlocal cptr
        if cptr >= CHUNK:
            _refill_chunk()
        rnd0_buf.uniform_(); rnd1_buf.uniform_()
        rnd_w0_buf.uniform_(); rnd_w1_buf.uniform_()
        cau_r_buf.uniform_(); nse_r_buf.normal_()
        fires  = fires_c[cptr]
        cx_z0  = cx_z0_c[cptr]
        cx_z1  = cx_z1_c[cptr]
        cy     = cy_c[cptr]
        cptr += 1
        return fires, cx_z0, cx_z1, cy

    # Build compiled step wrapper
    step_fn = build_step_for_graph(
        nb, cb0, cb1, col_g, row_g, z0f, z1f, n_z0, n_z1, L, B, N, r_w,
        s_buf, base_buf, mid_buf, phi_buf, streak_buf, wave_z_buf,
        M_out_buf, P_ten)

    # ── Warm up: 3 eager steps to compile kernels + stabilise allocations ─────
    if verbose:
        print(f'    Warming up (compile + 3 eager steps)...', flush=True)
    with torch.no_grad():
        for _ in range(3):
            fires, cx_z0, cx_z1, cy = _fill_randoms_and_get_wave(0)
            step_fn(rnd0_buf, rnd1_buf, fires, cx_z0, cx_z1, cy,
                    rnd_w0_buf, rnd_w1_buf, cau_r_buf, nse_r_buf)
    torch.cuda.synchronize()

    # ── Capture CUDA graph ────────────────────────────────────────────────────
    if verbose:
        print(f'    Capturing CUDA graph...', flush=True)

    # Graph capture requires fixed inputs -- use current chunk values
    # (randoms will be filled before each replay; these are just placeholder shapes)
    fires_g = fires_c[0]
    cx_z0_g = cx_z0_c[0]
    cx_z1_g = cx_z1_c[0]
    cy_g    = cy_c[0]

    g = torch.cuda.CUDAGraph()
    with torch.no_grad():
        with torch.cuda.graph(g):
            step_fn(rnd0_buf, rnd1_buf, fires_g, cx_z0_g, cx_z1_g, cy_g,
                    rnd_w0_buf, rnd_w1_buf, cau_r_buf, nse_r_buf)

    if verbose:
        print(f'    Graph captured. Running {nsteps} steps...', flush=True)

    # ── GPU accumulation buffer for M ─────────────────────────────────────────
    ss0    = int(nsteps * SS_FRAC)
    n_data = nsteps - ss0
    M_hist = torch.zeros(n_data, B, device=DEVICE, dtype=DTYPE)
    m_idx  = 0

    # ── Main loop: fill randoms + g.replay() + async M copy ───────────────────
    with torch.no_grad():
        for t in range(nsteps):
            # Fill per-step randoms (async CUDA kernels on default stream)
            if cptr >= CHUNK:
                _refill_chunk()
            fires_c_t = fires_c[cptr]
            cx_z0_c_t = cx_z0_c[cptr]
            cx_z1_c_t = cx_z1_c[cptr]
            cy_c_t    = cy_c[cptr]
            cptr += 1

            rnd0_buf.uniform_(); rnd1_buf.uniform_()
            rnd_w0_buf.uniform_(); rnd_w1_buf.uniform_()
            cau_r_buf.uniform_(); nse_r_buf.normal_()

            # Copy this step's wave coords into the graph's fixed input slots
            fires_g.copy_(fires_c_t, non_blocking=True)
            cx_z0_g.copy_(cx_z0_c_t, non_blocking=True)
            cx_z1_g.copy_(cx_z1_c_t, non_blocking=True)
            cy_g.copy_(cy_c_t, non_blocking=True)

            # Replay the captured graph (async -- no Python wait)
            g.replay()

            # Async accumulate M into GPU buffer (no sync)
            if t >= ss0 and m_idx < n_data:
                M_hist[m_idx].copy_(M_out_buf, non_blocking=True)
                m_idx += 1

            if verbose and (t+1) % 10_000 == 0:
                print(f'    step {t+1}/{nsteps}', flush=True)

    # Single sync + transfer at end
    torch.cuda.synchronize()
    M_stack = M_hist[:m_idx].cpu().numpy()

    results = []
    for b in range(B):
        M_arr = M_stack[:, b]
        absM  = float(np.mean(np.abs(M_arr)))
        M1    = float(np.mean(M_arr))
        M2    = float(np.mean(M_arr**2))
        M4    = float(np.mean(M_arr**4))
        chi   = float(M2 - M1**2)
        U4    = float(1. - M4/(3.*M2**2)) if M2 > 1e-12 else float('nan')
        lags, acf = _compute_autocorr(M_arr.tolist())
        results.append(dict(absM=absM, M1=M1, M2=M2, M4=M4,
                            chi=chi, U4=U4, lags=lags, acf=acf,
                            n_data=len(M_arr)))
    return results


# == Helpers ===================================================================

def _compute_autocorr(Mseries, max_lag=None):
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

compute_autocorr_absM = _compute_autocorr

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
        print(f'  GPU:   {torch.cuda.get_device_name(0)}')
        print(f'  VRAM:  {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB')
    print(f'  dtype: {DTYPE}  |  strategy: CUDAGraph + mode=default\n')

    # v1 baseline (no graph, mode='default') -- quick estimate from Paper74
    print('v1 baseline (Paper74/75, mode=default, float32, no graph):')
    print('  L=80, B=8: ~75k steps/min  (from paper run timing)')
    print()

    for L_bm, B_bm in [(80, 8), (80, 16), (160, 8)]:
        print(f'v3 CUDAGraph:  L={L_bm}, B={B_bm}')
        P_list = [0.001] * B_bm
        s_list = list(range(B_bm))
        N_BENCH = 30_000

        # Full run includes: compile + warmup + graph capture + N_BENCH steps
        t0 = time.time()
        run_batch_gpu_v3(L_bm, P_list, s_list, N_BENCH, verbose=True)
        torch.cuda.synchronize()
        dt_total = time.time() - t0
        print(f'  Total (incl compile+capture): {dt_total:.1f}s')

        # Second run: only graph replay (compile already cached)
        t0 = time.time()
        run_batch_gpu_v3(L_bm, P_list, s_list, N_BENCH, verbose=False)
        torch.cuda.synchronize()
        dt = time.time() - t0
        sps = N_BENCH / dt
        print(f'  Pure throughput: {sps:.0f} steps/sec  ({sps*60/1000:.1f}k steps/min)')
        print()

    print('Benchmark done.')
