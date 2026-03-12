"""
vcml_gpu_v4.py  --  Pure CUDA C backend for VCML simulation

Architecture:
  - All kernels in vcml_cuda/vcml_step.cu (cuRAND device API, no Python RNG)
  - Single pybind11 function vcml_full_step() captures into one CUDA graph
  - Python loop is pure g.replay() + async copy  ->  <5 μs overhead/step
  - Expected: 350-500k steps/min at L=80, B=8 on RTX 3060

Usage (same API as vcml_gpu_v3.run_batch_gpu_v3):
    from vcml_gpu_v4 import run_batch_gpu_v4
    results = run_batch_gpu_v4(L=80, P_causal_list=[0.005], seed_list=list(range(8)), nsteps=300_000)
"""

import os, time, math, warnings
import numpy as np
import torch
import torch.utils.cpp_extension as cpp_ext
from pathlib import Path

# ── compile / load extension ──────────────────────────────────────────
_CU_DIR  = Path(__file__).parent / 'vcml_cuda'
_CU_SRC  = str(_CU_DIR / 'vcml_step.cu')
_BUILD   = str(_CU_DIR / '_build')

def _find_msvc_bin():
    """Return the x64 MSVC bin dir if cl.exe is not already in PATH."""
    import shutil
    if shutil.which('cl'):
        return None
    for vs_root in [r'C:\Program Files\Microsoft Visual Studio',
                    r'C:\Program Files (x86)\Microsoft Visual Studio']:
        vs_path = Path(vs_root)
        if not vs_path.exists():
            continue
        for cl in sorted(vs_path.glob('*/*/VC/Tools/MSVC/*/bin/Hostx64/x64/cl.exe'),
                         reverse=True):
            return str(cl.parent)
    return None

def _find_cuda_include():
    """Return extra CUDA include paths needed when system CUDA headers are missing.
    Checks pip-installed nvidia-cuda-runtime-cu12 package as fallback."""
    import sys, glob
    for pattern in [
        str(Path(sys.prefix) / 'Lib/site-packages/nvidia/cuda_runtime/include'),
        str(Path(sys.prefix) / 'lib/python*/site-packages/nvidia/cuda_runtime/include'),
    ]:
        for p in glob.glob(pattern):
            if Path(p, 'cuda_runtime.h').exists():
                return p
    return None

def _load_ext():
    os.makedirs(_BUILD, exist_ok=True)
    msvc = _find_msvc_bin()
    if msvc:
        os.environ['PATH'] = msvc + os.pathsep + os.environ.get('PATH', '')
    extra_inc = []
    cuda_inc = _find_cuda_include()
    if cuda_inc:
        extra_inc = [f'-I{cuda_inc}']
    # sm_86 = RTX 3060/3080/3090; change to sm_89 for RTX 40-series
    return cpp_ext.load(
        name='vcml_cuda_ext',
        sources=[_CU_SRC],
        extra_cuda_cflags=['-O3', '-arch=sm_86', '--use_fast_math',
                           '-lcurand', '--expt-relaxed-constexpr'] + extra_inc,
        build_directory=_BUILD,
        verbose=False,
    )

# lazy-load on first call
_ext = None
def _get_ext():
    global _ext
    if _ext is None:
        print('[vcml_gpu_v4] Compiling CUDA extension … ', end='', flush=True)
        t0 = time.time()
        _ext = _load_ext()
        print(f'done ({time.time()-t0:.1f}s)')
    return _ext

# ── geometry helpers ──────────────────────────────────────────────────
def _build_geometry(L: int, device: torch.device):
    """Return (nb4, cb0, cb1, col_g, row_g, z0f) as CUDA int32/float32 tensors."""
    N    = L * L
    half = L // 2
    rows = torch.arange(L, dtype=torch.int32).repeat_interleave(L)   # [N]
    cols = torch.arange(L, dtype=torch.int32).repeat(L)              # [N]
    # 4-neighbours: up, down, left, right (toroidal)
    nb4 = torch.stack([
        ((rows - 1) % L) * L + cols,
        ((rows + 1) % L) * L + cols,
        rows * L + (cols - 1) % L,
        rows * L + (cols + 1) % L,
    ], dim=1).contiguous()                                            # [N, 4]
    # checkerboard: (row+col) % 2
    parity = (rows + cols) % 2
    cb0 = torch.where(parity == 0)[0].to(torch.int32)                # [n_cb0]
    cb1 = torch.where(parity == 1)[0].to(torch.int32)                # [n_cb1]
    # zone mask: zone-0 = left half (col < half)
    z0f = torch.where(cols < half,
                      torch.ones(N, dtype=torch.float32),
                      -torch.ones(N, dtype=torch.float32))           # [N]
    return (nb4.to(device),
            cb0.to(device),
            cb1.to(device),
            cols.to(device),
            rows.to(device),
            z0f.to(device))

# ── wave probability ──────────────────────────────────────────────────
def _wave_prob(L: int, L_base: int = 40, waves_per_step_base: float = 1.0) -> float:
    """Probabilistic wave firing: expected 1 wave/step at L=L_base, scaled by (L_base/L)^2."""
    return min(1.0, waves_per_step_base * (L_base / L) ** 2)

# ── autocorrelation ───────────────────────────────────────────────────
def _acf(x: np.ndarray, max_lag: int = 500) -> tuple[np.ndarray, np.ndarray]:
    x = x - x.mean()
    n = len(x)
    lags = np.arange(max_lag + 1)
    c0   = np.dot(x, x) / n
    if c0 < 1e-30:
        return lags, np.ones(max_lag + 1)
    acf  = np.array([np.dot(x[:n-k], x[k:]) / ((n - k) * c0) for k in lags])
    return lags, acf

def _tau_int(acf: np.ndarray, lags: np.ndarray) -> float:
    """Integrated autocorrelation time (truncated at first negative crossing)."""
    tau = 0.5
    for i in range(1, len(acf)):
        if acf[i] < 0:
            break
        tau += acf[i]
    return float(tau)

# ── single-P single-seed runner (graph capture + replay) ──────────────
def _run_one(ext, L: int, P_causal: float, seed: int, nsteps: int,
             r_w: int, device: torch.device, geom) -> dict:
    """Run one seed × one P_causal value. Returns stats dict."""
    N    = L * L
    B    = 1   # single seed per graph instance
    half = L // 2
    n_z0 = N // 2

    nb4, cb0, cb1, col_g, row_g, z0f = geom

    # RNG state size from extension
    rng_bytes = ext.rng_state_size()

    # ── allocate persistent state tensors ────────────────────────────
    s       = (2 * torch.randint(0, 2, (B * N,), device=device) - 1).float()
    base    = s.clone()
    mid     = torch.zeros(B * N, device=device)
    phi     = torch.zeros(B * N, device=device)
    streak  = torch.zeros(B * N, device=device, dtype=torch.int32)
    wave_z  = torch.zeros(B, device=device, dtype=torch.int32)
    M_out   = torch.zeros(B, device=device)

    # scratch
    s_wave  = s.clone()
    h_ext   = torch.zeros(B * N, device=device)
    fires   = torch.zeros(B, device=device, dtype=torch.int8)
    cx_z0   = torch.zeros(B, device=device, dtype=torch.int32)
    cx_z1   = torch.full((B,), half, device=device, dtype=torch.int32)
    cy      = torch.zeros(B, device=device, dtype=torch.int32)
    M_accum = torch.zeros(B, device=device)
    dev_sum = torch.zeros(B, device=device)
    dev_sq  = torch.zeros(B, device=device)
    dev_std = torch.ones(B, device=device)

    # RNG buffers
    cell_rng  = torch.zeros(B * N * rng_bytes, device=device, dtype=torch.uint8)
    wave_rng_t = torch.zeros(B * rng_bytes, device=device, dtype=torch.uint8)

    # geometry (must be on device already)
    P_ten = torch.full((B,), P_causal, device=device)

    wp     = float(_wave_prob(L))
    n_cb0  = cb0.shape[0]
    n_cb1  = cb1.shape[0]

    # ── initialise RNG states on device ──────────────────────────────
    ext.init_rng_states(cell_rng, wave_rng_t, int(seed * 1_000_003), B * N, B)

    # ── warmup (no graph): 1000 steps to reach steady state ──────────
    for _ in range(1000):
        ext.vcml_full_step(
            s, base, mid, phi, streak, wave_z, M_out,
            s_wave, h_ext, fires, cx_z0, cx_z1, cy,
            M_accum, dev_sum, dev_sq, dev_std,
            cell_rng, wave_rng_t,
            nb4, cb0, cb1, col_g, row_g, z0f,
            P_ten, wp, r_w, L, B, N, n_cb0, n_cb1, n_z0, h_field)
    torch.cuda.synchronize()

    # ── CUDA graph capture ────────────────────────────────────────────
    g = torch.cuda.CUDAGraph()
    # capture output buffer
    M_out_buf = torch.zeros(B, device=device)
    with torch.cuda.graph(g):
        ext.vcml_full_step(
            s, base, mid, phi, streak, wave_z, M_out,
            s_wave, h_ext, fires, cx_z0, cx_z1, cy,
            M_accum, dev_sum, dev_sq, dev_std,
            cell_rng, wave_rng_t,
            nb4, cb0, cb1, col_g, row_g, z0f,
            P_ten, wp, r_w, L, B, N, n_cb0, n_cb1, n_z0, h_field)
        M_out_buf.copy_(M_out)

    # ── measurement run ───────────────────────────────────────────────
    SS_BURN = 8 * 2000   # burn-in: 2000 × SS steps after warmup
    M_hist  = torch.zeros(nsteps - SS_BURN, device=device)
    for t in range(nsteps):
        g.replay()
        if t >= SS_BURN:
            M_hist[t - SS_BURN].copy_(M_out_buf[0], non_blocking=True)
    torch.cuda.synchronize()

    # ── statistics ────────────────────────────────────────────────────
    M_np  = M_hist.cpu().numpy()
    absM  = float(np.mean(np.abs(M_np)))
    M1    = float(np.mean(M_np))
    M2    = float(np.mean(M_np ** 2))
    M4    = float(np.mean(M_np ** 4))
    chi   = float(np.var(M_np))
    U4    = 1.0 - M4 / (3.0 * M2 ** 2) if M2 > 1e-14 else 0.0

    lags, acf = _acf(M_np, max_lag=500)
    tau_corr  = _tau_int(acf, lags)

    return {
        'absM': absM, 'M1': M1, 'M2': M2, 'M4': M4,
        'chi': chi, 'U4': U4,
        'lags': lags.tolist(), 'acf': acf.tolist(),
        'tau_corr': tau_corr,
        'n_data': len(M_np),
    }

# ── batched runner (multiple seeds × same P, B in parallel) ───────────
def _run_batch(ext, L: int, P_causal: float, seed_list: list[int],
               nsteps: int, r_w: int, device: torch.device, geom,
               h_field: float = 0.0) -> list[dict]:
    """Run all seeds for one P_causal value, B seeds batched in one graph."""
    N    = L * L
    B    = len(seed_list)
    half = L // 2
    n_z0 = N // 2

    nb4, cb0, cb1, col_g, row_g, z0f = geom
    rng_bytes = ext.rng_state_size()

    # ── allocate [B*N] batched state ─────────────────────────────────
    s       = (2 * torch.randint(0, 2, (B * N,), device=device) - 1).float()
    base    = s.clone()
    mid     = torch.zeros(B * N, device=device)
    phi     = torch.zeros(B * N, device=device)
    streak  = torch.zeros(B * N, device=device, dtype=torch.int32)
    wave_z  = torch.zeros(B, device=device, dtype=torch.int32)
    M_out   = torch.zeros(B, device=device)
    s_wave  = s.clone()
    h_ext   = torch.zeros(B * N, device=device)
    fires   = torch.zeros(B, device=device, dtype=torch.int8)
    cx_z0   = torch.zeros(B, device=device, dtype=torch.int32)
    cx_z1   = torch.full((B,), half, device=device, dtype=torch.int32)
    cy      = torch.zeros(B, device=device, dtype=torch.int32)
    M_accum = torch.zeros(B, device=device)
    dev_sum = torch.zeros(B, device=device)
    dev_sq  = torch.zeros(B, device=device)
    dev_std = torch.ones(B, device=device)
    cell_rng   = torch.zeros(B * N * rng_bytes, device=device, dtype=torch.uint8)
    wave_rng_t = torch.zeros(B * rng_bytes, device=device, dtype=torch.uint8)
    P_ten   = torch.full((B,), P_causal, device=device)

    wp    = float(_wave_prob(L))
    n_cb0 = cb0.shape[0]
    n_cb1 = cb1.shape[0]

    # init all B RNG states with different seeds
    for bi, seed in enumerate(seed_list):
        # init seed bi using a slice of cell_rng
        cell_slice = cell_rng[bi * N * rng_bytes : (bi + 1) * N * rng_bytes]
        wave_slice = wave_rng_t[bi * rng_bytes : (bi + 1) * rng_bytes]
        ext.init_rng_states(cell_slice, wave_slice, int(seed * 1_000_003), N, 1)
    torch.cuda.synchronize()

    # warmup
    for _ in range(1000):
        ext.vcml_full_step(
            s, base, mid, phi, streak, wave_z, M_out,
            s_wave, h_ext, fires, cx_z0, cx_z1, cy,
            M_accum, dev_sum, dev_sq, dev_std,
            cell_rng, wave_rng_t,
            nb4, cb0, cb1, col_g, row_g, z0f,
            P_ten, wp, r_w, L, B, N, n_cb0, n_cb1, n_z0, h_field)
    torch.cuda.synchronize()

    # graph capture
    g         = torch.cuda.CUDAGraph()
    M_out_buf = torch.zeros(B, device=device)
    with torch.cuda.graph(g):
        ext.vcml_full_step(
            s, base, mid, phi, streak, wave_z, M_out,
            s_wave, h_ext, fires, cx_z0, cx_z1, cy,
            M_accum, dev_sum, dev_sq, dev_std,
            cell_rng, wave_rng_t,
            nb4, cb0, cb1, col_g, row_g, z0f,
            P_ten, wp, r_w, L, B, N, n_cb0, n_cb1, n_z0, h_field)
        M_out_buf.copy_(M_out)

    # measurement
    SS_BURN = 8 * 2000
    n_meas  = nsteps - SS_BURN
    M_hist  = torch.zeros(n_meas, B, device=device)
    for t in range(nsteps):
        g.replay()
        if t >= SS_BURN:
            M_hist[t - SS_BURN].copy_(M_out_buf, non_blocking=True)
    torch.cuda.synchronize()

    # per-seed statistics
    results = []
    M_all   = M_hist.cpu().numpy()          # [n_meas, B]
    for bi in range(B):
        M_np  = M_all[:, bi]
        absM  = float(np.mean(np.abs(M_np)))
        M1    = float(np.mean(M_np))
        M2    = float(np.mean(M_np ** 2))
        M4    = float(np.mean(M_np ** 4))
        chi   = float(np.var(M_np))
        U4    = 1.0 - M4 / (3.0 * M2 ** 2) if M2 > 1e-14 else 0.0
        lags, acf = _acf(M_np, max_lag=500)
        tau_corr  = _tau_int(acf, lags)
        results.append({
            'absM': absM, 'M1': M1, 'M2': M2, 'M4': M4,
            'chi': chi, 'U4': U4,
            'lags': lags.tolist(), 'acf': acf.tolist(),
            'tau_corr': tau_corr,
            'n_data': len(M_np),
        })
    return results

# ── public API ────────────────────────────────────────────────────────
def run_batch_gpu_v4(L: int, P_causal_list: list[float], seed_list: list[int],
                     nsteps: int, r_w: int = 5, device_str: str = 'cuda',
                     h_field: float = 0.0) -> list[dict]:
    """
    Drop-in replacement for run_batch_gpu_v3.

    Parameters
    ----------
    L              : lattice side length
    P_causal_list  : list of P_causal values (one result dict per value × seed)
    seed_list      : list of integer seeds (all run in same batch for each P)
    nsteps         : total steps per seed (includes burn-in of ~16k)
    r_w            : wave radius (default 5)
    device_str     : 'cuda' or 'cuda:N'

    Returns
    -------
    List of result dicts (same format as v3), one per (P_causal, seed) pair,
    ordered as [(P0,s0),(P0,s1),...,(P1,s0),...].
    """
    ext    = _get_ext()
    device = torch.device(device_str)
    geom   = _build_geometry(L, device)

    all_results = []
    for P in P_causal_list:
        t0 = time.time()
        batch = _run_batch(ext, L, P, seed_list, nsteps, r_w, device, geom, h_field)
        dt    = time.time() - t0
        rate  = nsteps * len(seed_list) / dt / 1000
        print(f'  [v4] L={L} P={P:.4f} {len(seed_list)} seeds {nsteps}steps  '
              f'{dt:.0f}s  {rate:.0f}k seed-steps/min')
        all_results.extend(batch)

    return all_results

# ── benchmark ─────────────────────────────────────────────────────────
def benchmark(L: int = 80, B: int = 8, nsteps: int = 10_000, h_field: float = 0.0):
    """Quick benchmark: steps/min and seed-steps/min."""
    ext    = _get_ext()
    device = torch.device('cuda')
    geom   = _build_geometry(L, device)

    N    = L * L
    half = L // 2
    n_z0 = N // 2
    rng_bytes = ext.rng_state_size()
    nb4, cb0, cb1, col_g, row_g, z0f = geom
    n_cb0 = cb0.shape[0]; n_cb1 = cb1.shape[0]
    P_causal = 0.010
    wp = float(_wave_prob(L))
    r_w = 5

    s       = (2 * torch.randint(0, 2, (B * N,), device=device) - 1).float()
    base    = s.clone(); mid = torch.zeros_like(s)
    phi     = torch.zeros_like(s)
    streak  = torch.zeros(B * N, device=device, dtype=torch.int32)
    wave_z  = torch.zeros(B, device=device, dtype=torch.int32)
    M_out   = torch.zeros(B, device=device)
    s_wave  = s.clone(); h_ext = torch.zeros_like(s)
    fires   = torch.zeros(B, device=device, dtype=torch.int8)
    cx_z0   = torch.zeros(B, device=device, dtype=torch.int32)
    cx_z1   = torch.full((B,), half, device=device, dtype=torch.int32)
    cy      = torch.zeros(B, device=device, dtype=torch.int32)
    M_accum = torch.zeros(B, device=device)
    dev_sum = torch.zeros(B, device=device); dev_sq = dev_sum.clone()
    dev_std = torch.ones(B, device=device)
    cell_rng   = torch.zeros(B * N * rng_bytes, device=device, dtype=torch.uint8)
    wave_rng_t = torch.zeros(B * rng_bytes, device=device, dtype=torch.uint8)
    P_ten   = torch.full((B,), P_causal, device=device)

    ext.init_rng_states(cell_rng, wave_rng_t, 42, B * N, B)

    # warmup
    for _ in range(200):
        ext.vcml_full_step(s, base, mid, phi, streak, wave_z, M_out,
                            s_wave, h_ext, fires, cx_z0, cx_z1, cy,
                            M_accum, dev_sum, dev_sq, dev_std,
                            cell_rng, wave_rng_t,
                            nb4, cb0, cb1, col_g, row_g, z0f,
                            P_ten, wp, r_w, L, B, N, n_cb0, n_cb1, n_z0, h_field)
    torch.cuda.synchronize()

    g         = torch.cuda.CUDAGraph()
    M_out_buf = torch.zeros(B, device=device)
    with torch.cuda.graph(g):
        ext.vcml_full_step(s, base, mid, phi, streak, wave_z, M_out,
                            s_wave, h_ext, fires, cx_z0, cx_z1, cy,
                            M_accum, dev_sum, dev_sq, dev_std,
                            cell_rng, wave_rng_t,
                            nb4, cb0, cb1, col_g, row_g, z0f,
                            P_ten, wp, r_w, L, B, N, n_cb0, n_cb1, n_z0, h_field)
        M_out_buf.copy_(M_out)

    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(nsteps):
        g.replay()
    torch.cuda.synchronize()
    dt   = time.time() - t0
    rate = nsteps / dt * 60
    seed_rate = rate * B
    print(f'[v4 benchmark] L={L} B={B} {nsteps} steps: '
          f'{rate:.0f} steps/min  |  {seed_rate/1000:.0f}k seed-steps/min')
    return rate

if __name__ == '__main__':
    benchmark(L=80, B=8, nsteps=20_000)
    benchmark(L=80, B=16, nsteps=20_000)
    benchmark(L=160, B=8, nsteps=10_000)
