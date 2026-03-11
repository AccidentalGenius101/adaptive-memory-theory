"""
vcml_gpu_v5.py -- Persistent cooperative-kernel VCML runner.

Optimisations over v4:
  * Single persistent kernel: one cudaLaunchCooperativeKernel call covers
    all nsteps with zero kernel-scheduling overhead between phases.
  * FP16 phi field (half bandwidth for the memory field).
  * Block-level smem reduction for dev_std.
  * No Python step-loop at all: returns M_trace [nsteps, B] in one shot.

API:
    run_batch_gpu_v5(L, P_causal_list, seed_list, nsteps, warmup=2000)
    benchmark(L, B, nsteps)
"""

import os, sys, time
import torch
from torch.utils import cpp_extension as cpp_ext

# ── locate v5 source ──────────────────────────────────────────────────
_HERE    = os.path.dirname(os.path.abspath(__file__))
_CU_V5   = os.path.join(_HERE, 'vcml_cuda', 'vcml_step_v5.cu')
_EXT_V5  = None   # cached module

# ── compile-time constants (must match vcml_step_v5.cu) ──────────────
_BLOCK      = 256
_SS         = 8
_FA         = 0.30
_FIELD_DECAY= 0.999
_WAVE_DUR   = 5
_R_W        = 5
_L_BASE     = 40    # reference L for wave probability scaling

def _load_ext():
    msvc = r'C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC\14.42.34433\bin\Hostx64\x64'
    if msvc not in os.environ.get('PATH', ''):
        os.environ['PATH'] = msvc + ';' + os.environ.get('PATH', '')
    scripts = r'C:\Users\briel\AppData\Local\Programs\Python\Python312\Scripts'
    if scripts not in os.environ.get('PATH', ''):
        os.environ['PATH'] = scripts + ';' + os.environ.get('PATH', '')

    print('[vcml_gpu_v5] Compiling CUDA extension…', end=' ', flush=True)
    ext = cpp_ext.load(
        name='vcml_cuda_v5_ext',
        sources=[_CU_V5],
        extra_cuda_cflags=[
            '-O3', '--use_fast_math', '-arch=sm_86',
            '-lcurand', '-lcuda',       # cooperative launch needs -lcuda
            '--expt-relaxed-constexpr',
            '-std=c++17',
        ],
        verbose=False,
    )
    print('done')
    return ext

def _get_ext():
    global _EXT_V5
    if _EXT_V5 is None:
        _EXT_V5 = _load_ext()
    return _EXT_V5


# ── geometry helpers ──────────────────────────────────────────────────
def _build_geometry(L, device):
    N    = L * L
    rows = torch.arange(L, device=device).repeat_interleave(L)
    cols = torch.arange(L, device=device).repeat(L)

    # neighbour indices (periodic)
    nb4 = torch.stack([
        ((rows - 1) % L) * L + cols,
        ((rows + 1) % L) * L + cols,
        rows * L + (cols - 1) % L,
        rows * L + (cols + 1) % L,
    ], dim=1).to(torch.int32).contiguous()   # [N, 4]

    # checkerboard sublattices
    parity = (rows + cols) % 2
    cb0    = torch.where(parity == 0)[0].to(torch.int32).contiguous()
    cb1    = torch.where(parity == 1)[0].to(torch.int32).contiguous()

    # zone membership: left half = zone-0 (+1), right half = zone-1 (-1)
    z0f = torch.where(cols < L // 2,
                      torch.ones(N, device=device),
                      -torch.ones(N, device=device)).float().contiguous()

    return nb4, cb0, cb1, cols.to(torch.int32).contiguous(), \
           rows.to(torch.int32).contiguous(), z0f


def _wave_prob(L):
    """Probabilistic wave firing: keep wave density constant across L."""
    return min(1.0, (_L_BASE / L) ** 2)


# ── main run function ─────────────────────────────────────────────────
def run_batch_gpu_v5(L, P_causal_list, seed_list, nsteps, warmup=2000):
    """
    Run VCML for `nsteps` steps for every (P, seed) combination.

    Parameters
    ----------
    L             : grid side length
    P_causal_list : list of float, one per seed group
    seed_list     : list of int,   one per seed group
    nsteps        : number of production steps
    warmup        : additional warmup steps (equilibration, not recorded)

    Returns
    -------
    M_trace : torch.Tensor [nsteps, B] on CPU, each column = one seed's M(t)
    """
    ext    = _get_ext()
    device = torch.device('cuda')
    N      = L * L
    B      = len(seed_list)
    r_w    = _R_W
    wp     = float(_wave_prob(L))

    nb4, cb0, cb1, col_g, row_g, z0f = _build_geometry(L, device)
    n_cb0 = int(cb0.shape[0])
    n_cb1 = int(cb1.shape[0])
    n_z0  = int((z0f > 0).sum().item())

    rng_bytes = ext.rng_state_size()

    # ── allocate state tensors ────────────────────────────────────────
    def zeros(*shape, dtype=torch.float32):
        return torch.zeros(*shape, dtype=dtype, device=device)

    s       = (torch.randint(0, 2, (B * N,), device=device).float() * 2 - 1)
    base    = s.clone()
    mid     = zeros(B * N)
    phi     = zeros(B * N, dtype=torch.float16)   # FP16!
    streak  = torch.zeros(B * N, dtype=torch.int32, device=device)
    wave_z  = torch.zeros(B, dtype=torch.int32, device=device)
    s_wave  = s.clone()
    h_ext   = zeros(B * N)
    fires   = torch.zeros(B, dtype=torch.int8, device=device)
    cx_z0   = torch.zeros(B, dtype=torch.int32, device=device)
    cx_z1   = torch.zeros(B, dtype=torch.int32, device=device)
    cy      = torch.zeros(B, dtype=torch.int32, device=device)
    M_accum = zeros(B)
    dev_sum = zeros(B)
    dev_sq  = zeros(B)
    dev_std = torch.ones(B, device=device)

    cell_rng  = torch.zeros(B * N * rng_bytes, dtype=torch.uint8, device=device)
    wave_rng  = torch.zeros(B     * rng_bytes, dtype=torch.uint8, device=device)

    P_ten = torch.tensor(P_causal_list, dtype=torch.float32, device=device)

    # ── init RNG (one call per seed) ──────────────────────────────────
    for i, seed in enumerate(seed_list):
        # init each seed's slice independently
        seed_cell = cell_rng[i*N*rng_bytes : (i+1)*N*rng_bytes]
        seed_wave = wave_rng[i*rng_bytes   : (i+1)*rng_bytes]
        ext.init_rng_states(seed_cell, seed_wave, int(seed), N, 1)

    # ── warmup (not recorded) ─────────────────────────────────────────
    if warmup > 0:
        ext.vcml_run(
            s, base, mid, phi, streak, wave_z,
            s_wave, h_ext, fires, cx_z0, cx_z1, cy,
            M_accum, dev_sum, dev_sq, dev_std,
            cell_rng, wave_rng,
            nb4, cb0, cb1, col_g, row_g, z0f, P_ten,
            wp, r_w, L, B, N, n_cb0, n_cb1, n_z0, warmup)

    # ── production run ────────────────────────────────────────────────
    M_trace_gpu = ext.vcml_run(
        s, base, mid, phi, streak, wave_z,
        s_wave, h_ext, fires, cx_z0, cx_z1, cy,
        M_accum, dev_sum, dev_sq, dev_std,
        cell_rng, wave_rng,
        nb4, cb0, cb1, col_g, row_g, z0f, P_ten,
        wp, r_w, L, B, N, n_cb0, n_cb1, n_z0, nsteps)

    return M_trace_gpu.cpu()   # [nsteps, B]


# ── benchmark ─────────────────────────────────────────────────────────
def benchmark(L=80, B=8, nsteps=20_000, warmup=2000):
    ext    = _get_ext()
    device = torch.device('cuda')
    N      = L * L
    r_w    = _R_W
    wp     = float(_wave_prob(L))

    nb4, cb0, cb1, col_g, row_g, z0f = _build_geometry(L, device)
    n_cb0 = int(cb0.shape[0])
    n_cb1 = int(cb1.shape[0])
    n_z0  = int((z0f > 0).sum().item())
    rng_bytes = ext.rng_state_size()

    def zeros(*shape, dtype=torch.float32):
        return torch.zeros(*shape, dtype=dtype, device=device)

    s       = (torch.randint(0, 2, (B*N,), device=device).float()*2-1)
    base    = s.clone(); mid = zeros(B*N)
    phi     = zeros(B*N, dtype=torch.float16)
    streak  = torch.zeros(B*N, dtype=torch.int32, device=device)
    wave_z  = torch.zeros(B,   dtype=torch.int32, device=device)
    s_wave  = s.clone(); h_ext = zeros(B*N)
    fires   = torch.zeros(B, dtype=torch.int8, device=device)
    cx_z0   = torch.zeros(B, dtype=torch.int32, device=device)
    cx_z1   = torch.zeros(B, dtype=torch.int32, device=device)
    cy      = torch.zeros(B, dtype=torch.int32, device=device)
    M_accum = zeros(B); dev_sum = zeros(B); dev_sq = zeros(B)
    dev_std = torch.ones(B, device=device)
    cell_rng = torch.zeros(B*N*rng_bytes, dtype=torch.uint8, device=device)
    wave_rng = torch.zeros(B*rng_bytes,   dtype=torch.uint8, device=device)
    P_ten    = torch.full((B,), 0.01, dtype=torch.float32, device=device)

    ext.init_rng_states(cell_rng, wave_rng, 42, B*N, B)

    def _call(n):
        return ext.vcml_run(
            s, base, mid, phi, streak, wave_z,
            s_wave, h_ext, fires, cx_z0, cx_z1, cy,
            M_accum, dev_sum, dev_sq, dev_std,
            cell_rng, wave_rng,
            nb4, cb0, cb1, col_g, row_g, z0f, P_ten,
            wp, r_w, L, B, N, n_cb0, n_cb1, n_z0, n)

    _call(warmup)   # warmup
    torch.cuda.synchronize()
    t0 = time.time()
    _call(nsteps)
    torch.cuda.synchronize()
    dt   = time.time() - t0
    rate = nsteps / dt * 60
    print(f'[v5 benchmark] L={L} B={B} {nsteps} steps: '
          f'{rate/1e6:.3f}M steps/min  |  {rate*B/1e6:.2f}M seed-steps/min')
    return rate


if __name__ == '__main__':
    benchmark(L=80,  B=8,  nsteps=20_000)
    benchmark(L=80,  B=16, nsteps=20_000)
    benchmark(L=160, B=8,  nsteps=10_000)
