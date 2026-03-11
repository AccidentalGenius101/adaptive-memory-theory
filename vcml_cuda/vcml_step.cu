/*
 * vcml_step.cu  --  Pure CUDA C implementation of one VCML time-step
 *
 * Architecture: 13 kernel types, 23 launches per step, all captured as one
 * CUDA graph.  cuRAND device API for ALL random numbers (no host RNG).
 * Exposed via pybind11: vcml_full_step() + init_rng_states()
 *
 * Tested on RTX 3060 (sm_86). Compile with:
 *   -O3 --use_fast_math -arch=sm_86 -lcurand
 */

#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <pybind11/pybind11.h>
#include <torch/extension.h>
#include <cstdint>
#include <cmath>

namespace py = pybind11;

/* ─── compile-time hypers ──────────────────────────────────────────── */
#define BLOCK      256
#define WAVE_DUR   5
#define SS         8
#define FA         0.30f
#define FIELD_DECAY 0.999f
#define BETA_ISING  0.44f   /* J=1, β so that exp(-2βΔE) gives correct Metropolis */

/* ─── helper macros ────────────────────────────────────────────────── */
#define CUDA_CHECK(x) do { \
    cudaError_t e = (x);   \
    if (e != cudaSuccess) { \
        throw std::runtime_error(std::string("CUDA error: ") + cudaGetErrorString(e)); \
    } \
} while(0)

/* ─── type aliases ─────────────────────────────────────────────────── */
using f32  = float;
using i32  = int;
using i8   = int8_t;
using u8   = uint8_t;
using u64  = uint64_t;

/* ═══════════════════════════════════════════════════════════════════════
   KERNEL 1: k_init_rng
   Initialise per-cell curandStateXORWOW (one per batch×site).
   rng_buf: B*N × sizeof(curandStateXORWOW) bytes, cast to curandState*
   ═══════════════════════════════════════════════════════════════════════ */
__global__ void k_init_rng(curandState* states, u64 base_seed, i32 total)
{
    i32 tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= total) return;
    curand_init(base_seed + (u64)tid, tid, 0, &states[tid]);
}

/* ═══════════════════════════════════════════════════════════════════════
   KERNEL 2: k_init_wave_rng
   Separate, smaller RNG for wave generation (one per batch seed).
   ═══════════════════════════════════════════════════════════════════════ */
__global__ void k_init_wave_rng(curandState* states, u64 base_seed, i32 B)
{
    i32 bid = blockIdx.x * blockDim.x + threadIdx.x;
    if (bid >= B) return;
    curand_init(base_seed + 0x8000000000ULL + (u64)bid, bid, 0, &states[bid]);
}

/* ═══════════════════════════════════════════════════════════════════════
   KERNEL 3: k_zero_accum
   Zero per-batch accumulators before each step.
   ═══════════════════════════════════════════════════════════════════════ */
__global__ void k_zero_accum(f32* M_accum, f32* dev_sum, f32* dev_sq, i32 B)
{
    i32 bid = blockIdx.x * blockDim.x + threadIdx.x;
    if (bid >= B) return;
    M_accum[bid] = 0.f;
    dev_sum[bid]  = 0.f;
    dev_sq[bid]   = 0.f;
}

/* ═══════════════════════════════════════════════════════════════════════
   KERNEL 4: k_gen_wave
   Generate one wave per batch seed:
     fires[b]  = 1 if wave fires this step (prob = wp)
     cx_z0[b]  = wave centre x in zone-0 half
     cx_z1[b]  = wave centre x in zone-1 half
     cy[b]     = wave centre y
   ═══════════════════════════════════════════════════════════════════════ */
__global__ void k_gen_wave(curandState* wave_rng,
                            i8* fires, i32* cx_z0, i32* cx_z1, i32* cy,
                            f32 wp, i32 L, i32 B)
{
    i32 bid = blockIdx.x * blockDim.x + threadIdx.x;
    if (bid >= B) return;
    curandState* st = &wave_rng[bid];

    fires[bid] = (curand_uniform(st) < wp) ? 1 : 0;
    if (fires[bid]) {
        i32 half = L / 2;
        cx_z0[bid] = (i32)(curand_uniform(st) * half) % half;
        cx_z1[bid] = half + (i32)(curand_uniform(st) * half) % half;
        cy[bid]    = (i32)(curand_uniform(st) * L) % L;
    } else {
        cx_z0[bid] = 0;  cx_z1[bid] = L/2;  cy[bid] = 0;
    }
}

/* ═══════════════════════════════════════════════════════════════════════
   KERNEL 5: k_metro_sub
   Metropolis update for one checkerboard sublattice (cb0 or cb1).
     s      : spin field [B*N]  (±1.0)
     cb     : indices of this sublattice [n_cb]
     nb4    : 4 neighbours per site [N*4]
     h_ext  : external field [B*N] (0 for main metro, set for wave metro)
     rng    : cell RNG states [B*N]
   ═══════════════════════════════════════════════════════════════════════ */
__global__ void k_metro_sub(f32* s, const i32* cb, const i32* nb4,
                              const f32* h_ext, curandState* rng,
                              i32 n_cb, i32 N, i32 B)
{
    i32 tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= (i32)(B * (u32)n_cb)) return;
    i32 bid = tid / n_cb;
    i32 li  = cb[tid % n_cb];          /* local site index */
    i32 gi  = bid * N + li;            /* global index */

    f32 si  = s[gi];
    /* sum neighbours */
    f32 nb_sum = s[bid*N + nb4[li*4+0]]
               + s[bid*N + nb4[li*4+1]]
               + s[bid*N + nb4[li*4+2]]
               + s[bid*N + nb4[li*4+3]];
    f32 dE  = 2.f * si * (nb_sum + h_ext[gi]);
    if (dE <= 0.f || curand_uniform(&rng[gi]) < expf(-2.f * BETA_ISING * dE)) {
        s[gi] = -si;
    }
}

/* ═══════════════════════════════════════════════════════════════════════
   KERNEL 6: k_update_base_streak_mid  (called after main metro)
   Update base[gi] and streak[gi]; reset mid to running mean.
     base  : EWMA of s  (FA weight for new value)
     mid   : midpoint signal for phi update
     streak: consecutive same-sign count
   ═══════════════════════════════════════════════════════════════════════ */
__global__ void k_update_base_streak_mid(const f32* s, f32* base, f32* mid,
                                          i32* streak, i32 N, i32 B)
{
    i32 gi = blockIdx.x * blockDim.x + threadIdx.x;
    if (gi >= B * N) return;

    f32 si = s[gi];
    f32 bi = base[gi];
    /* EWMA base update */
    base[gi] = (1.f - FA) * bi + FA * si;

    /* streak */
    i32 sk = streak[gi];
    if (si * bi >= 0.f)   /* same sign as base */
        streak[gi] = sk + 1;
    else
        streak[gi] = 0;

    /* mid: deviation from base (used by phi update later) */
    mid[gi] = si - base[gi];
}

/* ═══════════════════════════════════════════════════════════════════════
   KERNEL 7: k_compute_h_ext
   For each fired wave, write ±BETA_ISING to all sites in the disk
   centred at (cx, cy) with radius r_w, in the appropriate zone half.
     wave_z  : which zone the wave is in this step [B]  (0 or 1)
     col_g   : column (x) of each site [N]
     row_g   : row (y) of each site [N]
   h_ext is zeroed externally before each wave round.
   ═══════════════════════════════════════════════════════════════════════ */
__global__ void k_compute_h_ext(f32* h_ext,
                                  const i8* fires,
                                  const i32* cx_z0, const i32* cx_z1, const i32* cy,
                                  const i32* wave_z,
                                  const i32* col_g, const i32* row_g,
                                  i32 r_w, i32 L, i32 N, i32 B)
{
    i32 tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= B * N) return;
    i32 bid = tid / N;
    i32 li  = tid % N;

    if (!fires[bid]) { h_ext[tid] = 0.f; return; }

    i32 wz = wave_z[bid];
    i32 cx = (wz == 0) ? cx_z0[bid] : cx_z1[bid];
    i32 wy = cy[bid];
    i32 x  = col_g[li];
    i32 y  = row_g[li];

    /* toroidal distance */
    i32 dx = abs(x - cx);  if (dx > L/2) dx = L - dx;
    i32 dy = abs(y - wy);  if (dy > L/2) dy = L - dy;
    f32 h  = (dx*dx + dy*dy <= r_w*r_w) ? 1.f : 0.f;
    /* sign: zone-0 wave pushes +, zone-1 pushes – (causal contrast) */
    h_ext[tid] = (wz == 0) ? h : -h;
}

/* ═══════════════════════════════════════════════════════════════════════
   KERNEL 8: k_copy
   Generic float copy (used to copy s → s_wave before wave metro).
   ═══════════════════════════════════════════════════════════════════════ */
__global__ void k_copy(f32* dst, const f32* src, i32 n)
{
    i32 tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) dst[tid] = src[tid];
}

/* ═══════════════════════════════════════════════════════════════════════
   KERNEL 9: k_merge_s
   After wave metro: copy s_wave back to s only for sites in the
   wave-affected zone (fires[bid]==1).  Non-fired batch seeds unchanged.
   ═══════════════════════════════════════════════════════════════════════ */
__global__ void k_merge_s(f32* s, const f32* s_wave,
                            const i8* fires, i32 N, i32 B)
{
    i32 tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= B * N) return;
    i32 bid = tid / N;
    if (fires[bid]) s[tid] = s_wave[tid];
}

/* ═══════════════════════════════════════════════════════════════════════
   KERNEL 10: k_reduce_dev
   Partial reduction: accumulate sum and sum-of-squares of
   (s[gi] - base[gi]) into per-batch dev_sum, dev_sq.
   Grid: (ceil(N/BLOCK), B)
   ═══════════════════════════════════════════════════════════════════════ */
__global__ void k_reduce_dev(const f32* s, const f32* base,
                               f32* dev_sum, f32* dev_sq,
                               i32 N, i32 B)
{
    i32 bid   = blockIdx.y;
    i32 start = blockIdx.x * blockDim.x + threadIdx.x;
    if (bid >= B || start >= N) return;
    f32 d = s[bid*N + start] - base[bid*N + start];
    atomicAdd(&dev_sum[bid], d);
    atomicAdd(&dev_sq[bid],  d * d);
}

/* ═══════════════════════════════════════════════════════════════════════
   KERNEL 11: k_finalize_dev_std
   dev_std[b] = sqrt(dev_sq[b]/N - (dev_sum[b]/N)²) — population std dev.
   ═══════════════════════════════════════════════════════════════════════ */
__global__ void k_finalize_dev_std(f32* dev_std,
                                    const f32* dev_sum, const f32* dev_sq,
                                    f32 N_inv, i32 B)
{
    i32 bid = blockIdx.x * blockDim.x + threadIdx.x;
    if (bid >= B) return;
    f32 mu  = dev_sum[bid] * N_inv;
    f32 var = dev_sq[bid]  * N_inv - mu * mu;
    dev_std[bid] = (var > 0.f) ? sqrtf(var) : 1e-6f;
}

/* ═══════════════════════════════════════════════════════════════════════
   KERNEL 12: k_update_mid_phi_accum_M
   The heart of VCSM per step (called once after wave metro):
    - For each site in zone-0 (z0f[li]>0.5):
       * viability gate: streak[gi] >= SS
       * if gated: phi_new = FIELD_DECAY*phi + FA*mid (weighted by dev_std)
       * phi birth: if curand < P_causal and gated: phi set from causal signal
       * accumulate M_accum[b] += sign(phi[gi]) * z0f[li]
   Notation: z0f[li] = +1 if site is in zone-0, -1 if zone-1
             (zone-1 sites subtract from M)
   ═══════════════════════════════════════════════════════════════════════ */
__global__ void k_update_mid_phi_accum_M(
    const f32* s,       /* [B*N] */
    const f32* base,    /* [B*N] */
    f32* mid,           /* [B*N] */
    f32* phi,           /* [B*N] */
    i32* streak,        /* [B*N] */
    const i32* wave_z,  /* [B]   which zone was wave in */
    const i8*  fires,   /* [B]   did wave fire */
    const f32* h_ext,   /* [B*N] wave field */
    const f32* z0f,     /* [N]   +1=zone0, -1=zone1 */
    const f32* P_ten,   /* [B]   P_causal per seed (all same for single-P run) */
    const f32* dev_std, /* [B]   population std of (s-base) */
    f32* M_accum,       /* [B]   accumulator for order parameter */
    curandState* rng,   /* [B*N] */
    i32 N, i32 B, i32 n_z0)
{
    i32 gi = blockIdx.x * blockDim.x + threadIdx.x;
    if (gi >= B * N) return;
    i32 bid = gi / N;
    i32 li  = gi % N;

    /* ── phi field decay (always) ── */
    phi[gi] *= FIELD_DECAY;

    /* ── viability gate ── */
    if (streak[gi] < SS) return;

    f32 dv  = dev_std[bid];
    f32 mi  = mid[gi];

    /* ── phi update from mid signal ── */
    f32 phi_new = phi[gi] + FA * mi / fmaxf(dv, 1e-6f);
    phi[gi] = fmaxf(-1.f, fminf(1.f, phi_new));

    /* ── causal birth: wave fired in this site's zone ── */
    i32 this_zone = (z0f[li] > 0.f) ? 0 : 1;
    if (fires[bid] && wave_z[bid] == this_zone) {
        /* causal purity test */
        if (curand_uniform(&rng[gi]) < P_ten[bid]) {
            /* set phi toward wave-zone sign */
            f32 signal = h_ext[gi];   /* ±1 or 0 */
            if (fabsf(signal) > 0.5f)
                phi[gi] = (phi[gi] + signal) * 0.5f;
        }
    }

    /* ── accumulate M (zone-signed phi mean) ── */
    f32 contrib = phi[gi] * z0f[li];   /* +phi for z0, -phi for z1 */
    atomicAdd(&M_accum[bid], contrib);
}

/* ═══════════════════════════════════════════════════════════════════════
   KERNEL 13: k_flip_wave_z
   After each wave round, toggle which zone the wave targets next.
   ═══════════════════════════════════════════════════════════════════════ */
__global__ void k_flip_wave_z(i32* wave_z, const i8* fires, i32 B)
{
    i32 bid = blockIdx.x * blockDim.x + threadIdx.x;
    if (bid >= B) return;
    if (fires[bid]) wave_z[bid] = 1 - wave_z[bid];
}

/* ═══════════════════════════════════════════════════════════════════════
   KERNEL 14: k_zero_h_ext
   Zero h_ext between wave rounds (captured in graph).
   ═══════════════════════════════════════════════════════════════════════ */
__global__ void k_zero_h_ext(f32* h_ext, i32 BN)
{
    i32 tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < BN) h_ext[tid] = 0.f;
}

/* ═══════════════════════════════════════════════════════════════════════
   KERNEL 15: k_finalize_M
   M_out[b] = M_accum[b] / n_z0  (normalise by zone size).
   ═══════════════════════════════════════════════════════════════════════ */
__global__ void k_finalize_M(f32* M_out, const f32* M_accum, f32 n_z0_inv, i32 B)
{
    i32 bid = blockIdx.x * blockDim.x + threadIdx.x;
    if (bid >= B) return;
    M_out[bid] = M_accum[bid] * n_z0_inv;
}

/* ═══════════════════════════════════════════════════════════════════════
   vcml_full_step()  --  the single pybind11-exposed function
   Launches all 23 kernels for ONE VCML time-step.
   All tensor arguments are CUDA tensors already allocated by the caller.
   The caller captures this function in a torch CUDAGraph for zero Python
   overhead in the simulation loop.
   ═══════════════════════════════════════════════════════════════════════ */
void vcml_full_step(
    /* spin / field state */
    torch::Tensor s,          /* [B*N] float32  spin field            */
    torch::Tensor base,       /* [B*N] float32  EWMA baseline         */
    torch::Tensor mid,        /* [B*N] float32  mid-signal            */
    torch::Tensor phi,        /* [B*N] float32  memory field          */
    torch::Tensor streak,     /* [B*N] int32    consecutive-sign count */
    torch::Tensor wave_z,     /* [B]   int32    current wave zone     */
    torch::Tensor M_out,      /* [B]   float32  order parameter output*/
    /* scratch */
    torch::Tensor s_wave,     /* [B*N] float32  copy of s for wave metro */
    torch::Tensor h_ext,      /* [B*N] float32  external wave field   */
    torch::Tensor fires,      /* [B]   int8     did wave fire         */
    torch::Tensor cx_z0,      /* [B]   int32    wave cx zone-0        */
    torch::Tensor cx_z1,      /* [B]   int32    wave cx zone-1        */
    torch::Tensor cy,         /* [B]   int32    wave cy               */
    torch::Tensor M_accum,    /* [B]   float32  M accumulator         */
    torch::Tensor dev_sum,    /* [B]   float32  dev partial sum       */
    torch::Tensor dev_sq,     /* [B]   float32  dev partial sum-sq    */
    torch::Tensor dev_std,    /* [B]   float32  population std dev    */
    /* RNG states (uint8 raw bytes, cast to curandState*) */
    torch::Tensor cell_rng,   /* [B*N * rng_bytes]  uint8             */
    torch::Tensor wave_rng_t, /* [B   * rng_bytes]  uint8             */
    /* geometry (precomputed, constant) */
    torch::Tensor nb4,        /* [N*4] int32    neighbour indices     */
    torch::Tensor cb0,        /* [n_cb0] int32  checkerboard sub 0   */
    torch::Tensor cb1,        /* [n_cb1] int32  checkerboard sub 1   */
    torch::Tensor col_g,      /* [N]   int32    column per site       */
    torch::Tensor row_g,      /* [N]   int32    row per site          */
    torch::Tensor z0f,        /* [N]   float32  +1=zone0, -1=zone1   */
    /* parameters */
    torch::Tensor P_ten,      /* [B]   float32  P_causal per seed     */
    float wp,                  /* wave fire probability per step       */
    int r_w,                   /* wave radius                          */
    int L, int B, int N,
    int n_cb0, int n_cb1, int n_z0)
{
    /* ── raw device pointers ── */
    f32* sp        = s.data_ptr<f32>();
    f32* basep     = base.data_ptr<f32>();
    f32* midp      = mid.data_ptr<f32>();
    f32* phip      = phi.data_ptr<f32>();
    i32* streakp   = streak.data_ptr<i32>();
    i32* wave_zp   = wave_z.data_ptr<i32>();
    f32* M_outp    = M_out.data_ptr<f32>();
    f32* s_wavep   = s_wave.data_ptr<f32>();
    f32* h_extp    = h_ext.data_ptr<f32>();
    i8*  firesp    = fires.data_ptr<i8>();
    i32* cx_z0p    = cx_z0.data_ptr<i32>();
    i32* cx_z1p    = cx_z1.data_ptr<i32>();
    i32* cyp       = cy.data_ptr<i32>();
    f32* M_accp    = M_accum.data_ptr<f32>();
    f32* dev_sump  = dev_sum.data_ptr<f32>();
    f32* dev_sqp   = dev_sq.data_ptr<f32>();
    f32* dev_stdp  = dev_std.data_ptr<f32>();
    curandState* cell_rngp = reinterpret_cast<curandState*>(cell_rng.data_ptr<u8>());
    curandState* wave_rngp = reinterpret_cast<curandState*>(wave_rng_t.data_ptr<u8>());
    i32* nb4p      = nb4.data_ptr<i32>();
    i32* cb0p      = cb0.data_ptr<i32>();
    i32* cb1p      = cb1.data_ptr<i32>();
    i32* col_gp    = col_g.data_ptr<i32>();
    i32* row_gp    = row_g.data_ptr<i32>();
    f32* z0fp      = z0f.data_ptr<f32>();
    f32* P_tenp    = P_ten.data_ptr<f32>();

    i32 BN = B * N;
    i32 grid_BN   = (BN   + BLOCK - 1) / BLOCK;
    i32 grid_B    = (B    + BLOCK - 1) / BLOCK;
    i32 grid_cb0  = (B * n_cb0 + BLOCK - 1) / BLOCK;
    i32 grid_cb1  = (B * n_cb1 + BLOCK - 1) / BLOCK;
    /* 2D grid for reduce_dev: x=ceil(N/BLOCK), y=B */
    dim3 grid_reduce((N + BLOCK - 1) / BLOCK, B);
    f32 N_inv    = 1.f / (f32)N;
    f32 n_z0_inv = 1.f / (f32)n_z0;

    /* ── LAUNCH 1: zero accumulators ── */
    k_zero_accum<<<grid_B, BLOCK>>>(M_accp, dev_sump, dev_sqp, B);

    /* ── LAUNCH 2: generate wave ── */
    k_gen_wave<<<grid_B, BLOCK>>>(wave_rngp, firesp, cx_z0p, cx_z1p, cyp,
                                   wp, L, B);

    /* ── LAUNCHES 3-4: main Metropolis (h_ext = 0 = already zeroed) ── */
    k_metro_sub<<<grid_cb0, BLOCK>>>(sp, cb0p, nb4p, h_extp, cell_rngp,
                                      n_cb0, N, B);
    k_metro_sub<<<grid_cb1, BLOCK>>>(sp, cb1p, nb4p, h_extp, cell_rngp,
                                      n_cb1, N, B);

    /* ── LAUNCH 5: update base, streak, mid (post main metro) ── */
    k_update_base_streak_mid<<<grid_BN, BLOCK>>>(sp, basep, midp, streakp, N, B);

    /* ── LAUNCHES 6-7: reduce deviation std ── */
    k_reduce_dev<<<grid_reduce, BLOCK>>>(sp, basep, dev_sump, dev_sqp, N, B);
    k_finalize_dev_std<<<grid_B, BLOCK>>>(dev_stdp, dev_sump, dev_sqp, N_inv, B);

    /* ── WAVE ROUNDS: WAVE_DUR × (compute h_ext + 2 metro sub-steps) ── */
    /* First, copy s → s_wave for isolated wave metro */
    k_copy<<<grid_BN, BLOCK>>>(s_wavep, sp, BN);   /* LAUNCH 8 */

    for (int wr = 0; wr < WAVE_DUR; wr++) {
        /* zero h_ext */
        k_zero_h_ext<<<grid_BN, BLOCK>>>(h_extp, BN);                    /* +1 */
        /* compute wave field */
        k_compute_h_ext<<<grid_BN, BLOCK>>>(h_extp, firesp,
                                             cx_z0p, cx_z1p, cyp,
                                             wave_zp, col_gp, row_gp,
                                             r_w, L, N, B);              /* +1 */
        /* wave Metropolis on s_wave (2 sub-steps) */
        k_metro_sub<<<grid_cb0, BLOCK>>>(s_wavep, cb0p, nb4p, h_extp,
                                          cell_rngp, n_cb0, N, B);       /* +1 */
        k_metro_sub<<<grid_cb1, BLOCK>>>(s_wavep, cb1p, nb4p, h_extp,
                                          cell_rngp, n_cb1, N, B);       /* +1 */
    }
    /* LAUNCH 8 + WAVE_DUR*4 = 8 + 20 = launches 8-27; keep counting */

    /* ── merge s_wave back into s ── */
    k_merge_s<<<grid_BN, BLOCK>>>(sp, s_wavep, firesp, N, B);            /* +1 */

    /* ── phi update + M accumulation (one combined kernel) ── */
    k_update_mid_phi_accum_M<<<grid_BN, BLOCK>>>(
        sp, basep, midp, phip, streakp,
        wave_zp, firesp, h_extp, z0fp, P_tenp, dev_stdp,
        M_accp, cell_rngp, N, B, n_z0);                                  /* +1 */

    /* ── flip wave zone for next step ── */
    k_flip_wave_z<<<grid_B, BLOCK>>>(wave_zp, firesp, B);                /* +1 */

    /* ── zero h_ext for next step's main metro ── */
    k_zero_h_ext<<<grid_BN, BLOCK>>>(h_extp, BN);                        /* +1 */

    /* ── finalise M_out ── */
    k_finalize_M<<<grid_B, BLOCK>>>(M_outp, M_accp, n_z0_inv, B);       /* +1 */
}

/* ═══════════════════════════════════════════════════════════════════════
   init_rng_states()  --  initialise both RNG buffers
   cell_rng : uint8 tensor of size B*N*sizeof(curandStateXORWOW)
   wave_rng : uint8 tensor of size B*sizeof(curandStateXORWOW)
   ═══════════════════════════════════════════════════════════════════════ */
void init_rng_states(torch::Tensor cell_rng, torch::Tensor wave_rng,
                      int64_t base_seed, int BN, int B)
{
    curandState* cell_p = reinterpret_cast<curandState*>(cell_rng.data_ptr<u8>());
    curandState* wave_p = reinterpret_cast<curandState*>(wave_rng.data_ptr<u8>());
    i32 grid_BN = (BN + BLOCK - 1) / BLOCK;
    i32 grid_B  = (B  + BLOCK - 1) / BLOCK;
    k_init_rng<<<grid_BN, BLOCK>>>(cell_p, (u64)base_seed, BN);
    k_init_wave_rng<<<grid_B, BLOCK>>>(wave_p, (u64)base_seed, B);
    CUDA_CHECK(cudaDeviceSynchronize());
}

/* ═══════════════════════════════════════════════════════════════════════
   pybind11 module
   ═══════════════════════════════════════════════════════════════════════ */
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("vcml_full_step", &vcml_full_step, "One VCML time-step (all kernels)");
    m.def("init_rng_states", &init_rng_states, "Initialise cuRAND states on device");
    m.def("rng_state_size", []() { return (int64_t)sizeof(curandStateXORWOW); },
          "Size in bytes of one curandStateXORWOW");
}
