/*
 * vcml_step_v5.cu  --  Persistent cooperative-kernel VCML implementation
 *
 * Optimisations over v4:
 *   1. Single persistent cooperative kernel: loops nsteps internally,
 *      uses grid.sync() between phases → zero kernel-scheduling overhead.
 *   2. FP16 phi field: half the memory bandwidth for the memory field.
 *   3. Block-level smem reduction for dev_std: ~250x fewer global atomicAdds.
 *   4. Returns full M_trace [nsteps*B] in one cudaLaunchCooperativeKernel call.
 *
 * Build: same flags as v4 plus -lcuda (for cooperative launch symbol).
 * Tested on RTX 3060 (sm_86).
 */

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <curand_kernel.h>
#include <cooperative_groups.h>
#include <pybind11/pybind11.h>
#include <torch/extension.h>
#include <cstdint>
#include <cmath>

namespace cg = cooperative_groups;
namespace py = pybind11;

typedef uint32_t  u32;
typedef int32_t   i32;
typedef uint8_t   u8;
typedef uint64_t  u64;
typedef float     f32;
typedef __half    f16;
typedef int8_t    i8;

/* ── compile-time hypers (must match Python side) ────────────────────── */
#define BLOCK        256
#define WAVE_DUR     5
#define SS           8
#define FA           0.30f
#define FIELD_DECAY  0.999f
#define BETA_ISING   0.44f

#define CUDA_CHECK(x) do { \
    cudaError_t _e = (x); \
    if (_e != cudaSuccess) \
        throw std::runtime_error(std::string("CUDA error: ") + cudaGetErrorString(_e)); \
} while(0)

/* ═══════════════════════════════════════════════════════════════════════
   All kernel arguments in one struct (passed by value to cooperative kernel)
   ═══════════════════════════════════════════════════════════════════════ */
struct VCMLArgs {
    /* main fields */
    f32* s;          /* [B*N]      spin field                  */
    f32* base;       /* [B*N]      EWMA baseline               */
    f32* mid;        /* [B*N]      mid-signal                  */
    f16* phi;        /* [B*N] FP16 memory field                */
    i32* streak;     /* [B*N]      consecutive-sign count      */
    i32* wave_z;     /* [B]        current wave zone (0 or 1) */
    /* output */
    f32* M_trace;    /* [nsteps*B] full M trace (written here) */
    /* scratch */
    f32* s_wave;     /* [B*N]      copy of s for wave metro    */
    f32* h_ext;      /* [B*N]      external wave field         */
    i8*  fires;      /* [B]        did wave fire this step     */
    i32* cx_z0;      /* [B]        wave centre x zone-0        */
    i32* cx_z1;      /* [B]        wave centre x zone-1        */
    i32* cy;         /* [B]        wave centre y               */
    f32* M_accum;    /* [B]        M accumulator per step      */
    f32* dev_sum;    /* [B]        partial dev sum             */
    f32* dev_sq;     /* [B]        partial dev sum-sq          */
    f32* dev_std;    /* [B]        population std dev          */
    /* RNG */
    curandState* cell_rng;  /* [B*N] per-cell cuRAND states    */
    curandState* wave_rng;  /* [B]   per-seed wave RNG states  */
    /* geometry (constant throughout run) */
    i32* nb4;        /* [N*4]      4 neighbours per site       */
    i32* cb0;        /* [n_cb0]    checkerboard sublattice 0   */
    i32* cb1;        /* [n_cb1]    checkerboard sublattice 1   */
    i32* col_g;      /* [N]        column per site             */
    i32* row_g;      /* [N]        row per site                */
    f32* z0f;        /* [N]        +1=zone0, -1=zone1          */
    f32* P_ten;      /* [B]        P_causal per seed           */
    /* scalars */
    f32 wp;          /* wave fire probability                  */
    i32 r_w;         /* wave radius                            */
    i32 L, B, N;
    i32 n_cb0, n_cb1, n_z0;
    f32 N_inv, n_z0_inv;
    i32 nsteps;
};

/* ═══════════════════════════════════════════════════════════════════════
   Inline metro update for a single site gi in spin array sp[].
   Reads neighbours from nb4; uses h_ext and cell_rng.
   ═══════════════════════════════════════════════════════════════════════ */
__device__ __forceinline__
void metro_site(f32* sp, const i32* nb4, const f32* h_ext,
                curandState* rng, i32 gi, i32 base_off)
{
    i32 li     = gi - base_off;          /* local site index within this batch */
    f32 si     = sp[gi];
    f32 nb_sum = sp[base_off + nb4[li*4+0]]
               + sp[base_off + nb4[li*4+1]]
               + sp[base_off + nb4[li*4+2]]
               + sp[base_off + nb4[li*4+3]];
    f32 dE = 2.f * si * (nb_sum + h_ext[gi]);
    if (dE <= 0.f || curand_uniform(&rng[gi]) < expf(-2.f * BETA_ISING * dE))
        sp[gi] = -si;
}

/* ═══════════════════════════════════════════════════════════════════════
   PERSISTENT COOPERATIVE KERNEL
   One launch covers all nsteps; grid.sync() replaces kernel boundaries.
   Shared memory layout per block: [2*B floats] = dev_sum_smem + dev_sq_smem
   ═══════════════════════════════════════════════════════════════════════ */
__global__ void __launch_bounds__(BLOCK, 4)
vcml_persistent(VCMLArgs a)
{
    cg::grid_group grid = cg::this_grid();

    i32 tid    = (i32)(blockIdx.x * blockDim.x + threadIdx.x);
    i32 stride = (i32)(gridDim.x  * blockDim.x);
    i32 BN     = a.B * a.N;

    /* shared memory: two arrays of B floats for block-level dev reduction */
    extern __shared__ f32 smem[];   /* 2*B floats */
    f32* s_dsum = smem;
    f32* s_dsq  = smem + a.B;

    for (i32 t = 0; t < a.nsteps; t++) {

        /* ── 1: zero accumulators ────────────────────────────────────── */
        for (i32 b = tid; b < a.B; b += stride) {
            a.M_accum[b] = 0.f;
            a.dev_sum[b] = 0.f;
            a.dev_sq[b]  = 0.f;
        }
        grid.sync();

        /* ── 2: generate wave per seed ───────────────────────────────── */
        for (i32 b = tid; b < a.B; b += stride) {
            curandState* st = &a.wave_rng[b];
            a.fires[b] = (curand_uniform(st) < a.wp) ? (i8)1 : (i8)0;
            if (a.fires[b]) {
                i32 half    = a.L / 2;
                a.cx_z0[b]  = (i32)(curand_uniform(st) * half) % half;
                a.cx_z1[b]  = half + (i32)(curand_uniform(st) * half) % half;
                a.cy[b]     = (i32)(curand_uniform(st) * a.L) % a.L;
            }
        }
        grid.sync();

        /* ── 3: main Metropolis cb0 (h_ext = 0 from previous step) ──── */
        for (i32 i = tid; i < a.B * a.n_cb0; i += stride) {
            i32 b   = i / a.n_cb0;
            i32 li  = a.cb0[i % a.n_cb0];
            i32 gi  = b * a.N + li;
            metro_site(a.s, a.nb4, a.h_ext, a.cell_rng, gi, b * a.N);
        }
        grid.sync();

        /* ── 4: main Metropolis cb1 ──────────────────────────────────── */
        for (i32 i = tid; i < a.B * a.n_cb1; i += stride) {
            i32 b   = i / a.n_cb1;
            i32 li  = a.cb1[i % a.n_cb1];
            i32 gi  = b * a.N + li;
            metro_site(a.s, a.nb4, a.h_ext, a.cell_rng, gi, b * a.N);
        }
        grid.sync();

        /* ── 5: update base / streak / mid ──────────────────────────── */
        for (i32 gi = tid; gi < BN; gi += stride) {
            f32 si       = a.s[gi];
            f32 bi       = a.base[gi];
            f32 b_new    = (1.f - FA) * bi + FA * si;
            a.base[gi]   = b_new;
            i32 sk       = a.streak[gi];
            a.streak[gi] = (si * bi >= 0.f) ? sk + 1 : 0;
            a.mid[gi]    = si - b_new;
        }
        grid.sync();

        /* ── 6: block-level dev reduction ──────────────────────────────
           Each block accumulates partial dev_sum / dev_sq in smem,
           then one atomicAdd per block per batch → huge contention drop. */
        /* init smem */
        for (i32 b = (i32)threadIdx.x; b < a.B; b += (i32)blockDim.x) {
            s_dsum[b] = 0.f;
            s_dsq[b]  = 0.f;
        }
        __syncthreads();
        for (i32 gi = tid; gi < BN; gi += stride) {
            i32 b = gi / a.N;
            f32 d = a.s[gi] - a.base[gi];
            atomicAdd(&s_dsum[b], d);
            atomicAdd(&s_dsq[b],  d * d);
        }
        __syncthreads();
        for (i32 b = (i32)threadIdx.x; b < a.B; b += (i32)blockDim.x) {
            atomicAdd(&a.dev_sum[b], s_dsum[b]);
            atomicAdd(&a.dev_sq[b],  s_dsq[b]);
        }
        grid.sync();

        /* ── 7: finalize dev_std ─────────────────────────────────────── */
        for (i32 b = tid; b < a.B; b += stride) {
            f32 mu       = a.dev_sum[b] * a.N_inv;
            f32 var      = a.dev_sq[b]  * a.N_inv - mu * mu;
            a.dev_std[b] = (var > 0.f) ? sqrtf(var) : 1e-6f;
        }
        grid.sync();

        /* ── 8: copy s → s_wave ─────────────────────────────────────── */
        for (i32 i = tid; i < BN; i += stride) a.s_wave[i] = a.s[i];
        grid.sync();

        /* ── 9–28: WAVE_DUR wave rounds ─────────────────────────────── */
        for (i32 wr = 0; wr < WAVE_DUR; wr++) {

            /* zero h_ext */
            for (i32 i = tid; i < BN; i += stride) a.h_ext[i] = 0.f;
            grid.sync();

            /* compute wave field */
            for (i32 gi = tid; gi < BN; gi += stride) {
                i32 b   = gi / a.N;
                i32 li  = gi % a.N;
                if (!a.fires[b]) { a.h_ext[gi] = 0.f; continue; }
                i32 wz  = a.wave_z[b];
                i32 cx  = (wz == 0) ? a.cx_z0[b] : a.cx_z1[b];
                i32 wy  = a.cy[b];
                i32 x   = a.col_g[li];
                i32 y   = a.row_g[li];
                i32 dx  = abs(x - cx); if (dx > a.L/2) dx = a.L - dx;
                i32 dy  = abs(y - wy); if (dy > a.L/2) dy = a.L - dy;
                f32 h   = (dx*dx + dy*dy <= a.r_w*a.r_w) ? 1.f : 0.f;
                a.h_ext[gi] = (wz == 0) ? h : -h;
            }
            grid.sync();

            /* wave metro cb0 */
            for (i32 i = tid; i < a.B * a.n_cb0; i += stride) {
                i32 b   = i / a.n_cb0;
                i32 li  = a.cb0[i % a.n_cb0];
                i32 gi  = b * a.N + li;
                metro_site(a.s_wave, a.nb4, a.h_ext, a.cell_rng, gi, b * a.N);
            }
            grid.sync();

            /* wave metro cb1 */
            for (i32 i = tid; i < a.B * a.n_cb1; i += stride) {
                i32 b   = i / a.n_cb1;
                i32 li  = a.cb1[i % a.n_cb1];
                i32 gi  = b * a.N + li;
                metro_site(a.s_wave, a.nb4, a.h_ext, a.cell_rng, gi, b * a.N);
            }
            grid.sync();
        }

        /* ── 29: merge s_wave → s ────────────────────────────────────── */
        for (i32 gi = tid; gi < BN; gi += stride)
            if (a.fires[gi / a.N]) a.s[gi] = a.s_wave[gi];
        grid.sync();

        /* ── 30: phi update + M accumulation (FP16 phi) ─────────────── */
        for (i32 gi = tid; gi < BN; gi += stride) {
            i32 b   = gi / a.N;
            i32 li  = gi % a.N;

            f32 phi_f = __half2float(a.phi[gi]) * FIELD_DECAY;

            if (a.streak[gi] >= SS) {
                f32 dv  = a.dev_std[b];
                phi_f  += FA * a.mid[gi] / fmaxf(dv, 1e-6f);
                phi_f   = fmaxf(-1.f, fminf(1.f, phi_f));

                i32 this_zone = (a.z0f[li] > 0.f) ? 0 : 1;
                if (a.fires[b] && a.wave_z[b] == this_zone) {
                    if (curand_uniform(&a.cell_rng[gi]) < a.P_ten[b]) {
                        f32 sig = a.h_ext[gi];
                        if (fabsf(sig) > 0.5f)
                            phi_f = (phi_f + sig) * 0.5f;
                    }
                }
            }

            a.phi[gi] = __float2half(phi_f);
            atomicAdd(&a.M_accum[b], phi_f * a.z0f[li]);
        }
        grid.sync();

        /* ── 31: flip wave_z ─────────────────────────────────────────── */
        for (i32 b = tid; b < a.B; b += stride)
            if (a.fires[b]) a.wave_z[b] ^= 1;
        grid.sync();

        /* ── 32: zero h_ext for next step's main metro ───────────────── */
        for (i32 i = tid; i < BN; i += stride) a.h_ext[i] = 0.f;
        grid.sync();

        /* ── 33: write M_trace[t, b] ─────────────────────────────────── */
        for (i32 b = tid; b < a.B; b += stride)
            a.M_trace[(i32)(t * a.B + b)] = a.M_accum[b] * a.n_z0_inv;
        grid.sync();
    }
}

/* ═══════════════════════════════════════════════════════════════════════
   vcml_run()  --  host launcher for the persistent cooperative kernel
   Returns M_trace as a CUDA float32 tensor [nsteps, B].
   ═══════════════════════════════════════════════════════════════════════ */
torch::Tensor vcml_run(
    torch::Tensor s, torch::Tensor base, torch::Tensor mid,
    torch::Tensor phi,         /* float16 [B*N] */
    torch::Tensor streak, torch::Tensor wave_z,
    torch::Tensor s_wave, torch::Tensor h_ext,
    torch::Tensor fires, torch::Tensor cx_z0, torch::Tensor cx_z1,
    torch::Tensor cy, torch::Tensor M_accum,
    torch::Tensor dev_sum, torch::Tensor dev_sq, torch::Tensor dev_std,
    torch::Tensor cell_rng, torch::Tensor wave_rng_t,
    torch::Tensor nb4, torch::Tensor cb0, torch::Tensor cb1,
    torch::Tensor col_g, torch::Tensor row_g, torch::Tensor z0f,
    torch::Tensor P_ten,
    float wp, int r_w, int L, int B, int N,
    int n_cb0, int n_cb1, int n_z0, int nsteps)
{
    /* allocate output trace */
    auto M_trace = torch::zeros({nsteps, B},
        torch::TensorOptions().dtype(torch::kFloat32).device(s.device()));

    VCMLArgs a;
    a.s        = s.data_ptr<f32>();
    a.base     = base.data_ptr<f32>();
    a.mid      = mid.data_ptr<f32>();
    a.phi      = reinterpret_cast<f16*>(phi.data_ptr<at::Half>());
    a.streak   = streak.data_ptr<i32>();
    a.wave_z   = wave_z.data_ptr<i32>();
    a.M_trace  = M_trace.data_ptr<f32>();
    a.s_wave   = s_wave.data_ptr<f32>();
    a.h_ext    = h_ext.data_ptr<f32>();
    a.fires    = fires.data_ptr<i8>();
    a.cx_z0    = cx_z0.data_ptr<i32>();
    a.cx_z1    = cx_z1.data_ptr<i32>();
    a.cy       = cy.data_ptr<i32>();
    a.M_accum  = M_accum.data_ptr<f32>();
    a.dev_sum  = dev_sum.data_ptr<f32>();
    a.dev_sq   = dev_sq.data_ptr<f32>();
    a.dev_std  = dev_std.data_ptr<f32>();
    a.cell_rng = reinterpret_cast<curandState*>(cell_rng.data_ptr<u8>());
    a.wave_rng = reinterpret_cast<curandState*>(wave_rng_t.data_ptr<u8>());
    a.nb4      = nb4.data_ptr<i32>();
    a.cb0      = cb0.data_ptr<i32>();
    a.cb1      = cb1.data_ptr<i32>();
    a.col_g    = col_g.data_ptr<i32>();
    a.row_g    = row_g.data_ptr<i32>();
    a.z0f      = z0f.data_ptr<f32>();
    a.P_ten    = P_ten.data_ptr<f32>();
    a.wp       = wp;  a.r_w = r_w;
    a.L = L;  a.B = B;  a.N = N;
    a.n_cb0 = n_cb0;  a.n_cb1 = n_cb1;  a.n_z0 = n_z0;
    a.N_inv    = 1.f / (f32)N;
    a.n_z0_inv = 1.f / (f32)n_z0;
    a.nsteps   = nsteps;

    /* query max resident blocks for cooperative launch */
    int max_blocks_per_sm;
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &max_blocks_per_sm, vcml_persistent, BLOCK,
        2 * B * sizeof(f32)));   /* smem = 2*B floats */

    int num_sms;
    CUDA_CHECK(cudaDeviceGetAttribute(&num_sms,
        cudaDevAttrMultiProcessorCount, 0));

    int grid_size = num_sms * max_blocks_per_sm;

    /* shared memory size */
    size_t smem_bytes = 2 * (size_t)B * sizeof(f32);

    void* kargs[] = { &a };
    CUDA_CHECK(cudaLaunchCooperativeKernel(
        (void*)vcml_persistent,
        grid_size, BLOCK,
        kargs,
        smem_bytes,
        /*stream=*/0));

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    return M_trace;   /* [nsteps, B] on CUDA */
}

/* ═══════════════════════════════════════════════════════════════════════
   init_rng_states()  --  same as v4 (called once at startup)
   ═══════════════════════════════════════════════════════════════════════ */
__global__ void k_init_rng(curandState* states, u64 base_seed, i32 total)
{
    i32 tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= total) return;
    curand_init(base_seed + (u64)tid, tid, 0, &states[tid]);
}

__global__ void k_init_wave_rng(curandState* states, u64 base_seed, i32 B)
{
    i32 bid = blockIdx.x * blockDim.x + threadIdx.x;
    if (bid >= B) return;
    curand_init(base_seed + 0x8000000000ULL + (u64)bid, bid, 0, &states[bid]);
}

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

int rng_state_size_v5() { return (int)sizeof(curandStateXORWOW); }

/* ═══════════════════════════════════════════════════════════════════════
   pybind11 module
   ═══════════════════════════════════════════════════════════════════════ */
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("vcml_run",       &vcml_run,       "Run nsteps of VCML (persistent kernel)");
    m.def("init_rng_states",&init_rng_states,"Initialise cuRAND states on device");
    m.def("rng_state_size", &rng_state_size_v5, "Bytes per curandStateXORWOW");
}
