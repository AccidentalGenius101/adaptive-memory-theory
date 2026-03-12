"""
vcml_cluster.py  --  Two-GPU cluster coordinator (desktop 3060 + laptop 3050 Ti)

Splits seed_list across two machines and runs them concurrently.
Drop-in replacement for run_batch_gpu_v4 for embarrassingly parallel workloads.

Usage:
    from vcml_cluster import run_batch_cluster
    results = run_batch_cluster(L=80, P_causal_list=[0.001],
                                seed_list=list(range(8)), nsteps=100_000)

Config:
    LAPTOP_HOST  -- SSH target (user@ip)
    LAPTOP_REPO  -- path to repo on laptop
    LAPTOP_PY    -- python executable on laptop
    LAPTOP_FRAC  -- fraction of seeds to send to laptop (0.5 = equal split)
"""

import subprocess, json, threading, time
from vcml_gpu_v4 import run_batch_gpu_v4

# ── cluster config ─────────────────────────────────────────────────────
LAPTOP_HOST = 'briel@192.168.2.151'
LAPTOP_REPO = r'C:\Users\briel\Documents\vcsm-theory'
LAPTOP_PY   = 'py'
LAPTOP_FRAC = 0.5   # give laptop half the seeds (adjust if speeds differ)

# ── remote runner ──────────────────────────────────────────────────────
def _run_remote(L, P, seeds, nsteps, r_w, h_field):
    seed_str = ','.join(map(str, seeds))
    cmd = [
        'ssh', LAPTOP_HOST,
        f'cd "{LAPTOP_REPO}" && {LAPTOP_PY} run_remote.py'
        f' --L {L} --P {P} --seeds {seed_str}'
        f' --nsteps {nsteps} --rw {r_w} --hfield {h_field}'
    ]
    try:
        raw = subprocess.check_output(cmd, stderr=subprocess.PIPE, timeout=3600)
        for line in raw.decode('utf-8', errors='replace').splitlines():
            if line.startswith('RESULT_JSON:'):
                return json.loads(line[len('RESULT_JSON:'):])
        raise RuntimeError(f'No RESULT_JSON in remote output:\n{raw.decode()[:500]}')
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f'Remote failed: {e.stderr.decode()[:500]}')


def run_batch_cluster(L, P_causal_list, seed_list, nsteps,
                      r_w=5, h_field=0.0, verbose=True):
    """
    Run batch across desktop + laptop concurrently.
    Returns list of result dicts (same format as run_batch_gpu_v4).
    Falls back to local-only if SSH fails.
    """
    all_results = []

    for P in P_causal_list:
        # Split seeds
        n_remote = max(1, round(len(seed_list) * LAPTOP_FRAC))
        local_seeds  = seed_list[n_remote:]
        remote_seeds = seed_list[:n_remote]

        if verbose:
            print(f'[cluster] P={P:.4f}  local seeds={local_seeds}  remote seeds={remote_seeds}')

        results = [None, None]
        errors  = [None, None]

        def _local():
            try:
                results[0] = run_batch_gpu_v4(L, [P], local_seeds, nsteps,
                                               r_w=r_w, h_field=h_field)
            except Exception as e:
                errors[0] = e

        def _remote():
            try:
                results[1] = _run_remote(L, P, remote_seeds, nsteps, r_w, h_field)
            except Exception as e:
                errors[1] = e
                if verbose:
                    print(f'[cluster] WARNING: remote failed ({e}), falling back to local')

        t0 = time.time()
        tl = threading.Thread(target=_local,  daemon=True)
        tr = threading.Thread(target=_remote, daemon=True)
        tl.start(); tr.start()
        tl.join();  tr.join()
        dt = time.time() - t0

        if errors[0]:
            raise errors[0]

        merged = (results[0] or []) + (results[1] or [])
        if verbose:
            n_total = len(merged)
            print(f'[cluster] P={P:.4f}  {n_total} seeds done in {dt:.0f}s'
                  + (f'  (remote failed, {len(results[0])} local only)' if errors[1] else ''))

        all_results.extend(merged)

    return all_results


# ── quick test ─────────────────────────────────────────────────────────
if __name__ == '__main__':
    import numpy as np
    print('Testing cluster with L=80, P=0.010, 8 seeds, 5k steps...')
    t0 = time.time()
    res = run_batch_cluster(80, [0.010], list(range(8)), 5_000, r_w=5)
    dt = time.time() - t0
    absMs = [r['absM'] for r in res]
    print(f'  {len(res)} results in {dt:.1f}s')
    print(f'  |M| = {np.mean(absMs):.4f} ± {np.std(absMs)/len(absMs)**0.5:.4f}')
    print(f'  seeds: {[r.get("seed", "?") for r in res]}')
