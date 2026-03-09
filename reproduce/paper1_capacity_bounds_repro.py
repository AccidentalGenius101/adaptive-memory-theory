"""
paper1_repro.py -- Reproduction script for Paper 1
"Geometric Capacity Laws for Minimum-Separation Routing on the Hypersphere"

Reproduces Experiment 1 (threshold sweep at HS=2): RSA capacity vs tau on S^1.

Expected output (SEEDS=5, steps=2000):
  tau=0.301  N_obs=15.0  eta=0.721
  tau=0.200  N_obs=23.0  eta=0.733
  tau=0.150  N_obs=30.4  eta=0.726

eta = N_obs / N_max converges to ~0.747 (Renyi's constant for 1D RSA).

Runtime: ~5 seconds.
Dependencies: numpy only.
"""
import numpy as np

def rsa_capacity(HS=2, tau=0.301, steps=2000, seed=0):
    rng = np.random.default_rng(seed)
    centroids = []                          # list of unit vectors

    for _ in range(steps):
        v = rng.standard_normal(HS)
        v /= np.linalg.norm(v)             # uniform on S^{HS-1}

        if len(centroids) == 0:
            centroids.append(v)
            continue

        C = np.array(centroids)
        dists = np.linalg.norm(C - v, axis=1)
        if dists.min() > tau:              # minimum-separation rule
            centroids.append(v)            # open new slot (RSA acceptance)
        else:
            # soft EMA update of nearest centroid (lr=0.01)
            i = dists.argmin()
            centroids[i] += 0.01 * (v - centroids[i])
            centroids[i] /= np.linalg.norm(centroids[i])

    return len(centroids)

# Threshold sweep: reproduce Table 1
N_max = lambda tau: np.pi / np.arcsin(tau / 2)   # circle packing bound
for tau in [0.301, 0.200, 0.150]:
    obs = np.mean([rsa_capacity(tau=tau, seed=s) for s in range(5)])
    eta = obs / N_max(tau)
    print(f"tau={tau:.3f}  N_obs={obs:.1f}  eta={eta:.3f}")
# Expected output:
# tau=0.301  N_obs=15.0  eta=0.721
# tau=0.200  N_obs=23.0  eta=0.733
# tau=0.150  N_obs=30.4  eta=0.726
