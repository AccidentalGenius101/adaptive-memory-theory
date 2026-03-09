"""
VCSM Core — Viability-Gated Contrastive State Memory

Shared learning rule logic and utilities for all substrate implementations.
Substrate-specific classes (VCSMLattice, VCSMGraph, VCSMEmbedding) are in substrates/.

================================================================================
VCSM LEARNING RULE (substrate-agnostic)
================================================================================

The four VCSM steps (all local, all unsupervised):

  [Calm]        baseline[i] += BETA * (hid[i] - baseline[i])
                Each site tracks its own running average. Half-life ~34 steps.

  [Perturb]     mid_mem[i] += ALPHA_MID * (hid[i] - baseline[i])
                mid_mem[i] *= MID_DECAY
                Accumulates the CONTRASTIVE DEVIATION — how much the site
                departs from its own baseline. Half-life ~23 steps.

  [Consolidate] IF survived AND calm_streak[i] >= STABLE_STEPS:
                    fieldM[i] += FIELD_ALPHA * (mid_mem[i] - fieldM[i])
                Writes surviving mid-term evidence into slow memory.
                The update is (mid_mem - fieldM): prediction error.
                fieldM converges toward what mid_mem has been observing.

  [Birth]       hid[i] = blend(debris_hid, fieldM[location], FIELD_SEED_BETA)
                New sites inherit location-appropriate slow memory at birth.
                This is the intergenerational seeding step.
"""

import random
import math
from statistics import mean


# ============================================================
# Utility functions (shared across substrates)
# ============================================================

def l2(v):
    """L2 norm of vector v."""
    return math.sqrt(sum(x * x for x in v))


def dot(a, b):
    """Dot product of vectors a and b."""
    return sum(a[k] * b[k] for k in range(len(a)))


def sigmoid(x):
    """Sigmoid function with clipping."""
    x = max(-8.0, min(8.0, x))
    return 1.0 / (1.0 + math.exp(-x))


def tanh_clipped(x):
    """Tanh function with clipping."""
    x = max(-8.0, min(8.0, x))
    e2 = math.exp(2 * x)
    return (e2 - 1.0) / (e2 + 1.0)


# ============================================================
# GRU cell (substrate-agnostic)
# ============================================================

def make_gru_weights(HS, input_size, scale=0.3):
    """Initialize GRU weight matrices (random Gaussian)."""
    def rand_matrix(r, c):
        return [[random.gauss(0, scale) for _ in range(c)] for _ in range(r)]
    return {
        "Wz": rand_matrix(HS, input_size), "Uz": rand_matrix(HS, HS), "bz": [0.5] * HS,
        "Wr": rand_matrix(HS, input_size), "Ur": rand_matrix(HS, HS), "br": [0.5] * HS,
        "Wh": rand_matrix(HS, input_size), "Uh": rand_matrix(HS, HS), "bh": [0.0] * HS,
        "Wo": [[random.gauss(0, 0.1) for _ in range(HS)]],
        "bo": [0.0],
    }


def copy_gru_weights(w):
    """Deep copy of GRU weight dictionary."""
    out = {}
    for k, v in w.items():
        if isinstance(v[0], list):
            out[k] = [row[:] for row in v]
        else:
            out[k] = v[:]
    return out


def gru_step(weights, h, x, HS, IS):
    """
    Standard GRU update (flat loops for Python performance).

    Args:
        weights: GRU weight dictionary
        h: hidden state (list of HS floats)
        x: input vector (list of IS floats)
        HS: hidden size
        IS: input size

    Returns:
        (new_h, output_scalar)
    """
    Wz = weights["Wz"]; Uz = weights["Uz"]; bz = weights["bz"]
    Wr = weights["Wr"]; Ur = weights["Ur"]; br = weights["br"]
    Wh = weights["Wh"]; Uh = weights["Uh"]; bh = weights["bh"]
    Wo = weights["Wo"][0]; bo0 = weights["bo"][0]

    z = [0.0] * HS
    r = [0.0] * HS
    g = [0.0] * HS

    for i in range(HS):
        acc = bz[i]
        Wz_i = Wz[i]; Uz_i = Uz[i]
        for j in range(IS): acc += Wz_i[j] * x[j]
        for j in range(HS): acc += Uz_i[j] * h[j]
        z[i] = sigmoid(acc)

    for i in range(HS):
        acc = br[i]
        Wr_i = Wr[i]; Ur_i = Ur[i]
        for j in range(IS): acc += Wr_i[j] * x[j]
        for j in range(HS): acc += Ur_i[j] * h[j]
        r[i] = sigmoid(acc)

    for i in range(HS):
        acc = bh[i]
        Wh_i = Wh[i]; Uh_i = Uh[i]
        for j in range(IS): acc += Wh_i[j] * x[j]
        for j in range(HS): acc += Uh_i[j] * (r[j] * h[j])
        g[i] = tanh_clipped(acc)

    h_new = [(1.0 - z[i]) * h[i] + z[i] * g[i] for i in range(HS)]
    out = tanh_clipped(sum(Wo[k] * h_new[k] for k in range(HS)) + bo0)
    return h_new, out
