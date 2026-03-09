"""
VCSM Configuration

Hyperparameters for Viability-Gated Contrastive State Memory learning rule.
Substrate-agnostic: these parameters apply to all implementations
(lattice, graph, embedding).

All values are Track 1 validated defaults.
"""


class VCSMConfig:
    """
    All hyperparameters for VCSM systems.
    Values are the Track 1 validated defaults.
    """

    # Grid geometry (substrate-specific, overridden per substrate)
    W: int = 40
    H: int = 20
    HALF: int = 20          # PASSIVE: x<HALF, ACTIVE: x>=HALF

    # GRU
    HS: int = 5             # hidden state dimension
    INPUT_SIZE: int = 3     # input feature dimension

    # Viability bounds (hard boundary: collapse if vals exits this range)
    BOUND_LOW:  float = 0.05
    BOUND_HIGH: float = 0.95
    NEAR_LOW:   float = 0.15
    NEAR_HIGH:  float = 0.85

    # Incremental vals dynamics (restores CML behaviour).
    # vals[i] is a dynamical state variable, NOT a direct GRU readout.
    VALS_DECAY: float = 0.92    # per-step persistence (half-life ~12 steps)
    VALS_NAV:   float = 0.08    # weight of neighbour mean in vals update
    ADJ_SCALE:  float = 0.03    # GRU adjustment scaling

    # Instability-based collapse (secondary mechanism, enforces T_agent < T_field)
    INST_COLLAPSE_THRESH: float = 0.45  # sum |hid-baseline| threshold
    INST_COLLAPSE_PROB:   float = 0.03  # probability per step when above threshold

    # Debris pool
    MAX_RESIDUE: int = 5            # max debris snapshots per site
    SEED_FRAG_NOISE: float = 0.01   # noise added to inherited hid at birth

    # --- VCSM timescales ---

    # [Calm] baseline tracker
    BASELINE_BETA: float = 0.02     # ~half-life 34 steps

    # [Perturb] mid-term contrastive accumulator
    ALPHA_MID:  float = 0.15        # write rate into mid_mem
    MID_DECAY:  float = 0.97        # per-step decay (~half-life 23 steps)

    # [Consolidate] slow memory
    FIELD_ALPHA:        float = 0.08    # write rate into fieldM
    FIELD_DECAY:        float = 0.9992  # per-step decay (~half-life 866 steps)
    FIELD_SEED_BETA:    float = 0.25    # birth blend: 25% fieldM, 75% debris
    STABLE_STEPS:       int   = 20      # calm streak required before consolidation

    # Diffusion (spatial smoothing of fieldM)
    FIELD_DIFFUSE_RATE: float = 0.02    # fraction of neighbour mean blended in
    FIELD_SMOOTH_EVERY: int   = 2       # diffuse every N steps

    # Zone classification
    ZONE_K: int = 80        # number of sites in stable/churn zone each
