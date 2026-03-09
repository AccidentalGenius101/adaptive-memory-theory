"""
VGRT (Viability-Gated Routing with Turnover) Framework

A mechanism for learning and adaptation under mandatory turnover.
VCSM (Viability-Gated Contrastive State Memory) is the concrete learning rule
implementing VGRT on various substrates.

Three substrate implementations:
  - VCSMLattice (VCSM-L): 2D coupled-map lattice
  - VCSMGraph (VCSM-G): Multi-region geographic relay
  - VCSMEmbedding (VCSM-E): Real embeddings

Core exports:
"""

from .config import VCSMConfig
from .vcsm_core import (
    l2, dot, sigmoid, tanh_clipped,
    make_gru_weights, copy_gru_weights, gru_step,
)
from .substrates import VCSMLattice, WaveEnvironment

__all__ = [
    "VCSMConfig",
    "l2", "dot", "sigmoid", "tanh_clipped",
    "make_gru_weights", "copy_gru_weights", "gru_step",
    "VCSMLattice",
    "WaveEnvironment",
]
