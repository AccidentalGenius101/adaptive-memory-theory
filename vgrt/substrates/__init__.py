"""
VCSM Substrate Implementations

Three concrete realizations of the VCSM learning rule on different substrates:
  - VCSMLattice (VCSM-L): 2D coupled-map lattice (primary, Track 1)
  - VCSMGraph (VCSM-G): Multi-region geographic relay coupling
  - VCSMEmbedding (VCSM-E): Real embeddings from sentence-transformers
"""

from .lattice import VCSMLattice, WaveEnvironment

__all__ = [
    "VCSMLattice",
    "WaveEnvironment",
]
