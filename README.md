# Adaptive Memory Theory

**Author:** [Gabriel Aubin-Moreau](https://github.com/AccidentalGenius101)

A constraint-based framework for adaptive memory systems, grounded in VCSM (Viability-Gated Contrastive State Memory) and instantiated in VCML (Viability-Constrained Coupled Map Lattice).

**Core idea:** Mandatory individual turnover is the learning mechanism, not an obstacle to overcome. VCML cells live, learn, collapse, and pass their field state to successors. Spatial structure emerges and is maintained without any global coordination. Paper 6 extends the framework to artificial AI architectures, deriving a constraint-based theory of adaptive memory that predicts the failure modes of transformers, RAG, continual learning, and recurrent networks.

---

## Papers

| # | Title | PDF |
|---|-------|-----|
| 0 | Spontaneous Spatial Self-Organization in Viability-Gated Memory Substrates | [paper0_bootstrap.pdf](papers/paper0_bootstrap.pdf) |
| 1 | Memory Capacity Bounds via Spherical Random Sequential Adsorption | [paper1_capacity.pdf](papers/paper1_capacity.pdf) |
| 2 | Memory Bandwidth as a Function of Turnover Rate | [paper2_bandwidth.pdf](papers/paper2_bandwidth.pdf) |
| 3 | Finite Correlation Length and Propagation Constraints in Adaptive Substrates | [paper3_propagation.pdf](papers/paper3_propagation.pdf) |
| 4 | Consolidation Gate and Noise Robustness in VCSM | [paper4_stability.pdf](papers/paper4_stability.pdf) |
| 5 | A Unified Physical Theory of Viability-Gated Adaptive Systems | [paper5_unified.pdf](papers/paper5_unified.pdf) |
| 6 | Adaptive Memory in Artificial Systems: A Constraint-Based Framework | [PDF](papers/paper6_adaptive_memory_theory/paper6_adaptive_memory_theory.pdf) · [TeX](papers/paper6_adaptive_memory_theory/paper6_adaptive_memory_theory.tex) · [Code](papers/paper6_adaptive_memory_theory/) |

---

## Quick start (~30 seconds)

```
py minimal/minimal_vcsm.py
```

Expected output:

```
VCSM minimal demonstration
Grid: 10x10 active sites, 4 zones, 800 steps, 5 seeds

Standard  (copy-forward ON):   sg4 = 0.0150 +- 0.0050
Ablation  (copy-forward OFF):  sg4 = 0.0005 +- 0.0004

Standard sg4 >> Ablation sg4: spatial structure requires copy-forward loop.
This confirms Paper 0 Fact 3: turnover is the mechanism, not the obstacle.
```

Requires: Python 3.9+, numpy. No other dependencies.

---

## Reproduce paper results

```
cd reproduce
py paper0_repro.py    # ~8 min
py paper1_repro.py    # ~5 sec  -- exact match
py paper2_repro.py    # ~30 sec
py paper3_repro.py    # ~5 min
py paper4_repro.py    # ~3 min
py paper5_repro.py    # ~4 min
```

See [`reproduce/README.md`](reproduce/README.md) for expected outputs and full runtime table.

---

## Use the framework (vgrt)

```python
from vgrt import VCSMLattice, VCSMConfig

cfg = VCSMConfig()
lat = VCSMLattice(cfg)
for _ in range(cfg.STEPS):
    lat.step()
print(lat.sg4())
```

---

## Repository structure

```
adaptive-memory-theory/
├── papers/
│   ├── paper0_bootstrap.*        Paper 0: spontaneous self-organization
│   ├── paper1_capacity.*         Paper 1: capacity law
│   ├── paper2_bandwidth.*        Paper 2: bandwidth law
│   ├── paper3_propagation.*      Paper 3: propagation constraint
│   ├── paper4_stability.*        Paper 4: consolidation gate
│   ├── paper5_unified.*          Paper 5: unified theory
│   └── paper6_adaptive_memory_theory/
│       ├── paper6_adaptive_memory_theory.pdf   Paper 6 (AI architectures)
│       ├── paper6_adaptive_memory_theory.tex
│       ├── paper6_exp1_phase_diagram.py
│       ├── paper6_exp2_scale_limit.py
│       ├── paper6_exp3_adaptive_vs_frozen.py
│       └── paper6_exp4_frozen_transition.py
├── minimal/           30-second self-contained demonstration (numpy only)
├── vgrt/              Core framework: VCSMConfig, VCSMLattice, WaveEnvironment
│   └── substrates/
└── reproduce/         Reproduction scripts for each paper's key experiments
```

---

## Key ideas

**The copy-forward loop** is the load-bearing mechanism: when a cell collapses, its successor inherits the local field memory (`fieldM`) from a surviving neighbor. This intergenerational transmission maintains spatial structure even under continuous turnover. Ablating it (Fact 3, Paper 0) collapses `sg4` to zero.

**Three timescales** operate simultaneously:
1. Fast: hidden state `hid` responds to perturbations each step
2. Medium: `mid_mem` accumulates contrastive signal across a wave episode
3. Slow: `fieldM` consolidates only when the site has been calm for `SS` consecutive steps (the consolidation gate)

**Turnover is not noise** -- it is the mechanism. Higher turnover rates give more copy-forward events per unit time, refreshing spatial structure. Too little turnover leads to crystallization; too much causes chaotic erasure. The adaptive phase sits between these extremes.

---

## Citation

If you use this work, please cite the relevant paper (see [`CITATION.cff`](CITATION.cff)):

```bibtex
@article{aubin2026vcsm_p0,
  author  = {Gabriel Aubin-Moreau},
  title   = {Spontaneous Spatial Self-Organization in
             Viability-Gated Memory Substrates},
  year    = {2026},
  note    = {VCSM series, Paper 0}
}
```

---

## License

Code: MIT. Papers: CC BY 4.0. See [`LICENSE`](LICENSE).
