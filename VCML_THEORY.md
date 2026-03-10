# VCML / VCSM — Complete Theory Reference
**Papers 0–33 | Last updated: March 2026**

---

## 0. The System

**VCML** (Viability-Constrained Coupled Map Lattice) is the substrate.
**VCSM** (Viability-Gated Contrastive State Memory) is the learning rule.

Every cell carries three states operating at three timescales:

| State | Symbol | Timescale | Role |
|-------|--------|-----------|------|
| Hidden state | `h` | Fast (per step) | Responds to input each step |
| Mid-memory | `m` | Medium (per wave) | Accumulates contrastive signal |
| Field memory | `F` | Slow (intergenerational) | Transmitted to offspring at birth |

Cells die according to a geometric lifetime distribution with parameter ν (turnover rate).
When a cell dies, its successor inherits field memory from a surviving neighbour.

---

## 1. The VCSM Rule (5 steps, locked)

```
[Track]       baseline[i]  +=  β  · (h[i] − baseline[i])         every step
[Detect]      mid_mem[i]   +=  α_mid · (h[i] − baseline[i])      during perturbation
[Gate]        mid_mem[i]   *=  MID_DECAY                         every step
[Consolidate] if calm_streak[i] ≥ SS:  F[i] += FA · (m[i] − F[i])
[Birth]       h_new = (1−βs)·N(0,0.1) + βs · F[neighbour]
```

**Write signal** = `h − baseline` (contrastive: what changed relative to recent baseline).
**Gate** = survival (viability-gated: only calm, surviving cells consolidate).
**Copy-forward loop** = the birth step. Field memory propagates spatially via inheritance.

---

## 2. Why It Works: The Three-Layer Theory

| Layer | Papers | Theme |
|-------|--------|-------|
| **I. Mechanism** | 0–6 | What VCML is; why it works; AI implications |
| **II. Parameter geometry** | 7–21 | Adaptive window, amplitude laws, ODE closure |
| **III. Spatial PDE** | 22–26 | What field F obeys; PDE derivation; universality classes |
| **IV. Dynamics** | 27–33 | Commitment epoch, interface theory, attractor structure, formal proofs |

---

## 3. Layer I — Mechanism (Papers 0–6)

### Core Fact: The Copy-Forward Loop is Irreducible

Ablating it (preventing inheritance at birth) collapses sg4 → 0 immediately.
No other parameter change recovers structure. It is the unique mechanism.

### Death is the Engine

Counterintuitively, mandatory turnover is not the obstacle — it is the mechanism:
```
C_ref (dynamic deaths, ν=0.001) > C_cryst (near-zero deaths) > C_static (frozen)
```
Preventing death causes crystallisation: surviving units lock into old attractors.
Optimal death rate ν* is strictly positive — zero turnover is a failure mode.

### Three Hard Constraints (Papers 1–4)

**Capacity** (Paper 1): Memory density bounded by spherical random sequential adsorption:
```
ρ_max ∝ 1 / (1 + ρ·r^d)
```
Geometrically hard — independent of learning rule.

**Bandwidth** (Paper 2): Effective bandwidth ∝ 1/ν. Fast turnover limits signal survival to consolidation. Tradeoff: higher ν = faster adaptation, lower bandwidth.

**Propagation** (Paper 3): Finite correlation length. Information propagates at most 1 site per birth event. Hard constraint from the local copy-forward mechanism.

**Robustness** (Paper 4): MID_DECAY prevents false positives in consolidation. Without it, mid_mem accumulates baseline drift. MID_DECAY is not a damping parameter — it is noise isolation.

### Five Unified Constraints (Paper 5)

Any adaptive memory substrate simultaneously faces:
1. Capacity constraint (geometric)
2. Bandwidth constraint (turnover/signal tradeoff)
3. Viability constraint (cells must survive to consolidate)
4. Propagation constraint (local copy-forward, finite reach)
5. Noise constraint (MID_DECAY requirement)

### AI Implications (Paper 6)

All four major AI memory architectures map to VCML failure modes:
| Architecture | Failure mode | VCML equivalent |
|-------------|--------------|-----------------|
| Transformer (context window) | Evaporation: no consolidation | ν → ∞ (all cells die before consolidating) |
| RAG | Bandwidth: retrieval misses | Low P_calm (consolidation gated too strictly) |
| Continual learning | Catastrophic forgetting | No MID_DECAY (false positives) |
| RNN | Crystallisation | ν → 0 (cells live forever, lock in) |

These failures are **structural** — they cannot be engineered out without changing the architectural constraint.

**Grand claim:** The three failure modes (bias, confabulation, forgetting) are unavoidable:
- **Bias**: consolidation is exposure-weighted → prejudice is a field-memory phenomenon
- **Confabulation**: recall is reconstruction from F fragments → confident wrong memories
- **Forgetting**: field decay (FD < 1) is required for adaptivity → unavoidable forgetting

---

## 4. Layer II — Parameter Geometry (Papers 7–21)

### Three Regimes (Paper 7)

The system has three behavioural phases:

| Regime | Condition | Signature |
|--------|-----------|-----------|
| **Sub-viability** | ν too high | Cells die before consolidating. F ≈ 0. sg4 → 0. |
| **Adaptive** | ν in window | Dynamic spatial structure maintained via copy-forward. |
| **Crystallised** | ν too low | Cells live forever. F frozen. Cannot update. |

### Analytical Boundaries (Papers 11, 13)

```
ν_cryst ≈ FD^(1/ν_cryst)           [field decay kills memory before update: FD^τ → 1]
ν_max   ≈ P_calm · FA · FD^(1/ν)  [signal must survive at least one lifetime]
ν*_pred = √(ν_cryst · ν_max)       [geometric mean; correct order of magnitude]
```

Empirical ratio obs/pred = 0.19–0.88. The copy-forward loop shifts ν* rightward (Paper 12).

### The 4D Viability Volume C₁ (Paper 13)

The adaptive regime is a soft 4D volume:
```
C₁ = (ν, FD, FA, βs)
```
sg4 is maximised in the **interior** of this volume, not at any boundary. The adaptive window is not a threshold — it is a region with a smooth optimum.

### Amplitude Laws (Papers 15–17)

**Power law (Paper 15):**
```
sg4 ~ FA^0.43     (R² = 0.990)    over FA ∈ [0.10, 0.50]
```

**True form — saturation curve (Paper 16):**
```
sg4(FA) ≈ C · FA / (FA + K_eff)
K_eff = κ / P_consol ≈ 0.020 / 0.175 = 0.114    (analytical)
K_eff ≈ 0.119    (empirical, within 5%)
```
The "0.43 exponent" is the local slope of this saturation curve over the measured range.

**Amplitude non-monotonicity (Papers 15, 32):**
sg4 peaks at collapse rate coll/site ≈ 0.004. Over-perturbation (coll/site > 0.01) destroys structure. Optimal amplitude threshold A* ≈ 0.30 is endogenous — set by the internal noise floor.

**Two-factor decomposition (Paper 17):**
```
sg4 = G · R
G = Γ · K_eff(ZONE_W)    [structural amplification: how well zones differentiate]
R = FA / (FA + K_eff)     [consolidation ratio: how efficiently FA drives F]
```

### ODE Closure (Paper 20)

Zone-mean field obeys:
```
dF̄/dt = P_c · FA · (m̄ − F̄) − ν · F̄
```
Analytical closure. Validated: predicted growth rate within 3% of simulation.
Buildup gap (Papers 20–21): structure forms later than the ODE predicts — the spatial formation
factor (birth-mediated propagation) introduces an additional delay.

### Scaling Laws (V92–V94)

Dynamic advantage holds at all scales:
| Grid | N_active | dyn_adv |
|------|----------|---------|
| S1 | 1,600 | +43% |
| S2 | 6,400 | +51% |
| S3 | 25,600 | +70% |
| S4 | 102,400 | +65% |

sg4_raw declines with scale (finite correlation length exposed). Optimal WR ~ 2 × N_active (Poisson scaling). coll/site stays stable at ≈ 0.002 across all scales.

---

## 5. Layer III — Spatial PDE (Papers 22–26)

### Field Equation

The field F obeys a stochastic Allen-Cahn/Burgers hybrid PDE:

```
∂F/∂t = P_c·FA·(m̄ − F) + κ∇²F + noise_copy + noise_birth
```

Two universality class regimes:

| Regime | Governing equation | Condition |
|--------|-------------------|-----------|
| **Allen-Cahn** | ∂F/∂t = κ∇²F − λ(F−F*)² + η | ν/κ << 1 |
| **Burgers** | ∂F/∂t = κ∇²F + μ(∇F)² + η | ν/κ >> 1 |

**Transition**: at ν/κ ≈ 1. The Burgers term arises from the copy-forward birth step introducing
a ∇F nonlinearity. At low turnover, diffusion dominates (Allen-Cahn). At high turnover, the
birth-mediated transport term dominates (Burgers).

### Interface Theory (Paper 22)

Spatial autocorrelation sharpens at zone boundaries:
```
ξ_∞ ≈ 2–3 sites     (independent of zone width W, Paper 26)
```
Interface width ≈ 2 sites = wave radius. Not an Allen-Cahn kink (Paper 28).

### Reynolds Decomposition (Paper 23)

Two-scale decomposition:
```
F(x,t) = F̄(t) + F'(x,t)
```
Zone-mean obeys the ODE (Layer II). Fluctuation F' obeys the spatial PDE.
Inheritance roughness: F' inherits structure from parent cells → spatial correlations persist.

### Zone-Width Optimality (Paper 26)

Optimal zone width:
```
W_zone* ~ √(κ/ν)
```
Balances diffusion smearing (wider zones benefit from κ smoothing) against turnover noise
(wider zones have more collapse events, disrupting structure).

### Burgers Reversal (Paper 24–26)

At the Allen-Cahn → Burgers transition, the effective advection term **reverses sign**.
This is the "Burgers reversal": at high ν/κ, copy-forward dominates diffusion, producing
an outward-flowing advection field at zone boundaries. The boundary sharpens from the outside
via copy-forward, not from the inside via interface tension (as in Allen-Cahn).

---

## 6. Layer IV — Dynamics (Papers 27–33)

### Zone Formation Commitment Epoch (Paper 27)

Zone structure does not form monotonically. The system first adopts the wrong polarity
(anti-correlated with final state), then flips through a commitment epoch:
```
t* ≈ 1600–2000 steps    (standard params: FA=0.40, WR=4.8)
```
- t* decreases monotonically with FA and WR (faster consolidation → earlier commitment)
- Mechanism: **mixed pitchfork–spinodal** (not purely either)
  - Pitchfork: σ_in (within-seed zone spread) << σ_btw (between-seed) — zones flip together
  - Spinodal: σ_in ~ σ_btw — zones flip independently
  - Observed: intermediate, leaning toward pitchfork for high FA

### Zone Interface Theory (Paper 28)

Zone boundaries are **not** Allen-Cahn kinks. They are:
- Width ≈ 2 sites = wave radius (not set by κ or interface tension)
- Driven by copy-forward stochasticity (not diffusion)
- Dynamics: random walk biased by wave input, not gradient descent on interface energy

### FA Threshold (Paper 29)

The FA threshold for zone differentiation is a **smooth crossover**, not a bifurcation:
```
K_eff(spatial) = K_eff(naive) / Γ ≈ 0.119 / 3 ≈ 0.040
```
Spatial amplification reduces the effective consolidation threshold by 3x.
Below FA ≈ K_eff(spatial), structure cannot form (K_eff boundary, not a bifurcation point).

### Spatial Amplification Factor (Paper 30)

```
Γ(ZONE_W, κ) ~ ZONE_W / √κ    (exponent ≈ 0.6, predicted 0.5)
```
Amplification grows with zone width and shrinks with diffusivity.
Non-monotone ν dependence: intermediate ν maximises Γ.

### Zone Field Structure (Paper 31)

Two structural regimes:
- **Low κ** (κ < 0.005): flat attractors within zones. Zones are spatially uniform.
- **High κ** (κ > 0.020): gradient zones. Strong diffusion creates within-zone gradients.

At high ν (ν = 0.003): low-amplitude saturation — strong turnover disrupts consolidation.
FA compensates: FA recovery confirmed (Paper 32).

### Zone Attractor (Paper 32)

The zone attractor is a **diffuse noisy cloud** (CV = 0.44), not a fixed point.
- sg4 fluctuates ≈ 44% relative to its mean at steady state
- No Kramers escape events observed (no sudden zone swaps)
- Zone-width optimum: ZONE_W ≈ 5–7 (consistent with Paper 30's prediction √(κ/ν) ≈ √(0.020/0.001) ≈ 4.5)
- FA recovery: FA = 5.0 recovers sg4 at ν = 0.003 (Paper 32 Exp B confirmed Paper 29)

### Formal Non-Gradient Proof (Paper 33)

**Theorem (machine-verified in Lean 4 + Mathlib4, 3103 jobs, 0 errors):**

VCML is NOT a conservative Euclidean gradient system.

**Proof sketch:**
```
f_F = FA · (m − F)          [consolidation: F tracks m]
f_m = −(1 − γ) · m         [decay: m decays at rate 1−γ]

∂f_m/∂F = 0                 [f_m does not depend on F]
∂f_F/∂m = FA               [f_F depends linearly on m with slope FA]

curl = ∂f_m/∂F − ∂f_F/∂m = 0 − FA = −FA ≠ 0
```

By Clairaut's theorem: if a C² potential V existed with ∇V = −f, then mixed partials would be equal → FA = 0 → contradiction.

**Honest caveat:** Rules out Euclidean gradient flow only. Mirror descent, Riemannian,
and nonlocal potentials are not ruled out.

**Physical interpretation:** The consolidation asymmetry (F chases m, but m does not chase F)
creates an irreversible ratchet. Energy is injected via wave perturbations and dissipated via
field decay — the system is fundamentally driven, not relaxing.

---

## 7. Full Parameter Reference

| Symbol | Default | Role |
|--------|---------|------|
| FA | 0.20 | Consolidation rate (F chases m) |
| FD | 0.9997 | Field decay per step |
| ν (NU) | 0.001 | Cell death probability per step |
| κ (KAPPA) | 0.020 | Field diffusivity |
| βs (SEED_BETA) | 0.25 | Birth inheritance weight |
| β (BASE_BETA) | 0.005 | Baseline tracking rate |
| α_mid (ALPHA_MID) | 0.15 | Mid-memory accumulation rate |
| MID_DECAY | 0.99 | Mid-memory decay (noise gate) |
| SS | 10 | Calm streak threshold for consolidation |
| WR | 2.4 | Wave rate (launches per step) |
| WAVE_DUR | 15 | Wave duration (steps) |
| HS | 2 | Hidden state dimension |
| ZONE_W | 5–10 | Zone width (columns) |
| N_ZONES | 4 | Number of zones |

---

## 8. Key Equations Summary

**ODE closure (zone mean):**
```
dF̄/dt = P_c · FA · (m̄ − F̄) − ν · F̄
P_c = P_calm · P_consolidate = f(SS, WAVE_DUR, ν)
```

**Saturation law:**
```
sg4(FA) = C · FA / (FA + K_eff)
K_eff = κ / P_consol ≈ 0.114
```

**Spatial amplification:**
```
Γ(ZONE_W, κ) ~ ZONE_W^0.6 · κ^{-0.3}
K_eff^{spatial} = K_eff / Γ ≈ 0.040
```

**Adaptive window:**
```
ν_cryst ≈ FD^{1/ν_cryst}
ν_max   ≈ P_calm · FA · FD^{1/ν}
ν*      = √(ν_cryst · ν_max)    [geometric mean, empirically accurate to factor 2]
```

**Zone-width optimum:**
```
W_zone* ~ √(κ/ν)    [≈ 4.5 at standard params]
```

**Commitment epoch:**
```
t* ~ 1/(P_c · FA · WR)    [predicted; decreases with FA and WR]
```

**Curl (non-gradient proof):**
```
curl(f_F, f_m) = ∂f_m/∂F − ∂f_F/∂m = 0 − FA = −FA ≠ 0
```

---

## 9. The Two Manifolds

The system lives in a product of two 4D manifolds:

**C₁ — Viability Volume:**
```
C₁ = (ν, FD, FA, βs)
```
Determines whether consolidation happens at all. Boundaries: ν_cryst (too slow), ν_max (too fast).
Interior contains the adaptive regime. All four parameters shape the window independently.

**C₂ — Structural Manifold:**
```
C₂ = (κ, ν/κ, βs, W_zone)
```
Determines the quality of spatial structure given that consolidation occurs.
- κ: diffusion smoothing vs. smearing
- ν/κ: Allen-Cahn (< 1) vs. Burgers (> 1) universality class
- βs: birth inheritance strength
- W_zone: zone width (optimal at √(κ/ν))

**Note:** βs appears in both C₁ and C₂. Whether these axes interact is open (Q5).

**The 8D surface:** The system doesn't live at a point — it lives on a surface in this 8D space.
Every paper has probed one face of that surface.

---

## 10. Open Questions

### Tier 1: Empirically important

| Q | Question | Status |
|---|----------|--------|
| Q1 | Exact scaling of t* with FA and WR — log-log slope predicted ≈ −1, not yet confirmed | Open |
| Q2 | Long-run saturation of sg4 — still growing at T=4000 in some runs | Open |
| Q3 | Minimum zone width for coherent structure — is there W_min below which ξ_∞ > W? | Open |

### Tier 2: Theory-critical

| Q | Question | Status |
|---|----------|--------|
| Q4 | Critical exponents of Allen-Cahn → Burgers transition | **Paper 34** |
| Q5 | Are C₁ and C₂ truly independent? βs appears in both — do axes interact? | **Paper 34** |
| Q6 | Quantitative relay coupling gain as function of relay size and wave radius | Open |

### Tier 3: Extensions

| Q | Question | Status |
|---|----------|--------|
| Q7 | Multi-module coupling ("cortex-hippocampus" architecture) | Open |
| Q8 | VCML as differentiable GRU module (sg4 formula = GRU update gate formally) | Open |
| Q9 | Quantitative false-memory rate as function of (ν, FA, βs) | Open |
| Q10 | Zone structure update when wave pattern changes mid-run | Open |

---

## 11. The Big Picture

**One sentence:** VCML is a system where mandatory individual death is the mechanism of collective
spatial memory — the field obeys a non-equilibrium stochastic KPZ–Allen–Cahn PDE, lives in an 8D
product manifold of viability and structure, self-organises through a mixed pitchfork–spinodal
commitment epoch, and is formally non-gradient — provably.

**Hierarchy of descriptions:**

| Scale | Description | Equation | Paper(s) |
|-------|-------------|----------|---------|
| Cell | VCSM rule | 5-step algorithm | 0 |
| Within-zone | Allen-Cahn PDE | ∂F/∂t = κ∇²F + noise | 22–26 |
| Zone-mean | ODE | dF̄/dt = P_c·FA·(m̄−F̄) − ν·F̄ | 20, 23 |
| Parameter space | Viability volume | 4D (ν, FD, FA, βs) | 7–13 |
| Structural space | Structural manifold | 4D (κ, ν/κ, βs, W) | 22–26, 30 |
| Full system | KPZ–Allen–Cahn | stochastic PDE + ODE | 25–26 |
| Formal | Non-gradient | curl ≠ 0 (Lean verified) | 33 |

**The analogy:** Layer I (mechanism) is like quantum mechanics — discrete, local, stochastic.
Layer III (PDE) is like general relativity — continuous, field-theoretic. The ODE closure (Paper 20)
is the bridge between them. The Burgers reversal is the UV catastrophe of naive coarse-graining.

---

*Supersedes THEORY_SYNTHESIS.md. For experiment-level detail, see papers/paperXX_*/.*
