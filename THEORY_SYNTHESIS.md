# VCML / VCSM Theory Synthesis
**Papers 0–26 — What we know, what we've derived, what's still open**

*Last updated: March 2026*

---

## 0. The System

**VCML** = Viability-Constrained Coupled Map Lattice (substrate).
**VCSM** = Viability-Gated Contrastive State Memory (learning rule).

Every cell carries a hidden state `h` (fast), a mid-memory `m` (medium), and a field `F`
(slow, intergenerational).  Cells have a finite lifetime drawn from Geometric(ν).  When a
cell dies it is replaced by a newborn seeded from a neighbour's field.

**The VCSM rule (5 steps, locked):**
```
[Track]       baseline[i]  +=  β · (h[i] − baseline[i])
[Detect]      mid_mem[i]   +=  α_mid · (h[i] − baseline[i])     (during perturbation)
[Gate]        mid_mem[i]   *=  MID_DECAY                        (every step)
[Consolidate] if calm_streak[i] ≥ SS:  F[i] += FA · (m[i] − F[i])
[Birth]       h_new = (1−βs)·N(0, 0.1) + βs · F[neighbour]
```

Write signal = `h − baseline` (contrastive).
Gate = survival (Viability-Gated).
The copy-forward loop at birth is **the** spatial memory mechanism.

---

## 1. Three-Layer Theory Structure

The 26 papers divide into three conceptually distinct layers:

| Layer | Papers | Theme |
|-------|--------|-------|
| **I. Mechanism** | 0–6 | What VCML is; why it works; AI implications |
| **II. Parameter geometry** | 7–21 | How the adaptive window is shaped; amplitude laws; ODE closure |
| **III. Spatial PDE** | 22–26 | What field F actually evolves as; universality classes; phase structure |

These layers are not sequential scaffolding — they are genuinely different levels of
description.  Layer I says "there is spatial self-organisation via copy-forward."
Layer II maps the parameter boundaries of that organisation.
Layer III says "the organisation obeys the stochastic KPZ–Allen–Cahn equation."

---

## 2. Layer I: Mechanism (Papers 0–6)

| Paper | Core result |
|-------|-------------|
| **0** | Spatial self-organisation exists. Copy-forward is necessary and sufficient (ablation: sg4 → 0 without it). |
| **1** | Capacity bound via spherical RSA. Memory density ∝ 1/(1 + ρ·r^d) — hard geometric limit independent of learning rule. |
| **2** | Bandwidth–turnover tradeoff: effective bandwidth ∝ 1/ν. Fast turnover limits how much signal survives to consolidation. |
| **3** | Finite correlation length. Information cannot propagate faster than 1 site/birth-event. Hard constraint. |
| **4** | MID_DECAY provides noise robustness. Without it, mid_mem accumulates baseline drift — false positives in consolidation. |
| **5** | Three constraints (capacity, bandwidth, viability) form a unified framework. Any memory substrate faces all three simultaneously. |
| **6** | Extension to AI: transformers, RAG, CL, RNNs all map to VCML failure modes. Failure modes are architectural, not design flaws. |

**Confirmed facts (Layer I):**
- Copy-forward loop is the irreducible mechanism. All structure flows through birth inheritance.
- Mandatory turnover is not the obstacle — it is *the mechanism*. Preventing death hurts structure (C_ref > C_cryst > C_static).
- Three failure modes are structurally unavoidable: bias (exposure-weighted consolidation), confabulation (reconstruction from fragments), forgetting (field decay needed for adaptivity).

---

## 3. Layer II: Parameter Geometry (Papers 7–21)

### 3.1 The Adaptive Window

The system has three regimes (Paper 7):
- **Sub-viability** (ν too high): cells die before consolidating. F ≈ 0.
- **Adaptive** (ν in window): spatial structure maintained dynamically.
- **Crystallised** (ν too low): cells live forever, lock into old attractors. Rigidity prevents both loss *and* gain.

Two analytically derived boundaries (Papers 11, 13):
```
ν_cryst  ≈  FD^(1/ν_cryst)                      [crystallisation: FD^(τ) → 1]
ν_max    ≈  P_calm · α_F · FD^(1/ν)             [consolidation: signal survives ≥ 1 lifetime]
```

Optimal turnover rate (Paper 13):
```
ν*_pred = √(ν_cryst · ν_max)    [geometric mean of window boundaries]
```
Empirical ratio obs/pred = 0.19–0.88. Geometric mean gives correct order of magnitude but
consistently overestimates (because copy-forward shifts ν* rightward — Paper 12).

**The 4D Viability Volume** (Paper 13):
```
C_1 = (ν, ν_cryst(FD), ν_max(P_calm, α_F), βs)
```
sg4 is maximised at the *interior* of this volume, not at any boundary.
The adaptive window is a soft 4D volume, not a sharp threshold.

### 3.2 Amplitude Laws

**Power-law regime (Paper 15):**
sg4 ~ FA^0.43 (R²=0.990) over FA ∈ [0.10, 0.50], standard params.

**Saturation formula (Paper 16):**
```
sg4(FA) ≈ C · FA / (FA + K_eff)
```
- K_eff = κ / P_consol ≈ 0.020 / 0.175 = 0.114 (analytical)
- K_eff ≈ 0.119 (empirical, within 5% of analytical)
- The "0.43 exponent" is the local slope of this saturation curve over [0.10, 0.50]
- Extending to FA=0.90: saturation confirmed (only 1.9× gain for 9× FA increase)
- Formally identical to the GRU update gate: z = σ(Wx + b). VCML discovers gated-memory dynamics through competition, not design.

**Anti-erasure effect (Paper 16):**
Phase 2 perturbations trigger births that *propagate* Phase 1 structure via copy-forward
rather than erasing it. Mixed-phase sg4 can exceed Phase 1 baseline by up to 16%.

**Two-factor decomposition (Paper 17):**
sg4 = factor₁(FA) × factor₂(ν/κ). The two factors are separable — FA controls amplitude,
ν/κ controls the within-zone coherence.

### 3.3 Dynamics

**Temporal structure (Paper 18):**
- Phase 1: consolidation burst (sg4 rises as mid_mem fills during perturbation)
- Phase 2: MID_DECAY-limited forgetting (exponential decay with τ = 1/|log(MID_DECAY)|)
- Recovery: copy-forward re-supplies structure from survivor field

**ODE closure (Paper 20):**
Four-reservoir model {F_active, F_blocked, F_mid, F_baseline} closed analytically.
Validated predictions: buildup rate, equilibrium sg4, saturation timescale.

**Spatial formation factor (Paper 21):**
Birth-mediated propagation is the primary spatial coupling mechanism.
The "buildup gap" (sg4 lags input onset) is explained by the birth rate × inheritance fidelity product.

**Blocked-site buffer (Paper 19):**
Third reservoir directly measured. Cells in streak<SS are a genuine information reservoir —
mid_mem accumulates during blocked phase, then consolidates in a burst when streak≥SS is reached.

---

## 4. Layer III: Spatial PDE Theory (Papers 22–26)

This layer answers: *what continuous PDE does F evolve under?*

### 4.1 The Two-Scale Reynolds Decomposition (Papers 22–23)

Decompose F into zone mean and within-zone residual:
```
F(x, t) = F̄_zone(t) + F_res(x, t)
```

**PDE 1** (zone mean, Paper 23):
```
dF̄/dt = P_c · FA · (m̄ − F̄) − ν · F̄
```
This is an ODE — no spatial derivatives. Zones evolve independently given m̄.

**PDE 2** (within-zone residual, Papers 22–23):
```
∂F_res/∂t = κ ∇²F_res − (ν + P_c·FA)·F_res
```
This is the **Allen–Cahn equation**: diffusive interface sharpening, ξ(t) decreasing
during zone formation. Confirmed in Paper 22 (ξ(t) signature = interface narrowing).

**Within-zone correlation length** (Paper 23):
```
ξ_∞ = √(κ / (P_c · FA))     [pure Allen–Cahn, βs = 0]
```

### 4.2 Full Stochastic KPZ–Allen–Cahn PDE (Papers 25–26)

Adding the stochastic source (inheritance roughness, βs) and fitness-proportionate birth (α):
```
∂F/∂t = κ∇²F + (α·ν/2)·(∇F)² + P_c·FA·(m−F) + √(ν·βs²·FA²)·η(x,t)
```
- α=0: Allen–Cahn (within-zone interface sharpening)
- α=1, κ large: Burgers / KPZ (fitness selection amplifies correlated structure)
- α=1, κ small: **REVERSAL** (extreme-value selection without diffusive smoothing)

### 4.3 Self-Consistency Equation for ξ_∞ (Papers 25–26)

```
P_c · FA · ξ² − ν·βs·|F̄|·ξ − κ = 0
```

Positive root:
```
ξ_∞ = [ν·βs·|F̄| + √((ν·βs·|F̄|)² + 4κ·P_c·FA)] / (2·P_c·FA)
```

- At βs=0: ξ_∞ = √(κ/P_c·FA)  — pure diffusion-consolidation balance
- At large βs: ξ_∞ ~ ν·βs·|F̄| / (P_c·FA)  — noise-dominated, linear in βs
- Noise threshold: ν·βs·|F̄| ≈ 2√(P_c·FA·κ)  — crossover near βs ≈ 0.15–0.20

Verified against Paper 25 data: 5.2× ratio of ξ²(βs=0.50)/ξ²(βs=0) matches quadratic curve.

### 4.4 The Burgers Reversal (Paper 26)

| ν/κ | Regime | Mechanism |
|-----|--------|-----------|
| < 1 | Normal: α=1 **helps** (ratio > 1) | Diffusion smooths extreme values; fitness selection amplifies correlated gradients |
| ≥ 1 | **Reversal: α=1 hurts** (ratio < 1) | Diffusion too weak; fitness selection creates high-variance but spatially incoherent fields |

Worst observed: ν/κ = 3.33, ratio = 0.417.

Standard parameters (ν=0.001, κ=0.020) → ν/κ = 0.05 → safely in Normal regime.

### 4.5 Zone Width Sweep (Paper 26)

- sg4 ∝ 1/W_zone (monotone: narrower zones → higher mean pairwise distance)
- ξ_∞ ≈ 2–3 sites, **independent** of W_zone
- Conclusion: within-zone structure is governed by κ alone; zone geometry is a separate parameter

### 4.6 Temporal Anti-Correlation and Zone Commitment (Paper 26)

Temporal correlation C(t_ref, T_end):
```
C(400,  3000) = −0.36   [strongly negative: pre-commitment]
C(1600, 3000) = −0.02   [near zero-crossing]
C(2000, 3000) = +0.03   [just committed]
C(2800, 3000) = +0.30   [locked]
```

Interpretation: the early zone pattern (t < 1600) is **anti-correlated** with the final
committed pattern. The system explores the "wrong" zone-polarity basin before tunnelling
to the correct one near t* ≈ 1600–2000. This is a **spatial bifurcation** happening within
a single run — the zone formation commitment epoch.

---

## 5. The Two 4D Manifolds

The project has discovered two distinct 4D parameter manifolds:

### Manifold 1: Viability Volume (Papers 7–13)
```
C_1 = (ν, FD, FA, βs)
```
Controls whether the adaptive regime is active at all.
Soft boundary surface in 4D parameter space.
Measured via sg4 vs. ablation.

### Manifold 2: Structural PDE Manifold (Papers 22–26)
```
C_2 = (κ, ν/κ, βs, W_zone)
```
Given that the adaptive regime is active, controls *what kind* of structure forms.
Different PDEs dominate in different regions.
Measured via ξ_∞, sg4, Burgers ratio, temporal correlations.

**Key insight (Paper 26 discussion):** These two manifolds share βs as a coordinate but are
otherwise genuinely independent. You can set the viability volume (fix ν, FA) and then
navigate the structural manifold (vary κ, W_zone) without leaving the adaptive regime.
This is why "each PDE isn't enough" — you're genuinely exploring a product of two 4D
manifolds: C_1 × C_2 ≅ 8D total. The system has no single-PDE description that captures all 8 axes.

---

## 6. Key Equations (Summary)

```
VCSM consolidation-diffusion balance:
  sg4_steady = C · FA / (FA + K_eff),   K_eff = κ / P_consol

Self-consistency (correlation length):
  P_c · FA · ξ² − ν·βs·|F̄|·ξ − κ = 0

Viability window:
  ν_cryst < ν < ν_max,   ν* ≈ √(ν_cryst · ν_max)

Allen-Cahn interface (βs=0):
  ξ_∞ = √(κ / (P_c · FA))

Burgers reversal threshold:
  ν/κ ≳ 1   →   fitness birth REDUCES ξ

Noise threshold (βs):
  ν·βs·|F̄| ≈ 2√(P_c·FA·κ)

Bandwidth-turnover:
  bandwidth ∝ 1/ν   [Paper 2]

Geometric mean law:
  ν* ≈ √(ν_cryst · ν_max)   [Paper 13, overestimates by 2–5×]
```

---

## 7. Confirmed Findings (Numbered)

1. **Spatial self-organisation requires copy-forward** (P0). Ablation (βs=0, no field inheritance) → sg4 → 0.
2. **Mandatory turnover is the learning mechanism** (P0, V77–V89). C_ref > C_cryst > C_static. Optimal coll/site ≈ 0.002–0.004.
3. **Finite correlation length is a hard constraint** (P3, P22–26). ξ_∞ set by κ/P_c·FA; cannot exceed zone width without geographic relay.
4. **Three failure modes are architectural** (P4–P6). Bias, confabulation, forgetting cannot be engineered out without breaking adaptivity.
5. **Adaptive window is a 4D viability volume** (P7–P13). No single parameter controls the regime; all four axes matter.
6. **Geometric mean law** (P13). ν* ≈ √(ν_cryst · ν_max) gives correct order of magnitude.
7. **sg4 saturation formula** (P16). sg4 = C·FA/(FA + K_eff), K_eff ≈ 0.114 (pred) / 0.119 (obs). The "0.43 power law" is a local slope in the crossover.
8. **ODE closure** (P20). 4-reservoir model analytically closed; predictions validated within 10%.
9. **Two-scale Reynolds decomposition** (P23). PDE 1 (zone-mean ODE) and PDE 2 (Allen–Cahn) are genuinely independent equations.
10. **Inheritance roughness explains FA slope flip** (P23–P25). βs=0 → slope −0.09; βs=0.25 → slope +0.21. Sign flip proven.
11. **Self-consistency quadratic for ξ_∞ verified** (P26). 5.2× ratio across βs axis matches analytical quadratic.
12. **Burgers reversal at ν/κ ≥ 1** (P26). Fitness birth reduces ξ when diffusion is too weak to smooth extreme values.
13. **ξ_∞ independent of zone width** (P26). W_zone controls sg4 but not ξ_∞; two-scale separation confirmed.
14. **Zone formation commitment epoch** (P26). C(t_ref, T_end) < 0 for t_ref < 1600; spatial bifurcation within a single run at t* ≈ 1600–2000.
15. **ESN comparison** (V85–V86). VCML 12× ESN sg4. ESN persistence ratio = 0 (no copy-forward); VCML = 0.884. Copy-forward is the differentiator.
16. **Scale invariance** (V92–V94). Dynamic advantage holds at all scales (+43%→+829%). ξ_∞ does not scale with N_active; structure forms in patches.
17. **Geographic coupling** (V99–V100). Same-wave-origin relay necessary AND sufficient for 3.48× structure boost at large scale.

---

## 8. Open Questions (Ranked by importance)

### Tier 1: Direct extensions of current theory

**Q1. Zone commitment epoch dependence on FA and WR**
- Current: t* ≈ 1600–2000 for standard parameters
- Prediction (from PDE): t* ~ 1/(P_c · FA · WR)
- Test: sweep FA × WR, measure zero-crossing of C(t_ref, T_end)
- Why important: links temporal dynamics (Layer II) to spatial PDE (Layer III)

**Q2. Long-run saturation of sg4**
- sg4 still growing at T=4000 in some seeds (V78)
- Prediction: saturation at T_sat ~ κ/(P_c · FA · ν · βs²)
- Test: run to T=8000–10000 for standard params; check if sg4 plateaus

**Q3. Does ξ_∞ saturate at zone boundary?**
- ξ_∞ ≈ 2–3 sites regardless of W_zone (Paper 26)
- But at W_zone=1, ξ_∞=nan (zone too narrow to measure)
- Question: is there a minimum zone width below which ξ_∞ > W (structure "leaks" between zones)?

### Tier 2: Theory-critical questions

**Q4. Critical exponents of the Allen-Cahn → Burgers transition**
- Current: "transition at ν/κ ≈ 1" (Paper 26)
- Unknown: is this a sharp phase transition? What are the critical exponents?
- Unknown: does the transition have finite-size scaling?

**Q5. The 8D product manifold: are C_1 and C_2 truly independent?**
- C_1 = (ν, FD, FA, βs) — viability volume
- C_2 = (κ, ν/κ, βs, W_zone) — structural manifold
- βs appears in both. Do the βs axes interact, or are they the same axis with different consequences?
- Test: joint sweep of (ν, κ) holding ν/κ constant — does viability volume shift with κ?

**Q6. Geographic relay coupling at finite scale**
- V99–V100: geographic coupling gives 3.48× boost at large N
- Theory: "structure-about-structure" — small relay region reads zone summaries
- Missing: quantitative prediction for relay gain as a function of relay region size and wave radius

**Q7. The commitment epoch mechanism**
- Current interpretation: subcritical pitchfork — system tunnels from wrong basin to correct basin
- Alternative: spinodal decomposition (both polarities co-exist, then one wins)
- Test: track *individual zone* trajectories through t*; do all zones flip simultaneously (pitchfork) or one by one (spinodal)?

### Tier 3: Extensions beyond current scope

**Q8. Multi-module coupling (V116–V117 thread)**
- Current: single VCML instance; all zones coupled by diffusion
- Open: two VCML instances with shared fieldM interface → "cortex-hippocampus" architecture
- Prediction: the inter-module interface should show zone commitment before intra-module structure

**Q9. GRU / attention formal equivalence**
- sg4 = C·FA/(FA+K_eff) is a sigmoid with midpoint K_eff — formally identical to GRU update gate
- Open: can VCML be formalised as a differentiable module with FA = gate parameter?
- If yes: VCML dynamics = GRU dynamics emergently, without design

**Q10. False memory: quantitative theory**
- Paper 6 + false_memory_experiment.py: confabulation confirmed
- Missing: analytical prediction for false-memory rate as function of (ν, FA, βs)
- Expected form: P_false ~ (βs · |F_corrupt| / (βs·|F_correct| + ε))

**Q11. Continual learning stability**
- Current: single long run, fixed wave pattern
- Open: what happens if wave pattern changes mid-run? Does committed zone structure update?
- Prediction: update speed ~ P_c·FA·ν; old zones resist change proportional to ξ_∞

---

## 9. Roadmap

```
DONE ████████████████████████████████████████████████ Papers 0-26
                                                           ↓
Q1 (commitment epoch) ─── Paper 27: FA × WR sweep on t*
Q2 (long-run saturation) ─ Paper 27 (same runs, long T)
Q7 (pitchfork vs spinodal) Paper 27 (zone-level trajectory analysis)
                                   ↓
Q4 (critical exponents) ─── Paper 28: ν/κ sweep near threshold
Q3 (ξ vs W_zone limit) ──── Paper 28: fine-grained W_zone=1,2,3
                                   ↓
Q5 (8D manifold) ─────────── Paper 29: joint (ν, κ) sweep
Q6 (relay coupling) ──────── Paper 29 (geographic relay theory)
                                   ↓
Q8–Q11 (multi-module, GRU, false memory, CL) ─ Papers 30+
```

---

## 10. The Big Picture

**What VCML is, in one sentence:**
A system where mandatory individual death is the mechanism of collective spatial memory —
not an obstacle to overcome, but the engine of adaptivity.

**The hierarchy of descriptions:**

| Scale | Description | Equation | Status |
|-------|-------------|----------|--------|
| Cell | VCSM rule | 5-step algorithm | Complete (P0) |
| Within-zone | Allen-Cahn PDE | ∂F/∂t = κ∇²F + noise | Derived and verified (P22–P26) |
| Zone-mean | ODE | dF̄/dt = P_c·FA·(m̄−F̄) − ν·F̄ | Derived and closed (P20, P23) |
| Parameter space | Viability volume | 4D (ν, FD, FA, βs) | Mapped (P7–P13) |
| Structural space | Structural manifold | 4D (κ, ν/κ, βs, W) | Mapped (P22–P26) |
| Full system | KPZ–Allen–Cahn | stochastic PDE | Written (P25–P26); not yet fully verified |

**The analogy that held throughout:**
QM (small scale, fixed background, Allen-Cahn) × GR (large scale, dynamic, zone-mean ODE) ×
the cross-scale coupling that neither equation contains alone (βs, α).
The Burgers reversal = the UV catastrophe of the coarse-grained description.

**What "each PDE isn't enough" means:**
You're navigating a product of two 4D manifolds. The system doesn't live at a single
point in parameter space — it lives on a *surface* in 8D. Every paper has been a probe
of one face of that surface. The surface is now mostly charted; the open questions
are about the boundaries, the critical exponents, and the multi-module extensions.
