# Physics Intuitions Parking Lot

Ideas worth keeping but not yet ready to formalize. Rough stage, not commitments.

---

## Causal Geometry (2026-03-10)

**The intuition:**
Geometry is not fundamental — causal order is. "Distance" between two points is a
summary of causal delay: how long influence takes to propagate between them. What appears
spatially stable is the slow layer of a timescale-coupled causal hierarchy. Solids are not
static objects; they are extremely slow causal processes relative to our perception timescale.

**What's already established (not novel):**
- Causal set theory (Bombelli, Lee, Myrheim, Sorkin 1987+): spacetime as a partially ordered
  set of causal events; geometry reconstructed from causal order, not assumed.
- Emergent geometry in quantum gravity (holography, ER=EPR, tensor network approaches):
  spacetime structure from entanglement/information connectivity.
- Condensed matter: emergent rigidity from slow structural relaxation of fast atomic dynamics.

**What VCML already demonstrates (not conjecture):**
The correlation length ξ ≈ r_wave is a causal propagation radius, not a geometric constant
of the lattice. Zone structure appears spatial because causal reach is local — not because
lattice geometry is load-bearing. This is a toy demonstration of causality → geometry:

| Causal set claim | VCML demonstration |
|---|---|
| Geometry emerges from causal order | Zone structure emerges from timescale-coupled causal dynamics |
| "Distance" = causal delay | ξ ≈ r_wave = causal propagation radius |
| Slow structure = frozen fast dynamics | fieldM = slow envelope of fast h perturbations |
| Causal events are primitive | Wave-cell interactions are primitive; lattice position is incidental |

**The C₁ × C₂ decomposition in this framing:**
- C₁ (viability volume) = causal control surface: governs whether influence *can* propagate (the gate).
- C₂ (structural manifold) = emergent geometry: governs what patterns that propagation produces.
- βs coupling is the only place C₁ touches C₂ — the gate is the only place causality writes to geometry.
- Implication: you cannot have stable emergent geometry without the viability gate because C₁ and C₂
  are otherwise independent manifolds.

**The neural network parallel:**
- Transformer attention: a_{ij} = softmax(q_i · k_j / sqrt(d)) defines causal proximity by
  information content, not position. Attention geometry is emergent from causal influence patterns.
- Transformer timescales: context window = h (fast); fine-tuning/in-context = m (medium);
  pre-trained weights = F (slow). The learned weight geometry = emergent slow-layer structure.
- RAG without a gate = static F without C₁ — geometry without the causal hierarchy that maintains it.

**What might be novel:**
The formal claim that the viability gate is the *only* coupling between C₁ and C₂ — and therefore
the necessary condition for causal events to write stable emergent geometry. If true as a theorem,
this would say: any system that produces stable structural patterns from causal dynamics must
have a gate between its causal control surface and its emergent geometry. The gate is not an
implementation detail; it is the structural interface between causality and geometry.

**The gap (what needs to close this):**
1. Formal proof that C₁ and C₂ are independent without the βs gate
2. Formal definition of "causal event writes to structural manifold"
3. That definition must be substrate-agnostic (graph, lattice, temporal sequence, etc.)
4. An observable that distinguishes this from the simpler claim "you need a threshold"

**What to park:**
The full physics extension (VCML → causal sets → spacetime geometry) requires reproducing GR + QM
from a causal network. Do not touch until the formal toy-level theorem is proven first.

**The Malament analog (VCML-specific):**
In causal set theory, causal structure almost uniquely determines conformal geometry
(Malament 1977: a bijection preserving causal order preserves the metric up to a local
scale factor). The VCML analog: ξ ≈ r_wave behaves as a causal invariant — Paper 36
showed it is unaffected by diffusion, field decay, FA, or any C₁/C₂ parameter.
Stated carefully: ξ behaves as a causal bound within the adaptive regime; an invariance
sweep (hold r_wave fixed, vary all else ±50%) would promote this to a structural law.

**The finite causal horizon and its three consequences:**
If ξ ≤ r_wave is structural, each cell has a causal horizon of radius r_wave:
cells beyond this radius have never shared a wave ancestor and cannot be directly
causally correlated.

Consequence 1 — V94 patch formation:
Zone width >> r_wave → zone contains (W/r_wave)^2 independent causal neighborhoods →
structure forms in patches of size ~r_wave, not zone-wide gradients.
Not a scaling failure. The correct behavior of a single-region system at trans-horizon
scale. Predicted by causal horizon; invisible to spatial framing.

Consequence 2 — N_crit = L/(5*r_wave) as causal averaging threshold:
Relay reads zone-mean summaries. Reliable estimation requires ≥5 independent causal
neighborhoods. N_crit = L/(5*r_wave) counts causal neighborhoods per zone.
The "5" is a signal/noise threshold on causal averaging — not a geometric constant.
This gives N_crit a derivation instead of just an empirical observation.

Consequence 3 — Hierarchical relay is structurally necessary:
A single region cannot produce coherent structure at scales > r_wave.
Trans-horizon coherence requires aggregation of causal neighborhoods — which is
exactly what relay coupling does. Relay crosses the causal horizon by aggregation,
not by propagating faster.

Unification table:
| Scale vs r_wave | Behavior | Mechanism |
|---|---|---|
| Zone ≈ r_wave (S1) | Zone-wide coherence | Single causal neighborhood |
| Zone >> r_wave (S4) | Patch formation | Independent causal neighborhoods |
| Relay: zone ≥ 5*r_wave | Relay succeeds | Causal averaging over ≥5 neighborhoods |
| Hierarchical relay | Trans-horizon coherence | Aggregation crosses horizon |

**Open direction — causal cone structure:**
The causal horizon may be a radius (isotropic) or may have a cone structure if wave
propagation is directional: a cell at time t cannot be influenced by events at
t' > t + r_wave/v_wave. Patch boundaries would propagate in a cone rather than a sphere,
explaining any directionality in patch formation at large scale.
Not yet tested. Formalize only after r_wave invariance sweep confirms the radius claim.

**Status:** Partially demonstrated in VCML (Paper 36 + V94 + Papers 35-37).
Three consequences already in experimental record; none yet explicitly named as
causal horizon consequences. Causal horizon + N_crit reinterpretation is in
VCML_THEORY_COMPLETE.tex (Open Frontiers). Invariance sweep needed to seal the claim.
The βs gate as unique C₁↔C₂ coupling needs formal treatment before publishing.

---

## MERA Causal Speed Hierarchy (2026-03-10)

**The intuition:**
MERA (Multi-scale Entanglement Renormalization Ansatz) layers don't just represent abstract
coarse-graining depth -- they represent a *causal speed hierarchy*. Causal propagation rate
c(ℓ) varies by MERA layer ℓ, analogous to how wave speed varies with scale in physical systems.
The RG depth parameter encodes something physical about how fast information can spread at
each scale.

**What's already established (not novel):**
- MERA causal cones are well-defined (Vidal 2007+)
- MERA geometry reproduces AdS via Swingle (2009/2012): entanglement entropy scales like
  Ryu-Takayanagi formula
- The layered MERA structure is already "AdS-like" -- this is not a new claim

**What might be novel:**
Defining c(ℓ) explicitly as the physical boundary-theory velocity corresponding to the
causal cone width at layer ℓ, and showing this produces predictions that differ from
assuming uniform propagation speed. Possible targets:
- How entanglement spreads after a local quench at different scales
- Entanglement velocity in non-equilibrium states (entanglement tsunami / quantum
  information scrambling)
- Connection to butterfly velocity and chaos bounds (OTOC literature)

**The gap (what needs to close this):**
1. Write down c(ℓ) explicitly
2. Derive at least one observable that follows from c(ℓ) being non-constant across layers
3. That observable must DIFFER from what uniform-speed assumption predicts
4. Must be in-principle measurable (even in tensor network simulation)

**Risk:** This might be a reinterpretation of what Swingle already showed, not new content.
"MERA has a causal hierarchy" might be equivalent to "MERA has a layered structure" --
i.e., renaming something already there without adding predictive power.

**Suggested reading before formalizing:**
- Vidal's original MERA papers
- Swingle (2012) "Entanglement renormalization and holography"
- OTOC / butterfly velocity literature (Kitaev, Maldacena-Shenker-Susskind)

**Status:** Paper 0 stage. Real formal grounding, mechanism-to-prediction gap not closed.
Same position VCML was before the copy-forward loop formalization. Worth 1-2 weeks reading,
not worth starting to write until c(ℓ) is defined and predicts something.

---

## Everything-Is-Timescales (2026-03-10)

**The intuition:**
The three-timescale architecture (h/m/F = fast/medium/slow) is not specific to VCML cells.
It is the universal structure of any system that maintains adaptive state under environmental
change:

| System | Fast (h) | Medium (m) | Slow (F) |
|---|---|---|---|
| VCML cell | Hidden state (per-step response) | Mid-memory (per-wave accumulation) | FieldM (intergenerational) |
| Neuron | Membrane potential | Synaptic weight trace | Structural synaptic change |
| Research session | Conversation context | Papers/MEMORY.md | Framework document |
| Cortex | Activation pattern | Short-term potentiation | Long-term potentiation |
| LLM in conversation | Context window tokens | [absent — the architectural gap] | [absent — the architectural gap] |
| Evolution | Individual genotype | Population allele frequencies | Species morphology |

**What makes this more than a metaphor:**
The viability gate between timescales is not optional. For information to move from h to m,
a consolidation criterion must be satisfied (calm streak, survival, signal strength).
For information to move from m to F, a stronger viability criterion must be met.
Without the gate, fast-timescale noise overwrites slow-timescale structure.
This is not a design choice — it is a mathematical requirement for any system where the
fast timescale has higher variance than the slow timescale.

**The LLM gap:** The current LLM architecture has h (context window) but is missing both m
and F as architectural components. This is why LLMs don't consolidate: there is no viability
gate, no medium accumulation, no slow intergenerational store. Adding RAG adds a static F
without a gate — the right structure but without the consolidation mechanism.
VCML (and its VCSM language prototype) adds m and the gate.

**What might be novel:**
Formalising the claim that *any* adaptive memory system must have at least three timescales
with viability gates between them, and that single-timescale systems (pure h or pure F)
are provably non-adaptive in the VCML sense.
This would be a theorem about adaptive memory in general, not just about VCML.

**The gap (what needs to close this):**
1. Define "adaptive memory" formally (can distinguish environments + update appropriately)
2. Prove that single-timescale systems fail this definition under turnover
3. Show three timescales are sufficient
4. This is probably Paper 44 or a companion theoretical note, not an experiment

**Risk:** This might be stating the obvious — of course memory needs multiple timescales.
The non-obvious part is the viability gate as a necessary architectural component, not
just a useful one. That's the load-bearing claim.

**Status:** Framework-level insight, not yet formalised. Belongs in VCML_FRAMEWORK.md
and possibly as a theoretical note after the single-region series is published.

---
