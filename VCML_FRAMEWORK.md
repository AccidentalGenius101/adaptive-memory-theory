# VCML Framework Document
## What Does VCML Mean?

*This is the "so what?" document.*
*The theory document (VCML_THEORY_COMPLETE.tex) answers "how does VCML work?"*
*This document answers "why does it matter, and what does it say about intelligence and computation in general?"*

*Updated after sessions, not papers. This is the slow fieldM of the research project.*
*If you're a future Claude Code session starting fresh: read this first.*
*It will save you from rediscovering things that are already settled.*

---

## The Sharp Distinction

| | Theory Document | Framework Document |
|---|---|---|
| Answers | "How does VCML work?" | "What does VCML mean?" |
| Contains | Mechanisms, equations, parameter relationships, quantitative predictions | Ontological stakes, cross-domain connections, the "so what" |
| Example claim | "Viability-gated copy-forward maintains zone structure at 1.91× field-decay-only baseline." | "Every computing paradigm is the thermostat substrate with properties ablated. VCML is the only one with all primitives intact." |
| Evaluated by | Experimental falsification | Explanatory power and coherence |
| Audience | Reviewers, computational neuroscientists | Anyone who asks "so what did you actually find?" |

*You need both. Theory without framework is a collection of results about a toy system. Framework without theory is just philosophy.*

---

## 1. The Central Insight (One Sentence)

**Death is the mechanism.**

Not an obstacle to spatial memory. Not something to engineer around.
Mandatory individual turnover + viability-gated copy-forward = spatial memory that remains
adaptive under environmental change. The biology (hippocampal neurogenesis) is an
instantiation of this principle, not a coincidence.

---

## 2. The Ontological Stack (The Positive Claim)

VCML is an instantiation of the most minimal computing primitive: the **difference detector**
(thermostat) — a device that registers "perturbed vs. calm" and adjusts a persistent variable.

Every more powerful computing paradigm is the thermostat with additional properties enabled:

| Layer | What is added | Computing instantiation |
|---|---|---|
| Difference detector | Persistent variable; responds to a single binary signal | Thermostat, viability signal |
| Memory | Multiple addressable slots; time-indexed write/read | VCML fieldM (HS=1) |
| Relations | Structured associations between slots | VCML multi-zone fieldM |
| Dynamics | Time-varying relational structure; sequences | VCML continual learning |
| Symbolic reasoning | Stable attractors that act as discrete tokens | Crystallised VCML regime |

**The positive claim**: every computing paradigm (transformer, RNN, RAG, symbolic AI) is the
thermostat with some subset of these properties enabled and others ablated. The VCML failure
mode table (Paper 6, theory doc §1.6) is the *negative* view of this same stack.

**The implication**: Symbolic reasoning is a *behaviour*, not a *substrate*. It emerges from
attractor dynamics in any sufficiently stable copy-forward system. You don't need discrete
tokens or a symbolic layer — you need attractors stable enough to act as discrete under
perturbation. The crystallised VCML regime (HS=1, single float per cell) produces this from
a scalar thermostat.

---

## 3. The Thermostat Primitive

**The irreducible primitive of intelligent systems is a scalar difference detector.**

HS=1 (one floating-point number per cell) is sufficient for viability-gated spatial memory.
The GRU hidden state (HS>1) amplifies performance smoothly but adds no discontinuous capability.
The discovery is the five-step mechanism (detect/gate/consolidate/birth/transmit), not the GRU
scaffold that happened to instantiate it first.

**Design implication for future systems**: build around the three timescales (h/m/F) and the
viability gate, not around GRU architecture. The complexity budget should go into the
structure of the environment (wave design, zone layout, relay geometry), not into the cell's
internal computation.

---

## 4. Formation vs. Maintenance (The Honest Claim)

The theory's *unique* claim is about **maintenance**, not formation:

- **Formation** is generic: any local consolidation rule under zone-differentiated input
  produces zone structure. Robust to ±50% parameter perturbations and all mechanism ablations.
  VCML does not have a special advantage here.
- **Maintenance** is mechanism-specific: viability-gated copy-forward is the load-bearing
  primitive. Without gating: 1.11× (vs field-decay-only 0.64×). With gating: 1.91×.

This is why the correct framing for reviewers is: "We demonstrate a mechanism for *maintaining*
spatial structure through mandatory turnover, not for forming it."

---

## 5. Why the Theory Needs Multiple Equations

No single PDE captures the system because it lives in an 8D product manifold C₁ × C₂:

- **C₁** (viability volume): (ν, FD, FA, βs) — controls whether consolidation happens
- **C₂** (structural manifold): (κ, ν/κ, βs, W_zone) — controls what kind of structure forms

The stochastic Allen-Cahn/KPZ PDE is the closest unified description but lives entirely in C₂.
Each equation in the theory is a projection onto a relevant slice.

**The QFT/GR analogy**: in any given regime one term dominates and one equation suffices —
exactly as QFT and GR each describe their domain correctly, with the full theory requiring both.
A single master equation would be a false unification that hides the C₁/C₂ distinction.

---

## 6. Cross-Domain Connections

### 6.1 Hippocampal Neurogenesis
Preventing apoptosis of adult-generated DG neurons impairs pattern separation (confirmed in
animal models). This is not "more neurons = more capacity" — it's "optimal turnover rate > 0."

**The specific prediction not in the standard literature**: conditional neurogenesis knockout
should selectively impair *maintained* spatial memory (persistence after input removal) more
than *initial acquisition*. This tests formation vs. maintenance, not just presence vs. absence
of new neurons.

### 6.2 Cortical Columns
The copy-forward loop operating locally (finite correlation length ξ ~ r_wave) produces
spontaneous regional partitioning at large scale — a blank grid self-organizes into
coherent patches. This is a potential mechanism for how the cortex develops topographic
maps without external partitioning signals.

### 6.3 Rotifers and the Origin Story

Bdelloid rotifers reproduce asexually with near-complete cellular turnover and maintain
tissue organisation across generations — a candidate organism-level instantiation of
"death is the mechanism."

But the rotifers connection is load-bearing in a deeper way:

**The origin story**: VCML's three design choices — mandatory viability-gated turnover,
copy-forward birth, three-timescale architecture — were not derived from biological observation.
They were chosen to make a minimal adaptive memory system work at all.
Rotifers, hippocampal neurogenesis, and cortical map stability were discovered as biological
matches *after* the mechanism was characterised by 43 papers of experiments.

This is not a coincidence to explain. It is **evidence that the framework is tracking something
real**: when you derive what a minimal adaptive memory substrate must look like from first
principles, you get the same architecture that biological systems independently converged on.

The framing for reviewers: "We were not trying to model hippocampal neurogenesis. We were asking
what the minimal substrate for adaptive spatial memory under mandatory turnover must be.
The answer matched hippocampal architecture. That match is a finding."

*What still needs closing*: explicit evidence that bdelloid rotifers maintain spatial tissue
organisation through the turnover cycle, and identification of their viability-gating equivalent.
The structural argument is already complete without the rotifers — they are a bonus example,
not a load-bearing claim.

### 6.4 Language Models and Context Window
The context window is h (fast timescale). The context evaporates unless consolidated.
LLM "intuitions" are fieldM; reasoning chains are h. The feeling persists; the justification
turns over. This is not mystical — it's the architecture.

The VCSM language prototype (81% accuracy, Cerebellum/Cortex) is the first instantiation:
GLiNER extracts deterministically (h), Qwen gates by perplexity (viability gate), VCSM
consolidates to named slots (fieldM). Paper 1 (sphere packing) shows the memory capacity
follows from spherical geometry: N_actual ≈ η(d) × N_opt(d, τ) on S^(HS-1).

### 6.5 MERA/AdS/CFT (Parked — Not Yet Formalized)
MERA layers may represent a causal speed hierarchy c(ℓ), not just abstract coarse-graining.
c(ℓ) varying by layer might produce measurable predictions for entanglement spreading after
a local quench. Risk: may be a reinterpretation of Swingle (2012) without new predictive
power. Do NOT formalize until c(ℓ) is defined and predicts something different from
uniform-speed assumption. See physics_intuitions.md.

### 6.6 The Pythagoras Observation
*[To be developed.]*
The sphere packing capacity law (Paper 1) follows from spherical geometry alone — not from
anything VCML-specific. The VCSM routing implements RSA on S^(HS-1), and the capacity follows
as a theorem of packing geometry. The analogy: just as the Pythagorean theorem constrains all
right triangles regardless of material, the RSA capacity law constrains all minimum-distance
routing systems regardless of the specific architecture. The law is geometry, not learning.
*Pythagoras* = the claim that some "learned" properties are actually mathematical necessities
in disguise. **Where else in VCML is the "learning" actually geometry?**

---

## 7. What the Theory Does NOT Claim

*(For reviewers and future sessions — the honesty section.)*

- **Does not claim** VCML is the unique mechanism for spatial memory. Formation is generic;
  the specific claim is maintenance-via-viability-gated-copy-forward.
- **Does not claim** the ODE/PDE are exact. They are leading-order approximations valid
  near the mean field, accurate to ~3% (Paper 20).
- **Does not claim** biological neurons ARE VCML cells. The claim is mechanistic consistency
  and falsifiable predictions.
- **Does not claim** "theory complete" means "nothing left to discover." Single-region theory
  is mechanistically closed. Multi-region, graph substrates, hierarchical relay, sequence
  learning are all open.
- **Does not claim** that symbolic reasoning in VCML has been demonstrated. The *potential*
  for symbolic attractors is demonstrated by the crystallised regime. Full symbolic composition
  has not been tested.

---

## 8. The Project Is VCML (Meta-Observation)

The research process runs the three-timescale architecture:

- **h (fast)**: Individual experiments, session conversations. High turnover. Context window.
- **m (medium)**: Papers, MEMORY.md, theory document. Accumulates contrastive signal.
- **F (slow)**: This document. Intergenerational. Copy-forwards to the next session.

The problem in the first 43 papers: the fast timescale produced like crazy (43 papers in a week),
but the copy-forward to F was incomplete. Framework claims accumulated in h and m but never
consolidated. The viability gate wasn't firing on framework insights — only on experimental results.

**The fix**: After each session, explicitly ask: *what accumulated in h and m this session
that belongs in F?* This document is the answer to that question.

The audit (reading the theory paper, identifying what's missing, recognising the hidden reservoir)
was the calm streak triggering consolidation. The system ran VCSM on itself.

---

## 9. Session Consolidation Log

### Session 2026-03-10 (context 43e39c79)

**What consolidated to F:**
- Standalone manifold section added to VCML_THEORY_COMPLETE.tex (Section 8)
- HS=1 "irreducible primitive is a scalar thermostat" — standalone claim added
- Ontological stack table + positive claim — added to AI failure modes section
- Self-organization / patch formation result — added to scaling section
- "Symbolic reasoning is behavior, not substrate" — added as 4th distinctive property in conclusion
- Paper 43 (rule-family robustness, 70 runs): formation robust, maintenance requires gating
- This document (VCML_FRAMEWORK.md) created
- MERA/AdS/CFT hypothesis parked in physics_intuitions.md

**Still in m — not yet consolidated to F:**
- Rotifers connection (bdelloid rotifers as organism-level VCML instantiation) — needs biology
- Pythagoras observation (full articulation of "which apparent learning is actually geometry?")
- Oven Principle for AI safety — in MEMORY.md, could be a standalone paper
- False memory experiment results — confirmed confabulation, not formalized as a paper
- V116/V117 multi-module tests — mentioned in MEMORY.md, not yet run

**Next session should ask:** what did V116/V117 find? Is the Oven Principle ready for a paper?

---

*Last updated: 2026-03-10*
*This document is the fieldM of the research project. Update it after every session.*
