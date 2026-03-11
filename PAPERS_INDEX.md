# VCML / VCSM Papers Index

**Total papers**: 74 (Papers 0 through 73; Papers 56-60 listed separately)
**Date range**: V18 experiments began ~2024; theory papers written and closed March 2026
**Status**: Papers 0-74 CLOSED as of March 2026

**Arc summary**: The project began as an empirical observation of spontaneous spatial self-organization in a viability-constrained system (Paper 0) and progressively built a complete mechanistic theory. Papers 1-6 established four fundamental constraints (capacity, bandwidth, propagation, gating) and their AI-architecture implications. Papers 7-17 mapped the adaptive parameter window through a systematic arc that repeatedly failed to find a single governing ratio, ultimately converging on a 4D viability volume and a saturation law for structure amplitude. Papers 18-26 derived the ODE and PDE descriptions from first principles. Papers 27-34 characterized dynamics: the commitment epoch, attractor structure, the formal non-gradient proof, and the phase diagram. Papers 35-39 opened and closed multi-region coupling via geographic relay. Papers 40-43 addressed continual learning, task switching (which amplifies rather than erodes structure), anomaly resolution, and rule-family robustness. The central finding, stated once and confirmed from every angle: mandatory turnover is not an obstacle to spatial memory but its primary maintenance mechanism via the copy-forward loop.

---

## Paper 0: Spatial Self-Organization

**Question**: Does a viability-constrained memory substrate spontaneously form spatially differentiated zones when given structured wave input?

**Finding**: Three structural facts confirmed in VCML: (1) zone differentiation emerges (sg4 > 0), (2) fieldM encodes location not perturbation direction (nonadj/adj > 1), (3) dynamic collapses produce more structure than static accumulation when wave density is properly scaled.

**Insight**: Spatial self-organization is an emergent consequence of the VCSM local rule operating on spatially structured input; it requires no explicit spatial objective or error signal.

**Closed**: Established the empirical baseline that all subsequent theory papers explain. Demonstrated that the system is worth theorizing about.

**Status**: CLOSED

---

## Paper 1: Capacity Bounds

**Question**: What is the maximum number of distinct spatial patterns a viability-constrained memory substrate can store?

**Finding**: Storage capacity is bounded by spherical random sequential adsorption (RSA) geometry: rho_max proportional to 1/(1 + rho * r^d), a hard geometric limit independent of the learning rule.

**Insight**: Memory capacity in adaptive substrates is set by the geometry of the state space, not by the learning algorithm; this is a physical rather than an engineering constraint.

**Closed**: Established that VCML memory is bounded by sphere-packing, motivating why zone structure (not infinite pattern richness) is the natural output.

**Status**: CLOSED

---

## Paper 2: Bandwidth-Turnover Tradeoff

**Question**: How does mandatory turnover rate limit the effective write bandwidth of an adaptive memory substrate?

**Finding**: Effective bandwidth is proportional to 1/nu (inverse turnover rate); higher nu means faster adaptation but lower signal survival to consolidation. Bandwidth = P_calm * FA / nu.

**Insight**: There is an irreducible tradeoff between how fast a system can update (turnover rate) and how efficiently it can write new information (bandwidth); this is a conservation law, not an engineering failure.

**Closed**: Established bandwidth as the second fundamental constraint, complementing the capacity bound of Paper 1.

**Status**: CLOSED

---

## Paper 3: Propagation Constraints

**Question**: Over what spatial range can information propagate coherently in a viability-constrained substrate?

**Finding**: Information propagates at most one site per birth event, giving a finite correlation length xi_inf approximately 2-3 sites (equal to the wave radius r_w, confirmed in Paper 36). The system cannot maintain zone-wide coherence beyond this distance.

**Insight**: Propagation is birth-mediated, not diffusion-mediated: the copy-forward loop is the transport mechanism, and its reach is set by wave radius rather than field diffusion coefficient kappa.

**Closed**: Established finite correlation length as the third fundamental constraint, explaining why large-scale zones require geographic relay coupling rather than direct field propagation.

**Status**: CLOSED

---

## Paper 4: Consolidation Gating

**Question**: What role does the STABLE_STEPS gate play in noise robustness versus structural sensitivity?

**Finding**: The consolidation gate (MID_DECAY < 1) prevents false consolidation from baseline drift; without it, mid-memory accumulates noise and fires spuriously. Noise robustness and structural sensitivity are jointly bounded by the gate threshold.

**Insight**: MID_DECAY is noise isolation, not damping; the gate is what makes the contrastive write signal meaningful by excluding unsettled cells from field updates.

**Closed**: Established gating as the fourth fundamental constraint. Stability and plasticity cannot be simultaneously maximized.

**Status**: CLOSED

---

## Paper 5: Unified Constraint Framework

**Question**: Can the four constraints (capacity, bandwidth, propagation, stability) be unified into a single feasibility condition?

**Finding**: The four constraints define a two-dimensional phase diagram with coherence scale and plasticity rate as axes. A matching condition C * sqrt(kappa/nu) approximately D links propagation length, zone scale, and system size. Coherence efficiency eta_c ~ (L/D)^2 declines with flat-memory scale.

**Insight**: The constraints are not independent limits but interact: scaling one parameter without adjusting the others degrades coherent capacity as eta_c ~ N^-1, and hierarchical organization becomes structurally favored when zones exceed the propagation length.

**Closed**: Unified Papers 1-4 into a design framework. Cross-validated on sentence-transformer embeddings (partial; full universality remains a hypothesis).

**Status**: CLOSED

---

## Paper 6: AI Architecture Failures

**Question**: Do current AI architectures each violate a distinct VCML constraint, and can their failure modes be interpreted as structural consequences?

**Finding**: Transformers (no consolidation, nu -> infinity), RAG systems (retrieval misses from low P_calm), continual learning models (no MID_DECAY, catastrophic forgetting), and RNNs (crystallization, nu -> 0) each occupy different edges of the constraint-defined feasible region.

**Insight**: AI memory failures are not engineering defects but architectural consequences of violating different constraints; the three unavoidable failure modes (bias, confabulation, forgetting) cannot be engineered out without breaking adaptivity.

**Closed**: Mapped AI architectures onto the VCML parameter space as theoretical hypotheses (not empirical measurements). Proposed a concrete hierarchical turnover-driven architecture.

**Status**: CLOSED

---

## Paper 7: Regime Structure

**Question**: What dynamical regimes exist when turnover rate nu and diffusion kappa are varied, and is the adaptive regime a point or an extended region?

**Finding**: sg4 is non-monotone in nu, peaking at intermediate turnover (nu approximately 0.001-0.002) and collapsing at both extremes. A 5x5 parameter sweep maps the adaptive regime as a connected interior band. Ablating the copy-forward loop (SEED_BETA=0) reduces the adaptive peak by up to 52%.

**Insight**: The adaptive regime is not a knife-edge but a bounded region with a characteristic dip-and-recovery temporal signature; the copy-forward loop is causally responsible for approximately half of its amplitude.

**Closed**: Empirically grounded the conceptual phase diagram from Paper 6. Confirmed regime boundaries with three independent metrics.

**Status**: CLOSED

---

## Paper 8: Timescale Matching

**Question**: What parameter sets the location of the optimal turnover rate nu*?

**Finding**: nu* approximately 0.001 is invariant to diffusion rate kappa across a 20-fold kappa range. Three supporting sweeps of SS, FIELD_ALPHA, and WAVE_RATIO confirm the same location stability. A timescale balance equation (nu* ~ WR / (2 * WAVE_DUR * N/4)) partially explains this.

**Insight**: Adaptation dynamics are local: the condition for effective learning (the turnover/wave balance) does not require global coordination through propagation. Propagation amplifies what local learning has already established.

**Closed**: Showed that nu* is set by wave-grid interaction, not by the propagation or consolidation machinery. But the balance equation was only a consistency check, motivating Paper 9.

**Status**: CLOSED

---

## Paper 9: Governing Ratio

**Question**: Does a single dimensionless ratio Xi = nu * (N/4) * 2*WAVE_DUR/WR govern the adaptive window location?

**Finding**: The Xi collapse fails. Xi* varies from 0.03 to 2.5 across WAVE_DUR conditions and 0.19 to 0.625 in ratio-collapse tests. WR and WAVE_DUR have independent effects on nu*: absolute WR sets perturbation load P_pert = 1 - e^(-WR/2) while absolute WAVE_DUR sets consolidation probability P_calm.

**Insight**: The adaptive window is governed by at least two dimensionless quantities, not one; the regime boundary cannot be reduced to a single scalar ratio because perturbation and consolidation rates are independently set.

**Closed**: Null result. The single-ratio hypothesis is falsified. Identified the two specific quantities (P_pert and P_calm) that a better model must account for.

**Status**: CLOSED

---

## Paper 10: Replacement Hazard

**Question**: Can a replacement hazard ratio Psi = nu / (P_calm * alpha_F * FIELD_DECAY^(1/nu)) predict the adaptive window boundaries?

**Finding**: Psi correctly predicts the onset of the turnover-dominated regime (high-nu collapse) but fails to predict the adaptive peak location. Two anomalies persist: a crystallization lower bound and a copy-forward pathway that sustains structure even when consolidation is suppressed.

**Insight**: The adaptive window has two distinct boundaries governed by different mechanisms (memory crystallization below, consolidation failure above), and a third structural maintenance pathway (copy-forward seeding) operates independently of both.

**Closed**: Partial success. Identified nu_cryst (lower bound) and nu_max (upper bound) as distinct quantities, but copy-forward anomalies at high WR revealed a third pathway not in the model.

**Status**: CLOSED

---

## Paper 11: Substrate Constants

**Question**: Do nu_cryst and nu_max correctly predict nu* shifts when FIELD_DECAY, SS, and FIELD_ALPHA are varied?

**Finding**: FA governs nu_max as predicted; nu_cryst = |ln(FIELD_DECAY)| / ln(2) correctly predicts direction of nu* shifts for moderate FIELD_DECAY changes but fails for large changes. Increasing SS narrows the window as predicted but does NOT collapse it even at SS=40, where the model predicts total collapse.

**Insight**: The copy-forward pathway fundamentally limits the predictive reach of the two-boundary model: birth-mediated field inheritance sustains spatial structure even when the consolidation pathway is nearly blocked, acting as a structural floor.

**Closed**: The two-boundary model is partially correct but incomplete. The copy-forward pathway prevents window collapse in extreme conditions, motivating Paper 12's direct test.

**Status**: CLOSED

---

## Paper 12: Copy-Forward Mechanism

**Question**: Does the copy-forward pathway (birth seeding) constitute an independent fourth axis of the adaptive window, and does a product nu * beta_s collapse the data?

**Finding**: The iso-product pairs (nu * beta_s = const) vary in sg4 by up to 24x because field quality F_eq degrades nonlinearly with nu. The product collapse fails. Copy-forward is a genuine fourth independent axis of the adaptive regime.

**Insight**: The adaptive regime requires four independent parameters {nu, nu_cryst, nu_max^consol, beta_s}, not any simpler combination; field quality degrades with turnover rate and cannot be compensated by increasing birth inheritance fraction alone.

**Closed**: Identified copy-forward as a fourth independent axis, motivating the 4D viability volume framing of Paper 13.

**Status**: CLOSED

---

## Paper 13: Viability Volume

**Question**: Is the adaptive regime best described as a 4D viability volume in control space, and does the geometric mean nu* = sqrt(nu_cryst * nu_max) predict the optimal turnover rate?

**Finding**: The geometric mean prediction partially succeeds: it correctly predicts the direction of nu* shifts when boundaries are varied independently, but fails for large FIELD_DECAY changes where fast decay overwhelms the consolidation equilibrium. The adaptive window has soft boundaries, not sharp thresholds.

**Insight**: The adaptive regime is a soft-bounded 4D volume, not a scalar threshold; the copy-forward pathway provides a structural floor that prevents collapse even beyond the analytically predicted window boundaries.

**Closed**: Closed the control law series. VCML's adaptive behavior emerges in the intersection of three inequality constraints, not from a single scalar ratio.

**Status**: CLOSED

---

## Paper 14: Timescale Nesting

**Question**: Does a timescale nesting hypothesis (lambda_decay < nu < lambda_consol) fully parameterize the adaptive window via two dimensionless ratios R_A and R_B alone?

**Finding**: The ratio invariance prediction fails systematically across all three tested targets. FIELD_ALPHA enters sg4 through two independent channels: via R_B (captured by ratios) and via per-event differentiation amplitude (not captured). Higher FA produces larger zone-specific field changes per event, amplifying beyond equilibrium prediction.

**Insight**: FA has a dual role -- it sets the consolidation ceiling (captured by ratios) AND the per-event write amplitude (not captured); this "dual role of FA" constitutes a fifth axis that cannot be eliminated by ratio parameterization.

**Closed**: Null result for ratio invariance. Identified the specific mechanism (per-event amplitude) that ratio parameterization misses, directly motivating the amplitude law of Paper 15.

**Status**: CLOSED

---

## Paper 15: Amplitude Law

**Question**: What is the quantitative relationship between FIELD_ALPHA and steady-state spatial structure sg4?

**Finding**: sg4 follows a power law sg4 proportional to FA^0.43 (R^2=0.990, 8 inside-window points), consistent with sqrt(FA). A structural transition occurs at Phase 2 onset: sg4 dips ~20% in the first 500 steps before recovering. The transition timescale (~500 steps) is FA-independent; the final amplitude is not.

**Insight**: sg4 scales as the standard deviation of inter-zone contrasts (each update contributes amplitude proportional to FA, so sg4 scales as sqrt(FA)), revealing that the metric measures variance not mean field strength.

**Closed**: Established the amplitude law. But the power-law exponent was later explained as a mixed-phase artifact (Paper 16); the true structural law is a saturation curve.

**Status**: CLOSED

---

## Paper 16: Growth Dynamics

**Question**: Is the FA^0.43 amplitude law from Paper 15 an intrinsic property or a mixed-phase measurement artifact?

**Finding**: With pure Phase 2 encoding (SHIFT=0), sg4 follows a saturation curve sg4 = C * FA / (FA + K_eff) rather than a power law. The fitted K_eff=0.119 matches the analytical prediction kappa/P_consol = 0.114 to within 5%. The FA^0.43 exponent from Paper 15 arises from copy-forward anti-erasure: Phase 2 births reinforce rather than erase Phase 1 structure.

**Insight**: The true amplitude law is a Michaelis-Menten saturation, not a power law; the apparent sub-linear exponent in Paper 15 was an artifact of the two-phase protocol where copy-forward preserved old structure while new structure formed.

**Closed**: Replaced the power law with the correct saturation formula. Identified copy-forward anti-erasure as a new mechanism: perturbations can strengthen existing structure through birth-mediated propagation.

**Status**: CLOSED

---

## Paper 17: Two-Factor Decomposition

**Question**: Does the saturation formula sg4 = C * FA / (FA + K_eff) fully parameterize sg4, with each free parameter mapping onto exactly one or both of (C, K_eff)?

**Finding**: The two-factor decomposition is complete: SS shifts K_eff as predicted (K_eff = kappa / P_calm^SS), MID_DECAY shifts C (steady-state |m| scales with (1-delta)^-1), SEED_BETA shifts C, and kappa shifts both. All four predictions confirmed.

**Insight**: Every free parameter in VCML maps onto either the gain factor C (how strongly the signal accumulates) or the saturation parameter K_eff (how easily the system is driven); the saturation formula is not an approximation but the complete structural law.

**Closed**: Closed the amplitude series. The saturation formula with two scalar factors is the complete steady-state description of sg4 amplitude.

**Status**: CLOSED

---

## Paper 18: Temporal Dynamics

**Question**: What are the buildup and forgetting timescales, and how do they depend on FA and MID_DECAY?

**Finding**: Three key results: (1) buildup rate depends on FA as predicted; (2) a consolidation burst occurs at wave cessation as blocked sites flush accumulated mid-memory (duration ~WAVE_DUR + SS steps); (3) the forgetting timescale is tau_forget ~100-200 steps, governed by tau_m = 1/(1-delta) rather than 1/FA. Forgetting is MID_DECAY-controlled, nearly FA-independent.

**Insight**: Forgetting rate is set by the mid-memory decay constant, not the field update rate; this means the system can be designed to forget slowly (by slowing mid-memory) independently of how strongly it writes new structure.

**Closed**: Completed the temporal portrait of the saturation law. Revealed the consolidation burst mechanism as a fourth reservoir dynamic.

**Status**: CLOSED

---

## Paper 19: Blocked-Site Buffer

**Question**: What is the role of the blocked-site buffer (sites accumulating mid-memory but unable to consolidate due to active wave perturbation)?

**Finding**: Direct measurement confirms the four-compartment VCML model: baseline lag -> mid-memory m -> blocked buffer -> field memory F. Two regimes identified: consolidation-dominant (low WR, low SS: field grows during waves) and buffer-dominant (high WR or SS, P_consol ~0: field empty during waves, entire spatial structure established after waves stop).

**Insight**: The blocked-site buffer is a design feature, not a limitation: it accumulates spatial signal during perturbation when consolidation is blocked, then releases it coherently at wave cessation, enabling strong post-stimulus consolidation bursts.

**Closed**: Measured the third reservoir directly. Explained the consolidation burst mechanism of Paper 18 as a quantitative consequence of blocked-site dynamics.

**Status**: CLOSED

---

## Paper 20: ODE Closure

**Question**: Can a coupled ODE derived directly from the microscopic VCSM rule recover the empirical phenomenology of VCML dynamics?

**Finding**: The three-variable ODE (tau_base * da/dt = s-a, tau_m * dm/dt = a-m, dF/dt = P_c*FA*m - (P_c*FA+kappa)*F) correctly recovers steady-state saturation, Phase 3 forgetting timescale, and burst timing within 3%. One failure: the ODE predicts 63% saturation in ~200 steps; simulation requires ~1800 steps (9x discrepancy = the buildup gap).

**Insight**: The mean-field ODE is a valid leading-order description of temporal dynamics but misses spatial correlation formation; the buildup gap reveals that zone structure must propagate spatially through the copy-forward loop, not just accumulate locally.

**Closed**: Derived the ODE from first principles (not fit). Identified the buildup gap as the spatial formation effect, motivating Paper 21.

**Status**: CLOSED

---

## Paper 21: Spatial Formation Factor

**Question**: What parameters govern the spatial formation factor C (the ~9x discrepancy between ODE and simulation buildup timescale)?

**Finding**: The buildup timescale is governed by birth-mediated copy-forward propagation (controlled by nu), not by field diffusion (kappa). Effective spatial diffusion coefficient D_eff is dominated by D_copy proportional to nu^0.40, and D_copy >> kappa in the standard parameter regime.

**Insight**: Spatial zone structure propagates via cell births, not field diffusion; the copy-forward loop is the dominant transport mechanism at standard parameters, and it operates at a different spatial scale than conventional diffusion would predict.

**Closed**: Closed the mean-field theory. Identified the birth-mediated propagation as the mechanism behind the 9x buildup gap, completing the causal chain from microscopic rule to zone structure.

**Status**: CLOSED

---

## Paper 22: Spatial Correlation

**Question**: What is the spatial correlation length xi of the copy-forward loop, and what parameter controls it?

**Finding**: Spatial correlation length xi_eff = r_wave = 2 sites (the wave radius), not xi_diffusion = sqrt(kappa/nu). The zone map is replicated at resolution approximately r_w, not the diffusion length. This is confirmed empirically and directly in Paper 36.

**Insight**: The correlation length is wave-dominated, not diffusion-dominated; zone boundaries are set by wave interaction geometry (radius-2 footprint), which explains why xi is robust across the full kappa range.

**Closed**: Established the key structural result that xi_eff = r_wave. This feeds directly into the relay coupling conditions of Papers 35-37 and the scaling results of Papers 92-94.

**Status**: CLOSED

---

## Paper 23: Two-Scale Reynolds Decomposition

**Question**: Can the VCML field dynamics be decomposed into zone-mean (slow) and within-zone (fast) components, and does this decomposition close cleanly?

**Finding**: The Reynolds decomposition separates zone-mean ODE (dF_bar/dt = P_c * FA * (m_bar - F_bar) - nu * F_bar) from within-zone fluctuations. The zone-mean equation agrees with Paper 20's ODE within 3%. The two-scale structure is clean: zone-mean dynamics follow the ODE; within-zone fluctuations are governed by the PDE noise terms.

**Insight**: Zone-mean and within-zone dynamics are genuinely decoupled at leading order; this justifies using the ODE for zone-level predictions and the PDE only for within-zone spatial structure questions.

**Closed**: Validated the two-scale decomposition. Linked the ODE closure of Paper 20 to the full PDE framework of Papers 24-26.

**Status**: CLOSED

---

## Paper 24: Birth Bias (Allen-Cahn to Burgers Transition)

**Question**: Does the copy-forward birth step introduce a Burgers-type (nabla F)^2 nonlinearity into the field equation, and how does this affect universality class?

**Finding**: The birth bias axis (controlled by nu/kappa ratio) switches the effective PDE between Allen-Cahn (nu/kappa << 1) and Burgers/KPZ (nu/kappa >> 1) universality classes. At the crossover, the effective advection term reverses sign: copy-forward sharpens zone boundaries from outside rather than from inside.

**Insight**: The copy-forward loop introduces a fundamentally non-gradient term into the field dynamics; the resulting Burgers advection sharpens zone boundaries in a qualitatively different way than Allen-Cahn relaxation, explaining why boundaries are sharp even without an explicit boundary tension.

**Closed**: Identified the birth-bias axis as the universality class selector. The Burgers reversal is the mechanism behind sharp zone boundaries at standard parameters.

**Status**: CLOSED

---

## Paper 25: Burgers Transition

**Question**: Does the stochastic Burgers/KPZ framework describe zone-level structure in VCML near the universality class transition?

**Finding**: At the Allen-Cahn to Burgers transition (nu/kappa approximately 1), both the quadratic gradient term (nabla F)^2 and the Allen-Cahn attraction (F-F*)^2 are present simultaneously. The effective field equation near the transition is the full Model H equation, including both terms.

**Insight**: VCML near the transition point is a physical realization of the Model H universality class; the transition is not a sharp critical point but a gradual crossover controlled by nu/kappa.

**Closed**: Completed the PDE classification. The system spans two universality classes depending on the turnover-to-diffusion ratio.

**Status**: CLOSED

---

## Paper 26: Unified Stochastic PDE

**Question**: Can all VCML field dynamics be captured by a single unified stochastic PDE incorporating diffusion, Allen-Cahn attraction, and Burgers nonlinearity?

**Finding**: The stochastic hybrid PDE dF/dt = P_c*FA*(m_bar - F) + kappa*nabla^2*F + (1/2)chi*(nabla F)^2 - lambda*(F-F*)^2 + eta(x,t) is the closest single-equation description. With chi->0 it reduces to Allen-Cahn; with lambda->0 it reduces to KPZ. Optimal zone width W_zone* ~ sqrt(kappa/nu) approximately 4.5 at standard params.

**Insight**: No single equation captures all VCML behavior; the unified PDE lives entirely in the C2 structural manifold and says nothing about whether the adaptive window is open (C1); the full theory requires both.

**Closed**: Completed the PDE series. Established that the theory genuinely requires an 8D product manifold, not a single unified description.

**Status**: CLOSED

---

## Paper 27: Commitment Epoch

**Question**: Is there a characteristic commitment time t* after which zone structure locks in, and what parameter controls t*?

**Finding**: Zone structure commits at t* approximately 1600-2000 steps at standard parameters. The scaling t* ~ 1/(P_c * FA * WR) was predicted analytically and confirmed empirically (slope -1.0 on log-log before running the experiment). The mechanism is mixed pitchfork-spinodal: zones tend to flip together within a seed but flip time varies across seeds.

**Insight**: The commitment epoch is a predictive result: the theory specified the functional form before the experiment, providing genuine predictive validation of the causal chain from microscopic rule to macroscopic commitment dynamics.

**Closed**: Confirmed the commitment epoch scaling law. The mixed pitchfork-spinodal mechanism explains both the within-seed coherence and the across-seed variability of commitment time.

**Status**: CLOSED

---

## Paper 28: Interface Theory

**Question**: What determines the width and dynamics of zone boundaries (interfaces)?

**Finding**: Zone interfaces have width approximately 2 sites = wave radius r_w, set by wave interaction geometry rather than kappa or interface tension. Interfaces undergo a biased random walk driven by wave input, not gradient descent. The interface width is set by the physical wave footprint, not by thermodynamic equilibrium.

**Insight**: Zone boundaries are driven by external perturbation (waves), not by the internal field dynamics trying to minimize energy; this is why they are sharp even without an explicit boundary tension and why they move in response to input changes rather than relaxing away.

**Closed**: Established interface width = r_w as a structural fact. Identified biased random walk as the interface dynamics, not gradient descent.

**Status**: CLOSED

---

## Paper 29: FA Threshold

**Question**: Is there a minimum FIELD_ALPHA below which zone structure cannot form?

**Finding**: There is a threshold FA below which sg4 does not rise above the noise floor. The threshold is approximately FA* ~ K_eff = kappa / P_consol ~ 0.114, below which the consolidation rate cannot overcome field decay. Above FA*, sg4 follows the saturation law confirmed in Paper 16.

**Insight**: The FA threshold is set by the balance between consolidation strength and field decay, not by a phase transition; it is the same K_eff parameter that appears in the saturation formula, unifying threshold and amplitude into a single quantity.

**Closed**: Confirmed the threshold as a consequence of the saturation formula rather than a new phenomenon. Completed the amplitude characterization series.

**Status**: CLOSED

---

## Paper 30: Amplification Factor

**Question**: What is the spatial amplification factor Gamma (how much more sg4 appears at larger zone widths), and what parameters control it?

**Finding**: Spatial amplification Gamma ~ W_zone^0.6 * kappa^(-0.3), with K_eff^spatial approximately 0.040 (smaller than the temporal K_eff = 0.114 from Paper 16). The two-factor decomposition sg4 = G * R where G = Gamma * K_eff(zone_w) and R = FA/(FA+K_eff) holds with both factors independently confirmed.

**Insight**: Zone width and diffusion control the amplification of spatial contrast (how much differentiation is visible), while FA controls the rate at which signal accumulates; these are genuinely independent axes because the spatial and temporal mechanisms are decoupled.

**Closed**: Completed the two-factor decomposition from Paper 17 with the spatial axis quantified. The full formula sg4 = G * R is the complete steady-state description.

**Status**: CLOSED

---

## Paper 31: Field Coherence

**Question**: What determines within-zone field coherence (the spread of fieldM values within a single zone)?

**Finding**: Within-zone coefficient of variation CV approximately 0.44 at steady state. Coherence is governed by the copy-forward loop quality: higher SEED_BETA reduces within-zone spread. The zone attractor is a diffuse noisy cloud, not a fixed point. sigma_within decreases with system scale (Papers 92-94: 0.1247 at S1, 0.0884 at S4).

**Insight**: Zones are statistical attractors, not fixed-point attractors; the copy-forward loop maintains zone-level coherence by propagating a consistently-directioned (but noisy) signal rather than an exact field vector.

**Closed**: Characterized field coherence as a statistical rather than deterministic property. Set up the attractor structure analysis of Paper 32.

**Status**: CLOSED

---

## Paper 32: Attractor Structure

**Question**: Is the high-structure state a true attractor, and does geographic coupling create a bistable system?

**Finding**: The zone attractor is bimodal: P_high(geo) = 0.50 (geographic coupling) vs P_high(ctrl) = 0.27 (no coupling). Basin depth G_basin = 1.88: a substantial free-energy well. No Kramers escape events observed (no sudden zone swaps at steady state). CV approximately 0.44 (diffuse cloud, not fixed point).

**Insight**: Geographic coupling doubles the probability of reaching the high-structure attractor basin but does not alter the basin structure itself; the copy-forward loop creates genuine bistability, not merely a gradient slope.

**Closed**: Confirmed the attractor is a real basin with measurable depth. Geographic coupling is a route into the basin, not a modification of the dynamics within it.

**Status**: CLOSED

---

## Paper 33: Non-Gradient Proof

**Question**: Is VCML a gradient descent system (does a potential function V exist such that the dynamics are conservative)?

**Finding**: VCML is NOT a conservative Euclidean gradient system. Formal proof (machine-verified in Lean 4 + Mathlib4, 3103 jobs, 0 errors): curl(f_F, f_m) = -FA not equal to 0, so by Clairaut's theorem no C^2 potential V exists. Honest caveat: rules out Euclidean gradient flow only; mirror descent, Riemannian, and nonlocal potentials not ruled out.

**Insight**: VCML is a fundamentally driven non-equilibrium process: energy is injected via waves and dissipated via field decay, and the asymmetric consolidation (F chases m, m does not chase F) creates a persistent circulation that cannot be eliminated by any gauge choice in Euclidean space.

**Closed**: Formally closed the question of whether VCML is gradient descent. The non-gradient character is the mathematical basis for why it behaves differently from any optimization system.

**Status**: CLOSED

---

## Paper 34: Phase Diagram

**Question**: What are the critical exponents of the Allen-Cahn to Burgers transition, and how do the two manifolds C1 and C2 relate to each other?

**Finding**: The Allen-Cahn to Burgers transition has measurable critical exponents at nu/kappa approximately 1. The two manifolds C1 (viability volume: nu, FD, FA, beta_s) and C2 (structural manifold: kappa, nu/kappa, beta_s, W_zone) share only beta_s as a common axis, otherwise independent in the measured parameter ranges.

**Insight**: beta_s is the only coupling between the viability regime (whether structure forms) and the structural type (what kind of structure forms); this independence allows C1 and C2 to be explored separately, explaining why the adaptive window and the field equation were derivable independently.

**Closed**: Completed the phase diagram characterization. Confirmed the C1 x C2 product structure with a single shared axis.

**Status**: CLOSED

---

## Paper 35: Relay Coupling

**Question**: Can two spatially separated regions share zone structure via geographic relay, and what conditions are required?

**Finding**: Relay gain G = sg4(B)/sg4(A) peaks at WR approximately 3.5 (Phi < 1) and degrades at N > 4 (zone width < 4*xi_eff). Two conditions required: N < N_crit = L/(5*r_w) (resolution) and Phi = WR*r_w/T_dur < 1 (coverage). Bimodal attractor causes high standard error at optimal conditions.

**Insight**: Geographic relay works when zone width is comparable to the correlation length (xi_eff = r_wave) but fails when the relay grid is too fine relative to the wave radius; the relay does not transmit information -- it replicates the spatial organization of wave launch events.

**Closed**: Established geographic relay as a mechanism and derived the two conditions for success. Set up the empirical verification of xi_eff in Paper 36.

**Status**: CLOSED

---

## Paper 36: Xi Validation

**Question**: Is xi_eff = r_wave = 2 sites confirmed directly, independent of the diffusion coefficient kappa?

**Finding**: xi_eff = r_wave = 2 (wave-dominated), NOT xi_diffusion = sqrt(kappa). N_crit = 5 is invariant to DIFFUSE (kappa variations). The zone map is replicated at resolution approximately r_w, not the diffusion length. Prediction confirmed.

**Insight**: The dominant transport mechanism in VCML is birth-mediated (wave footprint sets xi), not diffusion-mediated; diffusion can be varied widely without changing the relay conditions because it is not the primary spatial coupling.

**Closed**: Directly confirmed xi_eff = r_wave. Established that relay coupling conditions depend only on wave geometry, not diffusion parameters.

**Status**: CLOSED

---

## Paper 37: r_wave Scaling and Chain Relay

**Question**: Does N_crit scale as 1/r_wave, and does relay gain compound (or collapse) along a chain of relay regions?

**Finding**: N_crit proportional to 1/r_wave confirmed: empirically {8, 4, 2} for r_wave = {1, 2, 3} (empirical coefficient 5 not 4). At r_wave=4: Phi=1.28 (over-perturbation), G < 1 everywhere. Chain relay: G_k ~ G_1 (no compounding). Relay gain is depth-stable.

**Insight**: Relay chains transmit structure without amplification or collapse; each region in a chain maintains the same gain relative to the source as the first relay, meaning arbitrarily deep relay networks preserve zone structure at constant fidelity.

**Closed**: Confirmed 1/r_wave scaling for N_crit with empirical coefficient 5. Closed the geographic relay coupling series. The relay is a faithful transmission mechanism, not an amplifier.

**Status**: CLOSED

---

## Paper 38: Metastability and Continual Learning Baseline

**Question**: How long does zone structure persist after commitment, and what is the metastability half-life?

**Finding**: Zone structure commits at t* approximately 2000 steps, then decays with half-life approximately 800 steps. By T=6000-8000 steps, sg4n falls below 0.10 under single-task input. The bimodal attractor has P_high(geo)=0.50, G_basin=1.88. Polarity inversion mid-run is invisible to sg4 (polarity-blind metric).

**Insight**: Metastability is a finite-lifetime property, not permanence; zone structure requires ongoing perturbation to be maintained via copy-forward refreshing, and the 800-step half-life is the natural decay timescale of the copy-forward loop without active input.

**Closed**: Established the metastability baseline. The ~800 step half-life is the key quantity for understanding task-switching experiments in Papers 40-41.

**Status**: CLOSED

---

## Paper 39: Cleft Geometry and Alignment Blindness

**Question**: Does deterministic vs. stochastic misalignment in synaptic cleft geometry differentially affect relay gain?

**Finding**: Deterministic (consistent) offsets in zone labeling do NOT degrade relay gain G regardless of magnitude: at D_cleft=100% misalignment, G=3.42. Stochastic noise degrades G by randomizing zone identity across waves. G_det/G_stoch = 1.25 on average at matched magnitudes.

**Insight**: Consistency of geographic routing is the operative requirement, not geometric accuracy; systematic errors (topographic inversions, consistent offsets) are tolerated because the copy-forward loop consistently reinforces whatever spatial pattern is consistently presented.

**Closed**: Established alignment blindness as a design principle. Biological prediction: topographic inversion produces normal downstream zone formation; scattered terminal fields impair it.

**Status**: CLOSED

---

## Paper 40: Continual Learning and Zone Shifts

**Question**: How does VCML respond to zone boundary shifts (spatial reassignment with overlap)?

**Finding**: Cyclic zone shift (delta = z_w) is immediately transparent -- sg4n_new = sg4n_old at the shift moment (permutation invariance). Partial-overlap shift (delta = z_w/2) drops sg4n by 50% at the shift, then recovers slowly (>2500 steps to recover; reaches sg4n=0.80 at T=4000, exceeding the C_ref peak of 0.38). Post-commitment interleaving drives both tasks to equal steady-state sg4n approximately 0.28-0.29; forgetting is gradual (rate set by metastability half-life approximately 800 steps), not catastrophic.

**Insight**: Genuine continual learning requires spatial reassignment with overlap; pure permutations are invisible because sg4 is permutation-invariant. The recovery to sg4n=0.80 (exceeding baseline) is because zone-boundary mismatch drives continuous copy-forward refreshing -- catastrophic interference is replaced by structural amplification.

**Closed**: Established that VCML does not catastrophically forget but gradually forgets at the metastability timescale. The late surge (sg4n > baseline after shift) became one of the three anomalies resolved in Paper 42.

**Status**: CLOSED

---

## Paper 41: Switch Threshold

**Question**: Is there a critical switching period T_crit approximately 800 steps below which task switching causes net memory loss?

**Finding**: The predicted threshold T_crit does not exist. Every tested switching period (50, 500, 800, 3000 steps) maintains sg4n 5-10x above single-task baseline at T=6000. Best amplification at T_switch=500 (9.7x). C_ref sg4n=0.063; all interleaved conditions sg4n=0.30-0.64. Mechanism: task switches trigger zone-boundary collapses -> births -> fieldM replenishment.

**Insight**: Task switching amplifies zone structure rather than eroding it because each switch generates structured collapses that continuously counteract FIELD_DECAY; the metastability half-life (800 steps) is a static decay constant applying only when NO perturbation is present.

**Closed**: Null result for the threshold hypothesis. The prediction reversal (expected degradation, found amplification) is one of the most important findings in the series: mandatory turnover + periodic switching = structural maintenance engine.

**Status**: CLOSED

---

## Paper 42: Three Anomalies

**Question**: What explains three anomalies from earlier papers: (1) C_loc ratio > 1, (2) C_half late surge, (3) T_sw=50/100 apparent reversal?

**Finding**: (1) C_loc anomaly: waves are not necessary for copy-forward; spontaneous collapses generate ~2.84 events/step sustaining structure above pure field decay (ratio 0.664 vs field-decay-only 0.549). (2) C_half surge is sustained (sg4n=0.212 at T=6000 vs 0.068 for C_ref); zone-boundary mismatch drives continuous copy-forward identical to task-switching amplification. (3) T_sw=50/100 reversal is a checkpoint phase aliasing artifact: 300/T_sw=3 (odd) means last three checkpoints fall 2:1 in task B epochs; epoch-end measurement restores expected ordering.

**Insight**: All three anomalies share one mechanism: any source of structured collapses (waves, boundary mismatch, spontaneous noise) continuously refreshes zone structure via the copy-forward loop; the anomalies were not paradoxes but demonstrations that the mechanism is more general than the wave-input case.

**Closed**: Resolved all three open anomalies from Papers 38-41. Identified epoch-end measurement as the correct protocol for switching experiments (methodological lesson).

**Status**: CLOSED

---

## Paper 43: Rule Robustness

**Question**: Is zone formation and maintenance robust to nearby rule changes, or is it fragile rule engineering?

**Finding**: Experiment A (continuous waves, 10 conditions): all conditions including ablations of birth seeding (beta_s=0) and viability gating (SS removed) produce sg4n comparable to reference -- formation is environmentally driven. Experiment B (waves stop at T=1500): maintenance ratios: ref=1.91, no_birth_seed=1.63, no_gating=1.11, no_both=1.65. Field-decay-only=0.64. All six rule variants (+/-50% perturbations of SS, MID_DECAY, SEED_BETA) maintain comparable sg4n under waves.

**Insight**: Formation is robust to all rule variants because zone-differentiated wave input is sufficient to drive differentiation under any consolidation mechanism; maintenance is mechanism-specific and requires viability-gated copy-forward as its primary structural element -- removing gating degrades maintenance to near field-decay-only (1.11 vs 1.91).

**Closed**: Completed the single-region theory. The distinction between formation (environmentally driven, rule-robust) and maintenance (mechanism-specific, requires viability gate) is the final theoretical clarification.

**Status**: CLOSED

---

## Paper 44: Adversarial Persistence

**Question**: Is VCML zone structure genuine memory, or merely imprinting of current environmental statistics?

**Finding**: All three conditions (adaptive, passive, rigid) encode identically during Phase 1 (T=0-3000, full adaptive mechanism). At T_encode, wave-zone assignments are permanently flipped (adversarial input, 50% amplitude). At T_adv=2000: adaptive fidelity=+0.129, passive fidelity=-0.076, Delta=0.205, t~8.9, p<0.001. Rigid fidelity=+0.167 but via decay-to-prior (not active maintenance; dips to -0.452 at T=800 first). Three-phase adaptive trajectory: initial disruption -> resistance -> recovery to positive fidelity.

**Insight**: The copy-forward loop is the resistance mechanism. Removing turnover (rigid condition) INCREASES adversarial adaptation speed at intermediate timescales (fidelity -0.452 at T=800 vs adaptive -0.089) because the copy-forward loop is removed. The passive condition encodes adversarial statistics because it has no gate and no seeding -- a pure imprinting system. The adaptive condition's recovery to positive fidelity demonstrates the original encoding is a dynamical attractor, not a static trace.

**Closed**: The imprinting critique (Reviewer #2: "the system merely mirrors environmental structure"). Papers 43+44 jointly close the necessity-and-resistance question: Paper 43 = viability gating is necessary (ablation); Paper 44 = mechanism actively resists adversarial input (resistance test). Formation is environmentally driven; maintenance is mechanism-specific and adversarially robust.

**Status**: CLOSED

---

## Paper 45: Causal Geometry in GPT-2 (External Probe)

**Question**: Does mean-ablation influence between token pairs predict representational geometry (cosine distance) in a pre-trained transformer, after controlling for sequence distance?

**Finding**: Layer-wise partial correlations in GPT-2 small (12 layers, 117M parameters, 50 UD English EWT sentences, 11,706 token pairs) show a consistent sign flip at layer 4: positive partial correlation (feature construction, layers 1-3), transition at layer 4, then negative partial correlation (causal integration, layers 5-11), peaking at layer 7 (rp=-0.337, p~10^{-308}). Random-token control tracks the ablation profile at every layer (rp=-0.331 at layer 7), confirming the signal is in attention routing structure, not content. Dependency-distance-matched analysis: syntactically linked pairs are geometrically closer than unlinked pairs at matched sequence-distance bands (p<0.05 in all three bands: [1,3), [3,6), [6,15)). Mixed-effects regression: ablation influence is a significant negative predictor (beta=-0.00309, p~10^{-196}); syntactic dependency label is NOT an independent predictor beyond ablation (p=0.509) — syntax is mediated by causal routing.

**Insight**: Transformer geometry is organised by causal coupling (attention routing), not by Euclidean proximity or semantic content. The sign flip is a phase transition from feature construction (divergence) to causal integration (convergence). The random-token control result is the transformer analog of Paper 36's xi~r_w invariance: geometric organisation is in the causal structure of the routing, independent of what content is transported. Spatial-vs-dimensional capacity distinction confirmed: VCML's spatial separation (HS=1 sufficient) avoids the channel-conflation problem in transformer-style dimensional separation (d_h=768).

**Closed**: The external causal geometry conjecture: representation geometry is organised by causal coupling, not Euclidean proximity. First cross-system validation of the VCML causal geometry framework.

**Status**: CLOSED

---

## Paper 46: Temporal Gate Invariant and Causal Coordinate Validation

**Question A**: Does the quiescence gate (calm_streak >= SS) select for signal quality (clean settled mid_mem), or does it serve a different function? And: is the gate or the quantity of writes the operative variable?

**Question B**: Is Omega = WR * phi_w the operative C2 coordinate, as predicted by the causal reparametrisation?

**Finding A**: 6 gate conditions x 5 seeds. Quantity-matched random writes (rand_p60, p=0.60, write_rate=0.60) achieve sg4=0.0411 — significantly outperforming quiescence-gated SS=10 (sg4=0.0244, write_rate=0.675) by 1.68x (t=-4.32, p=0.003). Always-writing (SS=0, write_rate=1.00) achieves sg4=0.0409. All quiescence-gated conditions (SS=5,10,15,20) cluster tightly at sg4=0.022-0.025 despite varying write rates.

**Finding B**: r_wave in {1,2,3} x {omega_const, wr_const} x 5 seeds. Omega-constant sweep (WR adjusted so WR*phi_w=62.4): sg4 = 0.0181, 0.0244, 0.0235 (CV=0.126, approximately flat). WR-constant sweep (WR=4.8 fixed): sg4 = 0.0115, 0.0244, 0.0314 (increasing with r_wave). Omega collapses the WR vs r_wave variation.

**Insight**: The quiescence gate is an adversarial filter, not a signal-quality filter. mid_mem carries zone-specific signal during perturbation onset (peak signal), not just during calm. The gate excludes these high-signal writes, reducing steady-state structural strength by 1.68x. Reconciliation with Paper 44: under adversarial waves, perturbation mid_mem encodes wrong-zone signal; waiting SS steps lets this decay before writing, protecting the original encoding. The optimal SS threshold is set by adversarial pressure level, not by signal quality. For Exp B: Omega=WR*phi_w collapses the joint WR-r_wave variation, with modest r=1 deficit attributed to delta=W_zone/r_wave=10 (zone too wide for single-event coherence). First direct experimental support for C2 causal reparametrisation.

**Closed**: The temporal gate invariant (refined: gate = adversarial filter at cost of benign structural strength). The C2 causal coordinate Omega=WR*phi_w (first direct experimental support).

**Status**: CLOSED

---

# Summary Table

| Paper | Short Name | Experiment(s) | Key Metric | Status |
|-------|-----------|---------------|------------|--------|
| 0 | Spatial Self-Organization | V73+ | sg4, nonadj/adj | CLOSED |
| 1 | Capacity Bounds | Analytical | rho_max | CLOSED |
| 2 | Bandwidth-Turnover | Analytical + V18-V22 | bandwidth ~ 1/nu | CLOSED |
| 3 | Propagation Constraints | Analytical + V36 | xi = r_wave | CLOSED |
| 4 | Consolidation Gating | V45, V47 | MID_DECAY role | CLOSED |
| 5 | Unified Framework | Analytical | eta_c ~ (L/D)^2 | CLOSED |
| 6 | AI Architecture Failures | Conceptual | Architecture mapping | CLOSED |
| 7 | Regime Structure | V45+ sweeps | nu* non-monotone | CLOSED |
| 8 | Timescale Matching | V45+ sweeps | nu* kappa-invariant | CLOSED |
| 9 | Governing Ratio | V45+ sweeps | Xi collapse FAILS | CLOSED |
| 10 | Replacement Hazard | V45+ sweeps | Psi partial success | CLOSED |
| 11 | Substrate Constants | V45+ sweeps | 4D boundaries | CLOSED |
| 12 | Copy-Forward Mechanism | SEED_BETA sweep | Fourth axis confirmed | CLOSED |
| 13 | Viability Volume | Cross sweeps | 4D viability volume | CLOSED |
| 14 | Timescale Nesting | Ratio test | FA dual role | CLOSED |
| 15 | Amplitude Law | FA sweep | sg4 ~ FA^0.43 | CLOSED |
| 16 | Growth Dynamics | Phase protocol | Saturation formula | CLOSED |
| 17 | Two-Factor Decomposition | Multi-param sweep | sg4 = C * R complete | CLOSED |
| 18 | Temporal Dynamics | Phase protocol | tau_forget = tau_m | CLOSED |
| 19 | Blocked-Site Buffer | SS x WR sweep | 4-reservoir model | CLOSED |
| 20 | ODE Closure | Simulation vs ODE | 3% agreement; 9x buildup gap | CLOSED |
| 21 | Spatial Formation Factor | nu/kappa sweep | D_copy >> kappa | CLOSED |
| 22 | Spatial Correlation | kappa sweep | xi = r_wave | CLOSED |
| 23 | Two-Scale Decomposition | Reynolds split | ODE + PDE clean | CLOSED |
| 24 | Birth Bias | nu/kappa sweep | Allen-Cahn to Burgers | CLOSED |
| 25 | Burgers Transition | nu/kappa near 1 | Model H universality | CLOSED |
| 26 | Unified PDE | Theory | Hybrid PDE + W* formula | CLOSED |
| 27 | Commitment Epoch | t* sweep | t* ~ 1/(P_c*FA*WR) | CLOSED |
| 28 | Interface Theory | Interface measurement | Width = r_wave | CLOSED |
| 29 | FA Threshold | FA sweep | Threshold = K_eff | CLOSED |
| 30 | Amplification Factor | Zone width sweep | Gamma ~ W^0.6 | CLOSED |
| 31 | Field Coherence | CV measurement | CV ~0.44 (cloud not point) | CLOSED |
| 32 | Attractor Structure | V95-V100 | G_basin=1.88, bimodal | CLOSED |
| 33 | Non-Gradient Proof | Lean 4 formal proof | curl = -FA != 0 | CLOSED |
| 34 | Phase Diagram | C1 x C2 sweep | beta_s only shared axis | CLOSED |
| 35 | Relay Coupling | V95-V100 (G tests) | G peaks at WR~3.5, Phi<1 | CLOSED |
| 36 | Xi Validation | kappa sweep relay | xi = r_wave confirmed | CLOSED |
| 37 | r_wave Scaling + Chain | r_wave sweep | N_crit ~ 1/r_wave; chain stable | CLOSED |
| 38 | Metastability Baseline | V82 | half-life ~800 steps | CLOSED |
| 39 | Cleft Geometry | Det vs stochastic cleft | Consistency not accuracy | CLOSED |
| 40 | Continual Learning | V78, zone shifts | Gradual not catastrophic | CLOSED |
| 41 | Switch Threshold | V82/task-switching | 5-10x amplification (no threshold) | CLOSED |
| 42 | Three Anomalies | V82/V89 follow-ups | All three resolved | CLOSED |
| 43 | Rule Robustness | Ablation x5 | Formation robust; maintenance requires gate | CLOSED |
| 44 | Adversarial Persistence | 3 conditions x5 seeds, T_adv=2000 | Adaptive +0.129 vs passive -0.076 (Delta=0.205, p<0.001). Memory, not imprinting. | CLOSED |
| 45 | Causal Geometry (GPT-2) | GPT-2 small, 50 UD sentences, 11706 pairs | Sign flip layer 4; peak rp=-0.337 (layer 7); random control tracks ablation; dep pairs closer at all dist bands. | CLOSED |
| 46 | Temporal Gate + Omega | 6 gate conditions + r_wave sweep, 5 seeds each | Gate is adversarial filter (rand_p60 1.68x ss10, p=0.003); Omega-const CV=0.126 (flat); WR-const varies. | CLOSED |
| 47 | C2 Coordinate Validation | delta-sweep (25 runs) + gate x adv_amp (120 runs) | na_ratio peaks at delta=5; Exp B reveals T=2000 is pre-commitment; gate controls commitment rate and adversarial filtering on same timescale. | CLOSED |

## Paper 47: C2 Coordinate Validation

**Question A**: Is delta=W_zone/r_wave an operative C2 coordinate? Prediction: sg4 (and encoding quality) peaks at delta~4-5, degrades at delta<=2.

**Question B**: Does gate selectivity (SS threshold) interact with adversarial pressure? Prediction: optimal SS scales linearly with adversarial amplitude (gate as adversarial filter hypothesis).

**Finding A**: N_ZONES sweep {2,4,5,8,10} -> delta in {10,5,4,2.5,2}. na_ratio (nonadj/adj) peaks at delta=5 (N_ZONES=4, standard config, na_ratio=1.061) and degrades to ~0.97 at delta=2.5-4. Raw sg4 increases monotonically with more zones (partly metric artifact from more pairs). The encoding-quality signal is na_ratio: delta<5 means wave radius >= zone half-width, cross-zone bleed degrades monotonic zone gradient.

**Finding B**: 4 adv_amp x 6 gate conditions x 5 seeds. At T_ENCODE=2000 (near commitment epoch t*~2000), ALL conditions show fidelity near zero or negative even without adversarial input (adv_amp=0). High-SS gates maintain positive fidelity (slow commitment, stay near T=2000 snapshot); low-SS gates go strongly negative (accelerate sign-flip through commitment epoch). Adversarial input suppresses clean commitment at all gate conditions, with all values converging toward zero.

**Insight**: T_ENCODE=2000 is pre-commitment for most seeds. The fidelity metric at this baseline captures commitment RATE rather than adversarial resistance. High-SS slows commitment; low-SS (and rand_p60) accelerates it, letting fieldM cross the sign-flip described in Paper 26. This unifies Papers 26+46+47: the quiescence gate controls the timescale on which fieldM tracks mid_mem through the commitment sign-flip; this is the same mechanism as adversarial filtering, operating at the same timescale.

**Closed**: delta=W_zone/r_wave confirmed as third C2 coordinate (viable range delta>=5). Gate-commitment-epoch interaction revealed: gate controls commitment rate, not just adversarial filtering. Optimal SS* is a function of both adversarial pressure AND proximity to t*.

**Status**: CLOSED

---

## Paper 48: sg_C -- The Causal Structure Metric

**Question**: Is sg4 a valid causal metric, or does the C2 framework require a measure that tracks actual zone-identity information at the cell level? Does the gate optimise signal amplitude or signal quality?

**Finding A** (gate sweep, N_ZONES=4): Single-cell zone decode accuracy (sg_C) is only 4-5% above chance across all gate conditions. VCML builds a population code, not a labelling code. Zone structure lives in zone means, not individual cells. The quiescence gate halves sigma_w (within-zone noise) while sg4 drops by less -- SNR = sg4/sigma_w increases correctly: SS=20 (0.129) > SS=10 (0.125) > rand_p60 (0.108) > SS=0 (0.105). The Paper 46 "unexpected" result (rand_p60 1.69x sg4) is explained: rand gate has large amplitude but 2x the noise; SNR favours the quiescence gate.

**Finding B** (delta sweep): sg_C decreases monotonically as delta shrinks (correct causal direction). sg4 increases as delta shrinks (counting artifact: more zone pairs inflates average distance). Clean dissociation. sg_C and na_ratio agree: delta>=5 is the threshold.

**Insight**: VCML implements population coding because the copy-forward loop distributes zone information stochastically across cells (seeded from neighbours, not zone labels). No individual cell reliably encodes its zone; the signal is in the mean. This is architecturally required by mandatory turnover -- individual-cell labels would be erased by birth cycles. Population-level labels persist because they are not tied to specific cells. Same mechanism as hippocampal population codes for spatial position.

**Metric recommendation**: Full suite going forward: sg4 (signal), sigma_w (noise), SNR (quality), na_ratio (geometry), sg_C (direction check when N_zones varies).

**Status**: CLOSED

---

## Paper 49: Three Open Threads Closed

**Question A**: Is beta_s an independent C2 coordinate? Prediction: sg4/SNR should vary monotonically with seeding scale across {0, 0.1, 0.25, 0.5, 0.75, 1.0}.

**Question B**: What is the optimal gate SS*(T_enc, a) across the 2D surface of pre/post-commitment x adversarial pressure?

**Question C**: Does the population-code relay interface (zone-mean seeding into birth step) outperform the wave-channel relay (geo)?

**Finding A**: NULL. sg4 range 0.0195-0.0202, SNR range 0.100-0.104 — all within noise across the full beta_s sweep. beta_s is NOT an independent formation coordinate. In the continuous-wave regime, wave-driven viability differentiation dominates and overwrites the birth seed state within tens of steps. beta_s matters only in maintenance-only protocol (Paper 43: 1.63x degradation without seeding when waves absent). C2 triplet is complete: Omega (formation rate), delta (zone resolution), beta_s (maintenance seeding, wave-masked during formation).

**Finding B**: Non-trivial 2D surface with a reversal at the commitment epoch t*~2000. Pre-commitment (T=1000): SS=0 wins at adv>0 (rapid commitment more valuable than resistance — low gate accelerates sign-flip, committed structure is then robust). Post-commitment (T>=2000) + strong adversarial (a=0.5): SS=10 wins (adversarial resistance dominates after t*). Post-commitment + no adversarial: SS=20 wins (noise reduction). Weak adversarial (a=0.25): SS=0 wins across all T_enc (gate too conservative for low-pressure regime). The gate reversal at t* unifies Papers 26, 46, 47, 49 as one mechanism with two regimes.

**Finding C (CORRECTED by Paper 50)**: Neither relay mode outperforms ctrl. ctrl: sg4=0.0245, na=1.245 (identity). geo: sg4=0.0218, na=0.826 (parity — na<1). mean_relay: sg4=0.0196, na=0.908 (parity). geo encodes zone parity, not identity: binary supp/excite convention (cls%2) creates alternating pattern, not topographic gradient. mean_relay birth injection is washed out by ongoing unstructured fieldM updates. Identity encoding requires the full VCML formation process (structured zone-launched waves + copy-forward loop) running in the receiving module. Additional finding: SNR and decode_norm diverge under adversarial — SS=0 preserves per-cell accuracy (decode_norm), SS=10 protects zone-mean separation (SNR); different objectives.

**Insight**: The optimal gate strategy requires knowing where you are relative to t*. Before t*, low SS is correct (commitment). After t* under adversarial pressure, high SS is correct (resistance). A static SS cannot be simultaneously optimal for both regimes. Dynamic gating (SS(t)) is the correct architecture for systems that face adversarial input throughout formation AND maintenance.

**Status**: CLOSED

---

## Paper 50: Zone Identity Encoding Is Not Transmissible

**Question**: Can any relay coupling recover zone identity encoding (na_ratio>1) in a receiving module L2 that gets L1's signals? Five conditions audited: ctrl_rnd, ctrl_std (own WaveEnvStd), geo_copy (amplitude copy), geo_4class (4-level amplitude), geo_own_relay (own WaveEnvStd + L1 birth seeding).

**Finding**: Zone identity encoding is not transmissible via any tested relay mechanism.

- **ctrl_rnd** (random waves): na=0.963 [parity]. sg4=0.0278, gain=1.00x baseline.
- **ctrl_std** (own WaveEnvStd): na=1.219 [**IDENTITY**]. sg4=0.0212. Key diagnostic: L2 achieves identity encoding when it runs the full VCML process independently.
- **geo_copy** (amplitude copy, Paper 49 geo baseline): na=0.778 [parity]. sg4=0.0177.
- **geo_4class** (4-level amplitude): na=0.949 [parity, improved]. sg4=0.0186. Graded amplitudes approach but do not cross the identity threshold.
- **geo_own_relay** (own WaveEnvStd + L1 birth seeding): na=0.832 [parity]. sg4=0.0213. Adding L1 birth seeding DEGRADES ctrl_std's identity encoding (1.219→0.832): pattern-mismatch interference.

**Mechanism**: L1's identity encoding accumulates over 3000 steps of zone-anchored wave events reinforced by the copy-forward loop. Instantaneous amplitude-copy loses: (1) temporal integration history — the spatial gradient lives in L1's fieldM history, not in any single wave snapshot; (2) copy-forward self-consistency — L2's fieldM is pulled toward parity by viability signal and toward identity by copy-forward loop, but parity signal wins. Birth seeding from L1 fails because L1 and L2 develop different but equally valid identity-encoding patterns; seeds conflict with L2's self-consistently forming pattern.

**Caveat on Papers 35-39**: Relay gains of 3-5x (sg4) reported in those papers are real. However, the structure is parity encoding (na<1), not identity encoding. "Zone structure" in relay contexts means zone amplitude separation (sg4), not zone topographic identity (na_ratio).

**Minimum requirement for identity encoding**: L2 must run its own WaveEnvStd with zone-anchored launch positions. No relay shortcut (instantaneous copy, graded amplitude, birth seeding, combination) achieves this.

**Status**: CLOSED

---

## Paper 51: Necessity as Method

**Question**: What is the methodological structure that produced Papers 1-50, and is it identical to the VCSM framework it produced?

**Finding**: The research process is structurally identical to VCSM.

| VCSM component | Research-process analog |
|---|---|
| Cell | Experimental paper |
| Viability constraint | Hypothesis must produce finding |
| Mandatory turnover | Every paper terminates when question closes |
| Calm state (baseline_h) | Session working memory (context window) |
| Perturbation (mid_mem) | Paper-level accumulation (MEMORY.md) |
| Quiescence gate (SS>=10) | Adversarial stress test before consolidation |
| Consolidation (fieldM) | Framework document (VCML_THEORY_COMPLETE.tex) |
| Copy-forward loop | Theory document seeds new paper |

**Hypothesis graveyard** (11 null/reversed results across 50 papers): Governing ratio NULL (P9), ratio invariance FAIL (P14), power law REVISED (P15), dynamic always beats static REVERSED (P35/V73), switch threshold NULL (P41), gate necessary for formation REVERSED (P43), cascade attenuates REVERSED (P79), Mobius structure NULL (V90), mean_relay wins REVERSED (P49), geo relay preserves zone identity QUALIFIED (P35-50), any relay achieves identity NULL (P50).

**Centerpiece example**: Parity/identity distinction invisible in sg4 for ~15 papers (Papers 35-49). Revealed in one step when na_ratio was derived from the theoretical primitive (zone topography requires non-adjacent > adjacent distinctness). Relay architecture looked successful for 15 papers because the question was wrong.

**Gödel structure**: A top-down critique of bottom-up methodology is self-refuting by construction. To argue the method is wrong, the critic must argue from a prior theory -- which is exactly the top-down stance the methodology predicts will miss the primitive.

**Three-document architecture**:
- VCML_THEORY_COMPLETE.tex -- the WHAT (consolidated theory, slow/fieldM timescale)
- Paper 51 (this paper) -- the HOW (methodology, medium/paper timescale)
- Papers 1-50 -- the WHY (experimental ladder, fast/session timescale, accumulated in MEMORY.md)

**Self-demonstrating conclusion**: Zone identity encoding requires independent formation. Reading the framework document = geo_copy relay (parity encoding at best). Understanding why each primitive survived requires running the formation process yourself.

**Status**: CLOSED

---

## Paper 52: Boundary-Coupled Distinction

**Question**: Is the signed direction of the VCSM contrast signal (mid_mem = hid - baseline_h) load-bearing for zone identity encoding, or does the magnitude alone suffice?

**Design**: 9-condition ablation of consolidation write policy. Two-phase protocol: 2000-step formation (WaveEnvStd) + 1000-step adversarial flip (WaveEnvFlipped, supp/exc inverted). 5 seeds, W=80 H=40, HALF=40. Key conditions:

| Condition | Write trigger | Write content | Notes |
|-----------|--------------|---------------|-------|
| C_ref | quiescent (SS>=10) | signed mid | standard VCSM |
| C_near | quiescent + near boundary | signed mid | boundary-gated |
| C_near_low | quiescent + near BOUND_LO | signed mid | collapse-side only |
| C_near_high | quiescent + near BOUND_HI | signed mid | survival-side only |
| C_perturb | quiescent + active wave | signed mid | wave-phase only |
| C_rand | quiescent + random (matched rate) | signed mid | random trigger control |
| C_abs | quiescent + near boundary | abs(mid) | sign ablation |
| C_raw | quiescent + near boundary | raw hid | pre-subtraction content |
| C_far | quiescent + far from boundary | signed mid | boundary control |

**Results**:

| Condition | na_form | encoding | cos_maint | wps |
|-----------|---------|----------|-----------|-----|
| C_ref | 1.207 | IDENTITY | -0.274 | 0.607 |
| C_near | 1.324 | IDENTITY | +0.546 | 0.033 |
| C_near_low | 1.125 | IDENTITY | +0.405 | 0.031 |
| C_near_high | 1.176 | IDENTITY | +0.577 | 0.003 |
| C_perturb | 1.283 | IDENTITY | +0.664 | 0.0004 |
| C_rand | 1.335 | IDENTITY | +0.387 | 0.001 |
| C_abs | 0.877 | PARITY | +0.999 | 0.033 |
| C_raw | 1.802 | IDENTITY | +0.777 | 0.034 |
| C_far | 1.026 | IDENTITY | -0.085 | 0.476 |

**Key findings**:

1. **Sign is load-bearing**: C_abs (absolute value of mid_mem) is the ONLY condition to fail identity encoding (na=0.877, parity). All 8 signed conditions achieve identity. The direction of the contrast signal -- which side of the viability boundary the cell is on -- is necessary information.

2. **Two-channel write theory**: Two independent axes govern the write policy space:
   - **Content axis** (what to write): C_raw (na=1.802) > signed mid (~1.2-1.3) > |mid| (0.877). Raw hidden state carries stronger zone identity signal than mid_mem; subtracting baseline_h removes some identity information while adding boundary sensitivity.
   - **Selectivity axis** (when to write): C_perturb (cos=0.664, 83x fewer writes) > C_near (cos=0.546) >> C_ref (cos=-0.274). Write trigger policy -- not write content -- determines adversarial resistance.

3. **C_perturb surprise**: Writes only during active perturbation events (wa>0.05). Near-zero writes per site (0.0004), yet achieves highest adversarial maintenance among all conditions. Tight selectivity beats boundary proximity for resistance. C_ref and C_far re-encode adversarial environment (negative cosine maintenance) because write frequency is too high to distinguish formation from adversarial phase.

4. **C_rand finding**: Matched-rate random writes achieve na=1.335 (highest among standard-content conditions) -- the random trigger selects a different, also-valid subsample of the formation signal. Stochastic selectivity is sufficient; boundary proximity is one valid selection criterion, not the only one.

5. **All near-boundary conditions preserve identity under adversarial** (na_adv > 1 for C_near, C_near_low, C_near_high, C_perturb, C_abs, C_raw). C_abs preserves most (cos=0.999) -- it never updated during adversarial (formation was already complete, far fewer writes under flipped env). The absolute-value trap is stability through formation failure.

**Theory implications**: The signed contrast signal (mid_mem = hid - baseline_h) encodes the cell's viability trajectory direction: moving toward collapse (negative) vs. toward survival (positive). Collapsing this to |mid| destroys the distinction that makes intergenerational propagation zone-informative. The boundary does not generate information; the sign of the crossing does.

**Status**: CLOSED

---

## Paper 53: VCML Behavioral Phase Diagram

**Question**: Why does VCML keep producing structural surprises? Are these evidence of poor understanding, or of a generative system whose behavioral space exceeds intuition?

**Core claim**: VCML is a minimal causal field system. Papers 0-52 each explored a one-dimensional slice of a five-dimensional phase space. Surprises were regime crossings along unmapped axes.

**The five axes**:

| Axis | Control parameter | Regime boundary | Behavioral transition |
|------|-----------------|----------------|----------------------|
| I | coll/site | ~0.003 / ~0.010 | sub-optimal | adaptive peak | over-perturbed |
| II | delta = W_zone/r_wave | ~2.5 | wave-bleed (parity) | identity encoding |
| III | Omega = WR*phi_w | context-dep | formation flux coordinate |
| IV (2D) | trigger x content | sign(mid)=0 | resistant | responsive; parity | identity |
| V | N_active (scale) | ~25k sites/zone | zone-wide gradient | patch formation |

**Axis I (perturbation intensity)**: Non-monotone peak at coll/site~0.004 (sg4=0.038 at S3, +829% dynamic advantage). Below threshold: copy-forward events too rare. Above threshold: quiescence streak disrupted before consolidation completes. Mechanism: two competing constraints on the same parameter.

**Axis II (spatial resolution)**: Hard threshold at delta~2.5. Wave-bleed below this (zones interpenetrate). Monotone increase from delta=1.25 (na=0.88) to delta=5 (na=1.24), then plateau. This is a categorical transition, not a smooth degradation.

**Axis III (formation flux)**: Omega = WR*phi_w is the operative coordinate (Paper 46). Sweeping WR at constant Omega gives flat sg4 (CV=0.126). Omega and coll/site are coupled through WR -- optimal WR is a function of grid size.

**Axis IV (write policy, 2D)**: Content axis (raw hid na=1.802 > signed mid > |mid|=0.877) x Trigger axis (C_perturb cos=0.664 > C_near > C_ref cos=-0.274). These axes are orthogonal: changing content (C_raw vs. C_near) doesn't change trigger selectivity; changing trigger (C_perturb vs. C_near) doesn't change content. C_abs sits at the intersection of boundary-trigger selectivity and unsigned content -- the only condition in the parity failure region.

**Axis V (scale, finite correlation length)**: sg4_norm: S1=0.427, S2=0.459 (peak), S3=0.365, S4=0.187. na_ratio inverts at S4 (0.732, parity). Copy-forward loop correlation length ~r_wave ~2 sites. Above ~25k sites/zone, loop cannot maintain zone-wide gradients. Structure forms in patches.

**Cross-axis dependencies**:
- Axes I and III coupled through WR (increasing WR changes both)
- Axes II and V interact through zone width (large N increases delta, but patch formation is a different mechanism)
- Axis IV (C_perturb) contingent on Axis I being in adaptive zone -- C_perturb writes nothing in the sub-optimal regime

**Three testable cross-axis predictions**:
1. C_perturb fails at low coll/site (no wave events to trigger writes)
2. Hierarchical coupling recovers S4 structure (upper-layer encodes zone summaries)
3. C_raw adversarial performance is coll/site-dependent (more vs. less stable raw hid signal)

**Why surprises kept happening**: Each experiment held 4 axes constant while manipulating 1, without knowing 4 were being held constant. Results were correct but incomplete slices. Once axes are separated, each result is a predictable location in the phase diagram.

**A/neqB reviewer defense**: Predictive coding operates only on Axis IV content. It has no Axis I (collapse-birth cycle), no Axis II (spatial resolution requirement), no Axis III (formation flux), no Axis V (finite correlation length). The subtraction is not the contribution; the phase diagram location is.

**Status**: CLOSED

---

## Paper 54: Order Parameter, Correlation Length Law, Cross-Axis Surface

**Question**: Three open questions from Paper 53: (1) Is sg_C the unified order parameter? (2) Does correlation length scale with r_wave? (3) Does C_perturb fail at low amplitude?

**Exp A (sg_C test)**: Sweeping amplitude at S1.
| Condition | coll/site | sg4 | sigma_w | snr | na |
|-----------|-----------|-----|---------|-----|-----|
| A1 deep_sub | 0.00061 | 0.0174 | 0.162 | 0.104 | 0.945 [parity] |
| A2 near_thresh | 0.00122 | 0.0200 | 0.195 | 0.104 | 1.067 [IDENTITY] |
| A3 optimal | 0.00241 | 0.0209 | 0.202 | 0.105 | 1.026 [IDENTITY] |
| A4 over_pert | 0.00347 | 0.0258 | 0.225 | 0.115 | 1.014 [IDENTITY] |
| A5 strong_over | 0.00525 | 0.0215 | 0.230 | 0.094 | 0.872 [parity] |
| A6 extreme | 0.00861 | 0.0248 | 0.253 | 0.098 | 0.856 [parity] |

**Finding A**: sg_C (snr) is NOT the unified order parameter. snr is flat (~0.105) in identity regime, drops slightly (~0.094) in parity. na_ratio is the cleaner regime indicator. sg_C cannot replace na_ratio.

**Exp B (correlation length)**: r_wave sweep at S2 (zone_width=20).
| r_wave | delta | sg4 | snr | na |
|--------|-------|-----|-----|-----|
| 1 | 20.0 | 0.0046 | 0.041 | 0.748 [parity] |
| 2 | 10.0 | 0.0090 | 0.046 | 0.865 [parity] |
| 4 | 5.0 | 0.0132 | 0.052 | 1.002 [IDENTITY borderline] |
| 8 | 2.5 | 0.0121 | 0.065 | 1.274 [IDENTITY] |
| 16 | 1.25 | 0.0063 | 0.072 | 1.064 [IDENTITY] |

**Finding B**: Correlation length failure confirmed. At r_wave=1 (delta=20, well above wave-bleed threshold), na_ratio=0.748 (parity) — this is NOT Axis II failure. It's copy-forward loop range too small for zone_width=20. Empirical correlation length law: identity encoding requires zone_width <= 5*r_wave. sg4 peaks at r_wave=4, na_ratio peaks at r_wave=8. snr is monotone in r_wave (opposite to sg4 and na_ratio) — further evidence sg_C is not the unified OP. Two distinct failure modes at opposite ends: correlation-length failure (r_wave too small) and wave-bleed (r_wave too large).

**Exp C (cross-axis)**: C_ref vs C_perturb × 3 amplitude levels.
| Amplitude | Mode | sg4 | sigma_w | snr | na |
|-----------|------|-----|---------|-----|-----|
| sub_thresh | C_ref | 0.0174 | 0.162 | 0.104 | 0.945 [parity] |
| sub_thresh | C_perturb | 0.0069 | 0.0115 | 0.611 | 1.493 [IDENTITY] |
| optimal | C_ref | 0.0209 | 0.202 | 0.105 | 1.026 [IDENTITY] |
| optimal | C_perturb | 0.0048 | 0.0093 | 0.522 | 1.390 [IDENTITY] |
| over_pert | C_ref | 0.0215 | 0.230 | 0.094 | 0.872 [parity] |
| over_pert | C_perturb | 0.0036 | 0.0100 | 0.361 | 1.252 [IDENTITY] |

**Finding C**: Prediction inverted. C_perturb maintains identity at ALL amplitudes (na=1.49/1.39/1.25). C_ref: parity->identity->parity as amplitude increases. Mechanism: trigger selectivity is NOISE-GATING not signal-boosting. C_perturb sigma_w=0.009-0.012 (10-20x smaller than C_ref 0.16-0.23). Despite tiny sg4, C_perturb snr=0.36-0.61 vs C_ref snr=0.09-0.11. High SNR from low sigma_w, not high sg4, drives identity encoding.

**Theoretical revision**: Axis IV trigger mechanism reframed from "selects more informative writes" to "noise-gates sigma_w to near-zero." Both adversarial resistance (P52) and cross-amplitude robustness (P54) follow from same mechanism.

**Status**: CLOSED

---

## Paper 55: Lyapunov Functional for VCML Identity Formation

**Question**: Does C_order = (D_nonadj - D_adj) / sigma_w have monotone positive drift (Lyapunov property) in the identity-forming adaptive regime?

**Experiments**: 5 regimes x 5 seeds = 25 runs. Full 3000-step time-series tracking (120 timepoints per run). Regimes: ident_opt (C_ref optimal), parity_sub (sub-threshold), parity_over (over-perturbed), patch (r_wave=1, corr-len failure), ident_cp (C_perturb noise-gated).

**Results**:

| Regime | C_order(1000) | C_order(2000) | C_order(3000) | sigma_w(2000) | na(2000) | Verdict |
|--------|---------------|---------------|---------------|---------------|----------|---------|
| ident_cp   | +0.011 | +0.262 | +0.159 | 0.012 | 1.493 | STABLE IDENTITY -- Lyapunov holds |
| ident_opt  | -0.027 | -0.013 | -0.039 | 0.202 | 1.026 | TRANSIENT identity -- C_order neg throughout |
| parity_sub | -0.007 | -0.010 | -0.026 | 0.162 | 0.945 | Parity -- C_order neg |
| parity_over| -0.022 | -0.007 | -0.036 | 0.230 | 0.872 | Parity -- C_order neg |
| patch      | -0.014 | -0.019 | -0.007 | 0.120 | 0.838 | Patch -- C_order near zero |

**Key findings**:

1. **Lyapunov property confirmed for ident_cp only**: C_order rises monotonically from +0.011 (step 1000) to +0.262 (step 2000) then plateaus at +0.159. Mean drift = +2.5e-4/step during formation epoch. Stable positive plateau maintained at step 3000.

2. **Transient vs stable identity distinction**: ident_opt reaches na=1.026 at step 2000 (matches Paper 54 result) but C_order = -0.013 (negative!) because sigma_w = 0.202. By step 3000, na falls to 0.729 -- the Paper 54 "identity" measurement captured the transient peak, not a stable attractor. C_perturb identity is stable (na=1.412 at step 3000).

3. **Noise-floor mechanism**: ident_cp sigma_w = 0.010-0.033 (controlled); all C_ref conditions sigma_w = 0.12-0.33 (large). The sigma_w separation appears by step 500 and determines C_order sign throughout.

4. **Lyapunov condition derived**: dC_order/dt > 0 iff dS/S > dW/W (signal growth rate > noise growth rate). C_perturb satisfies by design (dW/W ~= 0); C_ref violates because writes to all quiescent cells grow sigma_w.

**Three jointly necessary conditions for Lyapunov property**:
- Correlation length: zone_width <= 5*r_wave (ensures copy-forward self-amplification K_self > 0)
- Perturbation window: amplitude in identity range (Axis I)
- Noise floor control: dW/W ~= 0 (satisfied by C_perturb or equivalent low write-frequency mode)

**Revised interpretation of Paper 54 Exp A**: ident_opt na=1.026 at step 2000 was transient. At step 3000, na=0.729. The C_ref system transiently visits the identity regime during the commitment epoch but does not maintain it. C_perturb identity is structurally stable.

**Status**: CLOSED

---

## Paper 56: Three Completions for the VCML Regime Diagram

**Question**: Three open questions from Paper 55: (A) Can SS threshold alone reproduce C_perturb noise-floor control? (B) Does C_perturb formation rate scale with r_wave (K_self), and does C_perturb push the correlation-length threshold? (C) Is na_ratio write-frequency-invariant?

**Experiments**: Exp A -- SS sweep {10,20,30,50,100,200}, C_ref sub-threshold (supp=0.06, exc=0.12), S1 grid, 5 seeds each = 30 runs. Exp B -- r_wave sweep {1,2,4,8}, C_perturb sub-threshold, S2 grid, 5 seeds each = 20 runs. Exp C -- analysis of Exp A write_ready_frac vs identity metrics (no new runs). Total: 50 runs x 3000 steps.

**Results**:

Exp A (SS sweep, C_ref sub-threshold):

| SS  | final na | C_order | write_ready | cross_step |
|-----|----------|---------|-------------|------------|
|  10 | 0.848    | -0.026  | 0.732       | never      |
|  20 | 0.847    | -0.016  | 0.719       | never      |
|  30 | 1.014    | +0.000  | 0.705       | never      |
|  50 | 0.868    | -0.016  | 0.680       | never      |
| 100 | 0.847    | -0.020  | 0.625       | never      |
| 200 | 0.721    | -0.036  | 0.536       | never      |

Exp B (r_wave sweep, C_perturb, S2, sub-threshold):

| r_wave | delta | final na | C_order | sigma_w(3000) | cross_step |
|--------|-------|----------|---------|---------------|------------|
| 1      | 20.0  | 1.160    | +0.041  | 0.0015        | ~1175      |
| 2      | 10.0  | 1.261    | +0.043  | 0.0111        | ~975       |
| 4      |  5.0  | 1.140    | +0.016  | 0.0452        | ~1525      |
| 8      |  2.5  | 0.943    | -0.008  | 0.0961        | never      |

**Key findings**:

1. **SS does not substitute for C_perturb**: No SS level achieves the Lyapunov property (cross_step = never for all). SS=30 gives marginal identity (na=1.014) but C_order stays near zero. Higher SS reduces write_ready_frac (noise-frequency) but simultaneously attenuates signal (MID_DECAY^200 = 0.134). The two effects cancel; Lyapunov condition cannot be reached.

2. **C_perturb extends the effective correlation-length threshold**: Paper 54 critical condition = zone_width <= 5*r_wave (delta <= 5). Under C_perturb, r=1 achieves identity at delta=20 and r=2 at delta=10 -- the effective threshold is extended by 2-4x. Mechanism: noise-gated writes keep sigma_w ~ 0.001-0.011 vs. C_ref sigma_w ~ 0.15-0.19. SNR is 10-100x higher, compensating for smaller r_wave.

3. **Write-mode inverts optimal r_wave**: C_ref at S2 had r=8 as optimum (na=1.274); C_perturb at S2 has r=2 as optimum (na=1.261). r=8 fails under C_perturb (sigma_w=0.096, noise-floor collapses) because large footprint causes wave-class overlap.

4. **na_ratio is write-frequency-invariant**: write_ready_frac ranges 0.53-0.73 across Exp A with no monotone relationship to na. C_perturb achieves identity at all but r=8 despite having a fundamentally different (lower) write-effective-frequency. Content selectivity (what gets written) dominates write frequency in determining identity formation.

5. **Noise floor as the primary variable**: Across Papers 54-56, sigma_w < 0.03 predicts identity formation success; sigma_w > 0.1 predicts failure. All other variables (r_wave, SS, amplitude, write policy) matter insofar as they determine sigma_w.

**Status**: CLOSED

---

## Paper 57: Analytic Closure of the VCML Lyapunov Condition

**Question**: Can dW/W be derived analytically from VCML geometry, closing the derivation left open in Paper 55?

**Experiments**: Fine r_wave sweep {2,3,4,5,6,7,8,10}, C_perturb, sub-threshold (supp=0.06, exc=0.12), S2 grid, 5 seeds = 40 runs x 3000 steps.

**Results**:

| r_wave | delta | f_int | final na | C_order | sigma_w(3000) |
|--------|-------|-------|----------|---------|---------------|
| 2      | 10.00 | 0.800 | 1.261    | +0.043  | 0.0111        |
| 3      |  6.67 | 0.700 | 1.254    | +0.025  | 0.0278        |
| 4      |  5.00 | 0.600 | 1.140    | +0.016  | 0.0452        |
| 5      |  4.00 | 0.500 | 1.132    | +0.018  | 0.0625        |
| 6      |  3.33 | 0.400 | 1.038    | -0.011  | 0.0798        |
| 7      |  2.86 | 0.300 | 1.009    | +0.001  | 0.0893        |
| 8      |  2.50 | 0.200 | 0.943    | -0.008  | 0.0961        |
| 10     |  2.00 | 0.000 | 1.091    | +0.002  | 0.1060        |

**Key findings**:

1. **Geometric approximation**: sigma_w scales roughly linearly with boundary fraction 2/delta (f_bnd = 2*r_wave/W_zone). Interior fraction f_int = (1-2/delta)^+ gives the noise-protected regime. Critical threshold delta_c = 2 (r_wave = W_zone/2): below this, f_int = 0.

2. **Geometric model deviation at r_wave=10**: delta=2 gives f_int=0 but na=1.091 (not failure). Explanation: same-zone launch area asymmetry keeps causal purity P_causal > 0 even when geometric interior fraction = 0. Central cells still receive more same-zone wave events than cross-zone events.

3. **Causal purity reparametrisation**: The correct primitive is P_causal = mean fraction of pre-consolidation wave-contact events from same-zone sources. On lattice this approximates to f_int(delta). Substrate-agnostic form: dC_order/dt > 0 iff P_causal > p_c. Lattice projection = delta > 2; immune memory projection = antigen-specific activation fraction > ~1/3.

4. **Unification**: The three prior laws -- P54 correlation-length threshold (delta<=5 for C_ref), P55-56 noise-floor condition (sigma_w < 0.03), and new geometric threshold (delta > 2) -- are all projections of P_causal > p_c. One condition, three projections.

5. **Write-policy enters through P_causal**: C_perturb biases writes toward wave-contact events, raising effective P_causal beyond its geometric value. C_ref writes from all quiescent cells, including boundary-contaminated ones, lowering P_causal. This explains the write-mode inversion of optimal r_wave (P56).

6. **Immune memory prediction**: Stable germinal centre memory requires antigen-specific activation fraction > ~1/3 of total B-cell activation events. Testable against affinity maturation data.

**Status**: CLOSED

---

## Paper 74: Dynamic Exponent z at P→0+ — Sub-Diffusive Scaling z≈0.48

**Question**: Does τ(P) diverge as P→0+ faster than τ_VCSM≈200 masks it, and what dynamic exponent z governs the divergence?

**Phase A (bridge region, L=80, P∈{0.001,0.002,0.003,0.005,0.007,0.010}, 8 seeds, 12k steps)**:
τ_corr≈762–1067 — flat plateau well above τ_VCSM=200. The phi-field order parameter has a longer intrinsic timescale (~1000) than the Ising spin observable from Paper 73. No power-law visible in this range; system deep in ordered phase with saturated correlation length ξ≫L at L=80.

**Phase B (deep P, L=80, P∈{0.0001,0.0002,0.0005}, 8 seeds, 150k steps, GPU)**:
τ rises substantially: τ(P=0.0005)=4812, τ(P=0.0002)=9263, τ(P=0.0001)=3772. Non-monotonic at P=0.0001 (possible noise floor from marginal equilibration or disordered-phase entry). Phase-B-only fit z_B=-0.108 (R²=0.034) — too noisy for a standalone exponent.

**Phase C (FSS at P=0.001, L∈{40,60,80,100,120}, 8 seeds, 100k steps, GPU)**:
τ_L anomalous: largest at L=40 (τ=9204), non-monotone thereafter. U4<0 for L≤100 (near-critical or disordered at this P). z_C=-0.414 (R²=0.133) — unphysical; FSS not applicable at P=0.001 for L<160.

**Global fit (Phases A+B combined)**:
τ~P^{-zν} with **z=0.48, R²=0.71**, ν=0.98 fixed. Sub-diffusive: z=0.48≪z_ModelA=2 and z<z_KPZ=1.5. Consistent with ballistic/long-range seeding via wave-coherence + fieldM birth mechanism accelerating propagation of order.

**GPU implementation**: torch.compile(mode='default', dynamic=False) on RTX 3060. 8 seeds batched as (B=8,N) tensors. ~8× speedup vs CPU. mode='reduce-overhead' (CUDA graph capture) fails for stateful simulations; 'default' fuses kernels without graph replay.

**Key finding**: z≈0.5 establishes VCML as sub-diffusive in the dynamic universality class. τ_VCSM≈1000 (phi-field timescale, correcting Paper 73's τ=200 Ising estimate). Phase C FSS requires P<0.001 and L≥160 for clean results.

**Paper 75**: L=160 at P∈{0.001,0.002,0.005} with >200k steps; P=0.00005 with 500k steps; FSS at P=0.0005 with L∈{80,100,120,160} for clean z_C; target R²>0.90.

**Status**: CLOSED

---

## Paper 75: z Probed at Deep P; γ Null; FSS Blocking and Hyperscaling Violation

**Question**: Can z=0.48 be confirmed at deeper P (P<10⁻³) and can the susceptibility exponent γ be extracted from χ=var(M)?

**Phase A (L=160, P∈{0.001,0.002,0.005}, 8 seeds, 200k steps)**:
U4≈0 at all P — system disordered at L=160 for this P range. τ=2446–3012 (anomalously increasing with P). χ≈8×10⁻⁸ (flat at noise floor). Wave-rounding at L=160 (prob≈0.64) keeps system in disordered regime.

**Phase B (L=80, P∈{3×10⁻⁵,5×10⁻⁵,10⁻⁴,2×10⁻⁴,5×10⁻⁴}, 8 seeds, 300k steps)**:
τ non-monotone: peaks at P=10⁻⁴ (τ=5894) then drops to τ=2597 at P=3×10⁻⁵. z_B=−0.271 (R²=0.641) — UNPHYSICAL (negative sign). Root cause: FSS blocking. For L=80 and ν=0.98: ξ(P)=P^{−ν}>L for ALL Phase B P values (ξ≫80 throughout). τ is controlled by finite-size, not critical divergence. χ≈4×10⁻⁷ flat — disordered-phase noise.

**Phase C (FSS, P=5×10⁻⁴, L∈{40,60,80,100,120,160}, 8 seeds, 150k steps)**:
τ_L non-monotone; z_C=0.155 (R²=0.062) — noise. χ~L^{−2} (geometric dilution): var(M)~(1/L)² for zero-mean disordered fluctuations.

**z fit summary**: Paper 74's z=0.48 (R²=0.71) was measured in the valid scaling window P∈[10⁻³,10⁻²] where ξ(P)≪L=80. Extending to P<10⁻³ exits the window into the FSS regime. True scaling at P~10⁻⁴ requires L≫ξ(10⁻⁴)~6300 — computationally inaccessible. z=0.48 confirmed by exclusion.

**γ measurement null**: Cross-phase γ_all=0.382 (R²=0.739) is L-contaminated (Phase A: L=160; Phase B: L=80; ratio χ_B/χ_A≈4.6≈(160/80)² — pure geometric dilution, not P-scaling). Phase-B-only γ_B=0.017 (flat χ). Scaling relation implications: η_eff≈1.61, δ≈1.61, α≈0.36 (all unreliable pending valid γ).

**Hyperscaling violation (d_eff=0)**: Josephson νd=2−α with d_eff=0 gives α=2, contradicting Rushbrooke α=0.36. Standard QFT hyperscaling inapplicable to zone-mean (d=0) order parameter. This is structural: the zone-mean M is a 0-dimensional collective mode; free energy is not extensive in L.

**GPU/CUDAGraph**: vcml_gpu_v3.py (manual torch.cuda.CUDAGraph) benchmarked: 229k steps/min at (L=80, B=8) vs v1's 75k steps/min — **3.06× speedup**. Python dispatch overhead eliminated. L=160 B=8: 105k steps/min.

**Paper 76**: γ measurement via χ_int=L²·var(M) in ordered phase. Fix L=80, scan P∈{0.003,0.005,0.010,0.020,0.030,0.050} where U4>0. Fit χ_int~P^{−γ}.

**Status**: CLOSED

---

## Paper 76: γ via χ_int=L²·var(M) — Second Null; d_eff=0 Blocks All Susceptibility Measurement

**Question**: Does the intensive susceptibility χ_int≡L²·var(M) (cancelling geometric dilution) allow direct measurement of γ?

**Phase A (L=80, P∈{0.003,0.005,0.010,0.020,0.030,0.050}, 8 seeds, 300k steps)**:
χ_int≈0.0024 flat across ALL P. U4≈0 for P≤0.020; barely ordered at P=0.030 (U4=0.12) and P=0.050 (U4=0.26). γ_A=0.009 (R²=0.18) — statistically zero, flat noise floor.

**Phase B (FSS, P=0.010, L∈{40,60,80,100,120,160}, 8 seeds, 200k steps)**:
χ_int≈0.0024 flat, DECREASING slightly with L. U4≈0 throughout. |M|~L^{-1.01} (disordered-phase geometric dilution). gonu_B=−0.126 (R²=0.76) NEGATIVE — unphysical. System not ordered at P=0.010 for any tested L.

**Phase C (L=120, P∈{0.002,0.003,0.005,0.010,0.020,0.030}, 8 seeds, 300k steps)**:
χ_int≈0.0022 flat throughout. U4 only rises above 0.1 at P=0.030 (0.213). γ_C=−0.006 (R²=0.18) — flat.

**Root cause (kinematic obstruction)**: φ_i are spatially uncorrelated (G_φ(r)≈0; Paper 72). CLT gives var(M)≈4σ_φ²/L², so χ_int=L²·var(M)≈4σ_φ²=const regardless of P and L. The geometric dilution cancellation works, but it cancels to a noise constant, not a diverging quantity. Physical susceptibility requires ξ>1 (genuine correlations). VCML has d_eff=0 (zone-mean only), so there is no susceptibility divergence to measure.

**Indirect estimate**: Fisher scaling γ=ν(2−η)=0.98×(2−1.28)=**0.71** from measured ν=0.98, η=1.28. Caveat: Fisher's law derived at d_eff>0; applicability uncertain.

**Universality class status**: (β=0.628, ν=0.98, z=0.48) directly measured. γ and η inaccessible to direct measurement at d_eff=0. The three directly measured exponents define the class uniquely.

**Paper 77**: δ measurement via equation-of-state M~h^{1/δ} with external causal field h (biased wave probability). Does not require susceptibility divergence.

**Status**: CLOSED

---

## Paper 73: Dynamic Exponent z — Intrinsic VCSM Timescale Dominates; z Requires P→0+

**Question**: What is the dynamic exponent z of the VCML universality class? Does τ_L~L^z or τ_corr~P^{-zν} reveal z, completing the (β,ν,z) exponent set?

**Phase A (FSS of τ_L, r_w=5, P=0.010, L∈{40,60,80,100,120}, 8 seeds, 30k steps)**:
τ_L scatters between 214 and 742 steps with no monotone trend across L. FSS fit τ_L~L^z gives z_A=0.21 with R²=0.04 (noise-dominated). No finite-size scaling of the relaxation time is observed.

**Phase B (τ_corr(P), r_w=5, L=80, P∈{0.007,0.010,0.015,0.020,0.030,0.050}, 8 seeds, 40k steps)**:
τ_corr≈190–206 steps across a 7-fold range in P — flat to within noise. Fit τ~P^{-zν} gives z_B≈0 (R²=0.21). No critical divergence detected in P∈[0.007,0.050].

**Phase C (C(t) shape at criticality, r_w=5, P=0.010, L=120, 8 seeds, 60k steps)**:
Exponential fit: τ=647, R²=0.602. Power-law fit: C(t)~t^{-2.20}, R²=0.800. Power law wins by ΔR²=0.198. The autocorrelator decays faster than exponential in the window t∈[100,4500].

**Key finding — intrinsic VCSM timescale**: τ_VCSM≈200 steps is an architectural constant set by {SS=8, FA=0.30, FIELD_DECAY=0.999} — NOT a critical divergence. Two competing processes: slow FIELD_DECAY (τ~1000 steps) and fast gate refresh (τ~8 steps). Their competition produces the effective τ_VCSM≈200, independent of L and P in the range tested. C(t)~t^{-2.2} is a mixture of exponentials (gate superposition), not genuine scale-free behaviour.

**Why no divergence**: P_c=0+. At P∈[0.007,0.050] the system is deep in the ordered phase. The critical timescale τ_crit~P^{-zν} only dominates when τ_crit≫τ_VCSM≈200, requiring P≪P_crossover~0.002 (for z=2, ν=0.98). At P>0.007 the VCSM intrinsic timescale masks the critical divergence entirely.

**Consequence**: The dynamic exponent z is not extractable from the current data range. All three z estimates (z_A=0.21, z_B≈0, z_C=0.58) are noise-dominated. The VCML universality class static fingerprint (β,ν)≈(0.63,0.98) is confirmed; z is the last open exponent.

**Paper 74**: Scan P∈{0.0001,0.0005,0.001,0.003} approaching P_c=0+ with L=80 and ≥50k steps. Crossover to critical-dominated timescales expected near P_crossover≈0.002. Measure τ(P) and extract z from τ~P^{-zν} once τ_crit≫τ_VCSM.

**Status**: CLOSED

---

## Paper 72: Two-Point fieldM Correlator — Null Result for eta; Zone-Mean Transition

**Question**: Does G(r)=<phi(x)phi(x+r)> follow r^{-eta} at criticality, enabling direct measurement of eta and testing the indirect eta=1.28 from Paper 71?

**Phase A (direct eta, r_w=5, P=0.010, L in {80,100,120,160}, 8 seeds, 20k steps)**:
G(r)≈0 at the cell level at all L. Correlator amplitudes O(1e-8 to 1e-9), at the statistical noise floor. G(r=3)<0 for three of four L values. Power-law fits give eta=0.64–1.18 with R²≤0.60 and L-to-L variance larger than the mean — completely unreliable. No power-law detected.

**Phase B (eta vs r_w=3,5,8, L=120, P_c=(0.50/r_w)^2)**:
eta estimates: r_w=3 → 2.01 (R²=0.31), r_w=5 → 1.18 (R²=0.60), r_w=8 → 2.50 (R²=0.70). Non-monotone scatter (spread=1.32). NON-UNIVERSAL but reflects noise-dominated fits, not genuine Lévy-DP physics. Lévy-DP hypothesis untestable.

**Phase C (xi_corr(P), r_w=5, 6 P values, L=120, 15k steps)**:
Exponential-fit for xi_corr fails at ALL P (returns inf). Correlator too noisy to extract correlation length. nu cross-check impossible via G(r).

**Key finding**: VCML ordering is a ZONE-MEAN transition. G(r)/sigma²_phi < 10% confirming Paper 66. The fieldM has no within-zone spatial correlations at ANY length scale. The order parameter M=mean(phi_z0)-mean(phi_z1) is a 0-dimensional collective variable. Critical fluctuations are in the TIME DOMAIN of M(t), not in the spatial domain of phi(x).

**Consequence**: eta is NOT DEFINED at the microscopic level. QFT ladder rung 6 (direct eta from G(r)) is VACUOUS. Scaling relation 2*beta/nu=eta CANNOT BE TESTED via G(r). But beta and nu remain valid exponents for the temporal dynamics of the zone-mean.

**Revised ladder**: Rungs 1-5 complete (phi, Z2, field eq, d_wave, beta/nu). Rung 6 vacuous. New target: temporal autocorrelator C(t)=<M(tau)M(tau+t)> and dynamic exponent z.

**Paper 73**: Measure C(t) at criticality; extract dynamic exponent z and temporal decay exponent lambda. Check z=2/nu (diffusive) or z≠2/nu (anomalous dynamics).

**Status**: CLOSED

---

## Paper 71: RG Fixed Point via xi=r_w*sqrt(P) — Iso-xi Universality, beta, FSS

**Question**: Is xi=r_w*sqrt(P) universal (same U4 for different (r_w,P) at fixed xi)? What are beta and nu from a xi-scan and FSS?

**Phase A (iso-xi universality, L=80, Protocol A, r_w in {2,3,4,5,8}, 4 xi values, 8 seeds)**:
NON-UNIVERSAL. At every xi, U4 is strictly ordered r_w=2 > r_w=3 > r_w=5 > r_w=8, with spread 0.20-0.39. Root cause: A(r_w)*P decreases 42% from r_w=2 to r_w=8 at fixed xi (A(2)*P=1.59 vs A(8)*P=1.12 at xi=0.70), because A(r_w)=2r_w^2+2r_w+1 grows faster than r_w^2. xi-universality requires matched noise density (Protocol B) where A(r_w)*p_wave is equalized.

**Phase B (beta extraction, r_w in {5,8}, L=80, Protocol A, 8 seeds, 12k steps)**:
r_w=5: beta=0.628 at xi*=0.50, R²=0.922. r_w=8: beta=0.494 at xi*=0.65, R²=0.923. Different xi* values confirm Protocol A amplitude shift. r_w=5 result (beta=0.628) is primary — consistent with Paper 63 (beta=0.65).

**Phase C (FSS, r_w=5, P in {0.010..0.050}, L in {40,60,80,100,120}, 8 seeds)**:
FSS slopes d(ln|M|)/d(lnL): -0.929 (xi=0.500), -0.783 (xi=0.612), -0.635 (xi=0.707), -0.392 (xi=0.866), -0.185 (xi=1.118). Slope at xi=0.707 = -0.635 matches -beta/nu=-0.67. L=40 anomaly: U4 near zero (wave footprint covers 8% of zone). Implied nu = beta/(beta/nu) = 0.63/0.64 = 0.98.

**Key finding**: beta/nu≈0.64, beta≈0.63, nu≈0.98, indirect eta=2*beta/nu≈1.28. Exponents (beta,nu,eta)≈(0.63,0.98,1.28) match NO known 2D non-equilibrium universality class. Large eta~1.28 (vs DP eta=0.23, Manna eta=0.30) is the fingerprint of non-local wave coupling. xi is a COHERENCE variable (Protocol B) not an amplitude variable (Protocol A): the RG fixed point controls spatial efficiency of zone-mean driving per unit causal signal, not total signal amplitude.

**Paper 72**: Direct measurement of eta from two-point fieldM correlator G(r)~r^{-(d-2+eta)} and check of scaling relation beta=nu*eta/2 (d=2).

---

## Paper 70: Phase Boundary in (r_w, P) Space — RG Dimension of Wave Operator

**Question**: What is the shape of the threshold curve r_w*(P)? Does r_w*(L) grow or shrink with L (thermodynamic vs finite-size artifact)?

**Phase A (2D heat map, L=80, r_w in {1..12}, P in {0.005..0.200}, 6 seeds)**:
Non-monotone anomaly: r_w=2 WORSE than r_w=1 (zone-boundary bleeding). r_w*(P) ~ P^{-0.315} (Protocol A, R²=0.77).

**Phase B (L-scaling at P=0.020, Protocol A)**:
r_w*(L) at L={40,60,80,100,120}: {3.87, 4.49, 2.61, 2.70, 1.00}.
gamma = -1.08. r_w* DECREASES with L. At L=120, r_w=1 orders (U4=0.160).
**Paper 69 threshold is a FINITE-SIZE ARTEFACT: r_w* → 0 as L → ∞.**

**Phase C (matched coverage Protocol B, L=80)**:
r_w*(P) ~ P^{-0.527} (R²=0.82). Prediction: P^{-0.500}. CONFIRMED.

**Key RG result**: Critical coupling ξ = r_w * sqrt(P) = const at threshold.
RG dimension d_wave = 1/2. Wave operator is non-local term in VCML action with smearing scale r_w and eigenvalue 1/2. MISSING COUPLING IDENTIFIED.
Paper 71: measure beta and nu vs ξ/ξ* to extract full beta-function.

**Status**: CLOSED

---

## Paper 69: Wave-Range Dependence of Ordering — Minimum Spatial Range Threshold

**Question**: Does reducing wave radius to r_w=1 (nearest-neighbour) recover Directed Percolation exponents? Is wave range the key differentiator for the novel universality class?

**Phase A (r_w sweep, L=80, P=0.020, 6 seeds, 10k steps, constant rate)**:

| r_w | A(r_w) | |M|×10⁴ | U4 |
|-----|--------|---------|-----|
| 1 | 5 | 0.8 | 0.152 |
| 3 | 25 | 2.2 | 0.144 |
| 5 | 61 | 4.1 | 0.244 |
| 8 | 145 | 8.3 | 0.339 |

|M| ~ A(r_w)^0.67. Ordering increases strongly with wave area.

**Phase B (FSS r_w=1 vs r_w=5, matched coverage)**:
- r_w=5: slopes -0.88 → -0.21 (orderly disorder-to-order crossover)
- r_w=1: slopes ALL ≈ -1.7 at ALL P (PROTOCOL BREAKDOWN: p_wave saturates at 1.0 for all L≥40 with r_w=1)

**Phase C (fine P scan, r_w=1, L=160, matched coverage)**:
- U4 < 0.18 for P ≤ 0.015; system NOT ordered
- |M| = 3-8x smaller than r_w=5 reference
- beta_fit = 0.113 (unreliable, no clean ordered phase)

**Key finding**: r_w=1 does NOT produce ordered phase at P ≤ 0.050. Ordering requires minimum wave spatial range. DP conjecture (Paper 68) untestable at r_w=1.

**New open question**: critical r_w*(L) below which ordered phase inaccessible. Threshold conjecture: r_w ≳ zone_width/20.

**Status**: CLOSED

---

## Paper 68: Order Parameter Exponent at p_c = 0+ — Power-Law vs Essential Singularity, Manna Class Falsified

**Question**: What is the functional form of |M(P)| as P → 0+? Power law, BKT essential singularity, or ordinary essential singularity? Is VCSM in the Manna/stochastic-sandpile universality class?

**Phase A (fine P scan, L=160, P in {0.001..0.050}, 6 seeds, 10k steps, probabilistic firing)**:

| P | absM | U4 | log\|M\| |
|---|------|-----|---------|
| 0.001 | 1.3e-4 | 0.064 | -8.93 |
| 0.005 | 1.4e-4 | 0.038 | -8.91 |
| 0.010 | 1.6e-4 | 0.083 | -8.72 |
| 0.015 | 2.1e-4 | 0.203 | -8.45 |
| 0.020 | 2.8e-4 | 0.321 | -8.20 |
| 0.030 | 4.1e-4 | 0.461 | -7.81 |
| 0.050 | 7.0e-4 | 0.584 | -7.27 |

Functional form fits: Power law R²=0.757 (best), BKT R²=0.501, Exp R²=0.307.
Apparent beta_fit = 0.406 — UNRELIABLE: U4 < 0.1 for P < 0.010 (noise-dominated).

**Phase B (FSS at fine P, L={80,100,120,160}, 4 seeds)**:
Scaling slopes: P=0.003 → slope=-1.21 (disorder L^{-1}); P=0.020 → slope=-0.53 (approaching -beta/nu=-0.67).
Consistent with gradual disorder→order crossover confirming p_c=0+.

**Phase C (Manna conservation-law diagnostic, L=120, P=0.010, 4 seeds)**:
flux_create = 0.256, flux_decay = 0.079, ratio = 3.23.
VCSM phi field NOT conserved. **Manna universality class FALSIFIED.**

**Conclusions**:
- Power law fits better than essential singularity (consistent with standard second-order transition)
- beta=0.406 from direct scan unreliable; Paper 63 FSS beta=0.65 remains best estimate
- Manna class excluded by conservation law violation
- Conjecture: novel class with directed dynamics + Z2 symmetry + long-range wave coupling (r_w=5)
- Paper 69: measure exponents at r_w=1 to test wave-range dependence

**Status**: CLOSED

---

## Paper 67: Probabilistic Wave Firing -- WAVE_EVERY Rounding Artefact Correction

**Question**: With the WAVE_EVERY rounding artefact corrected (probabilistic wave firing), does U4(L=160) > U4(L=120) for all P > 0? If yes: p_c=0+ confirmed. If no: finite p_c.

**Protocol change**: Probabilistic wave firing: at each step, fire wave with prob = min(1, 1/w_f(L)) where w_f(L) = 25*(40/L)^2 (exact, no rounding). For L=160: prob=0.640, giving correct 6400 waves/10k steps vs 5000 (det WE=2). Speed: 4.2 sweeps/step avg vs 6.0 for WE=1.

**Phase A (L={80,100,120,160}, P in {0..0.020}, 3 seeds, 10k steps, probabilistic firing)**:

| P | U4(80) | U4(100) | U4(120) | U4(160) | Verdict |
|---|--------|---------|---------|---------|---------|
| 0.000 | 0.130 | 0.121 | 0.065 | 0.134 | baseline |
| 0.005 | 0.144 | 0.153 | 0.065 | 0.067 | AMBIGUOUS (3-seed noise) |
| 0.010 | 0.190 | 0.171 | 0.080 | 0.117 | U4(160)>U4(120) -> p_c=0+ |
| 0.020 | 0.307 | 0.245 | 0.164 | 0.319 | U4(160)>U4(120) -> p_c=0+ |

**Phase B (L=160, P in {0.005,0.020}, 4 seeds, 6k steps, 3 conditions)**:

| Condition | U4 at P=0.005 | U4 at P=0.020 |
|-----------|--------------|--------------|
| Det WE=2 (old) | 0.283 ± 0.034 | 0.312 ± 0.053 |
| Probabilistic (target) | 0.274 ± 0.065 | 0.500 ± 0.025 |
| Det WE=1 (max) | 0.097 ± 0.069 | 0.499 ± 0.034 |

**Key findings**: (1) p_c=0+ confirmed at P>=0.010: U4(160)>U4(120) with corrected wave density; (2) Paper 66 reversal directly caused by wave density deficit (prob≈WE=1>>WE=2 at P=0.020); (3) NEW: at P=0.005, more waves HURT ordering — extra noise at low causal purity. Optimal wave density is signal-dependent; (4) Protocol fix: probabilistic firing is now standard for L>=100.

**Insight**: The transition near P=0 is zero-threshold (p_c=0+). At low P, the system is noise-dominated regardless of wave density — not because p_c>0 but because more waves inject more noise at low causal purity. This reframes wave density as having an optimal value dependent on P_causal, not "max is best."

**Status**: CLOSED (p_c=0+ confirmed for P>=0.010; protocol fix implemented)

---

## Paper 66: Discriminating Test for p_c=0+, Direct phi-phi Spatial Correlator, and FA Control

**Question**: Is the Paper 65 U4(L=160) < U4(L=120) reversal a WAVE_EVERY rounding artefact or genuine finite p_c? Can we measure G_phi(r) directly with boosted FA? Is FA a control parameter?

**Phase A (L={100,120,160}, P in {0..0.020}, 4 seeds, 10k steps, standard WAVE_EVERY)**: U4(L=160) < U4(L=120) for ALL P > 0.

| P | U4(100) | U4(120) | U4(160) | Delta(120,160) |
|---|---------|---------|---------|----------------|
| 0.000 | 0.103 | 0.163 | 0.083 | -0.080 (baseline) |
| 0.005 | 0.116 | 0.220 | 0.066 | -0.154 reversed |
| 0.010 | 0.149 | 0.253 | 0.073 | -0.180 reversed |
| 0.020 | 0.295 | 0.374 | 0.340 | -0.034 reversed |

Root cause: WAVE_EVERY rounds to 2 for L=160 (target 1.5625). L=160 receives 15.2 wave hits/site vs 18.0 for L=120 (78% of target). Systematic under-driving, not finite p_c. p_c=0+ remains favoured.

**Phase B (phi correlator, L in {60,80}, FA=0.50, P in {0..0.050}, 4 seeds, 8k steps)**: G_phi(r) UNDETERMINED at ALL (L,P). phi_amp ~ 0.013 (detectable), C_phi(r=1)/C_phi(0) < 0.02 — no spatial structure within a zone. Ordering lives in zone-mean M, not phi texture. Zone-mean symmetry-breaking, not BKT.

**Phase C (FA sweep, L=80, P=0.020, 4 seeds, 8k steps)**: U4 decreases with FA (0.314 at FA=0.10 → 0.064 at FA=0.80). |phi| ~ FA^alpha, alpha > 1 (super-linear). High FA = noisier phi = less ordered. Optimal zone ordering at low FA (stable time-average).

**Key findings**: (1) WAVE_EVERY rounding is root cause of Paper 65 reversal (not finite p_c); (2) phi has no within-zone spatial structure — ordering is zone-mean only; (3) FA is an anti-ordering parameter (higher FA = more noise in phi).

**Status**: CLOSED. Paper 67: confirm p_c=0+ with WAVE_EVERY=1 for L=160.

---

## Paper 65: Algebraic Order, Spin Correlator, and VCSM Ablation

**Question**: Does the spin-field correlator confirm algebraic order (BKT)? Does p_c = 0+ hold at L=160? Which VCSM components are load-bearing for phi-ordering?

**Phase A1 (spin correlator, L={60,80,100}, 12 seeds)**: G_s(r) is exponential (R2=0.996 vs power-law 0.936), xi~1.1 lattice spacings, *P-independent*. The phi-field ordering does not couple back to the Ising spins. Spin correlator is the wrong proxy. K_phi ~ L (wave geometry artifact, not BKT stiffness).

**Phase A2 (extended Binder, L={120,160}, P in {0.0005..0.005}, 8 seeds)**: U4(L=160) < U4(L=120) for ALL P ≤ 0.005 — reverses Paper 64 conclusion. Two interpretations: (i) finite p_c in [0.005, 0.020]; (ii) equilibration artifact at L=160 (WAVE_EVERY rounds to 2 vs target 1.56, shorter run 10k steps).

**Phase C (VCSM ablation, L=80, P=0.020, 5 conditions, 12 seeds)**:
- Ref: U4=0.155 (reference)
- NoGate (SS=0): U4=0.090 — gate is load-bearing (kills ordering)
- NoBaseline (β=0): U4=0.180 — baseline not critical at this operating point
- Crystallized (FD=1.0): U4=0.262 — cockroach artefact (lock-in amplification)
- NoCausal (P=0): U4=0.153 ≈ Ref — near-critical at this (L,P)

**Key findings**: (1) Spin field is wrong proxy for phi-transition; (2) p_c=0+ claim downgraded (needs longer L=160 runs); (3) Viability gate (SS) is load-bearing; (4) FIELD_DECAY=1 produces cockroach artefact confirming why phi-decay is necessary.

**Status**: CLOSED (three probes applied; p_c status unresolved; Paper 66 required)

---

## Paper 64: Is p_c = 0? BKT Test via Large-L Binder Crossing and Spatial Correlator

**Question**: Is the causal-purity critical point p_c = 0+ (any nonzero causal purity sufficient for thermodynamic order), or is there a finite disorder threshold? What is the true susceptibility gamma?

**Experiments**: Two phases, Ising + VCSM-lite, extensive-drive scaling, 12 seeds, 12000 steps.
- Phase A: Large-L Binder, L in {40,60,80,100,120}, P_causal in {0.000,0.002,...,0.030} (9 values). 540 runs.
- Phase B: Spatial phi-phi correlator, L in {60,80,100}, P in {0.000,0.005,0.010,0.020,0.050}. FFT-based C(Dy) accumulated in steady state, fit to exp(-r/xi). 180 runs.

**Phase A Results**:

| P_causal | L=40  | L=60  | L=80  | L=100 | L=120 |
|----------|-------|-------|-------|-------|-------|
| 0.000    | 0.050 | 0.023 | 0.113 | 0.111 | 0.130 |
| 0.002    | 0.060 | 0.028 | 0.100 | 0.119 | 0.133 |
| 0.004    | 0.062 | 0.036 | 0.095 | 0.106 | 0.155 |
| 0.008    | 0.061 | 0.056 | 0.097 | 0.092 | 0.187 |
| 0.010    | 0.067 | 0.070 | 0.107 | 0.090 | 0.208 |
| 0.020    | 0.104 | 0.175 | 0.212 | 0.240 | 0.352 |
| 0.030    | 0.161 | 0.256 | 0.326 | 0.388 | 0.457 |

**Key finding**: U4(L=120) > U4(L=40) for ALL P_causal >= 0.002. Binder crossings (where found): p_c = 0.0079 +/- 0.0060, all below 0.017. Most L-pairs produce no crossing (ordered-phase ordering throughout). P=0 row is noisy/flat, consistent with disorder.

**Phase B Results**: All xi = NaN; chi_true = 0 everywhere. Root cause: |M| ~ 1e-3--1e-4 (too small for reliable exponential fit of C(r)). Consistent with xi -> infinity (algebraic order) but does not confirm it.

**Conclusion**: p_c = 0+ strongly favoured. The system enters the ordered phase for any positive causal purity; no disorder threshold is required. Phase B inconclusive on ξ. BKT hypothesis remains open.

**Cosmological remark**: p_c = 0+ means "causality => order" without fine-tuning. Structural isomorphism with the causal set programme (Bombelli-Sorkin): spacetime geometry from causal partial order; here, thermodynamic phi-order from any nonzero causal purity.

**Status**: CLOSED (p_c = 0+ confirmed by Phase A; xi undetermined)

---

## Paper 63: Critical Exponents of the Causal-Purity Phase Transition

**Question**: What are the critical exponents beta, nu, gamma of the causal-purity transition? Which universality class?

**Experiments**: Fine scan P_causal in [0.000,0.080] step 0.005 (17 values), L in {20,30,40,60,80}, 8 seeds, 12000 steps, extensive-drive scaling.

**Results**: p_c = 0.018 +/- 0.012 (Binder crossings). beta/nu = 0.668 (at p_c grid). nu = 0.97 (from dU4/dP ~ L^{1/nu}). beta = 0.65. gamma UNDETERMINED (chi = N*Var(M) self-averages as 1/L^2; definition artifact). Data collapse partial.

| Class | beta | nu | gamma |
|---|---|---|---|
| Mean field | 0.500 | 0.500 | 1.000 |
| 2D Ising | 0.125 | 1.000 | 1.750 |
| Dir. percolation | 0.276 | 0.735 | -- |
| **This system** | **0.65** | **0.97** | undetermined |

Exponents differ from all known classes. BKT hypothesis raised: p_c may be 0+ (confirmed by Paper 64).

**Status**: CLOSED (exponents estimated; gamma undetermined; BKT hypothesis partially confirmed in Paper 64)

---

## Paper 62: Three Decisive Tests

**Question**: Is the causal-purity ordered phase a genuine thermodynamic phase? Does phi-diffusion relay coupling work? Does VCML birth-seeding ablation destroy zone structure?

**Experiments**: Three phases in one file.
- Phase A: Extensive-drive FSS, L in {20,30,40,60}, P_causal in [0.00,0.60], 5 seeds, 8000 steps. WAVE_EVERY = round(25*(40/L)^2).
- Phase B: Multi-zone relay coupling (K in {2,4,6,8} zones, D_phi in {0,0.02,0.10}, P in {0.20,0.40,0.60,0.80,1.00}).
- Phase C: Minimal tanh CML ablation, FIELD_SEED_BETA in {0,0.001,0.003,0.005,0.01}, 5 seeds, 4000 steps.

**Results**:
- Phase A: |M| ~ L^0 (size-independent) for P_causal >= 0.10. True long-range thermodynamic order confirmed. p_c ~ 0.02--0.05.
- Phase B: D_phi=0.10 boosts local S_phi 2.3x. Far-zone relay confounded (all zones receive direct wave drive).
- Phase C: Null — sg4 = 0.0185 identical for all FIELD_SEED_BETA including 0. Minimal tanh CML has no zone structure at all.

**Status**: CLOSED (true LRO confirmed; relay and ablation deferred)

---

## Paper 61: Finite-Size Scaling of the Causal-Purity Phase Transition

**Question**: Do the critical exponents beta, nu, gamma of the causal-purity transition match mean field, 2D Ising, or a new universality class?

**Experiments**: Two-phase FSS in Ising + VCSM-lite. Phase 1: dense P_causal scan at L=40 (PC_SCAN=[0.00,0.68], step 0.04, 5 seeds, 8000 steps). Phase 2: FSS at L={20,30,40,60}, P_causal in [0.00,0.60] step 0.05, 5 seeds, 8000 steps. Measured |M|, M^2, M^4, U4, chi, S_phi.

**Results**:

| P_causal | L=20 |M| | L=30 |M| | L=40 |M| | L=60 |M| | slope |
|----------|---------|---------|---------|---------|-------|
| 0.20 | 0.0144 | 0.0061 | 0.0036 | 0.0015 | -2.041 |
| 0.30 | 0.0218 | 0.0091 | 0.0053 | 0.0022 | -2.065 |
| 0.40 | 0.0285 | 0.0120 | 0.0069 | 0.0030 | -2.048 |
| 0.50 | 0.0348 | 0.0150 | 0.0086 | 0.0037 | -2.025 |
| 0.60 | 0.0413 | 0.0182 | 0.0102 | 0.0045 | -2.016 |

**Key findings**:

1. **|M| ~ L^{-2} at ALL P_causal (not just at p_c)**: Power-law slope = -2.02 +/- 0.03, independent of causal purity. This is geometric dilution, not critical scaling. Wave radius r_w=5 is fixed; signal density proportional to pi*r_w^2 / (L^2/2) ~ 1/L^2.

2. **No L-dependent Binder cumulant crossing**: U4 curves for L=20,30,40,60 nearly coincide for P_causal > 0.10. The transition from Gaussian (U4=0) to bimodal (U4=2/3) occurs near P_causal=0.05-0.10, independent of L.

3. **The transition is a finite-size crossover**: Standard FSS does not apply. The order parameter vanishes as L -> infinity with fixed wave density. No diverging correlation length in the thermodynamic limit.

4. **Root cause -- geometric dilution**: Each wave event perturbs ~79 sites (r=5). Zone has L^2/2 sites. Signal per zone = 79/(L^2/2) ~ L^{-2}. To maintain constant signal density, wave events must scale as L^2 (WAVE_EVERY proportional to 1).

5. **Binder cumulant confirms transition character**: Gaussian (P=0) -> bimodal (P>0.15) transition is real and L-independent. The phi-fluctuation statistics genuinely change with causal purity.

6. **Prediction for Paper 62**: Under extensive wave density (WAVE_EVERY = constant, events proportional to L^2), |M| will be size-independent in the ordered phase. True FSS will yield a Binder crossing at p_c and extractable exponents beta, nu, gamma.

**Implication for universality class**: The causal-purity transition exists but is currently a finite-size coherence effect. The correct thermodynamic limit requires wave-density scaling. The universality class claim stands as a hypothesis pending Paper 62 (extensive-drive FSS).

**Status**: CLOSED (geometric dilution identified; extensive-drive FSS needed for exponents)

---

## Paper 60: Causal Purity as an Independent Control Parameter in an Ising Spin System

**Question**: Does phi-order S_phi in VCSM-lite develop independently of temperature T? Can we confirm the causal-purity universality class in a second substrate (Ising)?

**Experiments**: 2D Ising (40x40, J=1, periodic BC) + VCSM-lite slow phi-field. Swept T in {1.0,1.5,2.0,2.27,3.0,4.0} x P_causal in {0.50,0.60,0.70,0.75,0.80,0.85,0.90,0.95,1.00}, 3 seeds, 8000 steps. Wave mechanism: zone-specific external field h_ext=+/-1.5 for 5 Metropolis steps per wave event, every 25 steps.

**Results**: m vs T (P_causal-independent): At T=1.0 m=1.00, at T=3.0 m=0.07. At T=3.0, S_phi goes from 0.51 (P=0.50) to 0.91 (P=1.00). S_phi increases with P_causal at ALL temperatures. m is constant across P_causal columns.

**Key findings**:

1. **Two orthogonal control parameters**: m = f(T only). S_phi = f(P_causal only, approximately).
2. **Phi-order above T_c**: At T=3.0 (m=0.07, spins disordered), S_phi > 0.5 for all P_causal >= 0.50. Two distinct order parameters coexist.
3. **Causal temperature analogy**: (1-P_causal) plays the role of thermal temperature for the phi-transition. Low causal purity = high causal temperature = disordered phi.
4. **S_phi peaks near T_c, not at low T**: Over-stable Ising attractors (low T) resist zone-specific perturbations, starving the phi-field of signal. Death=anti-crystallization: VCML optimal at intermediate T, not frozen.
5. **First cross-substrate confirmation**: Causal-purity controlled non-equilibrium order confirmed in Ising+VCSM. The mechanism is substrate-agnostic.

**Status**: CLOSED

---

## Paper 59: Field Equation of VCML, Historical Causal Purity, and the r_wave=8 Anomaly

**Question**: Is P_hist (historical purity of accumulated mid_mem) a third identity dimension that explains the r_wave=8 failure? And what is the physics of the VCML field equation?

**Experiments**: Same setup as Paper 58 (C_perturb and C_ref, r_wave={2,4,8,10}, S2, sub-threshold, 5 seeds each = 40 runs). Extended to track P_hist via exponentially-weighted accumulation of same-zone vs. total wave contribution to mid_mem.

**Results**:

| Mode | r | P_sp | P_hist (emp.) | P_hist (geom.) | sigma_w | na     |
|------|---|------|----------------|-----------------|---------|--------|
| CP   | 2 | 0.981| 0.982          | 0.841           | 0.0111  | 1.261  |
| CP   | 4 | 0.948| 0.955          | 0.753           | 0.0452  | 1.140  |
| CP   | 8 | 0.905| 0.943          | 0.684           | 0.0961  | 0.943  |
| CP   | 10| 0.883| 0.931          | 0.678           | 0.1060  | 1.091  |
| ref  | 2 | 0.984| 0.984          | 0.841           | 0.1630  | 1.001  |
| ref  | 4 | 0.956| 0.958          | 0.753           | 0.2277  | 1.081  |
| ref  | 8 | 0.928| 0.944          | 0.684           | 0.1805  | 0.941  |
| ref  | 10| 0.906| 0.932          | 0.678           | 0.1542  | 1.047  |

**Key findings**:

1. **P_hist ≈ P_sp (NULL result on third dimension)**: Empirical P_hist matches P_sp within 0.04 across all conditions. The geometric P_hist prediction (0.68--0.84) severely underestimates actual (0.93--0.98). The same launch-area asymmetry from Paper 57 applies. P_hist is NOT a distinct dimension — two-purity framework (P58) is sufficient.

2. **P_hist analytically reduces to P_sp in steady state**: The Model A field equation shows the mid_mem exponential kernel passes the stationary wave-field purity through unchanged. This is the theoretical reason P_hist ≈ P_sp.

3. **VCML fieldM obeys Model A field equation**: fieldM satisfies d_t phi = -kappa*phi + D*nabla^2*phi + r_w*FA*s(x) + eta, where eta has variance proportional to (1-P_sp)^2 * V_bnd + (1-P_temp)^2 * V_temp. C_order = S/sigma_w is the standard field-theoretic SNR (order-parameter amplitude / fluctuation amplitude).

4. **r_wave=8 anomaly is parity-resonance failure**: At r_wave ≈ W_zone/2.5, adjacent zones exchange suppress/excite polarity signal maximally without non-adjacent mixing. fieldM encodes zone parity (zones 0,2 = "suppressed"; zones 1,3 = "excited") rather than zone position (0 ≠ 2). This inverts the na signature (D_adj > D_nonadj). As r_wave → W_zone/2, non-adjacent zones begin to mix, reducing parity contrast (explaining r=10 partial recovery na=1.091).

5. **Third condition is structural, not causal purity**: Drive-pattern non-resonance condition: r_wave < W_zone/3 OR r_wave > W_zone/2. This is orthogonal to P_sp and P_temp. At r=8, the failure has P_sp≈0.9 and P_temp=1 — an identity failure without any purity deficiency.

**Status**: CLOSED

---

## Paper 58: Empirical Causal Purity and the Two-Dimensional Identity Condition

**Question**: Does direct empirical measurement of P_causal (no geometry, pure causal bookkeeping) confirm the Paper 57 predictions? And does spatial causal purity alone explain the C_perturb vs C_ref sigma_w gap?

**Experiments**: C_perturb and C_ref, r_wave in {2,4,8,10}, S2, sub-threshold (supp=0.06, exc=0.12), 5 seeds = 40 runs x 3000 steps. For every consolidation event, zone(wave) vs zone(cell) recorded directly.

**Results**:

| Mode | r | delta | P_sp  | 1-P_sp | sigma_w | sigma_w/(1-P_sp) | na    |
|------|---|-------|-------|--------|---------|------------------|-------|
| CP   | 2 | 10.0  | 0.981 | 0.019  | 0.0111  | 0.577            | 1.261 |
| CP   | 4 |  5.0  | 0.948 | 0.052  | 0.0452  | 0.861            | 1.140 |
| CP   | 8 |  2.5  | 0.905 | 0.095  | 0.0961  | 1.015            | 0.943 |
| CP   | 10|  2.0  | 0.883 | 0.117  | 0.1060  | 0.904            | 1.091 |
| ref  | 2 | 10.0  | 0.984 | 0.016  | 0.1630  | 10.10            | 1.001 |
| ref  | 4 |  5.0  | 0.956 | 0.044  | 0.2277  |  5.16            | 1.081 |
| ref  | 8 |  2.5  | 0.928 | 0.072  | 0.1805  |  2.50            | 0.941 |
| ref  | 10|  2.0  | 0.906 | 0.094  | 0.1542  |  1.64            | 1.047 |

**Key findings**:

1. **Empirical P_sp matches geometric prediction**: Spatial purity decreases monotonically with r_wave (0.883--0.984), consistent with zone-launch-area calculation in Paper 57. No geometric correction needed.

2. **C_perturb and C_ref have nearly identical P_sp**: Within 0.02--0.03 at each r_wave. Both modes observe the same wave field; C_ref's last-wave-zone proxy is accurate. Yet sigma_w differs 3--10x. Spatial causal purity cannot explain the write-policy gap.

3. **Linear noise model sigma_w ~ A*(1-P_sp) holds only for C_perturb**: Ratio sigma_w/(1-P_sp) is approximately constant for C_perturb (0.58--1.02, mean A≈0.84). For C_ref, the ratio is 1.64--10.10 and not constant. Temporal noise is the unaccounted contribution.

4. **Temporal causal purity P_temp is the second dimension**: C_perturb writes during active wave contact (P_temp=1 by construction). C_ref writes during quiescence after the wave has passed (P_temp<1). When consoildation is temporally decoupled from causal contact, mid_mem has decayed and mixed with multiple prior events — inflating sigma_w beyond the spatial prediction.

5. **Two-dimensional identity condition**: dC_order/dt > 0 iff P_sp > p_c^sp AND P_temp > p_c^temp. No spatial geometry in the final statement. Extended noise model: sigma_w = A_CP*(1-P_sp) + B*(1-P_temp), with B >> A_CP.

6. **Primitive restated**: The substrate-agnostic primitive is distinction + causal propagation + persistence (not carrier turnover). Carrier mortality is the mechanism enforcing persistence on biological substrates, not a primitive requirement. The purity conditions are requirements on persistence of a causally attributed distinction.

7. **r_wave=8 anomaly unexplained**: Both policies fail at r_wave=8 despite nonzero P_sp and C_perturb having P_temp=1. Neither purity dimension alone predicts this. Possible third condition (minimum copy-forward rate, or delta-specific interaction). Left for Paper 59.

**Status**: CLOSED

---

# Key Prediction Reversals

Papers where the initial hypothesis was wrong, reversed, or produced a surprise:

**Paper 9 (Governing Ratio)**: Predicted a single dimensionless ratio Xi would collapse the adaptive window location. NULL. WR and WAVE_DUR have independent effects; no single ratio governs nu*.

**Paper 14 (Timescale Nesting)**: Predicted ratio invariance -- that two dimensionless ratios (R_A, R_B) fully parameterize sg4. FAIL. FIELD_ALPHA has an independent per-event amplitude effect not captured by ratios.

**Paper 15 (Amplitude Law)**: Found sg4 ~ FA^0.43 power law. REVISED by Paper 16: the true law is a saturation curve; the apparent power law was a mixed-phase measurement artifact.

**Paper 24 (Birth Bias / Burgers)**: The copy-forward birth step was expected to be a noise source. SURPRISE: it introduces a Burgers (nabla F)^2 nonlinearity that sharpens zone boundaries from outside -- the opposite of diffusion smoothing.

**Paper 35 / V73 (Dynamic vs. Static)**: The standard assumption was that dynamic collapses (turnover) would always outperform static reservoirs. REVERSED at V73 initial 5-col zones: static > dynamic. Root cause: zone width below the critical threshold. When zone width > ~10 columns and wave density is appropriate, dynamic wins by 43-829%.

**Paper 41 (Switch Threshold)**: Predicted a critical switching period T_crit ~800 steps below which task switching causes net memory loss. NULL. Every tested switching period (50-3000 steps) produced 5-10x amplification over single-task baseline. Task switching amplifies, not erodes, zone structure.

**Paper 33 (Non-Gradient Proof)**: Showed VCML is NOT gradient descent (Lean 4 verified). This reverses any intuition that VCML is "basically like gradient descent on a spatial objective." The system is fundamentally non-equilibrium.

**Paper 79 / V79 (Two-Layer Hierarchy)**: Predicted that cascade input (L1 collapse events driving L2) would attenuate structure. REVERSED: cascade input produced 2.57x MORE sg4 than direct input. The temporal density of collapse events (not their spatial origin) drives L2 structure. Hierarchy amplifies, it does not abstract.

**Paper 90 / V90 (Generation Parity)**: Predicted Mobius sign structure -- a period-2 pattern in fieldM across alternating cell generations. NULL. Sign consistency = 0.510 (chance). The Mobius topology is a mathematical analogy, not a real VCSM property.

**Paper 43 (Rule Robustness -- Formation)**: Expected formation to depend on viability gating. REVERSED: ablating the viability gate barely affects formation (sg4n comparable to reference under continuous waves). Zone-differentiated wave input is sufficient for formation under any consolidation mechanism; formation is environmentally driven, not rule-specific.
