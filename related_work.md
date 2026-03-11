# Related Work & External Validation

Papers from the broader literature that independently converge on VCSM mechanisms.
These are NOT cited in our papers (independent discovery) but serve as external validation
that the architecture components are real and load-bearing.

---

## GRU-Mem: "When to Memorize and When to Stop" (Feb 2026)

**Paper:** [arXiv:2602.10560](https://arxiv.org/abs/2602.10560)
**Title:** *When to Memorize and When to Stop: Gated Recurrent Memory for Long-Context Reasoning*

### What they did
LLM processes long context chunk-by-chunk (RNN-style). Prior work (MemAgent) had two failure modes:
1. Memory explodes — updates indiscriminately on evidence-free chunks
2. No exit mechanism — keeps processing after sufficient evidence is already collected

Their fix: **GRU-Mem** — two text-controlled gates trained via RL:
- **Update gate** — only write to memory when the current chunk carries signal
- **Exit gate** — stop the loop once sufficient evidence is accumulated

Result: up to 400% inference speed acceleration over vanilla MemAgent.

### VCSM translation

| GRU-Mem concept | VCSM equivalent |
|---|---|
| Update gate open | Causal purity P high → phi-field write allowed |
| Update gate closed | Low-purity chunk → gate blocked, no fieldM update |
| Exit gate open | Quiescence condition met → target settled, consolidation happens |
| Evidence-free chunk | Low causal purity signal (noise floor) |
| Memory explosion | Ferromagnet / crystallization (FIELD_DECAY=1.0, Paper 65 Phase C) |
| No exit mechanism | Chasing moving target → gate never closes → never writes to fieldM |

### Why this validates VCSM

They rediscovered the **viability gate** (component 5 of the 7-component VCSM architecture)
from the engineering side via RL. Our Paper 65 Phase C ablation (`SS=0` → U4 drops to 0.09)
already showed the gate is load-bearing in 2025. They confirm it on language tasks in 2026.

The "exit gate" is the quiescence condition: don't consolidate a self-referential or
non-settled state. This is why `tau_VCSM ~ 200` steps exists (Paper 73) — the gate
waits for the signal to stabilise before writing.

### What they don't have that we do

- No theoretical reason *why* the gate is necessary (forced by 4 conditions)
- No prediction for gate threshold as a function of signal parameters (we have P_c = 0+)
- No universality class: they'll tune RL rewards per task indefinitely
- No phase diagram: they don't know that *any* nonzero causal purity eventually orders

### Status
External validation — independent confirmation of gating necessity.
Filed: 2026-03-11.

---
