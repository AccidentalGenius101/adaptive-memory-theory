# Origins

`substrate_origin.py` is the original script that started the VCSM project.
Written for Pythonista (iOS). No dependencies beyond the standard library.

It asked one question: *does the ability to see change produce qualitatively
different behavior than blind persistence?*

Every major concept in the 64-paper arc traces back to a line in this file:

| Origin concept | What it became |
|---|---|
| `trace` / `delta = value - trace` | Contrastive baseline in VCSM: `baseline_h`, `mid_mem = hid - baseline` |
| `can_see` flag (left/right split) | Zone differentiation; causal purity as control parameter |
| `death_and_renewal()` | Mandatory turnover; death = anti-crystallization principle |
| `adaptations` counter | Causal purity metric P_causal |
| `catastrophe()` | Adversarial persistence tests (Papers 44, 52) |
| Fixed vs. adaptive `mix` ratio | Viability gate (SS calm-streak threshold) |
| `diversity = variance` | sg4 / sigma_w / C_order |
| Left half blind, right half seeing | Two-zone Ising + VCSM-lite (Papers 60--64) |

The project ran for 64 papers across two years before confirming that yes,
the ability to see change (causal purity > 0) produces not just different
but *thermodynamically distinct* behavior — a genuine phase transition at
p_c = 0⁺, meaning any nonzero causal purity is sufficient for order to emerge.

The original question was answered. The answer is: yes, and it's a phase transition.
