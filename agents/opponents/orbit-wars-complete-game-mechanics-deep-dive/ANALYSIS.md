# orbit-wars-complete-game-mechanics-deep-dive  —  REFERENCE (Dingrong Xue)

## What it is
The best **mechanics study** of the game (not an agent). Extracted to `mechanics_reference.py`.
Working, annotated implementations + visualizations of every core rule:
- `fleet_speed(ships)` — speed scales with fleet size (log curve, cap 6).
- `resolve_combat` — exact multi-fleet combat resolution (top minus second survives, then garrison).
- `predict_planet_pos` — orbital position at a future step.
- `iterative_aim` — lead-aim solver for moving targets.
- `planet_score`, `comet_intercept`, `compute_reaction_turns`, fleet-consolidation speed analysis.
- Map-generation statistics and a 2P-vs-4P symmetry / FFA-bystander-advantage study.

## Why it matters
Ground-truth reference to validate our own physics and to sanity-check the orbit_lite constants.
The **FFA bystander-advantage** analysis is directly relevant to 4P coalition strategy (#3): the
player who stays out of early fights tends to win. Use as documentation, not as an opponent.
