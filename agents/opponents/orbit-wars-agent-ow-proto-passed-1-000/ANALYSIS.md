# orbit-wars-agent-ow-proto-passed-1-000  —  ~1000 (Djenk Ivanov)

## Core strategy
A clean, readable **formula heuristic** (~800 lines) — the ancestor of the search-value bot
(shares `get_custom_score`, `get_planets_under_attack`, reinforcement machinery). Per target,
`get_custom_score(m,t)` = `(100−dist) + 15·production + 10·enemy_bonus − 0.7·total_ships − 2·eta`.
Detects threats by **ray/trajectory collision** (`get_planets_under_attack` projects each enemy
fleet 60 turns and checks planet intersection), then `get_reinforcement_plans` to defend.
Lead-aims moving targets via `calculate_req_ships_moving` / `find_angle_to_moving_planet`.

## Edges
- Simple, robust, fast. Threat detection via actual fleet-trajectory collision is clean.
- Imports `kaggle_environments.envs.orbit_wars` directly to reuse engine constants.

## Weaknesses
- Linear formula scoring — no forward sim of combat, no phase awareness, no 4P opponent modeling.
- Caps at ~1000; outclassed by both the orbit_lite family and the ledger heuristics.

## Ideas to steal
The trajectory-collision threat detector is a tidy, cheap primitive. Good baseline/reference for
a from-scratch bot, and a readable map of the core mechanics. Low priority as an opponent.
