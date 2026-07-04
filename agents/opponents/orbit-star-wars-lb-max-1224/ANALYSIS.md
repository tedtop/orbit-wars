# orbit-star-wars-lb-max-1224  —  ~1224 (Roman Tamrazov)

## Core strategy
A **from-scratch stdlib heuristic** (NOT orbit_lite — Roman's other lineage; ~3300 lines). Forward-
simulates each planet's future via an **arrival ledger** (`build_arrival_ledger` →
`simulate_planet_timeline`): collects all in-flight fleets, resolves combat per turn, and projects
owner/ship trajectories. Picks targets by a giant phase-aware scoring config (~200 constants:
opening/early/late/total-war phases, neutral vs hostile value mults, margins that grow with
production & travel distance, snipe/swarm/reinforce/crash-exploit bonuses). Aims with
`search_safe_intercept` + `aim_with_prediction` against moving planets, avoiding the sun.

## Distinctive mechanics
- **Weakest-enemy targeting** (`_compute_weakest_enemy`) + **gang-up** bonus (pile onto a planet
  just after another battle weakens it) — real 4P opportunism the orbit_lite family lacks.
- **Exposed-enemy detection** (`detect_exposed_enemy_planets`): planets the enemy just stripped
  to launch an attack → counter-snipe them.
- Multi-source swarm timing (3-source ETA-tolerant coordinated strikes).
- Elimination drive (big bonus for removing a player in 4P).

## Edges over orbit_lite family
Genuinely models 4P opponent asymmetry (weakest/exposed/gang-up) — the single thing the Producer
family does worst. That's why it ties the family (~1224) despite a less-exact combat model.

## Weaknesses
- Hand-tuned constant soup → brittle, overfit to certain maps; no learned component.
- Combat projection is approximate vs orbit_lite's exact flow-diff; loses close ship-count races.

## Ideas to steal (high value)
The **4P opponent-asymmetry kit**: weakest-enemy focus, gang-up-after-battle, exposed-planet
counter-snipe, elimination bonus. Port these onto the orbit_lite engine (which has the better
combat model but naive 4P) → likely the single biggest top-10 lever. See strategy #3.
