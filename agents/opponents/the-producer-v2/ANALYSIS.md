> Shared engine: see `agents/opponents/ORBIT_LITE_FAMILY.md` for the full orbit_lite deep-dive.
> This file covers only what THIS fork changes and how to beat it.

# the-producer-v2  —  ~1248, #1 by score (Slawek Biel)

## Core strategy
The reference orbit_lite agent and the engine everyone else forked. Forward-simulates the board
over H=18 (2P) / 13 (4P), sends each source's `safe_drain` surplus to the best-scoring target by
exact competitive flow-diff (`Δnet_me − Σ_opp Δnet_opp`), max 6 waves/turn, roi gate 1.5, with
ETA-aware reinforcement risk (β=2.2) and pressure-gradient regroup.

## What it does better than a hand heuristic
- Exact combat-outcome scoring instead of distance/production proxies.
- Provable swept-segment aim against moving planets; never wastes a fleet on an unreachable shot.
- `safe_drain` guarantees it never over-commits a planet into being lost.
- Declines captures the enemy can reinforce mid-flight (most heuristics walk into these).

## Weaknesses / how to beat it
- Pure baseline: no terminal-phase dump, no anti-leader 4P bias — v44/i-m-stronger beat it on
  exactly those. In 4P it sums opponents equally and underplays coalition dynamics.
- Do-nothing opponent model → exploitable by feints and by timing attacks to land the turn
  after it commits its safe_drain elsewhere (it can't re-defend mid-flight).
- Static H=18 → blind to >18-turn rotating snipes.

## Ideas to steal
Adopt the whole engine. It is the strongest, cleanest base to build on.
