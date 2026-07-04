# Build plan — Tournament Gym + Prize-Zone Data Engine

Goal: the two infra pieces that gate everything (config search, 4P-kit validation, BC, RL).

## Part 1 — Subprocess Tournament Gym (`gym/`)
The orbit_lite bots can't be imported in-process together (every torch bot ships its own
`orbit_lite`/`main`/`submission` — name collisions) and need cwd=their folder. Solve with an
agent-server-per-bot pattern, fronted by an in-process proxy so existing arena.py works unchanged.

- `gym/agent_server.py` — persistent subprocess. Loads `./main.py` (cwd = bot folder), then loops:
  read one JSON line `{obs, config}` from stdin → `structify` → call `agent(obs, config)` → write
  JSON action line to stdout. Crash/timeout safe (emits `[]`).
- `gym/proxy.py` — `ProxyAgent(bot_dir, name)`: a callable `agent(obs, config)` that lazily spawns
  the server, restarts it on `step==0` (fresh per-game state), does line IPC with a per-turn
  timeout, returns `[]` on any failure. Drops into `arena.play_one` as a normal function spec.
- `gym/tournament.py` — discover `agents/opponents/*` (playable) + optionally our `agents/*.py`;
  build proxy specs; run 2P round-robin and/or 4P quads via arena's `play_one` + PlackettLuce
  OpenSkill; print a ranked table; save `strategy/gym_results.json`.

Validation: 1-game 2P match between two bots; then a small round-robin; confirm rankings track
known scores (producer-v2/v44 should top our champion).

## Part 2 — Move extractor (`pipeline/extract_moves.py`)
Recover `(state → action)` labels for behavior cloning. Orbit Wars is fully observable; a launch
appears as a NEW fleet next step. For each replay step t→t+1: any fleet present at t+1 not
traceable to t (by id) with `owner=team` is that team's action `[source_planet, angle, ships]`.
Emit per-team JSONL: `{step, player, obs, actions}`. Test on the 60 local replays in `replays/`.

## Part 3 — Prize-zone roster + targeted episode pull (`pipeline/pull_topbot_episodes.py`)
- Read latest `leaderboards/*.csv` → teams with Score ≥ ~1500 (rank ≤ ~12 + margin) = prize roster.
- Use `kaggle/orbit-wars-episodes-index` to find episode IDs featuring those teams; download just
  those (don't grab full 1.3GB/day dumps). Feed into extract_moves.py → BC dataset of top-bot play.

## Order
Gym first (stated first thing; self-contained; immediate baseline ranking). Then extract_moves
(testable now on local replays). Then prize-zone pull (needs Kaggle download of episode index).
