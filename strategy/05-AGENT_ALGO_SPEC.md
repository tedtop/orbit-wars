# Orbit Wars — Agent Implementation Spec (shared brief for algorithm bots)

You are implementing ONE strategy as a self-contained bot in `agents/<name>.py`.
Read `agents/coordinated_strike_interceptor.py` first — it is the reference implementation and a
library of correct, reusable geometry/physics helpers. Copy helpers you need.

## Hard requirements (every bot MUST satisfy)
1. File lives at `agents/<name>.py`. The arena auto-discovers `agents/*.py`.
2. The **last callable defined** in the file is `def agent(obs, config):` — the
   engine calls the last callable. It must be crash-safe: wrap your logic in
   try/except and `return []` on any error (see coordinated_strike_interceptor's `agent`).
3. **Pure standard library only** — `math`, `random`, `collections`, etc. NO
   numpy, NO torch, NO external packages. The Kaggle runner is sandboxed and
   has a 1-second/turn budget; a self-contained file is mandatory.
4. For the ML algorithms (GNN/LSTM/DQN/FNN/cascade): there is NO training data
   and no training step. Implement the model's **forward pass with fixed,
   hand-chosen weights** (a deterministic function of the features). Stay
   faithful to the architecture's structure and decision rule; the "learning"
   is replaced by sensible hand-tuned coefficients. Document this clearly in
   the file header.
5. Must run within the 1s/turn budget on a ~40-planet board. Keep per-turn work
   modest (no deep unbounded search; cap minimax depth/branching, etc.).
6. Module-level mutable state keyed by `player` is allowed (the engine reuses
   the module across turns AND across the 2 sides in a self-play game — see how
   coordinated_strike_interceptor keys `_prev_angles`/`_rotation_sign` by player id). History
   trackers (for LSTM/forecaster bots) MUST be keyed by player and reset when
   `step==0` for that player.

## Engine facts (ground truth)
- Board 100x100, origin top-left. Sun at (50,50), radius 10. Fleets crossing the
  sun (path within sunRadius) are destroyed. Out-of-bounds fleets are destroyed.
- Planets: `[id, owner, x, y, radius, ships, production]`. owner -1 = neutral.
  production 1..5; radius = 1 + ln(production); ships garrison.
- Fleets: `[id, owner, x, y, angle, from_planet_id, ships]`. angle in radians,
  0 = +x (right), pi/2 = down (+y). ships constant in flight.
- Fleet speed: `1 + (maxSpeed-1)*(log(ships)/log(1000))**1.5`, capped at maxSpeed
  (config `shipSpeed`, default 6.0). 1 ship => speed 1.
- Orbiting planets: those with `orbital_radius + planet_radius < 50` (i.e.
  distance from center + radius < ROTATION_RADIUS_LIMIT=50) rotate at constant
  `angular_velocity` rad/turn (obs field). Others are static. Use
  `initial_planets` + `angular_velocity` to predict positions. Rotation SIGN is
  not given — infer it by comparing current vs previous angles (coordinated_strike_interceptor does this).
- Comets: ids in `comet_planet_ids`; group data in `comets` (`planet_ids`,
  `paths`, `path_index`). They appear in `planets` too and follow normal rules.
  They leave the board and vanish with their ships; don't over-invest late.
- Combat (per planet, per tick): arriving fleets grouped by owner & summed.
  Largest attacker fights 2nd largest, difference survives. Survivor vs garrison:
  if attacker == owner, adds to garrison; else if survivors > garrison, planet
  flips and garrison = surplus. Ties annihilate. **To capture you must land
  strictly MORE than the garrison-at-arrival.** Garrison at arrival for an owned
  enemy planet = current_ships + production*ETA. Combat resolves per tick, so
  split waves arriving on different ticks let the garrison regrow — coordinate
  arrivals or send one sufficient wave.

## Observation / config access
Obs may be a dict or attr-object; config too. Use the `_get(obj, key, default)`
helper from coordinated_strike_interceptor. Key obs fields: `player`, `planets`, `fleets`,
`initial_planets`, `comets`, `comet_planet_ids`, `angular_velocity`, `step`,
`remainingOverageTime`. Config: `shipSpeed`, `episodeSteps` (500), `sunRadius`,
`boardSize`, `cometSpeed`.

## Action format
Return `[[from_planet_id, angle_radians, num_ships], ...]`. Only own planets.
num_ships integer, ≤ current garrison. `[]` = pass. The fleet spawns just
outside the planet radius along `angle`. Aim with a lead solution at the
target's FUTURE position (see `lead_solution` in coordinated_strike_interceptor).

## Reusable helpers available in coordinated_strike_interceptor.py (copy them)
`_get`, `fleet_speed`, `_pt_seg`, `segment_hits_sun`, `path_blocked_by_planet`,
`predict_planet_pos`, `predict_comet_pos`, `lead_solution`, plus the
rotation-sign inference block inside `_decide`.

## Verification (do this before declaring done)
Run a lightweight check from the repo root (`/Users/Ted/src/orbit_wars`). Do NOT
run the full arena or bench (other bots are being built in parallel — keep CPU
light). Use this snippet, replacing NAME:

```bash
.venv/bin/python - <<'PY'
import logging, os, sys
os.environ["KAGGLE_ENV_LOG_LEVEL"]="ERROR"; logging.disable(logging.INFO)
_dn=open(os.devnull,"w"); so,se=sys.stdout,sys.stderr; sys.stdout=sys.stderr=_dn
f1,f2=os.dup(1),os.dup(2); os.dup2(_dn.fileno(),1); os.dup2(_dn.fileno(),2)
try:
    from kaggle_environments import make
finally:
    os.dup2(f1,1); os.close(f1); os.dup2(f2,2); os.close(f2)
    sys.stdout,sys.stderr=so,se; _dn.close(); logging.disable(logging.NOTSET)
BOT="agents/NAME.py"
wins=0; n=6
for s in range(n):
    me = 0 if s%2==0 else 1
    order = [BOT,"random"] if me==0 else ["random",BOT]
    env=make("orbit_wars",configuration={"seed":s},debug=False)
    env.run(order)
    r=[x.reward for x in env.steps[-1]]
    if r[me]==1 and sum(1 for v in r if v==1)==1:  # strict win, not a draw
        wins+=1
print(f"{BOT}: beat random {wins}/{n} games")
PY
```
A correct expansion-capable bot should beat `random` most of the time (≥5/6).
If it crashes or loses to random, fix it. Report the result.

## Naming & style
Match the house style: module docstring/header explaining the strategy and any
engine-grounded reasoning, descriptive helper names, comments at the density of
coordinated_strike_interceptor. Keep it readable.
