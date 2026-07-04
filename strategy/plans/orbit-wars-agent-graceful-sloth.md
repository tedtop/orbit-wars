# Orbit Wars — Local Setup & Ambitious v1 Agent

## Context

We're building a bot for the Kaggle **Orbit Wars** competition. Two things prompted this plan:

1. **"Where is the game client to test locally?"** — There isn't a separate client. The
   entire Orbit Wars engine ships *inside* the `kaggle-environments` PyPI package
   (`kaggle_environments/envs/orbit_wars/`). You test locally purely by importing it:
   `make("orbit_wars")` then `env.run([...])`. The `kaggle competitions download orbit-wars`
   data is just docs/starter files, **not** the engine.

2. **A real install blocker on this machine:** the default interpreter is **Python 3.14**,
   but `open_spiel` (a hard dependency of `kaggle-environments`) only publishes macOS arm64
   wheels through **Python 3.13**. Installing on 3.14 fails trying to build from source. The
   project dir `/Users/Ted/src/orbit_wars` is currently empty. `uv` 0.11.21 is installed and
   `~/.kaggle/kaggle.json` already holds credentials.

Outcome: a working local game loop on a correct Python, plus an ambitious heuristic agent
that addresses the starter sniper's known weaknesses (sun avoidance, travel-time/lead,
target valuation, defense, coordination). **Submission is out of scope for now** (local only).

## 1. Environment setup (uv venv on Python 3.11)

Use Python **3.11** to match Kaggle's runtime and guarantee wheels exist (avoids the 3.14
build failure; 3.13 also works as a fallback). `uv` will fetch a standalone 3.11 build.

```
cd /Users/Ted/src/orbit_wars
uv venv --python 3.11 .venv
uv pip install --python .venv "kaggle-environments>=1.28.0"   # resolves to 1.30.1
```

Smoke test (confirms the engine is present and importable):
```
.venv/bin/python -c "from kaggle_environments import make; e=make('orbit_wars'); print(e.name, e.version)"
.venv/bin/python -c "from kaggle_environments.envs.orbit_wars.orbit_wars import Planet, Fleet, CENTER, ROTATION_RADIUS_LIMIT; print(CENTER, ROTATION_RADIUS_LIMIT)"
```

(Heavy deps — jax, open_spiel, transformers — get pulled in; that's expected and only needed
once. The agent itself uses only stdlib `math`.)

## 2. Project layout

- `main.py` — **self-contained** agent (the submission file later). Defines `agent(obs)` as
  the **last** function in the file (kaggle-environments invokes the last-defined callable).
  No local imports, so a future single-file submission stays trivial. Uses only `math` +
  the named-tuple convenience import, guarded for dict-or-object `obs` (per starter pattern).
  May keep **module-level state** across turns (the agent process persists) for rotation-
  direction inference and threat tracking.
- `run_local.py` — local harness (not submitted): runs the agent vs `random`, vs the
  starter sniper, and vs itself (4p); prints rewards/status.
- `bench.py` — runs N seeds vs each baseline and reports win-rate (the real quality signal;
  a single game is noisy).
- `.venv/`, plus a `.gitignore`. Optional `pyproject.toml` to pin the dep.

## 3. Agent design (ambitious heuristic, all in `main.py`)

Assumed config constants (not provided in obs; competition defaults): `CENTER=(50,50)`,
`SUN_RADIUS=10`, `BOARD=100`, `MAX_SPEED=6`, `COMET_SPEED=4`.

Helpers / building blocks:

- **`fleet_speed(ships)`** — `1 + (MAX_SPEED-1)*(log(ships)/log(1000))**1.5`, clamped.
- **`segment_hits_sun(p0, p1, margin)`** — point-to-segment distance from `CENTER`; if
  `< SUN_RADIUS + margin`, the straight path is destroyed. Used to **filter** any
  (source → target) pair whose direct line is blocked (fleets travel straight, so a blocked
  target simply can't be hit from that source this turn — try another source or skip).
- **`predict_planet_pos(planet, t)`** — orbiting planets (orbital_radius + planet_radius <
  `ROTATION_RADIUS_LIMIT`) rotate by `angular_velocity * t` about `CENTER`. **Rotation
  direction** is inferred from module-level previous-observation state (sign of observed
  angular delta); default `+` on turn 1. Static planets return current pos.
- **`predict_comet_pos(...)`** — read directly from `obs.comets` `paths[path_index + t]`
  (exact, no inference needed). Use `comet_planet_ids` to identify comets.
- **`lead_solution(source, target)`** — iterate a few times: guess `t` → future target pos →
  distance → required ships → speed → `t = dist/speed`; returns aim angle + ETA. Handles
  moving planets/comets.

Decision pipeline each turn:

1. **Required-ships model.** Neutral planets do **not** produce (only owned planets do), so a
   neutral needs `garrison + margin`. An enemy target grows in transit: need
   `garrison + production*ETA + margin`. Reinforcing our own planet just adds ships.
2. **Threat detection / defense.** For each owned planet, sum enemy fleets whose heading +
   ETA put them on a collision course; hold a defensive reserve / pull reinforcements so we
   don't lose planets we already own. Never empty a frontline planet below its threat level.
3. **Target valuation.** Score capturable planets by value/cost, e.g.
   `production / (required_ships + k*ETA)`, with an early-game bias toward cheap nearby
   neutrals (ramp production) and a modest, time-decayed bonus for comets (free production
   while they last; worthless once expiring — skip comets about to leave the board).
4. **Assignment.** Greedily match source planets to top-scored reachable targets within each
   source's spendable budget (garrison − reserve). Prefer the nearest/cheapest viable source;
   allow **stacking** multiple sources on one strong target when one source can't afford it;
   avoid redundant over-send once a target's requirement is met. Skip any pair blocked by the
   sun and re-route to an alternate source.
5. Emit `[from_planet_id, angle, num_ships]` moves. Keep well under the 1s `actTimeout`
   (trivial for ~40 planets); `remainingOverageTime` is slack for rare spikes.

### 3a. Engine-verified mechanics (v2 corrections)

The Section 3 constants were *assumed*; reading the shipped engine
(`kaggle_environments/envs/orbit_wars/orbit_wars.py`) confirmed them
(`CENTER=50`, `SUN_RADIUS=10`, `ROTATION_RADIUS_LIMIT=50`, `shipSpeed=6`,
speed `=1+(6-1)*(log(ships)/log(1000))**1.5`) **and** surfaced three mechanics the
v1 heuristics got wrong. Fixing them lifted the win-rate vs the starter sniper from
~67% to ~75–80% (two independent seed sweeps):

1. **Fleets are absorbed by the *first* planet their swept path crosses** (planets are
   tested before bounds/sun, breaking on first hit). v1 never checked for an intervening
   planet, so shots aimed "through" a planet were silently intercepted. Added
   `path_blocked_by_planet(src, aim, …)`; a blocked (source, target) pair is skipped so
   the assignment step re-routes from another source.
2. **Sun check shape.** A fleet dies only if a *per-tick* segment passes within
   `SUN_RADIUS` of centre, and it stops at the target on arrival — so only the segment
   **src → (lead point)** matters. v1 tested a 150-unit ray past the target, falsely
   rejecting valid close shots that merely *pointed* sunward. Now checks the real travel
   segment (`lead_solution` returns the aim point).
3. **Speed scales with the ships actually sent** (1→6×). v1 solved the lead with the full
   spendable budget, so a smaller real send arrived later than predicted (missing orbiting
   targets, under-counting enemy garrison growth). Now a two-pass solve: rough solve sizes
   the requirement, then re-solve with the real `ships_send`.

Also confirmed: combat sums *incoming fleets* per player (top − second survives), then the
survivor attacks/reinforces the stationary garrison — so required ships ≈ garrison + 1, with
owned/enemy garrisons growing `production`/tick in transit (v1's neutral vs enemy split is
correct). **Win = strictly-max total ships** (garrison + in-flight); ties lose, so production
that compounds is the real currency.

## 4. Local testing & verification

```
.venv/bin/python run_local.py          # single games vs random / sniper / self(4p)
.venv/bin/python bench.py --games 30    # win-rate vs each baseline across seeds
```

`run_local.py` mirrors the tutorial:
```python
from kaggle_environments import make
env = make("orbit_wars", configuration={"seed": 42}, debug=True)
env.run(["main.py", "random"])
print([(i, s.reward, s.status) for i, s in enumerate(env.steps[-1])])
```

**Success criteria:**
- Env imports and a full 500-turn game runs to `DONE` with no agent errors (`debug=True`
  surfaces exceptions — watch for any).
- v1 wins clearly vs `random` and beats the **starter sniper** across the `bench.py` seed
  sweep (target ≥70% win-rate vs sniper).
- No fleets lost to the sun from our own launches (assert in a debug pass: none of our
  emitted angles produce a sun-blocked segment), and lead-targeting actually connects with
  orbiting planets/comets (spot-check captures in replays).

## 5. Out of scope (for later)

- Kaggle submission (CLI install, `kaggle competitions submit`, episode/replay review).
  Credentials already exist at `~/.kaggle/kaggle.json`; wire this up once v1 is proven.
- Search/learning-based policy (rollouts, RL). v1 is heuristic; revisit after a baseline.
