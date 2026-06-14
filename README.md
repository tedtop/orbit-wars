# Orbit Wars — Local Development

Heuristic agents for the [Kaggle Orbit Wars](https://www.kaggle.com/competitions/orbit-wars)
competition, with a local test harness. Each iteration is kept as a **versioned bot** in
`agents/`; the arena pits any of them against each other, and the chosen bot is promoted to
`main.py` for submission.

## Setup

The `.venv` is already created (Python 3.11, `kaggle-environments==1.30.1`). If you need to recreate it:

```bash
cd /Users/Ted/src/orbit_wars
uv venv --python 3.11 .venv
uv pip install --python .venv "kaggle-environments>=1.28.0"
```

## Step 1 — Smoke test

Confirm the engine is installed and importable:

```bash
.venv/bin/python smoke_test.py
```

You should see:

```
Smoke test passed -- orbit_wars 1.0.9 loaded successfully.

Next: run single test games
  .venv/bin/python run_local.py
  .venv/bin/python run_local.py --debug   # verbose turn-by-turn output
```

## Step 2 — Single games

Run the agent against random, the starter sniper, and a 4-player game:

```bash
.venv/bin/python run_local.py
```

Add `--debug` for verbose turn-by-turn output:

```bash
.venv/bin/python run_local.py --debug
```

At the end you'll see the next command to run.

## Step 3 — Win-rate benchmark

Run 30 seeds against each baseline (~2–3 minutes). Targets: 100% vs random, ≥70% vs starter sniper.
Pick which bot to benchmark with `--bot` (a name/alias from `agents/`, `main`, or a builtin) —
no need to promote to `main.py` first. The run prints which bot it's testing.

```bash
.venv/bin/python bench.py --games 30                 # defaults to main.py
.venv/bin/python bench.py --bot coordinated_strike_interceptor      # benchmark a specific bot
.venv/bin/python bench.py --bot v3                   # short alias works too
```

## Step 4 — Arena (full tournament)

Every iteration is saved as a self-contained bot in `agents/` (filename = version name).
Running `arena.py` with no arguments plays a **fully automated round-robin**: every bot
fights every other bot (plus the `random` and `starter` baselines), side-swapped to cancel
first-mover bias, with results streaming live — a per-pairing tick stream (`W`/`L`/`·`),
running score, progress bar + ETA, and a final 🏆 leaderboard and head-to-head matrix.
Games run in parallel across all CPU cores.

```bash
# Adaptive tournament (recommended): each pairing plays only until its win-rate is
# statistically settled (95% CI half-width <= --margin), spending games on close
# matchups and stopping early on blowouts.
.venv/bin/python arena.py

# Faster / looser: accept a wider confidence band -> fewer games on lopsided pairs
.venv/bin/python arena.py --margin 0.10

# Fixed count instead of adaptive (exactly N games per pairing)
.venv/bin/python arena.py --games 30      # quick look   (~1.5 min)
.venv/bin/python arena.py --games 100     # thorough      (~3.5 min)

# Throttle parallelism (default: all cores) so you can keep working
.venv/bin/python arena.py -j 6

# One specific matchup instead of the whole field (2 or 4 players; a unique
# filename suffix works as an alias, e.g. "vulture" -> the_vulture)
.venv/bin/python arena.py --players coordinated_strike_interceptor,starter --games 50
.venv/bin/python arena.py --players coordinated_strike_interceptor,path_aware_lead_interceptor,greedy_lead_interceptor,starter --games 20   # 4-player mix

.venv/bin/python arena.py --list          # list discovered bots
```

Useful knobs for the adaptive mode: `--min-games` (games before the CI is allowed to stop a
pairing), `--max-games` (hard cap for dead-even matchups), `--seed-offset` (shift which board
layouts are used). Add `--no-color` if you're piping output to a file.

> **Tuning note:** games run up to `episodeSteps=500` turns but usually end far sooner —
> median ~200 turns, ~390 at the 90th percentile, occasionally reaching the 500 cap. Tune bots
> to win by ~200 turns, stay coherent through ~400, and be **ahead on the score tiebreak by
> turn 500** rather than mid-plan. Planning deeper than ~300 turns is mostly wasted.

## Step 5 — Submit a bot

Pick the bot to submit; it's copied to `main.py` and tagged with its version name:

```bash
.venv/bin/python arena.py --promote path_aware_lead_interceptor
kaggle competitions submit orbit-wars -f main.py -m 'path_aware_lead_interceptor'
```

## Adding a new bot version

Copy the current champion as a starting point, then iterate:

```bash
cp agents/coordinated_strike_interceptor.py agents/coordinated_strike_interceptor_v4.py   # edit the header name + logic
.venv/bin/python arena.py --players coordinated_strike_interceptor_v4,coordinated_strike_interceptor --games 50   # did it beat the champion?
.venv/bin/python arena.py                                     # then re-run the full ladder
```

New bots are picked up automatically — just dropping a `*.py` file in `agents/` adds it to the
tournament field next time you run `arena.py`.

## Files

| Path | Purpose |
|------|---------|
| `agents/*.py` | Versioned, self-contained bots (source of truth). Champion: `coordinated_strike_interceptor` (beats v2 ~64%, starter ~76%); history: `greedy_lead_interceptor` → `path_aware_lead_interceptor` → `coordinated_strike_interceptor` |
| `arena.py` | Pit bots against each other; round-robin; promote to `main.py` |
| `main.py` | **Generated** submission artifact (promoted from a bot) — don't edit directly |
| `smoke_test.py` | Verify the environment is installed |
| `run_local.py` | Single-game test harness |
| `bench.py` | Win-rate sweep of any bot (`--bot NAME`, default `main`) vs baselines + self-play |
