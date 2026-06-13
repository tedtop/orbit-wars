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
.venv/bin/python bench.py --bot comet_wraith_v3      # benchmark a specific bot
.venv/bin/python bench.py --bot v3                   # short alias works too
```

## Step 4 — Arena (pit bots against each other)

Every iteration is saved as a self-contained bot in `agents/` (filename = version name).
A game is **2 or 4 players** (engine hard limit), so you fill each slot with a bot. The
engine's `random` and `starter` are always available as baselines.

```bash
# Interactive "startup screen": choose players per slot, then run a series
.venv/bin/python arena.py

# Non-interactive
.venv/bin/python arena.py --list
.venv/bin/python arena.py --players graceful_sloth_v2,starter --games 10   # 2p (auto side-swap)
.venv/bin/python arena.py --players v2,v1,starter,random --games 5         # 4p mix (alias ok)
.venv/bin/python arena.py --round-robin --games 10                         # ladder of all bots
```

## Step 5 — Submit a bot

Pick the bot to submit; it's copied to `main.py` and tagged with its version name:

```bash
.venv/bin/python arena.py --promote graceful_sloth_v2
kaggle competitions submit orbit-wars -f main.py -m 'graceful_sloth_v2'
```

## Adding a new bot version

Copy the current champion as a starting point, then iterate:

```bash
cp agents/graceful_sloth_v2.py agents/graceful_sloth_v3.py   # edit the header name + logic
.venv/bin/python arena.py --players v3,v2 --games 20         # did it actually improve?
```

## Files

| Path | Purpose |
|------|---------|
| `agents/*.py` | Versioned, self-contained bots (source of truth). Champion: `comet_wraith_v3` (beats v2 ~64%, starter ~76%); history: `graceful_sloth_v1` → `v2` → `comet_wraith_v3` |
| `arena.py` | Pit bots against each other; round-robin; promote to `main.py` |
| `main.py` | **Generated** submission artifact (promoted from a bot) — don't edit directly |
| `smoke_test.py` | Verify the environment is installed |
| `run_local.py` | Single-game test harness |
| `bench.py` | Win-rate sweep of any bot (`--bot NAME`, default `main`) vs baselines + self-play |
