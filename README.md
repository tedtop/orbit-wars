# Orbit Wars — Local Development

Heuristic agent (Graceful Sloth v1) for the [Kaggle Orbit Wars](https://www.kaggle.com/competitions/orbit-wars) competition, with a local test harness.

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

```bash
.venv/bin/python bench.py --games 30
```

## Files

| File | Purpose |
|------|---------|
| `main.py` | Self-contained agent — submit this to Kaggle |
| `smoke_test.py` | Verify the environment is installed |
| `run_local.py` | Single-game test harness |
| `bench.py` | Win-rate sweep across seeds |
