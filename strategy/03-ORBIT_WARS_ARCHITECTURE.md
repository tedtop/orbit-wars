# Orbit Wars — Repository Architecture

A clean layout that separates the four concerns: **competing bots**, **evaluation**,
**the data/training pipeline**, and **tracking**. The goal is that six weeks from
now you can still find everything.

```
orbit_wars/
│
├── agents/                      # ACTIVE competition bots ONLY (keep tiny)
│   ├── markowitz.py             #   our submitted bot #1
│   ├── coordinated_strike.py    #   our submitted bot #2
│   └── opponents/               #   downloaded public bots for benchmarking
│       ├── the_producer_v2/     #     (its orbit_lite lib lives here if used)
│       └── midtier_bot.py
│
├── archive/                     # the other 21 bots — importable, out of the way
│   └── agents/
│       ├── the_vulture.py
│       ├── bayesian_wave_function_collapse.py
│       └── ... (all non-active bots)
│
├── core/                        # SHARED bot logic (your physics core)
│   ├── physics.py               #   lead-solution aiming, sun avoidance, fleet speed
│   ├── geometry.py              #   distance, intercept math
│   └── obs_utils.py             #   parse observation → convenient structs
│       # NOTE: this is for BOTS to import. NOT an evaluation engine.
│
├── arena/                       # EVALUATION harness (wraps the real engine)
│   ├── engine.py                #   thin wrapper over make("orbit_wars")
│   ├── arena_2p.py              #   2-player round-robin (your existing logic)
│   ├── arena_4p.py              #   NEW: 4-player FFA, seat rotation, placement scoring
│   ├── wilson.py                #   adaptive Wilson-CI stopping (shared)
│   └── run_arena.py             #   CLI entry: --players a,b,c,d --mode 4p
│
├── pipeline/                    # DATA pipeline (automate ingestion, not submission)
│   ├── pull_episodes.py         #   list new episodes per submission ID
│   ├── download_replays.py      #   fetch replay JSON (prioritize losses vs strong bots)
│   ├── parse_replays.py         #   replay → (state, action, placement) records
│   ├── build_dataset.py         #   assemble training-ready dataset
│   └── run_pipeline.sh          #   cron-able: pull → download → parse → append
│
├── training/                    # MODEL training (Phase 3 — empty until then)
│   ├── encode.py                #   state → feature vector  /  action encode-decode
│   ├── bc_train.py              #   behavioral cloning from replays
│   ├── rl_selfplay.py           #   4-player self-play RL (Jetstream2)
│   └── export_bot.py            #   trained weights → a submittable agents/ bot
│
├── data/                        # DATA (gitignored — large)
│   ├── replays/YYYY-MM-DD/      #   raw replay JSON by date
│   ├── datasets/                #   parsed (state,action) datasets, versioned
│   └── tracking.db              #   SQLite: submissions, episodes, experiments
│
├── scripts/                     # one-off utilities
│   ├── smoke_test.py            #   30-sec local liveness check before submitting
│   ├── submit.py                #   guided submission (prints order reminder)
│   └── leaderboard_snapshot.py  #   pull + log current standings
│
├── notebooks/                   # exploration, analysis, scratch
│   └── arena_analysis.ipynb     #   pairwise matrices, upset detection
│
├── ORBIT_WARS_STRATEGY.md       # the strategy doc (companion to this file)
├── README.md                    # how to run each piece
├── requirements.txt             # kaggle-environments>=1.28.0, torch, etc.
└── .gitignore                   # data/, *.db, replays, __pycache__
```

---

## The four boundaries that keep this clean

**1. `core/` is for bots; `arena/` is for evaluation. Never mix them.**
`core/physics.py` is logic your *bots* import to decide moves. `arena/engine.py`
wraps the *real Kaggle simulator* for scoring. If you ever find bot code importing
from `arena/`, or arena code importing bot strategy, something's wrong.

**2. `agents/` stays tiny.** Only what's actively competing or benchmarking.
Everything else lives in `archive/`. A bloated `agents/` folder is how arenas get
slow and how you lose track of what's actually submitted.

**3. The pipeline writes to `data/`, never to `agents/`.** Ingestion produces
datasets. A *human* runs `training/` on those datasets and `export_bot.py` to
produce a candidate, which a *human* moves into `agents/` and submits. The pipeline
never closes that loop automatically.

**4. `training/` stays empty until Phase 3.** Don't scaffold RL infrastructure
before you have a 4-player baseline and a replay dataset. Premature structure is
clutter too.

---

## Minimal first commit (tonight)

You don't need all of this tonight. Tonight you need:

```
agents/markowitz.py
agents/coordinated_strike.py
scripts/smoke_test.py
```

Get those working and submitted. Build `arena/arena_4p.py` and the `pipeline/`
this week. Leave `training/` for when the data justifies it.

---

## What to migrate from your existing repo

- Your current `arena.py` → split into `arena/arena_2p.py` + `arena/wilson.py`
- Your shared physics core → `core/physics.py`, `core/geometry.py`
- Your 23 bots → 2 into `agents/`, 21 into `archive/agents/`
- Everything else is new and built incrementally
