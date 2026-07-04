# Orbit Wars — the Montana Schmeekler campaign

A Kaggle AI competition (June 13–23, 2026) building bots that fight a
real-time orbital-mechanics strategy war. This repo is the full research
record — source for every bot, every experiment log, the autonomous AI lab
sessions that ran the research, and the interactive write-up built from it all.

### 🚀 Start here: **[kaggle-orbit-wars.vercel.app](https://kaggle-orbit-wars.vercel.app)**

| Final rank | Peak | Final submission |
|:---:|:---:|:---:|
| **#415** of 4,752 teams · top 9% | **#144** @ 1,243.8 Elo | `comet_reaper` · ~1,235 Elo |

---

## The Story

Start with [`TIMELINE.md`](TIMELINE.md) — every strategy pivot, dead end, and
breakthrough in chronological order, with the dead ends left visible. The
short version:

| When | What happened |
|-----|---------------|
| Day 1 | Built **23 bots from scratch, one per scientific field** (portfolio theory, epidemiology, control theory, ant colonies, …). The finance bot won our internal 80,000-game arena. Live ladder: mid-pack. |
| Days 2–3 | Studied the top of the leaderboard. The best bots all shared one engine: `orbit_lite`. Cloned it as **`comet_reaper`** → rank ~150. |
| Days 4–5 | Autonomous hypothesis-testing lab: Claude Code orchestrator + auditor agents running a fixed-gauntlet ratchet. **19 experiments, 2 KEEPs.** One genuinely novel win: `schmeekler` (grab the static planets first) beat the engine 72% locally. |
| Days 6–9 | Full RL pivot: 9-instance Jetstream2 fleet, then a from-scratch pure-JAX game engine on A100s. Every campaign plateaued at **23.3% vs a greedy baseline and 0% vs our own engine.** |
| Day 10 | Deadline. Final answer: `comet_reaper`, the day-three bot, unmodified — nothing ever beat it. |
| After | Post-deadline autopsies (v9/v10) and the post-mortem: the winners trained RL over a strong engine's *strategy knobs*, not raw ship commands. |

---

## Came here from the website?

| You watched… | The code is… |
|---|---|
| the 23-scientist arena | [`archive/agents/`](archive/agents/) + [`arena.py`](arena.py) |
| the engine discovery | [`agents/comet_reaper/`](agents/comet_reaper/) (attribution in its README) |
| schmeekler's 72% | [`agents/schmeekler/`](agents/schmeekler/) and siblings |
| the experiment ledger | [`experiments/`](experiments/) + [`TIMELINE.md`](TIMELINE.md) |
| the RL moonshot | [`agents/rl_ppo_cpu/`](agents/rl_ppo_cpu/) (v9) · [`agents/rl_ppo_jax/`](agents/rl_ppo_jax/) (v10 JAX engine) |
| the replay theater & charts | [`website_fable/`](website_fable/) (the site itself) |
| the screenshots | [`strategy/screenshots_highlights/`](strategy/screenshots_highlights/README.md) — 📸 captioned tour |

---

## Key Documents

| Document | What it is |
|---|---|
| [`TIMELINE.md`](TIMELINE.md) | Chronological research log — the primary narrative |
| [`strategy/`](strategy/) | 16 research write-ups: strategy, architecture, gym findings, RL plans |
| [`AUDITOR_LOG.md`](AUDITOR_LOG.md) | Raw session log from the independent auditor Claude agent |
| [`ORCHESTRATOR_STATE.md`](ORCHESTRATOR_STATE.md) | Session state records from the autonomous orchestrator agent |
| [`strategy/screenshots_highlights/`](strategy/screenshots_highlights/README.md) | The campaign in 30 captioned screenshots |

---

## Folder Map

```
agents/                Bot source
  archive/agents/      The 23 day-one "scientist" bots
  comet_reaper/        THE champion — orbit_lite engine clone (see its README for attribution)
  schmeekler*/         The static-planet-first family (our one novel win)
  comet_reaper_*/      Search / tuning / MCTS forks (all DISCARD, all preserved)
  opponents/           19 public Kaggle bots converted to runnable agents (attribution inside)
  rl_ppo_cpu/          v9 entity-transformer PPO (CPU fleet) + runs_archive/ metrics
  rl_ppo_jax/          v10 pure-JAX game engine + PPO + fleet scripts

arena.py               Round-robin evaluation harness (OpenSkill-rated, seat-balanced)
bench.py               Quick win-rate benchmark vs baselines
dashboard/             Streamlit ops dashboard used during the competition
experiments/           Per-experiment working directories (v5 tuning → v8 BC)
gym/                   Local match runner / tournament harness
pipeline/              Kaggle leaderboard + replay polling pipeline (every 15 min)
rl/                    Phase-4 behaviour cloning + self-play scripts (DEAD END, preserved)

strategy/              Research notes, write-ups, and the visual record
  screenshots_highlights/  30 curated screenshots with captions — start here
  screenshots/             Full 140-shot raw archive

website_fable/         ✨ The presentation site → kaggle-orbit-wars.vercel.app
website_opus/          First draft of the site (superseded, kept for the record)
```

Not in the repo (runtime data, re-pullable via `pipeline/`): `episodes/`
(~11 GB of raw episode JSONs) and `replays/`. The 713 leaderboard CSV
snapshots (polled every 15 minutes for nine days) **are** preserved as
[`leaderboards_snapshots.tar.xz`](leaderboards_snapshots.tar.xz) (15 MB) and
distilled into `website_fable/src/data/race.json`.

---

## Agents Lineage

```
[Day 1] 23 scientist bots → archive/agents/
    ↓  reverse-engineered the public meta's orbit_lite engine
comet_reaper  ← FINAL SUBMISSION (~1,235 Elo)
    ├── comet_reaper_mcts      (MCTS add-on — DISCARD)
    ├── comet_reaper_search    (forward-sim rollouts — DISCARD)
    ├── comet_reaper_tuned     (Optuna 37-trial sweep — base config already optimal)
    └── schmeekler             (static_target_bonus — 72% vs comet_reaper in the gym)
          ├── schmeekler_fmt       (format-aware — KEEP)
          ├── schmeekler_comet     (comet targeting — DISCARD)
          ├── schmeekler_elim      (elimination mode — DISCARD)
          └── schmeekler_potential (potential fields — DISCARD)

[Parallel] RL
    rl/           (Phase 4 BC + self-play — lost 0–16 to comet_reaper)
    rl_ppo_cpu    (v6–v9: up to 32 seeds × 69 h — 0% vs comet_reaper)
    rl_ppo_jax    (v10: pure-JAX engine, 2× A100, 107M+ steps — 0% vs comet_reaper)
```

---

## Run it yourself

Python 3.11 + `kaggle-environments`:

```bash
uv venv --python 3.11 .venv
uv pip install "kaggle-environments>=1.28.0"

.venv/bin/python smoke_test.py                # env sanity check
.venv/bin/python run_local.py --debug         # watch one game, turn by turn
.venv/bin/python bench.py --games 30          # quick win-rate benchmark
.venv/bin/python arena.py --list              # all discovered bots
.venv/bin/python arena.py --players schmeekler,comet_reaper --games 50
```

The website:

```bash
cd website_fable
npm install
npm run dev          # http://localhost:3000
```

---

## Attribution

- **`orbit_lite`** — the planning engine inside `comet_reaper` is our port of
  the engine behind The Producer, published on Kaggle by Slawek Biel
  (`producer-orbit-wars-utils` / "The Producer V2"). It defined the public
  meta; details in [`agents/comet_reaper/README.md`](agents/comet_reaper/README.md).
- **`agents/opponents/`** — 19 public notebooks by their respective authors,
  converted to runnable agents for local evaluation; original `.ipynb` files
  preserved in each folder.
- Compute: [Jetstream2](https://jetstream-cloud.org/) (NSF ACCESS) for the CPU
  fleet and A100s.
- The autonomous lab was run with [Claude Code](https://claude.com/claude-code)
  agents (orchestrator / auditor / worker tracks) — their raw logs are part of
  this record.
