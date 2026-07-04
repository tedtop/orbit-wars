# Opponent Bots — downloaded & standardized

> **Attribution:** everything in this folder is **other authors' public work**,
> shared as Kaggle notebooks during the competition. We converted each notebook
> into a runnable Python agent so we could battle against them in our local
> arena — that's the entire modification. Folder names are the original Kaggle
> kernel slugs, and each folder keeps the author's original `.ipynb` alongside
> the extracted code so provenance is never lost. Enormous thanks to these
> authors — the public field they created is what made local evaluation
> (and honestly, most of our progress) possible. See
> `strategy/15-Top Open Source Bots.md` for the notebook list with scores.

19 top public Kaggle notebooks, extracted from `.ipynb` into runnable agents. Each folder has
the author's original `<slug>.ipynb` plus extracted code. Re-run extraction any time with:

```bash
.venv/bin/python agents/opponents/extract_bots.py
```

## Playable bots (14) — all pass the smoke test
Every playable folder exposes `agent(obs, config)` via `main.py`. Verified with:

```bash
.venv/bin/python agents/opponents/_smoke_test.py        # PASS/FAIL table, all 14 green
```

| Bot | Family | Entry | Notes |
|---|---|---|---|
| the-producer-v2 | orbit_lite+torch | main.py | #1 by score (~1248). Shared `orbit_lite` engine. |
| orbit-wars-i-m-stronger | orbit_lite+torch | main.py | #2 (~1226). Opening hold (no actions until ~step 40). |
| orbit-wars-exp50 | orbit_lite+torch | main.py | Producer variant. |
| floor-matched-fleets-target-veto-evacuation | orbit_lite+torch | main.py | Producer variant w/ evac. |
| agent-lyonel-1200lb | orbit_lite+torch | main.py | Producer variant (~1200). |
| v2-gru | orbit_lite+torch | main.py | GRU scaffold is **disabled** in public ver (falls back to planner). |
| 1266-elo-the-v44-agent | orbit_lite (own copy) | main.py | Ships its own `orbit_lite/`. |
| orbit-star-wars-lb-max-1224 | stdlib heuristic | main.py→submission.py | Roman's tuned config (~1224). |
| lb-958-1-orbit-wars-2026-reinforce | stdlib heuristic | main.py→submission.py | Earlier Roman config (~958). |
| orbit-wars-heuristic-lb-1110 | stdlib heuristic | main.py→submission.py | vkhydras, fwd-sim search (~1110). |
| lb-highest-1000-search-learned-value-function | numpy heuristic | main.py | Search + learned value fn (~1000). |
| orbit-wars-agent-ow-proto-passed-1-000 | stdlib | main.py | Imports kaggle_environments.orbit_wars. |
| orbit-wars-rule-base-ml-shot-validator-hybrid | stdlib+MLP | main.py→submission.py | `weights.npz` decoded from base64. |
| orbit-wars-advanced-agent-target-1608-6 | stdlib | main.py→submission.py | **Real agent is ~15-line greedy**; MCTS/dashboard is theater. |

## Reference / non-playable
- `producer-orbit-wars-utils/orbit_lite/` — the Producer's torch forward-sim **engine** (the
  "orbit lite" library). Copied into each torch bot's folder.
- `orbit-wars-reinforcement-learning-tutorial/` — PPO self-play trainer (`src/`). Gold for our RL plan.
- `orbit-wars-complete-game-mechanics-deep-dive/mechanics_reference.py` — best mechanics study.
- `orbit-wars-physics-helper-module/physics_helper.py` — clean stdlib physics helpers.
- `reverse-engineering-agents-replay-analysis-tooli/replay_analysis.py` — opponent-modeling from replays.
- `simplified-orbit-wars-agent/NOTE.md` — not reconstructable (private Kaggle Model artifact).

## How to run one in a game
Module names collide across folders (every torch bot has its own `orbit_lite`/`main`/`submission`),
so **run each bot in its own subprocess with cwd = its folder** — the verified pattern:

```bash
cd agents/opponents/the-producer-v2
../../../.venv/bin/python ../_smoke_one.py 2 80     # num_players steps
```

For a real local tournament, build a subprocess-per-game runner (each bot launched with
cwd=its folder, 1s/turn timeout). Direct in-process `arena.py` paths work only for the
self-contained stdlib bots, not the `orbit_lite` ones.
