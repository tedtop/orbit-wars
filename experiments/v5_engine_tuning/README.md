# Experiment: v5_engine_tuning

**Goal:** climb from comet_reaper's #144 / 1243.8 toward the ~1500 prize zone.

**Status (2026-06-17):** decisive bench done → **comet_reaper already beats the whole public field**
(see `RESULTS.md`). So config-tuning is a low-ceiling side-bet (running overnight via Optuna); the real
lever is **deeper search** (`notes/deeper_search.md`). Submit `comet_reaper` meanwhile.

## Layout
- `RESULTS.md` — bench results + conclusion.
- `INTEL.md` — distilled competitor intel (forum notebooks + Discord).
- `notes/deeper_search.md` — the high-EV next build (multi-ply lookahead).
- `eval/eval_worker.py` — seat-fair game-eval worker (focal vs opponent; reads `CRT_CONFIG`).
- `tuning/tune_config.py` — Optuna study over the config knobs; `tuning/study.db` is the persistent store.
- `intel_kernels/` — the Kaggle notebooks pulled for intel.
- `logs/` — run logs (`gauntlet.log`, `optuna.log`).

## Bots (live in `agents/`, arena requirement)
- `agents/comet_reaper/` — baseline (our submission).
- `agents/comet_reaper_tuned/` — knob-exposed fork: reads `CRT_CONFIG` (JSON dict) to override any config
  knob, and re-adds `ffa_leader_attack_bonus` / `ffa_target_prod_bonus` (4P). `CRT_CONFIG` unset ⇒ identical
  to comet_reaper. Future variants get descriptive `snake_case` names (e.g. `comet_reaper_search`).

## How to run
- Bench a bot vs the field: `arena.py --players comet_reaper,<path> --games 50`.
- Tuning study: `.venv/bin/python experiments/v5_engine_tuning/tuning/tune_config.py 400`.
- Query Optuna: `.venv/bin/python -c "import optuna; s=optuna.load_study(study_name='crt_config_v1',
  storage='sqlite:///experiments/v5_engine_tuning/tuning/study.db'); print(s.best_value, s.best_params)"`.

## On conclusion
Move this folder **and** the `agents/comet_reaper_tuned*` bots → `archive/experiments/v5_engine_tuning/`.
Append milestones to root `TIMELINE.md`.
