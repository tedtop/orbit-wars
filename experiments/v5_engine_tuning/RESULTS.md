# v5_engine_tuning — Results

## Decisive gauntlet bench (2026-06-17, 2P seat-swapped, 50 games each)
**comet_reaper vs the public field:**

| opponent | result | comet_reaper win% |
|---|---|---|
| improved-heuristic-agent | 50–0 | **100%** (it's passive — the no-op trap) |
| floor-matched | 34–14 | 71% |
| i-m-stronger | 32–16 | 67% |
| 1266-elo-v44 | 33–17 | 66% |
| the-producer-v2 | 25–23 | 52% (tie — same engine) |

## Conclusion — this reframes the plan
- **comet_reaper already beats the entire *public* field** (ties the Producer, beats the rest). The
  "improved" 1500-target agents are *weaker* than comet_reaper head-to-head.
- ⇒ Porting public techniques and config-tuning are **lower-EV** — we're already top-of-public.
- The gap from **#144/1243 → the ~1500 prize zone is the *private* top teams (#1 ≈ 1793)**, who almost
  certainly use **deeper search** (souldrive: "#1 is a search agent; run the search, don't clone";
  inference is not the bottleneck — ~30× per-turn compute headroom unused).
- **The real lever is DEEPER SEARCH** (multi-ply lookahead). → `notes/deeper_search.md`.

## Running overnight
- **Optuna config study** (`tuning/tune_config.py` → `tuning/study.db`, `logs/optuna.log`): tuning ~19
  config knobs incl. re-added ffa bonuses; objective = win% of comet_reaper_tuned vs comet_reaper over
  seat-rotated 4P+2P. Mainly tests **whether the ffa bonuses help in 4P** (the 2P bench can't). **Low
  ceiling** (we're already top-of-public) but cheap + unattended. Validate any winner vs the full panel
  before submitting.

## Bottom line
Submit `comet_reaper` (still our best). Config tuning is a cheap overnight side-bet; **the prize-zone work
is the deeper-search build** (`notes/deeper_search.md`).
