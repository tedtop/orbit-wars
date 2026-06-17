# HANDOFF — Track B: Proper MCTS / deeper search (the prize-tier moonshot)

> Paste this whole file into a fresh Claude Code session opened in `/Users/Ted/src/orbit_wars`.
> You are running **Track B** on git branch **`track-b-mcts-search`**. A separate session runs Track A
> (structural scoring features) on its own branch — stay in your lane, don't touch theirs.

## Who you are / the goal
You're improving **Montana Schmeekler's** Orbit Wars Kaggle bot. Orbit Wars = real-time-strategy: planets
orbit a sun in continuous 2D; you launch ships to capture planets; final score = ships + planets; 2P and 4P
modes; ~500 turns; ~1 s/turn compute budget. We climb from ~#144 / 1243 toward the prize zone (~1500; #1 ≈ 1793).

The top public field is one engine — **orbit_lite** ("The Producer"), a **1-ply greedy flow-diff planner**.
Our clone is **`comet_reaper`** (vendors `orbit_lite/`); champion is **`schmeekler`** (comet_reaper +
static-planet bonus). **The gap to the 1793 prize tier is deeper search** — private teams search multiple plies;
comet_reaper looks **1 ply**. Forum evidence (souldrive: *"run the search, don't clone"*) + we have **~30× unused
per-turn compute** (1 s budget vs ~25 ms used). **Deeper search is THE prize lever** and your mandate.

## What's already been tried (so you don't repeat it)
- ❌ **Simple forward-sim rollout re-rank** (`agents/comet_reaper_search/`, Numba, ~80k rollouts/turn): the
  λ-blend added **no signal** — the aggressive rollout policy *reproduces* the candidate move, so the ranking
  doesn't change. **Lesson: a self-reproducing rollout is useless as a value fn.** Read its `main.py` +
  `search_engine.py` to see the dead end, then go a different way.
- ❌ 2-ply opponent-response bolt-on (`precog`, archived) was parity.
- ✅ The path that works: a **real tree** with the **engine's exact flow scorer as the leaf eval** (NOT a rollout).

## Your task — build a proper search bot `agents/comet_reaper_mcts/`
Fork comet_reaper (or schmeekler, to inherit the static bonus — your call; note it in LOG.md). Build an
**anytime tree search over our own move sequences** with a heuristic leaf value:

- **Node = board state.** Edges = our candidate wave-sets (reuse the engine's `score_candidates`/
  `_greedy_select` to propose the top-K children per node — don't branch over all moves).
- **Forward model:** apply a wave-set, roll the engine projection forward D turns. Reuse
  `garrison_status` projection (`orbit_lite/movement.py`), `sparse_launch_flow_delta`
  (`orbit_lite/garrison_launch.py` — `LaunchSet.owner` is per-launch, so opponent what-ifs are scorable).
- **Opponent reply (cheap):** a 1-ply planner reply (or `safe_drain`-to-nearest) so it's not a solo solitaire.
- **Leaf value = the EXACT flow scorer** (`competitive_score` / `score_candidates` on the leaf board) — NOT a
  rollout (that's the failed approach). This is the key design choice.
- **Anytime + iterative deepening / beam:** keep a beam of B board-lines, expand best-few, deepen until a
  **hard ~800 ms wall-clock cap**, then **fall back to the 1-ply greedy** (Kaggle timeout = instant loss).
- **Env-gate every knob:** beam width B, depth D, top-K, opponent fidelity, time budget — for later tuning.

Speed tips: orbit_lite is PyTorch and slowish; the forum tip is a **numpy/numba forward model** for the rollout
(there's a started Numba engine in `agents/comet_reaper_search/search_engine.py` you can repurpose for the
*projection*, but use the real scorer at leaves). Pre-compile any Numba at import (keep JIT off the turn-0 clock).

## Hard constraints (do not violate)
- **All logic in `agents/<name>/main.py`** (+ helper modules in the same folder). **NEVER edit vendored
  `orbit_lite/`** — shared module; editing corrupts every A/B comparison.
- **Hard per-turn wall-clock cap with anytime fallback to 1-ply.** Verify per-turn wall-clock empirically.
- **Crash-guard** the whole search in try/except → fall back to greedy. A crash/timeout = instant loss.
- No weights / no RL — this is classical search with a heuristic eval.

## How to evaluate (the FIXED yardstick — don't make it easier)
ALWAYS seat-swapped (2P seat-0 effect fakes ~+14%):
```
.venv/bin/python experiments/v5_engine_tuning/autoresearch/evaluate.py comet_reaper_mcts 50 SEARCH_DEPTH=3 SEARCH_BEAM=8
```
Runs the bot vs comet_reaper + the public panel; prints win% per opponent + overall. **A KEEP only if it beats
the champion `schmeekler` outside the noise (n≥150 final, ±~5% CI) AND stays within the time budget.** Also
report per-turn wall-clock — a search that wins but times out on Kaggle is worthless.

## Bookkeeping (every iteration)
- Append to `experiments/v5_engine_tuning/autoresearch/LOG.md` (design → gauntlet result + wall-clock → KEEP/DISCARD).
- Update "Accumulated knowledge" in `experiments/v5_engine_tuning/autoresearch/program.md`.
- New champion → append a milestone to root `TIMELINE.md` (append only).
- **Commit to branch `track-b-mcts-search` only. DO NOT `git push`** — repo is PUBLIC; keep strategy secret
  until the competition ends. Present the commit message for approval first.
- **End every work session with a brief paste-able "state of play" summary** so Ted can forward it for review.

## First moves
1. `git checkout track-b-mcts-search` (confirm).
2. Read `agents/comet_reaper_search/main.py` + `search_engine.py` (the dead end) and
   `experiments/v5_engine_tuning/notes/deeper_search.md`.
3. Scaffold `agents/comet_reaper_mcts/` with the 1-ply greedy as the fallback path FIRST (so you always have a
   safe baseline), then layer the tree + flow-scorer leaf eval on top.
