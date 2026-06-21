# Deeper search — the prize-zone lever (design)

## Why
comet_reaper (orbit_lite) is a **1-ply greedy**: each turn it scores candidate waves against a *do-nothing*
forward projection and fires the best by flow-diff. It beats the whole public field but caps ~1243. The
#1 (~1793) is a **search** agent. **We have ~30× per-turn compute headroom** (1 s budget vs ~25 ms used) —
currently wasted. Deeper search is the most likely path to the prize zone, and souldrive's evidence says
imitation/RL can't get there but *search* can.

## What "deeper" means here (cheapest → richest)
1. **2-ply with a 1-ply opponent reply (recommended v1).** For the top-K candidate wave-sets this turn:
   apply the wave → roll the engine projection forward → let a *cheap opponent model* (the same planner at
   1-ply, or the `safe_drain`-at-nearest proxy) pick the rivals' reply → score the resulting board with the
   exact flow scorer → pick the wave-set with the best post-reply value. This directly fixes the
   do-nothing blind spot AND adds one ply of our own lookahead. (This is "precog + helmsman, done right" —
   and unlike those bolt-ons, it's *real* search, not a heuristic nudge.)
2. **Beam / iterative-deepening over our own future turns.** Keep a beam of B board-lines, expand each by
   its best few waves, roll forward D turns, score the leaf; deepen D until the time budget is spent
   (anytime). Reuse the engine's `garrison_status` projection + `score_candidates`.
3. **Light MCTS** over wave-sets if 1–2 plateau.

## Build plan
- New bot `agents/comet_reaper_search/` (fork; all logic in `main.py`; never edit vendored `orbit_lite`).
- Reuse: `garrison_status` projection, `sparse_launch_flow_delta` (apply hypothetical launches — incl.
  opponent-owned, `LaunchSet.owner` is per-launch), `score_candidates`, `_greedy_select`.
- **Hard per-turn wall-clock cap** (e.g. 800 ms) with **anytime fallback to the 1-ply greedy** — a Kaggle
  timeout is an instant loss. Validate per-turn time in the gym.
- Knobs (env-exposed, Optuna-tunable): beam width B, depth D, top-K candidates, opponent-model fidelity,
  time budget.

## Validate
Seat-rotated gauntlet (n≥200) vs comet_reaper **and** the public panel; submit only if it beats
comet_reaper outside the CI **and** stays within the per-turn time budget. This is also the natural place
for the **autoresearch ratchet** (propose search-design variants → bench → keep-if-better) and **Jetstream2
150 cores** (search evals are heavier → more cores pay off; pair with a numba/numpy speedup of orbit_lite).

## Honest EV
Highest-ceiling lever we have, and the one the evidence points to — but the biggest build, and not
guaranteed. Build the scaffold supervised (search bugs are subtle), then let autoresearch + cloud tune it.

## Numba forward-model benchmark (2026-06-17) — GREEN LIGHT
Flat-array Numba simplified sim (`eval/numba_forward_bench.py`, economics only — drops orbital geometry):
- 20-turn rollout: **1.4 µs** → ~709K/s → **~567K rollouts in an 800ms turn budget**.
- 60-turn rollout: 2.4 µs → ~410K/s → ~328K/turn. Compile = one-time 2.7s (pre-trigger at import).
⇒ **MCTS is viable** — even a 10× heavier rollout policy leaves ~50K rollouts/turn. Build `comet_reaper_search`
on this. Open task: fidelity-check that the simplified model's state rankings track the real engine.

## Optuna (config tuning) — confirming low-EV
27 trials, best win% vs comet_reaper = **0.33** (< 0.5). The config is a tight optimum; tuning isn't beating
it (as predicted). Keep running as the cheap side-bet, but the real lever is the search build above.
