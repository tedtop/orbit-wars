# Orbit Wars — Autoresearch Program (the self-improving research prompt)

_This is the versioned "prompt" the agent (Claude Code) reads + **updates each iteration**. Commit it after
every experiment so the recursive self-improvement is tracked in git history. Pattern: Karpathy autoresearch
— propose → evaluate on a FIXED yardstick → keep-if-better → update this file._

## Goal
Climb comet_reaper (#144 / 1243) toward the ~1500 prize zone (#1 ≈ 1793).

## Fixed evaluator (the yardstick — do NOT make it easier mid-search)
`autoresearch/evaluate.py <bot> [STATIC_BONUS=..] [..env knobs]` → **seat-rotated** win% vs comet_reaper +
the public panel (the-producer-v2, i-m-stronger, floor-matched, 1266-elo), 2P + 4P, **n ≥ 150** (±~5% CI).
A candidate is a "keep" only if it beats the current champion **outside the CI** AND stays within the
**1 s/turn** budget. Always seat-swap (2P has a seat-0 effect that fakes ~+14%).

## Hard constraints
- Bots live in `agents/`; all logic in `main.py`; **never edit vendored `orbit_lite`** (shared-module
  collision would corrupt A/B). Each bot crash-guarded (a Kaggle error/timeout = instant loss).
- Validate on the gauntlet BEFORE submitting. comet_reaper/schmeekler are config-driven (no weights/RL).

## Accumulated knowledge (what works / what doesn't) — UPDATE EACH ITERATION
- ✅ **Structural scoring features WIN.** `schmeekler` (bonus for capturing static/non-rotating planets
  first) beats comet_reaper **72% 2P / best-in-pod 4P** at bonus 1.0–1.5 (≥2.0 over-commits). **Current
  champion.** → *the productive direction is more schmeekler-style features.*
- ❌ **Simple search is redundant with the 1-ply scorer.** Forward-sim rollout re-rank (`comet_reaper_search`)
  added no signal — the aggressive rollout policy reproduces the candidate moves. 2-ply opponent-response
  (`precog`) was parity too. Real gains need a proper MCTS *tree + value function* (big build).
- ⚠️ **Re-tuning the STOCK engine config is dead** (Optuna over ~19 base knobs, best 0.34 — already a tight
  optimum). But **sweeping the knob a NEW feature introduces is alive and IS the job** — schmeekler's
  `static_bonus` sweep (→1.0–1.5) is the model. Every new feature ships an env knob; sweep it (Optuna or grid).
- ❌ **BC / RL / cloning** is a dead end (0–16 vs the engine; forum-confirmed). Aggression must be
  *calibrated* (naive aggression over-extends), which is what search/structure provides — not knob-lowering.

## Ranked hypothesis queue (the self-improving part — re-rank by EV each iteration)
1. **More structural scoring features** (combine with schmeekler):
   - Potential-field target value using **future** planet positions (favor planets rotating toward us).
   - Enemy-fleet **interdiction** (race to contested neutrals).
   - **Phase-aware** dynamic ship sizing (early 1.2× → finishing 3.0×) + speed-bracket sizing.
   - Tune schmeekler's bonus jointly with aggression knobs (a NEW Optuna study, now that structure helps).
2. **Proper MCTS** — tree over our move sequences + a heuristic **value function** (use the engine's flow
   scorer as the leaf eval, NOT a self-reproducing rollout). The path to the 1793 tier; big, uncertain.
3. **4P meta-play** (coalition / weakest-enemy / don't-be-leader), validated seat-rotated.

## Ratchet rule
Each iteration: pick the top untried hypothesis → build `agents/<name>/` (main.py only) → `evaluate.py` →
if it beats the champion outside the CI, it's the new champion (consider submitting) → **append to LOG.md
and update "Accumulated knowledge" + re-rank this queue.**
