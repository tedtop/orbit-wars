# schmeekler_elim — Experiment Notes

## Design
`schmeekler_fmt` + elimination mode:
- When `n_enemy_planets <= 3` AND `n_my_planets > n_enemy_planets` (dominance guard):
  - Add `+8.0` bonus to all enemy planet candidate scores
  - Lower ROI threshold: `1.5 → 1.0`
- Normal schmeekler_fmt behavior otherwise (static_target_bonus=1.5 in 2P, 0 in 4P)

## Bug found and fixed (2026-06-17)
**Original bug:** `n_enemy_planets <= 3` fired immediately at game start (opponent starts with 1-2 planets).
Bot rushed opponent's start position instead of expanding to neutrals. Result: 1-4 (20%) vs floor-matched.

**Fix:** Added dominance guard: `n_my_planets > n_enemy_planets`. Mode only fires when we ALREADY OWN more planets than the enemy.

Confirmed fix: 17-3 (85%) vs floor-matched after fix (matches schmeekler_fmt baseline exactly).

## Arena results (n=150 each, seat-swapped, fixed bot)

| Opponent | schmeekler_elim | schmeekler_fmt baseline (n=20 ref) | Delta |
|---|---|---|---|
| comet_reaper | 85/150 = **56.7%** | ~80% | **-23pp** |
| the-producer-v2 | 86/150 = **57.3%** | ~85% | **-28pp** |
| i-m-stronger | 105/150 = **70.0%** | ~80% | **-10pp** |
| floor-matched | 104/150 = **69.3%** | ~85% | **-16pp** |
| 1266-elo | 108/150 = **72.0%** | ~60% | **+12pp** ← only win |

**OVERALL: 488/750 = 65.1%** vs schmeekler baseline 78/100 = 78% → **-12.9pp DISCARD**

## Direct matchup
schmeekler_elim vs schmeekler_fmt (n=20): **9-11 (45%)** — ELIM LOSES DIRECTLY

## Verdict: ❌ DISCARD

## Key insight: asymmetric pattern
Elim mode hurts vs weak/peer opponents but helps vs the hardest opponent:

| Opponent tier | Delta vs schmeekler_fmt |
|---|---|
| comet_reaper (peer) | -23pp |
| the-producer-v2 (peer) | -28pp |
| i-m-stronger (medium) | -10pp |
| floor-matched (weak) | -16pp |
| **1266-elo (hard)** | **+12pp ← same as stochastic** |

This is the SAME pattern as the stochastic bot (also +20pp vs 1266-elo, -30pp vs comet_reaper).
Root cause: orbit_lite greedy 1-ply is suboptimal in close/late games vs skilled opponents.
Elim mode and stochastic EV both partially compensate for this — but at the cost of peer-game performance.

Root cause of general regression: threshold=3 fires when opponent has 3 planets and we have 4-5.
That's "slightly ahead" not "dominant." Overcommit risk is high when evenly matched.
Would need threshold=1 (truly cornered — 1 planet) to be safe.

## What was preserved
- `agents/schmeekler_elim/main.py` kept in repo for reference (no orbit_lite edits)
- Insight: elim threshold=1 might work; threshold=3 is too early in contested games
- **Promising lead:** a bot that selectively uses hard-game skills (elim/EV) vs strong opponents
  while keeping schmeekler behavior vs medium/weak is the unexplored synthesis
- schmeekler_fmt remains current best gym bot
