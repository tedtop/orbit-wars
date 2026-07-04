# Local Gym — Baseline Findings

First head-to-head data from our private tournament gym (`gym/tournament.py`), the unfair
advantage no competitor has. All 14 public bots + our 2 bots, real 500-step games.

## 2P round-robin (240 games, 30/bot)
| # | bot | rating | win% | note |
|---|---|---|---|---|
| 1 | the-producer-v2 | 30.9 | 87% | **best 2P base** |
| 2 | agent-lyonel-1200lb | 28.9 | 87% | orbit_lite |
| 3 | floor-matched-fleets | 28.8 | 87% | orbit_lite |
| 4 | orbit-wars-exp50 | 28.5 | 77% | orbit_lite + terminal |
| 5 | v2-gru | 27.7 | 80% | orbit_lite (GRU dormant) |
| 6 | i-m-stronger | 21.4 | 70% | orbit_lite + anti-leader |
| 7 | v44 (1266 elo) | 18.7 | 70% | orbit_lite, aggressive |
| 8 | heuristic-1110 | 17.0 | 57% | search heuristic |
| 9 | ow-proto | 12.9 | 47% | formula heuristic |
| 10 | rule-base-ml | 7.3 | 40% | heuristic + MLP veto |
| 11 | search-learned-value | 3.2 | 33% | search + GBC value |
| 12 | lb-958-reinforce | -0.4 | 27% | Roman ledger |
| 13 | **markowitz (OURS)** | -4.5 | 17% | our bot |
| 14 | advanced-1608 | -9.0 | 7% | 15-line greedy (theater) |
| 15 | lb-max-1224 | -9.5 | 10% | Roman ledger (!) |
| 16 | **coordinated_strike (OURS)** | -11.3 | 7% | our champion |

## What this proves
1. **orbit_lite family = top 7, decisively.** Our analysis was right: it's the engine to adopt.
   The whole top cluster (18–31) is one codebase; the gap to #8 (heuristics) is real.
2. **Our bots are last.** coordinated_strike #16, markowitz #13. Green-field confirmed — do NOT
   iterate our heuristic; adopt orbit_lite.
3. **producer-v2 is the best 2P base** (not v44). v44's aggression (#7) and exp50's terminal (#4)
   don't help in 2P — those were tuned for Kaggle's 4P-heavy population.
4. **2P ≠ 4P ≠ Kaggle rating — the critical methodology lesson.** Roman's lb-1224 is #15 here
   despite ~1224 on Kaggle, because its entire edge (weakest-enemy / gang-up / crash-exploit) is
   **4P opportunism** a 2P round-robin can't reward. We must judge bots on 4P too (running now).
5. **Search+approx-eval < greedy+exact-eval** (heuristic-1110 #8, search-value #11): invest in the
   evaluation function before search depth.

## Caveats
- 30 games/bot → the top-7 cluster (18–31) is within noise; order there isn't reliable, but the
  top-7 vs rest split and our-bots-last are decisive.
- 2P only. The 4P run (`gym_4p_full.json`) is where Roman's bots and v44 should rise — that's the
  format that decides prize rating. **Read 4P results before picking the v4 base config.**

## Decision
v4 base = **producer-v2 config on the orbit_lite engine** (best, cleanest 2P base), then add the
4P opponent-asymmetry kit (strategy #2) — which the 2P test can't reward but 4P will. Validate
every change in the gym on both formats before submitting.
