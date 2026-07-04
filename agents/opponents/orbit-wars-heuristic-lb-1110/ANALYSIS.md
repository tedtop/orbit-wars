# orbit-wars-heuristic-lb-1110  —  ~1110 (vkhydras)

## Core strategy
The most elaborate **from-scratch search bot** (~4900 lines). Unlike the greedy heuristics, it
does real per-turn **action search**: `search_step_action` generates up to 3 candidate launches
per source, evaluates up to ~10 action sets, each via `forward_project` + `forward_score` /
`melis_evaluate` (a 12-turn lookahead state evaluation). Has `predict_defender_at_arrival`,
`effective_garrison_at_arrival`, `_fwd_capture_holds_2p` (verify a capture actually holds after
the enemy responds). Loaded with ablation-named feature flags (F14, SO1, SP1, TI1, AS1…) implying
heavy offline tuning.

## Edges
- Actual lookahead + "does this capture HOLD" verification — more principled than greedy heuristics.
- `melis_evaluate` aggregates score across several future turns (4,8,14,20), not just immediate.
- Anti-second-place logic (`AS1`) and tie-for-win endgame handling (`TI1`).

## Weaknesses (key lesson)
**It searches more than the orbit_lite family yet scores LOWER (~1110 vs ~1248).** Why: its
forward model & scoring are approximate, and the step-action search is shallow (1-ply over a
pruned candidate set). Conclusion: **exact evaluation + good greedy (orbit_lite) beats shallow
search + approximate evaluation.** Depth doesn't help if the leaf eval is wrong.

## Ideas to steal
- `_fwd_capture_holds_2p` (verify captures survive the enemy's response) is a good robustness gate.
- Multi-turn score aggregation (melis) and explicit anti-2nd / tie-for-win endgame rules.
- Mostly a cautionary tale: invest in the **evaluation function** before search depth.
