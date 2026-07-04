# orbit-wars-rule-base-ml-shot-validator-hybrid  —  (konbu17)

## Core strategy
Roman's lb-1224 **ledger heuristic** (same forward-sim core) + an **ML shot validator**. The
heuristic proposes candidate shots; `_encode_shot_np` featurizes each (geometry + board context)
and a small **MLP** (weights shipped as base64 → decoded to `weights.npz`, which loads locally)
scores/vetoes them. `_find_target_ray` checks what a shot actually hits. "Hybrid" = rules
generate, ML filters.

## Edges
- ML vetting of shots can prune the heuristic's bad launches (false-aim, walk-into-reinforcement)
  that pure rules miss — a learned safety net on top of a strong heuristic.
- We have the weights locally (decoded), so it's fully runnable for benchmarking.

## Weaknesses
- The ML only **validates** (filters) shots; it doesn't generate strategy or model opponents. The
  ceiling is still the underlying lb-1224 heuristic.
- Two-stage (propose→vet) adds latency; under the 1s/turn budget the search space stays small.

## Ideas to steal
The **propose-then-learned-vet** pattern is a low-risk way to inject learning into a heuristic
without full RL: keep a strong generator (orbit_lite), train a discriminator on
"did this launch improve my outcome" from replays, veto bad ones. A pragmatic stepping-stone to
strategy #2/#8.
