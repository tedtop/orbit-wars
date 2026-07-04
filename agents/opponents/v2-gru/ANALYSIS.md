> Shared engine: see `agents/opponents/ORBIT_LITE_FAMILY.md` for the full orbit_lite deep-dive.
> This file covers only what THIS fork changes and how to beat it.

# v2-gru  —  high score by leaderboard, "neural net" (SpeedSci)

## The headline finding
This is i-m-stronger PLUS a `TinyGRUStrategyController` (GRU(24→32) → modulates roi_threshold,
max_waves, ffa_leader_bonus from a 12-step history of game features like leader prod/ship gaps).
**BUT `GRU_WEIGHTS_AVAILABLE=False` and `GRU_EMBEDDED_STATE_DICT=None`** — the controller is
never loaded; `_load_gru_controller` returns None and it falls back to the static config. The
public "#4 neural net" runs **identically to i-m-stronger**. The NN is dead weight.

## Why this matters for us
The scaffolding proves the community believes a learned roi/wave meta-controller is worth points
— and then shipped it disabled. **Training that GRU (or any learned controller) on self-play is
free, unclaimed signal.** This is strategy #2 in our plan.

## How to beat it
Same as i-m-stronger (it IS i-m-stronger right now).
