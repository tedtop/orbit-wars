# orbit-wars-physics-helper-module  —  REFERENCE (Seventie/shaikabdussattar)

## What it is
A clean **pure-stdlib physics library** (not an agent). Extracted to `physics_helper.py`. ~29
well-factored helpers: `fleet_speed`, `segment_hits_sun`, `predict_planet_position`,
`predict_comet_position`, `comet_remaining_life`, `safe_angle_and_distance`, `arc_safe_angle`,
`search_safe_intercept`, `aim_with_prediction`, `travel_time`, `_verify_shot_hits`,
`_dynamic_tolerance`, and a `PhysicsStats` profiler.

## Why it matters
If we build a from-scratch (non-torch) bot, this is a ready-made, readable physics layer — no need
to reimplement sun-avoidance, lead-aim, or comet prediction. Lighter-weight than orbit_lite (no
torch dependency) and easy to audit. Good fallback engine for a stdlib heuristic (strategy #5).
