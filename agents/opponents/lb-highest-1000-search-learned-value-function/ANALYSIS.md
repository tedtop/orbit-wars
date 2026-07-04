# lb-highest-1000-search-learned-value-function  —  ~1000 (AidenSong123)

## Core strategy
The closest public bot to our RL ambition: **search + a learned value function.** Generates
candidate (source, angle, ships) launches, rolls each forward with `simulate_outcome`, and scores
the resulting state with `_value_score` — a **gradient-boosted decision-tree** value model
(`GBC_TREES`, `feature_list`/`threshold_list`) over features = ship/planet/production
differentials vs the strongest enemy + a 2P/4P one-hot. Falls back to a pure heuristic when the
model file isn't present (it lives in a Kaggle dataset).

## Edges
- A **learned** leaf evaluation (trained on outcomes) instead of hand-tuned constants — the right
  idea. Value features are sensible (relative economy vs best enemy).
- Clean separation: simulate → featurize → learned score. Easy to swap the model.

## Weaknesses (key lesson)
Scores only ~1000 despite the learned value. Why: the value model is **shallow GBC over ~8 coarse
global features** — it can't see geometry, threats, or who-attacks-whom. And the simulate_outcome
rollout is short. So the learned signal is weak. **The architecture is right; the model is too
small and the features too coarse.**

## Ideas to steal (directly feeds our plan)
This is the proof-of-concept for strategy #2 (learned value/controller) and #8 (BC→RL): keep the
simulate→featurize→learned-score loop, but (a) use the orbit_lite exact simulator, (b) replace
coarse GBC with a per-candidate NN (like the RL tutorial's PlanetPolicy), (c) train on replays +
self-play. We can leapfrog this bot by upgrading exactly the part it under-built.
