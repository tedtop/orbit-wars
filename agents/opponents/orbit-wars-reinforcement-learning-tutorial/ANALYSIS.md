# orbit-wars-reinforcement-learning-tutorial  —  REFERENCE (YumeNeko/kashiwaba)

## What it is (not a playable bot — a training pipeline)
A complete **PPO self-play trainer** in `src/` (config, env, features, policy, ppo, opponents,
train). This is the single most valuable reference for our RL plan.

## Architecture (`src/policy.py` — `PlanetPolicy`)
Per-candidate scoring network — exactly right for this game:
- `self_encoder` (my-planet features) + `global_encoder` (board state) + `candidate_encoder`
  (per candidate-launch features), each an MLP to hidden_size=128.
- `target_head` over concat(self, global, candidate) → which target to attack.
- (paired) ship-bucket head → how many ships (discretized into `ship_bucket_count=8` buckets).
- `candidate_count=8`, `max_planets=48`. PPO: rollout 64, gamma 0.99, clip 0.2, lr 3e-4.
  Public notebook runs only `total_updates=100` (toy); comment says use 2000+ for real training.

## Why it matters
It hands us a working action representation (candidate-set policy), feature extractor, env wrapper,
PPO loop, and self-play opponent pool. We can: (1) generate features from the orbit_lite exact
simulator, (2) **behavior-clone** the orbit_lite family's moves to warm-start this policy, then
(3) fine-tune with PPO self-play against the downloaded bots. See strategy #8 (BC→RL) and #2.

## Caveats
Toy training budget; candidate/feature dims are modest. The env reward is sparse (win/loss) — we'd
add shaped rewards (production/ship deltas) to speed learning.
