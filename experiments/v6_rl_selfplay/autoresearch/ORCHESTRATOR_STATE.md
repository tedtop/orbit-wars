# ORCHESTRATOR STATE — v6 RL Self-Play

_Last updated: 2026-06-18. Deadline: **2026-06-23 23:59 UTC** (~5 days)._

## STATUS: BUILDING

## 🔻 IMMEDIATE NEXT
- [ ] Run sanity check: `cd agents/rl_ppo && python sanity_check.py`
  - Validates obs encoder shape (272-dim)
  - Validates per-planet action heads fire correctly
  - Measures local SPS (target: 100+ on CPU)
- [ ] If SPS OK: launch full training on Jetstream2 / Lambda Labs GH200
  - Set `num_envs=128` for GPU server
  - Target: 150M steps (beats public agents), then 600M+ steps
- [ ] Monitor: checkpoint eval vs comet_reaper every 1000 updates
- [ ] Submit best checkpoint when win% vs comet_reaper > 55% (live LB calibration)

## Architecture
- **Policy**: Per-planet independent heads (ActorCritic in `agents/rl_ppo/train.py`)
  - Shared backbone: 3-layer MLP (256 hidden, LayerNorm + Tanh)
  - Per-planet: fire head (Bernoulli) + target head (Categorical/20) + fraction head (4 bins)
  - ~200K parameters
- **Observation**: 272-dim flat (20 planets × 10 feats + 12 fleets × 8 feats + 6 global)
- **Action**: Each owned planet independently: fire? → target planet → ship fraction [25/50/75/100%]
- **Reward**: ship-advantage delta × 0.001 (dense) + terminal win/loss × 1.0

## Training plan
- Phase 1 (updates 0-50): Pure self-play
- Phase 2 (updates 50-200): Add comet_reaper + schmeekler_fmt as fixed opponents
- Phase 3 (200+): Checkpoint pool (PFSP) — sample from past snapshots
- Evaluate vs greedy every 200 updates
- Add checkpoint to pool every 1000 updates

## Hyperparameters
- num_envs=64 (local) / 128 (GPU server)
- rollout_steps=64, ppo_epochs=2, lr=3e-4, clip_eps=0.2, ent_coef=0.05
- gamma=0.99, gae_lambda=0.95, vf_coef=0.5, grad_clip=1.0

## Submission packaging
- `python agents/rl_ppo/make_submission.py best_model.pt`
- Pure numpy inference (no torch import) — avoids NumPy 2.x / PyTorch incompatibility
- Weights embedded as base64 in main.py — stays well under 100MB limit

## Forum intel baseline
- Radek (68th): 1303 LB in $15 / 6.5hrs on GH200 (@1k SPS)
- Lin Myat Ko (#1): 1793 LB, 600M steps, $150 on RTX 5090
- Fanghhhh (261st): 8K SPS on RTX 6000 Ada with 128 envs
- Mendrika (48th): 1420 LB pure BC with per-planet fire head
- Abhyuday (31st): RL beat handcrafted in 1 day, +120 elo from 4P policy

## Live ladder at v6 start
- comet_reaper: 1234.7 (inactive, our floor — resubmit before Jun 23)
- schmeekler_fmt: ~1125 (active, converging)
- schmeekler: ~1098 (active)
- #1: ~1793. Prize zone: ~1500.

## Hard rules
- Repo PUBLIC — never git push
- Ted reviews every submission
- comet_reaper MUST be resubmitted before Jun 23 if RL bot doesn't beat it
