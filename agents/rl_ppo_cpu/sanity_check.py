"""Quick sanity check: obs shape, forward pass, SPS measurement."""
import sys, os, math, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from agents.rl_ppo.train import (
    encode_obs, get_owned_planets, collect_planet_actions,
    make_launches, VecEnv, ActorCritic, OBS_DIM, DEVICE, CFG
)
import torch
from kaggle_environments import make

print(f"Device: {DEVICE}")
print(f"OBS_DIM: {OBS_DIM} (expected 272)")

# 1. Obs encoder
env = make("orbit_wars", debug=False, configuration={"seed": 42, "episodeSteps": 10})
env.reset()
obs = env.state[0].observation
vec = encode_obs(obs)
assert vec.shape == (OBS_DIM,), f"Bad obs shape: {vec.shape}"
print(f"✓ Obs encoder: {vec.shape}")

# 2. Forward pass
policy = ActorCritic().to(DEVICE)
n_params = sum(p.numel() for p in policy.parameters())
print(f"✓ Model: {n_params:,} params")

owned = get_owned_planets(obs)
print(f"  Owned planets at start: {len(owned)}")

actions, exp = collect_planet_actions(policy, [obs])
launches = make_launches(obs, actions[0])
print(f"✓ Actions: {len(actions[0])} planet decisions, {len(launches)} launches")

# 3. SPS measurement
print("\nMeasuring SPS (10 rollout steps, 4 envs)...")
n_envs = 4
venv = VecEnv(n_envs, seed0=100)
obs_p0 = [venv.envs[i].state[0].observation for i in range(n_envs)]
obs_p1 = [venv.envs[i].state[1].observation for i in range(n_envs)]

t0 = time.time()
n_steps = 0
for _ in range(10):
    acts0, _ = collect_planet_actions(policy, obs_p0)
    acts1, _ = collect_planet_actions(policy, obs_p1)
    rewards, dones, obs_p0, obs_p1 = venv.step(acts0, acts1)
    n_steps += n_envs

elapsed = time.time() - t0
sps = n_steps / elapsed
print(f"✓ SPS: {sps:.0f} (with {n_envs} envs, sequential stepping)")
print(f"  Note: SPS is ~constant w.r.t. num_envs for sequential env stepping")
print(f"  Expected 150M steps in: {150e6 / sps / 3600:.1f} hours at this SPS")
print(f"  On GPU server (4-10x faster CPU): {150e6 / (sps*4) / 3600:.1f}–{150e6 / (sps*10) / 3600:.1f} hours")

print("\nAll checks passed. Ready to train.")
