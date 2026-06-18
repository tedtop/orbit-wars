"""
PPO self-play training for Orbit Wars.

Design choices (informed by forum intel):
- Per-planet independent action heads: each owned planet independently decides
  fire? -> target -> fraction. Achieves ~3 launches/turn (vs 1 for global heads).
- Self-play: both seats share one policy; observations are player-relative.
- Source masking: owned planets with >= MIN_SHIPS only.
- Reward: ship-advantage delta (dense) + terminal win/loss bonus.
- Opponent pool: pure self-play first, then checkpoint snapshots added.
- Submission inference: pure numpy (no torch import at runtime).
"""

import math, os, time, copy, random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from kaggle_environments import make

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

# ── Observation constants ──────────────────────────────────────────────────────
MAX_PLANETS = 20
MAX_FLEETS  = 12
PLANET_FEATS = 10   # x,y,r,ships,prod,is_mine,is_enemy,is_neutral,dist_sun,angle
FLEET_FEATS  = 8    # x,y,ships,angle,is_mine,is_enemy,dist_sun,speed_norm
GLOBAL_FEATS = 6    # step,my_ships,en_ships,my_planets,en_planets,ship_adv
OBS_DIM = MAX_PLANETS * PLANET_FEATS + MAX_FLEETS * FLEET_FEATS + GLOBAL_FEATS

FRAC_BINS = [0.25, 0.5, 0.75, 1.0]
N_FRACS   = len(FRAC_BINS)
MIN_SHIPS = 3.0
TOTAL_STEPS = 500

# ── Hyperparameters ────────────────────────────────────────────────────────────
CFG = dict(
    num_envs        = 64,     # reduce for local testing; use 128 on GPU server
    rollout_steps   = 64,
    ppo_epochs      = 2,
    num_minibatches = 8,
    lr              = 3e-4,
    gamma           = 0.99,
    gae_lambda      = 0.95,
    clip_eps        = 0.2,
    ent_coef        = 0.05,
    vf_coef         = 0.5,
    grad_clip       = 1.0,
    weight_decay    = 1e-4,
    total_updates   = 10000,
    eval_every      = 200,
    save_every      = 500,
    checkpoint_every= 1000,   # add checkpoint to opponent pool
    reward_scale    = 0.001,
    terminal_bonus  = 1.0,
)

HIDDEN   = 256
N_LAYERS = 3


# ── Observation encoder ────────────────────────────────────────────────────────

def encode_obs(obs) -> np.ndarray:
    if not isinstance(obs, dict):
        obs = {"player": obs.player, "planets": obs.planets,
               "fleets": obs.fleets, "step": getattr(obs, "step", 0)}
    player  = int(obs.get("player", 0))
    planets = obs.get("planets", [])
    fleets  = obs.get("fleets", [])
    step    = int(obs.get("step", 0))

    pf = np.zeros((MAX_PLANETS, PLANET_FEATS), dtype=np.float32)
    for i, p in enumerate(planets[:MAX_PLANETS]):
        pid, owner, x, y, r, ships, prod = p[:7]
        dx, dy = x - 50.0, y - 50.0
        dist_sun = math.sqrt(dx*dx + dy*dy) / 70.0
        angle    = math.atan2(dy, dx) / math.pi
        pf[i] = [
            x / 100.0, y / 100.0, r / 10.0,
            ships / 200.0, prod / 20.0,
            1.0 if owner == player else 0.0,
            1.0 if (owner >= 0 and owner != player) else 0.0,
            1.0 if owner == -1 else 0.0,
            dist_sun, angle,
        ]

    ff = np.zeros((MAX_FLEETS, FLEET_FEATS), dtype=np.float32)
    for i, f in enumerate(fleets[:MAX_FLEETS]):
        fid, owner, x, y, angle, from_id, ships = f[:7]
        spd = (1.0 + 5.0 * (max(ships, 1) / 1000.0) ** 1.5) if ships > 0 else 0.0
        dx, dy = x - 50.0, y - 50.0
        dist_sun = math.sqrt(dx*dx + dy*dy) / 70.0
        ff[i] = [
            x / 100.0, y / 100.0, ships / 200.0, angle / math.pi,
            1.0 if owner == player else 0.0,
            1.0 if (owner >= 0 and owner != player) else 0.0,
            dist_sun, spd / 6.0,
        ]

    my_s = sum(p[5] for p in planets if p[1] == player) + \
           sum(f[6] for f in fleets  if f[1] == player)
    en_s = sum(p[5] for p in planets if p[1] >= 0 and p[1] != player) + \
           sum(f[6] for f in fleets  if f[1] >= 0 and f[1] != player)
    my_p = sum(1 for p in planets if p[1] == player)
    en_p = sum(1 for p in planets if p[1] >= 0 and p[1] != player)

    gf = np.array([
        step / 500.0,
        my_s / 1000.0, en_s / 1000.0,
        my_p / 20.0,   en_p / 20.0,
        (my_s - en_s) / 1000.0,
    ], dtype=np.float32)

    return np.concatenate([pf.flatten(), ff.flatten(), gf])


def get_owned_planets(obs):
    """Return list of (index, planet) for owned planets with enough ships."""
    if not isinstance(obs, dict):
        obs = {"player": obs.player, "planets": obs.planets}
    player  = int(obs.get("player", 0))
    planets = obs.get("planets", [])
    return [(i, p) for i, p in enumerate(planets[:MAX_PLANETS])
            if p[1] == player and p[5] >= MIN_SHIPS]


def ship_advantage(obs):
    if not isinstance(obs, dict):
        obs = {"player": obs.player, "planets": obs.planets, "fleets": obs.fleets}
    player  = int(obs.get("player", 0))
    planets = obs.get("planets", [])
    fleets  = obs.get("fleets", [])
    my = sum(p[5] for p in planets if p[1] == player) + \
         sum(f[6] for f in fleets  if f[1] == player)
    en = sum(p[5] for p in planets if p[1] >= 0 and p[1] != player) + \
         sum(f[6] for f in fleets  if f[1] >= 0 and f[1] != player)
    return float(my), float(en)


# ── Actor-Critic (per-planet heads) ───────────────────────────────────────────

class ActorCritic(nn.Module):
    """
    Per-planet policy: given the global obs embedding, each owned planet
    independently selects a target and ship fraction.

    At each turn we make one forward pass per owned planet (or batch them).
    The source planet's own features are concatenated to the shared embedding
    so the network knows which planet is deciding.
    """
    def __init__(self):
        super().__init__()
        # Shared backbone over global obs
        layers = []
        d = OBS_DIM
        for _ in range(N_LAYERS):
            layers += [nn.Linear(d, HIDDEN), nn.LayerNorm(HIDDEN), nn.Tanh()]
            d = HIDDEN
        self.backbone = nn.Sequential(*layers)

        # Per-planet heads: input = backbone output + source planet features
        src_input_dim = HIDDEN + PLANET_FEATS
        self.fire_head = nn.Linear(src_input_dim, 1)   # should this planet fire?
        self.tgt_head  = nn.Linear(src_input_dim, MAX_PLANETS)
        self.frac_head = nn.Linear(src_input_dim, N_FRACS)
        self.val_head  = nn.Linear(HIDDEN, 1)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=math.sqrt(2))
                nn.init.constant_(m.bias, 0)
        for h in (self.fire_head, self.tgt_head, self.frac_head):
            nn.init.orthogonal_(h.weight, gain=0.01)
        nn.init.orthogonal_(self.val_head.weight, gain=1.0)

    def shared_embed(self, obs_t: torch.Tensor) -> torch.Tensor:
        return self.backbone(obs_t)

    def value(self, obs_t: torch.Tensor) -> torch.Tensor:
        return self.val_head(self.shared_embed(obs_t)).squeeze(-1)

    def forward(self, obs_t, planet_feats_t, tgt_mask=None):
        """
        obs_t:          (B, OBS_DIM)
        planet_feats_t: (B, PLANET_FEATS)  — features of the source planet
        tgt_mask:       (B, MAX_PLANETS) bool, True = valid target
        Returns: fire_dist, tgt_dist, frac_dist, value
        """
        h = self.shared_embed(obs_t)
        x = torch.cat([h, planet_feats_t], dim=-1)

        fire_logits = self.fire_head(x).squeeze(-1)          # (B,)
        tgt_logits  = self.tgt_head(x)                       # (B, MAX_PLANETS)
        frac_logits = self.frac_head(x)                      # (B, N_FRACS)
        val         = self.val_head(h).squeeze(-1)           # (B,)

        if tgt_mask is not None:
            tgt_logits = tgt_logits.masked_fill(~tgt_mask, -1e8)

        fire_dist = torch.distributions.Bernoulli(logits=fire_logits)
        tgt_dist  = Categorical(logits=tgt_logits)
        frac_dist = Categorical(logits=frac_logits)
        return fire_dist, tgt_dist, frac_dist, val


policy = ActorCritic().to(DEVICE)
n_params = sum(p.numel() for p in policy.parameters())
print(f"Model: {n_params:,} parameters  |  OBS_DIM={OBS_DIM}")


# ── Action execution ───────────────────────────────────────────────────────────

def make_launches(obs, per_planet_actions):
    """
    Convert per-planet (fire, tgt_idx, frac_idx) decisions to launch list.
    per_planet_actions: list of (planet_index, fire, tgt_idx, frac_idx)
    Returns list of [from_planet_id, angle, n_ships]
    """
    if not isinstance(obs, dict):
        obs = {"player": obs.player, "planets": obs.planets}
    planets = obs.get("planets", [])
    player  = int(obs.get("player", 0))
    launches = []
    for src_idx, fire, tgt_idx, frac_idx in per_planet_actions:
        if not fire:
            continue
        if src_idx >= len(planets) or tgt_idx >= len(planets):
            continue
        if src_idx == tgt_idx:
            continue
        src = planets[src_idx]
        tgt = planets[tgt_idx]
        if src[1] != player or src[5] < MIN_SHIPS:
            continue
        angle = math.atan2(tgt[3] - src[3], tgt[2] - src[2])
        n_ships = max(1, int(src[5] * FRAC_BINS[frac_idx]))
        launches.append([int(src[0]), angle, n_ships])
    return launches


# ── Vectorized environment ─────────────────────────────────────────────────────

class VecEnv:
    def __init__(self, n, seed0=0):
        self.n    = n
        self.envs = []
        self.obs  = [None] * n
        for i in range(n):
            e = make("orbit_wars", debug=False,
                     configuration={"seed": seed0 + i, "episodeSteps": TOTAL_STEPS})
            e.reset()
            self.envs.append(e)
            self.obs[i] = e.state[0].observation

    def reset_env(self, i, seed=None):
        s = seed if seed is not None else random.randint(0, 999999)
        self.envs[i] = make("orbit_wars", debug=False,
                            configuration={"seed": s, "episodeSteps": TOTAL_STEPS})
        self.envs[i].reset()
        self.obs[i] = self.envs[i].state[0].observation

    def step(self, actions_p0, actions_p1):
        """
        actions_p0/p1: list of length n, each = list of per_planet_actions
        Returns: rewards (2,n), dones (2,n), new_obs_p0, new_obs_p1
        """
        rewards = np.zeros((2, self.n), dtype=np.float32)
        dones   = np.zeros((2, self.n), dtype=np.float32)
        obs_p0, obs_p1 = [], []

        for i in range(self.n):
            launches0 = make_launches(self.obs[i], actions_p0[i])
            # For p1, we need p1's obs — stored separately
            pass  # handled below

        # We step each env once, passing both agents' actions
        # kaggle_environments needs callable agents
        for i in range(self.n):
            env = self.envs[i]
            l0 = actions_p0[i]
            l1 = actions_p1[i]

            obs0_pre = self.obs[i]
            # Get p1's current obs from env state
            obs1_pre_raw = env.state[1].observation

            def _agent0(obs, l=l0, o=obs0_pre):
                return make_launches(o, l)
            def _agent1(obs, l=l1, o=obs1_pre_raw):
                return make_launches(o, l)

            env.step([_agent0, _agent1])

            for pl in (0, 1):
                s = env.state[pl]
                r = s.reward if s.reward is not None else 0.0
                done = s.status not in ("ACTIVE", "")
                rewards[pl, i] = float(r)
                dones[pl, i]   = float(done)

            self.obs[i] = env.state[0].observation
            obs_p0.append(env.state[0].observation)
            obs_p1.append(env.state[1].observation)

        return rewards, dones, obs_p0, obs_p1


# ── Rollout collection (per-planet) ───────────────────────────────────────────

@torch.no_grad()
def collect_planet_actions(policy, obs_list):
    """
    For a batch of observations, run per-planet inference.
    Returns:
      per_planet_actions: list[list[(src_idx, fire, tgt_idx, frac_idx)]]
      experience: list of experience dicts for PPO buffer (one per planet-decision)
    """
    all_actions = [[] for _ in range(len(obs_list))]
    experience  = []  # (obs_vec, planet_feat, fire, tgt, frac, lp_fire, lp_tgt, lp_frac, val)

    # Batch all planet decisions across all envs for efficient GPU inference
    obs_batch, pf_batch, mask_batch = [], [], []
    env_planet_idx = []  # (env_i, src_planet_idx)

    for env_i, obs in enumerate(obs_list):
        owned = get_owned_planets(obs)
        if not isinstance(obs, dict):
            obs = {"player": obs.player, "planets": obs.planets,
                   "fleets": obs.fleets, "step": getattr(obs, "step", 0)}
        planets = obs.get("planets", [])
        obs_vec = encode_obs(obs)

        for src_idx, p in owned:
            pid, owner, x, y, r, ships, prod = p[:7]
            dx, dy = x - 50.0, y - 50.0
            pf = np.array([x/100, y/100, r/10, ships/200, prod/20,
                           1.0, 0.0, 0.0,  # is_mine=1
                           math.sqrt(dx*dx+dy*dy)/70,
                           math.atan2(dy, dx)/math.pi], dtype=np.float32)

            # Target mask: can't target self
            tgt_mask = np.ones(MAX_PLANETS, dtype=bool)
            tgt_mask[src_idx] = False
            # Mask empty planet slots
            for j in range(len(planets), MAX_PLANETS):
                tgt_mask[j] = False

            obs_batch.append(obs_vec)
            pf_batch.append(pf)
            mask_batch.append(tgt_mask)
            env_planet_idx.append((env_i, src_idx, obs_vec))

    if not obs_batch:
        return all_actions, experience

    obs_t  = torch.from_numpy(np.stack(obs_batch)).to(DEVICE)
    pf_t   = torch.from_numpy(np.stack(pf_batch)).to(DEVICE)
    mask_t = torch.from_numpy(np.stack(mask_batch)).to(DEVICE)

    fire_d, tgt_d, frac_d, vals = policy(obs_t, pf_t, mask_t)
    fire_a = fire_d.sample()
    tgt_a  = tgt_d.sample()
    frac_a = frac_d.sample()

    lp_fire = fire_d.log_prob(fire_a)
    lp_tgt  = tgt_d.log_prob(tgt_a)
    lp_frac = frac_d.log_prob(frac_a)

    for k, (env_i, src_idx, obs_vec) in enumerate(env_planet_idx):
        fire = bool(fire_a[k].item())
        tgt  = int(tgt_a[k].item())
        frac = int(frac_a[k].item())
        all_actions[env_i].append((src_idx, fire, tgt, frac))
        experience.append({
            "obs": obs_vec,
            "pf":  pf_batch[k],
            "mask": mask_batch[k],
            "fire": int(fire_a[k].item()),
            "tgt":  tgt,
            "frac": frac,
            "lp_fire": lp_fire[k].item(),
            "lp_tgt":  lp_tgt[k].item(),
            "lp_frac": lp_frac[k].item(),
            "val":     vals[k].item(),
            "env_i":   env_i,
        })

    return all_actions, experience


# ── Opponent pool ──────────────────────────────────────────────────────────────

class OpponentPool:
    """Holds frozen checkpoint policies. Returns one policy per call."""
    def __init__(self):
        self.checkpoints = []  # list of state_dicts

    def add(self, policy):
        sd = copy.deepcopy(policy.state_dict())
        self.checkpoints.append(sd)
        print(f"  [OpponentPool] {len(self.checkpoints)} checkpoints")

    def sample(self):
        if not self.checkpoints:
            return None
        sd = random.choice(self.checkpoints)
        p = ActorCritic().to(DEVICE)
        p.load_state_dict(sd)
        p.eval()
        return p


# ── PPO update ────────────────────────────────────────────────────────────────

def ppo_update(policy, optimizer, buffer, cfg):
    """Run PPO epochs over the collected buffer."""
    obs_np   = np.stack([e["obs"]   for e in buffer])
    pf_np    = np.stack([e["pf"]    for e in buffer])
    mask_np  = np.stack([e["mask"]  for e in buffer])
    fire_np  = np.array([e["fire"]  for e in buffer], dtype=np.int64)
    tgt_np   = np.array([e["tgt"]   for e in buffer], dtype=np.int64)
    frac_np  = np.array([e["frac"]  for e in buffer], dtype=np.int64)
    lp_f_np  = np.array([e["lp_fire"] for e in buffer], dtype=np.float32)
    lp_t_np  = np.array([e["lp_tgt"]  for e in buffer], dtype=np.float32)
    lp_fr_np = np.array([e["lp_frac"] for e in buffer], dtype=np.float32)
    adv_np   = np.array([e["adv"]   for e in buffer], dtype=np.float32)
    ret_np   = np.array([e["ret"]   for e in buffer], dtype=np.float32)

    # Normalize advantages
    adv_np = (adv_np - adv_np.mean()) / (adv_np.std() + 1e-8)

    B = len(buffer)
    mb = B // cfg["num_minibatches"]
    total_loss = clip_frac_total = 0.0
    n_mb = 0

    for _ in range(cfg["ppo_epochs"]):
        idx = np.random.permutation(B)
        for s in range(0, B, mb):
            i = idx[s:s+mb]
            o  = torch.from_numpy(obs_np[i]).to(DEVICE)
            pf = torch.from_numpy(pf_np[i]).to(DEVICE)
            mk = torch.from_numpy(mask_np[i]).bool().to(DEVICE)
            fa = torch.from_numpy(fire_np[i]).to(DEVICE)
            ta = torch.from_numpy(tgt_np[i]).to(DEVICE)
            fra= torch.from_numpy(frac_np[i]).to(DEVICE)
            old_lp_f  = torch.from_numpy(lp_f_np[i]).to(DEVICE)
            old_lp_t  = torch.from_numpy(lp_t_np[i]).to(DEVICE)
            old_lp_fr = torch.from_numpy(lp_fr_np[i]).to(DEVICE)
            adv = torch.from_numpy(adv_np[i]).to(DEVICE)
            ret = torch.from_numpy(ret_np[i]).to(DEVICE)

            fire_d, tgt_d, frac_d, val = policy(o, pf, mk)

            new_lp_f  = fire_d.log_prob(fa.float())
            new_lp_t  = tgt_d.log_prob(ta)
            new_lp_fr = frac_d.log_prob(fra)

            log_ratio = (new_lp_f - old_lp_f) + (new_lp_t - old_lp_t) + (new_lp_fr - old_lp_fr)
            ratio = log_ratio.exp()

            pg_loss = -torch.min(
                ratio * adv,
                ratio.clamp(1 - cfg["clip_eps"], 1 + cfg["clip_eps"]) * adv
            ).mean()

            v_loss = F.mse_loss(val, ret)
            ent = (fire_d.entropy() + tgt_d.entropy() + frac_d.entropy()).mean()
            loss = pg_loss + cfg["vf_coef"] * v_loss - cfg["ent_coef"] * ent

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), cfg["grad_clip"])
            optimizer.step()

            clip_frac_total += ((ratio.detach() - 1).abs() > cfg["clip_eps"]).float().mean().item()
            total_loss += loss.item()
            n_mb += 1

    return total_loss / max(n_mb, 1), clip_frac_total / max(n_mb, 1)


# ── GAE computation ────────────────────────────────────────────────────────────

def compute_gae(buffer, cfg):
    """Attach adv and ret to each buffer entry in-place."""
    # Group by env_i, preserving order
    from collections import defaultdict
    env_entries = defaultdict(list)
    for k, e in enumerate(buffer):
        env_entries[e["env_i"]].append((k, e))

    for env_i, entries in env_entries.items():
        vals  = np.array([e["val"]  for _, e in entries], dtype=np.float32)
        rews  = np.array([e["rew"]  for _, e in entries], dtype=np.float32)
        dones = np.array([e["done"] for _, e in entries], dtype=np.float32)
        T = len(entries)

        gae = 0.0
        adv = np.zeros(T, dtype=np.float32)
        for t in reversed(range(T)):
            nv = 0.0 if dones[t] or t == T-1 else vals[t+1]
            delta = rews[t] + cfg["gamma"] * nv * (1 - dones[t]) - vals[t]
            gae = delta + cfg["gamma"] * cfg["gae_lambda"] * (1 - dones[t]) * gae
            adv[t] = gae

        ret = adv + vals
        for t, (k, e) in enumerate(entries):
            buffer[k]["adv"] = adv[t]
            buffer[k]["ret"] = ret[t]


# ── Evaluation vs greedy ──────────────────────────────────────────────────────

@torch.no_grad()
def evaluate_vs_greedy(policy, n_games=20):
    def greedy_agent(obs):
        d = obs if isinstance(obs, dict) else {
            "player": obs.player, "planets": obs.planets, "fleets": obs.fleets}
        player  = int(d.get("player", 0))
        planets = d.get("planets", [])
        my = [p for p in planets if p[1] == player and p[5] >= MIN_SHIPS]
        if not my:
            return []
        src = max(my, key=lambda p: p[5])
        targets = [p for p in planets if p[1] != player]
        if not targets:
            return []
        tgt = min(targets, key=lambda p: math.hypot(p[2]-src[2], p[3]-src[3]))
        angle = math.atan2(tgt[3]-src[3], tgt[2]-src[2])
        return [[int(src[0]), angle, int(src[5])]]

    def rl_agent(obs):
        d = obs if isinstance(obs, dict) else {
            "player": obs.player, "planets": obs.planets,
            "fleets": obs.fleets, "step": getattr(obs, "step", 0)}
        actions, _ = collect_planet_actions(policy, [d])
        return make_launches(d, actions[0])

    wins = 0
    for seed in range(n_games):
        env = make("orbit_wars", debug=False,
                   configuration={"seed": seed, "episodeSteps": TOTAL_STEPS})
        env.run([rl_agent, greedy_agent])
        r0 = env.steps[-1][0].reward or 0
        r1 = env.steps[-1][1].reward or 0
        if r0 > r1:
            wins += 1
    return wins / n_games


# ── Training loop ─────────────────────────────────────────────────────────────

def train():
    optimizer    = torch.optim.Adam(policy.parameters(), lr=CFG["lr"],
                                    weight_decay=CFG["weight_decay"])
    opponent_pool= OpponentPool()
    vec_env      = VecEnv(CFG["num_envs"], seed0=0)

    # Track current obs for both seats
    obs_p0 = [vec_env.envs[i].state[0].observation for i in range(vec_env.n)]
    obs_p1 = [vec_env.envs[i].state[1].observation for i in range(vec_env.n)]
    prev_adv_p0 = [ship_advantage(o) for o in obs_p0]
    prev_adv_p1 = [ship_advantage(o) for o in obs_p1]

    update     = 0
    total_steps= 0
    best_wr    = 0.0
    t0         = time.time()

    print(f"{'='*64}")
    print(f"PPO Self-Play | {DEVICE} | {CFG['num_envs']} envs")
    print(f"{'='*64}\n")

    while update < CFG["total_updates"]:
        buffer   = []
        ep_count = 0

        for _ in range(CFG["rollout_steps"]):
            # Decide which policy controls p1 (self or pool)
            use_pool = len(opponent_pool.checkpoints) > 0 and random.random() < 0.5
            opp_policy = opponent_pool.sample() if use_pool else policy

            with torch.no_grad():
                acts_p0, exp_p0 = collect_planet_actions(policy,    obs_p0)
                acts_p1, _      = collect_planet_actions(opp_policy, obs_p1)

            rewards, dones, new_obs_p0, new_obs_p1 = vec_env.step(acts_p0, acts_p1)

            # Compute shaped rewards
            for exp in exp_p0:
                env_i = exp["env_i"]
                my0, en0 = prev_adv_p0[env_i]
                my1, en1 = ship_advantage(new_obs_p0[env_i])
                delta = (my1 - en1) - (my0 - en0)
                r = delta * CFG["reward_scale"]
                if dones[0, env_i]:
                    r += rewards[0, env_i] * CFG["terminal_bonus"]
                    ep_count += 1
                    vec_env.reset_env(env_i)
                    new_obs_p0[env_i] = vec_env.envs[env_i].state[0].observation
                    new_obs_p1[env_i] = vec_env.envs[env_i].state[1].observation
                    prev_adv_p0[env_i] = ship_advantage(new_obs_p0[env_i])
                    prev_adv_p1[env_i] = ship_advantage(new_obs_p1[env_i])
                exp["rew"]  = r
                exp["done"] = dones[0, env_i]
                buffer.append(exp)

            obs_p0 = new_obs_p0
            obs_p1 = new_obs_p1
            total_steps += CFG["num_envs"]

        if not buffer:
            continue

        compute_gae(buffer, CFG)
        loss, cf = ppo_update(policy, optimizer, buffer, CFG)
        update += 1

        sps = total_steps / (time.time() - t0)
        if update % 10 == 0:
            print(f"U{update:5d} | S:{total_steps:>10,d} | L:{loss:.4f} | CF:{cf:.3f} "
                  f"| EP:{ep_count} | SPS:{sps:.0f} | {(time.time()-t0)/3600:.1f}h")

        if update % CFG["eval_every"] == 0:
            wr = evaluate_vs_greedy(20)
            print(f"  -> Eval vs greedy: {wr:.0%}")
            if wr > best_wr:
                best_wr = wr
                torch.save({"state": policy.state_dict(), "update": update,
                            "steps": total_steps}, "best_model.pt")
                print(f"  *** New best: {wr:.0%} ***")

        if update % CFG["checkpoint_every"] == 0:
            opponent_pool.add(policy)
            torch.save({"state": policy.state_dict(), "update": update,
                        "steps": total_steps}, f"ckpt_{update}.pt")

        if update % CFG["save_every"] == 0:
            torch.save({"state": policy.state_dict(), "update": update,
                        "steps": total_steps}, "latest.pt")

    torch.save({"state": policy.state_dict(), "update": update,
                "steps": total_steps}, "final_model.pt")
    print(f"\nDone. Total steps: {total_steps:,d}")


if __name__ == "__main__":
    train()
