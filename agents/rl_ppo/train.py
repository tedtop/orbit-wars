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

import json, math, os, time, copy, random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from kaggle_environments import make

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

def log_metrics(row: dict, run_dir: str):
    os.makedirs(run_dir, exist_ok=True)
    path = os.path.join(run_dir, "metrics.jsonl")
    with open(path, "a") as f:
        f.write(json.dumps(row) + "\n")

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
    ent_coef        = 0.001,  # was 0.05 — entropy bonus 50× game signal, kept policy random
    vf_coef         = 0.5,
    grad_clip       = 1.0,
    weight_decay    = 1e-4,
    total_updates   = 10000,
    eval_every      = 100,
    save_every      = 200,
    checkpoint_every= 1000,
    reward_scale    = 0.01,   # was 0.001 — too small vs entropy bonus
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


def greedy_planet_actions(obs_list):
    """
    Greedy heuristic opponent: each owned planet fires 100% ships at the
    nearest non-owned planet. Returns same format as collect_planet_actions
    so it can be passed directly to vec_env.step as P1 actions.

    Used as cold-start opponent while the self-play pool is empty, so that
    P1 fires from step 0 and creates combat signal for the training policy.
    """
    all_actions = [[] for _ in range(len(obs_list))]
    for env_i, obs in enumerate(obs_list):
        if not isinstance(obs, dict):
            obs = {"player": obs.player, "planets": obs.planets}
        player  = int(obs.get("player", 0))
        planets = obs.get("planets", [])
        targets = [p for p in planets if p[1] != player]
        for src_idx, src in enumerate(planets[:MAX_PLANETS]):
            if src[1] != player or src[5] < MIN_SHIPS:
                continue
            if not targets:
                all_actions[env_i].append((src_idx, False, 0, 3))
                continue
            tgt = min(targets, key=lambda p: math.hypot(p[2]-src[2], p[3]-src[3]))
            tgt_idx = planets.index(tgt) if tgt in planets else 0
            all_actions[env_i].append((src_idx, True, tgt_idx, 3))  # frac=1.0
    return all_actions


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
        actions_p0: list[list[(src_idx, fire, tgt_idx, frac_idx)]] per env
        actions_p1: same format, OR a callable agent(obs)->[[planet,angle,ships]]
        Returns: rewards (2,n), dones (2,n), new_obs_p0, new_obs_p1
        """
        p1_callable = callable(actions_p1)
        rewards = np.zeros((2, self.n), dtype=np.float32)
        dones   = np.zeros((2, self.n), dtype=np.float32)
        obs_p0, obs_p1 = [], []

        for i in range(self.n):
            env = self.envs[i]
            l0 = actions_p0[i]

            # If env is done (episode finished last step but training loop
            # hasn't processed it yet), reset so we don't crash on step.
            if env.state and env.state[0].status not in ("ACTIVE", "IDLE", ""):
                self.reset_env(i)
                env = self.envs[i]

            obs0_pre = self.obs[i]
            obs1_pre_raw = env.state[1].observation

            def _agent0(obs, l=l0, o=obs0_pre):
                return make_launches(o, l)
            if p1_callable:
                def _agent1(obs, fn=actions_p1):
                    return fn(obs)
            else:
                l1 = actions_p1[i]
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

    # Compute explained variance and mean entropy over full batch (no grad needed)
    with torch.no_grad():
        o_all  = torch.from_numpy(obs_np).to(DEVICE)
        pf_all = torch.from_numpy(pf_np).to(DEVICE)
        mk_all = torch.from_numpy(mask_np).bool().to(DEVICE)
        ret_all = torch.from_numpy(ret_np).to(DEVICE)
        fd, td, frd, v_all = policy(o_all, pf_all, mk_all)
        ent_mean = (fd.entropy() + td.entropy() + frd.entropy()).mean().item()
        residual_var = torch.var(ret_all - v_all)
        total_var    = torch.var(ret_all)
        ev = (1.0 - residual_var / (total_var + 1e-8)).item()

    return total_loss / max(n_mb, 1), clip_frac_total / max(n_mb, 1), ev, ent_mean


# ── GAE computation ────────────────────────────────────────────────────────────

def compute_gae(buffer, cfg, bootstrap_vals=None):
    """Attach adv and ret to each buffer entry in-place.

    Bug fix: buffer entries are per-planet decisions, not per-timestep.
    Multiple planets fire per env per step — grouping by env_i alone and
    treating them as sequential steps causes the bootstrapped next-value
    to come from the NEXT PLANET at the same step instead of the next
    environmental state. Fix: tag each entry with step_i (rollout step
    index), group by (env_i, step_i) to get true timesteps, compute one
    advantage per timestep, broadcast to all planets at that step.
    """
    from collections import defaultdict

    # Map (env_i, step_i) -> list of buffer indices
    step_planet_idx = defaultdict(list)
    for k, e in enumerate(buffer):
        step_planet_idx[(e["env_i"], e["step_i"])].append(k)

    # Per-env: sorted list of step indices (true timestep sequence)
    env_steps = defaultdict(list)
    for (env_i, step_i) in step_planet_idx:
        env_steps[env_i].append(step_i)

    for env_i, steps in env_steps.items():
        steps = sorted(steps)
        T = len(steps)

        # All planets at the same (env_i, step_i) share the same value
        # (val_head depends only on global obs, not per-planet features).
        # Use the first planet's val/rew/done as the canonical timestep values.
        first_k = [step_planet_idx[(env_i, s)][0] for s in steps]
        vals  = np.array([buffer[k]["val"]  for k in first_k], dtype=np.float32)
        rews  = np.array([buffer[k]["rew"]  for k in first_k], dtype=np.float32)
        dones = np.array([buffer[k]["done"] for k in first_k], dtype=np.float32)

        # V(s_T): value of the state AFTER the last rollout step for this env.
        # Needed when the episode is still running at rollout boundary.
        boot_v = (bootstrap_vals or {}).get(env_i, 0.0)

        gae = 0.0
        adv = np.zeros(T, dtype=np.float32)
        for t in reversed(range(T)):
            if dones[t]:
                nv = 0.0
            elif t == T - 1:
                nv = boot_v  # use bootstrap instead of 0
            else:
                nv = vals[t + 1]
            delta = rews[t] + cfg["gamma"] * nv * (1 - dones[t]) - vals[t]
            gae   = delta + cfg["gamma"] * cfg["gae_lambda"] * (1 - dones[t]) * gae
            adv[t] = gae

        # Broadcast one advantage + return to all planets at each step
        for t, step_i in enumerate(steps):
            ret_t = adv[t] + vals[t]
            for k in step_planet_idx[(env_i, step_i)]:
                buffer[k]["adv"] = adv[t]
                buffer[k]["ret"] = ret_t


# ── Fixed opponent loader (comet_reaper cold-start) ──────────────────────────

def _load_fixed_opponent():
    """Load comet_reaper as a callable agent(obs)->launches for P1 cold-start."""
    import importlib.util, sys as _sys
    here = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(here, "..", "comet_reaper", "main.py")
    spec = importlib.util.spec_from_file_location("_comet_opp", path)
    mod  = importlib.util.module_from_spec(spec)
    _sys.modules["_comet_opp"] = mod   # required before @dataclass resolves __module__
    spec.loader.exec_module(mod)
    return mod.agent


# ── Evaluation vs greedy ──────────────────────────────────────────────────────

@torch.no_grad()
def evaluate_vs_greedy(policy, n_games=200):
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

    # Seat-balanced eval: alternate RL as P0 and P1
    wins_as_p0 = wins_as_p1 = 0
    half = n_games // 2
    for seed in range(half):
        env = make("orbit_wars", debug=False,
                   configuration={"seed": seed, "episodeSteps": TOTAL_STEPS})
        env.run([rl_agent, greedy_agent])
        r0 = env.steps[-1][0].reward or 0
        r1 = env.steps[-1][1].reward or 0
        if r0 > r1:
            wins_as_p0 += 1
    for seed in range(half, n_games):
        env = make("orbit_wars", debug=False,
                   configuration={"seed": seed, "episodeSteps": TOTAL_STEPS})
        env.run([greedy_agent, rl_agent])
        r0 = env.steps[-1][0].reward or 0
        r1 = env.steps[-1][1].reward or 0
        if r1 > r0:
            wins_as_p1 += 1
    wr = (wins_as_p0 + wins_as_p1) / n_games
    return wr, wins_as_p0 / half, wins_as_p1 / half


# ── Training loop ─────────────────────────────────────────────────────────────

def train():
    run_dir = CFG.get("run_dir", os.path.join(os.path.dirname(__file__), "runs", "default"))
    os.makedirs(run_dir, exist_ok=True)

    optimizer    = torch.optim.Adam(policy.parameters(), lr=CFG["lr"],
                                    weight_decay=CFG["weight_decay"])
    opponent_pool= OpponentPool()
    vec_env      = VecEnv(CFG["num_envs"], seed0=0)

    # Cold-start opponent: comet_reaper fires at us from update 1,
    # providing combat signal before the self-play pool has any checkpoints.
    _fixed_opp = None
    try:
        _fixed_opp = _load_fixed_opponent()
        print("  [Opponent] comet_reaper loaded as P1 cold-start opponent")
    except Exception as e:
        print(f"  [Opponent] comet_reaper unavailable ({e}), greedy fallback")

    # Track current obs for both seats
    obs_p0 = [vec_env.envs[i].state[0].observation for i in range(vec_env.n)]
    obs_p1 = [vec_env.envs[i].state[1].observation for i in range(vec_env.n)]
    prev_adv_p0 = [ship_advantage(o) for o in obs_p0]
    prev_adv_p1 = [ship_advantage(o) for o in obs_p1]

    update     = 0
    total_steps= 0
    best_wr    = 0.0
    t0         = time.time()

    # Auto-resume from latest checkpoint if one exists
    ckpt_path = os.path.join(run_dir, "latest.pt")
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=DEVICE)
        policy.load_state_dict(ckpt["state"])
        if "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
        update      = ckpt.get("update", 0)
        total_steps = ckpt.get("steps", 0)
        best_wr     = ckpt.get("best_wr", 0.0)
        print(f"  [Resume] Loaded checkpoint from U{update} ({total_steps:,} steps)")

    print(f"{'='*64}")
    print(f"PPO Self-Play | {DEVICE} | {CFG['num_envs']} envs")
    print(f"{'='*64}\n")

    while update < CFG["total_updates"]:
        buffer   = []
        ep_count = 0

        for step_i in range(CFG["rollout_steps"]):
            # P1 opponent selection:
            #   - Pool empty → comet_reaper (fires from step 1, breaks cold start)
            #   - Pool has checkpoints → 50% self-play, 50% pool sample
            pool_ready = len(opponent_pool.checkpoints) > 0
            if not pool_ready:
                acts_p1 = _fixed_opp if _fixed_opp is not None \
                          else greedy_planet_actions(obs_p1)
            elif random.random() < 0.5:
                acts_p1, _ = collect_planet_actions(opponent_pool.sample(), obs_p1)
            else:
                acts_p1, _ = collect_planet_actions(policy, obs_p1)

            with torch.no_grad():
                acts_p0, exp_p0 = collect_planet_actions(policy, obs_p0)

            rewards, dones, new_obs_p0, new_obs_p1 = vec_env.step(acts_p0, acts_p1)

            # Snapshot BOTH old and new advantages before the loop.
            # new_adv_snap must be captured now — the reset branch overwrites
            # new_obs_p0[env_i] mid-loop, corrupting delta for 2nd+ planets
            # of a done env if we call ship_advantage() inside the loop.
            baseline_snap = list(prev_adv_p0)
            new_adv_snap  = [ship_advantage(new_obs_p0[i]) for i in range(vec_env.n)]
            reset_this_step = set()

            for exp in exp_p0:
                env_i = exp["env_i"]
                my0, en0 = baseline_snap[env_i]
                my1, en1 = new_adv_snap[env_i]
                delta = (my1 - en1) - (my0 - en0)
                r = delta * CFG["reward_scale"]
                if dones[0, env_i]:
                    r += rewards[0, env_i] * CFG["terminal_bonus"]
                    if env_i not in reset_this_step:  # reset exactly once per done env
                        ep_count += 1
                        vec_env.reset_env(env_i)
                        new_obs_p0[env_i] = vec_env.envs[env_i].state[0].observation
                        new_obs_p1[env_i] = vec_env.envs[env_i].state[1].observation
                        prev_adv_p0[env_i] = ship_advantage(new_obs_p0[env_i])
                        prev_adv_p1[env_i] = ship_advantage(new_obs_p1[env_i])
                        reset_this_step.add(env_i)
                else:
                    # Update baseline every step so delta = one-step improvement,
                    # not cumulative drift from episode start
                    prev_adv_p0[env_i] = (my1, en1)
                exp["rew"]    = r
                exp["done"]   = dones[0, env_i]
                exp["step_i"] = step_i
                buffer.append(exp)

            obs_p0 = new_obs_p0
            obs_p1 = new_obs_p1
            total_steps += CFG["num_envs"]

        if not buffer:
            continue

        # Bootstrap: get V(s_T) for envs whose episode is still running at rollout end.
        # Without this, the last step of every rollout sets nv=0 and underestimates
        # future value for ~1/rollout_steps fraction of the buffer.
        with torch.no_grad():
            _, boot_exp = collect_planet_actions(policy, obs_p0)
        bootstrap_vals = {}
        for e in boot_exp:
            if e["env_i"] not in bootstrap_vals:
                bootstrap_vals[e["env_i"]] = e["val"]

        compute_gae(buffer, CFG, bootstrap_vals)
        loss, cf, ev, ent = ppo_update(policy, optimizer, buffer, CFG)
        update += 1

        sps = total_steps / (time.time() - t0)
        log_metrics({
            "update": update, "steps": total_steps, "loss": round(loss, 5),
            "clip_frac": round(cf, 4), "explained_variance": round(ev, 4),
            "entropy": round(ent, 4), "ep_count": ep_count,
            "sps": round(sps, 1), "elapsed_h": round((time.time() - t0) / 3600, 3),
            "ts": int(time.time()),
        }, run_dir)
        if update % 10 == 0:
            print(f"U{update:5d} | S:{total_steps:>10,d} | L:{loss:.4f} | CF:{cf:.3f} "
                  f"| EV:{ev:.3f} | Ent:{ent:.3f} | EP:{ep_count} | SPS:{sps:.0f} | {(time.time()-t0)/3600:.1f}h")

        if update % CFG["eval_every"] == 0:
            wr, wr_p0, wr_p1 = evaluate_vs_greedy(policy)
            print(f"  -> Eval vs greedy: {wr:.0%}  (as P0: {wr_p0:.0%}  as P1: {wr_p1:.0%})")
            log_metrics({"update": update, "steps": total_steps,
                         "eval_vs_greedy": round(wr, 4),
                         "eval_as_p0": round(wr_p0, 4), "eval_as_p1": round(wr_p1, 4),
                         "ts": int(time.time())}, run_dir)
            if wr > best_wr:
                best_wr = wr
                torch.save({"state": policy.state_dict(), "update": update,
                            "steps": total_steps}, os.path.join(run_dir, "best_model.pt"))
                print(f"  *** New best: {wr:.0%} ***")

        if update % CFG["checkpoint_every"] == 0:
            opponent_pool.add(policy)
            torch.save({"state": policy.state_dict(), "update": update,
                        "steps": total_steps}, os.path.join(run_dir, f"ckpt_{update}.pt"))

        if update % CFG["save_every"] == 0:
            torch.save({"state": policy.state_dict(), "optimizer": optimizer.state_dict(),
                        "update": update, "steps": total_steps, "best_wr": best_wr},
                       os.path.join(run_dir, "latest.pt"))

    torch.save({"state": policy.state_dict(), "update": update,
                "steps": total_steps}, os.path.join(run_dir, "final_model.pt"))
    print(f"\nDone. Total steps: {total_steps:,d}")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--num_envs",      type=int,   default=None)
    p.add_argument("--total_updates", type=int,   default=None)
    p.add_argument("--lr",            type=float, default=None)
    p.add_argument("--reward_scale",  type=float, default=None)
    p.add_argument("--run_name",      type=str,   default="default")
    args = p.parse_args()
    if args.num_envs      is not None: CFG["num_envs"]      = args.num_envs
    if args.total_updates is not None: CFG["total_updates"]  = args.total_updates
    if args.lr            is not None: CFG["lr"]             = args.lr
    if args.reward_scale  is not None: CFG["reward_scale"]   = args.reward_scale

    RUN_DIR = os.path.join(os.path.dirname(__file__), "runs", args.run_name)
    os.makedirs(RUN_DIR, exist_ok=True)
    CFG["run_dir"] = RUN_DIR
    train()
