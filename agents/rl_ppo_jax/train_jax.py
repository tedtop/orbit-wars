"""
PureJaxRL-style PPO for Orbit Wars — JAX/Flax on A100.

Architecture: ActorCriticET (Entity Transformer) ported to Flax.
  - Same hyperparameters as train.py CFG
  - Rollout: jax.vmap over N_ENVS parallel games (one GPU call per step)
  - Self-play: P1 uses the same current policy params with obs from P1's perspective
  - Policy: 2-layer 64-dim 4-head Transformer, per-planet action heads, scalar value
  - Log format matches train.py so monitor.sh works unchanged

Usage:
  python train_jax.py
  python train_jax.py --num_envs 4096 --run_name v10_a100

On fresh Jetstream2 A100 instance first run:
  bash setup_gpu.sh && python train_jax.py
"""

import argparse
import importlib.util
import json
import math
import os
import sys
import time
from functools import partial
from typing import NamedTuple

import numpy as np
from tensorboardX import SummaryWriter

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.normpath(os.path.join(HERE, "../.."))
sys.path.insert(0, REPO)
sys.path.insert(0, HERE)

# ── JAX/Flax imports ──────────────────────────────────────────────────────────
import jax
import jax.numpy as jnp
import flax.linen as nn
import optax
from flax.training.train_state import TrainState

import orbit_jax as oj

print(f"JAX backend: {jax.default_backend()}")
print(f"Devices: {jax.devices()}")

# ── Hyperparameters (matches train.py CFG) ─────────────────────────────────────
CFG = dict(
    num_envs        = 1024,
    rollout_steps   = 64,
    ppo_epochs      = 2,
    num_minibatches = 8,
    lr              = 3e-4,
    gamma           = 0.99,
    gae_lambda      = 0.95,
    clip_eps        = 0.2,
    ent_coef        = 0.001,
    vf_coef         = 0.5,
    grad_clip       = 1.0,
    weight_decay    = 1e-4,
    total_updates   = 3000,
    eval_every      = 100,
    save_every      = 200,
    checkpoint_every= 300,
    reward_scale    = 0.01,
    terminal_bonus  = 1.0,
    pool_size           = 64,
    cr_games_per_update = 16,   # Python games vs comet_reaper mixed into each PPO update
    embed_dim       = 64,
    n_heads         = 4,
    n_layers        = 2,
)

N_FRACS  = 4
MIN_SHIPS = 3.0
FRAC_BINS = jnp.array([0.25, 0.5, 0.75, 1.0])
FRAC_BINS_NP = np.array([0.25, 0.5, 0.75, 1.0])

# ── Flax ActorCriticET ────────────────────────────────────────────────────────

class TransformerBlock(nn.Module):
    embed_dim: int
    n_heads:   int

    @nn.compact
    def __call__(self, x):
        x = x + nn.MultiHeadDotProductAttention(
            num_heads=self.n_heads, qkv_features=self.embed_dim, out_features=self.embed_dim,
            kernel_init=nn.initializers.orthogonal(math.sqrt(2)),
        )(nn.LayerNorm()(x))
        y = nn.LayerNorm()(x)
        y = nn.Dense(self.embed_dim * 4, kernel_init=nn.initializers.orthogonal(math.sqrt(2)))(y)
        y = nn.gelu(y)
        y = nn.Dense(self.embed_dim, kernel_init=nn.initializers.orthogonal(math.sqrt(2)))(y)
        return x + y


class ActorCriticET(nn.Module):
    embed_dim: int = 64
    n_heads:   int = 4
    n_layers:  int = 2

    @nn.compact
    def __call__(self, obs, planet_feats, tgt_mask):
        """
        obs:          (B, OBS_DIM)
        planet_feats: (B, PLANET_FEATS)  — source planet features
        tgt_mask:     (B, MAX_OBS_PLANETS) bool — valid target slots
        Returns fire_logits (B,), tgt_logits (B,P), frac_logits (B,4), value (B,)
        """
        B  = obs.shape[0]
        E  = self.embed_dim
        P  = oj.MAX_OBS_PLANETS
        PF = oj.PLANET_FEATS
        FF = oj.FLEET_FEATS
        GF = oj.GLOBAL_FEATS

        planets_flat = obs[:, :P*PF].reshape(B, P, PF)
        fleets_flat  = obs[:, P*PF:P*PF+oj.MAX_OBS_FLEETS*FF].reshape(B, oj.MAX_OBS_FLEETS, FF)
        glob_flat    = obs[:, P*PF+oj.MAX_OBS_FLEETS*FF:]

        p_emb = nn.Dense(E, kernel_init=nn.initializers.orthogonal(math.sqrt(2)))(planets_flat)
        f_emb = nn.Dense(E, kernel_init=nn.initializers.orthogonal(math.sqrt(2)))(fleets_flat)
        g_emb = nn.Dense(E, kernel_init=nn.initializers.orthogonal(math.sqrt(2)))(glob_flat)[:, None, :]

        seq = jnp.concatenate([g_emb, p_emb, f_emb], axis=1)
        for _ in range(self.n_layers):
            seq = TransformerBlock(E, self.n_heads)(seq)

        global_rep  = seq.mean(axis=1)            # (B, E)
        planet_reps = seq[:, 1:1+P]               # (B, P, E)

        src_ctx = jnp.concatenate([global_rep, planet_feats], axis=-1)   # (B, E+PF)

        fire_logits = nn.Dense(1, kernel_init=nn.initializers.orthogonal(0.01))(src_ctx).squeeze(-1)
        frac_logits = nn.Dense(N_FRACS, kernel_init=nn.initializers.orthogonal(0.01))(src_ctx)

        q = nn.Dense(E, kernel_init=nn.initializers.orthogonal(0.01))(src_ctx)   # (B, E)
        tgt_logits = jnp.einsum("be,bpe->bp", q, planet_reps) / math.sqrt(E)    # (B, P)
        tgt_logits = jnp.where(tgt_mask, tgt_logits, jnp.float32(-1e8))

        value = nn.Dense(1, kernel_init=nn.initializers.orthogonal(1.0))(global_rep).squeeze(-1)

        return fire_logits, tgt_logits, frac_logits, value


# ── State stacking helpers ────────────────────────────────────────────────────

def _stack_states(states):
    """Stack list of N GameState objects → batched GameState with leading dim N."""
    return jax.tree_util.tree_map(lambda *xs: jnp.stack(xs), *states)


# ── Vectorised obs encoding and env step (compiled once at import) ─────────────
# These are defined here so JIT cache is shared across the training loop.

_encode_p0 = jax.jit(jax.vmap(lambda st: oj.encode_obs(st, 0)))
_encode_p1 = jax.jit(jax.vmap(lambda st: oj.encode_obs(st, 1)))
_step_batch = jax.jit(jax.vmap(oj.step))


@jax.jit
def _compute_delta_reward(st_old, st_new):
    """Per-env ship-delta reward for P0.  Returns (N,) float32."""
    def score(st, p):
        ps = jnp.sum(jnp.where((st.p_owner == p) & st.p_alive, st.p_ships, 0.), axis=-1)
        fs = jnp.sum(jnp.where((st.f_owner == p) & st.f_alive, st.f_ships, 0.), axis=-1)
        return ps + fs
    my0 = score(st_old, 0); en0 = score(st_old, 1)
    my1 = score(st_new, 0); en1 = score(st_new, 1)
    return (my1 - en1) - (my0 - en0)   # (N,)


# ── Pre-generate reset state pool ─────────────────────────────────────────────

def build_reset_pool(n: int, seed0: int = 0) -> list:
    """Generate n GameState objects (Python-side). Slow, done once."""
    print(f"  Building {n}-state reset pool (seed {seed0}..{seed0+n-1})...")
    pool = [oj.reset(seed0 + i) for i in range(n)]
    print(f"  Pool ready.")
    return pool


# ── Per-step action sampling ──────────────────────────────────────────────────

def _sample_actions(params, model, obs_batch, p_owner_batch, p_alive_batch,
                    p_x_batch, p_y_batch, p_ships_batch, key, player=0):
    """
    For a batch of environments sample per-planet actions for `player` (0 or 1).

    Returns:
      actions:  (N_ENVS, MAX_PLANETS, 3) float — [angle, n_ships, fire_flag]
      log_probs:(N_ENVS,) — summed log prob per env
      values:   (N_ENVS,) — V(s) estimate
      aux:      dict of arrays needed for PPO update  (empty when player==1)
    """
    N = obs_batch.shape[0]
    P = oj.MAX_OBS_PLANETS

    obs_list, pf_list, mask_list, env_idx_list, slot_idx_list = [], [], [], [], []

    obs_np       = np.array(obs_batch)
    p_owner_np   = np.array(p_owner_batch)
    p_alive_np   = np.array(p_alive_batch)
    p_x_np       = np.array(p_x_batch)
    p_y_np       = np.array(p_y_batch)
    p_ships_np   = np.array(p_ships_batch)

    for env_i in range(N):
        for slot in range(P):
            if not p_alive_np[env_i, slot]:
                continue
            if p_owner_np[env_i, slot] != player:
                continue
            if p_ships_np[env_i, slot] < MIN_SHIPS:
                continue
            x, y   = p_x_np[env_i, slot], p_y_np[env_i, slot]
            ships   = p_ships_np[env_i, slot]
            dx, dy  = x - 50., y - 50.
            pf = np.array([
                x/100., y/100., 0./10., ships/200., 0./20.,
                1., 0., 0.,
                math.sqrt(dx*dx+dy*dy)/70.,
                math.atan2(dy, dx)/math.pi,
            ], dtype=np.float32)
            n_alive   = int(np.sum(p_alive_np[env_i, :P]))
            tgt_mask  = np.zeros(P, dtype=bool)
            tgt_mask[:n_alive] = True
            tgt_mask[slot] = False

            obs_list.append(obs_np[env_i])
            pf_list.append(pf)
            mask_list.append(tgt_mask)
            env_idx_list.append(env_i)
            slot_idx_list.append(slot)

    if not obs_list:
        dummy_actions = np.zeros((N, oj.MAX_PLANETS, 3), np.float32)
        return jnp.array(dummy_actions), jnp.zeros(N, np.float32), jnp.zeros(N, np.float32), {}

    obs_t  = jnp.array(np.stack(obs_list))
    pf_t   = jnp.array(np.stack(pf_list))
    mask_t = jnp.array(np.stack(mask_list))

    fire_l, tgt_l, frac_l, vals = model.apply(params, obs_t, pf_t, mask_t)

    key, k1, k2, k3 = jax.random.split(key, 4)
    fire_a = (jax.random.bernoulli(k1, jax.nn.sigmoid(fire_l))).astype(jnp.int32)
    tgt_a  = jax.random.categorical(k2, tgt_l)
    frac_a = jax.random.categorical(k3, frac_l)

    lp_fire = jax.nn.log_sigmoid(fire_l) * fire_a + jax.nn.log_sigmoid(-fire_l) * (1-fire_a)
    lp_tgt  = tgt_l - jax.nn.logsumexp(tgt_l, axis=-1, keepdims=True)
    lp_tgt  = jnp.take_along_axis(lp_tgt, tgt_a[:, None], axis=-1).squeeze(-1)
    lp_frac = frac_l - jax.nn.logsumexp(frac_l, axis=-1, keepdims=True)
    lp_frac = jnp.take_along_axis(lp_frac, frac_a[:, None], axis=-1).squeeze(-1)
    lp_total = lp_fire + lp_tgt + lp_frac

    fire_np = np.array(fire_a)
    tgt_np  = np.array(tgt_a)
    frac_np = np.array(frac_a)
    vals_np = np.array(vals)
    lp_np   = np.array(lp_total)

    actions  = np.zeros((N, oj.MAX_PLANETS, 3), np.float32)
    env_lp   = np.zeros(N, np.float32)
    env_val  = np.zeros(N, np.float32)

    for k, (env_i, slot) in enumerate(zip(env_idx_list, slot_idx_list)):
        fire = fire_np[k]
        tgt  = int(tgt_np[k])
        frac = int(frac_np[k])
        env_lp[env_i] += lp_np[k]

        if fire and tgt < P and p_alive_np[env_i, tgt]:
            sx, sy = p_x_np[env_i, slot], p_y_np[env_i, slot]
            tx, ty = p_x_np[env_i, tgt],  p_y_np[env_i, tgt]
            angle   = math.atan2(ty - sy, tx - sx)
            n_ships = max(1., float(p_ships_np[env_i, slot]) * float(FRAC_BINS_NP[frac]))
            actions[env_i, slot, 0] = angle
            actions[env_i, slot, 1] = n_ships
            actions[env_i, slot, 2] = 1.0

    for k, env_i in enumerate(env_idx_list):
        env_val[env_i] = vals_np[k]

    # Only return aux for player 0 (used in PPO buffer)
    if player == 0:
        aux = {
            "obs":     np.stack(obs_list),
            "pf":      np.stack(pf_list),
            "mask":    np.stack(mask_list),
            "fire":    fire_np,
            "tgt":     tgt_np,
            "frac":    frac_np,
            "lp":      lp_np,
            "val":     vals_np,
            "env_idx": np.array(env_idx_list),
            "slot_idx":np.array(slot_idx_list),
        }
    else:
        aux = {}

    return jnp.array(actions), jnp.array(env_lp), jnp.array(env_val), aux


# ── GAE ───────────────────────────────────────────────────────────────────────

def compute_gae(buffer, cfg):
    """Attach 'adv' and 'ret' to each buffer entry in-place."""
    from collections import defaultdict
    step_map = defaultdict(list)
    for k, e in enumerate(buffer):
        step_map[(e["env_i"], e["step_i"])].append(k)
    env_steps = defaultdict(list)
    for (env_i, step_i) in step_map:
        env_steps[env_i].append(step_i)

    for env_i, steps in env_steps.items():
        steps = sorted(steps)
        T = len(steps)
        first_k = [step_map[(env_i, s)][0] for s in steps]
        vals  = np.array([buffer[k]["val"]  for k in first_k], np.float32)
        rews  = np.array([buffer[k]["rew"]  for k in first_k], np.float32)
        dones = np.array([buffer[k]["done"] for k in first_k], np.float32)
        boot  = buffer[first_k[-1]].get("boot_val", 0.)
        gae   = 0.
        adv   = np.zeros(T, np.float32)
        for t in reversed(range(T)):
            nv = 0. if dones[t] else (boot if t == T-1 else vals[t+1])
            delta = rews[t] + cfg["gamma"] * nv * (1-dones[t]) - vals[t]
            gae   = delta + cfg["gamma"] * cfg["gae_lambda"] * (1-dones[t]) * gae
            adv[t] = gae
        for t, s in enumerate(steps):
            ret_t = adv[t] + vals[t]
            for k in step_map[(env_i, s)]:
                buffer[k]["adv"] = adv[t]
                buffer[k]["ret"] = ret_t


# ── PPO update ────────────────────────────────────────────────────────────────

def ppo_update(train_state: TrainState, buffer, cfg):
    obs_np   = np.stack([e["obs"]  for e in buffer])
    pf_np    = np.stack([e["pf"]   for e in buffer])
    mask_np  = np.stack([e["mask"] for e in buffer])
    fire_np  = np.array([e["fire"] for e in buffer], np.int32)
    tgt_np   = np.array([e["tgt"]  for e in buffer], np.int32)
    frac_np  = np.array([e["frac"] for e in buffer], np.int32)
    old_lp   = np.array([e["lp"]   for e in buffer], np.float32)
    adv_np   = np.array([e["adv"]  for e in buffer], np.float32)
    ret_np   = np.array([e["ret"]  for e in buffer], np.float32)

    adv_np = (adv_np - adv_np.mean()) / (adv_np.std() + 1e-8)
    B  = len(buffer)
    mb = max(1, B // cfg["num_minibatches"])

    @jax.jit
    def _update_step(ts, batch):
        def loss_fn(params):
            fl, tl, frl, val = ts.apply_fn(params, batch["obs"], batch["pf"], batch["mask"])
            lp_f  = jax.nn.log_sigmoid(fl)*batch["fire"].astype(jnp.float32) + \
                    jax.nn.log_sigmoid(-fl)*(1-batch["fire"].astype(jnp.float32))
            lp_t_all = tl - jax.nn.logsumexp(tl, axis=-1, keepdims=True)
            lp_t  = jnp.take_along_axis(lp_t_all, batch["tgt"][:,None], 1).squeeze(-1)
            lp_fr_all = frl - jax.nn.logsumexp(frl, axis=-1, keepdims=True)
            lp_fr = jnp.take_along_axis(lp_fr_all, batch["frac"][:,None], 1).squeeze(-1)
            new_lp = lp_f + lp_t + lp_fr
            ratio  = jnp.exp(new_lp - batch["old_lp"])
            adv    = batch["adv"]
            pg = -jnp.minimum(ratio*adv, jnp.clip(ratio, 1-cfg["clip_eps"], 1+cfg["clip_eps"])*adv).mean()
            vl = 0.5 * jnp.mean((val - batch["ret"])**2)
            p_fire = jax.nn.sigmoid(fl)
            ent_fire = -(p_fire*jnp.log(p_fire+1e-8) + (1-p_fire)*jnp.log(1-p_fire+1e-8)).mean()
            ent_tgt  = -jnp.sum(jax.nn.softmax(tl)*jax.nn.log_softmax(tl), axis=-1).mean()
            ent_frac = -jnp.sum(jax.nn.softmax(frl)*jax.nn.log_softmax(frl), axis=-1).mean()
            ent = ent_fire + ent_tgt + ent_frac
            loss = pg + cfg["vf_coef"]*vl - cfg["ent_coef"]*ent
            return loss, (pg, vl, ent, val, new_lp, ratio)
        (loss, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(ts.params)
        grads = jax.tree_util.tree_map(lambda g: jnp.clip(g, -cfg["grad_clip"], cfg["grad_clip"]), grads)
        ts    = ts.apply_gradients(grads=grads)
        return ts, loss, aux

    total_loss = 0.; n_mb = 0
    ent_m = 0.

    for _ in range(cfg["ppo_epochs"]):
        idx = np.random.permutation(B)
        for s in range(0, B, mb):
            sl = idx[s:s+mb]
            batch = {
                "obs":    jnp.array(obs_np[sl]),
                "pf":     jnp.array(pf_np[sl]),
                "mask":   jnp.array(mask_np[sl]),
                "fire":   jnp.array(fire_np[sl]),
                "tgt":    jnp.array(tgt_np[sl]),
                "frac":   jnp.array(frac_np[sl]),
                "old_lp": jnp.array(old_lp[sl]),
                "adv":    jnp.array(adv_np[sl]),
                "ret":    jnp.array(ret_np[sl]),
            }
            train_state, loss, (pg, vl, ent, val, nlp, ratio) = _update_step(train_state, batch)
            total_loss += float(loss)
            n_mb += 1

    with jax.disable_jit():
        _, _, _, val_all = train_state.apply_fn(
            train_state.params,
            jnp.array(obs_np), jnp.array(pf_np), jnp.array(mask_np)
        )
    ret_t = jnp.array(ret_np)
    ev    = float(1.0 - jnp.var(ret_t - val_all) / (jnp.var(ret_t) + 1e-8))
    ent_m = float(ent.mean()) if hasattr(ent, "mean") else float(ent)

    return train_state, total_loss / max(n_mb, 1), ev, ent_m


# ── Logging ───────────────────────────────────────────────────────────────────

def log_metrics(row, run_dir):
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, "metrics.jsonl"), "a") as f:
        f.write(json.dumps(row) + "\n")


# ── Python-side obs encoder for eval (numpy, matches orbit_jax.encode_obs) ────

def encode_obs_python(obs, player=0):
    """
    Encode kaggle observation to the same 302-dim vector as orbit_jax.encode_obs.
    Planet format: [id, owner, x, y, r, ships, prod]
    Fleet format:  [id, owner, x, y, angle, from_id, ships]
    """
    if isinstance(obs, dict):
        planets = list(obs.get('planets', []))
        fleets  = list(obs.get('fleets', []))
        step    = int(obs.get('step', 0))
    else:
        planets = list(getattr(obs, 'planets', None) or [])
        fleets  = list(getattr(obs, 'fleets',  None) or [])
        step    = int(getattr(obs, 'step',  0) or 0)
    p = player

    # Planet features (first MAX_OBS_PLANETS)
    pfeat = np.zeros((oj.MAX_OBS_PLANETS, oj.PLANET_FEATS), np.float32)
    for i, pl in enumerate(planets[:oj.MAX_OBS_PLANETS]):
        pid, ow, x, y, r, sh, prod = pl[0], int(pl[1]), float(pl[2]), float(pl[3]), \
                                      float(pl[4]), float(pl[5]), float(pl[6])
        dx, dy = x - oj.CENTER, y - oj.CENTER
        pfeat[i] = [
            x/100., y/100., r/10., sh/200., prod/20.,
            float(ow == p), float(ow >= 0 and ow != p), float(ow < 0),
            math.sqrt(dx*dx + dy*dy) / 70.,
            math.atan2(dy, dx) / math.pi,
        ]

    # Fleet features (first MAX_OBS_FLEETS)
    # Fleet: [id, owner, x, y, angle, from_id, ships]
    ffeat = np.zeros((oj.MAX_OBS_FLEETS, oj.FLEET_FEATS), np.float32)
    for i, fl in enumerate(fleets[:oj.MAX_OBS_FLEETS]):
        ow, x, y, ang, sh = int(fl[1]), float(fl[2]), float(fl[3]), float(fl[4]), float(fl[6])
        dx, dy = x - oj.CENTER, y - oj.CENTER
        sp = min(1. + (oj.SHIP_SPEED_MAX-1.) * (math.log(max(sh, 1.)) / math.log(1000.))**1.5,
                 oj.SHIP_SPEED_MAX)
        ffeat[i] = [
            x/100., y/100., sh/200., ang/math.pi,
            float(ow == p), float(ow >= 0 and ow != p),
            math.sqrt(dx*dx + dy*dy) / 70.,
            sp / oj.SHIP_SPEED_MAX,
        ]

    # Global features
    my_ps = sum(float(pl[5]) for pl in planets if int(pl[1]) == p)
    en_ps = sum(float(pl[5]) for pl in planets if int(pl[1]) >= 0 and int(pl[1]) != p)
    my_fs = sum(float(fl[6]) for fl in fleets if int(fl[1]) == p)
    en_fs = sum(float(fl[6]) for fl in fleets if int(fl[1]) >= 0 and int(fl[1]) != p)
    my_s, en_s = my_ps + my_fs, en_ps + en_fs
    my_pl = sum(1 for pl in planets if int(pl[1]) == p)
    en_pl = sum(1 for pl in planets if int(pl[1]) >= 0 and int(pl[1]) != p)
    gfeat = np.array([
        step / 500., my_s / 1000., en_s / 1000.,
        my_pl / 20., en_pl / 20., (my_s - en_s) / 1000.,
    ], np.float32)

    return np.concatenate([pfeat.ravel(), ffeat.ravel(), gfeat])


# ── RL agent factory for Python-side eval ─────────────────────────────────────

def make_rl_agent(params, model):
    """
    Return a callable `agent(obs) -> launch_list` compatible with kaggle env.run().
    Planet format: [id, owner, x, y, r, ships, prod]
    """
    def rl_agent(obs):
        if isinstance(obs, dict):
            player  = int(obs.get('player', 0))
            planets = list(obs.get('planets', []))
        else:
            player  = int(getattr(obs, 'player', 0) or 0)
            planets = list(getattr(obs, 'planets', None) or [])
        obs_vec = encode_obs_python(obs, player)

        owned = [(i, pl) for i, pl in enumerate(planets[:oj.MAX_OBS_PLANETS])
                 if int(pl[1]) == player and float(pl[5]) >= MIN_SHIPS]
        if not owned:
            return []

        launches = []
        for src_idx, src in owned:
            x, y, ships = float(src[2]), float(src[3]), float(src[5])
            dx, dy = x - 50., y - 50.
            pf = np.array([
                x/100., y/100., float(src[4])/10., ships/200., float(src[6])/20.,
                1., 0., 0.,
                math.sqrt(dx*dx+dy*dy)/70.,
                math.atan2(dy, dx)/math.pi,
            ], dtype=np.float32)

            n_alive = sum(1 for pl in planets[:oj.MAX_OBS_PLANETS])
            tgt_mask = np.zeros(oj.MAX_OBS_PLANETS, dtype=bool)
            tgt_mask[:n_alive] = True
            tgt_mask[src_idx] = False

            obs_t  = jnp.array(obs_vec[None])
            pf_t   = jnp.array(pf[None])
            mask_t = jnp.array(tgt_mask[None])

            fire_l, tgt_l, frac_l, _ = model.apply(params, obs_t, pf_t, mask_t)

            fire = bool(jax.nn.sigmoid(fire_l[0]) > 0.5)
            if not fire:
                continue

            tgt_idx = int(jnp.argmax(tgt_l[0]))
            frac_idx = int(jnp.argmax(frac_l[0]))

            if tgt_idx >= len(planets) or tgt_idx == src_idx:
                continue
            tgt = planets[tgt_idx]
            tx, ty = float(tgt[2]), float(tgt[3])
            angle = math.atan2(ty - y, tx - x)
            n_ships = max(1, int(ships * FRAC_BINS_NP[frac_idx]))
            launches.append([int(src[0]), angle, n_ships])

        return launches

    return rl_agent


# ── Stochastic RL step + CR game collection ───────────────────────────────────

def _rl_step_recording(params, model, obs, player, env_i, step_i, rng):
    """
    One turn of the RL agent with stochastic sampling.
    Returns (launch_list, buffer_entries).
    buffer_entries have all fields needed by compute_gae/ppo_update except boot_val.
    """
    if isinstance(obs, dict):
        player_id = int(obs.get('player', player))
        planets   = list(obs.get('planets', []))
    else:
        player_id = int(getattr(obs, 'player', player) or player)
        planets   = list(getattr(obs, 'planets', None) or [])

    obs_vec = encode_obs_python(obs, player_id)
    owned = [(i, pl) for i, pl in enumerate(planets[:oj.MAX_OBS_PLANETS])
             if int(pl[1]) == player_id and float(pl[5]) >= MIN_SHIPS]
    if not owned:
        return [], []

    launches = []
    entries  = []
    for src_idx, src in owned:
        x, y, ships = float(src[2]), float(src[3]), float(src[5])
        dx, dy = x - 50., y - 50.
        pf = np.array([
            x/100., y/100., float(src[4])/10., ships/200., float(src[6])/20.,
            1., 0., 0.,
            math.sqrt(dx*dx+dy*dy)/70., math.atan2(dy, dx)/math.pi,
        ], dtype=np.float32)
        n_alive   = len([pl for pl in planets[:oj.MAX_OBS_PLANETS]])
        tgt_mask  = np.zeros(oj.MAX_OBS_PLANETS, dtype=bool)
        tgt_mask[:n_alive] = True
        tgt_mask[src_idx]  = False

        fire_l, tgt_l, frac_l, val_l = model.apply(
            params, jnp.array(obs_vec[None]), jnp.array(pf[None]), jnp.array(tgt_mask[None])
        )

        # Stochastic fire
        fire_prob = float(jax.nn.sigmoid(fire_l[0]))
        fire      = int(rng.random() < fire_prob)
        fire_lp   = math.log(fire_prob + 1e-8) if fire else math.log(1 - fire_prob + 1e-8)

        # Stochastic target
        tgt_logits = np.array(tgt_l[0], dtype=np.float64)
        tgt_logits[~tgt_mask] = -1e9
        tgt_logits -= tgt_logits.max()
        tgt_probs  = np.exp(tgt_logits)
        tgt_probs[~tgt_mask] = 0.
        tgt_sum = tgt_probs.sum()
        tgt_probs = tgt_probs / tgt_sum if tgt_sum > 0 else (tgt_mask / tgt_mask.sum()).astype(np.float64)
        tgt_idx   = int(rng.choice(len(tgt_probs), p=tgt_probs))
        tgt_lp    = math.log(float(tgt_probs[tgt_idx]) + 1e-8)

        # Stochastic fraction
        frac_logits = np.array(frac_l[0], dtype=np.float64)
        frac_logits -= frac_logits.max()
        frac_probs  = np.exp(frac_logits)
        frac_probs /= frac_probs.sum()
        frac_idx  = int(rng.choice(len(frac_probs), p=frac_probs))
        frac_lp   = math.log(float(frac_probs[frac_idx]) + 1e-8)

        lp  = fire_lp + (tgt_lp + frac_lp if fire else 0.)
        val = float(val_l[0])

        entries.append({
            "obs": obs_vec, "pf": pf, "mask": tgt_mask,
            "fire": fire, "tgt": tgt_idx, "frac": frac_idx,
            "lp": lp, "val": val,
            "rew": 0.0, "done": 0.0,
            "env_i": env_i, "step_i": step_i,
        })

        if not fire or tgt_idx >= len(planets) or tgt_idx == src_idx:
            continue
        tgt    = planets[tgt_idx]
        angle  = math.atan2(float(tgt[3]) - y, float(tgt[2]) - x)
        n_ship = max(1, int(ships * FRAC_BINS_NP[frac_idx]))
        launches.append([int(src[0]), angle, n_ship])

    return launches, entries


def collect_cr_games(params, model, n_games, cfg, rng=None):
    """
    Run n_games of RL agent (P0) vs comet_reaper (P1) in Python.
    Returns buffer entries compatible with compute_gae/ppo_update.
    Uses large negative env_i values to avoid colliding with JAX env indices.
    """
    from kaggle_environments import make
    try:
        cr_agent = _load_comet_reaper()
    except Exception as e:
        print(f"  [CR training] comet_reaper unavailable: {e}")
        return []

    if rng is None:
        rng = np.random.default_rng()

    all_entries = []
    base_env_i  = -100000

    for game_i in range(n_games):
        env_i      = base_env_i - game_i
        game_entries = []
        seed = int(rng.integers(0, 100000))
        env  = make("orbit_wars", debug=False,
                    configuration={"seed": seed, "episodeSteps": oj.EPISODE_STEPS})
        env.reset()

        step_i = 0
        while env.state[0].status == "ACTIVE":
            obs_p0   = env.state[0].observation
            obs_p1   = env.state[1].observation
            launches, entries = _rl_step_recording(
                params, model, obs_p0, player=0, env_i=env_i, step_i=step_i, rng=rng
            )
            game_entries.extend(entries)
            cr_action = cr_agent(obs_p1)
            env.step([launches, cr_action])
            step_i += 1

        if game_entries:
            r0 = env.state[0].reward or 0
            r1 = env.state[1].reward or 0
            term_rew = cfg["terminal_bonus"] * (1.0 if r0 > r1 else -1.0)
            game_entries[-1]["rew"]  = term_rew
            game_entries[-1]["done"] = 1.0
            all_entries.extend(game_entries)

    return all_entries


# ── Evaluation vs greedy ──────────────────────────────────────────────────────

def evaluate_vs_greedy(params, model, n_games=30, seed_offset=None):
    """Seat-balanced eval vs nearest-planet greedy. Returns win rate dict."""
    from kaggle_environments import make

    def greedy_agent(obs):
        player  = int(getattr(obs, 'player', 0))
        planets = list(getattr(obs, 'planets', []))
        my = [pl for pl in planets if int(pl[1]) == player and float(pl[5]) >= MIN_SHIPS]
        if not my:
            return []
        src = max(my, key=lambda pl: float(pl[5]))
        targets = [pl for pl in planets if int(pl[1]) != player]
        if not targets:
            return []
        tgt = min(targets, key=lambda pl: math.hypot(float(pl[2])-float(src[2]),
                                                       float(pl[3])-float(src[3])))
        angle = math.atan2(float(tgt[3])-float(src[3]), float(tgt[2])-float(src[2]))
        return [[int(src[0]), angle, int(float(src[5]))]]

    rl_agent = make_rl_agent(params, model)
    wins_p0 = wins_p1 = 0
    half = n_games // 2
    # Use random seed offset each call so deterministic policies don't lock
    base = seed_offset if seed_offset is not None else np.random.randint(0, 10000)

    for seed in range(half):
        env = make("orbit_wars", debug=False,
                   configuration={"seed": base + seed, "episodeSteps": oj.EPISODE_STEPS})
        env.run([rl_agent, greedy_agent])
        r0 = env.steps[-1][0].reward or 0
        r1 = env.steps[-1][1].reward or 0
        if r0 > r1:
            wins_p0 += 1

    for seed in range(half, n_games):
        env = make("orbit_wars", debug=False,
                   configuration={"seed": base + seed, "episodeSteps": oj.EPISODE_STEPS})
        env.run([greedy_agent, rl_agent])
        r0 = env.steps[-1][0].reward or 0
        r1 = env.steps[-1][1].reward or 0
        if r1 > r0:
            wins_p1 += 1

    wr = (wins_p0 + wins_p1) / n_games
    return {"wr": wr, "wr_p0": wins_p0 / half, "wr_p1": wins_p1 / half}


# ── Evaluation vs comet_reaper ────────────────────────────────────────────────

def _load_comet_reaper():
    """Load comet_reaper agent from agents/comet_reaper/main.py."""
    path = os.path.join(REPO, "agents", "comet_reaper", "main.py")
    spec = importlib.util.spec_from_file_location("_cr_agent", path)
    mod  = importlib.util.module_from_spec(spec)
    sys.modules["_cr_agent"] = mod
    spec.loader.exec_module(mod)
    return mod.agent


def evaluate_vs_comet_reaper(params, model, n_games=20):
    """Seat-balanced eval vs comet_reaper. Returns win rate dict or None."""
    from kaggle_environments import make
    try:
        cr_agent = _load_comet_reaper()
    except Exception as e:
        print(f"  [Eval CR] unavailable: {e}")
        return None

    rl_agent = make_rl_agent(params, model)
    wins_p0 = wins_p1 = 0
    half = n_games // 2

    for seed in range(half):
        env = make("orbit_wars", debug=False,
                   configuration={"seed": seed + 50000, "episodeSteps": oj.EPISODE_STEPS})
        env.run([rl_agent, cr_agent])
        r0 = env.steps[-1][0].reward or 0
        r1 = env.steps[-1][1].reward or 0
        if r0 > r1:
            wins_p0 += 1

    for seed in range(half, n_games):
        env = make("orbit_wars", debug=False,
                   configuration={"seed": seed + 50000, "episodeSteps": oj.EPISODE_STEPS})
        env.run([cr_agent, rl_agent])
        r0 = env.steps[-1][0].reward or 0
        r1 = env.steps[-1][1].reward or 0
        if r1 > r0:
            wins_p1 += 1

    wr = (wins_p0 + wins_p1) / n_games
    return {"wr": wr, "wr_p0": wins_p0 / half, "wr_p1": wins_p1 / half}


# ── Snapshot pool ─────────────────────────────────────────────────────────────

class SnapshotPool:
    """Rolling buffer of historical policy params for diverse P1 opponents.
    Breaks pure self-play Nash convergence by making P1 a lagged version
    of the current policy rather than always the current policy."""
    def __init__(self, max_size=10):
        self._snaps = []
        self._max   = max_size

    def add(self, params):
        self._snaps.append(jax.device_get(params))  # copy to CPU
        if len(self._snaps) > self._max:
            self._snaps.pop(0)

    def sample_params(self, current_params):
        """50% current params, 50% random historical snapshot."""
        if self._snaps and np.random.random() < 0.5:
            return jax.device_put(self._snaps[np.random.randint(len(self._snaps))])
        return current_params


# ── Training loop ─────────────────────────────────────────────────────────────

def train(cfg, run_dir, seed=0):
    os.makedirs(run_dir, exist_ok=True)
    writer = SummaryWriter(run_dir)
    key = jax.random.PRNGKey(seed)
    _cr_rng = np.random.default_rng(seed)  # separate RNG for Python CR games

    # Init model
    model  = ActorCriticET(cfg["embed_dim"], cfg["n_heads"], cfg["n_layers"])
    dummy_obs  = jnp.zeros((1, oj.OBS_DIM))
    dummy_pf   = jnp.zeros((1, oj.PLANET_FEATS))
    dummy_mask = jnp.ones((1, oj.MAX_OBS_PLANETS), jnp.bool_)
    key, subkey = jax.random.split(key)
    params = model.init(subkey, dummy_obs, dummy_pf, dummy_mask)
    n_params = sum(x.size for x in jax.tree_util.tree_leaves(params))
    print(f"Model: {n_params:,} parameters | OBS_DIM={oj.OBS_DIM}")

    tx = optax.chain(
        optax.clip_by_global_norm(cfg["grad_clip"]),
        optax.adamw(cfg["lr"], weight_decay=cfg["weight_decay"]),
    )
    train_state = TrainState.create(apply_fn=model.apply, params=params, tx=tx)

    # Resume from checkpoint if one exists
    update = 0; total_steps = 0; best_wr = 0.
    import pickle
    ckpt_path = os.path.join(run_dir, "latest.pkl")
    if os.path.exists(ckpt_path):
        with open(ckpt_path, "rb") as f:
            ckpt = pickle.load(f)
        train_state = train_state.replace(params=jax.device_put(ckpt["params"]))
        update      = ckpt.get("update", 0)
        total_steps = ckpt.get("steps",  0)
        best_wr     = ckpt.get("best_wr", 0.)
        print(f"Resumed from checkpoint: U={update}, steps={total_steps:,}")

    # Opponent snapshot pool — seed with initial weights so P1 diverges from U=0
    snap_pool = SnapshotPool(max_size=10)
    snap_pool.add(train_state.params)

    # Reset state pool
    pool = build_reset_pool(cfg["pool_size"])

    N = cfg["num_envs"]

    # batch_state is the canonical env state; we never maintain a Python list of N states.
    batch_state = _stack_states([pool[i % len(pool)] for i in range(N)])

    ep_count = 0; last_dones_np = np.zeros(N, bool)

    print(f"{'='*64}")
    print(f"PPO JAX | {jax.default_backend()} | {N} envs | {cfg['total_updates']} updates")
    print(f"{'='*64}\n")

    t0 = time.time()

    while update < cfg["total_updates"]:
        buffer = []

        # ── Rollout ───────────────────────────────────────────────────────────
        for step_i in range(cfg["rollout_steps"]):
            # Encode obs for both players in two vectorised GPU calls
            obs_p0 = np.array(_encode_p0(batch_state))   # (N, OBS_DIM)
            obs_p1 = np.array(_encode_p1(batch_state))   # (N, OBS_DIM)

            # Extract shared state arrays for action sampling
            p_owner_np = np.array(batch_state.p_owner)   # (N, MAX_PLANETS)
            p_alive_np = np.array(batch_state.p_alive)
            p_x_np     = np.array(batch_state.p_x)
            p_y_np     = np.array(batch_state.p_y)
            p_ships_np = np.array(batch_state.p_ships)

            key, k0, k1 = jax.random.split(key, 3)

            # P0 actions (policy + PPO aux)
            actions_p0, lp_batch, val_batch, aux = _sample_actions(
                train_state.params, model,
                jnp.array(obs_p0), jnp.array(p_owner_np), jnp.array(p_alive_np),
                jnp.array(p_x_np), jnp.array(p_y_np), jnp.array(p_ships_np),
                k0, player=0,
            )

            # P1 actions — 50% current policy, 50% historical snapshot
            p1_params = snap_pool.sample_params(train_state.params)
            actions_p1, _, _, _ = _sample_actions(
                p1_params, model,
                jnp.array(obs_p1), jnp.array(p_owner_np), jnp.array(p_alive_np),
                jnp.array(p_x_np), jnp.array(p_y_np), jnp.array(p_ships_np),
                k1, player=1,
            )

            # Step ALL N environments in a single GPU kernel
            batch_state_new, _, _, term_rew, dones = _step_batch(
                batch_state, actions_p0, actions_p1)

            # Dense reward: ship-count delta for P0
            delta       = np.array(_compute_delta_reward(batch_state, batch_state_new))
            dones_np    = np.array(dones)        # (N,) bool
            term_rew_np = np.array(term_rew)     # (N,) float32
            step_rewards = delta * cfg["reward_scale"] + \
                           term_rew_np * dones_np * cfg["terminal_bonus"]

            # Reset done envs: vectorised scatter (one .at[].set per GameState field)
            done_idxs = np.where(dones_np)[0]
            if len(done_idxs):
                ep_count += len(done_idxs)
                reset_pool_idxs = np.random.randint(0, len(pool), size=len(done_idxs))
                reset_sts = [pool[pi] for pi in reset_pool_idxs]
                reset_batch = _stack_states(reset_sts)
                jidxs = jnp.array(done_idxs)
                batch_state_new = jax.tree_util.tree_map(
                    lambda b, r: b.at[jidxs].set(r),
                    batch_state_new, reset_batch)

            batch_state = batch_state_new   # persistent — no per-env unstack
            last_dones_np = dones_np
            total_steps += N

            # Store P0 experience
            if aux:
                for k, (env_i, slot_i) in enumerate(
                        zip(aux["env_idx"].tolist(), aux["slot_idx"].tolist())):
                    buffer.append({
                        "obs":    aux["obs"][k],
                        "pf":     aux["pf"][k],
                        "mask":   aux["mask"][k],
                        "fire":   int(aux["fire"][k]),
                        "tgt":    int(aux["tgt"][k]),
                        "frac":   int(aux["frac"][k]),
                        "lp":     float(aux["lp"][k]),
                        "val":    float(aux["val"][k]),
                        "rew":    float(step_rewards[env_i]),
                        "done":   float(dones_np[env_i]),
                        "env_i":  env_i,
                        "step_i": step_i,
                    })

        if not buffer:
            continue

        # Bootstrap: one vectorised model call for ALL envs, zero out done ones
        boot_obs = np.array(_encode_p0(batch_state))          # (N, OBS_DIM)
        boot_pf  = np.zeros((N, oj.PLANET_FEATS), np.float32)
        boot_mk  = np.ones((N, oj.MAX_OBS_PLANETS), bool)
        _, _, _, boot_vals_j = model.apply(
            train_state.params,
            jnp.array(boot_obs), jnp.array(boot_pf), jnp.array(boot_mk),
        )
        boot_vals = np.array(boot_vals_j) * (~last_dones_np)  # (N,) zero for done envs

        # Attach bootstrap value to last buffer entry per env
        seen = set()
        for e in reversed(buffer):
            ei = e["env_i"]
            if ei not in seen and not e["done"]:
                e["boot_val"] = float(boot_vals[ei])
                seen.add(ei)

        # Mix in Python games vs comet_reaper — gives real opponent signal
        if cfg.get("cr_games_per_update", 0) > 0:
            cr_entries = collect_cr_games(
                train_state.params, model, cfg["cr_games_per_update"], cfg, rng=_cr_rng
            )
            buffer.extend(cr_entries)

        compute_gae(buffer, cfg)
        train_state, loss, ev, ent = ppo_update(train_state, buffer, cfg)
        update += 1

        sps = total_steps / (time.time() - t0)
        row = {
            "update": update, "steps": total_steps,
            "loss": round(loss, 5), "explained_variance": round(ev, 4),
            "entropy": round(ent, 4), "ep_count": ep_count,
            "sps": round(sps, 1), "elapsed_h": round((time.time()-t0)/3600, 3),
            "ts": int(time.time()),
        }
        log_metrics(row, run_dir)
        writer.add_scalar("train/loss", loss, update)
        writer.add_scalar("train/explained_variance", ev, update)
        writer.add_scalar("train/entropy", ent, update)
        writer.add_scalar("train/sps", sps, update)

        if update % 10 == 0:
            print(f"U{update:5d} | S:{total_steps:>10,d} | L:{loss:.4f} "
                  f"| EV:{ev:.3f} | Ent:{ent:.3f} | EP:{ep_count} | SPS:{sps:.0f} "
                  f"| {(time.time()-t0)/3600:.1f}h")

        # ── Periodic eval ─────────────────────────────────────────────────────
        if update % cfg["eval_every"] == 0:
            print(f"  [U{update}] Running eval...")
            gr = evaluate_vs_greedy(train_state.params, model, n_games=30)
            cr = evaluate_vs_comet_reaper(train_state.params, model, n_games=20)
            cr_wr = cr["wr"] if cr else None
            print(f"  eval_vs_greedy={gr['wr']:.3f}  comet_reaper_WR={cr_wr}")
            eval_row = {
                "update": update, "steps": total_steps,
                "eval_vs_greedy": round(gr["wr"], 4),
                "eval_vs_greedy_p0": round(gr["wr_p0"], 4),
                "eval_vs_greedy_p1": round(gr["wr_p1"], 4),
                "comet_reaper_WR": round(cr_wr, 4) if cr_wr is not None else None,
                "comet_reaper_WR_p0": round(cr["wr_p0"], 4) if cr else None,
                "comet_reaper_WR_p1": round(cr["wr_p1"], 4) if cr else None,
                "ts": int(time.time()),
            }
            log_metrics(eval_row, run_dir)
            writer.add_scalar("eval/greedy_WR", gr["wr"], update)
            if cr_wr is not None:
                writer.add_scalar("eval/comet_reaper_WR", cr_wr, update)
            if cr_wr is not None and cr_wr > best_wr:
                best_wr = cr_wr

        # ── Snapshot pool update ──────────────────────────────────────────────
        if update % 50 == 0:
            snap_pool.add(train_state.params)

        # ── Checkpoint ────────────────────────────────────────────────────────
        if update % cfg["save_every"] == 0:
            ckpt_data = {"params": train_state.params, "update": update,
                         "steps": total_steps, "best_wr": best_wr}
            with open(os.path.join(run_dir, "latest.pkl"), "wb") as f:
                pickle.dump(ckpt_data, f)
            # Keep a named snapshot every checkpoint_every updates for rollback
            if update % cfg["checkpoint_every"] == 0:
                named = os.path.join(run_dir, f"ckpt_U{update:05d}.pkl")
                with open(named, "wb") as f:
                    pickle.dump(ckpt_data, f)

    writer.close()
    print(f"\nDone. Total steps: {total_steps:,d}")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="PureJAX PPO for Orbit Wars")
    p.add_argument("--num_envs",      type=int,   default=None)
    p.add_argument("--total_updates", type=int,   default=None)
    p.add_argument("--lr",            type=float, default=None)
    p.add_argument("--run_name",      type=str,   default="default")
    p.add_argument("--pool_size",          type=int, default=None)
    p.add_argument("--cr_games_per_update",type=int, default=None)
    p.add_argument("--seed",               type=int, default=0)
    args = p.parse_args()

    if args.num_envs      is not None: CFG["num_envs"]      = args.num_envs
    if args.total_updates is not None: CFG["total_updates"]  = args.total_updates
    if args.lr            is not None: CFG["lr"]             = args.lr
    if args.pool_size          is not None: CFG["pool_size"]           = args.pool_size
    if args.cr_games_per_update is not None: CFG["cr_games_per_update"] = args.cr_games_per_update

    run_dir = os.path.join(HERE, "runs", args.run_name)
    train(CFG, run_dir, seed=args.seed)
