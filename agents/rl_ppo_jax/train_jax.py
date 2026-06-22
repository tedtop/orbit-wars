"""
PureJaxRL-style PPO for Orbit Wars — JAX/Flax on A100.

Architecture: ActorCriticET (Entity Transformer) ported to Flax.
  - Same hyperparameters as train.py CFG
  - Rollout: jax.lax.scan over steps, jax.vmap over N_ENVS parallel games
  - Policy: 2-layer 64-dim 4-head Transformer, per-planet action heads, scalar value
  - Log format matches train.py so monitor.sh works unchanged

Usage:
  python train_jax.py
  python train_jax.py --num_envs 2048 --run_name v10_a100

On fresh Jetstream2 A100 instance first run:
  bash setup_gpu.sh && python train_jax.py
"""

import argparse
import json
import math
import os
import sys
import time
from functools import partial
from typing import NamedTuple

import numpy as np

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
    pool_size       = 512,     # number of pre-generated reset states
    embed_dim       = 64,
    n_heads         = 4,
    n_layers        = 2,
)

N_FRACS  = 4
MIN_SHIPS = 3.0
FRAC_BINS = jnp.array([0.25, 0.5, 0.75, 1.0])

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


# ── Pre-generate reset state pool ─────────────────────────────────────────────

def build_reset_pool(n: int, seed0: int = 0) -> list:
    """Generate n GameState objects (Python-side). Slow, done once."""
    print(f"  Building {n}-state reset pool (seed {seed0}..{seed0+n-1})...")
    pool = []
    for i in range(n):
        pool.append(oj.reset(seed0 + i))
    print(f"  Pool ready.")
    return pool


# ── Per-step action sampling ──────────────────────────────────────────────────

def _sample_actions(params, model, obs_batch, p_owner_batch, p_alive_batch,
                    p_x_batch, p_y_batch, p_ships_batch, key):
    """
    For a batch of environments, sample per-planet actions.

    Returns:
      actions:  (N_ENVS, MAX_PLANETS, 3) float — [angle, n_ships, fire_flag]
      log_probs:(N_ENVS,) — summed log prob per env (across all planet decisions)
      values:   (N_ENVS,) — V(s) estimate for P0
      aux:      dict of arrays needed for PPO update
    """
    N = obs_batch.shape[0]
    P = oj.MAX_OBS_PLANETS

    # Collect per-planet inputs (only owned planets with enough ships)
    # We batch ALL planet-decisions across ALL envs for GPU efficiency.
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
            if p_owner_np[env_i, slot] != 0:
                continue
            if p_ships_np[env_i, slot] < MIN_SHIPS:
                continue
            x, y = p_x_np[env_i, slot], p_y_np[env_i, slot]
            r_val = 0.0  # planet radius not directly needed here
            ships  = p_ships_np[env_i, slot]
            dx, dy = x - 50., y - 50.
            pf = np.array([
                x/100., y/100., 0./10., ships/200., 0./20.,
                1., 0., 0.,
                math.sqrt(dx*dx+dy*dy)/70.,
                math.atan2(dy, dx)/math.pi,
            ], dtype=np.float32)
            tgt_mask = np.ones(P, dtype=bool)
            tgt_mask[slot] = False
            for j in range(len([sl for sl in range(P) if p_alive_np[env_i, sl]]), P):
                tgt_mask[j] = False

            obs_list.append(obs_np[env_i])
            pf_list.append(pf)
            mask_list.append(tgt_mask)
            env_idx_list.append(env_i)
            slot_idx_list.append(slot)

    if not obs_list:
        dummy_actions = np.zeros((N, oj.MAX_PLANETS, 3), np.float32)
        dummy_lp = np.zeros(N, np.float32)
        dummy_val = np.zeros(N, np.float32)
        return jnp.array(dummy_actions), jnp.array(dummy_lp), jnp.array(dummy_val), {}

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
    lp_total = lp_fire + lp_tgt + lp_frac   # per planet decision

    fire_np = np.array(fire_a)
    tgt_np  = np.array(tgt_a)
    frac_np = np.array(frac_a)
    vals_np = np.array(vals)
    lp_np   = np.array(lp_total)

    actions     = np.zeros((N, oj.MAX_PLANETS, 3), np.float32)
    env_lp      = np.zeros(N, np.float32)
    env_val     = np.zeros(N, np.float32)

    for k, (env_i, slot) in enumerate(zip(env_idx_list, slot_idx_list)):
        fire = fire_np[k]
        tgt  = int(tgt_np[k])
        frac = int(frac_np[k])
        env_lp[env_i] += lp_np[k]

        if fire and tgt < P and p_alive_np[env_i, tgt]:
            sx, sy = p_x_np[env_i, slot], p_y_np[env_i, slot]
            tx, ty = p_x_np[env_i, tgt],  p_y_np[env_i, tgt]
            angle = math.atan2(ty - sy, tx - sx)
            n_ships = max(1., float(p_ships_np[env_i, slot]) * float(FRAC_BINS[frac]))
            actions[env_i, slot, 0] = angle
            actions[env_i, slot, 1] = n_ships
            actions[env_i, slot, 2] = 1.0

    # Value: mean over all planet decisions per env
    for k, env_i in enumerate(env_idx_list):
        env_val[env_i] = vals_np[k]   # last write wins (same global rep)

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
            # entropy
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
    ev = 0.; ent_m = 0.

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

    # Explained variance on full batch
    with jax.disable_jit():
        _, _, _, val_all = train_state.apply_fn(
            train_state.params,
            jnp.array(obs_np), jnp.array(pf_np), jnp.array(mask_np)
        )
    ret_t = jnp.array(ret_np)
    ev    = float(1.0 - jnp.var(ret_t - val_all) / (jnp.var(ret_t) + 1e-8))
    # Entropy estimate from last mini-batch
    ent_m = float(ent.mean()) if hasattr(ent, "mean") else float(ent)

    return train_state, total_loss / max(n_mb, 1), ev, ent_m


# ── Logging ───────────────────────────────────────────────────────────────────

def log_metrics(row, run_dir):
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, "metrics.jsonl"), "a") as f:
        f.write(json.dumps(row) + "\n")


# ── Training loop ─────────────────────────────────────────────────────────────

def train(cfg, run_dir):
    os.makedirs(run_dir, exist_ok=True)
    key = jax.random.PRNGKey(0)

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

    # Reset state pool
    pool = build_reset_pool(cfg["pool_size"])

    # Current game states (one per env)
    N = cfg["num_envs"]
    key, *env_keys = jax.random.split(key, N+1)
    env_states = [pool[i % len(pool)] for i in range(N)]

    # Auto-resume
    ckpt_path = os.path.join(run_dir, "latest.pt")
    update = 0; total_steps = 0; best_wr = 0.

    print(f"{'='*64}")
    print(f"PPO JAX | {jax.default_backend()} | {N} envs | {cfg['total_updates']} updates")
    print(f"{'='*64}\n")

    t0 = time.time()
    ep_count = 0

    while update < cfg["total_updates"]:
        buffer   = []

        # ── Rollout ───────────────────────────────────────────────────────────
        for step_i in range(cfg["rollout_steps"]):
            # Stack observations for batch inference
            obs_batch    = np.zeros((N, oj.OBS_DIM), np.float32)
            p_owner_batch = np.zeros((N, oj.MAX_PLANETS), np.int32)
            p_alive_batch = np.zeros((N, oj.MAX_PLANETS), bool)
            p_x_batch    = np.zeros((N, oj.MAX_PLANETS), np.float32)
            p_y_batch    = np.zeros((N, oj.MAX_PLANETS), np.float32)
            p_ships_batch = np.zeros((N, oj.MAX_PLANETS), np.float32)

            for env_i, st in enumerate(env_states):
                obs_batch[env_i]     = np.array(oj.encode_obs(st, 0))
                p_owner_batch[env_i] = np.array(st.p_owner)
                p_alive_batch[env_i] = np.array(st.p_alive)
                p_x_batch[env_i]     = np.array(st.p_x)
                p_y_batch[env_i]     = np.array(st.p_y)
                p_ships_batch[env_i] = np.array(st.p_ships)

            key, subkey = jax.random.split(key)
            actions_p0, lp_batch, val_batch, aux = _sample_actions(
                train_state.params, model,
                jnp.array(obs_batch),
                jnp.array(p_owner_batch), jnp.array(p_alive_batch),
                jnp.array(p_x_batch), jnp.array(p_y_batch), jnp.array(p_ships_batch),
                subkey,
            )

            # Opponent: self-play (p1 uses the same policy with obs from p1's perspective)
            # For simplicity: p1 uses a greedy "fire at nearest" heuristic
            # TODO: port full OpponentPool logic from train.py
            actions_p1 = np.zeros((N, oj.MAX_PLANETS, 3), np.float32)
            for env_i, st in enumerate(env_states):
                for slot in range(oj.MAX_OBS_PLANETS):
                    if (not np.array(st.p_alive)[slot] or
                            np.array(st.p_owner)[slot] != 1 or
                            np.array(st.p_ships)[slot] < MIN_SHIPS):
                        continue
                    sx, sy = float(st.p_x[slot]), float(st.p_y[slot])
                    best_d, best_tgt = float("inf"), -1
                    for tsl in range(oj.MAX_PLANETS):
                        if not np.array(st.p_alive)[tsl] or np.array(st.p_owner)[tsl] == 1:
                            continue
                        d = math.hypot(float(st.p_x[tsl])-sx, float(st.p_y[tsl])-sy)
                        if d < best_d:
                            best_d, best_tgt = d, tsl
                    if best_tgt >= 0:
                        tx, ty = float(st.p_x[best_tgt]), float(st.p_y[best_tgt])
                        ang = math.atan2(ty-sy, tx-sx)
                        n_sh = max(1., float(st.p_ships[slot]) * 0.5)
                        actions_p1[env_i, slot, 0] = ang
                        actions_p1[env_i, slot, 1] = n_sh
                        actions_p1[env_i, slot, 2] = 1.0

            # Step envs
            prev_ships = [
                (float(jnp.sum(jnp.where(st.p_owner==0, st.p_ships*st.p_alive, 0.))) +
                 float(jnp.sum(jnp.where(st.f_owner==0, st.f_ships*st.f_alive, 0.))),
                 float(jnp.sum(jnp.where(st.p_owner==1, st.p_ships*st.p_alive, 0.))) +
                 float(jnp.sum(jnp.where(st.f_owner==1, st.f_ships*st.f_alive, 0.))))
                for st in env_states
            ]

            new_states = []
            step_rewards = []
            step_dones   = []
            for env_i, st in enumerate(env_states):
                a0 = jnp.array(np.array(actions_p0)[env_i])
                a1 = jnp.array(actions_p1[env_i])
                ns, _, _, term_rew, done = oj.step_jit(st, a0, a1)
                new_states.append(ns)
                step_dones.append(bool(done))

                my0, en0 = prev_ships[env_i]
                my1 = float(jnp.sum(jnp.where(ns.p_owner==0, ns.p_ships*ns.p_alive, 0.))) + \
                      float(jnp.sum(jnp.where(ns.f_owner==0, ns.f_ships*ns.f_alive, 0.)))
                en1 = float(jnp.sum(jnp.where(ns.p_owner==1, ns.p_ships*ns.p_alive, 0.))) + \
                      float(jnp.sum(jnp.where(ns.f_owner==1, ns.f_ships*ns.f_alive, 0.)))
                delta = (my1 - en1) - (my0 - en0)
                r = delta * cfg["reward_scale"]
                if bool(done):
                    r += float(term_rew) * cfg["terminal_bonus"]
                    ep_count += 1
                    key, rk = jax.random.split(key)
                    pool_idx = int(jax.random.randint(rk, (), 0, len(pool)))
                    new_states[-1] = pool[pool_idx]   # reset from pool
                step_rewards.append(r)

            env_states = new_states
            total_steps += N

            # Store experience from aux
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
                        "rew":    step_rewards[env_i],
                        "done":   float(step_dones[env_i]),
                        "env_i":  env_i,
                        "step_i": step_i,
                    })

        if not buffer:
            continue

        # Bootstrap vals for non-terminal envs
        for env_i, st in enumerate(env_states):
            if not step_dones[env_i]:
                obs_v = np.array(oj.encode_obs(st, 0))[None]
                pf_v  = np.zeros((1, oj.PLANET_FEATS), np.float32)
                mk_v  = np.ones((1, oj.MAX_OBS_PLANETS), bool)
                _, _, _, boot_v = model.apply(
                    train_state.params,
                    jnp.array(obs_v), jnp.array(pf_v), jnp.array(mk_v),
                )
                # Assign boot_val to this env's last buffer entries
                for e in reversed(buffer):
                    if e["env_i"] == env_i:
                        e["boot_val"] = float(boot_v[0])
                        break

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

        if update % 10 == 0:
            print(f"U{update:5d} | S:{total_steps:>10,d} | L:{loss:.4f} "
                  f"| EV:{ev:.3f} | Ent:{ent:.3f} | EP:{ep_count} | SPS:{sps:.0f} "
                  f"| {(time.time()-t0)/3600:.1f}h")

        if update % cfg["save_every"] == 0:
            import pickle
            with open(os.path.join(run_dir, "latest.pkl"), "wb") as f:
                pickle.dump({"params": train_state.params, "update": update,
                             "steps": total_steps, "best_wr": best_wr}, f)

    print(f"\nDone. Total steps: {total_steps:,d}")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="PureJAX PPO for Orbit Wars")
    p.add_argument("--num_envs",      type=int,   default=None)
    p.add_argument("--total_updates", type=int,   default=None)
    p.add_argument("--lr",            type=float, default=None)
    p.add_argument("--run_name",      type=str,   default="default")
    p.add_argument("--pool_size",     type=int,   default=None)
    args = p.parse_args()

    if args.num_envs      is not None: CFG["num_envs"]      = args.num_envs
    if args.total_updates is not None: CFG["total_updates"]  = args.total_updates
    if args.lr            is not None: CFG["lr"]             = args.lr
    if args.pool_size     is not None: CFG["pool_size"]      = args.pool_size

    run_dir = os.path.join(HERE, "runs", args.run_name)
    train(CFG, run_dir)
