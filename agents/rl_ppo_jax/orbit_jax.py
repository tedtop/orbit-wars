"""
Pure-JAX Orbit Wars engine — fully vectorised (no Python loops in step()).

State: NamedTuple of fixed-size JAX arrays (valid pytree → jit/vmap).
reset(seed) → GameState          — Python-side generation, returns JAX state.
step(state, a0, a1) → (state, obs0, obs1, reward, done)  — JIT/vmap target.

Action format: float32 (MAX_PLANETS, 3) per player
  action[slot] = [angle_rad, n_ships, fire_flag]
  fire_flag > 0.5 and p_alive[slot] and p_owner[slot]==player triggers launch.

Comet path_index semantics:
  c_path_idx[gi] = -2  →  never spawned
  c_path_idx[gi] = -1  →  spawned this step, first advance pending
  c_path_idx[gi] >= 0  →  active; planet currently at path[c_path_idx]
  c_path_idx[gi] >= c_path_len[gi]  →  expired (killed at start of next step)
"""

import math
import os, sys
import random as _random
import numpy as np
import jax
import jax.numpy as jnp
from typing import NamedTuple

# ── Constants ──────────────────────────────────────────────────────────────────
BOARD_SIZE     = 100.0
CENTER         = 50.0
SUN_RADIUS     = 10.0
ROT_LIMIT      = 50.0
SHIP_SPEED_MAX = 6.0
COMET_SPAWN_STEPS = [50, 150, 250, 350, 450]
EPISODE_STEPS  = 500

MAX_PLANETS    = 64
MAX_FLEETS     = 1024
N_COMET_GROUPS = 5
N_COMET_QUADS  = 4
MAX_COMET_LEN  = 50

MAX_OBS_PLANETS = 20
MAX_OBS_FLEETS  = 12
PLANET_FEATS    = 10
FLEET_FEATS     = 8
GLOBAL_FEATS    = 6
OBS_DIM = MAX_OBS_PLANETS * PLANET_FEATS + MAX_OBS_FLEETS * FLEET_FEATS + GLOBAL_FEATS

# Steps at which groups 0-4 spawn (cur_step = state.step + 1)
# Python interpreter spawns when (obs.step + 1) in [50,150,250,350,450]; obs.step
# goes 0→1 per call, so cur_step=50 is the right trigger (state.step=49 → cur_step=50).
_SPAWN_AT = jnp.array([50, 150, 250, 350, 450], dtype=jnp.int32)
_SPAWN_AT_NP = np.array([50, 150, 250, 350, 450], dtype=np.int32)


# ── State ──────────────────────────────────────────────────────────────────────

class GameState(NamedTuple):
    step:             jnp.ndarray   # () int32
    done:             jnp.ndarray   # () bool
    angular_velocity: jnp.ndarray   # () float32
    fleet_cursor:     jnp.ndarray   # () int32 — ring-buffer write pointer

    # Planets (MAX_PLANETS,)
    p_alive:      jnp.ndarray   # bool
    p_owner:      jnp.ndarray   # int32  (-1 = neutral)
    p_x:          jnp.ndarray   # float32
    p_y:          jnp.ndarray   # float32
    p_r:          jnp.ndarray   # float32
    p_ships:      jnp.ndarray   # float32
    p_prod:       jnp.ndarray   # float32
    p_orb_r:      jnp.ndarray   # float32  (0 if static)
    p_init_angle: jnp.ndarray   # float32
    p_is_orb:     jnp.ndarray   # bool

    # Fleets (MAX_FLEETS,) — ring buffer
    f_alive: jnp.ndarray   # bool
    f_owner: jnp.ndarray   # int32
    f_x:     jnp.ndarray   # float32
    f_y:     jnp.ndarray   # float32
    f_angle: jnp.ndarray   # float32
    f_ships: jnp.ndarray   # float32

    # Comets
    c_paths:    jnp.ndarray   # (N_COMET_GROUPS, N_COMET_QUADS, MAX_COMET_LEN, 2)
    c_path_len: jnp.ndarray   # (N_COMET_GROUPS,) int32
    c_path_idx: jnp.ndarray   # (N_COMET_GROUPS,) int32
    c_p_slot:   jnp.ndarray   # (N_COMET_GROUPS, N_COMET_QUADS) int32
    c_ships:    jnp.ndarray   # (N_COMET_GROUPS,) float32
    c_valid:    jnp.ndarray   # (N_COMET_GROUPS,) bool


# ── Python-side reset ─────────────────────────────────────────────────────────

def reset(seed: int) -> GameState:
    """Generate a new game using same RNG as kaggle interpreter. Not JIT-able."""
    try:
        from kaggle_environments.envs.orbit_wars.orbit_wars import (  # type: ignore
            generate_planets, generate_comet_paths,
        )
    except ImportError:
        _here = os.path.dirname(os.path.abspath(__file__))
        for _rel in ["../../.venv", "../../../orbit_wars/.venv"]:
            _venv = os.path.normpath(os.path.join(_here, _rel))
            _pkg  = os.path.join(_venv, "lib/python3.11/site-packages")
            if os.path.isdir(_pkg) and _pkg not in sys.path:
                sys.path.insert(0, _pkg)
        from kaggle_environments.envs.orbit_wars.orbit_wars import (  # type: ignore
            generate_planets, generate_comet_paths,
        )

    rng = _random.Random(seed)
    angular_velocity = rng.uniform(0.025, 0.05)
    planets = generate_planets(rng)

    ng = len(planets) // 4
    if ng > 0:
        hg   = rng.randint(0, ng - 1)
        base = hg * 4
        planets[base][1]     = 0;  planets[base][5]     = 10
        planets[base + 3][1] = 1;  planets[base + 3][5] = 10

    n_reg = len(planets)

    def _orb(p):
        dx, dy = p[2]-CENTER, p[3]-CENTER
        r = math.sqrt(dx*dx + dy*dy)
        if (r + p[4]) < ROT_LIMIT:
            return r, math.atan2(dy, dx), True
        return 0., 0., False

    # Pre-compute comet paths
    init_copy = [p[:] for p in planets]
    comet_pids = []
    next_cid   = (max(p[0] for p in planets) + 1) if planets else 0
    comet_groups = []
    for spawn_step in COMET_SPAWN_STEPS:
        crng  = _random.Random(f"orbit_wars-comet-{seed}-{spawn_step}")
        paths = generate_comet_paths(
            init_copy, angular_velocity, spawn_step, comet_pids, 4.0, rng=crng)
        valid = paths is not None
        ships = 0.
        if valid:
            ships = float(min(crng.randint(1,99), crng.randint(1,99),
                              crng.randint(1,99), crng.randint(1,99)))
            for i in range(4):
                pid = next_cid + i
                comet_pids.append(pid)
                init_copy.append([pid, -1, -99., -99., 1., ships, 1])
            next_cid += 4
        comet_groups.append({"valid": valid, "paths": paths, "ships": ships})

    P = MAX_PLANETS
    p_alive      = np.zeros(P, bool)
    p_owner      = np.full(P, -1, np.int32)
    p_x          = np.zeros(P, np.float32)
    p_y          = np.zeros(P, np.float32)
    p_r          = np.zeros(P, np.float32)
    p_ships      = np.zeros(P, np.float32)
    p_prod       = np.zeros(P, np.float32)
    p_orb_r      = np.zeros(P, np.float32)
    p_init_angle = np.zeros(P, np.float32)
    p_is_orb     = np.zeros(P, bool)

    for i, pl in enumerate(planets):
        _, owner, x, y, r, ships, prod = pl[:7]
        orb_r, ang, is_orb = _orb(pl)
        p_alive[i]=True; p_owner[i]=owner
        p_x[i]=x; p_y[i]=y; p_r[i]=r
        p_ships[i]=float(ships); p_prod[i]=float(prod)
        p_orb_r[i]=orb_r; p_init_angle[i]=ang; p_is_orb[i]=is_orb

    c_paths    = np.zeros((N_COMET_GROUPS, N_COMET_QUADS, MAX_COMET_LEN, 2), np.float32)
    c_path_len = np.zeros(N_COMET_GROUPS, np.int32)
    c_path_idx = np.full(N_COMET_GROUPS, -2, np.int32)
    c_p_slot   = np.zeros((N_COMET_GROUPS, N_COMET_QUADS), np.int32)
    c_ships    = np.zeros(N_COMET_GROUPS, np.float32)
    c_valid    = np.zeros(N_COMET_GROUPS, bool)

    for gi, g in enumerate(comet_groups):
        c_valid[gi]=g["valid"]; c_ships[gi]=g["ships"]
        if g["valid"]:
            paths = g["paths"]
            plen  = min(len(paths[0]), MAX_COMET_LEN)
            c_path_len[gi] = plen
            for qi in range(4):
                slot = n_reg + gi*4 + qi
                c_p_slot[gi,qi] = slot
                p_r[slot]=1.; p_prod[slot]=1.; p_ships[slot]=g["ships"]
                for ti in range(plen):
                    c_paths[gi,qi,ti,0]=paths[qi][ti][0]
                    c_paths[gi,qi,ti,1]=paths[qi][ti][1]

    return GameState(
        step=jnp.int32(0), done=jnp.bool_(False),
        angular_velocity=jnp.float32(angular_velocity),
        fleet_cursor=jnp.int32(0),
        p_alive=jnp.array(p_alive), p_owner=jnp.array(p_owner),
        p_x=jnp.array(p_x), p_y=jnp.array(p_y), p_r=jnp.array(p_r),
        p_ships=jnp.array(p_ships), p_prod=jnp.array(p_prod),
        p_orb_r=jnp.array(p_orb_r), p_init_angle=jnp.array(p_init_angle),
        p_is_orb=jnp.array(p_is_orb),
        f_alive=jnp.zeros(MAX_FLEETS, jnp.bool_),
        f_owner=jnp.zeros(MAX_FLEETS, jnp.int32),
        f_x=jnp.zeros(MAX_FLEETS, jnp.float32),
        f_y=jnp.zeros(MAX_FLEETS, jnp.float32),
        f_angle=jnp.zeros(MAX_FLEETS, jnp.float32),
        f_ships=jnp.zeros(MAX_FLEETS, jnp.float32),
        c_paths=jnp.array(c_paths), c_path_len=jnp.array(c_path_len),
        c_path_idx=jnp.array(c_path_idx), c_p_slot=jnp.array(c_p_slot),
        c_ships=jnp.array(c_ships), c_valid=jnp.array(c_valid),
    )


# ── Math helpers ───────────────────────────────────────────────────────────────

def _swept_hit(ax, ay, bx, by, p0x, p0y, p1x, p1y, r):
    """True iff swept segment A→B comes within r of swept segment P0→P1."""
    d0x, d0y = ax-p0x, ay-p0y
    dvx = (bx-ax)-(p1x-p0x);  dvy = (by-ay)-(p1y-p0y)
    a = dvx*dvx + dvy*dvy
    b = 2.*(d0x*dvx + d0y*dvy)
    c = d0x*d0x + d0y*d0y - r*r
    disc = b*b - 4.*a*c
    sq   = jnp.sqrt(jnp.maximum(disc, 0.))
    t1   = (-b-sq)/(2.*a+1e-12);  t2 = (-b+sq)/(2.*a+1e-12)
    moving = (disc>=0.) & (t2>=0.) & (t1<=1.)
    return jnp.where(a<1e-12, c<=0., moving)


def _pt_seg_dist_sq(px, py, vx, vy, wx, wy):
    dx, dy = wx-vx, wy-vy
    l2 = dx*dx+dy*dy
    t  = jnp.clip(((px-vx)*dx+(py-vy)*dy)/(l2+1e-12), 0., 1.)
    ex, ey = px-(vx+t*dx), py-(vy+t*dy)
    return jnp.where(l2<1e-12, (px-vx)**2+(py-vy)**2, ex*ex+ey*ey)


def _fleet_speed(ships):
    return jnp.minimum(
        1.+(SHIP_SPEED_MAX-1.)*(jnp.log(jnp.maximum(ships,1.))/math.log(1000.))**1.5,
        jnp.float32(SHIP_SPEED_MAX))


# ── Observation encoder ────────────────────────────────────────────────────────

def encode_obs(state: GameState, player: int) -> jnp.ndarray:
    """OBS_DIM obs, player-relative. Mirrors train.py:encode_obs()."""
    p = player

    def pfeat(i):
        alive = state.p_alive[i].astype(jnp.float32)
        ow, x, y = state.p_owner[i], state.p_x[i], state.p_y[i]
        r, sh, pr = state.p_r[i], state.p_ships[i], state.p_prod[i]
        dx, dy = x-CENTER, y-CENTER
        return jnp.array([
            x/100., y/100., r/10., sh/200., pr/20.,
            (ow==p).astype(jnp.float32),
            ((ow>=0)&(ow!=p)).astype(jnp.float32),
            (ow==-1).astype(jnp.float32),
            jnp.sqrt(dx*dx+dy*dy)/70.,
            jnp.arctan2(dy,dx)/math.pi,
        ])*alive

    pf = jax.vmap(pfeat)(jnp.arange(MAX_OBS_PLANETS))   # (20,10)

    def ffeat(i):
        alive = state.f_alive[i].astype(jnp.float32)
        ow, x, y = state.f_owner[i], state.f_x[i], state.f_y[i]
        ang, sh = state.f_angle[i], state.f_ships[i]
        dx, dy = x-CENTER, y-CENTER
        return jnp.array([
            x/100., y/100., sh/200., ang/math.pi,
            (ow==p).astype(jnp.float32),
            ((ow>=0)&(ow!=p)).astype(jnp.float32),
            jnp.sqrt(dx*dx+dy*dy)/70.,
            _fleet_speed(sh)/SHIP_SPEED_MAX,
        ])*alive

    ff = jax.vmap(ffeat)(jnp.arange(MAX_OBS_FLEETS))    # (12,8)

    my_ps = jnp.sum(jnp.where((state.p_owner==p)&state.p_alive, state.p_ships, 0.))
    my_fs = jnp.sum(jnp.where((state.f_owner==p)&state.f_alive, state.f_ships, 0.))
    en_ps = jnp.sum(jnp.where((state.p_owner>=0)&(state.p_owner!=p)&state.p_alive, state.p_ships, 0.))
    en_fs = jnp.sum(jnp.where((state.f_owner>=0)&(state.f_owner!=p)&state.f_alive, state.f_ships, 0.))
    my_s, en_s = my_ps+my_fs, en_ps+en_fs
    my_pl = jnp.sum(((state.p_owner==p)&state.p_alive).astype(jnp.float32))
    en_pl = jnp.sum(((state.p_owner>=0)&(state.p_owner!=p)&state.p_alive).astype(jnp.float32))

    gf = jnp.array([
        state.step.astype(jnp.float32)/500.,
        my_s/1000., en_s/1000., my_pl/20., en_pl/20., (my_s-en_s)/1000.,
    ])
    return jnp.concatenate([pf.ravel(), ff.ravel(), gf])


# ── Fully-vectorised step ──────────────────────────────────────────────────────

def step(state: GameState,
         action_p0: jnp.ndarray,
         action_p1: jnp.ndarray):
    """
    Advance one step.  Fully vectorised — no Python loops over state dims.
    Returns (next_state, obs_p0, obs_p1, reward_p0, done).
    reward_p0: +1 win, -1 loss/tie at terminal, 0 during game.
    """
    cur_step   = state.step + jnp.int32(1)
    p_alive    = state.p_alive
    p_owner    = state.p_owner
    p_x        = state.p_x
    p_y        = state.p_y
    p_ships    = state.p_ships
    c_path_idx = state.c_path_idx

    # Flat view of comet planet slots: (20,)
    flat_slots = state.c_p_slot.reshape(-1)                  # (N_COMET_GROUPS*4,)

    # ── Kill comets expired at end of LAST step ────────────────────────────────
    # (c_path_idx >= c_path_len means the group's path ran out)
    expired_last = state.c_valid & (c_path_idx >= state.c_path_len)   # (5,)
    flat_kill    = jnp.repeat(expired_last, N_COMET_QUADS)            # (20,)
    p_alive = p_alive.at[flat_slots].set(
        jnp.where(flat_kill, False, p_alive[flat_slots]))

    # ── Spawn new comet groups ─────────────────────────────────────────────────
    # Condition: cur_step matches each group's spawn step AND never spawned yet.
    # Python advances path_index -1→0 in the SAME step it spawns, so we place
    # comets at path[0] immediately (not -99,-99) and set c_path_idx=0.
    should_spawn = state.c_valid & (c_path_idx == jnp.int32(-2)) & (cur_step == _SPAWN_AT)  # (5,)
    flat_spawn   = jnp.repeat(should_spawn, N_COMET_QUADS)            # (20,)
    # Gather path[0] positions for all groups/quads: (5,4,2)
    spawn_pos    = state.c_paths[:, :, 0, :]                          # (5,4,2)
    flat_spawn_x = spawn_pos[:, :, 0].reshape(-1)                     # (20,)
    flat_spawn_y = spawn_pos[:, :, 1].reshape(-1)                     # (20,)
    flat_c_ships = jnp.repeat(state.c_ships, N_COMET_QUADS)           # (20,)
    p_alive  = p_alive.at[flat_slots].set(
        jnp.where(flat_spawn, True,          p_alive[flat_slots]))
    p_x      = p_x.at[flat_slots].set(
        jnp.where(flat_spawn, flat_spawn_x,  p_x[flat_slots]))
    p_y      = p_y.at[flat_slots].set(
        jnp.where(flat_spawn, flat_spawn_y,  p_y[flat_slots]))
    p_owner  = p_owner.at[flat_slots].set(
        jnp.where(flat_spawn, jnp.int32(-1), p_owner[flat_slots]))
    p_ships  = p_ships.at[flat_slots].set(
        jnp.where(flat_spawn, flat_c_ships,  p_ships[flat_slots]))
    # Set c_path_idx=0 for newly spawning groups (path[0] already placed above)
    c_path_idx = jnp.where(should_spawn, jnp.int32(0), c_path_idx)

    # ── Fleet launch — vectorised per player ───────────────────────────────────
    f_alive  = state.f_alive
    f_owner  = state.f_owner
    f_x      = state.f_x
    f_y      = state.f_y
    f_angle  = state.f_angle
    f_ships  = state.f_ships
    cursor   = state.fleet_cursor

    for player_id, action in ((0, action_p0), (1, action_p1)):
        ang   = action[:, 0]                                # (P,)
        n_sh  = action[:, 1]                                # (P,)
        fire  = (action[:, 2] > .5) & p_alive & (p_owner == player_id) & (p_ships >= n_sh) & (n_sh > 0.)
        fire_i = fire.astype(jnp.int32)
        # Unique write slot per firing planet via prefix-sum
        offsets  = jnp.cumsum(fire_i) - 1                  # (P,) 0-indexed
        ws       = (cursor + offsets) % MAX_FLEETS          # (P,)
        sx = p_x + jnp.cos(ang) * (state.p_r + .1)
        sy = p_y + jnp.sin(ang) * (state.p_r + .1)
        # Conditional scatter (fire=False → write same value back → no-op)
        f_alive = f_alive.at[ws].set(jnp.where(fire, True,         f_alive[ws]))
        f_owner = f_owner.at[ws].set(jnp.where(fire, jnp.int32(player_id), f_owner[ws]))
        f_x     = f_x.at[ws].set(jnp.where(fire, sx,              f_x[ws]))
        f_y     = f_y.at[ws].set(jnp.where(fire, sy,              f_y[ws]))
        f_angle = f_angle.at[ws].set(jnp.where(fire, ang,          f_angle[ws]))
        f_ships = f_ships.at[ws].set(jnp.where(fire, n_sh,         f_ships[ws]))
        p_ships = p_ships - jnp.where(fire, n_sh, 0.)
        cursor  = cursor + jnp.int32(jnp.sum(fire_i))

    # ── Production ────────────────────────────────────────────────────────────
    p_ships = jnp.where((p_owner >= 0) & p_alive, p_ships + state.p_prod, p_ships)

    # ── Planet next positions ─────────────────────────────────────────────────
    # Regular orbiting planets
    new_ang   = state.p_init_angle + state.angular_velocity * cur_step.astype(jnp.float32)
    new_px    = jnp.where(state.p_is_orb & p_alive,
                          CENTER + state.p_orb_r * jnp.cos(new_ang), p_x)
    new_py    = jnp.where(state.p_is_orb & p_alive,
                          CENTER + state.p_orb_r * jnp.sin(new_ang), p_y)

    # Comets: advance c_path_idx and look up new position from pre-computed paths.
    # active_c: group has been spawned (c_path_idx >= 0).
    # already_adv: advance only if spawned AND not the spawn tick (spawn tick
    # already placed planet at path[0]; advancing again would skip path[0]).
    active_c    = state.c_valid & (c_path_idx >= jnp.int32(0))  # (5,)
    already_adv = active_c & ~should_spawn                       # (5,)
    adv_idx     = c_path_idx + jnp.int32(1)                     # (5,)
    valid_adv   = adv_idx < state.c_path_len                    # (5,) bool
    safe_idx    = jnp.clip(adv_idx, 0, MAX_COMET_LEN-1)        # (5,) clipped for gather

    # Gather comet new positions: c_paths[gi, qi, safe_idx[gi], :] for all gi, qi
    # c_paths shape: (5, 4, MAX_COMET_LEN, 2)
    gi_idx   = jnp.arange(N_COMET_GROUPS)[:, None]             # (5,1)
    qi_idx   = jnp.arange(N_COMET_QUADS)[None, :]              # (1,4)
    safe_exp = safe_idx[:, None]                                # (5,1)
    c_new_pos = state.c_paths[gi_idx, qi_idx, safe_exp, :]     # (5,4,2)

    # Mask: advance only for already_adv groups with valid next step and live slot
    comet_slot_alive = p_alive[state.c_p_slot]                 # (5,4) bool
    use_c_new = (already_adv[:, None] & valid_adv[:, None] & comet_slot_alive)  # (5,4)

    old_comet_x = p_x[state.c_p_slot]                          # (5,4)
    old_comet_y = p_y[state.c_p_slot]                          # (5,4)
    c_nx = jnp.where(use_c_new, c_new_pos[:,:,0], old_comet_x)  # (5,4)
    c_ny = jnp.where(use_c_new, c_new_pos[:,:,1], old_comet_y)  # (5,4)

    # Sweep check: False on spawn tick (Python also skips check when old_pos off-board)
    comet_on_board   = old_comet_x >= 0.                         # (5,4)
    comet_check_mask = (active_c[:, None] & comet_on_board & comet_slot_alive
                        & ~should_spawn[:, None])                 # (5,4)

    # Scatter comet new positions and check masks into per-planet arrays
    flat_c_nx    = c_nx.reshape(-1)              # (20,)
    flat_c_ny    = c_ny.reshape(-1)              # (20,)
    flat_check   = comet_check_mask.reshape(-1)  # (20,)

    new_px = new_px.at[flat_slots].set(flat_c_nx)
    new_py = new_py.at[flat_slots].set(flat_c_ny)

    # Build per-planet sweep_check flag
    is_comet_slot = jnp.zeros(MAX_PLANETS, jnp.bool_).at[flat_slots].set(True)
    sweep_check   = jnp.where(is_comet_slot, False, p_alive)
    sweep_check   = sweep_check.at[flat_slots].set(flat_check)

    # ── Fleet movement + continuous collision ─────────────────────────────────
    speeds  = _fleet_speed(f_ships)                          # (F,) — element-wise, no vmap
    fa_f    = f_alive.astype(jnp.float32)
    new_fx  = f_x + jnp.cos(f_angle) * speeds * fa_f
    new_fy  = f_y + jnp.sin(f_angle) * speeds * fa_f

    pi_idx  = jnp.arange(MAX_PLANETS)

    # Vectorized swept collision via broadcasting: (F, P) — avoids nested vmap
    _d0x = f_x[:, None] - p_x[None, :]                          # (F, P)
    _d0y = f_y[:, None] - p_y[None, :]
    _dvx = (new_fx - f_x)[:, None] - (new_px - p_x)[None, :]   # (F, P)
    _dvy = (new_fy - f_y)[:, None] - (new_py - p_y)[None, :]
    _a   = _dvx*_dvx + _dvy*_dvy
    _b   = 2.*(_d0x*_dvx + _d0y*_dvy)
    _c   = _d0x*_d0x + _d0y*_d0y - state.p_r[None, :]**2
    _disc = _b*_b - 4.*_a*_c
    _sq   = jnp.sqrt(jnp.maximum(_disc, 0.))
    _t1   = (-_b - _sq) / (2.*_a + 1e-12)
    _t2   = (-_b + _sq) / (2.*_a + 1e-12)
    _mov  = (_disc >= 0.) & (_t2 >= 0.) & (_t1 <= 1.)
    hit_mat = (jnp.where(_a < 1e-12, _c <= 0., _mov)
               & f_alive[:, None] & p_alive[None, :] & sweep_check[None, :])  # (F, P)

    hit_any   = jnp.any(hit_mat, axis=1)                        # (F,)
    first_hit = jnp.argmin(
        jnp.where(hit_mat, pi_idx[None, :], MAX_PLANETS), axis=1
    ).astype(jnp.int32)                                          # (F,)

    oob      = (~((new_fx>=0.)&(new_fx<=BOARD_SIZE)&(new_fy>=0.)&(new_fy<=BOARD_SIZE))
                & f_alive)
    # Vectorized point-to-segment distance for sun avoidance
    _sv_x = new_fx - f_x;  _sv_y = new_fy - f_y
    _sl2  = _sv_x*_sv_x + _sv_y*_sv_y
    _st   = jnp.clip(((CENTER-f_x)*_sv_x + (CENTER-f_y)*_sv_y) / (_sl2+1e-12), 0., 1.)
    _ex   = CENTER - (f_x + _st*_sv_x)
    _ey   = CENTER - (f_y + _st*_sv_y)
    _d0xf = f_x - CENTER;  _d0yf = f_y - CENTER
    sun_sq  = jnp.where(_sl2<1e-12, _d0xf*_d0xf+_d0yf*_d0yf, _ex*_ex+_ey*_ey)  # (F,)
    hits_sun = (sun_sq < SUN_RADIUS**2) & f_alive

    fleet_gone   = f_alive & (hit_any | oob | hits_sun)
    still_moving = f_alive & ~fleet_gone
    f_x     = jnp.where(still_moving, new_fx, f_x)
    f_y     = jnp.where(still_moving, new_fy, f_y)
    f_alive = still_moving

    # ── Apply planet positions ────────────────────────────────────────────────
    p_x = jnp.where(p_alive, new_px, p_x)
    p_y = jnp.where(p_alive, new_py, p_y)

    # Advance comet path indices (only for groups that are advancing, not spawn tick)
    c_path_idx = jnp.where(already_adv, adv_idx, c_path_idx) # (5,)
    newly_exp  = active_c & (c_path_idx >= state.c_path_len)  # (5,) expired this step
    flat_new_exp = jnp.repeat(newly_exp, N_COMET_QUADS)      # (20,)
    p_alive = p_alive.at[flat_slots].set(
        jnp.where(flat_new_exp, False, p_alive[flat_slots]))

    # ── Combat — vectorised, no vmap ─────────────────────────────────────────
    _cp0     = hit_mat & (f_owner[:, None] == 0)                 # (F, P)
    _cp1     = hit_mat & (f_owner[:, None] == 1)                 # (F, P)
    arr0     = jnp.sum(jnp.where(_cp0, state.f_ships[:, None], 0.), axis=0)  # (P,)
    arr1     = jnp.sum(jnp.where(_cp1, state.f_ships[:, None], 0.), axis=0)  # (P,)
    has      = (arr0 + arr1) > 0.
    top      = jnp.maximum(arr0, arr1)
    sec      = jnp.minimum(arr0, arr1)
    surv     = top - sec
    tie      = arr0 == arr1
    surv_own = jnp.where(arr0 > arr1, jnp.int32(0),
               jnp.where(arr1 > arr0, jnp.int32(1), jnp.int32(-1)))  # (P,)
    applies   = has & (surv > 0.) & ~tie
    reinforce = applies & (p_owner == surv_own)
    attack    = applies & (p_owner != surv_own)
    after     = p_ships - surv
    flip      = after < 0.
    new_sh_all = jnp.where(reinforce, p_ships + surv,
                 jnp.where(attack, jnp.abs(after), p_ships))          # (P,)
    new_ow_all = jnp.where(attack & flip, surv_own, p_owner)          # (P,)
    p_ships = jnp.where(p_alive, new_sh_all, p_ships)
    p_owner = jnp.where(p_alive, new_ow_all, p_owner)

    # ── Terminal ──────────────────────────────────────────────────────────────
    p0h = jnp.any((p_owner==0)&p_alive) | jnp.any((f_owner==0)&f_alive)
    p1h = jnp.any((p_owner==1)&p_alive) | jnp.any((f_owner==1)&f_alive)
    elim   = (p0h.astype(jnp.int32)+p1h.astype(jnp.int32)) <= 1
    slimit = cur_step >= jnp.int32(EPISODE_STEPS-1)   # mirrors Python: step >= episodeSteps-2, obs.step starts at 0 and advances AFTER interpreter
    done   = elim | slimit | state.done

    sc0 = jnp.sum(jnp.where((p_owner==0)&p_alive,p_ships,0.))+jnp.sum(jnp.where((f_owner==0)&f_alive,state.f_ships,0.))
    sc1 = jnp.sum(jnp.where((p_owner==1)&p_alive,p_ships,0.))+jnp.sum(jnp.where((f_owner==1)&f_alive,state.f_ships,0.))
    win0   = done & (sc0>sc1) & (sc0>0.)
    win1   = done & (sc1>sc0) & (sc1>0.)
    reward = jnp.where(win0, 1., jnp.where(win1, -1., jnp.where(done, -1., 0.))).astype(jnp.float32)

    ns = GameState(
        step=state.step+jnp.int32(1), done=done,
        angular_velocity=state.angular_velocity, fleet_cursor=cursor,
        p_alive=p_alive, p_owner=p_owner,
        p_x=p_x, p_y=p_y, p_r=state.p_r,
        p_ships=p_ships, p_prod=state.p_prod,
        p_orb_r=state.p_orb_r, p_init_angle=state.p_init_angle, p_is_orb=state.p_is_orb,
        f_alive=f_alive, f_owner=f_owner,
        f_x=f_x, f_y=f_y, f_angle=f_angle, f_ships=f_ships,
        c_paths=state.c_paths, c_path_len=state.c_path_len,
        c_path_idx=c_path_idx, c_p_slot=state.c_p_slot,
        c_ships=state.c_ships, c_valid=state.c_valid,
    )
    obs0 = encode_obs(ns, 0)
    obs1 = encode_obs(ns, 1)
    return ns, obs0, obs1, reward, done


step_jit = jax.jit(step)


# ── Validation helpers ────────────────────────────────────────────────────────

def state_to_planet_list(state: GameState):
    """Returns list of dicts for all alive planet slots."""
    out = []
    alive = np.array(state.p_alive)
    for i in range(MAX_PLANETS):
        if alive[i]:
            out.append(dict(
                slot=i, owner=int(state.p_owner[i]),
                x=float(state.p_x[i]), y=float(state.p_y[i]),
                ships=float(state.p_ships[i]),
            ))
    return out


def launches_to_action(launches, slot_for_id: dict) -> np.ndarray:
    """[[from_id, angle, n_ships], ...] → (MAX_PLANETS, 3) action array."""
    action = np.zeros((MAX_PLANETS, 3), np.float32)
    for from_id, angle, n_ships in launches:
        slot = slot_for_id.get(int(from_id))
        if slot is not None:
            action[slot, 0] = float(angle)
            action[slot, 1] = float(n_ships)
            action[slot, 2] = 1.0
    return action
