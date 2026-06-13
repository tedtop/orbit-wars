"""
Orbit Wars — lstm_fleet_trajectory_forecaster
=============================================
SURROGATE NOTE (pure stdlib, no torch, no training data)
---------------------------------------------------------
A real LSTM would be trained on game replays to predict each planet's
per-turn ship-count delta. Without training data we implement an
*untrained LSTM surrogate* via two hand-tuned recurrent rules that
mimic what an LSTM would learn from data:

  1. Exponential-smoothed delta (EMA, α=0.35) — acts as the LSTM's
     hidden-state "velocity" estimate for each planet's garrison.
  2. Buildup-then-stall detector — tracks whether a planet accumulated
     ships well beyond organic production, then stalled (EMA delta < 50%
     of production rate). A trained LSTM would encode this staging →
     imminent-launch pattern; we replicate it with explicit thresholds.

When the surrogate flags a planet "about to launch", predicted garrison
at arrival is replaced with a post-launch estimate (small residual +
prod * eta) instead of the standard current_ships + prod * eta. This
lets the agent pre-empt undefended captures at much lower cost.

Fallback: standard production/cost expansion scoring keeps the bot
busy when no pre-emption targets are available.
"""

import math
from collections import deque

_BOARD = 100.0
_CX = _CY = 50.0
_SUN_R = 10.0
_ROT_LIM = 50.0
_EPISODE = 500
_EMA_ALPHA = 0.35  # LSTM-surrogate hidden-state smoothing coefficient

# Per-player state, keyed by player id; reset each time step==0
_prev_angles   = {}  # player_id → {planet_id → angle}
_rotation_sign = {}  # player_id → +1 or -1
_ship_history  = {}  # player_id → {planet_id → deque(maxlen=10)}
_smooth_delta  = {}  # player_id → {planet_id → float}  EMA of per-turn delta


# ---------------------------------------------------------------------------
# Helpers (ported verbatim from comet_wraith_v3)
# ---------------------------------------------------------------------------

def _get(obj, key, default=None):
    if isinstance(obj, dict):
        return obj.get(key, default)
    get_fn = getattr(obj, "get", None)
    if get_fn is not None and callable(get_fn) and not isinstance(obj, type):
        try:
            return get_fn(key, default)
        except TypeError:
            pass
    return getattr(obj, key, default)


def fleet_speed(ships, max_spd=6.0):
    n = max(int(ships), 1)
    return min(1.0 + (max_spd - 1.0) * (math.log(n) / math.log(1000)) ** 1.5,
               max_spd)


def _pt_seg(p, a, b):
    dx, dy = b[0] - a[0], b[1] - a[1]
    l2 = dx * dx + dy * dy
    if l2 < 1e-12:
        return math.hypot(p[0] - a[0], p[1] - a[1])
    t = max(0.0, min(1.0, ((p[0] - a[0]) * dx + (p[1] - a[1]) * dy) / l2))
    return math.hypot(a[0] + t * dx - p[0], a[1] + t * dy - p[1])


def segment_hits_sun(p0, p1, margin=1.0):
    return _pt_seg((_CX, _CY), p0, p1) < _SUN_R + margin


def path_blocked_by_planet(src, aim, blockers, exclude_ids, buffer=0.5):
    ax, ay = aim[0] - src[0], aim[1] - src[1]
    seg2 = ax * ax + ay * ay
    if seg2 < 1e-9:
        return False
    for b in blockers:
        if b[0] in exclude_ids:
            continue
        bx, by, br = b[2], b[3], b[4]
        t = ((bx - src[0]) * ax + (by - src[1]) * ay) / seg2
        if t <= 0.0 or t >= 1.0:
            continue
        projx = src[0] + t * ax
        projy = src[1] + t * ay
        if math.hypot(bx - projx, by - projy) < br + buffer:
            return True
    return False


def predict_planet_pos(init_x, init_y, radius, angular_velocity, abs_step,
                       rotation_sign=1):
    dx, dy = init_x - _CX, init_y - _CY
    r = math.hypot(dx, dy)
    if r + radius < _ROT_LIM:
        ang = (math.atan2(dy, dx) +
               rotation_sign * angular_velocity * abs_step)
        return (_CX + r * math.cos(ang), _CY + r * math.sin(ang))
    return (init_x, init_y)


def predict_comet_pos(pid, comets, step_ahead):
    for grp in comets:
        pids = grp["planet_ids"]
        if pid not in pids:
            continue
        i = pids.index(pid)
        path = grp["paths"][i]
        future_idx = grp["path_index"] + step_ahead
        if 0 <= future_idx < len(path):
            pos = path[future_idx]
            return (pos[0], pos[1])
        return None
    return None


def lead_solution(src_pos, tgt_pid, init_x, init_y, tgt_radius, is_comet,
                  comets, angular_velocity, current_step, ships, max_spd,
                  rotation_sign):
    """Returns (angle, eta, aim_pos) or None."""
    if is_comet:
        cur = predict_comet_pos(tgt_pid, comets, 0) or (init_x, init_y)
    else:
        cur = predict_planet_pos(init_x, init_y, tgt_radius, angular_velocity,
                                 current_step, rotation_sign)
    dist = math.hypot(cur[0] - src_pos[0], cur[1] - src_pos[1])
    t = max(1.0, dist / fleet_speed(ships, max_spd))
    for _ in range(8):
        t_int = max(1, int(round(t)))
        if is_comet:
            fut = predict_comet_pos(tgt_pid, comets, t_int)
            if fut is None:
                return None
        else:
            fut = predict_planet_pos(init_x, init_y, tgt_radius,
                                     angular_velocity, current_step + t_int,
                                     rotation_sign)
        dist = math.hypot(fut[0] - src_pos[0], fut[1] - src_pos[1])
        new_t = max(1.0, dist / fleet_speed(ships, max_spd))
        if abs(new_t - t) < 0.05:
            break
        t = new_t
    t_int = max(1, int(round(t)))
    if is_comet:
        aim = predict_comet_pos(tgt_pid, comets, t_int)
        if aim is None:
            return None
    else:
        aim = predict_planet_pos(init_x, init_y, tgt_radius, angular_velocity,
                                 current_step + t_int, rotation_sign)
    return (math.atan2(aim[1] - src_pos[1], aim[0] - src_pos[0]), t_int, aim)


# ---------------------------------------------------------------------------
# LSTM surrogate: history tracking and garrison forecasting
# ---------------------------------------------------------------------------

def _reset_history(player):
    _ship_history[player] = {}
    _smooth_delta[player] = {}


def _update_history(player, planets_raw):
    ph = _ship_history.setdefault(player, {})
    sd = _smooth_delta.setdefault(player, {})
    for p in planets_raw:
        pid, ships = p[0], p[5]
        if pid not in ph:
            ph[pid] = deque(maxlen=10)
            sd[pid] = 0.0
        hist = ph[pid]
        if hist:
            delta = ships - hist[-1]
            sd[pid] = _EMA_ALPHA * delta + (1.0 - _EMA_ALPHA) * sd[pid]
        hist.append(ships)


def _forecast_garrison(pid, current_ships, prod, eta, player):
    """
    LSTM surrogate forward pass: predict garrison at enemy planet `pid`
    after `eta` turns assuming we dispatch now.

    Detects two patterns:
      - Buildup-then-stall: excess ships accumulated; growth recently stalled →
        predict launch imminent, use post-launch residual + prod*eta.
      - Already-launched: large negative EMA delta + low current garrison →
        normal estimate is already low, still flag as pre-empt opportunity.

    Returns (predicted_garrison, is_preempt_opportunity).
    """
    ph = _ship_history.get(player, {}).get(pid)
    sd = _smooth_delta.get(player, {}).get(pid, 0.0)
    normal = current_ships + prod * eta

    if ph is None or len(ph) < 4:
        return normal, False

    hist = list(ph)
    n = len(hist)

    # Excess ships above what production alone would have added since first obs
    organic_gain = prod * (n - 1)
    excess = (hist[-1] - hist[0]) - organic_gain

    # Strong buildup: well above production and current garrison also elevated
    high_accumulation = (excess > max(prod * 3, 12) and
                         current_ships > prod * 4)

    # Stall: EMA delta dropped well below production rate
    stall = sd < prod * 0.5

    # Already launched: large negative EMA delta with low current garrison
    just_launched = (sd < -prod) and (current_ships < prod * 4)

    if just_launched:
        # current_ships is already the post-launch value; normal estimate is fine
        return normal, True

    if high_accumulation and stall:
        # Predict they'll empty the excess; residual ≈ current_ships - excess
        predicted_residual = max(1, int(current_ships - excess))
        return predicted_residual + prod * eta, True

    return normal, False


# ---------------------------------------------------------------------------
# Main decision logic
# ---------------------------------------------------------------------------

def _decide(obs, config):
    global _prev_angles, _rotation_sign

    max_spd = float(_get(config, "shipSpeed", 6.0))
    episode = int(_get(config, "episodeSteps", _EPISODE))
    player  = int(_get(obs, "player", 0))
    step    = int(_get(obs, "step", 0))

    if step == 0:
        _reset_history(player)
        _prev_angles.pop(player, None)
        _rotation_sign.pop(player, None)

    p_prev_angles   = _prev_angles.get(player, {})
    p_rotation_sign = _rotation_sign.get(player, 1)

    planets_raw = _get(obs, "planets", []) or []
    fleets_raw  = _get(obs, "fleets",  []) or []
    init_raw    = _get(obs, "initial_planets", []) or []
    comets      = _get(obs, "comets", []) or []
    comet_pids  = set(_get(obs, "comet_planet_ids", []) or [])
    ang_vel     = float(_get(obs, "angular_velocity", 0.035))
    turns_left  = max(1, episode - step - 2)

    # Update LSTM surrogate's sliding window
    _update_history(player, planets_raw)

    planets    = {p[0]: p for p in planets_raw}
    init_by_id = {p[0]: p for p in init_raw}

    # ---- Rotation-sign inference (per player, from comet_wraith_v3) ----
    cur_angles = {}
    for pid, p in planets.items():
        if pid in comet_pids:
            continue
        ip = init_by_id.get(pid)
        if ip is None:
            continue
        dx, dy = ip[2] - _CX, ip[3] - _CY
        if math.hypot(dx, dy) + ip[4] < _ROT_LIM:
            cur_angles[pid] = math.atan2(p[3] - _CY, p[2] - _CX)
    if p_prev_angles and cur_angles:
        deltas = []
        for pid, cur_a in cur_angles.items():
            prev_a = p_prev_angles.get(pid)
            if prev_a is None:
                continue
            d = cur_a - prev_a
            d -= 2.0 * math.pi * round(d / (2.0 * math.pi))
            deltas.append(d)
        if deltas:
            avg = sum(deltas) / len(deltas)
            if abs(avg) > 1e-4:
                p_rotation_sign = 1 if avg > 0 else -1
    _prev_angles[player]   = cur_angles
    _rotation_sign[player] = p_rotation_sign

    my_planets = [p for p in planets.values() if p[1] == player]
    if not my_planets:
        return []
    capturable = [p for p in planets.values() if p[1] != player]

    MARGIN = 1

    # ---- Threat → defensive reserve ----
    threat = {mp[0]: 0 for mp in my_planets}
    for f in fleets_raw:
        if f[1] == player:
            continue
        fx, fy, fangle, fships = f[2], f[3], f[4], f[6]
        for mp in my_planets:
            expected = math.atan2(mp[3] - fy, mp[2] - fx)
            diff = fangle - expected
            diff -= 2.0 * math.pi * round(diff / (2.0 * math.pi))
            if abs(diff) < 0.35:
                threat[mp[0]] += fships
    reserve = {mp[0]: max(threat[mp[0]] + MARGIN, MARGIN) for mp in my_planets}

    # ---- Score every (source, target) pair ----
    candidates = []
    for tgt in capturable:
        tid                             = tgt[0]
        tx, ty, tr, tships, tprod, towner = (
            tgt[2], tgt[3], tgt[4], tgt[5], tgt[6], tgt[1])
        is_comet = tid in comet_pids
        ip = init_by_id.get(tid)

        if is_comet:
            remaining = 0
            for grp in comets:
                if tid in grp["planet_ids"]:
                    i = grp["planet_ids"].index(tid)
                    remaining = len(grp["paths"][i]) - grp["path_index"] - 1
                    break
            if remaining < 15:
                continue

        for src in my_planets:
            sid = src[0]
            sx, sy = src[2], src[3]
            spendable_max = src[5] - reserve.get(sid, MARGIN)
            if spendable_max < 1:
                continue
            if segment_hits_sun((sx, sy), (tx, ty)):
                continue

            def _solve(n):
                if is_comet:
                    return lead_solution(
                        (sx, sy), tid, tx, ty, tr, True,
                        comets, ang_vel, step, n, max_spd, p_rotation_sign)
                if ip is not None:
                    return lead_solution(
                        (sx, sy), tid, ip[2], ip[3], tr, False,
                        comets, ang_vel, step, n, max_spd, p_rotation_sign)
                d = math.hypot(tx - sx, ty - sy)
                e = max(1, int(d / fleet_speed(n, max_spd)))
                return (math.atan2(ty - sy, tx - sx), e, (tx, ty))

            sol = _solve(spendable_max)
            if sol is None:
                continue
            angle, eta, aim = sol

            # --- LSTM surrogate: forecast garrison at arrival ---
            is_preempt = False
            if towner == -1:
                garrison = tships  # neutral: no production applied (matches ref)
            else:
                garrison, is_preempt = _forecast_garrison(
                    tid, tships, tprod, eta, player)

            required   = int(garrison) + MARGIN
            ships_send = min(spendable_max, required)
            if ships_send < 1:
                continue

            # Re-solve with actual fleet size (speed depends on ship count)
            sol = _solve(ships_send)
            if sol is None:
                continue
            angle, eta, aim = sol

            # Recompute for enemy targets with updated eta
            if towner != -1:
                garrison, is_preempt = _forecast_garrison(
                    tid, tships, tprod, eta, player)
                required   = int(garrison) + MARGIN
                ships_send = min(spendable_max, required)
                if ships_send < 1:
                    continue

            if segment_hits_sun((sx, sy), aim):
                continue
            if path_blocked_by_planet((sx, sy), aim, planets_raw, {sid, tid}):
                continue

            if eta + 2 >= turns_left:
                continue
            if turns_left < 45:
                mult = 2.0 if towner != -1 else 1.0
                if tprod * (turns_left - eta) * mult <= garrison:
                    continue

            score = tprod / (required + 0.3 * eta + 1.0)

            # Pre-emption bonus: an enemy about to empty is a steal opportunity
            if is_preempt and towner != -1:
                score *= 2.5

            if step < 60 and towner == -1:
                d = math.hypot(tx - sx, ty - sy)
                score *= 1.5 / (1.0 + d * 0.01)
            if is_comet:
                score *= 1.2

            candidates.append((score, sid, tid, angle, eta, ships_send,
                               required, towner == -1))

    candidates.sort(key=lambda c: -c[0])

    # ---- Greedy assignment ----
    used      = {mp[0]: 0 for mp in my_planets}
    committed = {}
    moves     = []
    for score, sid, tid, angle, eta, ships_send, required, _is_neutral in candidates:
        already = committed.get(tid, 0)
        if already >= required:
            continue
        avail = planets[sid][5] - reserve.get(sid, MARGIN) - used[sid]
        if avail < 1:
            continue
        s = min(avail, required - already)
        if s < 1:
            continue
        moves.append([sid, angle, int(s)])
        used[sid]       += s
        committed[tid]   = already + s

    return moves


def agent(obs, config):
    """Crash-safe entry point (last callable)."""
    try:
        return _decide(obs, config)
    except Exception:
        import sys
        import traceback
        traceback.print_exc(file=sys.stderr)
        return []
