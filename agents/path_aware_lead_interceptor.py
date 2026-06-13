# =============================================================================
# Orbit Wars bot: path_aware_lead_interceptor
# Builds on greedy_lead_interceptor with engine-grounded targeting fixes:
# intervening-planet path-blocking, a segment-correct sun check, and a
# speed-accurate two-pass lead solve. ~75-80% vs starter.
#
# Self-contained: agent() is the last callable, so kaggle-environments can run
# this file directly, and arena.py can promote it to main.py for submission.
# =============================================================================

import math

# ---- Game constants (match engine defaults) ----
_BOARD = 100.0
_CX = _CY = 50.0     # sun centre
_SUN_R = 10.0
_ROT_LIM = 50.0      # orbital_radius + planet_radius < this → planet orbits

# ---- Module-level state (persists across turns in the same process) ----
# Keyed by player ID so self-play (both sides sharing one module) stays isolated.
_prev_angles = {}    # player_id → {planet_id → angle}
_rotation_sign = {}  # player_id → +1 or -1


def _get(obj, key, default=None):
    """Read from dict or SimpleNamespace/Struct."""
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
    """Speed formula from the engine: 1 ship → 1.0/turn, 1000 ships → max_spd."""
    n = max(int(ships), 1)
    return min(1.0 + (max_spd - 1.0) * (math.log(n) / math.log(1000)) ** 1.5, max_spd)


def segment_hits_sun(p0, p1, margin=1.0):
    """True if the straight path p0→p1 passes within SUN_RADIUS+margin of the sun.

    The engine destroys a fleet whose per-tick movement segment comes within
    SUN_RADIUS of the centre (orbit_wars.py: point_to_segment_distance < SUN_RADIUS).
    Since a fleet is absorbed by the target planet on arrival, only the segment
    *up to the target* matters — checking a long ray past the target would
    falsely reject valid shots that merely point toward the sun.
    """
    dx, dy = p1[0] - p0[0], p1[1] - p0[1]
    l2 = dx * dx + dy * dy
    if l2 < 1e-12:
        return math.hypot(p0[0] - _CX, p0[1] - _CY) < _SUN_R + margin
    t = max(0.0, min(1.0, ((_CX - p0[0]) * dx + (_CY - p0[1]) * dy) / l2))
    return math.hypot(p0[0] + t * dx - _CX, p0[1] + t * dy - _CY) < _SUN_R + margin


def path_blocked_by_planet(src, aim, blockers, exclude_ids, buffer=0.5):
    """True if some planet body lies on the segment src→aim *before* the target.

    The engine absorbs a fleet into the first planet its swept path intersects
    (orbit_wars.py: planets are checked before bounds/sun, breaking on first hit).
    A shot whose straight line passes through an intervening planet therefore
    never reaches its intended target, so we skip that (source, target) pair.
    `blockers` is the planet list; `aim` is the target's (lead) position so any
    planet strictly between source and target counts.
    """
    ax, ay = aim[0] - src[0], aim[1] - src[1]
    seg2 = ax * ax + ay * ay
    if seg2 < 1e-9:
        return False
    for b in blockers:
        if b[0] in exclude_ids:
            continue
        bx, by, br = b[2], b[3], b[4]
        t = ((bx - src[0]) * ax + (by - src[1]) * ay) / seg2
        if t <= 0.0 or t >= 1.0:  # behind source, or at/beyond the target
            continue
        projx, projy = src[0] + t * ax, src[1] + t * ay
        if math.hypot(bx - projx, by - projy) < br + buffer:
            return True
    return False


def predict_planet_pos(init_x, init_y, radius, angular_velocity, abs_step,
                       rotation_sign=1):
    """
    Planet position at absolute game step `abs_step`.
    Orbiting planets revolve about (CX, CY); static ones stay fixed.
    """
    dx, dy = init_x - _CX, init_y - _CY
    r = math.hypot(dx, dy)
    if r + radius < _ROT_LIM:
        ang = math.atan2(dy, dx) + rotation_sign * angular_velocity * abs_step
        return (_CX + r * math.cos(ang), _CY + r * math.sin(ang))
    return (init_x, init_y)


def predict_comet_pos(pid, comets, step_ahead):
    """
    Comet position `step_ahead` turns from now.
    Returns (x, y) or None when the comet expires before then.
    `comets` is the obs.comets list; each group has planet_ids, paths, path_index.
    """
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
    """
    Iterative lead-targeting: find the angle and ETA for src_pos to intercept
    a moving target.  Converges in ~5-8 iterations for typical orbital speeds.
    `ships` is the number actually being sent (speed scales with fleet size, so
    this must match the real launch for the ETA/lead to be accurate).
    Returns (aim_angle_radians, eta_turns, aim_pos) or None if the comet expires.
    """
    # Seed the iteration with the target's current position.
    if is_comet:
        cur = predict_comet_pos(tgt_pid, comets, 0)
        if cur is None:
            cur = (init_x, init_y)
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
            fut = predict_planet_pos(init_x, init_y, tgt_radius, angular_velocity,
                                     current_step + t_int, rotation_sign)
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


def _decide(obs, config):
    """
    Path-Aware Lead Interceptor (was graceful_sloth_v2) — adds intervening-planet
    path-blocking, a segment-correct sun check, and a two-pass speed-accurate lead.
    Core decision logic; wrapped by agent() so a bad turn can't forfeit a game.
    """
    global _prev_angles, _rotation_sign

    # ---- Parse inputs (dict or SimpleNamespace) ----
    max_spd = float(_get(config, "shipSpeed", 6.0))
    player = int(_get(obs, "player", 0))

    # Per-player state (safe for self-play where both sides share this module)
    p_prev_angles = _prev_angles.get(player, {})
    p_rotation_sign = _rotation_sign.get(player, 1)
    planets_raw = _get(obs, "planets", []) or []
    fleets_raw = _get(obs, "fleets", []) or []
    init_raw = _get(obs, "initial_planets", []) or []
    comets = _get(obs, "comets", []) or []
    comet_pids = set(_get(obs, "comet_planet_ids", []) or [])
    ang_vel = float(_get(obs, "angular_velocity", 0.035))
    step = int(_get(obs, "step", 0))

    planets = {p[0]: p for p in planets_raw}
    init_by_id = {p[0]: p for p in init_raw}

    # ---- Infer rotation sign from consecutive observations ----
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

    _prev_angles[player] = cur_angles
    _rotation_sign[player] = p_rotation_sign

    # ---- Classify planets ----
    my_planets = [p for p in planets.values() if p[1] == player]
    if not my_planets:
        return []
    capturable = [p for p in planets.values() if p[1] != player]

    # ---- Step 2: Threat detection → defensive reserve ----
    MARGIN = 5
    threat = {mp[0]: 0 for mp in my_planets}
    for f in fleets_raw:
        if f[1] == player:
            continue
        fx, fy, fangle, fships = f[2], f[3], f[4], f[6]
        for mp in my_planets:
            expected = math.atan2(mp[3] - fy, mp[2] - fx)
            diff = fangle - expected
            diff -= 2.0 * math.pi * round(diff / (2.0 * math.pi))
            if abs(diff) < 0.35:   # ≈ 20° cone
                threat[mp[0]] += fships

    reserve = {mp[0]: max(threat[mp[0]] + MARGIN, MARGIN) for mp in my_planets}

    # ---- Steps 3 & 4: Score every (source, target) pair ----
    candidates = []

    for tgt in capturable:
        tid = tgt[0]
        tx, ty = tgt[2], tgt[3]
        tr = tgt[4]
        tships = tgt[5]
        tprod = tgt[6]
        towner = tgt[1]
        is_comet = tid in comet_pids
        ip = init_by_id.get(tid)

        # Skip comets too close to expiring (not worth the fleet cost)
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

            # Coarse sun-block check using current target position
            if segment_hits_sun((sx, sy), (tx, ty)):
                continue

            def _solve(n_ships):
                """Lead solution for sending n_ships from this source."""
                if is_comet:
                    return lead_solution(
                        (sx, sy), tid, tx, ty, tr, True,
                        comets, ang_vel, step, n_ships, max_spd, p_rotation_sign,
                    )
                if ip is not None:
                    return lead_solution(
                        (sx, sy), tid, ip[2], ip[3], tr, False,
                        comets, ang_vel, step, n_ships, max_spd, p_rotation_sign,
                    )
                # ip missing (shouldn't happen in normal play); aim at current pos
                dist = math.hypot(tx - sx, ty - sy)
                eta_ = max(1, int(dist / fleet_speed(n_ships, max_spd)))
                return (math.atan2(ty - sy, tx - sx), eta_, (tx, ty))

            # Pass 1: rough solve using the max budget to size the requirement.
            sol = _solve(spendable_max)
            if sol is None:
                continue
            angle, eta, aim = sol

            # Step 1: required ships. Enemy garrison grows by production each
            # tick in transit (engine adds production before combat); neutrals
            # don't produce. Survivor must exceed the garrison to flip it.
            if towner == -1:
                required = tships + MARGIN
            else:
                required = tships + tprod * eta + MARGIN
            ships_send = min(spendable_max, required)
            if ships_send < 1:
                continue

            # Pass 2: refine the lead/ETA with the ships we'll actually send,
            # since speed scales with fleet size. Re-cost enemy targets.
            sol = _solve(ships_send)
            if sol is None:
                continue
            angle, eta, aim = sol
            if towner != -1:
                required = tships + tprod * eta + MARGIN
                ships_send = min(spendable_max, required)
                if ships_send < 1:
                    continue

            # Sun must be clear along the actual travel segment (src → target).
            if segment_hits_sun((sx, sy), aim):
                continue

            # An intervening planet would absorb the fleet before the target.
            if path_blocked_by_planet((sx, sy), aim, planets_raw, {sid, tid}):
                continue

            # Step 3: Score = production / (cost + ETA decay)
            score = tprod / (required + 0.3 * eta + 1.0)

            # Early-game bonus: prioritise cheap nearby neutrals to ramp production
            if step < 60 and towner == -1:
                d = math.hypot(tx - sx, ty - sy)
                score *= 1.5 / (1.0 + d * 0.01)

            # Modest comet bonus (free production while they last)
            if is_comet:
                score *= 1.2

            candidates.append((score, sid, tid, angle, eta, ships_send, required))

    # Sort by score descending
    candidates.sort(key=lambda c: -c[0])

    # ---- Greedy assignment (Step 4) ----
    # Track ships committed per source and per target to avoid over-send.
    used = {mp[0]: 0 for mp in my_planets}
    committed = {}   # target pid → ships committed this turn
    moves = []

    for score, sid, tid, angle, eta, ships_send, required in candidates:
        already = committed.get(tid, 0)
        if already >= required:
            continue  # target already fully covered by earlier assignments

        src = planets[sid]
        avail = src[5] - reserve.get(sid, MARGIN) - used.get(sid, 0)
        if avail < 1:
            continue

        still_needed = required - already
        send = min(avail, still_needed)
        if send < 1:
            continue

        moves.append([sid, angle, int(send)])
        used[sid] = used.get(sid, 0) + send
        committed[tid] = already + send

    return moves


def agent(obs, config):
    """Crash-safe entry point (the last callable in this file).

    On Kaggle an unhandled exception or timeout forfeits the game, so never let
    one escape: on any error, log a traceback to stderr (so real bugs stay
    visible in local/Kaggle logs) and return no moves for this turn.
    """
    try:
        return _decide(obs, config)
    except Exception:  # noqa: BLE001 — a bad turn must not forfeit the match
        import sys
        import traceback
        traceback.print_exc(file=sys.stderr)
        return []
