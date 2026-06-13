# =============================================================================
# Orbit Wars bot: graceful_sloth_v1
# Baseline heuristic: sun avoidance, iterative lead targeting, threat reserve, greedy source->target assignment. ~67% vs starter.
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
_prev_angles = {}    # planet_id → last observed angle (radians)
_rotation_sign = 1   # +1 (CCW) or -1 (CW), inferred from successive turns


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


def segment_hits_sun(p0, p1, margin=1.5):
    """True if the straight path p0→p1 passes within SUN_RADIUS+margin of the sun."""
    dx, dy = p1[0] - p0[0], p1[1] - p0[1]
    l2 = dx * dx + dy * dy
    if l2 < 1e-12:
        return math.hypot(p0[0] - _CX, p0[1] - _CY) < _SUN_R + margin
    t = max(0.0, min(1.0, ((_CX - p0[0]) * dx + (_CY - p0[1]) * dy) / l2))
    return math.hypot(p0[0] + t * dx - _CX, p0[1] + t * dy - _CY) < _SUN_R + margin


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
    Returns (aim_angle_radians, eta_turns) or None if the comet expires first.
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

    return (math.atan2(aim[1] - src_pos[1], aim[0] - src_pos[0]), t_int)


def agent(obs, config):
    """
    Graceful Sloth v1 — ambitious heuristic agent for Orbit Wars.
    Must be the last callable defined in this file.
    """
    global _prev_angles, _rotation_sign

    # ---- Parse inputs (dict or SimpleNamespace) ----
    max_spd = float(_get(config, "shipSpeed", 6.0))
    player = int(_get(obs, "player", 0))
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

    if _prev_angles and cur_angles:
        deltas = []
        for pid, cur_a in cur_angles.items():
            prev_a = _prev_angles.get(pid)
            if prev_a is None:
                continue
            d = cur_a - prev_a
            d -= 2.0 * math.pi * round(d / (2.0 * math.pi))
            deltas.append(d)
        if deltas:
            avg = sum(deltas) / len(deltas)
            if abs(avg) > 1e-4:
                _rotation_sign = 1 if avg > 0 else -1

    _prev_angles = cur_angles

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

            # Lead targeting
            if is_comet:
                sol = lead_solution(
                    (sx, sy), tid, tx, ty, tr, True,
                    comets, ang_vel, step, spendable_max, max_spd, _rotation_sign,
                )
            elif ip is not None:
                sol = lead_solution(
                    (sx, sy), tid, ip[2], ip[3], tr, False,
                    comets, ang_vel, step, spendable_max, max_spd, _rotation_sign,
                )
            else:
                # ip missing (shouldn't happen in normal play); aim at current pos
                dist = math.hypot(tx - sx, ty - sy)
                eta = max(1, int(dist / fleet_speed(spendable_max, max_spd)))
                sol = (math.atan2(ty - sy, tx - sx), eta)

            if sol is None:
                continue

            angle, eta = sol

            # Verify the actual launch direction also clears the sun
            aim_x = sx + math.cos(angle) * 150.0
            aim_y = sy + math.sin(angle) * 150.0
            if segment_hits_sun((sx, sy), (aim_x, aim_y)):
                continue

            # Step 1: Required ships
            if towner == -1:
                required = tships + MARGIN
            else:
                # Enemy garrison grows while fleet is in transit
                required = tships + tprod * eta + MARGIN

            ships_send = min(spendable_max, required)
            if ships_send < 1:
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
