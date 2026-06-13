# =============================================================================
# Orbit Wars bot: frontline_consolidation  ("The Logistic Network")
#
# Ships flow from interior planets toward convex-hull "Frontline" hubs, where
# they feed real capture attacks.
#
# Algorithm per turn:
#   1. Convex hull (Andrew's monotone chain) of all owned planet coords.
#      Hull planets = "Frontline"; those inside = "Interior".
#      With <3 owned planets every planet is Frontline.
#   2. Interior: send spare ships (keep max(threat+1, 5) garrison) toward the
#      nearest Frontline planet via lead_solution aiming.
#   3. Frontline: run comet_wraith_v3-style capture scoring — score every
#      (source, target) pair by production/cost, greedy-assign to best targets.
#
# Engine grounding:
#   - lead_solution handles orbiting planets (iterative convergence).
#   - Garrison-at-arrival: send strictly > tships + tprod*ETA to capture.
#   - segment_hits_sun gates every candidate move.
#   - Crash-safe: agent() returns [] on any exception.
# =============================================================================

import math

_BOARD = 100.0
_CX = _CY = 50.0
_SUN_R = 10.0
_ROT_LIM = 50.0
_EPISODE = 500

_prev_angles = {}    # player_id → {planet_id → angle}
_rotation_sign = {}  # player_id → +1 or -1


# --- helpers (copied from comet_wraith_v3) -----------------------------------

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
    return min(1.0 + (max_spd - 1.0) * (math.log(n) / math.log(1000)) ** 1.5, max_spd)


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
        projx, projy = src[0] + t * ax, src[1] + t * ay
        if math.hypot(bx - projx, by - projy) < br + buffer:
            return True
    return False


def predict_planet_pos(init_x, init_y, radius, angular_velocity, abs_step,
                       rotation_sign=1):
    dx, dy = init_x - _CX, init_y - _CY
    r = math.hypot(dx, dy)
    if r + radius < _ROT_LIM:
        ang = math.atan2(dy, dx) + rotation_sign * angular_velocity * abs_step
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
    """Returns (angle, eta, aim_pos) or None. ships = what's actually sent."""
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


# --- convex hull (Andrew's monotone chain) -----------------------------------

def _cross(O, A, B):
    return (A[0] - O[0]) * (B[1] - O[1]) - (A[1] - O[1]) * (B[0] - O[0])


def convex_hull(points):
    """CCW convex hull of (x, y) pairs. Returns hull vertices."""
    pts = sorted(set(points))
    n = len(pts)
    if n <= 2:
        return list(pts)
    lower = []
    for p in pts:
        while len(lower) >= 2 and _cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper = []
    for p in reversed(pts):
        while len(upper) >= 2 and _cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return lower[:-1] + upper[:-1]


def hull_planet_ids(my_planets):
    """Return set of planet ids on the convex hull. <3 planets → all frontline."""
    if len(my_planets) < 3:
        return {p[0] for p in my_planets}
    pts = [(p[2], p[3]) for p in my_planets]
    hull_pts = set(convex_hull(pts))
    ids = {p[0] for p in my_planets if (p[2], p[3]) in hull_pts}
    return ids if ids else {my_planets[0][0]}


# --- main logic --------------------------------------------------------------

def _decide(obs, config):
    global _prev_angles, _rotation_sign

    max_spd = float(_get(config, "shipSpeed", 6.0))
    episode = int(_get(config, "episodeSteps", _EPISODE))
    player = int(_get(obs, "player", 0))

    p_prev_angles = _prev_angles.get(player, {})
    p_rotation_sign = _rotation_sign.get(player, 1)
    planets_raw = _get(obs, "planets", []) or []
    fleets_raw = _get(obs, "fleets", []) or []
    init_raw = _get(obs, "initial_planets", []) or []
    comets = _get(obs, "comets", []) or []
    comet_pids = set(_get(obs, "comet_planet_ids", []) or [])
    ang_vel = float(_get(obs, "angular_velocity", 0.035))
    step = int(_get(obs, "step", 0))
    turns_left = max(1, episode - step - 2)

    planets = {p[0]: p for p in planets_raw}
    init_by_id = {p[0]: p for p in init_raw}

    # ---- Rotation-sign inference (per player, same as comet_wraith_v3) ----
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

    my_planets = [p for p in planets.values() if p[1] == player]
    if not my_planets:
        return []
    capturable = [p for p in planets.values() if p[1] != player]

    MARGIN = 1
    INTERIOR_GARRISON = 5   # interior planets keep at least this many ships

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

    # ---- Convex hull classification ----
    frontline_ids = hull_planet_ids(my_planets)
    frontline_planets = [p for p in my_planets if p[0] in frontline_ids]
    interior_planets = [p for p in my_planets if p[0] not in frontline_ids]

    # Opponent scores for capture scoring (leader focus in 4p)
    scores_by_player = {}
    for p in planets.values():
        if p[1] != -1:
            scores_by_player[p[1]] = scores_by_player.get(p[1], 0) + p[5]
    for f in fleets_raw:
        scores_by_player[f[1]] = scores_by_player.get(f[1], 0) + f[6]
    opp_scores = {o: s for o, s in scores_by_player.items() if o != player}
    leader = max(opp_scores, key=opp_scores.get) if opp_scores else None
    multi_opp = len(opp_scores) > 1

    moves = []
    used = {mp[0]: 0 for mp in my_planets}

    # ========== Phase 1: Interior → nearest Frontline ==========
    for src in interior_planets:
        sid = src[0]
        sx, sy = src[2], src[3]
        min_keep = max(reserve.get(sid, MARGIN), INTERIOR_GARRISON)
        surplus = src[5] - min_keep - used[sid]
        if surplus < 1:
            continue

        # Nearest frontline by current observed distance
        best_fl = None
        best_dist = float("inf")
        for fl in frontline_planets:
            d = math.hypot(fl[2] - sx, fl[3] - sy)
            if d < best_dist:
                best_dist = d
                best_fl = fl
        if best_fl is None:
            continue

        fid = best_fl[0]
        ip = init_by_id.get(fid)
        if ip is not None:
            flt_x, flt_y, flt_r = ip[2], ip[3], ip[4]
        else:
            flt_x, flt_y, flt_r = best_fl[2], best_fl[3], best_fl[4]

        if segment_hits_sun((sx, sy), (best_fl[2], best_fl[3])):
            continue

        sol = lead_solution((sx, sy), fid, flt_x, flt_y, flt_r, False,
                            comets, ang_vel, step, surplus, max_spd, p_rotation_sign)
        if sol is None:
            continue
        angle, eta, aim = sol
        if segment_hits_sun((sx, sy), aim):
            continue

        ships_to_send = src[5] - min_keep - used[sid]
        if ships_to_send < 1:
            continue
        moves.append([sid, angle, int(ships_to_send)])
        used[sid] += ships_to_send

    # ========== Phase 2: Frontline → capture targets ==========
    candidates = []
    for tgt in capturable:
        tid = tgt[0]
        tx, ty, tr, tships, tprod, towner = (tgt[2], tgt[3], tgt[4], tgt[5],
                                             tgt[6], tgt[1])
        is_comet = tid in comet_pids
        ip_tgt = init_by_id.get(tid)

        if is_comet:
            remaining = 0
            for grp in comets:
                if tid in grp["planet_ids"]:
                    i = grp["planet_ids"].index(tid)
                    remaining = len(grp["paths"][i]) - grp["path_index"] - 1
                    break
            if remaining < 15:
                continue

        for src in frontline_planets:
            sid = src[0]
            sx, sy = src[2], src[3]

            if segment_hits_sun((sx, sy), (tx, ty)):
                continue

            # Capture default args bind loop-local values for each closure
            def _solve(n, _sx=sx, _sy=sy, _tid=tid, _tx=tx, _ty=ty,
                       _tr=tr, _is_comet=is_comet, _ip=ip_tgt):
                if _is_comet:
                    return lead_solution((_sx, _sy), _tid, _tx, _ty, _tr, True,
                                         comets, ang_vel, step, n, max_spd,
                                         p_rotation_sign)
                if _ip is not None:
                    return lead_solution((_sx, _sy), _tid, _ip[2], _ip[3], _tr,
                                         False, comets, ang_vel, step, n, max_spd,
                                         p_rotation_sign)
                d = math.hypot(_tx - _sx, _ty - _sy)
                e = max(1, int(d / fleet_speed(n, max_spd)))
                return (math.atan2(_ty - _sy, _tx - _sx), e, (_tx, _ty))

            avail_ships = src[5] - reserve.get(sid, MARGIN) - used[sid]
            if avail_ships < 1:
                continue

            sol = _solve(avail_ships)
            if sol is None:
                continue
            angle, eta, aim = sol

            garrison = tships if towner == -1 else tships + tprod * eta
            required = garrison + MARGIN
            ships_send = min(avail_ships, required)
            if ships_send < 1:
                continue

            # Re-solve with actual send count (speed changes with ship count)
            sol = _solve(ships_send)
            if sol is None:
                continue
            angle, eta, aim = sol
            if towner != -1:
                garrison = tships + tprod * eta
                required = garrison + MARGIN
                ships_send = min(avail_ships, required)
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
            if step < 60 and towner == -1:
                d = math.hypot(tx - sx, ty - sy)
                score *= 1.5 / (1.0 + d * 0.01)
            if is_comet:
                score *= 1.2
            if multi_opp and towner == leader:
                score *= 1.3

            candidates.append((score, sid, tid, angle, eta, ships_send, required,
                               towner == -1))

    candidates.sort(key=lambda c: -c[0])

    committed = {}
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
        used[sid] += s
        committed[tid] = already + s

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
