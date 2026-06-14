# =============================================================================
# Orbit Wars bot: predictive_kinematic_interceptor
#
# Fixes the Nearest Planet Sniper by computing exact orbital mechanics and
# travel times. Key idea: fleet speed scales with ship count, so the number
# of ships to send and the ETA are coupled. We iterate to the fixed point:
#
#   required*(n) = garrison(eta(n)) + 1   where n = required*
#
# Algorithm per (source, target) pair:
#   1. Solve with spendable_max → minimum possible ETA → lower bound on required
#   2. If lower-bound required > spendable_max: skip (can't guarantee capture)
#   3. Iterate: re-solve with `required` ships (slower fleet = longer ETA =
#      more garrison = more required). Repeat until stable (3-4 rounds typical).
#   4. Final required = exact ships that arrive with garrison_T + 1.
#
# Strict send-exactly-enough discipline: no partial waves (partial arrivals let
# the garrison regenerate between them). Skip any path crossing the Sun.
# Never-idle fallback for early game when nothing is affordable yet.
# =============================================================================

import math

_BOARD = 100.0
_CX = _CY = 50.0
_SUN_R = 10.0
_ROT_LIM = 50.0
_EPISODE = 500

_prev_angles = {}    # player_id → {planet_id → angle}
_rotation_sign = {}  # player_id → +1 or -1


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
    """Returns (angle, eta, aim_pos) or None. `ships` = what's actually sent."""
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

    # ---- Rotation-sign inference (per player) ----
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

    # ---- Defensive reserve: hold enough to repel tracked incoming threats ----
    MARGIN = 1
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

    # Standings for 4p focus
    scores = {}
    for p in planets.values():
        if p[1] != -1:
            scores[p[1]] = scores.get(p[1], 0) + p[5]
    for f in fleets_raw:
        scores[f[1]] = scores.get(f[1], 0) + f[6]
    opp_scores = {o: s for o, s in scores.items() if o != player}
    leader = max(opp_scores, key=opp_scores.get) if opp_scores else None
    multi_opp = len(opp_scores) > 1

    # ---- Score every (source, target) pair with exact kinematic solve ----
    candidates = []
    for tgt in capturable:
        tid = tgt[0]
        tx, ty, tr, tships, tprod, towner = (tgt[2], tgt[3], tgt[4], tgt[5],
                                              tgt[6], tgt[1])
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

            # Fast pre-filter: current-position sun check
            if segment_hits_sun((sx, sy), (tx, ty)):
                continue

            def _solve(n, _sx=sx, _sy=sy, _tx=tx, _ty=ty, _tr=tr,
                       _is_comet=is_comet, _ip=ip, _tid=tid):
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

            # Phase 1: fastest possible fleet → minimum ETA → lower bound on required
            sol = _solve(spendable_max)
            if sol is None:
                continue
            _, eta0, _ = sol
            garrison_t = tships if towner == -1 else tships + tprod * eta0
            required = int(garrison_t) + MARGIN

            # If even the fastest fleet can't afford capture, skip
            if required > spendable_max:
                continue

            # Phase 2: iterate to fixed point — slower fleet = longer ETA = more
            # garrison = more required. Typically converges in 2-3 rounds.
            angle, eta, aim = sol
            for _ in range(4):
                sol2 = _solve(required)
                if sol2 is None:
                    required = -1
                    break
                angle2, eta2, aim2 = sol2
                garrison_t = tships if towner == -1 else tships + tprod * eta2
                new_required = int(garrison_t) + MARGIN
                if new_required > spendable_max:
                    required = -1
                    break
                angle, eta, aim = angle2, eta2, aim2
                if new_required == required:
                    break  # fixed point reached
                required = new_required

            if required < 1:
                continue

            # Safety checks on the converged aim point
            if segment_hits_sun((sx, sy), aim):
                continue
            if path_blocked_by_planet((sx, sy), aim, planets_raw, {sid, tid}):
                continue

            # End-game gate: fleet must land before the episode ends
            if eta + 2 >= turns_left:
                continue
            if turns_left < 45:
                mult = 2.0 if towner != -1 else 1.0
                if tprod * (turns_left - eta) * mult <= garrison_t:
                    continue

            # Score: production ROI per ship-turn invested
            score = tprod / (required + 0.3 * eta + 1.0)
            if step < 60 and towner == -1:
                d = math.hypot(tx - sx, ty - sy)
                score *= 1.5 / (1.0 + d * 0.01)
            if is_comet:
                score *= 1.2
            if multi_opp and towner == leader:
                score *= 1.3

            candidates.append((score, sid, tid, angle, eta, required, garrison_t,
                               towner == -1))

    candidates.sort(key=lambda c: -c[0])

    # Greedy assignment: track what each source has spent and what each target has
    used = {mp[0]: 0 for mp in my_planets}
    committed = {}
    moves = []
    for score, sid, tid, angle, eta, required, garrison_t, is_neutral in candidates:
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

    # ---- Never-idle fallback: if strict discipline left us with nothing to do,
    # find the cheapest reachable target and send whatever we can afford ----
    if not moves:
        best = None
        best_score = -1.0
        for tgt in capturable:
            tid = tgt[0]
            if tid in comet_pids:
                continue
            tx, ty, tr, tships, tprod = tgt[2], tgt[3], tgt[4], tgt[5], tgt[6]
            ip = init_by_id.get(tid)
            for src in my_planets:
                sid = src[0]
                sx, sy = src[2], src[3]
                avail = src[5] - reserve.get(sid, MARGIN)
                if avail < 1:
                    continue
                if segment_hits_sun((sx, sy), (tx, ty)):
                    continue
                if ip is not None:
                    sol = lead_solution((sx, sy), tid, ip[2], ip[3], tr, False,
                                       comets, ang_vel, step, avail, max_spd,
                                       p_rotation_sign)
                else:
                    d = math.hypot(tx - sx, ty - sy)
                    angle_fb = math.atan2(ty - sy, tx - sx)
                    sol = (angle_fb, max(1, int(d / fleet_speed(avail, max_spd))), (tx, ty))
                if sol is None:
                    continue
                angle_fb, eta_fb, aim_fb = sol
                if segment_hits_sun((sx, sy), aim_fb):
                    continue
                if eta_fb + 2 >= turns_left:
                    continue
                score = tprod / (tships + eta_fb + 1.0)
                if score > best_score:
                    best_score = score
                    send = min(avail, max(1, tships + MARGIN))
                    best = [sid, angle_fb, int(send)]
        if best:
            moves.append(best)

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
