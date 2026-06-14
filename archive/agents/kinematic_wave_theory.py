# =============================================================================
# Orbit Wars bot: kinematic_wave_theory
#
# Inspired by the Lighthill-Whitham-Richards (LWR) traffic flow model.
# Instead of building one slow massive armada or leaking random singles:
#
#   Interior planets (far from enemies) dispatch a ~6-ship batch the instant
#   they have enough surplus, streaming a continuous supply flow to frontline
#   hub planets nearest the enemy.  Frontline hubs collect incoming waves and
#   launch coordinated capture strikes when they have enough to take a target.
#
# Engine note: fleet speed ≈ 1-2 units/turn for small fleets (the 6-cap only
# applies near 1000-ship fleets).  '6' is the BATCH SIZE for throughput
# regularity, not a speed target.  Continuous dispatch beats irregular bursts
# because each wave arrives before the opponent finishes building a big one.
#
# Capture scoring reuses coordinated_strike_interceptor's proven prod/cost ROI formula.
# =============================================================================

import math

_BOARD = 100.0
_CX = _CY = 50.0
_SUN_R = 10.0
_ROT_LIM = 50.0
_EPISODE = 500

_prev_angles = {}
_rotation_sign = {}

BATCH_SIZE = 6       # ships per interior supply wave
FRONT_RATIO = 0.35   # fraction of owned planets treated as frontline


# ---------------------------------------------------------------------------
# Helpers (copied from coordinated_strike_interceptor — do not modify)
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


# ---------------------------------------------------------------------------
# LWR-specific helper
# ---------------------------------------------------------------------------

def _route_to_owned(src_pos, hub, init_by_id, ships, ang_vel, step, max_spd,
                     rot_sign, comet_pids, comets):
    """Lead solution from src_pos to an owned hub planet."""
    hid, hx, hy, hr = hub[0], hub[2], hub[3], hub[4]
    is_comet = hid in comet_pids
    ip = init_by_id.get(hid)
    if is_comet:
        return lead_solution(src_pos, hid, hx, hy, hr, True,
                              comets, ang_vel, step, ships, max_spd, rot_sign)
    elif ip is not None:
        return lead_solution(src_pos, hid, ip[2], ip[3], hr, False,
                              comets, ang_vel, step, ships, max_spd, rot_sign)
    else:
        d = math.hypot(hx - src_pos[0], hy - src_pos[1])
        e = max(1, int(d / fleet_speed(ships, max_spd)))
        return (math.atan2(hy - src_pos[1], hx - src_pos[0]), e, (hx, hy))


# ---------------------------------------------------------------------------
# Core decision
# ---------------------------------------------------------------------------

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

    # ---- Rotation-sign inference (per player, from coordinated_strike_interceptor) ----
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

    # ---- Threat-based defensive reserve ----
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

    # ---- Multi-opponent standings ----
    scores_map = {}
    for p in planets.values():
        if p[1] != -1:
            scores_map[p[1]] = scores_map.get(p[1], 0) + p[5]
    for f in fleets_raw:
        scores_map[f[1]] = scores_map.get(f[1], 0) + f[6]
    opp_scores = {o: s for o, s in scores_map.items() if o != player}
    leader = max(opp_scores, key=opp_scores.get) if opp_scores else None
    multi_opp = len(opp_scores) > 1

    # ---- Classify planets: frontline (near enemy) vs interior (far) ----
    enemy_planets = [p for p in planets.values() if p[1] not in (-1, player)]

    if len(my_planets) <= 2 or not enemy_planets:
        # With few owned planets or no visible enemies, everyone is frontline.
        frontline_ids = set(p[0] for p in my_planets)
        interior_ids = set()
    else:
        def _dist_to_nearest_enemy(planet):
            return min(math.hypot(planet[2] - e[2], planet[3] - e[3])
                       for e in enemy_planets)
        # Sort ascending by distance to nearest enemy; closest = frontline.
        sorted_mine = sorted(my_planets, key=_dist_to_nearest_enemy)
        n_front = max(1, math.ceil(len(my_planets) * FRONT_RATIO))
        frontline_ids = set(p[0] for p in sorted_mine[:n_front])
        interior_ids = set(p[0] for p in sorted_mine[n_front:])

    frontline_planets = [p for p in my_planets if p[0] in frontline_ids]

    moves = []
    used = {mp[0]: 0 for mp in my_planets}

    # ---- Phase 1: Interior → frontline supply waves (the LWR supply line) ----
    # Each interior planet dispatches exactly BATCH_SIZE ships the moment it
    # has accumulated that surplus above its reserve.  This creates a steady
    # high-frequency stream rather than one huge slow convoy.
    for src in my_planets:
        sid = src[0]
        if sid not in interior_ids:
            continue
        sx, sy = src[2], src[3]
        avail = src[5] - reserve.get(sid, MARGIN) - used[sid]
        if avail < BATCH_SIZE:
            continue  # wait for the batch to fill — never send a trickle

        best_angle = None
        best_dist = float('inf')

        for hub in frontline_planets:
            hx, hy = hub[2], hub[3]
            sol = _route_to_owned((sx, sy), hub, init_by_id, BATCH_SIZE,
                                   ang_vel, step, max_spd, p_rotation_sign,
                                   comet_pids, comets)
            if sol is None:
                continue
            angle, eta, aim = sol
            if segment_hits_sun((sx, sy), aim):
                continue
            d = math.hypot(hx - sx, hy - sy)
            if d < best_dist:
                best_dist = d
                best_angle = angle

        if best_angle is None:
            continue

        moves.append([sid, best_angle, BATCH_SIZE])
        used[sid] += BATCH_SIZE

    # ---- Phase 2: Capture scoring — all owned planets can strike ----
    # Frontline planets get a score bonus so they are preferred as launch pads.
    # Interior planets that already sent a supply wave contribute whatever
    # remains; those that couldn't send (batch not ready) also participate
    # so nothing sits idle.
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
                    ci = grp["planet_ids"].index(tid)
                    remaining = len(grp["paths"][ci]) - grp["path_index"] - 1
                    break
            if remaining < 15:
                continue

        for src in my_planets:
            sid = src[0]
            sx, sy = src[2], src[3]
            spendable_max = src[5] - reserve.get(sid, MARGIN) - used[sid]
            if spendable_max < 1:
                continue
            if segment_hits_sun((sx, sy), (tx, ty)):
                continue

            # Closure captures loop-specific values via default args.
            def _solve(n, _sx=sx, _sy=sy, _tid=tid, _tx=tx, _ty=ty,
                       _tr=tr, _ic=is_comet, _ip=ip):
                if _ic:
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

            sol = _solve(spendable_max)
            if sol is None:
                continue
            angle, eta, aim = sol

            garrison = tships if towner == -1 else tships + tprod * eta
            required = garrison + MARGIN
            ships_send = min(spendable_max, required)
            if ships_send < 1:
                continue

            # Refine ETA with the actual send count.
            sol = _solve(ships_send)
            if sol is None:
                continue
            angle, eta, aim = sol
            if towner != -1:
                garrison = tships + tprod * eta
                required = garrison + MARGIN
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
            if step < 60 and towner == -1:
                d = math.hypot(tx - sx, ty - sy)
                score *= 1.5 / (1.0 + d * 0.01)
            if is_comet:
                score *= 1.2
            if multi_opp and towner == leader:
                score *= 1.3
            if sid in frontline_ids:
                score *= 1.25  # frontline hub preferred as launch pad

            candidates.append((score, sid, tid, angle, eta, ships_send,
                               required, towner == -1))

    candidates.sort(key=lambda c: -c[0])

    committed = {}
    for score, sid, tid, angle, eta, ships_send, required, _neutral in candidates:
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
    """Crash-safe entry point — the last callable in this file."""
    try:
        return _decide(obs, config)
    except Exception:
        import sys
        import traceback
        traceback.print_exc(file=sys.stderr)
        return []
