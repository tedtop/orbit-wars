# =============================================================================
# Orbit Wars bot: the_vulture
# Second-Mover Advantage — let enemies do the heavy lifting, then swoop in.
#
# Every turn:
#   1. Scan all enemy fleets in flight; infer each fleet's target planet by
#      smallest angular error between the fleet's heading and the bearing to
#      each candidate planet (same threat-cone heuristic as comet_wraith_v3).
#   2. Simulate the incoming battle: garrison-at-arrival vs enemy ship count.
#      If the enemy fleet flips the planet (captures it) or guts its garrison
#      below a cheap-capture threshold, mark it as a vulture opportunity.
#   3. For each opportunity, try each of our planets as a source. We must
#      arrive STRICTLY AFTER the enemy (natural lead-solution ETA > enemy ETA).
#      Compute the garrison we'll face (post-battle residual + production ×
#      arrival gap) and dispatch the minimal fleet to beat it by MARGIN=1.
#   4. Fall back to comet_wraith-style expansion for any ships not committed
#      by the vulture phase — so we keep growing even when no easy pickings exist.
# =============================================================================

import math

_BOARD = 100.0
_CX = _CY = 50.0
_SUN_R = 10.0
_ROT_LIM = 50.0
_EPISODE = 500

_prev_angles = {}    # player_id → {planet_id → angle}
_rotation_sign = {}  # player_id → +1 or -1


# ---- Helpers (copied from comet_wraith_v3) ----

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

    # ---- VULTURE PHASE ----
    # Post-battle garrison below this many production-turns → cheap enough to steal.
    GUT_TURNS = 4

    vulture_candidates = []  # (score, sid, tid, angle, our_eta, ships, required)

    for f in fleets_raw:
        if f[1] == player:
            continue
        f_owner, fx, fy, fangle, fships = f[1], f[2], f[3], f[4], f[6]

        # Infer target: smallest angular error (same threshold as threat cone)
        best_pid = None
        best_err = 0.35
        for pid, p in planets.items():
            px, py = p[2], p[3]
            expected = math.atan2(py - fy, px - fx)
            diff = fangle - expected
            diff -= 2.0 * math.pi * round(diff / (2.0 * math.pi))
            if abs(diff) < best_err:
                best_err = abs(diff)
                best_pid = pid

        if best_pid is None:
            continue
        tgt = planets.get(best_pid)
        if tgt is None:
            continue
        # Skip reinforcements to own planets and attacks on our planets (reserve handles those)
        if tgt[1] == f_owner or tgt[1] == player:
            continue

        tid = tgt[0]
        tx, ty, tr, tships, tprod, t_owner = (tgt[2], tgt[3], tgt[4], tgt[5],
                                               tgt[6], tgt[1])
        is_comet = tid in comet_pids
        ip = init_by_id.get(tid)

        # Enemy ETA (rough estimate using current planet position)
        dist_enemy = math.hypot(tx - fx, ty - fy)
        enemy_eta = max(1, int(round(dist_enemy / fleet_speed(fships, max_spd))))

        # Garrison at enemy arrival: neutral = no production, owned = grows
        garrison_at_arrival = tships if t_owner == -1 else tships + tprod * enemy_eta

        # Simulate battle
        if fships > garrison_at_arrival:
            # Enemy captures — new owner gets surplus ships as garrison
            post_garrison = fships - garrison_at_arrival
        else:
            # Enemy fails but might gut the defender
            post_garrison = garrison_at_arrival - fships
            if post_garrison >= tprod * GUT_TURNS:
                continue  # still well-defended, not a vulture opportunity

        # Try each of our planets as a launch source
        for src in my_planets:
            sid = src[0]
            sx, sy = src[2], src[3]
            spendable_max = src[5] - reserve.get(sid, MARGIN)
            if spendable_max < 1:
                continue
            if segment_hits_sun((sx, sy), (tx, ty)):
                continue

            # Lead solution using all available ships for ETA estimate
            def _v_solve(n, _sx=sx, _sy=sy, _tid=tid, _tx=tx, _ty=ty, _tr=tr,
                         _ic=is_comet, _ip=ip):
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

            sol = _v_solve(spendable_max)
            if sol is None:
                continue
            _, our_eta_0, _ = sol

            # We must arrive strictly AFTER the enemy — that's the whole point
            if our_eta_0 <= enemy_eta:
                continue

            # Iteratively refine ships_needed ↔ our_eta (converges in ≤3 steps)
            ships_needed = post_garrison + tprod * (our_eta_0 - enemy_eta) + MARGIN
            ships_needed = max(1, min(ships_needed, spendable_max))
            final_sol = None
            for _ in range(3):
                sol = _v_solve(ships_needed)
                if sol is None:
                    break
                angle, our_eta, aim = sol
                if our_eta <= enemy_eta:
                    break
                gap = our_eta - enemy_eta
                new_sn = post_garrison + tprod * gap + MARGIN
                if new_sn > spendable_max:
                    break
                final_sol = sol
                if new_sn == ships_needed:
                    ships_needed = new_sn
                    break
                ships_needed = new_sn
            else:
                # Loop finished without break — grab final sol
                sol = _v_solve(ships_needed)
                if sol is not None:
                    angle, our_eta, aim = sol
                    if our_eta > enemy_eta:
                        final_sol = sol

            if final_sol is None:
                continue
            angle, our_eta, aim = final_sol
            if our_eta <= enemy_eta:
                continue

            gap = our_eta - enemy_eta
            ships_needed = post_garrison + tprod * gap + MARGIN
            if ships_needed > spendable_max or ships_needed < 1:
                continue

            if segment_hits_sun((sx, sy), aim):
                continue
            if path_blocked_by_planet((sx, sy), aim, planets_raw, {sid, tid}):
                continue
            if our_eta + 2 >= turns_left:
                continue

            # Premium score: high-production planets captured cheaply are best
            score = tprod / (ships_needed + 0.3 * our_eta + 1.0) * 2.5
            if fships > garrison_at_arrival:
                score *= 1.5  # full capture-flip is tastiest

            vulture_candidates.append((score, sid, tid, angle, our_eta,
                                        ships_needed, ships_needed))

    vulture_candidates.sort(key=lambda c: -c[0])

    # Greedy assignment for vulture moves
    used = {mp[0]: 0 for mp in my_planets}
    committed = {}
    moves = []

    for score, sid, tid, angle, our_eta, ships, required in vulture_candidates:
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

    # ---- FALLBACK EXPANSION (comet_wraith-style) ----
    capturable = [p for p in planets.values() if p[1] != player]

    # Standings for multi-opponent suppression
    scores_map = {}
    for p in planets.values():
        if p[1] != -1:
            scores_map[p[1]] = scores_map.get(p[1], 0) + p[5]
    for f in fleets_raw:
        scores_map[f[1]] = scores_map.get(f[1], 0) + f[6]
    opp_scores = {o: s for o, s in scores_map.items() if o != player}
    leader = max(opp_scores, key=opp_scores.get) if opp_scores else None
    multi_opp = len(opp_scores) > 1

    expansion_candidates = []
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
            if segment_hits_sun((sx, sy), (tx, ty)):
                continue

            def _solve(n, _sx=sx, _sy=sy, _tid=tid, _tx=tx, _ty=ty, _tr=tr,
                       _ic=is_comet, _ip=ip):
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

            expansion_candidates.append((score, sid, tid, angle, eta, ships_send,
                                          required, towner == -1))

    expansion_candidates.sort(key=lambda c: -c[0])

    for score, sid, tid, angle, eta, ships_send, required, _is_neutral in expansion_candidates:
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
