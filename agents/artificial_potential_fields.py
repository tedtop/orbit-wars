# =============================================================================
# Orbit Wars bot: artificial_potential_fields
#
# Treats the board as a potential-energy landscape (robotics APF / fluid analogy):
#   - Sun           : strong repulsive source   (sun_k / d^3, pointing away)
#   - Enemy/neutral : attractive sinks scaled by weakness
#                     weakness = production / max(garrison, 1)  →  deep wells
#                     at low-garrison high-production planets
#   - Own interior  : mild repulsive bumpers (prevent ships stacking at home)
#
# At each owned planet we sum gradient vectors to get a resultant heading, then
# snap to the highest-APF-aligned capturable target (cosine-weighted value score).
# Fallback: greedy nearest viable target so ships never idle.
# Lead-solution aiming, rotation-sign inference, Sun segment-check, and defensive
# reserve are all copied verbatim from coordinated_strike_interceptor.
# =============================================================================

import math

_BOARD = 100.0
_CX = _CY = 50.0
_SUN_R = 10.0
_ROT_LIM = 50.0
_EPISODE = 500

_prev_angles = {}
_rotation_sign = {}


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


# APF tuning constants
_SUN_K  = 800.0   # sun repulsion strength
_ATTR_K = 12.0    # enemy/neutral attraction strength
_OWN_K  =  3.0    # own-planet mild repulsion strength


def _apf_resultant(sx, sy, my_planets, capturable):
    """
    Compute the APF gradient vector at position (sx, sy).

    Forces (all as 1/d^2 magnitude, 1/d^3 vector components):
      Sun repulsion:     pushes away from (CX, CY)
      Target attraction: pulls toward each enemy/neutral, weighted by weakness
      Own repulsion:     pushes away from own orbiting planets (interior only)
    """
    rx, ry = 0.0, 0.0

    # Sun: repel
    sdx, sdy = sx - _CX, sy - _CY
    sd = math.hypot(sdx, sdy)
    if sd > 1e-6:
        sd3 = sd * sd * sd
        rx += _SUN_K * sdx / sd3
        ry += _SUN_K * sdy / sd3

    # Capturable planets: attract, scaled by weakness = production / garrison
    for tgt in capturable:
        tx, ty, tships, tprod = tgt[2], tgt[3], tgt[5], tgt[6]
        tdx, tdy = tx - sx, ty - sy
        td = math.hypot(tdx, tdy)
        if td < 1e-6:
            continue
        weakness = tprod / max(tships, 1.0)
        strength = _ATTR_K * weakness / (td * td)
        rx += strength * tdx / td
        ry += strength * tdy / td

    # Own interior (orbiting) planets: mild repel so ships don't cluster
    for mp in my_planets:
        if mp[2] == sx and mp[3] == sy:
            continue
        dist_c = math.hypot(mp[2] - _CX, mp[3] - _CY)
        if dist_c + mp[4] >= _ROT_LIM:
            continue  # only repel from interior planets
        odx, ody = sx - mp[2], sy - mp[3]
        od = math.hypot(odx, ody)
        if od < 1e-6:
            continue
        strength = _OWN_K / (od * od)
        rx += strength * odx / od
        ry += strength * ody / od

    return rx, ry


def _decide(obs, config):
    global _prev_angles, _rotation_sign

    max_spd = float(_get(config, "shipSpeed", 6.0))
    episode = int(_get(config, "episodeSteps", _EPISODE))
    player  = int(_get(obs, "player", 0))

    p_prev_angles  = _prev_angles.get(player, {})
    p_rotation_sign = _rotation_sign.get(player, 1)
    planets_raw = _get(obs, "planets", []) or []
    fleets_raw  = _get(obs, "fleets",  []) or []
    init_raw    = _get(obs, "initial_planets", []) or []
    comets      = _get(obs, "comets", []) or []
    comet_pids  = set(_get(obs, "comet_planet_ids", []) or [])
    ang_vel     = float(_get(obs, "angular_velocity", 0.035))
    step        = int(_get(obs, "step", 0))
    turns_left  = max(1, episode - step - 2)

    planets    = {p[0]: p for p in planets_raw}
    init_by_id = {p[0]: p for p in init_raw}

    # ---- Rotation-sign inference (per player) — verbatim from coordinated_strike_interceptor ----
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

    # ---- Threat → defensive reserve (MARGIN=1 from coordinated_strike_interceptor) ----
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

    used  = {mp[0]: 0 for mp in my_planets}
    moves = []

    for src in my_planets:
        sid = src[0]
        sx, sy = src[2], src[3]
        spendable = src[5] - reserve.get(sid, MARGIN) - used[sid]
        if spendable < 1:
            continue

        # Compute APF resultant at this planet's position
        rx, ry = _apf_resultant(sx, sy, my_planets, capturable)
        rmag = math.hypot(rx, ry)
        if rmag < 1e-9:
            # Degenerate: fall straight through to fallback
            rx, ry, rmag = 1.0, 0.0, 1.0

        # ---- Find best target aligned with APF gradient ----
        best_score  = -1e18
        best_sol    = None
        best_send   = 0

        for tgt in capturable:
            tid    = tgt[0]
            tx, ty = tgt[2], tgt[3]
            tr, tships, tprod, towner = tgt[4], tgt[5], tgt[6], tgt[1]
            is_comet = tid in comet_pids
            ip = init_by_id.get(tid)

            # Skip near-expired comets
            if is_comet:
                remaining = 0
                for grp in comets:
                    if tid in grp["planet_ids"]:
                        i = grp["planet_ids"].index(tid)
                        remaining = len(grp["paths"][i]) - grp["path_index"] - 1
                        break
                if remaining < 15:
                    continue

            # Sun check on straight line to current target position
            if segment_hits_sun((sx, sy), (tx, ty)):
                continue

            # Lead solution with max spendable ships (for ETA estimate)
            if is_comet:
                sol = lead_solution((sx, sy), tid, tx, ty, tr, True, comets,
                                    ang_vel, step, spendable, max_spd, p_rotation_sign)
            elif ip is not None:
                sol = lead_solution((sx, sy), tid, ip[2], ip[3], tr, False, comets,
                                    ang_vel, step, spendable, max_spd, p_rotation_sign)
            else:
                d = math.hypot(tx - sx, ty - sy)
                e = max(1, int(d / fleet_speed(spendable, max_spd)))
                sol = (math.atan2(ty - sy, tx - sx), e, (tx, ty))
            if sol is None:
                continue
            _, eta, _ = sol

            if eta + 2 >= turns_left:
                continue

            garrison = tships if towner == -1 else tships + tprod * eta
            required = garrison + MARGIN
            ships_send = min(spendable, required)
            if ships_send < 1:
                continue

            # Re-solve with actual ship count for precise aim
            if is_comet:
                sol2 = lead_solution((sx, sy), tid, tx, ty, tr, True, comets,
                                     ang_vel, step, ships_send, max_spd, p_rotation_sign)
            elif ip is not None:
                sol2 = lead_solution((sx, sy), tid, ip[2], ip[3], tr, False, comets,
                                     ang_vel, step, ships_send, max_spd, p_rotation_sign)
            else:
                d = math.hypot(tx - sx, ty - sy)
                e = max(1, int(d / fleet_speed(ships_send, max_spd)))
                sol2 = (math.atan2(ty - sy, tx - sx), e, (tx, ty))
            if sol2 is None:
                continue
            angle2, eta2, aim2 = sol2

            # Recompute garrison with refined ETA
            if towner != -1:
                garrison = tships + tprod * eta2
                required = garrison + MARGIN
                ships_send = min(spendable, required)
                if ships_send < 1:
                    continue

            # Sun segment-check on actual aimed position (hard requirement)
            if segment_hits_sun((sx, sy), aim2):
                continue
            if path_blocked_by_planet((sx, sy), aim2, planets_raw, {sid, tid}):
                continue

            # Late-game ROI gate
            if turns_left < 45:
                mult = 2.0 if towner != -1 else 1.0
                if tprod * (turns_left - eta2) * mult <= garrison:
                    continue

            # APF score: cosine alignment with resultant × production value
            tdx, tdy = tx - sx, ty - sy
            td = math.hypot(tdx, tdy)
            if td < 1e-6:
                continue
            cosine = (rx * tdx + ry * tdy) / (rmag * td)
            cosine = max(-1.0, min(1.0, cosine))

            value = tprod / (required + 0.3 * eta2 + 1.0)
            # (cosine + 1) ∈ [0, 2] — never goes negative, rewards alignment
            score = (1.0 + cosine) * value

            if score > best_score:
                best_score = score
                best_sol   = (angle2, eta2, aim2)
                best_send  = ships_send

        if best_sol is None:
            # Fallback: greedy nearest viable target so we never idle
            for tgt in sorted(capturable,
                               key=lambda t: math.hypot(t[2] - sx, t[3] - sy)):
                tid    = tgt[0]
                tx, ty = tgt[2], tgt[3]
                tr, tships, tprod, towner = tgt[4], tgt[5], tgt[6], tgt[1]
                if tid in comet_pids:
                    continue
                if segment_hits_sun((sx, sy), (tx, ty)):
                    continue
                ip = init_by_id.get(tid)
                if ip is not None:
                    sol = lead_solution((sx, sy), tid, ip[2], ip[3], tr, False,
                                        comets, ang_vel, step, spendable,
                                        max_spd, p_rotation_sign)
                else:
                    d = math.hypot(tx - sx, ty - sy)
                    e = max(1, int(d / fleet_speed(spendable, max_spd)))
                    sol = (math.atan2(ty - sy, tx - sx), e, (tx, ty))
                if sol is None:
                    continue
                _, eta_fb, aim_fb = sol
                if eta_fb + 2 >= turns_left:
                    continue
                garrison_fb = tships if towner == -1 else tships + tprod * eta_fb
                req_fb = garrison_fb + MARGIN
                s_fb = min(spendable, req_fb)
                if s_fb < 1:
                    continue
                if segment_hits_sun((sx, sy), aim_fb):
                    continue
                angle_fb = math.atan2(aim_fb[1] - sy, aim_fb[0] - sx)
                best_sol  = (angle_fb, eta_fb, aim_fb)
                best_send = s_fb
                break

        if best_sol is None:
            continue

        angle, _eta, aim = best_sol
        # Final sun-check on launch heading (belt-and-suspenders)
        if segment_hits_sun((sx, sy), aim):
            continue

        moves.append([sid, angle, int(best_send)])
        used[sid] += best_send

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
