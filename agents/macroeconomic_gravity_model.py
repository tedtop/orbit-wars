"""
Macroeconomic Gravity Model — Orbit Wars agent

Applies the gravity model from migration studies: each planet carries a "mass"
encoding its strategic weight. For every (owned source i, candidate target j)
pair the attraction force is:

    F_ij = (M_i * M_j) / D_ij^2

Each owned planet independently distributes its spare ships (garrison minus
defensive reserve) across targets, weighted by normalised F_ij values.

Mass assignment
  Owned planets :   M = ships + production * 10   (size + productivity)
  Enemy / neutral : M = production                (strategic capture value)
  Threatened ally : M = excess threat             (reinforcement pull)

Emergent behaviour: high-production nearby enemy planets attract the heaviest
flows; distant, low-value outposts get starved; friendly planets under serious
threat draw defensive reinforcement.  Captures use a lead-solution ETA to
send strictly more ships than garrison-at-arrival.  End-game gate prevents
wasted launches.  Helpers and rotation-sign inference copied from comet_wraith_v3.
"""

import math

_BOARD = 100.0
_CX = _CY = 50.0
_SUN_R = 10.0
_ROT_LIM = 50.0
_EPISODE = 500

_prev_angles = {}    # player_id → {planet_id → angle}
_rotation_sign = {}  # player_id → +1 or -1


# ── helpers (verbatim from comet_wraith_v3) ───────────────────────────────────

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


# ── main strategy ─────────────────────────────────────────────────────────────

def _decide(obs, config):
    global _prev_angles, _rotation_sign

    max_spd  = float(_get(config, "shipSpeed", 6.0))
    episode  = int(_get(config, "episodeSteps", _EPISODE))
    player   = int(_get(obs, "player", 0))

    p_prev_angles   = _prev_angles.get(player, {})
    p_rotation_sign = _rotation_sign.get(player, 1)
    planets_raw  = _get(obs, "planets", []) or []
    fleets_raw   = _get(obs, "fleets", []) or []
    init_raw     = _get(obs, "initial_planets", []) or []
    comets       = _get(obs, "comets", []) or []
    comet_pids   = set(_get(obs, "comet_planet_ids", []) or [])
    ang_vel      = float(_get(obs, "angular_velocity", 0.035))
    step         = int(_get(obs, "step", 0))
    turns_left   = max(1, episode - step - 2)

    planets    = {p[0]: p for p in planets_raw}
    init_by_id = {p[0]: p for p in init_raw}

    # ── rotation-sign inference (per player) ──────────────────────────────────
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

    # ── threat → defensive reserve ────────────────────────────────────────────
    MARGIN      = 1
    RESERVE_MIN = 3
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
    reserve = {mp[0]: max(threat[mp[0]] + MARGIN, RESERVE_MIN) for mp in my_planets}

    # Planets whose incoming threat exceeds half their garrison — need backup
    reinforce_ids = {mp[0] for mp in my_planets
                     if threat.get(mp[0], 0) > mp[5] // 2 + 1}

    # ── lead-solution helper ──────────────────────────────────────────────────
    def _solve(src_xy, tid, n):
        tgt = planets.get(tid)
        if tgt is None:
            return None
        is_c = tid in comet_pids
        ip   = init_by_id.get(tid)
        if is_c:
            return lead_solution(src_xy, tid, tgt[2], tgt[3], tgt[4], True,
                                  comets, ang_vel, step, n, max_spd, p_rotation_sign)
        if ip is not None:
            return lead_solution(src_xy, tid, ip[2], ip[3], tgt[4], False,
                                  comets, ang_vel, step, n, max_spd, p_rotation_sign)
        # static fallback (no initial record)
        d = math.hypot(tgt[2] - src_xy[0], tgt[3] - src_xy[1])
        e = max(1, int(d / fleet_speed(n, max_spd)))
        return (math.atan2(tgt[3] - src_xy[1], tgt[2] - src_xy[0]), e, (tgt[2], tgt[3]))

    # ── gravity flow ──────────────────────────────────────────────────────────
    capturable = [p for p in planets.values() if p[1] != player]
    moves = []

    for src in my_planets:
        sid      = src[0]
        sx, sy   = src[2], src[3]
        src_ships = src[5]
        src_prod  = src[6]
        spare     = src_ships - reserve[sid]
        if spare < 1:
            continue

        M_src = float(src_ships + src_prod * 10)

        # ── compute gravity force to each candidate ───────────────────────────
        # entries: (force, planet_id, is_reinforce_flag)
        entries = []

        for tgt in capturable:
            tid  = tgt[0]
            is_c = tid in comet_pids

            if is_c:
                rem_path = 0
                for grp in comets:
                    if tid in grp["planet_ids"]:
                        idx = grp["planet_ids"].index(tid)
                        rem_path = len(grp["paths"][idx]) - grp["path_index"] - 1
                        break
                if rem_path < 15:
                    continue

            tx, ty = tgt[2], tgt[3]
            if segment_hits_sun((sx, sy), (tx, ty)):
                continue
            D = math.hypot(tx - sx, ty - sy)
            if D < 0.1:
                continue

            M_tgt = float(tgt[6])          # production = strategic capture value
            F = M_src * M_tgt / (D * D)
            entries.append((F, tid, False))

        for mp in my_planets:
            mid = mp[0]
            if mid == sid or mid not in reinforce_ids:
                continue
            tx, ty = mp[2], mp[3]
            if segment_hits_sun((sx, sy), (tx, ty)):
                continue
            D = math.hypot(tx - sx, ty - sy)
            if D < 0.1:
                continue
            pull  = float(max(threat.get(mid, 0) - mp[5], mp[6]))
            F = M_src * pull / (D * D)
            entries.append((F, mid, True))

        if not entries:
            continue

        total_F = sum(f for f, _, _ in entries)
        if total_F < 1e-9:
            continue

        entries.sort(key=lambda x: -x[0])
        remaining = spare

        for F, tid, is_reinforce in entries:
            if remaining < 1:
                break

            weight = F / total_F
            alloc  = max(1, int(spare * weight + 0.5))
            tgt    = planets.get(tid)
            if tgt is None:
                continue

            if is_reinforce:
                alloc = min(alloc, remaining)
                sol = _solve((sx, sy), tid, alloc)
                if sol is None:
                    continue
                angle, eta, aim = sol
                if segment_hits_sun((sx, sy), aim):
                    continue
                moves.append([sid, angle, int(alloc)])
                remaining -= alloc

            else:
                # ── capture branch ────────────────────────────────────────────
                towner = tgt[1]
                tships = tgt[5]
                tprod  = tgt[6]

                # Probe ETA with gravity-weighted allocation
                probe = max(1, min(alloc, remaining))
                sol   = _solve((sx, sy), tid, probe)
                if sol is None:
                    continue
                _, eta_probe, _ = sol

                garrison = tships if towner == -1 else tships + tprod * eta_probe
                needed   = garrison + MARGIN

                # Bump to the minimum viable fleet (never exceed remaining)
                send = min(remaining, max(probe, needed))
                if send < needed or send < 1:
                    continue

                if eta_probe + 2 >= turns_left:
                    continue

                # Refine aim with the actual fleet size
                sol = _solve((sx, sy), tid, send)
                if sol is None:
                    continue
                angle, eta, aim = sol

                # Re-verify garrison with refined ETA
                if towner != -1:
                    garrison = tships + tprod * eta
                    needed   = garrison + MARGIN
                    if send < needed:
                        continue

                if segment_hits_sun((sx, sy), aim):
                    continue
                if path_blocked_by_planet((sx, sy), aim, planets_raw, {sid, tid}):
                    continue

                moves.append([sid, angle, int(send)])
                remaining -= send

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
