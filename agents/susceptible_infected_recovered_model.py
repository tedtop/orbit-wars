"""
Orbit Wars bot: susceptible_infected_recovered_model

Strategy: SIR epidemiological model applied to the battlefield.

Planet classification:
  Susceptible (S): neutral planets OR lightly-held own planets (ships < threshold)
  Infected (I):    enemy-owned planets (the contagion spreading across the map)
  Recovered (R):   heavily-fortified own planets (stable, hard to flip)

Sector grid: the 100x100 board is divided into a 4x4 grid of 25x25 sectors.
Each sector's basic reproduction number R0 measures whether the enemy is
growing faster than we can suppress it:

  R0 = enemy_production_in_sector / our_delivery_rate_to_sector

where delivery_rate = Σ available_ships[src] / eta_to_sector_center across our
source planets.  (Think of it as: ships/turn we can pour into containment.)

When R0 ≥ 1 in a sector the enemy compounds faster than we contain it.
→ QUARANTINE: boost priority of attacking weak-garrison enemy planets in that
  sector to starve its production before it spirals.  The weaker the target
  (lower garrison at arrival) the more urgent the capture.

When no sector is critical (all R0 < 1) run normal greedy expansion:
score = production / (required_ships + 0.3 * eta + 1).

Never idles: every planet with spendable ships participates each turn.
"""

import math

_BOARD = 100.0
_CX = _CY = 50.0
_SUN_R = 10.0
_ROT_LIM = 50.0
_EPISODE = 500
_GRID_N = 4           # 4x4 sector grid → 25×25 units per cell
_SECTOR_SZ = _BOARD / _GRID_N

_prev_angles = {}     # player_id → {planet_id → float}
_rotation_sign = {}   # player_id → +1 or -1


# ---------------------------------------------------------------------------
# Helpers (copied from comet_wraith_v3 — the authoritative reference impl)
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
    """Returns (angle, eta, aim_pos) or None. ships = what is actually sent."""
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
# SIR-specific helpers
# ---------------------------------------------------------------------------

def _sector(x, y):
    """Map a position to a (gx, gy) grid bucket."""
    gx = min(int(x / _SECTOR_SZ), _GRID_N - 1)
    gy = min(int(y / _SECTOR_SZ), _GRID_N - 1)
    return (gx, gy)


def _sector_center(gx, gy):
    return (_SECTOR_SZ * (gx + 0.5), _SECTOR_SZ * (gy + 0.5))


def _compute_r0(sector, enemy_by_sector, my_capacity, max_spd):
    """
    R0 = enemy_production_in_sector / our_delivery_rate_to_sector_center.

    delivery_rate = Σ available[src] / eta(src → sector_center)  [ships/turn]

    R0 ≥ 1 means enemy production outpaces our suppression capacity.
    Returns inf when we have no capacity to reach the sector at all.
    """
    enemies = enemy_by_sector.get(sector, [])
    if not enemies:
        return 0.0
    enemy_prod = sum(p[6] for p in enemies)
    cx, cy = _sector_center(*sector)
    delivery = 0.0
    for src_pos, avail in my_capacity:
        if avail < 1:
            continue
        d = math.hypot(cx - src_pos[0], cy - src_pos[1])
        eta = max(1.0, d / fleet_speed(avail, max_spd))
        delivery += avail / eta
    if delivery < 1e-9:
        return float("inf")
    return enemy_prod / delivery


# ---------------------------------------------------------------------------
# Core decision logic
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

    # ---- Rotation-sign inference (per player, same method as comet_wraith) ----
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

    # ---- Threat → defensive reserve ----
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

    # ---- SIR sector analysis ----
    # Group Infected (enemy) planets by sector
    enemy_by_sector = {}
    for p in planets.values():
        if p[1] != player and p[1] != -1:
            sec = _sector(p[2], p[3])
            enemy_by_sector.setdefault(sec, []).append(p)

    # Our available delivery capacity: (pos, avail_ships) per planet
    my_capacity = []
    for mp in my_planets:
        avail = mp[5] - reserve.get(mp[0], MARGIN)
        my_capacity.append(((mp[2], mp[3]), max(0, avail)))

    # Compute R0 per infected sector and flag critical ones
    critical_sectors = {}  # sector → r0
    for sec in enemy_by_sector:
        r0 = _compute_r0(sec, enemy_by_sector, my_capacity, max_spd)
        if r0 >= 1.0:
            critical_sectors[sec] = r0

    # ---- Score (source, target) pairs ----
    capturable = [p for p in planets.values() if p[1] != player]
    candidates = []

    for tgt in capturable:
        tid = tgt[0]
        tx, ty, tr, tships, tprod, towner = (tgt[2], tgt[3], tgt[4],
                                              tgt[5], tgt[6], tgt[1])
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

        tgt_sec = _sector(tx, ty)
        # Quarantine: target is an enemy in a critical sector
        is_quarantine = (towner != player and towner != -1
                         and tgt_sec in critical_sectors)

        for src in my_planets:
            sid = src[0]
            sx, sy = src[2], src[3]
            spendable_max = src[5] - reserve.get(sid, MARGIN)
            if spendable_max < 1:
                continue
            if segment_hits_sun((sx, sy), (tx, ty)):
                continue

            # Capture the loop vars in default args to avoid closure trap
            def _solve(n, _sx=sx, _sy=sy, _tid=tid, _tx=tx, _ty=ty, _tr=tr,
                       _is_comet=is_comet, _ip=ip):
                if _is_comet:
                    return lead_solution((_sx, _sy), _tid, _tx, _ty, _tr, True,
                                         comets, ang_vel, step, n, max_spd,
                                         p_rotation_sign)
                if _ip is not None:
                    return lead_solution((_sx, _sy), _tid, _ip[2], _ip[3], _tr,
                                         False, comets, ang_vel, step, n,
                                         max_spd, p_rotation_sign)
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

            # Refine with actual send size (speed depends on ship count)
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

            # End-game gate: don't launch if fleet can't land before time runs out
            if eta + 2 >= turns_left:
                continue
            if turns_left < 45:
                mult = 2.0 if towner != -1 else 1.0
                if tprod * (turns_left - eta) * mult <= garrison:
                    continue

            # Base score: production yield per ship+time invested
            score = tprod / (required + 0.3 * eta + 1.0)

            # Early expansion: proximity bonus for neutral planets
            if step < 60 and towner == -1:
                d = math.hypot(tx - sx, ty - sy)
                score *= 1.5 / (1.0 + d * 0.01)

            # Quarantine boost: prioritise weak-garrison enemies in critical sectors.
            # Stronger R0 → bigger boost.  Lower garrison → easier quarantine win.
            if is_quarantine:
                r0 = critical_sectors[tgt_sec]
                score *= (1.5 + r0) / (tships + 1.0)

            candidates.append((score, sid, tid, angle, eta, ships_send, required))

    candidates.sort(key=lambda c: -c[0])

    # Greedy assignment: accumulate committed ships per target across sources
    used = {mp[0]: 0 for mp in my_planets}
    committed = {}
    moves = []
    for score, sid, tid, angle, eta, ships_send, required in candidates:
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
    """Crash-safe entry point — must be the last callable in the file."""
    try:
        return _decide(obs, config)
    except Exception:
        import sys
        import traceback
        traceback.print_exc(file=sys.stderr)
        return []
