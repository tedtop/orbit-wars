# =============================================================================
# Orbit Wars bot: stigmergic_pheromone_routing
#
# Ant Colony Optimization (ACO) inspired strategy:
#   - Maintains a 25x25 virtual pheromone grid over the 100x100 board.
#   - Each turn, 1-ship scout fleets probe target planets.
#   - A scout that arrives at its target (navigable path) deposits POSITIVE
#     pheromone along its grid path; a scout destroyed in transit (sun/OOB)
#     deposits NEGATIVE pheromone. Grid evaporates *0.95 per turn.
#   - Main battle fleets score targets with a pheromone bias multiplier on
#     the bearing toward each aim point. Analytic sun/planet segment checks
#     are ALWAYS applied independently of pheromone state.
#   - Scout fleet matching: each pending scout is keyed by (src_planet_id,
#     round(angle,6)); next turn we find the matching 1-ship fleet by those
#     fields and promote it to "active" (tracked by fleet_id).
# =============================================================================

import math
import random

_BOARD = 100.0
_CX = _CY = 50.0
_SUN_R = 10.0
_ROT_LIM = 50.0
_EPISODE = 500
_GRID = 25       # pheromone grid cells per axis
_EVAP = 0.95     # per-turn evaporation multiplier

# Per-player persistent state (keyed by player id; reset when step==0)
_pheromone = {}     # player → [[float]*_GRID for _ in range(_GRID)]
_pending   = {}     # player → {(src_pid, angle_key): (tgt_pid, cells, launch_step, exp_eta)}
_active    = {}     # player → {fleet_id: (tgt_pid, cells, launch_step, exp_eta)}
_prev_angles   = {} # player → {planet_id → float}
_rotation_sign = {} # player → +1 or -1


# ── Geometry helpers (verbatim from coordinated_strike_interceptor) ─────────────────────────

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


# ── Pheromone grid helpers ────────────────────────────────────────────────────

def _init_grid():
    return [[0.0] * _GRID for _ in range(_GRID)]


def _world_to_cell(x, y):
    ci = int(x / _BOARD * _GRID)
    cj = int(y / _BOARD * _GRID)
    return (max(0, min(_GRID - 1, ci)), max(0, min(_GRID - 1, cj)))


def _bresenham_cells(x0, y0, x1, y1):
    """Grid cells (Bresenham line) covering the world-space segment."""
    c0 = _world_to_cell(x0, y0)
    c1 = _world_to_cell(x1, y1)
    cells = []
    gx, gy = c0
    gx1, gy1 = c1
    dx = abs(gx1 - gx)
    dy = abs(gy1 - gy)
    sx = 1 if gx1 > gx else -1
    sy = 1 if gy1 > gy else -1
    err = dx - dy
    for _ in range(_GRID * 2 + 4):   # bounded to prevent infinite loop
        cells.append((gx, gy))
        if gx == gx1 and gy == gy1:
            break
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            gx += sx
        if e2 < dx:
            err += dx
            gy += sy
    return cells


def _deposit(grid, cells, amount):
    """Add amount to each cell; clamp to [-5, 10] to prevent runaway values."""
    for ci, cj in cells:
        grid[ci][cj] = max(-5.0, min(10.0, grid[ci][cj] + amount))


def _evaporate(grid):
    for i in range(_GRID):
        for j in range(_GRID):
            grid[i][j] *= _EVAP


def _path_pheromone(grid, x0, y0, x1, y1):
    """Average pheromone along the path from (x0,y0) to (x1,y1)."""
    cells = _bresenham_cells(x0, y0, x1, y1)
    return sum(grid[ci][cj] for ci, cj in cells) / max(1, len(cells))


# ── Main decision logic ───────────────────────────────────────────────────────

def _decide(obs, config):
    global _pheromone, _pending, _active, _prev_angles, _rotation_sign

    max_spd  = float(_get(config, "shipSpeed", 6.0))
    episode  = int(_get(config, "episodeSteps", _EPISODE))
    player   = int(_get(obs, "player", 0))
    step     = int(_get(obs, "step", 0))
    turns_left = max(1, episode - step - 2)

    # -- Reset per-player state at the start of each game --
    if step == 0:
        _pheromone[player]     = _init_grid()
        _pending[player]       = {}
        _active[player]        = {}
        _prev_angles[player]   = {}
        _rotation_sign[player] = 1

    grid       = _pheromone[player]
    p_pending  = _pending[player]
    p_active   = _active[player]
    p_prev_ang = _prev_angles.get(player, {})
    p_rot_sign = _rotation_sign.get(player, 1)

    planets_raw = _get(obs, "planets", []) or []
    fleets_raw  = _get(obs, "fleets", []) or []
    init_raw    = _get(obs, "initial_planets", []) or []
    comets_raw  = _get(obs, "comets", []) or []
    comet_pids  = set(_get(obs, "comet_planet_ids", []) or [])
    ang_vel     = float(_get(obs, "angular_velocity", 0.035))

    planets    = {p[0]: p for p in planets_raw}
    init_by_id = {p[0]: p for p in init_raw}

    # -- Rotation-sign inference (per player, same logic as coordinated_strike_interceptor) --
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
    if p_prev_ang and cur_angles:
        deltas = []
        for pid, cur_a in cur_angles.items():
            prev_a = p_prev_ang.get(pid)
            if prev_a is None:
                continue
            d = cur_a - prev_a
            d -= 2.0 * math.pi * round(d / (2.0 * math.pi))
            deltas.append(d)
        if deltas:
            avg = sum(deltas) / len(deltas)
            if abs(avg) > 1e-4:
                p_rot_sign = 1 if avg > 0 else -1
    _prev_angles[player]   = cur_angles
    _rotation_sign[player] = p_rot_sign

    my_planets = [p for p in planets.values() if p[1] == player]
    if not my_planets:
        return []
    capturable = [p for p in planets.values() if p[1] != player]

    # -- Evaporate pheromone grid --
    _evaporate(grid)

    # -- Process scout lifecycle --
    my_fleets    = {f[0]: f for f in fleets_raw if f[1] == player}
    my_fleet_ids = set(my_fleets.keys())
    tracked_ids  = set(p_active.keys())

    # Promote pending scouts → active once we see the matching fleet
    for fid, f in my_fleets.items():
        if fid in tracked_ids or f[6] != 1:
            continue
        key = (f[5], round(f[4], 6))   # (from_planet_id, angle)
        if key in p_pending:
            p_active[fid] = p_pending.pop(key)

    # Deposit pheromone for scouts that vanished this turn
    for fid in [fid for fid in list(p_active.keys()) if fid not in my_fleet_ids]:
        tgt_pid, cells, launch_step, exp_eta = p_active.pop(fid)
        tgt = planets.get(tgt_pid)
        if tgt is not None and tgt[1] == player:
            _deposit(grid, cells, +3.0)          # captured the planet
        elif step >= launch_step + exp_eta:
            _deposit(grid, cells, +1.0)          # arrived (path navigable)
        else:
            _deposit(grid, cells, -1.5)          # destroyed in transit

    # Purge pending scouts that never matched (source was too busy to launch)
    for k in [k for k, v in p_pending.items() if step - v[2] > 6]:
        del p_pending[k]

    # -- Threat map → per-planet reserve (same cone heuristic as coordinated_strike_interceptor) --
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

    # Opponent strength (for 4-player leader-focus)
    scores = {}
    for p in planets.values():
        if p[1] != -1:
            scores[p[1]] = scores.get(p[1], 0) + p[5]
    for f in fleets_raw:
        scores[f[1]] = scores.get(f[1], 0) + f[6]
    opp_scores = {o: s for o, s in scores.items() if o != player}
    leader    = max(opp_scores, key=opp_scores.get) if opp_scores else None
    multi_opp = len(opp_scores) > 1

    # -- Score every (source, target) pair --
    candidates = []
    for tgt in capturable:
        tid              = tgt[0]
        tx, ty, tr       = tgt[2], tgt[3], tgt[4]
        tships, tprod    = tgt[5], tgt[6]
        towner           = tgt[1]
        is_comet         = tid in comet_pids
        ip               = init_by_id.get(tid)

        if is_comet:
            remaining = 0
            for grp in comets_raw:
                if tid in grp["planet_ids"]:
                    i = grp["planet_ids"].index(tid)
                    remaining = len(grp["paths"][i]) - grp["path_index"] - 1
                    break
            if remaining < 15:
                continue

        for src in my_planets:
            sid = src[0]
            sx, sy        = src[2], src[3]
            spendable_max = src[5] - reserve.get(sid, MARGIN)
            if spendable_max < 1:
                continue
            if segment_hits_sun((sx, sy), (tx, ty)):
                continue

            # Closure captures current loop variables; called immediately each iter.
            def _solve(n):
                if is_comet:
                    return lead_solution((sx, sy), tid, tx, ty, tr, True,
                                        comets_raw, ang_vel, step, n, max_spd,
                                        p_rot_sign)
                if ip is not None:
                    return lead_solution((sx, sy), tid, ip[2], ip[3], tr, False,
                                        comets_raw, ang_vel, step, n, max_spd,
                                        p_rot_sign)
                d = math.hypot(tx - sx, ty - sy)
                e = max(1, int(d / fleet_speed(n, max_spd)))
                return (math.atan2(ty - sy, tx - sx), e, (tx, ty))

            sol = _solve(spendable_max)
            if sol is None:
                continue
            angle, eta, aim = sol

            garrison   = tships if towner == -1 else tships + tprod * eta
            required   = garrison + MARGIN
            ships_send = min(spendable_max, required)
            if ships_send < 1:
                continue

            # Re-solve with actual send count (fleet speed depends on ship count)
            sol = _solve(ships_send)
            if sol is None:
                continue
            angle, eta, aim = sol
            if towner != -1:
                garrison   = tships + tprod * eta
                required   = garrison + MARGIN
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

            # Base score (coordinated_strike_interceptor formula)
            score = tprod / (required + 0.3 * eta + 1.0)
            if step < 60 and towner == -1:
                d = math.hypot(tx - sx, ty - sy)
                score *= 1.5 / (1.0 + d * 0.01)
            if is_comet:
                score *= 1.2
            if multi_opp and towner == leader:
                score *= 1.3

            # Pheromone bias: multiply by (1 + 0.15*ph); ph can be negative
            ph = _path_pheromone(grid, sx, sy, aim[0], aim[1])
            score = max(1e-4, score * (1.0 + 0.15 * ph))

            candidates.append((score, sid, tid, angle, eta, ships_send, required,
                               towner == -1))

    candidates.sort(key=lambda c: -c[0])

    # Greedy fleet assignment (cumulative, same as coordinated_strike_interceptor)
    used      = {mp[0]: 0 for mp in my_planets}
    committed = {}
    moves     = []
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

    # -- Launch one scout per turn if below the cap --
    if len(p_pending) + len(p_active) < 4 and capturable:
        pending_tgts = {v[0] for v in p_pending.values()}
        active_tgts  = {v[0] for v in p_active.values()}
        unscouted = [p for p in capturable
                     if p[0] not in pending_tgts and p[0] not in active_tgts]
        random.shuffle(unscouted)

        # Only scout from planets with ≥2 free ships above reserve
        scout_srcs = [p for p in my_planets
                      if p[5] - reserve.get(p[0], MARGIN) - used.get(p[0], 0) >= 2]
        if scout_srcs and unscouted:
            src = random.choice(scout_srcs)
            sid = src[0]
            sx, sy = src[2], src[3]
            for tgt in unscouted[:4]:
                tid = tgt[0]
                tx, ty = tgt[2], tgt[3]
                if segment_hits_sun((sx, sy), (tx, ty)):
                    continue
                if path_blocked_by_planet((sx, sy), (tx, ty), planets_raw, {sid, tid}):
                    continue
                angle = math.atan2(ty - sy, tx - sx)
                key   = (sid, round(angle, 6))
                if key in p_pending:
                    continue
                dist    = math.hypot(tx - sx, ty - sy)
                exp_eta = max(1, int(dist))   # 1-ship speed = 1 unit/turn
                cells   = _bresenham_cells(sx, sy, tx, ty)
                p_pending[key] = (tid, cells, step, exp_eta)
                moves.append([sid, angle, 1])
                used[sid] = used.get(sid, 0) + 1
                break

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
