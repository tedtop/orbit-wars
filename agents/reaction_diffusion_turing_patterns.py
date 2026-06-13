# =============================================================================
# Orbit Wars bot: reaction_diffusion_turing_patterns
#
# Strategy — Turing-pattern reaction-diffusion on a coarse grid:
#   Activator (A) = our ship density / production; diffuses slowly (Da).
#   Inhibitor (B) = enemy presence;                diffuses fast  (Di > Da).
#
# The Da << Di imbalance is the Turing instability condition: stable
# stripes/spots of high activator emerge along the frontline, forming a
# self-repairing defense-in-depth perimeter.  Cells where A >> B are
# dominance zones — we attack out from them.  Cells where B encroaches on
# A signal planets that need reinforcement (covered by the threat-reserve
# computed from actual enemy fleet cones, same as coordinated_strike_interceptor).
#
# Per-turn work is bounded: GRID=20x20, ITER=4 RD steps.
# All helpers copied from coordinated_strike_interceptor (the reference bot).
# =============================================================================

import math

_BOARD = 100.0
_CX = _CY = 50.0
_SUN_R = 10.0
_ROT_LIM = 50.0
_EPISODE = 500

# ---- Reaction-diffusion parameters ----
_GRID = 20          # 20x20 coarse grid → 5x5 board-units per cell
_CELL = _BOARD / _GRID
_Da = 0.10          # activator diffusion rate (slow — short-range activation)
_Di = 0.40          # inhibitor diffusion rate (fast — long-range inhibition)
_ITER = 4           # RD iterations per turn (hard bound on CPU)
_DECAY = 0.7        # field persistence across turns (blends old state with sources)

# Per-player persistent state; keyed by player id and reset when step==0
_fields = {}        # player → {'A': grid, 'B': grid}
_prev_angles = {}   # player → {planet_id → angle}  (rotation-sign inference)
_rotation_sign = {} # player → +1 or -1


# ---------------------------------------------------------------------------
# Helpers (copied verbatim from coordinated_strike_interceptor)
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


def _predict_comet_pos(pid, comets, step_ahead):
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
    """Returns (angle, eta, aim_pos) or None.  `ships` = what's actually sent."""
    if is_comet:
        cur = _predict_comet_pos(tgt_pid, comets, 0) or (init_x, init_y)
    else:
        cur = predict_planet_pos(init_x, init_y, tgt_radius, angular_velocity,
                                 current_step, rotation_sign)
    dist = math.hypot(cur[0] - src_pos[0], cur[1] - src_pos[1])
    t = max(1.0, dist / fleet_speed(ships, max_spd))
    for _ in range(8):
        t_int = max(1, int(round(t)))
        if is_comet:
            fut = _predict_comet_pos(tgt_pid, comets, t_int)
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
        aim = _predict_comet_pos(tgt_pid, comets, t_int)
        if aim is None:
            return None
    else:
        aim = predict_planet_pos(init_x, init_y, tgt_radius, angular_velocity,
                                 current_step + t_int, rotation_sign)
    return (math.atan2(aim[1] - src_pos[1], aim[0] - src_pos[0]), t_int, aim)


# ---------------------------------------------------------------------------
# Reaction-diffusion grid helpers
# ---------------------------------------------------------------------------

def _make_grid(val=0.0):
    return [[val] * _GRID for _ in range(_GRID)]


def _world_to_cell(x, y):
    col = int(max(0, min(_GRID - 1, x / _CELL)))
    row = int(max(0, min(_GRID - 1, y / _CELL)))
    return (row, col)


def _laplacian(field, r, c):
    """4-neighbor discrete Laplacian with Neumann (no-flux) boundary."""
    center = field[r][c]
    n = field[r - 1][c] if r > 0 else center
    s = field[r + 1][c] if r < _GRID - 1 else center
    w = field[r][c - 1] if c > 0 else center
    e = field[r][c + 1] if c < _GRID - 1 else center
    return n + s + w + e - 4.0 * center


def _rd_step(A, B, src_A, src_B):
    """One Euler step of the diffusion-only RD model (no algebraic coupling).

    dA/dt = Da * ∇²A + src_A
    dB/dt = Di * ∇²B + src_B

    The Turing instability is purely structural: slow activator vs fast
    inhibitor, continuously injected from planet sources.  This keeps the
    per-step cost O(GRID²) without any multiplication tables.
    """
    new_A = [row[:] for row in A]
    new_B = [row[:] for row in B]
    for r in range(_GRID):
        for c in range(_GRID):
            new_A[r][c] = max(0.0, A[r][c] + _Da * _laplacian(A, r, c) + src_A[r][c])
            new_B[r][c] = max(0.0, B[r][c] + _Di * _laplacian(B, r, c) + src_B[r][c])
    return new_A, new_B


# ---------------------------------------------------------------------------
# Decision logic
# ---------------------------------------------------------------------------

def _decide(obs, config):
    global _prev_angles, _rotation_sign, _fields

    max_spd = float(_get(config, "shipSpeed", 6.0))
    episode = int(_get(config, "episodeSteps", _EPISODE))
    player = int(_get(obs, "player", 0))

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

    # ---- Rotation-sign inference (per player, from coordinated_strike_interceptor) ----
    p_prev_angles    = _prev_angles.get(player, {})
    p_rotation_sign  = _rotation_sign.get(player, 1)
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

    # ---- Reset RD fields on step 0 ----
    if step == 0 or player not in _fields:
        _fields[player] = {'A': _make_grid(), 'B': _make_grid()}

    A = _fields[player]['A']
    B = _fields[player]['B']

    # ---- Build source grids from current planet / fleet state ----
    # Activator source: our ships + production.
    # Inhibitor source: enemy ships + production; neutral planets add small inhibitor
    # (contested territory, slightly harder to activate).
    src_A = _make_grid()
    src_B = _make_grid()

    for p in planets.values():
        px, py       = p[2], p[3]
        pships, pprod, powner = p[5], p[6], p[1]
        r, c = _world_to_cell(px, py)
        if powner == player:
            src_A[r][c] += pships * 0.05 + pprod * 0.5
        elif powner != -1:
            src_B[r][c] += pships * 0.05 + pprod * 0.5
        else:
            src_B[r][c] += 0.1  # neutral: weak inhibitor

    for f in fleets_raw:
        if f[1] != player:
            fx, fy, fships = f[2], f[3], f[6]
            r, c = _world_to_cell(fx, fy)
            src_B[r][c] += fships * 0.03

    # Blend old fields with fresh sources, then iterate
    for r in range(_GRID):
        for c in range(_GRID):
            A[r][c] = A[r][c] * _DECAY + src_A[r][c]
            B[r][c] = B[r][c] * _DECAY + src_B[r][c]

    for _ in range(_ITER):
        A, B = _rd_step(A, B, src_A, src_B)

    _fields[player]['A'] = A
    _fields[player]['B'] = B

    # ---- Threat → defensive reserve (cone heuristic from coordinated_strike_interceptor) ----
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

    # ---- Score every (source, target) pair ----
    capturable = [p for p in planets.values() if p[1] != player]
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

        # RD dominance at target cell: A - B > 0 means we activate there
        trow, tcol = _world_to_cell(tx, ty)
        target_dominance = A[trow][tcol] - B[trow][tcol]

        for src in my_planets:
            sid = src[0]
            sx, sy = src[2], src[3]
            spendable_max = src[5] - reserve.get(sid, MARGIN)
            if spendable_max < 1:
                continue
            if segment_hits_sun((sx, sy), (tx, ty)):
                continue

            # RD field at source: prefer launching from high-activator cells
            srow, scol = _world_to_cell(sx, sy)
            source_strength = A[srow][scol]

            def _solve(n, _tx=tx, _ty=ty, _tr=tr, _tid=tid):
                if is_comet:
                    return lead_solution((sx, sy), _tid, _tx, _ty, _tr, True,
                                         comets, ang_vel, step, n, max_spd,
                                         p_rotation_sign)
                if ip is not None:
                    return lead_solution((sx, sy), _tid, ip[2], ip[3], _tr, False,
                                         comets, ang_vel, step, n, max_spd,
                                         p_rotation_sign)
                d = math.hypot(_tx - sx, _ty - sy)
                e = max(1, int(d / fleet_speed(n, max_spd)))
                return (math.atan2(_ty - sy, _tx - sx), e, (_tx, _ty))

            sol = _solve(spendable_max)
            if sol is None:
                continue
            angle, eta, aim = sol

            garrison = tships if towner == -1 else tships + tprod * eta
            required = garrison + MARGIN
            ships_send = min(spendable_max, required)
            if ships_send < 1:
                continue

            # Refine with actual send count
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

            # Base score: production ROI, same formula as coordinated_strike_interceptor
            base = tprod / (required + 0.3 * eta + 1.0)
            # RD bonus: reward attacking into cells where our activator dominates
            rd_bonus = max(0.0, target_dominance) * 0.04
            # Source-strength nudge: prefer dispatching from entrenched positions
            rd_src   = max(0.0, source_strength) * 0.01
            score = base + rd_bonus + rd_src

            if step < 60 and towner == -1:
                d = math.hypot(tx - sx, ty - sy)
                score *= 1.5 / (1.0 + d * 0.01)
            if is_comet:
                score *= 1.2

            candidates.append((score, sid, tid, angle, eta, ships_send, required,
                               towner == -1))

    candidates.sort(key=lambda c: -c[0])

    # Greedy assignment (same as coordinated_strike_interceptor)
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
        used[sid]       += s
        committed[tid]   = already + s

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
