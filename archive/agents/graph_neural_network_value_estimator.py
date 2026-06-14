# =============================================================================
# Orbit Wars bot: graph_neural_network_value_estimator
#
# UNTRAINED, HAND-WEIGHTED GCN FORWARD PASS — no training data or training
# step. Domain-knowledge coefficients replace learned weights:
#   · High production  → high Strategic Value  (f0 = prod/5)
#   · Low garrison     → accessible target     (f1 = 1 − ships/200)
#   · Capturable type  → enemy/neutral bonus   (f2/f3 indicator)
#   · Cluster context  → 1/travel_time-weighted neighbour aggregation amplifies
#                        value for planets inside productive, capturable clusters
#
# Architecture (2 message-passing layers, fully-connected graph, O(N²), N≤40):
#   Features (4D):  f = [prod/5, 1−ships/200, is_enemy, is_neutral]
#   Layer 1: h1_i = relu(W1 @ (f_i + 0.5 · weighted_mean_j(f_j, 1/tt_ij)))
#            tt_ij = travel_time(i→j) using 50-ship reference speed
#   Layer 2: val_i = (tanh(w2 · h1_i) + 1) / 2   →  StrategicValue ∈ [0, 1]
#
# Decision: score(src, tgt) = StrategicValue[tgt] / (garrison_at_arrival + MARGIN)
#   with a 0.3·eta time-discount to prefer closer targets.
#   Attack the highest-scoring capturable planet(s) with lead-solution aiming.
#   Neutral planets always in the candidate pool so the bot never idles.
# =============================================================================

import math

_BOARD = 100.0
_CX = _CY = 50.0
_SUN_R = 10.0
_ROT_LIM = 50.0
_EPISODE = 500
_MARGIN = 1

_prev_angles = {}     # player_id → {planet_id → angle}
_rotation_sign = {}   # player_id → +1 or -1

# ---------------------------------------------------------------------------
# GCN hand-chosen weights (untrained — domain-knowledge substitution)
# ---------------------------------------------------------------------------
# Feature vector (4D): [prod/5, 1−ships/200, is_enemy, is_neutral]
#   f0: production rate (0.2..1.0) — higher means planet is worth owning
#   f1: accessibility  (1=bare, 0=heavy garrison) — higher means cheaper capture
#   f2: is_enemy indicator — enemy disrupts our production growth
#   f3: is_neutral indicator — neutrals are expansion opportunities

# Layer 1 (4→4): each row is a strategic "detector"
#   h0: production-focused   h1: garrison-ease   h2: capturable-type   h3: balanced
_W1 = [
    [0.50, 0.15, 0.09, 0.09],
    [0.12, 0.45, 0.06, 0.12],
    [0.09, 0.12, 0.48, 0.24],
    [0.30, 0.24, 0.21, 0.21],
]
_b1 = [0.0, 0.0, 0.0, 0.0]

# Layer 2 (4→1 scalar): emphasise capturable-type (h2) and production (h0)
_w2 = [0.25, 0.17, 0.30, 0.20]
_b2 = 0.0


# ---------------------------------------------------------------------------
# Helpers (copied from coordinated_strike_interceptor)
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
# GCN forward pass
# ---------------------------------------------------------------------------

def _gnn_values(planets_list, player, max_spd):
    """
    2-layer GCN forward pass with fixed hand-chosen weights.
    Returns {planet_id: strategic_value ∈ [0, 1]}.
    O(N²), N ≤ 40.
    """
    n = len(planets_list)
    if n == 0:
        return {}

    # Encode node features: [prod/5, 1−ships/200, is_enemy, is_neutral]
    feats = []
    for p in planets_list:
        owner, ships, prod = p[1], p[5], p[6]
        f0 = prod / 5.0
        f1 = 1.0 - min(max(float(ships), 0.0), 200.0) / 200.0
        f2 = 1.0 if (owner != -1 and owner != player) else 0.0
        f3 = 1.0 if owner == -1 else 0.0
        feats.append([f0, f1, f2, f3])

    # Edge weights: 1 / travel_time using 50-ship reference fleet
    ref_spd = fleet_speed(50, max_spd)
    edge_w = []
    for i in range(n):
        xi, yi = float(planets_list[i][2]), float(planets_list[i][3])
        row = []
        for j in range(n):
            if i == j:
                row.append(0.0)
            else:
                xj, yj = float(planets_list[j][2]), float(planets_list[j][3])
                d = math.hypot(xj - xi, yj - yi)
                row.append(1.0 / max(d / ref_spd, 1.0))
        edge_w.append(row)

    # Layer 1: weighted neighbourhood mean → combine with self → linear + relu
    h1 = []
    for i in range(n):
        total_w = sum(edge_w[i])
        if total_w > 1e-9:
            agg = [
                sum(edge_w[i][j] * feats[j][k] for j in range(n)) / total_w
                for k in range(4)
            ]
        else:
            agg = [0.0] * 4
        combined = [feats[i][k] + 0.5 * agg[k] for k in range(4)]
        raw = [
            sum(_W1[r][k] * combined[k] for k in range(4)) + _b1[r]
            for r in range(4)
        ]
        h1.append([max(0.0, v) for v in raw])

    # Layer 2: linear → tanh → map to [0, 1]
    values = {}
    for i, p in enumerate(planets_list):
        raw = sum(_w2[k] * h1[i][k] for k in range(4)) + _b2
        values[p[0]] = (math.tanh(raw) + 1.0) / 2.0

    return values


# ---------------------------------------------------------------------------
# Main decision logic
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

    # Rotation-sign inference (per player, from coordinated_strike_interceptor)
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

    # Threat → defensive reserve
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
    reserve = {mp[0]: max(threat[mp[0]] + _MARGIN, _MARGIN) for mp in my_planets}

    # GCN forward pass — one call per turn, covers all planets
    gnn_vals = _gnn_values(planets_raw, player, max_spd)

    # Score every (source, target) pair
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

        tgt_gnn = gnn_vals.get(tid, 0.5)

        for src in my_planets:
            sid = src[0]
            sx, sy = src[2], src[3]
            spendable_max = src[5] - reserve.get(sid, _MARGIN)
            if spendable_max < 1:
                continue
            if segment_hits_sun((sx, sy), (tx, ty)):
                continue

            def _solve(n):
                if is_comet:
                    return lead_solution(
                        (sx, sy), tid, tx, ty, tr, True,
                        comets, ang_vel, step, n, max_spd, p_rotation_sign)
                if ip is not None:
                    return lead_solution(
                        (sx, sy), tid, ip[2], ip[3], tr, False,
                        comets, ang_vel, step, n, max_spd, p_rotation_sign)
                d = math.hypot(tx - sx, ty - sy)
                e = max(1, int(d / fleet_speed(n, max_spd)))
                return (math.atan2(ty - sy, tx - sx), e, (tx, ty))

            sol = _solve(spendable_max)
            if sol is None:
                continue
            angle, eta, aim = sol

            garrison = tships if towner == -1 else tships + tprod * eta
            required = garrison + _MARGIN
            ships_send = min(spendable_max, required)
            if ships_send < 1:
                continue

            # Refine with actual ship count (speed depends on ships sent)
            sol = _solve(ships_send)
            if sol is None:
                continue
            angle, eta, aim = sol
            if towner != -1:
                garrison = tships + tprod * eta
                required = garrison + _MARGIN
                ships_send = min(spendable_max, required)
                if ships_send < 1:
                    continue

            if segment_hits_sun((sx, sy), aim):
                continue
            if path_blocked_by_planet((sx, sy), aim, planets_raw, {sid, tid}):
                continue

            # End-game gate: skip fleets that can't land (and pay back) in time
            if eta + 2 >= turns_left:
                continue
            if turns_left < 45:
                mult = 2.0 if towner != -1 else 1.0
                if tprod * (turns_left - eta) * mult <= garrison:
                    continue

            # GNN score: StrategicValue / CaptureCost
            # 0.3·eta time-discounts long-range attacks (closer targets preferred)
            score = tgt_gnn / (required + 0.3 * eta + 1.0)

            candidates.append((score, sid, tid, angle, eta, ships_send, required,
                               towner == -1))

    candidates.sort(key=lambda c: -c[0])

    # Greedy assignment (cumulative, matching coordinated_strike_interceptor)
    used = {mp[0]: 0 for mp in my_planets}
    committed = {}
    moves = []
    for score, sid, tid, angle, eta, ships_send, required, _is_neutral in candidates:
        already = committed.get(tid, 0)
        if already >= required:
            continue
        avail = planets[sid][5] - reserve.get(sid, _MARGIN) - used[sid]
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
