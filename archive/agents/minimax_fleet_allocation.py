"""
Minimax Fleet Allocation
========================
Groups all non-comet planets into k=3 clusters via deterministic K-Means
(farthest-point seeding, 8 iterations). Identifies the most contested cluster
(highest combined our+enemy planet count) as the minimax arena.

Runs a depth-2 alpha-beta minimax over small candidate dispatches (branch=4)
within that cluster; leaf score = our ships+prod minus enemy ships+prod.
Instantaneous simplified combat is used inside the game tree (no turn-by-turn
flight simulation — fast enough for real-time use).

Outside the contested cluster and supplementing remaining planet capacity,
falls back to greedy expansion (coordinated_strike_interceptor style with lead solutions).
Falls back entirely to greedy when no contested cluster is found.

Timing: depth=2, branch=4 → 16 leaf evals; K-Means on ≤40 pts is negligible.
Well within the 1 s/turn budget.
"""

import math

_CX = _CY = 50.0
_SUN_R = 10.0
_ROT_LIM = 50.0
_EPISODE = 500

_prev_angles = {}    # player_id → {planet_id → angle}
_rotation_sign = {}  # player_id → +1 or -1


# ── Core helpers (copied from coordinated_strike_interceptor) ────────────────────────────────

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


def predict_planet_pos(init_x, init_y, radius, angular_velocity, abs_step,
                       rotation_sign=1):
    dx, dy = init_x - _CX, init_y - _CY
    r = math.hypot(dx, dy)
    if r + radius < _ROT_LIM:
        ang = math.atan2(dy, dx) + rotation_sign * angular_velocity * abs_step
        return (_CX + r * math.cos(ang), _CY + r * math.sin(ang))
    return (init_x, init_y)


def lead_solution(src_pos, init_x, init_y, tgt_radius, ang_vel,
                  current_step, ships, max_spd, rotation_sign):
    """Iterative lead solution for a static or orbiting planet.
    Returns (angle_rad, eta_ticks, aim_pos)."""
    cur = predict_planet_pos(init_x, init_y, tgt_radius, ang_vel,
                             current_step, rotation_sign)
    dist = math.hypot(cur[0] - src_pos[0], cur[1] - src_pos[1])
    t = max(1.0, dist / fleet_speed(ships, max_spd))
    for _ in range(8):
        t_int = max(1, int(round(t)))
        fut = predict_planet_pos(init_x, init_y, tgt_radius, ang_vel,
                                 current_step + t_int, rotation_sign)
        dist = math.hypot(fut[0] - src_pos[0], fut[1] - src_pos[1])
        new_t = max(1.0, dist / fleet_speed(ships, max_spd))
        if abs(new_t - t) < 0.05:
            break
        t = new_t
    t_int = max(1, int(round(t)))
    aim = predict_planet_pos(init_x, init_y, tgt_radius, ang_vel,
                             current_step + t_int, rotation_sign)
    return (math.atan2(aim[1] - src_pos[1], aim[0] - src_pos[0]), t_int, aim)


# ── K-Means (k=3, deterministic farthest-point seeding) ──────────────────────

def _kmeans(pts, k, iters=8):
    """pts = [(x, y, id), ...].  Returns {id: cluster_index}."""
    if not pts:
        return {}
    if len(pts) <= k:
        return {p[2]: i for i, p in enumerate(pts)}
    # Deterministic init: farthest-point seeding
    centers = [(pts[0][0], pts[0][1])]
    for _ in range(k - 1):
        best = max(pts, key=lambda p: min(
            (p[0] - c[0]) ** 2 + (p[1] - c[1]) ** 2 for c in centers))
        centers.append((best[0], best[1]))
    assign = {p[2]: 0 for p in pts}
    for _ in range(iters):
        new_assign = {}
        for p in pts:
            ci = min(range(k), key=lambda i: (
                (p[0] - centers[i][0]) ** 2 + (p[1] - centers[i][1]) ** 2))
            new_assign[p[2]] = ci
        sums = [[0.0, 0.0, 0] for _ in range(k)]
        for p in pts:
            c = new_assign[p[2]]
            sums[c][0] += p[0]; sums[c][1] += p[1]; sums[c][2] += 1
        new_centers = [
            (sums[i][0] / sums[i][2], sums[i][1] / sums[i][2])
            if sums[i][2] > 0 else centers[i]
            for i in range(k)
        ]
        if new_assign == assign:
            break
        assign, centers = new_assign, new_centers
    return assign


# ── Minimax game tree ─────────────────────────────────────────────────────────

def _make_state(planets_dict):
    """Snapshot for the game tree: {pid: [owner, ships, prod, x, y]}."""
    return {pid: [p[1], p[5], p[6], p[2], p[3]] for pid, p in planets_dict.items()}


def _apply_move(state, move, mover):
    """Shallow-copy state and apply one dispatch with instantaneous combat."""
    s = {pid: list(v) for pid, v in state.items()}
    if move is None:
        return s
    src_pid, tgt_pid, ships = move
    if src_pid not in s or tgt_pid not in s:
        return s
    src, tgt = s[src_pid], s[tgt_pid]
    if src[0] != mover or src[1] < ships or ships < 1:
        return s
    src[1] -= ships
    if tgt[0] == mover:
        tgt[1] += ships           # friendly reinforcement
    else:
        g = tgt[1]
        if ships > g:
            tgt[0] = mover        # capture; surplus stays
            tgt[1] = ships - g
        else:
            tgt[1] = max(0, g - ships)   # tie annihilates; defender keeps planet
    return s


def _cluster_score(state, player, cluster_pids):
    """Leaf eval: our (ships+prod) minus enemy (ships+prod) in the cluster."""
    our = enemy = 0
    for pid in cluster_pids:
        v = state.get(pid)
        if v is None:
            continue
        if v[0] == player:
            our += v[1] + v[2]
        elif v[0] != -1:
            enemy += v[1] + v[2]
    return our - enemy


def _gen_moves(state, mover, cluster_pids, max_spd, n):
    """Up to n candidate moves for `mover` attacking cluster targets.
    Always includes None (hold) as the first option."""
    moves = [None]
    targets = [(pid, v) for pid, v in state.items()
               if v[0] != mover and pid in cluster_pids]
    if not targets:
        return moves
    scored = []
    for src_pid, src in state.items():
        if src[0] != mover:
            continue
        avail = src[1] - 1      # keep at least 1 ship at home
        if avail < 1:
            continue
        sx, sy = src[3], src[4]
        for tgt_pid, tgt in targets:
            if segment_hits_sun((sx, sy), (tgt[3], tgt[4])):
                continue
            dist = math.hypot(tgt[3] - sx, tgt[4] - sy)
            ships = min(avail, tgt[1] + 1)
            if ships < 1:
                continue
            eta = max(1.0, dist / fleet_speed(ships, max_spd))
            score = (tgt[2] + 1.0) / (tgt[1] + 0.5 * eta + 1.0)
            scored.append((score, src_pid, tgt_pid, ships))
    scored.sort(key=lambda x: -x[0])
    for _, sp, tp, sh in scored[:n - 1]:
        moves.append((sp, tp, sh))
    return moves


def _minimax(state, depth, alpha, beta, is_max, player, enemy,
             cluster_pids, max_spd, branch):
    if depth == 0:
        return _cluster_score(state, player, cluster_pids), None
    mover = player if is_max else enemy
    moves = _gen_moves(state, mover, cluster_pids, max_spd, branch)
    best_val = -1e9 if is_max else 1e9
    best_move = moves[0]    # default to hold
    for mv in moves:
        ns = _apply_move(state, mv, mover)
        val, _ = _minimax(ns, depth - 1, alpha, beta, not is_max,
                          player, enemy, cluster_pids, max_spd, branch)
        if is_max:
            if val > best_val:
                best_val = val; best_move = mv
            alpha = max(alpha, val)
        else:
            if val < best_val:
                best_val = val; best_move = mv
            beta = min(beta, val)
        if beta <= alpha:
            break
    return best_val, best_move


# ── Greedy expansion (coordinated_strike_interceptor style) ──────────────────────────────────

def _greedy(planets, player, init_by_id, comet_pids, step, max_spd,
            ang_vel, rot_sign, turns_left, exclude_srcs=None):
    MARGIN = 1
    ex = exclude_srcs or set()
    my_pl = [p for p in planets.values() if p[1] == player]
    capturable = [p for p in planets.values()
                  if p[1] != player and p[0] not in comet_pids]
    cands = []
    for tgt in capturable:
        tid, towner = tgt[0], tgt[1]
        tx, ty, tr, tships, tprod = tgt[2], tgt[3], tgt[4], tgt[5], tgt[6]
        ip = init_by_id.get(tid)
        for src in my_pl:
            sid = src[0]
            if sid in ex:
                continue
            sx, sy = src[2], src[3]
            avail = src[5] - MARGIN
            if avail < 1:
                continue
            if segment_hits_sun((sx, sy), (tx, ty)):
                continue
            if ip is not None:
                sol = lead_solution((sx, sy), ip[2], ip[3], tr,
                                    ang_vel, step, avail, max_spd, rot_sign)
            else:
                d = math.hypot(tx - sx, ty - sy)
                e = max(1, int(d / fleet_speed(avail, max_spd)))
                sol = (math.atan2(ty - sy, tx - sx), e, (tx, ty))
            angle, eta, aim = sol
            if eta + 2 >= turns_left:
                continue
            garrison = tships if towner == -1 else tships + tprod * eta
            required = garrison + MARGIN
            ships_send = min(avail, required)
            if ships_send < 1:
                continue
            # Refine angle/eta with actual ship count
            if ip is not None:
                sol2 = lead_solution((sx, sy), ip[2], ip[3], tr,
                                     ang_vel, step, ships_send, max_spd, rot_sign)
                angle, eta, aim = sol2
                if towner != -1:
                    garrison = tships + tprod * eta
                    required = garrison + MARGIN
                    ships_send = min(avail, required)
                    if ships_send < 1:
                        continue
            if segment_hits_sun((sx, sy), aim):
                continue
            score = tprod / (required + 0.3 * eta + 1.0)
            cands.append((score, sid, angle, int(ships_send), required, tid))
    cands.sort(key=lambda c: -c[0])
    used = {}
    committed = {}
    out = []
    for _, sid, angle, ships_send, required, tid in cands:
        already = committed.get(tid, 0)
        if already >= required:
            continue
        src_p = planets.get(sid)
        if src_p is None:
            continue
        avail = src_p[5] - MARGIN - used.get(sid, 0)
        if avail < 1:
            continue
        s = min(avail, required - already)
        if s < 1:
            continue
        out.append([sid, angle, int(s)])
        used[sid] = used.get(sid, 0) + s
        committed[tid] = already + s
    return out


# ── Main decision logic ───────────────────────────────────────────────────────

def _decide(obs, config):
    global _prev_angles, _rotation_sign

    max_spd = float(_get(config, "shipSpeed", 6.0))
    episode = int(_get(config, "episodeSteps", _EPISODE))
    player = int(_get(obs, "player", 0))

    p_prev_angles = _prev_angles.get(player, {})
    p_rot_sign = _rotation_sign.get(player, 1)
    planets_raw = _get(obs, "planets", []) or []
    init_raw = _get(obs, "initial_planets", []) or []
    comet_pids = set(_get(obs, "comet_planet_ids", []) or [])
    ang_vel = float(_get(obs, "angular_velocity", 0.035))
    step = int(_get(obs, "step", 0))
    turns_left = max(1, episode - step - 2)

    planets = {p[0]: p for p in planets_raw}
    init_by_id = {p[0]: p for p in init_raw}

    # ── Rotation-sign inference (per player, same method as coordinated_strike_interceptor) ──
    cur_angles = {}
    for pid, p in planets.items():
        if pid in comet_pids:
            continue
        ip = init_by_id.get(pid)
        if ip is None:
            continue
        if math.hypot(ip[2] - _CX, ip[3] - _CY) + ip[4] < _ROT_LIM:
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
                p_rot_sign = 1 if avg > 0 else -1
    _prev_angles[player] = cur_angles
    _rotation_sign[player] = p_rot_sign

    my_planets = [p for p in planets.values() if p[1] == player]
    if not my_planets:
        return []

    enemy_set = {p[1] for p in planets.values() if p[1] not in (-1, player)}
    if not enemy_set:
        # No enemies on board — pure greedy expansion
        return _greedy(planets, player, init_by_id, comet_pids, step,
                       max_spd, ang_vel, p_rot_sign, turns_left)
    enemy = min(enemy_set)  # deterministic; focus on first enemy

    # ── K-Means clustering (k=3) ──────────────────────────────────────────────
    non_comets = [p for p in planets.values() if p[0] not in comet_pids]
    K = min(3, len(non_comets))
    if K < 2:
        return _greedy(planets, player, init_by_id, comet_pids, step,
                       max_spd, ang_vel, p_rot_sign, turns_left)

    pts = [(p[2], p[3], p[0]) for p in non_comets]
    assign = _kmeans(pts, K)

    # Count our and enemy planets per cluster
    stats = {}  # cid -> [our_count, enemy_count]
    for pid, cid in assign.items():
        p = planets.get(pid)
        if p is None:
            continue
        if cid not in stats:
            stats[cid] = [0, 0]
        if p[1] == player:
            stats[cid][0] += 1
        elif p[1] == enemy:
            stats[cid][1] += 1

    # Most contested = both sides present, maximise total presence
    contested = {c: s for c, s in stats.items() if s[0] > 0 and s[1] > 0}
    if not contested:
        return _greedy(planets, player, init_by_id, comet_pids, step,
                       max_spd, ang_vel, p_rot_sign, turns_left)

    best_cid = max(contested, key=lambda c: contested[c][0] + contested[c][1])
    cluster_pids = frozenset(pid for pid, cid in assign.items() if cid == best_cid)

    # ── Minimax (depth 2, branch 4 → 16 leaf evals) ───────────────────────────
    state = _make_state(planets)
    _, mm_move = _minimax(state, 2, -1e9, 1e9, True, player, enemy,
                          cluster_pids, max_spd, branch=4)

    moves = []
    mm_src = None

    if mm_move is not None:          # None means hold was the minimax choice
        src_pid, tgt_pid, ships = mm_move
        src = planets.get(src_pid)
        tgt = planets.get(tgt_pid)
        if (src is not None and tgt is not None
                and src[1] == player and src[5] >= ships >= 1):
            sx, sy = src[2], src[3]
            towner, tx, ty, tr, tships, tprod = (
                tgt[1], tgt[2], tgt[3], tgt[4], tgt[5], tgt[6])
            if not segment_hits_sun((sx, sy), (tx, ty)):
                ip = init_by_id.get(tgt_pid)
                if ip is not None:
                    sol = lead_solution((sx, sy), ip[2], ip[3], tr,
                                        ang_vel, step, ships, max_spd, p_rot_sign)
                else:
                    d = math.hypot(tx - sx, ty - sy)
                    e = max(1, int(d / fleet_speed(ships, max_spd)))
                    sol = (math.atan2(ty - sy, tx - sx), e, (tx, ty))
                angle, eta, aim = sol
                if not segment_hits_sun((sx, sy), aim) and eta + 2 < turns_left:
                    moves.append([src_pid, angle, int(ships)])
                    mm_src = src_pid

    # ── Greedy supplement for all other source planets ────────────────────────
    greedy = _greedy(planets, player, init_by_id, comet_pids, step,
                     max_spd, ang_vel, p_rot_sign, turns_left,
                     exclude_srcs={mm_src} if mm_src else None)
    used_srcs = {m[0] for m in moves}
    for gm in greedy:
        if gm[0] not in used_srcs:
            moves.append(gm)
            used_srcs.add(gm[0])

    return moves


def agent(obs, config):
    """Crash-safe entry point — last callable in this file."""
    try:
        return _decide(obs, config)
    except Exception:
        import sys, traceback
        traceback.print_exc(file=sys.stderr)
        return []
