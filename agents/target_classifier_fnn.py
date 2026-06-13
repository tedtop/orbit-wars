# =============================================================================
# Orbit Wars bot: target_classifier_fnn
#
# Strategy: Untrained hand-weighted FNN (2-layer, sigmoid output).
# No training data exists; weights are hand-tuned so P(Attack) is high when an
# enemy planet has a large garrison, is close to one of our frontier planets,
# and that frontier planet is weakly defended. The forward pass is pure Python
# arithmetic — no torch, no numpy.
#
# Decision loop each turn:
#   1. For every enemy planet extract 4 normalized features:
#      [enemy_garrison/100, proximity (1 − dist/141), our_weakness (1 − garrison/100), turn/500]
#   2. Run a 4-input → 4-hidden → 1-output sigmoid FNN with hand-chosen weights.
#   3. If P(Attack) > 0.85 for any enemy planet, reinforce our nearest frontier
#      planet by pulling spare ships from neighbors (lead-solution aiming).
#      Reinforced planet's reserve is raised to match the threat level.
#   4. Run normal greedy offensive expansion (comet_wraith_v3-style) from
#      remaining capacity — ensures the bot always expands and never idles.
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
    dist_cur = math.hypot(cur[0] - src_pos[0], cur[1] - src_pos[1])
    t = max(1.0, dist_cur / fleet_speed(ships, max_spd))
    for _ in range(8):
        t_int = max(1, int(round(t)))
        if is_comet:
            fut = predict_comet_pos(tgt_pid, comets, t_int)
            if fut is None:
                return None
        else:
            fut = predict_planet_pos(init_x, init_y, tgt_radius, angular_velocity,
                                     current_step + t_int, rotation_sign)
        dist_fut = math.hypot(fut[0] - src_pos[0], fut[1] - src_pos[1])
        new_t = max(1.0, dist_fut / fleet_speed(ships, max_spd))
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


# ---- Hand-weighted FNN (untrained, fixed weights) ----
# Input features x = [enemy_garrison/100, proximity, our_weakness, turn/500]
# where proximity = 1 - dist/141.4  (1.0 = touching, 0.0 = opposite corner)
# and   our_weakness = 1 - our_garrison/100  (1.0 = 0 ships, 0.0 = 100 ships)
#
# Hidden layer neurons detect:
#   h0 — large garrison + close (h_garrison_proximity)
#   h1 — our planet is weak vs sizeable enemy (h_weakness)
#   h2 — combined threat: garrison + proximity + weakness (h_combined)
#   h3 — general pressure including time (h_pressure)
#
# Output bias calibrated so P > 0.85 only fires for genuinely dangerous planets
# (e.g., 100 ships at distance 10 with our 5 ships nearby) and stays low for
# weak far-away planets (e.g., 10 ships at distance 70 with our 50 ships).
_W1 = [
    [2.5, 2.0, 0.5, 0.2],   # h0: garrison + proximity
    [0.5, 1.0, 3.0, 0.3],   # h1: our weakness
    [2.0, 2.5, 2.5, 0.5],   # h2: combined threat
    [1.0, 1.0, 1.0, 1.5],   # h3: general pressure with time
]
_b1 = [-1.5, -1.5, -3.5, -1.5]
_W2 = [3.5, 2.5, 4.0, 1.5]
_b2 = -6.5


def _sigmoid(x):
    if x < -20.0:
        return 0.0
    if x > 20.0:
        return 1.0
    return 1.0 / (1.0 + math.exp(-x))


def _fnn_forward(x):
    """Two-layer FNN forward pass. x: 4 normalized floats → P(Attack) in [0,1]."""
    h = [_sigmoid(sum(_W1[i][j] * x[j] for j in range(4)) + _b1[i])
         for i in range(4)]
    return _sigmoid(sum(_W2[i] * h[i] for i in range(4)) + _b2)


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
    enemy_planets = [p for p in planets.values()
                     if p[1] not in (-1, player) and p[0] not in comet_pids]

    # ---- Cone-based threat → defensive reserve (v3-proven) ----
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

    # ---- FNN threat classification → raise reserve on threatened frontline ----
    # Also accumulate how many extra ships each frontline planet needs so
    # neighbors know how many to divert.
    fnn_extra_needed = {}  # frontline planet id → extra ships wanted
    for ep in enemy_planets:
        ex, ey, eships = ep[2], ep[3], ep[5]
        nearest = min(my_planets,
                      key=lambda mp: math.hypot(mp[2] - ex, mp[3] - ey))
        dist_to_us = math.hypot(nearest[2] - ex, nearest[3] - ey)

        x0 = min(eships / 100.0, 2.0)                          # enemy garrison
        x1 = 1.0 - min(dist_to_us / 141.4, 1.0)               # proximity
        x2 = 1.0 - min(nearest[5] / 100.0, 1.0)               # our weakness
        x3 = step / 500.0                                       # turn progress

        p_attack = _fnn_forward([x0, x1, x2, x3])

        if p_attack > 0.85:
            nid = nearest[0]
            extra = max(1, eships - nearest[5] + MARGIN)
            fnn_extra_needed[nid] = max(fnn_extra_needed.get(nid, 0), extra)
            # Raise reserve so this planet keeps enough ships to survive
            reserve[nid] = max(reserve[nid], eships + MARGIN)

    # ---- Scores for 4-player focus ----
    scores = {}
    for p in planets.values():
        if p[1] != -1:
            scores[p[1]] = scores.get(p[1], 0) + p[5]
    for f in fleets_raw:
        scores[f[1]] = scores.get(f[1], 0) + f[6]
    opp_scores = {o: s for o, s in scores.items() if o != player}
    leader = max(opp_scores, key=opp_scores.get) if opp_scores else None
    multi_opp = len(opp_scores) > 1

    # ---- Shared ship-usage tracker ----
    used = {mp[0]: 0 for mp in my_planets}
    moves = []

    # ---- Reinforcement moves (FNN-triggered, highest priority) ----
    # Pull spare ships from neighbors toward each threatened frontline planet.
    for frontline_pid, extra_needed in fnn_extra_needed.items():
        frontline = planets.get(frontline_pid)
        if frontline is None:
            continue
        fp_init = init_by_id.get(frontline_pid)
        remaining_needed = extra_needed

        # Donate from closest neighbors first
        neighbors = sorted(
            [mp for mp in my_planets if mp[0] != frontline_pid],
            key=lambda mp: math.hypot(mp[2] - frontline[2], mp[3] - frontline[3])
        )
        for src in neighbors:
            if remaining_needed <= 0:
                break
            sid = src[0]
            avail = src[5] - reserve.get(sid, MARGIN) - used[sid]
            if avail < 1:
                continue
            s = min(avail, remaining_needed)

            if fp_init is not None:
                sol = lead_solution(
                    (src[2], src[3]), frontline_pid,
                    fp_init[2], fp_init[3], frontline[4], False,
                    comets, ang_vel, step, s, max_spd, p_rotation_sign
                )
            else:
                angle = math.atan2(frontline[3] - src[3], frontline[2] - src[2])
                sol = (angle, 1, (frontline[2], frontline[3]))

            if sol is None:
                continue
            angle, eta, aim = sol

            if segment_hits_sun((src[2], src[3]), aim):
                continue

            moves.append([sid, angle, int(s)])
            used[sid] += s
            remaining_needed -= s

    # ---- Offensive expansion (comet_wraith_v3-style greedy scoring) ----
    candidates = []
    for tgt in capturable:
        tid = tgt[0]
        tx, ty, tr, tships, tprod, towner = (tgt[2], tgt[3], tgt[4], tgt[5],
                                             tgt[6], tgt[1])
        is_comet = tid in comet_pids
        ip = init_by_id.get(tid)

        if is_comet:
            comet_remaining = 0
            for grp in comets:
                if tid in grp["planet_ids"]:
                    idx = grp["planet_ids"].index(tid)
                    comet_remaining = len(grp["paths"][idx]) - grp["path_index"] - 1
                    break
            if comet_remaining < 15:
                continue

        for src in my_planets:
            sid = src[0]
            sx, sy = src[2], src[3]
            spendable_max = src[5] - reserve.get(sid, MARGIN) - used[sid]
            if spendable_max < 1:
                continue
            if segment_hits_sun((sx, sy), (tx, ty)):
                continue

            def _solve(n):
                if is_comet:
                    return lead_solution((sx, sy), tid, tx, ty, tr, True, comets,
                                         ang_vel, step, n, max_spd, p_rotation_sign)
                if ip is not None:
                    return lead_solution((sx, sy), tid, ip[2], ip[3], tr, False,
                                         comets, ang_vel, step, n, max_spd,
                                         p_rotation_sign)
                d = math.hypot(tx - sx, ty - sy)
                e = max(1, int(d / fleet_speed(n, max_spd)))
                return (math.atan2(ty - sy, tx - sx), e, (tx, ty))

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

            candidates.append((score, sid, tid, angle, eta, ships_send, required,
                               towner == -1))

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
    """Crash-safe entry point — must be the last callable in the file."""
    try:
        return _decide(obs, config)
    except Exception:
        import sys
        import traceback
        traceback.print_exc(file=sys.stderr)
        return []
