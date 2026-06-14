# =============================================================================
# Orbit Wars bot: cascading_classifier_regressor
# =============================================================================
# Two-stage Cascading Classifier-Regressor Chain (CRC).
#
# STAGE 1 — Binary Classifier (hand-weighted logistic regression)
#   For each (enemy_source, our_planet) pair, compute a threat probability:
#     f1 = 1/(1 + dist/20)             — closeness (range-normalised)
#     f2 = enemy_ships / (our_ships+1) — enemy dominance ratio
#     f3 = enemy_prod / (our_prod+1)   — production advantage
#     f4 = our_prod / 5.0              — target desirability (max prod = 5)
#   logit = -1.5 + 4.0*f1 + 2.0*f2 + 0.5*f3 + 1.0*f4
#   sigmoid(logit) > 0.5 (i.e. logit > 0) → predicted attack.
#
# STAGE 2 — Linear Regressor (hand-set coefficients)
#   For Stage-1-positive pairs:
#     predicted_ships = 0.8 * enemy_garrison + 0.3 * enemy_prod * eta + 2
#   Models a rational enemy committing ~80% of garrison plus production earned
#   during flight, with a small base constant.
#
# EXECUTION
#   Engine constraint: fleets collide only with planets, never other fleets —
#   mid-flight interception is impossible. Response is pre-emptive:
#     1. Counter-attack the threat source with
#        max(garrison_at_arrival + 1, predicted_ships + 1) ships.
#     2. If source is unreachable, reinforce the threatened planet instead.
#   Sane expansion greedy (coordinated_strike_interceptor scoring) fills spare capacity so
#   the bot never idles when no threats are predicted.
#
# No training data exists — both stages are deterministic forward passes with
# hand-chosen weights grounded in game mechanics.
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


# =============================================================================
# Stage 1 — Binary classifier
# =============================================================================
# Weights grounded in game mechanics: closeness and enemy strength dominate;
# production signals both attacker's capability and target's value.

_W0 = -1.5   # bias: default prior is "no attack"
_W1 = 4.0    # closeness is the primary predictor
_W2 = 2.0    # enemy dominance (ships ratio)
_W3 = 0.5    # production advantage
_W4 = 1.0    # target value (our prod / 5, since max prod = 5)


def _sigmoid(x):
    return 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, x))))


def _stage1_prob(dist, e_ships, e_prod, o_ships, o_prod):
    """Return P(enemy at dist with e_ships/e_prod attacks our planet with o_ships/o_prod)."""
    f1 = 1.0 / (1.0 + dist / 20.0)
    f2 = e_ships / (o_ships + 1.0)
    f3 = e_prod / (o_prod + 1.0)
    f4 = o_prod / 5.0
    return _sigmoid(_W0 + _W1 * f1 + _W2 * f2 + _W3 * f3 + _W4 * f4)


# =============================================================================
# Stage 2 — Linear regressor
# =============================================================================
# Predicts fleet size enemy will commit: ~80% of garrison plus a share of
# production earned during flight.  +2 is a minimum-engagement baseline.

_RA = 0.8
_RB = 0.3
_RC = 2.0


def _stage2_ships(e_garrison, e_prod, eta):
    return max(1, int(_RA * e_garrison + _RB * e_prod * max(1.0, eta) + _RC))


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

    enemy_planets = [p for p in planets.values() if p[1] != player and p[1] != -1]

    # ships committed so far from each of my planets
    used = {p[0]: 0 for p in my_planets}

    def avail(pid):
        return planets[pid][5] - used[pid]

    # ---- Stage 1 + Stage 2: build ranked threat pairs ----
    threat_pairs = []
    for o_tgt in my_planets:
        ox, oy, o_ships, o_prod = o_tgt[2], o_tgt[3], o_tgt[5], o_tgt[6]
        for e_src in enemy_planets:
            ex, ey, e_ships, e_prod = e_src[2], e_src[3], e_src[5], e_src[6]
            dist = math.hypot(ex - ox, ey - oy)
            prob = _stage1_prob(dist, e_ships, e_prod, o_ships, o_prod)
            if prob <= 0.5:
                continue
            est_send = max(1, int(e_ships * _RA))
            eta = max(1.0, dist / fleet_speed(est_send, max_spd))
            pred_ships = _stage2_ships(e_ships, e_prod, eta)
            threat_pairs.append((prob, e_src, o_tgt, pred_ships))

    threat_pairs.sort(key=lambda x: -x[0])

    targeted_enemies = set()
    moves = []

    # ---- Execute threat responses ----
    for prob, e_src, o_tgt, pred_ships in threat_pairs:
        esid = e_src[0]
        otid = o_tgt[0]

        # Primary: pre-emptively counter-attack the threat source.
        # Send max(garrison_at_arrival + 1, predicted_ships + 1) per spec.
        if esid not in targeted_enemies:
            e_init = init_by_id.get(esid)
            best = None
            best_score = -1.0

            for src in my_planets:
                sid = src[0]
                sx, sy = src[2], src[3]
                if avail(sid) < 1:
                    continue
                if segment_hits_sun((sx, sy), (e_src[2], e_src[3])):
                    continue

                ref_n = pred_ships + 1
                if e_init is not None:
                    sol = lead_solution(
                        (sx, sy), esid, e_init[2], e_init[3], e_src[4], False,
                        comets, ang_vel, step, ref_n, max_spd, p_rotation_sign)
                else:
                    d_e = math.hypot(e_src[2] - sx, e_src[3] - sy)
                    our_eta_e = max(1, int(d_e / fleet_speed(ref_n, max_spd)))
                    sol = (math.atan2(e_src[3] - sy, e_src[2] - sx),
                           our_eta_e, (e_src[2], e_src[3]))

                if sol is None:
                    continue
                angle_e, our_eta_e, aim_e = sol

                if segment_hits_sun((sx, sy), aim_e):
                    continue
                if path_blocked_by_planet((sx, sy), aim_e, planets_raw, {sid, esid}):
                    continue
                if our_eta_e + 2 >= turns_left:
                    continue

                garrison_at_arrival = e_src[5] + e_src[6] * our_eta_e
                need = max(int(garrison_at_arrival) + 1, pred_ships + 1)
                can = avail(sid)
                if can < need:
                    continue  # need a single planet that can afford it

                score = prob / (need + 0.1 * our_eta_e + 1.0)
                if score > best_score:
                    best_score = score
                    best = (sid, angle_e, need)

            if best:
                sid, angle_e, send = best
                moves.append([sid, angle_e, send])
                used[sid] += send
                targeted_enemies.add(esid)
                continue

        # Fallback: reinforce the threatened planet.
        reinforcement = max(0, pred_ships + 1 - o_tgt[5])
        if reinforcement <= 0:
            continue
        o_init = init_by_id.get(otid)
        for src in my_planets:
            sid = src[0]
            if sid == otid:
                continue
            sx, sy = src[2], src[3]
            can = avail(sid)
            if can < 1:
                continue
            if segment_hits_sun((sx, sy), (o_tgt[2], o_tgt[3])):
                continue

            if o_init is not None:
                sol = lead_solution(
                    (sx, sy), otid, o_init[2], o_init[3], o_tgt[4], False,
                    comets, ang_vel, step, reinforcement, max_spd, p_rotation_sign)
            else:
                d_o = math.hypot(o_tgt[2] - sx, o_tgt[3] - sy)
                our_eta_o = max(1, int(d_o / fleet_speed(reinforcement, max_spd)))
                sol = (math.atan2(o_tgt[3] - sy, o_tgt[2] - sx),
                       our_eta_o, (o_tgt[2], o_tgt[3]))

            if sol is None:
                continue
            angle_o, our_eta_o, aim_o = sol

            if segment_hits_sun((sx, sy), aim_o):
                continue
            if our_eta_o + 2 >= turns_left:
                continue

            send = min(can, reinforcement)
            if send < 1:
                continue
            moves.append([sid, angle_o, send])
            used[sid] += send
            break  # one source per threatened planet is sufficient

    # ---- Expansion fallback (greedy, coordinated_strike_interceptor scoring) ----
    MARGIN = 1
    exp_candidates = []

    for tgt in planets.values():
        if tgt[1] == player:
            continue
        tid = tgt[0]
        tx, ty, tr, tships, tprod, towner = tgt[2], tgt[3], tgt[4], tgt[5], tgt[6], tgt[1]
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
            if avail(sid) < MARGIN:
                continue
            if segment_hits_sun((sx, sy), (tx, ty)):
                continue

            # capture sx, sy by value in default args to avoid closure issues
            def _solve(n, _ip=ip, _is_comet=is_comet, _tid=tid,
                       _tx=tx, _ty=ty, _tr=tr, _sx=sx, _sy=sy):
                if _is_comet:
                    return lead_solution(
                        (_sx, _sy), _tid, _tx, _ty, _tr, True,
                        comets, ang_vel, step, n, max_spd, p_rotation_sign)
                if _ip is not None:
                    return lead_solution(
                        (_sx, _sy), _tid, _ip[2], _ip[3], _tr, False,
                        comets, ang_vel, step, n, max_spd, p_rotation_sign)
                d = math.hypot(_tx - _sx, _ty - _sy)
                e = max(1, int(d / fleet_speed(n, max_spd)))
                return (math.atan2(_ty - _sy, _tx - _sx), e, (_tx, _ty))

            sol = _solve(avail(sid))
            if sol is None:
                continue
            _, eta_est, _ = sol

            garrison = tships if towner == -1 else tships + tprod * eta_est
            required = garrison + MARGIN
            ships_send = min(avail(sid), required)
            if ships_send < 1:
                continue

            sol2 = _solve(ships_send)
            if sol2 is None:
                continue
            angle, eta, aim = sol2
            if towner != -1:
                garrison = tships + tprod * eta
                required = garrison + MARGIN
                ships_send = min(avail(sid), required)
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

            exp_candidates.append((score, sid, tid, angle, eta, ships_send, required))

    exp_candidates.sort(key=lambda c: -c[0])

    committed_exp = {}
    for score, sid, tid, angle, eta, ships_send, required in exp_candidates:
        already = committed_exp.get(tid, 0)
        if already >= required:
            continue
        can = avail(sid)
        if can < 1:
            continue
        s = min(can, required - already)
        if s < 1:
            continue
        moves.append([sid, angle, int(s)])
        used[sid] += s
        committed_exp[tid] = already + s

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
