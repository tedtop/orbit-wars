# =============================================================================
# Orbit Wars bot: lyapunov_defense_heuristic
#
# Strategy: Lyapunov Exponent Defense + comet_wraith_v3 offense
#
# For each owned planet, tracks a Volatility Index each turn:
#   V = (enemy ships within 3*maxSpeed radius) / garrison
# where "enemy ships" includes in-flight enemy fleets (full weight) and latent
# garrison at enemy planets within range (half weight, haven't launched yet).
#
# A rolling history of V per planet is maintained. When a planet shows an
# exponential-divergence signature — V is high AND dV/dt is large positive
# (analogous to a positive Lyapunov exponent in chaotic systems) — it is
# flagged CRITICAL. The nearest safe (non-critical) owned planets immediately
# dispatch emergency reinforcements via lead_solution so they arrive before
# the projected attack.
#
# Offensive fallback: comet_wraith_v3's ROI scoring + greedy assignment
# ensures the bot always expands and never idles.
#
# Pure stdlib only. agent() is the last callable, crash-safe.
# =============================================================================

import math

_BOARD = 100.0
_CX = _CY = 50.0
_SUN_R = 10.0
_ROT_LIM = 50.0
_EPISODE = 500

_prev_angles = {}         # player_id → {planet_id → angle}
_rotation_sign = {}       # player_id → +1 or -1
_volatility_history = {}  # player_id → {planet_id → [V_t-n, ..., V_t]}

# ---- Lyapunov tuning ----
_V_CRIT        = 0.7   # V >= this AND rising fast → critical
_V_PANIC       = 1.5   # V >= this → critical regardless of slope (imminent attack)
_DV_CRIT       = 0.2   # dV/dt threshold for "exponentially rising"
_HIST_LEN      = 4     # rolling turns of V to keep
_MARGIN        = 1     # capture/reserve margin (comet_wraith_v3's lean edge)
_REINFORCE_FRAC = 0.55 # fraction of surplus to send as emergency reinforcement


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


def _decide(obs, config):
    global _prev_angles, _rotation_sign, _volatility_history

    max_spd    = float(_get(config, "shipSpeed", 6.0))
    episode    = int(_get(config, "episodeSteps", _EPISODE))
    player     = int(_get(obs, "player", 0))
    step       = int(_get(obs, "step", 0))
    turns_left = max(1, episode - step - 2)

    planets_raw = _get(obs, "planets", []) or []
    fleets_raw  = _get(obs, "fleets", []) or []
    init_raw    = _get(obs, "initial_planets", []) or []
    comets      = _get(obs, "comets", []) or []
    comet_pids  = set(_get(obs, "comet_planet_ids", []) or [])
    ang_vel     = float(_get(obs, "angular_velocity", 0.035))

    # Reset all per-game state at the start of each episode
    if step == 0:
        _prev_angles[player]        = {}
        _rotation_sign[player]      = 1
        _volatility_history[player] = {}

    p_prev_angles   = _prev_angles.get(player, {})
    p_rotation_sign = _rotation_sign.get(player, 1)

    planets    = {p[0]: p for p in planets_raw}
    init_by_id = {p[0]: p for p in init_raw}

    # ---- Rotation-sign inference (per player, copied from comet_wraith_v3) ----
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

    # ---- Standard flight-cone threat → per-planet reserve ----
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

    # ---- Lyapunov Volatility Index ----
    # Threat radius: how far ships can fly in 3 turns at maximum speed
    three_turn_r = 3.0 * max_spd

    p_vhist   = _volatility_history.setdefault(player, {})
    v_current = {}       # pid → V this turn
    critical_planets = set()

    for mp in my_planets:
        mpid      = mp[0]
        mpx, mpy  = mp[2], mp[3]
        garrison  = max(1, mp[5])

        # Active enemy fleet ships within 3-turn radius (full weight)
        enemy_power = 0.0
        for f in fleets_raw:
            if f[1] == player:
                continue
            if math.hypot(f[2] - mpx, f[3] - mpy) <= three_turn_r:
                enemy_power += f[6]

        # Latent enemy garrison within radius (half weight: not yet launched)
        for ep in planets.values():
            if ep[1] == player or ep[1] == -1:
                continue
            if math.hypot(ep[2] - mpx, ep[3] - mpy) <= three_turn_r:
                enemy_power += ep[5] * 0.5

        V = enemy_power / garrison
        v_current[mpid] = V

        # Rolling history
        hist = p_vhist.setdefault(mpid, [])
        hist.append(V)
        if len(hist) > _HIST_LEN:
            hist.pop(0)

        # dV/dt: compare current to previous tick
        dv = (hist[-1] - hist[-2]) if len(hist) >= 2 else 0.0

        # Positive-Lyapunov flag: exponential divergence or immediate panic
        if V >= _V_PANIC or (V >= _V_CRIT and dv >= _DV_CRIT):
            critical_planets.add(mpid)

    # ---- Emergency reinforcement for critical planets ----
    moves = []
    used  = {mp[0]: 0 for mp in my_planets}

    # Most endangered first
    critical_sorted = sorted(
        [mp for mp in my_planets if mp[0] in critical_planets],
        key=lambda mp: -v_current.get(mp[0], 0.0),
    )

    for tgt in critical_sorted:
        tpid       = tgt[0]
        tx, ty, tr = tgt[2], tgt[3], tgt[4]

        # Find nearest safe source: non-critical, has ships to spare
        sources = sorted(
            [mp for mp in my_planets
             if mp[0] != tpid and mp[0] not in critical_planets],
            key=lambda mp: math.hypot(mp[2] - tx, mp[3] - ty),
        )

        for src in sources:
            sid    = src[0]
            sx, sy = src[2], src[3]
            surplus = src[5] - used[sid] - reserve[sid]
            if surplus < 2:
                continue
            if segment_hits_sun((sx, sy), (tx, ty)):
                continue

            ships_to_send = max(1, int(surplus * _REINFORCE_FRAC))

            ip = init_by_id.get(tpid)
            if ip is not None:
                sol = lead_solution(
                    (sx, sy), tpid, ip[2], ip[3], tr, False,
                    comets, ang_vel, step, ships_to_send, max_spd, p_rotation_sign,
                )
            else:
                d = math.hypot(tx - sx, ty - sy)
                e = max(1, int(d / fleet_speed(ships_to_send, max_spd)))
                sol = (math.atan2(ty - sy, tx - sx), e, (tx, ty))

            if sol is None:
                continue
            angle, eta, aim = sol

            if segment_hits_sun((sx, sy), aim):
                continue

            moves.append([sid, angle, ships_to_send])
            used[sid] += ships_to_send
            break  # one reinforcing wave per critical planet per turn

    # ---- Offensive expansion (comet_wraith_v3 ROI scoring) ----
    scores = {}
    for p in planets.values():
        if p[1] != -1:
            scores[p[1]] = scores.get(p[1], 0) + p[5]
    for f in fleets_raw:
        scores[f[1]] = scores.get(f[1], 0) + f[6]
    opp_scores = {o: s for o, s in scores.items() if o != player}
    leader    = max(opp_scores, key=opp_scores.get) if opp_scores else None
    multi_opp = len(opp_scores) > 1

    candidates = []
    for tgt in capturable:
        tid                              = tgt[0]
        tx, ty, tr, tships, tprod, towner = (
            tgt[2], tgt[3], tgt[4], tgt[5], tgt[6], tgt[1])
        is_comet = tid in comet_pids
        ip       = init_by_id.get(tid)

        if is_comet:
            remaining = 0
            for grp in comets:
                if tid in grp["planet_ids"]:
                    i         = grp["planet_ids"].index(tid)
                    remaining = len(grp["paths"][i]) - grp["path_index"] - 1
                    break
            if remaining < 15:
                continue

        for src in my_planets:
            sid           = src[0]
            sx, sy        = src[2], src[3]
            spendable_max = src[5] - reserve[sid] - used[sid]
            if spendable_max < 1:
                continue
            if segment_hits_sun((sx, sy), (tx, ty)):
                continue

            def _solve(n):
                if is_comet:
                    return lead_solution(
                        (sx, sy), tid, tx, ty, tr, True,
                        comets, ang_vel, step, n, max_spd, p_rotation_sign,
                    )
                if ip is not None:
                    return lead_solution(
                        (sx, sy), tid, ip[2], ip[3], tr, False,
                        comets, ang_vel, step, n, max_spd, p_rotation_sign,
                    )
                d = math.hypot(tx - sx, ty - sy)
                e = max(1, int(d / fleet_speed(n, max_spd)))
                return (math.atan2(ty - sy, tx - sx), e, (tx, ty))

            sol = _solve(spendable_max)
            if sol is None:
                continue
            angle, eta, aim = sol

            garrison   = tships if towner == -1 else tships + tprod * eta
            required   = garrison + _MARGIN
            ships_send = min(spendable_max, required)
            if ships_send < 1:
                continue

            sol = _solve(ships_send)
            if sol is None:
                continue
            angle, eta, aim = sol
            if towner != -1:
                garrison   = tships + tprod * eta
                required   = garrison + _MARGIN
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
        avail = planets[sid][5] - reserve[sid] - used[sid]
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
