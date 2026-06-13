# =============================================================================
# Orbit Wars bot: distributed_pid_controllers
#
# Control-theory strategy: each owned planet's garrison is a controlled process
# variable regulated toward a setpoint by an independent PID controller.
#
# Setpoints:
#   Frontline planets (distance to nearest enemy < FRONTLINE_DIST): SP = 50
#     → accumulate ships before striking; act as sinks for interior flow.
#   Interior planets (far from enemies): SP = 5
#     → shed excess ships forward; act as supply depots.
#
# Each turn per planet:
#   e(t) = garrison − SP        (positive = surplus above setpoint)
#   dispatch = Kp·e + Ki·∫e dt + Kd·de/dt
#   dispatch > 0 → planet has surplus; emit ships outward.
#   dispatch ≤ 0 → planet is in deficit; hold and accumulate.
#
# Routing:
#   Frontline surplus → attack best capturable target (lead solution, enough to
#     capture: strictly more than garrison-at-arrival).
#   Interior surplus → reinforce nearest frontline planet (PID-gated amount);
#     if no reachable frontline, expand to nearest capturable.
#
# PID integral and previous-error are stored per (player, planet_id) in module
# globals, reset when step == 0 to handle multi-game reuse.
# =============================================================================

import math

_BOARD = 100.0
_CX = _CY = 50.0
_SUN_R = 10.0
_ROT_LIM = 50.0
_EPISODE = 500

# Per-player rotation tracking (same pattern as comet_wraith_v3)
_prev_angles   = {}   # player_id → {planet_id → angle}
_rotation_sign = {}   # player_id → +1 or -1

# PID persistent state, keyed by player then planet_id
_pid_integral = {}    # player_id → {planet_id → float}
_pid_prev_err = {}    # player_id → {planet_id → float}

# Tuning constants
SP_FRONTLINE    = 50.0    # garrison setpoint for frontline planets
SP_INTERIOR     = 5.0     # garrison setpoint for interior planets
FRONTLINE_DIST  = 32.0    # distance to nearest enemy → classified as frontline
Kp              = 0.5     # proportional gain
Ki              = 0.03    # integral gain (mild anti-stall, prevents perpetual hold)
Kd              = 0.15    # derivative gain (damps oscillation on sharp garrison swings)
INTEGRAL_CAP    = 250.0   # windup clamp
MARGIN          = 1       # extra ships above garrison_at_arrival needed to capture


# ---------------------------------------------------------------------------
# Helpers (copied from comet_wraith_v3 for correctness)
# ---------------------------------------------------------------------------

def _get(obj, key, default=None):
    if isinstance(obj, dict):
        return obj.get(key, default)
    fn = getattr(obj, "get", None)
    if fn is not None and callable(fn) and not isinstance(obj, type):
        try:
            return fn(key, default)
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
        px, py = src[0] + t * ax, src[1] + t * ay
        if math.hypot(bx - px, by - py) < br + buffer:
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
        idx = grp["path_index"] + step_ahead
        if 0 <= idx < len(path):
            pos = path[idx]
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


# ---------------------------------------------------------------------------
# Core decision logic
# ---------------------------------------------------------------------------

def _decide(obs, config):
    global _prev_angles, _rotation_sign, _pid_integral, _pid_prev_err

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

    planets    = {p[0]: p for p in planets_raw}
    init_by_id = {p[0]: p for p in init_raw}

    # ---- Rotation-sign inference (per player) ----
    p_prev_angles   = _prev_angles.get(player, {})
    p_rotation_sign = _rotation_sign.get(player, 1)
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

    my_planets    = [p for p in planets.values() if p[1] == player]
    if not my_planets:
        return []
    enemy_planets = [p for p in planets.values() if p[1] not in (-1, player)]
    capturable    = [p for p in planets.values() if p[1] != player]

    # ---- Reset PID state at step 0 ----
    if step == 0:
        _pid_integral[player] = {}
        _pid_prev_err[player] = {}
    p_integral = _pid_integral.setdefault(player, {})
    p_prev_err = _pid_prev_err.setdefault(player, {})

    # ---- Classify each owned planet as frontline or interior ----
    planet_sp = {}
    for mp in my_planets:
        sid = mp[0]
        if not enemy_planets:
            planet_sp[sid] = SP_INTERIOR
        else:
            min_e_dist = min(math.hypot(ep[2] - mp[2], ep[3] - mp[3])
                             for ep in enemy_planets)
            planet_sp[sid] = SP_FRONTLINE if min_e_dist < FRONTLINE_DIST else SP_INTERIOR

    # ---- Threat: incoming enemy fleets (cone check, matches comet_wraith_v3) ----
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

    # ---- PID dispatch signal per planet ----
    # e = garrison − SP  (positive → surplus above setpoint)
    # dispatch > 0 → emit ships; dispatch ≤ 0 → hold
    dispatch = {}
    for mp in my_planets:
        sid      = mp[0]
        garrison = float(mp[5])
        sp       = planet_sp[sid]
        e        = garrison - sp

        integral = p_integral.get(sid, 0.0) + e
        integral = max(-INTEGRAL_CAP, min(INTEGRAL_CAP, integral))
        p_integral[sid] = integral

        prev_e = p_prev_err.get(sid, e)
        deriv  = e - prev_e
        p_prev_err[sid] = e

        dispatch[sid] = Kp * e + Ki * integral + Kd * deriv

    # ---- Helper: find best attack on a capturable target ----
    def _attack_best(sid, sx, sy, spendable):
        best_score = -1e18
        best_move  = None
        for tgt in capturable:
            tid   = tgt[0]
            tx, ty, tr = tgt[2], tgt[3], tgt[4]
            tships, tprod, towner = tgt[5], tgt[6], tgt[1]
            is_comet = tid in comet_pids

            if segment_hits_sun((sx, sy), (tx, ty)):
                continue

            # Skip comets about to leave the board
            if is_comet:
                remaining = 0
                for grp in comets:
                    if tid in grp["planet_ids"]:
                        i = grp["planet_ids"].index(tid)
                        remaining = len(grp["paths"][i]) - grp["path_index"] - 1
                        break
                if remaining < 15:
                    continue

            ip = init_by_id.get(tid)
            if is_comet:
                sol = lead_solution((sx, sy), tid, tx, ty, tr, True, comets,
                                    ang_vel, step, spendable, max_spd, p_rotation_sign)
            elif ip is not None:
                sol = lead_solution((sx, sy), tid, ip[2], ip[3], tr, False,
                                    comets, ang_vel, step, spendable, max_spd,
                                    p_rotation_sign)
            else:
                d   = math.hypot(tx - sx, ty - sy)
                eta = max(1, int(d / fleet_speed(spendable, max_spd)))
                sol = (math.atan2(ty - sy, tx - sx), eta, (tx, ty))

            if sol is None:
                continue
            angle, eta, aim = sol

            if eta + 2 >= turns_left:
                continue

            garrison_arr = tships if towner == -1 else tships + tprod * eta
            required = garrison_arr + MARGIN
            if required > spendable:
                continue

            if segment_hits_sun((sx, sy), aim):
                continue
            if path_blocked_by_planet((sx, sy), aim, planets_raw, {sid, tid}):
                continue

            score = tprod / (required + 0.3 * eta + 1.0)
            if score > best_score:
                best_score = score
                best_move  = [sid, angle, int(required)]

        return best_move

    # ---- Helper: route ships to nearest frontline planet (reinforce own) ----
    frontline_ids = {sid for sid, sp in planet_sp.items() if sp == SP_FRONTLINE}

    def _reinforce_frontline(sid, sx, sy, n_send):
        if not frontline_ids:
            return None
        # Prefer frontline planets that are still in deficit (below SP)
        sorted_fl = sorted(
            frontline_ids,
            key=lambda fid: (
                dispatch.get(fid, 0) > 0,         # deficit first (dispatch > 0 = hold)
                math.hypot(planets[fid][2] - sx, planets[fid][3] - sy)
                if fid in planets else 1e18
            )
        )
        for fl_pid in sorted_fl:
            fl = planets.get(fl_pid)
            if fl is None:
                continue
            tid = fl[0]
            tx, ty, tr = fl[2], fl[3], fl[4]

            if segment_hits_sun((sx, sy), (tx, ty)):
                continue

            ip = init_by_id.get(tid)
            if ip is not None:
                sol = lead_solution((sx, sy), tid, ip[2], ip[3], tr, False,
                                    comets, ang_vel, step, n_send, max_spd,
                                    p_rotation_sign)
            else:
                d   = math.hypot(tx - sx, ty - sy)
                eta = max(1, int(d / fleet_speed(n_send, max_spd)))
                sol = (math.atan2(ty - sy, tx - sx), eta, (tx, ty))

            if sol is None:
                continue
            angle, eta, aim = sol

            if segment_hits_sun((sx, sy), aim):
                continue
            if path_blocked_by_planet((sx, sy), aim, planets_raw, {sid, tid}):
                continue

            return [sid, angle, int(n_send)]
        return None

    # ---- Route surplus per planet ----
    # Process frontline first (decisive attackers), then interior (suppliers).
    ordered = sorted(my_planets, key=lambda p: -planet_sp.get(p[0], 0))
    moves   = []

    for mp in ordered:
        sid      = mp[0]
        garrison = mp[5]
        disp     = dispatch.get(sid, 0.0)

        if disp <= 0.0:
            continue   # deficit: hold and accumulate toward SP

        reserve   = max(int(threat.get(sid, 0)) + MARGIN, MARGIN)
        avail_raw = garrison - reserve
        if avail_raw < 1:
            continue

        if sid in frontline_ids:
            # Frontline: attack with full available surplus (decisive strike)
            move = _attack_best(sid, mp[2], mp[3], avail_raw)
            if move is not None:
                moves.append(move)
        else:
            # Interior: route PID-gated amount to frontline sink
            ships_to_dispatch = max(1, min(int(disp), avail_raw))
            move = _reinforce_frontline(sid, mp[2], mp[3], ships_to_dispatch)
            if move is None:
                # No reachable frontline → expand (capture nearest neutral/enemy)
                move = _attack_best(sid, mp[2], mp[3], avail_raw)
            if move is not None:
                moves.append(move)

    return moves


def agent(obs, config):
    """Crash-safe entry point — last callable in the file."""
    try:
        return _decide(obs, config)
    except Exception:
        import sys
        import traceback
        traceback.print_exc(file=sys.stderr)
        return []
