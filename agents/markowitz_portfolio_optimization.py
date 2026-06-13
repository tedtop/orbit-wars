# =============================================================================
# Orbit Wars bot: markowitz_portfolio_optimization
#
# Quantitative-finance framing: attacks are treated as investments on a
# mean-variance efficient frontier.
#
#   Expected Return (mu)   = target's production value (dividend per turn)
#   Risk proxy (sigma^2)   = weighted sum of:
#       * ETA risk     — longer flight leaves more time for garrison growth
#       * Enemy threat — nearby enemy planets that could reinforce before arrival
#       * Fleet risk   — enemy fleets already en-route toward the target
#
# Instead of concentrating everything on one target (high variance), the bot
# solves the uncorrelated-asset Markowitz optimum:
#
#   maximise  sum_i [ w_i * mu_i  -  (lambda/2) * w_i^2 * sigma2_i ]
#
# Closed-form optimal weights (uncorrelated case):
#   w_i*  ∝  mu_i / (lambda * sigma2_i)
#
# A basket of up to 5 targets is selected (positive MV score only), ships are
# allocated proportionally to w_i* with a floor of the minimum required to
# guarantee capture (garrison-at-arrival + 1).  Surplus budget is divided by
# the same weights so high-Sharpe targets receive a larger top-up.
#
# Helpers copied verbatim from coordinated_strike_interceptor. Pure stdlib. Crash-safe.
# =============================================================================

import math

_BOARD = 100.0
_CX = _CY = 50.0
_SUN_R = 10.0
_ROT_LIM = 50.0
_EPISODE = 500

# Portfolio hyper-parameters
_LAMBDA = 0.8     # risk-aversion coefficient (higher → fewer, safer attacks)
_BASKET_MAX = 5   # maximum portfolio width
_MARGIN = 1       # ships above garrison-at-arrival needed to guarantee capture

_prev_angles = {}    # player_id → {planet_id → angle}
_rotation_sign = {}  # player_id → +1 or -1


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


# ---------------------------------------------------------------------------
# Decision logic
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

    # ---- Rotation-sign inference (identical to coordinated_strike_interceptor) ----
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

    # ---- Defensive reserve: match incoming threats + margin ----
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

    source_pool = {
        mp[0]: max(0, mp[5] - reserve.get(mp[0], _MARGIN))
        for mp in my_planets
    }
    total_available = sum(source_pool.values())
    if total_available < 1:
        return []

    # Inline lead solver capturing loop-local coords; called with explicit args to
    # avoid closure surprises when we reuse it in the dispatch phase.
    def _solve_lead(sx, sy, tid, tgt, n):
        tx, ty, tr = tgt[2], tgt[3], tgt[4]
        is_c = tid in comet_pids
        ip = init_by_id.get(tid)
        if is_c:
            return lead_solution((sx, sy), tid, tx, ty, tr, True, comets,
                                 ang_vel, step, n, max_spd, p_rotation_sign)
        if ip is not None:
            return lead_solution((sx, sy), tid, ip[2], ip[3], tr, False,
                                 comets, ang_vel, step, n, max_spd, p_rotation_sign)
        d = math.hypot(tx - sx, ty - sy)
        e = max(1, int(d / fleet_speed(n, max_spd)))
        return (math.atan2(ty - sy, tx - sx), e, (tx, ty))

    # ---- Phase 1: Score each target on the mean-variance frontier ----
    # For each target, find the best-fitting source planet (maximises spendable
    # minus required — the widest safety margin) and compute portfolio metrics.
    scored = []  # (mv_score, tid, sid, angle, eta, required, mu, sigma2)

    for tgt in capturable:
        tid = tgt[0]
        tx, ty, tr, tships, tprod, towner = (
            tgt[2], tgt[3], tgt[4], tgt[5], tgt[6], tgt[1]
        )
        is_comet = tid in comet_pids

        if is_comet:
            remaining = 0
            for grp in comets:
                if tid in grp["planet_ids"]:
                    idx = grp["planet_ids"].index(tid)
                    remaining = len(grp["paths"][idx]) - grp["path_index"] - 1
                    break
            if remaining < 15:
                continue

        best_sid = None
        best_angle = best_eta = best_aim = best_required = None
        best_excess = -1

        for src in my_planets:
            sid = src[0]
            sx, sy = src[2], src[3]
            spendable = source_pool[sid]
            if spendable < 1:
                continue
            if segment_hits_sun((sx, sy), (tx, ty)):
                continue

            sol = _solve_lead(sx, sy, tid, tgt, spendable)
            if sol is None:
                continue
            angle, eta, aim = sol

            garrison = tships if towner == -1 else tships + tprod * eta
            required = int(garrison) + _MARGIN
            ships_try = min(spendable, required)
            if ships_try < 1:
                continue

            # Refine with actual fleet size (affects speed → eta → garrison)
            sol = _solve_lead(sx, sy, tid, tgt, ships_try)
            if sol is None:
                continue
            angle, eta, aim = sol
            if towner != -1:
                garrison = tships + tprod * eta
                required = int(garrison) + _MARGIN
                ships_try = min(spendable, required)
                if ships_try < 1:
                    continue

            if spendable < required:
                continue  # can't fund the minimum capture cost from this source
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

            # Maximise surplus: wider margin → less chance of miscapture
            excess = spendable - required
            if excess > best_excess:
                best_excess = excess
                best_sid = sid
                best_angle = angle
                best_eta = eta
                best_aim = aim
                best_required = required

        if best_sid is None:
            continue

        # ---- Markowitz return and variance for this target ----
        mu = float(tprod)  # expected return = production dividend

        # ETA risk: quadratic penalty — farther targets grow their garrison longer
        eta_risk = (best_eta / 50.0) ** 2

        # Enemy-nearby risk: enemy planets that can reinforce before our fleet lands
        # (within 2 * ETA * maxSpeed effective radius from aim point)
        reinforce_range = best_eta * max_spd * 2.0
        enemy_nearby = 0.0
        for ep in planets.values():
            if ep[1] == player or ep[1] == -1:
                continue
            d_ep = math.hypot(ep[2] - best_aim[0], ep[3] - best_aim[1])
            if d_ep < reinforce_range:
                prox = 1.0 - d_ep / max(reinforce_range, 1.0)
                enemy_nearby += ep[5] * prox

        # Fleet risk: enemy fleets roughly aimed at the target
        fleet_risk = 0.0
        for f in fleets_raw:
            if f[1] == player:
                continue
            expected_fa = math.atan2(best_aim[1] - f[3], best_aim[0] - f[2])
            diff = f[4] - expected_fa
            diff -= 2.0 * math.pi * round(diff / (2.0 * math.pi))
            if abs(diff) < 0.5:
                fleet_risk += f[6]

        # Normalise threat components by minimum required fleet (scale-invariant)
        norm = max(float(best_required), 1.0)
        sigma2 = max(0.01,
                     eta_risk
                     + 0.3 * min(enemy_nearby / norm, 5.0)
                     + 0.1 * min(fleet_risk / norm, 5.0))

        mv_score = mu - _LAMBDA * sigma2

        # Early-game proximity boost for neutral planets (matches coordinated_strike_interceptor)
        if step < 60 and towner == -1:
            src_p = planets[best_sid]
            d = math.hypot(tx - src_p[2], ty - src_p[3])
            boost = 1.5 / (1.0 + d * 0.01)
            mv_score *= boost

        if is_comet:
            mv_score *= 1.2

        scored.append((mv_score, tid, best_sid, best_angle, best_eta,
                       best_required, mu, sigma2))

    if not scored:
        return []

    # ---- Phase 2: Select efficient-frontier basket ----
    scored.sort(key=lambda x: -x[0])
    viable = [s for s in scored if s[0] > 0] or scored[:1]
    basket = viable[:_BASKET_MAX]

    # ---- Phase 3: Markowitz weight allocation ----
    # Uncorrelated closed-form: w_i* ∝ mu_i / (lambda * sigma2_i)
    raw_w = [mu / (_LAMBDA * sigma2) for _, _, _, _, _, _, mu, sigma2 in basket]
    total_w = sum(raw_w) or 1.0
    norm_w = [w / total_w for w in raw_w]

    # Each target gets max(required_i, proportional share of total budget)
    allocs = [
        max(item[5], int(norm_w[i] * total_available))
        for i, item in enumerate(basket)
    ]

    # If total exceeds budget: retain minimums, scale down the surplus evenly
    total_alloc = sum(allocs)
    mins = [item[5] for item in basket]
    total_min = sum(mins)
    if total_alloc > total_available:
        if total_min <= total_available:
            surplus_budget = total_available - total_min
            excess = [max(0, a - m) for a, m in zip(allocs, mins)]
            total_excess = sum(excess) or 1.0
            allocs = [
                m + int(e * surplus_budget / total_excess)
                for m, e in zip(mins, excess)
            ]
        else:
            # Can't fund all minimums; trim basket greedily from the top
            trimmed_basket, trimmed_allocs = [], []
            remaining = total_available
            for i, item in enumerate(basket):
                if mins[i] <= remaining:
                    trimmed_basket.append(item)
                    trimmed_allocs.append(mins[i])
                    remaining -= mins[i]
            basket, allocs = trimmed_basket, trimmed_allocs

    # ---- Phase 4: Dispatch moves (one source per target) ----
    source_remaining = dict(source_pool)
    moves = []

    for i, item in enumerate(basket):
        if i >= len(allocs):
            break
        _, tid, preferred_sid, _, _, required, _, _ = item
        target_alloc = allocs[i]
        tgt = planets[tid]
        tships, tprod, towner = tgt[5], tgt[6], tgt[1]

        # Prefer the scored best source; fall back to richest alternative
        if source_remaining.get(preferred_sid, 0) >= required:
            use_sid = preferred_sid
        else:
            use_sid = None
            for mp in sorted(my_planets,
                             key=lambda p: -source_remaining.get(p[0], 0)):
                if source_remaining.get(mp[0], 0) >= required:
                    use_sid = mp[0]
                    break
            if use_sid is None:
                continue

        send = min(source_remaining[use_sid], target_alloc)
        if send < required:
            continue

        src = planets[use_sid]
        sx, sy = src[2], src[3]
        sol = _solve_lead(sx, sy, tid, tgt, send)
        if sol is None:
            continue
        move_angle, move_eta, aim = sol

        # Final safety checks with actual trajectory
        if segment_hits_sun((sx, sy), aim):
            continue
        if path_blocked_by_planet((sx, sy), aim, planets_raw, {use_sid, tid}):
            continue

        # Confirm capture is still guaranteed with this source's ETA
        garrison_at_arrival = tships if towner == -1 else tships + tprod * move_eta
        if send <= garrison_at_arrival:
            continue

        moves.append([use_sid, move_angle, int(send)])
        source_remaining[use_sid] -= send

    return moves


def agent(obs, config):
    """Crash-safe entry point (last callable)."""
    try:
        return _decide(obs, config)
    except Exception:  # noqa: BLE001
        import sys
        import traceback
        traceback.print_exc(file=sys.stderr)
        return []
