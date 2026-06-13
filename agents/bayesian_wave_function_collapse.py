# =============================================================================
# Orbit Wars bot: bayesian_wave_function_collapse
#
# Strategy: Bayesian Wave Function Collapse (quantum / statistics)
#
# Treats the board as a superposition of possible futures. Runs NUM_ROLLOUTS
# cheap Monte Carlo simulations over HORIZON turns using random-but-logical
# enemy moves. Builds a probabilistic heat map:
#   heat_map[planet_id][turn] -> (avg_garrison, enemy_ownership_fraction)
#
# Bayesian inference: observed enemy fleet counts each turn update a per-enemy
# aggression prior (exponential moving average). This prior calibrates the
# per-turn attack probability in each rollout — "collapsing the wave function"
# toward the most likely enemy behaviour as observations arrive.
#
# Decision rule: for each (source -> target) pair, read the expected garrison
# at arrival from the heat map. Neutral targets that enemies frequently claim
# (high enemy_fraction) are de-prioritised via success_probability weighting.
# We rank by (success_prob * production) / (cost + 0.3*eta).
#
# Time budget: 25 rollouts x 7 turns x ~40 planets ≈ 7 000 ops/turn — well
# under 1 s. Pure stdlib. No numpy / torch.
# =============================================================================

import collections
import math
import random

_BOARD = 100.0
_CX = _CY = 50.0
_SUN_R = 10.0
_ROT_LIM = 50.0
_EPISODE = 500

NUM_ROLLOUTS = 25  # hard-capped for time budget
HORIZON = 7        # turns ahead per rollout

_prev_angles = {}      # player -> {pid -> angle}
_rotation_sign = {}    # player -> +1 or -1
_enemy_aggression = {} # player -> {enemy_id -> float attack-prob prior}


# ---------------------------------------------------------------------------
# Helpers — copied / trimmed from coordinated_strike_interceptor
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
        if math.hypot(bx - src[0] - t * ax, by - src[1] - t * ay) < br + buffer:
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


def lead_solution(src_pos, init_x, init_y, tgt_radius,
                  angular_velocity, current_step, ships, max_spd, rotation_sign):
    """Return (angle, eta, aim_pos) for a non-comet planet, or None."""
    cur = predict_planet_pos(init_x, init_y, tgt_radius, angular_velocity,
                             current_step, rotation_sign)
    dist = math.hypot(cur[0] - src_pos[0], cur[1] - src_pos[1])
    t = max(1.0, dist / fleet_speed(ships, max_spd))
    for _ in range(8):
        t_int = max(1, int(round(t)))
        fut = predict_planet_pos(init_x, init_y, tgt_radius, angular_velocity,
                                 current_step + t_int, rotation_sign)
        dist = math.hypot(fut[0] - src_pos[0], fut[1] - src_pos[1])
        new_t = max(1.0, dist / fleet_speed(ships, max_spd))
        if abs(new_t - t) < 0.05:
            break
        t = new_t
    t_int = max(1, int(round(t)))
    aim = predict_planet_pos(init_x, init_y, tgt_radius, angular_velocity,
                             current_step + t_int, rotation_sign)
    return (math.atan2(aim[1] - src_pos[1], aim[0] - src_pos[0]), t_int, aim)


# ---------------------------------------------------------------------------
# Rollout engine
# ---------------------------------------------------------------------------

def _estimate_fleet_target(fl, planets_dict, max_spd):
    """Best-guess (planet_id, eta) for an in-transit fleet from angle matching."""
    fx, fy, fangle = fl[2], fl[3], fl[4]
    fships = fl[6]
    best_pid, best_score = None, float('-inf')
    for pid, p in planets_dict.items():
        if p[1] == fl[1]:
            continue
        px, py = p[2], p[3]
        expected = math.atan2(py - fy, px - fx)
        diff = fangle - expected
        diff -= 2.0 * math.pi * round(diff / (2.0 * math.pi))
        if abs(diff) > 0.6:
            continue
        dist = math.hypot(px - fx, py - fy)
        score = -abs(diff) * 5.0 - dist * 0.05
        if score > best_score:
            best_score, best_pid = score, pid
    if best_pid is None:
        return None, None
    px, py = planets_dict[best_pid][2], planets_dict[best_pid][3]
    dist = math.hypot(px - fx, py - fy)
    eta = max(1, int(dist / fleet_speed(fships, max_spd)))
    return best_pid, eta


def _run_rollouts(sim_p, sim_f_in, player, enemies, horizon, max_spd, attack_probs):
    """
    Monte Carlo rollout engine.

    sim_p      : {pid: [owner, ships, x, y, radius, prod]}
    sim_f_in   : [[owner, tgt_pid, ships, eta_remaining]]   in-transit enemy fleets
    attack_probs: {enemy_id: float}  per-turn attack probability (Bayesian prior)

    Returns {pid: {t: (avg_ships, enemy_frac)}} for t in 1..horizon.
    enemy_frac = fraction of rollouts where pid is owned by any enemy at time t.
    """
    pid_list = list(sim_p.keys())
    # acc[pid][t] = [sum_ships, sum_enemy_owned_indicator]
    acc = {pid: [[0.0, 0.0] for _ in range(horizon + 1)] for pid in pid_list}

    rng = random.Random()

    for roll in range(NUM_ROLLOUTS):
        rng.seed(roll * 6271 + 17)

        p = {pid: list(d) for pid, d in sim_p.items()}
        f = [list(fl) for fl in sim_f_in]

        for t in range(1, horizon + 1):
            # Each enemy randomly expands with probability from Bayesian prior
            for enemy in enemies:
                if rng.random() > attack_probs.get(enemy, 0.5):
                    continue
                ep = [pid for pid, pd in p.items() if pd[0] == enemy and pd[1] > 5]
                targets = [pid for pid, pd in p.items() if pd[0] != enemy]
                if not ep or not targets:
                    continue
                src_pid = rng.choice(ep)
                tgt_pid = rng.choice(targets)
                sx, sy = p[src_pid][2], p[src_pid][3]
                tx, ty = p[tgt_pid][2], p[tgt_pid][3]
                if segment_hits_sun((sx, sy), (tx, ty)):
                    continue
                ships = max(1, p[src_pid][1] // 2)
                dist = math.hypot(tx - sx, ty - sy)
                eta = max(1, int(dist / fleet_speed(ships, max_spd)))
                p[src_pid][1] = max(0, p[src_pid][1] - ships)
                f.append([enemy, tgt_pid, ships, eta])

            # Decrement fleet ETAs; collect arrivals
            arrivals = collections.defaultdict(list)
            next_f = []
            for fl in f:
                fl[3] -= 1
                if fl[3] <= 0:
                    if fl[1] in p:
                        arrivals[fl[1]].append((fl[0], fl[2]))
                else:
                    next_f.append(fl)
            f = next_f

            # Combat resolution (largest faction vs sum of rest)
            for pid, arrivers in arrivals.items():
                planet = p[pid]
                by_owner = collections.defaultdict(int)
                by_owner[planet[0]] += planet[1]
                for ow, sh in arrivers:
                    by_owner[ow] += sh
                sorted_ow = sorted(by_owner.items(), key=lambda x: -x[1])
                if len(sorted_ow) == 1:
                    planet[0], planet[1] = sorted_ow[0]
                else:
                    top_ow, top_sh = sorted_ow[0]
                    second_sh = sum(s for _, s in sorted_ow[1:])
                    surplus = top_sh - second_sh
                    if surplus > 0:
                        planet[0], planet[1] = top_ow, surplus
                    else:
                        planet[1] = 0   # mutual annihilation; owner unchanged

            # Production tick
            for planet in p.values():
                if planet[0] != -1:
                    planet[1] += planet[5]

            # Snapshot for heat map
            for pid in pid_list:
                planet = p.get(pid)
                if planet:
                    acc[pid][t][0] += planet[1]
                    if planet[0] in enemies:
                        acc[pid][t][1] += 1.0

    result = {}
    for pid in pid_list:
        result[pid] = {
            t: (acc[pid][t][0] / NUM_ROLLOUTS, acc[pid][t][1] / NUM_ROLLOUTS)
            for t in range(1, horizon + 1)
        }
    return result


# ---------------------------------------------------------------------------
# Decision
# ---------------------------------------------------------------------------

def _decide(obs, config):
    global _prev_angles, _rotation_sign, _enemy_aggression

    max_spd = float(_get(config, "shipSpeed", 6.0))
    episode = int(_get(config, "episodeSteps", _EPISODE))
    player = int(_get(obs, "player", 0))
    step = int(_get(obs, "step", 0))

    if step == 0:
        _prev_angles[player] = {}
        _rotation_sign[player] = 1
        _enemy_aggression[player] = {}

    p_prev_angles = _prev_angles.get(player, {})
    p_rotation_sign = _rotation_sign.get(player, 1)
    p_enemy_aggr = _enemy_aggression.get(player, {})

    planets_raw = _get(obs, "planets", []) or []
    fleets_raw = _get(obs, "fleets", []) or []
    init_raw = _get(obs, "initial_planets", []) or []
    comet_pids = set(_get(obs, "comet_planet_ids", []) or [])
    ang_vel = float(_get(obs, "angular_velocity", 0.035))
    turns_left = max(1, episode - step - 2)

    planets = {p[0]: p for p in planets_raw}
    init_by_id = {p[0]: p for p in init_raw}

    # ---- Rotation-sign inference (per-player, from coordinated_strike_interceptor) ----
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

    # ---- Identify enemies; update Bayesian aggression prior ----
    enemies = set()
    for p in planets.values():
        if p[1] not in (-1, player):
            enemies.add(p[1])
    for f in fleets_raw:
        if f[1] not in (-1, player):
            enemies.add(f[1])

    fleet_cnt = collections.Counter(f[1] for f in fleets_raw if f[1] in enemies)
    for enemy in enemies:
        n_owned = sum(1 for p in planets.values() if p[1] == enemy)
        obs_aggr = min(1.0, fleet_cnt.get(enemy, 0) / max(1, n_owned) * 0.7)
        prior = p_enemy_aggr.get(enemy, 0.5)
        p_enemy_aggr[enemy] = 0.7 * prior + 0.3 * obs_aggr
    _enemy_aggression[player] = p_enemy_aggr

    # ---- Build sim state (exclude comets — too volatile) ----
    sim_p = {
        pid: [p[1], p[5], p[2], p[3], p[4], p[6]]
        for pid, p in planets.items()
        if pid not in comet_pids
    }
    sim_f = []
    for fl in fleets_raw:
        if fl[1] == player:
            continue
        tgt_pid, eta = _estimate_fleet_target(fl, planets, max_spd)
        if tgt_pid is not None and tgt_pid in sim_p and 1 <= eta <= HORIZON:
            sim_f.append([fl[1], tgt_pid, fl[6], eta])

    # ---- Monte Carlo rollouts -> probabilistic heat map ----
    attack_probs = {e: p_enemy_aggr.get(e, 0.5) for e in enemies}
    heat_map = _run_rollouts(sim_p, sim_f, player, enemies, HORIZON, max_spd, attack_probs)

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

    # ---- Score (source, target) candidates ----
    capturable = [p for p in planets.values() if p[1] != player]
    candidates = []

    for tgt in capturable:
        tid = tgt[0]
        if tid in comet_pids:
            continue
        tx, ty, tr, tships, tprod, towner = tgt[2], tgt[3], tgt[4], tgt[5], tgt[6], tgt[1]
        ip = init_by_id.get(tid)
        if ip is None:
            continue

        for src in my_planets:
            sid = src[0]
            sx, sy = src[2], src[3]
            spendable_max = src[5] - reserve.get(sid, MARGIN)
            if spendable_max < 1:
                continue
            if segment_hits_sun((sx, sy), (tx, ty)):
                continue

            # First solve with max ships to estimate ETA for heat-map lookup
            sol = lead_solution((sx, sy), ip[2], ip[3], tr,
                                ang_vel, step, spendable_max, max_spd, p_rotation_sign)
            if sol is None:
                continue
            _, eta_est, _ = sol

            # Bayesian garrison estimate at arrival turn
            eta_key = min(eta_est, HORIZON)
            hm = heat_map.get(tid, {}).get(eta_key)
            if hm:
                avg_sh, enemy_frac = hm
            else:
                avg_sh = tships if towner == -1 else tships + tprod * eta_est
                enemy_frac = 0.0 if towner == -1 else 1.0

            garrison_est = max(0.0, avg_sh)
            required = max(1, int(garrison_est) + MARGIN)
            ships_send = min(spendable_max, required)
            if ships_send < 1:
                continue

            # Second solve with actual ships_send for accurate angle and ETA
            sol = lead_solution((sx, sy), ip[2], ip[3], tr,
                                ang_vel, step, ships_send, max_spd, p_rotation_sign)
            if sol is None:
                continue
            angle, eta, aim = sol

            if segment_hits_sun((sx, sy), aim):
                continue
            if path_blocked_by_planet((sx, sy), aim, planets_raw, {sid, tid}):
                continue
            if eta + 2 >= turns_left:
                continue
            if turns_left < 45:
                mult = 2.0 if towner != -1 else 1.0
                if tprod * (turns_left - eta) * mult <= garrison_est:
                    continue

            # Neutral targets contested by enemies → lower success probability
            if towner == -1:
                success_prob = max(0.05, 1.0 - enemy_frac)
            else:
                # Enemy planet: simulation always shows enemy owning it (no our attack sim);
                # success_prob=1 lets garrison_est drive the cost comparison instead.
                success_prob = 1.0

            score = success_prob * tprod / (required + 0.3 * eta + 1.0)
            if step < 60 and towner == -1:
                d = math.hypot(tx - sx, ty - sy)
                score *= 1.5 / (1.0 + d * 0.01)

            candidates.append((score, sid, tid, angle, ships_send, required))

    candidates.sort(key=lambda c: -c[0])

    # ---- Greedy ship assignment ----
    used = {mp[0]: 0 for mp in my_planets}
    committed = {}
    moves = []
    for score, sid, tid, angle, ships_send, required in candidates:
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
    """Crash-safe entry point (last callable)."""
    try:
        return _decide(obs, config)
    except Exception:
        import sys
        import traceback
        traceback.print_exc(file=sys.stderr)
        return []
