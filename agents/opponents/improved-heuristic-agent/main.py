"""Orbit Wars - Enhanced Heuristic Agent with Lead-Aim and Deception.

Techniques from top Kaggle leaderboard analysis + physics-based improvements:
1. Lead-aim: predict target position based on orbital mechanics
2. Fleet trajectory tracking (enemy fleet interception)
3. Reinforcement planning (defend under-attack planets)
4. Cooperative multi-source attacks with coordinated timing
5. Production snowball scoring with ownership multipliers
6. Dynamic ship count with game phase awareness
7. Sun collision avoidance
8. Deception: feints, diversionary attacks, overwhelm strikes
9. Mission priority: REINFORCE > FEINT > ATTACK
"""

import math

BOARD_SIZE = 100
SUN_X, SUN_Y = 50.0, 50.0
SUN_RADIUS = 10.0
VOID_MARGIN = 2.0
EPISODE_STEPS = 500
COOP_PLANET_CAP = 8

# Ownership multipliers for target scoring
ENEMY_MULT = 2.05
NEUTRAL_MULT = 1.4
CONTESTED_MULT = 0.7
FRIENDLY_MULT = 0.3

# Deception constants
FEINT_RATIO = 0.12
OVERWHELM_MULTIPLIER = 1.5
MIN_FEINT_SHIPS = 3
FEINT_COOLDOWN = 15  # Don't feint same target too often

# Planet position tracking for orbit prediction
_planet_history = {}  # planet_id -> [(step, x, y, owner)]
MAX_HISTORY = 8
_last_feint_target = None
_last_feint_step = -999


def _update_planet_history(planets, step):
    """Track planet positions over time for orbit prediction."""
    global _planet_history
    for p in planets:
        pid = p[0]
        if pid not in _planet_history:
            _planet_history[pid] = []
        history = _planet_history[pid]
        history.append((step, p[2], p[3], p[1]))
        if len(history) > MAX_HISTORY:
            _planet_history[pid] = history[-MAX_HISTORY:]


def _predict_planet_position(pid, future_step):
    """Predict where a planet will be at future_step using orbital mechanics.

    Planets orbit the sun in circular motion. We estimate angular velocity
    from the last two tracked positions and extrapolate.
    Returns (future_x, future_y) or None if prediction is impossible.
    """
    if pid not in _planet_history or len(_planet_history[pid]) < 2:
        return None

    history = _planet_history[pid]
    s1, x1, y1, _ = history[-2]
    s2, x2, y2, _ = history[-1]

    dt = s2 - s1
    if dt <= 0:
        return None

    a1 = math.atan2(y1 - SUN_Y, x1 - SUN_X)
    a2 = math.atan2(y2 - SUN_Y, x2 - SUN_X)

    da = a2 - a1
    if da > math.pi:
        da -= 2 * math.pi
    elif da < -math.pi:
        da += 2 * math.pi

    angular_velocity = da / dt
    steps_ahead = future_step - s2
    future_angle = a2 + angular_velocity * steps_ahead

    radius = math.hypot(x2 - SUN_X, y2 - SUN_Y)
    future_x = SUN_X + radius * math.cos(future_angle)
    future_y = SUN_Y + radius * math.sin(future_angle)

    return (future_x, future_y)


def estimate_fleet_speed(num_ships):
    """Fleet speed formula: speed = 1 + 5 * (log(ships)/log(1000))^1.5"""
    if num_ships <= 0:
        return 1.0
    return 1.0 + 5.0 * (math.log(max(num_ships, 1)) / math.log(1000)) ** 1.5


def detect_game_phase(step):
    """Return phase 0-3: early(0-15%), mid(15-50%), late(50-85%), finish(85-100%)"""
    r = step / EPISODE_STEPS
    if r < 0.15:
        return 0
    if r < 0.50:
        return 1
    if r < 0.85:
        return 2
    return 3


def point_to_segment_distance(p, a, b):
    """Minimum distance from point p to line segment ab."""
    dx, dy = b[0] - a[0], b[1] - a[1]
    len_sq = dx * dx + dy * dy
    if len_sq == 0:
        return math.hypot(p[0] - a[0], p[1] - a[1])
    t = max(0.0, min(1.0, ((p[0] - a[0]) * dx + (p[1] - a[1]) * dy) / len_sq))
    return math.hypot(p[0] - (a[0] + t * dx), p[1] - (a[1] + t * dy))


def shot_crosses_sun(sx, sy, angle, tx, ty, src_radius):
    """Check if a fleet launched at angle from (sx,sy) toward (tx,ty) crosses the sun."""
    start_x = sx + math.cos(angle) * (src_radius + 0.1)
    start_y = sy + math.sin(angle) * (src_radius + 0.1)
    return point_to_segment_distance((SUN_X, SUN_Y), (start_x, start_y), (tx, ty)) < SUN_RADIUS


def line_intersects_sun(x1, y1, x2, y2):
    """Check if straight line between two points crosses the sun."""
    return point_to_segment_distance((SUN_X, SUN_Y), (x1, y1), (x2, y2)) < SUN_RADIUS


def in_bounds(x, y):
    """Check if position is within the board with margin."""
    return VOID_MARGIN <= x <= BOARD_SIZE - VOID_MARGIN and VOID_MARGIN <= y <= BOARD_SIZE - VOID_MARGIN


def lead_aim(sx, sy, src_radius, target_id, num_ships, step):
    """Calculate aim angle accounting for target's orbital motion.

    Iteratively predicts where the target will be when the fleet arrives,
    using orbital angular velocity extrapolation.
    Returns (angle, predicted_x, predicted_y) or None if prediction fails or path crosses sun.
    """
    speed = estimate_fleet_speed(num_ships)

    predicted = _predict_planet_position(target_id, step + 5)
    if predicted is None:
        return None

    for _ in range(5):
        dist = math.hypot(predicted[0] - sx, predicted[1] - sy)
        travel_time = dist / speed if speed > 0 else 100
        new_predicted = _predict_planet_position(target_id, step + travel_time)
        if new_predicted is None:
            return None
        if abs(new_predicted[0] - predicted[0]) < 0.01 and abs(new_predicted[1] - predicted[1]) < 0.01:
            break
        predicted = new_predicted

    if not in_bounds(predicted[0], predicted[1]):
        return None

    angle = math.atan2(predicted[1] - sy, predicted[0] - sx)
    if shot_crosses_sun(sx, sy, angle, predicted[0], predicted[1], src_radius):
        return None

    return (angle, predicted[0], predicted[1])


def production_score(tgt_ships, tgt_prod, src_ships, src_prod,
                     tgt_owner, src_owner, dist, step):
    """Score a target by production snowball value."""
    remaining = max(EPISODE_STEPS - step, 1)
    ships_needed = tgt_ships + 1
    speed = estimate_fleet_speed(ships_needed)
    arrival = dist / speed if speed > 0 else 100.0
    effective_remaining = max(remaining - arrival, 1)
    base = tgt_prod * effective_remaining / (tgt_ships + arrival * 0.5 + 1)

    if tgt_owner == -1:
        base *= NEUTRAL_MULT
    elif tgt_owner != src_owner:
        base *= ENEMY_MULT
    else:
        base *= FRIENDLY_MULT
    return base


def track_fleet_arrivals(planets, fleets, player, step):
    """Build arrival map: planet_id -> [(owner, ships, arrival_turn)]."""
    arrivals = {}
    for f in fleets:
        fowner = f[1]
        dest = f[3]
        fships = f[6]
        fx, fy = f[4], f[5]
        if dest not in planets:
            continue
        tgt = planets[dest]
        tx, ty = tgt[2], tgt[3]
        dist = math.hypot(tx - fx, ty - fy)
        speed = estimate_fleet_speed(fships)
        arrival = step + dist / speed if speed > 0 else step + 100
        if dest not in arrivals:
            arrivals[dest] = []
        arrivals[dest].append((fowner, fships, arrival))
    return arrivals


def dynamic_ship_count(src_ships, src_prod, tgt_ships, tgt_prod, dist, step):
    """Determine how many ships to send based on game phase and distance."""
    min_needed = tgt_ships + 1
    speed = estimate_fleet_speed(min_needed)
    arrival = dist / speed if speed > 0 else 100.0
    prod_buffer = tgt_prod * int(arrival) + 5

    phase = detect_game_phase(step)
    multiplier = [1.2, 1.5, 2.0, 3.0][phase]
    needed = int((min_needed + prod_buffer) * multiplier)
    safety = max(int(src_prod * 5), 5)
    max_sendable = max(src_ships - safety, 0)
    return min(needed, max_sendable)


def agent(obs, cfg):
    """Main agent with lead-aim, deception, and tactical priorities.

    Mission priorities:
    1. REINFORCE - defend planets under imminent attack
    2. FEINT - send small diversionary fleets to trick opponents
    3. ATTACK - coordinated multi-source attacks with lead-aim

    Deception tactics:
    - Feint attacks: send ~12% of ships to secondary targets to draw enemy defenses
    - Lead-aim: predict where orbiting targets will be when fleet arrives
    - Overwhelm: send 1.5x needed ships in late game for decisive captures
    - Coordinated timing: bonus for attackable-from-many-sources targets
    """
    global _last_feint_target, _last_feint_step

    planets_raw = obs.get("planets", [])
    fleets_raw = obs.get("fleets", [])
    player = obs.get("player", 0)
    step = obs.get("step", 0)

    _update_planet_history(planets_raw, step)

    planets = {p[0]: p for p in planets_raw}
    my = {pid: p for pid, p in planets.items() if p[1] == player}
    enemies = {pid: p for pid, p in planets.items()
               if p[1] != player and p[1] != -1}
    neutral = {pid: p for pid, p in planets.items() if p[1] == -1}

    arrivals = track_fleet_arrivals(planets, fleets_raw, player, step)

    under_attack = {}
    for pid, p in my.items():
        if pid in arrivals:
            enemy_arriving = sum(
                s for o, s, t in arrivals[pid]
                if o != player and t <= step + 10
            )
            if enemy_arriving > p[5]:
                under_attack[pid] = enemy_arriving

    moves = []
    used_ships = {}

    # â”€â”€ PHASE 1: Reinforce under-attack planets (HIGHEST PRIORITY) â”€â”€
    for pid, enemy_ships in under_attack.items():
        p = my[pid]
        needed = enemy_ships - p[5] + 10

        reinf_candidates = []
        for sid, s in my.items():
            if sid == pid:
                continue
            available = s[5] - used_ships.get(sid, 0) - max(int(s[6] * 5), 5)
            if available <= 0:
                continue
            dist = math.hypot(s[2] - p[2], s[3] - p[3])
            if line_intersects_sun(s[2], s[3], p[2], p[3]) or not in_bounds(p[2], p[3]):
                continue
            reinf_candidates.append((dist, sid, s, available))

        reinf_candidates.sort()
        remaining = needed
        for dist, sid, s, available in reinf_candidates:
            if remaining <= 0:
                break
            send = min(available, remaining)
            if send > 0:
                angle = math.atan2(p[3] - s[3], p[2] - s[2])
                if not shot_crosses_sun(s[2], s[3], angle, p[2], p[3], s[4]):
                    moves.append([sid, angle, send])
                    used_ships[sid] = used_ships.get(sid, 0) + send
                    remaining -= send

    # â”€â”€ PHASE 2: Feint attacks (DECEPTION) â”€â”€
    phase = detect_game_phase(step)
    if phase >= 1 and len(enemies) >= 2:
        enemy_list = list(enemies.keys())
        enemy_scores = [(enemies[eid][6], eid) for eid in enemy_list]
        enemy_scores.sort(reverse=True)

        # Pick second most valuable enemy as feint target
        feint_tid = enemy_scores[1][1]
        if feint_tid != _last_feint_target or (step - _last_feint_step) > FEINT_COOLDOWN:
            ft = planets[feint_tid]
            feint_candidates = []
            for sid, s in my.items():
                available = s[5] - used_ships.get(sid, 0) - max(int(s[6] * 5), 5)
                feint_ships = max(int(available * FEINT_RATIO), MIN_FEINT_SHIPS)
                if feint_ships < MIN_FEINT_SHIPS or feint_ships > available:
                    continue
                dist = math.hypot(s[2] - ft[2], s[3] - ft[3])
                aim = lead_aim(s[2], s[3], s[4], feint_tid, feint_ships, step)
                if aim is not None:
                    feint_candidates.append((dist, sid, s, feint_ships, aim))

            if feint_candidates:
                feint_candidates.sort()
                _, sid, s, feint_ships, (angle, _, _) = feint_candidates[0]
                moves.append([sid, angle, feint_ships])
                used_ships[sid] = used_ships.get(sid, 0) + feint_ships
                _last_feint_target = feint_tid
                _last_feint_step = step

    # â”€â”€ PHASE 3: Score all targets and assign fleets (ATTACK) â”€â”€
    all_targets = list(enemies.keys()) + list(neutral.keys())
    target_assignments = []

    for sid, s in my.items():
        available = s[5] - used_ships.get(sid, 0)
        if available < 10:
            continue

        for tid in all_targets:
            t = planets[tid]
            sx, sy = s[2], s[3]

            # Lead-aim: predict future target position
            aim_result = lead_aim(sx, sy, s[4], tid, available, step)
            if aim_result is None:
                continue
            angle, predicted_x, predicted_y = aim_result
            dist = math.hypot(predicted_x - sx, predicted_y - sy)

            score = production_score(
                t[5], t[6], s[5], s[6], t[1], s[1], dist, step
            )

            # Penalize contested targets (enemy reinforcements coming)
            if tid in arrivals:
                enemy_reinf = sum(
                    ships for o, ships, _ in arrivals[tid] if o != player
                )
                if enemy_reinf > 0:
                    score *= CONTESTED_MULT

            # Bonus for coordinated attacks (multiple sources can reach)
            reachable_sources = sum(
                1 for sid2, s2 in my.items()
                if not line_intersects_sun(s2[2], s2[3], t[2], t[3])
            )
            if reachable_sources >= 3:
                score *= 1.3
            elif reachable_sources >= 2:
                score *= 1.15

            ships = dynamic_ship_count(
                s[5], s[6], t[5], t[6], dist, step
            )

            # Overwhelm: send extra ships in late game for decisive capture
            if phase >= 2 and t[1] != -1:
                ships = int(ships * OVERWHELM_MULTIPLIER)

            if 0 < ships <= available:
                target_assignments.append((score, sid, tid, ships, angle))

    target_assignments.sort(key=lambda x: -x[0])

    target_coop_count = {}
    for score, sid, tid, ships, angle in target_assignments:
        target_coop_count[tid] = target_coop_count.get(tid, 0) + 1
        if target_coop_count[tid] > COOP_PLANET_CAP:
            continue

        available = my[sid][5] - used_ships.get(sid, 0)
        send = min(ships, available)
        if send > 0:
            moves.append([sid, angle, send])
            used_ships[sid] = used_ships.get(sid, 0) + send

    return moves