# Reference code extracted from orbit-wars-physics-helper-module.ipynb (NOT a playable agent).
# May contain plotting/analysis cells; kept verbatim for study.

import math
import logging
from typing import Optional, Tuple

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)

# %% ---- next cell ----

# ── Official constants (from spec) ────────────────────────────────────────────
BOARD_SIZE            = 100.0
CENTER_X              = 50.0
CENTER_Y              = 50.0
CENTER                = 50.0
SUN_RADIUS            = 10.0
SUN_SAFETY            = 1.5        # conservative buffer on top of actual radius
MAX_SHIP_SPEED        = 6.0
ROTATION_LIMIT        = 50.0       # orbital_radius + planet_radius >= 50 → static
LAUNCH_CLEARANCE      = 0.1
INTERCEPT_TOLERANCE   = 1          # compat alias
ROUTE_SEARCH_HORIZON  = 150        # v7: raised from 90 (covers speed=1 over dist=150)
HORIZON               = 110
EPISODE_STEPS         = 500
COMET_MAX_CHASE_TURNS = 10

ANG_VEL_MIN = 0.025
ANG_VEL_MAX = 0.050

_FWD_ITER_MAX     = 16             # max convergence iterations for the iterative solver
_EDGE_AIM_FRACS   = (0.25, 0.50, 0.75, 0.95)  # fractional offsets used for arc aim-point sampling

# %% ---- next cell ----

def dist(ax: float, ay: float, bx: float, by: float) -> float:
    return math.hypot(ax - bx, ay - by)

# %% ---- next cell ----

def orbital_radius(px: float, py: float) -> float:
    return dist(px, py, CENTER_X, CENTER_Y)

# %% ---- next cell ----

def is_static_planet(px: float, py: float, radius: float) -> bool:
    """Static if orbital_radius + planet_radius >= ROTATION_LIMIT (50)."""
    return orbital_radius(px, py) + radius >= ROTATION_LIMIT

# %% ---- next cell ----

def fleet_speed(ships: int) -> float:
    """Exact formula from spec: speed = 1 + (maxSpeed-1) * (log(ships)/log(1000))^1.5"""
    if ships <= 1:
        return 1.0
    ratio = math.log(ships) / math.log(1000.0)
    ratio = max(0.0, min(1.0, ratio))
    return min(1.0 + (MAX_SHIP_SPEED - 1.0) * ratio ** 1.5, MAX_SHIP_SPEED)

# Quick illustration
for n in [1, 10, 50, 100, 500, 1000]:
    print(f"  {n:>5} ships → speed = {fleet_speed(n):.3f} units/turn")

# %% ---- next cell ----

def point_to_segment_distance(px: float, py: float,
                               x1: float, y1: float,
                               x2: float, y2: float) -> float:
    dx, dy   = x2 - x1, y2 - y1
    seg_sq   = dx * dx + dy * dy
    if seg_sq < 1e-9:
        return dist(px, py, x1, y1)
    t = ((px - x1) * dx + (py - y1) * dy) / seg_sq
    t = max(0.0, min(1.0, t))
    return dist(px, py, x1 + t * dx, y1 + t * dy)

# %% ---- next cell ----

def segment_intersects_circle(ax: float, ay: float,
                               bx: float, by: float,
                               cx: float, cy: float,
                               r: float) -> bool:
    """True if segment A→B passes within radius r of centre C."""
    return point_to_segment_distance(cx, cy, ax, ay, bx, by) <= r

# %% ---- next cell ----

def segment_hits_sun(x1: float, y1: float,
                     x2: float, y2: float,
                     safety: float = SUN_SAFETY) -> bool:
    return point_to_segment_distance(
        CENTER_X, CENTER_Y, x1, y1, x2, y2
    ) < SUN_RADIUS + safety

def is_path_clear(sx: float, sy: float,
                  tx: float, ty: float) -> bool:
    return not segment_hits_sun(sx, sy, tx, ty)

# %% ---- next cell ----

def launch_point(sx: float, sy: float,
                 sr: float, angle: float) -> Tuple[float, float]:
    c = sr + LAUNCH_CLEARANCE
    return sx + math.cos(angle) * c, sy + math.sin(angle) * c

# %% ---- next cell ----

def predict_planet_position(planet_id: int,
                             cur_x: float, cur_y: float, radius: float,
                             initial_by_id: dict,
                             angular_velocity: float,
                             turns_ahead: int) -> Tuple[float, float]:
    init = initial_by_id.get(planet_id)
    if init is None:
        return cur_x, cur_y
    ix, iy = init['x'], init['y']
    r = dist(ix, iy, CENTER_X, CENTER_Y)
    if r + radius >= ROTATION_LIMIT:
        return cur_x, cur_y
    cur_ang = math.atan2(cur_y - CENTER_Y, cur_x - CENTER_X)
    new_ang  = cur_ang + angular_velocity * turns_ahead
    return CENTER_X + r * math.cos(new_ang), CENTER_Y + r * math.sin(new_ang)

# %% ---- next cell ----

def predict_planet_pos(ix: float, iy: float, radius: float,
                       cur_x: float, cur_y: float,
                       angular_velocity: float,
                       turns_ahead: int) -> Tuple[float, float]:
    if radius >= 5.0:
        return cur_x, cur_y
    initial_by_id = {0: {'x': ix, 'y': iy}}
    return predict_planet_position(
        0, cur_x, cur_y, radius, initial_by_id, angular_velocity, turns_ahead)

# %% ---- next cell ----

def predict_comet_position(planet_id: int, comets: list,
                           turns: int) -> Optional[Tuple[float, float]]:
    for group in comets:
        pids = group.get('planet_ids', [])
        if planet_id not in pids:
            continue
        idx        = pids.index(planet_id)
        paths      = group.get('paths', [])
        path_index = group.get('path_index', 0)
        if idx >= len(paths):
            return None
        future = path_index + int(turns)
        path   = paths[idx]
        if 0 <= future < len(path):
            return path[future][0], path[future][1]
        return None
    return None

# %% ---- next cell ----

def comet_remaining_life(planet_id: int, comets: list) -> int:
    for group in comets:
        pids = group.get('planet_ids', [])
        if planet_id not in pids:
            continue
        idx        = pids.index(planet_id)
        paths      = group.get('paths', [])
        path_index = group.get('path_index', 0)
        if idx < len(paths):
            return max(0, len(paths[idx]) - path_index)
    return 0

# %% ---- next cell ----

def predict_target_position(planet_id: int,
                             cur_x: float, cur_y: float, radius: float,
                             initial_by_id: dict,
                             angular_velocity: float,
                             comets: list, comet_ids: set,
                             turns: int) -> Optional[Tuple[float, float]]:
    if planet_id in comet_ids:
        return predict_comet_position(planet_id, comets, turns)
    return predict_planet_position(
        planet_id, cur_x, cur_y, radius,
        initial_by_id, angular_velocity, turns)

# %% ---- next cell ----

def target_can_move(planet_id: int,
                    cur_x: float, cur_y: float, radius: float,
                    initial_by_id: dict, comet_ids: set) -> bool:
    if planet_id in comet_ids:
        return True
    init = initial_by_id.get(planet_id)
    if init is None:
        return False
    r = dist(init['x'], init['y'], CENTER_X, CENTER_Y)
    return r + radius < ROTATION_LIMIT

# %% ---- next cell ----

def safe_angle_and_distance(sx: float, sy: float, sr: float,
                             tx: float, ty: float, tr: float,
                             ) -> Optional[Tuple[float, float]]:
    angle    = math.atan2(ty - sy, tx - sx)
    lx, ly   = launch_point(sx, sy, sr, angle)
    hit_dist = max(0.0, dist(sx, sy, tx, ty) - sr - LAUNCH_CLEARANCE - tr)
    ex = lx + math.cos(angle) * hit_dist
    ey = ly + math.sin(angle) * hit_dist
    if segment_hits_sun(lx, ly, ex, ey):
        return None
    return angle, hit_dist

# %% ---- next cell ----

def _fractional_turns(total_d: float, ships: int) -> float:
    return total_d / fleet_speed(max(1, ships))

def estimate_arrival(sx: float, sy: float, sr: float,
                     tx: float, ty: float, tr: float,
                     ships: int) -> Optional[Tuple[float, int]]:
    result = safe_angle_and_distance(sx, sy, sr, tx, ty, tr)
    if result is None:
        return None
    angle, total_d = result
    turns = max(1, int(math.ceil(_fractional_turns(total_d, ships))))
    return angle, turns

def estimate_arrival_frac(sx: float, sy: float, sr: float,
                          tx: float, ty: float, tr: float,
                          ships: int) -> Optional[Tuple[float, float]]:
    result = safe_angle_and_distance(sx, sy, sr, tx, ty, tr)
    if result is None:
        return None
    angle, total_d = result
    return angle, max(1.0, _fractional_turns(total_d, ships))

# %% ---- next cell ----

def travel_time(sx: float, sy: float, sr: float,
                tx: float, ty: float, tr: float,
                ships: int) -> int:
    est = estimate_arrival(sx, sy, sr, tx, ty, tr, ships)
    return est[1] if est is not None else 10 ** 9

# %% ---- next cell ----

def _fwd_window(turns: int) -> int:
    """v7: window = max(8, turns // 2) — provides enough headroom for slow fleets."""
    return max(8, turns // 2)

# %% ---- next cell ----

def arc_safe_angle(sx: float, sy: float, sr: float,
                   tx: float, ty: float, tr: float,
                   ships: int) -> Optional[Tuple[float, int]]:
    dx, dy = tx - sx, ty - sy
    d = math.hypot(dx, dy)
    if d < 1e-9:
        return None
    ux, uy = dx / d, dy / d
    nx, ny = -uy, ux

    aim_points = [(tx, ty)]
    for frac in _EDGE_AIM_FRACS:
        off = tr * frac
        aim_points.append((tx + nx * off, ty + ny * off))
        aim_points.append((tx - nx * off, ty - ny * off))

    best = None
    for ax, ay in aim_points:
        angle      = math.atan2(ay - sy, ax - sx)
        lx, ly     = launch_point(sx, sy, sr, angle)
        rvx, rvy   = math.cos(angle), math.sin(angle)
        cx, cy     = tx - lx, ty - ly
        proj       = cx * rvx + cy * rvy
        closest_sq = cx * cx + cy * cy - proj * proj
        if proj <= 0.0 or closest_sq > tr * tr:
            continue
        entry_dist = max(0.0, proj - math.sqrt(max(0.0, tr * tr - closest_sq)))
        ex = lx + rvx * entry_dist
        ey = ly + rvy * entry_dist
        if segment_hits_sun(lx, ly, ex, ey):
            continue
        turns = max(1, int(math.ceil(entry_dist / fleet_speed(max(1, ships)))))
        score = (turns, entry_dist)
        if best is None or score < best[0]:
            best = (score, angle, turns)

    return (best[1], best[2]) if best else None

# %% ---- next cell ----

def _verify_shot_hits(sx: float, sy: float, sr: float,
                      angle: float, turns: int, ships: int,
                      target_id: int,
                      tx: float, ty: float, tr: float,
                      initial_by_id: dict,
                      angular_velocity: float,
                      comets: list, comet_ids: set) -> bool:
    """
    Ground-truth forward-sim: returns True only if the fleet physically
    hits the target within the scan window.
    Used to gate EVERY result before it leaves aim_with_prediction.
    """
    speed  = fleet_speed(max(1, ships))
    fx, fy = launch_point(sx, sy, sr, angle)
    vx, vy = math.cos(angle) * speed, math.sin(angle) * speed
    window = _fwd_window(turns)

    for t in range(1, turns + window + 1):
        pfx, pfy = fx, fy
        fx += vx
        fy += vy
        if segment_hits_sun(pfx, pfy, fx, fy):
            return False
        pos = predict_target_position(
            target_id, tx, ty, tr,
            initial_by_id, angular_velocity,
            comets, comet_ids, t)
        if pos is None:
            continue
        if segment_intersects_circle(pfx, pfy, fx, fy, pos[0], pos[1], tr):
            return True
    return False

# %% ---- next cell ----

def _dynamic_tolerance(target_id: int,
                       initial_by_id: dict,
                       angular_velocity: float,
                       comet_ids: set) -> int:
    """
    v7: max tolerance = 2 (was 3 in v6).
    Tolerance=3 allowed candidate_turns 3 off the real intercept, causing
    search_safe_intercept to pick the wrong orbital position.
    Comets get 2 (faster, less predictable). Orbiting >=1 unit/turn get 2. Else 1.
    """
    if target_id in comet_ids:
        return 2
    init = initial_by_id.get(target_id)
    if init is None:
        return 1
    orb_r     = dist(init['x'], init['y'], CENTER_X, CENTER_Y)
    orb_speed = orb_r * abs(angular_velocity)
    return 2 if orb_speed >= 1.0 else 1

# %% ---- next cell ----

def search_safe_intercept(sx: float, sy: float, sr: float,
                          target_id: int,
                          tx: float, ty: float, tr: float,
                          ships: int,
                          initial_by_id: dict,
                          angular_velocity: float,
                          comets: list, comet_ids: set,
                          tolerance: int = None,
                          ) -> Optional[Tuple[float, int, float, float]]:
    """
    Exhaustive scan: find earliest valid intercept window.
    v7: ROUTE_SEARCH_HORIZON=150, tolerance capped at 2.
    Every candidate is forward-sim verified before being accepted.
    """
    if tolerance is None:
        tolerance = _dynamic_tolerance(target_id, initial_by_id,
                                       angular_velocity, comet_ids)
    max_turns = ROUTE_SEARCH_HORIZON
    if target_id in comet_ids:
        max_turns = min(max_turns,
                        max(0, comet_remaining_life(target_id, comets) - 1))

    for candidate_turns in range(1, max_turns + 1):
        pos = predict_target_position(
            target_id, tx, ty, tr,
            initial_by_id, angular_velocity,
            comets, comet_ids, candidate_turns)
        if pos is None:
            continue

        est = estimate_arrival(sx, sy, sr, pos[0], pos[1], tr, ships)
        if est is None:
            est = arc_safe_angle(sx, sy, sr, pos[0], pos[1], tr, ships)
            if est is None:
                continue

        _, turns = est
        if abs(turns - candidate_turns) > tolerance:
            continue

        actual_turns = max(turns, candidate_turns)
        actual_pos   = predict_target_position(
            target_id, tx, ty, tr,
            initial_by_id, angular_velocity,
            comets, comet_ids, actual_turns)
        if actual_pos is None:
            continue

        confirm = estimate_arrival(sx, sy, sr, actual_pos[0], actual_pos[1], tr, ships)
        if confirm is None:
            confirm = arc_safe_angle(sx, sy, sr, actual_pos[0], actual_pos[1], tr, ships)
            if confirm is None:
                continue

        delta = abs(confirm[1] - actual_turns)
        if delta > tolerance:
            continue

        angle_out, turns_out = confirm[0], confirm[1]

        # v7: verify before accepting — stops false positives in exhaustive search
        if _verify_shot_hits(sx, sy, sr, angle_out, turns_out, ships,
                              target_id, tx, ty, tr,
                              initial_by_id, angular_velocity, comets, comet_ids):
            return (angle_out, turns_out, actual_pos[0], actual_pos[1])

    return None

# %% ---- next cell ----

def _aim_raw(sx: float, sy: float, sr: float,
             target_id: int,
             tx: float, ty: float, tr: float,
             ships: int,
             initial_by_id: dict,
             angular_velocity: float,
             comets: list, comet_ids: set,
             ) -> Optional[Tuple[float, int, float, float]]:
    """
    Iterative convergence. All results are UNVERIFIED — caller must verify.
    """
    tol = _dynamic_tolerance(target_id, initial_by_id, angular_velocity, comet_ids)

    est = estimate_arrival_frac(sx, sy, sr, tx, ty, tr, ships)
    if est is None:
        est_arc = arc_safe_angle(sx, sy, sr, tx, ty, tr, ships)
        if est_arc is None:
            if not target_can_move(target_id, tx, ty, tr, initial_by_id, comet_ids):
                angle   = math.atan2(ty - sy, tx - sx)
                total_d = max(0.0, dist(sx, sy, tx, ty) - sr - LAUNCH_CLEARANCE - tr)
                turns   = max(1, int(math.ceil(total_d / fleet_speed(max(1, ships)))))
                return angle, turns, tx, ty
            return None
        return est_arc[0], est_arc[1], tx, ty

    ox, oy = tx, ty
    for _ in range(_FWD_ITER_MAX):
        _, turns_f = est
        turns_i    = int(math.ceil(turns_f))
        pos        = predict_target_position(
            target_id, tx, ty, tr,
            initial_by_id, angular_velocity,
            comets, comet_ids, turns_i)
        if pos is None:
            return None
        ntx, nty = pos
        next_est = estimate_arrival_frac(sx, sy, sr, ntx, nty, tr, ships)
        if next_est is None:
            arc = arc_safe_angle(sx, sy, sr, ntx, nty, tr, ships)
            if arc:
                return arc[0], arc[1], ntx, nty
            return None
        _, next_turns_f = next_est
        if abs(next_turns_f - turns_f) <= tol:
            # Converged — return integer-turn result
            angle_int = estimate_arrival(sx, sy, sr, ntx, nty, tr, ships)
            if angle_int is None:
                arc = arc_safe_angle(sx, sy, sr, ntx, nty, tr, ships)
                return (arc[0], arc[1], ntx, nty) if arc else None
            return angle_int[0], angle_int[1], ntx, nty
        ox, oy = ntx, nty
        est    = next_est

    # Fallthrough: best effort with last position
    final_pos = predict_target_position(
        target_id, tx, ty, tr,
        initial_by_id, angular_velocity,
        comets, comet_ids, int(math.ceil(est[1])))
    if final_pos is None:
        return None
    refined = estimate_arrival(sx, sy, sr, final_pos[0], final_pos[1], tr, ships)
    if refined is None:
        arc = arc_safe_angle(sx, sy, sr, final_pos[0], final_pos[1], tr, ships)
        return (arc[0], arc[1], final_pos[0], final_pos[1]) if arc else None
    return refined[0], refined[1], final_pos[0], final_pos[1]

def _aim_with_prediction_raw(sx, sy, sr, target_id, tx, ty, tr, ships,
                             initial_by_id, angular_velocity, comets, comet_ids):
    """Backward-compatible alias."""
    return _aim_raw(sx, sy, sr, target_id, tx, ty, tr, ships,
                    initial_by_id, angular_velocity, comets, comet_ids)

# %% ---- next cell ----

def aim_with_prediction(sx: float, sy: float, sr: float,
                        target_id: int,
                        tx: float, ty: float, tr: float,
                        ships: int,
                        initial_by_id: dict,
                        angular_velocity: float,
                        comets: list, comet_ids: set,
                        ) -> Optional[Tuple[float, int, float, float]]:
    """
    Public solver. Returns (angle, turns, target_x, target_y) or None.

    GUARANTEE (v7): every non-None result is VERIFIED by _verify_shot_hits()
    before being returned. The model will NEVER send a fleet that this solver
    predicts will miss. False positives should be ~0.

    Pipeline:
      1. _aim_raw()          — fast iterative convergence (unverified)
      2. _verify_shot_hits() — mandatory forward-sim gate
      3. If raw fails verify → search_safe_intercept() (exhaustive, pre-verified)
      4. If both fail        → return None (shot correctly suppressed)
    """
    res = _aim_raw(sx, sy, sr, target_id, tx, ty, tr, ships,
                   initial_by_id, angular_velocity, comets, comet_ids)

    if res is not None:
        angle, turns, ntx, nty = res
        if _verify_shot_hits(sx, sy, sr, angle, turns, ships,
                              target_id, tx, ty, tr,
                              initial_by_id, angular_velocity, comets, comet_ids):
            return res   # ✅ verified hit
        logger.debug("aim: raw solver not verified, trying exhaustive search target=%d", target_id)

    # Raw failed or not verified — exhaustive search (already verifies internally)
    fallback = search_safe_intercept(
        sx, sy, sr, target_id, tx, ty, tr,
        ships, initial_by_id, angular_velocity, comets, comet_ids)
    if fallback is not None:
        return fallback   # ✅ verified by search_safe_intercept

    logger.debug("aim: NO verified hit for target=%d src=(%.1f,%.1f) ships=%d — suppressed",
                 target_id, sx, sy, ships)
    return None   # ✅ correctly suppressed

# %% ---- next cell ----

def probe_ship_candidates(need: int = None, avail: int = 0, ships: int = None) -> list:
    if need is None:
        need = ships
    if need is None:
        return []
    candidates = sorted(set([
        max(1, int(0.25 * need)),
        max(1, int(0.50 * need)),
        max(1, int(0.75 * need)),
        max(1, need - 5),
        need,
        min(avail, need + 5),
        min(avail, need + 10),
    ]))
    return [c for c in candidates if 1 <= c <= avail]

# %% ---- next cell ----

class PhysicsStats:
    __slots__ = ('aims_attempted', 'aims_succeeded', 'sun_blocked',
                 'fleets_sent', 'planets_captured')

    def __init__(self):
        self.reset()

    def reset(self):
        self.aims_attempted   = 0
        self.aims_succeeded   = 0
        self.sun_blocked      = 0
        self.fleets_sent      = 0
        self.planets_captured = 0

    def record_aim(self, success: bool, sun_blocked: bool = False):
        self.aims_attempted += 1
        if success:
            self.aims_succeeded += 1
        if sun_blocked:
            self.sun_blocked += 1

    def record_fleet_sent(self, count: int = 1):
        self.fleets_sent += count

    def record_planet_captured(self, count: int = 1):
        self.planets_captured += count

    @property
    def aim_rate(self) -> float:
        return self.aims_succeeded / self.aims_attempted if self.aims_attempted else 0.0

    @property
    def hit_rate(self) -> float:
        return self.planets_captured / self.fleets_sent if self.fleets_sent else 0.0

    def summary(self) -> str:
        return (
            f"physics | aims={self.aims_attempted} ok={self.aims_succeeded} "
            f"({self.aim_rate*100:.0f}%) sun_blocked={self.sun_blocked} | "
            f"fleets_sent={self.fleets_sent} captures={self.planets_captured} "
            f"hit_rate={self.hit_rate*100:.0f}%"
        )


# %% ---- next cell ----

