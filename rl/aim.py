#!/usr/bin/env python3
"""Lead-aim: solve the launch angle to intercept a MOVING (orbiting) planet.

Fleets fly straight at constant speed and can't steer, so to hit an orbiting
planet we must fire at where it WILL be when the fleet arrives. We don't know the
arrival time until we know the angle (and vice-versa), so iterate: guess travel
time from the current distance, predict the planet's future orbital position at
that time, recompute time to that point, repeat. (This mirrors orbit_lite's
intercept_angle / the mechanics deep-dive's iterative_aim.)

Static planets (that don't orbit) need no lead — aim straight at them.
"""
from __future__ import annotations

import math

CX = CY = 50.0
SUN_R = 10.0
MAX_SPEED = 6.0


def fleet_speed(ships: float) -> float:
    n = max(1.0, ships)
    return min(1.0 + (MAX_SPEED - 1.0) * (math.log(n) / math.log(1000)) ** 1.5, MAX_SPEED)


def orbits(x: float, y: float, radius: float) -> bool:
    # A planet orbits the sun iff its whole body stays inside the board's rotation zone.
    return math.hypot(x - CX, y - CY) + radius < 50.0


def future_pos(tx: float, ty: float, tr: float, t: float, omega: float) -> tuple[float, float]:
    if not orbits(tx, ty, tr):
        return tx, ty                       # static planet — no lead
    r = math.hypot(tx - CX, ty - CY)
    a = math.atan2(ty - CY, tx - CX) + omega * t   # angular_velocity is CCW (+)
    return CX + r * math.cos(a), CY + r * math.sin(a)


def crosses_sun(sx: float, sy: float, ax: float, ay: float, margin: float = 0.5) -> bool:
    """Does the straight segment src->aim pass through the sun (fleet destroyed)?"""
    dx, dy = ax - sx, ay - sy
    seg2 = dx * dx + dy * dy
    if seg2 < 1e-9:
        return False
    u = max(0.0, min(1.0, ((CX - sx) * dx + (CY - sy) * dy) / seg2))
    cx, cy = sx + u * dx, sy + u * dy
    return math.hypot(cx - CX, cy - CY) < SUN_R + margin


def lead_aim(sx: float, sy: float, tx: float, ty: float, tr: float,
             ships: float, omega: float, iters: int = 5):
    """Return (angle, aim_x, aim_y, eta, sun_blocked) for hitting the (moving) target."""
    spd = fleet_speed(ships)
    t = math.hypot(tx - sx, ty - sy) / spd
    px, py = tx, ty
    for _ in range(iters):
        px, py = future_pos(tx, ty, tr, t, omega)
        t = math.hypot(px - sx, py - sy) / spd
    angle = math.atan2(py - sy, px - sx)
    return angle, px, py, t, crosses_sun(sx, sy, px, py)


if __name__ == "__main__":
    # quick self-check: a rotating target should need a non-trivial lead.
    sx, sy = 20.0, 50.0
    tx, ty, tr = 70.0, 50.0, 1.5
    for omega in (0.0, 0.03):
        ang, ax, ay, eta, sun = lead_aim(sx, sy, tx, ty, tr, ships=30, omega=omega)
        naive = math.atan2(ty - sy, tx - sx)
        print(f"omega={omega}: lead_angle={ang:+.3f} naive={naive:+.3f} "
              f"lead_deg_off={math.degrees(ang-naive):+.1f} eta={eta:.1f} sun={sun} aim=({ax:.1f},{ay:.1f})")
