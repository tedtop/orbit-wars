# Reference code extracted from orbit-wars-complete-game-mechanics-deep-dive.ipynb (NOT a playable agent).
# May contain plotting/analysis cells; kept verbatim for study.

import math, random

print("=" * 68)
print("SECRET #1: Production fires on THE SAME TURN as your launch")
print("=" * 68)
print("""
Engine execution order each turn:
  1. Fleet Launch      (ships leave planet)
  2. Production        (planet[ships] += production)  <-- happens AFTER launch!
  3. Fleet Movement    (fleets advance one step)
  4. Planet Rotation + Sweep
  5. Combat Resolution

Result: If you launch ALL ships from a planet, it still gains +production
that very same turn. The planet will not be empty next turn.
""")

# Example: planet with 15 ships, prod=3, you launch 15
planet_ships_before = 15
prod = 3
launched = 15
remaining_after_launch = planet_ships_before - launched   # 0
remaining_after_production = remaining_after_launch + prod  # 3
print(f"  Planet: {planet_ships_before} ships, prod={prod}")
print(f"  You launch: {launched} ships")
print(f"  After launch (step A): {remaining_after_launch} ships on planet")
print(f"  After production (step B): {remaining_after_production} ships on planet")
print(f"  -> Max launchable this turn: {planet_ships_before} (not {planet_ships_before - prod})")

print()
print("=" * 68)
print("SECRET #2: Garrison does NOT fight arriving fleets")
print("=" * 68)
print("""
Fleet-vs-fleet resolution (Phase 1):
  player_ships = sum per arriving fleet owner
  survivors = top_ships - second_ships

Then survivor vs garrison (Phase 2):
  if survivor_owner == planet_owner: garrison += survivors
  else: garrison -= survivors; if garrison < 0: capture

The garrison only fights the WINNER of the fleet battle, alone.
""")

# Numeric example
print("  Example: Your planet has 40 ships.")
print("  Enemy A sends 60, Enemy B sends 55.")
print("  Fleet battle: A(60) - B(55) = 5 survivors for A")
print("  Garrison battle: 40 - 5 = 35 ships left -> planet NOT captured!")
print("  -> You defended with 40 ships against two simultaneous 60+55 attacks!")

print()
print("=" * 68)
print("SECRET #3: Rotating planets SWEEP passing fleets")
print("=" * 68)
print("""
The engine checks planet movement EVERY TURN with sweep_fleets():
  point_to_segment_distance(fleet_pos, planet_old_pos, planet_new_pos) < planet_radius

If your fleet is hovering in the arc swept by a rotating planet:
  -> The planet CAPTURES your fleet even if you never aimed at it.

This is especially dangerous for:
  - Stationary fleets waiting to intercept
  - Fleets launched at a wrong angle that ends up in a planet sweep zone
""")

# Angular velocity range
ang_min, ang_max = 0.025, 0.05
r_example = 30  # orbital radius of inner planet
print(f"  Inner planet at orbital r=30 sweeps: {r_example * ang_min:.2f} to {r_example * ang_max:.2f} units/turn")
print(f"  Full orbit: {2*math.pi/ang_max:.0f} to {2*math.pi/ang_min:.0f} turns")

print()
print("=" * 68)
print("SECRET #4: TIE at game end -> ALL tied players win (reward = 1)")
print("=" * 68)
print("""
from interpreter():
  max_score = max(scores)
  for i in range(num_agents):
      if scores[i] == max_score and max_score > 0:
          state[i].reward = 1

Late-game strategy: if you are losing, try to MATCH the leader's total
ships (planets + fleets in transit) for a shared win.
""")

print("=" * 68)
print("SECRET #5: 2P starting positions are point-symmetric through (50,50)")
print("=" * 68)
print("""
The engine picks planets[base] (Q1 = top-right quadrant copy) for Player 0
and planets[base+3] (Q4 = bottom-left quadrant copy) for Player 1.

Both planets have the same production, same garrison (always set to 10),
same radius — perfectly symmetric. Every neutral planet is also symmetric:
planet[k] mirrors planet[k+1], planet[k+2], planet[k+3] across center.

Implication: the opponent's nearest high-value planets are EXACTLY
symmetric to yours. The race for those planets is fair.
""")
sx, sy = 66.1, 83.2   # example Player 0 start
print(f"  Example: P0 @ ({sx}, {sy})")
print(f"  P1 must be @ ({100-sx:.1f}, {100-sy:.1f})")


# %% ---- next cell ----

import math
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Circle, FancyArrow
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

# ── Engine constants (from orbit_wars.py) ──
BOARD_SIZE        = 100.0
CENTER            = 50.0
SUN_RADIUS        = 10.0
ROTATION_RADIUS_LIMIT = 50.0   # orbital_r + planet_radius < 50 → rotating
MAX_SPEED         = 6.0        # shipSpeed config default
COMET_SPEED       = 4.0
COMET_SPAWN_STEPS = [50, 150, 250, 350, 450]
EPISODE_STEPS     = 500

print("Constants loaded. Engine version: kaggle_environments orbit_wars")

# %% ---- next cell ----

fig, ax = plt.subplots(1, 1, figsize=(8, 8))
ax.set_xlim(0, 100); ax.set_ylim(0, 100)
ax.set_aspect('equal'); ax.set_facecolor('#0d1117')
ax.set_title('Board Layout: Inner (rotating) vs Outer (static) planets', color='white', fontsize=13)
fig.patch.set_facecolor('#0d1117')

# Sun
sun = Circle((50, 50), SUN_RADIUS, color='#FFD700', alpha=0.9, zorder=5)
ax.add_patch(sun)
ax.text(50, 50, '☀', ha='center', va='center', fontsize=18, zorder=6)

# Rotation boundary circle
boundary = Circle((50, 50), ROTATION_RADIUS_LIMIT, color='cyan', fill=False,
                  linestyle='--', linewidth=1.2, alpha=0.5, label='Rotation boundary (r=50)')
ax.add_patch(boundary)

# Example inner planets (Q1 quadrant only for clarity)
inner_examples = [(65, 65, 1.5), (60, 72, 2.0), (72, 58, 1.8)]
for x, y, r in inner_examples:
    orb_r = math.sqrt((x-50)**2 + (y-50)**2)
    is_inner = (orb_r + r < 50)
    color = '#00FF88' if is_inner else '#FF6B6B'
    label = 'Inner (rotating)' if is_inner else 'Outer (static)'
    c = Circle((x, y), r, color=color, alpha=0.85, zorder=4)
    ax.add_patch(c)
    # orbit arc
    arc = Circle((50, 50), orb_r, color=color, fill=False,
                 linestyle=':', linewidth=0.8, alpha=0.4, zorder=3)
    ax.add_patch(arc)

# Example outer planets
outer_examples = [(80, 80, 2.5), (78, 60, 2.0), (62, 82, 1.5)]
for x, y, r in outer_examples:
    c = Circle((x, y), r, color='#FF6B6B', alpha=0.85, zorder=4)
    ax.add_patch(c)

# Legend
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], marker='o', color='w', markerfacecolor='#00FF88', markersize=10, label='Inner (rotating): orbital_r + r < 50'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='#FF6B6B', markersize=10, label='Outer (static): orbital_r + r ≥ 50'),
    Line2D([0], [0], color='cyan', linestyle='--', label='Rotation boundary (radius=50)')
]
ax.legend(handles=legend_elements, loc='lower left', facecolor='#1a1a2e', labelcolor='white', fontsize=9)
ax.tick_params(colors='white'); ax.spines[:].set_color('#333')
for label in ax.get_xticklabels() + ax.get_yticklabels():
    label.set_color('white')
plt.tight_layout()
plt.show()

print("Key: orbital_radius = sqrt((x-50)² + (y-50)²)")
print("     is_rotating    = (orbital_radius + planet_radius) < 50")

# %% ---- next cell ----

def fleet_speed(ships, max_speed=6.0):
    """Exact formula from orbit_wars.py"""
    if ships <= 0:
        return 0
    speed = 1.0 + (max_speed - 1.0) * (math.log(ships) / math.log(1000)) ** 1.5
    return min(speed, max_speed)

def arrival_turns(distance, ships, max_speed=6.0):
    """How many turns for a fleet to travel `distance` units"""
    s = fleet_speed(ships, max_speed)
    return math.ceil(distance / s)

ship_counts = np.arange(1, 1001)
speeds = [fleet_speed(n) for n in ship_counts]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
fig.patch.set_facecolor('#0d1117')

# Speed vs ships
ax1.set_facecolor('#0d1117')
ax1.plot(ship_counts, speeds, color='#00CFFF', linewidth=2.5)
ax1.axhline(6, color='#FF6B6B', linestyle='--', alpha=0.6, label='Max speed = 6')
for n, label in [(1, '1'), (10, '10'), (50, '50'), (100, '100'), (500, '500'), (1000, '1000')]:
    s = fleet_speed(n)
    ax1.plot(n, s, 'o', color='#FFD700', markersize=7, zorder=5)
    ax1.annotate(f'{n} ships\n→ {s:.2f}/turn', (n, s),
                 textcoords='offset points', xytext=(8, 4),
                 fontsize=7.5, color='#FFD700')
ax1.set_xlabel('Fleet size (ships)', color='white')
ax1.set_ylabel('Speed (units/turn)', color='white')
ax1.set_title('Fleet Speed Formula', color='white', fontsize=13)
ax1.legend(facecolor='#1a1a2e', labelcolor='white')
ax1.tick_params(colors='white'); ax1.spines[:].set_color('#333')
for label in ax1.get_xticklabels() + ax1.get_yticklabels():
    label.set_color('white')

# Arrival time at different distances
distances = [15, 30, 50, 70]
colors = ['#00FF88', '#FFD700', '#FF9500', '#FF4444']
ax2.set_facecolor('#0d1117')
for dist, col in zip(distances, colors):
    arrivals = [arrival_turns(dist, n) for n in ship_counts]
    ax2.plot(ship_counts, arrivals, color=col, linewidth=2, label=f'Distance = {dist}')
ax2.set_xlabel('Fleet size (ships)', color='white')
ax2.set_ylabel('Arrival time (turns)', color='white')
ax2.set_title('Arrival Time vs Fleet Size', color='white', fontsize=13)
ax2.legend(facecolor='#1a1a2e', labelcolor='white')
ax2.tick_params(colors='white'); ax2.spines[:].set_color('#333')
for label in ax2.get_xticklabels() + ax2.get_yticklabels():
    label.set_color('white')

plt.tight_layout()
plt.show()

print("Speed formula: 1 + 5 × (log(ships) / log(1000))^1.5")
print(f"{'Ships':>8} | {'Speed':>8} | {'Turns for dist=30':>18} | {'Turns for dist=70':>18}")
print('-' * 58)
for n in [1, 5, 10, 30, 50, 100, 200, 500, 1000]:
    s = fleet_speed(n)
    print(f"{n:>8} | {s:>8.3f} | {arrival_turns(30,n):>18} | {arrival_turns(70,n):>18}")

# %% ---- next cell ----

def resolve_combat(planet_owner, planet_ships, arriving_fleets):
    """
    Exact combat logic from orbit_wars.py interpreter().
    arriving_fleets: list of (owner, ships) tuples
    Returns: (new_owner, new_ships)
    """
    player_ships = {}
    for owner, ships in arriving_fleets:
        player_ships[owner] = player_ships.get(owner, 0) + ships

    if not player_ships:
        return planet_owner, planet_ships

    sorted_players = sorted(player_ships.items(), key=lambda x: x[1], reverse=True)
    top_player, top_ships = sorted_players[0]

    if len(sorted_players) > 1:
        second_ships = sorted_players[1][1]
        if sorted_players[0][1] == sorted_players[1][1]:
            survivor_ships = 0
            survivor_owner = -1  # tie → neutral
        else:
            survivor_ships = top_ships - second_ships
            survivor_owner = top_player
    else:
        survivor_ships = top_ships
        survivor_owner = top_player

    if survivor_ships == 0:
        return planet_owner, planet_ships  # no change to planet

    if planet_owner == survivor_owner:
        return planet_owner, planet_ships + survivor_ships
    else:
        remaining = planet_ships - survivor_ships
        if remaining < 0:
            return survivor_owner, abs(remaining)  # captured!
        else:
            return planet_owner, remaining  # defender holds


# ── Demonstrate key scenarios ──
print("COMBAT RESOLUTION EXAMPLES")
print("=" * 60)

scenarios = [
    ("1v1 capture",         1, 20, [(0, 50)]),
    ("1v1 failed",          1, 60, [(0, 50)]),
    ("1v1 exact tie",       1, 50, [(0, 50)]),
    ("3-way: A wins",       1, 10, [(0, 100), (2, 60), (3, 40)]),
    ("3-way: A/B tie",      1, 10, [(0, 80), (2, 80)]),
    ("FFA: C wastes fleets",1, 30, [(0, 80), (2, 40)]),
    ("Reinforce own",       0, 30, [(0, 50)]),
]

for name, p_owner, p_ships, fleets in scenarios:
    new_owner, new_ships = resolve_combat(p_owner, p_ships, fleets)
    fleet_str = ', '.join(f'P{o}×{s}' for o, s in fleets)
    status = '🔴 CAPTURED' if new_owner != p_owner and new_owner != -1 else \
             '🟡 NEUTRAL' if new_owner == -1 and p_owner != -1 else \
             '🟢 HELD'
    print(f"\n  [{name}]")
    print(f"    Planet: P{p_owner} with {p_ships} ships")
    print(f"    Fleets: {fleet_str}")
    print(f"    Result: Owner=P{new_owner}, Ships={new_ships}  {status}")

# %% ---- next cell ----

def predict_planet_pos(init_x, init_y, current_step, target_step, angular_velocity):
    """Predict planet position at target_step."""
    dx = init_x - CENTER
    dy = init_y - CENTER
    orb_r = math.sqrt(dx**2 + dy**2)
    init_angle = math.atan2(dy, dx)
    future_angle = init_angle + angular_velocity * target_step
    return CENTER + orb_r * math.cos(future_angle), CENTER + orb_r * math.sin(future_angle)


# Example: rotating planet at (65, 65), fleet at (20, 20)
omega = 0.035  # mid-range angular velocity
init_x, init_y = 65.0, 65.0
fleet_x, fleet_y = 20.0, 20.0
current_step = 50

# Show orbit over full game
steps = range(0, 500)
positions = [predict_planet_pos(init_x, init_y, 0, t, omega) for t in steps]
xs, ys = zip(*positions)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
fig.patch.set_facecolor('#0d1117')

# Left: orbit path
ax1.set_facecolor('#0d1117')
ax1.set_xlim(0, 100); ax1.set_ylim(0, 100); ax1.set_aspect('equal')
ax1.set_title(f'Rotating Planet Orbit (ω={omega} rad/turn)', color='white', fontsize=12)
sun = Circle((50, 50), 10, color='#FFD700', alpha=0.85, zorder=5)
ax1.add_patch(sun)
ax1.plot(xs, ys, '-', color='#00CFFF', alpha=0.3, linewidth=1)
# current position
cur_x, cur_y = predict_planet_pos(init_x, init_y, 0, current_step, omega)
ax1.plot(cur_x, cur_y, 'o', color='#00FF88', markersize=10, label=f'Step {current_step}')
# future positions
for future_step, col, lbl in [(70, '#FFD700', 'Step 70'), (100, '#FF9500', 'Step 100'), (150, '#FF4444', 'Step 150')]:
    fx, fy = predict_planet_pos(init_x, init_y, 0, future_step, omega)
    ax1.plot(fx, fy, 's', color=col, markersize=8, label=lbl)
ax1.plot(fleet_x, fleet_y, '^', color='white', markersize=10, label='Our planet')
ax1.legend(facecolor='#1a1a2e', labelcolor='white', fontsize=8)
ax1.tick_params(colors='white'); ax1.spines[:].set_color('#333')
for label in ax1.get_xticklabels() + ax1.get_yticklabels():
    label.set_color('white')

# Right: naive vs predictive aiming
ax2.set_facecolor('#0d1117')
ax2.set_xlim(35, 80); ax2.set_ylim(35, 80); ax2.set_aspect('equal')
ax2.set_title('Naive vs Predictive Aiming', color='white', fontsize=12)

n_ships = 30
flight_dist = math.sqrt((cur_x - fleet_x)**2 + (cur_y - fleet_y)**2)
flight_turns = math.ceil(flight_dist / fleet_speed(n_ships))
predicted_x, predicted_y = predict_planet_pos(init_x, init_y, 0, current_step + flight_turns, omega)

ax2.plot(xs, ys, '-', color='#00CFFF', alpha=0.2, linewidth=1)
ax2.plot(cur_x, cur_y, 'o', color='#00FF88', markersize=10, label=f'Current pos (step {current_step})')
ax2.plot(predicted_x, predicted_y, 's', color='#FFD700', markersize=10, label=f'Predicted pos (step {current_step+flight_turns})')
ax2.plot(fleet_x, fleet_y, '^', color='white', markersize=10, label='Our planet')
ax2.annotate('', (cur_x, cur_y), (fleet_x, fleet_y),
             arrowprops=dict(arrowstyle='->', color='#FF4444', lw=1.8))
ax2.text((fleet_x+cur_x)/2-2, (fleet_y+cur_y)/2+1, '❌ Naive\n(misses!)', color='#FF4444', fontsize=8)
ax2.annotate('', (predicted_x, predicted_y), (fleet_x, fleet_y),
             arrowprops=dict(arrowstyle='->', color='#00FF88', lw=1.8))
ax2.text((fleet_x+predicted_x)/2+1, (fleet_y+predicted_y)/2, '✅ Predictive', color='#00FF88', fontsize=8)
ax2.legend(facecolor='#1a1a2e', labelcolor='white', fontsize=8)
ax2.tick_params(colors='white'); ax2.spines[:].set_color('#333')
for label in ax2.get_xticklabels() + ax2.get_yticklabels():
    label.set_color('white')

plt.tight_layout()
plt.show()

print(f"Fleet of {n_ships} ships: speed={fleet_speed(n_ships):.2f}, travel time≈{flight_turns} turns")
print(f"Planet at step {current_step}: ({cur_x:.1f}, {cur_y:.1f})")
print(f"Planet at step {current_step+flight_turns}: ({predicted_x:.1f}, {predicted_y:.1f})")
print(f"Δ position = ({predicted_x-cur_x:.2f}, {predicted_y-cur_y:.2f}) — naive aiming MISSES")

# %% ---- next cell ----

# Simulate comet ship distribution
random.seed(42)
comet_ships_samples = [
    min(random.randint(1,99), random.randint(1,99),
        random.randint(1,99), random.randint(1,99))
    for _ in range(100_000)
]
single_samples = [random.randint(1,99) for _ in range(100_000)]

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.patch.set_facecolor('#0d1117')

# Left: distribution comparison
ax = axes[0]
ax.set_facecolor('#0d1117')
ax.hist(single_samples, bins=50, alpha=0.5, color='#FF6B6B', density=True, label='Uniform rand(1,99)')
ax.hist(comet_ships_samples, bins=50, alpha=0.7, color='#00CFFF', density=True, label='Comet: min(4×rand)')
ax.axvline(np.mean(comet_ships_samples), color='#00FF88', linestyle='--',
           label=f'Comet mean = {np.mean(comet_ships_samples):.1f}')
ax.axvline(np.median(comet_ships_samples), color='#FFD700', linestyle=':',
           label=f'Comet median = {np.median(comet_ships_samples):.0f}')
ax.set_xlabel('Ship count', color='white')
ax.set_ylabel('Density', color='white')
ax.set_title('Comet Ship Distribution\n(min of 4 × Uniform[1,99])', color='white', fontsize=12)
ax.legend(facecolor='#1a1a2e', labelcolor='white', fontsize=9)
ax.tick_params(colors='white'); ax.spines[:].set_color('#333')
for label in ax.get_xticklabels() + ax.get_yticklabels():
    label.set_color('white')

# Right: spawn timeline
ax2 = axes[1]
ax2.set_facecolor('#0d1117')
ax2.set_title('Comet Spawn Timeline (500-step game)', color='white', fontsize=12)
ax2.set_xlim(0, 510); ax2.set_ylim(0, 2)
ax2.axhline(1, color='#444', linewidth=2, alpha=0.5)
for step in COMET_SPAWN_STEPS:
    ax2.axvline(step, color='#00CFFF', linewidth=2, alpha=0.8)
    ax2.text(step, 1.3, f'Step\n{step}', ha='center', va='bottom', color='#00CFFF', fontsize=9)
    # Duration window (5-40 turns)
    ax2.barh(0.95, 35, left=step, height=0.1, color='#00CFFF', alpha=0.25)
ax2.text(500, 1.75, 'GAME END', ha='right', color='#FF6B6B', fontsize=10)
ax2.axvline(500, color='#FF6B6B', linewidth=2, alpha=0.7)
ax2.set_xlabel('Game Step', color='white')
ax2.set_yticks([])
ax2.tick_params(colors='white'); ax2.spines[:].set_color('#333')
for label in ax2.get_xticklabels():
    label.set_color('white')

plt.tight_layout()
plt.show()

# Percentiles
print("Comet ship distribution percentiles:")
for pct in [10, 25, 50, 75, 90]:
    print(f"  {pct}th percentile: {np.percentile(comet_ships_samples, pct):.0f} ships")
print(f"  Mean: {np.mean(comet_ships_samples):.1f} ships")
print(f"\nComet speed: 4 units/turn (vs fleet of 30 ships: {fleet_speed(30):.2f}/turn)")
print("→ You need ships nearby + aim at FUTURE comet position to capture it!")

# %% ---- next cell ----

# Production value analysis: when does capturing a planet pay off?
def capture_roi(cost_ships, production, turns_remaining):
    """
    Net ship gain from capturing a planet.
    cost_ships: ships spent to capture
    production: ships/turn the planet generates
    turns_remaining: turns left in game after capture
    """
    return production * turns_remaining - cost_ships

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.patch.set_facecolor('#0d1117')

# Left: break-even analysis
ax = axes[0]
ax.set_facecolor('#0d1117')
turns_remaining = np.arange(0, 500)
for cost, color in [(20, '#00FF88'), (50, '#FFD700'), (100, '#FF9500'), (200, '#FF4444')]:
    for prod in [2, 4]:
        roi = [capture_roi(cost, prod, t) for t in turns_remaining]
        style = '-' if prod == 4 else '--'
        ax.plot(turns_remaining, roi, style, color=color, alpha=0.8, linewidth=1.5,
                label=f'cost={cost}, prod={prod}')
ax.axhline(0, color='white', linewidth=1, linestyle=':')
ax.set_xlabel('Turns remaining after capture', color='white')
ax.set_ylabel('Net ship gain', color='white')
ax.set_title('Capture ROI: production × turns − cost', color='white', fontsize=12)
ax.legend(facecolor='#1a1a2e', labelcolor='white', fontsize=7.5, ncol=2)
ax.tick_params(colors='white'); ax.spines[:].set_color('#333')
ax.set_ylim(-250, 800)
for label in ax.get_xticklabels() + ax.get_yticklabels():
    label.set_color('white')

# Right: Break-even turns for various prod/cost combinations
ax2 = axes[1]
ax2.set_facecolor('#0d1117')
costs = np.arange(10, 200, 5)
for prod, col in [(1, '#FF4444'), (2, '#FF9500'), (3, '#FFD700'), (5, '#00FF88')]:
    breakeven = [c / prod for c in costs]
    ax2.plot(costs, breakeven, '-', color=col, linewidth=2, label=f'Production = {prod}/turn')
ax2.axhline(100, color='white', linestyle=':', alpha=0.5, label='Midgame (step 400, 100 left)')
ax2.axhline(200, color='#888', linestyle=':', alpha=0.5, label='Early (step 300, 200 left)')
ax2.set_xlabel('Ships spent to capture', color='white')
ax2.set_ylabel('Break-even turns needed', color='white')
ax2.set_title('Break-Even: Turns Until Profit', color='white', fontsize=12)
ax2.legend(facecolor='#1a1a2e', labelcolor='white', fontsize=9)
ax2.set_ylim(0, 300)
ax2.tick_params(colors='white'); ax2.spines[:].set_color('#333')
for label in ax2.get_xticklabels() + ax2.get_yticklabels():
    label.set_color('white')

plt.tight_layout()
plt.show()

print("Key insight: a production-5 planet costs at most 5×turns_remaining to justify")
print("At step 400 (100 turns left), even 100 ships spent on a prod-5 planet breaks even!")

# %% ---- next cell ----

def point_to_segment_distance(p, v, w):
    """Exact function from orbit_wars.py"""
    l2 = (v[0]-w[0])**2 + (v[1]-w[1])**2
    if l2 == 0:
        return math.sqrt((p[0]-v[0])**2 + (p[1]-v[1])**2)
    t = max(0, min(1, ((p[0]-v[0])*(w[0]-v[0]) + (p[1]-v[1])*(w[1]-v[1])) / l2))
    proj = (v[0] + t*(w[0]-v[0]), v[1] + t*(w[1]-v[1]))
    return math.sqrt((p[0]-proj[0])**2 + (p[1]-proj[1])**2)


def path_blocked_by_sun(start, end, buffer=1.0):
    return point_to_segment_distance((CENTER, CENTER), start, end) < SUN_RADIUS + buffer


def waypoint_detour(start, end, waypoint_scale=2.2):
    """Route around sun via a perpendicular waypoint."""
    mx = (start[0] + end[0]) / 2
    my = (start[1] + end[1]) / 2
    # Vector from sun to midpoint
    dx = mx - CENTER; dy = my - CENTER
    dist = math.sqrt(dx**2 + dy**2)
    if dist < 0.001:
        dx, dy = 1, 0; dist = 1
    # Waypoint at waypoint_scale × sun_radius from center
    wp_x = CENTER + (dx/dist) * SUN_RADIUS * waypoint_scale
    wp_y = CENTER + (dy/dist) * SUN_RADIUS * waypoint_scale
    return (wp_x, wp_y)


fig, ax = plt.subplots(1, 1, figsize=(9, 9))
ax.set_facecolor('#0d1117'); fig.patch.set_facecolor('#0d1117')
ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.set_aspect('equal')
ax.set_title('Sun Avoidance: Segment Check vs Point Check', color='white', fontsize=12)

sun = Circle((50, 50), SUN_RADIUS, color='#FFD700', alpha=0.85, zorder=5)
ax.add_patch(sun)
# Danger zone
danger = Circle((50, 50), SUN_RADIUS + 3, color='#FF4444', fill=False,
                linestyle='--', linewidth=1.5, alpha=0.6, label='Danger zone (buffer=3)')
ax.add_patch(danger)

# Route examples
routes = [
    ((20, 20), (80, 80), 'Cross through center', '#FF4444'),
    ((15, 60), (85, 40), 'Near miss — blocked!', '#FF9500'),
    ((20, 85), (80, 15), 'Far diagonal — safe', '#00FF88'),
]

for (start, end, label, color) in routes:
    blocked = path_blocked_by_sun(start, end)
    status = '❌ BLOCKED' if blocked else '✅ SAFE'
    dist_to_sun = point_to_segment_distance((CENTER, CENTER), start, end)
    ax.plot([start[0], end[0]], [start[1], end[1]], '-', color=color,
            linewidth=2, alpha=0.7, label=f'{label} — dist={dist_to_sun:.1f} {status}')
    ax.plot(*start, 'o', color=color, markersize=8, zorder=6)
    ax.plot(*end, 's', color=color, markersize=8, zorder=6)
    if blocked:
        # Show detour
        wp = waypoint_detour(start, end)
        ax.plot([start[0], wp[0], end[0]], [start[1], wp[1], end[1]],
                ':', color=color, linewidth=2, alpha=0.9)
        ax.plot(*wp, 'D', color='white', markersize=7, zorder=7)
        ax.annotate('waypoint', wp, xytext=(3, 5), textcoords='offset points',
                    color='white', fontsize=8)

ax.text(50, 50, '☀', ha='center', va='center', fontsize=18, zorder=10)
ax.legend(facecolor='#1a1a2e', labelcolor='white', fontsize=9, loc='upper right')
ax.tick_params(colors='white'); ax.spines[:].set_color('#333')
for label in ax.get_xticklabels() + ax.get_yticklabels():
    label.set_color('white')
plt.tight_layout()
plt.show()

print("How to check in your agent:")
print("  if point_to_segment_dist(sun, fleet_start, fleet_end) < SUN_RADIUS + buffer:")
print("      use waypoint detour")

# %% ---- next cell ----

print("═" * 65)
print("  ORBIT WARS ENGINE CHEAT SHEET")
print("═" * 65)

cheatsheet = [
    ("Fleet Speed",
     "1 + 5×(log(ships)/log(1000))^1.5",
     "Send ≥30 ships for speed>2; 100 ships→3.5/turn"),
    ("Planet Types",
     "orbital_r + radius < 50 → rotating",
     "Aim at future position for inner planets"),
    ("Planet Rotation",
     "angle(t) = init_angle + ω×t, ω∈[0.025,0.05]",
     "Iterative aim: solve for t where fleet arrives"),
    ("Combat (fleets)",
     "survivors = top_ships − second_ships",
     "3rd place loses everything; use in FFA!"),
    ("Combat (planet)",
     "garrison −= survivors; capture if <0",
     "You need ships > garrison to capture"),
    ("Comets",
     "Spawn at steps 50,150,250,350,450; speed=4",
     "Ships ~ min(4×rand[1,99]) ≈ 12-20 median"),
    ("Scoring",
     "All ships: planets + in-flight fleets",
     "Don't launch near game end; fleets = score"),
    ("Sun Avoidance",
     "Segment-to-point check (not endpoint)",
     "Chord of flight path must clear sun radius 10"),
    ("Continuous Collision",
     "Planets sweep-capture fleets as they move",
     "Rotating planets can scoop up in-transit fleets"),
    ("Game Length",
     "500 steps (or last player standing)",
     "Expansion only profitable if turns_left > cost/prod"),
]

for topic, formula, implication in cheatsheet:
    print(f"\n  [{topic}]")
    print(f"    Formula:     {formula}")
    print(f"    Implication: {implication}")

print("\n" + "═" * 65)
print("  SCORING FORMULA: value = production × turns_remaining")
print("  SEND SIZE RULE:   send enough for speed, keep garrison for defense")
print("  FFA RULE:         let enemies bleed each other; reinforce last")
print("═" * 65)

# %% ---- next cell ----

def iterative_aim(launcher_x, launcher_y, planet_init_x, planet_init_y,
                  current_step, ships, omega, max_iters=10, tol=0.1):
    """
    Returns (target_x, target_y, estimated_arrival_step)
    for an intercepting fleet launched at current_step.
    """
    t = 0.0  # initial travel time guess
    for _ in range(max_iters):
        future_step = current_step + int(t + 0.5)
        px, py = predict_planet_pos(planet_init_x, planet_init_y,
                                    0, future_step, omega)
        dist = math.sqrt((px - launcher_x)**2 + (py - launcher_y)**2)
        spd  = fleet_speed(ships)
        new_t = dist / spd
        if abs(new_t - t) < tol:
            break
        t = new_t
    arrival = current_step + math.ceil(t)
    return px, py, arrival


# ── Convergence demo ──
print('Iterative aim convergence demo')
print('Planet: init=(65,65), omega=0.035')
print('Launcher: (20, 20), 30 ships, step=50')
print()

for iter_n in range(1, 7):
    omega = 0.035
    planet_ix, planet_iy = 65.0, 65.0
    lx, ly = 20.0, 20.0
    step = 50
    ships = 30
    t = 0.0
    for i in range(iter_n):
        px, py = predict_planet_pos(planet_ix, planet_iy, 0, step + int(t + 0.5), omega)
        dist = math.sqrt((px - lx)**2 + (py - ly)**2)
        t = dist / fleet_speed(ships)
    print(f'  After {iter_n} iter(s): target=({px:.3f},{py:.3f}), travel_t={t:.3f} turns')

print()
# Final precise aim
tx, ty, arr = iterative_aim(20.0, 20.0, 65.0, 65.0, 50, 30, 0.035)
print(f'Final target: ({tx:.2f}, {ty:.2f}), arrives step {arr}')

# Visualise convergence
omega = 0.035
lx, ly = 20.0, 20.0
step = 50
ships_list = [5, 20, 100, 500]

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.patch.set_facecolor('#0d1117')

ax = axes[0]
ax.set_facecolor('#0d1117')
ax.set_title('Convergence Speed (iters vs error)', color='white', fontsize=12)
planet_ix, planet_iy = 65.0, 65.0
# ground truth: many iters
def get_t_true(ships):
    t = 0.0
    for _ in range(30):
        px, py = predict_planet_pos(planet_ix, planet_iy, 0, step + int(t + 0.5), omega)
        dist = math.sqrt((px - lx)**2 + (py - ly)**2)
        t = dist / fleet_speed(ships)
    return t

for ships, col in zip(ships_list, ['#00FF88','#FFD700','#FF9500','#FF4444']):
    t_true = get_t_true(ships)
    errors = []
    t = 0.0
    for i in range(8):
        px, py = predict_planet_pos(planet_ix, planet_iy, 0, step + int(t + 0.5), omega)
        dist = math.sqrt((px - lx)**2 + (py - ly)**2)
        t = dist / fleet_speed(ships)
        errors.append(abs(t - t_true))
    ax.semilogy(range(1, 9), errors, '-o', color=col, markersize=5, label=f'{ships} ships')

ax.set_xlabel('Iteration', color='white')
ax.set_ylabel('Error (turns)', color='white')
ax.legend(facecolor='#1a1a2e', labelcolor='white')
ax.tick_params(colors='white'); ax.spines[:].set_color('#333')
for label in ax.get_xticklabels() + ax.get_yticklabels(): label.set_color('white')

# Right: orbit + aim arrows
ax2 = axes[1]
ax2.set_facecolor('#0d1117'); ax2.set_aspect('equal')
ax2.set_xlim(10, 80); ax2.set_ylim(10, 80)
ax2.set_title('Iterative vs Naive Aiming (30 ships, step 50)', color='white', fontsize=12)
# orbit path
orbit_pts = [predict_planet_pos(planet_ix, planet_iy, 0, t, omega) for t in range(500)]
ox, oy = zip(*orbit_pts)
ax2.plot(ox, oy, '-', color='#00CFFF', alpha=0.2, linewidth=1)
cur_x2, cur_y2 = predict_planet_pos(planet_ix, planet_iy, 0, step, omega)
ax2.plot(cur_x2, cur_y2, 'o', color='#00FF88', markersize=10, zorder=5, label='Current pos')
# naive aim
naive_dist = math.sqrt((cur_x2 - lx)**2 + (cur_y2 - ly)**2)
naive_turns = math.ceil(naive_dist / fleet_speed(30))
naive_planet = predict_planet_pos(planet_ix, planet_iy, 0, step + naive_turns, omega)
ax2.plot(*naive_planet, 'x', color='#FF4444', markersize=12, zorder=5, linewidth=2, label='Planet when naive fleet arrives')
ax2.annotate('', (cur_x2, cur_y2), (lx, ly), arrowprops=dict(arrowstyle='->', color='#FF4444', lw=1.8))
# iterative aim
tx2, ty2, arr2 = iterative_aim(lx, ly, planet_ix, planet_iy, step, 30, omega)
ax2.plot(tx2, ty2, 's', color='#FFD700', markersize=10, zorder=5, label=f'Predicted pos (step {arr2})')
ax2.annotate('', (tx2, ty2), (lx, ly), arrowprops=dict(arrowstyle='->', color='#FFD700', lw=1.8))
ax2.plot(lx, ly, '^', color='white', markersize=10, zorder=5, label='Launcher')
sun2 = Circle((50, 50), 10, color='#FFD700', alpha=0.7, zorder=4)
ax2.add_patch(sun2)
ax2.legend(facecolor='#1a1a2e', labelcolor='white', fontsize=8)
ax2.tick_params(colors='white'); ax2.spines[:].set_color('#333')
for label in ax2.get_xticklabels() + ax2.get_yticklabels(): label.set_color('white')
plt.tight_layout(); plt.show()
print('Converges in 3-5 iterations for all fleet sizes!')

# %% ---- next cell ----

def compute_reaction_turns(my_planet_pos, enemy_planets, enemy_fleets):
    """
    Min turns before any enemy can reach my planet.
    """
    min_turns = float('inf')
    mx, my = my_planet_pos
    # From enemy planets (assume max speed)
    for (ex, ey, eships) in enemy_planets:
        dist = math.sqrt((ex - mx)**2 + (ey - my)**2)
        turns = dist / MAX_SPEED
        min_turns = min(min_turns, turns)
    # From enemy fleets already in transit
    for (fx, fy, fships, turns_remaining) in enemy_fleets:
        dist = math.sqrt((fx - mx)**2 + (fy - my)**2)
        turns = dist / fleet_speed(fships)
        min_turns = min(min_turns, turns)
    return min_turns if min_turns < float('inf') else 999


def safe_ships_to_send(garrison, production, threat_ships, reaction_turns):
    """
    How many ships can we send and still survive the incoming threat.
    """
    # Ships we'll have when enemy arrives (if we keep `keep` ships)
    # keep + production * reaction_turns >= threat_ships
    # => keep >= threat_ships - production * reaction_turns
    keep_needed = max(1, threat_ships - int(production * reaction_turns))
    return max(0, garrison - keep_needed)


# ── Visualise: reaction window vs safe ships ──
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.patch.set_facecolor('#0d1117')

# Left: reaction window heat map
ax = axes[0]
ax.set_facecolor('#0d1117')
ax.set_title('Reaction Window (turns) as Enemy Gets Closer', color='white', fontsize=11)
distances = np.linspace(1, 100, 200)
for threat_s, col, lbl in [(5,'#00FF88','5 ships (slow)'), (50,'#FFD700','50 ships'),
                             (200,'#FF9500','200 ships'), (800,'#FF4444','800 ships (fast)')]:
    speeds = [fleet_speed(threat_s)] * len(distances)
    windows = distances / fleet_speed(threat_s)
    ax.plot(distances, windows, '-', color=col, linewidth=2, label=lbl)
ax.axhline(5,  color='white', linestyle=':', alpha=0.4, label='5-turn window')
ax.axhline(15, color='white', linestyle='--', alpha=0.4, label='15-turn window')
ax.set_xlabel('Enemy distance (units)', color='white')
ax.set_ylabel('Reaction time (turns)', color='white')
ax.legend(facecolor='#1a1a2e', labelcolor='white', fontsize=8)
ax.tick_params(colors='white'); ax.spines[:].set_color('#333')
for label in ax.get_xticklabels() + ax.get_yticklabels(): label.set_color('white')

# Right: safe ships table visualised
ax2 = axes[1]
ax2.set_facecolor('#0d1117')
ax2.set_title('Safe Ships to Send (garrison=80, production=3)', color='white', fontsize=11)
reaction_turns_arr = np.linspace(1, 30, 100)
garrison_val = 80
production_val = 3
for threat_s, col, lbl in [(20,'#FF4444','Threat=20'), (40,'#FF9500','Threat=40'),
                             (60,'#FFD700','Threat=60'), (80,'#00FF88','Threat=80')]:
    safes = [safe_ships_to_send(garrison_val, production_val, threat_s, rt)
             for rt in reaction_turns_arr]
    ax2.plot(reaction_turns_arr, safes, '-', color=col, linewidth=2, label=lbl)
ax2.axhline(0, color='white', linestyle=':', linewidth=1)
ax2.set_xlabel('Reaction window (turns until enemy arrives)', color='white')
ax2.set_ylabel('Ships safe to send', color='white')
ax2.legend(facecolor='#1a1a2e', labelcolor='white', fontsize=9)
ax2.tick_params(colors='white'); ax2.spines[:].set_color('#333')
for label in ax2.get_xticklabels() + ax2.get_yticklabels(): label.set_color('white')

plt.tight_layout(); plt.show()

print('Safe ships calculation table (garrison=80, production=3):')
print(f'  {"Distance":>10} | {"Reaction(turns)":>16} | {"Threat":>7} | {"Safe to send":>12}')
print('-' * 56)
for dist in [10, 20, 30, 50]:
    rt = dist / MAX_SPEED
    for threat in [30, 60]:
        safe = safe_ships_to_send(80, 3, threat, rt)
        print(f'  {dist:>10} | {rt:>16.1f} | {threat:>7} | {safe:>12}')

# %% ---- next cell ----

def planet_score(production, garrison, distance, ships_to_send, turns_remaining,
                 contested_penalty=1.0):
    """
    Basic planet capture ROI score.
    Higher = better to attack.
    """
    if turns_remaining <= 0:
        return -999
    # Net ships gained by holding planet after capture
    arrival = arrival_turns(distance, ships_to_send)
    effective_turns = max(0, turns_remaining - arrival)
    gross_gain = production * effective_turns
    net_gain = gross_gain - ships_to_send  # ships spent
    if net_gain <= 0:
        return net_gain / (distance + 1)
    return (net_gain / (distance + 1)) * contested_penalty


# Visualise planet score landscape
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.patch.set_facecolor('#0d1117')

# Left: score vs distance at different game stages
ax = axes[0]
ax.set_facecolor('#0d1117')
ax.set_title('Planet Score vs Distance (prod=3, garrison=20, 60 ships sent)', color='white', fontsize=10)
dist_arr = np.linspace(5, 80, 200)
for turns_left, col, lbl in [(450,'#00FF88','Early (450 turns left)'),
                              (300,'#FFD700','Mid (300 left)'),
                              (150,'#FF9500','Late (150 left)'),
                              (50, '#FF4444','End (50 left)')]:
    scores = [planet_score(3, 20, d, 60, turns_left) for d in dist_arr]
    ax.plot(dist_arr, scores, '-', color=col, linewidth=2, label=lbl)
ax.axhline(0, color='white', linestyle=':', linewidth=1)
ax.set_xlabel('Distance to planet', color='white')
ax.set_ylabel('Score (higher = better target)', color='white')
ax.legend(facecolor='#1a1a2e', labelcolor='white', fontsize=9)
ax.tick_params(colors='white'); ax.spines[:].set_color('#333')
for label in ax.get_xticklabels() + ax.get_yticklabels(): label.set_color('white')

# Right: profitability threshold heat map (production × turns > cost?)
ax2 = axes[1]
ax2.set_facecolor('#0d1117')
prods = np.arange(1, 8)
turns_left_arr = np.arange(10, 500, 10)
Z = np.zeros((len(prods), len(turns_left_arr)))
for i, prod in enumerate(prods):
    for j, tl in enumerate(turns_left_arr):
        Z[i, j] = prod * tl  # gross gain if cost=0
im = ax2.contourf(turns_left_arr, prods, Z, levels=20, cmap='RdYlGn')
# overlay cost contours
for cost, style in [(30,':'), (60,'--'), (100,'-')]:
    cs = ax2.contour(turns_left_arr, prods, Z, levels=[cost],
                     colors='white', linestyles=style, linewidths=1.5)
    ax2.clabel(cs, fmt=f'cost={cost}', inline=True, fontsize=8, colors='white')
ax2.set_xlabel('Turns remaining', color='white')
ax2.set_ylabel('Planet production rate', color='white')
ax2.set_title('Gross Gain (prod x turns): white lines = break-even costs', color='white', fontsize=10)
plt.colorbar(im, ax=ax2, label='Gross gain')
ax2.tick_params(colors='white'); ax2.spines[:].set_color('#333')
for label in ax2.get_xticklabels() + ax2.get_yticklabels(): label.set_color('white')
plt.tight_layout(); plt.show()

# Rank example planets
print('Example: ranking 5 planets at step 200 (300 turns left)')
print(f'  {"Planet":>8} | {"Prod":>6} | {"Garr":>6} | {"Dist":>6} | {"Score":>8}')
print('-' * 46)
candidates = [
    ('Near-hi',  5, 10, 15),
    ('Far-hi',   5, 10, 60),
    ('Near-lo',  1, 5,  12),
    ('Mid-mid',  3, 30, 30),
    ('Far-lo',   1, 50, 70),
]
scored = [(name, prod, g, d, planet_score(prod, g, d, g+5, 300))
          for name, prod, g, d in candidates]
scored.sort(key=lambda x: x[4], reverse=True)
for name, prod, g, d, sc in scored:
    print(f'  {name:>8} | {prod:>6} | {g:>6} | {d:>6} | {sc:>8.2f}')

# %% ---- next cell ----

def comet_intercept(launcher_x, launcher_y, comet_spawn_x, comet_spawn_y,
                    comet_dx, comet_dy, spawn_step, current_step, ships,
                    max_iters=15, tol=0.5):
    """
    Find intercept point for a moving comet.
    comet_dx, comet_dy: unit direction vector × COMET_SPEED
    Returns (target_x, target_y, arrival_step) or None if unreachable.
    """
    if current_step > spawn_step:
        elapsed = current_step - spawn_step
        cx = comet_spawn_x + comet_dx * elapsed
        cy = comet_spawn_y + comet_dy * elapsed
    else:
        cx, cy = comet_spawn_x, comet_spawn_y
    
    tau = 0.0
    spd = fleet_speed(ships)
    for _ in range(max_iters):
        # Comet pos at current_step + tau
        future_elapsed = max(0, current_step + tau - spawn_step)
        tx = comet_spawn_x + comet_dx * future_elapsed
        ty = comet_spawn_y + comet_dy * future_elapsed
        dist = math.sqrt((tx - launcher_x)**2 + (ty - launcher_y)**2)
        new_tau = dist / spd
        if abs(new_tau - tau) < tol:
            break
        tau = new_tau
    # Check comet hasn't left board
    if not (0 <= tx <= 100 and 0 <= ty <= 100):
        return None
    return tx, ty, current_step + math.ceil(tau)


# ── Visualise comet intercept ──
random.seed(7)
comet_sx, comet_sy = 5.0, 30.0
comet_dx, comet_dy = COMET_SPEED * 0.866, COMET_SPEED * 0.5  # roughly NE direction
spawn_step = 50

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.patch.set_facecolor('#0d1117')

# Left: board view with comet path and intercept arrows
ax = axes[0]
ax.set_facecolor('#0d1117'); ax.set_aspect('equal')
ax.set_xlim(0, 100); ax.set_ylim(0, 100)
ax.set_title('Comet Interception (spawn step 50)', color='white', fontsize=12)
# Comet path
comet_steps = 35  # max comet duration
comet_xs = [comet_sx + comet_dx * t for t in range(comet_steps)]
comet_ys = [comet_sy + comet_dy * t for t in range(comet_steps)]
ax.plot(comet_xs, comet_ys, '-', color='#00CFFF', linewidth=3, alpha=0.5, label='Comet path')
ax.plot(comet_sx, comet_sy, 'D', color='#00CFFF', markersize=12, zorder=6, label='Spawn')
# Launcher positions
launchers = [(20, 80, 50, '#00FF88'), (80, 20, 80, '#FFD700'), (15, 50, 30, '#FF9500')]
for (lx, ly, ships, col) in launchers:
    result = comet_intercept(lx, ly, comet_sx, comet_sy, comet_dx, comet_dy,
                              spawn_step, spawn_step, ships)
    if result:
        ix, iy, arr = result
        ax.plot(lx, ly, '^', color=col, markersize=10, zorder=6)
        ax.annotate('', (ix, iy), (lx, ly),
                    arrowprops=dict(arrowstyle='->', color=col, lw=1.8))
        ax.plot(ix, iy, 'o', color=col, markersize=8, zorder=6)
        ax.text(ix+1, iy+1, f'{ships} ships\narrives step {arr}',
                color=col, fontsize=8)
        print(f'Launcher ({lx},{ly}), {ships} ships -> intercept ({ix:.1f},{iy:.1f}) step {arr}')
    else:
        print(f'Launcher ({lx},{ly}), {ships} ships -> CANNOT INTERCEPT (comet leaves board)')
        ax.plot(lx, ly, 'x', color=col, markersize=12, zorder=6, linewidth=3)
sun3 = Circle((50,50), 10, color='#FFD700', alpha=0.7, zorder=4)
ax.add_patch(sun3)
ax.legend(facecolor='#1a1a2e', labelcolor='white', fontsize=9)
ax.tick_params(colors='white'); ax.spines[:].set_color('#333')
for label in ax.get_xticklabels() + ax.get_yticklabels(): label.set_color('white')

# Right: 'Can you catch it?' — feasibility vs distance
ax2 = axes[1]
ax2.set_facecolor('#0d1117')
ax2.set_title('Comet Catchable? (Fleet speed vs Comet direction)', color='white', fontsize=11)
# For various fleet sizes, what is the max 'behind angle' that still catches?
angles_deg = np.linspace(0, 180, 200)  # angle from launch to comet direction
for ships2, col2 in [(10,'#FF4444'), (30,'#FF9500'), (100,'#FFD700'), (300,'#00FF88')]:
    catchable = []
    for angle in angles_deg:
        # Comet moves away at speed=4, fleet approaches at fleet_speed
        # Component of fleet velocity toward comet must exceed comet speed
        # Simplified: fleet speed > comet_speed * cos(0) for head-on
        # For catching a moving target at angle theta:
        # We can catch if fleet_speed > comet_speed * cos(theta_component)
        fs = fleet_speed(ships2)
        # Effective closing speed = fs - 4*cos(angle)
        closing = fs - COMET_SPEED * math.cos(math.radians(angle))
        catchable.append(1 if closing > 0 else 0)
    ax2.plot(angles_deg, catchable, '-', color=col2, linewidth=2, label=f'{ships2} ships (speed={fleet_speed(ships2):.2f})')
ax2.set_xlabel('Intercept angle (0=head-on, 180=chasing)', color='white')
ax2.set_ylabel('Catchable (1=yes, 0=no)', color='white')
ax2.axvline(90, color='white', linestyle=':', alpha=0.5, label='90 deg = perpendicular')
ax2.set_yticks([0, 1])
ax2.legend(facecolor='#1a1a2e', labelcolor='white', fontsize=9)
ax2.tick_params(colors='white'); ax2.spines[:].set_color('#333')
for label in ax2.get_xticklabels() + ax2.get_yticklabels(): label.set_color('white')
plt.tight_layout(); plt.show()
print(f'\nComet speed={COMET_SPEED}; only fleets >{COMET_SPEED}/turn can catch from behind!')
print('Small fleets (speed<4) MUST intercept from the front!')

# %% ---- next cell ----

# Fleet consolidation: speed benefit analysis
distance = 50

strategies = {
    '1x100':  [(100,)],
    '2x50':   [(50,), (50,)],
    '4x25':   [(25,), (25,), (25,), (25,)],
    '10x10':  [(10,)] * 10,
    '20x5':   [(5,)]  * 20,
}

print('Fleet Consolidation Analysis (total=100 ships, distance=50)')
print(f'  {"Strategy":>8} | {"Speed":>7} | {"Arrival":>9} | {"Turns saved vs 20x5":>20}')
print('-' * 55)

arrivals = {}
for name, fleets in strategies.items():
    # Assume all launched at same time, first fleet to arrive wins
    # For a single fleet, it's just arrival_turns(distance, ships)
    # For split, each fleet arrives separately
    if len(fleets) == 1:
        ships = fleets[0][0]
        arr = arrival_turns(distance, ships)
        spd = fleet_speed(ships)
    else:
        arr = min(arrival_turns(distance, f[0]) for f in fleets)
        spd = max(fleet_speed(f[0]) for f in fleets)
    arrivals[name] = arr
    print(f'  {name:>8} | {spd:>7.2f} | {arr:>9} | {"":>20}')

baseline = arrivals['20x5']
print('  (Turns saved vs 20x5):')
for name, arr in arrivals.items():
    saved = baseline - arr
    bar = '#' * max(0, saved)
    print(f'  {name:>8}: saves {saved:>3} turns  {bar}')

# Visual: speed curves with split-fleet overlay
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.patch.set_facecolor('#0d1117')

ax = axes[0]
ax.set_facecolor('#0d1117')
ax.set_title('Fleet Speed: Big vs Split (same total ships)', color='white', fontsize=12)
total_ships_arr = np.arange(10, 500)
for n_splits, col, lbl in [(1,'#00FF88','1 fleet (all together)'),
                             (2,'#FFD700','2 equal fleets'),
                             (4,'#FF9500','4 equal fleets'),
                             (10,'#FF4444','10 equal fleets')]:
    speeds = [fleet_speed(max(1, t // n_splits)) for t in total_ships_arr]
    ax.plot(total_ships_arr, speeds, '-', color=col, linewidth=2, label=lbl)
ax.set_xlabel('Total ships available', color='white')
ax.set_ylabel('Speed of each sub-fleet', color='white')
ax.legend(facecolor='#1a1a2e', labelcolor='white', fontsize=9)
ax.tick_params(colors='white'); ax.spines[:].set_color('#333')
for label in ax.get_xticklabels() + ax.get_yticklabels(): label.set_color('white')

# Right: arrival time difference
ax2 = axes[1]
ax2.set_facecolor('#0d1117')
ax2.set_title('Arrival Time: Consolidated vs Split (distance=50)', color='white', fontsize=11)
for n_splits, col, lbl in [(1,'#00FF88','1 fleet'), (2,'#FFD700','2 fleets'),
                             (4,'#FF9500','4 fleets'), (10,'#FF4444','10 fleets')]:
    arrivals2 = [arrival_turns(50, max(1, t // n_splits)) for t in total_ships_arr]
    ax2.plot(total_ships_arr, arrivals2, '-', color=col, linewidth=2, label=lbl)
ax2.set_xlabel('Total ships available', color='white')
ax2.set_ylabel('Arrival turn', color='white')
ax2.legend(facecolor='#1a1a2e', labelcolor='white', fontsize=9)
ax2.tick_params(colors='white'); ax2.spines[:].set_color('#333')
for label in ax2.get_xticklabels() + ax2.get_yticklabels(): label.set_color('white')
plt.tight_layout(); plt.show()

# Minimum fleet sizes for useful speeds
print('Minimum ships needed to reach a given speed:')
for target_speed in [2.0, 3.0, 4.0, 5.0, 5.5]:
    # Binary search
    lo, hi = 1, 1000
    while lo < hi:
        mid = (lo + hi) // 2
        if fleet_speed(mid) >= target_speed:
            hi = mid
        else:
            lo = mid + 1
    print(f'  Speed >= {target_speed}: need >= {lo} ships (actual speed={fleet_speed(lo):.3f})')

# %% ---- next cell ----

# Final sanity-check: print all key thresholds
print('ORBIT WARS — AGENT DESIGN QUICK REFERENCE')
print('=' * 60)

print('\n--- Fleet Speed Thresholds ---')
for target in [2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5]:
    lo, hi = 1, 1000
    while lo < hi:
        mid = (lo + hi) // 2
        if fleet_speed(mid) >= target: hi = mid
        else: lo = mid + 1
    print(f'  {target:.1f}/turn  requires >= {lo:4d} ships')

print('\n--- Capture Cost Safety Rule ---')
print('  send = garrison × 1.5 + 5 + production × arrival_turns')

print('\n--- Comet Timing (step of next comet) ---')
current_demo_step = 120
next_comet = next((s for s in COMET_SPAWN_STEPS if s > current_demo_step), None)
print(f'  At step {current_demo_step}: next comet spawns at step {next_comet}')
if next_comet:
    window = next_comet - current_demo_step
    print(f'  Turns until spawn: {window}')
    print(f'  Ideal fleet: preposition within dist={window * COMET_SPEED:.0f} of expected spawn zone')

print('\n--- Sun Danger Zone Routes ---')
danger_examples = [
    ((10,10),(90,90),'full diagonal'),
    ((10,90),(90,10),'anti-diagonal'),
    ((50,5),(50,95),'vertical'),
    ((5,50),(95,50),'horizontal'),
    ((10,70),(90,30),'cross-lane'),
]
for (s,e,name) in danger_examples:
    d = point_to_segment_distance((CENTER,CENTER), s, e)
    blocked = d < SUN_RADIUS
    print(f'  {name:15}: dist_to_sun={d:5.1f}  {"BLOCKED" if blocked else "safe"}')

# %% ---- next cell ----

fig, ax = plt.subplots(figsize=(14, 6))
fig.patch.set_facecolor('#0d1117')
ax.set_facecolor('#0d1117')
ax.set_xlim(0, 10)
ax.set_ylim(-0.5, 5.5)
ax.axis('off')
ax.set_title('Turn Execution Pipeline (per step)', color='white', fontsize=15, fontweight='bold', pad=15)

steps_pipeline = [
    (1, "① Fleet Launch",     "Ships subtracted from planet\nFleet placed at planet_edge + 0.1",    "#FF6B6B", "Remove ships from\nsource planet"),
    (2, "② Production",       "planet[ships] += planet[production]\n(only owned planets)",          "#FFD700", "Add +prod to all\nowned planets"),
    (3, "③ Fleet Movement",   "Each fleet advances speed units\nDestroyed if out-of-bounds or hits sun", "#00CFFF", "Fleets move, check\nsun/OOB kill"),
    (4, "④ Planet Rotation\n& Sweep", "Inner planets rotate by ω*step\nsweep_fleets() catches nearby fleets", "#00FF88", "Planets move, may\ncapture fleets"),
    (5, "⑤ Combat",           "Fleet-vs-fleet: top-second\nThen winner vs garrison",                "#FF9500", "Resolve all\narriving fleets"),
]

colors = [s[3] for s in steps_pipeline]
for i, (num, title, desc, color, note) in enumerate(steps_pipeline):
    y = 4 - i
    # Step box
    rect = plt.Rectangle((0.2, y-0.35), 2.5, 0.7, facecolor=color, alpha=0.25, edgecolor=color, linewidth=2)
    ax.add_patch(rect)
    ax.text(1.45, y, title, color=color, fontsize=10, fontweight='bold', ha='center', va='center')
    # Description
    ax.text(3.2, y, desc, color='#CCCCCC', fontsize=8, va='center')
    # Arrow to next
    if i < 4:
        ax.annotate('', xy=(1.45, y-0.35), xytext=(1.45, y-0.65),
                    arrowprops=dict(arrowstyle='->', color='#888888', lw=1.5))

# Annotations for the key "secrets"
ax.annotate('SECRET #1:\nProduction fires HERE\n(after launch, before combat)',
            xy=(2.7, 3), xytext=(6.5, 3.4),
            color='#FFD700', fontsize=9, fontweight='bold',
            arrowprops=dict(arrowstyle='->', color='#FFD700', lw=1.5))

ax.annotate('SECRET #3:\nPlanet sweeps\npassing fleets HERE',
            xy=(2.7, 1), xytext=(6.5, 1.2),
            color='#00FF88', fontsize=9, fontweight='bold',
            arrowprops=dict(arrowstyle='->', color='#00FF88', lw=1.5))

ax.annotate('GARRISON not in\nfleet-vs-fleet',
            xy=(2.7, 0), xytext=(6.5, -0.1),
            color='#FF9500', fontsize=9, fontweight='bold',
            arrowprops=dict(arrowstyle='->', color='#FF9500', lw=1.5))

plt.tight_layout()
plt.show()

print("\nKey consequence of the pipeline order:")
print("  - Production fires AFTER launch -> you always get +production even if you launch all ships")
print("  - max_ships_to_launch(planet) = planet.ships  (not planet.ships - production)")
print("  - A planet cannot 'produce and immediately re-defend' in the same turn combat hits")


# %% ---- next cell ----

import math
import numpy as np

fig, axes = plt.subplots(1, 2, figsize=(14, 7))
fig.patch.set_facecolor('#0d1117')
fig.suptitle('§15: Planet-Sweeps-Fleet Mechanic', color='white', fontsize=13, fontweight='bold')

# ── Left: show sweep arc ──
ax = axes[0]
ax.set_facecolor('#0d1117')
ax.set_aspect('equal')
ax.set_xlim(35, 75)
ax.set_ylim(35, 75)
ax.set_title('Sweep Zone for Rotating Planet', color='white', fontsize=11)

CENTER = 50.0
orbital_r = 20
planet_r = 2.39
ang_vel = 0.04  # typical

# Planet at step T and T+1
t = 80
init_angle = math.pi / 4  # 45 degrees
angle_t = init_angle + ang_vel * t
angle_t1 = init_angle + ang_vel * (t + 1)
px_t  = CENTER + orbital_r * math.cos(angle_t)
py_t  = CENTER + orbital_r * math.sin(angle_t)
px_t1 = CENTER + orbital_r * math.cos(angle_t1)
py_t1 = CENTER + orbital_r * math.sin(angle_t1)

# Draw orbit circle
theta = np.linspace(0, 2*np.pi, 200)
ax.plot(CENTER + orbital_r * np.cos(theta), CENTER + orbital_r * np.sin(theta),
        color='#444444', lw=1, linestyle='--', label='Orbit path')

# Draw sweep segment (planet swept this arc)
arc_theta = np.linspace(angle_t, angle_t1, 50)
sweep_outer = [(CENTER + (orbital_r + planet_r) * math.cos(a),
                CENTER + (orbital_r + planet_r) * math.sin(a)) for a in arc_theta]
sweep_inner = [(CENTER + (orbital_r - planet_r) * math.cos(a),
                CENTER + (orbital_r - planet_r) * math.sin(a)) for a in arc_theta]
sweep_x = [s[0] for s in sweep_outer] + [s[0] for s in reversed(sweep_inner)]
sweep_y = [s[1] for s in sweep_outer] + [s[1] for s in reversed(sweep_inner)]
ax.fill(sweep_x, sweep_y, color='#FF4444', alpha=0.35, label='Sweep zone (DANGER)')

# Draw planet at T and T+1
circle_t = plt.Circle((px_t, py_t), planet_r, color='#00CFFF', alpha=0.6, label=f'Planet @ step {t}')
circle_t1 = plt.Circle((px_t1, py_t1), planet_r, color='#00FF88', alpha=0.6, label=f'Planet @ step {t+1}')
ax.add_patch(circle_t)
ax.add_patch(circle_t1)

# Arrow from T to T+1
ax.annotate('', xy=(px_t1, py_t1), xytext=(px_t, py_t),
            arrowprops=dict(arrowstyle='->', color='white', lw=2))

# Plot a fleet caught in the sweep zone
fleet_x = (px_t + px_t1) / 2
fleet_y = (py_t + py_t1) / 2 + 0.2
ax.scatter([fleet_x], [fleet_y], s=120, color='#FF6B6B', marker='*', zorder=5, label='Fleet (CAPTURED!)')
ax.annotate('CAPTURED\nby sweep!', (fleet_x, fleet_y), (fleet_x+2.5, fleet_y+2),
            color='#FF6B6B', fontsize=9, arrowprops=dict(arrowstyle='->', color='#FF6B6B'))

ax.plot(CENTER, CENTER, 'o', color='#FFD700', markersize=8, label='Sun (50,50)')
ax.set_xlabel('X', color='#888888')
ax.set_ylabel('Y', color='#888888')
ax.tick_params(colors='#888888')
for spine in ax.spines.values(): spine.set_edgecolor('#333333')
ax.legend(fontsize=8, facecolor='#1a1a2e', labelcolor='white', loc='upper left')

# ── Right: sweep distance quantification ──
ax2 = axes[1]
ax2.set_facecolor('#0d1117')
ax2.set_title('Sweep Arc Width vs Orbital Radius & Angular Velocity', color='white', fontsize=11)

orb_radii = np.linspace(5, 47, 100)
for ang_v, color, label in [(0.025, '#00CFFF', 'ω=0.025 (slow rotation)'),
                              (0.0375, '#00FF88', 'ω=0.0375 (typical)'),
                              (0.05,  '#FF6B6B', 'ω=0.050 (fast rotation)')]:
    sweep_arc = orb_radii * ang_v  # arc length swept per turn
    ax2.plot(orb_radii, sweep_arc, color=color, lw=2, label=label)

ax2.axhline(y=2.39, color='#FFD700', linestyle='--', alpha=0.7, label='Planet radius 2.39 (prod=4)')
ax2.axhline(y=1.0,  color='#AAAAAA', linestyle=':', alpha=0.7, label='Planet radius 1.0 (prod=1)')
ax2.fill_between(orb_radii, 0, 2.39, alpha=0.08, color='#FF6B6B')

ax2.set_xlabel('Orbital Radius (units)', color='#888888')
ax2.set_ylabel('Arc swept per turn (units)', color='#888888')
ax2.tick_params(colors='#888888')
for spine in ax2.spines.values(): spine.set_edgecolor('#333333')
ax2.legend(fontsize=8, facecolor='#1a1a2e', labelcolor='white')
ax2.set_facecolor('#0d1117')

plt.tight_layout()
plt.show()

print("Practical rules:")
print("  - At orbital_r=20, ang_vel=0.04: sweep arc = 0.8 units/turn")
print("  - A fleet within planet_radius of the swept arc segment gets captured")
print("  - Risk is highest for INNER planets (small orbital_r) with FAST rotation (high ang_vel)")
print("  - To check safety: point_to_segment_dist(fleet_pos, planet_old, planet_new) >= planet_radius")
print()
print("Agent design implication:")
print("  - After launching a fleet, do NOT assume it is safe just because it did not hit a planet this turn")
print("  - A rotating planet CAN hit your stationary fleet next turn")
print("  - The sweep is checked AFTER planet rotation but BEFORE combat — so it is a capture, not combat")


# %% ---- next cell ----

# Simulate map generation statistics
import random, math, numpy as np

random.seed(42)
N_SIMS = 50_000

# 1. Starting planet garrison: always 10 (hardcoded)
print("=" * 60)
print("STARTING PLANET SHIPS: Always 10 (hardcoded)")
print("  obs0.planets[base][5] = 10  (line: home planet assignment)")
print("  This is true regardless of the planet's production value.")
print()

# 2. Neutral planet garrison: min(rand(5,99), rand(5,99))
neutral_garrisons_phase1 = [
    min(random.randint(5,99), random.randint(5,99))
    for _ in range(N_SIMS)
]
neutral_garrisons_phase2 = [
    random.randint(5, 30)
    for _ in range(N_SIMS)
]

# 3. Angular velocity distribution
ang_vels = [random.uniform(0.025, 0.05) for _ in range(N_SIMS)]

# 4. Orbital radius of starting planet (on y=x diagonal, orbiting)
# orbital_r = uniform(SUN_RADIUS + r + 10, 50 - r) where r = 1 + log(prod)
# prod can be 1-5, so r can be 1-2.61
orb_radii_start = []
for _ in range(N_SIMS):
    prod = random.randint(1, 5)
    r = 1 + math.log(prod)
    min_orb = 10 + r + 10  # SUN_RADIUS + r + 10
    max_orb = 50 - r        # ROTATION_RADIUS_LIMIT - r
    if min_orb < max_orb:
        orb_radii_start.append(random.uniform(min_orb, max_orb))

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.patch.set_facecolor('#0d1117')
fig.suptitle('§16: Map Generation Distributions', color='white', fontsize=13, fontweight='bold')

# Left: garrison distribution
ax = axes[0]
ax.set_facecolor('#0d1117')
ax.hist(neutral_garrisons_phase1, bins=40, alpha=0.7, color='#00CFFF', density=True,
        label=f'Phase1/2 planets\n(min of 2 rolls)\nmedian={int(np.median(neutral_garrisons_phase1))}')
ax.hist(neutral_garrisons_phase2, bins=25, alpha=0.6, color='#FFD700', density=True,
        label=f'Phase2 fill planets\n(uniform 5-30)\nmedian={int(np.median(neutral_garrisons_phase2))}')
ax.axvline(10, color='#FF4444', linestyle='--', lw=2, label='Starting planet: 10 (fixed)')
ax.set_xlabel('Initial garrison ships', color='#888888')
ax.set_ylabel('Density', color='#888888')
ax.set_title('Neutral Planet Garrisons', color='white')
ax.tick_params(colors='#888888')
for spine in ax.spines.values(): spine.set_edgecolor('#333333')
ax.legend(fontsize=8, facecolor='#1a1a2e', labelcolor='white')

# Middle: angular velocity -> orbit period
ax2 = axes[1]
ax2.set_facecolor('#0d1117')
orbit_periods = [2 * math.pi / v for v in ang_vels]
ax2.hist(orbit_periods, bins=40, color='#00FF88', alpha=0.8, density=True)
ax2.set_xlabel('Full orbit period (turns)', color='#888888')
ax2.set_ylabel('Density', color='#888888')
ax2.set_title('Orbit Period Distribution\nω ~ Uniform(0.025, 0.05)', color='white')
ax2.tick_params(colors='#888888')
for spine in ax2.spines.values(): spine.set_edgecolor('#333333')
ax2.axvline(np.mean(orbit_periods), color='#FF6B6B', linestyle='--',
            label=f'Mean={np.mean(orbit_periods):.0f} turns')
ax2.legend(fontsize=8, facecolor='#1a1a2e', labelcolor='white')

# Right: starting planet orbital radius
ax3 = axes[2]
ax3.set_facecolor('#0d1117')
ax3.hist(orb_radii_start, bins=40, color='#FF9500', alpha=0.8, density=True)
ax3.set_xlabel('Starting planet orbital radius', color='#888888')
ax3.set_ylabel('Density', color='#888888')
ax3.set_title('Starting Planet Orbital Radius\n(always on y=x diagonal)', color='white')
ax3.axvline(20, color='#FF4444', linestyle=':', alpha=0.7, label='r=20 reference')
ax3.axvline(np.mean(orb_radii_start), color='#00CFFF', linestyle='--',
            label=f'Mean={np.mean(orb_radii_start):.1f}')
ax3.tick_params(colors='#888888')
for spine in ax3.spines.values(): spine.set_edgecolor('#333333')
ax3.legend(fontsize=8, facecolor='#1a1a2e', labelcolor='white')

plt.tight_layout()
plt.show()

print("Key facts:")
print(f"  Neutral garrison (Phase1/2): median={int(np.median(neutral_garrisons_phase1))}, "
      f"mean={np.mean(neutral_garrisons_phase1):.1f}, 90th pctile={int(np.percentile(neutral_garrisons_phase1, 90))}")
print(f"  Neutral garrison (Phase2):   median={int(np.median(neutral_garrisons_phase2))}, "
      f"mean={np.mean(neutral_garrisons_phase2):.1f}")
print(f"  Orbit period:  mean={np.mean(orbit_periods):.0f} turns, range=[{min(orbit_periods):.0f}, {max(orbit_periods):.0f}]")
print(f"  Start orbital r: mean={np.mean(orb_radii_start):.1f}, range=[{min(orb_radii_start):.1f}, {max(orb_radii_start):.1f}]")
print()
print("2P map symmetry rules (from engine code):")
print("  - Starting planets: Q1 (top-right) for P0, Q4 (bottom-left) for P1")
print("  - They are planets[base] and planets[base+3] of the same group")
print("  - Every neutral planet k has a mirror at BOARD_SIZE - x, BOARD_SIZE - y")
print("  - Symmetric maps are GUARANTEED — no RNG asymmetry in 2P")


# %% ---- next cell ----

# 2P vs 4P visualization: map symmetry + FFA bystander advantage

import numpy as np

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.patch.set_facecolor('#0d1117')
fig.suptitle('§17: 2P vs 4P Strategic Differences', color='white', fontsize=14, fontweight='bold')

# ── Panel 1: 2P point symmetry ──
ax = axes[0]
ax.set_facecolor('#0d1117')
ax.set_xlim(0, 100); ax.set_ylim(0, 100)
ax.set_aspect('equal')
ax.set_title('2P: Point-Symmetric Map\n(Q1 vs Q4)', color='white', fontsize=11)

import matplotlib.patches as mpatches

# Sun
sun = plt.Circle((50, 50), 10, color='#FFD700', alpha=0.9, zorder=5)
ax.add_patch(sun)

# P0 home (Q1: top-right)
ax.scatter([68], [72], s=250, color='#00CFFF', zorder=6, marker='*')
ax.text(70, 74, 'P0 Home\n(Q1)', color='#00CFFF', fontsize=8, ha='left')

# P1 home (Q4: bottom-left, mirror of P0)
ax.scatter([32], [28], s=250, color='#FF6B6B', zorder=6, marker='*')
ax.text(22, 24, 'P1 Home\n(Q4)', color='#FF6B6B', fontsize=8, ha='left')

# Contested neutral (symmetric pairs)
neutrals_p0 = [(80, 60), (62, 85), (75, 78)]
neutrals_p1 = [(20, 40), (38, 15), (25, 22)]
for (x0, y0), (x1, y1) in zip(neutrals_p0, neutrals_p1):
    ax.scatter([x0], [y0], s=80, color='#00FF88', zorder=4, marker='o')
    ax.scatter([x1], [y1], s=80, color='#00FF88', zorder=4, marker='o')
    ax.plot([x0, x1], [y0, y1], color='#00FF88', linewidth=0.7, linestyle=':', alpha=0.5)
    ax.plot([50], [50], 'x', color='#00FF88', markersize=4)  # center marker

ax.text(50, 1, 'Identical race distance to every neutral', color='#888888',
        fontsize=8, ha='center')

# Quadrant dividers
ax.axhline(50, color='#333', linewidth=0.8, linestyle='--')
ax.axvline(50, color='#333', linewidth=0.8, linestyle='--')
ax.text(75, 95, 'Q1 (P0)', color='#00CFFF', fontsize=8, ha='center', alpha=0.6)
ax.text(25, 5, 'Q4 (P1)', color='#FF6B6B', fontsize=8, ha='center', alpha=0.6)
ax.tick_params(colors='#888888')
for spine in ax.spines.values(): spine.set_edgecolor('#333333')

# ── Panel 2: 4P quadrant layout ──
ax2 = axes[1]
ax2.set_facecolor('#0d1117')
ax2.set_xlim(0, 100); ax2.set_ylim(0, 100)
ax2.set_aspect('equal')
ax2.set_title('4P: Rotational-Symmetric Map\n(one player per quadrant)', color='white', fontsize=11)

sun2 = plt.Circle((50, 50), 10, color='#FFD700', alpha=0.9, zorder=5)
ax2.add_patch(sun2)

player_positions = [(72, 72), (28, 72), (28, 28), (72, 28)]  # Q1, Q2, Q3, Q4
player_colors = ['#00CFFF', '#00FF88', '#FF6B6B', '#FF9500']
player_labels = ['P0 (Q1)', 'P1 (Q2)', 'P2 (Q3)', 'P3 (Q4)']
for (px, py), color, label in zip(player_positions, player_colors, player_labels):
    ax2.scatter([px], [py], s=250, color=color, zorder=6, marker='*')
    offset = (4, 3) if px > 50 else (-4, 3)
    ha = 'left' if px > 50 else 'right'
    ax2.text(px + offset[0], py + offset[1], label, color=color, fontsize=8, ha=ha)

ax2.axhline(50, color='#333', linewidth=0.8, linestyle='--')
ax2.axvline(50, color='#333', linewidth=0.8, linestyle='--')
ax2.text(50, 1, 'Expand YOUR quadrant first, then cross-border', color='#888888',
         fontsize=8, ha='center')
ax2.tick_params(colors='#888888')
for spine in ax2.spines.values(): spine.set_edgecolor('#333333')

# ── Panel 3: FFA bystander advantage ──
ax3 = axes[2]
ax3.set_facecolor('#0d1117')
ax3.set_title('FFA Bystander Advantage\n(4P combat simulation)', color='white', fontsize=11)
ax3.axis('off')

scenarios = [
    ("Direct attack\n(2P style)",
     "You send 80 ships\nEnemy has 60 ships",
     "You spend 80, gain planet\nNet cost: 80 ships", "#FF6B6B"),
    ("FFA bystander\n(4P style)",
     "Enemy A: 80 ships\nEnemy B: 75 ships clash first",
     "Survivors: A gets 5 ships\nYou send 10 -> gain planet!\nNet cost: 10 ships", "#00FF88"),
]
for idx, (title, setup, result, color) in enumerate(scenarios):
    y_top = 0.85 - idx * 0.45
    ax3.text(0.05, y_top, title, transform=ax3.transAxes,
             color=color, fontsize=11, fontweight='bold', va='top')
    ax3.text(0.05, y_top - 0.08, setup, transform=ax3.transAxes,
             color='#CCCCCC', fontsize=9, va='top', family='monospace')
    ax3.text(0.05, y_top - 0.22, result, transform=ax3.transAxes,
             color=color, fontsize=9, va='top', family='monospace')

ax3.text(0.5, 0.05, '80 ships spent vs 10 ships spent\nfor the same result!',
         transform=ax3.transAxes, color='#FFD700', fontsize=10,
         ha='center', fontweight='bold',
         bbox=dict(boxstyle='round', facecolor='#1a1a2e', edgecolor='#FFD700'))

plt.tight_layout()
plt.show()

# Win-rate table from local experiments
print("Local 2P experiment results (10-game series):")
print("-" * 50)
data = [
    ("Snipe", "Tactical-Heuristic", 7, 10),
    ("Snipe", "Elite",              4, 10),
    ("Tactical", "Elite",           2, 10),
]
for a, b, w, n in data:
    bar = '#' * w + '.' * (n - w)
    print(f"  {a:10s} vs {b:22s}: {w}/{n}  [{bar}]")
print()
print("Key insight: forward-sim agents (Snipe) >> reactive heuristics (Tactical)")
print("             for 2P. Elite-level agents require macro-physics prediction.")
