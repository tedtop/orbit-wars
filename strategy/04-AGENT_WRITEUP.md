# Orbit Wars — Algorithm Agent Roster

Twenty agents, each implementing one strategy blueprint supplied by domain
scientists, split into two sets of ten. Every bot lives in `agents/<name>.py`,
is discovered automatically by `arena.py`, and is named for the strategy it
realizes.

## Ground rules every agent follows

- **Self-contained & pure stdlib.** No `numpy`, no `torch`, no external deps —
  the Kaggle runner is sandboxed with a ~1 second/turn budget. Each file is a
  single drop-in agent.
- **Crash-safe.** The last callable is `agent(obs, config)`, wrapped in
  try/except so a bug forfeits the turn (returns `[]`) instead of the match.
- **Shared physics core.** All bots reuse the engine-grounded helpers proven in
  `coordinated_strike_interceptor.py`: iterative **lead-solution** aiming at a target's *future*
  orbital position, sun line-segment avoidance, planet-blocking checks,
  rotation-sign inference, and the log-curve fleet-speed model.
- **Capture discipline.** Combat resolves per tick, so to flip a planet a fleet
  must land **strictly more** ships than the garrison-at-arrival
  (`current_ships + production × ETA`). Bots size fleets accordingly and never
  sit idle (each has an expansion fallback).
- **The ML blueprints (Set 1 #6–10) have no training data and can't ship
  `torch`.** They are implemented as **faithful fixed-weight forward passes** —
  the real architecture (graph convolution, recurrent trend, Q-grid, logistic
  FNN, two-stage cascade) with hand-tuned coefficients standing in for learned
  weights. This is documented in each file's header.

---

## Set 1 — Deterministic, Heuristic & ML blueprints

### 1. Predictive Kinematic Interceptor — `predictive_kinematic_interceptor.py`
**Blueprint:** Fix the "Nearest Planet Sniper" by solving exact orbital
mechanics. Iteratively solve travel time `T` (fleet speed scales with size,
inner-zone targets move), aim at the future position `(r·cos(θ₀+ωT),
r·sin(θ₀+ωT))`, compute true garrison at arrival `current + production·T`,
reject any path that crosses the Sun, and send exactly `Garrison_T + 1`.
**Implementation:** The iterative lead solver converges travel time against the
moving target; each candidate is filtered by a sun segment-intersection test and
sized to the minimum capturing wave (garrison+1) bounded by the source's
spendable surplus. The "send exactly enough" discipline is the whole point —
ships not needed for a capture stay home and keep expanding.

### 2. The Vulture (Second-Mover Advantage) — `the_vulture.py`
**Blueprint:** Let opponents weaken each other. Scan enemy fleets in flight,
project each one's arrival and target, subtract it from that planet's future
garrison; if it will flip or gut the planet, launch to arrive **one turn after**
the enemy and steal it with a minimal clean-up crew.
**Implementation:** Each enemy fleet's target is inferred from its heading
(smallest angular error along its path); a quick combat sim predicts the
post-impact owner and residual garrison. When a planet is about to fall, a
lead-solution fleet is timed to land just after, sized only to beat the depleted
residual plus the production that regrows in the gap. Falls back to normal
expansion when no carrion is available.

### 3. Frontline Consolidation (Logistic Network) — `frontline_consolidation.py`
**Blueprint:** Move dormant ships to the front. Convex-hull the owned planets;
perimeter planets are "Frontline", interior ones are "Interior". Interior
planets ship their surplus (keeping a small base garrison) to the nearest
frontline hub, which spends the massed ships on captures.
**Implementation:** An Andrew's monotone-chain hull (pure Python; <3 planets ⇒
all frontline) classifies the network each turn. Interior planets ferry surplus
to their nearest hull node via lead solution; hubs run the proven ROI capture
loop against nearby enemy/neutral targets.

### 4. Minimax Fleet Allocation — `minimax_fleet_allocation.py`
**Blueprint:** Treat a local cluster as a zero-sum game. K-Means the planets,
run a depth-2–3 minimax with alpha-beta pruning over candidate fleet dispatches,
evaluating leaves as our regional ships minus the enemy's, and play the maximin
move.
**Implementation:** A tiny deterministic K-Means picks the most contested
cluster; a **bounded** candidate set per side (hold / attack best 1–2 nearby
targets) is searched with alpha-beta over a fast production+combat forward sim —
branching and depth are capped so a 40-planet board stays well under the 1s
budget. Expansion fallback when no enemy is near.

### 5. Lyapunov Defense Heuristic (Chaos Theory) — `lyapunov_defense_heuristic.py`
**Blueprint:** Model volatility `V = enemy ships within a 3-turn radius / our
garrison`; track `dV/dt`; when a planet diverges exponentially (positive
Lyapunov exponent) flag it critical and have neighbors pre-emptively reinforce
before the attack lands.
**Implementation:** A per-player history of `V` (reset at step 0) feeds a short
linear regression for `dV/dt`; a planet is critical when `V` is high **and**
rising fast. Neighbors dispatch stabilizing reinforcements via lead solution;
otherwise the bot expands with `coordinated_strike_interceptor`-style scoring.

### 6. Graph Neural Network Value Estimator — `graph_neural_network_value_estimator.py`
**Blueprint:** Planets are nodes (`[owner,x,y,radius,ships,production]`), edges
fully connected weighted by travel time; a GCN emits a scalar Strategic Value
per node; attack the best Strategic-Value / Capture-Cost node.
**Implementation (fixed-weight GCN forward pass, no training):** Two
message-passing layers aggregate neighbor features weighted by `1/travel_time`
through hand-tuned `W1` (relu) and `w2` (tanh) into a value in [0,1] that rewards
high production, low garrison and proximity to our forces. Targets are ranked by
value ÷ (garrison-at-arrival + margin), with a small ETA discount, then assigned
greedily.

### 7. LSTM Fleet Trajectory Forecaster — `lstm_fleet_trajectory_forecaster.py`
**Blueprint:** Track a 10-turn sliding window of per-planet ship counts; an LSTM
predicts next-turn deltas; a large negative delta = a planet about to empty
itself to attack — so capture that soon-undefended planet first.
**Implementation (untrained-LSTM surrogate):** A per-player ring buffer feeds a
fixed-weight recurrent trend rule (exponential-smoothing trend + a
buildup-then-stall detector). When an enemy planet is flagged as about to
launch, a lead-solution capture is timed to arrive as it empties. Expansion
fallback otherwise.

### 8. Deep Q-Network Macro-Strategist — `deep_q_network_macro_strategist.py`
**Blueprint:** State = 10×10 downsampled grid (our ships / enemy ships /
production channels); action = `[source, target, percentage]`; rewards +10
capture, +1 net ship, −5 loss; a CNN emits Q-values, act greedily.
**Implementation (fixed-weight Q-function):** The board is downsampled to 10×10
feature maps; `Q(source,target)` is a hand-designed value whose terms mirror the
reward shaping (production gain, capturability, time cost, source-loss risk).
The greedy argmax action is executed with a capturing lead-solution fleet.

### 9. Target Classifier (FNN) — `target_classifier_fnn.py`
**Blueprint:** Per enemy planet, features `[garrison, distance to our nearest,
our nearest's garrison, turn]`; an FNN outputs `P(Attack)`; if `>0.85`, divert
production to the threatened frontline planet and turtle up.
**Implementation (fixed-weight 2-layer FNN, sigmoid):** Normalized features feed
hand-tuned weights so `P(Attack)` rises when an enemy planet is large, close, and
our nearest defense is thin. Above threshold it reinforces the threatened planet
from neighbors; it also runs a normal expansion loop so it still grows.

### 10. Cascading Classifier-Regressor Chain — `cascading_classifier_regressor.py`
**Blueprint:** Stage 1 binary classifier → threat map of who attacks whom next
turn; Stage 2 linear regressor → exact predicted fleet sizes; then intercept
mid-flight or counter the source with `PredictedShips + 1`.
**Implementation (two fixed-weight stages):** A hand-weighted logistic over each
(enemy-source, our-target) pair builds the threat map; a linear predictor
estimates attack size from enemy garrison + production. **Engine reality:** fleets
collide only with planets, never other fleets, so mid-flight interception is
impossible — the bot instead pre-empts by countering the predicted source or
reinforcing the predicted target.

---

## Set 2 — Cross-disciplinary blueprints

### 1. Macroeconomic Gravity Model — `macroeconomic_gravity_model.py`
**Blueprint:** Mass `M` = (ours) `ships + production·10`, (theirs) strategic
value; attraction `F_ij = M_i·M_j / D_ij²`; normalize to a distribution and let
ship flow follow it, fortifying clusters and starving outposts.
**Implementation:** `F_ij` becomes the allocation weight for distributing each
planet's spare ships (above a defensive reserve), biased toward weak
high-production targets; capturing flows are sized to take the planet, friendly
flows act as reinforcement.

### 2. Kinematic Wave Theory (LWR traffic flow) — `kinematic_wave_theory.py`
**Blueprint:** Optimize supply-line throughput instead of one slow armada or a
trickle: the instant a safe interior planet hits a small batch (~6 ships), launch
that batch — a relentless continuous stream.
**Implementation:** Interior planets stream fixed ~6-ship waves to their nearest
frontline hub the moment they reach the threshold; hubs accumulate the stream and
spend it on captures. (Engine note: "6" is the throughput batch size, not a
speed — only ~1000-ship fleets approach the 6.0 speed cap.)

### 3. Artificial Potential Fields — `artificial_potential_fields.py`
**Blueprint:** Sun = infinite repulsor; enemy planets = attractors scaling with
weakness; our interior planets = mild repulsors. Launch along the summed gradient
so ships slide past the Sun and pool into weak points like fluid.
**Implementation:** Each owned planet sums attractive pulls (`weakness/d²` toward
weak enemy/neutral nodes) with sun and interior repulsion; ships launch along the
resultant heading. A hard sun segment-check overrides the field so no fleet is
ever flung through the Sun.

### 4. Reaction-Diffusion Turing Patterns — `reaction_diffusion_turing_patterns.py`
**Blueprint:** Ship production = activator, enemy presence = inhibitor; diffuse
them on a grid (`Di > Da`) so stable spots/stripes form a self-repairing
defense-in-depth perimeter, with opportunistic captures where the activator
dominates.
**Implementation:** A coarse 20×20 grid runs ≤5 Gray-Scott iterations/turn
(bounded for the time budget); planets reinforce where the activator is high and
the inhibitor encroaches, and attack where our activator clearly dominates. Field
state persists per player and resets at step 0.

### 5. Stigmergic Pheromone Routing (Ant Colony) — `stigmergic_pheromone_routing.py`
**Blueprint:** A virtual pheromone grid; 1-ship scouts deposit positive trails
when they survive/capture and negative when destroyed; evaporate ×0.95/turn; main
fleets route along the strongest, battle-tested trails.
**Implementation:** A 25×25 grid (few scouts/turn, tracked by fleet id) records
+trail for survivors/captures and −trail for fleets lost to the Sun/out-of-bounds,
biasing main-fleet target bearings — while analytic sun/planet checks remain the
hard safety gate. State persists per player.

### 6. Markowitz Portfolio Optimization — `markowitz_portfolio_optimization.py`
**Blueprint:** Treat attacks as investments: return = target production, risk =
garrison-at-arrival uncertainty. Rather than one big attack, solve the efficient
frontier and diversify across a basket of 3–5 targets.
**Implementation:** Each candidate gets a return (production) and a risk proxy
(nearby enemy strength, distance, garrison-growth uncertainty); a mean-variance
objective `return − λ·risk` selects a 3–5 target basket under the ship budget,
each sized to capture. Diversification limits wipeout risk from a counter-surge.

### 7. Distributed PID Controllers — `distributed_pid_controllers.py`
**Blueprint:** Hold each garrison at a setpoint (Frontline ~50, Interior ~5).
Each turn `Dispatch = Kp·e + Ki·∫e + Kd·de/dt`; interior planets push surplus
out, frontline planets absorb it — smooth, non-oscillating.
**Implementation:** Per-planet integral and previous-error persist per player
(reset at step 0). Surplus above setpoint is routed to the nearest higher-setpoint
frontline planet, or spent on a nearby capture when the front is satisfied; small
gains keep the controller stable.

### 8. Susceptible-Infected-Recovered (SIR) Model — `susceptible_infected_recovered_model.py`
**Blueprint:** Enemy as contagion. Classify Susceptible / Infected / Recovered;
compute a sector reproduction number `R₀`; if `R₀ ≥ 1` quarantine — all nearby
planets stop other attacks and blockade that sector to starve its growth.
**Implementation:** A 4×4 sector grid classifies planets and estimates `R₀` as
enemy production density vs. our delivery rate to the sector. When a sector goes
critical it boosts scores for capturing that cluster's weakest fringe to choke
local production; otherwise it expands normally.

### 9. Comet-Riding Ephemeris Exploitation — `comet_riding_ephemeris_exploitation.py`
**Blueprint:** Ignore standard planets; capture comets as they enter the board,
ride their high production across the map, and just before they exit fire a
tangential strike at the enemy's soft interior planets.
**Implementation:** Comets are intercepted early (lead solution, `is_comet=True`)
and garrisoned to compound their production; when a comet's remaining path drops
below ~6 steps, its massed ships are launched at the best reachable enemy interior
planet. Falls back to normal expansion when no comets are present.

### 10. Bayesian Wave Function Collapse — `bayesian_wave_function_collapse.py`
**Blueprint:** Treat the future as a superposition; run Monte-Carlo rollouts to
build a probability heat map of enemy positions, update priors from observed
fleets (collapse the wave function), and target nodes that stay safe across the
most timelines.
**Implementation:** ~25 bounded rollouts over a 7-turn horizon (hard-capped for
the budget — measured ~4 ms/turn) tally which targets we capture and hold most
reliably; a Bayesian prior on enemy aggression updates via EMA from observed
fleet counts. Targets are weighted by `success_probability · production / cost`.

---

## First experiments — the homegrown lineage

Before the twenty scientist blueprints, three hand-built bots were iterated in
the arena. They are the evolutionary line that produced the shared physics core
every other agent copies, so they live in `agents/*.py` too. Each name now
describes its method (old codename in parentheses).

### A. Greedy Lead Interceptor — `greedy_lead_interceptor.py` (was `graceful_sloth_v1`)
The baseline. Iterative **lead-solution** targeting of moving planets, a
threat-cone **defensive reserve**, and **greedy ROI** (production ÷ cost)
source→target assignment, with sun avoidance. ~67% vs the starter sniper.

### B. Path-Aware Lead Interceptor — `path_aware_lead_interceptor.py` (was `graceful_sloth_v2`)
Adds the engine-grounded targeting fixes: **intervening-planet path-blocking**
(a shot is absorbed by the first planet its line crosses), a **segment-correct
sun check** (only the travel segment up to the target matters), and a **two-pass
speed-accurate lead** (fleet speed scales with size, so the ETA is re-solved with
the ships actually sent). ~75–80% vs the starter sniper.

### C. Coordinated Strike Interceptor — `coordinated_strike_interceptor.py` (was `comet_wraith_v3`)
The current **champion**. Keeps B's offense and adds **coordinated
simultaneous-arrival strikes** (combat resolves per tick, so it gangs up sources
that land the *same* tick instead of feeding an enemy garrison piecemeal), a
**lean +1 capture margin** (frees surplus ships for faster expansion), and an
**end-game efficiency gate** (late, ships are worth more at home than on a
capture that can't repay itself). Beats B ~64% and the starter sniper ~76%.

Lineage: `greedy_lead_interceptor` → `path_aware_lead_interceptor` →
`coordinated_strike_interceptor`.

---

## Baselines — meet `random` and `starter` (the sniper)

The arena also fields two opponents that are **not files in `agents/`** — they
are built into the kaggle-environments engine and referenced by name (see
`BUILTINS = ["random", "starter"]` in `arena.py`). Their source lives in:

```
.venv/lib/python3.11/site-packages/kaggle_environments/envs/orbit_wars/orbit_wars.py
    def random_agent(obs):   # line ~765
    def starter_agent(obs):  # line ~778  (the "Nearest Planet Sniper")
    agents = {"random": random_agent, "starter": starter_agent}
```

It matters how you read a result against each:

- **`random`** — for every owned planet with ships, fires **half** its ships at a
  uniformly **random angle** (only if that half is ≥ 20 ships). No targeting at
  all. It is the **floor, not a benchmark.** Random play essentially always loses
  to anything coherent, so a win over `random` proves only that a bot is *alive*:
  it imports, returns legal moves, doesn't crash, and doesn't sit idle. **If you
  lose to `random`, something is genuinely wrong with your bot.** That's the whole
  point of the per-agent smoke test — a pass/fail liveness check, not a measure of
  skill. Don't be impressed by a fat win rate over `random`; it has to lose.
- **`starter`** — the **"Nearest Planet Sniper,"** the bot the original blueprints
  were written to beat. Each owned planet sends **half** its ships (if ≥ 20)
  straight at the **nearest static (non-orbiting) planet it doesn't own**. It is a
  real strategy but deliberately naive, which is why it's only the first
  *meaningful* bar:
  - targets **static planets only** — it ignores every orbiting planet and comet;
  - **no lead** — aims at the target's current position (fine for static targets,
    useless against moving ones);
  - **no sun avoidance** — will fire a fleet straight through the Sun and lose it;
  - **no garrison math** — always sends a flat half, with no check that it's
    enough to actually capture.

Read results in tiers: beating `random` = **alive**; beating `starter` =
**competent**; beating `coordinated_strike_interceptor` and ranking high in the
full round-robin = **actually strong**.

---

## Running them

```bash
# Full round-robin of every agent in agents/ (plus engine baselines):
.venv/bin/python arena.py

# One specific 2- or 4-player match:
.venv/bin/python arena.py --players the_vulture,coordinated_strike_interceptor

# List discovered bots:
.venv/bin/python arena.py --list
```

The reigning champion to beat is `coordinated_strike_interceptor`. Every agent
above passes its `random` liveness smoke test — i.e. it is *alive*, not
necessarily strong — so the real standings are the arena's to decide.
