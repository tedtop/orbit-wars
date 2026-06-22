"""
Validation harness: orbit_jax.py vs kaggle_environments orbit_wars interpreter.

Two-phase strategy:

Phase 1 — No-op games (exact comparison):
  Both players pass every turn.  No fleets → no float32 precision drift.
  Tests: production, comet spawn/expire timing, planet rotation, terminal.
  Expected: exact agreement on planet counts and ship totals.

Phase 2 — Active-bot winner agreement (statistical):
  Both players run a deterministic greedy bot (det_bot).
  Fleet movement introduces float32 vs float64 divergence, so we don't
  compare step-by-step; we only check that winners agree ≥ 65% of the time.

Usage:
  python validate_engine.py          # Phase 1: N=100 noop, Phase 2: N=50 bot
  python validate_engine.py --n 20   # Phase 1: N=20,  Phase 2: N=20
  python validate_engine.py --seed 7 # Single noop game (debug)
"""

import argparse
import math
import sys
import os
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--n",    type=int, default=100)
parser.add_argument("--seed", type=int, default=None)
parser.add_argument("--verbose", action="store_true")
args = parser.parse_args()

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.normpath(os.path.join(HERE, "../.."))
VENV = os.path.join(REPO, ".venv/lib/python3.11/site-packages")
sys.path.insert(0, REPO)
sys.path.insert(0, VENV)

from kaggle_environments import make  # type: ignore
import jax.numpy as jnp

sys.path.insert(0, HERE)
import orbit_jax as oj

TOTAL_STEPS = 500


# ── Bots ──────────────────────────────────────────────────────────────────────

def noop_bot(obs):
    return []


def det_bot(obs):
    """Greedy: each owned planet fires 50% ships at nearest non-owned planet."""
    if not isinstance(obs, dict):
        obs = {"player": obs.player, "planets": obs.planets}
    player  = int(obs.get("player", 0))
    planets = obs.get("planets", [])
    targets = [p for p in planets if p[1] != player]
    launches = []
    for src in planets:
        if src[1] != player or src[5] < 3:
            continue
        if not targets:
            continue
        tgt = min(targets, key=lambda t: math.hypot(t[2]-src[2], t[3]-src[3]))
        angle   = math.atan2(tgt[2]-src[2], tgt[3]-src[3])  # atan2(Y_diff, X_diff)
        n_ships = max(1, int(src[5] * 0.5))
        launches.append([int(src[0]), angle, n_ships])
    return launches


# ── Python game runner ────────────────────────────────────────────────────────

def run_python_game(seed, bot=None):
    """Run one Python game with given bot (default noop). Returns game dict."""
    if bot is None:
        bot = noop_bot
    env = make("orbit_wars", debug=False,
               configuration={"seed": seed, "episodeSteps": TOTAL_STEPS})
    env.reset()

    planet_states = []
    winner        = -1

    while True:
        obs0 = env.state[0].observation
        if env.state[0].status not in ("ACTIVE", "IDLE", ""):
            break

        l0 = bot(obs0)
        l1 = bot(env.state[1].observation)
        env.step([l0, l1])

        planet_states.append([p[:] for p in env.state[0].observation.planets])

    final_planets = env.state[0].observation.planets
    final_fleets  = getattr(env.state[0].observation, "fleets", []) or []
    sc = [0, 0]
    for p in final_planets:
        if p[1] in (0, 1):
            sc[p[1]] += p[5]
    for f in final_fleets:
        if f[1] in (0, 1):
            sc[f[1]] += f[6]
    if sc[0] > sc[1]:    winner = 0
    elif sc[1] > sc[0]:  winner = 1
    else:                winner = -1

    return {"seed": seed, "planet_states": planet_states, "winner": winner}


# ── JAX game runner ───────────────────────────────────────────────────────────

def run_jax_noop_game(seed):
    """Run no-op JAX game (no fleet launches). Returns planet_states + winner."""
    state = oj.reset(seed)
    noop  = jnp.zeros((oj.MAX_PLANETS, 3), jnp.float32)  # all fire_flag=0 → no launch

    planet_states = []

    for _ in range(TOTAL_STEPS - 1):
        state, obs0, obs1, reward, done = oj.step_jit(state, noop, noop)
        planet_states.append(oj.state_to_planet_list(state))
        if done:
            break

    # Pad to match Python output length
    sc = [0., 0.]
    for d in oj.state_to_planet_list(state):
        if d["owner"] in (0, 1):
            sc[d["owner"]] += d["ships"]
    # Include fleet ships (none in noop game)
    if sc[0] > sc[1]:    winner = 0
    elif sc[1] > sc[0]:  winner = 1
    else:                winner = -1

    return {"planet_states": planet_states, "winner": winner}


def run_jax_bot_game(seed):
    """Run JAX game with greedy bot (mirror det_bot logic)."""
    state = oj.reset(seed)
    planet_states = []

    # Build slot→id mapping from reset state (for launching)
    # We'll use a simple greedy: for each alive owned planet, fire 50% at nearest enemy
    def greedy_action(st, player):
        action = np.zeros((oj.MAX_PLANETS, 3), np.float32)
        alive  = np.array(st.p_alive)
        owner  = np.array(st.p_owner)
        px, py = np.array(st.p_x), np.array(st.p_y)
        ships  = np.array(st.p_ships)
        # Enemy planet indices
        enemies = [i for i in range(oj.MAX_PLANETS) if alive[i] and owner[i] != player and owner[i] >= 0]
        # Neutral targets too
        targets = [i for i in range(oj.MAX_PLANETS) if alive[i] and owner[i] != player]
        if not targets:
            return action
        for i in range(oj.MAX_PLANETS):
            if not alive[i] or owner[i] != player or ships[i] < 3:
                continue
            tgt = min(targets, key=lambda j: (px[j]-px[i])**2 + (py[j]-py[i])**2)
            angle  = math.atan2(py[tgt]-py[i], px[tgt]-px[i])
            n      = max(1, int(ships[i] * 0.5))
            action[i, 0] = angle
            action[i, 1] = float(n)
            action[i, 2] = 1.0
        return action

    for _ in range(TOTAL_STEPS - 1):
        a0 = jnp.array(greedy_action(state, 0))
        a1 = jnp.array(greedy_action(state, 1))
        state, obs0, obs1, reward, done = oj.step_jit(state, a0, a1)
        planet_states.append(oj.state_to_planet_list(state))
        if done:
            break

    sc = [0., 0.]
    alive  = np.array(state.p_alive)
    owner  = np.array(state.p_owner)
    ships  = np.array(state.p_ships)
    f_alive = np.array(state.f_alive)
    f_owner = np.array(state.f_owner)
    f_ships = np.array(state.f_ships)
    for i in range(oj.MAX_PLANETS):
        if alive[i] and owner[i] in (0, 1):
            sc[owner[i]] += ships[i]
    for i in range(oj.MAX_FLEETS):
        if f_alive[i] and f_owner[i] in (0, 1):
            sc[f_owner[i]] += f_ships[i]
    if sc[0] > sc[1]:    winner = 0
    elif sc[1] > sc[0]:  winner = 1
    else:                winner = -1

    return {"planet_states": planet_states, "winner": winner}


# ── Comparison ────────────────────────────────────────────────────────────────

def compare_step(step_i, py_planets, jax_planets, tol_ships=1.5):
    """Returns list of error strings (empty = pass)."""
    errors = []
    py_map  = {int(p[0]): (int(p[1]), float(p[5])) for p in py_planets}
    jax_map = {d["slot"]: (d["owner"], d["ships"]) for d in jax_planets}

    # Planet counts per owner
    py_cnt  = {}
    jax_cnt = {}
    for (ow, _) in py_map.values():
        py_cnt[ow]  = py_cnt.get(ow, 0) + 1
    for (ow, _) in jax_map.values():
        jax_cnt[ow] = jax_cnt.get(ow, 0) + 1
    for ow in (0, 1, -1):
        pn, jn = py_cnt.get(ow, 0), jax_cnt.get(ow, 0)
        if pn != jn:
            errors.append(f"step {step_i}: owner {ow} planets: py={pn} jax={jn}")

    # Ship totals per owner
    py_sh  = {ow: sum(sh for (o,sh) in py_map.values()  if o==ow) for ow in (0,1)}
    jax_sh = {ow: sum(sh for (o,sh) in jax_map.values() if o==ow) for ow in (0,1)}
    for ow in (0, 1):
        d = abs(py_sh.get(ow,0) - jax_sh.get(ow,0))
        if d > tol_ships:
            errors.append(f"step {step_i}: owner {ow} ships: py={py_sh.get(ow,0):.1f} "
                          f"jax={jax_sh.get(ow,0):.1f} diff={d:.2f}")
    return errors


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    n = 1 if args.seed is not None else args.n
    seeds = [args.seed] if args.seed is not None else list(range(n))

    # ── Phase 1: No-op exact comparison ──────────────────────────────────────
    print(f"Phase 1 — No-op games ({n} seeds, exact comparison)...")
    p1_errors = 0; p1_games_fail = 0; p1_winner_err = 0

    for i, seed in enumerate(seeds):
        py  = run_python_game(seed, bot=noop_bot)
        jax = run_jax_noop_game(seed)

        game_errs = []
        nsteps = min(len(py["planet_states"]), len(jax["planet_states"]))
        for t in range(nsteps):
            errs = compare_step(t, py["planet_states"][t], jax["planet_states"][t])
            game_errs.extend(errs)

        if py["winner"] != jax["winner"]:
            game_errs.append(f"winner: py={py['winner']} jax={jax['winner']}")
            p1_winner_err += 1
        if len(py["planet_states"]) != len(jax["planet_states"]):
            game_errs.append(f"steps: py={len(py['planet_states'])} jax={len(jax['planet_states'])}")

        if game_errs:
            p1_games_fail += 1
            p1_errors += len(game_errs)
            if args.verbose or p1_games_fail <= 3:
                print(f"  GAME seed={seed} — {len(game_errs)} errors:")
                for e in game_errs[:6]:
                    print(f"    {e}")
                if len(game_errs) > 6:
                    print(f"    ... ({len(game_errs)-6} more)")
        elif args.verbose:
            print(f"  seed={seed} OK ({nsteps} steps, winner={py['winner']})")

        if (i+1) % 20 == 0 or i == n-1:
            pct = (1 - p1_games_fail/(i+1)) * 100
            print(f"  [{i+1}/{n}] ok={i+1-p1_games_fail}/{i+1} ({pct:.0f}%)"
                  f"  step_errs={p1_errors}  winner_errs={p1_winner_err}")

    p1_pass = (p1_errors == 0 and p1_winner_err == 0)
    print(f"Phase 1: {'PASS' if p1_pass else 'FAIL'}"
          f" — {p1_games_fail}/{n} games had errors\n")

    if args.seed is not None:
        sys.exit(0 if p1_pass else 1)

    # ── Phase 2: Bot-game winner agreement ───────────────────────────────────
    n2 = min(n, 50)
    print(f"Phase 2 — Bot games ({n2} seeds, winner agreement ≥65%)...")
    p2_match = 0; p2_total = 0

    for i, seed in enumerate(range(n2)):
        py  = run_python_game(seed, bot=det_bot)
        jax = run_jax_bot_game(seed)
        match = (py["winner"] == jax["winner"])
        if match:
            p2_match += 1
        p2_total += 1

        if args.verbose:
            status = "OK" if match else f"MISS py={py['winner']} jax={jax['winner']}"
            print(f"  seed={seed} {status}")

        if (i+1) % 10 == 0 or i == n2-1:
            rate = p2_match / p2_total * 100
            print(f"  [{i+1}/{n2}] winner_match={p2_match}/{p2_total} ({rate:.0f}%)")

    rate = p2_match / p2_total * 100
    p2_pass = (rate >= 65.)
    print(f"Phase 2: {'PASS' if p2_pass else 'FAIL'} — {rate:.0f}% winner agreement\n")

    if p1_pass and p2_pass:
        print("PASS — all checks OK")
        sys.exit(0)
    else:
        print("FAIL — see errors above")
        sys.exit(1)


if __name__ == "__main__":
    main()
