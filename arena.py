"""
Orbit Wars arena — unified 2P and 4P tournament with OpenSkill ratings.

Ratings use the PlackettLuce model (open-source TrueSkill variant), which
handles N-player games natively and accounts for opponent strength — closer
to what Kaggle's leaderboard computes than a raw win%.

Usage:
    python arena.py                       # 2P round-robin, all agents/ bots
    python arena.py --mode 4p             # 4P tournament, C(N,4) quads
    python arena.py --mode 4p --games 20  # fixed games per quad
    python arena.py --players a,b,c,d --mode 4p  # one specified matchup
    python arena.py --players a,b         # one 2P matchup
    python arena.py --list
    python arena.py --promote <bot>

To include archive bots, name them explicitly:
    python arena.py --players coordinated_strike,the_vulture,markowitz,bayesian_wave_function_collapse --mode 4p
"""

import argparse
import glob
import itertools
import logging
import math
import multiprocessing
import os
import sys
import time

# ---------------------------------------------------------------------------
# Silence kaggle-environments' noisy import (jax/open_spiel chatter)
# ---------------------------------------------------------------------------
os.environ["KAGGLE_ENV_LOG_LEVEL"] = "ERROR"
logging.disable(logging.INFO)
_devnull = open(os.devnull, "w")
_old_stdout, _old_stderr = sys.stdout, sys.stderr
sys.stdout = sys.stderr = _devnull
_saved_fd1, _saved_fd2 = os.dup(1), os.dup(2)
os.dup2(_devnull.fileno(), 1)
os.dup2(_devnull.fileno(), 2)
try:
    from kaggle_environments import make
finally:
    os.dup2(_saved_fd1, 1); os.close(_saved_fd1)
    os.dup2(_saved_fd2, 2); os.close(_saved_fd2)
    sys.stdout, sys.stderr = _old_stdout, _old_stderr
    _devnull.close()
    logging.disable(logging.NOTSET)

from openskill.models import PlackettLuce

ROOT       = os.path.dirname(os.path.abspath(__file__))
AGENTS_DIR = os.path.join(ROOT, "agents")
ARCHIVE_DIR = os.path.join(ROOT, "archive", "agents")
BUILTINS   = ["random", "starter"]

_model = PlackettLuce()


# ---------------------------------------------------------------------------
# Terminal output helpers
# ---------------------------------------------------------------------------
class C:
    """ANSI palette; blanked when not a TTY or --no-color is set."""
    enabled = sys.stdout.isatty()
    RESET = "\033[0m"; BOLD = "\033[1m"; DIM = "\033[2m"
    RED = "\033[31m"; GREEN = "\033[32m"; YELLOW = "\033[33m"
    BLUE = "\033[34m"; MAGENTA = "\033[35m"; CYAN = "\033[36m"; WHITE = "\033[37m"

    @classmethod
    def wrap(cls, code, text):
        return f"{code}{text}{cls.RESET}" if cls.enabled else str(text)


def out(text="", end="\n"):
    sys.stdout.write(text + end)
    sys.stdout.flush()


def rewrite(text):
    sys.stdout.write(("\r\033[K" if C.enabled else "\r") + text)
    sys.stdout.flush()


def fmt_dur(secs):
    secs = int(secs)
    if secs < 60:   return f"{secs}s"
    m, s = divmod(secs, 60)
    if m < 60:      return f"{m}m{s:02d}s"
    h, m = divmod(m, 60)
    return f"{h}h{m:02d}m"


def bar(frac, width=24):
    filled = int(round(frac * width))
    return "█" * filled + "░" * (width - filled)


# ---------------------------------------------------------------------------
# Bot discovery and resolution
# ---------------------------------------------------------------------------
def discover_bots():
    """Return {name: spec} — agents/ only.  No random/starter by default."""
    bots = {}
    for path in sorted(glob.glob(os.path.join(AGENTS_DIR, "*.py"))):
        name = os.path.splitext(os.path.basename(path))[0]
        if name == "__init__":
            continue
        bots[name] = path
    return bots


def resolve(name, bots):
    """Map a name to an agent spec.

    Priority: explicit file path → agents/ name → builtin → archive bot →
    short-suffix alias within agents/.
    """
    if os.path.exists(name):
        return name
    if name in bots:
        return bots[name]
    if name in BUILTINS:
        return name
    arc = os.path.join(ARCHIVE_DIR, name + ".py")
    if os.path.exists(arc):
        return arc
    matches = [n for n in bots if n.endswith(name) or n.endswith("_" + name)]
    if len(matches) == 1:
        return bots[matches[0]]
    raise SystemExit(f"Unknown bot '{name}'. Try --list to see available bots.")


def _ordered_names(bots):
    return list(bots.keys())


# ---------------------------------------------------------------------------
# Engine interface
# ---------------------------------------------------------------------------
def play_one(specs, seed):
    """Run one game; return (rewards, env) or None on engine error.

    Must stay silent — runs in worker processes whose stdout is shared."""
    env = make("orbit_wars", configuration={"seed": seed}, debug=False)
    try:
        env.run(list(specs))
    except Exception:
        return None
    rewards = [s.reward for s in env.steps[-1]]
    if any(r is None for r in rewards):
        return None
    return rewards, env


def _total_ships(obs, player_id):
    """Total ships (garrison + in-flight) for a given player in the final obs."""
    return (sum(p[5] for p in obs["planets"] if p[1] == player_id) +
            sum(f[5] for f in obs["fleets"]  if f[1] == player_id))


def _placements_4p(rewards, env):
    """Return placements[slot] = rank (1=winner) for each engine slot.

    Winner has reward=1; non-winners all get -1, so we use total ship count
    as a tiebreaker to assign 2nd/3rd/4th place."""
    n = len(rewards)
    obs = env.steps[-1][0].observation  # orbit-wars is fully observable
    ships = [_total_ships(obs, i) for i in range(n)]
    order = sorted(range(n), key=lambda i: (-rewards[i], -ships[i]))
    placements = [0] * n
    for rank, slot in enumerate(order):
        placements[slot] = rank + 1
    return placements


# ---------------------------------------------------------------------------
# OpenSkill helpers
# ---------------------------------------------------------------------------
def new_ratings(names):
    return {n: _model.rating(name=n) for n in names}


def update_ratings_2p(ratings, name_a, name_b, outcome):
    """Update ratings after a 2P game.
    outcome: True = A won, False = B won, None = draw."""
    ra, rb = ratings[name_a], ratings[name_b]
    if outcome is True:
        res = _model.rate([[ra], [rb]])
        ratings[name_a], ratings[name_b] = res[0][0], res[1][0]
    elif outcome is False:
        res = _model.rate([[rb], [ra]])
        ratings[name_a], ratings[name_b] = res[1][0], res[0][0]
    else:
        res = _model.rate([[ra], [rb]], ranks=[1, 1])
        ratings[name_a], ratings[name_b] = res[0][0], res[1][0]


def update_ratings_4p(ratings, seat_names, placements):
    """Update ratings after a 4P game.
    seat_names[i] = bot name in seat i; placements[i] = their rank (1=winner)."""
    pairs = sorted(zip(placements, seat_names), key=lambda x: x[0])
    names_by_place = [name for _, name in pairs]
    teams = [[ratings[n]] for n in names_by_place]
    result = _model.rate(teams)
    for i, name in enumerate(names_by_place):
        ratings[name] = result[i][0]


# ---------------------------------------------------------------------------
# Wilson CI (kept for 2P pairwise diagnostics)
# ---------------------------------------------------------------------------
def wilson_ci(wins, losses, draws, z=1.96):
    n = wins + losses + draws
    if n == 0:
        return 0.5, 0.5
    phat = (wins + 0.5 * draws) / n
    denom = 1.0 + z * z / n
    half = (z / denom) * math.sqrt(phat * (1 - phat) / n + z * z / (4 * n * n))
    return phat, half


# ---------------------------------------------------------------------------
# Worker functions (module-level so multiprocessing can pickle them)
# ---------------------------------------------------------------------------
def _play_2p_task(args):
    """Run one 2P game; return (a_is_p0, outcome) where outcome in {True,False,None}.
    Returns None if the engine errors."""
    spec_a, spec_b, seed, a_is_p0 = args
    specs = [spec_a, spec_b] if a_is_p0 else [spec_b, spec_a]
    result = play_one(specs, seed)
    if result is None:
        return a_is_p0, None  # engine error; treat as skip
    rewards, _ = result
    top = [i for i, r in enumerate(rewards) if r == 1]
    if len(top) != 1:
        return a_is_p0, "draw"
    slot_a = 0 if a_is_p0 else 1
    return a_is_p0, (top[0] == slot_a)  # True if A won


def _play_4p_task(args):
    """Run one 4P game; return (seat_names, placements) or None on engine error."""
    specs, seed, seat_names = args
    result = play_one(specs, seed)
    if result is None:
        return None
    rewards, env = result
    placements = _placements_4p(rewards, env)
    return seat_names, placements


def _iter_results(tasks, pool):
    if pool is None:
        for t in tasks:
            fn = t[0]
            yield fn(t[1])
    else:
        raise RuntimeError("use pool.imap_unordered directly")


# ---------------------------------------------------------------------------
# 2P Round-Robin Tournament
# ---------------------------------------------------------------------------
def run_2p_tournament(bots, seed_offset, jobs, margin, min_games, max_games, fixed_games):
    names = _ordered_names(bots)
    pairs = list(itertools.combinations(names, 2))
    adaptive = fixed_games is None
    cap = fixed_games if not adaptive else max_games

    ratings = new_ratings(names)
    stats  = {n: {"w": 0, "l": 0, "d": 0, "g": 0} for n in names}
    h2h    = {a: {b: [0, 0] for b in names if b != a} for a in names}

    out("\n" + C.wrap(C.BOLD + C.CYAN, "⚔  ORBIT WARS — 2P ROUND ROBIN  ⚔"))
    out(C.wrap(C.DIM, "─" * 60))
    out(f"  combatants : {C.wrap(C.BOLD, str(len(names)))}  ({', '.join(names)})")
    out(f"  pairings   : {len(pairs)}")
    if adaptive:
        out(f"  stop rule  : 95% CI ≤ ±{margin*100:.0f}%  (min {min_games}, max {max_games})")
    else:
        out(f"  per pairing: {fixed_games} games (fixed)")
    out(f"  each game  : ≤500 turns")
    out(C.wrap(C.DIM, "─" * 60))

    pool = multiprocessing.Pool(jobs) if jobs > 1 else None
    batch_size = max(jobs, 1) if pool else 8
    games_done = 0
    t_start = time.time()
    spinner = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"

    try:
        for idx, (a, b) in enumerate(pairs, 1):
            spec_a, spec_b = resolve(a, bots), resolve(b, bots)
            aw = bw = dr = skipped = 0
            ticks = []
            g = 0
            score = 0.5; half = 0.5; reason = "max games"

            head = (f"{C.wrap(C.BOLD, f'[{idx:>2}/{len(pairs)}]')} "
                    f"{C.wrap(C.GREEN, a)} {C.wrap(C.DIM, 'vs')} {C.wrap(C.RED, b)}")
            out("\n" + head)

            while g < cap:
                nb = min(batch_size, cap - g)
                # Side-swap every game, fresh board every 2 games
                tasks = []
                for k in range(nb):
                    gi = g + k
                    a_is_p0 = (gi % 2 == 0)
                    board_seed = seed_offset + gi // 2
                    tasks.append((spec_a, spec_b, board_seed, a_is_p0))
                g += nb

                if pool:
                    results = pool.imap_unordered(_play_2p_task, tasks)
                else:
                    results = (_play_2p_task(t) for t in tasks)

                for a_is_p0, outcome in results:
                    games_done += 1
                    if outcome is None:
                        skipped += 1
                        ticks.append(C.wrap(C.YELLOW, "?"))
                    elif outcome == "draw":
                        dr += 1
                        ticks.append(C.wrap(C.DIM, "·"))
                        update_ratings_2p(ratings, a, b, None)
                    elif outcome is True:
                        aw += 1
                        ticks.append(C.wrap(C.GREEN, "W"))
                        update_ratings_2p(ratings, a, b, True)
                    else:
                        bw += 1
                        ticks.append(C.wrap(C.RED, "L"))
                        update_ratings_2p(ratings, a, b, False)

                    score, half = wilson_ci(aw, bw, dr)
                    spin = C.wrap(C.CYAN, spinner[games_done % len(spinner)])
                    shown = "".join(ticks[-40:])
                    line = (f"  {spin} {shown}  "
                            f"{C.wrap(C.GREEN, str(aw))}–{C.wrap(C.RED, str(bw))}"
                            + (f" ({dr}d)" if dr else "")
                            + f"  {score*100:4.0f}% ±{half*100:.1f}%"
                            + f"  {C.wrap(C.DIM, f'n={aw+bw+dr}/{cap}')}")
                    rewrite(line)

                played = aw + bw + dr
                if adaptive and played >= min_games and half <= margin:
                    reason = "converged"
                    break

            played = aw + bw + dr
            rewrite("")
            if aw > bw:
                verdict = C.wrap(C.GREEN + C.BOLD, f"{a} wins")
            elif bw > aw:
                verdict = C.wrap(C.RED + C.BOLD, f"{b} wins")
            else:
                verdict = C.wrap(C.YELLOW, "tied")
            tag = C.wrap(C.DIM, f"[{reason}]") if adaptive else ""
            out(f"  {''.join(ticks[-60:])}")
            out(f"  → {a} {C.wrap(C.BOLD, f'{aw}–{bw}')} {b}"
                f"  {score*100:.0f}% ±{half*100:.1f}%"
                f"  ({played} games"
                f"{', %d skipped' % skipped if skipped else ''}"
                f"{', %d draws' % dr if dr else ''})  {verdict} {tag}")

            stats[a]["w"] += aw; stats[a]["l"] += bw; stats[a]["d"] += dr; stats[a]["g"] += played
            stats[b]["w"] += bw; stats[b]["l"] += aw; stats[b]["d"] += dr; stats[b]["g"] += played
            h2h[a][b][0] += aw; h2h[a][b][1] += played
            h2h[b][a][0] += bw; h2h[b][a][1] += played

            elapsed = time.time() - t_start
            rate = games_done / elapsed if elapsed else 0
            avg_per_pair = games_done / idx
            remaining = (len(pairs) - idx) * avg_per_pair
            eta = remaining / rate if rate else 0
            out(f"  {C.wrap(C.DIM, bar(idx / len(pairs)))} "
                f"pairing {idx}/{len(pairs)} · {games_done} games · "
                f"{C.wrap(C.CYAN, fmt_dur(elapsed))} elapsed · "
                f"~{C.wrap(C.YELLOW, fmt_dur(eta))} left · "
                f"{rate:.1f} g/s")
    finally:
        if pool:
            pool.close()
            pool.join()

    _print_leaderboard_2p(names, ratings, stats, time.time() - t_start)
    _print_matrix(names, h2h)


# ---------------------------------------------------------------------------
# 4P Tournament
# ---------------------------------------------------------------------------
def run_4p_tournament(bots, seed_offset, jobs, margin, min_games, max_games, fixed_games):
    names = _ordered_names(bots)
    if len(names) < 4:
        raise SystemExit(f"4P mode needs at least 4 bots; got {len(names)}. "
                         f"Add archive bots with --players a,b,c,d")
    quads = list(itertools.combinations(names, 4))
    adaptive = fixed_games is None
    cap = fixed_games if not adaptive else max_games

    ratings    = new_ratings(names)
    place_dist = {n: {1: 0, 2: 0, 3: 0, 4: 0} for n in names}
    games_ct   = {n: 0 for n in names}

    out("\n" + C.wrap(C.BOLD + C.CYAN, "⚔  ORBIT WARS — 4P TOURNAMENT  ⚔"))
    out(C.wrap(C.DIM, "─" * 60))
    out(f"  combatants : {C.wrap(C.BOLD, str(len(names)))}  ({', '.join(names)})")
    out(f"  quads      : {len(quads)}")
    if adaptive:
        out(f"  stop rule  : 95% CI on 1st-place rate ≤ ±{margin*100:.0f}%  "
            f"(min {min_games}, max {max_games})")
    else:
        out(f"  per quad   : {fixed_games} games (fixed)")
    out(f"  seat rotation: cycle 4 permutations per group of 4 games")
    out(C.wrap(C.DIM, "─" * 60))

    pool = multiprocessing.Pool(jobs) if jobs > 1 else None
    batch_size = max(jobs, 1) if pool else 8
    games_done = 0
    t_start = time.time()
    spinner = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"

    try:
        for qi, quad in enumerate(quads, 1):
            quad = list(quad)
            quad_specs = [resolve(n, bots) for n in quad]
            ticks = []
            g = 0
            reason = "max games"

            head = (f"{C.wrap(C.BOLD, f'[{qi:>2}/{len(quads)}]')} "
                    + C.wrap(C.CYAN, ' · '.join(quad)) + ")")
            out("\n" + head)

            while g < cap:
                nb = min(batch_size, cap - g)
                tasks = []
                for k in range(nb):
                    gi = g + k
                    rot = gi % 4
                    seat_names = [quad[(rot + i) % 4] for i in range(4)]
                    seat_specs = [quad_specs[(rot + i) % 4] for i in range(4)]
                    tasks.append((seat_specs, seed_offset + gi, seat_names))
                g += nb

                if pool:
                    results = list(pool.imap_unordered(_play_4p_task, tasks))
                else:
                    results = [_play_4p_task(t) for t in tasks]

                for r in results:
                    games_done += 1
                    if r is None:
                        ticks.append(C.wrap(C.YELLOW, "?"))
                        continue
                    seat_names, placements = r
                    update_ratings_4p(ratings, seat_names, placements)
                    for i, name in enumerate(seat_names):
                        p = placements[i]
                        place_dist[name][p] += 1
                        games_ct[name] += 1
                    winner_name = seat_names[placements.index(1)]
                    ticks.append(C.wrap(C.GREEN, winner_name[0].upper()))

                    spin = C.wrap(C.CYAN, spinner[games_done % len(spinner)])
                    shown = "".join(ticks[-40:])
                    rewrite(f"  {spin} {shown}  {C.wrap(C.DIM, f'n={sum(games_ct[n] for n in quad)//4}/{cap}')}")

                # Adaptive stopping: stop when all bots in this quad have tight 1st-place CI
                if adaptive and g >= min_games:
                    all_tight = True
                    for n in quad:
                        n1 = place_dist[n][1]
                        ng = games_ct[n]
                        if ng == 0:
                            all_tight = False
                            break
                        _, half = wilson_ci(n1, ng - n1, 0)
                        if half > margin:
                            all_tight = False
                            break
                    if all_tight:
                        reason = "converged"
                        break

            rewrite("")
            games_this_quad = sum(games_ct[n] for n in quad) // 4
            places_str = "  ".join(
                f"{C.wrap(C.BOLD, n[:6])}: "
                f"{C.wrap(C.GREEN, str(place_dist[n][1]))}/"
                f"{place_dist[n][2]}/"
                f"{place_dist[n][3]}/"
                f"{C.wrap(C.DIM, str(place_dist[n][4]))}"
                for n in quad
            )
            out(f"  {''.join(ticks[-60:])}")
            out(f"  1st/2nd/3rd/4th: {places_str}")
            out(f"  ({games_this_quad} games, {reason})")

            elapsed = time.time() - t_start
            rate = games_done / elapsed if elapsed else 0
            avg_per_quad = games_done / qi
            remaining = (len(quads) - qi) * avg_per_quad
            eta = remaining / rate if rate else 0
            out(f"  {C.wrap(C.DIM, bar(qi / len(quads)))} "
                f"quad {qi}/{len(quads)} · {games_done} games · "
                f"{C.wrap(C.CYAN, fmt_dur(elapsed))} elapsed · "
                f"~{C.wrap(C.YELLOW, fmt_dur(eta))} left · "
                f"{rate:.1f} g/s")
    finally:
        if pool:
            pool.close()
            pool.join()

    _print_leaderboard_4p(names, ratings, place_dist, games_ct, time.time() - t_start)


# ---------------------------------------------------------------------------
# Leaderboard display
# ---------------------------------------------------------------------------
def _print_leaderboard_2p(names, ratings, stats, elapsed):
    out("\n" + C.wrap(C.BOLD + C.CYAN, "═" * 60))
    out(C.wrap(C.BOLD + C.CYAN, "  🏆  FINAL LEADERBOARD — 2P"))
    out(C.wrap(C.BOLD + C.CYAN, "═" * 60))

    ranked = sorted(names,
                    key=lambda n: -ratings[n].ordinal())
    out(f"  {'#':>2}  {'bot':<28} {'μ':>6} {'σ':>5} {'ordinal':>8}  {'win%':>5}  {'games':>5}")
    out("  " + C.wrap(C.DIM, "─" * 62))
    medals = {0: "🥇", 1: "🥈", 2: "🥉"}
    for rank, n in enumerate(ranked):
        r   = ratings[n]
        s   = stats[n]
        pct = s["w"] / s["g"] * 100 if s["g"] else 0
        tag = medals.get(rank, f"{rank+1:>2}")
        col = C.GREEN if rank == 0 else C.WHITE if pct >= 50 else C.DIM
        out(f"  {tag:>2}  {C.wrap(col + C.BOLD, f'{n:<28}')} "
            f"{r.mu:>6.2f} {r.sigma:>5.2f} {r.ordinal():>8.2f}  "
            f"{pct:>4.0f}%  {s['g']:>5}")
    out("  " + C.wrap(C.DIM, "─" * 62))
    out(C.wrap(C.DIM, f"  finished in {fmt_dur(elapsed)}"))


def _print_leaderboard_4p(names, ratings, place_dist, games_ct, elapsed):
    out("\n" + C.wrap(C.BOLD + C.CYAN, "═" * 60))
    out(C.wrap(C.BOLD + C.CYAN, "  🏆  FINAL LEADERBOARD — 4P"))
    out(C.wrap(C.BOLD + C.CYAN, "═" * 60))

    ranked = sorted(names, key=lambda n: -ratings[n].ordinal())
    out(f"  {'#':>2}  {'bot':<28} {'μ':>6} {'σ':>5} {'ordinal':>8}  "
        f"{'1st':>5} {'2nd':>5} {'3rd':>5} {'4th':>5}  {'games':>5}")
    out("  " + C.wrap(C.DIM, "─" * 75))
    medals = {0: "🥇", 1: "🥈", 2: "🥉"}
    for rank, n in enumerate(ranked):
        r  = ratings[n]
        g  = games_ct[n]
        pd = place_dist[n]
        def pct(k): return f"{pd[k]/g*100:.0f}%" if g else "—"
        tag = medals.get(rank, f"{rank+1:>2}")
        col = C.GREEN if rank == 0 else C.WHITE
        out(f"  {tag:>2}  {C.wrap(col + C.BOLD, f'{n:<28}')} "
            f"{r.mu:>6.2f} {r.sigma:>5.2f} {r.ordinal():>8.2f}  "
            f"{pct(1):>5} {pct(2):>5} {pct(3):>5} {pct(4):>5}  {g:>5}")
    out("  " + C.wrap(C.DIM, "─" * 75))
    out(C.wrap(C.DIM, f"  finished in {fmt_dur(elapsed)}"))


def _short(name):
    parts = name.split("_")
    return "_".join(parts[-2:]) if len(parts) > 2 else name


def _print_matrix(names, h2h):
    out("\n" + C.wrap(C.BOLD, "  HEAD-TO-HEAD  ")
        + C.wrap(C.DIM, "(row's win% vs column)"))
    short = {n: _short(n) for n in names}
    colw = max(max(len(s) for s in short.values()) + 2, 7)
    header = " " * 22 + "".join(f"{short[n]:>{colw}}" for n in names)
    out(C.wrap(C.DIM, header))
    for a in names:
        cells = []
        for b in names:
            if a == b:
                cells.append(C.wrap(C.DIM, f"{'—':>{colw}}"))
            else:
                aw, games = h2h[a][b]
                pct = aw / games * 100 if games else 0
                col = C.GREEN if pct > 50 else C.RED if pct < 50 else C.YELLOW
                cells.append(C.wrap(col, f"{pct:>{colw-1}.0f}%"))
        out(f"  {C.wrap(C.BOLD, f'{a:<19}')} {''.join(cells)}")


# ---------------------------------------------------------------------------
# Single specified matchup (quick one-off)
# ---------------------------------------------------------------------------
def run_match(names, bots, games, mode, seed_offset):
    """Play a single specified 2P or 4P matchup for a fixed number of games."""
    if mode == "4p" or len(names) == 4:
        _run_single_4p(names, bots, games, seed_offset)
    else:
        _run_single_2p(names, bots, games, seed_offset)


def _run_single_2p(names, bots, games, seed_offset):
    a, b = names
    spec_a, spec_b = resolve(a, bots), resolve(b, bots)
    ratings = new_ratings([a, b])
    aw = bw = dr = 0

    out(f"\n{C.wrap(C.BOLD, f'{a} vs {b}')}  ({games} games)")
    for g in range(games):
        a_is_p0 = (g % 2 == 0)
        board_seed = seed_offset + g // 2
        result = _play_2p_task((spec_a, spec_b, board_seed, a_is_p0))
        _, outcome = result
        if outcome is None:
            continue
        elif outcome == "draw":
            dr += 1
            update_ratings_2p(ratings, a, b, None)
        elif outcome is True:
            aw += 1
            update_ratings_2p(ratings, a, b, True)
        else:
            bw += 1
            update_ratings_2p(ratings, a, b, False)
        out(f"  {C.wrap(C.DIM, f'{aw+bw+dr}/{games}...')}", end="\r")

    played = aw + bw + dr
    out(f"\nResults: {a} {aw}–{bw} {b}  ({dr} draws, {played} games)")
    out(f"  {a}: μ={ratings[a].mu:.2f} σ={ratings[a].sigma:.2f} ordinal={ratings[a].ordinal():.2f}")
    out(f"  {b}: μ={ratings[b].mu:.2f} σ={ratings[b].sigma:.2f} ordinal={ratings[b].ordinal():.2f}")


def _run_single_4p(names, bots, games, seed_offset):
    quad = list(names)
    quad_specs = [resolve(n, bots) for n in quad]
    ratings    = new_ratings(quad)
    place_dist = {n: {1: 0, 2: 0, 3: 0, 4: 0} for n in quad}
    games_ct   = {n: 0 for n in quad}

    out(f"\n{C.wrap(C.BOLD, ' · '.join(quad))}  ({games} games, 4P)")
    for g in range(games):
        rot = g % 4
        seat_names = [quad[(rot + i) % 4] for i in range(4)]
        seat_specs = [quad_specs[(rot + i) % 4] for i in range(4)]
        r = _play_4p_task((seat_specs, seed_offset + g, seat_names))
        if r is None:
            continue
        seat_names_r, placements = r
        update_ratings_4p(ratings, seat_names_r, placements)
        for i, name in enumerate(seat_names_r):
            place_dist[name][placements[i]] += 1
            games_ct[name] += 1
        out(f"  {C.wrap(C.DIM, f'{g+1}/{games}...')}", end="\r")

    out(f"\nResults over {games} games:")
    for n in sorted(quad, key=lambda n: -ratings[n].ordinal()):
        g = games_ct[n]
        pd = place_dist[n]
        out(f"  {n:<28} μ={ratings[n].mu:.2f} σ={ratings[n].sigma:.2f}  "
            f"1st:{pd[1]}  2nd:{pd[2]}  3rd:{pd[3]}  4th:{pd[4]}  (n={g})")


# ---------------------------------------------------------------------------
# Promote helper (unchanged from archive version)
# ---------------------------------------------------------------------------
PROMOTE_HEADER = '''# =============================================================================
# Orbit Wars SUBMISSION — promoted from bot: {name}
# Generated by arena.py --promote {name}. Do not edit here; edit agents/{name}.py.
# =============================================================================
'''


def promote(name, bots):
    spec = bots.get(name) or resolve(name, bots)
    if not spec or spec in BUILTINS:
        raise SystemExit(f"Can only promote a file-based bot. Got '{name}'.")
    name = os.path.splitext(os.path.basename(spec))[0]
    body = open(spec).read()
    out_text = PROMOTE_HEADER.format(name=name) + "\n" + body
    with open(os.path.join(ROOT, "main.py"), "w") as f:
        f.write(out_text)
    out(f"Promoted '{name}' → main.py")
    out(f"Submit with:  kaggle competitions submit orbit-wars -f main.py -m '{name}'")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    bots = discover_bots()

    p = argparse.ArgumentParser(description="Orbit Wars arena (2P + 4P, OpenSkill ratings)")
    p.add_argument("--mode", choices=["2p", "4p"], default="2p",
                   help="2P round-robin (default) or 4P placement tournament")
    p.add_argument("--margin", type=float, default=0.05,
                   help="95%% CI half-width for adaptive stopping (default 0.05 = ±5%%)")
    p.add_argument("--min-games", type=int, default=24,
                   help="min games before adaptive stop kicks in")
    p.add_argument("--max-games", type=int, default=400,
                   help="hard cap on games per pairing/quad")
    p.add_argument("--games", "-g", type=int, default=None,
                   help="play exactly N games per pairing/quad (disables adaptive)")
    p.add_argument("--seed-offset", type=int, default=0)
    p.add_argument("--jobs", "-j", type=int, default=os.cpu_count() or 1,
                   help="parallel workers (default: all cores; 1 = serial)")
    p.add_argument("--players", help="comma-separated bot names for a single matchup")
    p.add_argument("--no-color", action="store_true")
    p.add_argument("--list", action="store_true", help="list discoverable bots")
    p.add_argument("--promote", metavar="BOT", help="copy agents/BOT.py to main.py")
    args = p.parse_args()

    if args.no_color:
        C.enabled = False

    if args.list:
        out("\nagents/ bots:")
        for n, spec in bots.items():
            out(f"  {n:<30} {spec}")
        out("\narchive/ bots (pass by name or path):")
        for path in sorted(glob.glob(os.path.join(ARCHIVE_DIR, "*.py"))):
            n = os.path.splitext(os.path.basename(path))[0]
            if n != "__init__":
                out(f"  {n}")
        out("\nbuiltins (for smoke tests): " + ", ".join(BUILTINS))
        return

    if args.promote:
        promote(args.promote, bots)
        return

    if args.players:
        player_names = [x.strip() for x in args.players.split(",") if x.strip()]
        if len(player_names) not in (2, 4):
            raise SystemExit("A game needs exactly 2 or 4 players.")
        run_match(player_names, bots, args.games or 20, args.mode, args.seed_offset)
        return

    if args.mode == "4p":
        run_4p_tournament(bots, args.seed_offset, args.jobs,
                          args.margin, args.min_games, args.max_games, args.games)
    else:
        run_2p_tournament(bots, args.seed_offset, args.jobs,
                          args.margin, args.min_games, args.max_games, args.games)


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
