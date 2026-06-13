"""
Orbit Wars arena — fully automated round-robin tournament.

Every bot in agents/*.py fights every other bot (plus the engine baselines
"random" and "starter"), side-swapped to cancel first-mover bias. Results stream
live so you can watch it unfold; a leaderboard and head-to-head matrix print at
the end.

    .venv/bin/python arena.py                      # full auto tournament (default)
    .venv/bin/python arena.py --games 100          # more games per pairing
    .venv/bin/python arena.py --players v3,starter # one specific matchup
    .venv/bin/python arena.py --list
    .venv/bin/python arena.py --promote coordinated_strike_interceptor   # copy bot -> main.py

The unit everywhere is the *game*: one ≤500-turn match on one board. By default
the arena keeps playing each pairing until the win rate is statistically settled
(95% confidence interval within ±margin); pass --games N to play a fixed count.
"""
import argparse
import glob
import itertools
import logging
import math
import os
import sys
import time
from multiprocessing import Pool

os.environ["KAGGLE_ENV_LOG_LEVEL"] = "ERROR"
logging.disable(logging.INFO)
# Silence the heavy kaggle-environments import (jax/open_spiel chatter).
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

ROOT = os.path.dirname(os.path.abspath(__file__))
AGENTS_DIR = os.path.join(ROOT, "agents")
BUILTINS = ["random", "starter"]  # engine-provided baselines


# --------------------------------------------------------------------------- #
# Pretty output helpers
# --------------------------------------------------------------------------- #
class C:
    """ANSI palette; blanked out when not writing to a TTY (or --no-color)."""
    enabled = sys.stdout.isatty()
    RESET = "\033[0m"; BOLD = "\033[1m"; DIM = "\033[2m"
    RED = "\033[31m"; GREEN = "\033[32m"; YELLOW = "\033[33m"
    BLUE = "\033[34m"; MAGENTA = "\033[35m"; CYAN = "\033[36m"; WHITE = "\033[37m"

    @classmethod
    def wrap(cls, code, text):
        return f"{code}{text}{cls.RESET}" if cls.enabled else str(text)


def out(text="", end="\n"):
    """Write + flush immediately so nothing ever feels frozen."""
    sys.stdout.write(text + end)
    sys.stdout.flush()


def rewrite(text):
    """Overwrite the current terminal line in place (live tick stream)."""
    if C.enabled:
        sys.stdout.write("\r\033[K" + text)
    else:
        sys.stdout.write("\r" + text)
    sys.stdout.flush()


def fmt_dur(secs):
    secs = int(secs)
    if secs < 60:
        return f"{secs}s"
    m, s = divmod(secs, 60)
    if m < 60:
        return f"{m}m{s:02d}s"
    h, m = divmod(m, 60)
    return f"{h}h{m:02d}m"


def bar(frac, width=24):
    filled = int(round(frac * width))
    return "█" * filled + "░" * (width - filled)


# --------------------------------------------------------------------------- #
# Bot discovery / engine plumbing
# --------------------------------------------------------------------------- #
def discover_bots():
    """Return {name: spec} where spec is a file path (our bots) or a builtin name."""
    bots = {}
    for path in sorted(glob.glob(os.path.join(AGENTS_DIR, "*.py"))):
        name = os.path.splitext(os.path.basename(path))[0]
        if name == "__init__":
            continue
        bots[name] = path
    for b in BUILTINS:
        bots[b] = b
    return bots


def resolve(name, bots):
    """Map a user-typed name to an engine agent spec, tolerating short aliases."""
    if name in bots:
        return bots[name]
    matches = [n for n in bots if n.endswith(name) or n.endswith("_" + name)]
    if len(matches) == 1:
        return bots[matches[0]]
    raise SystemExit(f"Unknown bot '{name}'. Available: {', '.join(bots)}")


def play_one(specs, seed):
    """Run one game; return list of rewards per slot, or None on engine error.

    Must stay silent (no printing): it runs in worker processes whose stdout
    shares the terminal, and any output would corrupt the live tick line."""
    env = make("orbit_wars", configuration={"seed": seed}, debug=False)
    try:
        env.run(list(specs))
    except Exception:  # noqa: BLE001 — skip the seed, parent tallies it as '?'
        return None
    rewards = [s.reward for s in env.steps[-1]]
    if any(r is None for r in rewards):
        return None  # an agent errored/timed out -> skip, don't miscount
    return rewards


def _play_task(args):
    """Pool worker: run one game, preserve which slot bot A occupied."""
    order_specs, seed, a_is_p0 = args
    return a_is_p0, play_one(order_specs, seed)


def _winners(rewards):
    """Slot indices that strictly win (engine gives reward 1 to every top scorer;
    a shared top is a draw)."""
    top = [i for i, r in enumerate(rewards) if r == 1]
    return top if len(top) == 1 else []  # [] => draw


# --------------------------------------------------------------------------- #
# The tournament
# --------------------------------------------------------------------------- #
def _ordered_names(bots):
    """Our bots first (newest-looking last), then the builtin baselines."""
    ours = [n for n in bots if bots[n] != n]
    return list(dict.fromkeys(ours + BUILTINS))


def _iter_results(tasks, pool):
    """Yield (a_is_p0, rewards) for each game, in completion order if pooled."""
    if pool is None:
        for t in tasks:
            yield _play_task(t)
    else:
        for res in pool.imap_unordered(_play_task, tasks):
            yield res


def _game_task(spec_a, spec_b, seed_offset, g):
    """One game for index g: a fresh board every 2 games, sides alternated so
    first-mover effects cancel (A as P0 on even g, B as P0 on odd g)."""
    board_seed = seed_offset + g // 2
    if g % 2 == 0:
        return ([spec_a, spec_b], board_seed, True)    # A is P0
    return ([spec_b, spec_a], board_seed, False)       # B is P0


def wilson_ci(wins, losses, draws, z=1.96):
    """95% Wilson interval on bot A's score (a draw counts as half a win).
    Returns (score, half_width). Stays sensible at the 0%/100% extremes,
    unlike the plain normal approximation."""
    n = wins + losses + draws
    if n == 0:
        return 0.5, 0.5
    phat = (wins + 0.5 * draws) / n
    denom = 1.0 + z * z / n
    half = (z / denom) * math.sqrt(phat * (1 - phat) / n + z * z / (4 * n * n))
    return phat, half


def tournament(bots, seed_offset, jobs, margin, min_games, max_games, fixed_games):
    names = _ordered_names(bots)
    pairs = list(itertools.combinations(names, 2))
    adaptive = fixed_games is None

    stats = {n: {"w": 0, "l": 0, "d": 0, "g": 0} for n in names}
    # h2h[a][b] = [A's wins vs B, games A vs B] -> rendered as a win% so cells
    # stay comparable even when pairings play different numbers of games.
    h2h = {a: {b: [0, 0] for b in names if b != a} for a in names}

    title = C.wrap(C.BOLD + C.CYAN, "⚔  ORBIT WARS — FULL ROUND ROBIN  ⚔")
    out("\n" + title)
    out(C.wrap(C.DIM, "─" * 60))
    out(f"  combatants : {C.wrap(C.BOLD, str(len(names)))}  "
        f"({', '.join(names)})")
    out(f"  pairings   : {len(pairs)}")
    if adaptive:
        out(f"  stop rule  : play until 95% CI ≤ ±{margin*100:.0f}%  "
            f"(min {min_games}, max {max_games} games/pairing)")
        est = len(pairs) * (min_games + max_games) // 2
        out(f"  est. games : ~{est} (data-dependent)  "
            f"≈ {fmt_dur(est * 1.7 / max(jobs, 1))} on {jobs} core"
            f"{'s' if jobs != 1 else ''}")
    else:
        tot = len(pairs) * fixed_games
        out(f"  per pairing: {fixed_games} games (fixed)")
        out(f"  TOTAL      : {C.wrap(C.BOLD + C.YELLOW, str(tot))} games"
            f"  ≈ {fmt_dur(tot * 1.7 / max(jobs, 1))} on {jobs} core"
            f"{'s' if jobs != 1 else ''}")
    out(f"  each game  : ≤500 turns")
    out(C.wrap(C.DIM, "─" * 60))

    pool = Pool(jobs) if jobs > 1 else None
    cap = fixed_games if not adaptive else max_games
    batch = max(jobs, 1) if pool else 8
    games_done = 0
    t_start = time.time()
    spinner = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"

    try:
        for idx, (a, b) in enumerate(pairs, 1):
            spec_a, spec_b = resolve(a, bots), resolve(b, bots)
            aw = bw = dr = skipped = 0
            ticks = []  # one glyph per game, from A's perspective
            g = 0       # games launched for this pairing
            score = 0.5; half = 0.5; reason = "max games"

            head = (f"{C.wrap(C.BOLD, f'[{idx:>2}/{len(pairs)}]')} "
                    f"{C.wrap(C.GREEN, a)} {C.wrap(C.DIM, 'vs')} {C.wrap(C.RED, b)}")
            out("\n" + head)

            while g < cap:
                nb = min(batch, cap - g)
                batch_tasks = [_game_task(spec_a, spec_b, seed_offset, g + k)
                               for k in range(nb)]
                g += nb
                for a_is_p0, rewards in _iter_results(batch_tasks, pool):
                    games_done += 1
                    if rewards is None:
                        skipped += 1
                        ticks.append(C.wrap(C.YELLOW, "?"))
                    else:
                        w = _winners(rewards)
                        if not w:
                            dr += 1
                            ticks.append(C.wrap(C.DIM, "·"))
                        elif (w[0] == 0) == a_is_p0:
                            aw += 1
                            ticks.append(C.wrap(C.GREEN, "W"))
                        else:
                            bw += 1
                            ticks.append(C.wrap(C.RED, "L"))

                    score, half = wilson_ci(aw, bw, dr)
                    spin = C.wrap(C.CYAN, spinner[games_done % len(spinner)])
                    # keep the tick stream from running off the screen
                    shown = "".join(ticks[-40:])
                    line = (f"  {spin} {shown}  "
                            f"{C.wrap(C.GREEN, str(aw))}–{C.wrap(C.RED, str(bw))}"
                            + (f" ({dr}d)" if dr else "")
                            + f"  {score*100:4.0f}% ±{half*100:.1f}%"
                            + f"  {C.wrap(C.DIM, f'n={aw+bw+dr}/{cap}')}")
                    rewrite(line)

                # stop early once the estimate is precise enough
                played = aw + bw + dr
                if adaptive and played >= min_games and half <= margin:
                    reason = "converged"
                    break

            played = aw + bw + dr
            rewrite("")  # clear the live line
            if aw > bw:
                verdict = C.wrap(C.GREEN + C.BOLD, f"{a} wins")
            elif bw > aw:
                verdict = C.wrap(C.RED + C.BOLD, f"{b} wins")
            else:
                verdict = C.wrap(C.YELLOW, "tied")
            tag = (C.wrap(C.DIM, f"[{reason}]") if adaptive else "")
            out(f"  {''.join(ticks[-60:])}")
            out(f"  → {a} {C.wrap(C.BOLD, f'{aw}–{bw}')} {b}"
                f"  {score*100:.0f}% ±{half*100:.1f}%"
                f"  ({played} games{', %d skipped' % skipped if skipped else ''}"
                f"{', %d draws' % dr if dr else ''})  {verdict} {tag}")

            # tally
            stats[a]["w"] += aw; stats[a]["l"] += bw; stats[a]["d"] += dr; stats[a]["g"] += played
            stats[b]["w"] += bw; stats[b]["l"] += aw; stats[b]["d"] += dr; stats[b]["g"] += played
            h2h[a][b][0] += aw; h2h[a][b][1] += played
            h2h[b][a][0] += bw; h2h[b][a][1] += played

            # overall progress (total is data-dependent, so ETA is an estimate)
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
        if pool is not None:
            pool.close()
            pool.join()

    _print_leaderboard(names, stats, time.time() - t_start)
    _print_matrix(names, h2h)


def _print_leaderboard(names, stats, elapsed):
    out("\n" + C.wrap(C.BOLD + C.CYAN, "═" * 60))
    out(C.wrap(C.BOLD + C.CYAN, "  🏆  FINAL LEADERBOARD"))
    out(C.wrap(C.BOLD + C.CYAN, "═" * 60))

    ranked = sorted(names, key=lambda n: (
        -(stats[n]["w"] / stats[n]["g"] if stats[n]["g"] else 0), -stats[n]["w"]))
    out(f"  {'#':>2}  {'bot':<22} {'W':>4} {'L':>4} {'D':>4} {'win%':>6}  ")
    out("  " + C.wrap(C.DIM, "─" * 54))
    medals = {0: "🥇", 1: "🥈", 2: "🥉"}
    for rank, n in enumerate(ranked):
        s = stats[n]
        pct = s["w"] / s["g"] * 100 if s["g"] else 0
        tag = medals.get(rank, f"{rank+1:>2}")
        color = (C.GREEN if rank == 0 else
                 C.WHITE if pct >= 50 else C.DIM)
        winbar = C.wrap(C.GREEN, "▏" * int(pct / 5))
        out(f"  {tag:>2}  {C.wrap(color + C.BOLD, f'{n:<22}')} "
            f"{s['w']:>4} {s['l']:>4} {s['d']:>4} {pct:>5.0f}%  {winbar}")
    out("  " + C.wrap(C.DIM, "─" * 54))
    out(C.wrap(C.DIM, f"  finished in {fmt_dur(elapsed)}"))


def _short(name):
    """Compact but still-distinguishable label, e.g. greedy_lead_interceptor -> lead_interceptor."""
    parts = name.split("_")
    return "_".join(parts[-2:]) if len(parts) > 2 else name


def _print_matrix(names, h2h):
    out("\n" + C.wrap(C.BOLD, "  HEAD-TO-HEAD  ")
        + C.wrap(C.DIM, "(row's win% vs column)"))
    short = {n: _short(n) for n in names}
    colw = max(max(len(s) for s in short.values()) + 2, 7)
    header = " " * 20 + "".join(f"{short[n]:>{colw}}" for n in names)
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
        out(f"  {C.wrap(C.BOLD, f'{a:<17}')} {''.join(cells)}")


# --------------------------------------------------------------------------- #
# Single specified matchup (kept for quick one-off checks)
# --------------------------------------------------------------------------- #
def run_match(names, bots, games, seed_offset):
    """Play `games` games for one specified 2- or 4-player matchup.

    2-player games alternate sides every game (new board every 2 games).
    4-player games draw a fresh board each game (the 4-fold symmetry already
    makes the four starting slots equivalent)."""
    specs = [resolve(n, bots) for n in names]
    n = len(names)
    wins = {i: 0 for i in range(n)}
    draws = played = 0

    out(f"\n{C.wrap(C.BOLD, ' vs '.join(names))}  ({games} games)")
    for g in range(games):
        if n == 2:
            board_seed = seed_offset + g // 2
            order = [0, 1] if g % 2 == 0 else [1, 0]
        else:
            board_seed = seed_offset + g
            order = list(range(n))
        rewards = play_one([specs[o] for o in order], board_seed)
        if rewards is None:
            continue
        played += 1
        w = _winners(rewards)
        if not w:
            draws += 1
        else:
            wins[order[w[0]]] += 1
        out(f"  {C.wrap(C.DIM, f'{played}/{games} games...')}", end="\r")

    out(f"\nResults over {played} games:")
    for s in sorted(range(n), key=lambda s: -wins[s]):
        pct = wins[s] / played * 100 if played else 0
        out(f"  {names[s]:<22} {wins[s]:>3} wins  ({pct:4.0f}%)")
    out(f"  draws: {draws}  ({draws / played * 100 if played else 0:.0f}%)")


# --------------------------------------------------------------------------- #
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
    out(f"Promoted '{name}' -> main.py (tagged for submission).")
    out(f"Submit with:  kaggle competitions submit orbit-wars -f main.py -m '{name}'")


def main():
    bots = discover_bots()
    p = argparse.ArgumentParser(description="Orbit Wars arena (auto round-robin)")
    # By default the arena plays each pairing until the result is statistically
    # settled (95% CI within ±margin). --games forces a fixed count instead.
    p.add_argument("--margin", type=float, default=0.05,
                   help="target 95%% CI half-width on win%% (default 0.05 = ±5%%); "
                        "smaller = more games for tighter estimates")
    p.add_argument("--min-games", type=int, default=24,
                   help="games to play before the CI is allowed to stop a pairing")
    p.add_argument("--max-games", type=int, default=400,
                   help="hard cap on games per pairing (for dead-even matchups)")
    p.add_argument("--games", "-g", type=int, default=None,
                   help="play exactly N games per pairing (disables adaptive stop)")
    p.add_argument("--seed-offset", type=int, default=0,
                   help="shift which board layouts are used (for reproducibility)")
    p.add_argument("--jobs", "-j", type=int, default=os.cpu_count() or 1,
                   help="parallel games at once (default: all cores; 1 = serial)")
    p.add_argument("--players", help="comma-separated names for ONE 2/4-player match")
    p.add_argument("--no-color", action="store_true", help="disable ANSI colors")
    p.add_argument("--list", action="store_true", help="list discovered bots")
    p.add_argument("--promote", metavar="BOT", help="copy agents/BOT.py to main.py")
    args = p.parse_args()

    if args.no_color:
        C.enabled = False

    if args.list:
        for n in bots:
            kind = "builtin" if bots[n] in BUILTINS else bots[n]
            out(f"  {n:<24} {kind}")
        return
    if args.promote:
        promote(args.promote, bots)
        return
    if args.players:
        names = [x.strip() for x in args.players.split(",") if x.strip()]
        if len(names) not in (2, 4):
            raise SystemExit("A game needs 2 or 4 players (engine limit).")
        run_match(names, bots, args.games or 50, args.seed_offset)
        return

    tournament(bots, args.seed_offset, args.jobs,
               args.margin, args.min_games, args.max_games, args.games)


if __name__ == "__main__":
    main()
