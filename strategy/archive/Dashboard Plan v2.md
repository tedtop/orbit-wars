Orbit Wars — First Session Implementation Plan (Revised)

 Context

 Ted has two bots live on Kaggle (coordinated_strike ~546, markowitz ~535) with
 9 days until the submission deadline. This session builds the monitoring and
 evaluation infrastructure needed to iterate on bots intelligently:

 1. Archive non-competing bots (housekeeping)
 2. Replace arena.py with a unified 2P+4P arena
 3. Build the pipeline to pull episodes and leaderboard snapshots
 4. Build a Streamlit dashboard that reads pipeline data

 Deferred for later: physics core extraction, submit.py tar.gz bundling,
 downloading public bots (Ted will provide a list).

 ---
 Task 1 — Archive 21 non-active bots

 Move all agents/*.py except the two active ones to archive/agents/.

 Create:
 - archive/__init__.py (empty — makes archive importable)
 - archive/agents/__init__.py (empty)
 - archive/agents/<21 bots>

 Keep in agents/:
 - markowitz_portfolio_optimization.py
 - coordinated_strike_interceptor.py

 Execution: git mv agents/<bot>.py archive/agents/<bot>.py for each of
 the 21 non-active bots (everything except the two above).

 ---
 Task 2 — Replace arena.py with unified 2P + 4P arena

 Archive the existing arena.py, then create a new arena.py that handles both
 modes via --mode 2p|4p.

 Archive: git mv arena.py archive/arena_2p_round_robin.py

 New arena.py — key design:

 Common to both modes:
 - Bot discovery: agents/*.py + builtins (random, starter) + archive bots
 when passed explicitly by path
 - play_one(specs, seed) — unchanged from existing implementation (works for
 any player count)
 - wilson_ci(wins, losses, draws) — unchanged
 - Output helpers: C, out, rewrite, fmt_dur, bar — unchanged
 - Multiprocessing pool with --jobs flag

 2P mode (--mode 2p, the default):
 - Identical behavior to the existing arena.py round-robin
 - All pairs from discovered bots, adaptive Wilson CI stopping
 - Win% ranking leaderboard + head-to-head matrix

 4P mode (--mode 4p):
 - Enumerate all C(N, 4) quad combinations from the bot list
 - Seat rotation: For every group of 4 games within a combo, rotate seats:
 [A,B,C,D], [B,C,D,A], [C,D,A,B], [D,A,B,C] — ensures each bot
 sees each starting position equally over 4 games
 - Placement extraction: Engine rewards in 4P may not distinguish 2nd from
 4th (non-winners typically get 0). Determine placement by sorting players
 descending on reward first, then on total ships (planets + in-flight) as
 tiebreaker. Extract ship counts from env.steps[-1][i].observation.
 - Placement scoring (Borda): 1st=3pts, 2nd=2pts, 3rd=1pt, 4th=0pts
 - Stopping: Play until each bot in the combo has played enough games that
 its mean placement score has a narrow CI (adapt Wilson CI formula to
 placement scores: treat above-median as "win" equivalent for CI calculation)
 - Output: Leaderboard sorted by average Borda score + average placement
 distribution (% 1st, % 2nd, % 3rd, % 4th)

 CLI (same as existing):
 python arena.py                            # 2P round-robin (default)
 python arena.py --mode 4p                  # 4P tournament
 python arena.py --mode 4p --games 20      # fixed 20 games per combo
 python arena.py --players a,b,c,d --mode 4p  # one specific 4P matchup
 python arena.py --list
 python arena.py --promote <bot>

 Note on mixed-mode: Not implementing a --mode mixed now. Run --mode 2p
 and --mode 4p separately. Compare rankings to see if they diverge — that's
 the signal that Kaggle's mixed format matters.

 ---
 Task 3 — Pipeline scripts

 Folder: pipeline/ (new directory)

 pipeline/
   __init__.py
   pull_episodes.py
   download_replays.py
   leaderboard_snapshot.py
   run_pipeline.sh

 Data layout (use existing gitignored folders at root):
 - Replays → replays/YYYY-MM-DD/<episode_id>.json
 - Leaderboard → leaderboards/leaderboard_YYYY-MM-DD_HH-MM.csv
 - Tracking → strategy/tracking.db (SQLite, also writes JSON alongside)

 Submission IDs to track (hardcoded in pull_episodes.py, update manually
 as new bots are submitted):
 SUBMISSION_IDS = {
     "53676654": "coordinated_strike_interceptor_v1",
     "53676680": "markowitz_portfolio_optimization_v1",
 }

 pull_episodes.py:
 - For each submission ID, run kaggle competitions episodes <ID> (or parse
 kaggle competitions submissions orbit-wars to find IDs first if episodes
 command not available)
 - Compare against strategy/tracking.db table episodes to find new ones
 - Print new episode IDs to stdout; write to DB
 - Schema: episodes(episode_id TEXT PK, submission_id TEXT, discovered_at TEXT, downloaded INT DEFAULT 0)

 download_replays.py:
 - Reads undownloaded episodes from strategy/tracking.db
 - For each: kaggle competitions replay <EPISODE_ID> -p replays/YYYY-MM-DD/
 - Saves raw JSON; marks downloaded=1 in DB
 - Priority ordering: losses first (if placement info available), else FIFO
 - Also writes JSON alongside any DB record for human readability

 leaderboard_snapshot.py:
 - kaggle competitions leaderboard orbit-wars --download -p leaderboards/
 - Renames to leaderboard_<YYYY-MM-DD_HH-MM>.csv
 - Logs to strategy/tracking.db table leaderboard_snapshots(filename, taken_at)

 run_pipeline.sh:
 #!/usr/bin/env bash
 set -e
 .venv/bin/python pipeline/pull_episodes.py
 .venv/bin/python pipeline/download_replays.py
 .venv/bin/python pipeline/leaderboard_snapshot.py

 ---
 Task 4 — Streamlit dashboard

 File: dashboard/app.py (single file, no subdirs needed for now)

 Run: streamlit run dashboard/app.py

 Data sources:
 - Countdown: hardcoded deadlines
 - Leaderboard: latest CSV from leaderboards/
 - Submissions: strategy/tracking.db episode history
 - Notes: small JSON file strategy/submission_notes.json (editable in-app)

 Panels:

 1. Competition Timeline + Countdown (top of page)
 - Progress bar from competition start → entry deadline → sub deadline → games end
 - Countdown in days+hours to each milestone, shown in local time and UTC
 - Deadlines: entry deadline Jun 16 2026, sub deadline Jun 23 2026, games end ~Jul 8 2026

 2. My Submissions
 - Table: submission ID, name, current score, episode count, notes
 - Score sparkline per submission (if multiple leaderboard snapshots available)
 - Notes field: editable in-app, saved to strategy/submission_notes.json

 3. Leaderboard (top 20 + our position)
 - Read latest CSV from leaderboards/
 - Highlight "Montana Schmeekler" / "tedtop" row
 - Show score delta vs previous snapshot if available
 - Shows "Prize cutoff (top 10)" line

 4. Submission Log (all submissions with notes)
 - Table: version / description / score / notes / better or worse than prev

 Dependencies to add to requirements.txt:
 streamlit>=1.32.0
 pandas>=2.0.0

 ---
 Verification

 1. Archive check: ls agents/ archive/agents/ — agents/ has exactly 2 files
 2. Arena 2P smoke:
 python arena.py --players coordinated_strike_interceptor,markowitz_portfolio_optimization --games 4
 3. Arena 4P smoke:
 python arena.py --mode 4p --players coordinated_strike_interceptor,markowitz_portfolio_optimization,random,starter --games 8
 4. Pipeline smoke:
 python pipeline/leaderboard_snapshot.py
 ls leaderboards/
 5. Dashboard:
 streamlit run dashboard/app.py
 5. Open in browser, verify countdown shows correct dates, leaderboard loads.

 ---
 Files modified

 ┌──────────┬───────────────────────────────────────────┐
 │   File   │                  Action                   │
 ├──────────┼───────────────────────────────────────────┤
 │ arena.py │ Replace entirely with 2P+4P unified arena │
 └──────────┴───────────────────────────────────────────┘

 Files created

 ┌──────────────────────────────────┬──────────────────────────────────────┐
 │               File               │               Purpose                │
 ├──────────────────────────────────┼──────────────────────────────────────┤
 │ archive/__init__.py              │ Module marker                        │
 ├──────────────────────────────────┼──────────────────────────────────────┤
 │ archive/agents/__init__.py       │ Module marker                        │
 ├──────────────────────────────────┼──────────────────────────────────────┤
 │ archive/agents/<21 bots>         │ Non-competing bots                   │
 ├──────────────────────────────────┼──────────────────────────────────────┤
 │ archive/arena_2p_round_robin.py  │ Preserved original 2P arena          │
 ├──────────────────────────────────┼──────────────────────────────────────┤
 │ pipeline/__init__.py             │ Module marker                        │
 ├──────────────────────────────────┼──────────────────────────────────────┤
 │ pipeline/pull_episodes.py        │ Discover new episodes per submission │
 ├──────────────────────────────────┼──────────────────────────────────────┤
 │ pipeline/download_replays.py     │ Fetch replay JSON                    │
 ├──────────────────────────────────┼──────────────────────────────────────┤
 │ pipeline/leaderboard_snapshot.py │ Download + timestamp leaderboard     │
 ├──────────────────────────────────┼──────────────────────────────────────┤
 │ pipeline/run_pipeline.sh         │ Cron-able runner                     │
 ├──────────────────────────────────┼──────────────────────────────────────┤
 │ dashboard/app.py                 │ Streamlit dashboard                  │
 ├──────────────────────────────────┼──────────────────────────────────────┤
 │ requirements.txt                 │ Declare deps (streamlit, pandas)     │
 └──────────────────────────────────┴──────────────────────────────────────┘