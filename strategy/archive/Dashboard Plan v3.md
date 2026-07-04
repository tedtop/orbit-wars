Orbit Wars — First Session Implementation Plan (Revised v2)

 Context

 Ted has two bots live on Kaggle (coordinated_strike ~546, markowitz ~535) with
 9 days until the submission deadline. This session builds the monitoring and
 evaluation infrastructure needed to iterate on bots intelligently:

 1. Archive non-competing bots (housekeeping)
 2. Replace arena.py with a unified 2P+4P arena using OpenSkill ratings
 3. Build the pipeline to pull episodes and leaderboard snapshots
 4. Build a Streamlit dashboard that reads pipeline data

 Deferred: physics core extraction, submit.py tar.gz bundling, downloading public bots.

 ---
 Task 1 — Archive 21 non-active bots

 Move all agents/*.py except the two active ones to archive/agents/.

 Create: archive/__init__.py, archive/agents/__init__.py (both empty, for importability)

 Keep in agents/: markowitz_portfolio_optimization.py, coordinated_strike_interceptor.py

 Execution: git mv agents/<bot>.py archive/agents/<bot>.py for each of the 21.

 ---
 Task 2 — Replace arena.py with unified 2P+4P OpenSkill arena

 Archive the existing 2P arena, then build a new arena.py with:
 - Unified TrueSkill-style ratings (OpenSkill) for both 2P and 4P
 - --mode 2p|4p flag
 - No random/starter in default bot list

 Archive: git mv arena.py archive/arena_2p_round_robin.py

 Why OpenSkill instead of Wilson CI win-rate

 Kaggle's leaderboard uses a Gaussian N(μ, σ²) system (TrueSkill-flavored).
 Raw win% ignores strength of schedule — A at 90% vs weak bots can rank above
 B at 85% vs strong bots. OpenSkill (open-source TrueSkill variant) solves both:
 - Handles N-player games natively (4P match → 4 pairwise comparisons from
 placement order: 1st beat 2nd, 2nd beat 3rd, etc.)
 - Accounts for opponent strength in each rating update
 - Adaptive stopping becomes natural: play until all σ values drop below a
 threshold (mirrors Kaggle's own "new submissions climb fast, then converge")
 - A unified rating pool for 2P and 4P games simultaneously

 New dependency: openskill (pip installable, no licensing issues)

 New arena.py design

 Bot discovery:
 - agents/*.py only (no random/starter by default)
 - Archive bots loadable by explicit name: --players coordinated_strike,the_vulture
 - random/starter available explicitly for smoke tests: --players markowitz,random

 Rating system:
 from openskill.models import PlackettLuce  # handles N-player natively

 model = PlackettLuce()
 ratings = {bot: model.rating() for bot in bots}  # mu=25, sigma=8.33 initially

 # After each game (2P or 4P):
 # teams = [[rating_a], [rating_b], [rating_c], [rating_d]]
 # ranks = [placement_1, placement_2, placement_3, placement_4]  (1=winner)
 new_ratings = model.rate(teams, ranks=ranks)

 2P mode (--mode 2p, default):
 - Round-robin all pairs
 - Each game: side-swap to cancel positional bias (same as existing arena.py)
 - Adaptive stopping per pairing: play until both bots' σ has changed by less
 than 0.1 for 2 consecutive batches OR hit max_games
 - Also report: Wilson CI win% per pairing (kept as a secondary diagnostic)

 4P mode (--mode 4p):
 - Enumerate all C(N, 4) quad combinations
 - Seat rotation within each combo: cycle 4 permutations over every 4 games
 - Placement extraction: sort slots by descending reward, then by total ships
 (garrison + in-flight) as tiebreaker for non-winners
 - Adaptive stopping: play quads until σ is small enough for all participating bots

 Unified mode (future / nice-to-have): Not implementing now. Run --mode 2p
 and --mode 4p separately, both updating the same ratings dict — they can be
 run independently or sequentially. A --mode both that runs both in one pass
 is trivial to add later.

 Output leaderboard:
 Bot                      μ     σ    95% CI      Games  W/1st  L/4th
 coordinated_strike      27.4  1.2  [25.1–29.7]   84    62%    18%
 markowitz               26.8  1.4  [24.1–29.5]   80    58%    21%

 CLI (backward-compatible with existing flags):
 python arena.py                     # 2P round-robin (default)
 python arena.py --mode 4p           # 4P tournament
 python arena.py --mode 4p --games 20
 python arena.py --players a,b,c,d --mode 4p
 python arena.py --list
 python arena.py --promote <bot>

 ---
 Task 3 — Pipeline scripts

 pipeline/
   __init__.py
   pull_episodes.py
   download_replays.py
   leaderboard_snapshot.py
   run_pipeline.sh

 Data layout (existing gitignored folders at root):
 - Replays → replays/YYYY-MM-DD/<episode_id>.json
 - Leaderboard → leaderboards/leaderboard_YYYY-MM-DD_HH-MM.csv
 - Tracking → strategy/tracking.db (SQLite) — also human-readable via the JSON
 files in replays/

 Submission IDs (hardcoded in pull_episodes.py, update manually):
 SUBMISSION_IDS = {
     "53676654": "coordinated_strike_interceptor_v1",
     "53676680": "markowitz_portfolio_optimization_v1",
 }

 pull_episodes.py:
 - Run kaggle competitions episodes <ID> for each tracked submission
 - Compare against strategy/tracking.db to find new episode IDs
 - Schema: episodes(episode_id PK, submission_id, discovered_at, downloaded INT DEFAULT 0, our_placement, opponent_score)
 - Print new IDs; write to DB

 download_replays.py:
 - Read undownloaded episodes from DB
 - kaggle competitions replay <EPISODE_ID> -p replays/YYYY-MM-DD/
 - Mark downloaded=1 in DB; JSON is the source of truth for replay content

 leaderboard_snapshot.py:
 - kaggle competitions leaderboard orbit-wars --download -p leaderboards/
 - Rename to leaderboard_YYYY-MM-DD_HH-MM.csv
 - Log to strategy/tracking.db table leaderboard_snapshots(filename, taken_at)

 run_pipeline.sh:
 #!/usr/bin/env bash
 set -e
 .venv/bin/python pipeline/pull_episodes.py
 .venv/bin/python pipeline/download_replays.py
 .venv/bin/python pipeline/leaderboard_snapshot.py

 ---
 Task 4 — Streamlit dashboard

 File: dashboard/app.py

 Run: streamlit run dashboard/app.py

 Data sources:
 - Countdown: hardcoded deadlines
 - Leaderboard: latest CSV from leaderboards/
 - Episodes/scores: strategy/tracking.db
 - Submission notes: strategy/submission_notes.json (editable in-app)

 Panels:

 1. Competition Timeline + Countdown
 - Progress bar from start → entry deadline → sub deadline → games end
 - Countdown in days+hours (local time + UTC)
 - Entry deadline: Jun 16 2026 | Sub deadline: Jun 23 2026 | Games end: ~Jul 8 2026

 2. My Submissions
 - Table: submission name, current score, episode count, notes
 - Score sparkline per submission (across leaderboard snapshots)
 - In-app editable notes per submission

 3. Leaderboard (top 20 + our position)
 - Read latest CSV from leaderboards/
 - Highlight "Montana Schmeekler" / "tedtop" row in yellow
 - Score delta vs previous snapshot
 - "Prize cutoff (top 10)" marker

 4. Submission Log (historical)
 - All submissions with: version, description, score peak, notes, better/worse

 New deps for requirements.txt:
 openskill>=5.0.0
 streamlit>=1.32.0
 pandas>=2.0.0

 ---
 Verification

 1. Archive check: ls agents/ archive/agents/ — agents/ has exactly 2 files
 2. Arena 2P smoke:
 python arena.py --players coordinated_strike_interceptor,markowitz_portfolio_optimization --games 4 --mode 2p
 3. Arena 4P smoke:
 python arena.py --mode 4p --players coordinated_strike_interceptor,markowitz_portfolio_optimization,archive/agents/the_vulture.py,archive/agents/bayesian_wave_function_collapse.py --games
 8
 4. Pipeline smoke:
 python pipeline/leaderboard_snapshot.py
 ls leaderboards/
 5. Dashboard:
 streamlit run dashboard/app.py

 ---
 Files to modify

 ┌─────────────────────────────────────────────┬────────┐
 │                    File                     │ Action │
 ├─────────────────────────────────────────────┼────────┤
 │ (none — arena.py is replaced, not modified) │        │
 └─────────────────────────────────────────────┴────────┘

 Files to create/archive

 ┌──────────────────────────────────┬──────────────────────────────────────┐
 │               File               │               Purpose                │
 ├──────────────────────────────────┼──────────────────────────────────────┤
 │ archive/arena_2p_round_robin.py  │ Preserved original 2P arena (git mv) │
 ├──────────────────────────────────┼──────────────────────────────────────┤
 │ archive/__init__.py              │ Module marker                        │
 ├──────────────────────────────────┼──────────────────────────────────────┤
 │ archive/agents/__init__.py       │ Module marker                        │
 ├──────────────────────────────────┼──────────────────────────────────────┤
 │ archive/agents/<21 bots>         │ Non-competing bots (git mv)          │
 ├──────────────────────────────────┼──────────────────────────────────────┤
 │ arena.py                         │ New unified 2P+4P OpenSkill arena    │
 ├──────────────────────────────────┼──────────────────────────────────────┤
 │ pipeline/__init__.py             │ Module marker                        │
 ├──────────────────────────────────┼──────────────────────────────────────┤
 │ pipeline/pull_episodes.py        │ Discover new episodes                │
 ├──────────────────────────────────┼──────────────────────────────────────┤
 │ pipeline/download_replays.py     │ Fetch replay JSON                    │
 ├──────────────────────────────────┼──────────────────────────────────────┤
 │ pipeline/leaderboard_snapshot.py │ Download + timestamp leaderboard     │
 ├──────────────────────────────────┼──────────────────────────────────────┤
 │ pipeline/run_pipeline.sh         │ Cron-able runner                     │
 ├──────────────────────────────────┼──────────────────────────────────────┤
 │ dashboard/app.py                 │ Streamlit dashboard                  │
 ├──────────────────────────────────┼──────────────────────────────────────┤
 │ requirements.txt                 │ openskill, streamlit, pandas         │
 └──────────────────────────────────┴──────────────────────────────────────┘