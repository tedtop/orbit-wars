Orbit Wars — First Session Implementation Plan

 Context

 Ted has a Python-based Orbit Wars competition setup with 23 bots, two of which
 are live on Kaggle (markowitz_portfolio_optimization and
 coordinated_strike_interceptor). The codebase needs restructuring and new
 infrastructure: archive non-competing bots, extract a shared physics core,
 update the submission bundler, build a 4-player arena, build the data pipeline,
 and build a monitoring dashboard — all with 9 days until the competition
 submission deadline.

 The current state:
 - All 23 bots are in agents/ (should be 2 active + archive)
 - Both active bots share ~130 lines of verbatim physics helpers (no shared module)
 - submit.py just copies bot → main.py; needs tar.gz bundling once core/ exists
 - arena.py (root) handles 2P round-robins; no 4P placement-tracking harness
 - pipeline/, dashboard/, core/, archive/ directories do not yet exist

 ---
 Task 1 — Archive 21 non-active bots

 Create archive/agents/ and move all bots except the two active ones.

 archive/
   agents/
     artificial_potential_fields.py
     bayesian_wave_function_collapse.py   # 4P filler (64%)
     cascading_classifier_regressor.py
     comet_riding_ephemeris_exploitation.py
     deep_q_network_macro_strategist.py
     distributed_pid_controllers.py
     frontline_consolidation.py
     graph_neural_network_value_estimator.py
     greedy_lead_interceptor.py
     kinematic_wave_theory.py
     lstm_fleet_trajectory_forecaster.py
     lyapunov_defense_heuristic.py
     macroeconomic_gravity_model.py
     minimax_fleet_allocation.py
     path_aware_lead_interceptor.py
     predictive_kinematic_interceptor.py
     reaction_diffusion_turing_patterns.py
     stigmergic_pheromone_routing.py      # 4P filler (63%)
     susceptible_infected_recovered_model.py
     target_classifier_fnn.py
     the_vulture.py                       # 4P filler (60%)

 Keep in agents/: markowitz_portfolio_optimization.py,
 coordinated_strike_interceptor.py

 Add archive/__init__.py and archive/agents/__init__.py so they are
 importable as Python modules (for the 4P arena to use as filler bots).

 Update .gitignore: add data/, strategy/, .kaggle/ (already have
 strategy and leaderboards and replays; consolidate under data/).

 ---
 Task 2 — Extract shared physics core to core/physics.py

 Both active bots have identical copies of these at their tops:

 Constants (lines 16-20 in coordinated_strike):
 _BOARD = 100.0; _CX = _CY = 50.0; _SUN_R = 10.0; _ROT_LIM = 50.0; _EPISODE = 500

 Functions (coordinated_strike lines 26-132):
 - _get(obj, key, default=None)
 - fleet_speed(ships, max_spd=6.0)
 - _pt_seg(p, a, b)
 - segment_hits_sun(p0, p1, margin=1.0)
 - path_blocked_by_planet(src, aim, blockers, exclude_ids, buffer=0.5)
 - predict_planet_pos(init_x, init_y, radius, angular_velocity, abs_step, rotation_sign=1)
 - predict_comet_pos(pid, comets, step_ahead)
 - lead_solution(src_pos, tgt_pid, init_x, init_y, tgt_radius, is_comet, comets, angular_velocity,
 current_step, ships, max_spd, rotation_sign)

 Create:
 core/
   __init__.py  (empty)
   physics.py   (all constants + all 8 functions, verbatim from coordinated_strike)

 Update both active bots:
 - Remove the duplicate constants and function block
 - Add at the top: from core.physics import (_get, fleet_speed, _pt_seg, segment_hits_sun,
 path_blocked_by_planet, predict_planet_pos, predict_comet_pos, lead_solution)
 - Keep bot-local constants that differ (e.g., _EPISODE = 500 stays in coordinated_strike if it
 uses it separately)

 Verify: run smoke tests locally before proceeding to Task 3.

 ---
 Task 3 — Update submit.py for tar.gz bundling

 When a bot imports from core/, Kaggle needs a tarball. Update kaggle_submit():

 import tarfile

 def kaggle_submit(bot_path: Path, message: str) -> bool:
     core_dir = REPO_ROOT / "core"
     needs_bundle = core_dir.exists()

     with tempfile.TemporaryDirectory() as tmpdir:
         tmpdir = Path(tmpdir)
         shutil.copy2(bot_path, tmpdir / "main.py")

         if needs_bundle:
             shutil.copytree(core_dir, tmpdir / "core")
             tarball = tmpdir / "submission.tar.gz"
             with tarfile.open(tarball, "w:gz") as tar:
                 tar.add(tmpdir / "main.py", arcname="main.py")
                 tar.add(tmpdir / "core", arcname="core")
             submit_file = tarball
         else:
             submit_file = tmpdir / "main.py"

         cmd = ["kaggle", "competitions", "submit", COMPETITION,
                "-f", str(submit_file), "-m", message]
         result = subprocess.run(cmd, capture_output=False)
         return result.returncode == 0

 Also add import tarfile at top.

 ---
 Task 4 — Download The Producer V2 + orbit_lite

 Run these commands during implementation:
 mkdir -p agents/opponents/the_producer_v2
 kaggle kernels pull slawekbiel/the-producer-v2 -p agents/opponents/the_producer_v2
 kaggle datasets download slawekbiel/producer-orbit-wars-utils -p agents/opponents/the_producer_v2
 --unzip

 Study orbit_lite for physics improvements but do NOT modify core/physics.py
 based on it without running arena tests first.

 ---
 Task 5 — Build arena/arena_4p.py

 Create the directory:
 arena/
   __init__.py      (empty)
   arena_4p.py      (new 4P tournament harness)
   run_arena.py     (CLI dispatcher)

 Key design decisions for arena_4p.py:

 Seat rotation: Over every 4 games using the same 4 bots, cycle seats:
 game 0:  [A, B, C, D]
 game 1:  [B, C, D, A]
 game 2:  [C, D, A, B]
 game 3:  [D, A, B, C]
 This ensures each bot sees each starting position equally.

 Placement extraction: The engine rewards for 4P may not distinguish 2nd from
 4th (non-winners all get 0). Use the final observation's ship count to rank
 non-winners: sort players descending by reward first, then by total ships
 (planets + in-flight fleets).

 def _placements(env):
     """Return list of (original_slot, placement_1_to_4) sorted by slot."""
     last = env.steps[-1]
     rewards = [s.reward or 0.0 for s in last]
     # Use negative total ships as tiebreaker for non-winners
     ships = [_count_ships(last[i].observation) for i in range(len(rewards))]
     order = sorted(range(len(rewards)), key=lambda i: (-rewards[i], -ships[i]))
     placements = [0] * len(rewards)
     for rank, slot in enumerate(order):
         placements[slot] = rank + 1   # 1 = winner
     return placements

 Tournament structure: For N bots in a 4P tournament, enumerate all
 C(N, 4) quad combinations. For each quad, play until average placement score
 has converged (or hit max_games). This is the same adaptive stopping pattern as
 arena.py but applied to a mean placement score per bot.

 Scoring: Track placement_counts[bot][place] for place in 1..4. Compute:
 PLACEMENT_SCORE = {1: 3, 2: 2, 3: 1, 4: 0}  # simple Borda-count style
 avg_score = sum(PLACEMENT_SCORE[p] * count for p, count in counts.items()) / total_games

 Worker function: Same pattern as root arena.py _play_task — must be
 silent (no printing) and return None on engine error.

 Reuse from root arena.py: Import play_one, wilson_ci, C, out,
 rewrite, fmt_dur, bar from root arena.py (or duplicate the tiny ones).

 CLI arena/run_arena.py:
 python arena/run_arena.py --mode 4p --bots
 markowitz,coordinated_strike,the_vulture,bayesian_wave_function_collapse
 python arena/run_arena.py --mode 4p --games 50

 ---
 Task 6 — Build pipeline/

 pipeline/
   __init__.py
   pull_episodes.py
   download_replays.py
   leaderboard_snapshot.py
   run_pipeline.sh

 pull_episodes.py:
 - Reads submission IDs from config (hardcoded dict or JSON)
 - For each ID, runs kaggle competitions episodes <ID> → parse output
 - Compares against data/tracking.db (SQLite, creates if missing) to find new episodes
 - Prints new episode IDs; writes to DB

 download_replays.py:
 - Takes list of episode IDs (from pull_episodes output or --all-new flag)
 - For each: kaggle competitions replay <EPISODE_ID> -p data/replays/YYYY-MM-DD/
 - Prioritises: losses before wins, games vs high-rated opponents first
 - Tracks downloaded episodes in data/tracking.db

 leaderboard_snapshot.py:
 - Runs kaggle competitions leaderboard orbit-wars --download -p data/leaderboards/
 - Renames the file to leaderboard_YYYY-MM-DD_HH-MM.csv (timestamped)
 - Logs run timestamp to data/tracking.db

 SQLite schema (data/tracking.db):
 CREATE TABLE episodes (
     episode_id TEXT PRIMARY KEY,
     submission_id TEXT,
     downloaded_at TEXT,
     replay_path TEXT,
     our_placement INTEGER,
     opponent_score REAL
 );
 CREATE TABLE leaderboard_snapshots (
     filename TEXT PRIMARY KEY,
     taken_at TEXT
 );

 run_pipeline.sh:
 #!/usr/bin/env bash
 set -e
 .venv/bin/python pipeline/pull_episodes.py
 .venv/bin/python pipeline/download_replays.py --all-new
 .venv/bin/python pipeline/leaderboard_snapshot.py

 ---
 Task 7 — Build dashboard/

 Stack: FastAPI + vanilla HTML/JS (no build step; persistent service).

 dashboard/
   app.py          (FastAPI app, serves API + static)
   static/
     style.css
   templates/
     index.html    (single-page, vanilla JS polls API endpoints)

 API endpoints:
 GET /api/countdown          → {entry_deadline, sub_deadline, games_end, now_utc}
 GET /api/submissions        → [{id, name, score_history, episodes, win_pct, notes}]
 GET /api/leaderboard        → {top20: [...], our_position: {...}, last_updated}
 GET /api/submissions/{id}/notes  (POST to update notes)

 app.py data sources:
 - Countdown: hardcoded deadlines (entry Jun 16, submission Jun 23, games end ~Jul 8)
 - Submissions: reads data/tracking.db for episode history
 - Leaderboard: reads latest CSV from data/leaderboards/, highlights
 Montana Schmeekler row

 index.html panels:
 1. Timeline bar — progress bar entry→submission→end with countdown timers
 (local time + UTC)
 2. My Submissions — score history sparkline per bot, episode count, win%
 3. Leaderboard — top 20 table, our row highlighted in yellow, delta from
 last snapshot
 4. Submission Log — table with version / description / score peak / notes field

 Run:
 .venv/bin/python -m uvicorn dashboard.app:app --host 0.0.0.0 --port 8080
 (uvicorn + fastapi as new deps in requirements.txt)

 ---
 Task 8 — Update .gitignore and requirements.txt

 .gitignore additions:
 data/
 (replaces existing leaderboards and replays lines since they move under data/)

 Create requirements.txt:
 kaggle-environments>=1.28.0
 fastapi>=0.110.0
 uvicorn>=0.29.0

 ---
 Task 9 — Write README.md

 Sections:
 1. Setup (venv, deps)
 2. Running a local game
 3. Running the 2P arena (existing arena.py)
 4. Running the 4P arena (arena/run_arena.py)
 5. Submitting a bot (submit.py)
 6. Pipeline (how to run each script, cron suggestion)
 7. Dashboard (how to start, Tailscale access)

 ---
 Verification

 After all tasks:

 1. Physics extraction smoke test:
 python -c "from core.physics import fleet_speed, lead_solution; print(fleet_speed(100))"
 python -c "from kaggle_environments import make; env=make('orbit_wars');
 env.run(['agents/markowitz_portfolio_optimization.py','random']); print(env.steps[-1][0].reward)"
 python -c "from kaggle_environments import make; env=make('orbit_wars');
 env.run(['agents/coordinated_strike_interceptor.py','random']); print(env.steps[-1][0].reward)"
 2. Submit dry-run:
 python submit.py --dry-run --bot markowitz_portfolio_optimization
 2. Confirm tar.gz is created and contains main.py + core/.
 3. 4P arena quick run:
 python arena/run_arena.py --mode 4p --games 8 --bots
 markowitz,coordinated_strike,the_vulture,bayesian_wave_function_collapse
 4. Pipeline smoke:
 python pipeline/leaderboard_snapshot.py
 ls data/leaderboards/
 5. Dashboard:
 python -m uvicorn dashboard.app:app --host 0.0.0.0 --port 8080
 # open browser to http://localhost:8080

 ---
 Critical files to modify

 ┌────────────────────────────────────────────┬─────────────────────────────────────────────────┐
 │                    File                    │                     Action                      │
 ├────────────────────────────────────────────┼─────────────────────────────────────────────────┤
 │ agents/markowitz_portfolio_optimization.py │ Remove physics block, add from core.physics     │
 │                                            │ import ...                                      │
 ├────────────────────────────────────────────┼─────────────────────────────────────────────────┤
 │ agents/coordinated_strike_interceptor.py   │ Remove physics block, add from core.physics     │
 │                                            │ import ...                                      │
 ├────────────────────────────────────────────┼─────────────────────────────────────────────────┤
 │ submit.py                                  │ Add tarfile bundling logic                      │
 ├────────────────────────────────────────────┼─────────────────────────────────────────────────┤
 │ .gitignore                                 │ Add data/ line                                  │
 └────────────────────────────────────────────┴─────────────────────────────────────────────────┘

 Critical files to create

 ┌──────────────────────────────────┬──────────────────────────────────────────────────┐
 │               File               │                     Purpose                      │
 ├──────────────────────────────────┼──────────────────────────────────────────────────┤
 │ core/__init__.py                 │ Empty marker                                     │
 ├──────────────────────────────────┼──────────────────────────────────────────────────┤
 │ core/physics.py                  │ Shared physics helpers (8 functions + constants) │
 ├──────────────────────────────────┼──────────────────────────────────────────────────┤
 │ archive/__init__.py              │ Makes archive importable                         │
 ├──────────────────────────────────┼──────────────────────────────────────────────────┤
 │ archive/agents/__init__.py       │ Makes archive bots importable                    │
 ├──────────────────────────────────┼──────────────────────────────────────────────────┤
 │ arena/__init__.py                │ Module marker                                    │
 ├──────────────────────────────────┼──────────────────────────────────────────────────┤
 │ arena/arena_4p.py                │ 4P tournament with placement scoring             │
 ├──────────────────────────────────┼──────────────────────────────────────────────────┤
 │ arena/run_arena.py               │ CLI dispatcher                                   │
 ├──────────────────────────────────┼──────────────────────────────────────────────────┤
 │ pipeline/__init__.py             │ Module marker                                    │
 ├──────────────────────────────────┼──────────────────────────────────────────────────┤
 │ pipeline/pull_episodes.py        │ List new episodes per submission ID              │
 ├──────────────────────────────────┼──────────────────────────────────────────────────┤
 │ pipeline/download_replays.py     │ Fetch replay JSON                                │
 ├──────────────────────────────────┼──────────────────────────────────────────────────┤
 │ pipeline/leaderboard_snapshot.py │ Download + timestamp leaderboard                 │
 ├──────────────────────────────────┼──────────────────────────────────────────────────┤
 │ pipeline/run_pipeline.sh         │ Cron-able pipeline runner                        │
 ├──────────────────────────────────┼──────────────────────────────────────────────────┤
 │ dashboard/app.py                 │ FastAPI backend                                  │
 ├──────────────────────────────────┼──────────────────────────────────────────────────┤
 │ dashboard/templates/index.html   │ Single-page vanilla JS frontend                  │
 ├──────────────────────────────────┼──────────────────────────────────────────────────┤
 │ dashboard/static/style.css       │ Styles                                           │
 ├──────────────────────────────────┼──────────────────────────────────────────────────┤
 │ requirements.txt                 │ Declare deps                                     │
 ├──────────────────────────────────┼──────────────────────────────────────────────────┤
 │ README.md                        │ How to run everything                            │
 └──────────────────────────────────┴──────────────────────────────────────────────────┘


My response:

Update .gitignore: add data/, strategy/, .kaggle/ (already have
strategy and leaderboards and replays; consolidate under data/). -- don't do this, I like these folders at the root

I'm not sure if we need to do Task 2 at all, that was early on in the conversation, and we may not stick with my physics after we look at orbit lite, and with two bots the code repitition isn't hurting anyone

Task 3 — Update submit.py for tar.gz bundling -- ok I do agree with that, but I was kinda liking having the bots as a single file, though after they evolve we may need multi-file bots, I would say let's shelve this feature. I would like to focus on the dashboard for right now. And then getting the most competitive bots from the public leaderboard and running them locally

Task 4, sounds great, and I'm glad you can do that without my intervention. I will provide you a list of all the public code threads/bots with public scores -- please defer this item for right now, I want you to build my personal dashboard first

We'll need to get the orbit_lite and examine it. Let's do this interactively with claude chat perhaps. Right now let's focus on my personal dashboard

Task 5 -- I agree, we need a 4p arena, I think this would be a very quick thing for you to do, seems like you've already done it. Do we need to rename the other one arena_2p or just archive it? Don't forget that the real kaggle tournament can be 2p or 4p. So how do we account for that in our arena_4p? Do we have 2 arenas, or one that randomly does 2p and 4p?

CLI arena/run_arena.py:
 python arena/run_arena.py --mode 4p --bots
 markowitz,coordinated_strike,the_vulture,bayesian_wave_function_collapse
 python arena/run_arena.py --mode 4p --games 50

I don't agree so much with this naming convension, we already have arena.py which is a 2p arena, we're going to build an arena_4p. If we use "run" it should be run_tournament. but we don't really need to introduce another verb into this. I think arena is sufficient. I think we can archive the existing arena.py as archive/arena_2p_round_robin.py and create a new arena.py and that would be the whole thing, allowing it to take --mode 4p/2p and a list of bots and whatever else you think we need based on your knowledge of this project

Task 6 -- sounds great... pull_episodes, read submission ids:

(.venv)  ✘ Ted@TedBook  ~/src/orbit_wars   master  kaggle competitions submissions orbit-wars
     ref  fileName  date                        description                          status                     publicScore  privateScore
--------  --------  --------------------------  -----------------------------------  -------------------------  -----------  ------------
53676680  main.py   2026-06-14 14:07:06.697000  markowitz_portfolio_optimization v1  SubmissionStatus.COMPLETE  535.5
53676654  main.py   2026-06-14 14:06:12.947000  coordinated_strike_interceptor v1    SubmissionStatus.COMPLETE  546.1

I believe the submission ids come from running this command. I've only submitted 2 so far, but I'm guessing they will all appear here. We need to decide if we'll continuously pull episode history for all of my submissions or just the ones that are still in the running. We'll have to check that once I submit a few more submissions... I think the tracking.db should go into strategy folder, or my original thought was to save replays as json into replays/... sqlite isn't the worst idea, but if I wanted to look at json would be easier. What would be better speed wise for training models on this data, have it in sql lite or as json files. we can save as both and see how that goes... I spefifically made the replays folder for replays of games/episodes and gitignored it already

Oh I see. Episodes sounds like a db sort of situation, though I would still prefer that as human readable json, do both if you prefer. It sounds like you want to put everything in the data folder and I already have created replays/ leaderboard/ and strategy/ I would prefer you work in those folder, and if you need to create another entity, let's figure it out together


Task 7 -- excellent. Let's focus on this and the pipeline which we'll need for the data for the dashboard... do not use fastapi. Use Streamlit. Build locally first. Then I will decide if I will expose this with ngrok or Tailscale, but that's for later to decide, streamlit app first. I don't see any reason to use fastapi (unless there's a reason?)

Task 8 - Readme. I already have a readme that I don't want you to overwrite because I don't know what all you're about to do. So name it something differently and then we'll figure out how to consolidate.

Any questions?

