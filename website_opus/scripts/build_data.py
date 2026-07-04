#!/usr/bin/env python3
"""
Build all static JSON data for the Orbit Wars portfolio website.
Run from the orbit_wars/ root:
    python3 website/scripts/build_data.py
"""
import csv
import json
import re
import shutil
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent  # orbit_wars/
OUT  = Path(__file__).resolve().parent.parent / "public" / "data"
OUT.mkdir(parents=True, exist_ok=True)

OUR_NAME = "Montana Schmeekler"

SUBMISSION_TIMES = {
    "markowitz_portfolio_optimization v1": "2026-06-14 14:07:06",
    "coordinated_strike_interceptor v1":   "2026-06-14 14:06:13",
    "comet_reaper v1":                     "2026-06-15 10:52:59",
    "schmeekler@1.5":                      "2026-06-17 08:53:46",
    "schmeekler_fmt":                      "2026-06-17 18:55:51",
    "comet_reaper_1235":                   "2026-06-20 06:28:59",
}


# ── 1. Score history ──────────────────────────────────────────────────────────

def build_score_history():
    lb_dir = ROOT / "leaderboards"

    # Load submission timestamps to tag each snapshot
    sub_times = sorted(
        [(datetime.strptime(v, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc), k)
         for k, v in SUBMISSION_TIMES.items()]
    )

    def active_bot(ts):
        bot = "markowitz_portfolio_optimization v1"
        for sub_ts, name in sub_times:
            if ts >= sub_ts:
                bot = name
        return bot

    records = []
    for p in sorted(lb_dir.glob("leaderboard_*.csv")):
        try:
            ts = datetime.strptime(p.stem.replace("leaderboard_", ""), "%Y-%m-%d_%H-%M")
            ts = ts.replace(tzinfo=timezone.utc)
        except ValueError:
            continue
        try:
            with open(p, encoding="utf-8-sig") as f:
                rows = list(csv.DictReader(f))
            our = [r for r in rows if OUR_NAME.lower() in r.get("TeamName","").lower()]
            if not our:
                continue
            prize = next((r for r in rows if r.get("Rank") == "10"), None)
            records.append({
                "time":        ts.isoformat(),
                "rank":        int(our[0]["Rank"]),
                "score":       float(our[0]["Score"]),
                "prize_score": float(prize["Score"]) if prize else None,
                "bot_name":    active_bot(ts),
            })
        except Exception as e:
            print(f"  skip {p.name}: {e}")

    records.sort(key=lambda r: r["time"])
    out = OUT / "score_history.json"
    out.write_text(json.dumps(records, indent=2))
    print(f"✓ score_history.json — {len(records)} points")


# ── 2. Timeline ───────────────────────────────────────────────────────────────

PHASE_STATUS = {
    "Phase 0":  "breakthrough",
    "Phase 1":  "infrastructure",
    "Phase 2":  "submitted",
    "Phase 3":  "breakthrough",
    "Phase 4":  "dead_end",
    "Phase 5":  "dead_end",
    "Phase 6":  "breakthrough",
    "v5":       "closed",
    "v6":       "closed",
    "Plan B":   "ongoing",
}

def build_timeline():
    tl = ROOT / "TIMELINE.md"
    text = tl.read_text()

    phases = []
    # Split on ## headings
    chunks = re.split(r'\n## ', text)
    for chunk in chunks[1:]:  # skip preamble
        lines = chunk.strip().splitlines()
        heading = lines[0].strip()
        body = "\n".join(lines[1:]).strip()

        # Extract date from heading if present (e.g. "2026-06-17 — Track A …")
        date_m = re.search(r'(\d{4}-\d{2}-\d{2})', heading)
        date = date_m.group(1) if date_m else ""

        # Determine phase label
        phase_m = re.match(r'(Phase \d+|v\d+|Plan B|2026-\d{2}-\d{2}.*)', heading)
        if phase_m:
            phase_label = phase_m.group(1)
        else:
            phase_label = heading[:30]

        # Status heuristic
        status = "infrastructure"
        for key, val in PHASE_STATUS.items():
            if key in heading:
                status = val
                break
        if "DEAD END" in body or "dead end" in body.lower() or "DISCARD" in body:
            status = "dead_end"
        if "submitted" in body.lower() or "Submitted" in body:
            if status not in ("dead_end",):
                status = "submitted"
        if "breakthrough" in body.lower() or "CHAMPION" in body or "BEATS" in body:
            status = "breakthrough"
        if "CLOSED" in heading or "FAIL" in heading:
            status = "closed"

        # First sentence of body as headline
        first_para = body.split("\n\n")[0] if body else ""
        first_sent = re.split(r'[.!?]', first_para)[0].strip()
        headline = re.sub(r'\*\*|__|\[|\]|\(.*?\)', '', first_sent)[:120]

        phases.append({
            "phase":    phase_label,
            "date":     date,
            "title":    heading,
            "headline": headline,
            "body_md":  body,
            "status":   status,
        })

    out = OUT / "timeline.json"
    out.write_text(json.dumps(phases, indent=2, ensure_ascii=False))
    print(f"✓ timeline.json — {len(phases)} phases")


# ── 3. Experiments ────────────────────────────────────────────────────────────

EXPERIMENTS = [
    dict(epoch=1,  name="precog",               base="comet_reaper", track="A",       verdict="DISCARD",     win_pct=50,   delta=-16, category="Heuristic",       note="Parity vs CR seat-swapped; N≈50"),
    dict(epoch=2,  name="kingmaker",             base="comet_reaper", track="A",       verdict="DISCARD",     win_pct=50,   delta=-16, category="Heuristic",       note="Parity vs CR seat-swapped"),
    dict(epoch=3,  name="maestro",               base="comet_reaper", track="A",       verdict="DISCARD",     win_pct=50,   delta=-16, category="Heuristic",       note="Parity vs CR seat-swapped"),
    dict(epoch=4,  name="helmsman",              base="comet_reaper", track="A",       verdict="DISCARD",     win_pct=50,   delta=-16, category="Heuristic",       note="Parity vs CR seat-swapped"),
    dict(epoch=5,  name="oracle",                base="comet_reaper", track="A",       verdict="DISCARD",     win_pct=50,   delta=-16, category="Heuristic",       note="Parity vs CR seat-swapped"),
    dict(epoch=6,  name="comet_reaper_tuned",    base="comet_reaper", track="Config",  verdict="DISCARD",     win_pct=None, delta=None, category="Config tuning",  note="37 Optuna trials; best 0.34 — base config is a tight optimum"),
    dict(epoch=7,  name="schmeekler",            base="comet_reaper", track="A",       verdict="KEEP",        win_pct=74,   delta=+8,  category="Static bonus",    note="72% 2P vs CR; beats whole public panel; CHAMPION → submitted live"),
    dict(epoch=8,  name="schmeekler_potential",  base="schmeekler",   track="A",       verdict="DISCARD",     win_pct=74,   delta=0,   category="Scoring bonus",   note="≈0pp — flow scorer already encodes ETA/position"),
    dict(epoch=9,  name="schmeekler_interdict",  base="schmeekler",   track="A",       verdict="DISCARD",     win_pct=48,   delta=-18, category="Scoring bonus",   note="−26pp CATASTROPHIC — overrides flow scorer"),
    dict(epoch=10, name="schmeekler_phase",      base="schmeekler",   track="A",       verdict="DISCARD",     win_pct=22,   delta=-44, category="Phase sizing",    note="−52pp CATASTROPHIC — breaks ROI/floor/sizing interaction"),
    dict(epoch=11, name="schmeekler_fmt",        base="schmeekler",   track="A",       verdict="KEEP",        win_pct=66,   delta=0,   category="Format-aware",    note="2P identical; +3.72μ 4P; SUBMITTED live"),
    dict(epoch=12, name="comet_reaper_search",   base="comet_reaper", track="B",       verdict="DISCARD",     win_pct=50,   delta=-16, category="Search",          note="Rollout reproduces 1-ply moves exactly"),
    dict(epoch=13, name="CR_mcts_v1",            base="comet_reaper", track="B",       verdict="DISCARD",     win_pct=75,   delta=+1,  category="Search",          note="n=50 parity — 0–4 candidates/turn, de-mean correction = 0"),
    dict(epoch=14, name="CR_mcts_v2",            base="comet_reaper", track="B",       verdict="DISCARD",     win_pct=75,   delta=+1,  category="Search",          note="True depth-2 + state advance; candidate scarcity confirmed"),
    dict(epoch=15, name="schmeekler_orbit",      base="schmeekler",   track="B",       verdict="DISCARD",     win_pct=61,   delta=-5,  category="Orbit timing",    note="Hold fires 0% or 42% turns causing passivity; floor independent of orbit position"),
    dict(epoch=16, name="schmeekler_comet",      base="schmeekler",   track="A",       verdict="DISCARD",     win_pct=74,   delta=0,   category="Comet targeting", note="2×2 factorial — comet bonus +0pp; flow scorer handles ephemeral value implicitly"),
    dict(epoch=17, name="CR_stochastic",         base="comet_reaper", track="B",       verdict="DISCARD",     win_pct=61,   delta=-5,  category="Stochastic search", note="Boltzmann 2-ply 61% vs schmeekler 78%; base strength gap not overcome by EV"),
    dict(epoch=18, name="comet_reaper_vf",       base="comet_reaper", track="C",       verdict="DISCARD",     win_pct=50,   delta=-16, category="Value function",  note="AUC=0.9835 ✅ — 4P HURT; bolt-on VF wrong integration: needs policy-level not post-plan"),
    dict(epoch=19, name="schmeekler_elim",       base="schmeekler",   track="A",       verdict="DISCARD",     win_pct=65,   delta=-1,  category="Elim bonus",      note="65% vs schmeekler 78%; elim bonus adds edge vs 1266-elo but destroys medium-game"),
]

def build_experiments():
    out = OUT / "experiments.json"
    out.write_text(json.dumps(EXPERIMENTS, indent=2))
    print(f"✓ experiments.json — {len(EXPERIMENTS)} experiments")


# ── 4. RL training runs ───────────────────────────────────────────────────────

RL_LOG_PAT = re.compile(
    r"U\s*(\d+)\s*\|\s*S:\s*([\d,]+)\s*\|\s*L:([-\d.]+)\s*\|\s*CF:([\d.]+)"
    r"(?:\s*\|\s*EV:([-\d.]+))?(?:\s*\|\s*Ent:([-\d.]+))?"
    r"\s*\|\s*EP:(\d+)\s*\|\s*SPS:([\d.]+)"
)
EVAL_PAT = re.compile(r"eval.*?U=(\d+).*?WR=([\d.]+)", re.IGNORECASE)

def build_rl_runs():
    runs_dir = ROOT / "agents" / "rl_ppo" / "runs"
    if not runs_dir.exists():
        print("  rl runs dir not found — skipping")
        return

    all_runs = {}
    for log_path in sorted(runs_dir.rglob("*.log")):
        run_name = log_path.stem
        rows = []
        evals = {}
        try:
            for line in log_path.read_text(errors="replace").splitlines():
                m = RL_LOG_PAT.search(line)
                if m:
                    rows.append({
                        "run":      run_name,
                        "update":   int(m.group(1)),
                        "steps":    int(m.group(2).replace(",", "")),
                        "loss":     float(m.group(3)),
                        "clip_frac": float(m.group(4)),
                        "explained_variance": float(m.group(5)) if m.group(5) else None,
                        "entropy":  float(m.group(6)) if m.group(6) else None,
                        "sps":      float(m.group(8)),
                    })
                em = EVAL_PAT.search(line)
                if em:
                    evals[int(em.group(1))] = float(em.group(2))
        except Exception:
            continue
        if rows:
            for row in rows:
                row["eval_wr"] = evals.get(row["update"])
            all_runs[run_name] = rows

    out = OUT / "rl_runs.json"
    out.write_text(json.dumps(all_runs, indent=2))
    total = sum(len(v) for v in all_runs.values())
    print(f"✓ rl_runs.json — {len(all_runs)} runs, {total} data points")


# ── 5. Replays index + copy ───────────────────────────────────────────────────

def build_replays():
    replays_dir = ROOT / "replays"
    out_dir = OUT / "replays"
    out_dir.mkdir(exist_ok=True)

    index = []
    db_path = ROOT / "strategy" / "tracking.db"
    episode_meta = {}
    if db_path.exists():
        con = sqlite3.connect(db_path)
        for row in con.execute("SELECT episode_id, create_time, our_placement, num_players FROM episodes"):
            episode_meta[str(row[0])] = {
                "create_time":   row[1],
                "our_placement": row[2],
                "num_players":   row[3],
            }
        con.close()

    all_jsons = sorted(replays_dir.rglob("*.json"))

    # Curate: pick wins against good opponents + losses + longest games
    winners = []
    losers  = []
    for p in all_jsons:
        ep_id = p.stem
        meta  = episode_meta.get(ep_id, {})
        placement = meta.get("our_placement")
        try:
            d = json.loads(p.read_text())
            teams = d["info"].get("TeamNames", [])
            n_steps = len(d.get("steps", []))
            raw_rewards = d.get("rewards", [])
            # rewards is either a flat list [r0,r1] or a list-of-lists [[r0,r1],...]
            if raw_rewards and isinstance(raw_rewards[0], list):
                final_rewards = raw_rewards[-1]
            else:
                final_rewards = raw_rewards
            our_idx = next((i for i,t in enumerate(teams) if OUR_NAME.lower() in t.lower()), None)
            if our_idx is not None:
                our_reward = final_rewards[our_idx] if our_idx < len(final_rewards) else None
                result = "win" if placement == 1 else ("loss" if placement and placement > 1 else "unknown")
            else:
                result = "unknown"
                our_reward = None

            record = {
                "id":         ep_id,
                "teams":      teams,
                "our_result": result,
                "steps":      n_steps,
                "date":       meta.get("create_time", p.parent.name)[:10],
                "placement":  placement,
                "num_players": meta.get("num_players", 2),
            }
            index.append(record)
            if result == "win" and n_steps > 80:
                winners.append((n_steps, p, record))
            elif result == "loss":
                losers.append((n_steps, p, record))
        except Exception as e:
            print(f"  skip replay {p.name}: {e}")

    # Copy top 25 wins (longest) + top 25 losses (longest) for the gallery
    to_copy = set()
    for _, p, _ in sorted(winners, reverse=True)[:25]:
        to_copy.add(p)
    for _, p, _ in sorted(losers, reverse=True)[:25]:
        to_copy.add(p)

    copied_ids = set()
    for p in to_copy:
        shutil.copy2(p, out_dir / p.name)
        copied_ids.add(p.stem)

    for record in index:
        record["available"] = record["id"] in copied_ids

    out_idx = OUT / "replays_index.json"
    out_idx.write_text(json.dumps(index, indent=2))
    print(f"✓ replays_index.json — {len(index)} replays indexed, {len(to_copy)} copied")


# ── 6. 23 Scientists ─────────────────────────────────────────────────────────

SCIENTISTS = [
    # Physics / control
    dict(name="artificial_potential_fields",      field="Control Theory · APF",         cluster="physics",   description="Treats planets as attractive/repulsive potentials; fleets flow along gradient descent."),
    dict(name="distributed_pid_controllers",      field="Control Theory · PID",         cluster="physics",   description="Each fleet controlled by an independent PID loop targeting planet delta-ships."),
    dict(name="lyapunov_defense_heuristic",       field="Dynamical Systems · Lyapunov", cluster="physics",   description="Certifies stability of a fleet allocation via Lyapunov energy functions."),
    dict(name="kinematic_wave_theory",            field="Physics · Kinematic Waves",    cluster="physics",   description="Models fleet traffic as kinematic shock waves propagating through the galaxy."),
    dict(name="reaction_diffusion_turing_patterns", field="Physics · Turing Patterns",  cluster="physics",   description="Fleet allocation evolves via reaction-diffusion equations producing stable spatial patterns."),
    # Life sciences
    dict(name="susceptible_infected_recovered_model", field="Epidemiology · SIR",       cluster="life",      description="Planet ownership spreads like infection: susceptible neutrals, infected friendlies, recovered captured."),
    dict(name="stigmergic_pheromone_routing",     field="Biology · Stigmergy",          cluster="life",      description="Fleets deposit virtual pheromones on successful paths; others follow the scent gradient."),
    # Math / CS
    dict(name="minimax_fleet_allocation",         field="Game Theory · Minimax",        cluster="math_cs",   description="Two-ply minimax over fleet allocation decisions, pruned by alpha-beta."),
    dict(name="bayesian_wave_function_collapse",  field="Probability · Bayesian WFC",   cluster="math_cs",   description="Treats uncertain game states as a quantum wave function; collapses to best allocation via Bayes."),
    dict(name="graph_neural_network_value_estimator", field="ML · GNN",                cluster="math_cs",   description="Planet graph encoded by a GNN; value head scores each possible fleet dispatch."),
    dict(name="deep_q_network_macro_strategist",  field="ML · DQN",                    cluster="math_cs",   description="Deep Q-Network trained on synthetic episodes to select macro strategic moves."),
    dict(name="lstm_fleet_trajectory_forecaster", field="ML · LSTM",                   cluster="math_cs",   description="LSTM predicts opponent fleet trajectories; pre-empts captures with counter-launches."),
    dict(name="target_classifier_fnn",            field="ML · FNN Classifier",         cluster="math_cs",   description="Feedforward net classifies each planet as high/low priority target; ships flow to high."),
    dict(name="cascading_classifier_regressor",   field="ML · Cascade",                cluster="math_cs",   description="Two-stage model: classifier filters candidates, regressor sizes the fleet dispatch."),
    dict(name="predictive_kinematic_interceptor", field="Physics · Kinematics",        cluster="physics",   description="Predicts orbital intercept points analytically; launches to arrive exactly when planet is capturable."),
    dict(name="path_aware_lead_interceptor",      field="Pathfinding · Lead",          cluster="math_cs",   description="Computes lead-angle intercept path around the sun for orbiting targets."),
    dict(name="greedy_lead_interceptor",          field="Heuristic · Greedy",          cluster="math_cs",   description="Greedy nearest-available fleet always dispatched to highest-value intercept target."),
    # Economics
    dict(name="macroeconomic_gravity_model",      field="Economics · Gravity Model",   cluster="economics", description="Trade-gravity model: fleet flows proportional to planet GDP and inversely to orbital distance."),
    dict(name="frontline_consolidation",          field="Military · Consolidation",    cluster="other",     description="Identifies contested frontier planets; concentrates force at the weakest point in the line."),
    dict(name="comet_riding_ephemeris_exploitation", field="Astronomy · Ephemeris",    cluster="physics",   description="Exploits comet orbital ephemerides to predict lucrative short-window capture opportunities."),
    dict(name="the_vulture",                      field="Heuristic · Opportunistic",   cluster="other",     description="Waits for two opponents to battle; swoops in to capture depleted planets."),
    # The two submitted bots (lived in agents/, not archive/)
    dict(name="markowitz_portfolio_optimization", field="Economics · Markowitz MPT",   cluster="economics", description="Applies Markowitz mean-variance portfolio theory: allocates ships to maximize expected production return per unit risk. 🏆 Won the arena."),
    dict(name="coordinated_strike_interceptor",   field="Heuristic · Coordination",    cluster="other",     description="Coordinates multiple fleets to converge simultaneously on high-value targets."),
]

def build_scientists():
    out = OUT / "scientists.json"
    out.write_text(json.dumps(SCIENTISTS, indent=2, ensure_ascii=False))
    print(f"✓ scientists.json — {len(SCIENTISTS)} bots")


# ── 7. Submissions ────────────────────────────────────────────────────────────

def build_submissions():
    db_path = ROOT / "strategy" / "tracking.db"
    if not db_path.exists():
        return
    con = sqlite3.connect(db_path)
    rows = con.execute("SELECT * FROM submissions ORDER BY submitted_at").fetchall()
    cols = [d[1] for d in con.execute("PRAGMA table_info(submissions)").fetchall()]
    con.close()
    subs = [dict(zip(cols, r)) for r in rows]
    out = OUT / "submissions.json"
    out.write_text(json.dumps(subs, indent=2))
    print(f"✓ submissions.json — {len(subs)} submissions")


# ── 8. Screenshots index ──────────────────────────────────────────────────────

def build_screenshots():
    src = ROOT / "Progress Screenshots"
    out_dir = Path(__file__).resolve().parent.parent / "public" / "screenshots"
    out_dir.mkdir(parents=True, exist_ok=True)

    index = []
    if src.exists():
        for p in sorted(src.iterdir()):
            if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".gif", ".webp"):
                shutil.copy2(p, out_dir / p.name)
                index.append({"filename": p.name, "alt": p.stem.replace("_", " ")})

    out = OUT / "screenshots_index.json"
    out.write_text(json.dumps(index, indent=2))
    print(f"✓ screenshots_index.json — {len(index)} screenshots")


if __name__ == "__main__":
    print("Building Orbit Wars website data...")
    build_score_history()
    build_timeline()
    build_experiments()
    build_rl_runs()
    build_replays()
    build_scientists()
    build_submissions()
    build_screenshots()
    print("\nDone. All data in website/public/data/")
