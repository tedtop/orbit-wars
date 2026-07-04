#!/usr/bin/env python3
"""Build static data assets for the Orbit Wars site.

Reads from the repo's raw data (leaderboards/, website/public/data/) and writes:
  src/data/*.json          — small JSONs imported at build time
  public/data/replays/*    — compacted game replays fetched client-side
  public/data/rl_runs.json — curated RL training curves

Run from website_fable/:  python3 scripts/build_data.py
"""
import csv
import glob
import json
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
SITE = os.path.dirname(HERE)
REPO = os.path.dirname(SITE)
OLD_DATA = os.path.join(REPO, "website_opus", "public", "data")
LB_DIR = os.path.join(REPO, "leaderboards")
SRC_DATA = os.path.join(SITE, "src", "data")
PUB_DATA = os.path.join(SITE, "public", "data")

os.makedirs(SRC_DATA, exist_ok=True)
os.makedirs(os.path.join(PUB_DATA, "replays"), exist_ok=True)

OUR_TEAM = "Montana Schmeekler"


def write(path, obj, compact=False):
    with open(path, "w") as f:
        if compact:
            json.dump(obj, f, separators=(",", ":"))
        else:
            json.dump(obj, f, indent=1)
    print(f"wrote {os.path.relpath(path, SITE)} ({os.path.getsize(path)//1024} KB)")


# ---------------------------------------------------------------- leaderboard race
def build_race():
    """Per snapshot: leader score, prize line (rank 10), our best score + rank, field size."""
    rows = []
    pat = re.compile(r"leaderboard_(\d{4}-\d{2}-\d{2})_(\d{2})-(\d{2})\.csv$")
    for path in sorted(glob.glob(os.path.join(LB_DIR, "leaderboard_*.csv"))):
        m = pat.search(path)
        if not m:
            continue
        ts = f"{m.group(1)}T{m.group(2)}:{m.group(3)}:00Z"
        top1 = rank10 = our_score = None
        our_rank = None
        n = 0
        with open(path, newline="", encoding="utf-8-sig") as f:
            for r in csv.DictReader(f):
                n += 1
                rank = int(r["Rank"])
                score = float(r["Score"])
                if rank == 1:
                    top1 = score
                elif rank == 10:
                    rank10 = score
                if r["TeamName"] == OUR_TEAM:
                    our_score, our_rank = score, rank
        rows.append({
            "t": ts,
            "top1": top1,
            "prize": rank10,
            "us": our_score,
            "rank": our_rank,
            "teams": n,
        })
    write(os.path.join(SRC_DATA, "race.json"), rows, compact=True)
    return rows


# ---------------------------------------------------------------- final leaderboard
def build_final_lb():
    latest = sorted(glob.glob(os.path.join(LB_DIR, "leaderboard_*.csv")))[-1]
    teams = []
    with open(latest, newline="", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            teams.append({
                "rank": int(r["Rank"]),
                "name": r["TeamName"],
                "score": float(r["Score"]),
                "subs": int(r["SubmissionCount"]),
            })
    ours = next(t for t in teams if t["name"] == OUR_TEAM)
    out = {
        "snapshot": os.path.basename(latest),
        "n_teams": len(teams),
        "top": teams[:10],
        "us": ours,
        "around_us": [t for t in teams if abs(t["rank"] - ours["rank"]) <= 3],
    }
    write(os.path.join(SRC_DATA, "final_leaderboard.json"), out)


# ---------------------------------------------------------------- straight copies
def copy_small():
    for name in ["experiments", "scientists", "submissions", "timeline"]:
        with open(os.path.join(OLD_DATA, f"{name}.json")) as f:
            write(os.path.join(SRC_DATA, f"{name}.json"), json.load(f), compact=True)


# ---------------------------------------------------------------- RL runs (curated)
RL_KEEP = {
    # v6 per-planet PPO, CPU fleet — the long self-play runs
    "v6_greedy": "v6 · greedy cold-start",
    "v6_cr4": "v6 · diverse cold-start",
    # v9 entity-transformer PPO quad — entropy collapse exhibit
    "m3quad-v2_job1": "v9 · ET-PPO seed 1",
    "m3quad-v2_job2": "v9 · ET-PPO seed 2",
    "m3quad-v2_job3": "v9 · ET-PPO seed 3",
    "m3quad-v2_job4": "v9 · ET-PPO seed 4",
    # champion seed (only monotonic improver, peaked 37% vs greedy)
    "orbit-wars-ppo-1-of-3_job1": "v6 · champion seed j1",
}


def build_rl():
    with open(os.path.join(OLD_DATA, "rl_runs.json")) as f:
        runs = json.load(f)
    out = {}
    for key, label in RL_KEEP.items():
        pts = [
            {
                "u": p["update"],
                "ent": p.get("entropy"),
                "ev": p.get("explained_variance"),
                "cf": p.get("clip_frac"),
                "sps": p.get("sps"),
            }
            for p in runs[key]
        ]
        out[key] = {"label": label, "points": pts}
    write(os.path.join(PUB_DATA, "rl_runs.json"), out, compact=True)


# ---------------------------------------------------------------- replay compaction
# Curated mix: 2P wins/losses, 4P wins/losses, incl. the Mendrika game.
REPLAYS = [
    ("79984357", "2P · win vs shishaohua"),
    ("80518143", "2P · win vs Antonoof"),
    ("80852381", "2P · loss vs galaxy2025"),
    ("80032464", "4P · 1st of 4"),
    ("81027412", "4P · 2nd — pod with Mendrika (top-50 BC bot)"),
    ("81035800", "4P · 4th — a crushing"),
]


def r1(x):
    return round(x, 1)


def compact_replay(rid, label, index_meta):
    with open(os.path.join(OLD_DATA, "replays", f"{rid}.json")) as f:
        rep = json.load(f)
    info = rep.get("info", {})
    team_names = info.get("TeamNames", [])
    steps_out = []
    for step in rep["steps"]:
        obs = step[0]["observation"]
        # planets: [id, owner, x, y, radius, ships, production]
        planets = [[p[0], p[1], r1(p[2]), r1(p[3]), r1(p[4]), int(p[5]), p[6]] for p in obs["planets"]]
        # fleets: [id, owner, x, y, heading, ships, speed] -> drop speed, round heading
        fleets = [[fl[1], r1(fl[2]), r1(fl[3]), round(fl[4], 2), int(fl[5])] for fl in obs["fleets"]]
        steps_out.append({"p": planets, "f": fleets, "c": obs.get("comet_planet_ids", [])})
    out = {
        "id": rid,
        "label": label,
        "teams": team_names,
        "our_seat": team_names.index(OUR_TEAM) if OUR_TEAM in team_names else 0,
        "rewards": rep.get("rewards"),
        "result": index_meta.get("our_result"),
        "placement": index_meta.get("placement"),
        "date": index_meta.get("date"),
        "n_steps": len(steps_out),
        "steps": steps_out,
    }
    write(os.path.join(PUB_DATA, "replays", f"{rid}.json"), out, compact=True)
    return {
        "id": rid,
        "label": label,
        "teams": team_names,
        "our_seat": out["our_seat"],
        "result": out["result"],
        "placement": out["placement"],
        "date": out["date"],
        "n_steps": out["n_steps"],
        "players": len(team_names),
    }


def build_replays():
    with open(os.path.join(OLD_DATA, "replays_index.json")) as f:
        index = {r["id"]: r for r in json.load(f)}
    manifest = [compact_replay(rid, label, index.get(rid, {})) for rid, label in REPLAYS]
    write(os.path.join(SRC_DATA, "replays_manifest.json"), manifest)


if __name__ == "__main__":
    copy_small()
    build_race()
    build_final_lb()
    build_rl()
    build_replays()
    print("done")
