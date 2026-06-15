#!/usr/bin/env python3
"""Extract (observation -> action) behavior-cloning data from replay JSON.

Orbit Wars replays store each player's action per step, so extraction is exact:
for every step i and player p, `(steps[i][p].observation, steps[i][p].action)` is
a clean state->action label. (If a source ever lacks stored actions, we fall back
to reconstructing launches by diffing consecutive fleet lists.)

Every example is stamped with provenance (source file, episode id) and the acting
team's leaderboard rating, so prize-zone (>=1500) play can be filtered out for BC
even when our own weak games sit in the same corpus.

Output: gzipped JSONL to training/, plus a summary JSON.

Examples:
    # all local replays, rating-stamped
    python pipeline/extract_moves.py --src replays --out training/moves_local.jsonl.gz
    # only prize-zone play from scraped episodes
    python pipeline/extract_moves.py --src episodes --min-rating 1500 \
        --out training/moves_prizezone.jsonl.gz
"""
from __future__ import annotations

import argparse
import csv
import glob
import gzip
import json
import math
import os
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def load_ratings(leaderboard_csv: str | None) -> dict[str, float]:
    """Map lowercased TeamName AND each member username -> Score."""
    if not leaderboard_csv:
        cands = sorted(glob.glob(str(ROOT / "leaderboards" / "*.csv")))
        if not cands:
            return {}
        leaderboard_csv = cands[-1]
    ratings: dict[str, float] = {}
    with open(leaderboard_csv, encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            try:
                score = float(row["Score"])
            except (KeyError, ValueError):
                continue
            name = (row.get("TeamName") or "").strip().lower()
            if name:
                ratings[name] = score
            for u in (row.get("TeamMemberUserNames") or "").split(","):
                u = u.strip().lower()
                if u:
                    ratings.setdefault(u, score)
    return ratings


def rating_for(team: str, ratings: dict[str, float]) -> float | None:
    return ratings.get((team or "").strip().lower())


def diff_actions(prev_obs, obs, pid):
    """Fallback: recover player pid's launches as fleets that appear this step."""
    prev_ids = {f[0] for f in prev_obs.get("fleets", [])}
    moves = []
    for f in obs.get("fleets", []):
        # fleet = [id, owner, x, y, angle, ships, ...]; new fleet owned by pid = a launch
        if f[0] not in prev_ids and f[1] == pid:
            moves.append([f[6] if len(f) > 6 else -1, f[4], f[5]])  # [src?, angle, ships]
    return moves


def extract_file(path: str, ratings: dict, min_rating: float, embed_obs: bool):
    d = json.loads(Path(path).read_text())
    if not isinstance(d, dict) or "steps" not in d:   # skip non-replay JSON (e.g. _filelist cache)
        return [], []
    info = d.get("info", {})
    teams = info.get("TeamNames") or []
    episode = info.get("EpisodeId") or d.get("id")
    steps = d.get("steps") or []
    rel = os.path.relpath(path, ROOT)
    out = []
    for i, step in enumerate(steps):
        board0 = step[0].get("observation", {})
        for p, cell in enumerate(step):
            team = teams[p] if p < len(teams) else f"player_{p}"
            rt = rating_for(team, ratings)
            if min_rating > 0 and (rt is None or rt < min_rating):
                continue
            obs = cell.get("observation") or {}
            if not obs.get("planets"):
                # some formats only populate player 0's obs; board is shared (fully
                # observable) so reuse it, just re-pointing the player id.
                obs = dict(board0)
                obs["player"] = p
            action = cell.get("action")
            if action is None and i > 0:
                action = diff_actions(steps[i - 1][0].get("observation", {}), obs, p)
            action = action or []
            rec = {
                "episode": episode, "source": rel, "step": i, "player": p,
                "team": team, "rating": rt, "reward": cell.get("reward"),
                "n_actions": len(action), "action": action,
            }
            if embed_obs:
                rec["obs"] = obs
            out.append(rec)
    return out, teams


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", nargs="+", default=["replays"],
                    help="dirs (recursively globbed for *.json) — e.g. replays episodes")
    ap.add_argument("--out", default=str(ROOT / "training" / "moves.jsonl.gz"))
    ap.add_argument("--leaderboard", default=None, help="CSV for rating lookup (default: latest)")
    ap.add_argument("--min-rating", type=float, default=0.0, help="keep only teams rated >= this")
    ap.add_argument("--no-obs", action="store_true", help="omit obs (compact index only)")
    args = ap.parse_args()

    ratings = load_ratings(args.leaderboard)
    files = []
    for s in args.src:
        files += glob.glob(str(ROOT / s / "**" / "*.json"), recursive=True)
    files = sorted(set(files))
    print(f"Extracting from {len(files)} replays | min_rating={args.min_rating} | rating table={len(ratings)} names")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    n_ex = n_act = 0
    by_team = Counter(); by_band = Counter()
    teams_seen = set()
    with gzip.open(args.out, "wt") as gz:
        for fp in files:
            try:
                recs, teams = extract_file(fp, ratings, args.min_rating, not args.no_obs)
            except Exception as e:
                print(f"  skip {os.path.basename(fp)}: {type(e).__name__}: {e}")
                continue
            teams_seen.update(teams)
            for r in recs:
                gz.write(json.dumps(r) + "\n")
                n_ex += 1
                if r["n_actions"]:
                    n_act += 1
                by_team[r["team"]] += 1
                rt = r["rating"]
                band = ("unrated" if rt is None else
                        "1500+" if rt >= 1500 else "1200+" if rt >= 1200 else
                        "800+" if rt >= 800 else "<800")
                by_band[band] += 1

    summary = {
        "files": len(files), "examples": n_ex, "examples_with_action": n_act,
        "distinct_teams": len(teams_seen),
        "by_rating_band": dict(by_band),
        "top_teams_by_examples": by_team.most_common(15),
        "out": args.out, "min_rating": args.min_rating,
    }
    sp = Path(args.out).with_suffix("").with_suffix(".summary.json")
    sp.write_text(json.dumps(summary, indent=2))
    print(f"\nWrote {n_ex} examples ({n_act} with a launch) to {args.out}")
    print(f"By rating band: {dict(by_band)}")
    print(f"Summary: {sp}")


if __name__ == "__main__":
    main()
