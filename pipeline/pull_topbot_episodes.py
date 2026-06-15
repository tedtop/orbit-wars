#!/usr/bin/env python3
"""Pull prize-zone episodes from Kaggle's public daily dumps for BC/RL training.

Strategy (validated from the episode index):
  - `kaggle/orbit-wars-episodes-index` -> manifest.csv: per-day dump slug +
    `top_avg_score` / `median_avg_score`. In June the MEDIAN game already averages
    ~1400, so recent dumps are dense with strong-bot play.
  - Each daily dump (`kaggle/orbit-wars-episodes-YYYY-MM-DD`) holds individually
    downloadable `<episode_id>.json` files (~2-11 MB). We budget single-file
    downloads instead of grabbing the full ~20 GB/day dump.
  - Optionally keep only episodes containing a prize-zone team (roster from the
    latest leaderboard CSV, Score >= --require-rating), deleting the rest to save disk.

Downloads land in `episodes/<date>/<episode_id>.json`, ready for
`pipeline/extract_moves.py --src episodes --min-rating 1500`.

Examples:
    # 300 episodes/day from the last 3 prize-zone-era days, keep only games with a >=1500 team
    python pipeline/pull_topbot_episodes.py --days-back 3 --max-per-day 300 --require-rating 1500
    # everything available for one specific day (no roster filter)
    python pipeline/pull_topbot_episodes.py --dates 2026-06-14 --max-per-day 99999
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import os
import random
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EPISODES = ROOT / "episodes"
INDEX_DS = "kaggle/orbit-wars-episodes-index"


def _api():
    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi(); api.authenticate()
    return api


def _retry(fn, *a, tries=8, base=2.0, cap=90.0, **kw):
    """Call fn with exponential backoff (Kaggle throws intermittent 403/5xx)."""
    import time
    last = None
    for i in range(tries):
        try:
            return fn(*a, **kw)
        except Exception as e:
            last = e
            time.sleep(min(cap, base * (2 ** i)))
    raise last


def load_manifest(api) -> list[dict]:
    EPISODES.mkdir(exist_ok=True)
    mpath = EPISODES / "_index" / "manifest.csv"
    # Kaggle throttles consecutive calls hard; reuse the cached manifest so the
    # FIRST API hit of a run is the (reliable) single list call, not a download.
    if not mpath.exists():
        _retry(api.dataset_download_files, INDEX_DS, path=str(EPISODES / "_index"), unzip=True, quiet=True)
    with open(mpath, encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def prizezone_roster(min_rating: float) -> set[str]:
    cands = sorted((ROOT / "leaderboards").glob("*.csv"))
    roster: set[str] = set()
    if not cands:
        return roster
    with open(cands[-1], encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            try:
                if float(row["Score"]) >= min_rating:
                    nm = (row.get("TeamName") or "").strip().lower()
                    if nm:
                        roster.add(nm)
                    for u in (row.get("TeamMemberUserNames") or "").split(","):
                        if u.strip():
                            roster.add(u.strip().lower())
            except (KeyError, ValueError):
                continue
    return roster


def list_episode_files(api, slug: str, cap: int, dest_dir: Path) -> list[str]:
    """Paginate a dump's file list (cached to disk so we list each day only once)."""
    cache = dest_dir / "_filelist.json"
    if cache.exists():
        cached = json.loads(cache.read_text())
        if len(cached) >= cap:
            return cached
    import time
    names: list[str] = []
    token = None
    want = max(cap * 3, cap + 100)
    # The list endpoint throttles hard under PAGINATION (multiple calls), even though
    # a single call reliably returns up to 200 names. So for modest pulls we do ONE
    # large-page call (no pagination) on a FRESH client; only paginate if we truly
    # need more than one page can give.
    fresh = _api()
    first = _retry(fresh.dataset_list_files, slug, page_token=None, page_size=200)
    names += [f.name for f in getattr(first, "files", []) if f.name.endswith(".json")]
    token = getattr(first, "nextPageToken", None) or getattr(first, "next_page_token", None)
    while token and len(names) < want:
        time.sleep(3.0)  # heavy pacing between pages
        res = _retry(fresh.dataset_list_files, slug, page_token=token, page_size=200)
        page = [f.name for f in getattr(res, "files", []) if f.name.endswith(".json")]
        if not page:
            break
        names += page
        token = getattr(res, "nextPageToken", None) or getattr(res, "next_page_token", None)
    dest_dir.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(names))
    return names


def download_one(api, slug: str, name: str, dest_dir: Path) -> Path | None:
    dest = dest_dir / name
    if dest.exists():
        return dest
    try:
        _retry(api.dataset_download_file, slug, name, path=str(dest_dir), quiet=True)
    except Exception as e:
        print(f"    [warn] {name}: {e}")
        return None
    # Kaggle may deliver the single file zipped.
    z = dest_dir / (name + ".zip")
    if z.exists():
        with zipfile.ZipFile(z) as zf:
            zf.extractall(dest_dir)
        z.unlink()
    return dest if dest.exists() else None


def has_prizezone_team(path: Path, roster: set[str]) -> bool:
    try:
        d = json.loads(path.read_text())
        teams = d.get("info", {}).get("TeamNames", [])
        return any((t or "").strip().lower() in roster for t in teams)
    except Exception:
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dates", help="comma list YYYY-MM-DD; overrides --days-back")
    ap.add_argument("--days-back", type=int, default=3, help="use the N most recent dump days")
    ap.add_argument("--min-top-score", type=float, default=1500.0,
                    help="only use days whose top_avg_score >= this")
    ap.add_argument("--max-per-day", type=int, default=200)
    ap.add_argument("--require-rating", type=float, default=0.0,
                    help="keep only episodes containing a team rated >= this (else delete)")
    ap.add_argument("--full-dump", action="store_true",
                    help="download each day's ENTIRE dump in one request (robust; ~20GB/day, "
                         "no rate-limit-prone file listing). Then optionally prune by --require-rating.")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    api = _api()
    manifest = load_manifest(api)
    rows = []
    for r in manifest:
        try:
            r["_top"] = float(r.get("top_avg_score") or 0)
        except ValueError:
            r["_top"] = 0.0
        rows.append(r)

    if args.dates:
        want = set(args.dates.split(","))
        days = [r for r in rows if r["date"] in want]
    else:
        elig = [r for r in rows if r["_top"] >= args.min_top_score]
        days = sorted(elig, key=lambda r: r["date"])[-args.days_back:]

    roster = prizezone_roster(args.require_rating) if args.require_rating > 0 else set()
    print(f"Selected {len(days)} day(s): {[r['date'] for r in days]}")
    if args.require_rating > 0:
        print(f"Prize-zone roster (>= {args.require_rating}): {len(roster)} teams")

    rng = random.Random(args.seed)
    total_kept = total_dl = 0
    for r in days:
        slug, date = r["daily_dataset_slug"], r["date"]
        if "/" not in slug:        # manifest stores a bare slug; the API needs owner/slug
            slug = "kaggle/" + slug
        dest_dir = EPISODES / date
        dest_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n[{date}] {slug}  (top_avg={r['_top']:.0f}, {r.get('episode_count')} eps)", flush=True)

        if args.full_dump:
            print(f"  downloading FULL dump in one request (~{int(r.get('total_bytes',0))/1e9:.0f} GB)…", flush=True)
            _retry(api.dataset_download_files, slug, path=str(dest_dir), unzip=True, quiet=True, tries=4)
            eps = list(dest_dir.glob("*.json"))
            kept = 0
            for path in eps:
                if roster and not has_prizezone_team(path, roster):
                    path.unlink(); continue
                kept += 1
            total_kept += kept; total_dl += len(eps)
            print(f"  [{date}] dump unzipped: {len(eps)} episodes, kept {kept} prize-zone", flush=True)
            continue

        names = list_episode_files(api, slug, args.max_per_day, dest_dir)
        rng.shuffle(names)
        kept = dl = 0
        import time
        for name in names:
            if kept >= args.max_per_day:
                break
            path = download_one(api, slug, name, dest_dir)
            time.sleep(0.6)  # gentle pacing to avoid Kaggle rate limits
            if path is None:
                continue
            dl += 1
            if roster and not has_prizezone_team(path, roster):
                path.unlink()
                continue
            kept += 1
            if kept % 25 == 0:
                print(f"    kept {kept} / downloaded {dl} …", flush=True)
        total_kept += kept; total_dl += dl
        print(f"  [{date}] kept {kept} prize-zone episodes (downloaded {dl})")

    size = sum(f.stat().st_size for f in EPISODES.rglob("*.json")) / 1e9
    print(f"\nDone. kept={total_kept} downloaded={total_dl} | episodes/ now ~{size:.1f} GB")
    print("Next: .venv/bin/python pipeline/extract_moves.py --src episodes --min-rating 1500 "
          "--out training/moves_prizezone.jsonl.gz")


if __name__ == "__main__":
    main()
