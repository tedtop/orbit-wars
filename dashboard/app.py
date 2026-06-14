"""
Orbit Wars monitoring dashboard.

Run:
    .venv/bin/streamlit run dashboard/app.py
    # or via start.sh which also runs the pipeline loop

Layout:
  - Compact header: countdown to each deadline + progress bar
  - Left column: My Position (score, rank, trend charts)
  - Right column: Public leaderboard top 25 + our row
  - Below: My Submissions — episode stats split by 2P vs 4P
  - Below: Submission log with editable notes
  - Bottom: auto-refresh countdown (60s)
"""

import json
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

ROOT       = Path(__file__).resolve().parent.parent
DB_PATH    = ROOT / "strategy" / "tracking.db"
LB_DIR     = ROOT / "leaderboards"
NOTES_PATH = ROOT / "strategy" / "submission_notes.json"

OUR_NAME   = "Montana Schmeekler"

ENTRY_DEADLINE = datetime(2026, 6, 16, 23, 59, 0, tzinfo=timezone.utc)
SUB_DEADLINE   = datetime(2026, 6, 23, 23, 59, 0, tzinfo=timezone.utc)
GAMES_END      = datetime(2026, 7,  8, 23, 59, 0, tzinfo=timezone.utc)
COMP_START     = datetime(2026, 5,  1,  0,  0, 0, tzinfo=timezone.utc)

SUBMISSION_IDS = {
    "53676654": "coordinated_strike v1",
    "53676680": "markowitz v1",
}

REFRESH_SECS = 60

st.set_page_config(
    page_title="Orbit Wars",
    page_icon="🪐",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------

@st.cache_data(ttl=60)
def load_latest_leaderboard():
    """Return (df, snapshot_name, age_str) or (None, None, None)."""
    csvs = sorted(LB_DIR.glob("leaderboard_*.csv"))
    if not csvs:
        return None, None, None
    path = csvs[-1]
    df = pd.read_csv(path, encoding="utf-8-sig")
    # parse age from filename
    try:
        ts = datetime.strptime(path.stem.replace("leaderboard_", ""), "%Y-%m-%d_%H-%M")
        ts = ts.replace(tzinfo=timezone.utc)
        age_secs = (datetime.now(timezone.utc) - ts).total_seconds()
        if age_secs < 3600:
            age_str = f"{int(age_secs // 60)}m ago"
        else:
            age_str = f"{age_secs / 3600:.1f}h ago"
    except ValueError:
        age_str = "unknown age"
    return df, path.stem, age_str


@st.cache_data(ttl=60)
def load_prev_leaderboard():
    csvs = sorted(LB_DIR.glob("leaderboard_*.csv"))
    if len(csvs) < 2:
        return None
    return pd.read_csv(csvs[-2], encoding="utf-8-sig")


@st.cache_data(ttl=60)
def load_rank_history() -> pd.DataFrame:
    """Parse every leaderboard snapshot to build rank/score over time."""
    records = []

    # New format: leaderboard_YYYY-MM-DD_HH-MM.csv
    for p in sorted(LB_DIR.glob("leaderboard_*.csv")):
        try:
            ts = datetime.strptime(
                p.stem.replace("leaderboard_", ""), "%Y-%m-%d_%H-%M"
            ).replace(tzinfo=timezone.utc)
        except ValueError:
            continue
        _add_our_row(p, ts, records)

    # Old format: orbit-wars-publicleaderboard-YYYY-MM-DDTHH:MM:SS.csv
    for p in sorted(LB_DIR.glob("orbit-wars-publicleaderboard-*.csv")):
        try:
            ts_str = p.stem.replace("orbit-wars-publicleaderboard-", "")
            ts = datetime.fromisoformat(ts_str).replace(tzinfo=timezone.utc)
        except ValueError:
            continue
        _add_our_row(p, ts, records)

    if not records:
        return pd.DataFrame()
    df = pd.DataFrame(records).sort_values("time").drop_duplicates("time").reset_index(drop=True)
    df["time_local"] = df["time"].dt.tz_convert(None)  # for chart display
    return df


def _add_our_row(path: Path, ts: datetime, records: list):
    try:
        df = pd.read_csv(path, encoding="utf-8-sig")
        our = df[df["TeamName"].str.contains(OUR_NAME, case=False, na=False)]
        if not our.empty:
            records.append({
                "time":  ts,
                "rank":  int(our.iloc[0]["Rank"]),
                "score": float(our.iloc[0]["Score"]),
            })
    except Exception:
        pass


@st.cache_data(ttl=60)
def load_episodes() -> pd.DataFrame:
    if not DB_PATH.exists():
        return pd.DataFrame()
    con = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("SELECT * FROM episodes ORDER BY create_time DESC", con)
    con.close()
    return df


def load_notes() -> dict:
    return json.loads(NOTES_PATH.read_text()) if NOTES_PATH.exists() else {}


def save_notes(notes: dict):
    NOTES_PATH.write_text(json.dumps(notes, indent=2))


# ---------------------------------------------------------------------------
# Header — compact countdown
# ---------------------------------------------------------------------------

def section_header():
    now = datetime.now(timezone.utc)

    def countdown(dt):
        d = dt - now
        if d.total_seconds() < 0:
            return "✅ passed"
        days = d.days
        hrs  = d.seconds // 3600
        mins = (d.seconds % 3600) // 60
        return f"{days}d {hrs}h {mins}m"

    def local(dt):
        return dt.astimezone().strftime("%b %-d %-I:%M %p %Z")

    st.markdown("## 🪐 Orbit Wars")
    c1, c2, c3 = st.columns(3)
    c1.metric("Entry deadline",   local(ENTRY_DEADLINE),   countdown(ENTRY_DEADLINE),   delta_color="off")
    c2.metric("Submission lock",  local(SUB_DEADLINE),     countdown(SUB_DEADLINE),     delta_color="off")
    c3.metric("Games end",        local(GAMES_END),         countdown(GAMES_END),         delta_color="off")

    total   = (GAMES_END - COMP_START).total_seconds()
    elapsed = (now - COMP_START).total_seconds()
    st.progress(max(0.0, min(1.0, elapsed / total)),
                text=f"Competition: {elapsed/total*100:.1f}% through")


# ---------------------------------------------------------------------------
# My Position + Leaderboard (two columns)
# ---------------------------------------------------------------------------

def section_position_and_leaderboard():
    lb_df, snap_name, age_str = load_latest_leaderboard()
    prev_df = load_prev_leaderboard()
    hist    = load_rank_history()

    our_row   = lb_df[lb_df["TeamName"].str.contains(OUR_NAME, case=False, na=False)] if lb_df is not None else pd.DataFrame()
    our_rank  = int(our_row.iloc[0]["Rank"])  if not our_row.empty else None
    our_score = float(our_row.iloc[0]["Score"]) if not our_row.empty else None

    # Score/rank delta vs previous snapshot
    score_delta = rank_delta = None
    if prev_df is not None and our_score is not None:
        prev_our = prev_df[prev_df["TeamName"].str.contains(OUR_NAME, case=False, na=False)]
        if not prev_our.empty:
            score_delta = our_score - float(prev_our.iloc[0]["Score"])
            rank_delta  = int(prev_our.iloc[0]["Rank"]) - our_rank  # positive = improved

    left, right = st.columns([0.55, 0.45])

    # ── Left: My Position ──────────────────────────────────────────────────
    with left:
        st.subheader("My Position")
        m1, m2 = st.columns(2)
        m1.metric("Score", f"{our_score:.1f}" if our_score else "—",
                  delta=f"{score_delta:+.1f}" if score_delta is not None else None)
        m2.metric("Rank", f"#{our_rank}" if our_rank else "—",
                  delta=f"{rank_delta:+d} places" if rank_delta is not None else None,
                  delta_color="normal")

        if not hist.empty and len(hist) >= 2:
            st.caption("Score over time")
            score_chart = (
                alt.Chart(hist)
                .mark_line(point=True, color="#4c78a8")
                .encode(
                    x=alt.X("time:T", title=None, axis=alt.Axis(format="%m/%d %H:%M")),
                    y=alt.Y("score:Q", title="Score", scale=alt.Scale(zero=False)),
                    tooltip=[
                        alt.Tooltip("time:T", title="Time", format="%m/%d %H:%M"),
                        alt.Tooltip("score:Q", title="Score", format=".1f"),
                        alt.Tooltip("rank:Q",  title="Rank"),
                    ],
                )
                .properties(height=140)
            )
            st.altair_chart(score_chart, use_container_width=True)

            st.caption("Rank over time (lower = better)")
            rank_chart = (
                alt.Chart(hist)
                .mark_line(point=True, color="#f58518")
                .encode(
                    x=alt.X("time:T", title=None, axis=alt.Axis(format="%m/%d %H:%M")),
                    y=alt.Y("rank:Q", title="Rank (#)",
                            scale=alt.Scale(reverse=True, zero=False)),
                    tooltip=[
                        alt.Tooltip("time:T", title="Time", format="%m/%d %H:%M"),
                        alt.Tooltip("rank:Q",  title="Rank"),
                        alt.Tooltip("score:Q", title="Score", format=".1f"),
                    ],
                )
                .properties(height=140)
            )
            st.altair_chart(rank_chart, use_container_width=True)
        elif not hist.empty:
            st.caption("Score history will appear after 2+ snapshots")
            st.dataframe(hist[["time", "rank", "score"]], hide_index=True, use_container_width=True)
        else:
            st.info("No leaderboard history yet. Run the pipeline to fetch snapshots.")

    # ── Right: Leaderboard ─────────────────────────────────────────────────
    with right:
        if lb_df is None:
            st.warning("No leaderboard snapshot. Run `pipeline/leaderboard_snapshot.py`")
            return

        prize_row   = lb_df[lb_df["Rank"] == 10]
        prize_score = float(prize_row.iloc[0]["Score"]) if not prize_row.empty else 1534.0
        gap_to_prize = prize_score - our_score if our_score else None

        st.subheader("Leaderboard")
        st.caption(f"Snapshot: {age_str}  ·  {len(lb_df):,} teams  ·  Prize cutoff: {prize_score:.0f}")

        # Build display table (top 25 + our row if outside top 25)
        display = lb_df.head(25).copy()
        if our_rank and our_rank > 25:
            display = pd.concat([display, our_row])

        # Score delta column
        if prev_df is not None:
            prev_scores = prev_df.set_index("TeamName")["Score"].to_dict()
            display = display.copy()
            display["Δ"] = display.apply(
                lambda r: f"{r['Score'] - prev_scores[r['TeamName']]:+.1f}"
                if r["TeamName"] in prev_scores else "new",
                axis=1
            )

        # "You" marker column — works on dark and light themes
        display["  "] = display["TeamName"].apply(
            lambda n: "⬅ you" if OUR_NAME.lower() in str(n).lower() else ""
        )

        cols = ["Rank", "TeamName", "Score"]
        if "Δ" in display.columns:
            cols.append("Δ")
        cols.append("  ")

        st.dataframe(
            display[cols].reset_index(drop=True),
            hide_index=True,
            use_container_width=True,
            height=520,
            column_config={
                "Rank":     st.column_config.NumberColumn("Rank", format="%d"),
                "Score":    st.column_config.NumberColumn("Score", format="%.1f"),
                "TeamName": st.column_config.TextColumn("Team"),
                "  ":       st.column_config.TextColumn(""),
            },
        )

        if gap_to_prize is not None:
            st.caption(f"Gap to prizes: **{gap_to_prize:+.0f} pts** · Our rank: **#{our_rank}**")


# ---------------------------------------------------------------------------
# My Submissions — 2P and 4P stats split
# ---------------------------------------------------------------------------

def section_submissions():
    st.subheader("My Submissions")
    episodes = load_episodes()

    if episodes.empty:
        st.info("No episodes yet — run the pipeline.")
        return

    for sub_id, sub_name in SUBMISSION_IDS.items():
        sub_eps = episodes[episodes["submission_id"] == sub_id]
        n_total = len(sub_eps)

        twop  = sub_eps[(sub_eps["num_players"] == 2) & sub_eps["our_placement"].notna()]
        fourp = sub_eps[(sub_eps["num_players"] == 4) & sub_eps["our_placement"].notna()]
        n_downloaded = int(sub_eps["downloaded"].sum())

        with st.expander(f"**{sub_name}**  ·  {n_total} episodes  ·  {n_downloaded} downloaded", expanded=True):
            if n_downloaded == 0:
                st.caption("No replays downloaded yet.")
                continue

            cols = st.columns(2)

            # ── 2P stats ──────────────────────────────────────────────────
            with cols[0]:
                st.markdown("**2-player games**")
                if twop.empty:
                    st.caption("No 2P replays downloaded yet.")
                else:
                    wins2   = int((twop["our_placement"] == 1).sum())
                    losses2 = int((twop["our_placement"] == 2).sum())
                    n2 = wins2 + losses2
                    pct = wins2 / n2 * 100 if n2 else 0
                    m1, m2, m3 = st.columns(3)
                    m1.metric("Games",  n2)
                    m2.metric("Wins",   wins2)
                    m3.metric("Win %",  f"{pct:.0f}%")
                    place2_df = pd.DataFrame({
                        "Result": ["1st (W)", "2nd (L)"],
                        "Count":  [wins2, losses2],
                    })
                    st.caption("Win / Loss")
                    st.bar_chart(place2_df.set_index("Result"), height=140)

            # ── 4P stats ──────────────────────────────────────────────────
            with cols[1]:
                st.markdown("**4-player games**")
                if fourp.empty:
                    st.caption("No 4P replays downloaded yet.")
                else:
                    counts = {1: 0, 2: 0, 3: 0, 4: 0}
                    for p in fourp["our_placement"]:
                        k = int(p)
                        if k in counts:
                            counts[k] += 1
                    n4 = sum(counts.values())
                    avg4 = fourp["our_placement"].mean()
                    m1, m2 = st.columns(2)
                    m1.metric("Games",         n4)
                    m2.metric("Avg placement", f"{avg4:.2f}")
                    place4_df = pd.DataFrame({
                        "Place": ["1st", "2nd", "3rd", "4th"],
                        "Count": [counts[1], counts[2], counts[3], counts[4]],
                    })
                    st.caption("Placement distribution")
                    st.bar_chart(place4_df.set_index("Place"), height=140)


# ---------------------------------------------------------------------------
# Submission log with notes
# ---------------------------------------------------------------------------

def section_log():
    st.subheader("Submission Log")
    notes    = load_notes()
    episodes = load_episodes()

    cols = st.columns([2, 1, 1, 1, 3])
    cols[0].markdown("**Submission**")
    cols[1].markdown("**Episodes**")
    cols[2].markdown("**Best placement**")
    cols[3].markdown("**Win %**")
    cols[4].markdown("**Notes**")
    st.divider()

    for sub_id, sub_name in SUBMISSION_IDS.items():
        sub_eps = episodes[episodes["submission_id"] == sub_id] if not episodes.empty else pd.DataFrame()
        placed  = sub_eps[sub_eps["our_placement"].notna()] if not sub_eps.empty else pd.DataFrame()
        best    = int(placed["our_placement"].min()) if not placed.empty else None
        twop    = placed[placed["num_players"] == 2] if not placed.empty else pd.DataFrame()
        win_pct = f"{(twop['our_placement']==1).mean()*100:.0f}%" if not twop.empty else "—"

        cols = st.columns([2, 1, 1, 1, 3])
        cols[0].markdown(f"**{sub_name}**  \n`{sub_id}`")
        cols[1].write(len(sub_eps))
        cols[2].write(f"{best}{'st' if best==1 else 'nd' if best==2 else 'rd' if best==3 else 'th'}" if best else "—")
        cols[3].write(win_pct)
        new_note = cols[4].text_input(
            "Notes", value=notes.get(sub_id, ""),
            key=f"note_{sub_id}", label_visibility="collapsed",
            placeholder="e.g. increased aggression, better 4P…"
        )
        if new_note != notes.get(sub_id, ""):
            notes[sub_id] = new_note
            save_notes(notes)


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

def main():
    section_header()
    st.divider()
    section_position_and_leaderboard()
    st.divider()
    section_submissions()
    st.divider()
    section_log()

    # Auto-refresh countdown
    st.divider()
    refresh_slot = st.empty()
    for remaining in range(REFRESH_SECS, 0, -1):
        refresh_slot.caption(f"⏱ Auto-refresh in {remaining}s")
        time.sleep(1)
    st.cache_data.clear()
    st.rerun()


if __name__ == "__main__":
    main()
