"""
Orbit Wars monitoring dashboard.

Run:
    .venv/bin/streamlit run dashboard/app.py
    # or via start.sh which also runs the pipeline loop
"""

import json
import re
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

ROOT    = Path(__file__).resolve().parent.parent
DB_PATH = ROOT / "strategy" / "tracking.db"
LB_DIR  = ROOT / "leaderboards"

OUR_NAME = "Montana Schmeekler"

ENTRY_DEADLINE = datetime(2026, 6, 16, 23, 59, 0, tzinfo=timezone.utc)
SUB_DEADLINE   = datetime(2026, 6, 23, 23, 59, 0, tzinfo=timezone.utc)
GAMES_END      = datetime(2026, 7,  8, 23, 59, 0, tzinfo=timezone.utc)
COMP_START     = datetime(2026, 5,  1,  0,  0, 0, tzinfo=timezone.utc)

REFRESH_SECS = 60

# Fallback map for when submissions table is empty (bootstrap run)
SUBMISSION_IDS_FALLBACK = {
    "53676654": "coordinated_strike_interceptor v1",
    "53676680": "markowitz_portfolio_optimization v1",
}

# Map submission name slug → agent source file (for docstring extraction)
AGENT_FILE_MAP = {
    "coordinated_strike_interceptor": "agents/coordinated_strike_interceptor.py",
    "markowitz_portfolio_optimization": "agents/markowitz_portfolio_optimization.py",
}

st.set_page_config(
    page_title="Orbit Wars Dashboard",
    page_icon="🪐",
    layout="wide",
    initial_sidebar_state="collapsed",
)



# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------

@st.cache_data(ttl=60)
def load_latest_leaderboard():
    csvs = sorted(LB_DIR.glob("leaderboard_*.csv"))
    if not csvs:
        return None, None, None
    path = csvs[-1]
    df = pd.read_csv(path, encoding="utf-8-sig")
    try:
        ts = datetime.strptime(path.stem.replace("leaderboard_", ""), "%Y-%m-%d_%H-%M")
        ts = ts.replace(tzinfo=timezone.utc)
        age = (datetime.now(timezone.utc) - ts).total_seconds()
        age_str = f"{int(age // 60)}m ago" if age < 3600 else f"{age/3600:.1f}h ago"
    except ValueError:
        age_str = "unknown"
    return df, path.stem, age_str


@st.cache_data(ttl=60)
def load_prev_leaderboard():
    csvs = sorted(LB_DIR.glob("leaderboard_*.csv"))
    if len(csvs) < 2:
        return None
    return pd.read_csv(csvs[-2], encoding="utf-8-sig")


@st.cache_data(ttl=60)
def load_rank_history() -> pd.DataFrame:
    records = []

    def _add(path, ts):
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

    for p in sorted(LB_DIR.glob("leaderboard_*.csv")):
        try:
            ts = datetime.strptime(p.stem.replace("leaderboard_", ""), "%Y-%m-%d_%H-%M")
            _add(p, ts.replace(tzinfo=timezone.utc))
        except ValueError:
            pass

    for p in sorted(LB_DIR.glob("orbit-wars-publicleaderboard-*.csv")):
        try:
            ts = datetime.fromisoformat(
                p.stem.replace("orbit-wars-publicleaderboard-", "")
            ).replace(tzinfo=timezone.utc)
            _add(p, ts)
        except ValueError:
            pass

    if not records:
        return pd.DataFrame()
    return (pd.DataFrame(records)
            .sort_values("time")
            .drop_duplicates("time")
            .reset_index(drop=True))


@st.cache_data(ttl=60)
def load_episodes() -> pd.DataFrame:
    if not DB_PATH.exists():
        return pd.DataFrame()
    con = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("SELECT * FROM episodes ORDER BY create_time DESC", con)
    con.close()
    return df


@st.cache_data(ttl=300)
def load_submissions() -> pd.DataFrame:
    if DB_PATH.exists():
        try:
            con = sqlite3.connect(DB_PATH)
            df = pd.read_sql_query(
                "SELECT * FROM submissions ORDER BY submitted_at DESC", con
            )
            con.close()
            if not df.empty:
                return df
        except Exception:
            pass
    rows = [{"submission_id": k, "name": v, "status": "active", "submitted_at": ""}
            for k, v in SUBMISSION_IDS_FALLBACK.items()]
    return pd.DataFrame(rows)


@st.cache_data
def load_replay_planets(episode_id: str) -> pd.DataFrame:
    paths = list((ROOT / "replays").glob(f"*/{episode_id}.json"))
    if not paths:
        return pd.DataFrame()
    d = json.loads(paths[0].read_text())
    names = d["info"]["TeamNames"]
    rows = []
    for step_num, step in enumerate(d["steps"]):
        obs = step[0]["observation"]
        for pid, name in enumerate(names):
            count = sum(1 for p in obs["planets"] if p[1] == pid)
            rows.append({"step": step_num, "player": name, "planets": count})
    return pd.DataFrame(rows)


def read_agent_description(sub_name: str) -> str:
    slug = re.sub(r"\s+v\d+$", "", sub_name).strip().replace(" ", "_").lower()
    rel  = AGENT_FILE_MAP.get(slug)
    if not rel:
        return ""
    path = ROOT / rel
    if not path.exists():
        return ""
    lines = path.read_text().splitlines()
    desc_lines = []
    in_block = False
    for line in lines[:60]:
        if line.startswith("# ==="):
            in_block = not in_block
            continue
        if in_block and line.startswith("#"):
            text = line.lstrip("# ").rstrip()
            if text:
                desc_lines.append(text)
    return " ".join(desc_lines[:4]) if desc_lines else ""


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

YOU_COLOR    = "#f58518"   # orange — always used for our line
OTHER_COLORS = ["#4c78a8", "#72b7b2", "#54a24b", "#e45756", "#b279a2"]


def section_header(refresh_slot_ref: list):
    now = datetime.now(timezone.utc)

    def countdown(dt):
        d = dt - now
        if d.total_seconds() < 0:
            return "✅ passed"
        days, rem = divmod(int(d.total_seconds()), 86400)
        hrs, rem  = divmod(rem, 3600)
        mins      = rem // 60
        return f"{days}d {hrs}h {mins}m"

    def local(dt):
        return dt.astimezone().strftime("%b %-d %-I:%M %p %Z")

    h1, h2, h3, h4, h5 = st.columns([3, 2, 2, 2, 1])
    h1.markdown("## 🪐 Orbit Wars")
    h2.metric(f"Entry deadline", countdown(ENTRY_DEADLINE), local(ENTRY_DEADLINE), delta_color="off")
    h3.metric(f"Submission lock", countdown(SUB_DEADLINE),  local(SUB_DEADLINE),   delta_color="off")
    h4.metric(f"Games end",       countdown(GAMES_END),     local(GAMES_END),       delta_color="off")
    refresh_slot_ref.append(h5.empty())

    total   = (GAMES_END - COMP_START).total_seconds()
    elapsed = (now - COMP_START).total_seconds()
    st.progress(max(0.0, min(1.0, elapsed / total)),
                text=f"Competition: {elapsed/total*100:.1f}% through")


# ---------------------------------------------------------------------------
# My Position + Leaderboard
# ---------------------------------------------------------------------------

_TD  = "padding:5px 10px;border:1px solid #333;"
_TDR = _TD + "text-align:right;"
_TDN = _TD + "font-variant-numeric:tabular-nums;text-align:right;"


def _leaderboard_html(display_rows: list[dict]) -> str:
    rows_html = ""
    for r in display_rows:
        rank  = r["rank"]
        team  = r["team"]
        score = r["score"]
        is_us = OUR_NAME.lower() in team.lower()

        if is_us:
            row_bg = "background:rgba(99,102,241,0.25);"
            left   = "border-left:4px solid #6366f1;"
        elif rank <= 10:
            row_bg = ""
            left   = "border-left:4px solid #22c55e;"
        else:
            row_bg = ""
            left   = ""

        team_display = team if len(team) <= 30 else team[:28] + "…"
        rows_html += (
            f'<tr style="{row_bg}">'
            f'<td style="{_TDR}color:#888;{left}">{rank}</td>'
            f'<td style="{_TD}">{team_display}</td>'
            f'<td style="{_TDN}">{score:.1f}</td>'
            f'</tr>'
        )

    th = "padding:5px 10px;border:1px solid #444;color:#888;font-weight:600;background:rgba(255,255,255,0.04);"
    return (
        f'<table style="width:100%;border-collapse:collapse;font-size:0.85rem">'
        f'<thead><tr>'
        f'<th style="{th}text-align:right;">#</th>'
        f'<th style="{th}">Team</th>'
        f'<th style="{th}text-align:right;">Score</th>'
        f'</tr></thead>'
        f'<tbody>{rows_html}</tbody>'
        f'</table>'
    )


def section_position_and_leaderboard():
    lb_df, snap_name, age_str = load_latest_leaderboard()
    prev_df = load_prev_leaderboard()
    hist    = load_rank_history()

    our_row   = lb_df[lb_df["TeamName"].str.contains(OUR_NAME, case=False, na=False)] if lb_df is not None else pd.DataFrame()
    our_rank  = int(our_row.iloc[0]["Rank"])   if not our_row.empty else None
    our_score = float(our_row.iloc[0]["Score"]) if not our_row.empty else None

    score_delta = rank_delta = None
    if prev_df is not None and our_score is not None:
        prev_our = prev_df[prev_df["TeamName"].str.contains(OUR_NAME, case=False, na=False)]
        if not prev_our.empty:
            score_delta = our_score - float(prev_our.iloc[0]["Score"])
            rank_delta  = int(prev_our.iloc[0]["Rank"]) - our_rank

    left, right = st.columns([0.55, 0.45])

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
            st.altair_chart(
                alt.Chart(hist).mark_line(point=True, color="#4c78a8").encode(
                    x=alt.X("time:T", title=None, axis=alt.Axis(format="%m/%d %H:%M")),
                    y=alt.Y("score:Q", title="Score", scale=alt.Scale(zero=False)),
                    tooltip=[alt.Tooltip("time:T", format="%m/%d %H:%M"),
                             alt.Tooltip("score:Q", format=".1f"),
                             alt.Tooltip("rank:Q")],
                ).properties(height=130),
                use_container_width=True,
            )
            st.caption("Rank over time (lower = better)")
            st.altair_chart(
                alt.Chart(hist).mark_line(point=True, color="#f58518").encode(
                    x=alt.X("time:T", title=None, axis=alt.Axis(format="%m/%d %H:%M")),
                    y=alt.Y("rank:Q", title="Rank", scale=alt.Scale(reverse=True, zero=False)),
                    tooltip=[alt.Tooltip("time:T", format="%m/%d %H:%M"),
                             alt.Tooltip("rank:Q"),
                             alt.Tooltip("score:Q", format=".1f")],
                ).properties(height=130),
                use_container_width=True,
            )
        elif not hist.empty:
            st.caption("2+ snapshots needed for trend charts")
        else:
            st.info("No leaderboard history yet — run the pipeline.")

    with right:
        if lb_df is None:
            st.warning("No leaderboard snapshot. Run `pipeline/leaderboard_snapshot.py`")
            return

        prize_row   = lb_df[lb_df["Rank"] == 10]
        prize_score = float(prize_row.iloc[0]["Score"]) if not prize_row.empty else 1534.0
        gap         = prize_score - our_score if our_score else None

        st.subheader("Leaderboard")
        st.caption(
            f"Snapshot: {age_str}  ·  {len(lb_df):,} teams"
            + (f"  ·  Gap to prizes: **{gap:+.0f} pts**" if gap is not None else "")
        )

        # Top 14 + our row always last
        top14 = lb_df.head(14)
        display_rows = [
            {"rank": int(r["Rank"]), "team": r["TeamName"], "score": float(r["Score"])}
            for _, r in top14.iterrows()
        ]
        if our_rank and our_rank > 14:
            display_rows.append({"rank": our_rank, "team": OUR_NAME, "score": our_score})

        st.html(_leaderboard_html(display_rows))


# ---------------------------------------------------------------------------
# Episode card grid
# ---------------------------------------------------------------------------

def _placement_badge(place, nplayer):
    if place is None:
        return "?"
    p    = int(place)
    icon = {1: "🥇", 2: "🥈", 3: "🥉"}.get(p, "💀")
    return f"{icon} {p}/{int(nplayer)}"


def _render_episode_card(ep, col):
    ep_id   = str(ep["episode_id"])
    place   = ep.get("our_placement")
    nplayer = ep.get("num_players", 2)
    date    = str(ep.get("create_time", ""))[:10]

    with col:
        with st.container(border=True):
            st.caption(_placement_badge(place, nplayer))

            df = load_replay_planets(ep_id)
            if not df.empty:
                df["player"] = df["player"].apply(
                    lambda n: "You" if OUR_NAME.lower() in n.lower() else n[:12]
                )
                players = df["player"].unique().tolist()
                others  = [p for p in players if p != "You"]
                domain  = ["You"] + others
                colors  = [YOU_COLOR] + OTHER_COLORS[:len(others)]
                chart = (
                    alt.Chart(df)
                    .mark_line()
                    .encode(
                        x=alt.X("step:Q", axis=None),
                        y=alt.Y("planets:Q", title=None,
                                axis=alt.Axis(tickCount=3, labelFontSize=9)),
                        color=alt.Color(
                            "player:N",
                            scale=alt.Scale(domain=domain, range=colors),
                            legend=alt.Legend(
                                orient="bottom", labelLimit=70,
                                labelFontSize=9, symbolSize=60,
                                padding=0, rowPadding=0,
                            ),
                        ),
                    )
                    .properties(height=110)
                    .configure_view(strokeWidth=0)
                )
                st.altair_chart(chart, use_container_width=True)
            else:
                st.caption("no replay")

            st.caption(date)


def _render_card_grid(sub_eps: pd.DataFrame, cards_per_row: int = 4):
    if sub_eps.empty:
        st.caption("No episodes with placements downloaded yet.")
        return
    eps_list = list(sub_eps.iterrows())
    for i in range(0, len(eps_list), cards_per_row):
        chunk = eps_list[i:i + cards_per_row]
        cols  = st.columns(cards_per_row)
        for j, (_, ep) in enumerate(chunk):
            _render_episode_card(ep, cols[j])


def section_episodes():
    subs     = load_submissions()
    episodes = load_episodes()

    active  = subs[subs["status"] == "active"]
    retired = subs[subs["status"] != "active"]

    st.subheader("Active Agents")
    for _, sub in active.iterrows():
        sub_id  = str(sub["submission_id"])
        name    = sub.get("name", sub_id)
        if not episodes.empty:
            sub_eps = (episodes[episodes["submission_id"] == sub_id]
                       .loc[lambda df: df["our_placement"].notna()]
                       .sort_values("create_time", ascending=False))
        else:
            sub_eps = pd.DataFrame()

        desc = read_agent_description(name)
        with st.expander(f"**{name}**  ·  {len(sub_eps)} episodes", expanded=True):
            if desc:
                st.caption(desc)
            _render_card_grid(sub_eps)

    if not retired.empty:
        st.subheader("Retired Agents")
        for _, sub in retired.iterrows():
            sub_id  = str(sub["submission_id"])
            name    = sub.get("name", sub_id)
            if not episodes.empty:
                sub_eps = (episodes[episodes["submission_id"] == sub_id]
                           .loc[lambda df: df["our_placement"].notna()]
                           .sort_values("create_time", ascending=False))
            else:
                sub_eps = pd.DataFrame()

            with st.expander(f"**{name}**  ·  {len(sub_eps)} episodes", expanded=True):
                _render_card_grid(sub_eps)


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

def main():
    refresh_slot_ref: list = []
    section_header(refresh_slot_ref)
    st.divider()
    section_position_and_leaderboard()
    st.divider()
    section_episodes()

    if refresh_slot_ref:
        refresh_slot = refresh_slot_ref[0]
        for remaining in range(REFRESH_SECS, 0, -1):
            refresh_slot.caption(f"⏱ {remaining}s")
            time.sleep(1)
        st.cache_data.clear()
        st.rerun()


if __name__ == "__main__":
    main()
