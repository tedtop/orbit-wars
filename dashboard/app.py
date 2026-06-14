"""
Orbit Wars monitoring dashboard.

Run:
    .venv/bin/streamlit run dashboard/app.py
    # or via start.sh
"""

import json
import re
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

ROOT    = Path(__file__).resolve().parent.parent
DB_PATH = ROOT / "strategy" / "tracking.db"
LB_DIR  = ROOT / "leaderboards"

OUR_NAME = "Montana Schmeekler"

ENTRY_DEADLINE = datetime(2026, 6, 16, 23, 59, 0, tzinfo=timezone.utc)
SUB_DEADLINE   = datetime(2026, 6, 23, 23, 59, 0, tzinfo=timezone.utc)
GAMES_END      = datetime(2026, 7,  8, 23, 59, 0, tzinfo=timezone.utc)
COMP_START     = datetime(2026, 5,  1,  0,  0, 0, tzinfo=timezone.utc)


SUBMISSION_IDS_FALLBACK = {
    "53676654": "coordinated_strike_interceptor v1",
    "53676680": "markowitz_portfolio_optimization v1",
}

AGENT_FILE_MAP = {
    "coordinated_strike_interceptor": "agents/coordinated_strike_interceptor.py",
    "markowitz_portfolio_optimization": "agents/markowitz_portfolio_optimization.py",
}

YOU_COLOR    = "#f58518"
OTHER_COLORS = ["#4c78a8", "#72b7b2", "#54a24b", "#e45756", "#b279a2"]

st.set_page_config(
    page_title="Orbit Wars Dashboard",
    page_icon="🪐",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    "<style>"
    "[data-stale='true'],[data-stale='true'] *{opacity:1!important;transition:none!important}"
    ".vega-actions{display:none!important}"
    ".vega-embed details summary{display:none!important}"
    "[data-testid='stElementToolbar']{display:none!important;visibility:hidden!important;opacity:0!important;pointer-events:none!important}"
    "[data-testid='stVegaLiteChart']{cursor:pointer}"
    "[data-testid='stVegaLiteChart']{margin-bottom:-0.9rem!important}"
    "[role='tab'] p{color:rgba(255,255,255,0.9)!important}"
    "</style>",
    unsafe_allow_html=True,
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
                prize_row = df[df["Rank"] == 10]
                prize_score = float(prize_row.iloc[0]["Score"]) if not prize_row.empty else None
                records.append({
                    "time":        ts,
                    "rank":        int(our.iloc[0]["Rank"]),
                    "score":       float(our.iloc[0]["Score"]),
                    "prize_score": prize_score,
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
    rows = [{"submission_id": k, "name": v, "status": "active",
             "submitted_at": "", "public_score": None}
            for k, v in SUBMISSION_IDS_FALLBACK.items()]
    return pd.DataFrame(rows)


@st.cache_data
def load_replay_full(episode_id: str) -> tuple[pd.DataFrame, list]:
    paths = list((ROOT / "replays").glob(f"*/{episode_id}.json"))
    if not paths:
        return pd.DataFrame(), []
    d = json.loads(paths[0].read_text())
    names = d["info"]["TeamNames"]
    rows = []
    for step_num, step in enumerate(d["steps"]):
        obs = step[0]["observation"]
        for pid, name in enumerate(names):
            planet_count = sum(1 for p in obs["planets"] if p[1] == pid)
            production   = sum(p[6] for p in obs["planets"] if p[1] == pid)
            garrison     = sum(p[5] for p in obs["planets"] if p[1] == pid)
            fleets       = sum(f[5] for f in obs["fleets"]  if f[1] == pid)
            rows.append({
                "step":         step_num,
                "player":       name,
                "planet_count": planet_count,
                "production":   production,
                "total_ships":  garrison + fleets,
            })
    return pd.DataFrame(rows), names


@st.cache_data(ttl=300)
def compute_opponent_quality(sub_id: str) -> tuple:
    """Returns (avg_rank_on_wins, avg_rank_on_losses) by joining replays with leaderboard."""
    lb_df, _, _ = load_latest_leaderboard()
    if lb_df is None:
        return None, None
    episodes = load_episodes()
    if episodes.empty:
        return None, None
    sub_eps = (episodes[(episodes["submission_id"] == sub_id) &
                        episodes["our_placement"].notna()])
    win_ranks, loss_ranks = [], []
    lb_lower = lb_df.copy()
    lb_lower["_name_lower"] = lb_lower["TeamName"].str.lower()
    for _, ep in sub_eps.iterrows():
        df, names = load_replay_full(str(ep["episode_id"]))
        if df.empty:
            continue
        opponents = [n for n in names if OUR_NAME.lower() not in n.lower()]
        for opp in opponents:
            match = lb_lower[lb_lower["_name_lower"] == opp.lower()]
            if not match.empty:
                rank = int(match.iloc[0]["Rank"])
                if ep["our_placement"] == 1:
                    win_ranks.append(rank)
                else:
                    loss_ranks.append(rank)
    win_avg  = int(sum(win_ranks)  / len(win_ranks))  if win_ranks  else None
    loss_avg = int(sum(loss_ranks) / len(loss_ranks)) if loss_ranks else None
    return win_avg, loss_avg


def read_agent_description(sub_name: str) -> str:
    slug = re.sub(r"\s+v\d+$", "", sub_name).strip().replace(" ", "_").lower()
    rel  = AGENT_FILE_MAP.get(slug)
    if not rel:
        return ""
    path = ROOT / rel
    if not path.exists():
        return ""
    lines = path.read_text().splitlines()
    desc_lines, in_block, started = [], False, False
    for line in lines[:200]:
        if line.startswith("# ==="):
            if in_block:
                break   # end of block
            in_block = True
            continue
        if in_block:
            if not line.startswith("#"):
                break
            text = line[2:].rstrip() if line.startswith("# ") else line[1:].rstrip()
            if not started and text.lower().startswith("orbit wars bot:"):
                continue  # skip the bot-name header line
            if text or started:
                started = True
                desc_lines.append(text)
    # Strip leading/trailing blank lines
    while desc_lines and not desc_lines[0]:
        desc_lines.pop(0)
    while desc_lines and not desc_lines[-1]:
        desc_lines.pop()
    return "\n".join(desc_lines)


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

def section_header():
    now = datetime.now(timezone.utc)

    def countdown(dt):
        d = dt - now
        if d.total_seconds() < 0:
            return "✅ passed"
        days, rem = divmod(int(d.total_seconds()), 86400)
        hrs, rem  = divmod(rem, 3600)
        return f"{days}d {hrs}h {rem//60}m"

    def local(dt):
        return dt.astimezone().strftime("%b %-d %-I:%M %p %Z")

    h1, h2, h3, h4, h5 = st.columns([3, 2, 2, 2, 1])
    h1.markdown("## 🪐 Orbit Wars")
    _ngrok = Path("/tmp/orbit_wars_ngrok_url.txt")
    _url   = _ngrok.read_text().strip() if _ngrok.exists() else ""
    if _url:
        components.html(
            f"""<script>
            (function() {{
                function patch() {{
                    var a = window.parent.document.querySelector('a[href="#orbit-wars"]');
                    if (a) {{ a.href = "{_url}"; a.target = "_blank"; a.rel = "noopener"; }}
                }}
                patch();
                setTimeout(patch, 400);
            }})();
            </script>""",
            height=0, scrolling=False,
        )
    h2.metric("Entry deadline",  countdown(ENTRY_DEADLINE), local(ENTRY_DEADLINE),  delta_color="off")
    h3.metric("Submission lock", countdown(SUB_DEADLINE),   local(SUB_DEADLINE),    delta_color="off")
    h4.metric("Games end",       countdown(GAMES_END),      local(GAMES_END),       delta_color="off")

    total     = (GAMES_END    - COMP_START).total_seconds()
    elapsed   = (now          - COMP_START).total_seconds()
    lock_pos  = (SUB_DEADLINE - COMP_START).total_seconds()
    pct       = max(0.0, min(1.0, elapsed  / total)) * 100
    lock_pct  = max(0.0, min(1.0, lock_pos / total)) * 100
    sub_passed = now >= SUB_DEADLINE

    # Custom progress bar with submission-lock marker
    fill_color  = "#4c78a8"
    lock_color  = "#f59e0b" if not sub_passed else "#6b7280"
    lock_label  = "🔒 Submissions locked" if not sub_passed else "🔒 Submissions locked"
    # Compact single-line HTML to avoid markdown code-block treatment
    bar_html = (
        f'<div style="position:relative;height:14px;background:#1e1e1e;border-radius:4px;margin-bottom:4px">'
        f'<div style="height:100%;width:{pct:.1f}%;background:{fill_color};border-radius:4px;transition:width 0.3s"></div>'
        f'<div style="position:absolute;top:0;left:{lock_pct:.2f}%;width:2px;height:100%;background:{lock_color};border-radius:1px"></div>'
        f'<div style="position:absolute;top:-18px;left:{lock_pct:.2f}%;transform:translateX(-50%);font-size:0.68rem;color:{lock_color};white-space:nowrap">{lock_label}</div>'
        f'</div>'
        f'<div style="font-size:0.72rem;color:#555;margin-bottom:4px">'
        f'Competition: {pct:.1f}% through'
        f'{"  ·  submissions locked" if sub_passed else f"  ·  {countdown(SUB_DEADLINE)} until submissions locked"}'
        f'</div>'
    )
    st.markdown(bar_html, unsafe_allow_html=True)



# ---------------------------------------------------------------------------
# My Position + Leaderboard
# ---------------------------------------------------------------------------

_TD  = "padding:5px 10px;border:1px solid #333;"
_TDR = _TD + "text-align:right;"
_TDN = _TD + "font-variant-numeric:tabular-nums;text-align:right;"


def _leaderboard_html(display_rows: list[dict]) -> str:
    rows_html = ""
    for r in display_rows:
        rank, team, score = r["rank"], r["team"], r["score"]
        is_us = OUR_NAME.lower() in team.lower()
        row_bg = "background:rgba(99,102,241,0.25);" if is_us else ""
        left   = ("border-left:4px solid #6366f1;" if is_us
                  else "border-left:4px solid #22c55e;" if rank <= 10
                  else "")
        team_d = team if len(team) <= 30 else team[:28] + "…"
        rows_html += (f'<tr style="{row_bg}">'
                      f'<td style="{_TDR}color:#888;{left}">{rank}</td>'
                      f'<td style="{_TD}">{team_d}</td>'
                      f'<td style="{_TDN}">{score:.1f}</td>'
                      f'</tr>')

    th = "padding:5px 10px;border:1px solid #444;color:#888;font-weight:600;background:rgba(255,255,255,0.04);"
    return (f'<table style="width:100%;border-collapse:collapse;font-size:0.85rem">'
            f'<thead><tr>'
            f'<th style="{th}text-align:right;">#</th>'
            f'<th style="{th}">Team</th>'
            f'<th style="{th}text-align:right;">Score</th>'
            f'</tr></thead>'
            f'<tbody>{rows_html}</tbody>'
            f'</table>')


def section_position_and_leaderboard():
    lb_df, _, age_str = load_latest_leaderboard()
    prev_df = load_prev_leaderboard()
    hist    = load_rank_history()

    our_row   = lb_df[lb_df["TeamName"].str.contains(OUR_NAME, case=False, na=False)] if lb_df is not None else pd.DataFrame()
    our_rank  = int(our_row.iloc[0]["Rank"])    if not our_row.empty else None
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
                  delta=f"{score_delta:+.1f} since last snapshot" if score_delta is not None else None)
        m2.metric("Rank", f"#{our_rank}" if our_rank else "—",
                  delta=f"{rank_delta:+d} places since last snapshot" if rank_delta is not None else None,
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
                ).properties(height=120), width="stretch")

            st.caption("Rank over time (lower = better)")
            st.altair_chart(
                alt.Chart(hist).mark_line(point=True, color="#f58518").encode(
                    x=alt.X("time:T", title=None, axis=alt.Axis(format="%m/%d %H:%M")),
                    y=alt.Y("rank:Q", title="Rank", scale=alt.Scale(reverse=True, zero=False)),
                    tooltip=[alt.Tooltip("time:T", format="%m/%d %H:%M"),
                             alt.Tooltip("rank:Q"),
                             alt.Tooltip("score:Q", format=".1f")],
                ).properties(height=120), width="stretch")

            gap_df = hist.dropna(subset=["prize_score"]).copy()
            if not gap_df.empty:
                gap_df["gap"] = gap_df["prize_score"] - gap_df["score"]
                st.caption("Gap to prize cutoff (rank #10)")
                st.altair_chart(
                    alt.Chart(gap_df).mark_line(point=True, color="#e45756").encode(
                        x=alt.X("time:T", title=None, axis=alt.Axis(format="%m/%d %H:%M")),
                        y=alt.Y("gap:Q", title="Points behind #10"),
                        tooltip=[alt.Tooltip("time:T", format="%m/%d %H:%M"),
                                 alt.Tooltip("gap:Q", title="Gap", format=".1f")],
                    ).properties(height=110), width="stretch")
        else:
            st.info("No leaderboard history yet — run the pipeline.")

    with right:
        if lb_df is None:
            st.warning("No leaderboard snapshot found.")
            return

        prize_row   = lb_df[lb_df["Rank"] == 10]
        prize_score = float(prize_row.iloc[0]["Score"]) if not prize_row.empty else None
        gap         = prize_score - our_score if (prize_score and our_score) else None

        st.subheader("Leaderboard")
        st.caption(
            f"Snapshot: {age_str}  ·  {len(lb_df):,} teams"
            + (f"  ·  Gap to prizes: **{gap:+.0f} pts**" if gap is not None else "")
        )

        top14 = lb_df.head(14)
        display_rows = [
            {"rank": int(r["Rank"]), "team": r["TeamName"], "score": float(r["Score"])}
            for _, r in top14.iterrows()
        ]
        if our_rank and our_rank > 14:
            display_rows.append({"rank": our_rank, "team": OUR_NAME, "score": our_score})

        st.html(_leaderboard_html(display_rows))


# ---------------------------------------------------------------------------
# Episode detail modal
# ---------------------------------------------------------------------------

@st.dialog("Episode Detail", width="large")
def episode_detail_modal(episode_id: str, episode_meta: dict):
    place   = episode_meta.get("our_placement")
    nplayer = episode_meta.get("num_players", 2)
    date    = _fmt_card_time(episode_meta.get("create_time", ""))

    with st.spinner("Loading replay…"):
        df, names = load_replay_full(episode_id)
    if df.empty:
        st.warning("Replay not downloaded yet — run the pipeline.")
        return

    total_steps = int(df["step"].max()) + 1
    st.markdown(
        f"**{_placement_badge(place, nplayer)}** · {date} · "
        f"Episode `{episode_id}` · {total_steps} steps"
    )

    name_map = {n: ("You" if OUR_NAME.lower() in n.lower() else n[:20]) for n in names}
    df["player"] = df["player"].map(name_map)
    players     = ["You"] + [v for v in name_map.values() if v != "You"]
    others      = [p for p in players if p != "You"]
    domain      = ["You"] + others
    colors      = [YOU_COLOR] + OTHER_COLORS[:len(others)]
    color_scale = alt.Scale(domain=domain, range=colors)

    def make_chart(y_field, y_title):
        return (
            alt.Chart(df).mark_line(strokeWidth=2).encode(
                x=alt.X("step:Q", title="Step"),
                y=alt.Y(f"{y_field}:Q", title=y_title),
                color=alt.Color("player:N",
                                scale=color_scale,
                                legend=alt.Legend(title=None, orient="bottom")),
                tooltip=["step:Q", "player:N",
                         alt.Tooltip(f"{y_field}:Q", title=y_title)],
            ).properties(height=300)
        )

    tab1, tab2, tab3, tab4 = st.tabs(["🌍 Planets", "⚙️ Production", "🚀 Total Ships", "📋 Table"])

    with tab1:
        st.caption("Planets owned per player over time")
        st.altair_chart(make_chart("planet_count", "Planets"), width="stretch")
    with tab2:
        st.caption("Ships/turn generated from owned planets — the true economic indicator")
        st.altair_chart(make_chart("production", "Ships/turn"), width="stretch")
    with tab3:
        st.caption("Planet garrisons + fleets in transit — total military strength")
        st.altair_chart(make_chart("total_ships", "Total ships"), width="stretch")
    with tab4:
        metric = st.selectbox(
            "Metric", ["planet_count", "production", "total_ships"],
            format_func=lambda x: {"planet_count": "Planets",
                                   "production":   "Production (ships/turn)",
                                   "total_ships":  "Total Ships"}[x]
        )
        pivot = df.pivot(index="step", columns="player", values=metric)
        pivot.columns.name = None
        st.dataframe(pivot, width="stretch", height=420)


# ---------------------------------------------------------------------------
# Episode cards
# ---------------------------------------------------------------------------

def _ordinal(n: int) -> str:
    return {1: "1st", 2: "2nd", 3: "3rd"}.get(n, f"{n}th")


def _placement_badge(place, nplayer):
    if place is None:
        return "?"
    p = int(place)
    n = int(nplayer)
    if p == 1:
        icon = "🟢"
    elif n == 4 and p < n:
        icon = "🟡"
    else:
        icon = "🔴"
    return f"{icon} {p}/{n}"


def _render_form_strip(sub_eps: pd.DataFrame) -> None:
    """Episode history bubbles: hover for tooltip, click to scroll + flash the card."""
    spans = ""
    for _, ep in sub_eps.iterrows():
        ep_id = str(ep["episode_id"])
        p = int(ep["our_placement"])
        n = int(ep["num_players"])
        icon = "🟢" if p == 1 else ("🟡" if n == 4 and p < n else "🔴")
        tip = f"{p}/{n}P · {_fmt_card_time(ep.get('create_time', ''))}"
        try:
            _, names = load_replay_full(ep_id)
            if names:
                opps = ", ".join(nm[:18] for nm in names if OUR_NAME.lower() not in nm.lower())
                if opps:
                    tip += f" · {opps}"
        except Exception:
            pass
        tip_safe = tip.replace('"', "&quot;").replace("'", "&#39;")
        spans += (
            f'<span title="{tip_safe}" onclick="fc(\'card_{ep_id}\')" '
            f'style="cursor:pointer;font-size:1.1rem;user-select:none">{icon}</span> '
        )
    n_eps = len(sub_eps)
    h = max(1, n_eps // 34 + 1) * 34
    html = (
        f'<style>body{{margin:0;background:transparent;font-family:sans-serif}}</style>'
        f'<div style="line-height:1.8;word-wrap:break-word">'
        f'<span style="font-size:.95rem;color:#fff;font-weight:500;margin-right:6px">Snapshots ({n_eps}):</span>'
        f'{spans}'
        f'</div>'
        f'<script>'
        f'function fc(id){{'
        f'try{{'
        f'var a=window.parent.document.getElementById(id);'
        f'if(!a)return;'
        f'a.scrollIntoView({{behavior:"smooth",block:"center"}});'
        f'var md=a.closest(\'[data-testid="stMarkdown"]\');'
        f'var card=md?md.nextElementSibling:null;'
        f'if(!card)return;'
        f'card.animate(['
        f'{{boxShadow:"0 0 0 0 rgba(245,133,24,0)"}},'
        f'{{boxShadow:"0 0 0 8px rgba(245,133,24,.8)"}},'
        f'{{boxShadow:"0 0 0 0 rgba(245,133,24,0)"}},'
        f'{{boxShadow:"0 0 0 8px rgba(245,133,24,.8)"}},'
        f'{{boxShadow:"0 0 0 0 rgba(245,133,24,0)"}}'
        f'],{{duration:1400,easing:"ease-in-out"}});'
        f'}}catch(e){{}}'
        f'}}'
        f'</script>'
    )
    components.html(html, height=h, scrolling=False)


def _fmt_card_time(create_time) -> str:
    """Parse UTC timestamp from DB and return as local 12-hour string."""
    raw = str(create_time or "")
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M"):
        try:
            dt_utc = datetime.strptime(raw[:19], fmt).replace(tzinfo=timezone.utc)
            return dt_utc.astimezone().strftime("%b %-d %-I:%M %p")
        except ValueError:
            continue
    return raw[:16]


def _render_episode_card(ep, col):
    ep_id   = str(ep["episode_id"])
    place   = ep.get("our_placement")
    nplayer = ep.get("num_players", 2)
    ep_meta = dict(ep)

    with col:
        with st.container(border=True):
            badge_col, date_col = st.columns([3, 2])
            df, names = load_replay_full(ep_id)
            total_steps = int(df["step"].max()) + 1 if not df.empty else None
            steps_str = f" · {total_steps} steps" if total_steps else ""
            badge_col.caption(f"{_placement_badge(place, nplayer)}{steps_str}")
            date_col.markdown(
                f'<div style="text-align:right;font-size:0.8rem;color:#888">'
                f'{_fmt_card_time(ep.get("create_time",""))}</div>',
                unsafe_allow_html=True,
            )
            if not df.empty:
                name_map = {n: ("You" if OUR_NAME.lower() in n.lower() else n[:10])
                            for n in names}
                df["player"] = df["player"].map(name_map)
                players = ["You"] + [v for v in name_map.values() if v != "You"]
                others  = [p for p in players if p != "You"]
                domain  = ["You"] + others
                colors  = [YOU_COLOR] + OTHER_COLORS[:len(others)]
                color_enc = alt.Color("player:N",
                                      scale=alt.Scale(domain=domain, range=colors),
                                      legend=alt.Legend(title=None, orient="bottom",
                                                        labelLimit=120, labelFontSize=12,
                                                        symbolSize=80, padding=0, rowPadding=0,
                                                        offset=4))
                # Wide-format for shared tooltip (one row per step)
                df_wide = (df.pivot(index="step", columns="player", values="planet_count")
                             .reset_index())
                df_wide.columns.name = None
                tip_players = ["You"] + [c for c in df_wide.columns if c not in ("step", "You")]
                tip_fields  = [alt.Tooltip("step:Q", title="Step")] + [
                    alt.Tooltip(f"{p}:Q", title=p) for p in tip_players if p in df_wide.columns
                ]

                # Click selection → opens Episode Detail modal
                sel = alt.selection_point(name="click", on="click", nearest=True, clear="dblclick")
                # Hover selection → drives crosshair (separate from click)
                hover_sel = alt.selection_point(
                    name="hover", on="mouseover", nearest=True,
                    clear="mouseout", fields=["step"], empty=False,
                )
                base = alt.Chart(df)
                line = base.mark_line(strokeWidth=1.5).encode(
                    x=alt.X("step:Q", axis=None),
                    y=alt.Y("planet_count:Q", title=None,
                            axis=alt.Axis(tickCount=3, labelFontSize=10,
                                          gridColor="#333", gridOpacity=0.6)),
                    color=color_enc,
                )
                # Invisible points for click (modal) — long-format data
                click_pts = base.mark_point(opacity=0, size=300).encode(
                    x="step:Q", y="planet_count:Q",
                    color=alt.Color("player:N",
                                    scale=alt.Scale(domain=domain, range=colors), legend=None),
                    tooltip=alt.value(None),
                ).add_params(sel)
                # Hover capture uses y=value(0) — avoids conflicting with line's y-axis config
                wide_base = alt.Chart(df_wide)
                hover_pts = wide_base.mark_point(opacity=0, size=300).encode(
                    x="step:Q",
                    y=alt.value(0),
                    tooltip=alt.value(None),
                ).add_params(hover_sel)
                crosshair = wide_base.mark_rule(color="#ffffff", strokeWidth=1).encode(
                    x="step:Q",
                    opacity=alt.condition(hover_sel, alt.value(0.3), alt.value(0)),
                    tooltip=tip_fields,
                )
                chart = alt.layer(line, click_pts, hover_pts, crosshair).properties(height=200)
                event = st.altair_chart(chart, width="stretch", on_select="rerun",
                                        key=f"chart_{ep_id}")
                ev_sel = event.selection if event else {}
                if ev_sel.get("click"):
                    if not st.session_state.get(f"_mopen_{ep_id}"):
                        st.session_state[f"_mopen_{ep_id}"] = True
                        episode_detail_modal(ep_id, ep_meta)
                else:
                    st.session_state.pop(f"_mopen_{ep_id}", None)
            else:
                st.caption("no replay")


def _render_card_grid(sub_eps: pd.DataFrame, cards_per_row: int = 4):
    if sub_eps.empty:
        st.caption("No episodes with placements downloaded yet.")
        return
    eps_list = list(sub_eps.iterrows())
    for i in range(0, len(eps_list), cards_per_row):
        chunk = eps_list[i:i + cards_per_row]
        cols  = st.columns(cards_per_row)
        for j, (_, ep) in enumerate(chunk):
            ep_id = str(ep["episode_id"])
            with cols[j]:
                # Scroll anchor targeted by form-strip bubble clicks
                st.markdown(
                    f'<div id="card_{ep_id}" style="height:0;margin:0;padding:0;overflow:hidden"></div>',
                    unsafe_allow_html=True,
                )
            _render_episode_card(ep, cols[j])


def _bot_summary_strip(sub_id: str, sub_eps: pd.DataFrame, public_score):
    from datetime import timedelta
    twop  = sub_eps[sub_eps["num_players"] == 2]
    fourp = sub_eps[sub_eps["num_players"] == 4]

    wins2   = int((twop["our_placement"] == 1).sum())
    total2  = len(twop)
    losses2 = total2 - wins2
    avg4    = fourp["our_placement"].mean() if not fourp.empty else None

    since24 = (datetime.now(timezone.utc) - timedelta(hours=24)).isoformat()
    episodes_all = load_episodes()
    recent_count = 0
    if not episodes_all.empty:
        sub_all = episodes_all[episodes_all["submission_id"] == sub_id]
        recent_count = int((sub_all["create_time"] > since24).sum())

    # Build sub-line HTML for 2P
    if total2:
        twop_sub = (
            f'<span style="color:#22c55e;font-weight:600">{wins2}W</span>'
            f'<span style="color:#666"> · </span>'
            f'<span style="color:#ef4444;font-weight:600">{losses2}L</span>'
            f'<span style="color:#666;font-size:.78rem"> ({total2} games)</span>'
        )
    else:
        twop_sub = ""

    # Build sub-line HTML for 4P (full placement breakdown)
    if not fourp.empty:
        _medal = {1: "🥇", 2: "🥈", 3: "🥉"}
        p_counts = fourp["our_placement"].value_counts().sort_index()
        parts = " · ".join(
            f'{int(c)}×{_medal.get(int(p), "💀")}' for p, c in p_counts.items()
        )
        fourp_sub = f'<span style="color:#aaa;font-size:.8rem">{parts} ({len(fourp)} games)</span>'
    else:
        fourp_sub = ""

    def _m(label, value, sub=""):
        return (
            f'<div style="padding:0.25rem 0 0.5rem">'
            f'<div style="font-size:.75rem;color:#999;font-weight:500">{label}</div>'
            f'<div style="font-size:1.75rem;font-weight:700;line-height:1.15;margin-top:1px">{value}</div>'
            f'<div style="margin-top:2px;line-height:1.4">{sub}</div>'
            f'</div>'
        )

    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(_m("Official score", f"{public_score:.1f}" if public_score else "—"), unsafe_allow_html=True)
    c2.markdown(_m("2P win rate", f"{wins2/total2*100:.0f}%" if total2 else "—", twop_sub), unsafe_allow_html=True)
    c3.markdown(_m("4P avg placement", f"{avg4:.2f}" if avg4 else "—", fourp_sub), unsafe_allow_html=True)
    c4.markdown(_m("Games / 24h", str(recent_count)), unsafe_allow_html=True)


def section_episodes():
    subs     = load_submissions()
    episodes = load_episodes()

    active  = subs[subs["status"] == "active"]
    retired = subs[subs["status"] != "active"]

    st.subheader("Active Agents")
    for _, sub in active.iterrows():
        sub_id  = str(sub["submission_id"])
        name    = sub.get("name", sub_id)
        pub_score = sub.get("public_score")

        sub_eps = (episodes[episodes["submission_id"] == sub_id]
                   .loc[lambda df: df["our_placement"].notna()]
                   .sort_values("create_time", ascending=True)
                   if not episodes.empty else pd.DataFrame())

        desc = read_agent_description(name)

        with st.expander(f"**{name}**  ·  {len(sub_eps)} episodes", expanded=True):
            if not sub_eps.empty:
                _bot_summary_strip(sub_id, sub_eps, pub_score)
                _render_form_strip(sub_eps)
            if desc:
                flat = " ".join(desc.split())
                if len(flat) > 120:
                    blurb = flat[:120].rsplit(" ", 1)[0] + " …"
                    label = f"Description — {blurb}"
                else:
                    label = f"Description — {flat}"
                with st.expander(label, expanded=False):
                    st.code(desc, language=None)

            _render_card_grid(sub_eps)

    if not retired.empty:
        st.subheader("Retired Agents")
        for _, sub in retired.iterrows():
            sub_id  = str(sub["submission_id"])
            name    = sub.get("name", sub_id)
            pub_score = sub.get("public_score")

            sub_eps = (episodes[episodes["submission_id"] == sub_id]
                       .loc[lambda df: df["our_placement"].notna()]
                       .sort_values("create_time", ascending=True)
                       if not episodes.empty else pd.DataFrame())

            with st.expander(f"**{name}**  ·  {len(sub_eps)} episodes", expanded=True):
                if not sub_eps.empty:
                    _bot_summary_strip(sub_id, sub_eps, pub_score)
                _render_card_grid(sub_eps)


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

def main():
    section_header()
    st.divider()
    section_position_and_leaderboard()
    st.divider()
    section_episodes()


if __name__ == "__main__":
    main()
