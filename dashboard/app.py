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

    # ── Top row: position metrics (left) + leaderboard (right) ───────────────
    m_col, lb_col = st.columns([0.3, 0.7])

    with m_col:
        st.subheader("My Position")
        c1, c2 = st.columns(2)
        c1.metric("Score", f"{our_score:.1f}" if our_score else "—",
                  delta=f"{score_delta:+.1f} since last snapshot" if score_delta is not None else None)
        c2.metric("Rank", f"#{our_rank}" if our_rank else "—",
                  delta=f"{rank_delta:+d} places since last snapshot" if rank_delta is not None else None,
                  delta_color="normal")

    with lb_col:
        if lb_df is not None:
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

    # ── Full-width trend charts (below the top row) ───────────────────────────
    if hist.empty or len(hist) < 2:
        st.info("No leaderboard history yet — run the pipeline.")
        return

    _range_opts = {"All": None, "7d": 7 * 24, "3d": 3 * 24, "24h": 24, "6h": 6}
    _sel = st.pills("Range", list(_range_opts.keys()), default="All",
                    label_visibility="collapsed", key="chart_range")
    _cutoff_h = _range_opts.get(_sel or "All")
    if _cutoff_h:
        hist_view = hist[hist["time"] >= hist["time"].max() - pd.Timedelta(hours=_cutoff_h)]
    else:
        hist_view = hist

    # Submission metadata
    _subs_ann  = load_submissions()
    _ann_df    = pd.DataFrame()
    _sub_times = pd.DataFrame()
    if not _subs_ann.empty:
        _a = _subs_ann[
            _subs_ann["submitted_at"].notna() &
            (_subs_ann["submitted_at"].astype(str).str.strip() != "")
        ].copy()
        if not _a.empty:
            _a["time"]  = pd.to_datetime(_a["submitted_at"], utc=True, errors="coerce")
            _a          = _a.dropna(subset=["time"])
            _a["label"] = _a["name"].str[:22]
            _SHORT_NAMES = {
                "comet_reaper v1": "comet_reaper",
                "schmeekler@1.5":  "schmeekler",
                "schmeekler_fmt":  "schmeekler_fmt",
            }
            _a["short_name"]     = _a["name"].map(_SHORT_NAMES).fillna("")
            _a["released_label"] = _a["short_name"].apply(
                lambda s: f"{s} released" if s else ""
            )
            _ann_df = _a[["time", "name", "label", "short_name",
                           "released_label", "public_score"]].reset_index(drop=True)
        _sub_times = (
            _subs_ann[_subs_ann["submitted_at"].notna()]
            .assign(sub_time=lambda d: pd.to_datetime(d["submitted_at"], utc=True, errors="coerce"))
            .dropna(subset=["sub_time"])
            .sort_values("sub_time")
            [["sub_time", "name"]]
            .reset_index(drop=True)
        )

    # Tag each snapshot with the active submission at that time
    _hb = hist_view.copy()
    if not _sub_times.empty:
        _hb["bot_name"] = _sub_times.iloc[0]["name"]
        for _, _sr in _sub_times.iterrows():
            _hb.loc[_hb["time"] >= _sr["sub_time"], "bot_name"] = _sr["name"]
    else:
        _hb["bot_name"] = "unknown"

    # Stable color palette
    _BOT_PALETTE = {
        "comet_reaper v1":                     "#f59e0b",
        "schmeekler@1.5":                      "#4c78a8",
        "schmeekler_fmt":                      "#54a24b",
        "markowitz_portfolio_optimization v1": "#555",
        "coordinated_strike_interceptor v1":   "#555",
    }
    _hb_names  = sorted(_hb["bot_name"].unique().tolist())
    _c_domain  = _hb_names
    _c_range   = [_BOT_PALETTE.get(n, "#888") for n in _hb_names]
    _color_enc = alt.Color(
        "bot_name:N",
        scale=alt.Scale(domain=_c_domain, range=_c_range),
        legend=alt.Legend(title=None, orient="bottom", labelFontSize=11,
                          symbolSize=120, padding=4, rowPadding=3),
    )
    _x_enc = alt.X("time:T", title=None,
                   axis=alt.Axis(format="%m/%d %H:%M", labelFontSize=10))

    # Colored vertical submission rules + "{bot} released" label at top
    def _sub_rules():
        if _ann_df.empty or hist_view.empty:
            return []
        _lo  = hist_view["time"].min()
        _hi  = hist_view["time"].max()
        _vis = _ann_df[(_ann_df["time"] >= _lo) & (_ann_df["time"] <= _hi)].copy()
        if _vis.empty:
            return []
        _rd  = _vis["name"].tolist()
        _rr  = [_BOT_PALETTE.get(n, "#888") for n in _rd]
        _ce  = alt.Color("name:N", scale=alt.Scale(domain=_rd, range=_rr), legend=None)
        rules = (
            alt.Chart(_vis)
            .mark_rule(strokeDash=[6, 4], strokeWidth=2, opacity=0.75)
            .encode(
                x=alt.X("time:T"),
                color=_ce,
                tooltip=[
                    alt.Tooltip("time:T",         format="%m/%d %H:%M", title="Submitted"),
                    alt.Tooltip("name:N",         title="Bot"),
                    alt.Tooltip("public_score:Q", format=".1f",         title="Score at sub"),
                ],
            )
        )
        # Horizontal "{bot} released" label anchored to top of chart
        _vis_lbl = _vis[_vis["released_label"] != ""] if "released_label" in _vis.columns else _vis.iloc[:0]
        if _vis_lbl.empty:
            return [rules]
        labels = (
            alt.Chart(_vis_lbl)
            .mark_text(angle=0, align="left", baseline="top",
                       fontSize=9, fontWeight="bold", dx=5, dy=3)
            .encode(
                x=alt.X("time:T"),
                y=alt.value(10),
                text=alt.Text("released_label:N"),
                color=_ce,
            )
        )
        return [rules, labels]

    # Reference horizontals: CR best + prize zone
    _ref_df = pd.DataFrame([
        {"y": 1234.7, "ref": "CR best  1234"},
        {"y": 1500.0, "ref": "Prize zone  ~1500"},
    ])
    _ref_lines = (
        alt.Chart(_ref_df)
        .mark_rule(strokeDash=[4, 3], strokeWidth=1.5, opacity=0.5)
        .encode(
            y=alt.Y("y:Q"),
            color=alt.Color(
                "ref:N",
                scale=alt.Scale(
                    domain=["CR best  1234", "Prize zone  ~1500"],
                    range=["#f59e0b", "#a855f7"],
                ),
                legend=alt.Legend(title=None, orient="bottom", labelFontSize=10,
                                  symbolSize=80, padding=2, rowPadding=2),
            ),
            tooltip=["ref:N", alt.Tooltip("y:Q", format=".0f")],
        )
    )

    # ── Score over time: area gradient + line ───────────────────────────────
    # Explicit Y domain so the area doesn't fill to 0 (which wrecks the scale)
    _y_floor = float(_hb["score"].min()) - 40
    _y_ceil  = float(_hb["score"].max()) + 70
    _s_scale = alt.Scale(domain=[_y_floor, _y_ceil], zero=False, nice=False)

    _s_base = alt.Chart(_hb).encode(x=_x_enc, color=_color_enc)
    st.caption("Score over time — each color = one submission")
    st.altair_chart(
        alt.layer(
            # Area fill: from score line DOWN to y_floor (not to 0)
            _s_base.mark_area(opacity=0.18, interpolate="monotone").encode(
                y=alt.Y("score:Q", scale=_s_scale),
                y2=alt.Y2(datum=_y_floor),
            ),
            # Line on top
            _s_base.mark_line(strokeWidth=2.5, interpolate="monotone").encode(
                y=alt.Y("score:Q", title="Score", scale=_s_scale),
                tooltip=[alt.Tooltip("time:T", format="%m/%d %H:%M"),
                         alt.Tooltip("score:Q", format=".1f"),
                         alt.Tooltip("rank:Q"),
                         alt.Tooltip("bot_name:N", title="Bot")],
            ),
            _ref_lines,
            *_sub_rules(),
        ).properties(height=320),
        use_container_width=True,
    )

    # ── Rank over time: clean line only (area fill misbehaves on reversed scale)
    _r_base = alt.Chart(_hb).encode(x=_x_enc, color=_color_enc)
    st.caption("Rank over time — lower is better")
    st.altair_chart(
        alt.layer(
            _r_base.mark_line(strokeWidth=2.5, interpolate="monotone").encode(
                y=alt.Y("rank:Q", title="Rank  ↓ better",
                        scale=alt.Scale(reverse=True, zero=False, nice=True)),
                tooltip=[alt.Tooltip("time:T", format="%m/%d %H:%M"),
                         alt.Tooltip("rank:Q"),
                         alt.Tooltip("score:Q", format=".1f"),
                         alt.Tooltip("bot_name:N", title="Bot")],
            ),
            *_sub_rules(),
        ).properties(height=200),
        use_container_width=True,
    )

    # ── Per-bot zoom: individual local Y-scale for oscillation detail ────────
    st.caption("Per-bot convergence (local scale — shows oscillation within each bot's range)")
    _bot_names_ordered = sorted(
        _hb["bot_name"].unique().tolist(),
        key=lambda n: _hb.loc[_hb["bot_name"] == n, "score"].mean(),
        reverse=True,
    )
    _per_bot_cols = st.columns(min(len(_bot_names_ordered), 3))
    for _i, _bn in enumerate(_bot_names_ordered):
        _bd = _hb[_hb["bot_name"] == _bn].copy()
        if len(_bd) < 2:
            continue
        _bc     = _BOT_PALETTE.get(_bn, "#888")
        _bx_enc = alt.X("time:T", title=None,
                        axis=alt.Axis(format="%m/%d %H:%M", labelFontSize=9, tickCount=3))

        # Score mini-chart with local scale
        _bs_floor = float(_bd["score"].min()) - 8
        _bs_ceil  = float(_bd["score"].max()) + 12
        _bs_scale = alt.Scale(domain=[_bs_floor, _bs_ceil], zero=False, nice=False)
        _bs_base  = alt.Chart(_bd).encode(x=_bx_enc)
        _bscore   = alt.layer(
            _bs_base.mark_area(opacity=0.25, color=_bc, interpolate="monotone").encode(
                y=alt.Y("score:Q", scale=_bs_scale, title="Score"),
                y2=alt.Y2(datum=_bs_floor),
            ),
            _bs_base.mark_line(strokeWidth=2, color=_bc, interpolate="monotone").encode(
                y=alt.Y("score:Q", scale=_bs_scale),
                tooltip=[alt.Tooltip("time:T", format="%m/%d %H:%M"),
                         alt.Tooltip("score:Q", format=".1f"),
                         alt.Tooltip("rank:Q")],
            ),
        ).properties(height=110)

        # Rank mini-chart with local scale
        _br_floor = float(_bd["rank"].min()) - 2
        _br_ceil  = float(_bd["rank"].max()) + 2
        _br_scale = alt.Scale(domain=[_br_ceil, _br_floor], zero=False, nice=False)  # reversed
        _brank    = (
            alt.Chart(_bd).encode(x=_bx_enc)
            .mark_line(strokeWidth=1.5, color=_bc, interpolate="monotone", strokeDash=[])
            .encode(
                y=alt.Y("rank:Q", scale=_br_scale, title="Rank"),
                tooltip=[alt.Tooltip("time:T", format="%m/%d %H:%M"),
                         alt.Tooltip("rank:Q"),
                         alt.Tooltip("score:Q", format=".1f")],
            )
            .properties(height=70)
        )

        _cur_score = float(_bd.sort_values("time").iloc[-1]["score"])
        _cur_rank  = int(_bd.sort_values("time").iloc[-1]["rank"])
        with _per_bot_cols[_i % 3]:
            st.markdown(
                f"<span style='color:{_bc};font-weight:700;font-size:13px'>{_bn}</span>"
                f"<span style='color:#999;font-size:11px;margin-left:8px'>"
                f"score {_cur_score:.0f} · rank #{_cur_rank}</span>",
                unsafe_allow_html=True,
            )
            st.altair_chart(alt.vconcat(_bscore, _brank, spacing=4),
                            use_container_width=True)

    # ── Current score by submission (bar) ───────────────────────────────────
    if not _subs_ann.empty:
        _sub_bar = (
            _subs_ann[_subs_ann["public_score"].notna() & (_subs_ann["public_score"] > 800)]
            .sort_values("public_score", ascending=False)
            .copy()
        )
        if not _sub_bar.empty:
            _sb_dom   = _sub_bar["name"].tolist()
            _sb_range = [_BOT_PALETTE.get(n, "#888") for n in _sb_dom]
            _sb_color = alt.Color("name:N",
                                   scale=alt.Scale(domain=_sb_dom, range=_sb_range),
                                   legend=None)
            _x_max = max(1550, float(_sub_bar["public_score"].max()) + 80)
            st.caption("Current score by submission")
            st.altair_chart(
                alt.layer(
                    alt.Chart(_sub_bar).mark_bar(cornerRadiusEnd=4).encode(
                        x=alt.X("public_score:Q", title="Score",
                                scale=alt.Scale(domain=[900, _x_max])),
                        y=alt.Y("name:N", title=None, sort="-x"),
                        color=_sb_color,
                        tooltip=["name:N",
                                 alt.Tooltip("public_score:Q", format=".1f", title="Score")],
                    ),
                    alt.Chart(_sub_bar).mark_text(align="left", dx=5, fontSize=11,
                                                   fontWeight="bold").encode(
                        x=alt.X("public_score:Q"),
                        y=alt.Y("name:N", sort="-x"),
                        text=alt.Text("public_score:Q", format=".0f"),
                        color=_sb_color,
                    ),
                    alt.Chart(pd.DataFrame({"x": [1234.7], "r": ["comet_reaper"]})).mark_rule(
                        color="#f59e0b", strokeDash=[4, 3], strokeWidth=1.5, opacity=0.7
                    ).encode(x=alt.X("x:Q"), tooltip=["r:N"]),
                    alt.Chart(pd.DataFrame({"x": [1500], "r": ["Prize zone"]})).mark_rule(
                        color="#a855f7", strokeDash=[4, 3], strokeWidth=1.5, opacity=0.7
                    ).encode(x=alt.X("x:Q"), tooltip=["r:N"]),
                ).properties(height=max(70, len(_sub_bar) * 32)),
                use_container_width=True,
            )


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


def _lazy_card_grid(sub_id: str, sub_eps: pd.DataFrame,
                    auto_load: bool = False, page_size: int = 24):
    """Render episode cards with lazy loading + pagination.

    auto_load=True  → render immediately (top agent), paginated to page_size
    auto_load=False → show a load button; only render after user clicks it
    """
    load_key = f"eps_loaded_{sub_id}"
    page_key = f"eps_page_{sub_id}"

    if not auto_load and not st.session_state.get(load_key, False):
        st.button(
            f"Load {len(sub_eps)} episode charts",
            key=f"btn_load_{sub_id}",
            on_click=lambda: st.session_state.update({load_key: True}),
        )
        return

    if page_key not in st.session_state:
        st.session_state[page_key] = page_size

    shown = st.session_state[page_key]
    # sub_eps is sorted ascending; tail gives the most-recent N
    _render_card_grid(sub_eps.tail(shown))

    if len(sub_eps) > shown:
        remaining = len(sub_eps) - shown
        st.button(
            f"Show {remaining} more episodes",
            key=f"more_{sub_id}",
            on_click=lambda: st.session_state.update({page_key: shown + page_size}),
        )


def section_episodes():
    subs     = load_submissions()
    episodes = load_episodes()

    active  = subs[subs["status"] == "active"]
    retired = subs[subs["status"] != "active"]

    st.subheader("Active Agents")
    for i, (_, sub) in enumerate(active.iterrows()):
        sub_id    = str(sub["submission_id"])
        name      = sub.get("name", sub_id)
        pub_score = sub.get("public_score")
        is_top    = (i == 0)

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

            _lazy_card_grid(sub_id, sub_eps, auto_load=is_top)

    if not retired.empty:
        st.subheader("Retired Agents")
        for _, sub in retired.iterrows():
            sub_id    = str(sub["submission_id"])
            name      = sub.get("name", sub_id)
            pub_score = sub.get("public_score")

            sub_eps = (episodes[episodes["submission_id"] == sub_id]
                       .loc[lambda df: df["our_placement"].notna()]
                       .sort_values("create_time", ascending=True)
                       if not episodes.empty else pd.DataFrame())

            with st.expander(f"**{name}**  ·  {len(sub_eps)} episodes", expanded=False):
                if not sub_eps.empty:
                    _bot_summary_strip(sub_id, sub_eps, pub_score)
                _lazy_card_grid(sub_id, sub_eps, auto_load=False)


# ---------------------------------------------------------------------------
# Autoresearch
# ---------------------------------------------------------------------------

# Registry of every experiment run — update as results land.
# win_pct: "overall" win% vs public panel where measured; None = pending.
# delta:   vs schmeekler n=150 baseline (66% overall). None = pending/N/A.
_EXPERIMENTS = [
    # --- epoch 1: overnight bolt-ons (seat-swapped, all parity) ---
    dict(epoch=1, name="precog",          base="comet_reaper", track="A (pre)",  verdict="DISCARD",     win_pct=50,  delta=-16, category="Heuristic",        note="Parity vs CR seat-swapped; N≈50"),
    dict(epoch=2, name="kingmaker",        base="comet_reaper", track="A (pre)",  verdict="DISCARD",     win_pct=50,  delta=-16, category="Heuristic",        note="Parity vs CR seat-swapped"),
    dict(epoch=3, name="maestro",          base="comet_reaper", track="A (pre)",  verdict="DISCARD",     win_pct=50,  delta=-16, category="Heuristic",        note="Parity vs CR seat-swapped"),
    dict(epoch=4, name="helmsman",         base="comet_reaper", track="A (pre)",  verdict="DISCARD",     win_pct=50,  delta=-16, category="Heuristic",        note="Parity vs CR seat-swapped"),
    dict(epoch=5, name="oracle",           base="comet_reaper", track="A (pre)",  verdict="DISCARD",     win_pct=50,  delta=-16, category="Heuristic",        note="Parity vs CR seat-swapped"),
    # --- epoch 2: Optuna config tuning ---
    dict(epoch=6, name="comet_reaper_tuned", base="comet_reaper", track="Config", verdict="DISCARD",    win_pct=None, delta=None, category="Config tuning",   note="37 Optuna trials; best score 0.34 — base config is a tight optimum"),
    # --- epoch 3: schmeekler (the breakthrough) ---
    dict(epoch=7, name="schmeekler",       base="comet_reaper", track="A",        verdict="KEEP",        win_pct=74,  delta=+8,  category="Static bonus",     note="72% 2P vs CR; beats whole public panel; CHAMPION → submitted live"),
    # --- epoch 4: Track A bonus features ---
    dict(epoch=8, name="schmeekler_potential", base="schmeekler", track="A",      verdict="DISCARD",    win_pct=74,  delta=0,   category="Scoring bonus",    note="Noise ≈0pp — flow scorer already encodes ETA/position"),
    dict(epoch=9, name="schmeekler_interdict", base="schmeekler", track="A",      verdict="DISCARD",    win_pct=48,  delta=-18, category="Scoring bonus",    note="−26pp CATASTROPHIC — overrides flow scorer; wins race, loses hold"),
    dict(epoch=10, name="schmeekler_phase",   base="schmeekler", track="A",       verdict="DISCARD",    win_pct=22,  delta=-44, category="Phase sizing",     note="−52pp CATASTROPHIC — breaks ROI/floor/sizing interaction"),
    dict(epoch=11, name="schmeekler_fmt",     base="schmeekler", track="A",       verdict="KEEP",       win_pct=66,  delta=0,   category="Format-aware",     note="2P identical at n=150; +3.72μ 4P; SUBMITTED live"),
    # --- epoch 5: Track B search ---
    dict(epoch=12, name="comet_reaper_search", base="comet_reaper", track="B",    verdict="DISCARD",    win_pct=50,  delta=-16, category="Search",           note="Rollout policy reproduces the 1-ply candidate moves exactly"),
    dict(epoch=13, name="CR_mcts_v1",         base="comet_reaper", track="B",     verdict="DISCARD",    win_pct=75,  delta=+1,  category="Search",           note="n=50 parity — 0–4 candidates/turn, de-mean correction = 0"),
    dict(epoch=14, name="CR_mcts_v2",         base="comet_reaper", track="B",     verdict="DISCARD",    win_pct=75,  delta=+1,  category="Search",           note="True depth-2 + state advance; still parity — candidate scarcity confirmed"),
    # --- epoch 6: overnight (tonight) ---
    dict(epoch=15, name="schmeekler_orbit",   base="schmeekler", track="B",       verdict="DISCARD",    win_pct=61,  delta=-5,  category="Orbit timing",     note="Hold fires 0% (threshold=0.6) or 42% turns causing passivity; capture floor independent of orbit position for neutrals; enemy floors INCREASE with delay"),
    dict(epoch=16, name="schmeekler_comet",   base="schmeekler", track="A",       verdict="IN PROGRESS", win_pct=None, delta=None, category="Comet targeting", note="2×2 factorial — comet targeting bonus; evals running"),
    dict(epoch=17, name="CR_comet",           base="comet_reaper", track="A",     verdict="IN PROGRESS", win_pct=None, delta=None, category="Comet targeting", note="2×2 factorial (comet on, static off); evals running"),
    dict(epoch=18, name="CR_stochastic",      base="comet_reaper", track="Search", verdict="IN PROGRESS", win_pct=None, delta=None, category="Stochastic search", note="Boltzmann opponent model; 192 candidates; τ sweep pending"),
    dict(epoch=19, name="comet_reaper_vf",    base="comet_reaper", track="C",     verdict="IN PROGRESS", win_pct=None, delta=None, category="Value function",  note="AUC=0.99 ✅ PASS — gym arena eval vs CR pending; 19ms/turn"),
]

_VERDICT_COLOR = {"KEEP": "#22c55e", "DISCARD": "#ef4444", "IN PROGRESS": "#f59e0b"}
_TRACK_COLOR   = {"A": "#4c78a8", "A (pre)": "#6b8dbf", "B": "#72b7b2",
                  "Config": "#b279a2", "Search": "#f58518", "C": "#e45756"}


def _autoresearch_scatter() -> alt.Chart:
    """Scatter: epoch × delta-vs-baseline, colored by verdict."""
    rows = []
    for e in _EXPERIMENTS:
        rows.append({
            "epoch":   e["epoch"],
            "name":    e["name"],
            "delta":   e["delta"] if e["delta"] is not None else 0,
            "pending": e["delta"] is None,
            "verdict": e["verdict"],
            "track":   e["track"],
            "note":    e["note"],
            "category": e["category"],
            "win_pct": e["win_pct"] if e["win_pct"] is not None else 0,
        })
    df = pd.DataFrame(rows)

    color_scale = alt.Scale(
        domain=list(_VERDICT_COLOR.keys()),
        range=list(_VERDICT_COLOR.values()),
    )
    shape_scale = alt.Scale(
        domain=["KEEP", "DISCARD", "IN PROGRESS"],
        range=["circle", "cross", "square"],
    )

    base = alt.Chart(df).encode(
        x=alt.X("epoch:Q", title="Experiment #", axis=alt.Axis(tickMinStep=1)),
        y=alt.Y("delta:Q", title="Δ vs schmeekler baseline (pp)",
                scale=alt.Scale(domain=[-55, 15]),
                axis=alt.Axis(gridColor="#333")),
        color=alt.Color("verdict:N", scale=color_scale,
                        legend=alt.Legend(title="Verdict", orient="top-left")),
        shape=alt.Shape("verdict:N", scale=shape_scale),
        tooltip=[
            alt.Tooltip("epoch:Q", title="#"),
            alt.Tooltip("name:N",  title="Bot"),
            alt.Tooltip("category:N", title="Category"),
            alt.Tooltip("delta:Q", title="Δ vs baseline (pp)"),
            alt.Tooltip("win_pct:Q", title="Overall win%"),
            alt.Tooltip("track:N",  title="Track"),
            alt.Tooltip("note:N",   title="Finding"),
        ],
    )

    points = base.mark_point(size=100, strokeWidth=2, filled=True).encode(
        opacity=alt.condition(
            alt.datum.pending == True,
            alt.value(0.4),
            alt.value(0.9),
        )
    )

    labels = base.mark_text(dy=-12, fontSize=9, angle=0).encode(
        text=alt.Text("name:N"),
        opacity=alt.condition(
            alt.datum.verdict == "KEEP",
            alt.value(1.0),
            alt.value(0.0),
        )
    )

    zero_line = alt.Chart(pd.DataFrame({"y": [0]})).mark_rule(
        color="#aaa", strokeDash=[4, 4], strokeWidth=1
    ).encode(y="y:Q")

    return (
        alt.layer(zero_line, points, labels)
        .properties(height=280, title=alt.TitleParams(
            "Δ win% vs schmeekler (n=150 baseline = 66% overall)",
            fontSize=11, color="#aaa"
        ))
        .configure_view(strokeWidth=0)
    )


def _live_score_bar() -> alt.Chart:
    """Horizontal bar chart of live submission scores + prize zone reference."""
    data = pd.DataFrame([
        {"bot": "comet_reaper",  "score": 1234.7, "status": "Inactive (score preserved)", "color": "#4c78a8"},
        {"bot": "schmeekler",    "score": 1083.0, "status": "Active · converging ↑",      "color": "#22c55e"},
        {"bot": "schmeekler_fmt","score": 1040.7, "status": "Active · cold-start",         "color": "#f59e0b"},
    ])
    bars = alt.Chart(data).mark_bar(cornerRadiusEnd=4).encode(
        y=alt.Y("bot:N", sort="-x", title=None, axis=alt.Axis(labelFontSize=12)),
        x=alt.X("score:Q", title="Live Kaggle score",
                scale=alt.Scale(domain=[800, 1800]),
                axis=alt.Axis(gridColor="#333")),
        color=alt.Color("color:N", scale=None, legend=None),
        tooltip=["bot:N", alt.Tooltip("score:Q", format=".1f"), "status:N"],
    )
    prize_line = alt.Chart(pd.DataFrame({"x": [1500]})).mark_rule(
        color="#e45756", strokeDash=[5, 4], strokeWidth=1.5
    ).encode(x="x:Q")
    prize_label = alt.Chart(pd.DataFrame({"x": [1500], "y": ["schmeekler_fmt"], "label": ["🎯 Prize zone ~1500"]})).mark_text(
        align="left", dx=6, dy=-6, fontSize=10, color="#e45756"
    ).encode(x="x:Q", y=alt.Y("y:N"), text="label:N")
    producer_line = alt.Chart(pd.DataFrame({"x": [1259]})).mark_rule(
        color="#b279a2", strokeDash=[3, 3], strokeWidth=1
    ).encode(x="x:Q")
    producer_label = alt.Chart(pd.DataFrame({"x": [1259], "y": ["schmeekler"], "label": ["Producer ~1259"]})).mark_text(
        align="left", dx=6, dy=-6, fontSize=10, color="#b279a2"
    ).encode(x="x:Q", y=alt.Y("y:N"), text="label:N")
    return (
        alt.layer(bars, prize_line, prize_label, producer_line, producer_label)
        .properties(height=130, title=alt.TitleParams("Live scores (Jun 17 poll)", fontSize=11, color="#aaa"))
        .configure_view(strokeWidth=0)
    )


def _lineage_html() -> str:
    """SVG-based bot lineage tree."""
    return """
<div style="font-family:monospace;font-size:12px;line-height:1.7;color:#ccc;background:#111;
            padding:14px 16px;border-radius:8px;border:1px solid #2a2a2a;overflow-x:auto">
<div style="color:#888;font-size:10px;margin-bottom:8px;letter-spacing:1px">BOT LINEAGE</div>
<span style="color:#6b8dbf">orbit_lite engine</span><br>
└── <span style="color:#4c78a8;font-weight:600">comet_reaper</span>
    <span style="color:#22c55e;font-size:10px"> ✅ live 1234.7</span><br>
&nbsp;&nbsp;&nbsp;&nbsp;├── <span style="color:#ef4444">precog / kingmaker / maestro / helmsman / oracle</span>
    <span style="color:#555;font-size:10px"> ❌ all ≈parity</span><br>
&nbsp;&nbsp;&nbsp;&nbsp;├── <span style="color:#ef4444">comet_reaper_tuned</span>
    <span style="color:#555;font-size:10px"> ❌ Optuna 37 trials — tight optimum</span><br>
&nbsp;&nbsp;&nbsp;&nbsp;├── <span style="color:#ef4444">comet_reaper_search</span>
    <span style="color:#555;font-size:10px"> ❌ rollout reproduces 1-ply moves</span><br>
&nbsp;&nbsp;&nbsp;&nbsp;├── <span style="color:#ef4444">CR_mcts_v1 → v2</span>
    <span style="color:#555;font-size:10px"> ❌ 2-ply dead — 0–4 candidates/turn, nothing to re-rank</span><br>
&nbsp;&nbsp;&nbsp;&nbsp;├── <span style="color:#f59e0b">CR_stochastic</span>
    <span style="color:#f59e0b;font-size:10px"> 🔄 Boltzmann 2-ply, 192 cands — τ sweep running</span><br>
&nbsp;&nbsp;&nbsp;&nbsp;├── <span style="color:#f59e0b">CR_comet</span>
    <span style="color:#f59e0b;font-size:10px"> 🔄 2×2 factorial (comet on, static off)</span><br>
&nbsp;&nbsp;&nbsp;&nbsp;├── <span style="color:#e45756;font-weight:600">comet_reaper_vf</span>
    <span style="color:#22c55e;font-size:10px"> 🧠 AUC=0.99 ✅ — gym arena eval pending</span><br>
&nbsp;&nbsp;&nbsp;&nbsp;└── <span style="color:#22c55e;font-weight:600">schmeekler</span>
    <span style="color:#22c55e;font-size:10px"> ✅ CHAMPION · live 1083 ↑</span><br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├── <span style="color:#ef4444">schmeekler_potential</span>
    <span style="color:#555;font-size:10px"> ❌ ≈0pp — scorer already encodes ETA</span><br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├── <span style="color:#ef4444">schmeekler_interdict</span>
    <span style="color:#555;font-size:10px"> ❌ −26pp — overrides scorer</span><br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├── <span style="color:#ef4444">schmeekler_phase</span>
    <span style="color:#555;font-size:10px"> ❌ −52pp — breaks ROI/floor interaction</span><br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├── <span style="color:#ef4444">schmeekler_orbit</span>
    <span style="color:#555;font-size:10px"> ❌ floor independent of orbit position; delay → more expensive</span><br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├── <span style="color:#f59e0b">schmeekler_comet</span>
    <span style="color:#f59e0b;font-size:10px"> 🔄 2×2 factorial (comet on, static on)</span><br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;└── <span style="color:#22c55e">schmeekler_fmt</span>
    <span style="color:#22c55e;font-size:10px"> ✅ KEEP · live 1040 cold-start ↑</span>
</div>
"""


def section_autoresearch():
    # ── headline KPIs ──────────────────────────────────────────────────────
    keep  = sum(1 for e in _EXPERIMENTS if e["verdict"] == "KEEP")
    done  = sum(1 for e in _EXPERIMENTS if e["verdict"] != "IN PROGRESS")
    prog  = sum(1 for e in _EXPERIMENTS if e["verdict"] == "IN PROGRESS")
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Experiments done",  str(done))
    k2.metric("KEEP rate",         f"{keep}/{done}  ({keep/done*100:.0f}%)")
    k3.metric("Running overnight", str(prog))
    k4.metric("Track C AUC",       "0.99 ✅",  delta=">> 0.65 threshold")
    k5.metric("Deadline",          "6 days",   delta="Jun 23 23:59 UTC", delta_color="off")

    st.divider()

    # ── live scores + lineage ───────────────────────────────────────────────
    lcol, rcol = st.columns([0.44, 0.56])
    with lcol:
        st.caption("**Bot lineage** — every experiment forked from the base")
        st.html(_lineage_html())
    with rcol:
        st.caption("**Experiment results** — Δ win% vs schmeekler n=150 baseline (66% overall)")
        st.altair_chart(_autoresearch_scatter(), use_container_width=True)

    st.divider()

    # ── live ladder ────────────────────────────────────────────────────────
    st.caption("**Live ladder** — current Kaggle scores")
    st.altair_chart(_live_score_bar(), use_container_width=True)

    st.divider()

    # ── active track status ────────────────────────────────────────────────
    st.caption("**Active tracks — overnight status**")
    t1, t2, t3, t4 = st.columns(4)

    with t1:
        st.markdown("**🧠 Track C — ValueNet**")
        st.markdown("Phase A ✅ labeler\nPhase B ✅ encoder\nPhase D ✅ AUC=**0.99**\nPhase E 🔄 gym arena")
        st.progress(0.85, text="85% — awaiting gym rating")

    with t2:
        st.markdown("**🎲 Search — Stochastic**")
        st.markdown("Bot built ✅\nτ sweep 🔄 running\nn=30 directional ⏳\nn=150 ⏳")
        st.progress(0.25, text="25% — τ sweep in progress")

    with t3:
        st.markdown("**☄️ Comet 2×2**")
        st.markdown("schmeekler_comet ✅\nCR_comet ✅\nEvals 🔄 running\nVerdict ⏳")
        st.progress(0.40, text="40% — evals running")

    with t4:
        st.markdown("**📡 Live calibration**")
        st.markdown("schmeekler 1083 ↑\nschmeekler_fmt 1040 cold\ncomet_reaper 1234 (inactive)\nConvergence ⏳")
        st.progress(0.55, text="schmeekler converging")

    # ── mechanistic insight box ────────────────────────────────────────────
    st.divider()
    st.info(
        "**🔑 Core mechanistic finding** — The orbit_lite engine collapses to **0–4 candidates/turn** via "
        "`capture_floor`/`clears_floor` filters. This is why every shallow re-ranker (bonuses, 2-ply exact "
        "search, orbit timing) lands at parity: there is nothing to re-rank. **The only untested lever is "
        "EXPANDING the candidate set to aggressive/floor-blocked moves and judging them by learned outcome "
        "(Track C VF) or stochastic EV (stochastic search).** Track C AUC=0.99 suggests the VF has real "
        "discriminative power — gym arena will tell us if this translates to a rating gain."
    )


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

def main():
    section_header()
    st.divider()
    tab_pos, tab_eps, tab_ar = st.tabs(["📊 Position & Ladder", "📡 Episodes", "🔬 Autoresearch"])
    with tab_pos:
        section_position_and_leaderboard()
    with tab_eps:
        section_episodes()
    with tab_ar:
        section_autoresearch()


if __name__ == "__main__":
    main()
