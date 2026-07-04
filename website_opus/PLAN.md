# Orbit Wars Portfolio Website — Implementation Plan

## Context

Ted ran a 7-day Kaggle AI competition ("Orbit Wars") building and iterating bots that compete in a 2/4-player orbital-mechanics strategy game. The project produced rich, documentary-quality data: 491 leaderboard snapshots every 15 minutes, 511 actual game replay JSONs, 125 RL training logs across a 9-instance HPC cluster, 19 catalogued experiments with win-rate measurements, and a detailed narrative TIMELINE.md. The goal is to turn all of this into a beautiful, interactive Next.js portfolio piece that tells the story end-to-end — both the technical substance AND the meta-narrative of autonomous AI development. Lives in `orbit_wars/website/`.

---

## Tech Stack

| Layer | Choice | Why |
|---|---|---|
| Framework | Next.js 16 (App Router, static export) | Portfolio-deployable to Vercel or GitHub Pages |
| Language | TypeScript | |
| Styling | Tailwind CSS + CSS variables for theming | |
| UI kit | shadcn/ui + Radix primitives | |
| Charts | **Recharts** (line/area/bar/scatter) | Easiest React integration |
| Game canvas | **D3.js** (SVG-based orbital replay) | Precise control over planet/fleet physics |
| Animations | **Framer Motion** (scroll-triggered reveals) | |
| Icons | Lucide React | |
| Fonts | Space Grotesk (display) + Inter (body) + JetBrains Mono (code/stats) | |

**No backend.** All data is static JSON, pre-built by `scripts/build_data.py` at site-build time.

---

## Visual Design

**Theme:** Deep space dark — `#08080f` background, glassmorphism cards (`bg-white/5 backdrop-blur border-white/10`), animated CSS starfield.

**Accent palette:**
- Electric blue `#3b82f6` — primary / bot highlight
- Amber `#f59e0b` — comet_reaper / Elo history
- Emerald `#10b981` — wins / KEEP experiments
- Rose `#ef4444` — losses / DISCARD
- Purple `#a855f7` — prize zone reference line
- Slate `#94a3b8` — neutral / text

**Typography:** Headings use Space Grotesk weight 700. Numbers/stats use JetBrains Mono. Body uses Inter.

---

## File Structure

```
orbit_wars/website/
├── scripts/
│   └── build_data.py          # pre-processing: CSV → JSON, log → JSON, timeline parse
├── public/
│   └── data/
│       ├── score_history.json         # 491-point Elo time-series (our row from each CSV)
│       ├── timeline.json              # parsed TIMELINE.md → structured phase objects
│       ├── experiments.json           # 19 experiments with win%, delta, verdict
│       ├── rl_runs.json               # PPO training curves (clip_frac, entropy, eval_wr)
│       ├── replays_index.json         # metadata for all 511 replays (teams, winner, length)
│       └── replays/                   # copy of interesting game replay JSONs (~50 curated)
├── src/
│   ├── app/
│   │   ├── layout.tsx                 # font loading, metadata, starfield wrapper
│   │   ├── page.tsx                   # single-page composition of all sections
│   │   └── globals.css
│   ├── components/
│   │   ├── sections/                  # one file per scroll section (see below)
│   │   ├── charts/                    # Recharts + D3 chart components
│   │   └── ui/                        # StarfieldBg, MetricBadge, PhaseChip, etc.
│   ├── data/
│   │   └── types.ts                   # shared TypeScript interfaces
│   └── lib/
│       └── utils.ts                   # cn(), formatScore(), etc.
├── package.json
├── tailwind.config.ts
└── next.config.ts                     # output: 'export' for static deploy
```

---

## Sections (top → bottom, single-page scroll)

### 1. Hero
- Animated CSS starfield background
- Large title: "Orbit Wars" with orbital-path SVG decorations
- Tagline: "7 days. 19 experiments. 491 leaderboard snapshots. One AI battle."
- 4 headline stats: Final Rank, Final Elo, # Experiments, # Games Played
- CTA: "See the journey ↓" scroll button
- Submission date range chip

### 2. The 23 Scientists ⭐ (Night One)
- The origin story: before any Kaggle data, we invented 23 bots from scratch — each applying a different scientific field to an orbital strategy game
- Visual: a 5×5ish grid of "scientist cards", one per bot
  - Each card: bot name, discipline (e.g. "Epidemiology · SIR Model", "Economics · Markowitz Portfolio"), a 1-sentence description
  - Colored by field cluster (physical sciences / life sciences / math/CS / economics)
- Big stat callout: "10 hours 10 minutes. 23 bots. One arena."
- OpenSkill round-robin arena results: bar chart of final ratings after the overnight run
- Winner callout: `markowitz_portfolio_optimization` (economics) beat every physics bot
- Transition: "Then we looked at the actual leaderboard. None of our 23 would have cracked the top 400."
- Data source: `timeline.json` Phase 0 + bot descriptions from `archive/agents/`

### 3. The Game (what is Orbit Wars?)
- Brief explanation of the game: planets orbit a sun, send fleets to capture, 2P/4P
- Two-column: text on left, Kaggle competition link + embed preview on right
- Key mechanics callout boxes (orbital mechanics, fleet timing, production economy)
- Link to play in-browser on Kaggle

### 4. Journey Timeline
- Full-width vertical timeline, Phase 0 through Plan B
- Each phase is a card with: phase label, date, 1-line headline, expandable body
- Status chips: "Submitted" / "Dead End" / "Breakthrough" / "Infrastructure"
- Key ones get icon treatments: 🔬 experiment, 🚀 submission, 💀 dead end, ✨ breakthrough
- Data source: `timeline.json` (parsed from TIMELINE.md)

### 5. Score Progression ⭐ (flagship chart)
- Full-width animated area chart: Elo score over 6 days (491 points)
- Colored by active submission (amber = comet_reaper, blue = schmeekler, green = schmeekler_fmt)
- Vertical dashed lines for each submission event with labels
- Reference lines: "Prize zone ~1500" (purple) and "comet_reaper best 1234" (amber)
- Toggle: Score view / Rank view (rank chart below, inverted Y)
- Range pills: All / 3d / 24h
- Data source: `score_history.json`

### 6. Bot Lineage + Experiment Lab ⭐
- Left panel: SVG bot lineage tree (orbit_lite → comet_reaper → 5 forks → schmeekler → schmeekler forks)
  - Green nodes = KEEP, red = DISCARD, amber = running
  - Hover reveals win% and finding
- Right panel: Recharts scatter plot (experiment # × Δwin% vs baseline)
  - Color = verdict, shape = track
  - Click expands a detail drawer with the full experiment finding
- Summary KPIs row: "19 experiments · 3 KEEP · 17% hit rate"
- Data source: `experiments.json`

### 7. The Breakthrough
- Dedicated section for the schmeekler discovery (the one experiment that worked)
- Story format: "We reverse-engineered the #1 engine. Then we found its blind spot."
- Code snippet showing the static_target_bonus addition (syntax-highlighted)
- Small chart: Win rate sweep across static_bonus values (0 → 2.0), showing the 1.0–1.5 sweet spot
- Result callout: "72% vs comet_reaper · #144 on live leaderboard"

### 8. The RL Attempt (v6)
- Story: "We tried to train a neural net to replace the engine. It failed."
- Multi-line Recharts chart: entropy decay + clip_frac + eval_win_rate over PPO updates
  - Best seed trajectory highlighted
  - Pre-committed gate (U=500) shown as vertical line
  - "20% ceiling" annotation
- Fleet stats: 9 Jetstream2 instances, 32 seeds, ~1,312 SPS
- Failure analysis callout: "Policy converged to a local optimum. Not still exploring — committed to something bad."
- Data source: `rl_runs.json`

### 9. Game Replay Gallery ⭐
- Grid of replay cards (6–8 curated highlights)
- Each card: opponent name, result (W/L), game length, a mini sparkline of planet counts
- Click → modal with the **orbital canvas visualizer**:
  - SVG canvas showing planets as circles sized by production, colored by owner
  - Planets orbit the sun (use actual position data from replay JSON)
  - Fleets shown as moving dots along straight-line paths
  - Playback controls: play/pause, speed (1×/5×/10×), step slider
  - Planet ownership timeline chart below canvas
- Data source: `replays/[id].json` + `replays_index.json`
- Kaggle link: "Watch on Kaggle" button for each episode (Kaggle replay URL pattern)

### 10. Autonomous AI Development
- The meta-narrative: "This project was built with Claude Code running autonomously overnight"
- Screenshot gallery (lightbox) from `Progress Screenshots/` folder
- Key stats: Autonomous sessions, longest unattended run, experiments auto-run
- Quote from AUDITOR_LOG.md: the independent auditor catching the CF=0 bug
- ORCHESTRATOR_STATE.md excerpts showing structured session logs
- Stat: "Bugs found during RL: 8 critical bugs in GAE, entropy scaling, eval logic"

### 11. Learnings & Final Standings
- Numbered lessons learned (from TIMELINE.md closing entries)
- Final result: Elo / Rank / Prize zone gap
- "What I'd do differently" 3-bullet callout:
  1. Wrap kaggle-environments in a Gym interface from day one → CleanRL handles the PPO algorithm, Tensorboard handles visualization, custom work is only obs encoding + JAX engine
  2. Lock the visualization stack before writing any training code — Tensorboard is 3 lines; rebuilding dashboards session-by-session with an LLM produces 3 inconsistent systems
  3. Explicitly seed every run and log the seed — JAX requires explicit PRNGKey diversity; CPU gets it free from OS entropy but you get no reproducibility; always pass --seed
- Links: GitHub repo (post-competition), Kaggle competition page
- Personal/contact footer

**Lesson 07 content (tooling/observability) captured in LessonsSection.tsx** — the core narrative: LLM is stateless on visualization, generates different output each session, leads to tmux boards + Streamlit + inline tables all coexisting. Standard tools (Tensorboard, CleanRL, MLFlow) have persistent consistent behavior the LLM can't replicate.

---

## Data Build Script (`scripts/build_data.py`)

Seven outputs, run once before `next build`:

1. **`score_history.json`** — iterate all 491 leaderboard CSVs, find "Montana Schmeekler" row, emit `[{time, rank, score, prize_score, bot_name}]`. Bot name assigned by matching timestamps to submission events.

2. **`timeline.json`** — parse TIMELINE.md headings + body into `[{phase, date, title, headline, body_md, status}]` objects.

3. **`experiments.json`** — hardcode the `_EXPERIMENTS` list from `dashboard/app.py` as JSON (it's already structured there with epoch, name, verdict, win_pct, delta, category, note).

4. **`rl_runs.json`** — parse up to 125 log files from `agents/rl_ppo/runs/`, extract per-run `[{run, update, steps, clip_frac, explained_variance, entropy, eval_wr}]` time series. Use the same regex already in `dashboard/app.py:_parse_log_file()`.

5. **`replays_index.json`** — for each of the 511 replay JSONs, read just the metadata (episode ID, TeamNames, winner from final rewards, step count, date bucket) → lightweight index.

6. **`replays/[id].json`** — copy ~50 curated replays to `public/data/replays/` (selection criteria: games where we won against strong opponents, games we lost badly, the longest games).

7. **`screenshots_index.json`** — list all files in `Progress Screenshots/` → copy to `public/screenshots/` for the gallery section.

---

## Key Reusable Patterns from Existing Code

- **Score history loading logic** from `dashboard/app.py:load_rank_history()` — directly ports to Python build script
- **RL log parser regex** from `dashboard/app.py:_parse_log_file()` — identical regex in build script
- **Experiment list** from `dashboard/app.py:_EXPERIMENTS` — export verbatim as JSON
- **Bot color palette** (`_BOT_PALETTE`) — reuse same hex values in Tailwind/CSS vars
- **Replay JSON structure** — `steps[i][0].observation.{planets, fleets}` with `[id, owner, x, y, angular_vel, garrison, prod]` for planets

---

## Orbital Canvas Visualizer (D3)

Planet positions: each planet has `[id, owner, x, y, angular_velocity, garrison, production]`. The `x,y` values are absolute coordinates that advance each step based on `angular_velocity`. We can recompute positions per step client-side or pre-bake them.

Rendering approach (SVG not Canvas — easier to style with CSS):
- Sun at center, drawn as yellow glow circle
- Planets: circles scaled by `production`, colored by owner (blue/red/neutral gray/comet orange)
- Garrison count: text label inside planet
- Fleets: small triangles moving linearly from source planet to destination
- Step slider drives which frame of `steps[]` to display
- Smooth interpolation between steps using Framer Motion

---

## Build & Deploy

```bash
# After competition ends and screenshots are added:
cd orbit_wars/website
python3 ../scripts/build_data.py   # build all static JSON
npm install
npm run build                      # next build + static export → out/
# Deploy: push out/ to Vercel or GitHub Pages
```

`next.config.ts`: `output: 'export'` for static — no server needed.

---

## Implementation Order

1. **Scaffold**: `create-next-app` (Next.js 16), install deps, Tailwind config, font setup, starfield component
2. **Data pipeline**: `build_data.py` → generate all 7 JSON outputs (including 23-bot arena ratings from DB if available, else hardcode from archive README)
3. **Score Progression chart** — the flagship; validates data pipeline is correct
4. **Journey Timeline** — content-heavy, no complex interactivity
5. **The 23 Scientists section** — bot grid cards + OpenSkill bar chart
6. **Experiment Lab** — scatter + lineage tree
7. **Hero section** — polish after content sections are working
8. **Orbital canvas visualizer** — most complex; build last with a subset of replays
9. **Replay gallery** — wraps the canvas in cards
10. **RL deep dive** — chart assembly from `rl_runs.json`
11. **Autonomous AI / Screenshots** — gallery + prose (screenshots added post-competition)
12. **Final Standings + Footer** — once competition result is known
13. **Polish pass**: Framer Motion scroll reveals, mobile responsiveness, OG image

---

## Open Questions (fill in after June 23)

- Final rank/Elo once the leaderboard locks
- Kaggle replay URL pattern for public linking (verify once competition goes public)
- Which screenshots to feature in the Autonomous AI section
- Personal/contact info to put in footer
