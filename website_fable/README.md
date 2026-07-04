# orbit wars. — the Montana Schmeekler campaign site

A single-page presentation website telling the full story of competing in the
[Kaggle Orbit Wars](https://www.kaggle.com/competitions/orbit-wars/overview)
competition — 23 bots, 19 experiments, the orbit_lite engine discovery, five
failed RL campaigns, and the lessons that came out the other side.

Built with **Next.js 16** (App Router, fully static output), **Tailwind CSS v4**,
and zero chart/animation libraries — the orbital hero, replay theater, and all
charts are hand-rolled canvas/SVG.

## Sections

| Anchor | What it shows |
|---|---|
| hero | Live orbital canvas — sun, orbit trails, fleets in transit |
| `#game` | Competition rules with an animated board schematic |
| `#scientists` | The 23 phase-0 bots, one per scientific field |
| `#engine` | Reverse-engineering The Producer / orbit_lite → comet_reaper |
| `#climb` | 712 leaderboard snapshots: us vs the prize line vs the leader |
| `#lab` | 19-experiment KEEP/DISCARD ledger + the zero-choice discovery |
| `#rl` | The RL moonshot: 0% hero number, plateau + entropy-collapse charts |
| `#theater` | Canvas playback of six real ranked games from episode logs |
| `#lessons` | Seven lessons, paid in full |
| `#finale` | Final standings, full 24-phase mission log, Kaggle links |

## Development

```bash
npm install
npm run dev        # http://localhost:3000
npm run build      # static production build
npm run lint
```

## Data pipeline

All numbers on the site come from the repo's raw mission data. To regenerate:

```bash
python3 scripts/build_data.py
```

This reads `../leaderboards/*.csv` (712 pipeline snapshots) and
`../website/public/data/` (experiments, timeline, scientists, RL metrics, raw
replays) and writes:

- `src/data/*.json` — small datasets imported at build time
  (race, experiments, timeline, scientists, submissions, final leaderboard, replay manifest)
- `public/data/rl_runs.json` — curated PPO training curves (fetched client-side)
- `public/data/replays/*.json` — six curated episodes compacted from ~15 MB
  each to ~500 KB (per-step planet/fleet/comet arrays only)

The generated files are committed, so deploys don't need the raw data.

## Deploying to Vercel

No configuration required — it's a stock Next.js app:

```bash
npx vercel        # or import the repo in the Vercel dashboard,
                  # set the project root to website_fable/
```

Everything is prerendered static; `public/data/` is served from the CDN.

## QA scripts

`scripts/qa_shots.mjs` and `scripts/qa_theater.mjs` drive headless Chrome
(via `puppeteer-core`, using the system Chrome binary) to screenshot every
section at desktop and mobile widths against a local `next start` server on
port 3199.
