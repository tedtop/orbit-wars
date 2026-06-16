# Orbit Wars — Overnight Run Readout

_Live log. Driver appends timestamped tuning rows at the bottom._

## ☀️ MORNING SUMMARY (read this first)

**TL;DR — submit `comet_reaper` unchanged.** It's our best bot and is **at/above the entire public field**
(ties the real Producer 14–14, beats the other top public bots ~67%). The five experimental bots are all
**≈parity or worse** once a measurement bias was removed. The real value delivered is the *infrastructure*
+ an *insight*, not a new champion. Nothing is committed.

### Built overnight (all in the working tree)
- `arena.py`: folder-bot discovery (4-line additive change) — reused as the harness, no new one.
- **5 engine-layer bots** (forks of comet_reaper; one idea each; all logic in `main.py`; env-tunable;
  crash-guarded): `precog` (opponent-response modeling), `kingmaker` (4P leader-weighting), `maestro`
  (standing-aware roi_threshold), `helmsman` (deeper horizon), `oracle` (recovered-objective target bias).
- `rl/objective_recovery.py` + `training/objective_weights.json` — Track A (learn what top teams target).
- `pipeline/overnight.py` + `pipeline/_eval_worker.py` — self-looping knob-tuning + self-play driver.

### Honest results — seat-swapped win% vs comet_reaper (50% = parity)
| bot | best knob | clean win% vs comet_reaper | verdict |
|-----|-----------|----------------------------|---------|
| maestro | gain 0.25 / 0.5 | **50% / 52%** (25–25, 26–24; n=60 each) | parity |
| precog | strength 0–0.5 | ~50% (noisy 43–57%) | parity |
| kingmaker | lw 0.1 (4P) | ~parity | parity |
| helmsman | horizon ×1.0 | parity (×>1.0 **regresses**) | idea fails |
| oracle | bias 3 | **~17%** | **regresses** |

**`comet_reaper` is at/above the entire public field** (seat-swapped, 30g each): **ties** the real
Producer `the-producer-v2` **14–14**, and **beats** `floor-matched` **19–9**, `i-m-stronger` **19–9**,
`1266-elo-v44` **20–10** (~67–68%). So comet_reaper is a strong submit — it doesn't just match the top
public bot, it beats most of the field.

### What worked / didn't / the lesson
- **Worked:** the infra (arena folder bots, tuning driver, objective-recovery) and a real **insight** —
  top teams target **enemy ≫ neutral, near-dead enemies, nearby, higher-production** planets.
- **Didn't:** every bolt-on heuristic is parity-or-worse. **The orbit_lite engine is a tight, well-tuned
  local optimum**, and it *already* encodes the recovered preferences (its zero-sum score prefers enemy
  captures), so adding more regresses (oracle). The neural BC/self-play track is a **dead end** (0–16 vs
  the engine).
- **Methodology catch (important):** 2P has a **seat-0 advantage**; my first driver eval didn't seat-swap,
  which falsely inflated several bots to ~61%. Fixed `_eval_worker.py` to seat-swap; the parity numbers
  above are clean.

### ✅ SUBMIT RECOMMENDATION
**Ship `comet_reaper`** (current best, ties the Producer). No experiment beats it cleanly — don't risk a
regression. Keep the five forks as a tunable bench.

### Highest-EV next steps (where real gains likely are)
1. **Multi-knob config search (CMA-ES/Optuna)** over the engine's ~20 knobs vs a fixed gauntlet — the top
   teams have tuned for months; the driver only sweeps one knob at a time, so this is the realistic path.
2. **4P meta-play** done properly (the prize is in 4P; kingmaker's single lever was too crude) — rework
   the competitive scorer, validate with seat-rotated 4P.
3. To make learning viable: **add the engine (comet_reaper) as a self-play league opponent** (selfplay.py
   currently only loads PlanetPolicy opponents, so the policy never trains against real strength).

---

## What got built
- **arena.py**: `discover_bots()` now also finds folder bots (`agents/*/main.py`) — additive,
  backward-compatible. comet_reaper + the four forks are addressable by name.
- **Four bots** (each a fork of comet_reaper, one change, **all logic in `main.py`** so the shared
  `sys.modules["orbit_lite"]` collision can't corrupt A/B; each gated by an **env-tunable knob**,
  knob=0 ⇒ comet_reaper, each wrapped so it can never crash):
  - `precog` — opponent-response modeling. Predicts each rival's launches (cheap `safe_drain` proxy),
    scores my waves vs a *moving* opponent: `score(mine+opp) − score(opp)`. Env `PRECOG_OPP_STRENGTH`.
  - `kingmaker` — 4P leader-weighting. Reweights the equal-weight opponent sum in the competitive score
    to hit the leader / let trailers bleed. Env `KINGMAKER_LEADER_WEIGHT`, `KINGMAKER_BLEED`.
  - `maestro` — standing-aware `roi_threshold` (ahead ⇒ bank lead, behind ⇒ press). Env `MAESTRO_GAIN`.
  - `helmsman` — deeper lookahead via a longer projection horizon. Env `HELMSMAN_HORIZON_MULT`.
- **pipeline/overnight.py** — self-looping driver: keeps self-play alive + sweeps each bot's knob vs
  comet_reaper in parallel; appends results here.

## Key empirical finding (matters)
The orbit_lite engine (comet_reaper) is a **tight local optimum**. Every naive structural addition
**regresses at its default strength**, and the best knob value found so far is ≈0 (i.e. ≈comet_reaper):
- precog vs comet_reaper (2P, 30g): strength 0.0 → **50%** (parity ✓), 0.25 → 43%, 1.0 → 40%.
- kingmaker focal-4p vs 3×comet_reaper (16g, seat-0): default lw=0.35 → 25% vs comet_reaper-focal 75%
  (note strong seat-0 advantage inflates both; the relative drop is the signal). Regresses.
- maestro / helmsman: built; being swept by the driver.

**Implication:** beating the engine with hand-designed heuristics is hard — it's already well-tuned.
The realistic wins are (a) finding any non-regressing knob sweet spots (driver is searching), and
(b) **objective-recovery (Track A)** — fitting the scorer to what the top teams actually do, from the
168k prize-zone decisions. That's the highest-EV remaining work and is next.

## Running overnight
- **Self-play** (`rl/selfplay.py`): neural PlanetPolicy, warm-started from `bc_prizezone_v2.pt`, league =
  BC clones + self snapshots. Training fine (checkpoints to `training/selfplay_overnight.pt`). Caveat: it
  wins ~100% vs its own league (weak opponents) — it is NOT yet tested vs the engine; likely weaker than
  comet_reaper (BC ceiling). Needs the engine added as a league opponent to push it (TODO).
- **Knob tuning** (`pipeline/overnight.py`): sweeps precog/maestro/helmsman/kingmaker knobs, logged below.

## Nothing committed
All changes are in the working tree, uncommitted (commits need your approval). Diff: `git status`.

---
## Driver log
[2026-06-15 15:07:03] === overnight driver START ===
[2026-06-15 15:18:50] maestro   MAESTRO_GAIN [2P vs comet_reaper] 0.0=43%  0.1=46%  0.25=61%  0.5=61% | best=0.25 (61%, n=28)  <-- BEATS control
[2026-06-15 15:27:46] helmsman  HELMSMAN_HORIZON_MULT [2P vs comet_reaper] 1.0=64%  1.15=46%  1.3=32%  1.5=14% | best=1.0 (64%, n=28)  <-- BEATS control

### Supervision tick — 15:36
- **maestro WINS (preliminary):** MAESTRO_GAIN 0.25→61%, 0.5→61% vs comet_reaper (n=28). 0.25 is the
  default, so maestro looks like a genuine improvement. Confirming with 60 games (overnight/maestro_confirm.txt).
- **helmsman idea FAILS:** longer horizon regresses (mult 1.3→32%, 1.5→14%); best=1.0=comet_reaper.
  Fixed helmsman default 1.4→1.0 (parity). The deeper-horizon hypothesis is wrong — H=18 is well-tuned.
- **precog:** parity at strength 0, regresses above (40-43%). kingmaker: regresses at 4P default. Both ≈
  comet_reaper at best — the engine resists naive heuristics.
- **comet_reaper vs the REAL Producer (the-producer-v2): 11–11 parity** — our baseline is at the
  top-public-bot level (expected: same engine + our v4 hardening).
- **Objective-recovery (Track A) BUILT + validated** (`rl/objective_recovery.py`): recovered what top
  teams value in a target — **nearby, high-production, enemy-owned, weakly-defended** (dist −0.60,
  tgt_prod +0.52, enemy +0.49, tgt_ships −0.20). Top-1 target prediction **25% vs 18% nearest vs 13%
  random** → learned real preferences beyond proximity. Final 20k-sample run + weights saving in progress.
- **self-play:** still 100% vs its weak neural league; BC-policy-vs-comet_reaper eval running (expect the
  neural policy << engine — the BC ceiling). To make self-play useful it needs the engine as a league
  opponent (selfplay.py change) — flagged for a later tick.
- **Net:** the one promising hand-built bot is **maestro**; the one promising data path is
  **objective-recovery** (next: bias the engine's target shortlist by the recovered weights → a 5th bot).
[2026-06-15 15:45:52] kingmaker KINGMAKER_LEADER_WEIGHT [focal-4P vs 3x comet_reaper] 0.0=25%  0.1=29%  0.25=11%  0.5=7% | best=0.1 (29%, n=28)
[2026-06-15 15:56:45] precog    PRECOG_OPP_STRENGTH [2P vs comet_reaper] 0.0=54%  0.1=50%  0.25=57%  0.5=50% | best=0.25 (57%, n=28)  <-- BEATS control
[2026-06-15 15:56:47] self-play: launched chunk (init=selfplay_overnight.pt)
[2026-06-15 16:07:58] maestro   MAESTRO_GAIN [2P vs comet_reaper] 0.0=50%  0.1=50%  0.25=50%  0.5=71% | best=0.5 (71%, n=28)  <-- BEATS control
[2026-06-15 16:09:31] === overnight driver START ===

### Supervision tick — 16:05
- **Objective-recovery FINAL (20k decisions, 14,983 examples):** top teams target **enemy ≫ neutral**
  (+0.58/−0.58), **near-dead enemies** (+0.37), **outer/rotating** planets (+0.47), **nearby** (−0.43),
  higher production (+0.17). Top-1 **21.4% vs 15.8% nearest vs 12.8% random** — real, interpretable
  signal. Weights saved to `training/objective_weights.json`.
- **NEW bot `oracle`** built from that: biases candidate scores toward enemy / near-dead / high-prod
  targets (env `ORACLE_BIAS`, 0 ⇒ comet_reaper, crash-guarded). Runs clean; sweep in progress; added to
  the driver rotation.
- **Self-play / neural BC is a DEAD END vs the engine: BC policy lost 0–16 to comet_reaper.** The neural
  track can't approach the engine (BC ceiling ≪ orbit_lite). Self-play keeps running per request but
  will not produce an engine-beating bot; the real levers are the engine-layer bots + objective-recovery.
- **Correction (kingmaker):** the 4P seat-0 baseline for 4 identical bots is ~25% (symmetric), not the
  75% I mis-measured earlier from 16 seeds. So kingmaker ≈ parity (lw 0.1→29%, noisy), not a regression.
- **Status of the 5 bots vs comet_reaper:** maestro = most promising (~61% in the sweep, needs clean
  confirmation — earlier confirm got starved by core over-subscription, now fixed); precog ≈ parity
  (noisy 43–57%); kingmaker ≈ parity (4P); helmsman = parity only at default (deeper horizon fails);
  oracle = measuring. **Honest takeaway so far: the engine is hard to beat; maestro is the one real
  candidate, and it needs more games to be sure.**
[2026-06-15 16:21:34] maestro   MAESTRO_GAIN [2P vs comet_reaper] 0.0=43%  0.1=46%  0.25=61%  0.5=61% | best=0.25 (61%, n=28)  <-- BEATS control
[2026-06-15 16:30:36] helmsman  HELMSMAN_HORIZON_MULT [2P vs comet_reaper] 1.0=64%  1.15=46%  1.3=32%  1.5=14% | best=1.0 (64%, n=28)  <-- BEATS control

### Supervision tick — 16:36
- **METHODOLOGY FIX (important):** the driver's eval worker always put the candidate in **seat 0**, and
  2P Orbit Wars has a **seat-0 advantage** — visible because "parity" knobs (helmsman 1.0, oracle 0, both
  = comet_reaper) scored 61-64%, not 50%. So all driver "BEATS control" flags were **inflated ~10-14%**.
  Fixed `pipeline/_eval_worker.py` to **alternate the focal's seat** in 2P. Re-measuring maestro cleanly
  with seat-swapped arena (overnight/maestro_clean.txt) — driver paused for clean numbers, relaunches next tick.
- **oracle FAILS:** ORACLE_BIAS 0→61% (=comet_reaper, seat-inflated), 3→**17%**. The recovered enemy/
  near-dead bias *regresses* — the engine's zero-sum competitive score already prefers enemy captures, so
  adding more over-aggresses. oracle default stays 0 (=comet_reaper). (Objective-recovery still gave real
  insight; it just doesn't improve an already-enemy-aware engine when bolted on naively.)
- **maestro:** 3 driver sweeps had 0.5 ≈ 61-71%, but that's seat-inflated; the clean seat-swapped run will
  say if it's real. **Tempered expectation:** after the seat correction, maestro may also be ≈parity.
- **Reality check:** with the seat-0 bias removed, it's likely that NO bolt-on beats comet_reaper — the
  engine is genuinely strong (and ties the real Producer 11-11). If so, the honest recommendation is to
  **submit comet_reaper** (already our best), with maestro as a candidate only if the clean run confirms.
[2026-06-15 17:04:13] === overnight driver START ===

### Supervision tick — 17:04
- **Clean seat-swapped baseline:** maestro gain=0.0 (=comet_reaper) → **22–22 (exactly 50%)** over 50
  games. Confirms the seat-swap fix removes the bias (the earlier 61% was the seat-0 artifact).
- Launched a batched **final eval** (driver paused, self-play continues) → `overnight/final_eval.txt`:
  (a) maestro gain 0.25 & 0.5 vs comet_reaper, 60 games seat-swapped — the decisive maestro test;
  (b) comet_reaper vs the 4 real public bots (the-producer-v2, floor-matched, i-m-stronger, 1266-elo-v44),
  30 games each — the "big tournament" of our champion vs the public field.
- Next tick writes the MORNING SUMMARY + submit recommendation from these numbers.
[2026-06-15 17:30:37] === overnight driver START ===
[2026-06-15 17:33:09] maestro   MAESTRO_GAIN [2P vs comet_reaper] 0.0=64%  0.1=46%  0.25=61%  0.5=46% | best=0.0 (64%, n=28)  <-- BEATS control
[2026-06-15 17:35:11] helmsman  HELMSMAN_HORIZON_MULT [2P vs comet_reaper] 1.0=50%  1.15=46%  1.3=29%  1.5=11% | best=1.0 (50%, n=28)
[2026-06-15 17:39:05] kingmaker KINGMAKER_LEADER_WEIGHT [focal-4P vs 3x comet_reaper] 0.0=25%  0.1=29%  0.25=11%  0.5=7% | best=0.1 (29%, n=28)
[2026-06-15 17:41:28] oracle    ORACLE_BIAS [2P vs comet_reaper] 0.0=46%  2.0=54%  5.0=25%  10.0=14% | best=2.0 (54%, n=28)  <-- BEATS control
[2026-06-15 17:43:54] precog    PRECOG_OPP_STRENGTH [2P vs comet_reaper] 0.0=50%  0.1=54%  0.25=50%  0.5=54% | best=0.1 (54%, n=28)  <-- BEATS control
[2026-06-15 17:46:35] maestro   MAESTRO_GAIN [2P vs comet_reaper] 0.0=36%  0.1=54%  0.25=36%  0.5=54% | best=0.1 (54%, n=28)  <-- BEATS control
[2026-06-15 17:48:47] helmsman  HELMSMAN_HORIZON_MULT [2P vs comet_reaper] 1.0=57%  1.15=39%  1.3=32%  1.5=25% | best=1.0 (57%, n=28)  <-- BEATS control
[2026-06-15 17:52:36] kingmaker KINGMAKER_LEADER_WEIGHT [focal-4P vs 3x comet_reaper] 0.0=29%  0.1=32%  0.25=21%  0.5=14% | best=0.1 (32%, n=28)
[2026-06-15 17:54:41] oracle    ORACLE_BIAS [2P vs comet_reaper] 0.0=50%  2.0=36%  5.0=21%  10.0=21% | best=0.0 (50%, n=28)
[2026-06-15 17:57:00] precog    PRECOG_OPP_STRENGTH [2P vs comet_reaper] 0.0=54%  0.1=46%  0.25=54%  0.5=43% | best=0.0 (54%, n=28)  <-- BEATS control
[2026-06-15 17:59:19] maestro   MAESTRO_GAIN [2P vs comet_reaper] 0.0=46%  0.1=61%  0.25=46%  0.5=36% | best=0.1 (61%, n=28)  <-- BEATS control
[2026-06-15 17:59:48] heartbeat — driver+selfplay alive; final_eval DONE (comet_reaper ties Producer 14-14, beats floor-matched/i-m-stronger/1266-v44 ~67-68%; maestro 0.5=52% parity); load ~10.6, no cruft. Core work complete; monitoring.
[2026-06-15 18:01:14] helmsman  HELMSMAN_HORIZON_MULT [2P vs comet_reaper] 1.0=43%  1.15=25%  1.3=39%  1.5=18% | best=1.0 (43%, n=28)
[2026-06-15 18:04:50] kingmaker KINGMAKER_LEADER_WEIGHT [focal-4P vs 3x comet_reaper] 0.0=21%  0.1=21%  0.25=14%  0.5=21% | best=0.0 (21%, n=28)
[2026-06-15 18:06:49] oracle    ORACLE_BIAS [2P vs comet_reaper] 0.0=61%  2.0=46%  5.0=29%  10.0=36% | best=0.0 (61%, n=28)  <-- BEATS control
[2026-06-15 18:09:06] precog    PRECOG_OPP_STRENGTH [2P vs comet_reaper] 0.0=50%  0.1=61%  0.25=46%  0.5=36% | best=0.1 (61%, n=28)  <-- BEATS control
[2026-06-15 18:11:28] maestro   MAESTRO_GAIN [2P vs comet_reaper] 0.0=54%  0.1=36%  0.25=54%  0.5=54% | best=0.0 (54%, n=28)  <-- BEATS control
[2026-06-15 18:11:31] self-play: launched chunk (init=selfplay_overnight.pt)
[2026-06-15 18:13:28] helmsman  HELMSMAN_HORIZON_MULT [2P vs comet_reaper] 1.0=50%  1.15=29%  1.3=25%  1.5=11% | best=1.0 (50%, n=28)
[2026-06-15 18:17:06] kingmaker KINGMAKER_LEADER_WEIGHT [focal-4P vs 3x comet_reaper] 0.0=32%  0.1=18%  0.25=21%  0.5=11% | best=0.0 (32%, n=28)
[2026-06-15 18:19:02] oracle    ORACLE_BIAS [2P vs comet_reaper] 0.0=36%  2.0=54%  5.0=29%  10.0=39% | best=2.0 (54%, n=28)  <-- BEATS control
[2026-06-15 18:21:22] precog    PRECOG_OPP_STRENGTH [2P vs comet_reaper] 0.0=50%  0.1=50%  0.25=50%  0.5=43% | best=0.0 (50%, n=28)
[2026-06-15 18:23:46] maestro   MAESTRO_GAIN [2P vs comet_reaper] 0.0=61%  0.1=46%  0.25=36%  0.5=54% | best=0.0 (61%, n=28)  <-- BEATS control
[2026-06-15 18:26:01] helmsman  HELMSMAN_HORIZON_MULT [2P vs comet_reaper] 1.0=50%  1.15=46%  1.3=21%  1.5=25% | best=1.0 (50%, n=28)
[2026-06-15 18:26:32] heartbeat — driver+selfplay alive; load 9.6, no cruft. Seat-fair driver rows confirm all bots ≈parity (noisy n=28; clean 60g numbers in summary are authoritative). No action; awaiting user.
[2026-06-15 18:29:32] kingmaker KINGMAKER_LEADER_WEIGHT [focal-4P vs 3x comet_reaper] 0.0=39%  0.1=14%  0.25=18%  0.5=18% | best=0.0 (39%, n=28)
[2026-06-15 18:31:28] oracle    ORACLE_BIAS [2P vs comet_reaper] 0.0=64%  2.0=36%  5.0=18%  10.0=11% | best=0.0 (64%, n=28)  <-- BEATS control
[2026-06-15 18:34:01] precog    PRECOG_OPP_STRENGTH [2P vs comet_reaper] 0.0=64%  0.1=50%  0.25=39%  0.5=39% | best=0.0 (64%, n=28)  <-- BEATS control
[2026-06-15 18:36:35] maestro   MAESTRO_GAIN [2P vs comet_reaper] 0.0=36%  0.1=54%  0.25=54%  0.5=46% | best=0.1 (54%, n=28)  <-- BEATS control
[2026-06-15 18:38:39] helmsman  HELMSMAN_HORIZON_MULT [2P vs comet_reaper] 1.0=61%  1.15=36%  1.3=39%  1.5=39% | best=1.0 (61%, n=28)  <-- BEATS control
[2026-06-15 18:42:20] kingmaker KINGMAKER_LEADER_WEIGHT [focal-4P vs 3x comet_reaper] 0.0=21%  0.1=25%  0.25=7%  0.5=32% | best=0.5 (32%, n=28)
[2026-06-15 18:44:13] oracle    ORACLE_BIAS [2P vs comet_reaper] 0.0=50%  2.0=43%  5.0=25%  10.0=11% | best=0.0 (50%, n=28)
[2026-06-15 18:46:34] precog    PRECOG_OPP_STRENGTH [2P vs comet_reaper] 0.0=54%  0.1=61%  0.25=29%  0.5=32% | best=0.1 (61%, n=28)  <-- BEATS control
[2026-06-15 18:48:58] maestro   MAESTRO_GAIN [2P vs comet_reaper] 0.0=46%  0.1=36%  0.25=54%  0.5=46% | best=0.25 (54%, n=28)  <-- BEATS control
[2026-06-15 18:50:56] helmsman  HELMSMAN_HORIZON_MULT [2P vs comet_reaper] 1.0=50%  1.15=32%  1.3=29%  1.5=25% | best=1.0 (50%, n=28)
[2026-06-15 18:52:15] heartbeat — driver+selfplay alive; load 11.20; no cruft; all bots still ≈parity. Awaiting user.
[2026-06-15 18:54:02] kingmaker KINGMAKER_LEADER_WEIGHT [focal-4P vs 3x comet_reaper] 0.0=21%  0.1=32%  0.25=21%  0.5=32% | best=0.1 (32%, n=28)
[2026-06-15 18:55:57] oracle    ORACLE_BIAS [2P vs comet_reaper] 0.0=39%  2.0=46%  5.0=21%  10.0=29% | best=2.0 (46%, n=28)
[2026-06-15 18:58:49] precog    PRECOG_OPP_STRENGTH [2P vs comet_reaper] 0.0=50%  0.1=39%  0.25=43%  0.5=36% | best=0.0 (50%, n=28)
[2026-06-15 19:01:22] maestro   MAESTRO_GAIN [2P vs comet_reaper] 0.0=54%  0.1=54%  0.25=46%  0.5=68% | best=0.5 (68%, n=28)  <-- BEATS control
[2026-06-15 19:03:39] helmsman  HELMSMAN_HORIZON_MULT [2P vs comet_reaper] 1.0=57%  1.15=43%  1.3=21%  1.5=21% | best=1.0 (57%, n=28)  <-- BEATS control
[2026-06-15 19:07:31] kingmaker KINGMAKER_LEADER_WEIGHT [focal-4P vs 3x comet_reaper] 0.0=29%  0.1=7%  0.25=29%  0.5=29% | best=0.0 (29%, n=28)
[2026-06-15 19:09:37] oracle    ORACLE_BIAS [2P vs comet_reaper] 0.0=54%  2.0=36%  5.0=21%  10.0=18% | best=0.0 (54%, n=28)  <-- BEATS control
[2026-06-15 19:12:06] precog    PRECOG_OPP_STRENGTH [2P vs comet_reaper] 0.0=68%  0.1=32%  0.25=43%  0.5=39% | best=0.0 (68%, n=28)  <-- BEATS control
[2026-06-15 19:14:30] maestro   MAESTRO_GAIN [2P vs comet_reaper] 0.0=36%  0.1=54%  0.25=46%  0.5=50% | best=0.1 (54%, n=28)  <-- BEATS control
[2026-06-15 19:16:24] helmsman  HELMSMAN_HORIZON_MULT [2P vs comet_reaper] 1.0=61%  1.15=29%  1.3=25%  1.5=4% | best=1.0 (61%, n=28)  <-- BEATS control
[2026-06-15 19:16:26] self-play: launched chunk (init=selfplay_overnight.pt)
[2026-06-15 19:18:38] heartbeat — jobs alive; user awake; holding for direction (stop loops / CMA-ES tuning / prep submission).
[2026-06-15 19:19:41] kingmaker KINGMAKER_LEADER_WEIGHT [focal-4P vs 3x comet_reaper] 0.0=25%  0.1=29%  0.25=32%  0.5=29% | best=0.25 (32%, n=28)
[2026-06-15 19:21:42] oracle    ORACLE_BIAS [2P vs comet_reaper] 0.0=46%  2.0=43%  5.0=18%  10.0=18% | best=0.0 (46%, n=28)
[2026-06-15 19:24:29] precog    PRECOG_OPP_STRENGTH [2P vs comet_reaper] 0.0=36%  0.1=57%  0.25=57%  0.5=46% | best=0.1 (57%, n=28)  <-- BEATS control
[2026-06-15 19:26:52] maestro   MAESTRO_GAIN [2P vs comet_reaper] 0.0=54%  0.1=46%  0.25=71%  0.5=71% | best=0.25 (71%, n=28)  <-- BEATS control
[2026-06-15 19:28:59] helmsman  HELMSMAN_HORIZON_MULT [2P vs comet_reaper] 1.0=43%  1.15=46%  1.3=29%  1.5=25% | best=1.15 (46%, n=28)
[2026-06-15 19:32:50] kingmaker KINGMAKER_LEADER_WEIGHT [focal-4P vs 3x comet_reaper] 0.0=18%  0.1=25%  0.25=32%  0.5=21% | best=0.25 (32%, n=28)
[2026-06-15 19:34:42] oracle    ORACLE_BIAS [2P vs comet_reaper] 0.0=36%  2.0=29%  5.0=18%  10.0=25% | best=0.0 (36%, n=28)
[2026-06-15 19:37:01] precog    PRECOG_OPP_STRENGTH [2P vs comet_reaper] 0.0=25%  0.1=50%  0.25=46%  0.5=36% | best=0.1 (50%, n=28)
[2026-06-15 19:39:34] maestro   MAESTRO_GAIN [2P vs comet_reaper] 0.0=54%  0.1=46%  0.25=50%  0.5=43% | best=0.0 (54%, n=28)  <-- BEATS control
[2026-06-15 19:41:56] helmsman  HELMSMAN_HORIZON_MULT [2P vs comet_reaper] 1.0=29%  1.15=46%  1.3=18%  1.5=21% | best=1.15 (46%, n=28)
[2026-06-15 19:44:30] auto-monitoring loop PAUSED — user awake and directing. Background driver+selfplay still running but low-value; awaiting user call (analyze live replays / stop jobs / tune).
[2026-06-15 19:45:43] kingmaker KINGMAKER_LEADER_WEIGHT [focal-4P vs 3x comet_reaper] 0.0=36%  0.1=39%  0.25=25%  0.5=36% | best=0.1 (39%, n=28)
[2026-06-15 19:47:51] oracle    ORACLE_BIAS [2P vs comet_reaper] 0.0=61%  2.0=50%  5.0=29%  10.0=14% | best=0.0 (61%, n=28)  <-- BEATS control
[2026-06-15 19:50:25] precog    PRECOG_OPP_STRENGTH [2P vs comet_reaper] 0.0=57%  0.1=57%  0.25=29%  0.5=57% | best=0.0 (57%, n=28)  <-- BEATS control
[2026-06-15 19:52:44] maestro   MAESTRO_GAIN [2P vs comet_reaper] 0.0=46%  0.1=71%  0.25=71%  0.5=43% | best=0.1 (71%, n=28)  <-- BEATS control
[2026-06-15 19:54:28] helmsman  HELMSMAN_HORIZON_MULT [2P vs comet_reaper] 1.0=57%  1.15=43%  1.3=39%  1.5=11% | best=1.0 (57%, n=28)  <-- BEATS control
[2026-06-15 19:58:14] kingmaker KINGMAKER_LEADER_WEIGHT [focal-4P vs 3x comet_reaper] 0.0=29%  0.1=29%  0.25=29%  0.5=32% | best=0.5 (32%, n=28)
[2026-06-15 20:00:00] oracle    ORACLE_BIAS [2P vs comet_reaper] 0.0=46%  2.0=36%  5.0=21%  10.0=21% | best=0.0 (46%, n=28)
[2026-06-15 20:02:26] precog    PRECOG_OPP_STRENGTH [2P vs comet_reaper] 0.0=64%  0.1=43%  0.25=50%  0.5=29% | best=0.0 (64%, n=28)  <-- BEATS control
[2026-06-15 20:04:46] maestro   MAESTRO_GAIN [2P vs comet_reaper] 0.0=46%  0.1=50%  0.25=43%  0.5=46% | best=0.1 (50%, n=28)
[2026-06-15 20:06:42] helmsman  HELMSMAN_HORIZON_MULT [2P vs comet_reaper] 1.0=57%  1.15=32%  1.3=36%  1.5=21% | best=1.0 (57%, n=28)  <-- BEATS control
[2026-06-15 20:10:16] kingmaker KINGMAKER_LEADER_WEIGHT [focal-4P vs 3x comet_reaper] 0.0=29%  0.1=29%  0.25=39%  0.5=25% | best=0.25 (39%, n=28)
[2026-06-15 20:12:19] oracle    ORACLE_BIAS [2P vs comet_reaper] 0.0=57%  2.0=29%  5.0=18%  10.0=11% | best=0.0 (57%, n=28)  <-- BEATS control
[2026-06-15 20:14:37] precog    PRECOG_OPP_STRENGTH [2P vs comet_reaper] 0.0=57%  0.1=46%  0.25=61%  0.5=50% | best=0.25 (61%, n=28)  <-- BEATS control
[2026-06-15 20:16:51] maestro   MAESTRO_GAIN [2P vs comet_reaper] 0.0=71%  0.1=71%  0.25=43%  0.5=39% | best=0.0 (71%, n=28)  <-- BEATS control
[2026-06-15 20:19:02] helmsman  HELMSMAN_HORIZON_MULT [2P vs comet_reaper] 1.0=46%  1.15=43%  1.3=21%  1.5=14% | best=1.0 (46%, n=28)
[2026-06-15 20:22:05] kingmaker KINGMAKER_LEADER_WEIGHT [focal-4P vs 3x comet_reaper] 0.0=32%  0.1=25%  0.25=39%  0.5=32% | best=0.25 (39%, n=28)
[2026-06-15 20:22:07] self-play: launched chunk (init=selfplay_overnight.pt)
[2026-06-15 20:24:23] oracle    ORACLE_BIAS [2P vs comet_reaper] 0.0=29%  2.0=57%  5.0=21%  10.0=18% | best=2.0 (57%, n=28)  <-- BEATS control
[2026-06-15 20:27:10] precog    PRECOG_OPP_STRENGTH [2P vs comet_reaper] 0.0=39%  0.1=36%  0.25=39%  0.5=25% | best=0.0 (39%, n=28)
[2026-06-15 20:29:29] maestro   MAESTRO_GAIN [2P vs comet_reaper] 0.0=50%  0.1=43%  0.25=46%  0.5=39% | best=0.0 (50%, n=28)
[2026-06-15 20:31:22] helmsman  HELMSMAN_HORIZON_MULT [2P vs comet_reaper] 1.0=54%  1.15=36%  1.3=39%  1.5=25% | best=1.0 (54%, n=28)  <-- BEATS control
[2026-06-15 20:34:54] kingmaker KINGMAKER_LEADER_WEIGHT [focal-4P vs 3x comet_reaper] 0.0=36%  0.1=29%  0.25=29%  0.5=14% | best=0.0 (36%, n=28)
[2026-06-15 20:37:05] oracle    ORACLE_BIAS [2P vs comet_reaper] 0.0=36%  2.0=36%  5.0=11%  10.0=11% | best=0.0 (36%, n=28)
[2026-06-15 20:39:38] precog    PRECOG_OPP_STRENGTH [2P vs comet_reaper] 0.0=54%  0.1=61%  0.25=54%  0.5=36% | best=0.1 (61%, n=28)  <-- BEATS control
[2026-06-15 20:42:05] maestro   MAESTRO_GAIN [2P vs comet_reaper] 0.0=71%  0.1=43%  0.25=39%  0.5=36% | best=0.0 (71%, n=28)  <-- BEATS control
[2026-06-15 20:43:59] helmsman  HELMSMAN_HORIZON_MULT [2P vs comet_reaper] 1.0=71%  1.15=46%  1.3=32%  1.5=7% | best=1.0 (71%, n=28)  <-- BEATS control
[2026-06-15 20:47:13] kingmaker KINGMAKER_LEADER_WEIGHT [focal-4P vs 3x comet_reaper] 0.0=32%  0.1=43%  0.25=36%  0.5=29% | best=0.1 (43%, n=28)
[2026-06-15 21:13:35] oracle    ORACLE_BIAS [2P vs comet_reaper] 0.0=54%  2.0=61%  5.0=21%  10.0=14% | best=2.0 (61%, n=28)  <-- BEATS control
[2026-06-15 21:16:20] precog    PRECOG_OPP_STRENGTH [2P vs comet_reaper] 0.0=32%  0.1=57%  0.25=36%  0.5=39% | best=0.1 (57%, n=28)  <-- BEATS control
[2026-06-15 21:18:47] maestro   MAESTRO_GAIN [2P vs comet_reaper] 0.0=43%  0.1=46%  0.25=39%  0.5=43% | best=0.1 (46%, n=28)
[2026-06-15 21:21:06] helmsman  HELMSMAN_HORIZON_MULT [2P vs comet_reaper] 1.0=46%  1.15=43%  1.3=32%  1.5=21% | best=1.0 (46%, n=28)
[2026-06-15 21:24:56] kingmaker KINGMAKER_LEADER_WEIGHT [focal-4P vs 3x comet_reaper] 0.0=32%  0.1=18%  0.25=21%  0.5=14% | best=0.0 (32%, n=28)
[2026-06-15 21:26:52] oracle    ORACLE_BIAS [2P vs comet_reaper] 0.0=57%  2.0=39%  5.0=14%  10.0=11% | best=0.0 (57%, n=28)  <-- BEATS control
[2026-06-15 21:29:27] precog    PRECOG_OPP_STRENGTH [2P vs comet_reaper] 0.0=61%  0.1=39%  0.25=39%  0.5=36% | best=0.0 (61%, n=28)  <-- BEATS control
[2026-06-15 21:31:54] maestro   MAESTRO_GAIN [2P vs comet_reaper] 0.0=43%  0.1=39%  0.25=36%  0.5=46% | best=0.5 (46%, n=28)
[2026-06-15 21:33:59] helmsman  HELMSMAN_HORIZON_MULT [2P vs comet_reaper] 1.0=57%  1.15=57%  1.3=25%  1.5=14% | best=1.0 (57%, n=28)  <-- BEATS control
[2026-06-15 21:37:36] kingmaker KINGMAKER_LEADER_WEIGHT [focal-4P vs 3x comet_reaper] 0.0=36%  0.1=36%  0.25=21%  0.5=25% | best=0.0 (36%, n=28)
[2026-06-15 21:39:35] oracle    ORACLE_BIAS [2P vs comet_reaper] 0.0=54%  2.0=68%  5.0=29%  10.0=14% | best=2.0 (68%, n=28)  <-- BEATS control
[2026-06-15 21:42:06] precog    PRECOG_OPP_STRENGTH [2P vs comet_reaper] 0.0=61%  0.1=36%  0.25=36%  0.5=39% | best=0.0 (61%, n=28)  <-- BEATS control
[2026-06-15 21:44:28] maestro   MAESTRO_GAIN [2P vs comet_reaper] 0.0=46%  0.1=39%  0.25=43%  0.5=36% | best=0.0 (46%, n=28)
[2026-06-15 21:46:34] helmsman  HELMSMAN_HORIZON_MULT [2P vs comet_reaper] 1.0=36%  1.15=39%  1.3=46%  1.5=11% | best=1.3 (46%, n=28)
[2026-06-15 21:46:36] self-play: launched chunk (init=selfplay_overnight.pt)
[2026-06-15 21:50:14] kingmaker KINGMAKER_LEADER_WEIGHT [focal-4P vs 3x comet_reaper] 0.0=29%  0.1=25%  0.25=29%  0.5=25% | best=0.0 (29%, n=28)
[2026-06-15 21:52:08] oracle    ORACLE_BIAS [2P vs comet_reaper] 0.0=46%  2.0=57%  5.0=11%  10.0=25% | best=2.0 (57%, n=28)  <-- BEATS control
[2026-06-15 21:54:46] precog    PRECOG_OPP_STRENGTH [2P vs comet_reaper] 0.0=43%  0.1=39%  0.25=46%  0.5=36% | best=0.25 (46%, n=28)
[2026-06-15 21:57:15] maestro   MAESTRO_GAIN [2P vs comet_reaper] 0.0=39%  0.1=36%  0.25=39%  0.5=39% | best=0.0 (39%, n=28)
[2026-06-15 21:59:19] helmsman  HELMSMAN_HORIZON_MULT [2P vs comet_reaper] 1.0=57%  1.15=32%  1.3=46%  1.5=14% | best=1.0 (57%, n=28)  <-- BEATS control
[2026-06-15 22:02:53] kingmaker KINGMAKER_LEADER_WEIGHT [focal-4P vs 3x comet_reaper] 0.0=36%  0.1=18%  0.25=18%  0.5=21% | best=0.0 (36%, n=28)
[2026-06-15 22:04:50] oracle    ORACLE_BIAS [2P vs comet_reaper] 0.0=75%  2.0=54%  5.0=29%  10.0=0% | best=0.0 (75%, n=28)  <-- BEATS control
[2026-06-15 22:07:25] precog    PRECOG_OPP_STRENGTH [2P vs comet_reaper] 0.0=36%  0.1=54%  0.25=36%  0.5=36% | best=0.1 (54%, n=28)  <-- BEATS control
[2026-06-15 22:09:52] maestro   MAESTRO_GAIN [2P vs comet_reaper] 0.0=39%  0.1=43%  0.25=36%  0.5=50% | best=0.5 (50%, n=28)
[2026-06-15 22:12:02] helmsman  HELMSMAN_HORIZON_MULT [2P vs comet_reaper] 1.0=43%  1.15=36%  1.3=21%  1.5=14% | best=1.0 (43%, n=28)
[2026-06-15 22:15:44] kingmaker KINGMAKER_LEADER_WEIGHT [focal-4P vs 3x comet_reaper] 0.0=29%  0.1=25%  0.25=25%  0.5=18% | best=0.0 (29%, n=28)
