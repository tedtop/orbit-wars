# Orbit Wars — Competition Strategy & Architecture

**Deadline:** June 23, 2026 (final submission) · games run through ~July 8
**Prize:** Top 10 each win $5,000 · prize cutoff currently ~1539 Elo
**Current field:** ~4,448 teams · leader ~1746
**Today:** June 14 — 9 days to deadline

---

## Guiding principles

1. **The Kaggle engine is ground truth.** Evaluate only against `kaggle-environments`. Never build a substitute physics engine for evaluation — divergence means optimizing for the wrong game.
2. **Submission slots are scarce.** 5/day, only latest 2 tracked. Every submission is reviewed by a human before upload. Automate data, not decisions.
3. **Don't sit stagnant, but don't thrash.** Get bots on the field tonight so σ converges. Iterate deliberately, not randomly.
4. **2-player results ≠ 4-player results.** Our local arena was all 2-player. The competition is both. Treat 4-player performance as an open question until measured.

---

## What we actually know vs. assume

| Claim | Status |
|---|---|
| markowitz & coordinated_strike are best | True **for 2-player only** — our arena had no FFA games |
| markowitz is "the best bot" | Overstated — won one format, one tournament |
| The Producer = 1259, public, copyable | Confirmed from Kaggle |
| Prize cutoff ~1539 | From leaderboard screenshot, will drift |
| Replay JSON has all players' full state/actions | Per competition spec — this is our opponent-data source |
| kaggle-environments == the real engine | Confirmed — local sim is the actual simulator |

The biggest unknown: **how do our bots do in 4-player FFA?** Resolve this first.

---

## Phase 0 — TONIGHT (submit + smoke test)

**Goal:** get two bots on the ladder so they start accumulating episodes and σ drops.

1. Smoke-test both bots locally against the real engine (2P and 4P):
   ```python
   from kaggle_environments import make
   env = make("orbit_wars", debug=True)
   env.run(["main.py", "random"])              # 2-player liveness
   env.run(["main.py", "random", "random", "random"])  # 4-player liveness
   ```
2. Add a **lightweight** logger to the agent (see logging note below). Keep it crash-safe and cheap.
3. Submit in correct order so best bot is most-recent (latest 2 are tracked):
   ```bash
   kaggle competitions submit orbit-wars -f coordinated_strike.py -m "coordinated strike v1 +logging"
   kaggle competitions submit orbit-wars -f markowitz.py -m "markowitz v1 +logging"   # most recent
   ```
4. Confirm both are live, not errored:
   ```bash
   kaggle competitions submissions orbit-wars
   ```

**Logging note:** The replay JSON already gives full state+actions for ALL players. In-bot logging is only needed for our bot's *internal* reasoning (scores, chosen mission). Keep it minimal — a few fields per turn, wrapped in try/except, never risking the 1s budget. Do not over-build this.

---

## Phase 1 — THIS WEEK (local 4-player arena + opponent benchmarking)

**Goal:** Measure our bots in 4-player FFA against real competition-level opponents.

### Archive, don't delete
Move the 21 non-submitted bots to `archive/agents/`. They stay importable for future arenas but don't clutter the active set. Keep `agents/` to our 2 + downloaded opponents.

### Download a SMALL set of public opponents (3–4, not 20)
- The Producer V2 (1259) — the strong public ceiling
- One mid-tier public bot (~1180) for variety
- `starter` and `random` are built into the engine (free)

Rationale: a tight opponent set keeps the arena fast so we can iterate. We add opponents only when they teach us something new.

### Build the 4-player arena harness (around the real engine)
This is a *harness*, not a physics engine. It wraps `make("orbit_wars")`, runs N games with seed control, rotates seat positions (FFA has positional effects), and logs outcomes. Reuse the adaptive Wilson-CI stopping from your existing `arena.py` — that logic is sound and format-agnostic.

Key 4-player design points:
- **Seat rotation:** in a 4-player game, run each unique 4-bot combo across all seat permutations (or a fixed rotation) to cancel positional bias, the same way side-swapping worked in 2P.
- **Mixed fields:** test our bot vs 3 copies of an opponent, vs mixed opponents, and in self-play.
- **Score by placement,** not just win/loss — 4-player needs 1st/2nd/3rd/4th tracking.

**Deliverable:** a ranking of our 2 bots + opponents in true FFA. *This may change which bot we favor.*

### On orbit_lite (The Producer's library)
Optional. It's public, so everyone already has it — copying it gives no edge. Use it only to (a) understand mechanics, or (b) bootstrap a planning bot faster. Our own physics core may be just as good. Don't treat acquiring it as a priority.

---

## Phase 2 — ONGOING (data pipeline: automate ingestion, NOT submission)

**Goal:** Turn every Kaggle game our bots play into training data, automatically.

```
[Kaggle plays our bots] → pull episodes → download replays →
parse to (state, action, outcome) → dataset on disk → [human reviews] → train → [human submits]
```

Automate everything up to the dataset. A human reviews and triggers training/submission.

### Pipeline components (cron or manual trigger, every few hours)
1. `pull_episodes.py` — for each live submission ID, list new episodes
2. `download_replays.py` — fetch replay JSON for episodes vs high-rated opponents (prioritize games we lost to strong bots — most informative)
3. `parse_replays.py` — convert replays → `(state, action, placement)` records, keyed by player rating where known
4. Append to a versioned dataset (`data/replays/YYYY-MM-DD/...`)

### Why not auto-submit?
- Only 2 tracked slots — a bad auto-submission can knock off a good bot
- 5/day cap — wasted submissions are gone
- A regression caught locally is free; caught on the ladder costs a day

### Tracking (keep it simple)
A single SQLite DB (`tracking.db`) with tables: `submissions` (id, name, score, σ, date), `episodes` (id, submission, opponents, result), `experiments` (local arena runs). **No MLflow yet** — add it only when you have multiple trained models to compare.

---

## Phase 3 — IMPROVEMENT (only after Phases 0–1 give a baseline)

Decide the improvement path *after* you know your 4-player baseline and have opponent data. Options, roughly in order of effort/payoff:

1. **Rule-distillation meta-bot** (no ML, buildable fast): switch strategy by game phase / state. Early-game markowitz expansion → vulture behavior when opponents fight → coordinated_strike finishing. The pairwise matrix already suggests this.
2. **Behavioral cloning** from opponent replays: supervised learning to imitate winning bots. Needs the Phase-2 dataset. Fast to converge.
3. **BC → RL fine-tuning** (self-play, 4-player): the high-ceiling path. Warm-start from BC or a heuristic, then improve via self-play. Jetstream2 makes this feasible. Highest effort.

Gate each on the previous paying off. Don't jump to RL before a meta-bot or BC is exhausted.

---

## Decision log (update as we learn)

- *[June 14]* Submitting markowitz + coordinated_strike based on 2P arena. **Caveat: no 4P data yet — revisit after Phase 1.**

---

## Anti-goals (things NOT to do)

- ❌ Build a custom physics engine for evaluation
- ❌ Auto-submit bots without human review
- ❌ Download 20 opponents (slows iteration)
- ❌ Set up MLflow/heavy infra before having models
- ❌ Over-engineer in-bot logging (replay JSON already has opponent data)
- ❌ Over-anchor on the 2-player ranking
