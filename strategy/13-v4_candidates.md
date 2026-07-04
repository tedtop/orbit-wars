# Orbit Wars — Strategy to Reach Top-10 by July 8

*Synthesis of the 19 downloaded bots + our own analysis. Green-field: we are not bound to the
current `coordinated_strike_interceptor`. Optimize purely for final rating, not code reuse.*

---

## 0. The strategic situation (read this first)

**Where we are:** ~552. **Prize cutoff (rank 10):** ~1533. **Best PUBLIC bot:** ~1266 (v44, an
orbit_lite fork). 

**The uncomfortable implication:** the top-10 is **above every public bot**. The ~1533 wall is
held by *private* agents — almost certainly serious self-play RL or heavily-tuned private
orbit_lite descendants. **Therefore: cloning the best public bot gets us to ~1266, which is NOT
top-10.** We must beat bots we cannot see. That rules out "tune a heuristic a bit" and forces one
of: (a) a clearly-better engine, (b) self-play learning that surpasses hand-tuned play, or (c)
exploiting *structural* weaknesses every bot shares (4P coalition, do-nothing opponent models,
OpenSkill variance).

**The single biggest finding:** the entire visible top of the leaderboard is **one codebase** —
Slawek Biel's `orbit_lite` engine — in config variants (see `agents/opponents/ORBIT_LITE_FAMILY.md`).
That engine has a research-grade combat/physics core (exact flow-diff scoring, provable aim,
safe_drain defense) and **three shared blind spots, none addressed by anyone**:
1. **Opponents are modeled as do-nothing.** No move prediction. (Universal.)
2. **4P is played shallow** — opponents summed with equal weight; almost no coalition/anti-leader play.
3. **No working learned control** — v2-gru shipped its GRU *disabled*; the public learned-value bot
   uses a tiny gradient-boosted tree on 8 coarse features. The community tried learning and under-built it.

**Our thesis:** *Adopt the orbit_lite combat core (it's the best, and free). Win on the three
layers the whole field neglects — real 4P opponent-asymmetry, opponent-response modeling, and a
self-play-trained control/value layer — and exploit the OpenSkill meta. Use the 14 bots we now run
locally as a private benchmark gym no competitor has.*

**Our unfair advantages:** (1) all 14 top bots runnable locally for head-to-head benchmarking;
(2) a replay ingestion pipeline + dashboard already built; (3) `orbit_lite` (torch self-play
engine) + the RL tutorial's PPO `PlanetPolicy` in hand. We can train and measure; most competitors
are hand-tuning blind.

---

## The 10 strategies (ranked by expected value toward top-10)

Each: **Concept · Mechanism · Why it beats the field · Impact · Effort · Risk.**
Scores are rough rating deltas relative to adopting the orbit_lite baseline (~1248).

### 1. Adopt orbit_lite + automated self-play config search  ⭐ FOUNDATION (do tonight)
- **Concept:** Make orbit_lite our engine. Then beat v44 (best public config, ~1266) by *searching*
  the ~12 config knobs with CMA-ES / coordinate-ascent, evaluated in our local arena against the 14
  downloaded bots — not by hand-tuning like everyone else.
- **Mechanism:** Wrap `the-producer-v2/main.py` as our bot. Build a subprocess tournament runner
  (each bot in its folder, 1s/turn). Objective = win-rate vs the field (2P + 4P placement). Sweep
  `horizon, roi_threshold, max_waves, min_ships, reinforce_beta, terminal-phase, ffa_leader_bonus`.
- **Why it beats the field:** the family plateaued at ~1248–1266 via *manual* tuning (44 iterations
  for v44's +18). Automated search over a real opponent gym finds configs humans didn't, and we
  validate on fresh seeds (avoids the overfit that burned our v2→v3 work).
- **Impact:** +0 to ~+30 (reach/slightly exceed v44). **Effort:** Low–Med (tonight → 2 days).
  **Risk:** Low. This is the safe floor and the substrate for #2–#6.

### 2. Port Roman's 4P opponent-asymmetry kit onto orbit_lite  ⭐ HIGHEST SINGLE LEVER
- **Concept:** The orbit_lite family's worst weakness is naive 4P; Roman's *separate* ledger bot
  (lb-1224) has the best 4P opportunism in public. **Combine the best combat model with the best 4P
  logic.** Nobody has done this — the two lineages are different authors.
- **Mechanism:** Add to orbit_lite's candidate scoring: **weakest-enemy focus** (bonus ∝ how much
  weaker a target's owner is), **gang-up-after-battle** (pile onto a planet just weakened by a
  fight), **exposed-planet counter-snipe** (hit planets an enemy just stripped to attack),
  **crash-exploit** (grab the empty planet after two enemies collide), **elimination bonus** (large
  reward for removing a player). These are concrete functions in `lb-max-1224`/`lb-958` to port.
- **Why it beats the field:** **4P FFA is where the prize rating is won and lost** (huge variance),
  and the entire orbit_lite family plays it with a flat `Σ_opp` sum + a token 0.035 anti-leader
  nudge. Real asymmetry play is a step-change in 4P, directly attacking the population's blind spot.
- **Impact:** +30 to +80, concentrated in 4P (the high-variance, high-reward format). **Effort:**
  Med. **Risk:** Low–Med (must validate it doesn't hurt 2P). Stacks on #1.

### 3. Behavior-clone the field → PPO self-play (the path PAST 1266)  ⭐ THE BIG BET
- **Concept:** The only proven route above hand-tuned play is self-play RL. Warm-start by **cloning
  the orbit_lite family's moves** (free expert data — we can generate unlimited labeled
  (state→action) pairs by running the bots), then **fine-tune with PPO self-play** against the 14
  downloaded bots + past versions of ourselves until we exceed them.
- **Mechanism:** Use the RL tutorial's `PlanetPolicy` (self/global/candidate encoders → target +
  ship-bucket heads — the correct architecture). Phase 1: supervised BC on producer-v2/v44 rollouts
  → instantly ~matches them with a fast NN (no 1s torch-sim per turn). Phase 2: PPO with shaped
  rewards (production/ship deltas, not just win/loss) and a self-play opponent pool. Phase 3:
  league play vs all downloaded bots.
- **Why it beats the field:** BC gives a strong, *fast* baseline; self-play then discovers play the
  hand-tuned engine can't express (true multi-turn setups, opponent-adaptive aggression). This is
  almost certainly what the private ~1533 bots are doing. It's our realistic top-10 path.
- **Impact:** +50 to +250 (the wide bet; this is what can actually clear 1533). **Effort:** High
  (the multi-day/week project). **Risk:** High (RL is finicky) — but BC alone is low-risk insurance
  that already lands near SOTA. Build on the RL tutorial; don't start from scratch.

### 4. Replace the do-nothing projection with 1-ply opponent-response modeling
- **Concept:** Every top bot plans assuming opponents sit still. Predict that each opponent plays
  ~orbit_lite-greedy, simulate their best response to our move, and plan against *that* (minimax-lite).
- **Mechanism:** Within the orbit_lite turn, after generating our candidate waves, run a cheap
  opponent-policy forward (reuse orbit_lite itself as the opponent model) to project their likely
  launches over the next few turns, fold those into `garrison_status` (replacing the do-nothing
  trajectory), then re-score. Effectively turns the planner reactive without full search.
- **Why it beats the field:** directly removes blind spot #1 — the do-nothing assumption is the
  single most exploitable thing in the game. Our planner stops walking into reinforcements/counters
  the whole field walks into.
- **Impact:** +20 to +60. **Effort:** Med–High (must stay under 1s/turn — opponent model must be
  cheap). **Risk:** Med (compute budget). Pairs naturally with #1/#2; also a strong feature for #3's reward.

### 5. Self-play-trained control & value layer (revive the dead GRU; upgrade the GBC value)
- **Concept:** Two public bots left learning on the table: v2-gru's roi/wave **GRU controller**
  (disabled) and the search-value bot's **gradient-boosted value** (8 coarse features). Build the
  version they didn't: a learned controller that modulates `roi_threshold / max_waves /
  aggression / ffa_weight` from game-state history, and/or a learned **leaf value** over the
  orbit_lite exact simulator with rich (geometry-aware) features.
- **Mechanism:** Train on self-play + replay outcomes. Controller = small GRU/MLP on a feature
  history (leader gaps, board control, phase) → continuous knob deltas. Value = per-candidate NN
  scoring simulated next-states (the "right" version of the GBC bot).
- **Why it beats the field:** stacks a *learned*, state-adaptive policy on top of the strong static
  engine — exactly the gap between #1 (static config) and dynamic play. Lower-risk than full #3
  because the engine still guarantees legal, safe moves; learning only tunes/scores.
- **Impact:** +20 to +70. **Effort:** Med. **Risk:** Med. Natural bridge from #1 → #3.

### 6. 4P coalition / threat-aware target selection ("don't poke the bear; bleed them")
- **Concept:** Explicit FFA meta-play the field lacks: **don't attack a player who is busy fighting
  your bigger threat**, bandwagon the weakest, and avoid being the player who spends ships while two
  rivals trade (the mechanics deep-dive's own data shows the FFA bystander tends to win).
- **Mechanism:** Per turn, estimate each opponent's threat (production + ship mass + proximity to
  us) and their current engagement (who's fighting whom, from the arrival ledger). Down-weight
  attacking an opponent engaged with our top threat; up-weight finishing the weakest; hold a reserve
  when two rivals are about to collide so we arrive to mop up.
- **Why it beats the field:** wins 4P on positioning rather than combat. Compounds with #2 (asymmetry
  kit) and #4 (response modeling). Can ship rule-based first, then let #3/#5 learn it.
- **Impact:** +20 to +60 (4P). **Effort:** Med. **Risk:** Med (hard to validate without a 4P gym —
  build one). 

### 7. OpenSkill meta-management — variance control & submission timing  ⭐ FREE RATING
- **Concept:** The leaderboard is a *skill rating from games against the field*, not an absolute
  score. Manage it like one: in 4P, a catastrophic last place hurts rating more than a win helps, so
  **minimize variance / avoid blowups** rather than maximize peak aggression; time submissions to
  accumulate games before the June 23 lock; keep two tracked slots diversified (one safe, one
  experimental) so a bad experimental bot can't sink us.
- **Mechanism:** Add a "don't lose badly" guardrail (never strip the homeworld below a survival
  floor; bail to a defensive shell when hopeless to salvage 3rd over 4th in FFA). Track per-format
  rating in the dashboard; submit early enough that rating converges before lock.
- **Why it beats the field:** pure rating efficiency that costs no in-game skill. Many strong-but-
  volatile bots under-rate because of 4P blowups; consistency climbs past them.
- **Impact:** +10 to +40 (and protects everything else). **Effort:** Low. **Risk:** Very low.
  **Do this regardless of which engine we ship.**

### 8. Comet & sun hazard weaponization
- **Concept:** Turn the map hazards into weapons — an edge only floor-matched (comet evac) partially
  touches. **Sun-herding:** aim so an enemy's likely intercept path crosses the sun (they lose the
  fleet). **Comet lifecycle:** harvest our comet planets before expiry, deny/avoid sinking ships
  into comets about to vanish, and time captures to the comet production windows (spawn steps
  50/150/250/350/450).
- **Mechanism:** Extend aim scoring to reward shots that force the opponent into sun-crossing
  responses; add comet-remaining-life gating to target value (port from floor-matched +
  physics-helper's `segment_hits_sun`/`comet_remaining_life`).
- **Why it beats the field:** unclaimed micro-edge; small per-instance but free and compounding,
  and occasionally swings a game by deleting an enemy fleet for nothing.
- **Impact:** +5 to +25. **Effort:** Low–Med. **Risk:** Low. Cheap bolt-on to #1/#2.

### 9. Endgame terminal optimizer + tie-break exploitation
- **Concept:** Final standings tie-break is **total ships**; exp50/v44 already dump in a terminal
  phase but crudely. Build an *exact* end-state optimizer for the last ~25 turns: maximize final
  (garrison + in-flight) ships and planet count, accounting for which fleets actually land before
  step 500.
- **Mechanism:** When `steps_remaining < H`, switch objective from competitive-flow to terminal
  ship/planet maximization with a hard "fleet must arrive by 500" gate; never launch ships that
  can't land in time (a known orbit_lite efficiency leak our own v3 also had).
- **Why it beats the field:** squeezes guaranteed points from every game's last phase; in close 4P
  games the tie-break decides 2nd vs 3rd (big rating swing). Stacks on everything.
- **Impact:** +5 to +25. **Effort:** Low. **Risk:** Low.

### 10. Opening book + never-crash robustness
- **Concept:** Two cheap, reliable gains: (a) precompute strong **openings** per starting
  configuration (starts are structured; the first ~15 moves are nearly map-determined and can be
  solved offline), banking early production the slow-opening bots (i-m-stronger holds to ~turn 40)
  cede; (b) **never forfeit** — a crash/timeout loses the game outright, and several public bots are
  fragile. A bot that always returns a legal move in <1s banks free rating against them.
- **Mechanism:** Offline-search opening lines vs the field, store as a small lookup keyed by start
  layout; wrap the agent in a hard try/except + time guard that always emits a safe move.
- **Why it beats the field:** exploits the opening-hold meta and the fragility tax. Low glamour,
  high reliability.
- **Impact:** +5 to +20. **Effort:** Low–Med. **Risk:** Very low.

---

## Sourcing prize-zone (>1533) expert data — fuel for #3, #4, #5, #6  ⭐ (Ted's directive)

To train a policy that *exceeds* the public field, behavior-clone the **top-10/prize-zone bots**,
not the ~550-rated bots our own games currently match against. OpenSkill pairs similar ratings, so
our replays are mostly weak opponents — useless as expert data. We need the strong bots' games.

**The data exists and is public.** Kaggle publishes daily episode dumps —
`kaggle/orbit-wars-episodes-YYYY-MM-DD` (≈1.3 GB/day; we already saw them) plus
`kaggle/orbit-wars-episodes-index`. These contain **every ranked game including the prize-zone
bots playing each other.** Pipeline:

1. **Identify the prize-zone teams.** From our `leaderboards/*.csv` snapshots, list every TeamName
   with Score ≥ ~1500 (rank ≤ ~12, with margin). That's the target roster of "bots above us we
   can't beat yet."
2. **Pull their episodes.** Use `orbit-wars-episodes-index` to find episode IDs featuring those
   teams; download just those from the daily dumps (filter by team, don't grab all 1.3 GB blindly).
   These are real prize-zone games — the exact play we must learn.
3. **Recover their MOVES (state→action labels).** Orbit Wars is fully observable and a launch
   materializes as a new fleet. Diff consecutive step observations: any fleet appearing at step
   t+1 with `owner=team, source_planet=p, angle=θ, ships=n` that wasn't in-flight at step t **is
   that team's action** `[p, θ, n]`. This reconstructs every top bot's exact per-turn action set —
   clean supervised `(obs_t → action_t)` pairs for behavior cloning. (Adapt
   `reverse-engineering-agents-replay-analysis-tooli/replay_analysis.py`, which already loads
   replays and detects launches/coordination.)
4. **Profile each prize-zone bot** (reaction time, opening tempo, 4P coalition behavior, comet use,
   aggression curve) with the same tooling → per-opponent fingerprints. These feed: opponent
   modeling (#4), coalition logic (#6), and *opponent-conditioned* BC (clone "what the BEST bot
   does in THIS situation").
5. **Train.** BC the aggregate prize-zone policy (#3 phase 1) → strong fast baseline; then PPO
   self-play (#3 phase 2) with these bots' profiles as the opponent league; use recovered states
   for the value model (#5). The replays also tell us *which* private approaches dominate (do the
   >1533 bots look like souped-up orbit_lite, or something else?) — directly de-risking our big bet.

**Infra to build (early, gates #3):** extend `pipeline/` with `pull_topbot_episodes.py` (leaderboard
→ team roster → episode-index → targeted download) and `extract_moves.py` (obs-diff → labeled
action dataset). We already have the ingestion/replay scaffolding — this is an extension, not a rebuild.

**Reality check on our own replays:** they become useful expert data only as *we* climb. Until
then, the public episode dumps are the prize-zone data source. Also mine our own losses for
*counter-exploits*: every game we lose to a strong bot is a labeled example of a situation our
policy mishandles — high-value for targeted fine-tuning.

---

## Recommended roadmap (timeline to July 8; submission lock June 23)

**Tonight / this session's follow-up (ship a real v4):**
- #1 baseline: wrap orbit_lite (start from v44's config — best known point) as our bot.
- #7 guardrails + #9 terminal gate + #10 never-crash wrapper (all cheap, all pure upside).
- Build the subprocess tournament runner (the gym) — needed for everything downstream.
- **Submit:** orbit_lite@v44-config + guardrails. Expected ~1250–1280, an instant jump from ~552.

**Days 2–5 (climb toward / past SOTA + start the data engine):**
- #1 automated config search in the gym (push past v44).
- #2 port the 4P asymmetry kit (the highest single lever).
- #8 comet/sun bolt-ons. Re-submit the tuned + 4P-aware build. Target ~1300+.
- **Start the prize-zone data engine** (gates #3): `pull_topbot_episodes.py` +
  `extract_moves.py` → begin accumulating the labeled (obs→action) dataset from >1533 bots while we
  iterate on the heuristic. Profile the prize-zone roster to learn what actually holds the top-10.

**Week 2 → June 23 lock (the top-10 bet):**
- #3 BC from the field → fast NN baseline (low-risk, ~SOTA), then #5 learned control, then #4
  opponent-response modeling, then #3 phase-2 PPO self-play.
- #6 coalition logic folded into the learned policy.
- Keep one tracked slot on the safe tuned-orbit_lite bot; the other on the experimental learned bot.
  Submit the learned bot only once it beats the tuned one in the gym on fresh seeds.

**Validation discipline (hard-won lesson, see [[orbit-wars-setup]]):** always evaluate on fresh
seeds (25+ seeds / 50+ games), 2P *and* 4P; tuning-seed win-rates overfit. The local gym of 14 real
opponents is our edge — use it as the source of truth, not the public leaderboard (slow feedback).

## What I'd build first if forced to pick one
**#1 + #2 + #7.** Adopt orbit_lite, port the 4P asymmetry kit, add variance guardrails. That's a
concrete, low-risk bot that should land ~1300 (clear of all public bots) and is the launchpad for
the #3 RL bet that can actually reach top-10. Everything is gated behind building the **subprocess
tournament gym** — that's the first piece of infrastructure to write.
