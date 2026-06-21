# Autoresearch Log — ratchet history

Append-only. Each experiment: hypothesis → gauntlet result → KEEP/DISCARD. The champion is the best KEEP.

| date | experiment (bot) | hypothesis | gauntlet result | verdict |
|------|------------------|-----------|-----------------|---------|
| 2026-06-15 | precog/kingmaker/maestro/helmsman/oracle | bolt-on heuristics beat the engine | all ≈parity (seat-swapped) | ❌ discard (archived) |
| 2026-06-16 | comet_reaper_tuned (Optuna, ~19 knobs) | config tuning beats the engine | best 0.34 | ❌ discard (base config is a tight optimum) |
| 2026-06-17 | **schmeekler** | capture static (non-rotating) planets first | **72% 2P vs comet_reaper; beats whole panel (producer-v2 77%); best-in-pod 4P** | ✅ **KEEP → champion, submitted** |
| 2026-06-17 | comet_reaper_search | forward-sim rollout re-rank (λ-blend) | no signal — rollout policy reproduces the candidate moves | ❌ discard (needs a real MCTS tree + value fn) |
| 2026-06-17 | schmeekler_potential (Track A) | potential-field closing-rate bonus | w0.5/1.0 noise, w1.5 −5pp; flow scorer already encodes ETA/position | ❌ discard |
| 2026-06-17 | schmeekler_interdict (Track A) | enemy-fleet interdiction bonus | w1.0 **−26pp CATASTROPHIC**; additive bonus overrides the flow scorer (wins race, loses the hold) | ❌ discard |
| 2026-06-17 | comet_reaper_mcts v1 (Track B) | flat 2-ply combined-LaunchSet re-rank, exact flow leaf | n=20 looked 80% but **n=50 = 75% vs schmeekler 74% (parity)**; de-meaned corr has zero variance w/ fixed opp | ❌ discard |
| 2026-06-17 | comet_reaper_mcts v2 (Track B) | true depth-2 beam + state-advancement opp model (13.7ms/turn) | n=20 looked +2pp; **n=50 = 75% = schmeekler 74%** | ❌ DISCARD — **Track B FINAL: 2-ply search dead; default back to depth=1** |
| 2026-06-17 | schmeekler_fmt (Track A) | format-aware static bonus (off in 4P) | **CONFIRMED at matched n=150: identical to schmeekler in 2P (byte-identical code) + 3.72μ in 4P** → schmeekler_fmt ≥ schmeekler, the best schmeekler VARIANT. (Earlier "66% collapse" was an unmatched-baseline artifact.) Still 2P-identical so 2P-bound below comet_reaper. | ✅ KEEP over schmeekler (gym); natural replacement for the schmeekler slot + the live-calibration candidate. Not a comet_reaper-beater. |
| 2026-06-17 | schmeekler_phase (Track A) | phase-aware ship sizing (early 1.2× → late 3.0×) | 1.2=19%, 1.5=22% — breaks the engine's ROI/speed/floor sizing interaction | ❌ DISCARD |

| 2026-06-17 | **schmeekler_orbit** (Track B repurposed) | delay launch until planet closer/cheaper | HOLD_THRESHOLD=0.6 fires 0% turns; 0.8 fires 42% → passivity catastrophe. Root cause: capture floor is INDEPENDENT of orbital position for neutrals (garrison constant); enemy floors INCREASE with delay. Closer ≠ cheaper in this game. | ❌ DISCARD |
| 2026-06-17 | comet_reaper_vf (Track C Phase D) | learned value function on real episodes | AUC=0.9905, Pearson=0.9145, 19ms/turn — massive PASS. Phase E bot built. | ✅ PASS gate → Phase E eval |
| 2026-06-17 | comet_reaper_vf (Track C Phase E) | bolt-on VF expansion beats diverse field | 2P: 17-17-6 parity vs comet_reaper (both engine → duplicate moves); 4P: 12 vs 23 firsts — **HURT** (aggressive moves expose defensive holes to 3 opponents); 2P-gated variant still parity | ❌ **DISCARD** — bolt-on integration doesn't transfer; V has genuine signal (AUC=0.98) but needs to REPLACE `score_candidates`, not layer above it |
| 2026-06-17 | schmeekler_comet + comet_reaper_comet (Track A 2×2) | comet-targeting bonus (additive 1.5 to comet-planet candidates) | schmeekler_comet 74% OVERALL = schmeekler baseline 74% (+0pp); comet_reaper_comet 61% OVERALL ≈ comet_reaper baseline (+0pp); bonus sweep not run (no gain at n=50) | ❌ DISCARD — flow scorer already handles ephemeral target valuation; flat bonus is noise |
| 2026-06-17 | comet_reaper_stochastic (comet_reaper base + Boltzmann EV τ=2.0) | 2-ply Boltzmann opponent model beats schmeekler | τ=2.0: 61/100=61% vs schmeekler baseline 78/100=78% (-17pp); 80% vs 1266-elo but 50% vs comet_reaper — base too weak | ❌ DISCARD — wrong base; **key insight: apply EV on schmeekler base instead** |
| 2026-06-17 | **schmeekler_elim** | elimination mode (opponent ≤3 planets + dominance guard → +8 enemy bonus, ROI→1.0) | 488/750=65.1% vs schmeekler_fmt baseline ~78% (-13pp); direct: 9-11 (45%) vs fmt. **Pattern:** +12pp vs 1266-elo (hard) but -16 to -28pp vs peers. threshold=3 fires in "slightly ahead" states causing overcommit. | ❌ DISCARD — insight: same asymmetry as stochastic (hard-game edge, medium-game regression) |
| 2026-06-17 | **schmeekler_stochastic** | schmeekler base + Boltzmann EV (fixes base-weakness root cause) | n=20: 7/20 (35%) vs CR [baseline 80%], 16/20 (80%) vs 1266-elo [+20pp]. Controls: EV_BONUS=0→80% (base correct), any non-zero EV→hurts. Root cause: EV penalizes ships-in-transit (offensive attacks look like garrison loss → EV inverts schmeekler's aggressive strategy). | ❌ DISCARD — EV mechanism structurally incompatible with schmeekler's static-grab |
| 2026-06-17 | **multi_fleet_coordination** (`comet_reaper_combo`) | coordinated 2-src attacks on targets neither fleet can take alone (staggered arrival) | 4 iterations (crude ETA → tensor intercept_angle); COMBO_ON: 35% vs CR (vs baseline 48%); economic analysis: 95% of combos negative EV (avg -75.6 ships). Staggered arrival = sacrificial softening hurts. Simultaneous arrival (L=2 LaunchSet) would be right path. | ❌ DISCARD — action-space gap real but staggered exploit not profitable |

**Current champion:** `schmeekler` (static_target_bonus=1.5) — but **live ≈ comet_reaper PARITY**; gym overstated it.
**Live (2026-06-17):** schmeekler ~1066–1085–**1075** = PLATEAUED ~170 below comet_reaper 1245 (not near 1259). Cold-start
lag ended → schmeekler genuinely a touch *below* CR live; the gym overstated it.
**Gym v2 (clean, valid anchors):** hard inversion FIXED — CR 25.53 ≳ schmeekler 25.17 (≈tie); weak bots far below.
Spearman 0.43 is an artifact of the 4 top anchors being within 66 live pts (1200–1266) — uncrackable noise, wrong metric.
Residual: gym shows CR≈schmeekler but live is CR≫schmeekler (+170) → gym understates CR's margin, so a SMALL gym edge over
CR ≠ a live gain. (low-p4 run testing whether less 4P sharpens CR>schmeekler.)
**Submission discipline:** comet_reaper (1245) stays our best; no candidate beats it live. BUT schmeekler_fmt ≥ schmeekler
(confirmed) → if we run a schmeekler-family slot, fmt is the better choice. Ted opting to live-test (live data = gym
calibration); safest is to replace the schmeekler slot with fmt, keep comet_reaper. Verify active-slot rule first.
**Tracks:** A COMPLETE (all bolt-ons explored; fmt the only keep-over-schmeekler). B COMPLETE (2-ply search dead).
Remaining bets: value function on Jetstream2 (the real swing); optional cheap comet kill-test (2×2 factorial).
**Read so far:** structural-bonus features (potential/interdict) and 2-ply exact-flow re-rank all ≈parity-or-worse —
the orbit_lite 1-ply scorer is a tight optimum even vs correct 2-ply lookahead. New live-evidence lever = the
engine-wide **comet blind spot** (Track A's next build) + a **gym v2** that predicts the ladder (calibrating now).
**Next:** comet_reaper_mcts v2 @ n=50; Track A comet-aware feature; validate everything on gym v2 before submitting.
