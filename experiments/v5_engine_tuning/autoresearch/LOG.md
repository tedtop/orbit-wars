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
| 2026-06-17 | schmeekler_fmt (Track A) | format-aware static bonus (off in 4P) | +3.72μ at small n **collapsed to 66% at n=150** (below schmeekler ~74%) | ❌ DISCARD (small-sample mirage) |
| 2026-06-17 | schmeekler_phase (Track A) | phase-aware ship sizing (early 1.2× → late 3.0×) | 1.2=19%, 1.5=22% — breaks the engine's ROI/speed/floor sizing interaction | ❌ DISCARD |

**Current champion:** `schmeekler` (static_target_bonus=1.5) — but **live ≈ comet_reaper PARITY**; gym overstated it.
**Live (2026-06-17):** schmeekler ~1066–1085–**1075** = PLATEAUED ~170 below comet_reaper 1245 (not near 1259). Cold-start
lag ended → schmeekler genuinely a touch *below* CR live; the gym overstated it.
**Gym v2 (clean, valid anchors):** hard inversion FIXED — CR 25.53 ≳ schmeekler 25.17 (≈tie); weak bots far below.
Spearman 0.43 is an artifact of the 4 top anchors being within 66 live pts (1200–1266) — uncrackable noise, wrong metric.
Residual: gym shows CR≈schmeekler but live is CR≫schmeekler (+170) → gym understates CR's margin, so a SMALL gym edge over
CR ≠ a live gain. (low-p4 run testing whether less 4P sharpens CR>schmeekler.)
**Submission discipline:** HOLD. No candidate has a credible *live*-beats-comet_reaper case yet; comet_reaper (1245) stays our
best submitted bot. Also verify Kaggle's active-slot rule before submitting (a 3rd sub may retire comet_reaper → LB drop).
**Read so far:** structural-bonus features (potential/interdict) and 2-ply exact-flow re-rank all ≈parity-or-worse —
the orbit_lite 1-ply scorer is a tight optimum even vs correct 2-ply lookahead. New live-evidence lever = the
engine-wide **comet blind spot** (Track A's next build) + a **gym v2** that predicts the ladder (calibrating now).
**Next:** comet_reaper_mcts v2 @ n=50; Track A comet-aware feature; validate everything on gym v2 before submitting.
