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
| 2026-06-17 | comet_reaper_mcts v2 (Track B) | true depth-2 beam + state-advancement opp model (13.7ms/turn) | n=20 80% vs schmeekler 78% (+2pp, within noise) | ⏳ PENDING n=50 |
| 2026-06-17 | schmeekler_fmt (Track A) | format-aware static bonus (off in 4P) to fix 4P | disabling bonus in 4P does NOT rescue 4P (μ20.06 vs baseline μ23.11 — worse); 4P weakness ≠ static bonus | ⏳ PENDING (likely discard; points to fleet-SIZING in multiway fights) |

**Current champion:** `schmeekler` (static_target_bonus=1.5) — but **live ≈ comet_reaper PARITY**; gym overstated it.
**Live (2026-06-17):** schmeekler 1057→1079→**1085** (slope flattening ~160 *below* comet_reaper 1245, not near 1259).
**Read so far:** structural-bonus features (potential/interdict) and 2-ply exact-flow re-rank all ≈parity-or-worse —
the orbit_lite 1-ply scorer is a tight optimum even vs correct 2-ply lookahead. New live-evidence lever = the
engine-wide **comet blind spot** (Track A's next build) + a **gym v2** that predicts the ladder (calibrating now).
**Next:** comet_reaper_mcts v2 @ n=50; Track A comet-aware feature; validate everything on gym v2 before submitting.
