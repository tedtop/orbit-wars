# AUDITOR LOG — Independent Orbit Wars RL Audit
*Auditor reports to Ted directly. Evidence-first, no cheerleading.*

---

## 2026-06-19 05:25 MT — Cycle 1 (First independent cycle)

### v6_cr4 LOCAL: ⛔ DEAD — ESCALATE

- U=1156, CF=0.000, Ent=0.086, SPS=199
- CF=0/50 for last 50 updates (>30 consecutive threshold breached — RED)
- Eval trajectory: 20.0% (U=900) → 12.0% (U=1000) → 9.5% (U=1100) — **actively declining**
- Entropy=0.086: policy fully committed, zero exploration. Not learning, collapsing.
- **This run should be killed.** It is now producing negative signal (declining eval while training).
- Orchestrator has not acted on this. Escalating to Ted.

### Fleet: 108 seeds synced, 35 with meaningful training

| Seed | U | bestG | lastG | Ent | CF50 |
|------|---|-------|-------|-----|------|
| 149.165.170.73/job1 | 200 | **32.5%** | 32.5% | 4.06 (U=240) | 0.055 |
| 149.165.175.182/quad_job4 | 300 | 28% | 16% | 1.96 | 0.087 |
| 149.165.175.182/quad_job3 | 300 | 24% | 22% | 1.01 | 0.044 |
| 149.165.175.182/quad_job1 | 337 | 26% | 18% | 0.80 | 0.013 |
| m3.2xl cluster (31 seeds) | 200 | 18–28% | 16–28% | 2.6–4.5 | varies |
| ~72 new seeds | 1–4 | ? | ? | ~5.0 | — |

**One standout: 149.165.170.73/job1 at 32.5% greedy (U=100).** Only one eval point — seed ran from U=0 to U=200+ after sync at 04:04. The U=200 eval hasn't been pulled yet. This is the best single-seed result in the campaign. CF50=0.055 (in-band), entropy decaying (not collapsed). This seed is alive.

**Quad seeds are dying:** U=300-337, all showing lastG declining from bestG. job1 dropped from 26% to 18% — same collapse pattern as local runs.

**CR eval: ABSENT from all fleet metrics.** Fleet code predates the `evaluate_vs_comet_reaper` addition. No CR signal exists from fleet.

### Promotion Loop — Phase 1 Findings

**05:02 MT: First promotion attempt. Symmetry gate fired correctly.**
```
challenger=U200  WR=0.754  slot0=0.372  slot1=0.384  seat_delta=0.028  vs_comet_WR=0.000
SYMMETRY GATE FAILED
```

✅ Gate ran — confirm mechanism works  
✅ Episode lengths unknown but likely real (decorator bug comment in eval_checkpoints.py says it was fixed)  
⚠️ **Draw rate ~24.4%** (slot0+slot1=0.756, draws=0.244) — see finding below  
❌ **Challenger was U=200; quad seeds are at U=300-337 and should have been selected** — selection is depth-based, but something picked U=200 over U=300+. May be a sync timing issue (quad not synced at 05:02) or the depth comparison found U=200 as deepest at sync time.

**Critical finding on symmetry gate calibration:**
With 24.4% draw rate, even a PERFECTLY symmetric game would produce slot0 = slot1 = 0.38 < 0.46 threshold. The gate threshold was designed assuming near-zero draws. At U=200 with weak policies playing each other, stalemates are common. The gate is mathematically correct but may be blocking a legitimate promotion by a threshold that doesn't account for draw-heavy matchups. The 32.5% seed's terminal_win_pct=23.1% shows it DOES win by elimination sometimes — not a pure turtler.

**Alternative: draw rate could indicate the 2-step fake-draw bug.** Cannot confirm without episode length from that specific eval JSON. Recommend: orchestrator re-run the promotion eval with `--json` and log `mean_episode_length`. If ~119 steps: draws are real stalemates. If ~2 steps: decorator bug present.

### Structural Flaws in Promotion Loop (Phase 1 verdict)

1. **Depth-based challenger selection** — picks deepest seed, not best-performing. The 32.5% seed at U=100 would have lost to a 20% seed at U=300 if both were available. Best-performing challenger by greedy WR would correctly elevate the 32.5% seed.

2. **Symmetry gate threshold uncalibrated for draws** — threshold 0.46 fails all weak-vs-weak matchups with >8% draw rate. Consider adding draw-aware check: `(slot0 + 0.5*draw_rate) in [0.46, 0.54]`.

3. **CR eval absent from fleet training** — all fleet seeds report CR=? in metrics. The only CR signal is the post-sync champion eval (which requires the fleet sync to run). Fleet seeds have no self-correcting CR signal during training.

4. **9 of 10 sync cycles returned vs_comet_WR=?** — the comet eval was silently failing. 10th cycle (05:02) returned WR=0.000. Local champion vs CR confirmed 0-200. The earlier "?" is likely a stderr-suppression issue in the sync script on fleet machines.

### Agreement/Disagreement with Orchestrator

**Disagree:** "Campaign FAIL — RL bet closed" (my earlier recording). Best fleet seed is 32.5% vs greedy at U=100 with CF in-band and entropy still decaying. This is above the 25% threshold. The pre-committed gate was U=500 WITH a verified-working league. The league was NOT verified-working at U=500 (symmetry gate failure prevented any valid promotion). The gate should not have been declared failed based on local seeds alone when the fleet has a seed showing >25%.

**Agree:** v6_cr4 local is dead and should be killed. Quad seeds are dying. The 20% ceiling is real for most seeds. CR eval = 0% for every checkpoint where it ran.

**Flagged risk:** The 32.5% seed's U=200 eval has not been pulled. If it shows continued improvement or holds >25%, it becomes the primary candidate for the pre-committed gate test. If it declines (like all local seeds), the FAIL verdict stands. **This one data point is the most important unknown right now.**

### Next Recommended Orchestrator Actions (not mine to take)

1. Kill v6_cr4 (dead, consuming resources, now producing negative signal)
2. Run fleet sync now to pull U=200+ eval from the 32.5% seed
3. Verify episode lengths from the 05:02 promotion eval (re-run with JSON log)
4. Consider adjusting symmetry gate for draw-aware threshold OR investigate draw cause

---

## 2026-06-19 05:45 MT — Cycle 2

### v6_cr4 LOCAL: Still running at U=1496. Not killed.

- CF=0.0000, Ent=0.153, CF zeros 46/50 last updates — policy frozen
- Eval: 9.5%→9.0%→17.0%→25.0% (U=1100–1400) — **noise**, not recovery. Frozen policy produces high-variance 200-game samples. The 25% at U=1400 is a statistical outlier; CF=0 and entropy near-zero confirm no learning.
- **Orchestrator has not acted on my Cycle 1 escalation.** This run has been dead since ~U=1100, ~2 hours ago. Flagging again.

### FLEET: Fully restarted. 32.5% seed gone.

- All 4 jobs on 149.165.170.73 at U=36, elapsed_h=0.64 — restarted ~05:00 MT
- The 32.5% seed (ppo-4-of-6_job1) was killed before its U=200 eval was captured. **We never learned if it held >25% at depth.** New run back at U=36.
- Sync at 05:35 confirms fresh restarts across fleet.

### ORCHESTRATOR STATE: Phase 1 COMPLETE — Substantively correct.

The orchestrator addressed all Phase 1 structural flaws I identified:
- ✅ Challenger ranked by greedy WR (not depth) — the critical fix
- ✅ Decorator bug → context manager fix; episode lengths now ~445-457 (confirmed: the 05:02 24% draw rate WAS caused by the bug, not encoding asymmetry)
- ✅ Draw-adjusted symmetry gate (`slot_0_decisive` replacing raw slot rates)
- ✅ Champion re-seeded from 32% seed (ppo-4-of-6_job1 U=100)
- ✅ Quad excluded from promotion
- ✅ Bench proved: 1 clean promotion (WR=78%, ep_len=457, gate PASS)
- ✅ Updated code pushed to all 8 hosts, 32 seeds restarted

**These fixes are correct.** The league mechanism is now structurally sound. Phase 1 verdict: VALID.

### PROMOTION LOOP: No new attempts since 05:02. (Fleet at U=36 — no challengers yet.)

### CR eval: 0.000 for current champion (expected). 0% for all seeds. No change.

### NEW FLAG: Submission floor is schmeekler_fmt (1149.4), NOT comet_reaper.

ORCHESTRATOR_STATE lists comet_reaper as "defended floor (sub 53707586)" — but this is the 3rd most recent submission. The "highest of latest 2" rule means the current selected score is **schmeekler_fmt at 1149.4**, not 1234.7. Ted is aware and plans to manage timing. Orchestrator may not know.

### Agreement / Disagreement

**Agree:** Phase 1 fixes are complete and correct. The league now works mechanically. U=500 gate at ~16:00 MT Jun 19 is the next decision point.

**Disagree:** "Phase 1 complete" framing implicitly closes the repair loop, but the 32.5% seed's U=200 result — the most important data point in the campaign — was destroyed in the restart. The new wave's first data arrives at U=100 ~06:00-07:00 MT. **Whether any seed in the fresh fleet replicates 32.5% at U=100 is unknown and is the next critical read.**

### Gate status

- Pre-committed gate: U=500, greedy >25% AND comet WR >0% — fires ~16:00-17:00 MT Jun 19
- Current best: champion = 32% greedy at U=100 (from OLD run, now in champion.pt only)
- New fleet: U=36, no evals yet

---

## 2026-06-19 06:05 MT — Cycle 3

### ⛔ ESCALATION #3: v6_cr4 still running (PID 34456, cpu=155min)

CF=0, Ent=0.153, eval bouncing 9–25% as noise. Dead. Orchestrator has not killed it across three cycles.

### Fleet evals — OLD DATA, not new run

The U=100 evals visible via SSH are from the **initial overnight run (02:18 MT)**, 8.7 hours before the fleet restart. Confirmed by timestamp delta: ts=1781857133 vs restart ~11:07 UTC = 8.7h prior.

metrics.jsonl has 378 lines = old run (U=1–200+) + new run (U=1–58, appended). grep found old-run eval data.

**Current new run state (job1, 149.165.170.73):** U=58, Ent=4.688, ts=1781870375. No evals yet. U=100 fires ~06:45 MT.

### Old-run eval data (confirmed real, not new):

| Host | Job | U=100 greedy | sym_ok | mean_ep_win | Notes |
|------|-----|-------------|--------|-------------|-------|
| 149.165.170.73 | job1 | **32.5%** | ✅ | 382 steps | Champion seed |
| 149.165.170.73 | job2 | 20.0% | ✅ | 341 steps | — |
| 149.165.170.73 | job3 | 24.0% | ✅ | 354 steps | — |
| 149.165.170.73 | job4 | 20.5% | ✅ | 343 steps | — |
| 149.165.174.18 | job1 | 16.5% | ✅ | 368 steps | — |
| 149.165.174.18 | job2 | 21.5% | ✅ | 406 steps | — |
| 149.165.174.18 | job3 | 22.5% | ✅ | 392 steps | — |
| 149.165.174.18 | job4 | 20.5% | ✅ | 363 steps | — |

**Key:** sym_ok=true for all 8 seeds, mean_ep_win 341–406 steps. Episode lengths are real (not 2-step draws). The internal `evaluate_vs_greedy` in train.py was clean throughout. The decorator bug only affected eval_checkpoints.py (now fixed).

**Distribution:** 1 seed at 32.5%, 7 seeds at 16.5–24%. One outlier, not a fleet-wide signal. This is important context for the pre-committed gate: even the best single result (32.5%) is above 25%, but it's a single seed in a right-tailed distribution where the median is ~21%.

### Promotion loop: No new attempts. Fleet at U=58 — first challenger at ~U=200, ~07:45 MT.

### CR eval: All zero or absent. No change.

### Gate trajectory

- Pre-committed gate: U=500, greedy >25% AND comet >0%
- Fires ~16:00-17:00 MT on new run's seeds
- Only 1 of 8 sampled old-run seeds hit >25% — need new run to confirm distribution

---

## 2026-06-19 06:22 MT — Cycle 4

### v6_cr4: KILLED. Last entry U=1758. ✅

Finally killed after 3 escalations. Process no longer in ps aux.

### New fleet run: U=79-80 on both sampled hosts. Evals fire ~06:45 MT.

| Host | Job | U | Ent@U80 | elapsed_h |
|------|-----|---|---------|-----------|
| 149.165.174.18 | job1 | 80 | **3.873** | 1.39 |
| 149.165.174.18 | job2 | 80 | 4.554 | 1.40 |
| 149.165.174.18 | job3 | 80 | 4.663 | 1.39 |
| 149.165.174.18 | job4 | 79 | 4.817 | 1.39 |
| 149.165.170.73 | job1 | 80 | 4.627 | 1.40 |
| 149.165.170.73 | job2 | 79 | 4.488 | 1.40 |
| 149.165.170.73 | job3 | 80 | 4.344 | 1.40 |
| 149.165.170.73 | job4 | 80 | 4.505 | 1.39 |

Entropy spread at U=80: 3.87–4.82. .174.18/job1 is the fastest decayer (3.873). .170.73/job1 (the 32.5% seed) is at 4.627 — barely started. Old run's 32.5% seed apparently had faster entropy decay by U=80; the new run appears slower on the same job slot. **Different decay trajectory likely means a different U=100 result.**

No new-run evals (ts>1781867000) on either host — confirmed U=100 not yet reached.

### Promotion loop: No new challengers. 3 recent sync entries all "no challenger, vs_comet=0.000". 

vs_comet_WR=0.000 (not "?") in all recent entries — comet eval fix is confirmed working consistently.

### CR eval: 0.000 consistently. No change.

### Gate trajectory: On track mechanically. U=100 evals in ~20 min, U=500 gate at ~16:00 MT.

---

## 2026-06-19 06:44 MT — Cycle 5

### Evals running now — none written to disk yet

12 seeds sampled across 3 hosts. All at U=99-100. Zero new-run evals logged (ts>1781867000). The eval (200-game vs greedy) is executing right now but hasn't flushed to metrics.jsonl.

Entropy at U=100 in new run — key comparison:

| Seed | Ent@U100 (new) | Ent@U100 (old) | Notes |
|------|----------------|----------------|-------|
| .170.73/job1 | 4.639 | ~4.163 | +0.476 slower — predicts lower WR |
| .174.18/job1 | 4.021 | ? | Faster decay, comparable to old job1 |
| .171.142/job3 | 3.916 | ? | Fastest of sample |

.170.73/job1 (the 32.5% seed) has higher entropy at U=100 in the new run than in the old run. Old run had 4.163 at U=100, new has 4.639. Slower decay → weaker gradient signal → likely lower U=100 eval. **Predicting the 32.5% does NOT replicate on this seed this run.**

Fast-decaying seeds (.174.18/job1 Ent=4.021, .171.142/job3 Ent=3.916) are more promising candidates for the top result this wave.

### Promotion loop: no challengers. Last entry 06:30 "no challenger". Fleet at U=100, challenger threshold is U=150+. First challenger expected ~07:35 MT.

### CR: still 0.000. No change.

### Gate trajectory

Seeds at U=100 now. U=500 gate fires ~16:00 MT. First challenger appears ~07:35 MT. The entropy spread (3.916–4.639 at U=100) suggests performance spread will be wider than the old run's 16.5–32.5%. Fastest decayers may outperform; slowest (including the prior 32.5% seed) may underperform.

---

## 2026-06-19 07:05 MT — Cycle 6

### U=100 evals still running — all 12 seeds showing NO_EVAL_YET

All 12 seeds across 3 hosts at exactly U=100, entropy ranging 3.859–4.639. Evals executing but not written. Fleet machines run eval games sequentially at ~65 SPS equivalent → 200 games × ~330 steps = ~66K steps at ~10-20 steps/sec = estimated 55-110 min eval wall time. **Expected write: 07:30-08:00 MT.**

Entropy at U=100, new run (key seeds):
- .171.142/job3: **3.859** (fastest decay — top candidate)
- .174.18/job1: 4.021
- .170.73/job2: 4.111
- .170.73/job3: 4.232 / job4: 4.178
- .174.18/job2: 4.337
- .171.142/job2: 4.353
- .171.142/job1: 4.401
- .174.18/job4: 4.411
- .171.142/job4: 4.501
- .170.73/job1: **4.639** (slowest — prior 32.5% seed, predicts weakest result)

Entropy pattern in new run is universally higher than old run at U=100. Old run's champion seed had 4.163; new run's same seed is at 4.639. **Predicting the new-run distribution shifts left vs old run** — median likely <21%, fewer seeds above 25%.

### Promotion loop: no challengers. 06:45/06:59/07:00 syncs all "no challenger". First challenger at U=200 (~08:00-08:30 MT, accounting for eval delay).

### vs_comet_WR=0.000 consistently. No CR eval change.

---

## 2026-06-19 07:32 MT — Cycle 7

### U=100 evals still running. Timing pinned.

All 12 primary seeds at U=100, elapsed_h=1.74-1.77 (frozen — no new training updates). U=100 training finished at ~06:51 MT (job_start 05:07 + 1.74h). Eval has been running **41 minutes** since 06:51. Expected completion: 07:46-08:41 MT (55-110 min total eval wall time).

Extra seeds spotted: .174.18/job5 (U=2, elapsed=0.05h) and job6 (U=1, elapsed=0.03h) — new seeds just launched on that host, beyond the original 4 per host.

### No new eval_vs_cr field visible. All "WAIT" — can't check CR yet.

### Promotion loop: No challengers. 07:10/07:15/07:16 syncs clean. vs_comet_WR=0.000.

### Gate trajectory: On mechanical track. Evals ~08:00-09:00 MT. First challenger at U=200 follows ~1h after evals complete.

---

## 2026-06-19 08:02 MT — Cycle 8 ⚠️ ESCALATION

### ⚠️ FLEET RESTARTED AGAIN — second time. All U=100 eval data permanently lost.

Seeds that were at U=100 running 70-minute evals (since 06:51 MT) are now at **U=3-6, elapsed_h=0.11h**. Fleet restarted at ~07:55 MT. The evals that were running for 71 minutes never wrote results.

**Data destroyed twice:**
- Restart 1 (~05:00 MT): killed old overnight run (U=200+, 32.5% seed). Lost U=200 eval.
- Restart 2 (~07:55 MT): killed new run at U=100 with evals running 71 min. Lost entire U=100 distribution.

**Fleet capacity change:** .174.18 now has 8 jobs (job1-8), up from 4. .170.73 and .171.142 unchanged at 4 jobs each.

**Gate impact:** U=500 gate slipped from ~16:00 MT to ~20:00 MT.
- U=100 eval re-fires: ~07:55 + 1.74h + 70min eval = ~10:30 MT
- U=500 arrives: ~07:55 + 8.75h training + ~2h eval overhead = ~18:45-20:00 MT
- Deadline is 2026-06-23. 4 days - 12h slip leaves plenty of runway, but this is the second slip.

**Pattern:** Fleet restarts happen when orchestrator finds bugs or adds capacity. Each restart kills an eval cycle in progress. The auditor has now been through 8 cycles without a single new-run eval point from the fleet. All reads are from the old overnight run (8 hours ago).

### No U=100 evals in any new-run seed. Zero CR data. Promotion loop running syncs every 15 min, all "no challenger."

### Recommended escalation to Ted: the fleet restart pattern is destroying data before it can be read. Third restart would push the U=500 gate past midnight. The fleet needs to run to U=500 WITHOUT interruption to produce the gate decision.

---

## 2026-06-19 09:04 MT — Cycle 9

### No third restart. Fleet progressing normally.

| Host | Jobs | U range | Ent range | elapsed_h |
|------|------|---------|-----------|-----------|
| .170.73 | 4 | 63–66 | 4.39–4.92 | 1.13–1.15h |
| .174.18 wave 1 | 4 | 50–51 | 4.38–4.85 | 1.14–1.15h |
| .174.18 wave 2 | 4 | 12–13 | 4.87–4.95 | 0.45–0.47h |
| .171.142 | 4 | 63–64 | 4.07–4.38 | 1.13–1.15h |

No third restart confirmed. .174.18 now runs 8 jobs (wave 2 added ~08:32 MT). Fastest decayer: .171.142/job1 at Ent=4.065. Slowest: .170.73/job4 at 4.920 (near-random at U=65).

**U=100 evals write: ~10:51–11:07 MT** (35-50 more updates to U=100, then ~70 min eval).

### Promotion loop: 5 clean syncs (08:17–09:02), all "no challenger, vs_comet=0.000". No anomalies.

### Gate: U=500 at ~20:00 MT. No third restart → holds.

---

## 2026-06-19 10:06 MT — Cycle 10

### No third restart. All primary seeds at U=100 with evals running.

| Host | Jobs | U | elapsed_h | Ent@U100 | Status |
|------|------|---|-----------|----------|--------|
| .170.73 | 4 | 100 | 1.74–1.80h | 4.083–4.640 | EVAL RUNNING |
| .174.18 wave 1 | 4 | 100 | 1.99–2.02h | **3.765–4.556** | EVAL RUNNING |
| .174.18 wave 2 | 4 | 12–13 | 0.45–0.47h | 4.87–4.95 | TRAINING |
| .171.142 | 4 | 100 | 1.75–1.77h | 3.954–4.244 | EVAL RUNNING |

No third restart: elapsed_h on primary seeds is 1.74–2.02h, consistent with 07:55 restart and 100 updates at ~63 sec/update.

**Entropy signals at U=100 — best wave yet:**

| Seed | Ent@U100 | vs old champ (4.163) |
|------|----------|----------------------|
| .174.18/job4 | **3.765** | −0.398 (much faster decay) |
| .174.18/job1 | **3.831** | −0.332 |
| .171.142/job1 | 3.954 | −0.209 |
| .171.142/job2 | 3.964 | −0.199 |
| .170.73/job1 (prior 32.5% seed) | **4.083** | −0.080 |
| .170.73/job3 | 4.438 | +0.275 |
| .170.73/job4 | 4.640 | +0.477 (weak seed) |

**Six seeds have lower entropy at U=100 than the old champion run (4.163).** The two fastest decayers (.174.18/job4 at 3.765, job1 at 3.831) have never been seen before in this campaign. The prior 32.5% seed (.170.73/job1) is at 4.083 — BELOW its old overnight value of 4.163, meaning it's more committed this run than the run that produced 32.5%.

If entropy-decay correlates with greedy eval performance (as observed across prior runs), this is the strongest U=100 cohort so far. Cautioulsy optimistic but holding judgment until results write.

**Eval write estimate:** ~70-90 min from eval start.
- .170.73 eval started ~09:41 MT → writes ~10:51–11:11 MT
- .171.142 eval started ~09:44 MT → writes ~10:54–11:14 MT
- .174.18 wave 1 eval started ~09:56 MT → writes ~11:06–11:26 MT

### Promotion loop: 5 clean syncs (09:38–10:01). No challengers. vs_comet=0.000.

### Gate: U=500 ~20:00 MT. Entropy signals suggest this wave could outperform prior wave at U=100.

---

## 2026-06-19 11:08 MT — Cycle 11

### No third restart. FIRST U=100 evals written. FIRST promotion validated.

**IMPORTANT — eval game count was reduced at 07:50 restart:**
n_games bug: 200+100 sequential games = ~125 min per eval. Fix: n_games=30 (greedy) / 20 (CR). All fleet evals from this run use these smaller counts. 95% CI at 30 games ≈ ±18% — results are noisy, gate threshold defensible but training signal weaker.

### U=100 eval distribution (30-game, ts > 1781877000):

| Seed | g WR | CR WR | ep_win | Ent@now |
|------|------|-------|--------|---------|
| .170.73/job1 | **0.433** | 0.000 | 394 | 3.860 |
| .170.73/job2 | 0.367 | 0.000 | 344 | 4.291 |
| .170.73/job3 | 0.333 | 0.000 | 361 | 4.293 |
| .170.73/job4 | 0.300 | 0.000 | 399 | 4.650 |
| .174.18/job1 | 0.367 | 0.000 | 417 | 3.401 |
| .174.18/job2 | **0.400** | 0.000 | 363 | 4.433 |
| .174.18/job3 | 0.333 | 0.000 | 375 | 4.661 |
| .174.18/job4 | **0.400** | 0.000 | 366 | 3.723 |
| .171.142/job1 | 0.267 | 0.000 | 402 | 3.764 |
| .171.142/job2 | **0.433** | 0.000 | 413 | 3.374 |
| .171.142/job3 | **0.433** | 0.000 | 423 | 2.807 |
| .171.142/job4 | 0.300 | 0.000 | 408 | 4.252 |

Max=0.433 (×3), Median=0.367, Min=0.267 | ≥0.333: 10/12 | ≥0.400: 5/12
Episode lengths 344–423 ✅. CR=0.000 all (expected at U=100, 20 games).
Note: Old overnight champion was 32.5% in 200 games — not directly comparable to 30-game results here.

### FIRST PROMOTION — 10:45 MT ✅

Challenger: ppo-1-of-3_job1 (xl host, NOT quad), U=300, WR=33%
vs Champion: U=100, 32.5%
Result: WR=0.751 | slot0_dec=0.488 ✅ | slot1_dec=0.512 ✅ | seat_delta=0.034 ✅ | sym_gate=PASS
term=0.02 (score-based wins) | vs_comet=0.000 (expected)
Champion archive: champion_U100_20260619_1045.pt in pool ✅
Double-log at 10:45 is script artifact (sync bug fixed 10:03), not double-promotion.

Source: xl hosts ran since 05:00 MT restart (not 07:50 restart) → reached U=300 in 6h ✓

### Current state:
Champion = U=300, 33% greedy, 0% CR. 2xl seeds at U=133-155 (elapsed=3.20h).
Next challengers: xl/job2 WR=40% at U=100 (30-game) → watch at U=200 ~11:45 MT.
2xl seeds reach U=350 (eligible) at ~12:00-12:30 MT.

.174.18 jobs 5-8: STUCK at U=13 / elapsed=0.47h — same as Cycle 10. Appear dead. Not escalating (primary seeds healthy).

### Gate: U=500 ETA ~17:00-18:00 MT. 6 more hours of training.

---

## 2026-06-19 12:12 MT — Cycle 12

### No third restart. No new promotions. 4 dead seeds confirmed on .174.18.

**Restart check:** 2xl seeds show only U=100 eval entries. Seeds must currently be between U=100 and U=200 (no U=200 evals written yet). This means they've been running since 07:50 MT restart (~4.4h) and have NOT crashed. Estimated current U: 190-220.

**No new promotions:** ORCHESTRATOR_STATE shows 7 clean syncs (10:57–12:10 MT), all "no challenger, WR=0.3333, vs_comet=0.000". Expected — 2xl seeds not yet at U=350 (promotion threshold = champion_U 300 + 50).

**Champion.pt verified:** U=300, best_wr=0.333 ✅

**Minor audit flag — pool archival content mismatch:**
pool/champion_U100_20260619_1045.pt shows U=300, best_wr=0.333 — SAME as current champion. Filename says "U100" but content is U=300. Archival appears to have saved the new champion's file, not the old U=100 snapshot. Not critical (U=300 > U=100 WR anyway) but if rollback to old champion were needed, it's not available. Flag to orchestrator.

**xl host .175.105 status:**
| Seed | Last eval U | g WR | CR |
|------|-------------|------|-----|
| job1 | U=300 | 0.333 | 0.000 | ← promoted champion (running but last eval at U=300) |
| job2 | U=100 | **0.400** | 0.000 | ← highest WR seed, likely in U=200 eval NOW |
| job3 | U=300 | 0.333 | 0.000 | ← not eligible (WR ≤ champion) |
| job4 | U=100 | 0.333 | 0.000 | ← same as champion WR at U=100 |

job2 (40% WR at U=100) is the strongest seed identified. If it sustains or improves at U=200, it becomes next challenger. If U=200 WR > 33.3% champion, eligible at U=350.

**Dead seeds — .174.18 jobs 5-8 CONFIRMED dead:**
Showing U=13/elapsed=0.47h for 4 consecutive cycles (09:04 → 12:12 MT, 3+ hours with zero progress). These 4 seeds are not training. Effective fleet count: 12 primary 2xl + ~12 xl seeds (some dead) = fewer active seeds than designed. Not an escalation trigger but the fleet is running at reduced capacity.

### Gate trajectory
- 2xl seeds at estimated U=200 right now → eval fires → U=200 results expected ~12:15–12:30 MT
- U=350 (first 2xl challenger eligible): ~13:00–13:30 MT
- U=500 gate: ~17:30 MT on fastest 2xl seeds
- CR signal still 0.000 across all evals

---

## 2026-06-19 13:17 MT — Cycle 13

### No restart. U=200 evals complete. No new promotions. Diverging seed trajectories.

**Restart check:** All primary seeds elapsed_h=5.34–5.36h. ✅ (07:50 MT restart = 5.45h ago. Consistent.)

---

### U=200 eval distribution (30-game):

| Seed | g@U100 | g@U200 | Δ | CR | ep@200 | Ent@U~230 |
|------|--------|--------|---|----|--------|-----------|
| .170.73/job1 | 0.433 | **0.467** | +0.03 | 0 | 384 | 3.376 |
| .170.73/job2 | 0.367 | 0.367 | 0 | 0 | 336 | 4.341 |
| .170.73/job3 | 0.333 | 0.367 | +0.03 | 0 | 429 | 4.114 |
| .170.73/job4 | 0.300 | 0.333 | +0.03 | 0 | 400 | 3.780 |
| .174.18/job1 | 0.367 | 0.367 | 0 | 0 | 396 | **2.980** |
| .174.18/job2 | 0.400 | 0.433 | +0.03 | 0 | 385 | 3.814 |
| .174.18/job3 | 0.333 | 0.400 | +0.07 | 0 | 374 | 4.398 |
| .174.18/job4 | 0.400 | **0.300** | -0.10 | 0 | 444 | **2.712** ← BAD collapse |
| .171.142/job1 | 0.267 | 0.267 | 0 | 0 | 399 | **2.879** |
| .171.142/job2 | 0.433 | 0.400 | -0.03 | 0 | 409 | **2.538** |
| .171.142/job3 | 0.433 | 0.433 | 0 | 0 | 467 | **2.279** ← LOWEST Ent |
| .171.142/job4 | 0.300 | 0.400 | +0.10 | 0 | 416 | 3.365 |

Stats: Max=0.467, Median=0.383, Min=0.267 | ≥0.400: 7/12 | ≥0.433: 3/12
All CR=0.000. Episode lengths 336–467 ✅.

**Distribution shift U=100→U=200:** Max +0.034, Median +0.016. Slight improvement across the fleet.

**Two diverging patterns emerging:**
1. **Strong-commit-good-strategy group** (Ent 2.3–3.4, WR holding or improving):
   - .170.73/job1: 43.3%→46.7%, Ent=3.376 — TOP CANDIDATE for next champion
   - .171.142/job3: 43.3%→43.3% flat, Ent=2.279 — MOST COMMITTED, WR holding
   - .174.18/job2: 40%→43.3%, Ent=3.814 — improving
2. **Entropy-collapse-bad-strategy group** (Ent < 3.0, WR falling):
   - .174.18/job4: 40%→30%, Ent=2.712 — committed to BAD strategy, ep_win=444 (defensive)
   - .171.142/job1: 26.7% flat, Ent=2.879 — lowest WR, highly committed, stuck
   - .174.18/job1: 36.7% flat, Ent=2.980 — plateau at 36.7%

---

### XL host .175.105 — job2 trajectory update

| Seed | g@U100 | g@U200 | Now | Ent |
|------|--------|--------|-----|-----|
| job2 | 0.400 | **0.367** | U=234 | 3.249 |

**xl/job2 went DOWN 40%→36.7%.** The 40% at U=100 was noise (12/30 games). Not the champion candidate we flagged last cycle. WR=36.7% > champion 33.3% — still a challenger if it holds at U=400.

xl/job1 (the promoted seed): now at U=371, Ent=2.384. Running toward U=400 eval. Its U=300 eval was 33.3% (= current champion WR). Whether it improves at U=400 is the key question.
xl/job3: U=370, Ent=2.395. Same WR trajectory as job1.

---

### Promotion window analysis
Champion: U=300, 33.3%. Next 2xl challengers eligible at U=400 checkpoint (threshold = champion_U 300 + 50 = 350, but evals fire at U=400 for 2xl seeds).
- 2xl seeds currently at U=220–241. Reach U=400 at ~14:00–14:30 MT.
- At U=400, 3 strong candidates: .170.73/job1, .171.142/job3, .174.18/job2

xl/job1 eval fires at U=400 (already at U=371, eligible since U>350). If U=400 WR > 33.3%, it will self-promote (champion.pt snapshot at U=300 being beaten by the same seed at U=400).

**No promotions in ORCHESTRATOR_STATE since 10:45 MT.** 8 clean syncs (12:10–13:10 MT), all "no challenger."

### Gate: U=500 on best seed ~17:30 MT. On track. CR still 0.000 everywhere.

---

## 2026-06-19 14:19 MT — Cycle 14

### No restart. U=300 evals WRITING NOW. Extreme entropy events in 3 seeds.

**Restart check:** elapsed_h=6.33–6.40h on all primary seeds. ✅

**U=300 evals are writing right now — too early to read results.**
2xl seeds at U=278-300 hit the U=300 checkpoint simultaneously this cycle. Eval games ~5-8 min. Results will be in AUDITOR_LOG by next cycle (~15:15 MT).

---

### Entropy alarms — 3 seeds approaching Ent < 2.0

| Seed | Ent@U~280 | g@U200 | Status |
|------|-----------|--------|--------|
| .174.18/job4 | **1.867** | 0.300 | ⚠ COLLAPSED — low entropy + falling WR |
| .171.142/job3 | **1.905** | 0.433 | ❓ AMBIGUOUS — low entropy + HOLDING WR |
| .171.142/job2 | 2.045 | 0.400 | ❓ committed + holding |
| .174.18/job2 | 2.414 | 0.433 | promising |
| .170.73/job1 | 2.919 | 0.467 | TOP CANDIDATE |

.174.18/job4 is the confirmed collapse pattern: Ent=1.867 with WR at 30% (down from 40%) and ep_win=444 (long defensive games). It has committed deeply to a bad strategy.

.171.142/job3 is the campaign's most interesting seed: Ent=1.905 (lower than any other PRODUCTIVE seed) with WR at 43.3% for TWO consecutive evals. If the U=300 eval also shows 43.3%+, this is a genuine discovery. If it drops, it's collapsing.

---

### XL host .175.105 — U=400 evals imminent

| Seed | Last eval | WR | Now U | Ent | Status |
|------|-----------|-----|-------|-----|--------|
| job1 (champion seed) | U=300 | 0.333 | 394 | 2.174 | U=400 eval WRITING NOW |
| job2 | U=200 | 0.367 | 286 | **1.949** | low ent + declining WR |
| job3 | U=300 | 0.333 | 394 | **1.550** | LOWEST ENT IN FLEET — U=400 eval WRITING NOW |
| job4 | U=200 | 0.333 | 288 | 3.836 | stable |

xl/job3 Ent=1.550 is the lowest entropy in the entire fleet. Its WR at U=300 was 33.3% (= current champion). Whether it improves or falls at U=400 answers the question of whether extreme commitment helps or hurts.

xl/job1 and xl/job3 are at U=394, eligible after U=400 eval writes if WR > 33.3%. They are the first promotion candidates from the xl queue.

---

### Promotion log: 10 clean "no challenger" syncs (12:58–14:10 MT). No promotions. ✅

**First promotion window:**
- xl/job1 + job3: U=400 eval writes ~14:30 MT → if WR > 33.3%, become challengers
- 2xl seeds: U=300 eval writes ~14:25-35 MT → NOT YET eligible (need U > 350 checkpoint). Eligible at U=400+ (~15:30 MT)
- Best 2xl candidate: .170.73/job1 (46.7% at U=200) currently writing U=300 eval

### Gate: U=500 ETA ~17:45-18:00 MT. Primary 2xl seeds at U=278-300.

---

## 2026-06-19 15:21 MT — Cycle 15

### No restart. U=300 distribution is DECLINING. ep_win=500 stall-strategy emerging. No promotions.

**Restart check:** elapsed_h=7.37–7.43h. ✅

---

### U=300 eval distribution — SIGNIFICANT REGRESSION

| Seed | g@100 | g@200 | g@300 | Δ200→300 | ep@300 | Ent@now |
|------|-------|-------|-------|----------|--------|---------|
| .170.73/job1 | 0.433 | 0.467 | **0.300** | -0.167 | **500** | 2.668 |
| .170.73/job2 | 0.367 | 0.367 | **0.300** | -0.067 | 340 | 3.395 |
| .170.73/job3 | 0.333 | 0.367 | 0.367 | 0 | 387 | 3.937 |
| .170.73/job4 | 0.300 | 0.333 | **0.300** | -0.033 | 410 | 2.998 |
| .174.18/job1 | 0.367 | 0.367 | **0.300** | -0.067 | **500** | 2.460 |
| .174.18/job2 | 0.400 | 0.433 | 0.367 | -0.067 | 448 | 2.087 |
| .174.18/job3 | 0.333 | 0.400 | **0.300** | -0.100 | 453 | 3.758 |
| .174.18/job4 | 0.400 | 0.300 | **0.267** | -0.033 | 492 | 1.964 |
| .171.142/job1 | 0.267 | 0.267 | 0.267 | 0 | 391 | 2.555 |
| .171.142/job2 | 0.433 | 0.400 | 0.333 | -0.067 | 381 | 1.879 |
| .171.142/job3 | 0.433 | 0.433 | **0.300** | -0.133 | 417 | **1.466** |
| .171.142/job4 | 0.300 | 0.400 | 0.367 | -0.033 | 420 | 3.287 |

U=300 stats: **Max=0.367, Median=0.300, Min=0.267**
Beats champion (0.333): 3/12 seeds (.170.73/job3, .174.18/job2, .171.142/job4, all at 0.367)
10 of 12 seeds DECLINED from U=200 to U=300.

**Comparison to prior checkpoints:**
| | Max | Median |
|---|---|---|
| U=100 | 0.433 | 0.367 |
| U=200 | 0.467 | 0.383 |
| U=300 | **0.367** | **0.300** |

The fleet peaked at U=200 and has deteriorated at U=300. This is the key finding of this cycle.

---

### ep_win=500 STALL STRATEGY — RED FLAG

Three seeds show ep_win=500 at U=300 (the maximum episode length):
- .170.73/job1: ep_win=500 (was 394/384 at U=100/200) — WR crashed from 0.467 to 0.300
- .174.18/job1: ep_win=500 (was 417/396 at U=100/200) — WR stable at 0.300
- .174.18/job4: ep_win=492 (was 444 at U=200) — WR declining: 0.300 → 0.267

ep_win=500 means the agent is winning games only at the max time limit — "stall for points" strategy. This behavior is likely reinforced by mixed cold-start vs comet_reaper (which may make time-limit endings more common). It is a degenerate local optimum: beats greedy marginally by outlasting it, but would be demolished by comet_reaper's aggressive play.

.171.142/job3 (the "most committed" seed at Ent=1.466): WR crashed 43.3%→43.3%→**30.0%** at U=300. The extreme entropy commitment was to a WRONG strategy. ep_win=417 (not max, but long). This is the fastest collapse in the campaign.

---

### XL host .175.105 — U=400 results

| Seed | U=300 WR | U=400 WR | Now U | Ent | Status |
|------|----------|----------|-------|-----|--------|
| job1 (champion seed) | 0.333 | **0.367** | 405 | 2.174 | IMPROVED — eligible challenger |
| job2 | (U=200: 0.367) | 0.400@U=300 | 303 | 1.695 | See below |
| job3 | 0.333 | **0.300** | 405 | **1.032** | COLLAPSED — Ent near zero |
| job4 | 0.333 | 0.300@U=200 | 304 | 3.694 | stable |

**xl/job1 scored 0.367 at U=400** (36.7% > champion 33.3%, U=400 > 350 threshold). Should be an eligible challenger. Yet sync shows "no challenger" through 15:06 MT. Either:
a) Sync hasn't picked up the updated best_model.pt yet (timing)
b) best_model.pt wasn't updated to U=400 (if WR tracking has a bug)
**Flag to orchestrator: xl/job1 U=400 WR=0.367 should be a challenger but isn't appearing in sync.**

**xl/job3 Ent=1.032** — the lowest entropy in the entire fleet. U=400 WR=0.300 (down from 0.333). Extreme commitment → bad strategy confirmed.

**xl/job2 WR=0.400 at U=300** — actually recovered vs U=200. Currently at U=303, Ent=1.695. Not yet eligible (need U > 350). Reaches U=350 at ~16:10 MT. Could be the next real promotion candidate if it holds.

---

### Gate trajectory — PRE-CALL WARNING

Gate condition: greedy >25% AND CR >0% at U=500 on best seed.

Current best trajectory (.170.73/job1 or .174.18/job2 or .171.142/job4):
- All three at 0.367 at U=300. Trend is declining at U=300. 
- If decline continues: U=400 result ~0.300-0.333, U=500 result ~0.267-0.333.
- greedy gate (>25%): LIKELY PASS even at 0.267 (8/30 > 25% = 7.5 needed)
- CR gate: 0.000 across ALL evals, all seeds. 20 games each. Nearly certain to be 0 at U=500.
- **Pre-call: LIKELY FAIL (CR gate)**. Greedy gate will probably pass but CR gate almost certainly fails.

### Promotions: 12 clean "no challenger" syncs (12:58–15:06 MT). Champion U=300 holds.

---

## 2026-06-19 16:25 MT — Cycle 16

### No restart. SECOND PROMOTION at 15:35 (U=400, 36.7%). Gate slips to ~20:45 MT.

**Restart check:** elapsed_h=8.44–8.49h on all primary 2xl seeds. ✅

---

### SECOND PROMOTION — 15:35 MT

Champion promoted from U=300 (33.3%) → U=400 (36.7%).
ORCHESTRATOR_STATE:
- 15:35: "PROMOTED U400 | WR=0.753 slot0=0.371 slot1=0.385 term=0.01 vs_comet_WR=0.000"
- 15:56: "PROMOTED U400 | WR=0.752 slot0=0.372 slot1=0.384 term=0.01 vs_comet_WR=0.000" (duplicate)
- 15:47+: "champion=U400 WR=0.3666..." (11/30 games)

Draw-adjusted gate: slot0_dec=0.371/0.756=0.491, slot1_dec=0.385/0.756=0.509. Both ∈[0.46,0.54]. ✅
term=0.01 (1% terminal wins — same degenerate pattern, score-based not elimination-based). 0% CR.

Source: xl/job1 (now at U=425, Ent=2.076). xl/job1 EP=435 at U=400 (reasonable game length, not stall).
Double-log is script artifact (same as first promotion).

**Champion progression:** U=100 (32.5%) → U=300 (33.3%) → U=400 (36.7%). Marginal improvement.

---

### 2xl seeds — U=400 evals NOT YET written

All primary 2xl seeds at U=319–329 at 16:25 MT. No U=400 eval rows written.
| Seed | Last eval | Last WR | Now U | Ent | Trend |
|------|-----------|---------|-------|-----|-------|
| .170.73/job1 | U=300 | 0.300 | 324 | 3.020 | ↓ from 0.467@200 |
| .170.73/job2 | U=300 | 0.300 | 325 | 3.098 | flat |
| .170.73/job3 | U=300 | 0.367 | 328 | 3.693 | best on host |
| .170.73/job4 | U=300 | 0.300 | 327 | 2.630 | flat |
| .174.18/job1 | U=300 | 0.300 | 320 | 2.588 | stall |
| .174.18/job2 | U=300 | 0.367 | 319 | 2.472 | best on host |
| .174.18/job3 | U=300 | 0.300 | 320 | 3.570 | flat |
| .174.18/job4 | U=300 | 0.267 | 321 | **1.219** | collapsed |
| .171.142/job1 | U=300 | 0.267 | 329 | **1.949** | STUCK |
| .171.142/job2 | U=300 | 0.333 | 322 | **1.713** | fast decay |
| .171.142/job3 | U=300 | 0.300 | 321 | **1.929** | collapsed (was 0.433) |
| .171.142/job4 | U=300 | 0.367 | 326 | 3.288 | best on host |

New collapse: .174.18/job4 Ent=1.219 (was 1.867 last cycle — continuing to drop). FOUR seeds now below Ent=2.0.

**U=400 eval ETA for 2xl seeds:** ~71-81 more updates at 38.7/hr = ~1.85-2.1h → ~18:15-18:30 MT.

---

### xl host .175.105 state

| Seed | Last eval | WR | Now U | Ent |
|------|-----------|-----|-------|-----|
| job1 (new champion) | U=400 | 0.367 | 425 | 2.076 |
| job2 | U=300 | **0.400** | 323 | 2.026 |
| job3 | U=400 | 0.300 | 427 | **0.920** |
| job4 | U=300 | 0.300 | 325 | 3.578 |

xl/job3 Ent=0.920 — almost deterministic. U=400 WR=0.300 and heading toward near-zero entropy. Confirmed bad strategy commitment.

xl/job2 WR=0.400 at U=300. Now at U=323. Next checkpoint at U=400, but new champion is U=400 so threshold is U>450. xl/job2 CANNOT promote at U=400 — needs U=500 checkpoint (eligible at U>450). ETA for xl/job2 U=500 checkpoint: from U=323 at ~18.5/hr → 177 updates → 9.6h → ~02:00 MT Jun 20. Not a meaningful gate candidate.

---

### GATE SLIP — CRITICAL REVISION

**New U=500 ETA for 2xl seeds:** Seeds at U=319-329. To U=500 = ~171-181 updates at 38.7/hr = 4.4-4.7h → **~20:50-21:10 MT** (was 17:45 MT). SLIP OF 3+ HOURS.

Root cause: eval overhead is ~20-25 min per eval cycle (4 jobs sharing CPU at ~25s/game × 50 games), not the 5-8 min initially assumed. Three eval cycles = ~65-75 min of wall time lost to evals. Plus actual training rate is ~38.7/hr vs assumed 57/hr.

**Updated champion threshold:** U=400 (36.7%). Next challengers eligible at U>450. All current seeds need U=500 checkpoint for next promotion.

---

### Promotion syncs: 12 clean "no challenger" (15:06–16:23 MT) after double-promotion. Champion U=400 confirmed.

### Pre-gate call stands:
- greedy WR at U=500: ~30-37% (30-game noise) → likely >25% → **PASS**
- CR WR at U=500: 0.000 at every single checkpoint across all seeds, all rounds → **FAIL**
- Gate outcome: **PREDICTED FAIL** (CR gate). Fires ~20:50-21:10 MT.

---

## 2026-06-19 17:28 MT — Cycle 17

### No restart. U=400 evals still pending for 2xl seeds. Gate slips further to ~21:30 MT.

**Restart check:** elapsed_h=9.49–9.54h all primary seeds. ✅

**U=400 evals for 2xl seeds: NOT YET written.** Seeds at U=337–347. Need ~53-63 more updates at 38.7/hr = 1.4-1.6h → **U=400 eval writes ~19:00 MT.**

Then U=500 eval: 100 more updates at 38.7/hr = 2.58h → **U=500 eval writes ~21:35 MT.**

**xl/job1 (champion seed) at U=446, Ent=2.205:** Needs only 54 more updates to U=500. At xl rate 18.5/hr = 2.9h → **xl/job1 U=500 eval fires ~20:23 MT** (earliest gate trigger).

Gate now expected: **~20:23 MT (xl/job1) or ~21:35 MT (2xl seeds)**, whichever counts as "best seed."

---

### No new promotions. 8 clean syncs (16:23–17:23 MT). Champion U=400 WR=36.7% holds.

Challenger eligibility window: new threshold = champion U=400 + 50 = 450.
- xl/job1 at U=446: next eval at U=500 (>450 ✓ — FIRST eligible candidate after gate, or IS the gate if it's the best seed)
- xl/job2 at U=344, WR=0.400@U=300, Ent=1.984: needs U=500 checkpoint, ETA ~20:28 MT for U=400 eval then further
- xl/job3 at U=446, Ent=1.184 (very low): WR=0.300@U=400 — likely below champion; won't promote
- 2xl seeds: need U=500 checkpoint → ETA ~21:35 MT

---

### Entropy state at U=337-347

Significant: .174.18/job2 Ent=1.885 — this is the seed that held 0.367 at U=300 (best on that host). Very fast entropy decay. If committed to a GOOD strategy (like job3 was, which then collapsed at U=300), its U=400 will confirm or deny.

| Seeds below Ent=2.0 | Ent | Last WR | Risk |
|---------------------|-----|---------|------|
| .171.142/job3 | 1.579 | 0.300@U=300 | collapsed |
| .174.18/job4 | 1.691 | 0.267@U=300 | worst in fleet |
| .174.18/job2 | 1.885 | 0.367@U=300 | AMBIGUOUS |
| .171.142/job2 | 2.176 | 0.333@U=300 | declining |
| .171.142/job4 | 2.478 | 0.367@U=300 | best trajectory |

xl/job3 Ent=1.184 (up slightly from 0.920 last cycle — entropy is NOT monotonically decreasing; small oscillation is normal).

---

### Gate pre-call summary (unchanged):

All CR evals = 0.000. 30+ eval points across all seeds. ~600 total CR games played (12 seeds × 3+ checkpoints × 20 games). Zero wins vs comet_reaper.

**Greedy gate at U=500:** likely 30-40% (champion trending up: 33.3% → 36.7%) → **PASS**
**CR gate at U=500:** 0/20 games with overwhelming evidence of 0% true WR → **FAIL**
**Pre-committed gate outcome: FAIL**

Gate fires at champion xl/job1 U=500: ~20:23 MT. Ted should be available.

---

## 2026-06-19 18:30 MT — Cycle 18

### No restart. U=400 evals writing shortly. Gate on xl/job1 at ~20:40 MT. xl/job3 Ent=0.710.

**Restart check:** elapsed_h=10.53–10.57h. ✅

**U=400 evals for 2xl seeds: NOT YET written.** Seeds at U=355–365. Need ~35-45 more updates at 38.7/hr = ~55-70 min. U=400 evals write ~**19:25-19:40 MT**.

**xl/job1 (champion): U=466, Ent=2.312.** Needs 34 more updates to U=500 at 18.5/hr = 1.84h. U=500 eval fires + 20-25 min eval → **gate on xl/job1 at ~20:40-20:45 MT.**

**xl/job3: Ent=0.710 at U=466.** New campaign entropy low. Near-deterministic policy. WR=0.300@U=400. This is terminal commitment to a wrong strategy. Its U=500 result will be ≤ 0.300.

**xl/job2: U=364, Ent=1.937, WR=0.400@U=300.** From U=364 to U=500 via xl rate (~18.5/hr × 136 updates) = 7.4h → ~02:00 MT Jun 20. NOT a gate candidate.

---

### No new promotions. 8 clean syncs (17:22–18:23 MT). Champion U=400 WR=0.367 holds.

Next challenger requires U > 450 checkpoint. Earliest eligible:
- xl/job1 at U=500 (champion seed) — gate trigger at ~20:40 MT
- 2xl seeds at U=500 — ~22:25-22:35 MT

---

### Entropy collapse tracking

| Seed | Ent@now | Last WR | Pattern |
|------|---------|---------|---------|
| xl/job3 | **0.710** | 0.300@U=400 | Near-zero — worst in fleet |
| .171.142/job3 | 1.305 | 0.300@U=300 | Collapsed |
| .174.18/job4 | 1.663 | 0.267@U=300 | Worst 2xl WR |
| .174.18/job2 | **1.792** | 0.367@U=300 | WATCH: still has the best recent WR of fast-decaying seeds |
| .171.142/job2 | 2.520 | 0.333@U=300 | Declining |

.174.18/job2 at Ent=1.792 with WR=0.367 at U=300 is the only fast-decaying seed that might sustain performance. Its U=400 eval (writing ~19:30 MT) will be critical.

---

### Gate countdown

| Event | ETA |
|-------|-----|
| 2xl U=400 evals write | ~19:30–19:45 MT |
| xl/job1 reaches U=500 | ~20:19 MT |
| xl/job1 U=500 eval completes | ~**20:40-20:45 MT** |
| **GATE FIRES** | **~20:40-20:45 MT** |
| 2xl seeds U=500 eval writes | ~22:25-22:35 MT |

**Gate verdict preview:** greedy WR → PASS (xl/job1 trending 33.3%→36.7%, will likely stay >25%). CR WR → FAIL (0/20 games at every eval point, true WR ≈ 0%). **Pre-committed gate: FAIL.**

---

## 2026-06-19 19:32 MT — Cycle 19

### No restart. Gate 68 min out. U=400 evals not yet written. xl/job1 at U=485.

**Restart check:** elapsed_h=11.56–11.61h. ✅

**U=400 evals for 2xl seeds: NOT YET written.** Seeds at U=373–383. Need ~17-27 more updates at 38.7/hr = ~26-42 min. Evals write ~**20:00-20:15 MT**.

**xl/job1 (champion seed): U=485, Ent=1.994.** Needs 15 more updates to U=500 at 18.5/hr = ~49 min. U=500 eval fires ~20:21 MT → eval runs 20-25 min → **GATE FIRES ~20:41-20:46 MT.**

No U=500 eval written yet. Gate has NOT yet fired.

---

### No new promotions. 8 clean syncs (18:11–19:23 MT). No challengers.

---

### Entropy sweep — 7 seeds now below Ent=2.0

| Seed | Ent@now | Last WR | Status |
|------|---------|---------|--------|
| xl/job3 | 1.252 | 0.300@U=400 | Worst XL — collapsed |
| .171.142/job1 | **1.165** | 0.267@U=300 | Worst 2xl WR + low Ent |
| .171.142/job3 | 1.275 | 0.300@U=300 | Collapsed |
| .174.18/job2 | **1.318** | 0.367@U=300 | Best-WR fast-decay seed |
| .174.18/job4 | 1.413 | 0.267@U=300 | Collapsed |
| xl/job1 | 1.994 | 0.367@U=400 | Champion seed, gate trigger |
| xl/job2 | 2.013 | 0.400@U=300 | Just above threshold |

.174.18/job2 at Ent=1.318 is the most critical watch: it had the second-best trajectory at U=300 (36.7%) and is now committing very fast. Its U=400 eval writes in ~30 min and will be the last data point before the gate.

---

### Gate status

- xl/job1 U=500 eval: fires ~20:21 MT, completes ~20:41-20:46 MT
- 2xl U=400 evals: write ~20:00-20:15 MT (20-25 min before gate fires)
- 2xl U=500 evals: write ~22:25-22:35 MT (post-gate confirmation)
- CR at every eval point: 0.000 — no change possible at gate
- **Gate verdict: PREDICTED FAIL (CR gate)**

---

## 2026-06-19 20:18 MT — Cycle 20 — GATE EVAL IN PROGRESS

### xl/job1 AT U=500. Gate eval running. Results in ~15-20 min.

**Restart check:** elapsed_h=12.27–12.37h. ✅

---

### GATE STATUS — ACTIVE

xl/job1 (champion seed): `now: U=500, elapsed=12.35h, Ent=1.907`
U=500 eval row NOT YET written — eval games in progress.
Expected write: **~20:35-20:40 MT.**

xl/job3: also at U=500, Ent=0.696 (extreme collapse). WR=0.300@U=400. Not a gate trigger (below champion WR).

**All CR evals = 0.000. Gate pre-call: FAIL remains.**

---

### 2xl U=400 evals: NOT YET written. Seeds at U=387-396.

ETA for U=400 eval write: ~4-13 more updates at 38.7/hr = ~6-20 min → **~20:25-20:40 MT.**

These will write simultaneously with or just after the gate result. CR will be 0.000 for all.

| Seed | Last WR | Now U | Ent |
|------|---------|-------|-----|
| .170.73/job3 | 0.367@U=300 | 396 | 3.408 |
| .174.18/job2 | 0.367@U=300 | 387 | 1.810 |
| .171.142/job4 | 0.367@U=300 | 395 | 1.734 |

The three 0.367 seeds are about to write their U=400 results. .174.18/job2 Ent=1.810 (fast decay, last 3 at same WR) is the most uncertain.

---

### No new promotions. 8 clean syncs (19:23–20:15 MT). No challengers.

---

### GATE RESULT — PENDING (write in ~15-20 min)

Watching for xl/job1 U=500 eval row. Will report result to Ted immediately upon read.

---

## 2026-06-19 20:40 MT — Cycle 21 — GATE PARTIAL DATA

### xl/job1 U=500 eval STILL RUNNING. xl/job3 U=500 written: 23.3% greedy, 0% CR.

**Restart check:** elapsed_h=12.35–12.74h. ✅

---

### xl host .175.105 state:

| Seed | U=300 | U=400 | U=500 | Now U | Ent |
|------|-------|-------|-------|-------|-----|
| job1 (CHAMPION) | 0.333 | 0.367 | **RUNNING** | 500 | 1.907 |
| job2 | 0.400 | — | — | 400 | 1.816 |
| job3 | 0.333 | 0.300 | **0.233** ep=500 | 502 | 0.969 |
| job4 | 0.300 | — | — | 400 | 3.693 |

**xl/job3 at U=500:** g=0.233, CR=0.000, ep=500.
This is BELOW the greedy gate threshold (0.233 < 0.25). ep=500 = max stall strategy fully locked in.
Trajectory: 33.3% → 33.3% → 30.0% → **23.3%** — monotonic decline.
xl/job3 would be a DOUBLE FAIL (greedy AND CR).

xl/job1 eval: still running at 22 min elapsed (expected 20-25 min). Should write imminently.

---

### 2xl U=400 evals: RUNNING NOW (not yet written)

All primary 2xl seeds at U=394-400. Most showing "now: U=400" — in eval at this moment.
Earliest write: ~21:00-21:05 MT.

Best 2xl candidates at U=300: .170.73/job3 (36.7%), .174.18/job2 (36.7%), .171.142/job4 (36.7%).
All three about to write U=400 results.

---

### No new promotions. 5 syncs (19:59-20:35). No challengers.

---

## 2026-06-19 20:57 MT — Cycle 22 — GATE RESULT

```
═══════════════════════════════════════════════════════════════
PRE-COMMITTED GATE VERDICT — 2026-06-19 20:57 MT
Seed: xl/job1 (ppo-1-of-3_job1) — champion seed, best performer

  Greedy WR at U=500:   36.7%   [threshold >25%  → PASS ✓]
  Comet WR at U=500:     0.0%   [threshold >0%   → FAIL ✗]
  Episode length (wins): 350 steps (real combat, not stall)

  Condition: BOTH must pass.
  Result: FAIL — CR gate not met.
═══════════════════════════════════════════════════════════════
```

**ep=350 note:** The champion seed wins its greedy games at 350 steps (real combat). This distinguishes it from the collapsed seeds (ep=500 stall strategy). It learned real fighting strategies but cannot beat comet_reaper.

---

### Full campaign CR record (pre-committed gate evidence)

Every eval checkpoint, every seed, every CR result:

| Checkpoint | Seeds | Games each | Wins |
|------------|-------|------------|------|
| U=100 (all 12 primary) | 12 | 20 | 0 |
| U=200 (all 12 primary) | 12 | 20 | 0 |
| U=300 (all 12 primary) | 12 | 20 | 0 |
| U=400 (xl/job1,job3,job4) | 3 | 20 | 0 |
| U=500 (xl/job1,job3) | 2 | 20 | 0 |
| xl/job2 (U=100,200,300,400) | 4 | 20 | 0 |
| **Total** | **~45 eval rounds** | **~900 games** | **0** |

**0 wins vs comet_reaper in ~900 evaluation games.** The true CR WR is effectively 0%.

---

### Champion greedy trajectory

| Checkpoint | Champion | Greedy WR | CR WR | Notes |
|------------|---------|-----------|-------|-------|
| Overnight run | U=100 | 32.5% | 0% | Original seed |
| Post-Restart 2 | U=100 | 32.5%→fleet avg 36.7% | 0% | New run seeds |
| **PROMOTION 1** | U=300 | 33.3% | 0% | xl/job1 |
| **PROMOTION 2** | U=400 | 36.7% | 0% | xl/job1 |
| **GATE** | U=500 | **36.7%** | **0%** | xl/job1 (champion) |

Greedy ceiling: ~36-37% (30-game, high noise). Held flat from U=400 to U=500 — no further improvement. CR ceiling: exactly 0%.

---

### Why CR=0% persisted

The RL agents learned to beat the greedy bot at ~37% WR. They could NOT beat comet_reaper at any update depth (U=100 through U=500) across all seeds. This is consistent with the v6 hypothesis failure: PPO from scratch, even with 500 updates and mixed cold-start vs comet_reaper, cannot crack comet_reaper's combinatorial planning. The training signal from the game reward is insufficient to develop the long-horizon look-ahead that comet_reaper uses.

---

### 2xl U=400 evals (running now — will write ~21:15 MT)

All 12 primary 2xl seeds at U=400, running evals. Best candidates (.170.73/job3, .174.18/job2, .171.142/job4) held 36.7% at U=300 — consistent with xl/job1's ceiling but not exceeding it. Confirmation data only; gate is called.

### ORCHESTRATOR_STATE: 5 syncs through 20:47 MT. No promotions. No challengers.

---

## AUDITOR RECOMMENDATION TO TED

**GATE: FAIL. The pre-committed RL experiment condition is not met.**

Greedy WR passed (36.7% > 25%) but means nothing if comet_reaper cannot be beaten.

**Immediate action required:**
1. **Submit comet_reaper (sub 53707586) before June 22 backstop** — 3 days remain. Current selected submission is schmeekler_fmt (~1149 Elo). comet_reaper at ~1235 Elo needs to be re-submitted to restore the protected floor.
2. The RL fleet can continue running (Ted's prior call: "we need to continue the RL training loop") but without expectation of gate passage. If any subsequent CR result comes in >0%, report immediately.
3. No new orchestrator actions recommended from the audit side.

---

## 2026-06-19 21:59 MT — Cycle 23 — Post-Gate Confirmation

### No restart. 2xl U=400 distribution confirms ceiling. Gate verdict stands: FAIL.

**Restart check:** elapsed_h=14.02–14.06h. ✅

---

### 2xl U=400 full distribution (30-game)

| Seed | g@100 | g@200 | g@300 | **g@400** | ep@400 | Ent |
|------|-------|-------|-------|-----------|--------|-----|
| .170.73/job1 | 0.433 | 0.467 | 0.300 | 0.333 | **500** | 2.407 |
| .170.73/job2 | 0.367 | 0.367 | 0.300 | 0.267 | 391 | 2.219 |
| .170.73/job3 | 0.333 | 0.367 | 0.367 | 0.333 | 386 | 3.301 |
| .170.73/job4 | 0.300 | 0.333 | 0.300 | 0.300 | 385 | 2.995 |
| .174.18/job1 | 0.367 | 0.367 | 0.300 | 0.267 | **500** | 2.402 |
| .174.18/job2 | 0.400 | 0.433 | 0.367 | 0.300 | 435 | 1.401 |
| .174.18/job3 | 0.333 | 0.400 | 0.300 | **0.367** | 446 | 2.927 |
| .174.18/job4 | 0.400 | 0.300 | 0.267 | 0.267 | 468 | 1.490 |
| .171.142/job1 | 0.267 | 0.267 | 0.267 | 0.300 | 387 | 1.564 |
| .171.142/job2 | 0.433 | 0.400 | 0.333 | 0.333 | 376 | 1.870 |
| .171.142/job3 | 0.433 | 0.433 | 0.300 | 0.267 | 476 | **0.764** |
| .171.142/job4 | 0.300 | 0.400 | 0.367 | 0.300 | 421 | 1.524 |

**Stats:** Max=0.367 (1 seed), Median=0.300, Min=0.267
- Beats champion WR (0.367): 0/12 — no 2xl seed exceeds champion
- At champion WR (0.367): 1/12 (.174.18/job3) — ties, ineligible as challenger
- ep=500 (stall): 2 seeds (.170.73/job1, .174.18/job1) — fully locked in

**All CR=0.000 at U=400.** Total CR games now ~900+ with zero wins.

**Trend:** Median flat at 0.300 from U=300 to U=400. 10 of 12 seeds declined or held at U=400.

**Confirmed:** Greedy ceiling of ~36-37% is real and consistent across 2xl and xl hosts, across U=300-500. No 2xl seed improved beyond U=200's best result (46.7% was noise on .170.73/job1 that returned to 30-33% by U=300-400).

---

### No new promotions. 5 syncs (21:23–21:58 MT). Champion U=400 holds.

---

*Gate verdict stands: FAIL. Fleet continues per Ted's call. Any CR > 0% will escalate immediately.*

---
*Next (final) auditor read: ~22:30 MT — 2xl U=500 confirmation data*

---

## 2026-06-19 22:31 MT — Cycle 24 — FINAL FLEET READ

**No restart. 2xl U=500 not yet written (overnight). CR=0 across all seeds. Campaign closed.**

Restart check: elapsed_h=14.54–14.60h on all active seeds. ✅ Dead wave-2 .174.18 seeds still stuck at U=12-13.

2xl U=500 status: Evals NOT yet written. All 2xl seeds at U=415–426. Rate ~29 updates/hr → U=500 evals write ~01:00–02:00 MT Jun 20. Confirmation-only; gate is already called.

xl update: job1 U=531, job3 U=538 (Ent=0.502, further collapse), job2 U=430, job4 U=432.

All CR=0.000 everywhere. ORCHESTRATOR_STATE: 5 clean syncs (21:47–22:22 MT). Champion U=400, 36.7%, 0% CR. No promotions, no challengers.

---

## FINAL AUDITOR REPORT — v6 RL Campaign

**Gate: FAIL (called 20:57 MT Jun 19). Monitoring: CLOSED.**

| Metric | Value |
|--------|-------|
| Total CR eval games | ~900+ (~45 rounds × 20 games) |
| CR wins | **0** |
| Greedy ceiling (best seed, U=500) | **36.7%** (xl/job1, ep=350 — real combat) |
| Fleet median at U=400 | 30.0% |
| ep=500 stall seeds | ≥3 locked in |
| Entropy-collapsed seeds | ≥2 (xl/job3 Ent=0.502, .171.142/job3 Ent=0.764) |
| Champion at close | U=400, 36.7%, 0% CR |

What was needed: greedy >25% **AND** CR >0%. Greedy passed; CR failed by every measure.

### Auditor's closing recommendation to Ted

1. **Submit comet_reaper (sub 53707586) before June 22 backstop.** 2 days remain. "Highest of latest 2 submissions" rule — this restores the ~1235 Elo floor.
2. RL fleet may continue idle. No further auditor cycles unless CR >0% appears or Ted extends scope.
3. 2xl U=500 data will write ~01:00–02:00 MT but is confirmation-only — trajectory guarantees CR=0.

**24 cycles. 2026-06-18 evening through 2026-06-19 22:31 MT. Audit complete.**

---
*AUDITOR_LOG.md closed — monitoring ended*
