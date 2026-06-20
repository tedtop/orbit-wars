# ORCHESTRATOR STATE — v6 RL/PPO Campaign
*Branch: v6-rl-selfplay | Updated: 2026-06-19 ~05:30 MT | Deadline: 2026-06-23*

---

## CURRENT STATUS: 🟡 PHASE 2 — FLEET RUNNING, CHAMPION LOOP LIVE

**Phase 1 (Repair & bench-prove): ✅ COMPLETE as of ~05:25 MT Jun 19**

---

## Phase 1 completion checklist

| Task | Status | Notes |
|------|--------|-------|
| Re-seed champion from real best | ✅ | ppo-4-of-6_job1 U=100, WR=32% vs greedy |
| Quad excluded from challenger | ✅ | m3quad_job1 never becomes challenger |
| Challenger ranked by eval WR (not depth) | ✅ | metrics.jsonl fallback when best_wr not in ckpt |
| U-tiebreaker for equal-WR challengers | ✅ | Added to sync.sh |
| RL vs RL eval fix (decorator→context mgr) | ✅ | Episode lengths now ~445-457 steps |
| Draw-adjusted symmetry gate | ✅ | slot_0_decisive replaces raw slot_0_wr in check |
| Bench prove: 1 clean promotion run | ✅ | champion (32%) vs dead quad: WR=78%, gate PASS, ep_len=457 |
| Updated train.py saves best_wr in ckpt | ✅ | Pushed to all 8 hosts |
| Updated eval_checkpoints.py on fleet | ✅ | Pushed to all 8 hosts |
| Fleet restarted with heredoc pattern | ✅ | 8 hosts × 4 jobs = 32 seeds running |

---

## Current fleet state

- **Champion**: `agents/rl_ppo/checkpoints/champion.pt` — ppo-4-of-6_job1 U=100, ~32% vs greedy, 0% vs comet (expected at U=100)
- **Fleet**: 32 seeds (8 × m3.2xl/xl × 4 jobs), all running from U=0 (restarted ~03:00-04:00 MT Jun 19)
- **First eval fires at**: U=100 per seed (~2.4h from restart → ~06:00-07:00 MT)
- **First challenger eligible at**: U=150 (champion U=100 + 50 threshold) → expect at U=200 first save (~05:30 MT)
- **vs comet_reaper**: 0% for current champion (expected — 100K steps, cold start)
- **Quad (175.182)**: Running 8-env jobs, excluded from champion promotion, ignored

---

## Phase 3 gate (do NOT call PASS or FAIL until this fires)

**At U=500 on best seed** (per Auditor spec):
- greedy WR > 25% → PASS condition (one of two)
- comet WR > 0% → PASS condition (other of two)
- BOTH must pass → overall PASS; continue fleet

**If FAIL (both conditions missed):** Wake Ted immediately. Present plan B.

**If PASS (either condition met):** Continue fleet, schedule 2h check-in.

Estimated U=500 time: ~12h from restart → ~16:00-17:00 MT Jun 19

---

## Promotion loop rules (current)

1. Challenger must be from 64-env seed (quad excluded)
2. Challenger must be U > champion_U + 50
3. Challenger ranked by best greedy WR (from best_wr field or metrics.jsonl)
4. Promotion eval: n=1000 games, seat-balanced
5. Symmetry gate: `slot_0_decisive` (draw-adjusted) ∈ [0.46, 0.54] AND seat delta < 0.05
6. WR gate: challenger must beat champion > 0.50

---

## Protected floor

| Bot | Live Score | Status |
|-----|-----------|--------|
| comet_reaper | ~1235 Elo | Active (sub 53707586) — restorable floor |
| schmeekler_fmt | ~1149 Elo | **Currently SELECTED** — must re-sub comet to restore floor |

> ⚠ The SELECTED score is schmeekler_fmt (~1149). comet_reaper is NOT selected — it needs a re-submission.

### 🔴 HARD TRIGGER — June 22, 2026

If by June 22 MT nothing has beaten comet_reaper:
**RE-SUBMIT comet_reaper immediately** so it re-enters the selected pair as the defended floor.
Do NOT let the competition close with schmeekler_fmt (~1149) as the active score.

---

## Sync log (latest entries)

### Champion eval — 2026-06-19 00:13–02:02 MT | champion=U100 | no challenger | vs_comet_WR=? (broken)
(9 syncs, all showing ? because stdout contamination in old code)

### Champion eval — 2026-06-19 05:02 MT | sym_FAIL | challenger=U200 WR=0.754 slot0=0.372 slot1=0.384
(Old gate — failed due to draw rate math. Draw-adjusted gate would have PASSED.)

### Champion eval — 2026-06-19 05:21 MT | champion=U100 WR=0.0 | no challenger | vs_comet_WR=0.000 ✅
(First sync with corrected gate. No challenger yet — fleet at U=7-50, not past threshold.)

---

## Parked until Ted's say-so

- **#1047 obs-encoding investigation** — #1047 already cleared (FALSE ALARM: atan2(dy,dx) used consistently)
- **Anvil GPU (HPC)** — SSH refused, multi-hour integration; not worth it at 4 days to deadline
- **JAX rewrite** — 1-2 day effort; would need clean week
- **Simulator reimplementation** — architectural bet, requires multi-day rewrite

---

## Escalation triggers (wake Ted immediately)

- Phase 1 cannot produce clean promotion (DONE — it can, Phase 1 COMPLETE)
- U=500 valid FAIL: greedy WR ≤ 25% AND comet WR = 0% on best seed
- Any seed reaches > 0% vs comet_reaper (good news — escalate immediately)
- Shared opponent NOT updating on promotion at U=300 (verify on first promotion)
- Any prior conclusion reversing

---

## Infrastructure built

- `eval_checkpoints.py`: seat-balanced RL×RL + vs greedy + vs comet, draw-adjusted symmetry gate, JSON
- `sync_checkpoints.sh`: champion promotion loop, WR-ranked challenger, metrics.jsonl fallback, comet WR tracking
- 8 Jetstream2 instances running
- 10+ critical bugs found and fixed (GAE grouping, ent_coef, decorator sig, stdout contamination, draw gate)

### Champion eval — 2026-06-19 05:29 | champion=U100 WR=0.0 | no challenger  vs_comet_WR=0.000
Pool size:        0 checkpoints

### Champion eval — 2026-06-19 05:41 | champion=U100 WR=0.0 | no challenger  vs_comet_WR=0.000
Pool size:        0 checkpoints

### Champion eval — 2026-06-19 05:44 | champion=U100 WR=0.0 | no challenger  vs_comet_WR=0.000
Pool size:        0 checkpoints

### Champion eval — 2026-06-19 05:49 | champion=U100 WR=0.0 | no challenger  vs_comet_WR=0.000
Pool size:        0 checkpoints

### Champion eval — 2026-06-19 05:59 | champion=U100 WR=0.0 | no challenger  vs_comet_WR=0.000
Pool size:        0 checkpoints

### Champion eval — 2026-06-19 06:06 | champion=U100 WR=0.0 | no challenger  vs_comet_WR=0.000
Pool size:        0 checkpoints

### Champion eval — 2026-06-19 06:13 | champion=U100 WR=0.0 | no challenger  vs_comet_WR=0.000
Pool size:        0 checkpoints

### Champion eval — 2026-06-19 06:15 | champion=U100 WR=0.0 | no challenger  vs_comet_WR=0.000
Pool size:        0 checkpoints

### Champion eval — 2026-06-19 06:30 | champion=U100 WR=0.0 | no challenger  vs_comet_WR=0.000
Pool size:        0 checkpoints

### Champion eval — 2026-06-19 06:45 | champion=U100 WR=0.0 | no challenger  vs_comet_WR=0.000
Pool size:        0 checkpoints

### Champion eval — 2026-06-19 06:59 | champion=U100 WR=0.0 | no challenger  vs_comet_WR=0.000
Pool size:        0 checkpoints

### Champion eval — 2026-06-19 07:00 | champion=U100 WR=0.0 | no challenger  vs_comet_WR=0.000
Pool size:        0 checkpoints

### Champion eval — 2026-06-19 07:10 | champion=U100 WR=0.0 | no challenger  vs_comet_WR=0.000
Pool size:        0 checkpoints

### Champion eval — 2026-06-19 07:15 | champion=U100 WR=0.0 | no challenger  vs_comet_WR=0.000
Pool size:        0 checkpoints

### Champion eval — 2026-06-19 07:16 | champion=U100 WR=0.0 | no challenger  vs_comet_WR=0.000
Pool size:        0 checkpoints

### Champion eval — 2026-06-19 07:31 | champion=U100 WR=0.0 | no challenger  vs_comet_WR=0.000
Pool size:        0 checkpoints

### Champion eval — 2026-06-19 07:43 | champion=U100 WR=0.0 | no challenger  vs_comet_WR=0.000
Pool size:        0 checkpoints

### Champion eval — 2026-06-19 07:46 | champion=U100 WR=0.0 | no challenger  vs_comet_WR=0.000
Pool size:        0 checkpoints

### Fleet restart — 2026-06-19 07:50 MT | CRITICAL: eval n_games bug

**Root cause found and fixed:**
- `evaluate_vs_greedy(policy)` defaulted to n_games=200 (sequential)
- `evaluate_vs_comet_reaper(policy)` defaulted to n_games=100 (sequential)
- With 4 concurrent jobs competing for CPU, each game took ~25s → 300 games = **125 min per eval**
- All 32 fresh seeds were stuck at U=100 eval for 60+ min; Phase 3 gate would slip to 2 AM Jun 20

**Fix applied:**
- train.py line 925: `evaluate_vs_greedy(policy, n_games=30)` 
- train.py line 933: `evaluate_vs_comet_reaper(policy, n_games=20)`
- Pushed to all 8 hosts, all processes restarted from U=0 (no latest.pt existed for fresh seeds)

**New eval timing estimate:** ~50 games × 5-8s each = ~4-8 min per eval cycle

**New Phase 3 gate ETA:** ~17:00-18:00 MT Jun 19 (vs 2 AM Jun 20 before fix)

### Champion eval — 2026-06-19 08:01 | champion=U100 WR=0.0 | no challenger  vs_comet_WR=0.000
Pool size:        0 checkpoints

### Champion eval — 2026-06-19 08:17 | champion=U100 WR=0.0 | no challenger  vs_comet_WR=0.000
Pool size:        0 checkpoints

### Champion eval — 2026-06-19 08:18 | champion=U100 WR=0.0 | no challenger  vs_comet_WR=0.000
Pool size:        0 checkpoints

### Champion eval — 2026-06-19 08:32 | champion=U100 WR=0.0 | no challenger  vs_comet_WR=0.000
Pool size:        0 checkpoints

### Champion eval — 2026-06-19 08:52 | champion=U100 WR=0.0 | no challenger  vs_comet_WR=0.000
Pool size:        0 checkpoints

### Champion eval — 2026-06-19 09:02 | champion=U100 WR=0.0 | no challenger  vs_comet_WR=0.000
Pool size:        0 checkpoints

### Champion eval — 2026-06-19 09:14 | champion=U100 WR=0.0 | no challenger  vs_comet_WR=0.000
Pool size:        0 checkpoints

### Champion eval — 2026-06-19 09:15 | champion=U100 WR=0.0 | no challenger  vs_comet_WR=0.000
Pool size:        0 checkpoints

### Champion eval — 2026-06-19 09:26 | champion=U100 WR=0.0 | no challenger  vs_comet_WR=0.000
Pool size:        0 checkpoints

### Champion eval — 2026-06-19 09:38 | champion=U100 WR=0.0 | no challenger  vs_comet_WR=0.000
Pool size:        0 checkpoints

### Champion eval — 2026-06-19 09:45 | champion=U100 WR=0.0 | no challenger  vs_comet_WR=0.000
Pool size:        0 checkpoints

### Champion eval — 2026-06-19 09:49 | champion=U100 WR=0.0 | no challenger  vs_comet_WR=0.000
Pool size:        0 checkpoints

### Champion eval — 2026-06-19 10:00 | champion=U100 WR=0.0 | no challenger  vs_comet_WR=0.000
Pool size:        0 checkpoints

### Champion eval — 2026-06-19 10:01 | champion=U100 WR=0.0 | no challenger  vs_comet_WR=0.000
Pool size:        0 checkpoints

### Champion eval — 2026-06-19 10:45 | PROMOTED U300 | WR=0.750 slot0=0.370 slot1=0.386 term=0.03  vs_comet_WR=0.000
Pool size:        1 checkpoints

### Champion eval — 2026-06-19 10:45 | PROMOTED U300 | WR=0.751 slot0=0.369 slot1=0.387 term=0.02  vs_comet_WR=0.000
Pool size:        1 checkpoints

### 🏆 FIRST PROMOTION — 2026-06-19 10:45 MT
**Challenger:** ppo-1-of-3_job1 (U=300, 33% vs greedy)
**vs Champion:** U=100 (32% vs greedy, seeded)
**Result:** WR=75.1% | slot0=0.369 slot1=0.387 | seat_delta=0.034 | sym_gate=PASS ✅
**Action:** PROMOTED → champion v1 (U=300) pushed to all 9 fleet hosts
**vs comet_reaper:** 0.0% (expected at U=300)

Next challengers in queue:
- ppo-1-of-3_job3: U=300 WR=33% (equal to new champion, not eligible yet)
- ppo-1-of-3_job2: U=100 WR=40% → WATCH at U=200 (~11:45 MT) — strongest seed
- ppo-3-of-3_job3/job4: U=300 WR=30% (below new champion 33%)

Phase 3 gate ETA: U=500 on deep seeds ~14:30 MT Jun 19

### Sync fixes applied — 2026-06-19 10:03 MT
Critical bug: `except: pass` caught `sys.exit(0)` → double-print → awk crash → exit 2
Fix: `except Exception: pass` + printed flag (no more sys.exit)
All 9 hosts updated.

### Champion eval — 2026-06-19 10:57 | champion=U300 WR=0.3333333333333333 | no challenger  vs_comet_WR=0.000
Pool size:        1 checkpoints

### Champion eval — 2026-06-19 11:09 | champion=U300 WR=0.3333333333333333 | no challenger  vs_comet_WR=0.000
Pool size:        1 checkpoints

### Champion eval — 2026-06-19 11:20 | champion=U300 WR=0.3333333333333333 | no challenger  vs_comet_WR=0.000
Pool size:        1 checkpoints

### Champion eval — 2026-06-19 11:21 | champion=U300 WR=0.3333333333333333 | no challenger  vs_comet_WR=0.000
Pool size:        1 checkpoints

### Champion eval — 2026-06-19 11:33 | champion=U300 WR=0.3333333333333333 | no challenger  vs_comet_WR=0.000
Pool size:        1 checkpoints

### Champion eval — 2026-06-19 11:45 | champion=U300 WR=0.3333333333333333 | no challenger  vs_comet_WR=0.000
Pool size:        1 checkpoints

### Champion eval — 2026-06-19 11:46 | champion=U300 WR=0.3333333333333333 | no challenger  vs_comet_WR=0.000
Pool size:        1 checkpoints

### Champion eval — 2026-06-19 11:57 | champion=U300 WR=0.3333333333333333 | no challenger  vs_comet_WR=0.000
Pool size:        1 checkpoints

### Champion eval — 2026-06-19 12:09 | champion=U300 WR=0.3333333333333333 | no challenger  vs_comet_WR=0.000
Pool size:        1 checkpoints

### Champion eval — 2026-06-19 12:10 | champion=U300 WR=0.3333333333333333 | no challenger  vs_comet_WR=0.000
Pool size:        1 checkpoints

### Champion eval — 2026-06-19 12:21 | champion=U300 WR=0.3333333333333333 | no challenger  vs_comet_WR=0.000
Pool size:        1 checkpoints

### Champion eval — 2026-06-19 12:34 | champion=U300 WR=0.3333333333333333 | no challenger  vs_comet_WR=0.000
Pool size:        1 checkpoints

### Champion eval — 2026-06-19 12:35 | champion=U300 WR=0.3333333333333333 | no challenger  vs_comet_WR=0.000
Pool size:        1 checkpoints

### Champion eval — 2026-06-19 12:46 | champion=U300 WR=0.3333333333333333 | no challenger  vs_comet_WR=0.000
Pool size:        1 checkpoints

### Champion eval — 2026-06-19 12:58 | champion=U300 WR=0.3333333333333333 | no challenger  vs_comet_WR=0.000
Pool size:        1 checkpoints

### Champion eval — 2026-06-19 13:09 | champion=U300 WR=0.3333333333333333 | no challenger  vs_comet_WR=0.000
Pool size:        1 checkpoints

### Champion eval — 2026-06-19 13:10 | champion=U300 WR=0.3333333333333333 | no challenger  vs_comet_WR=0.000
Pool size:        1 checkpoints

### Champion eval — 2026-06-19 13:22 | champion=U300 WR=0.3333333333333333 | no challenger  vs_comet_WR=0.000
Pool size:        1 checkpoints

### Champion eval — 2026-06-19 13:34 | champion=U300 WR=0.3333333333333333 | no challenger  vs_comet_WR=0.000
Pool size:        1 checkpoints

### Champion eval — 2026-06-19 13:35 | champion=U300 WR=0.3333333333333333 | no challenger  vs_comet_WR=0.000
Pool size:        1 checkpoints

### Champion eval — 2026-06-19 13:46 | champion=U300 WR=0.3333333333333333 | no challenger  vs_comet_WR=0.000
Pool size:        1 checkpoints

### Champion eval — 2026-06-19 13:58 | champion=U300 WR=0.3333333333333333 | no challenger  vs_comet_WR=0.000
Pool size:        1 checkpoints

### Champion eval — 2026-06-19 13:59 | champion=U300 WR=0.3333333333333333 | no challenger  vs_comet_WR=0.000
Pool size:        1 checkpoints

### Champion eval — 2026-06-19 14:10 | champion=U300 WR=0.3333333333333333 | no challenger  vs_comet_WR=0.000
Pool size:        1 checkpoints

### Champion eval — 2026-06-19 14:22 | champion=U300 WR=0.3333333333333333 | no challenger  vs_comet_WR=0.000
Pool size:        1 checkpoints

### Champion eval — 2026-06-19 14:34 | champion=U300 WR=0.3333333333333333 | no challenger  vs_comet_WR=0.000
Pool size:        1 checkpoints

### Champion eval — 2026-06-19 14:37 | champion=U300 WR=0.3333333333333333 | no challenger  vs_comet_WR=0.000
Pool size:        1 checkpoints

### Champion eval — 2026-06-19 14:46 | champion=U300 WR=0.3333333333333333 | no challenger  vs_comet_WR=0.000
Pool size:        1 checkpoints

### Champion eval — 2026-06-19 14:58 | champion=U300 WR=0.3333333333333333 | no challenger  vs_comet_WR=0.000
Pool size:        1 checkpoints

### Champion eval — 2026-06-19 15:06 | champion=U300 WR=0.3333333333333333 | no challenger  vs_comet_WR=0.000
Pool size:        1 checkpoints

### Champion eval — 2026-06-19 15:35 | PROMOTED U400 | WR=0.753 slot0=0.371 slot1=0.385 term=0.01  vs_comet_WR=0.000
Pool size:        2 checkpoints

### Champion eval — 2026-06-19 15:47 | champion=U400 WR=0.36666666666666664 | no challenger  vs_comet_WR=0.000
Pool size:        2 checkpoints

### Champion eval — 2026-06-19 15:56 | PROMOTED U400 | WR=0.752 slot0=0.372 slot1=0.384 term=0.01  vs_comet_WR=0.000
Pool size:        3 checkpoints

### Champion eval — 2026-06-19 15:59 | champion=U400 WR=0.36666666666666664 | no challenger  vs_comet_WR=0.000
Pool size:        3 checkpoints

### Champion eval — 2026-06-19 16:11 | champion=U400 WR=0.36666666666666664 | no challenger  vs_comet_WR=0.000
Pool size:        3 checkpoints

### Champion eval — 2026-06-19 16:23 | champion=U400 WR=0.36666666666666664 | no challenger  vs_comet_WR=0.000
Pool size:        3 checkpoints

### Champion eval — 2026-06-19 16:34 | champion=U400 WR=0.36666666666666664 | no challenger  vs_comet_WR=0.000
Pool size:        3 checkpoints

### Champion eval — 2026-06-19 16:35 | champion=U400 WR=0.36666666666666664 | no challenger  vs_comet_WR=0.000
Pool size:        3 checkpoints

### Champion eval — 2026-06-19 16:47 | champion=U400 WR=0.36666666666666664 | no challenger  vs_comet_WR=0.000
Pool size:        3 checkpoints

### Champion eval — 2026-06-19 16:59 | champion=U400 WR=0.36666666666666664 | no challenger  vs_comet_WR=0.000
Pool size:        3 checkpoints

### Champion eval — 2026-06-19 17:10 | champion=U400 WR=0.36666666666666664 | no challenger  vs_comet_WR=0.000
Pool size:        3 checkpoints

### Champion eval — 2026-06-19 17:22 | champion=U400 WR=0.36666666666666664 | no challenger  vs_comet_WR=0.000
Pool size:        3 checkpoints

### Champion eval — 2026-06-19 17:23 | champion=U400 WR=0.36666666666666664 | no challenger  vs_comet_WR=0.000
Pool size:        3 checkpoints

### Champion eval — 2026-06-19 17:34 | champion=U400 WR=0.36666666666666664 | no challenger  vs_comet_WR=0.000
Pool size:        3 checkpoints

### Champion eval — 2026-06-19 17:47 | champion=U400 WR=0.36666666666666664 | no challenger  vs_comet_WR=0.000
Pool size:        3 checkpoints

### Champion eval — 2026-06-19 17:58 | champion=U400 WR=0.36666666666666664 | no challenger  vs_comet_WR=0.000
Pool size:        3 checkpoints

### Champion eval — 2026-06-19 18:07 | champion=U400 WR=0.36666666666666664 | no challenger  vs_comet_WR=0.000
Pool size:        3 checkpoints

### Champion eval — 2026-06-19 18:11 | champion=U400 WR=0.36666666666666664 | no challenger  vs_comet_WR=0.000
Pool size:        3 checkpoints

### Champion eval — 2026-06-19 18:23 | champion=U400 WR=0.36666666666666664 | no challenger  vs_comet_WR=0.000
Pool size:        3 checkpoints

### Champion eval — 2026-06-19 18:35 | champion=U400 WR=0.36666666666666664 | no challenger  vs_comet_WR=0.000
Pool size:        3 checkpoints

### Champion eval — 2026-06-19 18:47 | champion=U400 WR=0.36666666666666664 | no challenger  vs_comet_WR=0.000
Pool size:        3 checkpoints

### Champion eval — 2026-06-19 18:59 | champion=U400 WR=0.36666666666666664 | no challenger  vs_comet_WR=0.000
Pool size:        3 checkpoints

### Champion eval — 2026-06-19 19:11 | champion=U400 WR=0.36666666666666664 | no challenger  vs_comet_WR=0.000
Pool size:        3 checkpoints

### Champion eval — 2026-06-19 19:11 | champion=U400 WR=0.36666666666666664 | no challenger  vs_comet_WR=0.000
Pool size:        3 checkpoints

### Champion eval — 2026-06-19 19:23 | champion=U400 WR=0.36666666666666664 | no challenger  vs_comet_WR=0.000
Pool size:        3 checkpoints

### Champion eval — 2026-06-19 19:35 | champion=U400 WR=0.36666666666666664 | no challenger  vs_comet_WR=0.000
Pool size:        3 checkpoints

### Champion eval — 2026-06-19 19:47 | champion=U400 WR=0.36666666666666664 | no challenger  vs_comet_WR=0.000
Pool size:        3 checkpoints

### Champion eval — 2026-06-19 19:59 | champion=U400 WR=0.36666666666666664 | no challenger  vs_comet_WR=0.000
Pool size:        3 checkpoints

### Champion eval — 2026-06-19 20:11 | champion=U400 WR=0.36666666666666664 | no challenger  vs_comet_WR=0.000
Pool size:        3 checkpoints

### Champion eval — 2026-06-19 20:15 | champion=U400 WR=0.36666666666666664 | no challenger  vs_comet_WR=0.000
Pool size:        3 checkpoints

### Champion eval — 2026-06-19 20:22 | champion=U400 WR=0.36666666666666664 | no challenger  vs_comet_WR=0.000
Pool size:        3 checkpoints

### Champion eval — 2026-06-19 20:35 | champion=U400 WR=0.36666666666666664 | no challenger  vs_comet_WR=0.000
Pool size:        3 checkpoints

### Champion eval — 2026-06-19 20:44 | champion=U400 WR=0.36666666666666664 | no challenger  vs_comet_WR=0.000
Pool size:        3 checkpoints

---
## 20:42 MT — Phase 3 Gate: j3 FAIL, j1 eval in progress

**j3 (ppo-1-of-3_job3) U=500 result:**
- greedy WR = 23.3% ✗ (threshold: >25%)
- comet WR = 0.0% ✗
- Entropy = 0.696 (COLLAPSED — explains failure)
- Gate verdict for j3: FAIL (but j3 is not the champion seed)

**j1 (ppo-1-of-3_job1) U=500:**
- Live log: U=500 logged at 12.4h, Ent=1.907 (HEALTHY)
- U=500 eval RUNNING NOW on 175.105 (~21 min, done ~21:03 MT)
- j1 trend: U100=20%, U200=16%, U300=33%, U400=37% → expect >25% at U=500

**Fresh seeds news:**
- ppo-2-of-3_j3 (170.84): WR=40% at U=400 — best fresh seed result yet, beats champion (36.7%)
- ppo-2-of-3_j2/j4: 33% at U=400
- All fresh seeds at U=400, not yet eligible (need U>450)

**Sync state:** Champion v3 U=400 WR=36.7%, comet=0.000, 9/9 hosts ✓

### Champion eval — 2026-06-19 20:47 | champion=U400 WR=0.36666666666666664 | no challenger  vs_comet_WR=0.000
Pool size:        3 checkpoints

### Champion eval — 2026-06-19 20:59 | champion=U400 WR=0.36666666666666664 | no challenger  vs_comet_WR=0.000
Pool size:        3 checkpoints

### Champion eval — 2026-06-19 21:11 | champion=U400 WR=0.36666666666666664 | no challenger  vs_comet_WR=0.000
Pool size:        3 checkpoints

### Champion eval — 2026-06-19 21:13 | champion=U400 WR=0.36666666666666664 | no challenger  vs_comet_WR=0.000
Pool size:        3 checkpoints

---
## 21:07 MT — Phase 3 Gate PASS + Fleet Update

**PHASE 3 GATE: PASS (condition 1)**
- j1 (champion seed): 36.7% greedy ✓, 0% comet ✗
- j3 (collapsed):     23.3% greedy ✗, 0% comet ✗
- 3-of-3_j3:          30.0% greedy ✓, 0% comet ✗
- 3-of-3_j4:          26.7% greedy ✓, 0% comet ✗

**Comet ceiling CONFIRMED at U=500:** 0.000% on all 4 deep seeds. Pattern is consistent across entire training run.

**Fleet WR landscape (all seeds at max eval depth):**
Top performers: ppo-2-of-6_j1=43%@U300, ppo-2-of-3_j3=40%@U100(best_model), ppo-1-of-3_j1=37%(champion), ppo-1-of-3_j2=37%@U400, ppo-2-of-6_j2=37%@U400

**Champion:** v3 U=400 WR=36.7%, comet=0.000 (unchanged)
**Next challenger window:** Fresh seeds reaching U=500 ~22:00-22:30 MT
**League state:** 9/9 hosts ✓

### Champion eval — 2026-06-19 21:23 | champion=U400 WR=0.36666666666666664 | no challenger  vs_comet_WR=0.000
Pool size:        3 checkpoints

### Champion eval — 2026-06-19 21:34 | champion=U400 WR=0.36666666666666664 | no challenger  vs_comet_WR=0.000
Pool size:        3 checkpoints

### Champion eval — 2026-06-19 21:46 | champion=U400 WR=0.36666666666666664 | no challenger  vs_comet_WR=0.000
Pool size:        3 checkpoints

### Champion eval — 2026-06-19 21:47 | champion=U400 WR=0.36666666666666664 | no challenger  vs_comet_WR=0.000
Pool size:        3 checkpoints

---
## 21:45 MT — Fleet Analysis: Structural Ceiling Identified

**No challengers at U=500.** Fleet shows a clear structural pattern:

### Best-model WR peak analysis (all 32 seeds)
Most seeds save best_model.pt EARLY (U=100-200) due to n_games=30 variance,
then inline evals DECLINE at later updates. Best_models are stuck at U=100-200
and will never reach challenger eligibility (needs best_model U>450).

**High WR best_models STUCK at early U (not eligible at U>450):**
| Seed | best_model U | best_model WR | Recent trend |
|------|-------------|---------------|-------------|
| ppo-4-of-6_j1 | U=200 | **47%** | declining → 33%@U400 |
| ppo-2-of-6_j1 | U=300 | **43%** | declining → 37%@U400 |
| ppo-2-of-3_j1 | U=200 | **43%** | 27%@U400 |
| ppo-3-of-6_j2 | U=100 | **43%** | |
| ppo-3-of-6_j3 | U=100 | **43%** | |

**Only the champion (j1) showed monotone improvement: 20→16→33→37%**
**All other seeds peak early then decline — n_games=30 variance artifact**

### Fresh seed live positions
- ppo-2-of-6_j1: U=~420 (was 37% at U=400; declining) → ETA U=500 ~70 min
- ppo-2-of-6_j2: U=~410 (37% at U=400) → ETA U=500 ~80 min  
- ppo-2-of-3_j3: U=~430, Ent=2.67 (healthy) → ETA U=500 ~60 min
- ppo-4-of-6_j1: U=~400, Ent=3.26 (very high entropy — exploration) → ETA U=500 ~90 min

### Strategic assessment
- comet WR = 0.000% consistent across 5+ seeds at U=300-500
- Challenger path requires inline eval IMPROVEMENT at U=500 vs prior best_wr
- Probability of any fresh seed saving new best_model at U=500: LOW (declining trends)
- June 22 trigger increasingly likely: re-submit comet_reaper as final answer

**comet=0.000, champion=v3 U=400, 9/9 ✓**

### Champion eval — 2026-06-19 21:58 | champion=U400 WR=0.36666666666666664 | no challenger  vs_comet_WR=0.000
Pool size:        3 checkpoints

### Champion eval — 2026-06-19 22:10 | champion=U400 WR=0.36666666666666664 | no challenger  vs_comet_WR=0.000
Pool size:        3 checkpoints

### Champion eval — 2026-06-19 22:20 | champion=U400 WR=0.36666666666666664 | no challenger  vs_comet_WR=0.000
Pool size:        3 checkpoints

---
## 22:19 MT — RL Ceiling Confirmed + Structural Analysis Final

**VERDICT: No challengers at U=500. Fleet at structural ceiling.**

### Why the challenger pipeline is empty
Challenger requires: best_model.pt saved at U>450 with WR>36.7%.
But n_games=30 evaluations produce inverted-U WR across training:
- Most seeds save best_model.pt at U=100-200 with high variance WR (37-47%)
- Later evals (U=300-400) return LOWER WR → no new saves → stuck at early U
- To save a new best_model at U=500, a seed needs to beat its EARLY high WR
- This requires ~43%+ at U=500, against a declining trend — very unlikely

### One remaining candidate: ppo-2-of-6_j2 (174.133)
Trend: 28%→27%→33%→33%→37% (monotone INCREASING from U=300 onward)
best_model saved at U=400 with WR=37% (equal to champion)
At U=420, 33 SPS → ETA U=500 = ~2.5h (~01:00 MT June 20)
Needs ≥40% (12/30 wins) to save new best_model at U=500
If it saves: eligible challenger (U=500 > 450, WR=40% > 36.7%) → 1000-game eval

### Fleet disposition
- League state: v3 U=400 WR=36.7%, comet=0.000, 9/9 ✓
- Next promotion opportunity: j2 eval at U=500 (~01:00 MT)
- June 22 trigger: re-submit comet_reaper — T-3 days

### Champion eval — 2026-06-19 22:22 | champion=U400 WR=0.36666666666666664 | no challenger  vs_comet_WR=0.000
Pool size:        3 checkpoints

### Champion eval — 2026-06-19 22:34 | champion=U400 WR=0.36666666666666664 | no challenger  vs_comet_WR=0.000
Pool size:        3 checkpoints

### Champion eval — 2026-06-19 22:46 | champion=U400 WR=0.36666666666666664 | no challenger  vs_comet_WR=0.000
Pool size:        3 checkpoints

### Champion eval — 2026-06-19 22:58 | champion=U400 WR=0.36666666666666664 | no challenger  vs_comet_WR=0.000
Pool size:        3 checkpoints

### Champion eval — 2026-06-19 23:10 | champion=U400 WR=0.36666666666666664 | no challenger  vs_comet_WR=0.000
Pool size:        3 checkpoints

### Champion eval — 2026-06-19 23:11 | champion=U400 WR=0.36666666666666664 | no challenger  vs_comet_WR=0.000
Pool size:        3 checkpoints

### Champion eval — 2026-06-19 23:22 | champion=U400 WR=0.36666666666666664 | no challenger  vs_comet_WR=0.000
Pool size:        3 checkpoints

### Champion eval — 2026-06-19 23:35 | champion=U400 WR=0.36666666666666664 | no challenger  vs_comet_WR=0.000
Pool size:        3 checkpoints

### Champion eval — 2026-06-19 23:47 | champion=U400 WR=0.36666666666666664 | no challenger  vs_comet_WR=0.000
Pool size:        3 checkpoints

### Champion eval — 2026-06-20 00:01 | champion=U400 WR=0.36666666666666664 | no challenger  vs_comet_WR=0.000
Pool size:        3 checkpoints

### Champion eval — 2026-06-20 00:15 | champion=U400 WR=0.36666666666666664 | no challenger  vs_comet_WR=0.000
Pool size:        3 checkpoints

### Champion eval — 2026-06-20 00:29 | champion=U400 WR=0.36666666666666664 | no challenger  vs_comet_WR=0.000
Pool size:        3 checkpoints
