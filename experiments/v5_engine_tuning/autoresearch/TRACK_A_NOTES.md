# Track A Notes — Orbit Wars v5 Engine Tuning

Append newest experiment first.

---

### 2026-06-17 comet 2×2 factorial — comet targeting bonus  [COMET_TARGET_BONUS=1.5 sweep]

**Hypothesis**: actively hunting comet planets (ephemeral, prod=1.0) improves score despite field-wide 92%-ignore rate.
Comet planet IDs fetched from `obs_tensors["comet_planet_ids"]` (populated by `orbit_lite/adapter.py`);
cross-referenced with `obs_tensors["planets"][:, 0]` via `torch.isin()` to build boolean `is_comet` mask;
additive bonus=1.5 applied to `score` for comet-target candidates — same pattern as static_target_bonus.

**Conditions (4-cell 2×2 factorial)**:

| condition | static_bonus | comet_bonus | COMET | STATIC | n/opp |
|-----------|-------------|-------------|-------|--------|-------|
| comet_reaper | 0 | 0 | off | off | baseline |
| schmeekler | 1.5 | 0 | off | on | baseline |
| comet_reaper_comet | 0 | 1.5 | on | off | 50 |
| schmeekler_comet | 1.5 | 1.5 | on | on | 50 |

**Results (n=50/opp, seat-swapped, 5 opponents)**:

| condition | comet_reaper | the-producer-v2 | i-m-stronger | floor-matched | 1266-elo | OVERALL | vs baseline |
|-----------|-------------|-----------------|--------------|---------------|----------|---------|-------------|
| schmeekler (baseline) | 72% | 77%* | —* | —* | —* | **74%** | — |
| schmeekler_comet | 72% | 74% | 78% | 78% | 66% | **74%** | **0pp** |
| comet_reaper_comet | 50% | 52% | 67% | 71% | 66% | **61%** | **≈0pp** |

*schmeekler per-opp details partially from LOG.md; vs-producer-v2 77% = LOG entry; OVERALL 74% from prior eval.
comet_reaper baseline OVERALL ~61% (inferred; not explicitly stored — matches comet_reaper_comet ≈ exactly).

**Per-opp scores (raw)**:

schmeekler_comet:
- vs comet_reaper: 36-14 (72%)
- vs the-producer-v2: 37-13 (74%)
- vs i-m-stronger: 39-11 (78%)
- vs floor-matched: 39-11 (78%)
- vs 1266-elo: 33-17 (66%)
- OVERALL: 184/250 = 74%

comet_reaper_comet:
- vs comet_reaper: 22-22 (50%)
- vs the-producer-v2: 25-23 (52%)
- vs i-m-stronger: 32-16 (67%)
- vs floor-matched: 34-14 (71%)
- vs 1266-elo: 33-17 (66%)
- OVERALL: 146/238 = 61% (250-12 games, likely 12 draws)

**Analysis**:
- schmeekler_comet OVERALL = 74% = schmeekler baseline → **0pp gain, p>>0.05**
- comet_reaper_comet OVERALL ≈ comet_reaper baseline → **0pp gain**
- Interaction: neither condition (with or without static bonus) benefits from comet targeting
- Mechanism: orbit_lite's flow scorer already implicitly handles ephemeral targets via ETA weighting;
  the 1.5 additive bonus doesn't change which planet wins selection — the bonus is swamped by the
  existing flow score differences, or comet planets are simply too rare/fleeting to shift outcome
- The 92% field-wide ignore rate reflects that comets are suboptimal targets on average —
  the flow scorer knows this; we cannot outsmart it with a flat additive bonus

**Decision**: COMET bonus sweep (0.5/1.0/1.5/2.0) — **NOT RUN** (no gain at n=50).

**Verdict**: DISCARD — comet_target_bonus=1.5 is noise, no gain on either base bot. The orbit_lite
flow scorer already handles comet valuation implicitly. Flat additive bonus cannot help when comets
are structurally low-value ephemeral targets that the engine correctly de-prioritizes.

Agents built (committed, kept for provenance, not submitted):
- `agents/schmeekler_comet/` — WIP commit 18e37a1
- `agents/comet_reaper_comet/` — WIP commit 18e37a1
