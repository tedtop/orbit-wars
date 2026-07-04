# strategy/

Research notes, experiment context, and visual analysis collected during the competition. For the primary narrative, start with [`../TIMELINE.md`](../TIMELINE.md).

---

## Primary references

| File | What it is |
|---|---|
| [`EXPERIMENTS_CONTEXT.md`](EXPERIMENTS_CONTEXT.md) | Deep-dive on every experiment: why it was tried, what the sub-findings were, what didn't make it into TIMELINE.md |
| [`SCREENSHOT_ANALYSIS.md`](SCREENSHOT_ANALYSIS.md) | Phase-by-phase analysis of 139 progress screenshots — maps visual evidence to the timeline narrative |

---

## Session research notes

Written during the competition, in order. Rougher than TIMELINE.md — these are working documents.

| File | Written | Topic |
|---|---|---|
| [`01-SUMMARY_FOR_HUMANS.md`](01-SUMMARY_FOR_HUMANS.md) | Jun 17 | Mid-competition strategic summary |
| [`02-ORBIT_WARS_STRATEGY.md`](02-ORBIT_WARS_STRATEGY.md) | Jun 14 | High-level strategy framing |
| [`03-ORBIT_WARS_ARCHITECTURE.md`](03-ORBIT_WARS_ARCHITECTURE.md) | Jun 14 | orbit_lite engine architecture analysis |
| [`04-AGENT_WRITEUP.md`](04-AGENT_WRITEUP.md) | Jun 13 | The 23-scientist first-night agents — design rationale for each |
| [`05-AGENT_ALGO_SPEC.md`](05-AGENT_ALGO_SPEC.md) | Jun 13 | Algorithmic specification for the heuristic agents |
| [`09-rl_strategy.md`](09-rl_strategy.md) | Jun 14 | RL strategy notes written before the fleet launched |
| [`10-ppo_mlp_to_entity_transformer.md`](10-ppo_mlp_to_entity_transformer.md) | Jun 22 | Architecture design: MLP baseline → Entity Transformer |
| [`11-gym_and_data_plan.md`](11-gym_and_data_plan.md) | Jun 14 | Gym wrapper + training data plan |
| [`12-gym_findings.md`](12-gym_findings.md) | Jun 14 | Gym environment findings |
| [`13-v4_candidates.md`](13-v4_candidates.md) | Jun 14 | v4 candidate bot analysis |
| [`14-Scoring - Wilson vs TrueSkill.md`](<14-Scoring - Wilson vs TrueSkill.md>) | Jun 14 | Evaluation methodology: which rating system to use |
| [`15-Top Open Source Bots.md`](<15-Top Open Source Bots.md>) | Jun 14 | Survey of public leaderboard bots |
| [`16-session-status.md`](16-session-status.md) | Jun 14 | Session status snapshot |

---

## Folders

| Folder | Contents |
|---|---|
| [`portfolio_screenshots/`](portfolio_screenshots/) | 30 hand-curated screenshots — the visual story of the competition, chronological |
| [`raw_notes/`](raw_notes/) | Raw conversation snippets from autonomous sessions. Authentic record of how the orchestrator was prompted and how it responded. Not polished. |
| [`archive/`](archive/) | Dashboard design evolution (3 versions), Optuna config snapshot, submission tracking database (`tracking.db` — SQLite, queryable) |
| [`scripts/`](scripts/) | Helper scripts: screenshot curation pipeline, OCR runner |
