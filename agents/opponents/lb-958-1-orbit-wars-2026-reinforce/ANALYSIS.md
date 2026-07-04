# lb-958-1-reinforce  —  ~958 (Roman Tamrazov)

## Core strategy
The **earlier version of the lb-1224 lineage** (~1900 lines, same arrival-ledger forward sim).
Headline feature is **reinforcement missions**: proactively shuttle ships to frontier planets the
timeline shows under threat, plus `_multi_enemy_proactive_keep` (in 4P, hold back a defensive
fraction when multiple enemies can reach a planet) and `detect_enemy_crashes` (exploit planets
left empty right after two enemies collide).

## Edges
- Proactive reinforcement before the hit lands (vs reactive defense).
- 4P crash-exploitation: snipe the empty planet after enemy A and enemy B trade.

## Weaknesses
- ~290 points below lb-1224 — superseded. The later version added weakest-enemy/gang-up/exposed
  targeting on top of this, which is where the points came from.

## Ideas to steal
Crash-exploitation (`detect_enemy_crashes`) and proactive reinforcement are clean 4P ideas; both
were carried into lb-1224. Study here is mainly to see the lineage's evolution. Lower priority
than lb-1224 itself.
