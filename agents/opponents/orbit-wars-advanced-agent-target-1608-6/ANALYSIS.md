# orbit-wars-advanced-agent-target-1608-6  —  (RAHUL CHAUHAN)

## Core strategy (the real one)
**Cautionary finding.** The notebook is 840 lines of MCTS (`AdvancedMCTS`, `MCTSNode`),
`OpponentModel`, `StrategyEngine`, `EliteEvaluator`, and a "cyberpunk tactical dashboard" with
matplotlib. **None of it is submitted.** The actual `submission.py` written to disk is a ~15-line
greedy bot: *for each of my planets with >15 ships, send half to the nearest neutral/enemy planet
if the path doesn't hit the sun.* The "Target: 1608.6" is aspirational marketing.

## Lesson
Vote count and notebook flashiness ≠ bot strength. Always extract and run the **actually
submitted** code. The MCTS here is dead code (never wired to the submission). Several high-vote
notebooks are like this.

## Ideas to steal
None from the submission (it's trivially weak — easy win for any real bot). The dead MCTS scaffold
is readable if we ever want a from-scratch MCTS reference, but it's untested theater.
