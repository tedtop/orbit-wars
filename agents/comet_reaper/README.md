# comet_reaper — the final answer

Our champion and final submission (~1,235 Elo, rank #415 of 4,752). Built on
day three of the competition and never beaten by anything we tried in the 19
experiments and five RL campaigns that followed.

## Attribution

The vendored [`orbit_lite/`](./orbit_lite/) planning engine is our port of the
engine behind **The Producer** — the top public bot lineage, published on
Kaggle by Slawek Biel and shared through the `producer-orbit-wars-utils`
dataset and the "The Producer V2" notebook family. The engine simulates ~18
turns of future board state (every fleet in flight, every garrison at every
tick) and scores candidate attacks by competitive flow differential. Our
`main.py` wraps it with submission plumbing; the planning core is their
design. Credit where it's due — this engine defined the public meta.

See [`../opponents/ORBIT_LITE_FAMILY.md`](../opponents/ORBIT_LITE_FAMILY.md)
for the family tree, and the project write-up at
**https://kaggle-orbit-wars.vercel.app** for how it held off everything we
threw at it.
