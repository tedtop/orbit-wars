> Shared engine: see `agents/opponents/ORBIT_LITE_FAMILY.md` for the full orbit_lite deep-dive.
> This file covers only what THIS fork changes and how to beat it.

# floor-matched-fleets  —  ~1175 (AnthonyTherrien)

## Delta vs producer-v2
Two additions: (1) **comet evacuation** — own planets sitting on a comet that is ~4 steps from
expiry evacuate their ships before the comet (and its production bonus) vanishes; (2) **target
veto** — declines attacks on comet planets that will expire before/around fleet arrival; (3) the
anti-leader 4P bias. Post-processes the engine's moves with `_comet_evac_moves`.

## How to beat it
- Outside comet windows it is baseline producer-v2.
- Its evac is reactive and threshold-based (4 steps); time a strike to arrive exactly as it
  evacuates a comet planet — it leaves with a thin garrison and the destination is contested.

## Ideas to steal
Comet lifecycle awareness (don't sink ships into a planet whose production is about to vanish;
harvest your own before expiry) is a clean, cheap edge most bots ignore (strategy #6).
