# v9 — Entity Transformer Benchmark

**Opened:** 2026-06-22
**Branch:** v9-entity-transformer
**Deadline:** 2026-06-23 (competition end)

---

## Hypothesis

All prior RL failures (v6, v7) shared one root cause: flat-MLP over a flattened 302-dim
planet vector can't express the relational structure of the game. Per Lin Myat Ko (41st,
~top 20 via self-play):

> "EV should go up to at least 0.8 in 100 iters. I checked my earliest run. It got to 0.9
> in 20 iters. If explained variance never gets past 0.5, you should check your obs
> representation or architecture."

Our v6/v7 EV capped at 0.75–0.84 — below Lin's threshold. The flat MLP (inherited from
kashiwaba's tutorial) was the unchallenged bottleneck across both campaigns.

**`ActorCriticET`** replaces the flat MLP backbone with a Transformer encoder that treats
planets and fleets as a *set of objects*, each attending to all others. The value function
now has access to genuine relational structure → should achieve EV > 0.9 early.

---

## Architecture: ActorCriticET

| Component | Old (ActorCritic) | New (ActorCriticET) |
|-----------|-------------------|---------------------|
| Input | 302-dim flat vector | 20 planet objects + 12 fleet objects + 1 global token |
| Backbone | 3× Linear(256)+LayerNorm+Tanh | TransformerEncoder (2 layers, 64-dim, 4 heads) |
| Params | 217,620 | 106,936 |
| Target head | Linear(src_ctx, 20) | dot-product attention: source query vs planet reps |
| Value input | flat MLP output | mean-pool of all entity outputs |

Key: target selection is now a proper cross-attention between source and all planet
representations — structurally matching what the game requires.

---

## Diagnostic Gate (LOCAL, ~1h)

Run 50 updates locally with `num_envs=32`:

```bash
python agents/rl_ppo/train.py --model-type et --num_envs 32 \
    --total_updates 50 --run_name et_benchmark
```

**Pass:** EV ≥ 0.90 by update 20  
**Fail:** EV stays below 0.84 (same as flat MLP)

If pass → deploy to Jetstream2 instances (still running), run overnight, eval comet_reaper_WR at dawn.  
If fail → representation change insufficient; close v9, comet_reaper is final.

---

## Results Log

| Update | EV (ET) | EV (MLP baseline) | Notes |
|--------|---------|-------------------|-------|
| | | | |

## Verdict

OPEN
