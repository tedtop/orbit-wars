# 📸 The screenshot tour — 30 curated moments

The campaign in pictures, in story order. The full unfiltered archive (140
shots) lives in [`../screenshots/`](../screenshots/).

Companion pieces: the interactive write-up at
**[kaggle-orbit-wars.vercel.app](https://kaggle-orbit-wars.vercel.app)** and the
dated research log in [`../../TIMELINE.md`](../../TIMELINE.md).

---

## Act I — The game

Total domination: step 186, our 5,806 ships vs the starter bot's 5.

![game board domination](./01-game-board-domination-step186-you-5806-vs-starter-5.png)

Mid-game grind, step 165 — 4,293 vs 136.

![mid domination](./02-game-board-mid-domination-step165-you-4293-vs-starter-136.png)

Early expansion, step 154.

![early game](./03-game-board-early-game-step154-you-3606-vs-starter-279.png)

The opening land-grab, step 109.

![early game](./04-game-board-early-game-step109-you-1041-vs-starter-47.png)

And what losing looks like — 296 ships vs 9,002, step 279.

![losing state](./05-game-board-losing-state-step279-you-296-vs-easy-9002.png)

## Act II — The climb

Peak form: **rank #140, score 1,248** on the ops dashboard.

![rank 140](./06-dashboard-rank-140-score-1248-leaderboard.png)

Active agents and episode history, live from the pipeline.

![active agents](./07-dashboard-active-agents-episode-history.png)

Earlier on the ladder: 1,141 at rank #653 and climbing.

![climbing](./08-dashboard-early-score-1141-rank-653-climbing.png)

Second angle on the summit.

![rank 140 again](./30-dashboard-rank-140-score-1248-angle2.png)

## Act III — schmeekler, the one that worked

The orchestrator watches the live ladder as schmeekler beats comet_reaper.

![schmeekler beats comet_reaper](./12-orchestrator-live-ladder-schmeekler-beats-comet-reaper.png)

Confirmed: ~70% win rate across the entire public field.

![win rate table](./13-schmeekler-confirmed-win-rate-table-70pct-entire-field.png)

Validation sweep running in parallel.

![validation sweep](./27-schmeekler-validation-sweep-running-parallel.png)

The submission moment.

![submitting schmeekler](./19-submission-moment-python-submit-schmeekler-kaggle.png)

Picking the bot in `submit.py`.

![bot selection](./26-submit-py-bot-selection-list.png)

schmeekler_fmt climbing the live ladder, orchestrator tick by tick.

![schmeekler_fmt climbing](./22-orchestrator-tick-live-ladder-schmeekler-fmt-climbing.png)

## Act IV — The autoresearch lab

Claude agents running the experiment tracks in parallel terminals:
Track A (structural features)…

![track A session](./14-multi-agent-terminal-track-a-structural-features-session.png)

…and Track B (MCTS search).

![track B session](./15-multi-agent-terminal-track-b-mcts-search-session.png)

The ratchet's hypothesis queue in the program log.

![hypothesis queue](./16-autoresearch-program-log-ratchet-hypothesis-queue.png)

The value function posts AUC 0.99 — a massive pass on prediction, which still
didn't move the win rate.

![value function AUC](./11-orchestrator-track-c-value-function-auc-0-99-massive-pass.png)

VF metrics landing in an orchestrator wake event.

![VF wake event](./23-orchestrator-wake-event-track-c-vf-metrics-table.png)

The orbit_lite inflection — the moment the value-function insight crystallized.

![orbit_lite inflection](./20-orchestrator-orbit-lite-inflection-value-function-insight.png)

The agent-variants scoreboard.

![variants table](./21-orchestrator-agent-variants-table.png)

Track B closed — state of play.

![track B closed](./25-track-b-mcts-search-closed-state-of-play.png)

The sober conclusion: cheap hypothesis space exhausted.

![hypotheses exhausted](./17-autoresearch-cheap-hypothesis-space-exhausted-conclusion.png)

15 experiments on the Streamlit autoresearch tab.

![autoresearch tab](./18-streamlit-autoresearch-tab-15-experiments-track-c-auc.png)

A fresh orchestrator relaunch, hard rules pinned.

![orchestrator relaunch](./24-orchestrator-fresh-relaunch-prompt-hard-rules.png)

## Act V — The RL fleet

Launch moment: 8 Jetstream2 instances deploying.

![fleet launch](./29-fleet-launch-moment-8-instances-deploying.png)

All nine instances in a 9-pane tmux monitor.

![9-pane monitor](./28-fleet-monitor-9-pane-all-jetstream2-instances.png)

Per-instance training metrics, 4-pane view.

![4-pane metrics](./10-fleet-monitor-4-pane-per-instance-metrics.png)

The RL dashboard mid-campaign — red gate, fleet at 1,612 SPS.

![red gate](./09-rl-training-dashboard-red-gate-fleet-sps-1612.png)
