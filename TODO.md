# Orbit Wars — Active TODO

Updated: 2026-06-22 07:10 MT

## Post-competition cleanup (do after June 23)

- [ ] Delete stale v6 run dirs from `agents/rl_ppo/runs/` (v6_cr2/cr3/cr4, v6_fresh/greedy/main/sym, ent_fix_test, et_benchmark, eval_fix_test, gae_fix_test, gae_full, h1_test, remote/)
- [ ] Delete loose .log files at `agents/rl_ppo/` root level
- [ ] Delete `agents/rl_ppo/checkpoints/pool/` MLP champion relics
- [ ] Delete `agents/rl_ppo/launch_gpu.sh` (speculative, never used, superseded by JAX rewrite)
- [ ] Add `agents/rl_ppo/runs/` and `agents/rl_ppo/checkpoints/` to `.gitignore`
- [ ] Commit actual code changes: train.py, monitor.sh, sync_checkpoints.sh, eval_checkpoints.py, launch_cpu.sh
- [ ] Future experiments: use versioned subfolders (experiments/v9_et/runs/ + /checkpoints/) to avoid cross-contamination
- [ ] Review v10-jax worktree (`../orbit_wars_jax/`) — merge or close branch

---

## Stale entries (from 2026-06-18, kept for reference)

## Blocking / high priority

- [ ] **Fleet eval gate** — wait for U=100 eval on remote fleet (~27 min from now).
      Need >25% to confirm fleet is learning. Currently 32 jobs running.
- [ ] **Local eval climbing** — v6_main at U=280, eval stuck at 20%. Check again
      at U=300 eval. If still 20%, investigate reward signal strength.
- [ ] **m3.quad not working** — stuck on "waiting for logs". The 4-CPU instance
      never had training deployed successfully. Either fix (too slow for 4 jobs,
      try 1 job with --num_envs 4) or repurpose as dashboard-only host.

## Infrastructure

- [ ] **Separate pip install from relaunch** — launch_cpu.sh reinstalls pip deps
      every restart, causing 5-10 min downtime. Split into setup.sh (first-time)
      and restart.sh (just kill + relaunch existing venv). Use heredoc SSH instead.
- [ ] **Restart start.sh** — version running for days (PID 20677) lacks the RL
      sync function. Ctrl-C the terminal tab and run `bash start.sh` to activate
      10-min checkpoint sync.
- [ ] **Wire m3.quad as dashboard host** — once repurposed, point it at Streamlit
      so Ted has a persistent web UI without using a local tab.

## Deferred (after proof of learning)

- [ ] **Anvil SLURM setup** — 93 idle 128-CPU wholenode nodes at Purdue. Write
      and submit script. Do NOT touch until fleet shows >30% eval and we trust
      the training code.
- [ ] **TIMELINE.md** — append tonight's 7-bug-fix milestone and first learning
      signal.
- [ ] **Competitor analysis** — Discord shows someone at U=2468, 69% vs own
      checkpoint. Track their public submissions to gauge gap.
