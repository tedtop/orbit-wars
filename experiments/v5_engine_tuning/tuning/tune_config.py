#!/usr/bin/env python3
"""Optuna study: tune comet_reaper_tuned's config to BEAT comet_reaper on a seat-fair gauntlet.

Objective = win-rate of comet_reaper_tuned (CRT_CONFIG = trial config) vs comet_reaper, over
seat-rotated 4P-focal + 2P games (n≈100/trial → ±~10% CI; Optuna tolerates the noise). Persistent
SQLite study (resumable, queryable). A trial > 0.55 cleanly beats the baseline — validate the winner
vs the full public panel before submitting.

    .venv/bin/python experiments/v5_engine_tuning/tuning/tune_config.py [n_trials]
"""
import json
import os
import subprocess
import sys

import optuna

ROOT = "/Users/Ted/src/orbit_wars"
PY = os.path.join(ROOT, ".venv/bin/python")
WORKER = os.path.join(ROOT, "experiments/v5_engine_tuning/eval/eval_worker.py")
FOCAL = "agents/comet_reaper_tuned/main.py"
GAMES_4P, GAMES_2P, WORKERS = 60, 40, 4   # per trial, split across WORKERS subprocs per mode

# knob: (low, high, "int"|"float") — the full tunable surface incl. re-added ffa bonuses
SPACE = {
    "horizon": (8, 24, "int"),
    "max_sources_per_lane": (4, 16, "int"),
    "max_offensive_targets": (4, 16, "int"),
    "max_defensive_targets": (1, 8, "int"),
    "max_waves_per_turn": (3, 10, "int"),
    "roi_threshold": (1.0, 2.0, "float"),
    "min_ships_to_launch": (2.0, 8.0, "float"),
    "reinforce_size_beta": (0.0, 4.0, "float"),
    "reinforce_eta_free": (1.0, 6.0, "float"),
    "reinforce_eta_scale": (4.0, 20.0, "float"),
    "max_regroup_time": (3.0, 12.0, "float"),
    "regroup_pressure_delta_min": (0.0, 1.0, "float"),
    "max_regroup_targets_per_source": (3, 12, "int"),
    "terminal_phase_turns": (10, 80, "int"),
    "terminal_roi_threshold": (0.5, 1.5, "float"),
    "terminal_max_waves_per_turn": (4, 14, "int"),
    "comet_evac_steps": (1, 8, "int"),
    "ffa_leader_attack_bonus": (0.0, 0.1, "float"),
    "ffa_target_prod_bonus": (0.0, 0.15, "float"),
}


def suggest(trial):
    cfg = {}
    for k, (lo, hi, t) in SPACE.items():
        cfg[k] = trial.suggest_int(k, lo, hi) if t == "int" else round(trial.suggest_float(k, lo, hi), 4)
    return cfg


def eval_winrate(cfg):
    env = dict(os.environ)
    env["CRT_CONFIG"] = json.dumps(cfg)
    procs = []
    for mode, tot, base in (("4p", GAMES_4P, 100000), ("2p", GAMES_2P, 0)):
        per = max(1, tot // WORKERS)
        for w in range(WORKERS):
            seed = 1000 + base + w * per
            procs.append(subprocess.Popen(
                [PY, WORKER, FOCAL, mode, str(per), str(seed)],
                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, env=env, cwd=ROOT, text=True))
    wins = games = 0
    for p in procs:
        try:
            out, _ = p.communicate(timeout=3000)
            a = out.split()
            if len(a) >= 3 and a[0] == "winrate":
                wins += int(a[1]); games += int(a[2])
        except Exception:
            with __import__("contextlib").suppress(Exception):
                p.kill()
    return wins / max(1, games)


def objective(trial):
    return eval_winrate(suggest(trial))


if __name__ == "__main__":
    n_trials = int(sys.argv[1]) if len(sys.argv) > 1 else 300
    study = optuna.create_study(
        direction="maximize",
        storage=f"sqlite:///{ROOT}/experiments/v5_engine_tuning/tuning/study.db",
        study_name="crt_config_v1", load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=0, n_startup_trials=20),
    )
    study.optimize(objective, n_trials=n_trials)
    print("BEST win-rate vs comet_reaper:", round(study.best_value, 3))
    print("BEST config:", json.dumps(study.best_params))
