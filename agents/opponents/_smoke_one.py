#!/usr/bin/env python3
"""Smoke-test ONE extracted bot. Run with cwd set to the bot's folder.

Loads ./main.py, plays a short orbit_wars game vs `random`, and reports whether
the agent ran without error and emitted actions. Usage:
    cd agents/opponents/<slug> && python ../_smoke_one.py [num_players] [steps]
"""
import importlib.util
import os
import sys
import warnings

warnings.filterwarnings("ignore")

NUM_PLAYERS = int(sys.argv[1]) if len(sys.argv) > 1 else 2
STEPS = int(sys.argv[2]) if len(sys.argv) > 2 else 8


def load_agent():
    sys.path.insert(0, os.getcwd())
    spec = importlib.util.spec_from_file_location("botmain", "main.py")
    m = importlib.util.module_from_spec(spec)
    sys.modules["botmain"] = m
    spec.loader.exec_module(m)
    return m.agent


def main():
    import contextlib
    import io

    sink = io.StringIO()
    try:
        with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
            agent = load_agent()
            from kaggle_environments import make

            env = make("orbit_wars", configuration={"episodeSteps": STEPS})
            agents = [agent] + ["random"] * (NUM_PLAYERS - 1)
            env.run(agents)
    except Exception as e:
        print(f"IMPORT/RUN ERROR: {type(e).__name__}: {e}")
        return 2

    s0 = env.state[0]
    status = s0.status
    # Did our agent ever emit a non-empty action over the game?
    emitted = False
    for st in env.steps:
        act = st[0].get("action")
        if act:
            emitted = True
            break
    if status == "DONE" or status == "ACTIVE":
        print(f"PASS status={status} emitted_actions={emitted} reward={s0.reward}")
        return 0
    print(f"FAIL status={status} emitted_actions={emitted} reward={s0.reward}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
