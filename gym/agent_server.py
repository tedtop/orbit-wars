#!/usr/bin/env python3
"""Persistent agent server — runs ONE bot in its own process for the gym.

Launched with cwd = the bot's folder so its local `orbit_lite` / `submission`
modules resolve and don't collide with other bots. Protocol (line-delimited JSON
over stdin/stdout):

    stdin  : {"obs": {...}, "config": {...}}   one per turn
    stdout : [[from_planet, angle, ships], ...] one per turn (JSON)

Any error inside the bot yields `[]` (a legal no-op) rather than crashing the game.
The first stdout line after import is `{"ready": true}` (or an error object).
"""
import contextlib
import importlib.util
import inspect
import io
import json
import os
import sys
import warnings

warnings.filterwarnings("ignore")

# kaggle_environments prints env-load failures (e.g. the `cabt` dlopen error) to
# fd 1 at import time, which would corrupt our line-delimited JSON protocol.
# Redirect fd 1 -> stderr for ALL heavy imports, then restore a clean stdout.
_saved_stdout_fd = os.dup(1)
sys.stdout.flush()
os.dup2(2, 1)
try:
    from kaggle_environments.utils import structify  # triggers env registration (noisy)
finally:
    sys.stdout.flush()
    os.dup2(_saved_stdout_fd, 1)
    os.close(_saved_stdout_fd)


def _load_agent():
    sys.path.insert(0, os.getcwd())
    entry = os.environ.get("BOT_MAIN", "main.py")
    spec = importlib.util.spec_from_file_location("botmain", entry)
    m = importlib.util.module_from_spec(spec)
    sys.modules["botmain"] = m
    # Silence any import-time chatter so it can't corrupt the JSON stream.
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        spec.loader.exec_module(m)
    return m.agent


def _sanitize(o):
    if hasattr(o, "tolist"):
        return o.tolist()
    if isinstance(o, (list, tuple)):
        return [_sanitize(x) for x in o]
    try:
        return float(o) if isinstance(o, float) else o
    except Exception:
        return o


def main():
    out = sys.stdout
    try:
        agent = _load_agent()
        n_params = len(inspect.signature(agent).parameters)
    except Exception as e:  # import/load failure
        out.write(json.dumps({"ready": False, "error": f"{type(e).__name__}: {e}"}) + "\n")
        out.flush()
        return
    out.write(json.dumps({"ready": True}) + "\n")
    out.flush()

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
            if msg.get("reset"):
                # Reload the bot module to clear per-game state without re-importing
                # torch/kaggle_environments (which stay cached). Cheap fresh start.
                agent = _load_agent()
                n_params = len(inspect.signature(agent).parameters)
                out.write(json.dumps({"reset_ok": True}) + "\n")
                out.flush()
                continue
            obs = structify(msg["obs"])
            cfg = structify(msg.get("config", {}))
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                action = agent(obs, cfg) if n_params >= 2 else agent(obs)
            action = _sanitize(action) if action is not None else []
            out.write(json.dumps(action, default=lambda o: _sanitize(o)) + "\n")
        except Exception as e:
            out.write(json.dumps([], default=str) + "\n")
            sys.stderr.write(f"agent_server error: {type(e).__name__}: {e}\n")
        out.flush()


if __name__ == "__main__":
    main()
