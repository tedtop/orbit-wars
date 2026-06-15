#!/usr/bin/env python3
"""In-process proxy that fronts a subprocess-isolated bot as a normal agent fn.

`ProxyAgent(bot_dir)` is callable as `agent(obs, config)` and can be passed
straight to `kaggle_environments` / `arena.play_one` as an agent spec. Each
proxy owns a persistent `agent_server.py` subprocess (cwd = bot_dir); it
restarts the subprocess whenever it sees `step == 0` so every game starts from
clean bot state. On timeout/crash it returns `[]` (legal no-op) and is marked
broken so it stops eating the per-turn budget.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SERVER = str(Path(__file__).resolve().parent / "agent_server.py")
PY = sys.executable


class ProxyAgent:
    def __init__(self, bot_path: str, name: str | None = None, turn_timeout: float = 8.0,
                 env_extra: dict | None = None):
        self.env_extra = env_extra or {}
        p = Path(bot_path).resolve()
        if p.is_file():            # single-file bot (e.g. our agents/foo.py)
            self.cwd = str(p.parent)
            self.entry = p.name
            self.name = name or p.stem
        else:                      # bot folder with main.py (+ deps like orbit_lite)
            self.cwd = str(p)
            self.entry = "main.py"
            self.name = name or p.name
        self.turn_timeout = turn_timeout
        self.proc: subprocess.Popen | None = None
        self.broken = False

    # -- subprocess lifecycle ------------------------------------------------
    def _start(self):
        self._stop()
        env = dict(os.environ, BOT_MAIN=self.entry, **{k: str(v) for k, v in self.env_extra.items()})
        self.proc = subprocess.Popen(
            [PY, SERVER], cwd=self.cwd, env=env,
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            text=True, bufsize=1,
        )
        ready = self._readline(timeout=60.0)  # import can be slow (torch)
        if not ready:
            self.broken = True
            return
        try:
            info = json.loads(ready)
            self.broken = not info.get("ready", False)
        except Exception:
            self.broken = True

    def _stop(self):
        if self.proc is not None:
            try:
                self.proc.kill()
            except Exception:
                pass
            self.proc = None

    def _readline(self, timeout: float) -> str | None:
        """Read one line with a wall-clock timeout (portable, thread-free)."""
        import select
        end = time.time() + timeout
        if self.proc is None or self.proc.stdout is None:
            return None
        while time.time() < end:
            r, _, _ = select.select([self.proc.stdout], [], [], max(0.0, end - time.time()))
            if r:
                return self.proc.stdout.readline()
            if self.proc.poll() is not None:  # process died
                return None
        return None  # timeout

    # -- agent interface -----------------------------------------------------
    def _reset_bot(self) -> bool:
        """Reload the bot's per-game state in the live process (keeps torch loaded)."""
        if self.proc is None:
            return False
        try:
            self.proc.stdin.write(json.dumps({"reset": True}) + "\n")
            self.proc.stdin.flush()
        except Exception:
            return False
        ack = self._readline(timeout=60.0)
        if not ack:
            return False
        try:
            return bool(json.loads(ack).get("reset_ok"))
        except Exception:
            return False

    def __call__(self, obs, config=None):
        step = obs.get("step", 1) if isinstance(obs, dict) else getattr(obs, "step", 1)
        if self.proc is None:
            self.broken = False
            self._start()
        elif int(step) == 0:
            # New game: reuse the warm process, just reload bot state. Fall back
            # to a full restart if the in-process reset fails.
            self.broken = False
            if not self._reset_bot():
                self._start()
        if self.broken or self.proc is None:
            return []
        try:
            payload = json.dumps({
                "obs": dict(obs) if not isinstance(obs, dict) else obs,
                "config": dict(config) if config is not None and not isinstance(config, dict) else (config or {}),
            })
            self.proc.stdin.write(payload + "\n")
            self.proc.stdin.flush()
        except Exception:
            self.broken = True
            return []
        resp = self._readline(timeout=self.turn_timeout)
        if resp is None:
            self.broken = True
            self._stop()
            return []
        try:
            return json.loads(resp)
        except Exception:
            return []

    def close(self):
        self._stop()
