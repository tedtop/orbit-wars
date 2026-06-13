"""Verify the orbit_wars environment is installed and importable."""
import logging
import os
import sys

# Suppress startup noise (same technique as run_local.py)
logging.disable(logging.INFO)
_devnull = open(os.devnull, "w")
_old_stdout, _old_stderr = sys.stdout, sys.stderr
sys.stdout = sys.stderr = _devnull
_saved_fd1, _saved_fd2 = os.dup(1), os.dup(2)
os.dup2(_devnull.fileno(), 1)
os.dup2(_devnull.fileno(), 2)
try:
    from kaggle_environments import make
finally:
    os.dup2(_saved_fd1, 1); os.close(_saved_fd1)
    os.dup2(_saved_fd2, 2); os.close(_saved_fd2)
    sys.stdout, sys.stderr = _old_stdout, _old_stderr
    _devnull.close()
    logging.disable(logging.NOTSET)

env = make("orbit_wars")
print(f"Smoke test passed -- orbit_wars {env.version} loaded successfully.")
print()
print("Next: run single test games (pick one)")
print("  .venv/bin/python run_local.py           # summary output")
print("  .venv/bin/python run_local.py --debug   # verbose turn-by-turn output")
