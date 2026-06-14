#!/usr/bin/env python3
"""
Orbit Wars submission helper.

Usage:
    python submit.py              # interactive: pick a bot, smoke test, submit
    python submit.py --list       # just list available bots
    python submit.py --dry-run    # smoke test only, no Kaggle upload
"""

import argparse
import importlib.util
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT  = Path(__file__).resolve().parent
AGENTS_DIR = REPO_ROOT / "agents"
COMPETITION = "orbit-wars"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def list_bots() -> list[Path]:
    """Return .py files in agents/ that aren't __init__ or opponents."""
    bots = sorted(
        p for p in AGENTS_DIR.glob("*.py")
        if p.stem != "__init__"
    )
    return bots


def smoke_test(bot_path: Path, modes: tuple[str, ...] = ("2p", "4p")) -> bool:
    """Run bot locally with kaggle-environments. Returns True if all pass."""
    try:
        from kaggle_environments import make
    except ImportError:
        print("  ⚠  kaggle-environments not installed — skipping smoke test.")
        print("     pip install 'kaggle-environments>=1.28.0'")
        return True  # don't block submission over missing dep

    spec = importlib.util.spec_from_file_location("bot_under_test", bot_path)
    mod  = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
    except Exception as e:
        print(f"  ✗  Import failed: {e}")
        return False

    if not callable(getattr(mod, "agent", None)):
        print("  ✗  No callable 'agent' found in the file.")
        return False

    print("  • Import OK, agent is callable")

    passed = True
    if "2p" in modes:
        try:
            env = make("orbit_wars", debug=False)
            out = env.run([mod.agent, "random"])
            r   = out[-1][0].reward
            print(f"  • 2-player smoke test: reward={r}  {'✓' if r is not None else '?'}")
        except Exception as e:
            print(f"  ✗  2-player smoke test crashed: {e}")
            passed = False

    if "4p" in modes:
        try:
            env = make("orbit_wars", debug=False)
            out = env.run([mod.agent, "random", "random", "random"])
            r   = out[-1][0].reward
            print(f"  • 4-player smoke test: reward={r}  {'✓' if r is not None else '?'}")
        except Exception as e:
            print(f"  ✗  4-player smoke test crashed: {e}")
            passed = False

    return passed


def kaggle_submit(bot_path: Path, message: str) -> bool:
    """Copy bot to a temp main.py and submit to Kaggle."""
    with tempfile.TemporaryDirectory() as tmpdir:
        main_py = Path(tmpdir) / "main.py"
        shutil.copy2(bot_path, main_py)

        cmd = [
            "kaggle", "competitions", "submit",
            COMPETITION,
            "-f", str(main_py),
            "-m", message,
        ]
        print(f"\n  Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=False)
        return result.returncode == 0


def show_submissions():
    """Print current submission status."""
    print("\n─── Current submissions ───────────────────────────────────")
    subprocess.run(["kaggle", "competitions", "submissions", COMPETITION])
    print("────────────────────────────────────────────────────────────")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Orbit Wars submission helper")
    parser.add_argument("--list",    action="store_true", help="List bots and exit")
    parser.add_argument("--dry-run", action="store_true", help="Smoke test only, no upload")
    parser.add_argument("--bot",     type=str,            help="Bot filename (no path needed)")
    args = parser.parse_args()

    bots = list_bots()
    if not bots:
        print(f"No bots found in {AGENTS_DIR}/")
        sys.exit(1)

    # -- list mode -----------------------------------------------------------
    if args.list:
        print(f"\nBots in {AGENTS_DIR.relative_to(REPO_ROOT)}/")
        for i, b in enumerate(bots, 1):
            print(f"  {i:2d}.  {b.name}")
        print()
        return

    # -- select bot ----------------------------------------------------------
    if args.bot:
        # Accept bare name (markowitz) or full filename
        stem = Path(args.bot).stem
        matches = [b for b in bots if b.stem == stem]
        if not matches:
            print(f"Bot '{args.bot}' not found in {AGENTS_DIR}/")
            print("Available:", [b.name for b in bots])
            sys.exit(1)
        chosen = matches[0]
    else:
        print(f"\nBots available in {AGENTS_DIR.relative_to(REPO_ROOT)}/\n")
        for i, b in enumerate(bots, 1):
            print(f"  {i:2d}.  {b.stem}")
        print()
        while True:
            raw = input("Select bot number: ").strip()
            if raw.isdigit() and 1 <= int(raw) <= len(bots):
                chosen = bots[int(raw) - 1]
                break
            print(f"  Enter a number between 1 and {len(bots)}")

    print(f"\n{'─'*56}")
    print(f"  Bot : {chosen.name}")
    print(f"{'─'*56}")

    # -- smoke test ----------------------------------------------------------
    print("\nRunning smoke tests …")
    ok = smoke_test(chosen)
    if not ok:
        print("\n✗  Smoke test failed. Fix the bot before submitting.")
        sys.exit(1)
    print("✓  Smoke tests passed.\n")

    if args.dry_run:
        print("--dry-run: stopping here, nothing submitted.")
        return

    # -- submission message --------------------------------------------------
    default_msg = f"{chosen.stem} v1"
    msg_input   = input(f"Submission message [{default_msg}]: ").strip()
    message     = msg_input if msg_input else default_msg

    # -- reminder about slot order ------------------------------------------
    print("""
  ⚠  SLOT ORDER REMINDER
  Only your latest 2 submissions are tracked for the leaderboard.
  Submit your LESS PREFERRED bot first, then your BEST bot last
  so the best is most-recent.
""")
    confirm = input("  Submit now? [y/N] ").strip().lower()
    if confirm != "y":
        print("Aborted.")
        return

    # -- submit --------------------------------------------------------------
    success = kaggle_submit(chosen, message)

    if success:
        print(f"\n✓  Submitted '{chosen.stem}'")
        show_submissions()
    else:
        print("\n✗  Kaggle CLI returned an error. Check output above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
