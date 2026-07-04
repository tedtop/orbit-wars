#!/usr/bin/env python3
"""Extract runnable agent code from downloaded Kaggle .ipynb notebooks.

For each `agents/opponents/<slug>/<slug>.ipynb` it scans code cells for
`%%writefile <path>` cell magics and writes the cell body to `<slug>/<path>`
(absolute paths like /kaggle/working/foo.py are mapped to the basename).

This reconstructs each bot's submission files exactly as the author shipped them:
  - main.py / submission.py        -> the agent policy
  - orbit_lite/*.py                -> self-contained engine (v44 bot)
  - decode_weights.py + weights    -> ML hybrids
  - default_cfg.yaml, src/*.py     -> training notebooks

It then normalizes the entrypoint: if a folder has `submission.py` but no
`main.py`, a thin `main.py` re-exporting `agent` is created so every playable
bot is importable as `<slug>.main`.

Run:  .venv/bin/python agents/opponents/extract_bots.py
"""
from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent
WRITEFILE_RE = re.compile(r"^\s*%%writefile\s+(-a\s+)?(?P<path>\S+)\s*$")


def cell_source(cell: dict) -> str:
    src = cell.get("source", "")
    return "".join(src) if isinstance(src, list) else src


def extract_notebook(nb_path: Path) -> dict:
    """Return {written: [paths], magics: [target_strings]} for one notebook."""
    folder = nb_path.parent
    nb = json.loads(nb_path.read_text())
    written, magics = [], []

    for cell in nb.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        body = cell_source(cell)
        lines = body.splitlines(keepends=True)
        if not lines:
            continue
        m = WRITEFILE_RE.match(lines[0])
        if not m:
            continue

        target = m.group("path")
        append = bool(m.group(1))
        magics.append(target)

        # Map /kaggle/working/foo.py or absolute paths to a folder-relative path.
        rel = target
        for prefix in ("/kaggle/working/", "/kaggle/input/", "/kaggle/"):
            if rel.startswith(prefix):
                rel = rel[len(prefix):]
        rel = rel.lstrip("/")

        out_path = folder / rel
        out_path.parent.mkdir(parents=True, exist_ok=True)
        content = "".join(lines[1:])
        mode = "a" if append and out_path.exists() else "w"
        with open(out_path, mode) as f:
            f.write(content)
        written.append(str(out_path.relative_to(ROOT)))

    return {"written": written, "magics": magics}


def main():
    summary = {}
    for nb_path in sorted(ROOT.glob("*/*.ipynb")):
        slug = nb_path.parent.name
        res = extract_notebook(nb_path)
        summary[slug] = res
        files = res["written"]
        print(f"[{slug}]")
        if files:
            for fp in files:
                print(f"    wrote {fp}")
        else:
            print(f"    (no %%writefile cells — inline agent or reference notebook)")
    print("\n=== entrypoint normalization ===")
    for nb_path in sorted(ROOT.glob("*/*.ipynb")):
        folder = nb_path.parent
        has_main = (folder / "main.py").exists()
        has_sub = (folder / "submission.py").exists()
        if has_sub and not has_main:
            # thin re-export so every bot imports as <slug>.main
            (folder / "main.py").write_text(
                "# auto-generated entrypoint: re-exports agent from submission.py\n"
                "from submission import *  # noqa\n"
                "from submission import agent  # noqa\n"
            )
            print(f"[{folder.name}] created main.py -> submission.py")
    return summary


if __name__ == "__main__":
    main()
