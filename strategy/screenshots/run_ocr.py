#!/usr/bin/env python3
"""
Orbit Wars screenshot OCR pipeline.

Steps:
  1. Deduplicate images by perceptual hash (exact pixel duplicates).
  2. Rename all surviving images sequentially: 001-short-description.png
     (description generated from filename using simple slug rules — no model call needed).
  3. Send each image to LM Studio local API (OpenAI-compatible, multimodal) with the OCR prompt.
  4. Write results to:
       strategy/08-SCREENSHOT_OCR_FULL.json   (structured, one object per image)
       strategy/08-SCREENSHOT_OCR_FULL.txt    (human-readable, one section per image)

Usage:
  uv pip install pillow imagehash openai tqdm
  python run_ocr.py [--dry-run] [--lmstudio-url http://localhost:1234]

Set LMSTUDIO_MODEL env var to override the model name, e.g.:
  LMSTUDIO_MODEL="mistralai/ministral-3-14b-reasoning" python run_ocr.py
"""

import argparse
import base64
import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path

try:
    import imagehash
    from PIL import Image
except ImportError:
    print("Missing deps. Run: uv pip install pillow imagehash openai tqdm")
    sys.exit(1)

try:
    from openai import OpenAI
except ImportError:
    print("Missing openai. Run: uv pip install openai")
    sys.exit(1)

try:
    from tqdm import tqdm
except ImportError:
    tqdm = lambda x, **kw: x  # noqa: E731

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).parent.resolve()
SCREENSHOTS_DIR = SCRIPT_DIR          # images live alongside this script
OUTPUT_DIR = SCRIPT_DIR.parent        # strategy/
JSON_OUT = OUTPUT_DIR / "08-SCREENSHOT_OCR_FULL.json"
TXT_OUT  = OUTPUT_DIR / "08-SCREENSHOT_OCR_FULL.txt"

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
LMSTUDIO_URL   = os.environ.get("LMSTUDIO_URL", "http://localhost:1234")
MODEL_NAME     = os.environ.get("LMSTUDIO_MODEL", "moondream/moondream-2b-2025-04-14-4bit")
HASH_THRESHOLD = 5   # hamming distance — images within this are considered duplicates
MAX_RETRIES    = 3
RETRY_DELAY    = 5   # seconds between retries

SYSTEM_PROMPT = (
    "You are a precise OCR and analysis assistant. Your job is to extract ALL text visible "
    "in screenshots and provide a structured summary. Be exhaustive — capture every word, "
    "number, table row, terminal line, and UI label you can see. Transcribe verbatim; "
    "do not paraphrase or omit text."
)

USER_PROMPT = """\
For this screenshot, produce a JSON object with exactly these fields:

{{
  "filename": "<the renamed filename of this image>",
  "short_filename": "<3-5 hyphen-separated lowercase words describing what is visible — used as a filename slug. Examples: leaderboard-rank-140, autoresearch-morning-session, rl-entropy-collapse, kaggle-submission-score-1234, game-board-5v5-domination, streamlit-dashboard-winrate. No extension, no spaces, no special characters other than hyphens.>",
  "ocr_text": "<verbatim transcription of ALL visible text, preserving structure with newlines>",
  "summary": "<2-4 sentence description: what this shows and why it matters in a Kaggle competition project>",
  "category": "<one of: game_board | dashboard | terminal_orchestrator | terminal_agent | autoresearch | fleet_monitor | submission | leaderboard | code>",
  "key_numbers": "<any important metrics, scores, dates, percentages — comma separated>"
}}

The filename for this image is: {filename}

Return ONLY the JSON object — no markdown fences, no explanation before or after.
If the image is blank or unreadable, set ocr_text to "[UNREADABLE]", summary to "[blank or unreadable image]", and short_filename to "unreadable-image".
"""

# ---------------------------------------------------------------------------
# Step 1: collect & deduplicate
# ---------------------------------------------------------------------------

def collect_images(folder: Path) -> list[Path]:
    """Return all image files sorted by creation date (birth time on macOS)."""
    exts = {".png", ".jpg", ".jpeg", ".gif", ".webp"}
    candidates = [
        p for p in folder.iterdir()
        if p.suffix.lower() in exts and p.name != Path(__file__).name
    ]

    def birth_time(p: Path) -> float:
        st = p.stat()
        return getattr(st, "st_birthtime", st.st_mtime)

    return sorted(candidates, key=birth_time)


def deduplicate(files: list[Path], threshold: int = HASH_THRESHOLD) -> list[Path]:
    """Remove near-duplicate images. Keeps the first seen; deletes the rest."""
    seen: dict[imagehash.ImageHash, Path] = {}
    kept: list[Path] = []
    removed = 0

    for p in files:
        try:
            h = imagehash.phash(Image.open(p))
        except Exception as e:
            print(f"  [WARN] Could not hash {p.name}: {e} — keeping it")
            kept.append(p)
            continue

        duplicate_of = None
        for existing_hash, existing_path in seen.items():
            if abs(h - existing_hash) <= threshold:
                duplicate_of = existing_path
                break

        if duplicate_of:
            print(f"  [DUP] {p.name}  ≈  {duplicate_of.name}  → deleting duplicate")
            p.unlink()
            removed += 1
        else:
            seen[h] = p
            kept.append(p)

    print(f"\nDeduplication: kept {len(kept)}, removed {removed} duplicates.\n")
    return kept


# ---------------------------------------------------------------------------
# Step 2: OCR via LM Studio
# ---------------------------------------------------------------------------

def image_to_b64(path: Path) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def safe_slug(s: str, maxlen: int = 50) -> str:
    """Sanitise a model-generated short_filename to be safe as a path component."""
    s = s.lower().strip()
    s = re.sub(r"[^a-z0-9\-]", "-", s)
    s = re.sub(r"-+", "-", s).strip("-")
    return s[:maxlen] or "screenshot"


# ---------------------------------------------------------------------------
# macOS Vision OCR (no API, no model download)
# ---------------------------------------------------------------------------

def _categorize_and_slug(text: str) -> tuple[str, str]:
    """Derive category + short_filename from OCR text using keyword rules."""
    t = text.lower()
    nums = re.findall(r'\b(\d{3,5})\b', text)
    first_num = nums[0] if nums else ""

    if any(w in t for w in ["leaderboard", "public score", "private score", "rank #", "elo"]):
        return "leaderboard", f"leaderboard-rank-{first_num}" if first_num else "kaggle-leaderboard"
    if any(w in t for w in ["submission", "submitted", "submit agent"]):
        return "submission", "kaggle-submission"
    if any(w in t for w in ["entropy", "clip_frac", "explained_var", "ppo update", "rollout"]):
        return "terminal_agent", f"rl-training-step-{first_num}" if first_num else "rl-training-metrics"
    if any(w in t for w in ["orchestrator", "auditor", "hypothesis", "verdict", "experiment log"]):
        return "autoresearch", "autoresearch-session"
    if any(w in t for w in ["win rate", "winrate", "gauntlet", "vs comet", "schmeekler"]):
        return "dashboard", "dashboard-winrate-chart"
    if any(w in t for w in ["streamlit", "dashboard", "score history"]):
        return "dashboard", "streamlit-dashboard"
    if any(w in t for w in ["tmux", "fleet", "seed", "instance", "jetstream"]):
        return "fleet_monitor", "tmux-fleet-monitor"
    if any(w in t for w in ["planet", "garrison", "production", "orbit", "game over"]):
        return "game_board", "game-board-state"
    if any(w in t for w in ["def ", "class ", "import ", "return ", "function"]):
        return "code", "code-editor"
    return "terminal_agent", "terminal-session"


def ocr_image_macos(image_path: Path) -> dict:
    """OCR using Apple Vision framework — no API, no model download needed."""
    try:
        import Vision
        import Quartz
    except ImportError:
        print("Missing pyobjc. Run: uv pip install pyobjc-framework-Vision pyobjc-framework-Quartz")
        sys.exit(1)

    try:
        url = Quartz.NSURL.fileURLWithPath_(str(image_path.resolve()))
        handler = Vision.VNImageRequestHandler.alloc().initWithURL_options_(url, {})
        request = Vision.VNRecognizeTextRequest.alloc().init()
        request.setRecognitionLevel_(Vision.VNRequestTextRecognitionLevelAccurate)
        request.setUsesLanguageCorrection_(True)
        handler.performRequests_error_([request], None)

        lines = []
        for obs in (request.results() or []):
            candidates = obs.topCandidates_(1)
            if candidates:
                lines.append(candidates[0].string())

        ocr_text = "\n".join(lines) if lines else "[UNREADABLE]"
        category, short_filename = _categorize_and_slug(ocr_text)
        key_numbers = ", ".join(re.findall(r'\b\d{3,5}(?:\.\d+)?\b', ocr_text)[:8])

        return {
            "filename": image_path.name,
            "short_filename": short_filename,
            "ocr_text": ocr_text,
            "summary": f"[{category}] {ocr_text[:300].strip()}",
            "category": category,
            "key_numbers": key_numbers,
        }
    except Exception as e:
        return {
            "filename": image_path.name,
            "short_filename": "vision-error",
            "ocr_text": f"[ERROR: {e}]",
            "summary": f"[Vision error: {e}]",
            "category": "unknown",
            "key_numbers": "",
        }


def ocr_image(client: OpenAI, image_path: Path, dry_run: bool = False, model: str = MODEL_NAME) -> dict:
    if dry_run:
        return {
            "filename": image_path.name,
            "short_filename": "dry-run-placeholder",
            "ocr_text": "[DRY RUN]",
            "summary": "[DRY RUN]",
            "category": "unknown",
            "key_numbers": "",
        }

    b64 = image_to_b64(image_path)
    prompt = USER_PROMPT.format(filename=image_path.name)

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{b64}",
                                    "detail": "high",
                                },
                            },
                            {"type": "text", "text": prompt},
                        ],
                    },
                ],
                temperature=0.1,
                max_tokens=2048,
            )
            raw = response.choices[0].message.content.strip()

            # Strip markdown fences if model added them anyway
            if raw.startswith("```"):
                raw = re.sub(r"^```[a-z]*\n?", "", raw)
                raw = re.sub(r"\n?```$", "", raw)

            result = json.loads(raw)
            result["filename"] = image_path.name  # always use actual filename
            return result

        except json.JSONDecodeError as e:
            print(f"    [WARN] JSON parse failed (attempt {attempt}/{MAX_RETRIES}): {e}")
            print(f"    Raw response: {raw[:300]}")
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY)
            else:
                return {
                    "filename": image_path.name,
                    "short_filename": "parse-error",
                    "ocr_text": raw,
                    "summary": "[JSON parse failed — raw text stored in ocr_text]",
                    "category": "unknown",
                    "key_numbers": "",
                }
        except Exception as e:
            print(f"    [ERROR] API call failed (attempt {attempt}/{MAX_RETRIES}): {e}")
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY)
            else:
                return {
                    "filename": image_path.name,
                    "short_filename": "api-error",
                    "ocr_text": "[API ERROR]",
                    "summary": f"[API error: {e}]",
                    "category": "unknown",
                    "key_numbers": "",
                }


# ---------------------------------------------------------------------------
# Step 4: write outputs
# ---------------------------------------------------------------------------

def write_outputs(results: list[dict]) -> None:
    # JSON
    with open(JSON_OUT, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nJSON written → {JSON_OUT}")

    # TXT
    with open(TXT_OUT, "w", encoding="utf-8") as f:
        for r in results:
            f.write(f"{'='*80}\n")
            f.write(f"FILE: {r['filename']}\n")
            f.write(f"CATEGORY: {r.get('category','')}\n")
            f.write(f"KEY NUMBERS: {r.get('key_numbers','')}\n")
            f.write(f"\nSUMMARY:\n{r.get('summary','')}\n")
            f.write(f"\nOCR TEXT:\n{r.get('ocr_text','')}\n\n")
    print(f"TXT  written → {TXT_OUT}")


# ---------------------------------------------------------------------------
# Checkpoint helpers (resume interrupted runs)
# ---------------------------------------------------------------------------

def load_checkpoint() -> dict[str, dict]:
    """Load already-processed results keyed by filename."""
    if not JSON_OUT.exists():
        return {}
    try:
        with open(JSON_OUT, encoding="utf-8") as f:
            data = json.load(f)
        return {r["filename"]: r for r in data}
    except Exception:
        return {}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

GEMINI_BASE_URL    = "https://generativelanguage.googleapis.com/v1beta/openai/"
GEMINI_MODEL       = "gemini-2.0-flash"
OLLAMA_BASE_URL    = "http://localhost:11434/v1"
OLLAMA_MODEL       = "qwen3-vl:235b-cloud"


def main():
    parser = argparse.ArgumentParser(description="Orbit Wars screenshot OCR pipeline")
    parser.add_argument("--dry-run", action="store_true", help="Skip API calls, test pipeline only")
    parser.add_argument("--api", choices=["lmstudio", "gemini", "ollama", "macos"], default="lmstudio",
                        help="Which API backend to use (default: lmstudio)")
    parser.add_argument("--lmstudio-url", default=LMSTUDIO_URL)
    parser.add_argument("--model", default=None, help="Override model name")
    parser.add_argument("--no-rename", action="store_true", help="Skip post-OCR renaming (keep original filenames)")
    parser.add_argument("--no-dedup", action="store_true", help="Skip deduplication")
    args = parser.parse_args()

    use_macos_vision = args.api == "macos"

    if args.api == "gemini":
        api_key = os.environ.get("GOOGLE_API_KEY", "")
        if not api_key:
            print("ERROR: set GOOGLE_API_KEY env var to your Google AI API key")
            sys.exit(1)
        base_url = GEMINI_BASE_URL
        model = args.model or os.environ.get("GEMINI_MODEL", GEMINI_MODEL)
    elif args.api == "ollama":
        api_key = "ollama"
        base_url = OLLAMA_BASE_URL
        model = args.model or OLLAMA_MODEL
    elif args.api == "macos":
        api_key = model = base_url = None  # unused
    else:
        api_key = "lm-studio"
        base_url = f"{args.lmstudio_url}/v1"
        model = args.model or MODEL_NAME

    print("=" * 60)
    print("Orbit Wars Screenshot OCR Pipeline")
    print("=" * 60)
    print(f"Screenshots folder : {SCREENSHOTS_DIR}")
    print(f"API backend        : {args.api}")
    if not use_macos_vision:
        print(f"Model              : {model}")
    print(f"Dry run            : {args.dry_run}")
    print()

    # Collect
    files = collect_images(SCREENSHOTS_DIR)
    print(f"Found {len(files)} images.\n")

    # Deduplicate
    if not args.no_dedup:
        files = deduplicate(files)

    # Load checkpoint (skip already-done images on resume)
    # Keyed by current filename — already-renamed files match by their new name.
    done = load_checkpoint()
    remaining = [f for f in files if f.name not in done]
    print(f"Already processed : {len(done)}")
    print(f"To process now    : {len(remaining)}\n")

    if not remaining:
        print("Nothing left to process. Writing outputs from checkpoint...")
        results = list(done.values())
        write_outputs(results)
        return

    # OCR — sequence numbers are position in the full sorted list (stable across resumes)
    client = None if use_macos_vision else OpenAI(base_url=base_url, api_key=api_key)

    # Build a map of seq number for every file (including already-done ones)
    seq_map = {p.name: i for i, p in enumerate(files, start=1)}

    results = list(done.values())
    for image_path in tqdm(remaining, desc="OCR"):
        seq = seq_map[image_path.name]
        print(f"\nProcessing {seq}/{len(files)}: {image_path.name}")
        if use_macos_vision:
            result = ocr_image_macos(image_path)
        else:
            result = ocr_image(client, image_path, dry_run=args.dry_run, model=model)

        # Rename file to 001-model-generated-description.png (skip in dry-run)
        if not args.no_rename and not args.dry_run:
            slug = safe_slug(result.get("short_filename", "") or "screenshot")
            new_name = f"{seq:03d}-{slug}.png"
            new_path = image_path.parent / new_name
            if image_path.name != new_name:
                if new_path.exists() and new_path != image_path:
                    new_path = image_path.parent / f"{seq:03d}-{slug}-b.png"
                image_path.rename(new_path)
                print(f"  → renamed to {new_path.name}")
            result["filename"] = new_path.name
        else:
            result["filename"] = image_path.name

        results.append(result)

        # Save checkpoint after every image (skip in dry-run — nothing real was done)
        if not args.dry_run:
            write_outputs(results)

    print(f"\nDone! Processed {len(results)} images total.")
    print(f"  JSON → {JSON_OUT}")
    print(f"  TXT  → {TXT_OUT}")


if __name__ == "__main__":
    main()
