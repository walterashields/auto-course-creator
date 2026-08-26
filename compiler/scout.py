#!/usr/bin/env python3
"""
compiler/scout.py

Environment scouting pass: observe the real application + database state before
script generation so scripts can assert only observed facts and skip actions
that are already satisfied by the default launch state.
"""

from __future__ import annotations

import json
import os
import sqlite3
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import anthropic
from PIL import Image

from .discovery import APP_NAME, TARGET_LONG_EDGE, _capture_screenshot

MODEL = os.environ.get("DISCOVERY_MODEL", "claude-sonnet-5")


def _quit_app_cleanly() -> None:
    """Best-effort polite quit of DB Browser so the real run starts fresh."""
    try:
        subprocess.run(
            ["osascript", "-e", f'tell application "{APP_NAME}" to quit'],
            capture_output=True,
            timeout=5,
        )
    except Exception:
        pass
    try:
        subprocess.run(
            ["pkill", "-x", APP_NAME],
            capture_output=True,
            timeout=5,
        )
    except Exception:
        pass
    time.sleep(2)


def _list_tables_and_counts(db_path: str) -> Tuple[List[str], Dict[str, List[str]], Dict[str, int]]:
    """Read table names, columns per table, and exact row counts from SQLite."""
    tables: List[str] = []
    columns: Dict[str, List[str]] = {}
    row_counts: Dict[str, int] = {}
    if not db_path or not Path(db_path).exists():
        return tables, columns, row_counts

    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()
    cursor.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY rowid"
    )
    tables = [row[0] for row in cursor.fetchall()]
    for table in tables:
        cursor.execute(f"PRAGMA table_info({table})")
        columns[table] = [row[1] for row in cursor.fetchall()]
        try:
            cursor.execute(f"SELECT COUNT(*) FROM {table}")
            row_counts[table] = int(cursor.fetchone()[0])
        except Exception as exc:
            print(f"Warning: could not count rows in {table}: {exc}", file=sys.stderr)
            row_counts[table] = 0
    conn.close()
    return tables, columns, row_counts


def _launch_app_for_scout(db_path: str) -> None:
    """Open DB Browser for SQLite on the provided database file."""
    subprocess.run(
        ["open", "-a", APP_NAME, str(db_path)],
        check=True,
        capture_output=True,
        timeout=30,
    )
    time.sleep(6)
    subprocess.run(
        ["osascript", "-e", f'tell application "{APP_NAME}" to activate'],
        check=True,
        capture_output=True,
        timeout=10,
    )
    time.sleep(1)

    # Maximize/front the window to give the model a consistent canvas.
    try:
        subprocess.run(
            [
                "osascript",
                "-e",
                f"""
                tell application "{APP_NAME}" to activate
                delay 0.5
                tell application "System Events"
                    tell process "{APP_NAME}"
                        if exists (window 1) then
                            set position of window 1 to {{0, 0}}
                            set size of window 1 to {{1920, 1200}}
                        end if
                    end tell
                end tell
                """,
            ],
            check=True,
            capture_output=True,
            timeout=15,
        )
        time.sleep(1)
    except Exception:
        pass


def _vision_scout(client: anthropic.Anthropic, output_dir: Path) -> Dict[str, Any]:
    """Ask Claude one question about the current DB Browser screenshot."""
    b64, _, _, _, _, _ = _capture_screenshot(output_dir)
    prompt = (
        "You are observing DB Browser for SQLite immediately after launch. "
        "Look at the screenshot and answer with ONLY a JSON object in this exact shape:\n\n"
        "{\n"
        '  "active_tab": "Name of the currently selected tab",\n'
        '  "available_tabs": ["Browse Data", "Edit Pragmas", ...],\n'
        '  "browse_data_default_table": "Name of the table shown in Browse Data, or null",\n'
        '  "notable": "Any visible UI state worth mentioning (open panels, dialogs, error messages, etc.)"\n'
        "}\n\n"
        "Use null for unknown values. Do not add any other text."
    )
    response = client.messages.create(
        model=MODEL,
        max_tokens=512,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": b64}},
                    {"type": "text", "text": prompt},
                ],
            }
        ],
    )
    text_parts = [block.text for block in response.content if block.type == "text"]
    raw = "\n".join(text_parts).strip()

    # Extract JSON from fenced block or raw text.
    data: Dict[str, Any] = {
        "active_tab": None,
        "available_tabs": [],
        "browse_data_default_table": None,
        "notable": "",
    }
    fenced = __import__("re").search(r"```(?:json)?\s*(\{.*\})\s*```", raw, __import__("re").DOTALL)
    payload = fenced.group(1) if fenced else raw
    try:
        parsed = json.loads(payload)
        if isinstance(parsed, dict):
            data.update(parsed)
    except json.JSONDecodeError:
        print(f"Warning: could not parse scout vision response as JSON: {raw[:200]}", file=sys.stderr)

    return data


def scout_environment(
    db_path: str,
    application: str,
    video_id: Optional[str] = None,
    output_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Scout the environment for a single video.

    Returns an EnvironmentMap dict with database facts (from sqlite3) and UI
    facts (from one Claude vision call). Also writes the map to
    <output_dir>/<video_id>_env.json if both are provided.

    The application is launched, observed, and then quit cleanly so the real
    discovery run starts from a fresh state.
    """
    if application != "db_browser_sqlite":
        raise ValueError(f"Scout not implemented for application: {application}")

    discovery_output_dir = Path(__file__).resolve().parent / "discovery_output"
    discovery_output_dir.mkdir(parents=True, exist_ok=True)
    if output_dir is None:
        output_dir = discovery_output_dir

    tables, columns, row_counts = _list_tables_and_counts(db_path)

    # Launch, observe, quit.
    _launch_app_for_scout(db_path)
    client = anthropic.Anthropic()
    vision_facts = _vision_scout(client, discovery_output_dir)
    _quit_app_cleanly()

    env_map: Dict[str, Any] = {
        "video_id": video_id,
        "application": application,
        "db_path": str(Path(db_path).resolve()),
        "tables": tables,
        "columns": columns,
        "row_counts": row_counts,
        "ui": vision_facts,
    }

    if video_id:
        env_path = output_dir / f"{video_id}_env.json"
        env_path.write_text(json.dumps(env_map, indent=2), encoding="utf-8")

    return env_map
