#!/usr/bin/env python3
"""
compiler/curriculum.py

CourseManifest and VideoManifest schemas plus the multi-video pipeline.
"""

from __future__ import annotations

import argparse
import difflib
import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple

from pydantic import BaseModel, Field

from .discovery import EndStateDiscovery
from .frame_analysis import (
    compute_video_metrics,
    count_error_signature_frames,
    format_gate_table,
    frozen_share_percent,
    reconcile_summary,
    run_acceptance_gates,
)
from .graph_store import GraphStore
from .lesson_builder import LessonBuilder
from .narrator import ScriptBeat
from .renderer import GraphRenderer
from .schemas import DiscoveryResult, EnvironmentProfile
from .scout import scout_environment
from .tts import TTSGenerator
from .vision_agent import VisionAgent
from .cost_tracker import CostExhaustedError, get_tracker, reset_tracker


# ---------------------------------------------------------------------------
# Loop ceilings (documented for attempt reports)
# ---------------------------------------------------------------------------

_LOOP_CEILINGS: List[Dict[str, Any]] = [
    {"file": "compiler/curriculum.py", "line": 551, "loop": "_resolve_controlling_terminal process walk", "ceiling": "PID <= 1"},
    {"file": "compiler/curriculum.py", "line": 1170, "loop": "run_course seed-DB loop", "ceiling": "len(ordered_videos)"},
    {"file": "compiler/curriculum.py", "line": 1220, "loop": "run_course video pipeline", "ceiling": "len(ordered_videos)"},
    {"file": "compiler/curriculum.py", "line": 1586, "loop": "main --only-video iteration", "ceiling": "args.max_iterations (default 10)"},
    {"file": "compiler/curriculum_designer.py", "line": 616, "loop": "CurriculumDesigner.design validation/fix", "ceiling": "2"},
    {"file": "compiler/curriculum_designer.py", "line": 763, "loop": "_generate_design LLM retries", "ceiling": "3"},
    {"file": "compiler/curriculum_designer.py", "line": 849, "loop": "_enrich_design LLM retries", "ceiling": "3"},
    {"file": "compiler/discovery.py", "line": 227, "loop": "ScreenRecorder._capture_loop", "ceiling": "MAX_SCREEN_RECORDER_FRAMES (~3h @ 10fps)"},
    {"file": "compiler/discovery.py", "line": 414, "loop": "_ScreenCaptureKitRecorder._writer_loop", "ceiling": "MAX_SC_RECORDER_SAMPLES (~3h @ 10fps)"},
    {"file": "compiler/discovery.py", "line": 1972, "loop": "_execute_action_script recipe steps", "ceiling": "len(actions)"},
    {"file": "compiler/discovery.py", "line": 2053, "loop": "_execute_action_script target fallback", "ceiling": "len(targets_to_try)"},
    {"file": "compiler/discovery.py", "line": 2374, "loop": "EndStateDiscovery.discover vision loop", "ceiling": "max_attempts (default 10)"},
    {"file": "compiler/discovery.py", "line": 2523, "loop": "_looks_like_text_input type forcing", "ceiling": "len(type_values)"},
    {"file": "compiler/discovery.py", "line": 3061, "loop": "_execute_beats_with_agent beat loop", "ceiling": "len(beats)"},
    {"file": "compiler/discovery.py", "line": 3173, "loop": "_execute_beats_with_agent per-beat retries", "ceiling": "MAX_BEAT_RETRIES = WSDA_MAX_ATTEMPTS (default 3)"},
    {"file": "compiler/discovery.py", "line": 3172, "loop": "_execute_beats_with_agent inner action attempts", "ceiling": "1 for typing, 3 otherwise"},
    {"file": "compiler/discovery.py", "line": 3655, "loop": "_wait_for_visual_stability", "ceiling": "timeout_seconds (default 4.0s)"},
    {"file": "compiler/discovery.py", "line": 3787, "loop": "_trim_clip_to_motion frame decode", "ceiling": "MAX_TRIM_FRAMES (1_000_000)"},
    {"file": "compiler/discovery.py", "line": 3895, "loop": "_auto_fit_columns", "ceiling": "4"},
    {"file": "compiler/discovery.py", "line": 3971, "loop": "_auto_fit_if_truncated", "ceiling": "3"},
    {"file": "compiler/discovery.py", "line": 4040, "loop": "_results_grid_visible", "ceiling": "max_retries + 1 (default 3)"},
    {"file": "compiler/lesson_builder.py", "line": 272, "loop": "_complete_beat_text padding", "ceiling": "20"},
    {"file": "compiler/lesson_builder.py", "line": 1124, "loop": "validation word padding", "ceiling": "wc < 10"},
    {"file": "compiler/lesson_builder.py", "line": 1581, "loop": "SQL validation word padding", "ceiling": "wc < 15"},
    {"file": "compiler/lesson_builder.py", "line": 2629, "loop": "generate_script LLM retries", "ceiling": "max_attempts (default 1)"},
    {"file": "compiler/renderer.py", "line": 696, "loop": "_build_frames", "ceiling": "len(beats)"},
    {"file": "compiler/tts.py", "line": 209, "loop": "_synthesize_with_curl", "ceiling": "3"},
    {"file": "compiler/vision_agent.py", "line": 234, "loop": "_ensure_frontmost", "ceiling": "max_attempts (default 3)"},
    {"file": "compiler/vision_agent.py", "line": 471, "loop": "_assess_and_maybe_repair", "ceiling": "max_attempts (default 2)"},
    {"file": "compiler/vision_agent.py", "line": 1410, "loop": "type_segments segment loop", "ceiling": "len(segments)"},
    {"file": "compiler/vision_agent.py", "line": 1450, "loop": "type_segments non-recording retry", "ceiling": "2"},
    {"file": "compiler/vision_agent.py", "line": 1512, "loop": "type_block retry", "ceiling": "2"},
    {"file": "compiler/vision_agent.py", "line": 1584, "loop": "append_block retry", "ceiling": "2"},
    {"file": "compiler/vision_agent.py", "line": 2198, "loop": "is_end_state_already_present line parse", "ceiling": "len(lines)"},
    {"file": "compiler/vision_agent.py", "line": 2580, "loop": "verify_app_visible_in_frames batches", "ceiling": "len(frame_paths) in batches of 5"},
]


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------


class VideoManifest(BaseModel):
    """A single video in a course."""

    video_id: str
    title: str
    learning_objective: str
    discovery_objective: str
    application: str
    prerequisite_videos: List[str] = Field(default_factory=list)
    exercise_artifact: dict = Field(default_factory=dict)
    format_tier: Literal["micro", "short", "mid", "long", "full"]
    estimated_duration_seconds: int = 0
    # Fields populated by the curriculum designer.
    video_type: Literal[
        "orientation", "concept", "demo", "exercise", "anti-pattern", "capstone"
    ] = "demo"
    new_capability: Optional[str] = None
    key_concept: Optional[str] = None
    prerequisite_knowledge: Optional[str] = None
    running_example_usage: Optional[str] = None
    proof_numbers: Optional[str] = None
    estimated_word_count: int = 0
    has_recap: bool = False
    has_preview: bool = False
    recap_text_hint: Optional[str] = None
    preview_text_hint: Optional[str] = None
    # Serialized script beats (Path A lesson-first architecture).
    script_beats: List[dict] = Field(default_factory=list)
    # Ground-truth SELECT queries to run during the scout pass.
    planned_queries: List[str] = Field(default_factory=list)


class CourseManifest(BaseModel):
    """A complete course composed of one or more videos."""

    course_id: str
    title: str
    description: str
    target_audience: str
    videos: List[VideoManifest]
    running_example: dict = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def _courses_dir() -> Path:
    """Return the directory where course manifests are persisted."""
    return Path(__file__).resolve().parent / "courses"


def save_manifest(manifest: CourseManifest) -> Path:
    """Save a CourseManifest to compiler/courses/<course_id>.json."""
    courses_dir = _courses_dir()
    courses_dir.mkdir(parents=True, exist_ok=True)
    path = courses_dir / f"{manifest.course_id}.json"
    path.write_text(manifest.model_dump_json(indent=2), encoding="utf-8")
    return path


def load_manifest(course_id: str) -> Optional[CourseManifest]:
    """Load a CourseManifest by course id, or None if not found."""
    path = _courses_dir() / f"{course_id}.json"
    if not path.exists():
        return None
    return CourseManifest.model_validate_json(path.read_text(encoding="utf-8"))


def list_manifests() -> List[str]:
    """Return a list of persisted course ids."""
    courses_dir = _courses_dir()
    if not courses_dir.exists():
        return []
    return sorted(
        p.stem for p in courses_dir.glob("*.json") if p.is_file()
    )


# ---------------------------------------------------------------------------
# Hardcoded template
# ---------------------------------------------------------------------------


def create_sql_sorting_fundamentals() -> CourseManifest:
    """
    A 3-video course using the same Orders database.
    Each video teaches exactly one new capability.
    """
    exercise = {
        "db_path": str((Path(__file__).resolve().parent / "discovery_output" / "sample.db").resolve()),
        "table_name": "Orders",
        "description": "Sample Orders table with id, region, order_date, and amount columns.",
    }

    videos = [
        VideoManifest(
            video_id="video_1_1",
            title="Browse the Orders Table",
            learning_objective="Open the Orders table in DB Browser and display all rows",
            discovery_objective=(
                "Open the Orders table in the Browse Data tab and display all rows "
                "with columns id, region, order_date, and amount visible"
            ),
            application="db_browser_sqlite",
            prerequisite_videos=[],
            exercise_artifact=exercise,
            format_tier="short",
            has_preview=True,
            preview_text_hint="Next we'll sort the Orders table by amount to see the highest and lowest values.",
        ),
        VideoManifest(
            video_id="video_1_2",
            title="Sort by Amount Ascending",
            learning_objective=(
                "Sort the Orders table by the amount column in ascending order "
                "(smallest amount 85.0 at the top, largest 340.0 at the bottom)"
            ),
            discovery_objective=(
                "Click the Amount column header in the Orders table so the smallest "
                "amount 85.0 appears at the top and the largest 340.0 appears at the bottom"
            ),
            application="db_browser_sqlite",
            prerequisite_videos=["video_1_1"],
            exercise_artifact=exercise,
            format_tier="short",
            has_recap=True,
            recap_text_hint="In the last video we opened the Orders table and saw all rows with id, region, order_date, and amount columns.",
            has_preview=True,
            preview_text_hint="Next we'll flip the sort to descending so the largest amounts appear at the top.",
        ),
        VideoManifest(
            video_id="video_1_3",
            title="Sort by Amount Descending",
            learning_objective=(
                "Sort the Orders table by the amount column in descending order "
                "(largest amount 340.0 at the top, smallest 85.0 at the bottom)"
            ),
            discovery_objective=(
                "Click the Amount column header in the Orders table a second time so "
                "the largest amount 340.0 appears at the top and the smallest 85.0 appears at the bottom"
            ),
            application="db_browser_sqlite",
            prerequisite_videos=["video_1_2"],
            exercise_artifact=exercise,
            format_tier="short",
            has_recap=True,
            recap_text_hint="In the last video we sorted the Orders table by amount ascending, with the smallest amount 85.0 at the top.",
            has_preview=True,
            preview_text_hint="Next we'll filter the Orders table to show only rows from one region.",
        ),
    ]

    return CourseManifest(
        course_id="sql_sorting_fundamentals",
        title="SQL Sorting Fundamentals",
        description=(
            "Learn to browse, sort ascending, and sort descending in DB Browser for SQLite"
        ),
        target_audience="Beginner data analysts with no SQL experience",
        videos=videos,
        running_example={
            "name": "Orders",
            "description": "A simple e-commerce orders table.",
            "schema": {
                "id": "INTEGER PRIMARY KEY",
                "region": "TEXT",
                "order_date": "TEXT",
                "amount": "REAL",
            },
        },
    )


def create_sql_essential_training_ch4() -> CourseManifest:
    """
    Phase 2 chapter 4: five videos teaching a first SELECT query, aliases,
    ORDER BY, LIMIT, and a recap query against the WSDA Music database.
    """
    db_path = str((Path(__file__).resolve().parent / "data" / "wsda_music.db").resolve())
    exercise = {
        "db_path": db_path,
        "table_name": "Customer",
        "description": "WSDA Music Customer table with FirstName, LastName, Email.",
    }

    videos = [
        VideoManifest(
            video_id="video_1_1",
            title="Your First Query",
            learning_objective="Write and run your first SELECT query to return a customer contact list",
            discovery_objective=(
                "Open the Execute SQL tab, type the commented query SELECT FirstName, LastName, "
                "Email FROM Customer, run it with F5, and show the result pane with the contact list"
            ),
            application="db_browser_sqlite",
            prerequisite_videos=[],
            exercise_artifact=exercise,
            format_tier="long",
            planned_queries=[
                "SELECT FirstName, LastName, Email FROM Customer;",
            ],
        ),
        VideoManifest(
            video_id="video_1_2",
            title="Aliases — Speaking Management's Language",
            learning_objective="Use the AS keyword to give query results readable column headers",
            discovery_objective=(
                "Open the Execute SQL tab, type SELECT FirstName AS First Name, LastName AS Last Name, "
                "Email AS Email Address FROM Customer, run it with F5, and show the aliased headers"
            ),
            application="db_browser_sqlite",
            prerequisite_videos=["video_1_1"],
            exercise_artifact=exercise,
            format_tier="long",
            planned_queries=[
                'SELECT FirstName AS "First Name", LastName AS "Last Name", Email AS "Email Address" FROM Customer;',
            ],
        ),
        VideoManifest(
            video_id="video_1_3",
            title="Sorting Results with ORDER BY",
            learning_objective="Sort query results by a specific column using ORDER BY",
            discovery_objective=(
                "Open the Execute SQL tab, type SELECT FirstName, LastName, Email FROM Customer "
                "ORDER BY LastName, run it with F5, and show the result pane sorted by LastName"
            ),
            application="db_browser_sqlite",
            prerequisite_videos=["video_1_2"],
            exercise_artifact=exercise,
            format_tier="long",
            planned_queries=[
                "SELECT FirstName, LastName, Email FROM Customer ORDER BY LastName;",
            ],
        ),
        VideoManifest(
            video_id="video_1_4",
            title="Limiting Results with LIMIT",
            learning_objective="Limit the number of returned rows using the LIMIT clause",
            discovery_objective=(
                "Open the Execute SQL tab, type SELECT FirstName, LastName, Email FROM Customer "
                "ORDER BY LastName LIMIT 5, run it with F5, and show exactly five rows in the result pane"
            ),
            application="db_browser_sqlite",
            prerequisite_videos=["video_1_3"],
            exercise_artifact=exercise,
            format_tier="long",
            planned_queries=[
                "SELECT FirstName, LastName, Email FROM Customer ORDER BY LastName LIMIT 5;",
            ],
        ),
        VideoManifest(
            video_id="video_1_5",
            title="Query Etiquette Recap",
            learning_objective="Combine comment headers, aliases, ORDER BY, and LIMIT in one clean query",
            discovery_objective=(
                "Open the Execute SQL tab, type a commented query that aliases columns, orders by "
                "LastName, and limits to 5 rows, run it with F5, and show the documented preview"
            ),
            application="db_browser_sqlite",
            prerequisite_videos=["video_1_4"],
            exercise_artifact=exercise,
            format_tier="long",
            planned_queries=[
                'SELECT FirstName AS "First Name", LastName AS "Last Name", Email AS "Email Address" FROM Customer ORDER BY LastName LIMIT 5;',
            ],
        ),
    ]

    return CourseManifest(
        course_id="sql_essential_training_ch4",
        title="SQL Essential Training Chapter 4",
        description=(
            "Write and run SELECT queries with aliases, ORDER BY, and LIMIT against the WSDA Music database."
        ),
        target_audience="Beginner data analysts",
        videos=videos,
        running_example={
            "name": "Customer",
            "description": "WSDA Music customer contact list.",
            "schema": {
                "CustomerId": "INTEGER PRIMARY KEY",
                "FirstName": "TEXT",
                "LastName": "TEXT",
                "Email": "TEXT",
            },
        },
    )


# ---------------------------------------------------------------------------
# Dependency ordering
# ---------------------------------------------------------------------------


def _close_application(app_name: str) -> None:
    """Best-effort attempt to quit the target application between videos."""
    # Try a polite AppleScript quit first, then force-kill if it is still running.
    try:
        subprocess.run(
            ["osascript", "-e", f'tell application "{app_name}" to quit'],
            capture_output=True,
            timeout=5,
        )
    except Exception:
        pass
    try:
        subprocess.run(
            ["pkill", "-x", app_name],
            capture_output=True,
            timeout=5,
        )
    except Exception:
        pass
    # Give the process time to release the screen and files.
    time.sleep(2)


def _full_sql_from_video(video: VideoManifest) -> Optional[str]:
    """
    Return the full cumulative SQL text represented by a video's demo beats.

    Segmented beats split a single query across multiple actions; this helper
    concatenates those segments in order so the continuity history and prior
    query extraction see the complete query, not just the last clause.
    """
    if video.script_beats:
        parts: List[str] = []
        has_sql_action = False
        for beat_dict in video.script_beats:
            action = beat_dict.get("action") or {}
            action_type = action.get("type")
            if action_type == "type_block":
                parts.append(action.get("text") or "")
                has_sql_action = True
            elif action_type == "type_segments":
                segments = action.get("segments") or []
                parts.append(
                    "".join(
                        (seg.get("text", "") if isinstance(seg, dict) else str(seg))
                        for seg in segments
                    )
                )
                has_sql_action = True
            elif action_type == "execute_query":
                parts.append(action.get("query") or "")
                has_sql_action = True
        if has_sql_action:
            return "".join(parts).strip() or None
    if video.planned_queries:
        return video.planned_queries[0]
    return None


def _derive_opening_state_query(
    manifest: CourseManifest, video: VideoManifest
) -> Optional[str]:
    """
    Return the full cumulative query from the immediate prerequisite video, if any.

    This lets the discovery harness establish the UI state that the opening
    state-beat describes (execution tab open with the previous query).
    """
    if not video.prerequisite_videos:
        return None
    prereq_id = video.prerequisite_videos[0]
    prereq_video = next((v for v in manifest.videos if v.video_id == prereq_id), None)
    if prereq_video is None:
        return None
    return _full_sql_from_video(prereq_video)


def _strip_outer_comment_block(text: str) -> str:
    """Remove a leading /* ... */ wrapper so the text can be re-wrapped safely."""
    text = text.strip()
    if text.startswith("/*"):
        end = text.find("*/")
        if end != -1:
            text = text[end + 2 :].strip()
    return text


def _wrap_query_as_history(query_text: str) -> str:
    """Return a prior query wrapped in a block comment for continuity display."""
    stripped = _strip_outer_comment_block(query_text)
    return f"/*\n{stripped}\n*/"


def _derive_sql_history(
    manifest: CourseManifest, video: VideoManifest
) -> Tuple[Optional[str], Optional[str]]:
    """
    Return (commented_history, new_query) for continuity-by-design.

    ``commented_history`` is the concatenation of all prerequisite videos'
    final SQL queries, each wrapped in /* ... */. ``new_query`` is the current
    video's final SQL query (comment block + query) as it should appear below
    the history.
    """

    by_id = {v.video_id: v for v in manifest.videos}

    # Collect the full prerequisite chain in dependency order.
    prior: List[VideoManifest] = []
    seen: set = set()

    def _visit(vid: str) -> None:
        if vid in seen:
            return
        seen.add(vid)
        v = by_id.get(vid)
        if not v:
            return
        for prereq in v.prerequisite_videos:
            _visit(prereq)
        prior.append(v)

    for prereq_id in video.prerequisite_videos:
        _visit(prereq_id)

    history_parts: List[str] = []
    for v in prior:
        q = _full_sql_from_video(v)
        if q:
            history_parts.append(_wrap_query_as_history(q))

    history = "\n\n".join(history_parts) if history_parts else None
    new_query = _full_sql_from_video(video)
    return history, new_query


def _expected_editor_content_for_video(
    manifest: CourseManifest,
    video_id: str,
) -> Optional[str]:
    """
    Return the SQL text that should appear in the editor for ``video_id``.

    For videos with prerequisites, the expected editor content is the commented
    continuity history (one blank line before the current block) followed by the
    current video's full cumulative query. For the first video it is just the
    current block. This mirrors what the discovery harness pastes before typing.
    """
    video = next((v for v in manifest.videos if v.video_id == video_id), None)
    if video is None:
        return None
    history, new_query = _derive_sql_history(manifest, video)
    if not new_query:
        return None
    if history:
        return f"{history}\n\n{new_query}"
    return new_query


def _extract_final_editor_content(
    script_beats: List[ScriptBeat],
) -> Optional[str]:
    """Return the last verified editor content from the script beats, if any."""
    for beat in reversed(script_beats):
        observed = beat.observed_state or {}
        content = observed.get("editor_content")
        if content:
            return str(content)
        action = beat.action or {}
        if action.get("type") in ("type_block", "append_block", "type_segments"):
            text = action.get("text") or ""
            if action.get("type") == "type_segments":
                segments = action.get("segments") or []
                text = "".join(
                    s.get("text", "") if isinstance(s, dict) else str(s)
                    for s in segments
                )
            if text:
                return text
    return None


def _resolve_controlling_terminal() -> str:
    """Return the name of the process that launched this Python process."""
    import os

    override = os.environ.get("WSDA_CONTROLLING_TERMINAL", "").strip()
    if override:
        return override
    try:
        import psutil

        proc = psutil.Process()
        while proc.pid > 1:
            parent = proc.parent()
            if parent is None:
                break
            proc = parent
        return proc.name()
    except Exception:
        pass
    try:
        result = subprocess.run(
            ["osascript", "-e", 'tell application "System Events" to get name of first process whose frontmost is true'],
            capture_output=True,
            text=True,
            timeout=5,
        )
        return result.stdout.strip() or "Terminal"
    except Exception:
        return "Terminal"


def _application_to_app_name(application: str) -> str:
    """Map canonical application ids to the human-facing process/window name."""
    if application == "db_browser_sqlite":
        return "DB Browser for SQLite"
    return application


def _query_dnd_state() -> str:
    """Return the current Do Not Disturb state as a normalized string.

    macOS 12+ replaced the System Events ``do not disturb`` property with Focus
    modes. We first try the legacy AppleScript API for older macOS, then fall
    back to the ``com.apple.notificationcenterui`` defaults key that is still
    honored on modern systems.
    """
    # Legacy AppleScript property (macOS <= 11).
    try:
        result = subprocess.run(
            ["osascript", "-e", 'tell application "System Events" to get do not disturb'],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip().lower()
    except Exception:
        pass

    # Modern fallback: notification center defaults key.
    try:
        prefs_path = os.path.expanduser(
            "~/Library/Preferences/ByHost/com.apple.notificationcenterui"
        )
        result = subprocess.run(
            ["defaults", "-currentHost", "read", prefs_path, "doNotDisturb"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip().lower()
    except Exception:
        pass

    return "unknown"


def _visible_window_owners() -> set:
    """Return the set of process names that own at least one visible window."""
    script = '''
    tell application "System Events"
        set owners to {}
        repeat with proc in (every process whose background only is false and visible is true)
            set procName to name of proc
            try
                set winList to every window of proc whose value of attribute "AXMinimized" is false
                if (count of winList) > 0 then
                    set end of owners to procName
                end if
            on error
                -- Some processes do not expose window attributes; ignore.
            end try
        end repeat
        return owners
    end tell
    '''
    try:
        result = subprocess.run(
            ["osascript", "-e", script],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0:
            text = result.stdout.strip()
            if text:
                return {p.strip() for p in text.split(",") if p.strip()}
            return set()
    except Exception:
        pass
    return set()


def _preflight_system_state(target_app_name: str, controlling_terminal: str) -> None:
    """
    Hard preflight via AppleScript System Events.

    Asserts Do Not Disturb is on and only the target app + controlling terminal
    have visible windows. Refuses to record otherwise.
    """
    # 1. Do Not Disturb / Focus must be on.
    dnd = _query_dnd_state()
    if dnd not in {"true", "1", "yes", "on"}:
        raise RuntimeError(
            f"[PREFLIGHT] Do Not Disturb is off ({dnd!r}). Enable it before recording."
        )

    # 2. Enumerate owners of visible windows.
    visible = _visible_window_owners()
    allowed = {
        target_app_name,
        controlling_terminal,
        "Finder",  # desktop windows
    }
    offenders = visible - allowed
    if offenders:
        raise RuntimeError(
            f"[PREFLIGHT] Disallowed visible windows from: {sorted(offenders)}. "
            "Close/hide them or set WSDA_CONTROLLING_TERMINAL if needed."
        )

    # 3. Optional dedicated render-account check.
    import os

    render_user = os.environ.get("WSDA_RENDER_USER")
    if render_user:
        current_user = os.environ.get("USER", "")
        if current_user != render_user:
            raise RuntimeError(
                f"[PREFLIGHT] WSDA_RENDER_USER is set to {render_user!r} but the "
                f"current console user is {current_user!r}. Switch to the render account."
            )
    else:
        print(
            "[PREFLIGHT] WSDA_RENDER_USER is not set; running base preflight. "
            "Set WSDA_RENDER_USER to enforce a dedicated render account.",
            file=sys.stderr,
        )


def _assert_recording_hygiene(profile: "EnvironmentProfile") -> None:
    """
    Pre-flight assertion that the recording environment is clean.

    Raises RuntimeError if an overlay window or notification is visible, because
    the off-app frame gate can only cut frames that are already recorded; a
    notification at run start must be cleared before recording begins.
    """
    _preflight_system_state(
        target_app_name=profile.app_name,
        controlling_terminal=_resolve_controlling_terminal(),
    )

    # Do Not Disturb / Focus state (best-effort on macOS).
    dnd_state = _query_dnd_state()
    print(f"[RECORDING HYGIENE] Do Not Disturb/Focus: {dnd_state}", file=sys.stderr)

    # Overlay window check via VLM.
    overlay_detected = False
    try:
        agent = VisionAgent(profile=profile)
        b64 = agent.screenshot()
        app_name = profile.app_name
        prompt = (
            f"The target application is {app_name}, which is allowed to be on "
            "screen. Look at this screenshot and answer: is there any notification banner, "
            "Messages conversation window, FaceTime overlay, Character Viewer, or window "
            f"from any OTHER application covering {app_name}? "
            "Reply exactly YES or NO, nothing else."
        )
        result = agent._call_vlm(prompt, expect_json=False, max_tokens=32)
        overlay_detected = result.text.strip().upper().startswith("YES")
    except Exception as exc:
        # If we cannot verify, treat as a blocker to avoid shipping private UI.
        raise RuntimeError(f"[RECORDING HYGIENE] Could not verify overlay state: {exc}")

    print(
        f"[RECORDING HYGIENE] Overlay windows detected: {overlay_detected}",
        file=sys.stderr,
    )
    if overlay_detected:
        raise RuntimeError(
            "[RECORDING HYGIENE] Overlay window or notification detected. "
            "Clear all notifications and non-target windows, then retry."
        )


def _verify_video_frames_show_app(
    video_path: str,
    profile: EnvironmentProfile,
    interval: float = 5.0,
    bad_frame_dir: Optional[Path] = None,
) -> List[float]:
    """
    Sample the rendered video every ``interval`` seconds and ask the VLM whether
    each frame shows the target application. Return a list of offending timestamps.

    Raises RuntimeError if any sampled frame does not show the target application.
    """
    if not Path(video_path).exists():
        return []

    if bad_frame_dir is None:
        bad_frame_dir = Path(__file__).resolve().parent / "discovery_output" / "guard_bad_frames"
    bad_frame_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="wsda_frames_") as tmpdir:
        # Extract frames at the requested interval.
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-i",
                video_path,
                "-vf",
                f"fps=1/{interval}",
                "-pix_fmt",
                "rgb24",
                f"{tmpdir}/frame_%04d.png",
            ],
            check=True,
            capture_output=True,
            timeout=120,
        )

        frame_paths = sorted(Path(tmpdir).glob("frame_*.png"))
        if not frame_paths:
            return []

        agent = VisionAgent(profile=profile)
        visible = agent.verify_app_visible_in_frames([str(p) for p in frame_paths])

        bad_timestamps: List[float] = []
        for idx, is_visible in enumerate(visible):
            if not is_visible:
                ts = round(idx * interval, 2)
                bad_timestamps.append(ts)
                src = frame_paths[idx]
                dst = bad_frame_dir / f"{Path(video_path).stem}_frame_{idx:04d}_{ts:.2f}s.png"
                shutil.copy(str(src), str(dst))

    if bad_timestamps:
        print(
            f"[POST-RENDER GUARD] bad frames at timestamps (s): {bad_timestamps}",
            file=sys.stderr,
        )
        raise RuntimeError(
            f"Video {video_path} contains off-application frames at {bad_timestamps}"
        )

    print(
        f"[POST-RENDER GUARD] all {len(visible)} sampled frames show DB Browser",
        file=sys.stderr,
    )
    return bad_timestamps


def _cleanup_dir_contents(directory: Path) -> None:
    """
    Remove screenshots, videos, and temp files inside ``directory`` while
    preserving SQLite seed databases (``.db`` files).
    """
    if not directory.exists():
        return
    for item in directory.iterdir():
        if item.is_file() and item.suffix.lower() == ".db":
            continue
        try:
            if item.is_dir():
                shutil.rmtree(item)
            else:
                item.unlink()
        except Exception as exc:
            print(f"Warning: could not clean up {item}: {exc}", file=sys.stderr)


def _video_order(manifest: CourseManifest) -> List[VideoManifest]:
    """Return videos in dependency order (topological sort)."""
    by_id = {v.video_id: v for v in manifest.videos}
    completed: set = set()
    ordered: List[VideoManifest] = []

    def visit(video: VideoManifest) -> None:
        if video.video_id in completed:
            return
        for prereq in video.prerequisite_videos:
            if prereq not in by_id:
                raise ValueError(f"Unknown prerequisite video {prereq!r}")
            visit(by_id[prereq])
        ordered.append(video)
        completed.add(video.video_id)

    for video in manifest.videos:
        visit(video)

    return ordered


# ---------------------------------------------------------------------------
# Script beat serialization helpers
# ---------------------------------------------------------------------------


_SCRIPT_BEAT_FIELDS = {
    "beat_id", "kind", "text", "action", "visual_check",
    "attaches_to", "target_id", "video_clip_path", "observed_state",
    "choreography", "planned_duration",
}


def _script_beat_to_dict(beat: ScriptBeat) -> dict:
    """Serialize a ScriptBeat to a plain dict for Pydantic storage."""
    return {
        k: getattr(beat, k)
        for k in _SCRIPT_BEAT_FIELDS
        if getattr(beat, k) is not None
    }


def _dict_to_script_beat(data: dict) -> ScriptBeat:
    """Deserialize a ScriptBeat from a plain dict."""
    filtered = {k: v for k, v in data.items() if k in _SCRIPT_BEAT_FIELDS}
    return ScriptBeat(**filtered)


# ---------------------------------------------------------------------------
# Render quality gates
# ---------------------------------------------------------------------------


def _run_quality_gates(
    beats: List[ScriptBeat],
    discovery_result: DiscoveryResult,
    video: VideoManifest,
) -> List[str]:
    """
    Render-time quality warnings. All issues are logged as warnings; this
    function returns an empty list so rendering never fails for content reasons.

    Hard gates are applied after rendering to the actual output files.
    """
    demo_beats = [b for b in beats if b.kind == "demo"]

    # Missing clips are still worth noting, but the renderer will hold a still frame.
    for beat in demo_beats:
        if not beat.video_clip_path or not Path(beat.video_clip_path).exists():
            print(
                f"Warning: demo beat {beat.beat_id} has no recorded video clip; "
                "renderer will hold a still frame.",
                file=sys.stderr,
            )

    # Duration mismatch is handled by the renderer (pad/trim per beat).
    script_duration = sum(len(b.text.split()) / 2.5 for b in beats)
    clip_duration = 0.0
    for beat in demo_beats:
        if beat.video_clip_path:
            dur = _media_duration(beat.video_clip_path)
            if dur is not None:
                clip_duration += dur
    if abs(script_duration - clip_duration) > 2.0:
        print(
            f"Warning: script duration ({script_duration:.1f}s) differs from clip duration "
            f"({clip_duration:.1f}s) — renderer will pad/trim clips to match beat timings.",
            file=sys.stderr,
        )

    return []


def _media_duration(path: str) -> Optional[float]:
    """Return duration in seconds using ffprobe, or None if unavailable."""
    import shutil

    if shutil.which("ffprobe") is None:
        return None
    result = subprocess.run(
        [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            path,
        ],
        capture_output=True, text=True, timeout=30,
    )
    try:
        return float(result.stdout.strip())
    except Exception:
        return None


def _verify_final_frame_matches_locked_state(
    video_path: str,
    discovery_result: Any,
    similarity_threshold: float = 0.90,
) -> None:
    """
    Extract the last frame of the rendered silent MP4 and compare it to the
    locked end-state screenshot using SSIM. Print a loud warning if they differ.

    SSIM is used instead of MSE because screen-capture and screen-recording
    frames of the same UI state can differ in color rendering, timing text, and
    anti-aliasing while still being semantically identical.

    Diagnostics:
      - Both images are normalized to 640 px wide grayscale before comparison.
      - Their original dimensions are logged.
      - On mismatch, both compared images are saved to discovery_output/.
    """
    locked_state = getattr(discovery_result, "locked_state", None)
    if not locked_state:
        return
    screenshot_path = getattr(locked_state, "screenshot_path", None)
    if not screenshot_path or not Path(screenshot_path).exists():
        return
    if not video_path or not Path(video_path).exists():
        return

    try:
        from PIL import Image
        import numpy as np

        discovery_output_dir = Path(__file__).resolve().parent / "discovery_output"
        discovery_output_dir.mkdir(parents=True, exist_ok=True)

        video_frame = discovery_output_dir / "guard_video_last.png"
        subprocess.run(
            [
                "ffmpeg", "-y", "-sseof", "-0.5", "-i", video_path,
                "-vframes", "1", "-pix_fmt", "rgb24", str(video_frame),
            ],
            check=True, capture_output=True, timeout=30,
        )

        img_video_raw = Image.open(video_frame).convert("L")
        img_state_raw = Image.open(screenshot_path).convert("L")

        print(
            f"[POST-RENDER GUARD] comparing video frame {img_video_raw.size} "
            f"vs locked screenshot {img_state_raw.size}",
            file=sys.stderr,
        )

        # Normalize both to the same fixed width before comparing.
        target_width = 640
        def _normalize(img: Image.Image) -> Image.Image:
            w, h = img.size
            if w == target_width:
                return img
            ratio = target_width / w
            return img.resize((target_width, int(h * ratio)), Image.Resampling.LANCZOS)

        img_video = _normalize(img_video_raw)
        img_state = _normalize(img_state_raw)
        arr_video = np.array(img_video).astype(np.float32)
        arr_state = np.array(img_state).astype(np.float32)

        try:
            from skimage.metrics import structural_similarity as ssim
            score = float(ssim(arr_video, arr_state, data_range=255.0))
            metric_name = "SSIM"
        except Exception:
            # Fallback to MSE if scikit-image is unavailable.
            score = float(np.mean((arr_video - arr_state) ** 2))
            metric_name = "MSE"
            similarity_threshold = 10.0

        if metric_name == "SSIM":
            verdict = "MATCH" if score >= similarity_threshold else "MISMATCH"
        else:
            verdict = "MATCH" if score < similarity_threshold else "MISMATCH"
        print(
            f"[POST-RENDER GUARD] final-frame vs locked-state: {verdict} ({metric_name}={score:.4f})",
            file=sys.stderr,
        )
        if verdict == "MISMATCH":
            final_save = discovery_output_dir / "guard_final_frame.png"
            locked_save = discovery_output_dir / "guard_locked_reference.png"
            img_video_raw.save(final_save)
            img_state_raw.save(locked_save)
            print(
                "[POST-RENDER GUARD] WARNING: rendered video does not end on the discovered objective state! "
                f"video={video_path} screenshot={screenshot_path} "
                f"saved={final_save}, {locked_save}",
                file=sys.stderr,
            )
    except Exception as exc:
        print(
            f"[POST-RENDER GUARD] WARNING: could not run final-frame check: {exc}",
            file=sys.stderr,
        )


# ---------------------------------------------------------------------------
# Build pipeline
# ---------------------------------------------------------------------------


class _Tee:
    """Write to a file and the original stream at the same time."""

    def __init__(self, stream, file_path: Path):
        self.stream = stream
        self.file = open(str(file_path), "a", encoding="utf-8")

    def write(self, data: str) -> int:
        self.stream.write(data)
        self.stream.flush()
        self.file.write(data)
        self.file.flush()
        return len(data)

    def flush(self) -> None:
        self.stream.flush()
        self.file.flush()

    def close(self) -> None:
        self.file.close()


def _setup_run_log(course_output_dir: Path):
    """
    Persist console output to a timestamped run log inside the course output directory.

    The filename always includes a timestamp so successive runs never clobber
    previous logs. The most recent log is also symlinked as run_latest.log.
    """
    course_output_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    log_path = course_output_dir / f"run_{ts}.log"
    # If a log already exists for this second, append a microsecond suffix.
    if log_path.exists():
        ts = time.strftime("%Y%m%d_%H%M%S") + f"_{time.time_ns() // 1_000_000 % 1000:03d}"
        log_path = course_output_dir / f"run_{ts}.log"
    log_path.write_text(
        f"# Run log started {time.strftime('%Y-%m-%d %H:%M:%S')}\n",
        encoding="utf-8",
    )

    # Tee stdout/stderr so print() and logging both end up in the run log.
    sys.stdout = _Tee(sys.stdout, log_path)  # type: ignore[assignment]
    sys.stderr = _Tee(sys.stderr, log_path)  # type: ignore[assignment]

    handler = logging.FileHandler(str(log_path), mode="a", encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s: %(message)s"))
    handler.setLevel(logging.INFO)
    root = logging.getLogger()
    root.setLevel(min(root.level or logging.INFO, logging.INFO))
    root.addHandler(handler)

    # Stable symlink to the current run log for live tailing.
    latest_link = course_output_dir / "run_latest.log"
    try:
        if latest_link.exists() or latest_link.is_symlink():
            latest_link.unlink()
        latest_link.symlink_to(log_path.name)
    except Exception:
        pass

    print(f"[RUN LOG] {log_path}", file=sys.stderr)
    return handler


def run_course(
    manifest: CourseManifest,
    output_dir: str = "output/courses",
    min_reliability: float = 0.8,
    output_mode: Literal["auto", "hybrid", "raw"] = "auto",
    only_video: Optional[str] = None,
    dry_run: bool = False,
) -> dict:
    """
    Run the full compiler pipeline for every video in the manifest.

    ``min_reliability`` sets the discovery quality gate. If a video's discovery
    reliability score is below this threshold, the pipeline stops and reports
    the failure.

    ``output_mode`` controls what is produced per video:
      - "auto": one final MP4 with burned-in highlights + TTS audio (default).
      - "hybrid": raw MP4, highlights JSON, TTS audio, and reference script.
      - "raw": raw MP4 + reference script only.

    ``only_video`` restricts the run to a single video_id for iteration discipline.

    ``dry_run`` skips VLM/screen recording and exercises the attempt-report path.

    Returns a summary dict with per-video outputs and aggregate duration.
    """
    reset_tracker()

    if dry_run:
        return _dry_run_video(manifest, output_dir=output_dir, only_video=only_video)

    course_output_dir = Path(output_dir) / manifest.course_id
    course_output_dir.mkdir(parents=True, exist_ok=True)
    discovery_output_dir = Path(__file__).resolve().parent / "discovery_output"

    # Clean up old artifacts for this course so each run starts fresh.
    # Seed databases (.db files) are preserved.
    _cleanup_dir_contents(course_output_dir)
    _cleanup_dir_contents(discovery_output_dir)

    # Set up run.log after cleanup so it survives the fresh-start wipe.
    _setup_run_log(course_output_dir)

    # Resolve video order early for hygiene assertion and seed-DB checks.
    ordered_videos = _video_order(manifest)

    # C9 iteration discipline: optionally restrict to a single video.
    if only_video:
        ordered_videos = [v for v in ordered_videos if v.video_id == only_video]
        if not ordered_videos:
            raise ValueError(f"Video {only_video!r} not found in manifest")
        print(f"[ITERATION] restricted run to {only_video}", file=sys.stderr)

    # Recording-hygiene assertion: DND/Focus state and overlay-window check.
    # We use a default profile here because the first video's profile is not
    # scouted until the loop begins; the app name is enough for the overlay gate.
    _assert_recording_hygiene(
        EnvironmentProfile(
            application=ordered_videos[0].application if ordered_videos else "unknown",
            app_name=_application_to_app_name(
                ordered_videos[0].application if ordered_videos else "unknown"
            ),
            focus_target=_application_to_app_name(
                ordered_videos[0].application if ordered_videos else "unknown"
            ),
        )
    )
    for video in ordered_videos:
        db_path_str = video.exercise_artifact.get("db_path")
        if db_path_str and not Path(db_path_str).exists():
            from .curriculum_designer import generate_seed_database

            schema = manifest.running_example.get("schema")
            if schema:
                generated = generate_seed_database(
                    manifest.course_id, schema, str(discovery_output_dir)
                )
                video.exercise_artifact["db_path"] = str(generated)
            else:
                from .discovery import _ensure_sample_db

                video.exercise_artifact["db_path"] = str(_ensure_sample_db(discovery_output_dir))

    # Determine whether TTS is available.
    tts_available = bool(
        os.environ.get("ELEVENLABS_API_KEY", "").strip()
        and os.environ.get("ELEVENLABS_VOICE_ID", "").strip()
    )
    if not tts_available:
        print(
            "Note: ElevenLabs credentials not set; producing silent videos + reference scripts.",
            file=sys.stderr,
        )
    elif output_mode in ("auto", "hybrid"):
        # Preflight the credentials once, up front, instead of discovering a
        # 401 at mux time for every video (which silently degrades to silent
        # output). A failure here is a warning — rendering continues.
        try:
            from .tts import TTSGenerator

            TTSGenerator().check_credentials()
        except Exception as exc:
            print(
                f"Warning: ElevenLabs credential check failed: {exc}\n"
                "Videos will fall back to silent output unless this is fixed.",
                file=sys.stderr,
            )

    graph_store = GraphStore()
    renderer = GraphRenderer(output_dir=str(course_output_dir))
    lesson_builder = LessonBuilder()

    logging.info("Starting course %s with %d video(s)", manifest.course_id, len(ordered_videos))

    video_outputs: List[dict] = []
    total_duration = 0.0

    for idx, video in enumerate(ordered_videos, start=1):
        try:
            print(f"Video {idx}/{len(ordered_videos)}: {video.title} ... ", end="", flush=True)

            graph_id = f"{manifest.course_id}_{video.video_id}"
            db_path = video.exercise_artifact.get("db_path")

            # Phase 1b: scout the environment so the script only asserts observed facts.
            profile: Optional[EnvironmentProfile] = None
            if db_path and video.application:
                try:
                    profile = scout_environment(
                        db_path=str(db_path),
                        application=video.application,
                        video_id=video.video_id,
                        output_dir=discovery_output_dir,
                        planned_queries=video.planned_queries,
                    )
                except Exception as exc:
                    print(f"Warning: environment scout failed: {exc}", file=sys.stderr)
            if profile is None:
                app_name = _application_to_app_name(video.application)
                profile = EnvironmentProfile(
                    application=video.application,
                    app_name=app_name,
                    focus_target=app_name,
                )

            # Phase 2: generate or load the narration script.
            if video.script_beats:
                script_beats = [_dict_to_script_beat(b) for b in video.script_beats]
                # Normalize legacy recipe/coordinate actions to the vision-agent format.
                script_beats = lesson_builder._validate_script_beats(script_beats, video)
            else:
                script_beats = lesson_builder.generate_script(
                    video, env_map=profile.model_dump()
                )

            # C4.1: enforce sentence integrity on every script, whether generated or loaded.
            lesson_builder._enforce_sentence_integrity(script_beats)
            # C6: budget narration to the planned action duration so scripts fit the demo.
            lesson_builder._enforce_word_limits(script_beats, video)
            video.script_beats = [_script_beat_to_dict(b) for b in script_beats]

            if not script_beats:
                print("FAILED (script generation)")
                raise RuntimeError(f"Script generation failed for {video.video_id}")

            # Phase 2b: quality gate — only hard failures stop the pipeline.
            ok, errors, warnings = lesson_builder.validate_script(script_beats, video)
            for warning in warnings:
                print(f"  Warning: {warning}", file=sys.stderr)
            if not ok:
                print("FAILED (script quality gate)")
                for err in errors:
                    print(f"  - {err}", file=sys.stderr)
                raise RuntimeError(f"Script quality gate failed for {video.video_id}")

            # Phase 3: execute the script beats via the vision agent and record clips.
            opening_state_query = _derive_opening_state_query(manifest, video)
            opening_state_history, new_query = _derive_sql_history(manifest, video)
            if opening_state_history:
                print(
                    f"  [CONTINUITY] {video.video_id}: pasted history length "
                    f"{len(opening_state_history)} chars, new query length {len(new_query or '')} chars",
                    file=sys.stderr,
                )
            discovery = EndStateDiscovery(
                objective=video.discovery_objective,
                application=video.application,
                db_path=db_path,
                opening_state_query=opening_state_query,
                profile=profile,
            )
            discovery_result = lesson_builder.execute_script(
                beats=script_beats,
                discovery=discovery,
                db_path=db_path,
                opening_state_query=opening_state_query,
                opening_state_history=opening_state_history,
                new_query=new_query,
            )

            if not discovery_result.success:
                print("FAILED (script execution did not reach objective)")
                raise RuntimeError(
                    f"Script execution failed for {video.video_id}: {video.discovery_objective}"
                )

            if discovery_result.reliability_score < min_reliability:
                print(
                    f"FAILED (reliability {discovery_result.reliability_score:.2f} < {min_reliability:.2f})"
                )
                raise RuntimeError(
                    f"Discovery reliability too low for {video.video_id}: "
                    f"{discovery_result.reliability_score:.2f}"
                )

            # C10: canonical editor-content gate.
            canonical_ok, canonical_reason, _ = _canonical_match_editor_content(
                manifest, video.video_id, discovery_result.final_editor_content
            )
            if not canonical_ok:
                raise RuntimeError(
                    f"Canonical editor content mismatch for {video.video_id}: {canonical_reason}"
                )

            # Phase 4b: per-video adaptation log (continuity-aware rendering).
            adapt_log_path = course_output_dir / f"{graph_id}_adaptation.jsonl"
            with open(adapt_log_path, "w", encoding="utf-8") as adapt_f:
                for beat in script_beats:
                    observed = beat.observed_state or {}
                    adapt_f.write(
                        json.dumps(
                            {
                                "video_id": video.video_id,
                                "beat_id": beat.beat_id,
                                "kind": beat.kind,
                                "text": beat.text,
                                "opening_state_strategy": observed.get("opening_state_strategy"),
                                "opening_state_log": observed.get("opening_state_log"),
                                "observed_state_summary": observed.get("summary"),
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )

            # Phase 5: build the ExecutionGraph from script beats and recorded clips.
            graph = lesson_builder.build_graph(
                video=video,
                beats=script_beats,
                discovery_result=discovery_result,
            )
            graph.graph_id = graph_id
            graph_store.save(graph)

            # Phase 5b: quality gates before rendering.
            gate_errors = _run_quality_gates(script_beats, discovery_result, video)
            if gate_errors:
                print("FAILED (render quality gates)")
                for err in gate_errors:
                    print(f"  - {err}", file=sys.stderr)
                raise RuntimeError(f"Render quality gates failed for {video.video_id}")

            # Phase 6: Render from the script beats.
            output_path = str(course_output_dir / f"{graph_id}.mp4")
            render_result = renderer.render_from_script(
                video_manifest=video,
                script_beats=script_beats,
                output_path=output_path,
                output_mode=output_mode,
                graph=graph,
            )
            if render_result is None:
                print("FAILED (render from script)")
                raise RuntimeError(f"Render from script failed for {video.video_id}")

            if render_result.get("needs_reshoot"):
                print("FAILED (recording deficit >4s/beat; re-record before TTS)")
                print(
                    f"  timing_report: {render_result.get('timing_report_path')}",
                    file=sys.stderr,
                )
                raise RuntimeError(
                    f"Recording deficit for {video.video_id}; see timing_report.json"
                )

            # Post-render hard gates: whatever files the renderer claimed to produce
            # must actually exist on disk.
            video_path = render_result.get("video_path")
            if not video_path or not Path(video_path).exists():
                print("FAILED (rendered video file missing)")
                raise RuntimeError(f"Rendered video missing for {video.video_id}")

            audio_path = render_result.get("audio_path")
            if audio_path and not Path(audio_path).exists():
                print("FAILED (TTS audio file missing)")
                raise RuntimeError(f"TTS audio missing for {video.video_id}")

            final_path = render_result.get("final_path")
            if final_path and not Path(final_path).exists():
                print("FAILED (muxed final MP4 missing)")
                raise RuntimeError(f"Muxed final MP4 missing for {video.video_id}")

            if not final_path:
                final_path = video_path

            # Post-render guard: final frame of the silent MP4 should match the
            # locked end-state screenshot. A mismatch means the renderer did not end
            # on the discovered objective state.
            _verify_final_frame_matches_locked_state(video_path, discovery_result)

            # Whole-video off-application frame gate: sample the silent/raw MP4 and
            # fail if any frame does not show the target application.
            try:
                _verify_video_frames_show_app(video_path, profile=profile, interval=5.0)
            except RuntimeError as exc:
                print(f"FAILED (whole-video off-app frame gate): {exc}", file=sys.stderr)
                raise

            # Part B acceptance gates + A4 reconciliation.
            final_path_obj = Path(final_path)
            audio_path_obj = (
                Path(audio_path) if audio_path and Path(audio_path).exists() else None
            )
            reference_path_obj = Path(render_result.get("script_path") or "")

            # C10: canonical grounding gate — the rendered query must produce the
            # canonical result set for this video.
            if new_query:
                grounding_errors = lesson_builder._assert_canonical_grounding(
                    video.video_id, new_query, db_path
                )
                if grounding_errors:
                    raise RuntimeError(
                        f"Canonical grounding failed for {video.video_id}: "
                        f"{'; '.join(grounding_errors)}"
                    )

            # C10: zero error-signature frames in the rendered output.
            error_frame_count = count_error_signature_frames(final_path_obj, profile)
            if error_frame_count > 0:
                raise RuntimeError(
                    f"Rendered video contains {error_frame_count} error-signature frame(s) "
                    f"for {video.video_id}"
                )

            computed_metrics = compute_video_metrics(
                final_path_obj, audio_path_obj, reference_path_obj, profile
            )
            renderer_summary = {
                "duration_seconds": render_result.get("duration", 0.0),
                "audio_duration_seconds": _media_duration(audio_path) if audio_path else 0.0,
                "word_count": sum(len(b.text.split()) for b in script_beats),
                "frozen_pct": frozen_share_percent(final_path_obj),
                "error_frames": count_error_signature_frames(final_path_obj, profile),
            }
            discrepancies = reconcile_summary(renderer_summary, computed_metrics)
            if discrepancies:
                print(
                    f"FAILED (summary reconciliation): {discrepancies}",
                    file=sys.stderr,
                )
                raise RuntimeError(f"Summary reconciliation failed: {discrepancies}")

            gate_result = run_acceptance_gates(
                final_path_obj, audio_path_obj, reference_path_obj, profile
            )
            print(f"[GATES] {video.video_id}", file=sys.stderr)
            print(format_gate_table(gate_result), file=sys.stderr)
            if not gate_result["passed"]:
                failed: List[str] = []
                for g in gate_result["gates"]:
                    if g["passed"]:
                        continue
                    thr = g["threshold"]
                    if thr.startswith("<"):
                        op = "≥"
                        thr = thr.lstrip("<")
                    elif thr.startswith(">="):
                        op = "<"
                        thr = thr.lstrip(">=")
                    elif "-" in thr:
                        op = "outside"
                    else:
                        op = "≠"
                    failed.append(f"{g['gate']}: {g['value']} {op} {thr}")
                raise RuntimeError(
                    f"Acceptance gates failed for {video.video_id}: " + "; ".join(failed)
                )

            duration = computed_metrics["duration_seconds"]
            logging.info(
                "Completed video %s (%s): duration=%.3fs final=%s",
                video.video_id,
                video.title,
                duration,
                final_path,
            )
            editor_content = _extract_final_editor_content(script_beats)
            video_outputs.append(
                {
                    "video_id": video.video_id,
                    "final_path": final_path,
                    "raw_path": render_result.get("video_path"),
                    "audio_path": render_result.get("audio_path"),
                    "reference_path": render_result.get("script_path"),
                    "duration": round(duration, 3),
                    "graph_id": graph_id,
                    "metrics": computed_metrics,
                    "gates": gate_result["gates"],
                    "editor_content": editor_content,
                }
            )

            video.estimated_duration_seconds = int(round(duration))
            total_duration += duration

            print(f"done ({duration:.1f}s)")

        except Exception as exc:
            expected = _expected_editor_content_for_video(manifest, video.video_id)
            actual: Optional[str] = None
            screenshot_paths: List[str] = []
            attempts = 0
            vlm_assessment = str(exc)
            local_discovery_result = locals().get("discovery_result")
            if local_discovery_result is not None:
                actual = getattr(local_discovery_result, "final_editor_content", None)
                attempts = getattr(local_discovery_result, "attempts", 0)
                locked_state = getattr(local_discovery_result, "locked_state", None)
                if locked_state:
                    path = getattr(locked_state, "screenshot_path", None)
                    if path:
                        screenshot_paths.append(str(path))
            _write_attempt_report(
                manifest=manifest,
                video_id=video.video_id,
                error=str(exc),
                expected_editor_content=expected,
                actual_editor_content=actual,
                vlm_assessment=vlm_assessment,
                screenshot_paths=screenshot_paths,
                attempts=attempts,
            )
            raise

        # Close the application so the next video starts from a fresh state.
        _close_application(profile.app_name)

    # Save the updated manifest with actual durations.
    save_manifest(manifest)

    logging.info(
        "Course %s finished: %d video(s), total_duration=%.3fs",
        manifest.course_id,
        len(video_outputs),
        total_duration,
    )

    return {
        "course_id": manifest.course_id,
        "videos_completed": len(video_outputs),
        "total_duration_seconds": round(total_duration, 3),
        "video_outputs": video_outputs,
    }


def _canonical_normalize_editor_content(text: Optional[str]) -> str:
    """
    Normalize editor text for canonical comparison.

    - rstrip each line
    - drop trailing blank lines
    - drop a single trailing newline
    """
    if text is None:
        text = ""
    lines = [line.rstrip() for line in text.split("\n")]
    while lines and lines[-1] == "":
        lines.pop()
    return "\n".join(lines)


def _canonical_match_editor_content(
    manifest: CourseManifest,
    video_id: str,
    editor_content: Optional[str],
) -> Tuple[bool, str, Optional[str]]:
    """
    Compare editor content to the manifest-derived expected text using canonical
    normalization. Returns (ok, reason, unified_diff_or_none).
    """
    expected = _expected_editor_content_for_video(manifest, video_id)
    if expected is None:
        return False, "no canonical reference for video", None
    if editor_content is None:
        return False, "no editor content captured", None
    expected_norm = _canonical_normalize_editor_content(expected)
    actual_norm = _canonical_normalize_editor_content(editor_content)
    if expected_norm == actual_norm:
        return True, "canonical match", None
    diff = "\n".join(
        difflib.unified_diff(
            expected_norm.split("\n"),
            actual_norm.split("\n"),
            fromfile="expected",
            tofile="actual",
            lineterm="",
        )
    )
    print(f"[CANONICAL] mismatch for {video_id}:\n{diff}", file=sys.stderr)
    return False, "canonical mismatch", diff


def _write_attempt_report(
    manifest: CourseManifest,
    video_id: str,
    error: str,
    expected_editor_content: Optional[str],
    actual_editor_content: Optional[str],
    vlm_assessment: Optional[str],
    screenshot_paths: List[str],
    attempts: int = 0,
    output_path: Optional[str] = None,
) -> Path:
    """Write the C10 attempt report to output/attempt_report.json."""
    report_path = Path(output_path or "output/attempt_report.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    _, _, diff = _canonical_match_editor_content(
        manifest, video_id, actual_editor_content
    )
    if diff is None:
        diff = ""
    report = {
        "success": False,
        "course_id": manifest.course_id,
        "video_id": video_id,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "error": error,
        "expected_editor_content": expected_editor_content,
        "actual_editor_content": actual_editor_content,
        "diff": diff,
        "vlm_assessment": vlm_assessment or "",
        "cost_summary": get_tracker().summary(),
        "screenshot_paths": screenshot_paths,
        "loop_ceilings": _LOOP_CEILINGS,
        "attempts": attempts,
    }
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[ATTEMPT REPORT] wrote {report_path}", file=sys.stderr)
    return report_path


def _dry_run_video(
    manifest: CourseManifest,
    output_dir: str = "output/courses",
    only_video: Optional[str] = None,
) -> dict:
    """
    Dry-run path: deterministic script, forced canonical mismatch, report, no API calls.

    Exercises the attempt-report path without VLM calls or screen recording.
    Always raises RuntimeError after writing the report so the CLI exits non-zero.
    """
    tracker = get_tracker()
    tracker.budget_usd = min(tracker.budget_usd, 0.5)
    print(
        f"[DRY RUN] budget capped at ${tracker.budget_usd:.2f}; no API calls will be made",
        file=sys.stderr,
    )

    ordered_videos = _video_order(manifest)
    if only_video:
        ordered_videos = [v for v in ordered_videos if v.video_id == only_video]
    if not ordered_videos:
        raise ValueError(f"Video {only_video!r} not found in manifest")

    video = ordered_videos[0]
    lesson_builder = LessonBuilder()
    script_beats = lesson_builder.generate_script(video, env_map={})
    if not script_beats:
        raise RuntimeError(f"Script generation failed for {video.video_id}")

    expected = _expected_editor_content_for_video(manifest, video.video_id)
    actual = _extract_final_editor_content(script_beats) or ""
    # Force a canonical mismatch so the report path is exercised.
    actual = f"{actual}\n-- dry-run forced mismatch".strip()

    error = f"[DRY RUN] forced canonical mismatch for {video.video_id}"
    print(error, file=sys.stderr)

    _write_attempt_report(
        manifest=manifest,
        video_id=video.video_id,
        error=error,
        expected_editor_content=expected,
        actual_editor_content=actual,
        vlm_assessment=(
            "[DRY RUN] no VLM assessment; deterministic script path exercised "
            "without API calls."
        ),
        screenshot_paths=[],
        attempts=1,
    )
    raise RuntimeError(error)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description="Build all videos for a course.")
    parser.add_argument(
        "--output-dir",
        default="output/courses",
        help="Directory for rendered outputs (default: output/courses)",
    )
    parser.add_argument(
        "--min-reliability",
        type=float,
        default=0.0,
        help=(
            "Minimum discovery reliability score (default: 0.0 for the demo "
            "course; set to 0.8 to enforce stricter quality gating)"
        ),
    )
    parser.add_argument(
        "--output-mode",
        choices=["auto", "hybrid", "raw"],
        default="auto",
        help="Output mode: auto (default), hybrid, or raw",
    )
    parser.add_argument(
        "--course-id",
        choices=["sql_sorting_fundamentals", "sql_essential_training_ch4"],
        default="sql_essential_training_ch4",
        help="Course manifest to build (default: sql_essential_training_ch4)",
    )
    parser.add_argument(
        "--only-video",
        default=None,
        help="Iterate on a single video_id until it passes gates once and matches the canonical reference.",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=10,
        help="Maximum single-video iteration attempts (default: 10)",
    )
    parser.add_argument(
        "--then-full-course",
        action="store_true",
        help="After single-video iteration succeeds, render the full course to output/course_ch4_v5/.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Dry run: deterministic script, forced canonical mismatch, attempt report, "
            "no API calls. Also enabled by WSDA_DRY_RUN=1."
        ),
    )
    args = parser.parse_args()

    dry_run = args.dry_run or os.environ.get("WSDA_DRY_RUN", "").strip() in (
        "1",
        "true",
        "yes",
    )

    if args.course_id == "sql_sorting_fundamentals":
        manifest = create_sql_sorting_fundamentals()
    else:
        manifest = create_sql_essential_training_ch4()

    # Preserve an existing manifest (e.g., the validation13 script with intentional
    # mid-sentence beats) instead of clobbering it. run_course will save the final
    # enriched manifest at the end of the pipeline.
    existing_manifest = load_manifest(manifest.course_id)
    if existing_manifest is None:
        save_manifest(manifest)
    else:
        manifest = existing_manifest

    # C9/C10 iteration discipline: single-video loop with raw gate output per experiment.
    if args.only_video:
        consecutive_passes = 0
        final_single_result: Optional[dict] = None
        for experiment in range(1, args.max_iterations + 1):
            print(
                f"\n[ITERATION] experiment {experiment}/{args.max_iterations} for {args.only_video}",
                file=sys.stderr,
            )
            try:
                results = run_course(
                    manifest,
                    output_dir=args.output_dir,
                    min_reliability=args.min_reliability,
                    output_mode=args.output_mode,
                    only_video=args.only_video,
                    dry_run=dry_run,
                )
            except Exception as exc:
                print(f"[ITERATION] experiment {experiment} failed: {exc}", file=sys.stderr)
                consecutive_passes = 0
                continue

            if not results.get("video_outputs"):
                print(f"[ITERATION] experiment {experiment} produced no video", file=sys.stderr)
                consecutive_passes = 0
                continue

            video_result = results["video_outputs"][0]
            gate_result = {
                "passed": all(g.get("passed") for g in video_result.get("gates", [])),
                "gates": video_result.get("gates", []),
            }
            canonical_ok, canonical_reason, _ = _canonical_match_editor_content(
                manifest, args.only_video, video_result.get("editor_content")
            )

            print(f"[ITERATION] experiment {experiment} gate result:", file=sys.stderr)
            print(format_gate_table(gate_result), file=sys.stderr)
            print(
                f"[ITERATION] experiment {experiment} canonical match: {canonical_ok} ({canonical_reason})",
                file=sys.stderr,
            )

            if gate_result["passed"] and canonical_ok:
                consecutive_passes += 1
                final_single_result = results
                print(
                    f"[ITERATION] experiment {experiment} PASS ({consecutive_passes}/1)",
                    file=sys.stderr,
                )
                if consecutive_passes >= 1:
                    print(
                        f"[ITERATION] {args.only_video} passed with canonical match.",
                        file=sys.stderr,
                    )
                    break
            else:
                consecutive_passes = 0
                print(
                    f"[ITERATION] experiment {experiment} FAIL; resetting consecutive pass counter.",
                    file=sys.stderr,
                )
        else:
            print(
                f"[ITERATION] {args.only_video} did not pass within "
                f"{args.max_iterations} experiments.",
                file=sys.stderr,
            )
            return 1

        if not args.then_full_course:
            print(json.dumps(final_single_result, indent=2))
            return 0

        # Fall through to full-course render.
        print("[ITERATION] proceeding to full course render.", file=sys.stderr)

    try:
        # C9 full course target directory.
        output_dir = args.output_dir
        if args.then_full_course:
            output_dir = "output/course_ch4_v5"
        results = run_course(
            manifest,
            output_dir=output_dir,
            min_reliability=args.min_reliability,
            output_mode=args.output_mode,
            dry_run=dry_run,
        )
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
