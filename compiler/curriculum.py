#!/usr/bin/env python3
"""
compiler/curriculum.py

CourseManifest and VideoManifest schemas plus the multi-video pipeline.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import List, Literal, Optional, Tuple

from pydantic import BaseModel, Field

from .discovery import EndStateDiscovery
from .graph_store import GraphStore
from .lesson_builder import LessonBuilder
from .narrator import ScriptBeat
from .renderer import GraphRenderer
from .schemas import EnvironmentProfile
from .scout import scout_environment
from .tts import TTSGenerator
from .vision_agent import VisionAgent


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


def _preflight_system_state(target_app_name: str, controlling_terminal: str) -> None:
    """
    Hard preflight via AppleScript System Events.

    Asserts Do Not Disturb is on and only the target app + controlling terminal
    (+ essential macOS UI processes) are running. Refuses to record otherwise.
    """
    import os

    if os.environ.get("WSDA_SKIP_PREFLIGHT"):
        return

    # 1. Do Not Disturb must be on.
    try:
        result = subprocess.run(
            ["osascript", "-e", 'tell application "System Events" to get do not disturb'],
            capture_output=True,
            text=True,
            timeout=5,
        )
        dnd = result.stdout.strip().lower()
    except Exception as exc:
        raise RuntimeError(f"[PREFLIGHT] Could not query Do Not Disturb: {exc}")
    if dnd != "true":
        raise RuntimeError(
            f"[PREFLIGHT] Do Not Disturb is off ({dnd!r}). Enable it before recording."
        )

    # 2. Enumerate visible (non-background) processes.
    try:
        result = subprocess.run(
            [
                "osascript",
                "-e",
                'tell application "System Events" to get name of every process whose background only is false',
            ],
            capture_output=True,
            text=True,
            timeout=5,
        )
        visible = {p.strip() for p in result.stdout.split(",") if p.strip()}
    except Exception as exc:
        raise RuntimeError(f"[PREFLIGHT] Could not enumerate running processes: {exc}")

    allowed = {
        target_app_name,
        controlling_terminal,
        "Finder",
        "SystemUIServer",
        "Dock",
        "WindowServer",
        "loginwindow",
        "ControlCenter",
        "Spotlight",
        "Python",  # test runners / python -m compiler.curriculum
        "python",
        "python3",
        "osascript",
    }
    offenders = visible - allowed
    if offenders:
        raise RuntimeError(
            f"[PREFLIGHT] Disallowed visible processes are running: {sorted(offenders)}. "
            "Close them or set WSDA_CONTROLLING_TERMINAL if needed."
        )


def _assert_recording_hygiene(profile: "EnvironmentProfile") -> None:
    """
    Pre-flight assertion that the recording environment is clean.

    Raises RuntimeError if an overlay window or notification is visible, because
    the off-app frame gate can only cut frames that are already recorded; a
    notification at run start must be cleared before recording begins.
    """
    import os

    if os.environ.get("WSDA_SKIP_PREFLIGHT"):
        print("[RECORDING HYGIENE] preflight skipped via WSDA_SKIP_PREFLIGHT", file=sys.stderr)
        return

    _preflight_system_state(
        target_app_name=profile.app_name,
        controlling_terminal=_resolve_controlling_terminal(),
    )

    # Do Not Disturb / Focus state (best-effort on macOS).
    dnd_state = "unknown"
    try:
        result = subprocess.run(
            ["defaults", "read", "com.apple.controlcenter", "NSStatusItem Visible FocusModes"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        dnd_state = result.stdout.strip() or "unknown"
    except Exception as exc:
        dnd_state = f"could not determine ({exc})"

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
    """Persist console output to run.log inside the course output directory."""
    log_path = course_output_dir / "run.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    # Ensure the file exists with a run header even if no log records are emitted.
    if not log_path.exists():
        log_path.write_text(
            f"# Run log started {time.strftime('%Y-%m-%d %H:%M:%S')}\n",
            encoding="utf-8",
        )

    # Tee stdout/stderr so print() and logging both end up in run.log.
    sys.stdout = _Tee(sys.stdout, log_path)  # type: ignore[assignment]
    sys.stderr = _Tee(sys.stderr, log_path)  # type: ignore[assignment]

    handler = logging.FileHandler(str(log_path), mode="a", encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s: %(message)s"))
    handler.setLevel(logging.INFO)
    root = logging.getLogger()
    root.setLevel(min(root.level or logging.INFO, logging.INFO))
    root.addHandler(handler)
    return handler


def run_course(
    manifest: CourseManifest,
    output_dir: str = "output/courses",
    min_reliability: float = 0.8,
    output_mode: Literal["auto", "hybrid", "raw"] = "auto",
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

    Returns a summary dict with per-video outputs and aggregate duration.
    """
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

    # Recording-hygiene assertion: DND/Focus state and overlay-window check.
    # We use a default profile here because the first video's profile is not
    # scouted until the loop begins; the app name is enough for the overlay gate.
    _assert_recording_hygiene(
        EnvironmentProfile(
            application=ordered_videos[0].application if ordered_videos else "unknown",
            app_name=ordered_videos[0].application if ordered_videos else "unknown",
            focus_target=ordered_videos[0].application if ordered_videos else "unknown",
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
            profile = EnvironmentProfile(
                application=video.application,
                app_name=video.application,
                focus_target=video.application,
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

        duration = render_result.get("duration", 0.0)
        logging.info(
            "Completed video %s (%s): duration=%.3fs final=%s",
            video.video_id,
            video.title,
            duration,
            final_path,
        )
        video_outputs.append(
            {
                "video_id": video.video_id,
                "final_path": final_path,
                "raw_path": render_result.get("video_path"),
                "audio_path": render_result.get("audio_path"),
                "reference_path": render_result.get("script_path"),
                "duration": round(duration, 3),
                "graph_id": graph_id,
            }
        )

        video.estimated_duration_seconds = int(round(duration))
        total_duration += duration

        print(f"done ({duration:.1f}s)")

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
    args = parser.parse_args()

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

    try:
        results = run_course(
            manifest,
            output_dir=args.output_dir,
            min_reliability=args.min_reliability,
            output_mode=args.output_mode,
        )
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
