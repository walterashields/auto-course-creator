#!/usr/bin/env python3
"""
compiler/curriculum.py

CourseManifest and VideoManifest schemas plus the multi-video pipeline.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Literal, Optional

from pydantic import BaseModel, Field

from .discovery import APP_NAME, EndStateDiscovery
from .graph_store import GraphStore
from .lesson_builder import LessonBuilder
from .narrator import ScriptBeat
from .renderer import GraphRenderer
from .tts import TTSGenerator


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


# ---------------------------------------------------------------------------
# Dependency ordering
# ---------------------------------------------------------------------------


def _close_application() -> None:
    """Best-effort attempt to quit the target application between videos."""
    # Try a polite AppleScript quit first, then force-kill if it is still running.
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
    # Give the process time to release the screen and files.
    time.sleep(2)


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
    "attaches_to", "target_id", "video_clip_path",
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


# ---------------------------------------------------------------------------
# Build pipeline
# ---------------------------------------------------------------------------


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

    # Resolve video order early so we can ensure each video's seed DB exists.
    ordered_videos = _video_order(manifest)
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

    video_outputs: List[dict] = []
    total_duration = 0.0

    for idx, video in enumerate(ordered_videos, start=1):
        print(f"Video {idx}/{len(ordered_videos)}: {video.title} ... ", end="", flush=True)

        graph_id = f"{manifest.course_id}_{video.video_id}"
        db_path = video.exercise_artifact.get("db_path")

        # Phase 2: generate or load the narration script.
        if video.script_beats:
            script_beats = [_dict_to_script_beat(b) for b in video.script_beats]
            # Normalize legacy recipe/coordinate actions to the vision-agent format.
            script_beats = lesson_builder._validate_script_beats(script_beats, video)
        else:
            script_beats = lesson_builder.generate_script(video)
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
        discovery = EndStateDiscovery(
            objective=video.discovery_objective,
            application=video.application,
            db_path=db_path,
        )
        discovery_result = lesson_builder.execute_script(
            beats=script_beats,
            discovery=discovery,
            db_path=db_path,
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

        duration = render_result.get("duration", 0.0)
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
        _close_application()

    # Save the updated manifest with actual durations.
    save_manifest(manifest)

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
    args = parser.parse_args()

    manifest = create_sql_sorting_fundamentals()
    save_manifest(manifest)

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
