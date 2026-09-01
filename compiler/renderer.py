#!/usr/bin/env python3
"""
compiler/renderer.py

GraphRenderer: renders a verified ExecutionGraph into a silent MP4 with
burned-in highlights and a Markdown reference script with beat timings.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image, ImageDraw, ImageFilter

from .graph_store import GraphStore
from .narrator import ScriptBeat
from .schemas import ActionEdge, ExecutionGraph, NarrationBeat, ScreenState
from .tts import TTSGenerator


HIGHLIGHT_COLOR = "#FF2D95"
HIGHLIGHT_RGB = (255, 45, 149)
HIGHLIGHT_WIDTH = 3
HIGHLIGHT_SIZE = 40  # width/height of the highlight box centered on the click point
WORDS_PER_SECOND = 2.2
MIN_STATE_DURATION = 2.0
MIN_EDGE_DURATION = 1.0
MAX_CLONE_PAD_SECONDS = 4.0
FPS = 30
VIDEO_MAX_WIDTH = 1280


class GraphRenderer:
    """
    Renders a verified ExecutionGraph into a silent video with burned-in highlights
    and a reference script with beat timings.
    """

    def __init__(self, output_dir: str = "output"):
        self.output_dir = Path(output_dir).resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def render(
        self, graph: ExecutionGraph, recompute_timings: bool = True
    ) -> Optional[Dict[str, Any]]:
        """
        Compute beat timings, build a silent MP4 with highlighted transitions,
        write a Markdown reference script, save the updated graph, and return
        paths plus total duration.

        When ``recompute_timings`` is False, the existing ``start_time`` /
        ``end_time`` values on each beat are preserved. This is used by
        ``render_with_audio`` after timings have been set from actual TTS
        durations.
        """
        if not self._ffmpeg_available():
            print("Error: FFmpeg is not installed or not on PATH.", file=sys.stderr)
            return None

        if recompute_timings:
            self._compute_timings(graph)

        frames = self._build_frames(graph)
        if not frames:
            print("Error: No frames could be built from the graph.", file=sys.stderr)
            return None

        video_path = self.output_dir / f"{graph.graph_id}_silent.mp4"
        script_path = self.output_dir / f"{graph.graph_id}_reference.md"

        self._assemble_video(frames, str(video_path))
        self._write_reference_script(graph, str(script_path))

        # Save the graph with computed timings back to the store.
        store = GraphStore()
        store.save(graph)

        total_duration = graph.narration_beats[-1].end_time if graph.narration_beats else 0.0

        return {
            "video_path": str(video_path),
            "script_path": str(script_path),
            "duration_seconds": round(total_duration, 3),
        }

    def render_with_audio(
        self, graph: ExecutionGraph, output_dir: str
    ) -> Optional[Dict[str, Any]]:
        """
        Two-pass render that uses actual TTS durations as the master clock.

        Pass 1: Generate TTS audio for every beat and measure actual durations.
        Pass 2: Re-compute beat timings from the measured durations (respecting
                minimum state/edge hold times).
        Pass 3: Re-render the silent video with the exact timings.
        Pass 4: Mux the silent video and continuous audio into the final MP4.

        Returns a dict with video_path, audio_path, final_path, and duration.
        """
        if not self._ffmpeg_available():
            print("Error: FFmpeg is not installed or not on PATH.", file=sys.stderr)
            return None

        try:
            tts = TTSGenerator()
        except ValueError as exc:
            print(f"Error: {exc}", file=sys.stderr)
            return None

        self.output_dir = Path(output_dir).resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)

        audio_path = self.output_dir / f"{graph.graph_id}_audio.mp3"
        final_path = self.output_dir / f"{graph.graph_id}_final.mp4"

        try:
            with tempfile.TemporaryDirectory(prefix="wsda_tts_") as tmpdir:
                # Pass 1: synthesize individual beat clips and measure durations.
                beat_clips = tts.generate_clips(graph, Path(tmpdir))

                # Pass 2: rebuild the timeline from actual audio durations.
                # Ceiling: bounded by the number of narration beats.
                cursor = 0.0
                for beat, _clip_path, clip_duration in beat_clips:
                    minimum = (
                        MIN_STATE_DURATION
                        if beat.attaches_to == "state"
                        else MIN_EDGE_DURATION
                    )
                    actual_duration = max(clip_duration / 1000.0, minimum)
                    beat.start_time = round(cursor, 3)
                    cursor += actual_duration
                    beat.end_time = round(cursor, 3)

                # Save the re-timed graph before rendering.
                store = GraphStore()
                store.save(graph)

                # Pass 3: render the silent video using the exact timings.
                render_result = self.render(graph, recompute_timings=False)
                if render_result is None:
                    return None

                # Assemble the continuous audio using the same exact timings.
                tts.assemble_clips(beat_clips, audio_path)

            # Pass 4: mux video and audio.
            self.mux(render_result["video_path"], str(audio_path), str(final_path))

        except Exception as exc:
            print(f"Error during audio/video render: {exc}", file=sys.stderr)
            return None

        total_duration = graph.narration_beats[-1].end_time if graph.narration_beats else 0.0

        return {
            "video_path": render_result["video_path"],
            "audio_path": str(audio_path),
            "final_path": str(final_path),
            "duration": round(total_duration, 3),
        }

    def render_script(
        self, graph: ExecutionGraph, output_dir: str
    ) -> Optional[Dict[str, Any]]:
        """
        Render a script-driven ExecutionGraph into a final MP4 with TTS audio.

        This method assembles the recorded action clips stored on each edge
        (``edge.video_path``) and the state screenshots into a continuous video,
        synthesizes the narration script through TTS, and muxes them with padding
        so neither stream cuts off.
        """
        if not graph.narration_beats:
            print("Error: Graph has no narration beats.", file=sys.stderr)
            return None
        if not any(edge.video_path for edge in graph.edges):
            print(
                "Warning: No recorded action clips found on edges; rendering from screenshots only.",
                file=sys.stderr,
            )
        return self.render_with_audio(graph, output_dir)

    def render_from_script(
        self,
        video_manifest: Any,
        script_beats: List[ScriptBeat],
        output_path: str,
        output_mode: Literal["auto", "hybrid", "raw"] = "auto",
        graph: Optional[ExecutionGraph] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Render a video directly from script beats and their recorded clips.

        Builds an ExecutionGraph (or uses the supplied one), computes timings,
        concatenates recorded demo-beat clips, holds still frames for non-demo
        beats, optionally synthesizes TTS, and muxes the final output.

        Returns {video_path, audio_path, final_path, duration, script_path}.
        """
        if not self._ffmpeg_available():
            print("Error: FFmpeg is not installed or not on PATH.", file=sys.stderr)
            return None

        if not script_beats:
            print("Error: No script beats provided.", file=sys.stderr)
            return None

        out_path = Path(output_path).resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        output_dir = out_path.parent
        graph_id = out_path.stem

        # Use the supplied graph or build a minimal one from beats.
        if graph is None:
            graph = self._build_graph_from_beats(
                graph_id, video_manifest, script_beats, output_dir
            )

        # Ensure graph narration beats match the script beats.
        if not graph.narration_beats:
            graph.narration_beats = self._narration_beats_from_script(script_beats)

        # Compute timings from word count.
        self._compute_timings(graph)

        # TTS pass for hybrid/auto modes.
        audio_path: Optional[str] = None
        beat_clips: List[Any] = []
        if output_mode in ("auto", "hybrid") and self._tts_available():
            try:
                tts = TTSGenerator()
                with tempfile.TemporaryDirectory(prefix="wsda_tts_") as tmpdir:
                    beat_clips = tts.generate_clips(graph, Path(tmpdir))

                    # Collect actual narration durations per beat.
                    audio_durations: Dict[str, float] = {}
                    for beat, _clip_path, clip_duration_ms in beat_clips:
                        minimum = (
                            MIN_STATE_DURATION
                            if beat.attaches_to == "state"
                            else MIN_EDGE_DURATION
                        )
                        audio_durations[beat.beat_id] = max(
                            clip_duration_ms / 1000.0, minimum
                        )

                    # Build ONE master timeline: each beat lasts at least as
                    # long as its narration AND its recorded action clip, so
                    # no recorded content is ever trimmed and audio/picture
                    # share the same clock.
                    frame_durations, debt_by_beat, timing_details = self._synced_beat_durations(
                        graph, script_beats, audio_durations
                    )
                    self._print_timing_table(graph_id, timing_details)

                    # Sync the graph timeline to the actual frame durations so the
                    # reported duration matches the rendered output.
                    cursor = 0.0
                    for beat in graph.narration_beats:
                        duration = frame_durations.get(beat.beat_id, MIN_STATE_DURATION)
                        beat.start_time = round(cursor, 3)
                        cursor += duration
                        beat.end_time = round(cursor, 3)

                    # Write timing report and flag videos that exceed the pad cap.
                    timing_report_path = output_dir / f"{graph_id}_timing_report.json"
                    self._write_timing_report(
                        graph_id, audio_durations, script_beats, debt_by_beat, timing_report_path
                    )
                    if debt_by_beat:
                        graph.render_status = "NEEDS_RESHOOT"
                        raise RuntimeError(
                            f"NEEDS_RESHOOT: {len(debt_by_beat)} beat(s) exceed the "
                            f"{MAX_CLONE_PAD_SECONDS}s padding cap. See "
                            f"{timing_report_path.resolve()}"
                        )

                    # Persist the re-timed graph before rendering.
                    store = GraphStore()
                    store.save(graph)

                    # Build frames on the SAME timeline (was word-count based,
                    # which caused audio/video drift and cut-offs).
                    frames = self._build_frames_from_beats(
                        script_beats, output_dir, durations=frame_durations
                    )
                    video_path = str(output_dir / f"{graph_id}_silent.mp4")
                    self._assemble_video(frames, video_path)

                    audio_path = str(output_dir / f"{graph_id}_audio.mp3")
                    tts.assemble_clips(beat_clips, audio_path)

                    if output_mode == "auto":
                        final_path = str(output_dir / f"{graph_id}_final.mp4")
                        self.mux(video_path, audio_path, final_path)
                    else:
                        final_path = None

                    script_path = self._write_script_reference(
                        graph_id, video_manifest, script_beats, output_dir
                    )
                    total_duration = graph.narration_beats[-1].end_time if graph.narration_beats else 0.0
                    result = {
                        "video_path": video_path,
                        "audio_path": audio_path,
                        "final_path": final_path,
                        "duration": round(total_duration, 3),
                        "script_path": script_path,
                        "timing_report_path": str(timing_report_path.resolve()),
                    }
                    if debt_by_beat:
                        result["status"] = "NEEDS_RESHOOT"
                        result["needs_reshoot"] = True
                        result["timing_debt_seconds"] = round(sum(debt_by_beat.values()), 3)
                    return result
            except RuntimeError as exc:
                if "NEEDS_RESHOOT" in str(exc):
                    raise
                print(f"Warning: TTS pass failed ({exc}); falling back to silent.", file=sys.stderr)
            except Exception as exc:
                print(f"Warning: TTS pass failed ({exc}); falling back to silent.", file=sys.stderr)

        # Silent (raw/hybrid without TTS or TTS failure fallback).
        # Use the recorded clip durations as the master timeline so the video
        # shows the real action pace and avoids long frozen padding caused by
        # word-count-based timings in silent mode.
        frame_durations, _debt_by_beat, _timing_details = self._synced_beat_durations(
            graph, script_beats, audio_durations=None
        )
        cursor = 0.0
        for beat in graph.narration_beats:
            duration = frame_durations.get(beat.beat_id, MIN_STATE_DURATION)
            beat.start_time = round(cursor, 3)
            cursor += duration
            beat.end_time = round(cursor, 3)

        frames = self._build_frames_from_beats(
            script_beats, output_dir, durations=frame_durations
        )
        video_path = str(output_dir / f"{graph_id}_raw.mp4")
        self._assemble_video(frames, video_path)

        script_path = self._write_script_reference(
            graph_id, video_manifest, script_beats, output_dir
        )
        total_duration = graph.narration_beats[-1].end_time if graph.narration_beats else 0.0

        result: Dict[str, Any] = {
            "video_path": video_path,
            "audio_path": None,
            "final_path": None,
            "duration": round(total_duration, 3),
            "script_path": script_path,
        }

        if output_mode == "hybrid":
            highlights_path = None
            if graph.edges:
                highlights_path = self.export_highlights(graph, str(output_dir))
            result["highlights_path"] = highlights_path

        return result

    def mux(self, video_path: str, audio_path: str, output_path: str) -> str:
        """
        Mux the silent video with the continuous TTS audio using FFmpeg.

        - Video is copied without re-encoding.
        - Audio is encoded to AAC 192k.
        - Both streams are padded to exactly the same duration; nothing is cut.
        """
        if not self._ffmpeg_available():
            print("Error: FFmpeg is not installed or not on PATH.", file=sys.stderr)
            return ""

        out = Path(output_path).resolve()
        out.parent.mkdir(parents=True, exist_ok=True)

        video_path_obj = Path(video_path).resolve()
        audio_path_obj = Path(audio_path).resolve()

        video_duration = self._media_duration(str(video_path_obj))
        audio_duration = self._media_duration(str(audio_path_obj))

        if video_duration is None:
            raise RuntimeError(f"Could not determine duration of {video_path_obj}")
        if audio_duration is None:
            raise RuntimeError(f"Could not determine duration of {audio_path_obj}")

        padded_video = str(video_path_obj)
        padded_audio = str(audio_path_obj)

        with tempfile.TemporaryDirectory(prefix="wsda_mux_") as tmpdir:
            tmpdir_path = Path(tmpdir)

            if video_duration < audio_duration:
                # Extend the video by holding the final frame for the gap.
                gap = audio_duration - video_duration
                padded_video = self._extend_video_with_last_frame(
                    str(video_path_obj), tmpdir_path, gap
                )
            elif audio_duration < video_duration:
                # Pad the audio with silence for the gap.
                gap = video_duration - audio_duration
                padded_audio = str(tmpdir_path / "padded_audio.mp3")
                subprocess.run(
                    [
                        "ffmpeg",
                        "-y",
                        "-i",
                        str(audio_path_obj),
                        "-af",
                        f"apad=pad_dur={gap:.3f}",
                        "-c:a",
                        "libmp3lame",
                        "-b:a",
                        "192k",
                        padded_audio,
                    ],
                    check=True,
                    capture_output=True,
                    timeout=300,
                )

            subprocess.run(
                [
                    "ffmpeg",
                    "-y",
                    "-i",
                    padded_video,
                    "-i",
                    padded_audio,
                    "-c:v",
                    "copy",
                    "-c:a",
                    "aac",
                    "-b:a",
                    "192k",
                    str(out),
                ],
                check=True,
                capture_output=True,
                timeout=300,
            )

        return str(out)

    # ------------------------------------------------------------------
    # Timing
    # ------------------------------------------------------------------

    @staticmethod
    def _beat_duration(beat: NarrationBeat) -> float:
        """Duration in seconds derived from word count and minimum hold times."""
        base = beat.word_count / WORDS_PER_SECOND
        minimum = MIN_STATE_DURATION if beat.attaches_to == "state" else MIN_EDGE_DURATION
        return max(base, minimum)

    def _compute_timings(self, graph: ExecutionGraph) -> None:
        """Set start_time / end_time on each beat sequentially."""
        cursor = 0.0
        for beat in graph.narration_beats:
            duration = self._beat_duration(beat)
            beat.start_time = round(cursor, 3)
            cursor += duration
            beat.end_time = round(cursor, 3)

    def _synced_beat_durations(
        self,
        graph: ExecutionGraph,
        beats: List[ScriptBeat],
        audio_durations: Optional[Dict[str, float]] = None,
    ) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, Dict[str, Any]]]:
        """
        Compute the final on-screen duration for every script beat.

        Rules (Part D timing contract):
          - beat_duration = max(actual_tts_duration, clip_duration, beat_minimum).
          - Demo-beat clone padding is capped at MAX_CLONE_PAD_SECONDS per beat.
          - If narration exceeds clip + MAX_CLONE_PAD_SECONDS, the excess is a
            recording deficit; it is recorded as timing debt and the render is
            marked NEEDS_RESHOOT. The video duration for that beat stays at
            clip + MAX_CLONE_PAD_SECONDS so silent padding never exceeds 4s.
          - If clip exceeds narration + MAX_CLONE_PAD_SECONDS, the excess is a
            recording overrun; it is recorded as timing debt and the render is
            marked NEEDS_RESHOOT. Clips are otherwise kept intact.

        Returns (
            {beat_id: seconds},
            {beat_id: timing_debt_seconds},
            {beat_id: timing_detail_dict},
        ).
        """
        minimums: Dict[str, float] = {
            nb.beat_id: (
                MIN_STATE_DURATION if nb.attaches_to == "state" else MIN_EDGE_DURATION
            )
            for nb in graph.narration_beats
        }

        durations: Dict[str, float] = {}
        debt_by_beat: Dict[str, float] = {}
        details: Dict[str, Dict[str, Any]] = {}
        for beat in beats:
            min_dur = minimums.get(beat.beat_id, MIN_STATE_DURATION)
            audio_dur: Optional[float] = None
            if audio_durations and beat.beat_id in audio_durations:
                audio_dur = audio_durations[beat.beat_id]
            clip_dur: Optional[float] = None
            if beat.video_clip_path and Path(beat.video_clip_path).exists():
                clip_dur = self._media_duration(beat.video_clip_path)

            detail: Dict[str, Any] = {
                "clip_dur": round(clip_dur, 3) if clip_dur is not None else 0.0,
                "tts_dur": round(audio_dur, 3) if audio_dur is not None else 0.0,
                "pad_applied": 0.0,
                "action": "no_media",
            }

            if clip_dur is not None:
                # Keep the full clip unless it overruns the narration budget.
                base = max(min_dur, clip_dur)
                if audio_dur is not None:
                    under_allowed = clip_dur + MAX_CLONE_PAD_SECONDS
                    over_allowed = audio_dur + MAX_CLONE_PAD_SECONDS
                    if audio_dur > under_allowed:
                        debt_by_beat[beat.beat_id] = round(audio_dur - under_allowed, 3)
                        detail["pad_applied"] = round(MAX_CLONE_PAD_SECONDS, 3)
                        detail["action"] = "NEEDS_RESHOOT"
                        duration = max(base, audio_dur)
                    elif clip_dur > over_allowed:
                        debt_by_beat[beat.beat_id] = round(clip_dur - over_allowed, 3)
                        detail["pad_applied"] = round(clip_dur - audio_dur, 3)
                        detail["action"] = "NEEDS_RESHOOT"
                        duration = base
                    elif audio_dur > clip_dur:
                        detail["pad_applied"] = round(audio_dur - clip_dur, 3)
                        detail["action"] = "pad"
                        duration = max(base, audio_dur)
                    elif audio_dur == clip_dur:
                        detail["action"] = "exact"
                        duration = base
                    else:
                        detail["action"] = "keep_clip"
                        duration = base
                else:
                    detail["action"] = "clip_only"
                    duration = base
            elif audio_dur is not None:
                detail["action"] = "tts_only"
                duration = max(min_dur, audio_dur)
            else:
                detail["action"] = "hold"
                duration = min_dur

            durations[beat.beat_id] = round(duration, 3)
            details[beat.beat_id] = detail
        return durations, debt_by_beat, details

    @staticmethod
    def _print_timing_table(
        graph_id: str,
        details: Dict[str, Dict[str, Any]],
    ) -> None:
        """Print a per-beat timing table to stderr."""
        print(
            f"[TIMING] {graph_id}: beat_id | clip_dur | tts_dur | pad_applied | action",
            file=sys.stderr,
        )
        for beat_id, d in details.items():
            print(
                f"[TIMING] {beat_id} | {d['clip_dur']:.3f} | {d['tts_dur']:.3f} | "
                f"{d['pad_applied']:.3f} | {d['action']}",
                file=sys.stderr,
            )

    @staticmethod
    def _write_timing_report(
        graph_id: str,
        audio_durations: Dict[str, float],
        beats: List[ScriptBeat],
        debt_by_beat: Dict[str, float],
        out_path: Path,
    ) -> None:
        """Write a JSON report of per-beat timing debt."""
        beat_clip_durations: Dict[str, float] = {}
        for beat in beats:
            if beat.video_clip_path and Path(beat.video_clip_path).exists():
                try:
                    import subprocess

                    result = subprocess.run(
                        [
                            "ffprobe",
                            "-v",
                            "error",
                            "-show_entries",
                            "format=duration",
                            "-of",
                            "default=noprint_wrappers=1:nokey=1",
                            str(beat.video_clip_path),
                        ],
                        capture_output=True,
                        text=True,
                        timeout=30,
                    )
                    beat_clip_durations[beat.beat_id] = float(result.stdout.strip())
                except Exception:
                    beat_clip_durations[beat.beat_id] = 0.0
        report = {
            "graph_id": graph_id,
            "max_clone_pad_seconds": MAX_CLONE_PAD_SECONDS,
            "total_debt_seconds": round(sum(debt_by_beat.values()), 3),
            "beats": [
                {
                    "beat_id": beat_id,
                    "audio_seconds": round(audio_durations.get(beat_id, 0.0), 3),
                    "clip_seconds": round(beat_clip_durations.get(beat_id, 0.0), 3),
                    "allowed_video_seconds": round(
                        beat_clip_durations.get(beat_id, 0.0) + MAX_CLONE_PAD_SECONDS, 3
                    ),
                    "debt_seconds": debt,
                }
                for beat_id, debt in debt_by_beat.items()
            ],
        }
        out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    # ------------------------------------------------------------------
    # Frame building
    # ------------------------------------------------------------------

    def render_raw(
        self, graph: ExecutionGraph, output_dir: str
    ) -> Optional[Dict[str, Any]]:
        """
        Produce a clean screen recording video with NO burned-in highlights.

        Uses the same timing and cuts as ``render()`` but skips all PIL highlight
        drawing. Output: ``<output_dir>/<graph_id>_raw.mp4``.
        """
        if not self._ffmpeg_available():
            print("Error: FFmpeg is not installed or not on PATH.", file=sys.stderr)
            return None

        self.output_dir = Path(output_dir).resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self._compute_timings(graph)

        frames = self._build_frames(graph, highlight=False)
        if not frames:
            print("Error: No frames could be built from the graph.", file=sys.stderr)
            return None

        video_path = self.output_dir / f"{graph.graph_id}_raw.mp4"
        self._assemble_video(frames, str(video_path))

        # Save the graph with computed timings back to the store.
        store = GraphStore()
        store.save(graph)

        total_duration = graph.narration_beats[-1].end_time if graph.narration_beats else 0.0

        return {
            "video_path": str(video_path),
            "duration": round(total_duration, 3),
        }

    def export_highlights(
        self, graph: ExecutionGraph, output_dir: str
    ) -> Optional[str]:
        """
        Export a JSON file with timestamped highlight metadata for manual editing.

        One entry per ActionEdge, using the edge beat's start/end times and the
        edge's target bounding box. Coordinates are scaled to the output video
        width (VIDEO_MAX_WIDTH) so they match the raw rendered MP4.
        """
        if not graph.narration_beats:
            print("Error: Graph has no narration beats.", file=sys.stderr)
            return None

        self.output_dir = Path(output_dir).resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Ensure timings exist without overwriting actual TTS timings.
        if graph.narration_beats[-1].end_time == 0.0:
            self._compute_timings(graph)

        # Map edge IDs to their beat timings.
        edge_beats: Dict[str, NarrationBeat] = {
            beat.target_id: beat
            for beat in graph.narration_beats
            if beat.attaches_to == "edge"
        }

        highlights: List[Dict[str, Any]] = []
        for edge in graph.edges:
            beat = edge_beats.get(edge.edge_id)
            if beat is None:
                continue

            target = self._scaled_target(graph, edge)
            label = self._highlight_label(edge)

            highlights.append(
                {
                    "beat_id": beat.beat_id,
                    "edge_id": edge.edge_id,
                    "start_time": round(float(beat.start_time), 3),
                    "end_time": round(float(beat.end_time), 3),
                    "target": target,
                    "style": {"color": HIGHLIGHT_COLOR, "width": HIGHLIGHT_WIDTH},
                    "label": label,
                }
            )

        total_duration = round(float(graph.narration_beats[-1].end_time), 3)
        payload = {
            "graph_id": graph.graph_id,
            "video_duration_seconds": total_duration,
            "highlights": highlights,
        }

        out_path = self.output_dir / f"{graph.graph_id}_highlights.json"
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return str(out_path)

    def write_reference_script(self, graph: ExecutionGraph, script_path: str) -> None:
        """Public wrapper for the reference-script writer."""
        self._write_reference_script(graph, script_path)

    def _build_frames(
        self, graph: ExecutionGraph, highlight: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Return a list of frame descriptors in playback order. Each descriptor has:
          - state_id: the state this frame represents
          - media_path: path to a PNG screenshot or an MP4 clip
          - duration: seconds to show this media
          - media_type: "image" or "video"

        Highlighted frames are written into the output directory when
        ``highlight`` is True.
        """
        frames: List[Dict[str, Any]] = []
        beats = graph.narration_beats

        for i, beat in enumerate(beats):
            # Prefer the stored start/end timing when it has been set (e.g., from
            # actual TTS durations). Otherwise fall back to the word-count estimate.
            if beat.end_time > beat.start_time:
                duration = beat.end_time - beat.start_time
            else:
                duration = self._beat_duration(beat)

            if beat.attaches_to == "state":
                state = self._resolve_state(graph, beat.target_id)

                # C11: prefer a recorded motion clip for state beats so the renderer
                # never holds a static screenshot during long narration. Fall back to
                # the screenshot only when no clip exists.
                if beat.video_clip_path and Path(beat.video_clip_path).exists():
                    frames.append(
                        {
                            "state_id": state.state_id,
                            "media_path": beat.video_clip_path,
                            "duration": duration,
                            "media_type": "video",
                        }
                    )
                else:
                    image_path = Path(state.screenshot_path)

                    # If the previous beat is an edge with click coordinates, burn a
                    # highlight onto this state's frame (the frame shown *during* the
                    # state narration).
                    if highlight and i > 0 and beats[i - 1].attaches_to == "edge":
                        prev_edge = self._resolve_edge(graph, beats[i - 1].target_id)
                        image_path = self._maybe_highlight(
                            image_path,
                            prev_edge.target,
                            state.state_id,
                            state.platform_snapshot,
                        )

                    frames.append(
                        {
                            "state_id": state.state_id,
                            "media_path": str(image_path.resolve()),
                            "duration": duration,
                            "media_type": "image",
                        }
                    )

            elif beat.attaches_to == "edge":
                edge = self._resolve_edge(graph, beat.target_id)
                target_state = self._resolve_state(graph, edge.to_state_id)

                # Prefer a recorded screen-capture clip for the action.
                if edge.video_path and Path(edge.video_path).exists():
                    frames.append(
                        {
                            "state_id": target_state.state_id,
                            "media_path": edge.video_path,
                            "duration": duration,
                            "media_type": "video",
                        }
                    )
                else:
                    image_path = Path(target_state.screenshot_path)
                    if highlight:
                        image_path = self._maybe_highlight(
                            image_path,
                            edge.target,
                            f"{target_state.state_id}_{edge.edge_id}",
                            target_state.platform_snapshot,
                        )
                    frames.append(
                        {
                            "state_id": target_state.state_id,
                            "media_path": str(image_path.resolve()),
                            "duration": duration,
                            "media_type": "image",
                        }
                    )

        return frames

    def _maybe_highlight(
        self,
        image_path: Path,
        target: Dict[str, Any],
        suffix: str,
        platform_snapshot: Dict[str, Any],
    ) -> Path:
        """Return a highlighted version of the image if target has coordinates."""
        x = target.get("x")
        y = target.get("y")
        if x is None or y is None:
            return image_path

        out_path = self.output_dir / f"{suffix}_highlight.png"

        with Image.open(image_path) as img:
            raw_w, raw_h = img.size

        api_w = platform_snapshot.get("api_width_px")
        api_h = platform_snapshot.get("api_height_px")
        if api_w and api_h:
            scale_x = raw_w / float(api_w)
            scale_y = raw_h / float(api_h)
        else:
            # Fallback: assume coordinates are already in screenshot pixel space.
            scale_x = scale_y = 1.0

        draw_x = int(round(float(x) * scale_x))
        draw_y = int(round(float(y) * scale_y))
        if "w" in target and "h" in target:
            draw_w = int(round(float(target["w"]) * scale_x))
            draw_h = int(round(float(target["h"]) * scale_y))
        else:
            draw_w = draw_h = HIGHLIGHT_SIZE

        # 5px padding around the bounding box, applied after scaling.
        box = [
            draw_x - 5,
            draw_y - 5,
            draw_x + draw_w + 5,
            draw_y + draw_h + 5,
        ]

        with Image.open(image_path) as img:
            highlighted = self._draw_highlight_box(img, box)
            highlighted.save(out_path)

        return out_path

    @staticmethod
    def _draw_highlight_box(img: Image.Image, box: List[int]) -> Image.Image:
        """Burn a magenta border + Gaussian glow around a bounding box."""
        rgba = img.convert("RGBA")
        overlay = Image.new("RGBA", rgba.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)

        # Outer glow: slightly larger magenta outline, then blur it.
        glow_box = [box[0] - 2, box[1] - 2, box[2] + 2, box[3] + 2]
        draw.rectangle(
            glow_box, outline=HIGHLIGHT_RGB + (255,), width=HIGHLIGHT_WIDTH + 4
        )
        overlay = overlay.filter(ImageFilter.GaussianBlur(2))

        # Sharp border on top of the glow.
        border_draw = ImageDraw.Draw(overlay)
        border_draw.rectangle(box, outline=HIGHLIGHT_RGB + (255,), width=HIGHLIGHT_WIDTH)

        return Image.alpha_composite(rgba, overlay).convert("RGB")

    def _scaled_target(
        self, graph: ExecutionGraph, edge: ActionEdge
    ) -> Dict[str, int]:
        """
        Return the edge target bounding box scaled to the output video width.

        The raw/rendered MP4 is scaled to VIDEO_MAX_WIDTH, so editors need
        coordinates in that space. If the edge has no target, returns zeros.
        """
        target = edge.target or {}
        x = target.get("x")
        y = target.get("y")

        if x is None or y is None:
            return {"x": 0, "y": 0, "w": 0, "h": 0}

        # Use the from_state's platform snapshot to determine API-to-video scale.
        from_state = self._resolve_state(graph, edge.from_state_id)
        snapshot = from_state.platform_snapshot if from_state else {}
        api_w = snapshot.get("api_width_px")

        if api_w:
            scale = VIDEO_MAX_WIDTH / float(api_w)
        else:
            scale = 1.0

        def _int(v: Any) -> int:
            try:
                return int(round(float(v) * scale))
            except (TypeError, ValueError):
                return 0

        return {
            "x": _int(x),
            "y": _int(y),
            "w": _int(target.get("w", HIGHLIGHT_SIZE)),
            "h": _int(target.get("h", HIGHLIGHT_SIZE)),
        }

    def _highlight_label(self, edge: ActionEdge) -> str:
        """Generate a short human-readable label for an edge highlight."""
        action = edge.action_type or "action"
        payload = edge.payload
        target = edge.target or {}

        if action == "click":
            return f"Click at ({target.get('x', 0)}, {target.get('y', 0)})"
        if action == "type":
            return f"Type '{payload}' into input field"
        if action == "hotkey":
            return f"Press {payload or 'key'}"
        if action == "wait":
            return "Wait"
        return f"{action}: {payload or target}"

    # ------------------------------------------------------------------
    # Video assembly
    # ------------------------------------------------------------------

    @staticmethod
    def _ffmpeg_available() -> bool:
        return shutil.which("ffmpeg") is not None

    @staticmethod
    def _media_duration(path: str) -> Optional[float]:
        """Return duration in seconds using ffprobe, or None if unavailable."""
        if shutil.which("ffprobe") is None:
            return None
        result = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                path,
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        try:
            return float(result.stdout.strip())
        except Exception:
            return None

    def _extend_video_with_last_frame(
        self, video_path: str, tmpdir: Path, extra_seconds: float
    ) -> str:
        """
        Append a hold on the last frame for extra_seconds.

        Single-pass re-encode with tpad. This replaces the old concat
        demuxer + ``-c copy`` approach, which could silently corrupt the
        output when the still clip's codec parameters differed from the
        main video's.
        """
        extended_video_path = tmpdir / "extended.mp4"
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-i",
                video_path,
                "-vf",
                f"tpad=stop_mode=clone:stop_duration={extra_seconds:.6f}",
                "-c:v",
                "libx264",
                "-preset",
                "fast",
                "-pix_fmt",
                "yuv420p",
                str(extended_video_path),
            ],
            check=True,
            capture_output=True,
            timeout=300,
        )
        return str(extended_video_path)

    def _assemble_video(self, frames: List[Dict[str, Any]], out_path: str) -> None:
        """
        Assemble frames via FFmpeg's filter_complex.

        Consecutive beats that reference the same state are grouped and their
        durations summed. Each group becomes one FFmpeg input: a PNG is looped
        at 1 fps, while an MP4 clip is played once and held on its last frame
        (via tpad) if the target duration exceeds the clip length. Everything is
        scaled to VIDEO_MAX_WIDTH, normalized to FPS, and concatenated.
        """
        if not frames:
            raise ValueError("Cannot assemble video with no frames.")

        # Group consecutive frames with the same state_id. If any frame in the
        # group has a recorded video clip, prefer it so the action is shown.
        grouped: List[Dict[str, Any]] = []
        # Ceiling: bounded by the number of assembled frames.
        for frame in frames:
            if grouped and grouped[-1]["state_id"] == frame["state_id"]:
                grouped[-1]["duration"] += frame["duration"]
                if frame["media_type"] == "video" and grouped[-1]["media_type"] != "video":
                    grouped[-1]["media_path"] = frame["media_path"]
                    grouped[-1]["media_type"] = "video"
            else:
                grouped.append(dict(frame))

        cmd: List[str] = ["ffmpeg", "-y"]
        for frame in grouped:
            if frame["media_type"] == "image":
                cmd.extend(["-loop", "1", "-r", "1", "-i", frame["media_path"]])
            else:
                cmd.extend(["-i", frame["media_path"]])

        filter_parts: List[str] = []
        concat_inputs: List[str] = []
        for idx, frame in enumerate(grouped):
            duration = frame["duration"]
            if frame["media_type"] == "image":
                filter_parts.append(
                    f"[{idx}:v]scale={VIDEO_MAX_WIDTH}:-2,fps={FPS},trim=duration={duration:.6f}[v{idx}]"
                )
            else:
                # Video: loop the recorded clip if the target duration exceeds
                # the clip length, capped at MAX_CLONE_PAD_SECONDS. Repeating
                # legitimate recorded motion (cursor emphasis, progressive typing)
                # avoids frozen last-frame padding while staying honest to what
                # was actually recorded.
                clip_dur = self._media_duration(frame["media_path"]) or 0.0
                pad = min(max(duration - clip_dur, 0.0), MAX_CLONE_PAD_SECONDS)
                segment_dur = clip_dur + pad
                if pad > 0:
                    filter_parts.append(
                        f"[{idx}:v]loop=loop=-1:size=0:start=0,"
                        f"scale={VIDEO_MAX_WIDTH}:-2,fps={FPS},"
                        f"setpts=PTS-STARTPTS,trim=duration={segment_dur:.6f}[v{idx}]"
                    )
                else:
                    filter_parts.append(
                        f"[{idx}:v]scale={VIDEO_MAX_WIDTH}:-2,fps={FPS},"
                        f"setpts=PTS-STARTPTS,trim=duration={segment_dur:.6f}[v{idx}]"
                    )
            concat_inputs.append(f"[v{idx}]")

        concat_n = len(grouped)
        filter_complex = (
            ";".join(filter_parts)
            + ";"
            + "".join(concat_inputs)
            + f"concat=n={concat_n}:v=1:a=0,format=yuv420p"
        )

        cmd.extend(
            [
                "-filter_complex",
                filter_complex,
                "-c:v",
                "libx264",
                "-preset",
                "ultrafast",
                "-pix_fmt",
                "yuv420p",
                out_path,
            ]
        )

        try:
            subprocess.run(cmd, check=True, capture_output=True, timeout=300)
        except subprocess.CalledProcessError as exc:
            stderr = (
                exc.stderr.decode("utf-8", errors="replace")
                if exc.stderr is not None
                else ""
            )
            raise RuntimeError(f"FFmpeg failed: {stderr[:500]}") from exc

    # ------------------------------------------------------------------
    # Reference script
    # ------------------------------------------------------------------

    def _write_reference_script(self, graph: ExecutionGraph, script_path: str) -> None:
        lines = [
            f"# Reference Script: {graph.graph_id}",
            "",
            f"**Learning objective:** {graph.learning_objective}",
            f"**Application:** {graph.application}",
            "",
            "| Beat | Time | Type | Target | Words | Text |",
            "|------|------|------|--------|-------|------|",
        ]

        for beat in graph.narration_beats:
            time_range = f"{beat.start_time:.1f}s – {beat.end_time:.1f}s"
            target = self._format_target(graph, beat)
            text = beat.text.replace("|", "\\|")
            lines.append(
                f"| {beat.beat_id} | {time_range} | {beat.attaches_to} | {target} | {beat.word_count} | {text} |"
            )

        lines.append("")
        lines.append("## State Screenshots")
        lines.append("")
        for beat in graph.narration_beats:
            if beat.attaches_to == "state":
                state = self._resolve_state(graph, beat.target_id)
                lines.append(f"- **{beat.beat_id}** ({beat.target_id}): `{state.screenshot_path}`")

        Path(script_path).write_text("\n".join(lines), encoding="utf-8")

    def _format_target(self, graph: ExecutionGraph, beat: NarrationBeat) -> str:
        if beat.attaches_to == "state":
            state = self._resolve_state(graph, beat.target_id)
            return f"{beat.target_id} ({Path(state.screenshot_path).name})"
        if beat.attaches_to == "edge":
            edge = self._resolve_edge(graph, beat.target_id)
            x = edge.target.get("x")
            y = edge.target.get("y")
            if x is not None and y is not None:
                return f"{beat.target_id} ({x}, {y})"
            return beat.target_id
        return beat.target_id

    # ------------------------------------------------------------------
    # render_from_script helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _tts_available() -> bool:
        return bool(
            os.environ.get("ELEVENLABS_API_KEY", "").strip()
            and os.environ.get("ELEVENLABS_VOICE_ID", "").strip()
        )

    def _build_graph_from_beats(
        self,
        graph_id: str,
        video_manifest: Any,
        beats: List[ScriptBeat],
        output_dir: Path,
    ) -> ExecutionGraph:
        """Build a minimal ExecutionGraph from script beats and their clips."""
        from PIL import Image

        application = getattr(video_manifest, "application", "db_browser_sqlite")
        demo_beats = [b for b in beats if b.kind == "demo"]

        # Placeholder states/screenshots.
        placeholder_path = output_dir / f"{graph_id}_blank.png"
        self._blank_frame(placeholder_path)

        start_state = ScreenState(
            state_id="state_start",
            screenshot_path=str(placeholder_path.resolve()),
            timestamp=0.0,
            application=application,  # type: ignore[arg-type]
            platform_snapshot={},
            visual_summary="Start",
        )
        end_state = ScreenState(
            state_id="state_end",
            screenshot_path=str(placeholder_path.resolve()),
            timestamp=0.0,
            application=application,  # type: ignore[arg-type]
            platform_snapshot={},
            visual_summary="End",
        )

        states: List[ScreenState] = []
        edges: List[ActionEdge] = []
        prev_state = start_state

        for i, beat in enumerate(demo_beats, start=1):
            state_id = f"state_{i:03d}"
            state = ScreenState(
                state_id=state_id,
                screenshot_path=str(placeholder_path.resolve()),
                timestamp=0.0,
                application=application,  # type: ignore[arg-type]
                platform_snapshot={},
                visual_summary=f"After {beat.beat_id}",
            )
            states.append(state)
            edges.append(
                ActionEdge(
                    edge_id=f"edge_{i:03d}",
                    from_state_id=prev_state.state_id,
                    to_state_id=state.state_id,
                    action_type="click",
                    target=beat.action.get("target", {}) if beat.action else {},
                    payload=None,
                    expected_duration=2.0,
                    video_path=beat.video_clip_path,
                )
            )
            prev_state = state

        if prev_state.state_id != end_state.state_id:
            edges.append(
                ActionEdge(
                    edge_id=f"edge_{len(edges) + 1:03d}",
                    from_state_id=prev_state.state_id,
                    to_state_id=end_state.state_id,
                    action_type="wait",
                    target={},
                    payload=None,
                    expected_duration=1.0,
                    video_path=None,
                )
            )

        narration_beats = self._narration_beats_from_script(beats)
        return ExecutionGraph(
            graph_id=graph_id,
            learning_objective=getattr(video_manifest, "learning_objective", ""),
            application=application,
            start_state=start_state,
            end_state=end_state,
            states=states,
            edges=edges,
            narration_beats=narration_beats,
        )

    @staticmethod
    def _narration_beats_from_script(beats: List[ScriptBeat]) -> List[NarrationBeat]:
        """Create NarrationBeat objects from ScriptBeat objects."""
        narration_beats: List[NarrationBeat] = []
        demo_idx = 0
        for beat in beats:
            if beat.kind == "demo":
                beat.attaches_to = "edge"
                beat.target_id = f"edge_{demo_idx + 1:03d}"
                demo_idx += 1
            else:
                beat.attaches_to = "state"
                beat.target_id = "state_start"
            narration_beats.append(
                NarrationBeat(
                    beat_id=beat.beat_id,
                    attaches_to=beat.attaches_to,  # type: ignore[arg-type]
                    target_id=beat.target_id or "",
                    text=beat.text,
                    word_count=len(beat.text.split()),
                    start_time=0.0,
                    end_time=0.0,
                )
            )
        return narration_beats

    def _build_frames_from_beats(
        self,
        beats: List[ScriptBeat],
        output_dir: Path,
        durations: Optional[Dict[str, float]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Build frame descriptors from script beats and recorded clips.

        When ``durations`` is provided (the TTS-driven master timeline from
        ``_synced_beat_durations``), those exact per-beat durations are used.
        Otherwise durations fall back to a word-count estimate that is still
        extended to cover the full recorded clip, so silent/raw renders never
        trim recorded actions either.
        """
        frames: List[Dict[str, Any]] = []
        clip_beats = [
            b for b in beats if b.video_clip_path and Path(b.video_clip_path).exists()
        ]
        current_still: Optional[Path] = None

        for i, beat in enumerate(beats):
            if durations is not None and beat.beat_id in durations:
                duration = durations[beat.beat_id]
            else:
                minimum = (
                    MIN_EDGE_DURATION if beat.kind == "demo" else MIN_STATE_DURATION
                )
                duration = max(len(beat.text.split()) / WORDS_PER_SECOND, minimum)
                if beat.video_clip_path and Path(beat.video_clip_path).exists():
                    clip_dur = self._media_duration(beat.video_clip_path)
                    if clip_dur:
                        duration = max(duration, clip_dur)

            if beat.video_clip_path and Path(beat.video_clip_path).exists():
                frames.append(
                    {
                        "state_id": beat.beat_id,
                        "media_path": beat.video_clip_path,
                        "duration": duration,
                        "media_type": "video",
                    }
                )
                current_still = self._extract_last_frame(
                    beat.video_clip_path,
                    output_dir / f"{beat.beat_id}_last.png",
                )
            else:
                # Missing clip: hold on the nearest recorded frame or blank.
                if current_still is None and clip_beats:
                    first_clip = clip_beats[0]
                    current_still = self._extract_first_frame(
                        first_clip.video_clip_path,
                        output_dir / f"{first_clip.beat_id}_first.png",
                    )
                still = current_still or self._blank_frame(
                    output_dir / "blank_frame.png"
                )
                frames.append(
                    {
                        "state_id": beat.beat_id,
                        "media_path": str(still.resolve()),
                        "duration": duration,
                        "media_type": "image",
                    }
                )

        # Warn if total clip time differs from target beat time, but never fail.
        # _assemble_video pads shorter clips and trims longer ones per beat.
        script_duration = sum(f["duration"] for f in frames)
        clip_duration = 0.0
        for f in frames:
            if f["media_type"] == "video":
                dur = self._media_duration(f["media_path"])
                if dur is not None:
                    clip_duration += dur
        if abs(script_duration - clip_duration) > 2.0:
            print(
                f"Warning: total beat duration ({script_duration:.1f}s) differs from "
                f"total clip duration ({clip_duration:.1f}s). Clips will be padded/trimmed "
                "to match beat timings.",
                file=sys.stderr,
            )

        return frames

    def _write_script_reference(
        self,
        graph_id: str,
        video_manifest: Any,
        beats: List[ScriptBeat],
        output_dir: Path,
    ) -> str:
        """Write a simple Markdown reference script for the rendered video."""
        script_path = output_dir / f"{graph_id}_reference.md"
        lines = [
            f"# Reference Script: {graph_id}",
            "",
            f"**Title:** {getattr(video_manifest, 'title', '')}",
            f"**Learning objective:** {getattr(video_manifest, 'learning_objective', '')}",
            "",
            "| Beat | Kind | Words | Text |",
            "|------|------|-------|------|",
        ]
        for beat in beats:
            text = beat.text.replace("|", "\\|")
            lines.append(
                f"| {beat.beat_id} | {beat.kind} | {len(beat.text.split())} | {text} |"
            )
        script_path.write_text("\n".join(lines), encoding="utf-8")
        return str(script_path)

    @staticmethod
    def _extract_first_frame(video_path: str, out_path: Path) -> Path:
        """Extract the first frame of a video clip."""
        if not Path(video_path).exists():
            return GraphRenderer._blank_frame(out_path)
        try:
            subprocess.run(
                [
                    "ffmpeg", "-y", "-i", video_path,
                    "-ss", "0", "-vframes", "1", "-pix_fmt", "rgb24",
                    str(out_path),
                ],
                check=True, capture_output=True, timeout=60,
            )
            return out_path
        except Exception as exc:
            print(f"Warning: could not extract first frame: {exc}", file=sys.stderr)
            return GraphRenderer._blank_frame(out_path)

    @staticmethod
    def _extract_last_frame(video_path: str, out_path: Path) -> Path:
        """Extract the last frame of a video clip."""
        if not Path(video_path).exists():
            return GraphRenderer._blank_frame(out_path)
        try:
            subprocess.run(
                [
                    "ffmpeg", "-y", "-sseof", "-0.5", "-i", video_path,
                    "-vframes", "1", "-pix_fmt", "rgb24", str(out_path),
                ],
                check=True, capture_output=True, timeout=60,
            )
            return out_path
        except Exception as exc:
            print(f"Warning: could not extract last frame: {exc}", file=sys.stderr)
            return GraphRenderer._blank_frame(out_path)

    @staticmethod
    def _blank_frame(out_path: Path) -> Path:
        """Create a small black placeholder PNG."""
        try:
            from PIL import Image
            img = Image.new("RGB", (VIDEO_MAX_WIDTH, 720), color=(0, 0, 0))
            img.save(out_path)
        except Exception:
            out_path.write_bytes(
                b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\x0cIDATx\x9cc\xf8\x0f\x00\x00\x01\x01\x00\x05\x18\xd8N\x00\x00\x00\x00IEND\xaeB`\x82"
            )
        return out_path

    # ------------------------------------------------------------------
    # Graph lookups
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_state(graph: ExecutionGraph, state_id: str) -> ScreenState:
        if graph.start_state.state_id == state_id:
            return graph.start_state
        if graph.end_state.state_id == state_id:
            return graph.end_state
        for state in graph.states:
            if state.state_id == state_id:
                return state
        raise ValueError(f"State {state_id!r} not found in graph.")

    @staticmethod
    def _resolve_edge(graph: ExecutionGraph, edge_id: str) -> ActionEdge:
        for edge in graph.edges:
            if edge.edge_id == edge_id:
                return edge
        raise ValueError(f"Edge {edge_id!r} not found in graph.")
