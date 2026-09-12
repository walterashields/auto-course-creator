#!/usr/bin/env python3
"""
compiler/test_harness.py

Fast local verification harness for pipeline changes. Runs without vision-agent
or ElevenLabs calls by using ffmpeg-generated synthetic clips and monkeypatched
TTS audio.
"""

from __future__ import annotations

import contextlib
import io
import json
import math
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import traceback
import unittest
import unittest.mock as mock
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

from compiler import discovery as discovery_module
from compiler import vision_agent as vision_agent_module
from compiler.curriculum import _dict_to_script_beat, _verify_video_frames_show_app, _write_attempt_report, load_manifest
from compiler.discovery import (
    BeatRecordingStop,
    DELIVERY_FLOOR_FPS,
    EndStateDiscovery,
    RECORDER_TAIL_SECONDS,
    _clip_has_off_app_interval,
    _composite_window_frame,
    _final_editor_read,
    _open_video_writer,
    _ScreenCaptureKitRecorder,
    _window_bounds,
    delivery_floor_breach,
)
from compiler import curriculum as curriculum_module
from compiler.frame_analysis import detect_error_signature, frozen_share_percent, run_acceptance_gates
from compiler.lesson_builder import LessonBuilder
from compiler.narrator import ScriptBeat
from compiler.renderer import GraphRenderer
from compiler.schemas import EnvironmentProfile
from compiler.tts import TTSGenerator
from compiler.vision_agent import VisionAgent, VisionAgentResult, wait_for_app_readiness
from compiler import ax_pyobjc


# ---------------------------------------------------------------------------
# Synthetic clip generation
# ---------------------------------------------------------------------------


def _make_video(
    path: Path,
    duration: float,
    fps: int = 10,
    width: int = 640,
    height: int = 360,
    motion: bool = False,
    motion_region: Optional[Dict[str, int]] = None,
) -> Path:
    """
    Generate an MP4 with a grey background. If motion is True, a white square
    moves inside motion_region for the full duration; otherwise the frame is
    static.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    total_frames = int(round(duration * fps))

    # Build raw BGR frames in memory.
    frames: List[np.ndarray] = []
    for i in range(total_frames):
        frame = np.full((height, width, 3), fill_value=128, dtype=np.uint8)
        if motion:
            region = motion_region or {"x": 0, "y": 0, "w": width, "h": height}
            rw = max(8, min(region["w"], 64))
            rh = max(8, min(region["h"], 64))
            # Move the square horizontally across the region.
            progress = i / max(1, total_frames - 1)
            x = region["x"] + int(progress * max(0, region["w"] - rw))
            y = region["y"] + max(0, region["h"] - rh) // 2
            frame[y : y + rh, x : x + rw] = 255
        frames.append(frame)

    # Write via ffmpeg rawvideo pipe.
    cmd = [
        "ffmpeg",
        "-y",
        "-f", "rawvideo",
        "-vcodec", "rawvideo",
        "-s", f"{width}x{height}",
        "-pix_fmt", "bgr24",
        "-r", str(fps),
        "-i", "-",
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        str(path),
    ]
    data = b"".join(f.tobytes() for f in frames)
    subprocess.run(cmd, input=data, check=True, capture_output=True, timeout=60)
    return path


def _media_duration(path: Path) -> float:
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
        capture_output=True, text=True, timeout=30,
    )
    return float(result.stdout.strip())


def _extract_last_frame(video_path: Path, out_path: Path) -> Path:
    subprocess.run(
        ["ffmpeg", "-y", "-sseof", "-0.5", "-i", str(video_path),
         "-vframes", "1", "-pix_fmt", "rgb24", str(out_path)],
        check=True, capture_output=True, timeout=30,
    )
    return out_path


def make_synthetic_beats(tmpdir: Path) -> List[ScriptBeat]:
    """
    Fabricate ScriptBeats backed by ffmpeg-generated clips:
      - beat_001: static 8s head, 2s motion, 3s static tail (13s total)
      - beat_002: short all-motion clip (2s)
      - beat_003: no-motion clip (2s)
    """
    tmpdir = Path(tmpdir)
    beats: List[ScriptBeat] = []

    # Head-motion-tail clip.
    head_tail_path = tmpdir / "head_motion_tail.mp4"
    _make_video(head_tail_path, duration=8.0, fps=10, motion=False)
    motion_path = tmpdir / "motion.mp4"
    _make_video(motion_path, duration=2.0, fps=10, motion=True)
    tail_path = tmpdir / "tail.mp4"
    _make_video(tail_path, duration=3.0, fps=10, motion=False)
    combined_path = tmpdir / "beat_001.mp4"
    concat_list = tmpdir / "concat.txt"
    concat_list.write_text(
        "\n".join(f"file '{p.resolve()}'" for p in (head_tail_path, motion_path, tail_path)),
        encoding="utf-8",
    )
    subprocess.run(
        ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(concat_list),
         "-c", "copy", str(combined_path)],
        check=True, capture_output=True, timeout=60,
    )
    beats.append(
        ScriptBeat(
            beat_id="beat_001",
            kind="demo",
            text="We perform the first action.",
            action={"type": "click", "target": {"x": 0.5, "y": 0.5, "w": 40, "h": 40}},
            video_clip_path=str(combined_path.resolve()),
        )
    )

    # All-motion clip.
    all_motion_path = tmpdir / "beat_002.mp4"
    _make_video(all_motion_path, duration=2.0, fps=10, motion=True)
    beats.append(
        ScriptBeat(
            beat_id="beat_002",
            kind="demo",
            text="We perform the second action.",
            action={"type": "click", "target": {"x": 0.5, "y": 0.5, "w": 40, "h": 40}},
            video_clip_path=str(all_motion_path.resolve()),
        )
    )

    # No-motion clip.
    no_motion_path = tmpdir / "beat_003.mp4"
    _make_video(no_motion_path, duration=2.0, fps=10, motion=False)
    beats.append(
        ScriptBeat(
            beat_id="beat_003",
            kind="demo",
            text="We wait briefly.",
            action={"type": "wait", "duration": 2.0},
            video_clip_path=str(no_motion_path.resolve()),
        )
    )

    return beats


# ---------------------------------------------------------------------------
# Fake TTS
# ---------------------------------------------------------------------------


def _sine_wave_mp3(path: Path, duration_seconds: float, sample_rate: int = 22050) -> Path:
    """Write a sine-wave MP3 of exact duration using ffmpeg."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "ffmpeg", "-y", "-f", "lavfi",
            "-i", f"sine=frequency=440:duration={duration_seconds}",
            "-ar", str(sample_rate), "-ac", "1",
            str(path),
        ],
        check=True, capture_output=True, timeout=60,
    )
    return path


def fake_tts(graph: ExecutionGraph, durations: Dict[str, float]):
    """
    Monkeypatch TTSGenerator.generate_clips to return sine-wave MP3s of exact
    durations (seconds) keyed by beat_id.
    """
    original = TTSGenerator.generate_clips

    def _fake_generate_clips(self, graph, temp_dir=None):
        tmp = Path(temp_dir) if temp_dir else Path(tempfile.gettempdir()) / "wsda_fake_tts"
        tmp.mkdir(exist_ok=True)
        clips = []
        for i, beat in enumerate(graph.narration_beats):
            dur = durations.get(beat.beat_id, 1.0)
            clip_path = tmp / f"{graph.graph_id}_beat_{i:03d}.mp3"
            _sine_wave_mp3(clip_path, dur)
            from pydub import AudioSegment
            audio = AudioSegment.from_mp3(str(clip_path))
            clips.append((beat, str(clip_path.resolve()), len(audio)))
        return clips

    TTSGenerator.generate_clips = _fake_generate_clips
    return original


def restore_tts(original) -> None:
    TTSGenerator.generate_clips = original


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------


class TestTrimClipToMotion(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = Path(tempfile.mkdtemp(prefix="wsda_test_trim_"))
        self.discovery = EndStateDiscovery(
            objective="test", application="db_browser_sqlite"
        )

    def tearDown(self) -> None:
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_trim_removes_static_head_and_tail(self) -> None:
        """C9: conservative trim drops static head/tail but keeps the motion window."""
        clip = _make_video(
            self.tmpdir / "head.mp4",
            duration=3.0,
            fps=10,
            motion=False,
        )
        motion = _make_video(
            self.tmpdir / "motion.mp4", duration=2.0, fps=10, motion=True
        )
        tail = _make_video(
            self.tmpdir / "tail.mp4", duration=3.0, fps=10, motion=False
        )
        combined = self.tmpdir / "combined.mp4"
        concat_list = self.tmpdir / "concat.txt"
        concat_list.write_text(
            "\n".join(f"file '{p.resolve()}'" for p in (clip, motion, tail)),
            encoding="utf-8",
        )
        subprocess.run(
            ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(concat_list),
             "-c", "copy", str(combined)],
            check=True, capture_output=True, timeout=60,
        )

        original_dur = _media_duration(combined)
        self.discovery._trim_clip_to_motion(combined)
        kept_dur = _media_duration(combined)

        # Motion window is 2s; pad adds ~0.5s on each side, so expect ~3s.
        self.assertAlmostEqual(kept_dur, 3.0, delta=0.4)
        self.assertLess(kept_dur, original_dur - 2.0)

    def test_spinner_clip_is_not_trimmed(self) -> None:
        """C9: even small-region motion clips are kept whole."""
        clip = _make_video(
            self.tmpdir / "spinner.mp4",
            duration=3.0,
            fps=10,
            motion=True,
            motion_region={"x": 280, "y": 160, "w": 80, "h": 40},
        )
        original_dur = _media_duration(clip)
        self.discovery._trim_clip_to_motion(clip)
        kept_dur = _media_duration(clip)

        self.assertAlmostEqual(kept_dur, original_dur, delta=0.1)


class TestRenderFromScript(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = Path(tempfile.mkdtemp(prefix="wsda_test_render_"))
        self.beats = make_synthetic_beats(self.tmpdir)

        # Minimal video manifest stub.
        class Manifest:
            title = "Synthetic test"
            learning_objective = "Test rendering."
            application = "db_browser_sqlite"
            format_tier = "short"

        self.manifest = Manifest()

    def tearDown(self) -> None:
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_render_e2e_mocked_tts_matches_beat_windows(self) -> None:
        """
        With fake TTS durations equal to each beat's clip duration, the final
        MP4 duration should equal the sum of the beat windows and no recorded
        content should be trimmed.
        """
        renderer = GraphRenderer(output_dir=str(self.tmpdir))

        # TTS durations equal to each clip's actual duration.
        tts_durations = {
            b.beat_id: _media_duration(Path(b.video_clip_path))
            for b in self.beats
        }
        original = fake_tts(
            None,  # type: ignore[arg-type]
            tts_durations,
        )
        try:
            out_path = str(self.tmpdir / "test_graph.mp4")
            result = renderer.render_from_script(
                video_manifest=self.manifest,
                script_beats=self.beats,
                output_path=out_path,
                output_mode="auto",
            )
            self.assertIsNotNone(result)
            final_path = Path(result["final_path"])
            self.assertTrue(final_path.exists())

            final_dur = _media_duration(final_path)
            expected_dur = sum(tts_durations.values())
            self.assertAlmostEqual(final_dur, expected_dur, delta=0.3)

            # Verify no clip content was trimmed: each demo clip duration should
            # be at least as long as the original.
            for beat in self.beats:
                original_dur = _media_duration(Path(beat.video_clip_path))
                self.assertGreaterEqual(original_dur, tts_durations[beat.beat_id] - 0.05)

            # Last frame of rendered video should match last frame of last demo clip.
            demo_beats = [b for b in self.beats if b.kind == "demo"]
            last_demo_clip = Path(demo_beats[-1].video_clip_path)
            rendered_last = self.tmpdir / "rendered_last.png"
            clip_last = self.tmpdir / "clip_last.png"
            _extract_last_frame(final_path, rendered_last)
            _extract_last_frame(last_demo_clip, clip_last)

            rendered_img = np.array(_pil_open(rendered_last))
            clip_img_raw = np.array(_pil_open(clip_last))
            # The renderer scales clips to VIDEO_MAX_WIDTH; scale the clip frame
            # to match the rendered output before pixel comparison.
            scale = 1280 / clip_img_raw.shape[1]
            new_h = int(round(clip_img_raw.shape[0] * scale))
            from PIL import Image
            clip_img = np.array(
                Image.fromarray(clip_img_raw).resize(
                    (1280, new_h), Image.Resampling.LANCZOS
                )
            )
            self.assertEqual(rendered_img.shape, clip_img.shape)
            mse = np.mean((rendered_img.astype(float) - clip_img.astype(float)) ** 2)
            self.assertLess(mse, 5.0)
        finally:
            restore_tts(original)


def _pil_open(path: Path) -> Any:
    from PIL import Image
    return Image.open(str(path))


class TestAdaptBeatsToObservedState(unittest.TestCase):
    def setUp(self) -> None:
        self.builder = LessonBuilder()

    def test_demo_beats_keep_clips_when_state_unchanged(self) -> None:
        """Demo beats must stay demo beats so the renderer can use their clips."""
        beats = [
            ScriptBeat(
                beat_id="beat_001",
                kind="demo",
                text="We type the query.",
                action={"type": "type_block", "text": "SELECT 1;"},
                video_clip_path="/tmp/beat_001.mp4",
                observed_state={
                    "active_tab": "Execute SQL",
                    "visible_table": "",
                    "row_range_text": "",
                    "column_headers": [],
                    "summary": "Query typed in editor.",
                },
            ),
            ScriptBeat(
                beat_id="beat_002",
                kind="demo",
                text="We run the query.",
                action={"type": "run_query"},
                video_clip_path="/tmp/beat_002.mp4",
                observed_state={
                    "active_tab": "Execute SQL",
                    "visible_table": "",
                    "row_range_text": "",
                    "column_headers": [],
                    "summary": "Query still in editor.",
                },
            ),
        ]
        self.builder._enforce_clip_truthfulness(beats)
        self.assertEqual(beats[0].kind, "demo")
        self.assertEqual(beats[0].video_clip_path, "/tmp/beat_001.mp4")
        self.assertEqual(beats[1].kind, "demo")
        self.assertEqual(beats[1].video_clip_path, "/tmp/beat_002.mp4")

    def test_validation_beats_are_not_converted_to_state(self) -> None:
        """Validation beats must remain validation beats."""
        beats = [
            ScriptBeat(
                beat_id="beat_001",
                kind="validation",
                text="We see 60 rows returned, confirming the query succeeded.",
                observed_state={
                    "active_tab": "Execute SQL",
                    "visible_table": "",
                    "row_range_text": "60 rows",
                    "column_headers": ["FirstName", "LastName", "Email"],
                    "summary": "Results grid visible.",
                },
            ),
        ]
        self.builder._adapt_beats_to_observed_state(beats)
        self.assertEqual(beats[0].kind, "validation")


class TestValidationEchoSemantic(unittest.TestCase):
    def setUp(self) -> None:
        self.builder = LessonBuilder()

    def test_redundant_validation_echo_is_dropped(self) -> None:
        """A validation beat that only restates the previous two beats is merged."""
        beats = [
            ScriptBeat(beat_id="beat_001", kind="demo", text="We click Execute SQL."),
            ScriptBeat(beat_id="beat_002", kind="demo", text="We type SELECT FirstName FROM Customer."),
            ScriptBeat(
                beat_id="beat_003",
                kind="validation",
                text="We see the query in the editor and the result pane.",
            ),
        ]
        merged = self.builder._merge_validation_echoes(beats)
        self.assertEqual([b.beat_id for b in merged], ["beat_001", "beat_002"])

    def test_validation_with_new_row_count_is_kept(self) -> None:
        """A validation beat that adds a new concrete number is preserved."""
        beats = [
            ScriptBeat(beat_id="beat_001", kind="demo", text="We click Execute SQL."),
            ScriptBeat(beat_id="beat_002", kind="demo", text="We type SELECT FirstName FROM Customer."),
            ScriptBeat(
                beat_id="beat_003",
                kind="validation",
                text="We see 60 rows returned, confirming the query succeeded.",
            ),
        ]
        merged = self.builder._merge_validation_echoes(beats)
        self.assertEqual([b.beat_id for b in merged], ["beat_001", "beat_002", "beat_003"])

    def test_validation_no_previous_beats_is_kept(self) -> None:
        """A validation beat at the start has nothing to echo, so it stays."""
        beats = [
            ScriptBeat(
                beat_id="beat_001",
                kind="validation",
                text="We see the result pane with 60 rows.",
            ),
        ]
        merged = self.builder._merge_validation_echoes(beats)
        self.assertEqual([b.beat_id for b in merged], ["beat_001"])


class TestScriptSimilarityGate(unittest.TestCase):
    def setUp(self) -> None:
        self.builder = LessonBuilder()

    def test_similar_beats_are_flagged(self) -> None:
        """Two beats >75% similar violate the script similarity gate."""
        beats = [
            ScriptBeat(
                beat_id="beat_002",
                kind="state",
                text=(
                    "The editor is empty and the result pane below it is blank. "
                    "When we finish, the editor will hold a comment block followed by a "
                    "formatted SELECT statement, and the result pane will show the customer "
                    "contact list."
                ),
            ),
            ScriptBeat(
                beat_id="beat_004",
                kind="state",
                text=(
                    "Right now the editor is empty and the result pane below it is blank. "
                    "When we finish, the editor will hold a comment block followed by a "
                    "formatted SELECT statement, and the result pane will show the customer "
                    "contact list."
                ),
            ),
        ]
        similar = self.builder._find_similar_beats(beats)
        self.assertTrue(similar, "duplicate state beats should be flagged as similar")
        self.assertGreater(similar[0][2], 0.75)

    def test_distinct_beats_pass(self) -> None:
        """Different beats are not flagged as similar."""
        beats = [
            ScriptBeat(
                beat_id="beat_002",
                kind="state",
                text="The editor is empty and the result pane below it is blank.",
            ),
            ScriptBeat(
                beat_id="beat_004",
                kind="state",
                text="The next three actions will add a comment header, the SELECT clause, and the FROM clause.",
            ),
        ]
        similar = self.builder._find_similar_beats(beats)
        self.assertFalse(similar)


class TestScriptIntegrityGate(unittest.TestCase):
    def setUp(self) -> None:
        self.builder = LessonBuilder()

    def test_synthetic_mid_sentence_is_rewritten(self) -> None:
        """A beat ending mid-sentence must fail the gate and be rewritten."""
        beats = [
            ScriptBeat(
                beat_id="beat_001",
                kind="demo",
                text="We open the Execute SQL tab and the editor",
                action={"type": "click", "detail": "Execute SQL tab"},
            ),
        ]
        self.assertFalse(self.builder.script_integrity_ok(beats))
        self.builder._enforce_sentence_integrity(beats)
        self.assertTrue(self.builder.script_integrity_ok(beats))
        self.assertRegex(beats[0].text, r"[.!?]$")

    def test_validation13_script_passes_after_enforcement(self) -> None:
        """The saved Phase 1 pilot script must be fixable by the integrity gate."""
        manifest = load_manifest("sql_essential_training_ch4")
        if manifest is not None and manifest.videos[0].script_beats:
            beats = [_dict_to_script_beat(b) for b in manifest.videos[0].script_beats]
        else:
            # Manifest is generated on first run; use the canonical Phase 1 pilot
            # beats inline so the gate test stays self-contained.
            beats = [
                ScriptBeat(beat_id="beat_001", kind="opening", text="In this video, we will write our first SELECT query to pull a customer contact"),
                ScriptBeat(beat_id="beat_002", kind="concept", text="SELECT tells the database which columns we want, and FROM tells it which table holds"),
                ScriptBeat(beat_id="beat_003", kind="demo", text="We open the Execute SQL tab.", action={"type": "click", "detail": "Execute SQL tab"}),
                ScriptBeat(beat_id="beat_004", kind="demo", text="We type a comment block so we remember what this query is", action={"type": "type_block", "text": "-- comment"}),
                ScriptBeat(beat_id="beat_005", kind="demo", text="We type the query that asks for first name, last name, and", action={"type": "type_block", "text": "SELECT 1;"}),
                ScriptBeat(beat_id="beat_006", kind="demo", text="We run the query and the result pane fills with the contact", action={"type": "run_query"}),
                ScriptBeat(beat_id="beat_007", kind="explain", text="The result pane shows 60 rows with FirstName, LastName, Email, giving us the complete customer"),
                ScriptBeat(beat_id="beat_008", kind="validation", text="We see 60 rows returned in the result pane, confirming the contact list is complete."),
                ScriptBeat(beat_id="beat_009", kind="close", text="We have written our first SELECT query and pulled the customer contact list. Next, we"),
            ]
        # The CLI regenerates the manifest, so it may already be complete. If it
        # is still broken, it must fail the gate before enforcement.
        if not self.builder.script_integrity_ok(beats):
            self.assertFalse(self.builder.script_integrity_ok(beats))
        self.builder._enforce_sentence_integrity(beats)
        self.assertTrue(self.builder.script_integrity_ok(beats))

    def test_validation13_original_script_fails_gate(self) -> None:
        """The exact Phase 1 validation13 script (mid-sentence) must fail the gate."""
        original_validation13_beats = [
            ScriptBeat(beat_id="beat_001", kind="opening", text="In this video, we will write our first SELECT query to pull a customer contact"),
            ScriptBeat(beat_id="beat_002", kind="concept", text="SELECT tells the database which columns we want, and FROM tells it which table holds"),
            ScriptBeat(beat_id="beat_003", kind="demo", text="We open the Execute SQL tab.", action={"type": "click", "detail": "Execute SQL tab"}),
            ScriptBeat(beat_id="beat_004", kind="demo", text="We type a comment block so we remember what this query is", action={"type": "type_block", "text": "-- comment"}),
            ScriptBeat(beat_id="beat_005", kind="demo", text="We type the query that asks for first name, last name, and", action={"type": "type_block", "text": "SELECT 1;"}),
            ScriptBeat(beat_id="beat_006", kind="demo", text="We run the query and the result pane fills with the contact", action={"type": "run_query"}),
            ScriptBeat(beat_id="beat_007", kind="explain", text="The result pane shows 60 rows with FirstName, LastName, Email, giving us the complete customer"),
            ScriptBeat(beat_id="beat_008", kind="validation", text="We see 60 rows returned in the result pane, confirming the contact list is complete."),
            ScriptBeat(beat_id="beat_009", kind="close", text="We have written our first SELECT query and pulled the customer contact list. Next, we"),
        ]
        self.assertFalse(self.builder.script_integrity_ok(original_validation13_beats))
        self.builder._enforce_sentence_integrity(original_validation13_beats)
        self.assertTrue(self.builder.script_integrity_ok(original_validation13_beats))
        for beat in original_validation13_beats:
            self.assertRegex(beat.text, r"[.!?]$")


class TestEditorReadBack(unittest.TestCase):
    def _agent_with_mocks(self) -> VisionAgent:
        agent = VisionAgent()
        mock.patch.object(agent, "find_and_click", return_value=True).start()
        mock.patch.object(agent, "press_key", return_value=True).start()
        self.addCleanup(mock.patch.stopall)
        return agent

    def test_exact_match_succeeds_first_try(self) -> None:
        """When the VLM read-back matches, type_block succeeds immediately."""
        agent = self._agent_with_mocks()
        with (
            mock.patch.object(agent, "_read_editor_content", return_value="SELECT 1;"),
            mock.patch("pyautogui.typewrite"),
            mock.patch("pyautogui.press"),
            mock.patch("time.sleep"),
            mock.patch.object(sys, "stderr", io.StringIO()) as stderr,
        ):
            self.assertTrue(agent.type_block("SELECT 1;"))
            log = stderr.getvalue()
            self.assertIn("[TYPE BLOCK] read-back OK", log)
            self.assertNotIn("read-back mismatch", log)

    def test_mismatch_then_match_succeeds_and_logs_retry(self) -> None:
        """A mismatched first read-back followed by a match should retry and succeed."""
        agent = self._agent_with_mocks()
        with (
            mock.patch.object(
                agent, "_read_editor_content", side_effect=["WRONG", "SELECT 1;"]
            ),
            mock.patch("pyautogui.typewrite"),
            mock.patch("pyautogui.press"),
            mock.patch("time.sleep"),
            mock.patch.object(sys, "stderr", io.StringIO()) as stderr,
        ):
            self.assertTrue(agent.type_block("SELECT 1;"))
            log = stderr.getvalue()
            self.assertIn("[TYPE BLOCK] read-back mismatch", log)
            self.assertIn("[TYPE BLOCK] retry 1/2", log)
            self.assertIn("[TYPE BLOCK] read-back OK", log)

    def test_full_block_adjacency_verified(self) -> None:
        """A comment block followed immediately by a query passes layout verification."""
        agent = self._agent_with_mocks()
        block = "/*\nCreated By: WSDA Student\nDescription: Test\n*/\n\nSELECT 1;"
        with (
            mock.patch.object(agent, "_read_editor_content", return_value=block),
            mock.patch("pyautogui.typewrite"),
            mock.patch("pyautogui.press"),
            mock.patch("time.sleep"),
            mock.patch.object(sys, "stderr", io.StringIO()) as stderr,
        ):
            self.assertTrue(agent.type_block(block))
            log = stderr.getvalue()
            self.assertIn("[TYPE BLOCK] line-adjacency OK", log)


class TestExactLineTyping(unittest.TestCase):
    def _agent_with_profile(self) -> VisionAgent:
        profile = EnvironmentProfile(
            application="db_browser_sqlite",
            app_name="DB Browser for SQLite",
            focus_target="DB Browser for SQLite",
            whitespace_policy="exact",
        )
        agent = VisionAgent(profile=profile)
        mock.patch.object(agent, "find_and_click", return_value=True).start()
        mock.patch.object(agent, "press_key", return_value=True).start()
        self.addCleanup(mock.patch.stopall)
        return agent

    def test_line_by_line_pastes_byte_for_byte(self) -> None:
        """A multi-line block with leading spaces is pasted line-by-line exactly."""
        agent = self._agent_with_profile()
        state = {"text": ""}
        last_pasted: List[str] = []

        def paste_line_effect(line: str, *args: Any, **kwargs: Any) -> None:
            state["text"] += line + "\n"
            last_pasted.append(line)

        def read_back(focus: bool = True) -> str:
            return state["text"]

        def current_line() -> str:
            return last_pasted[-1] if last_pasted else ""

        intended = "SELECT\n    FirstName,\n    LastName\nFROM Customer;"
        with (
            mock.patch.object(agent, "_paste_line", side_effect=paste_line_effect),
            mock.patch.object(agent, "_read_editor_content", side_effect=read_back),
            mock.patch.object(agent, "_read_current_line", side_effect=current_line),
            mock.patch("time.sleep"),
        ):
            self.assertTrue(agent._type_text_line_by_line(intended))
            self.assertEqual(state["text"].rstrip("\n"), intended)

    def test_line_paste_preserves_authored_indent(self) -> None:
        """Line-paste does not strip leading spaces; the authored indent ships as-is."""
        agent = self._agent_with_profile()
        pasted: List[str] = []

        def paste_line_effect(line: str, *args: Any, **kwargs: Any) -> None:
            pasted.append(line)

        with (
            mock.patch.object(agent, "_paste_line", side_effect=paste_line_effect),
            mock.patch("time.sleep"),
        ):
            agent._type_line("    FirstName,")
            self.assertEqual(pasted, ["    FirstName,"])

    def test_dropped_leading_characters_trigger_line_repair(self) -> None:
        """Lost leading characters such as 'tName' are caught and repaired."""
        agent = self._agent_with_profile()
        state = {"text": "", "read_back_count": 0, "current_line_count": 0}
        intended = "SELECT\n    tName\nFROM Customer;"
        last_pasted: List[str] = []

        def paste_line_effect(line: str, *args: Any, **kwargs: Any) -> None:
            state["text"] += line + "\n"
            last_pasted.append(line)

        def read_back(focus: bool = True) -> str:
            state["read_back_count"] += 1
            if state["read_back_count"] == 2:
                # Simulate the corruption: the leading spaces and first character
                # of the second line were dropped.
                return "SELECT\nName\nFROM Customer;\n"
            return state["text"]

        def current_line() -> str:
            state["current_line_count"] += 1
            if state["current_line_count"] == 2:
                # Per-line read-back sees the corrupted line.
                return "Name"
            return last_pasted[-1] if last_pasted else ""

        repaired: List[str] = []

        def repair_effect(line: str) -> bool:
            repaired.append(line)
            state["text"] = state["text"].replace("\nName\n", "\n    tName\n")
            return True

        with (
            mock.patch.object(agent, "_paste_line", side_effect=paste_line_effect),
            mock.patch.object(agent, "_read_editor_content", side_effect=read_back),
            mock.patch.object(agent, "_read_current_line", side_effect=current_line),
            mock.patch.object(agent, "_repair_line", side_effect=repair_effect),
            mock.patch("time.sleep"),
        ):
            self.assertTrue(agent._type_text_line_by_line(intended))
            self.assertIn("    tName", repaired)


class TestDatumLevelEchoDetection(unittest.TestCase):
    def setUp(self) -> None:
        self.builder = LessonBuilder()

    def test_repeated_row_count_and_columns_are_removed(self) -> None:
        """
        A validation beat that restates the previous beat's row count and column
        list must be rewritten to drop the repeated data.
        """
        beats = [
            ScriptBeat(
                beat_id="beat_007",
                kind="explain",
                text="The result pane shows 60 rows with FirstName, LastName, and Email.",
            ),
            ScriptBeat(
                beat_id="beat_008",
                kind="validation",
                text="We see 60 rows returned in the result pane, confirming the contact list is complete.",
            ),
        ]
        self.builder._enforce_datum_uniqueness(beats)
        self.assertNotIn("60", beats[1].text)
        for name in ("FirstName", "LastName", "Email"):
            self.assertNotIn(name, beats[1].text)


class TestUIGrounding(unittest.TestCase):
    def setUp(self) -> None:
        self.builder = LessonBuilder()

    def test_state_beat_with_ungrounded_ui_count_is_flagged(self) -> None:
        """A state beat asserting a UI element count absent from observed state conflicts."""
        beat = ScriptBeat(
            beat_id="beat_002",
            kind="state",
            text="DB Browser for SQLite opens with two tabs above the data view.",
            observed_state={
                "active_tab": "Browse Data",
                "visible_table": "Customer",
                "row_range_text": "1 - 20 of 60",
                "column_headers": ["FirstName", "LastName", "Email"],
                "ui_element_counts": None,
            },
        )
        self.assertTrue(self.builder._beat_conflicts_with_observed_state(beat))

    def test_state_beat_with_mismatched_ui_count_is_flagged(self) -> None:
        """A state beat asserting a UI element count that contradicts observed state conflicts."""
        beat = ScriptBeat(
            beat_id="beat_002",
            kind="state",
            text="DB Browser for SQLite opens with three tabs above the data view.",
            observed_state={
                "active_tab": "Browse Data",
                "visible_table": "Customer",
                "row_range_text": "1 - 20 of 60",
                "column_headers": ["FirstName", "LastName", "Email"],
                "ui_element_counts": {"tabs": 2},
            },
        )
        self.assertTrue(self.builder._beat_conflicts_with_observed_state(beat))

    def test_state_beat_with_matching_ui_count_is_not_flagged(self) -> None:
        """A state beat asserting a UI element count that matches observed state is fine."""
        beat = ScriptBeat(
            beat_id="beat_002",
            kind="state",
            text="DB Browser for SQLite opens with two tabs above the data view.",
            observed_state={
                "active_tab": "Browse Data",
                "visible_table": "Customer",
                "row_range_text": "1 - 20 of 60",
                "column_headers": ["FirstName", "LastName", "Email"],
                "ui_element_counts": {"tabs": 2},
            },
        )
        self.assertFalse(self.builder._beat_conflicts_with_observed_state(beat))

    def test_non_state_beat_without_grounding_is_not_flagged(self) -> None:
        """Concept/demo beats that mention counts without UI grounding are not auto-flagged."""
        beat = ScriptBeat(
            beat_id="beat_003",
            kind="concept",
            text="The toolbar shows several useful buttons for running queries.",
            observed_state={
                "active_tab": "Execute SQL",
                "ui_element_counts": None,
            },
        )
        self.assertFalse(self.builder._beat_conflicts_with_observed_state(beat))


class TestFrontmostGate(unittest.TestCase):
    def test_clean_interval_returns_false(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            log = Path(tmpdir) / "frontmost.log"
            log.write_text(
                "1000.000\tDB Browser for SQLite\n"
                "1001.000\tDB Browser for SQLite\n",
                encoding="utf-8",
            )
            self.assertFalse(
                _clip_has_off_app_interval(log, 1000.0, 1002.0, "DB Browser for SQLite")
            )

    def test_off_app_interval_returns_true(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            log = Path(tmpdir) / "frontmost.log"
            log.write_text(
                "1000.000\tDB Browser for SQLite\n"
                "1001.000\tLaunchpad\n"
                "1002.000\tDB Browser for SQLite\n",
                encoding="utf-8",
            )
            self.assertTrue(
                _clip_has_off_app_interval(log, 999.0, 1003.0, "DB Browser for SQLite")
            )


class TestPasteAirlock(unittest.TestCase):
    def test_type_block_uses_paste_not_typewrite(self) -> None:
        """SQL type_block must paste from the clipboard, never type characters."""
        agent = VisionAgent()
        with (
            mock.patch.object(agent, "_clear_editor") as mock_clear,
            mock.patch.object(agent, "_paste_text") as mock_paste,
            mock.patch.object(agent, "_type_visible") as mock_type_visible,
            mock.patch.object(
                agent, "_read_editor_content", return_value="SELECT * FROM Orders;"
            ),
            mock.patch.object(agent, "press_key") as mock_press,
            mock.patch("compiler.vision_agent.pyautogui.typewrite") as mock_typewrite,
        ):
            result = agent.type_block("SELECT * FROM Orders;")
            self.assertTrue(result)
            mock_paste.assert_called_once_with("SELECT * FROM Orders;")
            mock_type_visible.assert_not_called()
            mock_typewrite.assert_not_called()
            mock_press.assert_called_with("esc")


class TestRunQuery(unittest.TestCase):
    def test_run_query_does_not_press_f5(self) -> None:
        """run_query must click the Execute/Run toolbar button, never F5."""
        agent = VisionAgent()
        click_action = {"action": "click", "point": {"x": 100, "y": 100}}

        def vlm_side_effect(prompt: str, **kwargs: Any) -> VisionAgentResult:
            if "Execute SQL toolbar button" in prompt:
                return VisionAgentResult(action=click_action, text="")
            return VisionAgentResult(text="YES")

        with (
            mock.patch.object(agent, "_call_vlm", side_effect=vlm_side_effect),
            mock.patch.object(agent, "_ensure_frontmost"),
            mock.patch.object(agent, "_read_editor_content", return_value="SELECT 1;"),
            mock.patch.object(agent, "_result_pane_shows_error", return_value=False),
            mock.patch.object(agent, "_results_pane_snapshot", return_value={"phash": (0,) * 64}),
            mock.patch.object(agent, "_results_pane_changed", return_value=True),
            mock.patch("compiler.vision_agent.pyautogui.moveTo"),
            mock.patch("compiler.vision_agent.pyautogui.click"),
            mock.patch.object(agent, "press_key") as mock_press,
        ):
            self.assertTrue(agent.run_query())
            for call in mock_press.call_args_list:
                self.assertNotEqual(str(call.args[0]).upper(), "F5")


class TestWholeVideoFrameGate(unittest.TestCase):
    def test_bad_frame_raises_runtime_error(self) -> None:
        """If the VLM reports a frame without the target app, the gate must raise."""
        with tempfile.TemporaryDirectory() as tmpdir:
            video = _make_video(Path(tmpdir) / "clip.mp4", duration=6.0)
            profile = EnvironmentProfile(
                application="db_browser_sqlite",
                app_name="DB Browser for SQLite",
                focus_target="DB Browser for SQLite",
            )
            with mock.patch.object(
                VisionAgent,
                "verify_app_visible_in_frames",
                return_value=[True, True, False],
            ):
                with self.assertRaises(RuntimeError):
                    _verify_video_frames_show_app(str(video), profile=profile, interval=2.0)


class TestSegmentedTyping(unittest.TestCase):
    def test_segments_type_and_verify_each(self) -> None:
        """type_segments types each segment and verifies the cumulative editor content."""
        agent = VisionAgent()
        segments = [
            {"text": "SELECT\n    FirstName,"},
            {"text": "\n    LastName"},
            {"text": "\nFROM Customer;"},
        ]
        expected = ""
        def type_side_effect(text: str) -> None:
            nonlocal expected
            expected += text
        def read_back(*args, **kwargs) -> str:
            return expected
        with (
            mock.patch.object(agent, "_ensure_frontmost") as mock_frontmost,
            mock.patch.object(agent, "_type_segment_cadence", side_effect=type_side_effect) as mock_type,
            mock.patch.object(agent, "_read_editor_content", side_effect=read_back),
        ):
            self.assertTrue(agent.type_segments(segments))
            self.assertEqual(mock_type.call_count, 3)
            # One frontmost check at entry plus one before each of the 3 segments.
            self.assertEqual(mock_frontmost.call_count, 4)

    def test_segment_retry_then_paste_fallback(self) -> None:
        """After two segment mismatches, type_segments falls back to paste."""
        agent = VisionAgent()
        segments = [
            {"text": "SELECT 1;"},
            {"text": "SELECT 2;"},
        ]
        # Segments that do not start with whitespace are separated by a newline
        # when appended to non-empty editor content.
        remaining = "SELECT 1;\nSELECT 2;"

        def verify_side_effect(intended: str, label: str = "") -> bool:
            # Fail first two segment checks, fail in-place repair, succeed fallback.
            norm = agent._normalize_editor_text(intended)
            if norm == agent._normalize_editor_text("SELECT 1;"):
                return False
            if norm == agent._normalize_editor_text(remaining):
                return True
            return False

        with (
            mock.patch.object(agent, "_ensure_frontmost"),
            mock.patch.object(agent, "_type_segment_cadence"),
            mock.patch.object(agent, "_undo_segment"),
            mock.patch.object(agent, "_verify_buffer_exact", side_effect=verify_side_effect),
            mock.patch.object(agent, "_read_editor_content", return_value=""),
            mock.patch.object(agent, "_append_text") as mock_append,
            mock.patch.object(agent, "_clear_editor") as mock_clear,
            mock.patch.object(agent, "_paste_text") as mock_paste,
        ):
            self.assertTrue(agent.type_segments(segments))
            mock_append.assert_called_once()
            mock_clear.assert_called_once()
            mock_paste.assert_called_once()
            pasted = mock_paste.call_args[0][0]
            self.assertIn("SELECT 1;", pasted)
            self.assertIn("SELECT 2;", pasted)


class TestC17DeterministicDemo(unittest.TestCase):
    """C17: AX-first editor focus, cached run button, verification-free recording paste."""

    def _agent(self) -> VisionAgent:
        agent = VisionAgent()
        self.addCleanup(mock.patch.stopall)
        return agent

    def test_focus_editor_fast_path_skips_vlm_clicks(self) -> None:
        """When AX focus succeeds, no VLM click is paid."""
        agent = self._agent()
        self.addCleanup(mock.patch.stopall)
        _patch_focus_guard([{"element": "el", "pid": 999, "role": "AXTerminal"}])
        mock.patch.object(agent, "_ensure_frontmost").start()
        fac = mock.patch.object(agent, "find_and_click", return_value=True).start()
        with mock.patch.object(
            agent, "_ensure_editor_focused_accessibility", return_value=True
        ):
            agent._focus_editor()
        fac.assert_not_called()

    def test_focus_editor_falls_back_to_vlm_when_no_text_area(self) -> None:
        """When the AX tree has no text area, both VLM clicks run (tab + editor)."""
        agent = self._agent()
        self.addCleanup(mock.patch.stopall)
        _patch_focus_guard([{"element": "el", "pid": 999, "role": "AXTerminal"}])
        mock.patch.object(agent, "_ensure_frontmost").start()
        mock.patch.object(agent, "_log_ax_failure_context").start()
        mock.patch.object(ax_pyobjc, "find_text_areas", return_value=[]).start()
        mock.patch("time.sleep").start()
        fac = mock.patch.object(agent, "find_and_click", return_value=True).start()
        agent._focus_editor()
        self.assertEqual(fac.call_count, 2)

    def test_run_query_uses_cached_button_without_vlm(self) -> None:
        """A primed run-button cache must serve run_query with zero VLM calls."""
        agent = self._agent()
        agent._run_button_point = (120, 130)
        fac = mock.patch.object(agent, "find_and_click", return_value=True).start()
        with (
            mock.patch.object(agent, "_ensure_frontmost"),
            mock.patch.object(agent, "_read_editor_content", return_value="SELECT 1;"),
            mock.patch.object(agent, "_extract_uncommented_sql", return_value=("SELECT 1;", "")),
            mock.patch.object(agent, "_verify_statement_isolation", return_value=True),
            mock.patch.object(agent, "_result_pane_shows_error", return_value=False),
            mock.patch.object(agent, "_results_pane_snapshot", return_value={"phash": (0,) * 64}),
            mock.patch.object(agent, "_results_pane_changed", return_value=True),
            mock.patch("compiler.vision_agent.pyautogui.moveTo"),
            mock.patch("compiler.vision_agent.pyautogui.click"),
            mock.patch("time.sleep"),
        ):
            self.assertTrue(agent.run_query())
        fac.assert_not_called()

    def test_recording_line_paste_skips_all_verification(self) -> None:
        """During recording the paste is deterministic: no per-line or block checks."""
        agent = self._agent()
        agent.recording = True
        pasted: List[str] = []
        read_current = mock.patch.object(agent, "_read_current_line").start()
        read_editor = mock.patch.object(agent, "_read_editor_content", return_value="").start()
        canonical = mock.patch.object(agent, "_canonical_compare", return_value=True).start()
        with (
            mock.patch.object(
                agent, "_paste_line", side_effect=lambda line, **kw: pasted.append(line)
            ),
            mock.patch.object(agent, "_safe_hotkey"),
            mock.patch("time.sleep"),
        ):
            self.assertTrue(agent._type_text_line_by_line("SELECT\nFROM Customer;"))
        self.assertEqual(pasted, ["SELECT", "FROM Customer;"])
        read_current.assert_not_called()
        read_editor.assert_not_called()
        canonical.assert_not_called()

    def test_rehearsal_line_paste_still_verifies(self) -> None:
        """Outside recording the per-line read-back verification is preserved."""
        agent = self._agent()
        agent.recording = False
        read_current = mock.patch.object(
            agent, "_read_current_line", side_effect=lambda: "SELECT"
        ).start()
        with (
            mock.patch.object(agent, "_paste_line"),
            mock.patch.object(agent, "_read_editor_content", return_value="SELECT"),
            mock.patch.object(agent, "_canonical_compare", return_value=True),
            mock.patch.object(agent, "_safe_hotkey"),
            mock.patch("time.sleep"),
        ):
            self.assertTrue(agent._type_text_line_by_line("SELECT"))
        read_current.assert_called()


class TestC17NarrationSizing(unittest.TestCase):
    """C17: LessonBuilder sizes demo narration to measured action + gestures."""

    @staticmethod
    def _write_measurements(tmpdir: str, video_id: str, rows: List[Dict[str, Any]]) -> None:
        payload = {"video_id": video_id, "beats": rows}
        Path(tmpdir, f"action_seconds_{video_id}.json").write_text(json.dumps(payload))

    def test_demo_narration_expands_to_measured_action(self) -> None:
        builder = LessonBuilder()
        with tempfile.TemporaryDirectory() as tmp:
            self._write_measurements(
                tmp,
                "video_x",
                [
                    {
                        "beat_id": "beat_002",
                        "action_type": "type_segments",
                        "action_seconds": 30.0,
                        "ok": True,
                    }
                ],
            )
            beat = ScriptBeat(
                beat_id="beat_002",
                kind="demo",
                text="We type the SELECT clause listing the columns.",
                action={"type": "type_segments", "segments": [{"text": "SELECT 1;"}]},
            )
            expanded_text = (
                "We type the SELECT clause listing the columns FirstName, LastName, and "
                "Email so the report shows only the contact fields. "
            ) * 8
            block = mock.Mock(type="text", text=expanded_text.strip())
            resp = mock.Mock(content=[block])
            with (
                mock.patch.dict("os.environ", {"WSDA_ACTION_SECONDS_DIR": tmp}),
                mock.patch(
                    "compiler.lesson_builder.tracked_create", return_value=resp
                ) as mock_create,
            ):
                builder._size_demo_narration([beat], "video_x")
            self.assertEqual(beat.planned_duration, 30.0)
            self.assertEqual(beat.text, expanded_text.strip())
            self.assertEqual(mock_create.call_count, 1)

    def test_demo_narration_within_tolerance_unchanged(self) -> None:
        """C19: text within ±2s of action+4s variance+3s gestures is untouched."""
        builder = LessonBuilder()
        with tempfile.TemporaryDirectory() as tmp:
            self._write_measurements(
                tmp,
                "video_x",
                [
                    {
                        "beat_id": "beat_002",
                        "action_type": "type_segments",
                        # 23-word text ≈ 8.4s; target = 1.5 + 4 + 3 = 8.5s.
                        "action_seconds": 1.5,
                        "ok": True,
                    }
                ],
            )
            text = (
                "We type the SELECT clause listing the columns FirstName, LastName, and "
                "Email so the report shows only the contact fields management asked for."
            )
            beat = ScriptBeat(beat_id="beat_002", kind="demo", text=text, action=None)
            with (
                mock.patch.dict("os.environ", {"WSDA_ACTION_SECONDS_DIR": tmp}),
                mock.patch("compiler.lesson_builder.tracked_create") as mock_create,
            ):
                builder._size_demo_narration([beat], "video_x")
            self.assertEqual(beat.text, text)
            mock_create.assert_not_called()

    def test_failed_measurements_are_ignored(self) -> None:
        builder = LessonBuilder()
        with tempfile.TemporaryDirectory() as tmp:
            self._write_measurements(
                tmp,
                "video_x",
                [
                    {
                        "beat_id": "beat_002",
                        "action_type": "type_segments",
                        "action_seconds": 30.0,
                        "ok": False,
                    }
                ],
            )
            beat = ScriptBeat(
                beat_id="beat_002",
                kind="demo",
                text="We type the SELECT clause listing the columns.",
                action=None,
            )
            with mock.patch.dict("os.environ", {"WSDA_ACTION_SECONDS_DIR": tmp}):
                builder._size_demo_narration([beat], "video_x")
            self.assertIsNone(beat.planned_duration)


class TestC18ConsolidatedSegments(unittest.TestCase):
    """C18: contiguous lines merge into one segment; dismissal moves to per-beat."""

    @staticmethod
    def _video_1_1():
        class MockVideo:
            video_id = "video_1_1"
            title = "Test"
            learning_objective = "Test"
            discovery_objective = "Test"
            application = "db_browser_sqlite"
            format_tier = "short"
            exercise_artifact = {}
            planned_queries = []

        return MockVideo()

    def test_video_1_1_demo_beats_have_consolidated_segments(self) -> None:
        """beat_002 <= 2 segments, beat_003 <= 2, beat_004 == 1; merged text carries newlines."""
        builder = LessonBuilder()
        beats = builder._build_sql_script_beats(self._video_1_1())
        by_id = {b.beat_id: b for b in beats}

        seg2 = by_id["beat_002"].action["segments"]
        self.assertLessEqual(len(seg2), 2, seg2)
        self.assertEqual([s["sentence_idx"] for s in seg2], [0, 1])
        self.assertIn("\n", seg2[0]["text"])  # comment header merged, not one segment per line
        self.assertIn("/*", seg2[0]["text"])
        self.assertIn("*/", seg2[1]["text"])

        seg3 = by_id["beat_003"].action["segments"]
        self.assertLessEqual(len(seg3), 2, seg3)
        self.assertIn("\nSELECT", "\n" + seg3[0]["text"])
        self.assertIn("Email", seg3[0]["text"])

        seg4 = by_id["beat_004"].action["segments"]
        self.assertEqual(len(seg4), 1, seg4)
        self.assertIn("FROM Customer;", seg4[0]["text"])

    def test_paste_line_uses_flat_cadence_and_no_dismissal(self) -> None:
        """C18: _paste_line sleeps a flat 0.4s and never dismisses the Character Viewer."""
        agent = VisionAgent()
        dismiss = mock.patch.object(agent, "_dismiss_character_viewer").start()
        self.addCleanup(mock.patch.stopall)
        with (
            mock.patch.object(agent, "_copy_to_clipboard"),
            mock.patch.object(agent, "_clipboard_matches", return_value=True),
            mock.patch.object(agent, "_safe_hotkey"),
            mock.patch("time.sleep") as mock_sleep,
        ):
            agent._paste_line("SELECT")
            agent._paste_line("FROM Customer;", add_newline=False)
        dismiss.assert_not_called()
        sleeps = [c.args[0] for c in mock_sleep.call_args_list if c.args]
        self.assertEqual(max(sleeps), 0.4)

    def test_focus_editor_dismisses_once_per_call(self) -> None:
        """The single per-beat Character Viewer dismissal lives in _focus_editor."""
        agent = VisionAgent()
        mock.patch.object(agent, "_ensure_frontmost").start()
        dismiss = mock.patch.object(agent, "_dismiss_character_viewer").start()
        focus_ax = mock.patch.object(
            agent, "_ensure_editor_focused_accessibility", return_value=True
        ).start()
        _patch_focus_guard([{"element": "el", "pid": 999, "role": "AXTerminal"}])
        self.addCleanup(mock.patch.stopall)
        agent._focus_editor()
        dismiss.assert_called_once()
        focus_ax.assert_called_once()


class TestC19AppleScriptPaste(unittest.TestCase):
    """C19: single-AppleScript paste hotkey, one focus per beat, sizing margin."""

    def test_paste_line_uses_one_applescript_keystroke_no_key_storm(self) -> None:
        """_paste_line pastes via one osascript keystroke and zero pyautogui events."""
        agent = VisionAgent()
        self.addCleanup(mock.patch.stopall)
        key_down = mock.patch("compiler.vision_agent.pyautogui.keyDown").start()
        key_up = mock.patch("compiler.vision_agent.pyautogui.keyUp").start()
        run = mock.patch("compiler.vision_agent.subprocess.run").start()
        with (
            mock.patch.object(agent, "_copy_to_clipboard"),
            mock.patch.object(agent, "_clipboard_matches", return_value=True),
            mock.patch("time.sleep"),
        ):
            agent._paste_line("SELECT")
        key_down.assert_not_called()
        key_up.assert_not_called()
        self.assertEqual(run.call_count, 1)
        argv = run.call_args.args[0]
        self.assertEqual(argv[0], "osascript")
        self.assertIn('keystroke "v" using command down', argv[2])

    def test_execute_beat_type_segments_focuses_editor_once(self) -> None:
        """The pre-read already focused the editor; type_segments must skip its own."""
        agent = VisionAgent()
        self.addCleanup(mock.patch.stopall)
        mock.patch.object(agent, "_read_editor_content", return_value="").start()
        ts = mock.patch.object(agent, "type_segments", return_value=True).start()
        assess = mock.patch.object(agent, "_assess_and_maybe_repair", return_value=True).start()
        ok = agent.execute_beat(
            {"type": "type_segments", "segments": [{"text": "SELECT"}]}
        )
        self.assertTrue(ok)
        self.assertFalse(ts.call_args.kwargs["focus_editor"])
        assess.assert_called_once()

    def test_narration_target_includes_variance_and_gesture_margin(self) -> None:
        """A beat already sized to action+4s+3s is left untouched (no LLM call)."""
        builder = LessonBuilder()
        with tempfile.TemporaryDirectory() as tmp:
            payload = {
                "video_id": "video_x",
                "beats": [
                    {
                        "beat_id": "beat_002",
                        "action_type": "type_segments",
                        "action_seconds": 10.0,
                        "ok": True,
                    }
                ],
            }
            Path(tmp, "action_seconds_video_x.json").write_text(json.dumps(payload))
            # (10.0 action + 4.0 variance + 3.0 gestures) * 2.75 wps = 46 words.
            text = " ".join(["word"] * 45 + ["end."])
            self.assertEqual(builder._word_count(text), 46)
            beat = ScriptBeat(
                beat_id="beat_002", kind="demo", text=text, action=None
            )
            with (
                mock.patch.dict("os.environ", {"WSDA_ACTION_SECONDS_DIR": tmp}),
                mock.patch("compiler.lesson_builder.tracked_create") as mock_create,
            ):
                builder._size_demo_narration([beat], "video_x")
            mock_create.assert_not_called()
            self.assertEqual(beat.text, text)
            self.assertEqual(beat.planned_duration, 10.0)


def _patch_focus_guard(infos: List[Any]) -> None:
    """Patch the C22 pyobjc guard seam: ax_pyobjc focused-element reads.

    ``infos`` is consumed one dict (or None) per focused_element_info call.
    """
    queue = list(infos)

    mock.patch.object(ax_pyobjc, "app_pid_for_name", return_value=4242).start()
    mock.patch.object(ax_pyobjc, "create_application", return_value="app-el").start()
    mock.patch.object(
        ax_pyobjc, "focused_element_info",
        side_effect=lambda app_el: queue.pop(0) if queue else None,
    ).start()
    mock.patch.object(ax_pyobjc, "app_name_for_pid", return_value="OtherApp").start()


class TestC20IdempotentFocus(unittest.TestCase):
    """C20: focus is idempotent — an already-focused editor skips the focus cycle."""

    def test_focus_editor_early_exits_when_already_focused(self) -> None:
        """Two focus calls with an already-focused editor: zero dismissal, zero AX writes."""
        agent = VisionAgent()
        self.addCleanup(mock.patch.stopall)
        _patch_focus_guard(
            [
                {"element": "el", "pid": 4242, "role": "AXTextArea"},
                {"element": "el", "pid": 4242, "role": "AXTextArea"},
            ]
        )
        dismiss = mock.patch.object(agent, "_dismiss_character_viewer").start()
        frontmost = mock.patch.object(agent, "_ensure_frontmost").start()
        ax_write = mock.patch.object(
            agent, "_ensure_editor_focused_accessibility"
        ).start()
        click = mock.patch.object(agent, "find_and_click").start()
        buf = io.StringIO()
        with contextlib.redirect_stderr(buf):
            agent._focus_editor()
            agent._focus_editor()
        dismiss.assert_not_called()
        frontmost.assert_not_called()
        ax_write.assert_not_called()
        click.assert_not_called()
        self.assertEqual(buf.getvalue().count("editor already focused; skipping"), 2)

    def test_second_focus_call_skips_duplicate_dismissal(self) -> None:
        """First call runs one full cycle; the second early-exits with no dismissal."""
        agent = VisionAgent()
        self.addCleanup(mock.patch.stopall)
        _patch_focus_guard(
            [
                {"element": "el", "pid": 999, "role": "AXTerminal"},
                {"element": "el", "pid": 4242, "role": "AXTextArea"},
            ]
        )
        dismiss = mock.patch.object(agent, "_dismiss_character_viewer").start()
        frontmost = mock.patch.object(agent, "_ensure_frontmost").start()
        ax_write = mock.patch.object(
            agent, "_ensure_editor_focused_accessibility", return_value=True
        ).start()
        click = mock.patch.object(agent, "find_and_click").start()
        buf = io.StringIO()
        with contextlib.redirect_stderr(buf):
            agent._focus_editor()
            agent._focus_editor()
        dismiss.assert_called_once()
        frontmost.assert_called_once()
        ax_write.assert_called_once()
        click.assert_not_called()
        self.assertEqual(buf.getvalue().count("editor already focused; skipping"), 1)


class TestC23ClipboardInterlock(unittest.TestCase):
    """C23: paste keystroke fires only when the clipboard read-back verifies."""

    def test_match_fires_keystroke_once(self) -> None:
        """Clipboard verifies on the first try: exactly one paste keystroke."""
        agent = VisionAgent()
        self.addCleanup(mock.patch.stopall)
        copy = mock.patch.object(agent, "_copy_to_clipboard").start()
        matches = mock.patch.object(agent, "_clipboard_matches", return_value=True).start()
        keystroke = mock.patch.object(agent, "_hotkey_paste").start()
        mock.patch("time.sleep").start()
        with contextlib.redirect_stderr(io.StringIO()) as buf:
            agent._verified_clipboard_paste("SELECT 1;\n")
        keystroke.assert_called_once()
        self.assertEqual(copy.call_count, 1)
        self.assertEqual(matches.call_count, 1)
        self.assertIn("wsda-paste-verified", buf.getvalue())

    def test_persistent_mismatch_zero_keystrokes_raises(self) -> None:
        """Clipboard never verifies: zero keystrokes, PasteInterlockError."""
        from compiler.vision_agent import PasteInterlockError

        agent = VisionAgent()
        self.addCleanup(mock.patch.stopall)
        copy = mock.patch.object(agent, "_copy_to_clipboard").start()
        mock.patch.object(agent, "_clipboard_matches", return_value=False).start()
        keystroke = mock.patch.object(agent, "_hotkey_paste").start()
        mock.patch("time.sleep").start()
        with contextlib.redirect_stderr(io.StringIO()) as buf:
            with self.assertRaises(PasteInterlockError):
                agent._verified_clipboard_paste("SELECT 1;\n")
        keystroke.assert_not_called()
        self.assertEqual(copy.call_count, 3)
        self.assertIn("wsda-paste-interlock-fail", buf.getvalue())

    def test_fail_then_match_fires_once_after_retry(self) -> None:
        """First read-back mismatches, second matches: one retry, one keystroke."""
        agent = VisionAgent()
        self.addCleanup(mock.patch.stopall)
        copy = mock.patch.object(agent, "_copy_to_clipboard").start()
        matches = mock.patch.object(
            agent, "_clipboard_matches", side_effect=[False, True]
        ).start()
        keystroke = mock.patch.object(agent, "_hotkey_paste").start()
        mock.patch("time.sleep").start()
        with contextlib.redirect_stderr(io.StringIO()):
            agent._verified_clipboard_paste("SELECT 1;\n")
        keystroke.assert_called_once()
        self.assertEqual(copy.call_count, 2)
        self.assertEqual(matches.call_count, 2)

    def test_paste_line_routes_through_interlock(self) -> None:
        """_paste_line uses the interlock: no direct _hotkey_paste call."""
        agent = VisionAgent()
        self.addCleanup(mock.patch.stopall)
        interlock = mock.patch.object(agent, "_verified_clipboard_paste").start()
        keystroke = mock.patch.object(agent, "_hotkey_paste").start()
        mock.patch("time.sleep").start()
        agent._paste_line("SELECT 1;")
        interlock.assert_called_once()
        keystroke.assert_not_called()


class TestC21FocusedElementGuard(unittest.TestCase):
    """C21: the early-exit guard reads the app's focused element (C22: via AX API)."""

    def test_focused_axtextarea_in_db_browser_early_exits_without_enumeration(self):
        """Guard reports an AXTextArea focused in DB Browser: skip, never enumerate."""
        agent = VisionAgent()
        self.addCleanup(mock.patch.stopall)
        _patch_focus_guard([{"element": "el", "pid": 4242, "role": "AXTextArea"}])
        traverse = mock.patch.object(ax_pyobjc, "find_text_areas").start()
        enum = mock.patch.object(agent, "_ensure_editor_focused_accessibility").start()
        dismiss = mock.patch.object(agent, "_dismiss_character_viewer").start()
        frontmost = mock.patch.object(agent, "_ensure_frontmost").start()
        click = mock.patch.object(agent, "find_and_click").start()
        buf = io.StringIO()
        with contextlib.redirect_stderr(buf):
            agent._focus_editor()
        traverse.assert_not_called()
        enum.assert_not_called()
        dismiss.assert_not_called()
        frontmost.assert_not_called()
        click.assert_not_called()
        self.assertIn("editor already focused; skipping", buf.getvalue())
        self.assertIn("wsda-focused-editor", buf.getvalue())

    def test_guard_failure_falls_through_to_full_path(self):
        """Guard cannot confirm focus (other app focused): run the full path once."""
        agent = VisionAgent()
        self.addCleanup(mock.patch.stopall)
        _patch_focus_guard([{"element": "el", "pid": 999, "role": "AXTerminal"}])
        enum = mock.patch.object(
            agent, "_ensure_editor_focused_accessibility", return_value=True
        ).start()
        dismiss = mock.patch.object(agent, "_dismiss_character_viewer").start()
        frontmost = mock.patch.object(agent, "_ensure_frontmost").start()
        click = mock.patch.object(agent, "find_and_click").start()
        buf = io.StringIO()
        with contextlib.redirect_stderr(buf):
            agent._focus_editor()
        enum.assert_called_once()
        dismiss.assert_called_once()
        frontmost.assert_called_once()
        click.assert_not_called()
        self.assertIn("wsda-focused-other|OtherApp|AXTerminal", buf.getvalue())


class TestC25PidFallback(unittest.TestCase):
    """C25: app_pid_for_name falls back to ps when NSWorkspace is stale."""

    def _fake_app(self, name: str, pid: int):
        app = mock.MagicMock()
        app.localizedName.return_value = name
        app.processIdentifier.return_value = pid
        return app

    def test_ps_is_primary_nsworkspace_stale(self) -> None:
        """ps wins even when NSWorkspace reports a different (stale) pid."""
        self.addCleanup(mock.patch.stopall)
        mock.patch.object(
            ax_pyobjc, "_running_apps",
            return_value=[self._fake_app("DB Browser for SQLite", 111)],
        ).start()
        ps_out = "  28001 /Applications/DB Browser for SQLite.app/Contents/MacOS/DB Browser for SQLite\n"
        mock.patch(
            "subprocess.run", return_value=mock.MagicMock(stdout=ps_out),
        ).start()
        self.assertEqual(ax_pyobjc.app_pid_for_name("DB Browser for SQLite"), 28001)

    def test_ps_fallback_when_nsworkspace_stale(self) -> None:
        self.addCleanup(mock.patch.stopall)
        mock.patch.object(ax_pyobjc, "_running_apps", return_value=[]).start()
        ps_out = (
            "  100 /Applications/Safari.app/Contents/MacOS/Safari\n"
            "  27913 /Applications/DB Browser for SQLite.app/Contents/MacOS/DB Browser for SQLite\n"
            "28298 /usr/bin/osascript -e tell application \"DB Browser for SQLite\" to activate\n"
        )
        mock.patch(
            "subprocess.run",
            return_value=mock.MagicMock(stdout=ps_out),
        ).start()
        self.assertEqual(ax_pyobjc.app_pid_for_name("DB Browser for SQLite"), 27913)

    def test_not_running_returns_none(self) -> None:
        self.addCleanup(mock.patch.stopall)
        mock.patch.object(ax_pyobjc, "_running_apps", return_value=[]).start()
        mock.patch(
            "subprocess.run",
            return_value=mock.MagicMock(stdout="  100 /Applications/Safari.app/Contents/MacOS/Safari\n"),
        ).start()
        self.assertIsNone(ax_pyobjc.app_pid_for_name("DB Browser for SQLite"))


class TestC22PyobjcTraversal(unittest.TestCase):
    """C22: the direct AX API traversal finds the top-most AXTextArea."""

    def test_finds_top_most_text_area_and_focuses_it(self) -> None:
        """Fake AX tree: top-most (smallest y) AXTextArea receives AXFocused."""
        self.addCleanup(mock.patch.stopall)
        tree = {
            "app": {"AXWindows": ["w"]},
            "w": {"AXRole": "AXWindow", "AXChildren": ["toolbar", "split"]},
            "toolbar": {"AXRole": "AXGroup", "AXChildren": []},
            "split": {"AXRole": "AXGroup", "AXChildren": ["ta_editor", "results"]},
            "results": {"AXRole": "AXScrollArea", "AXChildren": ["ta_results"]},
            "ta_editor": {"AXRole": "AXTextArea"},
            "ta_results": {"AXRole": "AXTextArea"},
        }
        y_positions = {"ta_editor": 180.0, "ta_results": 724.0}
        mock.patch.object(
            ax_pyobjc, "copy_attribute",
            side_effect=lambda el, name: tree.get(el, {}).get(name),
        ).start()
        mock.patch.object(
            ax_pyobjc, "element_position",
            side_effect=lambda el: (0.0, y_positions[el]) if el in y_positions else None,
        ).start()
        set_focused = mock.patch.object(ax_pyobjc, "set_focused").start()
        found = ax_pyobjc.find_text_areas("app")
        self.assertEqual(
            sorted(el for el, _ in found), ["ta_editor", "ta_results"]
        )
        top_el, top_y = min(found, key=lambda t: t[1] if t[1] is not None else 1e9)
        self.assertEqual((top_el, top_y), ("ta_editor", 180.0))

        # Through the vision-agent path: the top-most area is what gets focused.
        agent = VisionAgent()
        mock.patch.object(ax_pyobjc, "app_pid_for_name", return_value=7).start()
        mock.patch.object(ax_pyobjc, "create_application", return_value="app").start()
        with contextlib.redirect_stderr(io.StringIO()):
            ok = agent._ensure_editor_focused_accessibility()
        self.assertTrue(ok)
        set_focused.assert_called_once_with("ta_editor")

    def test_traversal_error_and_empty_are_distinct_markers(self):
        """AxCallError -> wsda-pyobjc-error; empty tree -> wsda-no-text-area."""
        self.addCleanup(mock.patch.stopall)
        agent = VisionAgent()
        mock.patch.object(ax_pyobjc, "app_pid_for_name", return_value=7).start()
        mock.patch.object(ax_pyobjc, "create_application", return_value="app").start()
        mock.patch.object(agent, "_log_ax_failure_context").start()
        mock.patch("time.sleep").start()
        err = ax_pyobjc.AxCallError("copy AXWindows", -25204)
        buf = io.StringIO()
        with mock.patch.object(ax_pyobjc, "find_text_areas", side_effect=err):
            with contextlib.redirect_stderr(buf):
                self.assertFalse(agent._ensure_editor_focused_accessibility())
        self.assertIn("wsda-pyobjc-error", buf.getvalue())
        buf2 = io.StringIO()
        with mock.patch.object(ax_pyobjc, "find_text_areas", return_value=[]):
            with contextlib.redirect_stderr(buf2):
                self.assertFalse(agent._ensure_editor_focused_accessibility())
        self.assertIn("wsda-no-text-area", buf2.getvalue())


class TestC22PyobjcRetry(unittest.TestCase):
    """C22: the AX traversal retries before VLM fallback (C21 discipline)."""

    def _patch_full_path(self, find_side_effect) -> Any:
        _patch_focus_guard([{"element": "el", "pid": 999, "role": "AXTerminal"}])
        find = mock.patch.object(
            ax_pyobjc, "find_text_areas", side_effect=find_side_effect
        ).start()
        mock.patch.object(ax_pyobjc, "set_focused").start()
        return find

    def test_two_errors_then_success_three_attempts_no_vlm(self) -> None:
        """First two traversals raise, third focuses: 3 attempts, no VLM clicks."""
        agent = VisionAgent()
        self.addCleanup(mock.patch.stopall)
        err = ax_pyobjc.AxCallError("copy AXWindows", -25204)
        find = self._patch_full_path([err, err, [("ta", 50.0)]])
        mock.patch.object(agent, "_log_ax_failure_context").start()
        mock.patch.object(agent, "_dismiss_character_viewer").start()
        mock.patch.object(agent, "_ensure_frontmost").start()
        click = mock.patch.object(agent, "find_and_click").start()
        sleep = mock.patch("time.sleep").start()
        with contextlib.redirect_stderr(io.StringIO()):
            agent._focus_editor()
        self.assertEqual(find.call_count, 3)
        click.assert_not_called()
        self.assertEqual(sleep.call_count, 2)

    def test_zero_text_areas_after_retries_vlm_once(self):
        """Traversal succeeds but finds nothing: VLM fallback exactly once."""
        agent = VisionAgent()
        self.addCleanup(mock.patch.stopall)
        # Round 1 (fast path) and round 2 (post-VLM hard guarantee) all empty.
        find = self._patch_full_path([[], [], [], [], [], []])
        mock.patch.object(agent, "_log_ax_failure_context").start()
        mock.patch.object(agent, "_dismiss_character_viewer").start()
        mock.patch.object(agent, "_ensure_frontmost").start()
        click = mock.patch.object(agent, "find_and_click").start()
        mock.patch("time.sleep").start()
        with contextlib.redirect_stderr(io.StringIO()):
            agent._focus_editor()
        self.assertEqual(click.call_count, 2)
        self.assertEqual(find.call_count, 6)

    def test_pyobjc_raises_vlm_fallback_exactly_once(self):
        """Every traversal raises: VLM fallback exactly once."""
        agent = VisionAgent()
        self.addCleanup(mock.patch.stopall)
        err = ax_pyobjc.AxCallError("copy AXWindows", -25204)
        find = self._patch_full_effect([err] * 6)
        mock.patch.object(agent, "_log_ax_failure_context").start()
        mock.patch.object(agent, "_dismiss_character_viewer").start()
        mock.patch.object(agent, "_ensure_frontmost").start()
        click = mock.patch.object(agent, "find_and_click").start()
        mock.patch("time.sleep").start()
        with contextlib.redirect_stderr(io.StringIO()):
            agent._focus_editor()
        self.assertEqual(click.call_count, 2)
        self.assertEqual(find.call_count, 6)

    def _patch_full_effect(self, effects: List[Any]) -> Any:
        _patch_focus_guard([{"element": "el", "pid": 999, "role": "AXTerminal"}])
        return mock.patch.object(
            ax_pyobjc, "find_text_areas", side_effect=list(effects)
        ).start()


class TestC24EditorAutoClear(unittest.TestCase):
    """C24: run-start editor hygiene — clean proceeds, dirty self-clears."""

    def test_clean_editor_zero_keystrokes_proceeds(self) -> None:
        """A clean editor logs wsda-editor-clean and never fires a keystroke."""
        agent = VisionAgent()
        self.addCleanup(mock.patch.stopall)
        mock.patch.object(agent, "_editor_text_length", return_value=0).start()
        clear = mock.patch.object(agent, "_clear_editor_once").start()
        run = mock.patch("compiler.vision_agent.subprocess.run").start()
        buf = io.StringIO()
        with contextlib.redirect_stderr(buf):
            agent.ensure_editor_clean()
        clear.assert_not_called()
        run.assert_not_called()
        self.assertIn("wsda-editor-clean", buf.getvalue())

    def test_dirty_editor_one_clear_proceeds_when_re_read_zero(self) -> None:
        """Dirty editor: exactly one clear sequence; re-read 0 proceeds."""
        agent = VisionAgent()
        self.addCleanup(mock.patch.stopall)
        length = mock.patch.object(
            agent, "_editor_text_length", side_effect=[500, 0]
        ).start()
        mock.patch.object(agent, "_ensure_frontmost").start()
        mock.patch.object(
            agent, "_ensure_editor_focused_accessibility", return_value=True
        ).start()
        run = mock.patch("compiler.vision_agent.subprocess.run").start()
        mock.patch("time.sleep").start()
        buf = io.StringIO()
        with contextlib.redirect_stderr(buf):
            agent.ensure_editor_clean()
        self.assertEqual(length.call_count, 2)
        self.assertEqual(run.call_count, 1)  # one osascript: cmd+a + key 51
        argv = run.call_args.args[0]
        self.assertEqual(argv[0], "osascript")
        self.assertIn('keystroke "a" using command down', argv[2])
        self.assertIn("key code 51", argv[2])
        self.assertIn("wsda-editor-dirty:500chars", buf.getvalue())
        self.assertIn("wsda-editor-cleared:500chars", buf.getvalue())

    def test_dirty_editor_retry_once_then_halt_marker(self) -> None:
        """First re-read still non-zero: one retry; still dirty => halt."""
        agent = VisionAgent()
        self.addCleanup(mock.patch.stopall)
        length = mock.patch.object(
            agent, "_editor_text_length", side_effect=[500, 300, 100]
        ).start()
        mock.patch.object(agent, "_ensure_frontmost").start()
        mock.patch.object(
            agent, "_ensure_editor_focused_accessibility", return_value=True
        ).start()
        run = mock.patch("compiler.vision_agent.subprocess.run").start()
        mock.patch("time.sleep").start()
        buf = io.StringIO()
        with contextlib.redirect_stderr(buf):
            with self.assertRaises(RuntimeError) as ctx:
                agent.ensure_editor_clean()
        self.assertIn("wsda-editor-clear-fail", str(ctx.exception))
        self.assertEqual(run.call_count, 2)  # clear tried exactly twice
        self.assertEqual(length.call_count, 3)


class TestC25AppReadinessWait(unittest.TestCase):
    """C25: app-readiness polling after launch; halt on timeout."""

    def _fake_clock(self) -> List[Any]:
        clock = [0.0]
        mock.patch("time.time", side_effect=lambda: clock[0]).start()
        mock.patch("time.sleep", side_effect=lambda s: clock.__setitem__(0, clock[0] + s)).start()
        return clock

    def test_poll_succeeds_on_attempt_n(self) -> None:
        """Transient AX failures then success: proceeds, marker logged with elapsed."""
        agent = VisionAgent()
        self.addCleanup(mock.patch.stopall)
        self._fake_clock()
        pid_calls = [0]

        def _pid(name):
            pid_calls[0] += 1
            return None if pid_calls[0] == 1 else 4242

        pid = mock.patch.object(ax_pyobjc, "app_pid_for_name", side_effect=_pid).start()
        mock.patch.object(ax_pyobjc, "create_application", return_value="app").start()
        copy = mock.patch.object(
            ax_pyobjc, "copy_attribute",
            side_effect=[
                ax_pyobjc.AxCallError("copy AXWindows", -25204),
                ["w"],
                ["w"],
            ],
        ).start()
        find = mock.patch.object(
            ax_pyobjc, "find_text_areas", side_effect=[[("ta", 5.0)]]
        ).start()
        run = mock.patch("compiler.vision_agent.subprocess.run").start()
        buf = io.StringIO()
        with contextlib.redirect_stderr(buf):
            wait_for_app_readiness(agent, db_path="/tmp/x.db")
        self.assertIn("wsda-app-ready:1.0s", buf.getvalue())
        self.assertEqual(copy.call_count, 2)
        self.assertEqual(find.call_count, 1)
        # Launch path fired because the first pid lookup returned None.
        self.assertTrue(
            any(c.args[0][:2] == ["open", "-a"] for c in run.call_args_list)
        )

    def test_all_polls_fail_halt_zero_beats(self) -> None:
        """Every poll fails: RuntimeError with wsda-app-not-ready."""
        agent = VisionAgent()
        self.addCleanup(mock.patch.stopall)
        self._fake_clock()
        mock.patch.object(ax_pyobjc, "app_pid_for_name", return_value=4242).start()
        mock.patch.object(ax_pyobjc, "create_application", return_value="app").start()
        copy = mock.patch.object(
            ax_pyobjc, "copy_attribute",
            side_effect=ax_pyobjc.AxCallError("copy AXWindows", -25204),
        ).start()
        buf = io.StringIO()
        with contextlib.redirect_stderr(buf):
            with self.assertRaises(RuntimeError) as ctx:
                wait_for_app_readiness(agent, db_path="/tmp/x.db", timeout=3.0)
        self.assertIn("wsda-app-not-ready", str(ctx.exception))
        self.assertGreaterEqual(copy.call_count, 3)

    def test_no_text_areas_presses_execute_tab(self) -> None:
        """Windows up but no AXTextArea: opens DB once, presses Execute SQL
        radio, then succeeds."""
        agent = VisionAgent()
        self.addCleanup(mock.patch.stopall)
        self._fake_clock()
        mock.patch.object(ax_pyobjc, "app_pid_for_name", return_value=4242).start()
        mock.patch.object(ax_pyobjc, "create_application", return_value="app").start()
        mock.patch.object(ax_pyobjc, "copy_attribute", return_value=["w"]).start()
        find = mock.patch.object(
            ax_pyobjc, "find_text_areas", side_effect=[[], [], [("ta", 5.0)]]
        ).start()
        press = mock.patch.object(ax_pyobjc, "press_execute_tab", return_value=True).start()
        run = mock.patch("compiler.vision_agent.subprocess.run").start()
        buf = io.StringIO()
        with contextlib.redirect_stderr(buf):
            wait_for_app_readiness(agent, db_path="/tmp/x.db")
        # Poll 1: no areas -> one-time DB open (no press yet). Poll 2: press.
        # Poll 3: enumeration succeeds.
        open_calls = [c for c in run.call_args_list if c.args[0][:2] == ["open", "-a"]]
        self.assertEqual(len(open_calls), 1)
        self.assertEqual(press.call_count, 1)
        self.assertEqual(find.call_count, 3)
        self.assertIn("wsda-db-open", buf.getvalue())
        self.assertIn("wsda-execute-tab-pressed", buf.getvalue())
        self.assertIn("wsda-app-ready:2.0s", buf.getvalue())

    def test_no_text_areas_no_db_path_presses_tab(self) -> None:
        """No db_path: skips the DB-open branch, presses the radio directly."""
        agent = VisionAgent()
        self.addCleanup(mock.patch.stopall)
        self._fake_clock()
        mock.patch.object(ax_pyobjc, "app_pid_for_name", return_value=4242).start()
        mock.patch.object(ax_pyobjc, "create_application", return_value="app").start()
        mock.patch.object(ax_pyobjc, "copy_attribute", return_value=["w"]).start()
        mock.patch.object(
            ax_pyobjc, "find_text_areas", side_effect=[[], [("ta", 5.0)]]
        ).start()
        press = mock.patch.object(ax_pyobjc, "press_execute_tab", return_value=True).start()
        run = mock.patch("compiler.vision_agent.subprocess.run").start()
        buf = io.StringIO()
        with contextlib.redirect_stderr(buf):
            wait_for_app_readiness(agent, db_path=None)
        self.assertEqual(press.call_count, 1)
        self.assertFalse(
            any(c.args[0][:2] == ["open", "-a"] for c in run.call_args_list)
        )
        self.assertIn("wsda-app-ready:1.0s", buf.getvalue())


class TestC26StopInvariant(unittest.TestCase):
    """C26: recorder stop is anchored to measured audio end + tail, HARD.

    Fake clock: work pending at the deadline must not delay the stop; the
    canonical gate still runs (post-stop) and its result is still reported.
    """

    class _Clock:
        def __init__(self, t: float = 100.0):
            self.t = t

        def now(self) -> float:
            return self.t

        def sleep(self, s: float) -> None:
            self.t += s

    class _Recorder:
        def __init__(self, clock: "TestC26StopInvariant._Clock"):
            self.clock = clock
            self.stopped_at: Optional[float] = None

        def stop(self) -> None:
            self.stopped_at = self.clock.now()

    def test_stop_at_deadline_with_work_pending(self) -> None:
        clock = self._Clock(100.0)
        audio_end = 110.0
        ctl = BeatRecordingStop(audio_end, now_fn=clock.now)
        self.assertEqual(ctl.deadline, audio_end + RECORDER_TAIL_SECONDS)
        clock.t = ctl.deadline  # at the deadline, deferred work still pending
        rec = self._Recorder(clock)
        events: List[Any] = []
        results = ctl.stop_recorder(
            rec,
            pending_work=[
                lambda: events.append(("gate", clock.now())) or True,
                lambda: events.append(("post", clock.now())) or None,
            ],
        )
        # HARD invariant: stop at audio_actual_end + 1.0s regardless of work.
        self.assertEqual(rec.stopped_at, audio_end + RECORDER_TAIL_SECONDS)
        # Deferred work executed strictly post-stop, in order; gate result kept.
        self.assertEqual([e[0] for e in events], ["gate", "post"])
        self.assertTrue(all(e[1] >= rec.stopped_at for e in events))
        self.assertEqual(results, [True, None])

    def test_gate_still_evaluated_when_stop_is_late(self) -> None:
        clock = self._Clock(100.0)
        ctl = BeatRecordingStop(110.0, now_fn=clock.now)
        clock.t = 113.5  # 2.5s past the deadline, work pending
        rec = self._Recorder(clock)
        order: List[str] = []
        results = ctl.stop_recorder(
            rec, pending_work=[lambda: order.append("gate") or True]
        )
        self.assertEqual(rec.stopped_at, 113.5)
        self.assertTrue(ctl.stop_flagged)  # lateness flags the beat
        self.assertEqual(order, ["gate"])  # gate still evaluated, post-stop
        self.assertEqual(results, [True])

    def test_wait_until_deadline_never_overshoots(self) -> None:
        clock = self._Clock(100.0)
        ctl = BeatRecordingStop(110.25, now_fn=clock.now)
        choreo_steps: List[float] = []
        ctl.wait_until_deadline(
            clock.sleep,
            on_interval=lambda step: choreo_steps.append(step),
        )
        self.assertAlmostEqual(clock.t, 111.25, places=6)
        self.assertAlmostEqual(sum(choreo_steps), 11.25, places=6)
        self.assertLessEqual(max(choreo_steps, default=0.0), 0.1)


class TestC27WallClockWriter(unittest.TestCase):
    """C27: the SCK writer is wall-clock anchored — output length equals wall
    span at any delivery rate; ticks with no new delivery duplicate the latest
    frame."""

    def _run_writer(
        self, output_path: Path, feed_schedule: List[Any]
    ) -> Any:
        """Start the writer loop; feed_schedule is [(time_offset, count), ...].
        Returns (recorder, writer_thread); caller sets stop_event and joins."""
        rec = _ScreenCaptureKitRecorder(str(output_path), fps=10, app_name="")
        generation = [0]

        def _fake_bgr(_sample):
            generation[0] += 1
            return np.full((80, 128, 3), generation[0] % 200 + 1, dtype=np.uint8)

        rec._sample_buffer_to_bgr = _fake_bgr  # type: ignore[assignment]
        t0 = time.monotonic()

        def _feeder():
            for offset, count in feed_schedule:
                delay = t0 + offset - time.monotonic()
                if delay > 0:
                    time.sleep(delay)
                with rec._lock:
                    rec._samples[:] = [object() for _ in range(count)]

        threading.Thread(target=_feeder, daemon=True).start()
        wt = threading.Thread(target=rec._writer_loop, daemon=True)
        wt.start()
        return rec, wt

    def test_low_delivery_and_stall_keep_wall_clock_length(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "wc.mp4"
            # Deliver at 2fps with a 3s total stall in [1.0, 4.0); stop ~4.05s.
            rec, wt = self._run_writer(
                out, [(0.0, 1), (0.5, 1), (4.0, 1)]
            )
            time.sleep(4.05)
            rec._stop_event.set()
            wt.join(timeout=10)
            if rec._writer is not None:
                rec._writer.release()
            summary = rec.delivery_summary
            cap = cv2.VideoCapture(str(out))
            frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            self.assertIsNotNone(summary)
            # Invariant: written frames == written span x fps exactly.
            self.assertEqual(frames, summary["expected_written"])
            # 2fps delivery over a ~4s span: only a few deliveries, never 40.
            self.assertLessEqual(summary["frames_delivered"], 8)
            # Duplication filled the gap: frame count is wall-anchored.
            self.assertGreaterEqual(frames, 35)
            self.assertLessEqual(frames, 42)

            # Stall-window frames are pixel-identical duplicates of the last
            # pre-stall frame (ticks 1.0s..3.7s => indices 9..37 at 10fps).
            cap = cv2.VideoCapture(str(out))
            ok, prev = cap.read()
            idx = 0
            duplicates = 0
            while ok:
                ok, frame = cap.read()
                if not ok:
                    break
                idx += 1
                if 9 <= idx <= 37 and np.array_equal(prev, frame):
                    duplicates += 1
                prev = frame
            cap.release()
            self.assertEqual(duplicates, 29)


class TestC27DeliveryFloor(unittest.TestCase):
    """C27: delivered fps below the floor flags wsda-low-delivery; at/above
    (or unknown, e.g. MSS fallback) it does not."""

    def test_below_floor_flags(self) -> None:
        marker = delivery_floor_breach(DELIVERY_FLOOR_FPS - 0.1)
        self.assertIsNotNone(marker)
        self.assertIn("wsda-low-delivery", marker)

    def test_at_or_above_floor_passes(self) -> None:
        self.assertIsNone(delivery_floor_breach(DELIVERY_FLOOR_FPS))
        self.assertIsNone(delivery_floor_breach(DELIVERY_FLOOR_FPS + 2.5))
        self.assertIsNone(delivery_floor_breach(None))


class TestC27TeardownSettle(unittest.TestCase):
    """C27: a new SCK stream waits out the teardown settle window."""

    def test_start_waits_after_recent_teardown(self) -> None:
        rec = _ScreenCaptureKitRecorder("/tmp/wsda_settle_test.mp4", fps=10, app_name="")
        sleeps: List[float] = []
        # A teardown just happened: quiet time ~0, so start must settle.
        _ScreenCaptureKitRecorder._last_teardown_mono = time.monotonic()
        try:
            with mock.patch.object(
                discovery_module.time, "sleep", side_effect=lambda s: sleeps.append(s)
            ):
                with mock.patch.object(rec, "_start_stream", return_value=True):
                    rec.start()
            rec._stop_event.set()
            if rec._thread is not None:
                rec._thread.join(timeout=3)
        finally:
            _ScreenCaptureKitRecorder._last_teardown_mono = 0.0
        self.assertTrue(sleeps)
        self.assertGreaterEqual(
            sleeps[0], discovery_module.SCK_TEARDOWN_SETTLE_SECONDS - 0.6
        )


class TestC28CaptureWarmup(unittest.TestCase):
    """C28: the once-per-run sacrificial warmup capture runs BEFORE beat_001's
    recorder; a warmup failure halts the run with the wsda-capture-warmup-fail
    marker and zero beats executed."""

    def _discovery(self, events: List[str]) -> EndStateDiscovery:
        d = EndStateDiscovery(objective="test", application="db_browser_sqlite")
        d._launch_app = mock.Mock()
        d._auto_fit_columns = mock.Mock()
        d._assert_stage_resources = mock.Mock(return_value={"ok": True})
        d._prepare_opening_state = mock.Mock()
        d._wait_for_visual_stability = mock.Mock(return_value=True)
        d._capture_warmup = mock.Mock(side_effect=lambda run_id: events.append("warmup"))
        return d

    def _run(self, d: EndStateDiscovery, events: List[str]) -> Any:
        beat = ScriptBeat(
            beat_id="beat_001",
            kind="opening",
            text="In this video we introduce the course.",
            action={"type": "wait", "duration": 1.5},
        )
        recorder = mock.MagicMock()
        recorder.delivery_summary = None
        recorder._fallback = None
        recorder.first_frame_time.return_value = None

        def factory(*args: Any, **kwargs: Any) -> Any:
            events.append("recorder")
            return recorder

        agent_cls = mock.MagicMock()
        agent_cls.return_value._read_editor_content.return_value = ""

        with mock.patch.multiple(
            discovery_module,
            _find_db_browser=mock.Mock(return_value="/tmp/DB Browser.app"),
            _ensure_sample_db=mock.Mock(return_value=Path("/tmp/x.db")),
            VisionAgent=agent_cls,
            TTSGenerator=mock.Mock(side_effect=RuntimeError("no tts")),
            _ScreenCaptureKitRecorder=mock.Mock(side_effect=factory),
            _clip_has_off_app_interval=mock.Mock(return_value=False),
            _capture_screenshot=mock.Mock(
                return_value=("", 100, 100, 1.0, mock.MagicMock(), b"bytes")
            ),
        ):
            return d._execute_beats_with_agent([beat], "summary", False)

    def test_warmup_once_and_before_first_beat_recorder(self) -> None:
        events: List[str] = []
        d = self._discovery(events)
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            result = self._run(d, events)
        self.assertTrue(result.success, err.getvalue())
        # Warmup ran exactly once, strictly before the beat recorder existed.
        self.assertEqual(d._capture_warmup.call_count, 1)
        self.assertEqual(events, ["warmup", "recorder"])
        self.assertIn("wsda-capture-warmup:ok", err.getvalue())

    def test_warmup_exception_halts_zero_beats(self) -> None:
        events: List[str] = []
        d = self._discovery(events)
        d._capture_warmup = mock.Mock(side_effect=RuntimeError("boom"))
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            result = self._run(d, events)
        self.assertFalse(result.success)
        self.assertEqual(result.attempts, 0)
        self.assertEqual(d._capture_warmup.call_count, 1)
        self.assertEqual(events, [])  # no beat recorder was ever constructed
        self.assertIn("wsda-capture-warmup-fail", err.getvalue())


class TestC29GroundingReadsRecorderFrame(unittest.TestCase):
    """C29: while recording, VLM grounding reads the recorder's latest frame
    and NEVER invokes an independent capture API; with the provider unset
    (dry-run) the live-capture path stays."""

    def test_screenshot_uses_recorder_frame(self) -> None:
        from PIL import Image

        rec = _ScreenCaptureKitRecorder("/tmp/wsda_c29_frame.mp4", fps=10, app_name="")
        rec.app_name = "Fake App"  # provider composites via window bounds
        rec._latest_frame = np.full((800, 1280, 3), 200, np.uint8)
        agent = VisionAgent(model="test-model", output_dir="/tmp")
        agent.set_frame_provider(rec.latest_frame_provider())

        calls: List[int] = []

        def fake_capture() -> Image.Image:
            calls.append(1)
            return Image.new("RGB", (10, 10))

        bounds = {"x": 0.0, "y": 56.0, "w": 1470.0, "h": 900.0}
        with mock.patch.object(VisionAgent, "_capture_screen", staticmethod(fake_capture)), \
             mock.patch.object(discovery_module, "_window_bounds", return_value=bounds), \
             mock.patch.object(discovery_module.pyautogui, "size", return_value=(1470, 956)):
            agent.screenshot()
        self.assertEqual(calls, [])  # zero independent captures during recording
        # C30: composited full-screen canvas, window at its real origin.
        self.assertEqual(agent.last_raw_image.size, (1280, 832))
        self.assertEqual(agent.last_raw_image.getpixel((10, 10)), (200, 200, 200))

        # Provider cleared (recorder stopped / dry-run): live capture returns.
        agent.set_frame_provider(None)
        with mock.patch.object(VisionAgent, "_capture_screen", staticmethod(fake_capture)):
            agent.screenshot()
        self.assertEqual(len(calls), 1)
        self.assertEqual(agent.last_raw_image.size, (10, 10))

    def test_no_frame_yet_falls_back_to_live_capture(self) -> None:
        from PIL import Image

        rec = _ScreenCaptureKitRecorder("/tmp/wsda_c29_frame2.mp4", fps=10, app_name="")
        agent = VisionAgent(model="test-model", output_dir="/tmp")
        agent.set_frame_provider(rec.latest_frame_provider())  # no frame written yet

        calls: List[int] = []

        def fake_capture() -> Image.Image:
            calls.append(1)
            return Image.new("RGB", (10, 10))

        with mock.patch.object(VisionAgent, "_capture_screen", staticmethod(fake_capture)):
            agent.screenshot()
        self.assertEqual(len(calls), 1)


class TestC29CodecFallback(unittest.TestCase):
    """C29: writer prefers avc1 (hardware H.264); if it cannot open, falls
    back to mp4v with a logged marker."""

    def test_avc1_used_when_available(self) -> None:
        tmp = Path(tempfile.mkdtemp()) / "c29_avc1.mp4"
        opened = mock.MagicMock()
        opened.isOpened.return_value = True
        with mock.patch.object(
            discovery_module.cv2, "VideoWriter", return_value=opened
        ) as vw:
            err = io.StringIO()
            with contextlib.redirect_stderr(err):
                writer = _open_video_writer(tmp, 10, (1280, 800))
        self.assertIs(writer, opened)
        vw.assert_called_once()
        fourcc_arg = vw.call_args[0][1]
        self.assertEqual(fourcc_arg, cv2.VideoWriter_fourcc(*"avc1"))
        self.assertNotIn("wsda-codec:avc1-unavailable", err.getvalue())

    def test_avc1_failure_falls_back_to_mp4v_with_marker(self) -> None:
        tmp = Path(tempfile.mkdtemp()) / "c29_mp4v.mp4"
        refused = mock.MagicMock()
        refused.isOpened.return_value = False
        opened = mock.MagicMock()
        opened.isOpened.return_value = True

        def fake_writer(path: str, fourcc: int, fps: int, size: Any) -> Any:
            if fourcc == cv2.VideoWriter_fourcc(*"avc1"):
                return refused
            return opened

        with mock.patch.object(
            discovery_module.cv2, "VideoWriter", side_effect=fake_writer
        ) as vw:
            err = io.StringIO()
            with contextlib.redirect_stderr(err):
                writer = _open_video_writer(tmp, 10, (1280, 800))
        self.assertIs(writer, opened)
        self.assertEqual(vw.call_count, 2)
        self.assertIn("wsda-codec:avc1-unavailable", err.getvalue())


class TestC29EncodeOffDelivery(unittest.TestCase):
    """C29: the delivery path (latest-sample swap under the lock) never blocks
    on encode — a catastrophically slow encoder leaves feeder latency tiny."""

    def test_slow_encoder_delivery_unaffected(self) -> None:
        rec = _ScreenCaptureKitRecorder("/tmp/wsda_c29_slow.mp4", fps=10, app_name="")
        rec._sample_buffer_to_bgr = lambda s: np.zeros((64, 64, 3), np.uint8)

        class SlowWriter:
            def __init__(self) -> None:
                self.writes = 0

            def isOpened(self) -> bool:
                return True

            def write(self, frame: np.ndarray) -> None:
                self.writes += 1
                time.sleep(0.4)  # 4x slower than the 0.1s tick budget

            def release(self) -> None:
                pass

        slow = SlowWriter()
        feeder_ms: List[float] = []

        def feeder() -> None:
            end = time.monotonic() + 3.0
            while time.monotonic() < end:
                t0 = time.monotonic()
                with rec._lock:
                    rec._samples[:] = [object()]
                feeder_ms.append((time.monotonic() - t0) * 1000.0)
                time.sleep(0.05)

        with mock.patch.object(
            discovery_module, "_open_video_writer", return_value=slow
        ):
            wt = threading.Thread(target=rec._writer_loop, daemon=True)
            wt.start()
            ft = threading.Thread(target=feeder, daemon=True)
            ft.start()
            ft.join(timeout=10)
            rec._stop_event.set()
            wt.join(timeout=10)

        self.assertFalse(ft.is_alive())
        self.assertGreaterEqual(slow.writes, 1)
        self.assertIsNotNone(rec.delivery_summary)
        # Encode (400ms/write, on the writer thread) never stalled the
        # delivery path: worst lock acquisition is orders of magnitude below.
        self.assertLess(max(feeder_ms), 50.0)


class TestC30ProviderFrameGeometry(unittest.TestCase):
    """C30: window-only provider frames are composited onto a full-screen
    canvas at the window's real CGWindowList origin. Grounded logical
    coordinates then match the screencapture full-screen path exactly —
    uniform scale on both axes, menu-bar offset included."""

    def test_known_origin_point_maps_like_full_screen(self) -> None:
        frame = np.zeros((800, 1280, 3), np.uint8)
        frame[100, 200] = (255, 255, 255)  # marker at window-frame px (200, 100)
        bounds = {"x": 100.0, "y": 150.0, "w": 800.0, "h": 500.0}  # Quartz pts
        screen_w, screen_h = 1440.0, 900.0
        canvas = _composite_window_frame(frame, bounds, screen_w, screen_h)
        ch, cw = canvas.shape[:2]
        scale = 1280 / 800  # 1.6 px per point
        # Canvas is the FULL SCREEN at uniform scale: aspect preserved.
        self.assertEqual((cw, ch), (int(round(screen_w * scale)), int(round(screen_h * scale))))
        self.assertAlmostEqual(cw / ch, screen_w / screen_h, places=3)
        # Window origin: x=100pt -> 160px; top-left y = 900-150-500 = 250pt -> 400px.
        ox, oy = 160, 400
        self.assertEqual(tuple(canvas[oy + 100, ox + 200]), (255, 255, 255))
        # Menu-bar region exists above the window origin.
        self.assertGreater(oy, 0)
        # Uniform scale maps both axes exactly (what screenshot() assumes);
        # any logical point — menu bar, desktop, or in-window — round-trips.
        scale_to_logical = screen_w / cw
        for lx, ly in [(720.0, 10.0), (500.0, 450.0), (100.0, 880.0), (1400.0, 850.0)]:
            ax, ay = lx / scale_to_logical, ly / scale_to_logical
            self.assertAlmostEqual(ax * scale_to_logical, lx, places=6)
            self.assertAlmostEqual(ay * scale_to_logical, ly, places=6)

    def test_window_only_frame_without_composite_misses_by_tens_of_px(self) -> None:
        # Pin the C29 regression: raw window frame (1.6:1) vs true screen
        # aspect (~1.54) makes the width-derived scale wrong for Y by enough
        # to miss a toolbar button; compositing restores uniformity.
        screen_w, screen_h = 1470.0, 956.0
        bad_scale = screen_w / 1280  # used for BOTH axes pre-C30
        true_y_scale = screen_h / 800
        err_px = 700 * abs(bad_scale - true_y_scale)
        self.assertGreater(err_px, 20)  # the observed ~35px miss class
        bounds = {"x": 0.0, "y": 56.0, "w": 1470.0, "h": 900.0}
        canvas = _composite_window_frame(
            np.zeros((800, 1280, 3), np.uint8), bounds, screen_w, screen_h
        )
        cw, ch = canvas.shape[1], canvas.shape[0]
        # Uniform up to integer pixel rounding (<=2px on ~830px height).
        self.assertAlmostEqual(screen_w / cw, screen_h / ch, delta=0.002)


class TestC30FreshResultsRequired(unittest.TestCase):
    """C30: run_query success requires a results-pane STATE CHANGE, and a
    passing cheap check never overrides a failing VLM assess."""

    def _agent(self) -> VisionAgent:
        agent = VisionAgent(model="test-model", output_dir="/tmp")
        agent._run_button_point = (139, 136)  # primed: no VLM locate
        agent._ensure_frontmost = mock.Mock()
        agent._read_editor_content = mock.Mock(return_value="SELECT 1;")
        agent._extract_uncommented_sql = mock.Mock(return_value=("SELECT 1;", None))
        agent._verify_statement_isolation = mock.Mock(return_value=True)
        agent._result_pane_shows_error = mock.Mock(return_value=False)
        agent._results_pane_snapshot = mock.Mock(return_value={"phash": (1,) * 64})
        return agent

    def test_identical_pane_snapshots_fail(self) -> None:
        agent = self._agent()
        agent._results_pane_changed = lambda before, timeout=4.0: False
        with mock.patch.object(vision_agent_module.time, "sleep", return_value=None):
            self.assertFalse(agent.run_query())

    def test_changed_pane_snapshot_passes(self) -> None:
        agent = self._agent()
        agent._results_pane_changed = lambda before, timeout=4.0: True
        with mock.patch.object(vision_agent_module.time, "sleep", return_value=None):
            self.assertTrue(agent.run_query())

    def test_cheap_pass_assess_fail_is_failure(self) -> None:
        agent = self._agent()
        agent._cheap_checks_ok = mock.Mock(return_value=(True, ""))
        agent.assess_screen_state = mock.Mock(
            return_value={
                "objective": "o",
                "intended_state": "i",
                "description": "d",
                "serves_objective": False,
                "anomaly": "wrong state",
                "anomaly_class": "wrong app state",
                "corrective_action": None,
            }
        )
        self.assertFalse(agent._assess_and_maybe_repair("o", "i"))


class TestC30FinalReadResolvesFresh(unittest.TestCase):
    """C30: a single empty final read never declares the editor empty — one
    re-resolve + retry via the standard focus path recovers real content."""

    def test_stale_empty_read_recovers_via_reresolve(self) -> None:
        agent = mock.Mock()
        agent._read_editor_content.side_effect = ["", "SELECT 1;"]
        content = _final_editor_read(agent)
        self.assertEqual(content, "SELECT 1;")
        self.assertEqual(agent._focus_editor.call_count, 2)

    def test_populated_first_read_skips_retry(self) -> None:
        agent = mock.Mock()
        agent._read_editor_content.return_value = "SELECT 1;"
        self.assertEqual(_final_editor_read(agent), "SELECT 1;")
        self.assertEqual(agent._focus_editor.call_count, 1)


class TestC31BoundsValidation(unittest.TestCase):
    """C31: _window_bounds accepts only a candidate owned by the target pid
    that passes sanity (min area, wide aspect, layer 0), largest preferred.
    Otherwise None + 'wsda-bounds-rejected' (all candidate rects logged) and
    the caller falls back to the live-capture path."""

    _PID = 4321

    def _entry(self, pid, rect, layer=0):
        x, y, w, h = rect
        return {
            "kCGWindowOwnerPID": pid,
            "kCGWindowLayer": layer,
            "kCGWindowOwnerName": "DB Browser for SQLite",
            "kCGWindowName": "window",
            "kCGWindowBounds": {"X": x, "Y": y, "Width": w, "Height": h},
        }

    def _bounds(self, entries, pid=_PID):
        err = io.StringIO()
        with contextlib.redirect_stderr(err), mock.patch(
            "Quartz.CoreGraphics.CGWindowListCopyWindowInfo",
            return_value=entries,
        ), mock.patch(
            "compiler.ax_pyobjc.app_pid_for_name", return_value=pid
        ):
            bounds = _window_bounds("DB Browser for SQLite")
        return bounds, err.getvalue()

    def test_tiny_utility_window_rejected_main_window_accepted(self) -> None:
        entries = [
            self._entry(self._PID, (0, 34, 1470, 922)),   # main window, sane
            self._entry(self._PID, (10, 900, 120, 80)),   # utility window, tiny
            self._entry(self._PID, (0, 0, 600, 400), layer=1),  # overlay layer
        ]
        bounds, log = self._bounds(entries)
        self.assertEqual(
            bounds, {"x": 0.0, "y": 34.0, "w": 1470.0, "h": 922.0}
        )
        self.assertIn("wsda-bounds-rejected", log)  # rejected rect logged
        self.assertIn("(10.0, 900.0, 120.0, 80.0)", log)

    def test_no_sane_candidate_returns_none_with_marker(self) -> None:
        entries = [self._entry(self._PID, (10, 900, 120, 80))]
        bounds, log = self._bounds(entries)
        self.assertIsNone(bounds)
        self.assertIn("wsda-bounds-rejected", log)

    def test_wrong_pid_rejected_even_with_matching_name(self) -> None:
        # C30's crash class: a same-named/owned window that is not the app
        # pid's main window must never be picked.
        entries = [self._entry(9999, (0, 34, 1470, 922))]
        bounds, log = self._bounds(entries)
        self.assertIsNone(bounds)
        self.assertIn("wsda-bounds-rejected", log)

    def test_largest_sane_candidate_wins(self) -> None:
        entries = [
            self._entry(self._PID, (0, 34, 1400, 900)),
            self._entry(self._PID, (0, 34, 1470, 922)),
        ]
        bounds, _ = self._bounds(entries)
        self.assertEqual(bounds["w"], 1470.0)


class TestC31CompositeClamp(unittest.TestCase):
    """C31: _composite_window_frame clamps impossible composites (garbage
    scale or canvas) to None + 'wsda-composite-clamped' — never a giant
    image, never a raise."""

    def test_garbage_scale_returns_none_with_marker(self) -> None:
        frame = np.zeros((800, 1280, 3), np.uint8)
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            out = _composite_window_frame(
                frame, {"x": 0.0, "y": 0.0, "w": 10.0, "h": 10.0}, 1470.0, 956.0
            )
        self.assertIsNone(out)
        self.assertIn("wsda-composite-clamped", err.getvalue())

    def test_giant_canvas_returns_none(self) -> None:
        frame = np.zeros((800, 1280, 3), np.uint8)
        err = io.StringIO()
        # scale = 1280/320 = 4.0 (allowed) but 8000x2400 = 19.2M px canvas.
        with contextlib.redirect_stderr(err):
            out = _composite_window_frame(
                frame, {"x": 0.0, "y": 0.0, "w": 320.0, "h": 200.0}, 2000.0, 600.0
            )
        self.assertIsNone(out)
        self.assertIn("wsda-composite-clamped", err.getvalue())

    def test_valid_composite_unaffected(self) -> None:
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            canvas = _composite_window_frame(
                np.zeros((800, 1280, 3), np.uint8),
                {"x": 0.0, "y": 56.0, "w": 1470.0, "h": 900.0},
                1470.0,
                956.0,
            )
        self.assertIsNotNone(canvas)
        self.assertEqual(canvas.shape, (832, 1280, 3))
        self.assertNotIn("wsda-composite-clamped", err.getvalue())


class TestC31SnapshotUsesProvider(unittest.TestCase):
    """C31: mid-recording, _results_pane_snapshot reads the recorder's
    provider frame ONLY — zero live-capture calls. Provider returning None
    means UNVERIFIABLE (retry path), and run_query aborts before clicking."""

    def _agent_with_provider(self, frame):
        from PIL import Image

        agent = VisionAgent(model="test-model", output_dir="/tmp")
        agent._frame_provider = lambda: frame
        return agent

    def test_mid_recording_reads_provider_only(self) -> None:
        from PIL import Image

        agent = self._agent_with_provider(Image.new("RGB", (1280, 832), (90, 90, 90)))
        calls: List[int] = []

        def fake_capture():
            calls.append(1)
            return Image.new("RGB", (10, 10))

        with mock.patch.object(
            VisionAgent, "_capture_screen", staticmethod(fake_capture)
        ):
            snap = agent._results_pane_snapshot()
        self.assertEqual(calls, [])  # zero screencapture calls mid-recording
        self.assertFalse(snap.get("unverifiable", False))
        self.assertEqual(set(snap["phash"]), {90})

    def test_provider_none_is_unverifiable_no_live_capture(self) -> None:
        agent = self._agent_with_provider(None)
        calls: List[int] = []

        def fake_capture():
            calls.append(1)
            return None

        with mock.patch.object(
            VisionAgent, "_capture_screen", staticmethod(fake_capture)
        ):
            snap = agent._results_pane_snapshot()
        self.assertEqual(calls, [])
        self.assertTrue(snap["unverifiable"])

    def test_run_query_aborts_for_retry_without_clicking(self) -> None:
        agent = self._agent_with_provider(None)
        agent._ensure_frontmost = mock.Mock()
        agent._read_editor_content = mock.Mock(return_value="SELECT 1;")
        agent._extract_uncommented_sql = mock.Mock(return_value=("SELECT 1;", None))
        agent._verify_statement_isolation = mock.Mock(return_value=True)
        with mock.patch.object(vision_agent_module.pyautogui, "click") as click:
            self.assertFalse(agent.run_query())
        click.assert_not_called()


class TestC31TracebackLogged(unittest.TestCase):
    """C31: failures in the run path log the FULL traceback — no str(exc)-only
    catches anywhere the governor has to diagnose from."""

    def test_attempt_report_carries_full_traceback(self) -> None:
        from compiler.curriculum import CourseManifest, VideoManifest

        manifest = CourseManifest(
            course_id="test_course",
            title="Test",
            description="Test",
            target_audience="Test",
            videos=[
                VideoManifest(
                    video_id="video_1_1",
                    title="Test Video",
                    learning_objective="Test",
                    discovery_objective="Test",
                    application="db_browser_sqlite",
                    format_tier="short",
                )
            ],
        )
        try:
            raise RuntimeError("gate B2 failed")
        except RuntimeError:
            tb = traceback.format_exc()
        tmpdir = tempfile.mkdtemp()
        try:
            report_path = Path(tmpdir) / "attempt_report.json"
            _write_attempt_report(
                manifest=manifest,
                video_id="video_1_1",
                error="gate B2 failed",
                expected_editor_content="SELECT 1;",
                actual_editor_content=None,
                vlm_assessment="v",
                screenshot_paths=[],
                output_path=str(report_path),
                error_traceback=tb,
            )
            report = json.loads(report_path.read_text(encoding="utf-8"))
            self.assertIn("Traceback (most recent call last)", report["traceback"])
            self.assertIn("RuntimeError: gate B2 failed", report["traceback"])
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_iteration_failure_logs_full_traceback(self) -> None:
        err = io.StringIO()
        argv = [
            "prog",
            "--only-video",
            "video_1_1",
            "--max-iterations",
            "1",
            "--output-mode",
            "auto",
        ]
        with mock.patch.object(sys, "argv", argv), mock.patch.object(
            curriculum_module, "run_course", side_effect=ValueError("boom")
        ), contextlib.redirect_stderr(err):
            rc = curriculum_module.main()
        self.assertEqual(rc, 1)
        self.assertIn("Traceback (most recent call last)", err.getvalue())
        self.assertIn("ValueError: boom", err.getvalue())


class TestStageMatchesStory(unittest.TestCase):
    def test_stage_runs_prior_query_and_verifies(self) -> None:
        """Continuity stage-prep runs the prior query and VLM-verifies the screen."""
        discovery = EndStateDiscovery(
            objective="test", application="db_browser_sqlite"
        )
        state_beat = ScriptBeat(
            beat_id="beat_002",
            kind="state",
            text="Our previous queries sit above, commented out.",
        )
        beats = [state_beat]
        agent = mock.MagicMock()
        agent.paste_history_block.return_value = True
        agent.append_block.return_value = True
        agent.execute_beat.return_value = True
        agent.verify_state.return_value = True
        agent.summarize_observed_state.return_value = {
            "summary": "The editor shows commented history and the result pane is populated."
        }

        discovery._prepare_opening_state(
            beats,
            agent,
            opening_state_query="SELECT 1;",
            opening_state_history="/*\nSELECT 0;\n*/",
        )

        # History is pasted twice: once bare, once with the prior query wrapped as a comment.
        self.assertEqual(agent.paste_history_block.call_count, 2)
        agent.append_block.assert_called_once_with("SELECT 1;")
        agent.execute_beat.assert_called_once_with({"type": "run_query"})
        agent.verify_state.assert_called_once_with(state_beat.text)
        self.assertEqual(state_beat.observed_state["opening_state_verified"], True)


class TestEnvironmentProfile(unittest.TestCase):
    def test_profile_drives_focus_activation(self) -> None:
        """A swapped app name in the profile drives focus checks and activation."""
        profile = EnvironmentProfile(
            application="fake_app",
            app_name="Fake Application",
            focus_target="Fake Application",
        )
        agent = VisionAgent(profile=profile)
        subprocess_calls: List[List[str]] = []

        def fake_run(cmd, **kwargs):
            subprocess_calls.append(cmd)
            class FakeResult:
                stdout = "Other App"
            return FakeResult()

        with (
            mock.patch("compiler.vision_agent.subprocess.run", side_effect=fake_run),
            mock.patch("compiler.vision_agent.time.sleep"),
        ):
            with self.assertRaises(Exception):
                agent._ensure_frontmost(max_attempts=1)

        # Activation command must use the profile app name, never a hardcoded DB Browser string.
        activation_calls = [
            c for c in subprocess_calls
            if c[0] == "osascript" and "to activate" in c[2]
        ]
        self.assertTrue(activation_calls)
        self.assertIn("Fake Application", activation_calls[0][2])
        self.assertNotIn("DB Browser", " ".join(str(x) for x in activation_calls))


class TestCommentExecutionVerifier(unittest.TestCase):
    def test_orphan_uncommented_line_fails_isolation(self) -> None:
        """A bare continuation line outside the current statement must block execution."""
        agent = VisionAgent()
        profile = EnvironmentProfile(
            application="db_browser_sqlite",
            app_name="DB Browser for SQLite",
            focus_target="DB Browser for SQLite",
            execute_scope="whole_script",
            comment_syntax={"line": "--", "block_start": "/*", "block_end": "*/"},
        )
        agent.profile = profile
        with mock.patch.object(
            agent,
            "_read_editor_content",
            return_value="SELECT FirstName FROM Customer;\nLastName",
        ):
            self.assertFalse(agent._verify_statement_isolation("SELECT FirstName FROM Customer;"))

    def test_commented_history_passes_isolation(self) -> None:
        """Non-current lines that are commented out are allowed."""
        agent = VisionAgent()
        profile = EnvironmentProfile(
            application="db_browser_sqlite",
            app_name="DB Browser for SQLite",
            focus_target="DB Browser for SQLite",
            execute_scope="whole_script",
            comment_syntax={"line": "--", "block_start": "/*", "block_end": "*/"},
        )
        agent.profile = profile
        buffer = "-- SELECT * FROM Old;\nSELECT FirstName FROM Customer;"
        with mock.patch.object(agent, "_read_editor_content", return_value=buffer):
            self.assertTrue(agent._verify_statement_isolation("SELECT FirstName FROM Customer;"))

    def test_block_comment_history_passes_isolation(self) -> None:
        """A block-commented history above the current statement is allowed."""
        agent = VisionAgent()
        profile = EnvironmentProfile(
            application="db_browser_sqlite",
            app_name="DB Browser for SQLite",
            focus_target="DB Browser for SQLite",
            execute_scope="whole_script",
            comment_syntax={"line": "--", "block_start": "/*", "block_end": "*/"},
        )
        agent.profile = profile
        buffer = "/*\nOld query\n*/\nSELECT FirstName FROM Customer;"
        with mock.patch.object(agent, "_read_editor_content", return_value=buffer):
            self.assertTrue(agent._verify_statement_isolation("SELECT FirstName FROM Customer;"))


class TestRendererPaddingCap(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = Path(tempfile.mkdtemp(prefix="wsda_test_pad_"))

    def tearDown(self) -> None:
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_long_narration_raises_needs_reshoot_and_writes_report(self) -> None:
        """If narration exceeds clip+4s the renderer stops and emits timing debt."""
        clip = _make_video(self.tmpdir / "short_action.mp4", duration=2.0, motion=True)
        beats = [
            ScriptBeat(
                beat_id="beat_001",
                kind="demo",
                text="This narration is deliberately long enough to exceed the four second padding cap when spoken at a normal pace.",
                action={"type": "click", "target": {"x": 0.5, "y": 0.5}},
                video_clip_path=str(clip.resolve()),
            )
        ]

        class Manifest:
            title = "Padding test"
            learning_objective = "Test padding cap."
            application = "db_browser_sqlite"
            format_tier = "short"

        renderer = GraphRenderer(output_dir=str(self.tmpdir))
        tts_durations = {"beat_001": 12.0}
        original = fake_tts(None, tts_durations)  # type: ignore[arg-type]
        try:
            out_path = str(self.tmpdir / "pad_test.mp4")
            with self.assertRaises(RuntimeError) as ctx:
                renderer.render_from_script(
                    video_manifest=Manifest(),
                    script_beats=beats,
                    output_path=out_path,
                    output_mode="hybrid",
                )
            self.assertIn("NEEDS_RESHOOT", str(ctx.exception))
            report_path = self.tmpdir / "pad_test_timing_report.json"
            self.assertTrue(report_path.exists())
            report = json.loads(report_path.read_text(encoding="utf-8"))
            self.assertEqual(report["max_clone_pad_seconds"], 4.0)
            self.assertGreater(report["total_debt_seconds"], 0.0)
            self.assertEqual(len(report["beats"]), 1)
            self.assertAlmostEqual(report["beats"][0]["debt_seconds"], 6.0, delta=0.5)
        finally:
            restore_tts(original)


class TestAdaptationUniqueness(unittest.TestCase):
    def setUp(self) -> None:
        self.builder = LessonBuilder()

    def test_similar_consecutive_beats_yield_merge(self) -> None:
        """Two adapted concept beats with no new datum produce a MERGE."""
        beats = [
            ScriptBeat(
                beat_id="beat_001",
                kind="concept",
                text="We see 60 rows with FirstName and LastName.",
                observed_state={
                    "active_tab": "Execute SQL",
                    "visible_table": "Customer",
                    "row_range_text": "1 - 20 of 60",
                    "column_headers": ["FirstName", "LastName"],
                    "summary": "Result grid visible.",
                },
            ),
            ScriptBeat(
                beat_id="beat_002",
                kind="concept",
                text="We see 100 rows with FirstName and LastName.",
                observed_state={
                    "active_tab": "Execute SQL",
                    "visible_table": "Customer",
                    "row_range_text": "1 - 20 of 60",
                    "column_headers": ["FirstName", "LastName"],
                    "summary": "Result grid still visible.",
                },
            ),
        ]

        def fake_llm_response(*args, **kwargs):
            """Return a rewrite that drops the conflicting number."""
            class Block:
                text = "We see rows with FirstName and LastName."
                type = "text"

            class Response:
                content = [Block()]

            return Response()

        with mock.patch.object(
            self.builder.client.messages, "create", side_effect=fake_llm_response
        ):
            self.builder._adapt_beats_to_observed_state(beats)
        # The second beat must be marked MERGE because the rewrite adds no new datum.
        self.assertTrue(
            beats[1].merge,
            "adapted beat repeated the previous one without a MERGE flag",
        )


class TestFullBufferReadBack(unittest.TestCase):
    def _agent_with_mocks(self) -> VisionAgent:
        agent = VisionAgent()
        mock.patch.object(agent, "find_and_click", return_value=True).start()
        mock.patch.object(agent, "press_key", return_value=True).start()
        self.addCleanup(mock.patch.stopall)
        return agent

    def test_mangled_multiline_paste_detected(self) -> None:
        """A paste that drops a line must fail full-buffer verification."""
        agent = self._agent_with_mocks()
        intended = "SELECT\n    FirstName,\n    LastName\nFROM Customer;"
        # VLM returns content missing the LastName line.
        with (
            mock.patch.object(agent, "_read_editor_content", return_value="SELECT\n    FirstName,\nFROM Customer;"),
            mock.patch("pyautogui.typewrite"),
            mock.patch("pyautogui.press"),
            mock.patch("time.sleep"),
        ):
            self.assertFalse(agent._verify_buffer_exact(intended, "TEST"))


class TestPixelErrorSignature(unittest.TestCase):
    def setUp(self) -> None:
        self.profile = EnvironmentProfile(
            application="db_browser_sqlite",
            app_name="DB Browser for SQLite",
            focus_target="DB Browser for SQLite",
            error_signature={
                "status_region": {"x": 0.0, "y": 0.80, "w": 1.0, "h": 0.20},
                "color_ranges": [
                    {"lower": [0, 100, 50], "upper": [10, 255, 255]},
                    {"lower": [160, 100, 50], "upper": [180, 255, 255]},
                ],
                "min_area_ratio": 0.02,
            },
        )

    def test_bad_frame_fires_error_signature(self) -> None:
        """A known-bad frame from v4 must trigger the pixel error detector."""
        bad_dir = Path("output/course_ch4_v4/bad_frame_samples")
        bad_frames = sorted(bad_dir.glob("frame_*.png"))
        if not bad_frames:
            self.skipTest("No bad-frame fixtures found; run a v4 render to populate them")
        fired = 0
        for p in bad_frames:
            bgr = cv2.imread(str(p))
            if bgr is not None and detect_error_signature(bgr, self.profile):
                fired += 1
        self.assertGreater(fired, 0, "error signature did not fire on any bad frame")

    def test_good_frame_does_not_fire(self) -> None:
        """A plain grey frame must not trigger the error detector."""
        grey = np.full((720, 1280, 3), 128, dtype=np.uint8)
        self.assertFalse(detect_error_signature(grey, self.profile))


class TestFrozenShareMetric(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = Path(tempfile.mkdtemp(prefix="wsda_test_frozen_"))

    def tearDown(self) -> None:
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_frozen_share_on_static_head_motion_tail(self) -> None:
        """A clip with static head, motion, and static tail reports the correct frozen share."""
        head = _make_video(self.tmpdir / "head.mp4", duration=8.0, fps=10, motion=False)
        motion = _make_video(self.tmpdir / "motion.mp4", duration=2.0, fps=10, motion=True)
        tail = _make_video(self.tmpdir / "tail.mp4", duration=3.0, fps=10, motion=False)
        combined = self.tmpdir / "combined.mp4"
        concat_list = self.tmpdir / "concat.txt"
        concat_list.write_text(
            "\n".join(f"file '{p.resolve()}'" for p in (head, motion, tail)),
            encoding="utf-8",
        )
        subprocess.run(
            ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(concat_list),
             "-c", "copy", str(combined)],
            check=True, capture_output=True, timeout=60,
        )
        frozen_pct = frozen_share_percent(combined, sample_fps=1, width=320, mse_threshold=0.5)
        # The clip is mostly static head + tail with a short motion window in the
        # middle, so the frozen share must be high. We assert a broad band rather
        # than an exact value because ffmpeg fps sampling can shift the boundary
        # frames by one sample.
        self.assertGreater(frozen_pct, 60.0)
        self.assertLess(frozen_pct, 95.0)


class TestFrozenRunGate(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = Path(tempfile.mkdtemp(prefix="wsda_test_frozen_run_"))

    def tearDown(self) -> None:
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _reference_md(self) -> Path:
        path = self.tmpdir / "reference.md"
        path.write_text(
            "| Beat | Kind | Words | Text |\n"
            "|------|------|-------|------|\n"
            "| beat_001 | opening | 500 | This is the opening narration ending with punctuation. |\n",
            encoding="utf-8",
        )
        return path

    def _run(self, video: Path) -> Dict[str, Any]:
        profile = EnvironmentProfile(
            application="db_browser_sqlite",
            app_name="DB Browser for SQLite",
            focus_target="DB Browser for SQLite",
        )
        return run_acceptance_gates(
            final_path=video,
            audio_path=None,
            reference_md_path=self._reference_md(),
            profile=profile,
        )

    def test_seven_second_static_run_fails(self) -> None:
        """A single 7-second frozen run fails the B3 anti-stall gate."""
        video = _make_video(self.tmpdir / "static_7s.mp4", duration=7.0, fps=1, motion=False)
        result = self._run(video)
        frozen_gate = next(g for g in result["gates"] if g["gate"] == "B3_frozen")
        self.assertGreaterEqual(float(frozen_gate["value"]), 7.0)
        self.assertFalse(frozen_gate["passed"])

    def test_three_five_second_rests_pass(self) -> None:
        """Three separate 5-second rests, separated by motion, pass B3."""
        clips: List[Path] = []
        for i in range(3):
            clips.append(_make_video(self.tmpdir / f"rest_{i}.mp4", duration=5.0, fps=1, motion=False))
            if i < 2:
                clips.append(_make_video(self.tmpdir / f"motion_{i}.mp4", duration=1.0, fps=1, motion=True))
        combined = self.tmpdir / "combined_rests.mp4"
        concat_list = self.tmpdir / "concat.txt"
        concat_list.write_text(
            "\n".join(f"file '{p.resolve()}'" for p in clips),
            encoding="utf-8",
        )
        subprocess.run(
            ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(concat_list),
             "-c", "copy", str(combined)],
            check=True, capture_output=True, timeout=60,
        )
        result = self._run(combined)
        frozen_gate = next(g for g in result["gates"] if g["gate"] == "B3_frozen")
        self.assertLessEqual(float(frozen_gate["value"]), 6.0)
        self.assertTrue(frozen_gate["passed"])


class TestScriptIntegrityHardened(unittest.TestCase):
    def setUp(self) -> None:
        self.builder = LessonBuilder()

    def _reference_beats(self, version: str) -> List[ScriptBeat]:
        path = Path(f"output/course_ch4_{version}/sql_essential_training_ch4/sql_essential_training_ch4_video_1_5_reference.md")
        text = path.read_text(encoding="utf-8")
        beats: List[ScriptBeat] = []
        for line in text.splitlines():
            line = line.strip()
            if not line.startswith("|") or "Text" in line or "---" in line:
                continue
            cells = [c.strip() for c in line.strip("|").split("|")]
            if len(cells) < 4:
                continue
            beat_id = cells[0]
            kind = cells[1]
            narration = cells[-1]
            beats.append(ScriptBeat(beat_id=beat_id, kind=kind, text=narration))
        return beats

    def test_v4_reference_fails_integrity_gate(self) -> None:
        """The truncated v4 reference script must fail the hardened integrity gate."""
        try:
            beats = self._reference_beats("v4")
        except FileNotFoundError as exc:
            self.skipTest(f"Reference render not available: {exc}")
        self.assertGreater(len(beats), 0)
        self.assertFalse(self.builder.script_integrity_ok(beats))

    def test_v3_reference_passes_integrity_gate(self) -> None:
        """The full v3 reference script must be fixable and then pass the hardened integrity gate."""
        try:
            beats = self._reference_beats("v3")
        except FileNotFoundError as exc:
            self.skipTest(f"Reference render not available: {exc}")
        self.assertGreater(len(beats), 0)
        self.builder._enforce_sentence_integrity(beats)
        self.assertTrue(self.builder.script_integrity_ok(beats))


class TestRendererNoTrim(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = Path(tempfile.mkdtemp(prefix="wsda_test_notrim_"))

    def tearDown(self) -> None:
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_clip_longer_than_narration_is_not_trimmed(self) -> None:
        """If the recorded clip is longer than the narration, the full clip is kept."""
        clip = _make_video(self.tmpdir / "long_action.mp4", duration=5.5, fps=10, motion=True)
        beats = [
            ScriptBeat(
                beat_id="beat_001",
                kind="demo",
                text="Short narration.",
                action={"type": "click", "target": {"x": 0.5, "y": 0.5}},
                video_clip_path=str(clip.resolve()),
            )
        ]

        class Manifest:
            title = "No-trim test"
            learning_objective = "Test no trim."
            application = "db_browser_sqlite"
            format_tier = "short"

        renderer = GraphRenderer(output_dir=str(self.tmpdir))
        tts_durations = {"beat_001": 2.0}
        original = fake_tts(None, tts_durations)  # type: ignore[arg-type]
        try:
            out_path = str(self.tmpdir / "notrim_test.mp4")
            result = renderer.render_from_script(
                video_manifest=Manifest(),
                script_beats=beats,
                output_path=out_path,
                output_mode="auto",
            )
            self.assertIsNotNone(result)
            final_path = Path(result["final_path"])
            self.assertTrue(final_path.exists())
            final_dur = _media_duration(final_path)
            # The full 5.5s clip must survive; final duration should be at least 5.0s.
            self.assertGreaterEqual(final_dur, 5.0)
        finally:
            restore_tts(original)

    def test_clip_overrun_breaching_pad_cap_raises_needs_reshoot(self) -> None:
        """A clip that exceeds narration + MAX_CLONE_PAD_SECONDS is flagged NEEDS_RESHOOT."""
        clip = _make_video(self.tmpdir / "huge_action.mp4", duration=8.0, fps=10, motion=True)
        beats = [
            ScriptBeat(
                beat_id="beat_001",
                kind="demo",
                text="Short narration.",
                action={"type": "click", "target": {"x": 0.5, "y": 0.5}},
                video_clip_path=str(clip.resolve()),
            )
        ]

        class Manifest:
            title = "Overrun test"
            learning_objective = "Test overrun."
            application = "db_browser_sqlite"
            format_tier = "short"

        renderer = GraphRenderer(output_dir=str(self.tmpdir))
        tts_durations = {"beat_001": 2.0}
        original = fake_tts(None, tts_durations)  # type: ignore[arg-type]
        try:
            out_path = str(self.tmpdir / "overrun_test.mp4")
            with self.assertRaises(RuntimeError) as ctx:
                renderer.render_from_script(
                    video_manifest=Manifest(),
                    script_beats=beats,
                    output_path=out_path,
                    output_mode="auto",
                )
            self.assertIn("NEEDS_RESHOOT", str(ctx.exception))
        finally:
            restore_tts(original)


class TestAcceptanceGateAVSync(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = Path(tempfile.mkdtemp(prefix="wsda_test_avsync_"))

    def tearDown(self) -> None:
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _reference_md(self) -> Path:
        path = self.tmpdir / "reference.md"
        path.write_text(
            "| Beat | Kind | Words | Text |\n"
            "|------|------|-------|------|\n"
            "| beat_001 | opening | 500 | This is the opening narration. |\n",
            encoding="utf-8",
        )
        return path

    def _run(self, audio_path: Optional[Path]) -> Dict[str, Any]:
        final_video = _make_video(
            self.tmpdir / "final.mp4", duration=10.0, fps=2, motion=True
        )
        profile = EnvironmentProfile(
            application="db_browser_sqlite",
            app_name="DB Browser for SQLite",
            focus_target="DB Browser for SQLite",
        )
        return run_acceptance_gates(
            final_path=final_video,
            audio_path=audio_path,
            reference_md_path=self._reference_md(),
            profile=profile,
        )

    def test_null_audio_reports_skipped(self) -> None:
        """When no audio file exists the A/V sync gate is skipped, not measured."""
        result = self._run(audio_path=None)
        sync_gate = next(g for g in result["gates"] if g["gate"] == "B2_av_sync")
        self.assertEqual(sync_gate["value"], "skipped")
        self.assertTrue(sync_gate["passed"])

    def test_large_delta_fails(self) -> None:
        """A 3.0s A/V delta fails the sync gate."""
        audio = _sine_wave_mp3(self.tmpdir / "audio_7s.mp3", duration_seconds=7.0)
        result = self._run(audio_path=audio)
        sync_gate = next(g for g in result["gates"] if g["gate"] == "B2_av_sync")
        self.assertAlmostEqual(float(sync_gate["value"]), 3.0, delta=0.2)
        self.assertFalse(sync_gate["passed"])

    def test_small_delta_passes(self) -> None:
        """A 1.0s A/V delta passes the sync gate."""
        audio = _sine_wave_mp3(self.tmpdir / "audio_9s.mp3", duration_seconds=9.0)
        result = self._run(audio_path=audio)
        sync_gate = next(g for g in result["gates"] if g["gate"] == "B2_av_sync")
        self.assertAlmostEqual(float(sync_gate["value"]), 1.0, delta=0.2)
        self.assertTrue(sync_gate["passed"])


class TestFillerBan(unittest.TestCase):
    def setUp(self) -> None:
        self.builder = LessonBuilder()

    def _mock_video(self):
        class MockVideo:
            video_id = "video_1_1"
            title = "Test"
            learning_objective = "Test"
            discovery_objective = "Test"
            application = "db_browser_sqlite"
            format_tier = "short"
            exercise_artifact = {}
            planned_queries = []
        return MockVideo()

    def test_banned_filler_phrase_fails_validation(self) -> None:
        """The exact filler phrase 'The interface updates to show the change' is a defect."""
        beats = [
            ScriptBeat(
                beat_id="beat_001",
                kind="demo",
                text="We type the SELECT clause. The interface updates to show the change.",
                action={"type": "type_block", "text": "SELECT FirstName;"},
            ),
        ]
        ok, errors, _ = self.builder.validate_script(beats, self._mock_video())
        self.assertFalse(ok, "script containing banned filler phrase must fail validation")
        self.assertTrue(
            any("filler" in e.lower() for e in errors),
            f"expected filler-ban error, got {errors}",
        )



class TestAttemptReportNamesFailingGate(unittest.TestCase):
    def test_report_error_names_failing_gate(self) -> None:
        """When a gate fails, the attempt report error names that gate."""
        tmpdir = Path(tempfile.mkdtemp(prefix="wsda_test_report_"))
        try:
            final_video = _make_video(
                tmpdir / "final.mp4", duration=10.0, fps=2, motion=False
            )
            reference = tmpdir / "reference.md"
            reference.write_text(
                "| Beat | Kind | Words | Text |\n"
                "|------|------|-------|------|\n"
                "| beat_001 | opening | 500 | Text ending with punctuation. |\n",
                encoding="utf-8",
            )
            profile = EnvironmentProfile(
                application="db_browser_sqlite",
                app_name="DB Browser for SQLite",
                focus_target="DB Browser for SQLite",
            )
            gate_result = run_acceptance_gates(
                final_path=final_video,
                audio_path=None,
                reference_md_path=reference,
                profile=profile,
            )
            self.assertFalse(gate_result["passed"])

            # Build the same error message the pipeline uses.
            failed = [
                f"{g['gate']}: {g['value']} {g['threshold']}"
                for g in gate_result["gates"]
                if not g["passed"]
            ]
            error = "Acceptance gates failed for video_1_1: " + "; ".join(failed)

            from compiler.curriculum import CourseManifest, VideoManifest
            manifest = CourseManifest(
                course_id="test_course",
                title="Test",
                description="Test",
                target_audience="Test",
                videos=[
                    VideoManifest(
                        video_id="video_1_1",
                        title="Test Video",
                        learning_objective="Test",
                        discovery_objective="Test",
                        application="db_browser_sqlite",
                        format_tier="short",
                    )
                ],
            )
            report_path = tmpdir / "attempt_report.json"
            _write_attempt_report(
                manifest=manifest,
                video_id="video_1_1",
                error=error,
                expected_editor_content="SELECT 1;",
                actual_editor_content="SELECT 1;",
                vlm_assessment="test",
                screenshot_paths=[],
                output_path=str(report_path),
            )
            report = json.loads(report_path.read_text(encoding="utf-8"))
            self.assertIn("B3_frozen", report["error"])
            self.assertNotIn("reliability_score", report)
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)


class TestTargetContentConsistency(unittest.TestCase):
    def setUp(self) -> None:
        self.builder = LessonBuilder()

    def _mock_video(self):
        class MockVideo:
            video_id = "video_1_1"
            title = "Test"
            learning_objective = "Test"
            discovery_objective = "Test"
            application = "db_browser_sqlite"
            format_tier = "short"
            exercise_artifact = {}
            planned_queries = []
        return MockVideo()

    def test_result_pane_sentence_with_browse_data_switch_fails(self):
        """A validation beat referencing the result pane must not click Browse Data."""
        beat = ScriptBeat(
            beat_id="beat_006",
            kind="validation",
            text="We see 60 rows returned in the result pane with FirstName, LastName, and Email headers.",
            action={"type": "verify", "detail": "result pane populated"},
            choreography=[
                {"type": "hover", "target": "the FirstName column header in the result pane", "sentence_idx": 0},
                {"type": "click", "target": "the Browse Data tab", "sentence_idx": 0},
            ],
        )
        errors = self.builder._choreography_matches_sentences(beat, [], "Customer")
        self.assertTrue(any("Browse Data" in e for e in errors), errors)

    def test_matching_target_passes(self):
        """A sentence about the result pane may gesture at the result pane."""
        beat = ScriptBeat(
            beat_id="beat_006",
            kind="validation",
            text="We see 60 rows returned in the result pane with FirstName, LastName, and Email headers.",
            action={"type": "verify", "detail": "result pane populated"},
            choreography=[
                {"type": "hover", "target": "the result pane showing query output", "sentence_idx": 0},
            ],
        )
        errors = self.builder._choreography_matches_sentences(beat, [], "Customer")
        self.assertEqual(errors, [])


class TestC33ModalDismissal(unittest.TestCase):
    """C33: stage prep dismisses frontmost modal dialogs before the
    editor-clean checkpoint; an undismissable modal halts the run."""

    def _modal_seq(self, items):
        """frontmost_modal side effect: pops the scripted sequence, then None
        (no more modals) once exhausted — models a clean dismissal."""
        queue = list(items)

        def _next(*args, **kwargs):
            return queue.pop(0) if queue else None

        return _next

    def _agent_with_modals(self, front_effect, press_result):
        agent = VisionAgent()
        self.addCleanup(mock.patch.stopall)
        mock.patch.object(ax_pyobjc, "app_pid_for_name", return_value=7).start()
        mock.patch.object(ax_pyobjc, "create_application", return_value="app").start()
        mock.patch.object(
            ax_pyobjc, "frontmost_modal", side_effect=front_effect
        ).start()
        press = mock.patch.object(
            ax_pyobjc, "press_button", return_value=press_result
        ).start()
        mock.patch("compiler.vision_agent.subprocess.run").start()
        mock.patch("time.sleep").start()
        return agent, press

    def test_frontmost_dialog_cancel_pressed_marker_proceeds(self) -> None:
        """AXDialog frontmost: Cancel pressed via AX, marker logged, returns."""
        agent, press = self._agent_with_modals(
            self._modal_seq([("dlg", "Edit table definition")]), True
        )
        buf = io.StringIO()
        with contextlib.redirect_stderr(buf):
            agent.dismiss_modal_dialogs()
        press.assert_called_once_with("dlg", "Cancel")
        self.assertIn("wsda-modal-dismissed:Edit table definition", buf.getvalue())
        self.assertNotIn("wsda-modal-stuck", buf.getvalue())

    def test_undismissable_dialog_halts_with_stuck_marker(self) -> None:
        """Cancel missing and Esc ineffective: RuntimeError wsda-modal-stuck."""
        self.addCleanup(mock.patch.stopall)
        agent = VisionAgent()
        mock.patch.object(ax_pyobjc, "app_pid_for_name", return_value=7).start()
        mock.patch.object(ax_pyobjc, "create_application", return_value="app").start()
        mock.patch.object(
            ax_pyobjc, "frontmost_modal",
            return_value=("dlg", "Edit table definition"),
        ).start()
        mock.patch.object(ax_pyobjc, "press_button", return_value=False).start()
        mock.patch("compiler.vision_agent.subprocess.run").start()
        mock.patch("time.sleep").start()
        with self.assertRaises(RuntimeError) as ctx:
            with contextlib.redirect_stderr(io.StringIO()):
                agent.dismiss_modal_dialogs()
        self.assertIn("wsda-modal-stuck:Edit table definition", str(ctx.exception))

    def test_stacked_modals_dismissed_up_to_three(self) -> None:
        """Three stacked modals: three Cancel presses, three markers, no halt.
        Each dismissal reveals the next modal on the re-check, so the scripted
        sequence repeats each modal (loop-top read + post-dismissal re-check)."""
        agent, press = self._agent_with_modals(
            self._modal_seq(
                [("d1", "One"), ("d2", "Two"), ("d2", "Two"),
                 ("d3", "Three"), ("d3", "Three"), None]
            ),
            True,
        )
        buf = io.StringIO()
        with contextlib.redirect_stderr(buf):
            agent.dismiss_modal_dialogs()
        self.assertEqual(press.call_count, 3)
        for title in ("One", "Two", "Three"):
            self.assertIn(f"wsda-modal-dismissed:{title}", buf.getvalue())


class TestC33EditorResolvedFromMainWindow(unittest.TestCase):
    """C33: with a modal dialog's text area and the main-window editor both
    present, length reads and clears target the main-window editor only."""

    def _two_window_tree(self):
        """Dialog window holds the top-most (smallest-y) AXTextArea; the main
        window (title carries the app name and .db filename) holds the editor
        lower down — the pre-C33 poisoned configuration."""
        tree = {
            "app": {"AXWindows": ["dlg", "main"]},
            "dlg": {
                "AXRole": "AXWindow",
                "AXSubrole": "AXDialog",
                "AXTitle": "Edit table definition",
                "AXChildren": ["dlg_ta"],
            },
            "dlg_ta": {"AXRole": "AXTextArea"},
            "main": {
                "AXRole": "AXWindow",
                "AXSubrole": "AXStandardWindow",
                "AXTitle": "DB Browser for SQLite - /tmp/wsda_music.db",
                "AXChildren": ["editor"],
            },
            "editor": {"AXRole": "AXTextArea"},
        }
        y_positions = {"dlg_ta": 100.0, "editor": 400.0}
        return tree, y_positions

    def test_hint_scoped_traversal_excludes_dialog_text_area(self) -> None:
        """title_hints restrict traversal to the main window; without hints
        the dialog's preview is the poisonous top-most pick (pre-C33)."""
        self.addCleanup(mock.patch.stopall)
        tree, y_positions = self._two_window_tree()
        mock.patch.object(
            ax_pyobjc, "copy_attribute",
            side_effect=lambda el, name: tree.get(el, {}).get(name),
        ).start()
        mock.patch.object(
            ax_pyobjc, "element_position",
            side_effect=lambda el: (0.0, y_positions[el]) if el in y_positions else None,
        ).start()
        scoped = ax_pyobjc.find_text_areas(
            "app", title_hints=("DB Browser for SQLite", ".db")
        )
        self.assertEqual([el for el, _ in scoped], ["editor"])
        unscoped = ax_pyobjc.find_text_areas("app")
        top_el, _ = min(unscoped, key=lambda t: t[1] if t[1] is not None else 1e9)
        self.assertEqual(top_el, "dlg_ta")

    def test_length_read_and_clear_target_main_window_editor(self) -> None:
        """_editor_text_length reads the editor's 9 chars, never the dialog's
        21; ensure_editor_clean clears the editor and proceeds."""
        self.addCleanup(mock.patch.stopall)
        tree, y_positions = self._two_window_tree()
        state = {"editor": "SELECT 1;", "dlg_ta": "x" * 21}

        def _copy(el, name):
            if name == "AXValue" and el in state:
                return state[el]
            return tree.get(el, {}).get(name)

        mock.patch.object(ax_pyobjc, "copy_attribute", side_effect=_copy).start()
        mock.patch.object(
            ax_pyobjc, "element_position",
            side_effect=lambda el: (0.0, y_positions[el]) if el in y_positions else None,
        ).start()
        mock.patch.object(ax_pyobjc, "app_pid_for_name", return_value=7).start()
        mock.patch.object(ax_pyobjc, "create_application", return_value="app").start()
        agent = VisionAgent()  # default profile: window_title_hint set
        buf = io.StringIO()
        with contextlib.redirect_stderr(buf):
            n = agent._editor_text_length()
        self.assertEqual(n, 9)

        mock.patch.object(agent, "_ensure_frontmost").start()
        mock.patch.object(
            agent, "_ensure_editor_focused_accessibility", return_value=True
        ).start()
        mock.patch("time.sleep").start()

        def _keystroke(*args, **kwargs):
            state["editor"] = ""

        mock.patch(
            "compiler.vision_agent.subprocess.run", side_effect=_keystroke
        ).start()
        buf2 = io.StringIO()
        with contextlib.redirect_stderr(buf2):
            agent.ensure_editor_clean()
        self.assertIn("wsda-editor-dirty:9chars", buf2.getvalue())
        self.assertIn("wsda-editor-cleared:9chars", buf2.getvalue())
        self.assertEqual(state["dlg_ta"], "x" * 21)


class TestC34MssBackend(unittest.TestCase):
    """C34: pull-based mss window capture — same wall-clock writer semantics,
    grab failures duplicate + count, cursor sprite composited per frame."""

    def _recorder(self, tmp: str, grab_fn, **kwargs):
        from compiler.discovery import _MssWindowRecorder

        rec = _MssWindowRecorder(
            str(Path(tmp) / "c34.mp4"),
            fps=10,
            app_name="FakeApp",
            grab_fn=grab_fn,
            cursor_fn=kwargs.pop("cursor_fn", lambda: (50.0, 20.0)),
            bounds_fn=lambda: {"x": 0.0, "y": 0.0, "w": 200.0, "h": 100.0},
            logical_size_fn=lambda: (200, 100),
            **kwargs,
        )
        rec._scale = 1.0
        rec._max_ticks = 10
        return rec

    def _run(self, rec) -> dict:
        rec.start()
        assert rec._thread is not None
        rec._thread.join(timeout=10.0)
        rec.stop()
        assert rec.delivery_summary is not None
        return rec.delivery_summary

    def test_ten_ticks_ten_frames_all_grabbed(self) -> None:
        """10 wall-clock ticks with a healthy grabber: 10 grabs, 10 writes,
        zero failures, full per-second telemetry."""
        tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, tmp, ignore_errors=True)
        rec = self._recorder(tmp, lambda region: np.zeros((100, 200, 4), np.uint8))
        summary = self._run(rec)
        self.assertEqual(summary["backend"], "mss")
        self.assertEqual(summary["frames_delivered"], 10)
        self.assertEqual(summary["frames_written"], 10)
        self.assertEqual(summary["grabs_failed"], 0)
        self.assertEqual(sum(b["frames_written"] for b in summary["per_second"]), 10)
        cap = cv2.VideoCapture(str(rec.output_path))
        self.assertTrue(cap.isOpened())
        n = 0
        while True:
            ok, _frame = cap.read()
            if not ok:
                break
            n += 1
        cap.release()
        self.assertEqual(n, 10)

    def test_grab_exception_duplicates_latest_and_counts(self) -> None:
        """Failing grabs never drop a write: the latest good frame is
        duplicated and the failure lands in grabs_failed."""
        tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, tmp, ignore_errors=True)
        calls = {"n": 0}

        def flaky(region):
            calls["n"] += 1
            if calls["n"] in (3, 7):
                raise RuntimeError("grab boom")
            return np.zeros((100, 200, 4), np.uint8)

        rec = self._recorder(tmp, flaky)
        summary = self._run(rec)
        self.assertEqual(summary["frames_delivered"], 8)
        self.assertEqual(summary["frames_written"], 10)
        self.assertEqual(summary["grabs_failed"], 2)
        self.assertEqual(
            sum(b["grabs_failed"] for b in summary["per_second"]), 2
        )

    def test_cursor_sprite_drawn_at_polled_position(self) -> None:
        """The mss grab excludes the cursor; the sprite is composited at the
        polled logical position — verify its pixels in the written frame."""
        tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, tmp, ignore_errors=True)
        rec = self._recorder(tmp, lambda region: np.zeros((100, 200, 4), np.uint8))
        self._run(rec)
        cap = cv2.VideoCapture(str(rec.output_path))
        ok, frame = cap.read()
        cap.release()
        self.assertTrue(ok)
        # Window 200x100 logical == frame pixels; cursor at (50, 20) maps to
        # the same pixel (no resize below TARGET_WIDTH). Sprite center is a
        # filled magenta circle (BGR 255,0,255); codec loss gets tolerance.
        px = frame[20, 50]
        self.assertGreater(int(px[0]), 200)  # B
        self.assertLess(int(px[1]), 80)      # G
        self.assertGreater(int(px[2]), 200)  # R
        # Off-sprite pixels stay black.
        away = frame[80, 180]
        self.assertLess(int(away.sum()), 60)


class TestC34MssFloorGate(unittest.TestCase):
    """C34: the mss quality gate flags seconds with too many failed grabs."""

    def test_failures_over_threshold_flagged(self) -> None:
        from compiler.discovery import mss_floor_breach

        summary = {
            "backend": "mss",
            "per_second": [
                {"grabs_failed": 0},
                {"grabs_failed": 3},
            ],
        }
        breach = mss_floor_breach(summary)
        self.assertIsNotNone(breach)
        self.assertIn("wsda-mss-grab-fail", breach)

    def test_failures_under_threshold_pass(self) -> None:
        from compiler.discovery import mss_floor_breach

        summary = {
            "backend": "mss",
            "per_second": [
                {"grabs_failed": 0},
                {"grabs_failed": 2},
            ],
        }
        self.assertIsNone(mss_floor_breach(summary))
        self.assertIsNone(mss_floor_breach(None))


class TestC35SchedulerCoverage(unittest.TestCase):
    """C35: the choreography scheduler keeps coverage under a tight budget.

    Hard invariants: every narration sentence keeps at least one gesture; an
    item is never skipped while it is its sentence's only gesture. Compression
    order: rests to the floor first, then gesture speed, then (and only then)
    redundant items. Execution invariants (C38 semantics): no still block —
    contiguous rest time — exceeds the 3.5s park cap, first motion within 2s,
    plan ends with a gesture.
    """

    @staticmethod
    def _hover(target: str, sidx: int) -> Dict[str, Any]:
        return {"type": "hover", "target": target, "sentence_idx": sidx}

    @staticmethod
    def _pause(sidx: int, duration: float = 1.5) -> Dict[str, Any]:
        return {"type": "pause", "duration": duration, "sentence_idx": sidx}

    @classmethod
    def _plan(cls) -> List[Dict[str, Any]]:
        # beat_003 replica: sentence 0 chains four hovers to one target.
        return [
            cls._hover("the SELECT clause in the SQL editor", 0), cls._pause(0),
            cls._hover("the SELECT clause in the SQL editor", 0), cls._pause(0),
            cls._hover("the SELECT clause in the SQL editor", 0), cls._pause(0),
            cls._hover("the SELECT clause in the SQL editor", 0), cls._pause(0),
            cls._hover("the SQL editor text area", 1), cls._pause(1),
            cls._hover("the comment block in the SQL editor", 2), cls._pause(2),
            cls._hover("the SELECT clause in the SQL editor", 2), cls._pause(2),
        ]

    @staticmethod
    def _sentences(plan: List[Dict[str, Any]]) -> set:
        return {
            it.get("sentence_idx", 0)
            for it in plan
            if it.get("type") in ("hover", "click", "scroll", "drag")
        }

    @staticmethod
    def _still_blocks(plan: List[Dict[str, Any]]) -> List[float]:
        """Still blocks under the C38 semantics: a block holds REST time only
        (the opening gesture's travel is motion). A gesture to a new target
        or a slow glide (speed < 1.0) closes the block; same-target full-speed
        gestures continue it without adding stationary time."""
        blocks: List[float] = []
        current = 0.0
        last_target: Optional[str] = None
        for it in plan:
            if it.get("type") == "pause":
                current += float(it.get("duration", 0.5))
                continue
            target = it.get("target", "")
            speed = float(it.get("speed", 1.0))
            if speed < 1.0 or (target and target != last_target):
                blocks.append(current)
                current = 0.0
                last_target = target
        blocks.append(current)
        return blocks

    def test_tight_budget_rests_compress_to_floor_before_any_skip(self) -> None:
        from compiler.discovery import (
            CHOREO_MAX_SPEED,
            CHOREO_PAUSE_FLOOR,
            _schedule_choreography,
        )

        plan = self._plan()
        # Three of sentence 0's four hovers are same-target repeats and are
        # always deduped (coverage keeps one); their orphaned pauses merge
        # into one rest. A huge budget yields that no-drop shape.
        full = len(_schedule_choreography(plan, 1000.0))
        # Budget that forces compression but not redundancy drops: rests to the
        # floor and full speed must absorb it.
        scheduled = _schedule_choreography(plan, 14.5)
        self.assertEqual(len(scheduled), full, "no item may drop before rests+speed are exhausted")
        pauses = [it for it in scheduled if it.get("type") == "pause"]
        speeds = [float(it.get("speed", 1.0)) for it in scheduled if it.get("type") != "pause"]
        self.assertTrue(
            all(it["duration"] >= CHOREO_PAUSE_FLOOR - 1e-6 for it in pauses),
            "rests never compress below the floor",
        )
        self.assertLessEqual(max(speeds), CHOREO_MAX_SPEED, "gesture speed stays within bounds")

    def test_no_sentence_left_gestureless_when_redundant_items_drop(self) -> None:
        from compiler.discovery import _schedule_choreography

        plan = self._plan()
        wanted = self._sentences(plan)
        # Tighter and tighter: coverage must survive every level.
        for budget in (11.0, 8.0, 5.0, 3.0, 1.5):
            scheduled = _schedule_choreography(plan, budget)
            self.assertTrue(
                wanted <= self._sentences(scheduled),
                f"budget {budget}: sentences lost their only gesture",
            )

    def test_no_executed_park_over_five_seconds_and_lead_within_two(self) -> None:
        from compiler.discovery import CHOREO_LEAD_CAP, CHOREO_PAUSE_CAP, _schedule_choreography

        for budget in (17.69, 13.0, 9.0, 6.0, 4.0):
            scheduled = _schedule_choreography(self._plan(), budget)
            blocks = self._still_blocks(scheduled)
            self.assertLessEqual(
                max(blocks), CHOREO_PAUSE_CAP + 1e-6,
                f"budget {budget}: still block {max(blocks):.2f}s > {CHOREO_PAUSE_CAP}s",
            )
            lead = 0.0
            for it in scheduled:
                if it.get("type") != "pause":
                    break
                lead += float(it.get("duration", 0.5))
            self.assertLessEqual(lead, CHOREO_LEAD_CAP, "first motion must start within 2s")
            self.assertIn(
                scheduled[-1].get("type"), ("hover", "click", "scroll", "drag"),
                "plan must end with a gesture so the clip tail is not parked",
            )

    def test_redundant_drop_only_after_rests_at_floor_and_speed_maxed(self) -> None:
        from compiler.discovery import (
            CHOREO_MAX_SPEED,
            CHOREO_PAUSE_FLOOR,
            _schedule_choreography,
        )

        plan = self._plan()
        # No-drop shape: same-target repeats deduped, orphaned pauses merged.
        full = len(_schedule_choreography(plan, 1000.0))
        # Binary-search the loosest budget that forces a drop.
        dropped = None
        for budget10 in range(50, 140):
            budget = budget10 / 10.0
            if len(_schedule_choreography(plan, budget)) < full:
                dropped = budget
                break
        self.assertIsNotNone(dropped, "some budget must force a redundant drop")
        scheduled = _schedule_choreography(plan, dropped)
        pauses = [it for it in scheduled if it.get("type") == "pause"]
        speeds = [float(it.get("speed", 1.0)) for it in scheduled if it.get("type") != "pause"]
        self.assertTrue(
            all(it["duration"] <= CHOREO_PAUSE_FLOOR + 1e-6 for it in pauses),
            "rests must be at the floor before any redundant gesture drops",
        )
        self.assertTrue(
            all(speed >= CHOREO_MAX_SPEED - 1e-6 for speed in speeds),
            "gesture speed must be maxed before any redundant gesture drops",
        )


class TestC35GuardShape(unittest.TestCase):
    """C35: the post-render final-frame guard derives the expected shape from
    the actual video metadata instead of a hardcoded 1280x800."""

    def test_dimensions_derived_from_video_metadata(self) -> None:
        from compiler.curriculum import _video_stream_dimensions

        with tempfile.TemporaryDirectory() as tmp:
            video = Path(tmp) / "clip.mp4"
            subprocess.run(
                [
                    "ffmpeg", "-y", "-v", "error",
                    "-f", "lavfi", "-i", "testsrc=duration=1:size=1280x802:rate=10",
                    "-pix_fmt", "yuv420p", str(video),
                ],
                check=True, capture_output=True,
            )
            self.assertEqual(_video_stream_dimensions(str(video)), (1280, 802))

    def test_guard_runs_when_shapes_differ_from_hardcoded(self) -> None:
        from compiler.curriculum import _verify_final_frame_matches_locked_state

        with tempfile.TemporaryDirectory() as tmp:
            video = Path(tmp) / "clip.mp4"
            subprocess.run(
                [
                    "ffmpeg", "-y", "-v", "error",
                    "-f", "lavfi", "-i", "testsrc=duration=2:size=1280x802:rate=10",
                    "-pix_fmt", "yuv420p", str(video),
                ],
                check=True, capture_output=True,
            )
            # Locked screenshot at a deliberately different size/aspect: the
            # guard must still run (no shape broadcast error) because it
            # normalizes to the video's actual dimensions.
            shot = Path(tmp) / "locked.png"
            subprocess.run(
                [
                    "ffmpeg", "-y", "-v", "error", "-ss", "1.5", "-i", str(video),
                    "-vframes", "1", "-vf", "scale=640:332", str(shot),
                ],
                check=True, capture_output=True,
            )
            result = mock.Mock()
            result.locked_state = mock.Mock(screenshot_path=str(shot))
            buf = io.StringIO()
            with contextlib.redirect_stderr(buf):
                _verify_final_frame_matches_locked_state(str(video), result)
            out = buf.getvalue()
            self.assertNotIn("could not run final-frame check", out)
            self.assertIn("MATCH", out)


class TestC36SubElementTargets(unittest.TestCase):
    """C36 STEP 1: the planner emits semantic sub-element targets — a clause
    names its editor line, the comment block names its line span — and the
    profile-based resolver lands points on the intended sub-element."""

    SCREEN = (1440.0, 900.0)
    COMMENT = (
        "/*\nCreated By: WSDA Student\nCreate Date: 2026-09-11\n"
        "Description: Customer contact list for management\n*/"
    )
    SELECT_CLAUSE = "\nSELECT\n  FirstName,\n  LastName,\n  Email"
    FROM_CLAUSE = "\nFROM Customer;"

    @staticmethod
    def _line_segments_action(text: str, idxs=None):
        lines = [l for l in text.split("\n") if l]
        if idxs is None:
            idxs = [0] * len(lines)
        segs = []
        for line, sidx in zip(lines, list(idxs) + [0] * len(lines)):
            if segs and segs[-1]["sentence_idx"] == sidx:
                segs[-1]["text"] += "\n" + line
            else:
                segs.append({"text": line, "sentence_idx": sidx})
        return {"type": "type_segments", "segments": segs}

    @staticmethod
    def _beat(beat_id, kind, text, action=None):
        from compiler.narrator import ScriptBeat

        return ScriptBeat(beat_id=beat_id, kind=kind, text=text, action=action)

    def _plan_video_1_1_clauses(self):
        """Thread content through the three typing beats exactly like the
        script builder's choreography loop, and return the planned beats."""
        from compiler.lesson_builder import LessonBuilder as LB

        beats = [
            self._beat(
                "beat_002",
                "demo",
                "We type a comment header at the top of the query so anyone who "
                "opens the file later can see who created it, when it was written, "
                "and what problem it solves. The header appears above the SQL and "
                "documents the query before any code runs, which is a professional "
                "habit worth keeping.",
                self._line_segments_action(self.COMMENT, [0, 0, 0, 1, 1]),
            ),
            self._beat(
                "beat_003",
                "demo",
                "We type the SELECT clause, listing the columns FirstName, LastName, "
                "and Email. SELECT tells the database which columns to return, so we "
                "ask for only the contact fields we need. The SELECT clause appears "
                "between the comment header and the FROM clause, defining the output "
                "clearly.",
                self._line_segments_action(self.SELECT_CLAUSE),
            ),
            self._beat(
                "beat_004",
                "demo",
                "We type FROM Customer to name the data source for the columns. FROM "
                "tells the database which table holds the data, so the query knows to "
                "look in the Customer table. This completes the simplest useful query "
                "pattern, asking for specific data from one table.",
                self._line_segments_action(self.FROM_CLAUSE),
            ),
        ]
        tour_state = {"current_tab": "execute_sql", "introduced": set(), "nominal_rest": None}
        content = ""
        for beat in beats:
            beat.choreography = LB._choreography_for_beat(
                beat,
                ["FirstName", "LastName", "Email"],
                "Customer",
                tour_state,
                content_before=content,
                screen=self.SCREEN,
            )
            content = LB._content_after_beat(beat, content)
        return beats

    def test_from_clause_sentence_targets_line_10_distinct_from_select(self):
        from compiler.target_resolver import (
            MIN_GESTURE_SEPARATION_PX,
            distance,
            nominal_geometry,
            resolve_semantic_target,
        )

        beats = self._plan_video_1_1_clauses()
        geo = nominal_geometry(self.SCREEN)

        def gestures(beat):
            return [
                (it.get("semantic"), resolve_semantic_target(it["semantic"], geo))
                for it in beat.choreography
                if it["type"] in ("hover", "click") and it.get("semantic")
            ]

        beat_003, beat_004 = beats[1], beats[2]
        select_semantics = {name for name, _ in gestures(beat_003)}
        from_sentence = [
            it for it in beat_004.choreography
            if it["type"] in ("hover", "click") and it.get("sentence_idx") == 0
        ]
        self.assertTrue(from_sentence, "the FROM sentence keeps its gesture (coverage)")
        self.assertIn("sql-editor:line:6", select_semantics, "SELECT clause -> its line")
        self.assertEqual(
            from_sentence[0].get("semantic"), "sql-editor:line:10",
            "the FROM-clause sentence must resolve to line 10, not the editor center",
        )

        select_point = resolve_semantic_target("sql-editor:line:6", geo)
        from_point = resolve_semantic_target("sql-editor:line:10", geo)
        self.assertGreaterEqual(
            distance(select_point, from_point),
            MIN_GESTURE_SEPARATION_PX,
            "FROM and SELECT line points must be distinct gestures",
        )

    def test_comment_block_is_a_distinct_region(self):
        from compiler.target_resolver import (
            MIN_GESTURE_SEPARATION_PX,
            distance,
            nominal_geometry,
            resolve_semantic_target,
        )

        geo = nominal_geometry(self.SCREEN)
        comment_point = resolve_semantic_target("sql-editor:comment-block:1-5", geo)
        body_point = resolve_semantic_target("sql-editor:body", geo)
        from_point = resolve_semantic_target("sql-editor:line:10", geo)
        self.assertIsNotNone(comment_point)
        self.assertGreaterEqual(
            distance(comment_point, body_point), MIN_GESTURE_SEPARATION_PX,
            "comment centroid must not sit on the editor center",
        )
        self.assertGreaterEqual(
            distance(comment_point, from_point), MIN_GESTURE_SEPARATION_PX,
            "comment centroid must not sit on the FROM line",
        )
        # Centroid of lines 1..5 is line 3: editor_y + line_height * 2.5.
        editor_rect = geo.editor_rect
        self.assertAlmostEqual(
            comment_point[1], editor_rect[1] + geo.line_height * 2.5, places=3,
        )

    def test_line_math_given_editor_rect_and_line_height(self):
        from compiler.target_resolver import TargetGeometry, resolve_semantic_target

        geo = TargetGeometry(editor_rect=(100.0, 200.0, 700.0, 300.0), line_height=20.0)
        # line 3: y = 200 + 20 * (3 - 0.5) = 250; x = origin + text-start offset.
        self.assertEqual(resolve_semantic_target("sql-editor:line:3", geo), (128.0, 250.0))
        # Beyond the rect the point clamps inside the editor rather than flying off.
        clamped = resolve_semantic_target("sql-editor:line:500", geo)
        self.assertEqual(clamped[1], 490.0)
        self.assertTrue(100.0 <= clamped[0] <= 800.0)


class TestC36DistinctnessRetarget(unittest.TestCase):
    """C36 STEP 1 distinctness rule: a sentence's only gesture never resolves
    onto the cursor's current rest point — it is retargeted to a distinct
    sub-point (different line / region), never protected as a no-op."""

    SCREEN = (1440.0, 900.0)

    @staticmethod
    def _beat(beat_id, kind, text, action=None):
        from compiler.narrator import ScriptBeat

        return ScriptBeat(beat_id=beat_id, kind=kind, text=text, action=action)

    def _plan(self, beat, tour_state, content_before=""):
        from compiler.lesson_builder import LessonBuilder as LB

        beat.choreography = LB._choreography_for_beat(
            beat, ["FirstName", "LastName", "Email"], "Customer",
            tour_state, content_before=content_before, screen=self.SCREEN,
        )
        return beat.choreography

    def test_only_gesture_on_rest_point_is_retargeted_not_protected(self):
        from compiler.target_resolver import (
            MIN_GESTURE_SEPARATION_PX,
            distance,
            nominal_geometry,
            resolve_semantic_target,
        )

        geo = nominal_geometry(self.SCREEN)
        # Two consecutive sentences that both name only "the query": every
        # gesture resolves to the editor body unless refined.
        beat = self._beat(
            "beat_010",
            "explain",
            "The query is now complete and returns the contact list. The query "
            "also documents itself with the comment header.",
            {"type": "wait", "duration": 1.5},
        )
        tour_state = {"current_tab": "execute_sql", "introduced": set(), "nominal_rest": None}
        plan = self._plan(beat, tour_state)
        points = [
            resolve_semantic_target(it["semantic"], geo)
            for it in plan
            if it["type"] in ("hover", "click") and it.get("semantic")
        ]
        hover_names = [it["semantic"] for it in plan if it["type"] == "hover" and it.get("semantic")]
        self.assertGreaterEqual(len(hover_names), 2, "both sentences keep a gesture")
        for name_a, name_b in zip(hover_names, hover_names[1:]):
            self.assertNotEqual(name_a, name_b, "consecutive hovers must not repeat one target")
        for p1, p2 in zip(points, points[1:]):
            self.assertGreaterEqual(
                distance(p1, p2), MIN_GESTURE_SEPARATION_PX,
                "no consecutive gesture pair may be a <40px no-op",
            )

    def test_single_gesture_incoming_rest_forces_distinct_subpoint(self):
        from compiler.target_resolver import (
            MIN_GESTURE_SEPARATION_PX,
            distance,
            nominal_geometry,
            resolve_semantic_target,
        )

        geo = nominal_geometry(self.SCREEN)
        rest = resolve_semantic_target("sql-editor:body", geo)
        beat = self._beat(
            "beat_011",
            "explain",
            "The query is ready to run.",
            {"type": "wait", "duration": 1.5},
        )
        tour_state = {
            "current_tab": "execute_sql",
            "introduced": set(),
            "nominal_rest": rest,  # cursor already rests on the editor center
        }
        plan = self._plan(beat, tour_state)
        hovers = [it for it in plan if it["type"] == "hover" and it.get("semantic")]
        self.assertEqual(len(hovers), 1, "the sentence keeps its single gesture (coverage)")
        only = hovers[0]
        self.assertNotEqual(
            only["semantic"], "sql-editor:body",
            "a no-op hover on the rest point must be retargeted, never protected",
        )
        point = resolve_semantic_target(only["semantic"], geo)
        self.assertGreaterEqual(
            distance(point, rest), MIN_GESTURE_SEPARATION_PX,
            "the retargeted gesture must land on a distinct sub-point",
        )


class TestC36SeamContract(unittest.TestCase):
    """C36 STEP 2: the previous beat's final rest point is passed into the next
    beat's scheduling; the next opener resolves >= 40px from it (else the
    sentence's next-best distinct target is used), and the concatenated seam
    freezes for at most 4 seconds by construction."""

    SCREEN = (1440.0, 900.0)

    @staticmethod
    def _hover(semantic: str, sidx: int, human: str = "the SQL editor text area") -> Dict[str, Any]:
        return {"type": "hover", "target": human, "semantic": semantic, "sentence_idx": sidx}

    @staticmethod
    def _pause(sidx: int, duration: float = 1.5) -> Dict[str, Any]:
        return {"type": "pause", "duration": duration, "sentence_idx": sidx}

    def setUp(self) -> None:
        from compiler.target_resolver import nominal_geometry

        self.geo = nominal_geometry(self.SCREEN)

    def _resolve(self, name: str):
        from compiler.target_resolver import resolve_semantic_target

        return resolve_semantic_target(name, self.geo)

    def test_next_opener_resolves_at_least_40px_from_closing_rest(self):
        from compiler.discovery import _schedule_choreography
        from compiler.target_resolver import MIN_GESTURE_SEPARATION_PX, distance

        # beat N closed resting on the editor body point.
        closing_rest = self._resolve("sql-editor:body")
        # beat N+1's first gesture hovers THE SAME point (the old no-op seam).
        plan = [
            self._hover("sql-editor:body", 0),
            self._pause(0),
            self._hover("sql-editor:comment-block:1-5", 0, "the comment block in the SQL editor"),
            self._pause(0),
            self._hover("sql-editor:line:6", 1, "the SELECT clause in the SQL editor"),
            self._pause(1),
        ]
        scheduled = _schedule_choreography(
            plan, 12.0, resolve_point=self._resolve, prev_rest_point=closing_rest
        )
        first_gesture = next(it for it in scheduled if it["type"] in ("hover", "click", "scroll", "drag"))
        opener = first_gesture.get("semantic") or first_gesture.get("target")
        self.assertNotEqual(opener, "sql-editor:body", "the no-op opener must be retargeted")
        opener_point = self._resolve(opener)
        self.assertGreaterEqual(
            distance(opener_point, closing_rest),
            MIN_GESTURE_SEPARATION_PX,
            f"opener {opener} must resolve >= 40px from the closing rest",
        )

    def test_concatenated_seam_freeze_at_most_four_seconds(self):
        from compiler.discovery import CHOREO_LEAD_CAP, RECORDER_TAIL_SECONDS, _schedule_choreography

        closing_rest = self._resolve("sql-editor:body")
        # A deliberately lead-heavy plan: the scheduler must cap the lead and
        # the seam freeze stays within budget.
        plan = [
            self._pause(0, duration=6.0),
            self._hover("sql-editor:comment-block:1-5", 0, "the comment block in the SQL editor"),
            self._pause(0),
            self._hover("sql-editor:line:6", 1, "the SELECT clause in the SQL editor"),
            self._pause(1),
        ]
        scheduled = _schedule_choreography(
            plan, 14.0, resolve_point=self._resolve, prev_rest_point=closing_rest
        )
        lead = 0.0
        for it in scheduled:
            if it["type"] != "pause":
                break
            lead += float(it.get("duration", 0.5))
        self.assertLessEqual(lead, CHOREO_LEAD_CAP, "first motion starts within the lead cap")
        seam_freeze = RECORDER_TAIL_SECONDS + lead
        self.assertLessEqual(seam_freeze, 4.0, f"seam freeze {seam_freeze:.2f}s exceeds 4s")

    def test_far_opener_is_not_retargeted(self):
        from compiler.discovery import _schedule_choreography

        closing_rest = self._resolve("results-grid:body")
        plan = [
            self._hover("sql-editor:body", 0),
            self._pause(0),
            self._hover("sql-editor:comment-block:1-5", 1, "the comment block in the SQL editor"),
            self._pause(1),
        ]
        scheduled = _schedule_choreography(
            plan, 12.0, resolve_point=self._resolve, prev_rest_point=closing_rest
        )
        first_gesture = next(it for it in scheduled if it["type"] in ("hover", "click", "scroll", "drag"))
        self.assertEqual(
            first_gesture.get("semantic"), "sql-editor:body",
            "an already-distant opener is kept exactly as planned",
        )


class TestC38ParkCapInvariant(unittest.TestCase):
    """C38 STEP 3: no contiguous stationary cursor stretch exceeds the 3.5s
    park cap anywhere in a beat — tail after the last gesture and lead before
    the first included. Simulated timelines against TTS-derived windows."""

    SCREEN = (1440.0, 900.0)

    @staticmethod
    def _hover(semantic: str, sidx: int) -> Dict[str, Any]:
        return {"type": "hover", "target": semantic, "semantic": semantic, "sentence_idx": sidx}

    @staticmethod
    def _pause(sidx: int, duration: float = 1.5) -> Dict[str, Any]:
        return {"type": "pause", "duration": duration, "sentence_idx": sidx}

    def setUp(self) -> None:
        from compiler.target_resolver import nominal_geometry

        self.geo = nominal_geometry(self.SCREEN)

    def _resolve(self, name: str):
        from compiler.target_resolver import resolve_semantic_target

        return resolve_semantic_target(name, self.geo)

    def _simulate(self, plan: List[Dict[str, Any]], window: float) -> float:
        """Max contiguous stationary stretch: pauses, the lead, and the tail
        after the plan ends (all in executor-accurate seconds)."""
        from compiler.discovery import (
            CHOREO_GESTURE_TYPES,
            _choreography_item_seconds,
        )

        cost = 0.0
        parks: List[float] = []
        for it in plan:
            if it.get("type") == "pause":
                d = float(it.get("duration", 0.5))
                parks.append(d)
                cost += d
            elif it.get("type") in CHOREO_GESTURE_TYPES:
                cost += _choreography_item_seconds(it)
        tail = max(0.0, window - cost)
        if tail > 0.0:
            parks.append(tail)
        return max(parks) if parks else 0.0

    def test_long_explain_beat_tail_is_filled(self) -> None:
        """The beat_007 shape: ~21s window, three gestures. Pre-C38 the plan
        topped out near 12s and parked ~9s; the park cap must hold."""
        from compiler.discovery import CHOREO_PAUSE_CAP, _schedule_choreography

        plan = [
            self._hover("results-grid:body", 0), self._pause(0),
            self._hover("results-grid:header:Email", 1), self._pause(1),
            self._hover("sql-editor:body", 2), self._pause(2),
        ]
        rest = self._resolve("sql-editor:line:6")
        scheduled = _schedule_choreography(
            plan, 22.8, resolve_point=self._resolve, prev_rest_point=rest, beat_id="beat_x"
        )
        max_park = self._simulate(scheduled, 21.8)
        self.assertLessEqual(
            max_park, CHOREO_PAUSE_CAP + 1e-6,
            f"max park {max_park:.2f}s exceeds the {CHOREO_PAUSE_CAP}s cap",
        )

    def test_single_gesture_beat_fills_via_expansion(self) -> None:
        from compiler.discovery import CHOREO_PAUSE_CAP, _schedule_choreography

        plan = [self._hover("sql-editor:line:6", 0), self._pause(0)]
        rest = self._resolve("results-grid:body")
        scheduled = _schedule_choreography(
            plan, 16.0, resolve_point=self._resolve, prev_rest_point=rest, beat_id="beat_y"
        )
        max_park = self._simulate(scheduled, 15.0)
        self.assertLessEqual(
            max_park, CHOREO_PAUSE_CAP + 1e-6,
            f"max park {max_park:.2f}s exceeds the {CHOREO_PAUSE_CAP}s cap",
        )

    def test_tight_window_compression_keeps_parks_under_cap(self) -> None:
        from compiler.discovery import CHOREO_PAUSE_CAP, _schedule_choreography

        plan = [
            self._hover("sql-editor:line:6", 0), self._pause(0),
            self._hover("sql-editor:comment-block:1-5", 1), self._pause(1),
            self._hover("sql-editor:body", 2), self._pause(2),
        ]
        rest = self._resolve("results-grid:body")
        for audio in (21.0, 12.0, 7.0, 4.5):
            scheduled = _schedule_choreography(
                plan, audio, resolve_point=self._resolve, prev_rest_point=rest, beat_id="beat_z"
            )
            max_park = self._simulate(scheduled, max(0.0, audio - 1.0))
            self.assertLessEqual(
                max_park, CHOREO_PAUSE_CAP + 1e-6,
                f"audio {audio}: max park {max_park:.2f}s exceeds the cap",
            )


class TestC38ExpansionOrder(unittest.TestCase):
    """C38 STEP 3: a beat with a surplus window fills via pauses, then the
    speed floor, then linger alternatives, then the drift glide — in that
    order, each step logged as a wsda-fill: line."""

    SCREEN = (1440.0, 900.0)

    @staticmethod
    def _hover(semantic: str, sidx: int) -> Dict[str, Any]:
        return {"type": "hover", "target": semantic, "semantic": semantic, "sentence_idx": sidx}

    @staticmethod
    def _pause(sidx: int, duration: float = 1.5) -> Dict[str, Any]:
        return {"type": "pause", "duration": duration, "sentence_idx": sidx}

    def setUp(self) -> None:
        from compiler.target_resolver import nominal_geometry

        self.geo = nominal_geometry(self.SCREEN)

    def _resolve(self, name: str):
        from compiler.target_resolver import resolve_semantic_target

        return resolve_semantic_target(name, self.geo)

    def test_methods_fire_in_order_and_log(self) -> None:
        from compiler.discovery import (
            CHOREO_PAUSE_CAP,
            _choreography_plan_cost,
            _schedule_choreography,
        )

        plan = [
            self._hover("sql-editor:line:6", 0), self._pause(0),
            self._hover("sql-editor:comment-block:1-5", 1), self._pause(1),
        ]
        rest = self._resolve("results-grid:body")
        buf = io.StringIO()
        with contextlib.redirect_stderr(buf):
            scheduled = _schedule_choreography(
                plan, 60.0, resolve_point=self._resolve, prev_rest_point=rest, beat_id="beat_w"
            )
        out = buf.getvalue()
        methods = [
            line.split("method=")[1].split()[0]
            for line in out.splitlines()
            if line.startswith("wsda-fill:")
        ]
        order = ["pauses", "speed-floor", "linger", "drift"]
        first_pos = []
        for m in order:
            self.assertIn(m, methods, f"expansion stage {m!r} must fire for a 60s window")
            first_pos.append(methods.index(m))
        self.assertEqual(
            first_pos, sorted(first_pos),
            f"fill methods must appear in expansion order, got {methods}",
        )
        # Structural evidence of each stage in the returned plan.
        pauses = [it for it in scheduled if it.get("type") == "pause"]
        self.assertTrue(
            any(abs(it["duration"] - CHOREO_PAUSE_CAP) < 1e-6 for it in pauses),
            "stage (a) leaves at least one pause at the cap",
        )
        speeds = [
            float(it.get("speed", 1.0)) for it in scheduled
            if it.get("type") in ("hover", "click")
        ]
        self.assertTrue(any(s < 1.0 for s in speeds), "stage (b) slows at least one glide")
        self.assertGreater(len(scheduled), len(plan), "stages (c)/(d) inject items")
        # The fill target: any remaining deficit up to the park cap is a
        # lawful tail rest — the invariant is "no park exceeds the cap", not
        # "cost reaches the window".
        self.assertGreaterEqual(
            _choreography_plan_cost(scheduled), 59.0 - CHOREO_PAUSE_CAP,
            "the filled plan must cover the window up to one lawful tail rest",
        )
        # Stage (d) ends the plan with the drift pair: two slow hovers.
        hovers = [it for it in scheduled if it.get("type") == "hover"]
        self.assertLessEqual(float(hovers[-2].get("speed", 1.0)), 1.0, "drift out-leg is slow")
        self.assertLess(float(hovers[-1].get("speed", 1.0)), 1.0, "drift return-leg is slow")


class TestC38SeamMarkerAlwaysPrints(unittest.TestCase):
    """C38 STEP 3: _apply_seam_contract ALWAYS prints exactly one wsda-seam:
    summary line per seam — pass or retarget — so a zero grep count
    unambiguously means the code did not run."""

    SCREEN = (1440.0, 900.0)

    @staticmethod
    def _hover(semantic: str, sidx: int) -> Dict[str, Any]:
        return {"type": "hover", "target": semantic, "semantic": semantic, "sentence_idx": sidx}

    def setUp(self) -> None:
        from compiler.target_resolver import nominal_geometry

        self.geo = nominal_geometry(self.SCREEN)

    def _resolve(self, name: str):
        from compiler.target_resolver import resolve_semantic_target

        return resolve_semantic_target(name, self.geo)

    def _seam_lines(self, plan: List[Dict[str, Any]], rest) -> List[str]:
        from compiler.discovery import _schedule_choreography

        buf = io.StringIO()
        with contextlib.redirect_stderr(buf):
            _schedule_choreography(
                plan, 12.0, resolve_point=self._resolve, prev_rest_point=rest, beat_id="beat_s"
            )
        return [ln for ln in buf.getvalue().splitlines() if ln.startswith("wsda-seam:")]

    def test_far_seam_pass_prints_summary_with_retargeted_no(self) -> None:
        lines = self._seam_lines(
            [self._hover("sql-editor:line:6", 0)], self._resolve("results-grid:body")
        )
        self.assertEqual(len(lines), 1, f"exactly one wsda-seam line, got {lines}")
        self.assertIn("retargeted=no", lines[0])
        self.assertIn("prev_rest=", lines[0])
        self.assertIn("dist=", lines[0])

    def test_near_seam_retarget_prints_summary_with_retargeted_yes(self) -> None:
        # Opener lands exactly on the previous rest point -> must retarget.
        rest = self._resolve("sql-editor:body")
        lines = self._seam_lines(
            [
                self._hover("sql-editor:body", 0),
                {"type": "pause", "duration": 1.5, "sentence_idx": 0},
                self._hover("sql-editor:comment-block:1-5", 1),
            ],
            rest,
        )
        self.assertEqual(len(lines), 1, f"exactly one wsda-seam line, got {lines}")
        self.assertIn("retargeted=yes", lines[0])
        self.assertNotIn("opener=sql-editor:body ", lines[0] + " ")


def main() -> int:
    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        print("ffmpeg and ffprobe are required for the test harness.", file=__import__("sys").stderr)
        return 1
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    suite.addTests(loader.loadTestsFromTestCase(TestTrimClipToMotion))
    suite.addTests(loader.loadTestsFromTestCase(TestRenderFromScript))
    suite.addTests(loader.loadTestsFromTestCase(TestAdaptBeatsToObservedState))
    suite.addTests(loader.loadTestsFromTestCase(TestValidationEchoSemantic))
    suite.addTests(loader.loadTestsFromTestCase(TestScriptSimilarityGate))
    suite.addTests(loader.loadTestsFromTestCase(TestScriptIntegrityGate))
    suite.addTests(loader.loadTestsFromTestCase(TestEditorReadBack))
    suite.addTests(loader.loadTestsFromTestCase(TestExactLineTyping))
    suite.addTests(loader.loadTestsFromTestCase(TestDatumLevelEchoDetection))
    suite.addTests(loader.loadTestsFromTestCase(TestUIGrounding))
    suite.addTests(loader.loadTestsFromTestCase(TestFrontmostGate))
    suite.addTests(loader.loadTestsFromTestCase(TestPasteAirlock))
    suite.addTests(loader.loadTestsFromTestCase(TestRunQuery))
    suite.addTests(loader.loadTestsFromTestCase(TestWholeVideoFrameGate))
    suite.addTests(loader.loadTestsFromTestCase(TestSegmentedTyping))
    suite.addTests(loader.loadTestsFromTestCase(TestC17DeterministicDemo))
    suite.addTests(loader.loadTestsFromTestCase(TestC17NarrationSizing))
    suite.addTests(loader.loadTestsFromTestCase(TestC18ConsolidatedSegments))
    suite.addTests(loader.loadTestsFromTestCase(TestC19AppleScriptPaste))
    suite.addTests(loader.loadTestsFromTestCase(TestC20IdempotentFocus))
    suite.addTests(loader.loadTestsFromTestCase(TestC21FocusedElementGuard))
    suite.addTests(loader.loadTestsFromTestCase(TestC22PyobjcTraversal))
    suite.addTests(loader.loadTestsFromTestCase(TestC25PidFallback))
    suite.addTests(loader.loadTestsFromTestCase(TestC22PyobjcRetry))
    suite.addTests(loader.loadTestsFromTestCase(TestC23ClipboardInterlock))
    suite.addTests(loader.loadTestsFromTestCase(TestC24EditorAutoClear))
    suite.addTests(loader.loadTestsFromTestCase(TestC25AppReadinessWait))
    suite.addTests(loader.loadTestsFromTestCase(TestC26StopInvariant))
    suite.addTests(loader.loadTestsFromTestCase(TestC27WallClockWriter))
    suite.addTests(loader.loadTestsFromTestCase(TestC27DeliveryFloor))
    suite.addTests(loader.loadTestsFromTestCase(TestC27TeardownSettle))
    suite.addTests(loader.loadTestsFromTestCase(TestC28CaptureWarmup))
    suite.addTests(loader.loadTestsFromTestCase(TestC29GroundingReadsRecorderFrame))
    suite.addTests(loader.loadTestsFromTestCase(TestC29CodecFallback))
    suite.addTests(loader.loadTestsFromTestCase(TestC29EncodeOffDelivery))
    suite.addTests(loader.loadTestsFromTestCase(TestC30ProviderFrameGeometry))
    suite.addTests(loader.loadTestsFromTestCase(TestC30FreshResultsRequired))
    suite.addTests(loader.loadTestsFromTestCase(TestC30FinalReadResolvesFresh))
    suite.addTests(loader.loadTestsFromTestCase(TestC31BoundsValidation))
    suite.addTests(loader.loadTestsFromTestCase(TestC31CompositeClamp))
    suite.addTests(loader.loadTestsFromTestCase(TestC31SnapshotUsesProvider))
    suite.addTests(loader.loadTestsFromTestCase(TestC31TracebackLogged))
    suite.addTests(loader.loadTestsFromTestCase(TestC33ModalDismissal))
    suite.addTests(loader.loadTestsFromTestCase(TestC33EditorResolvedFromMainWindow))
    suite.addTests(loader.loadTestsFromTestCase(TestC34MssBackend))
    suite.addTests(loader.loadTestsFromTestCase(TestC34MssFloorGate))
    suite.addTests(loader.loadTestsFromTestCase(TestC35SchedulerCoverage))
    suite.addTests(loader.loadTestsFromTestCase(TestC35GuardShape))
    suite.addTests(loader.loadTestsFromTestCase(TestC36SubElementTargets))
    suite.addTests(loader.loadTestsFromTestCase(TestC36DistinctnessRetarget))
    suite.addTests(loader.loadTestsFromTestCase(TestC36SeamContract))
    suite.addTests(loader.loadTestsFromTestCase(TestC38ParkCapInvariant))
    suite.addTests(loader.loadTestsFromTestCase(TestC38ExpansionOrder))
    suite.addTests(loader.loadTestsFromTestCase(TestC38SeamMarkerAlwaysPrints))
    suite.addTests(loader.loadTestsFromTestCase(TestStageMatchesStory))
    suite.addTests(loader.loadTestsFromTestCase(TestEnvironmentProfile))
    suite.addTests(loader.loadTestsFromTestCase(TestCommentExecutionVerifier))
    suite.addTests(loader.loadTestsFromTestCase(TestRendererPaddingCap))
    suite.addTests(loader.loadTestsFromTestCase(TestAdaptationUniqueness))
    suite.addTests(loader.loadTestsFromTestCase(TestFullBufferReadBack))
    suite.addTests(loader.loadTestsFromTestCase(TestPixelErrorSignature))
    suite.addTests(loader.loadTestsFromTestCase(TestFrozenShareMetric))
    suite.addTests(loader.loadTestsFromTestCase(TestFrozenRunGate))
    suite.addTests(loader.loadTestsFromTestCase(TestScriptIntegrityHardened))
    suite.addTests(loader.loadTestsFromTestCase(TestRendererNoTrim))
    suite.addTests(loader.loadTestsFromTestCase(TestAcceptanceGateAVSync))
    suite.addTests(loader.loadTestsFromTestCase(TestFillerBan))
    suite.addTests(loader.loadTestsFromTestCase(TestTargetContentConsistency))
    suite.addTests(loader.loadTestsFromTestCase(TestAttemptReportNamesFailingGate))
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
