#!/usr/bin/env python3
"""
compiler/test_harness.py

Fast local verification harness for pipeline changes. Runs without vision-agent
or ElevenLabs calls by using ffmpeg-generated synthetic clips and monkeypatched
TTS audio.
"""

from __future__ import annotations

import io
import math
import shutil
import subprocess
import sys
import tempfile
import unittest
import unittest.mock as mock
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from compiler.curriculum import _dict_to_script_beat, load_manifest
from compiler.discovery import EndStateDiscovery
from compiler.lesson_builder import LessonBuilder
from compiler.narrator import ScriptBeat
from compiler.renderer import GraphRenderer
from compiler.tts import TTSGenerator
from compiler.vision_agent import VisionAgent


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

    def test_trim_removes_static_head_and_keeps_tail(self) -> None:
        """Motion starts at 8s and ends at 10s; trimmed window should keep tail."""
        clip = _make_video(
            self.tmpdir / "head_motion_tail.mp4",
            duration=8.0,
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
        trimmed_dur = _media_duration(combined)

        # Motion window: 8.0-10.0s with 0.7s pad on each side -> 7.6s to 10.7s -> ~3.1s.
        self.assertAlmostEqual(trimmed_dur, 3.1, delta=0.4)
        # Tail is included because pad extends past motion end.
        self.assertGreater(trimmed_dur, 2.0)
        # Head is removed.
        self.assertLess(trimmed_dur, original_dur - 5.0)

    def test_spinner_small_area_motion_does_not_count(self) -> None:
        """A tiny moving square should be below the motion threshold."""
        clip = _make_video(
            self.tmpdir / "spinner.mp4",
            duration=3.0,
            fps=10,
            motion=True,
            motion_region={"x": 280, "y": 160, "w": 80, "h": 40},
        )
        original_dur = _media_duration(clip)
        self.discovery._trim_clip_to_motion(clip)
        trimmed_dur = _media_duration(clip)

        # The small-region motion is averaged over the downscaled frame and
        # should fall below MOTION_DIFF_THRESHOLD, so the clip falls back to
        # the 2.0s middle slice.
        self.assertAlmostEqual(trimmed_dur, 2.0, delta=0.3)
        self.assertLess(trimmed_dur, original_dur - 0.5)


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
        self.assertIsNotNone(manifest)
        beats = [_dict_to_script_beat(b) for b in manifest.videos[0].script_beats]
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
            self.assertIn("[TYPE BLOCK] read-back mismatch, retry 1/2", log)
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
    suite.addTests(loader.loadTestsFromTestCase(TestScriptIntegrityGate))
    suite.addTests(loader.loadTestsFromTestCase(TestEditorReadBack))
    suite.addTests(loader.loadTestsFromTestCase(TestDatumLevelEchoDetection))
    suite.addTests(loader.loadTestsFromTestCase(TestUIGrounding))
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
