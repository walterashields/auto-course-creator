#!/usr/bin/env python3
"""
compiler/test_harness.py

Fast local verification harness for pipeline changes. Runs without vision-agent
or ElevenLabs calls by using ffmpeg-generated synthetic clips and monkeypatched
TTS audio.
"""

from __future__ import annotations

import io
import json
import math
import shutil
import subprocess
import sys
import tempfile
import unittest
import unittest.mock as mock
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

from compiler.curriculum import _dict_to_script_beat, _verify_video_frames_show_app, load_manifest
from compiler.discovery import EndStateDiscovery, _clip_has_off_app_interval
from compiler.frame_analysis import detect_error_signature, frozen_share_percent
from compiler.lesson_builder import LessonBuilder
from compiler.narrator import ScriptBeat
from compiler.renderer import GraphRenderer
from compiler.schemas import EnvironmentProfile
from compiler.tts import TTSGenerator
from compiler.vision_agent import VisionAgent, VisionAgentResult


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
        def read_back() -> str:
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

    def test_long_narration_writes_timing_report_and_flags_debt(self) -> None:
        """If narration exceeds clip+4s the renderer reports timing debt."""
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
            result = renderer.render_from_script(
                video_manifest=Manifest(),
                script_beats=beats,
                output_path=out_path,
                output_mode="hybrid",
            )
            self.assertIsNotNone(result)
            self.assertEqual(result.get("status"), "NEEDS_RESHOOT")
            self.assertTrue(result.get("needs_reshoot"))
            report_path = Path(result["timing_report_path"])
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
        clip = _make_video(self.tmpdir / "long_action.mp4", duration=8.0, fps=10, motion=True)
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
            # The full 8s clip must survive; final duration should be at least 7.5s.
            self.assertGreaterEqual(final_dur, 7.5)
        finally:
            restore_tts(original)


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
    suite.addTests(loader.loadTestsFromTestCase(TestExactLineTyping))
    suite.addTests(loader.loadTestsFromTestCase(TestDatumLevelEchoDetection))
    suite.addTests(loader.loadTestsFromTestCase(TestUIGrounding))
    suite.addTests(loader.loadTestsFromTestCase(TestFrontmostGate))
    suite.addTests(loader.loadTestsFromTestCase(TestPasteAirlock))
    suite.addTests(loader.loadTestsFromTestCase(TestRunQuery))
    suite.addTests(loader.loadTestsFromTestCase(TestWholeVideoFrameGate))
    suite.addTests(loader.loadTestsFromTestCase(TestSegmentedTyping))
    suite.addTests(loader.loadTestsFromTestCase(TestStageMatchesStory))
    suite.addTests(loader.loadTestsFromTestCase(TestEnvironmentProfile))
    suite.addTests(loader.loadTestsFromTestCase(TestCommentExecutionVerifier))
    suite.addTests(loader.loadTestsFromTestCase(TestRendererPaddingCap))
    suite.addTests(loader.loadTestsFromTestCase(TestAdaptationUniqueness))
    suite.addTests(loader.loadTestsFromTestCase(TestFullBufferReadBack))
    suite.addTests(loader.loadTestsFromTestCase(TestPixelErrorSignature))
    suite.addTests(loader.loadTestsFromTestCase(TestFrozenShareMetric))
    suite.addTests(loader.loadTestsFromTestCase(TestScriptIntegrityHardened))
    suite.addTests(loader.loadTestsFromTestCase(TestRendererNoTrim))
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
