#!/usr/bin/env python3
"""
compiler/discovery.py

EndStateDiscovery harness: given a learning objective and a target application,
launch the app, explore it with a vision model, and lock the first screen state
that satisfies the objective.

For now only "db_browser_sqlite" is supported.  All artifacts (screenshots,
telemetry, and the sample database) live under compiler/discovery_output/.
"""

from __future__ import annotations

import base64
import hashlib
import io
import json
import logging
import math
import os
import re
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import anthropic
import cv2
import mss
import numpy as np
import pyautogui
from PIL import Image

from .frame_analysis import count_error_signature_frames
from .narrator import ScriptBeat
from .schemas import DiscoveryResult, EnvironmentProfile, ScreenState
from .sql_formatter import extract_first_query, format_sql_in_text, format_sql_query
from .vision_agent import VisionAgent
from .cost_tracker import get_tracker, tracked_create

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SUPPORTED_APPLICATIONS = {"db_browser_sqlite"}
MODEL = os.environ.get("DISCOVERY_MODEL", "claude-sonnet-5")
TARGET_LONG_EDGE = 1568
MOTION_DIFF_THRESHOLD = 0.05  # Mean absolute grayscale frame diff; catches cursor and small-region motion.
MIN_CLIP_DURATION_SECONDS = 1.5
MOTION_PAD_SECONDS = 0.5
NO_MOTION_KEEP_SECONDS = 1.5

# C10 hard ceilings for unbounded / weakly-bounded loops.
MAX_SCREEN_RECORDER_FRAMES = 1_080_000  # ~3 hours at 10 fps
MAX_SC_RECORDER_SAMPLES = 1_080_000     # ~3 hours at 10 fps
MAX_TRIM_FRAMES = 1_000_000
MAX_BEAT_RETRIES = max(1, int(os.environ.get("WSDA_MAX_ATTEMPTS", "3")))

logger = logging.getLogger(__name__)
if not logger.handlers:
    _handler = logging.StreamHandler(sys.stderr)
    _handler.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
    logger.addHandler(_handler)
    logger.setLevel(logging.INFO)


# ---------------------------------------------------------------------------
# Focus / frontmost-app helpers
# ---------------------------------------------------------------------------


def _frontmost_app_name() -> Optional[str]:
    """Return the name of the current frontmost process via System Events."""
    try:
        result = subprocess.run(
            [
                "osascript",
                "-e",
                'tell application "System Events" to get name of first process whose frontmost is true',
            ],
            capture_output=True,
            text=True,
            timeout=5,
        )
        return result.stdout.strip() or None
    except Exception as exc:
        print(f"Warning: could not determine frontmost app: {exc}", file=sys.stderr)
        return None


def _log_frontmost(frontmost_log_path: Path) -> None:
    """Append the current frontmost app and timestamp to the run sidecar."""
    frontmost = _frontmost_app_name()
    try:
        frontmost_log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(frontmost_log_path, "a", encoding="utf-8") as f:
            f.write(f"{time.time():.3f}\t{frontmost}\n")
    except Exception as exc:
        print(f"Warning: could not write frontmost log: {exc}", file=sys.stderr)


def _clip_has_off_app_interval(
    frontmost_log_path: Path, t0: float, t1: float, target_app_name: str
) -> bool:
    """Return True if the sidecar records any interval in [t0, t1] not on the target app."""
    if not frontmost_log_path.exists():
        return False
    try:
        with open(frontmost_log_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split("\t", 1)
                if len(parts) != 2:
                    continue
                try:
                    ts = float(parts[0])
                except ValueError:
                    continue
                if t0 <= ts <= t1 and parts[1] != target_app_name:
                    return True
    except Exception as exc:
        print(f"Warning: could not read frontmost log: {exc}", file=sys.stderr)
    return False


def _extract_frame(video_path: str, out_path: str, offset: float = 0.5) -> bool:
    """Extract a single PNG frame from ``video_path`` at ``offset`` seconds."""
    try:
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-ss",
                str(offset),
                "-i",
                video_path,
                "-vframes",
                "1",
                "-pix_fmt",
                "rgb24",
                out_path,
            ],
            check=True,
            capture_output=True,
            timeout=30,
        )
        return Path(out_path).exists()
    except Exception as exc:
        print(f"Warning: could not extract sample frame: {exc}", file=sys.stderr)
        return False


# ---------------------------------------------------------------------------
# Screen recording and cursor animation
# ---------------------------------------------------------------------------


class ScreenRecorder:
    """Capture the screen to an MP4 file using cv2.VideoWriter."""

    # Record at the same width the renderer uses so we do not waste cycles
    # writing full-retina frames.
    TARGET_WIDTH = 1280

    def __init__(self, output_path: str, fps: int = 10):
        self.output_path = Path(output_path)
        self.fps = fps
        self._writer: Optional[cv2.VideoWriter] = None
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._frame_shape: Optional[Tuple[int, int]] = None
        self._logical_size = pyautogui.size()

    def _resize_frame(self, frame: np.ndarray) -> np.ndarray:
        """Resize a frame to TARGET_WIDTH while keeping aspect ratio."""
        h, w = frame.shape[:2]
        if w <= self.TARGET_WIDTH:
            return frame
        scale = self.TARGET_WIDTH / w
        new_w = int(w * scale)
        new_h = int(h * scale)
        # Ensure dimensions are even (required by many codecs).
        new_w = new_w - (new_w % 2)
        new_h = new_h - (new_h % 2)
        return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)

    def _draw_cursor(self, frame: np.ndarray) -> np.ndarray:
        """Overlay the mouse cursor so it is visible in the recording."""
        try:
            cursor_x, cursor_y = pyautogui.position()
        except Exception:
            return frame

        logical_w, logical_h = self._logical_size
        if logical_w == 0 or logical_h == 0:
            return frame

        frame_h, frame_w = frame.shape[:2]
        scale = frame_w / logical_w
        x = int(round(cursor_x * scale))
        y = int(round(cursor_y * scale))

        # Keep inside frame bounds.
        x = max(0, min(x, frame_w - 1))
        y = max(0, min(y, frame_h - 1))

        # Draw a magenta ring with a black dot center (highly visible on most UIs).
        cv2.circle(frame, (x, y), 8, (0, 0, 0), 2)
        cv2.circle(frame, (x, y), 5, (255, 0, 255), -1)
        return frame

    def _capture_loop(self) -> None:
        interval = 1.0 / self.fps
        frame_count = 0
        with mss.MSS() as sct:
            monitor = sct.monitors[0]  # Full screen
            # Ceiling: MAX_SCREEN_RECORDER_FRAMES (~3h at 10fps); loud abort if exceeded.
            while not self._stop_event.is_set() and frame_count < MAX_SCREEN_RECORDER_FRAMES:
                frame_count += 1
                start = time.time()
                try:
                    raw = sct.grab(monitor)
                except Exception as exc:
                    print(f"Warning: screenshot capture failed during recording: {exc}", file=sys.stderr)
                    time.sleep(interval)
                    continue

                # mss.grab returns BGRA; convert to BGR for OpenCV.
                frame = np.array(raw)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
                frame = self._resize_frame(frame)
                # C9: do not draw a synthetic cursor overlay; only real cursor
                # motion from emphasis actions should appear in the recording.

                if self._writer is None:
                    h, w = frame.shape[:2]
                    self._frame_shape = (w, h)
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    self._writer = cv2.VideoWriter(
                        str(self.output_path), fourcc, self.fps, (w, h)
                    )

                self._writer.write(frame)

                elapsed = time.time() - start
                sleep_time = max(0.0, interval - elapsed)
                if sleep_time > 0:
                    time.sleep(sleep_time)
            if frame_count >= MAX_SCREEN_RECORDER_FRAMES:
                print(
                    f"[CIRCUIT BREAKER] ScreenRecorder hit frame ceiling "
                    f"({MAX_SCREEN_RECORDER_FRAMES}); stopping capture loop.",
                    file=sys.stderr,
                )

    def start(self) -> None:
        """Start recording in a background thread."""
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Stop recording and release the video writer."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        if self._writer is not None:
            self._writer.release()
            self._writer = None


def animate_cursor_to(x: float, y: float, duration: float = 0.6) -> None:
    """Move the mouse cursor smoothly to (x, y) in macOS logical points."""
    pyautogui.moveTo(x, y, duration=duration, tween=pyautogui.easeInOutQuad)


def _find_window_id(app_name: str) -> Optional[int]:
    """Return the window ID of the frontmost window owned by ``app_name``."""
    try:
        from Quartz import CoreGraphics

        window_list = CoreGraphics.CGWindowListCopyWindowInfo(
            CoreGraphics.kCGWindowListOptionOnScreenOnly,
            CoreGraphics.kCGNullWindowID,
        )
        for entry in window_list:
            owner = entry.get(CoreGraphics.kCGWindowOwnerName, "")
            name = entry.get(CoreGraphics.kCGWindowName, "")
            if owner == app_name or name == app_name:
                return int(entry[CoreGraphics.kCGWindowNumber])
    except Exception as exc:
        print(f"Warning: could not find window id for {app_name}: {exc}", file=sys.stderr)
    return None


def _load_sc_class(name: str) -> Any:
    """Import a ScreenCaptureKit class, falling back to objc.lookUpClass."""
    try:
        import ScreenCaptureKit as sc

        cls = getattr(sc, name, None)
        if cls is not None:
            return cls
    except Exception:
        pass
    try:
        import objc

        return objc.lookUpClass(name)
    except Exception as exc:
        raise ImportError(f"ScreenCaptureKit class {name!r} is not available: {exc}")


class _ScreenCaptureKitRecorder:
    """
    Window-targeted screen recorder using ScreenCaptureKit (macOS 12.3+).

    Captures only the window owned by the profile app so occluding windows never
    appear in the recording. Falls back to full-display capture with a warning if
    the framework is unavailable.
    """

    TARGET_WIDTH = 1280

    def __init__(self, output_path: str, fps: int = 10, app_name: str = ""):
        self.output_path = Path(output_path)
        self.fps = fps
        self.app_name = app_name
        self._writer: Optional[cv2.VideoWriter] = None
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._frame_shape: Optional[Tuple[int, int]] = None
        self._stream: Any = None
        self._delegate: Any = None
        self._samples: List[Any] = []
        self._lock = threading.Lock()
        self._fallback: Optional[ScreenRecorder] = None

    def _resize_frame(self, frame: np.ndarray) -> np.ndarray:
        h, w = frame.shape[:2]
        if w <= self.TARGET_WIDTH:
            return frame
        scale = self.TARGET_WIDTH / w
        new_w = int(w * scale)
        new_h = int(h * scale)
        new_w = new_w - (new_w % 2)
        new_h = new_h - (new_h % 2)
        return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)

    def _draw_cursor(self, frame: np.ndarray) -> np.ndarray:
        try:
            cursor_x, cursor_y = pyautogui.position()
        except Exception:
            return frame
        logical_w, logical_h = pyautogui.size()
        if logical_w == 0 or logical_h == 0:
            return frame
        frame_h, frame_w = frame.shape[:2]
        scale = frame_w / logical_w
        x = int(round(cursor_x * scale))
        y = int(round(cursor_y * scale))
        x = max(0, min(x, frame_w - 1))
        y = max(0, min(y, frame_h - 1))
        cv2.circle(frame, (x, y), 8, (0, 0, 0), 2)
        cv2.circle(frame, (x, y), 5, (255, 0, 255), -1)
        return frame

    def _sample_buffer_to_bgr(self, sample_buffer) -> Optional[np.ndarray]:
        """Convert a CMSampleBuffer to a BGR numpy array."""
        try:
            import CoreMedia
            from Quartz.CoreVideo import (
                CVPixelBufferGetBaseAddress,
                CVPixelBufferGetBytesPerRow,
                CVPixelBufferGetHeight,
                CVPixelBufferGetWidth,
                CVPixelBufferLockBaseAddress,
                CVPixelBufferUnlockBaseAddress,
                kCVPixelBufferLock_ReadOnly,
            )

            image_buffer = CoreMedia.CMSampleBufferGetImageBuffer(sample_buffer)
            if image_buffer is None:
                return None
            CVPixelBufferLockBaseAddress(image_buffer, kCVPixelBufferLock_ReadOnly)
            width = CVPixelBufferGetWidth(image_buffer)
            height = CVPixelBufferGetHeight(image_buffer)
            base_address = CVPixelBufferGetBaseAddress(image_buffer)
            bytes_per_row = CVPixelBufferGetBytesPerRow(image_buffer)
            # BGRA 32-bit. PyObjC returns the base address as an objc.varlist;
            # as_buffer gives a memoryview we can feed directly to numpy.
            buf = np.frombuffer(
                base_address.as_buffer(bytes_per_row * height), dtype=np.uint8
            ).reshape((height, bytes_per_row // 4, 4))[:, :width, :]
            CVPixelBufferUnlockBaseAddress(image_buffer, kCVPixelBufferLock_ReadOnly)
            return cv2.cvtColor(buf, cv2.COLOR_BGRA2BGR)
        except Exception as exc:
            print(f"Warning: could not convert sample buffer: {exc}", file=sys.stderr)
            return None

    def _writer_loop(self) -> None:
        interval = 1.0 / self.fps
        sample_count = 0
        # Ceiling: MAX_SC_RECORDER_SAMPLES (~3h at 10fps); loud abort if exceeded.
        while not self._stop_event.is_set() and sample_count < MAX_SC_RECORDER_SAMPLES:
            sample_count += 1
            start = time.time()
            sample = None
            with self._lock:
                if self._samples:
                    sample = self._samples.pop(0)
            if sample is not None:
                frame = self._sample_buffer_to_bgr(sample)
                if frame is not None:
                    frame = self._resize_frame(frame)
                    # C9: no synthetic cursor overlay; keep only real cursor motion.
                    if self._writer is None:
                        h, w = frame.shape[:2]
                        self._frame_shape = (w, h)
                        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                        self._writer = cv2.VideoWriter(
                            str(self.output_path), fourcc, self.fps, (w, h)
                        )
                    self._writer.write(frame)
            elapsed = time.time() - start
            sleep_time = max(0.0, interval - elapsed)
            if sleep_time > 0:
                time.sleep(sleep_time)
        if sample_count >= MAX_SC_RECORDER_SAMPLES:
            print(
                f"[CIRCUIT BREAKER] _ScreenCaptureKitRecorder hit sample ceiling "
                f"({MAX_SC_RECORDER_SAMPLES}); stopping writer loop.",
                file=sys.stderr,
            )

    def _build_delegate(self):
        import ScreenCaptureKit as sc
        NSObject = _load_sc_class("NSObject")

        recorder = self

        # Each delegate must have a unique Objective-C class name; reusing the
        # same name across recorder instances causes a registration conflict.
        class_name = f"_WSDAScreenCaptureDelegate_{uuid.uuid4().hex}"

        def _stream_didOutputSampleBuffer_ofType_(
            self, stream, sample_buffer, output_type
        ):
            if output_type == sc.SCStreamOutputTypeScreen:
                with recorder._lock:
                    recorder._samples.append(sample_buffer)

        DelegateClass = type(
            class_name,
            (NSObject,),
            {"stream_didOutputSampleBuffer_ofType_": _stream_didOutputSampleBuffer_ofType_},
        )
        return DelegateClass.alloc().init()

    def _get_shareable_content(self):
        """Fetch SCShareableContent synchronously from the async API."""
        SCShareableContent = _load_sc_class("SCShareableContent")
        import threading

        event = threading.Event()
        result = {}

        def handler(content, error):
            result["content"] = content
            result["error"] = error
            event.set()

        SCShareableContent.getShareableContentWithCompletionHandler_(handler)
        event.wait(timeout=5.0)
        return result.get("content"), result.get("error")

    def _start_stream(self) -> bool:
        try:
            import objc

            SCStream = _load_sc_class("SCStream")
            SCStreamConfiguration = _load_sc_class("SCStreamConfiguration")
            SCContentFilter = _load_sc_class("SCContentFilter")
            import ScreenCaptureKit as sc
            from Quartz import CoreGraphics
            from Quartz.CoreVideo import kCVPixelFormatType_32BGRA
            import CoreMedia

            window_id = _find_window_id(self.app_name)
            if window_id is None:
                print(
                    f"[WINDOW-CAPTURE] could not find window for {self.app_name}; falling back to full screen",
                    file=sys.stderr,
                )
                return False

            filter_ = SCContentFilter.alloc().initWithWindowId_(window_id)
            if filter_ is None:
                # Fallback to the shareable-content path if initWithWindowId_ is unavailable.
                content, error = self._get_shareable_content()
                if content is None:
                    print(
                        f"[WINDOW-CAPTURE] could not get shareable content: {error}; falling back",
                        file=sys.stderr,
                    )
                    return False

                target_window = None
                for window in content.windows():
                    if int(window.windowID()) == window_id:
                        target_window = window
                        break
                if target_window is None:
                    print(
                        f"[WINDOW-CAPTURE] window id {window_id} not in SCShareableContent; falling back",
                        file=sys.stderr,
                    )
                    return False

                filter_ = SCContentFilter.alloc().initWithDesktopIndependentWindow_(target_window)
            config = SCStreamConfiguration.alloc().init()
            config.setCapturesAudio_(False)
            config.setPixelFormat_(kCVPixelFormatType_32BGRA)
            config.setWidth_(1920)
            config.setHeight_(1200)
            config.setShowsCursor_(True)
            config.setMinimumFrameInterval_(CoreMedia.CMTimeMake(1, self.fps))

            self._delegate = self._build_delegate()
            self._stream = SCStream.alloc().initWithFilter_configuration_delegate_(filter_, config, self._delegate)
            self._stream.addStreamOutput_type_sampleHandlerQueue_error_(
                self._delegate,
                sc.SCStreamOutputTypeScreen,
                None,
                None,
            )
            self._stream.startCaptureWithCompletionHandler_(lambda error: None)
            return True
        except Exception as exc:
            print(
                f"[WINDOW-CAPTURE] ScreenCaptureKit failed: {exc}; falling back to full screen",
                file=sys.stderr,
            )
            return False

    def start(self) -> None:
        self._stop_event.clear()
        if self._start_stream():
            self._thread = threading.Thread(target=self._writer_loop, daemon=True)
            self._thread.start()
        else:
            print(
                "[WINDOW-CAPTURE] falling back to full-display MSS capture",
                file=sys.stderr,
            )
            self._fallback = ScreenRecorder(str(self.output_path), fps=self.fps)
            self._fallback.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._fallback is not None:
            self._fallback.stop()
            return
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        if self._stream is not None:
            try:
                self._stream.stopCaptureWithCompletionHandler_(lambda error: None)
            except Exception:
                pass
        if self._writer is not None:
            self._writer.release()
            self._writer = None


def test_window_capture_occlusion(
    app_name: str = "DB Browser for SQLite",
    duration: float = 6.0,
    occlusion_duration: float = 3.0,
    output_path: Optional[str] = None,
) -> bool:
    """
    Real occlusion test for the ScreenCaptureKit window recorder.

    Records ``app_name`` for ``duration`` seconds. Halfway through, Finder is
    brought to the front for ``occlusion_duration`` seconds. The test passes
    only when the captured clip shows no trace of Finder (i.e. window-targeted
    capture excluded the occluding window).
    """
    if output_path is None:
        output_path = (
            Path(__file__).resolve().parent
            / "discovery_output"
            / f"occlusion_test_{uuid.uuid4().hex[:8]}.mp4"
        )
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    def _activate(app: str) -> None:
        subprocess.run(
            ["osascript", "-e", f'tell application "{app}" to activate'],
            capture_output=True,
            text=True,
            timeout=5,
        )

    _activate(app_name)
    time.sleep(0.5)

    recorder = _ScreenCaptureKitRecorder(str(output_path), fps=10, app_name=app_name)
    recorder.start()
    if recorder._fallback is not None:
        recorder.stop()
        raise RuntimeError(
            "[OCCLUSION TEST] Window capture fell back to full display; cannot test occlusion"
        )

    try:
        lead = (duration - occlusion_duration) / 2.0
        time.sleep(lead)
        print("[OCCLUSION TEST] Raising Finder", file=sys.stderr)
        _activate("Finder")
        time.sleep(occlusion_duration)
        print("[OCCLUSION TEST] Restoring target app", file=sys.stderr)
        _activate(app_name)
        time.sleep(lead)
    finally:
        recorder.stop()

    cap = cv2.VideoCapture(str(output_path))
    if not cap.isOpened():
        raise RuntimeError(f"[OCCLUSION TEST] Could not open {output_path}")
    frames: List[np.ndarray] = []
    # Ceiling: MAX_TRIM_FRAMES; abort loudly if a corrupt file loops forever.
    while len(frames) < MAX_TRIM_FRAMES:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    else:
        print(
            f"[CIRCUIT BREAKER] occlusion test frame read hit ceiling "
            f"({MAX_TRIM_FRAMES}); aborting.",
            file=sys.stderr,
        )
    cap.release()

    if len(frames) < 10:
        raise RuntimeError(
            f"[OCCLUSION TEST] Too few frames captured ({len(frames)})"
        )

    # Compare a central content ROI so the window's own title-bar chrome
    # changing between active/inactive does not false-positive the test.
    h, w = frames[0].shape[:2]
    y0, y1 = int(h * 0.12), int(h * 0.92)
    x0, x1 = int(w * 0.05), int(w * 0.95)
    baseline = cv2.cvtColor(frames[0][y0:y1, x0:x1], cv2.COLOR_BGR2GRAY)
    occlusion_start = int(len(frames) * 0.33)
    occlusion_end = int(len(frames) * 0.66)
    max_mse = 0.0
    for i in range(occlusion_start, occlusion_end):
        gray = cv2.cvtColor(frames[i][y0:y1, x0:x1], cv2.COLOR_BGR2GRAY)
        mse = float(np.mean((gray.astype(np.float32) - baseline.astype(np.float32)) ** 2))
        if mse > max_mse:
            max_mse = mse

    print(
        f"[OCCLUSION TEST] captured {len(frames)} frames; max MSE during occlusion = {max_mse:.2f}",
        file=sys.stderr,
    )
    if max_mse > 10.0:
        print(
            f"[OCCLUSION TEST] FAIL: occluding window leaked into capture (max MSE {max_mse:.2f})",
            file=sys.stderr,
        )
        return False
    print("[OCCLUSION TEST] PASS: no occluding window trace", file=sys.stderr)
    return True


class DiscoveryRecipes:
    """Deterministic action sequences for common DB Browser UI patterns."""

    @staticmethod
    def browse_table(table_name: str, db_path: Optional[str] = None) -> List[Dict[str, Any]]:
        """Return action sequence to open a table in the Browse Data tab."""
        actions: List[Dict[str, Any]] = [
            {
                "action": "click",
                "target": {"x": 0.516, "y": 0.110, "w": 120, "h": 30},
                "fallback_targets": [
                    {"x": 0.500, "y": 0.110},
                    {"x": 0.532, "y": 0.110},
                    {"x": 0.516, "y": 0.095},
                ],
                "description": "Click Browse Data tab",
                "animate": True,
            },
            {
                "action": "click",
                "target": {"x": 0.075, "y": 0.143, "w": 150, "h": 25},
                "description": "Open table dropdown",
                "animate": True,
            },
        ]

        # Use the database schema to determine the table's position in the
        # dropdown list, then navigate to it with Home + Down arrows. This
        # avoids relying on type-ahead or fragile item coordinates.
        tables = _list_tables(db_path) if db_path else []
        if table_name in tables:
            index = tables.index(table_name)
            actions.append({
                "action": "key",
                "text": "Home",
                "description": "Jump to first dropdown item",
            })
            for _ in range(index):
                actions.append({
                    "action": "key",
                    "text": "Down",
                    "description": f"Navigate dropdown down toward {table_name}",
                })

        actions.extend([
            {
                "action": "key",
                "text": "Return",
                "description": "Press Return to select table",
            },
            {
                "action": "key",
                "text": "Escape",
                "description": "Dismiss open dropdown menu",
            },
            {
                "action": "wait",
                "duration": 0.5,
                "description": "Wait for UI to settle",
            },
        ])
        return actions

    @staticmethod
    def sort_column(table_name: str, column: str, direction: str) -> List[Dict[str, Any]]:
        """Return action sequence to sort a column in the Browse Data tab."""
        header_target = _column_header_target(column)
        actions: List[Dict[str, Any]] = [
            {
                "action": "click",
                "target": {"x": 0.516, "y": 0.110, "w": 120, "h": 30},
                "fallback_targets": [
                    {"x": 0.500, "y": 0.110},
                    {"x": 0.532, "y": 0.110},
                    {"x": 0.516, "y": 0.095},
                ],
                "description": "Click Browse Data tab",
                "animate": True,
            },
            {
                "action": "click",
                "target": {"x": 0.075, "y": 0.143, "w": 150, "h": 25},
                "description": f"Select {table_name} from table dropdown",
                "animate": True,
            },
            # Dismiss any open dropdown menu so the next click reaches the header.
            {
                "action": "key",
                "text": "Escape",
                "description": "Dismiss open dropdown menu",
            },
            {
                "action": "click",
                "target": {**header_target, "w": 130, "h": 28},
                "description": f"Click {column} column header",
                "animate": True,
            },
        ]
        if direction == "desc":
            actions.append(
                {
                    "action": "click",
                    "target": {**header_target, "w": 130, "h": 28},
                    "description": f"Click {column} column header again to sort descending",
                    "animate": True,
                }
            )
        return actions

    @staticmethod
    def filter_column(table_name: str, column: str, value: str) -> List[Dict[str, Any]]:
        """Return action sequence to filter a column in the Browse Data tab."""
        filter_target = _filter_target_for_column(column)
        return [
            {
                "action": "click",
                "target": {"x": 0.516, "y": 0.110, "w": 120, "h": 30},
                "fallback_targets": [
                    {"x": 0.500, "y": 0.110},
                    {"x": 0.532, "y": 0.110},
                    {"x": 0.516, "y": 0.095},
                ],
                "description": "Click Browse Data tab",
                "animate": True,
            },
            {
                "action": "click",
                "target": {"x": 0.075, "y": 0.143, "w": 150, "h": 25},
                "description": f"Select {table_name} from table dropdown",
                "animate": True,
            },
            # Dismiss any open dropdown menu so the filter box click lands correctly.
            {
                "action": "key",
                "text": "Escape",
                "description": "Dismiss open dropdown menu",
            },
            {
                "action": "click",
                "target": {**filter_target, "w": 120, "h": 24},
                "description": f"Click {column} filter box",
                "animate": True,
            },
            {
                "action": "type",
                "target": {**filter_target, "w": 120, "h": 24},
                "text": value,
                "description": f"Type {value!r} into {column} filter box",
            },
            {
                "action": "key",
                "text": "Return",
                "description": "Press Return to apply filter",
            },
        ]

    @staticmethod
    def execute_query(query: str) -> List[Dict[str, Any]]:
        """Return action sequence to run a query in the Execute SQL tab."""
        formatted = format_sql_query(query)
        # Use the first non-comment line for short action labels.
        short_label = formatted.splitlines()[-1].strip() or "SQL query"
        return [
            {
                "action": "click",
                "target": {"x": 0.620, "y": 0.110, "w": 120, "h": 30},
                "fallback_targets": [
                    {"x": 0.600, "y": 0.110},
                    {"x": 0.640, "y": 0.110},
                    {"x": 0.620, "y": 0.095},
                    {"x": 0.650, "y": 0.130},
                    {"x": 0.600, "y": 0.130},
                ],
                "description": "Click Execute SQL tab",
                "animate": True,
            },
            {
                "action": "wait",
                "duration": 0.5,
                "description": "Wait for Execute SQL tab to load",
            },
            {
                "action": "click",
                "target": {"x": 0.500, "y": 0.450, "w": 700, "h": 300},
                "description": "Click SQL editor text area",
                "animate": True,
            },
            {
                "action": "wait",
                "duration": 0.3,
                "description": "Wait for editor focus",
            },
            {
                "action": "key",
                "text": "cmd+a",
                "description": "Select all existing SQL text",
            },
            {
                "action": "key",
                "text": "delete",
                "description": "Delete selected SQL text",
            },
            {
                "action": "type",
                "target": {"x": 0.500, "y": 0.450, "w": 700, "h": 300},
                "text": formatted,
                "click_first": False,
                "description": f"Type query: {short_label}",
            },
            {
                "action": "key",
                "text": "F5",
                "description": "Press F5 to execute query",
            },
            {
                "action": "wait",
                "duration": 1.5,
                "description": "Wait for results to appear",
            },
        ]


def _column_header_target(column: str) -> Dict[str, float]:
    """Return normalized center coordinates for a column header in the Orders table."""
    lowered = column.lower()
    positions = {
        "id": {"x": 0.026, "y": 0.162},
        "order_id": {"x": 0.026, "y": 0.162},
        "region": {"x": 0.060, "y": 0.162},
        "customer_id": {"x": 0.054, "y": 0.162},
        "customer": {"x": 0.054, "y": 0.162},
        "order_date": {"x": 0.110, "y": 0.162},
        "date": {"x": 0.110, "y": 0.162},
        "amount": {"x": 0.169, "y": 0.162},
        "total": {"x": 0.169, "y": 0.162},
        "status": {"x": 0.223, "y": 0.162},
    }
    return positions.get(lowered, {"x": 0.169, "y": 0.162})


def _filter_target_for_column(column: str) -> Dict[str, float]:
    """Return normalized center coordinates for a column filter box."""
    lowered = column.lower()
    positions = {
        "id": {"x": 0.026, "y": 0.195},
        "order_id": {"x": 0.026, "y": 0.195},
        "region": {"x": 0.060, "y": 0.195},
        "customer_id": {"x": 0.054, "y": 0.195},
        "customer": {"x": 0.054, "y": 0.195},
        "order_date": {"x": 0.110, "y": 0.195},
        "date": {"x": 0.110, "y": 0.195},
        "amount": {"x": 0.169, "y": 0.195},
        "total": {"x": 0.169, "y": 0.195},
        "status": {"x": 0.223, "y": 0.195},
    }
    return positions.get(lowered, {"x": 0.169, "y": 0.195})


def _match_recipe(objective: str, db_path: Optional[str] = None) -> Optional[List[Dict[str, Any]]]:
    """
    Match common DB Browser objectives to deterministic recipes.
    Returns None if no recipe matches (fall back to vision exploration).
    """
    lowered = objective.lower()

    # Explicit "sort X by Y [direction]"
    sort_match = re.search(
        r"sort\s+(?:the\s+)?(\w+)\s+(?:table\s+)?by\s+(?:the\s+)?(\w+)(?:\s+(?:column|header))?",
        lowered,
    )
    if sort_match:
        table = sort_match.group(1).capitalize()
        column = sort_match.group(2).capitalize()
        direction = "desc" if any(w in lowered for w in ("descending", "desc", "largest", "biggest", "highest")) else "asc"
        return DiscoveryRecipes.sort_column(table, column, direction)

    # "Click the X column header in the Y table ..."
    header_match = re.search(
        r"(?:the\s+)?(\w+)\s+(?:column\s+)?header\s+(?:in\s+(?:the\s+)?)?(\w+)\s+(?:table)?",
        lowered,
    )
    if header_match:
        column = header_match.group(1).capitalize()
        table = header_match.group(2).capitalize()
        direction = "desc" if any(
            w in lowered for w in ("descending", "desc", "largest", "biggest", "highest", "second time", "again")
        ) else "asc"
        return DiscoveryRecipes.sort_column(table, column, direction)

    # "Filter X by typing Y into the Z column filter box"
    filter_match = re.search(
        r"filter\s+(?:the\s+)?(\w+)\s+(?:table\s+)?by\s+(?:typing\s+)?(.+?)\s+(?:into|in)\s+(?:the\s+)?(\w+)\s+(?:column\s+)?filter",
        lowered,
    )
    if filter_match:
        table = filter_match.group(1).capitalize()
        column = filter_match.group(3).capitalize()
        # Preserve the original casing of the typed value from the objective.
        orig_match = re.search(
            r"filter\s+(?:the\s+)?(\w+)\s+(?:table\s+)?by\s+(?:typing\s+)?(.+?)\s+(?:into|in)\s+(?:the\s+)?(\w+)\s+(?:column\s+)?filter",
            objective,
            re.IGNORECASE,
        )
        value = (orig_match.group(2) if orig_match else filter_match.group(2)).strip().strip("'\"")
        return DiscoveryRecipes.filter_column(table, column, value)

    # "Run a SELECT ... query in the Execute SQL tab" / "execute sql query"
    if any(phrase in lowered for phrase in ("execute sql", "select ", "run a query", "query in the execute sql")):
        query = _extract_query_from_objective(objective)
        return DiscoveryRecipes.execute_query(query)

    # "Open/browse/show/display the X table"
    browse_match = re.search(
        r"(?:open|browse|show|display)\s+(?:the\s+)?(\w+)\s+(?:table)",
        lowered,
    )
    if browse_match:
        return DiscoveryRecipes.browse_table(browse_match.group(1).capitalize(), db_path)

    return None


def _describe_recipe_outcome(objective: str) -> Optional[str]:
    """Return a deterministic human-readable summary for known recipe objectives."""
    lowered = objective.lower()

    sort_match = re.search(
        r"sort\s+(?:the\s+)?(\w+)\s+(?:table\s+)?by\s+(?:the\s+)?(\w+)",
        lowered,
    )
    if sort_match:
        table = sort_match.group(1).capitalize()
        column = sort_match.group(2).capitalize()
        direction = (
            "descending"
            if any(w in lowered for w in ("descending", "desc", "largest", "biggest", "highest"))
            else "ascending"
        )
        return f"The {table} table is sorted by the {column} column in {direction} order."

    filter_match = re.search(
        r"filter\s+(?:the\s+)?(\w+)\s+(?:table\s+)?by\s+(?:typing\s+)?(.+?)\s+(?:into|in)\s+(?:the\s+)?(\w+)\s+(?:column\s+)?filter",
        lowered,
    )
    if filter_match:
        table = filter_match.group(1).capitalize()
        column = filter_match.group(3).capitalize()
        value = filter_match.group(2).strip().strip("'\"")
        return f"The {table} table is filtered to show only rows where {column} is {value}."

    if any(phrase in lowered for phrase in ("execute sql", "select ", "run a query", "query in the execute sql")):
        query = _extract_query_from_objective(objective)
        return f"Executed query '{query}' in the Execute SQL tab and displayed the result grid."

    browse_match = re.search(r"(?:open|browse|show|display)\s+(?:the\s+)?(\w+)\s+(?:table)", lowered)
    if browse_match:
        return f"The {browse_match.group(1).capitalize()} table is open in the Browse Data tab."

    return None


def _is_browse_objective(objective: str) -> bool:
    """Return True if the objective is a simple table-browse recipe."""
    lowered = objective.lower()
    return bool(
        re.search(r"(?:open|browse|show|display)\s+(?:the\s+)?(\w+)\s+(?:table)", lowered)
    )


def _is_execute_query_objective(objective: str) -> bool:
    """Return True if the objective is an Execute SQL query recipe."""
    lowered = objective.lower()
    return any(
        phrase in lowered
        for phrase in ("execute sql", "select ", "run a query", "query in the execute sql")
    )

# Approximate Claude Sonnet vision API pricing (USD per token).
# These are estimates; the harness uses them for cost telemetry only.
INPUT_PRICE_PER_TOKEN = 3.0 / 1_000_000
OUTPUT_PRICE_PER_TOKEN = 15.0 / 1_000_000

PROMPT_TEMPLATE = """You are evaluating whether the current screen state satisfies the learning objective.

Objective: {objective}

Look at the screenshot. Does this screen state satisfy the objective?
Respond in exactly this format on the first line:
YES: <concise reason>
or
NO: <concise reason>

If NO, also provide exactly one next UI action to move toward the objective as JSON on its own line:
{{"action": "click", "point": {{"x": int, "y": int}}, "element_type": "tab|button|column_header|table_cell|icon|menu_item|other", "description": "brief label of what was clicked"}}
{{"action": "type", "point": {{"x": int, "y": int}}, "text": "...", "element_type": "table_cell|filter_box|other", "description": "brief label of where to type"}}
{{"action": "key", "text": "..."}}  (examples: "Return", "Tab", "cmd+f")
{{"action": "wait", "duration": 1}}

Rules for actions:
- Use 1-3 plain left-clicks per step. NO shift-click, ctrl-click, or drag.
- For filter/search tasks, use a multi-step sequence: click the input box, type the value, then press Return or click an apply button. Return each step as a separate action.
- The point must be the CENTER of the clickable UI element, measured in the provided screenshot coordinate space (top-left is 0,0; x increases to the right, y increases downward).
- You do NOT need to return a bounding box; just return the center point and element type.

Example sequence for "Click the filter box under the region column and type 'West'":
Step 1 (current view shows empty filter box): {{"action": "click", "point": {{"x": 300, "y": 200}}, "element_type": "filter_box", "description": "region filter box"}}
Step 2 (filter box is now focused): {{"action": "type", "point": {{"x": 300, "y": 200}}, "text": "West", "element_type": "filter_box", "description": "region filter box"}}
Step 3 (text "West" is in the box but filter not applied): {{"action": "key", "text": "Return"}}

Do not add any other explanation.
"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _screenshots_similar(
    img1: Image.Image, img2: Image.Image, threshold: float = 0.98
) -> bool:
    """Return True if two PIL images are visually nearly identical."""
    if img1.size != img2.size:
        return False
    # Resize to small grayscale thumbnails for fast, robust comparison.
    small1 = img1.resize((64, 64), Image.Resampling.LANCZOS).convert("L")
    small2 = img2.resize((64, 64), Image.Resampling.LANCZOS).convert("L")
    pixels1 = list(small1.getdata())
    pixels2 = list(small2.getdata())
    if len(pixels1) != len(pixels2):
        return False
    matching = sum(1 for a, b in zip(pixels1, pixels2) if abs(a - b) <= 5)
    return (matching / len(pixels1)) >= threshold


def _logical_screen_size() -> Tuple[int, int, float]:
    """Return (logical_width, logical_height, backing_scale_factor) for the main screen."""
    script = (
        "from AppKit import NSScreen\n"
        "s = NSScreen.mainScreen()\n"
        "f = s.frame()\n"
        "print(int(f.size.width), int(f.size.height), s.backingScaleFactor())"
    )
    try:
        out = subprocess.run(
            ["python3", "-c", script],
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        )
        w, h, scale = out.stdout.strip().split()
        return int(w), int(h), float(scale)
    except Exception as exc:
        raise RuntimeError(f"Could not determine screen size: {exc}")


def _find_db_browser(app_name: str = "DB Browser for SQLite") -> Optional[Path]:
    """Return the path to the target database browser app if it appears installed."""
    # Try AppleScript's canonical path lookup first.
    try:
        out = subprocess.run(
            [
                "osascript",
                "-e",
                f'POSIX path of (path to application "{app_name}")',
            ],
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        )
        path = Path(out.stdout.strip())
        if path.exists():
            return path
    except Exception:
        pass

    # Fall back to common install locations.
    candidates = [
        Path("/Applications") / f"{app_name}.app",
        Path.home() / "Applications" / f"{app_name}.app",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _ensure_sample_db(output_dir: Path) -> Path:
    """Create a minimal SQLite database with an Orders table if one does not exist."""
    db_path = output_dir / "sample.db"
    if db_path.exists():
        return db_path

    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()
    cursor.execute(
        """
        CREATE TABLE Orders (
            id INTEGER PRIMARY KEY,
            region TEXT,
            order_date TEXT,
            amount REAL
        )
        """
    )
    rows = [
        (1, "North", "2024-01-15", 120.50),
        (2, "South", "2024-02-10", 85.00),
        (3, "East", "2024-03-05", 210.25),
        (4, "West", "2024-01-22", 340.00),
        (5, "North", "2024-04-18", 95.75),   # Outside Q1 2024; useful for filter tasks.
        (6, "South", "2024-02-28", 150.00),
    ]
    cursor.executemany(
        "INSERT INTO Orders (id, region, order_date, amount) VALUES (?, ?, ?, ?)",
        rows,
    )
    conn.commit()
    conn.close()
    return db_path


def _list_tables(db_path: str) -> List[str]:
    """Return user tables in the SQLite database in schema order."""
    try:
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()
        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY rowid"
        )
        tables = [row[0] for row in cursor.fetchall()]
        conn.close()
        return tables
    except Exception as exc:
        print(f"Warning: could not list tables in {db_path}: {exc}", file=sys.stderr)
        return []


def _capture_screenshot(output_dir: Path) -> Tuple[str, int, int, float, Image.Image, bytes]:
    """
    Capture the screen, resize it for the vision API, and return:
      - base64 PNG string (API-sized)
      - API width / height
      - scale factor from API coordinates to macOS logical points
      - raw full-resolution PIL Image
      - raw full-resolution PNG bytes (for hashing and saving)
    """
    logical_w, logical_h, _ = _logical_screen_size()
    tmp_path = output_dir / f"_tmp_screenshot_{uuid.uuid4().hex}.png"
    try:
        subprocess.run(["screencapture", "-x", str(tmp_path)], check=True, timeout=30)
        raw_bytes = tmp_path.read_bytes()
        raw_img = Image.open(io.BytesIO(raw_bytes))
        raw_w, raw_h = raw_img.size

        long_edge = max(raw_w, raw_h)
        resize_scale = min(1.0, TARGET_LONG_EDGE / long_edge)
        api_w = int(raw_w * resize_scale)
        api_h = int(raw_h * resize_scale)
        api_img = raw_img.resize((api_w, api_h), Image.Resampling.LANCZOS)

        buf = io.BytesIO()
        api_img.save(buf, format="PNG")
        b64 = base64.standard_b64encode(buf.getvalue()).decode("utf-8")

        # API coordinate (x) * scale_to_logical = macOS logical point.
        scale_to_logical = logical_w / api_w
        return b64, api_w, api_h, scale_to_logical, raw_img, raw_bytes
    finally:
        try:
            tmp_path.unlink()
        except FileNotFoundError:
            pass


# ---------------------------------------------------------------------------
# Action execution (cliclick preferred, AppleScript fallback)
# ---------------------------------------------------------------------------


_KEY_MAP = {
    "return": "kp:return",
    "enter": "kp:return",
    "tab": "kp:tab",
    "escape": "kp:esc",
    "esc": "kp:esc",
    "delete": "kp:delete",
    "backspace": "kp:delete",
    "space": "kp:space",
    "up": "kp:arrow-up",
    "down": "kp:arrow-down",
    "left": "kp:arrow-left",
    "right": "kp:arrow-right",
    "home": "kp:home",
    "end": "kp:end",
}

_MOD_MAP = {
    "shift": "shift",
    "ctrl": "ctrl",
    "control": "ctrl",
    "alt": "alt",
    "option": "alt",
    "super": "cmd",
    "cmd": "cmd",
    "command": "cmd",
}

_APPLESCRIPT_MOD_MAP = {
    "shift": "shift",
    "ctrl": "control",
    "control": "control",
    "alt": "option",
    "option": "option",
    "super": "command",
    "cmd": "command",
    "command": "command",
}

_APPLESCRIPT_KEY_CODE = {
    "return": 36,
    "enter": 36,
    "tab": 48,
    "escape": 53,
    "esc": 53,
    "delete": 51,
    "backspace": 51,
    "space": 49,
    "up": 126,
    "down": 125,
    "left": 123,
    "right": 124,
}


def _run_cliclick(*args: str) -> None:
    subprocess.run(["cliclick", *args], check=True, capture_output=True, timeout=30)


def _run_applescript(source: str) -> None:
    subprocess.run(["osascript", "-e", source], check=True, capture_output=True, timeout=30)


def _has_cliclick() -> bool:
    return shutil.which("cliclick") is not None


def _click(x: float, y: float) -> None:
    if _has_cliclick():
        _run_cliclick(f"c:{x:.0f},{y:.0f}")
    else:
        _run_applescript(f'tell application "System Events" to click at {{{x:.0f},{y:.0f}}}')


def _double_click(x: float, y: float) -> None:
    if _has_cliclick():
        _run_cliclick(f"dc:{x:.0f},{y:.0f}")
    else:
        _run_applescript(
            f'tell application "System Events" to double click at {{{x:.0f},{y:.0f}}}'
        )


def _type_text(text: str) -> None:
    if _has_cliclick():
        _run_cliclick(f"t:{text}")
    else:
        # Escape double quotes for AppleScript.
        safe = text.replace('"', '\\"')
        _run_applescript(f'tell application "System Events" to keystroke "{safe}"')


def _type_text_pyg(text: str, interval: float = 0.01) -> None:
    """Type text using pyautogui, which correctly handles newlines and special characters."""
    pyautogui.typewrite(text, interval=interval)


def _press_key(key_str: str) -> None:
    parts = [p.strip() for p in key_str.split("+")]
    modifiers = [p.lower() for p in parts[:-1] if p.lower() in _MOD_MAP]
    base = parts[-1]

    if _has_cliclick():
        mod_tokens = [_MOD_MAP[m] for m in modifiers]
        if mod_tokens:
            _run_cliclick("kd:" + ",".join(mod_tokens))
        if base.lower() in _KEY_MAP:
            _run_cliclick(_KEY_MAP[base.lower()])
        elif len(base) == 1:
            _run_cliclick(f"t:{base}")
        else:
            _run_cliclick(f"kp:{base.lower()}")
        if mod_tokens:
            _run_cliclick("ku:" + ",".join(mod_tokens))
        return

    # AppleScript fallback.
    mod_list = ", ".join(f"{_APPLESCRIPT_MOD_MAP[m]} down" for m in modifiers)
    base_lower = base.lower()
    if base_lower in _APPLESCRIPT_KEY_CODE:
        code = _APPLESCRIPT_KEY_CODE[base_lower]
        if mod_list:
            _run_applescript(
                f'tell application "System Events" to key code {code} using {{{mod_list}}}'
            )
        else:
            _run_applescript(f'tell application "System Events" to key code {code}')
    else:
        safe = base.replace('"', '\\"')
        if mod_list:
            _run_applescript(
                f'tell application "System Events" to keystroke "{safe}" using {{{mod_list}}}'
            )
        else:
            _run_applescript(f'tell application "System Events" to keystroke "{safe}"')


def _execute_action(action: Dict[str, Any], scale_to_logical: float) -> str:
    """Execute a single parsed action and return a human-readable description."""
    name = action.get("action")

    if name in ("click", "double_click"):
        if "point" in action:
            point = action["point"]
            x, y = point["x"], point["y"]
        elif "bbox" in action:
            bbox = action["bbox"]
            x = bbox["x"] + bbox["w"] / 2
            y = bbox["y"] + bbox["h"] / 2
        elif "coordinate" in action:
            x, y = action["coordinate"]
        else:
            raise ValueError(f"Click action missing point, bbox, or coordinate: {action}")
        lx, ly = x * scale_to_logical, y * scale_to_logical
        if action.get("animate"):
            animate_cursor_to(lx, ly, duration=action.get("animate_duration", 0.6))
        if name == "click":
            _click(lx, ly)
            return f"click at ({lx:.0f}, {ly:.0f})"
        _double_click(lx, ly)
        return f"double-click at ({lx:.0f}, {ly:.0f})"

    if name == "type":
        text = action["text"]
        # Click the target first to focus it, then type the text, unless the
        # recipe explicitly asks to type without clicking (e.g. type-ahead in
        # an already-open dropdown menu).
        if action.get("click_first", True) and ("point" in action or "bbox" in action or "coordinate" in action):
            if "point" in action:
                point = action["point"]
                x, y = point["x"], point["y"]
            elif "bbox" in action:
                bbox = action["bbox"]
                x = bbox["x"] + bbox["w"] / 2
                y = bbox["y"] + bbox["h"] / 2
            else:
                x, y = action["coordinate"]
            lx, ly = x * scale_to_logical, y * scale_to_logical
            if action.get("animate"):
                animate_cursor_to(lx, ly, duration=action.get("animate_duration", 0.6))
            _click(lx, ly)
            time.sleep(0.2)
            desc = f"type {text!r} at ({lx:.0f}, {ly:.0f})"
        else:
            desc = f"type {text!r}"
        # Multi-line or long text (e.g. formatted SQL) is unreliable with cliclick/
        # AppleScript keystroke because newlines and special characters may be
        # mangled or interpreted as shortcuts. Use pyautogui for those cases.
        if "\n" in text or len(text) > 200:
            _type_text_pyg(text)
        else:
            segments = text.split("\n")
            for idx, segment in enumerate(segments):
                if segment:
                    _type_text(segment)
                if idx < len(segments) - 1:
                    _press_key("Return")
        return desc

    if name == "key":
        key = action["text"]
        _press_key(key)
        return f"press {key!r}"

    if name == "wait":
        duration = action.get("duration", 1)
        time.sleep(duration)
        return f"wait {duration}s"

    raise ValueError(f"Unsupported action: {name!r}")


# ---------------------------------------------------------------------------
# Model interaction
# ---------------------------------------------------------------------------


def _extract_action(text: str) -> Optional[Dict[str, Any]]:
    """Try to pull a single JSON action object out of the model response."""
    # Look for a fenced JSON block first.
    fenced = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, re.DOTALL)
    if fenced:
        try:
            parsed = json.loads(fenced.group(1))
            if isinstance(parsed, dict) and "action" in parsed:
                return parsed
        except json.JSONDecodeError:
            pass

    # Find all standalone JSON objects and pick the first one that looks like
    # an action. Using a non-greedy match prevents spanning multiple objects.
    for match in re.finditer(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", text):
        try:
            parsed = json.loads(match.group(0))
            if isinstance(parsed, dict) and "action" in parsed:
                return parsed
        except json.JSONDecodeError:
            continue

    return None


def _extract_type_values(objective: str) -> List[str]:
    """
    Parse the value to type from the objective.
    Handles quoted strings and bare values before a preposition.
    Examples:
      "type 'West'" -> ["West"]
      "type \"hello world\"" -> ["hello world"]
      "type Shipped into the filter box" -> ["Shipped"]
      "type >200 into the filter box" -> [">200"]
    """
    values: List[str] = []
    # Quoted values first.
    for match in re.finditer(r"type\s+(['\"])(.+?)\1", objective, re.IGNORECASE):
        values.append(match.group(2))
    # Bare value followed by a preposition like "into", "in", "to".
    for match in re.finditer(
        r"type\s+(?!['\"])([^\s,;]+)(?:\s+(?:into|in|to|under|for|on))",
        objective,
        re.IGNORECASE,
    ):
        values.append(match.group(1))
    return values


def _extract_query_from_objective(objective: str) -> str:
    """
    Extract a raw SELECT query from the objective, or return a safe default.
    Examples:
      "Run a SELECT * FROM Orders query..." -> "SELECT * FROM Orders"
      "execute SELECT id, amount FROM Orders" -> "SELECT id, amount FROM Orders"
    """
    query = extract_first_query(objective)
    if query:
        # Remove any trailing prose punctuation/semicolon normalization is left
        # to the formatter in execute_query.
        return query.rstrip(";").strip()
    # No SELECT statement found; try to infer a table from "from X".
    table_match = re.search(r"\bfrom\s+(\w+)", objective, re.IGNORECASE)
    table = table_match.group(1) if table_match else "Orders"
    return f"SELECT * FROM {table}"


def _extract_sql_query(beat: "ScriptBeat") -> Optional[str]:
    """
    Pull the SQL query associated with a beat, including ORDER BY and LIMIT.

    Prefer an explicit type_block or type_segments action, then fall back to a
    SELECT statement embedded in the beat narration only when the beat is a
    run_query beat. Prose explain/state/concept beats frequently mention SQL
    keywords and must not be executed as queries.
    """
    candidates: List[str] = []
    action = beat.action or {}
    if action.get("type") == "type_block":
        candidates.append(action.get("text") or action.get("detail") or "")
    elif action.get("type") == "type_segments":
        segments = action.get("segments") or []
        candidates.append(
            "".join(
                (seg.get("text", "") if isinstance(seg, dict) else str(seg))
                for seg in segments
            )
        )
    # Only treat narration as a runnable query for actual run_query/demo beats.
    if action.get("type") == "run_query":
        candidates.append(beat.text)

    for candidate in candidates:
        query = extract_first_query(candidate)
        if query:
            return query
    return None


def _execute_query_ground_truth(db_path: str, query: str) -> Dict[str, Any]:
    """Run a SELECT query through sqlite3 and return a structured summary."""
    result: Dict[str, Any] = {"columns": [], "row_count": 0, "first_rows": []}
    try:
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()
        cursor.execute(query)
        rows = cursor.fetchall()
        result["columns"] = [desc[0] for desc in cursor.description] if cursor.description else []
        result["row_count"] = len(rows)
        result["first_rows"] = [list(row) for row in rows[:5]]
        conn.close()
    except Exception as exc:
        print(f"Warning: could not ground query against {db_path}: {exc}", file=sys.stderr)
        result["error"] = str(exc)
    return result


def _load_canonical_grounding(video_id: Optional[str]) -> Dict[str, Any]:
    """Load expected rows/first row/headers from the canonical reference file."""
    result: Dict[str, Any] = {}
    if not video_id:
        return result
    ref_path = Path(__file__).resolve().parent / "courses" / "ch4_canonical_reference.md"
    if not ref_path.exists():
        return result
    text = ref_path.read_text(encoding="utf-8")
    video_num = video_id.replace("video_", "").split("_")[0]
    pattern = re.compile(
        rf"## Video\s+{re.escape(video_num)}\b.*?```sql\n(.*?)```",
        re.DOTALL,
    )
    match = pattern.search(text)
    if not match:
        return result
    block = match.group(0)
    row_match = re.search(r"Expected result:\s*\*\*(\d+)\s+rows?\*\*", block, re.IGNORECASE)
    if row_match:
        result["expected_rows"] = int(row_match.group(1))
    first_match = re.search(r"first row\s+\*\*(.+?)\*\*", block, re.IGNORECASE)
    if first_match:
        result["expected_first_row"] = first_match.group(1).strip()
    headers_match = re.search(r"columns?\s+(.+?)\.", block, re.IGNORECASE)
    if headers_match:
        header_text = headers_match.group(1)
        result["expected_headers"] = [
            h.strip() for h in re.split(r"[,|]", header_text) if h.strip()
        ]
    return result


def _ground_query_result(
    beat: "ScriptBeat",
    db_path: Optional[str],
    video_id: Optional[str] = None,
) -> bool:
    """
    Compare a beat's SQL result to the actual sqlite3 result.

    Returns True only when:
      - sqlite3 executes the beat's SQL without error.
      - The VLM-reported row count and columns match sqlite3 (when available).
      - Any row-count expectation declared in the beat text matches sqlite3.
      - Canonical reference expectations (rows, first row, headers) match.

    Any mismatch or sqlite3 error is a FAIL. The sqlite3 result is stored as
    the authoritative ground truth, but GROUNDING OK is never printed on failure.
    """
    query = _extract_sql_query(beat)
    if not query or not db_path or not Path(db_path).exists():
        return False

    vlm_result = (beat.observed_state or {}).get("query_result") or {}
    ground = _execute_query_ground_truth(db_path, query)

    if beat.observed_state is None:
        beat.observed_state = {}
    beat.observed_state["query_result"] = ground

    if ground.get("error"):
        print(
            f"[GROUNDING FAIL] sqlite3 error for beat {beat.beat_id}: {ground['error']}",
            file=sys.stderr,
        )
        return False

    ground_count = ground.get("row_count")
    ground_columns = list(ground.get("columns") or [])

    vlm_count = vlm_result.get("row_count")
    vlm_columns = list(vlm_result.get("columns") or [])

    count_mismatch = False
    try:
        if isinstance(vlm_count, str) and "of" in vlm_count.lower():
            vlm_count_int = int(vlm_count.lower().split("of")[0].strip())
        else:
            vlm_count_int = int(vlm_count) if vlm_count is not None else None
        count_mismatch = vlm_count_int is not None and vlm_count_int != ground_count
    except Exception:
        pass

    column_mismatch = bool(vlm_columns) and [
        c.lower() for c in vlm_columns
    ] != [c.lower() for c in ground_columns]

    if count_mismatch or column_mismatch:
        print(
            f"[GROUNDING FAIL] sqlite3={ground_count} rows, VLM={vlm_count} rows "
            f"(columns sqlite3={ground_columns}, VLM={vlm_columns})",
            file=sys.stderr,
        )
        return False

    # Assert manifest-declared expectations from the beat narration.
    expected = _extract_expected_row_count(beat.text)
    if expected is not None and expected != ground_count:
        print(
            f"[GROUNDING FAIL] beat expects {expected} rows, sqlite3 returned {ground_count}",
            file=sys.stderr,
        )
        return False

    # C9: assert canonical reference expectations (rows, first row, headers).
    canonical = _load_canonical_grounding(video_id)
    if canonical.get("expected_rows") is not None:
        if ground_count != canonical["expected_rows"]:
            print(
                f"[GROUNDING FAIL] canonical reference expects {canonical['expected_rows']} rows, "
                f"sqlite3 returned {ground_count}",
                file=sys.stderr,
            )
            return False
    if canonical.get("expected_first_row") and ground.get("first_rows"):
        first_row = ground["first_rows"][0]
        first_row_str = " ".join(str(c) for c in first_row[:3])
        expected_first = canonical["expected_first_row"]
        if expected_first.lower() not in first_row_str.lower():
            print(
                f"[GROUNDING FAIL] canonical first row mismatch: expected {expected_first!r}, "
                f"sqlite3 returned {first_row!r}",
                file=sys.stderr,
            )
            return False
    if canonical.get("expected_headers") and ground_columns:
        if [c.strip() for c in ground_columns] != [c.strip() for c in canonical["expected_headers"]]:
            print(
                f"[GROUNDING FAIL] canonical header mismatch: expected {canonical['expected_headers']!r}, "
                f"sqlite3 returned {ground_columns!r}",
                file=sys.stderr,
            )
            return False

    print(
        f"[GROUNDING OK] sqlite3={ground_count} rows, columns={ground_columns}",
        file=sys.stderr,
    )
    return True


def _extract_expected_row_count(text: str) -> Optional[int]:
    """Return the first integer followed by 'row' or 'rows' in ``text``, if any."""
    match = re.search(r"\b(\d+)\s+rows?\b", text, re.IGNORECASE)
    if match:
        return int(match.group(1))
    return None


def _parse_row_count(value: Any) -> Optional[int]:
    """Normalize a row_count value from a result summary to an int."""
    if value is None:
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        lowered = value.lower().strip()
        if "of" in lowered:
            try:
                return int(lowered.split("of")[0].strip())
            except Exception:
                return None
        try:
            return int(lowered)
        except Exception:
            return None
    return None


# Programmatic bounding box dimensions (in API screenshot pixels) for each
# element type.  The vision model returns only a center point; we expand it to
# a full clickable area centered on that point.
_ELEMENT_BBOX_SIZES: Dict[str, Tuple[int, int]] = {
    "tab": (120, 30),
    "button": (120, 30),
    "column_header": (130, 28),
    "table_cell": (100, 24),
    "filter_box": (120, 28),
    "icon": (40, 40),
    "menu_item": (150, 25),
    "other": (120, 30),
}


def _infer_element_type(action: Dict[str, Any]) -> str:
    """
    Determine the element type for a click action.
    Prefer an explicit element_type from the model, then fall back to heuristics.
    """
    explicit = str(action.get("element_type", "")).lower().strip()
    if explicit in _ELEMENT_BBOX_SIZES:
        return explicit

    point = action.get("point") or action
    y = point.get("y", 0) if isinstance(point, dict) else 0
    description = str(action.get("description", "")).lower()

    # Heuristic: top-of-screen short labels are usually tabs or buttons.
    if y < 200:
        return "tab" if "tab" in description or "browse" in description else "button"

    # Heuristic: column headers sit in the table header row, usually below tabs.
    if "column" in description or "header" in description or 200 <= y <= 260:
        return "column_header"

    # Heuristic: grid area below the header is table cells.
    if y > 260:
        return "table_cell"

    return "other"


def _center_to_bbox(point: Dict[str, Any], element_type: str) -> Dict[str, int]:
    """Expand a center point into a full bounding box based on element type."""
    cx = int(round(float(point.get("x", 0))))
    cy = int(round(float(point.get("y", 0))))
    w, h = _ELEMENT_BBOX_SIZES.get(element_type, _ELEMENT_BBOX_SIZES["other"])
    return {
        "x": cx - w // 2,
        "y": cy - h // 2,
        "w": w,
        "h": h,
    }


def _expand_click_action(action: Dict[str, Any]) -> Dict[str, Any]:
    """
    Given a click action with a center point, add a programmatic full-element
    bounding box and return the action with both 'point' and 'bbox'.
    """
    action = dict(action)
    point = action.get("point") or action
    if not isinstance(point, dict) or "x" not in point or "y" not in point:
        raise ValueError(f"Click action missing center point: {action}")

    element_type = _infer_element_type(action)
    bbox = _center_to_bbox(point, element_type)

    action["element_type"] = element_type
    action["point"] = {"x": point.get("x"), "y": point.get("y")}
    action["bbox"] = bbox
    return action


def _call_vision_model(
    client: anthropic.Anthropic,
    objective: str,
    b64_image: str,
) -> Tuple[bool, str, Optional[Dict[str, Any]], int, int, float]:
    """
    Ask the model whether the screenshot satisfies the objective.

    The model returns a center point and element type for click actions; this
    function expands the point into a full programmatic bounding box.

    Returns:
      success: True if the model answered YES.
      reason: the model's reason string.
      action: the next action dict (only when success is False), with a full bbox.
      input_tokens, output_tokens, latency_seconds
    """
    prompt = PROMPT_TEMPLATE.format(objective=objective)
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": b64_image}},
                {"type": "text", "text": prompt},
            ],
        }
    ]

    started = time.time()
    response = tracked_create(
        client,
        model=MODEL,
        max_tokens=1024,
        messages=messages,
    )
    latency = time.time() - started

    text_parts = [block.text for block in response.content if block.type == "text"]
    full_text = "\n".join(text_parts).strip()

    # Parse yes/no from the response. Prefer an explicit YES:/NO: line.
    lines = [line.strip() for line in full_text.splitlines() if line.strip()]
    yes_no_line = next(
        (
            line
            for line in lines
            if line.upper().startswith("YES:") or line.upper().startswith("NO:")
        ),
        "",
    )
    if yes_no_line:
        upper = yes_no_line.upper()
        success = upper.startswith("YES")
        reason = yes_no_line.split(":", 1)[1].strip()
    else:
        # The model did not follow the YES/NO format; treat as failure.
        success = False
        reason = "Model did not respond with YES/NO format."

    action = None if success else _extract_action(full_text)
    if action and action.get("action") in ("click", "double_click", "type"):
        action = _expand_click_action(action)

    usage = getattr(response, "usage", None) or {}
    input_tokens = int(getattr(usage, "input_tokens", None) or usage.get("input_tokens", 0))
    output_tokens = int(getattr(usage, "output_tokens", None) or usage.get("output_tokens", 0))

    return success, reason, action, input_tokens, output_tokens, latency


# ---------------------------------------------------------------------------
# EndStateDiscovery
# ---------------------------------------------------------------------------


class EndStateDiscovery:
    """
    Given a learning objective and target application, explore the application
    using a vision model (Claude computer-use API), lock the first screen state
    that satisfies the objective, and return a DiscoveryResult with telemetry.
    """

    def __init__(
        self,
        objective: str,
        application: str,
        db_path: Optional[str] = None,
        opening_state_query: Optional[str] = None,
        profile: Optional[EnvironmentProfile] = None,
        video_id: Optional[str] = None,
    ):
        self.objective = objective
        self.application = application
        self.db_path = Path(db_path) if db_path else None
        self.opening_state_query = opening_state_query
        self.profile = profile or self._default_profile_for_app(application)
        self.video_id = video_id
        self.client = anthropic.Anthropic()
        self.output_dir = Path(__file__).resolve().parent / "discovery_output"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.telemetry_path = self.output_dir / "telemetry.jsonl"
        self.attempt_logs: List[Dict[str, Any]] = []

    @staticmethod
    def _default_profile_for_app(application: str) -> EnvironmentProfile:
        """Fallback profile for callers that do not inject one."""
        if application == "db_browser_sqlite":
            return EnvironmentProfile(
                application=application,
                app_name="DB Browser for SQLite",
                focus_target="DB Browser for SQLite",
                window_title_hint="DB Browser for SQLite",
                landmarks={
                    "editor": "the editable SQL text area in the Execute SQL tab",
                    "run_button": "the Execute SQL toolbar button (blue play triangle / right-pointing arrow icon) above the SQL editor",
                    "result_pane": "the lower result pane showing query output",
                    "result_tab": "the Result tab in the lower results pane",
                    "execute_tab": "Execute SQL tab",
                    "browse_tab": "Browse Data tab",
                },
                grounding_channel={"type": "sqlite3"},
                action_vocabulary=[
                    "click", "type", "type_block", "append_block", "type_segments",
                    "key", "run_query", "wait", "verify", "scroll",
                ],
            )
        return EnvironmentProfile(
            application=application,
            app_name=application,
            focus_target=application,
            landmarks={},
            grounding_channel={},
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def _execute_action_script(
        self,
        actions: List[Dict[str, Any]],
        run_id: str,
        save_all_screenshots: bool,
    ) -> Dict[str, Any]:
        """
        Execute a list of UI actions, recording a video clip per action and
        logging telemetry. Returns a dict with the final capture data and
        execution metadata.
        """
        last_action: Optional[str] = None
        last_target: Optional[Dict[str, Any]] = None
        prev_video_path: Optional[str] = None

        # Ceiling: one iteration per action (bounded by len(actions)).
        for step_idx, step in enumerate(actions):
            attempt = step_idx + 1
            try:
                b64, api_w, api_h, scale_to_logical, raw_img, raw_bytes = _capture_screenshot(
                    self.output_dir
                )
            except Exception as exc:
                msg = f"Screenshot capture failed: {exc}"
                print(f"Error: {msg}", file=sys.stderr)
                return {"success": False, "reason": msg}

            screenshot_hash = hashlib.sha256(raw_bytes).hexdigest()
            if save_all_screenshots:
                hash_path = self.output_dir / f"{screenshot_hash}.png"
                raw_img.save(hash_path)

            # Log the state BEFORE executing this step's action.
            self._log_attempt(
                run_id=run_id,
                attempt=attempt,
                success=False,
                reason=f"Action step {attempt}: {step.get('description', '')}",
                action=last_action,
                target=last_target,
                screenshot_hash=screenshot_hash,
                tokens=(0, 0),
                cost=0.0,
                latency=0.0,
                error=None,
                api_width_px=api_w,
                api_height_px=api_h,
                video_path=prev_video_path,
            )

            # Build the action with API-pixel coordinates.
            target = step.get("target", {})
            cx_norm = float(target.get("x", 0.5))
            cy_norm = float(target.get("y", 0.5))
            w = float(target.get("w", 120))
            h = float(target.get("h", 30))

            cx_api = cx_norm * api_w
            cy_api = cy_norm * api_h
            bbox_api = {
                "x": int(round(cx_api - w / 2)),
                "y": int(round(cy_api - h / 2)),
                "w": int(round(w)),
                "h": int(round(h)),
            }

            action: Dict[str, Any] = {
                "action": step.get("action", "click"),
                "point": {"x": int(round(cx_api)), "y": int(round(cy_api))},
                "bbox": bbox_api,
                "description": step.get("description", ""),
            }
            if action["action"] in ("type", "key"):
                action["text"] = step.get("text", "")

            fallback_targets = step.get("fallback_targets", [])
            enforce_screen_change = action["action"] in ("click", "double_click") and bool(fallback_targets)
            targets_to_try = [bbox_api]
            if enforce_screen_change:
                for fb in fallback_targets:
                    fb_api = {
                        "x": int(round(float(fb.get("x", 0.5)) * api_w)),
                        "y": int(round(float(fb.get("y", 0.5)) * api_h)),
                        "w": int(round(float(fb.get("w", 120)))),
                        "h": int(round(float(fb.get("h", 30)))),
                    }
                    targets_to_try.append(fb_api)

            step_video_path = self.output_dir / f"{run_id}_step_{step_idx}.mp4"
            recorder: Any = ScreenRecorder(str(step_video_path), fps=10)
            if self.profile and self.profile.app_name:
                recorder = _ScreenCaptureKitRecorder(
                    str(step_video_path), fps=10, app_name=self.profile.app_name
                )
            recorder.start()
            try:
                action_succeeded = False
                # Ceiling: primary + fallback_targets (bounded by len(targets_to_try)).
                for try_idx, try_target in enumerate(targets_to_try):
                    action["point"] = {
                        "x": try_target["x"] + try_target["w"] // 2,
                        "y": try_target["y"] + try_target["h"] // 2,
                    }
                    action["bbox"] = try_target

                    logical_x = action["point"]["x"] * scale_to_logical
                    logical_y = action["point"]["y"] * scale_to_logical
                    target_label = "primary" if try_idx == 0 else f"fallback_{try_idx}"
                    action_name = action.get("action", "click")
                    if action_name in ("click", "double_click"):
                        action_desc = f"Clicking at ({logical_x:.0f}, {logical_y:.0f})"
                    else:
                        action_desc = f"Executing {action_name} action"
                    print(
                        f"  [{target_label}] {action_desc} for: {step.get('description', '')}",
                        file=sys.stderr,
                    )

                    try:
                        last_action = _execute_action(action, scale_to_logical)
                        last_target = try_target
                    except Exception as exc:
                        msg = f"Action failed: {exc}"
                        print(f"Error: {msg}", file=sys.stderr)
                        self._log_attempt(
                            run_id=run_id,
                            attempt=attempt,
                            success=False,
                            reason="Action failed",
                            action=last_action,
                            target=last_target,
                            screenshot_hash=screenshot_hash,
                            tokens=(0, 0),
                            cost=0.0,
                            latency=0.0,
                            error=msg,
                            api_width_px=api_w,
                            api_height_px=api_h,
                        )
                        return {"success": False, "reason": msg}

                    wait_seconds = 1.5 if action["action"] != "wait" else float(action.get("duration", 1))
                    time.sleep(wait_seconds)

                    if enforce_screen_change:
                        print(
                            f"  Waited {wait_seconds}s, taking verification screenshot",
                            file=sys.stderr,
                        )
                        try:
                            _, _, _, _, raw_img_after, raw_bytes_after = _capture_screenshot(
                                self.output_dir
                            )
                        except Exception as exc:
                            msg = f"Post-action screenshot capture failed: {exc}"
                            print(f"Error: {msg}", file=sys.stderr)
                            return {"success": False, "reason": msg}

                        if _screenshots_similar(raw_img, raw_img_after):
                            print(
                                f"  Screen did not change after click; trying next target...",
                                file=sys.stderr,
                            )
                            continue
                        else:
                            print(
                                f"  Screen changed after click; action succeeded.",
                                file=sys.stderr,
                            )

                        raw_img = raw_img_after
                        raw_bytes = raw_bytes_after
                        action_succeeded = True
                        break
                    else:
                        print(
                            f"  Waited {wait_seconds}s (screen-change check skipped for routine step).",
                            file=sys.stderr,
                        )
                        action_succeeded = True
                        break

                if not action_succeeded:
                    msg = (
                        f"Warning: screen did not change after all click targets for "
                        f"'{step.get('description', '')}'; continuing on the assumption "
                        f"the UI was already in the target state."
                    )
                    print(msg, file=sys.stderr)
                    action_succeeded = True
            finally:
                recorder.stop()
                if action_succeeded:
                    prev_video_path = str(step_video_path.resolve())

        return {
            "success": True,
            "last_action": last_action,
            "last_target": last_target,
            "prev_video_path": prev_video_path,
        }

    def _run_recipe(
        self,
        recipe: List[Dict[str, Any]],
        run_id: str,
        save_all_screenshots: bool,
    ) -> DiscoveryResult:
        """Execute a deterministic recipe and lock the resulting end state."""
        run_start = time.time()
        self.attempt_logs = []

        result = self._execute_action_script(
            actions=recipe,
            run_id=run_id,
            save_all_screenshots=save_all_screenshots,
        )
        if not result.get("success"):
            return self._make_result(success=False, reason=result.get("reason", "unknown"))

        last_action = result.get("last_action")
        last_target = result.get("last_target")
        prev_video_path = result.get("prev_video_path")

        # Capture and lock the final end state.
        try:
            b64, api_w, api_h, scale_to_logical, raw_img, raw_bytes = _capture_screenshot(
                self.output_dir
            )
        except Exception as exc:
            msg = f"Final screenshot capture failed: {exc}"
            print(f"Error: {msg}", file=sys.stderr)
            return self._make_result(success=False, reason=msg)

        screenshot_hash = hashlib.sha256(raw_bytes).hexdigest()
        if save_all_screenshots:
            hash_path = self.output_dir / f"{screenshot_hash}.png"
            raw_img.save(hash_path)

        # Execute SQL recipes must show a populated results grid, not just the
        # empty "Results of the last executed statements" text. Retry F5 once.
        if _is_execute_query_objective(self.objective):
            if not self._results_grid_visible(b64):
                print(
                    "Warning: no results grid visible after query execution; retrying F5...",
                    file=sys.stderr,
                )
                _press_key("F5")
                time.sleep(1.5)
                try:
                    b64, api_w, api_h, scale_to_logical, raw_img, raw_bytes = _capture_screenshot(
                        self.output_dir
                    )
                except Exception as exc:
                    msg = f"Retry screenshot capture failed: {exc}"
                    print(f"Error: {msg}", file=sys.stderr)
                    return self._make_result(success=False, reason=msg)

                screenshot_hash = hashlib.sha256(raw_bytes).hexdigest()
                if save_all_screenshots:
                    hash_path = self.output_dir / f"{screenshot_hash}.png"
                    raw_img.save(hash_path)

                if not self._results_grid_visible(b64):
                    return self._make_result(
                        success=False,
                        reason="Execute SQL recipe did not display a results grid",
                    )

        final_attempt = len(recipe) + 1
        self._log_attempt(
            run_id=run_id,
            attempt=final_attempt,
            success=True,
            reason=f"Recipe completed: {self.objective}",
            action=last_action,
            target=last_target,
            screenshot_hash=screenshot_hash,
            tokens=(0, 0),
            cost=0.0,
            latency=0.0,
            error=None,
            api_width_px=api_w,
            api_height_px=api_h,
            video_path=prev_video_path,
        )

        state_id = f"{self.application}_{uuid.uuid4().hex[:12]}"
        screenshot_path = self.output_dir / f"{state_id}.png"
        raw_img.save(screenshot_path)

        visual_summary = _describe_recipe_outcome(self.objective)
        if not visual_summary:
            visual_summary = f"Recipe completed: {self.objective}"

        screen_state = ScreenState(
            state_id=state_id,
            screenshot_path=str(screenshot_path.resolve()),
            timestamp=0.0,
            application=self.application,
            platform_snapshot={
                "api_width_px": api_w,
                "api_height_px": api_h,
                "scale_to_logical": scale_to_logical,
                "screenshot_sha256": screenshot_hash,
                "run_duration_seconds": round(time.time() - run_start, 3),
            },
            visual_summary=visual_summary,
        )

        print(f"End state discovered via recipe in {final_attempt} attempts: {self.objective}")
        return self._make_result(success=True, locked_state=screen_state, recipe=True)

    def discover(
        self,
        max_attempts: int = 10,
        cost_ceiling_usd: float = 5.0,
        save_all_screenshots: bool = False,
    ) -> DiscoveryResult:
        """
        1. Launch the target application (DB Browser for SQLite) with a known database.
        2. Run a setup step to auto-fit column widths if a table is visible.
        3. Use the computer-use API to explore the UI.
        4. At each step, capture a screenshot and ask a vision model whether the
           current screen satisfies the objective.
        5. If yes, save the screenshot and return a DiscoveryResult with the locked
           ScreenState.
        6. If no after max_attempts, return a DiscoveryResult with success=False.
        7. Track every attempt: tokens, cost, latency, success/failure, screenshot hash.

        When ``save_all_screenshots`` is True, every captured screenshot is saved to
        ``discovery_output/<sha256>.png`` so the pathfinder can reconstruct the full
        state sequence.
        """
        self.attempt_logs = []
        run_id = uuid.uuid4().hex[:12]

        # -- Validate prerequisites --------------------------------------
        errors: List[str] = []

        if not os.environ.get("ANTHROPIC_API_KEY"):
            errors.append("ANTHROPIC_API_KEY environment variable is not set.")

        if self.application not in SUPPORTED_APPLICATIONS:
            errors.append(
                f"Application {self.application!r} is not supported. "
                f"Supported: {', '.join(sorted(SUPPORTED_APPLICATIONS))}"
            )

        db_browser_path = _find_db_browser()
        if self.application == "db_browser_sqlite" and db_browser_path is None:
            errors.append(f"{APP_NAME} does not appear to be installed.")

        if errors:
            for msg in errors:
                print(f"Error: {msg}", file=sys.stderr)
            return self._make_result(success=False, reason="; ".join(errors))

        # -- Launch application with the requested database --------------
        try:
            if self.db_path:
                db_path = self.db_path
                if not db_path.exists():
                    raise FileNotFoundError(f"Requested database not found: {db_path}")
            else:
                db_path = _ensure_sample_db(self.output_dir)
            self._launch_app(db_path)
        except Exception as exc:
            msg = f"Failed to launch {APP_NAME}: {exc}"
            print(f"Error: {msg}", file=sys.stderr)
            return self._make_result(success=False, reason=msg)

        # -- Setup: auto-fit column widths if a table is visible -----------
        self._auto_fit_columns()

        # -- Use a deterministic recipe if the objective matches a known pattern.
        recipe = _match_recipe(self.objective, str(db_path) if db_path else None)
        if recipe:
            print(f"Using deterministic recipe for objective: {self.objective}")
            recipe_result = self._run_recipe(
                recipe=recipe,
                run_id=run_id,
                save_all_screenshots=save_all_screenshots,
            )
            if recipe_result.success:
                return recipe_result
            recipe_reason = (
                recipe_result.attempt_logs[-1].get("reason")
                if recipe_result.attempt_logs
                else "unknown"
            )
            print(
                f"Recipe did not reach the objective ({recipe_reason}). "
                f"Falling back to vision exploration.",
                file=sys.stderr,
            )

        # -- Exploration loop (vision-based fallback) --------------------
        total_cost = 0.0
        run_start = time.time()
        last_action: Optional[str] = None
        last_target: Optional[Dict[str, Any]] = None
        type_values = _extract_type_values(self.objective)
        typed_values: set = set()
        forced_action: Optional[Dict[str, Any]] = None
        forced_reason = ""
        force_return_after_type = False

        def _looks_like_text_input(action: Dict[str, Any]) -> bool:
            elem_type = str(action.get("element_type", "")).lower()
            desc = str(action.get("description", "")).lower()
            return (
                elem_type in ("filter_box", "table_cell")
                or "filter" in desc
                or "input" in desc
                or "search" in desc
                or ("box" in desc and "filter" in self.objective.lower())
            )

        # Ceiling: max_attempts (default 10) exploration steps.
        for attempt in range(1, max_attempts + 1):
            if total_cost >= cost_ceiling_usd:
                print(
                    f"Cost ceiling ${cost_ceiling_usd:.2f} reached after "
                    f"${total_cost:.4f}; stopping discovery.",
                    file=sys.stderr,
                )
                break

            try:
                b64, api_w, api_h, scale_to_logical, raw_img, raw_bytes = _capture_screenshot(
                    self.output_dir
                )
                screenshot_hash = hashlib.sha256(raw_bytes).hexdigest()
                if save_all_screenshots:
                    hash_path = self.output_dir / f"{screenshot_hash}.png"
                    raw_img.save(hash_path)
            except Exception as exc:
                msg = f"Screenshot capture failed: {exc}"
                print(f"Error: {msg}", file=sys.stderr)
                self._log_attempt(
                    run_id=run_id,
                    attempt=attempt,
                    success=False,
                    reason=msg,
                    action=last_action,
                    target=last_target,
                    screenshot_hash="",
                    tokens=(0, 0),
                    cost=0.0,
                    latency=0.0,
                    error=msg,
                )
                break

            # Use a forced action (e.g., type value after clicking a filter box)
            # instead of calling the vision model.
            if forced_action:
                action = forced_action
                reason = forced_reason
                success = False
                input_tokens = 0
                output_tokens = 0
                latency = 0.0
                cost = 0.0
                forced_action = None
                forced_reason = ""
            else:
                try:
                    success, reason, action, input_tokens, output_tokens, latency = _call_vision_model(
                        self.client, self.objective, b64
                    )
                except Exception as exc:
                    msg = f"Vision model call failed: {exc}"
                    print(f"Error: {msg}", file=sys.stderr)
                    self._log_attempt(
                        run_id=run_id,
                        attempt=attempt,
                        success=False,
                        reason=msg,
                        action=last_action,
                        target=last_target,
                        screenshot_hash=screenshot_hash,
                        tokens=(0, 0),
                        cost=0.0,
                        latency=0.0,
                        error=msg,
                        api_width_px=api_w,
                        api_height_px=api_h,
                    )
                    break

                cost = (input_tokens * INPUT_PRICE_PER_TOKEN) + (
                    output_tokens * OUTPUT_PRICE_PER_TOKEN
                )
                total_cost += cost

            self._log_attempt(
                run_id=run_id,
                attempt=attempt,
                success=success,
                reason=reason,
                action=last_action,
                target=last_target,
                screenshot_hash=screenshot_hash,
                tokens=(input_tokens, output_tokens),
                cost=cost,
                latency=latency,
                error=None,
                api_width_px=api_w,
                api_height_px=api_h,
            )

            if success:
                state_id = f"{self.application}_{uuid.uuid4().hex[:12]}"
                screenshot_path = self.output_dir / f"{state_id}.png"
                raw_img.save(screenshot_path)

                screen_state = ScreenState(
                    state_id=state_id,
                    screenshot_path=str(screenshot_path.resolve()),
                    timestamp=0.0,
                    application=self.application,
                    platform_snapshot={
                        "api_width_px": api_w,
                        "api_height_px": api_h,
                        "scale_to_logical": scale_to_logical,
                        "screenshot_sha256": screenshot_hash,
                        "run_duration_seconds": round(time.time() - run_start, 3),
                    },
                    visual_summary=reason,
                )
                print(f"End state discovered on attempt {attempt}: {reason}")
                return self._make_result(success=True, locked_state=screen_state)

            # Otherwise execute the suggested action and continue.
            if action is None:
                print(
                    f"Attempt {attempt}: objective not satisfied and no action suggested; stopping.",
                    file=sys.stderr,
                )
                break

            try:
                last_action = _execute_action(action, scale_to_logical)
                last_target = action.get("bbox") or action.get("coordinate")

                # Track typed values and schedule follow-up typing/Return actions
                # for filter/search objectives.
                if action.get("action") == "type":
                    typed_values.add(str(action.get("text", "")))
                    if force_return_after_type:
                        val = str(action.get("text", ""))
                        forced_action = {
                            "action": "key",
                            "text": "Return",
                        }
                        # The screenshot logged with this forced action shows the
                        # value typed but the filter not yet applied.
                        forced_reason = (
                            f"The filter box now contains {val!r}; the table still "
                            "shows all rows because the filter has not been applied yet."
                        )
                        force_return_after_type = False
                elif (
                    action.get("action") in ("click", "double_click")
                    and type_values
                    and _looks_like_text_input(action)
                ):
                    for val in type_values:
                        if val not in typed_values:
                            point = action.get("point") or last_target
                            forced_action = {
                                "action": "type",
                                "point": point,
                                "bbox": action.get("bbox"),
                                "text": val,
                                "element_type": "filter_box",
                                "description": f"type {val!r} into filter box",
                            }
                            # The screenshot logged with this forced action shows
                            # the filter box focused and empty.
                            forced_reason = (
                                "The filter box is now focused and empty; the table still "
                                "shows all rows because no filter value has been entered yet."
                            )
                            force_return_after_type = True
                            break
                elif action.get("action") == "key":
                    # A Return/Enter keypress likely applied a filter; the next
                    # screenshot will reveal whether the objective is satisfied.
                    pass

                # Give the UI a moment to settle before the next screenshot.
                time.sleep(0.8)

                # After a state change, check whether a table is visible with
                # truncated column headers and auto-fit if needed. Skip this check
                # after text-entry actions; the auto-fit helper can misinterpret a
                # focused filter box as a column border and insert an unwanted click.
                action_name = action.get("action")
                if action_name not in ("type", "key"):
                    try:
                        b64_after, _, _, scale_after, _, _ = _capture_screenshot(self.output_dir)
                        if self._auto_fit_if_truncated(b64_after, scale_after):
                            scale_to_logical = scale_after
                    except Exception as exc:
                        print(f"Warning: post-action auto-fit check failed: {exc}", file=sys.stderr)
            except Exception as exc:
                msg = f"Action execution failed: {exc}"
                print(f"Error: {msg}", file=sys.stderr)
                self._log_attempt(
                    run_id=run_id,
                    attempt=attempt,
                    success=False,
                    reason=reason,
                    action=last_action,
                    target=last_target,
                    screenshot_hash=screenshot_hash,
                    tokens=(input_tokens, output_tokens),
                    cost=cost,
                    latency=latency,
                    error=msg,
                    api_width_px=api_w,
                    api_height_px=api_h,
                )
                break

        return self._make_result(success=False, reason="Objective not satisfied within attempt budget")

    def execute_script(
        self,
        actions: Optional[List[Dict[str, Any]]] = None,
        visual_summary: Optional[str] = None,
        save_all_screenshots: bool = True,
        script_beats: Optional[List[ScriptBeat]] = None,
        beat_for_action: Optional[List[Optional[ScriptBeat]]] = None,
        beats: Optional[List[ScriptBeat]] = None,
        opening_state_query: Optional[str] = None,
        opening_state_history: Optional[str] = None,
        new_query: Optional[str] = None,
    ) -> DiscoveryResult:
        """
        Execute a script and lock the resulting end state.

        Two modes are supported:

        1. Lesson-first vision-agent mode (preferred): pass ``beats``. Each beat
           is executed by the VisionAgent, which locates UI elements dynamically
           using Claude, and a video clip is recorded for every beat.

        2. Legacy action-script mode: pass ``actions`` (and optionally
           ``script_beats``/``beat_for_action`` for per-beat clip grouping).
        """
        self.attempt_logs = []

        if beats is not None:
            return self._execute_beats_with_agent(
                beats=beats,
                visual_summary=visual_summary,
                save_all_screenshots=save_all_screenshots,
                opening_state_query=opening_state_query,
                opening_state_history=opening_state_history,
                new_query=new_query,
            )

        if actions is None:
            return self._make_result(
                success=False, reason="execute_script called with neither beats nor actions"
            )

        run_id = uuid.uuid4().hex[:12]
        run_start = time.time()

        errors: List[str] = []
        if self.application not in SUPPORTED_APPLICATIONS:
            errors.append(
                f"Application {self.application!r} is not supported. "
                f"Supported: {', '.join(sorted(SUPPORTED_APPLICATIONS))}"
            )

        db_browser_path = _find_db_browser(self.profile.app_name)
        if self.application == "db_browser_sqlite" and db_browser_path is None:
            errors.append(f"{self.profile.app_name} does not appear to be installed.")

        if errors:
            for msg in errors:
                print(f"Error: {msg}", file=sys.stderr)
            return self._make_result(success=False, reason="; ".join(errors))

        try:
            if self.db_path:
                db_path = self.db_path
                if not db_path.exists():
                    raise FileNotFoundError(f"Requested database not found: {db_path}")
            else:
                db_path = _ensure_sample_db(self.output_dir)
            self._launch_app(db_path)
        except Exception as exc:
            msg = f"Failed to launch {self.profile.app_name}: {exc}"
            print(f"Error: {msg}", file=sys.stderr)
            return self._make_result(success=False, reason=msg)

        self._auto_fit_columns()

        stage_assertion = self._assert_stage_resources()
        if not stage_assertion["ok"]:
            print(f"Error: {stage_assertion['reason']}", file=sys.stderr)
            return self._make_result(success=False, reason=stage_assertion["reason"])

        result = self._execute_action_script(
            actions=actions,
            run_id=run_id,
            save_all_screenshots=save_all_screenshots,
        )
        if not result.get("success"):
            return self._make_result(success=False, reason=result.get("reason", "unknown"))

        last_action = result.get("last_action")
        last_target = result.get("last_target")
        prev_video_path = result.get("prev_video_path")

        # Lesson-first path: group per-action clips into one clip per demo beat.
        if script_beats is not None and beat_for_action is not None:
            self._concatenate_clips_per_beat(
                run_id=run_id,
                script_beats=script_beats,
                beat_for_action=beat_for_action,
            )

        # Capture and lock the final end state.
        try:
            b64, api_w, api_h, scale_to_logical, raw_img, raw_bytes = _capture_screenshot(
                self.output_dir
            )
        except Exception as exc:
            msg = f"Final screenshot capture failed: {exc}"
            print(f"Error: {msg}", file=sys.stderr)
            return self._make_result(success=False, reason=msg)

        screenshot_hash = hashlib.sha256(raw_bytes).hexdigest()
        if save_all_screenshots:
            hash_path = self.output_dir / f"{screenshot_hash}.png"
            raw_img.save(hash_path)

        # Execute SQL recipes must show a populated results grid.
        if _is_execute_query_objective(self.objective):
            if not self._results_grid_visible(b64):
                print(
                    "Warning: no results grid visible after query execution; retrying F5...",
                    file=sys.stderr,
                )
                _press_key("F5")
                time.sleep(1.5)
                try:
                    b64, api_w, api_h, scale_to_logical, raw_img, raw_bytes = _capture_screenshot(
                        self.output_dir
                    )
                except Exception as exc:
                    msg = f"Retry screenshot capture failed: {exc}"
                    print(f"Error: {msg}", file=sys.stderr)
                    return self._make_result(success=False, reason=msg)

                screenshot_hash = hashlib.sha256(raw_bytes).hexdigest()
                if save_all_screenshots:
                    hash_path = self.output_dir / f"{screenshot_hash}.png"
                    raw_img.save(hash_path)

                if not self._results_grid_visible(b64):
                    return self._make_result(
                        success=False,
                        reason="Execute SQL script did not display a results grid",
                    )

        final_attempt = len(actions) + 1
        self._log_attempt(
            run_id=run_id,
            attempt=final_attempt,
            success=True,
            reason=f"Script completed: {self.objective}",
            action=last_action,
            target=last_target,
            screenshot_hash=screenshot_hash,
            tokens=(0, 0),
            cost=0.0,
            latency=0.0,
            error=None,
            api_width_px=api_w,
            api_height_px=api_h,
            video_path=prev_video_path,
        )

        state_id = f"{self.application}_{uuid.uuid4().hex[:12]}"
        screenshot_path = self.output_dir / f"{state_id}.png"
        raw_img.save(screenshot_path)

        if not visual_summary:
            visual_summary = _describe_recipe_outcome(self.objective)
        if not visual_summary:
            visual_summary = f"Script completed: {self.objective}"

        screen_state = ScreenState(
            state_id=state_id,
            screenshot_path=str(screenshot_path.resolve()),
            timestamp=0.0,
            application=self.application,
            platform_snapshot={
                "api_width_px": api_w,
                "api_height_px": api_h,
                "scale_to_logical": scale_to_logical,
                "screenshot_sha256": screenshot_hash,
                "run_duration_seconds": round(time.time() - run_start, 3),
            },
            visual_summary=visual_summary,
        )

        print(f"End state discovered via script in {final_attempt} attempts: {self.objective}")
        return self._make_result(success=True, locked_state=screen_state, recipe=True)

    @staticmethod
    def _wrap_query_as_history(query_text: str) -> str:
        """Wrap a prior query in a block comment for continuity display."""
        text = query_text.strip()
        if text.startswith("/*"):
            end = text.find("*/")
            if end != -1:
                text = text[end + 2 :].strip()
        return f"/*\n{text}\n*/"

    def _prepare_opening_state(
        self,
        beats: List[ScriptBeat],
        agent: VisionAgent,
        opening_state_query: Optional[str] = None,
        opening_state_history: Optional[str] = None,
        new_query: Optional[str] = None,
    ) -> None:
        """
        Establish or adapt the UI state described by the first state beat.

        Stage-matches-story: the opening screen must fully instantiate what the
        state beat narrates. For continuity videos we paste the commented history,
        run the immediate prerequisite query so the result pane is populated, then
        re-paste the history with that query commented out. Finally we VLM-verify
        the screen against the beat; on mismatch we try once to fix, then rewrite
        the beat to match reality.
        """
        first_state = next((b for b in beats if b.kind == "state"), None)
        if first_state is None:
            return

        # Persist for beat retries.
        self.opening_state_history = opening_state_history
        self.new_query = new_query

        history_to_paste = opening_state_history
        query_to_type = opening_state_query or self.opening_state_query
        text_lower = first_state.text.lower()
        observed: Dict[str, Any] = {}
        execute_tab = self.profile.landmarks.get("execute_tab", "Execute SQL tab")

        def _log_and_store(strategy: str, reason: str) -> None:
            log_line = f"[OPENING STATE] {strategy}: {reason}"
            print(log_line, file=sys.stderr)
            if first_state.observed_state is None:
                first_state.observed_state = {}
            first_state.observed_state.update(observed)
            first_state.observed_state["opening_state_strategy"] = strategy
            first_state.observed_state["opening_state_log"] = log_line

        def _verify_or_rewrite(already_ran_query: bool = False) -> None:
            """VLM-verify the screen against the state beat and rewrite on mismatch."""
            verified = agent.verify_state(first_state.text)
            observed["opening_state_verified"] = verified
            if verified:
                return
            print(
                f"[OPENING STATE] verification mismatch for {first_state.beat_id}, "
                "attempting one fix",
                file=sys.stderr,
            )
            # Only append and re-run the prior query as a fix when it has not
            # already been executed as part of the continuity setup. Re-running
            # after the commented history has been staged would leave an
            # uncommented duplicate in the editor and break statement isolation.
            if query_to_type and not already_ran_query:
                agent.append_block(query_to_type)
                agent.execute_beat({"type": "run_query"})
                verified = agent.verify_state(first_state.text)
                observed["opening_state_verified"] = verified
            if not verified:
                summary = observed.get("summary", "")
                if summary:
                    first_state.text = (
                        f"Our previous queries sit above, commented out. {summary}"
                    )
                else:
                    first_state.text = (
                        "Our previous queries sit above, commented out, and the "
                        "result pane shows the prior query output."
                    )
                observed["opening_state_rewritten"] = True

        # Case A: continuity-by-design. Paste the commented history, run the prior
        # query to populate the result pane, then re-paste the history with the
        # prior query commented out so the editor shows the full progression.
        if history_to_paste:
            agent.dismiss_transient_ui()
            agent.prepare_sql_editor()
            if agent.paste_history_block(history_to_paste):
                if query_to_type:
                    agent.append_block(query_to_type)
                    agent.execute_beat({"type": "run_query"})
                    # The history already contains the prior query commented out;
                    # clear the active query and re-paste only the commented history
                    # so the editor matches the state beat (previous queries above,
                    # commented out) without duplicating the just-run query.
                    agent.prepare_sql_editor()
                    agent.paste_history_block(history_to_paste)
                agent.dismiss_transient_ui()
                observed = agent.summarize_observed_state()
                observed["history_pasted"] = True
                _verify_or_rewrite(already_ran_query=True)
                _log_and_store(
                    "established" if observed.get("opening_state_verified") else "adapted",
                    "continuity history + prior query results staged",
                )
                return
            else:
                agent.prepare_sql_editor()
                observed = agent.summarize_observed_state()
                _log_and_store(
                    "adapted",
                    "failed to paste SQL history; editor cleared",
                )
                return

        # Case B: the script claims the editor is empty/blank.
        if "editor is empty" in text_lower or "editor is blank" in text_lower:
            agent.dismiss_transient_ui()
            agent.prepare_sql_editor()
            agent.dismiss_transient_ui()
            observed = agent.summarize_observed_state()
            _verify_or_rewrite()
            _log_and_store(
                "established" if observed.get("opening_state_verified") else "adapted",
                "opening state describes an empty editor; editor cleared",
            )
            return

        # Case C: the script assumes the execution tab is open with a prior query.
        establish_phrases = (
            f"{execute_tab.lower()} is still open",
            f"{execute_tab.lower()} is open",
            "editor still holds",
            "editor holds our",
            "editor shows our previous",
        )
        should_establish = any(phrase in text_lower for phrase in establish_phrases)

        if should_establish and query_to_type:
            agent.dismiss_transient_ui()
            agent.find_and_click(f"Click the {execute_tab} tab", execute_tab)
            agent.prepare_sql_editor()
            if agent.type_block(query_to_type):
                agent.execute_beat({"type": "run_query"})
                agent.dismiss_transient_ui()
                observed = agent.summarize_observed_state()
                _verify_or_rewrite()
                _log_and_store(
                    "established" if observed.get("opening_state_verified") else "adapted",
                    f"{execute_tab} open, prior query executed",
                )
                return
            else:
                agent.prepare_sql_editor()
                observed = agent.summarize_observed_state()
                _log_and_store(
                    "adapted",
                    "failed to type prerequisite query; editor cleared",
                )
                return

        # Case D: fresh or unknown orientation; summarize actual screen.
        agent.dismiss_transient_ui()
        observed = agent.summarize_observed_state()
        _verify_or_rewrite()
        _log_and_store(
            "established" if observed.get("opening_state_verified") else "adapted",
            "orientation summarized from actual screen",
        )

    def _reset_editor_for_retry(
        self,
        agent: VisionAgent,
        opening_state_history: Optional[str],
    ) -> None:
        """Clear the SQL editor and, if a continuity history exists, re-paste it."""
        agent.prepare_sql_editor()
        if opening_state_history:
            agent.type_block(opening_state_history)

    def _execute_beats_with_agent(
        self,
        beats: List[ScriptBeat],
        visual_summary: Optional[str],
        save_all_screenshots: bool,
        opening_state_query: Optional[str] = None,
        opening_state_history: Optional[str] = None,
        new_query: Optional[str] = None,
    ) -> DiscoveryResult:
        """
        Execute a lesson-first script using the VisionAgent.

        Every beat gets its own recorded video clip. Demo beats drive dynamic UI
        actions; validation beats ask the VLM to confirm the screen state; opening
        and close beats record a short pause.

        A frontmost-app sidecar log is kept so any clip that captures off-
        application frames can be discarded and re-recorded.
        """
        run_id = uuid.uuid4().hex[:12]
        run_start = time.time()

        errors: List[str] = []
        if self.application not in SUPPORTED_APPLICATIONS:
            errors.append(
                f"Application {self.application!r} is not supported. "
                f"Supported: {', '.join(sorted(SUPPORTED_APPLICATIONS))}"
            )

        db_browser_path = _find_db_browser(self.profile.app_name)
        if self.application == "db_browser_sqlite" and db_browser_path is None:
            errors.append(f"{self.profile.app_name} does not appear to be installed.")

        if errors:
            for msg in errors:
                print(f"Error: {msg}", file=sys.stderr)
            return self._make_result(success=False, reason="; ".join(errors))

        try:
            if self.db_path:
                db_path = self.db_path
                if not db_path.exists():
                    raise FileNotFoundError(f"Requested database not found: {db_path}")
            else:
                db_path = _ensure_sample_db(self.output_dir)
            self._launch_app(db_path)
        except Exception as exc:
            msg = f"Failed to launch {self.profile.app_name}: {exc}"
            print(f"Error: {msg}", file=sys.stderr)
            return self._make_result(success=False, reason=msg)

        self._auto_fit_columns()

        # Stage-prep resource assertion: the correct database is open and the
        # expected seed table has the expected row count before any beat runs.
        stage_assertion = self._assert_stage_resources()
        if not stage_assertion["ok"]:
            print(f"Error: {stage_assertion['reason']}", file=sys.stderr)
            return self._make_result(success=False, reason=stage_assertion["reason"])

        agent = VisionAgent(model=MODEL, output_dir=str(self.output_dir), profile=self.profile)
        frontmost_log_path = self.output_dir / f"frontmost_{run_id}.log"
        if frontmost_log_path.exists():
            frontmost_log_path.unlink()

        self._prepare_opening_state(
            beats,
            agent,
            opening_state_query=opening_state_query,
            opening_state_history=opening_state_history,
            new_query=new_query,
        )

        # C10: snapshot the editor state after stage prep so beat-scoped retries
        # never re-compose beats that already passed.
        last_good_editor = opening_state_history or ""
        try:
            current_editor = agent._read_editor_content(focus=False) or ""
            if current_editor.strip():
                last_good_editor = current_editor
        except Exception:
            pass

        # Continuity-by-design: if the first typing demo is a monolithic type_block,
        # rewrite it to append the new query to the pasted history instead of
        # replacing it. Segmented type_segments beats naturally append because the
        # editor already contains the commented history after stage prep.
        if new_query and opening_state_history:
            for beat in beats:
                if beat.kind == "demo" and (beat.action or {}).get("type") == "type_block":
                    beat.action = {
                        "type": "append_block",
                        "text": new_query,
                        "detail": new_query,
                    }
                    print(
                        f"[CONTINUITY] Rewrote {beat.beat_id} to append_block "
                        f"for continuity-by-design",
                        file=sys.stderr,
                    )
                    break

        # Caller-provided intended cumulative block for segmented typing. If a
        # segment fails, the fallback paste uses this instead of the live editor
        # content, so a corrupted editor cannot poison the recovery.
        segment_fallback_text = ""
        if opening_state_history:
            segment_fallback_text += opening_state_history
        if new_query:
            segment_fallback_text += ("\n\n" if segment_fallback_text else "") + new_query

        failed_reason = ""
        previous_observed_state: Optional[Dict[str, Any]] = None
        executed_demo_count = 0

        # Ceiling: one pass per beat (bounded by len(beats)).
        for idx, beat in enumerate(beats):
            action = beat.action
            if not action:
                # Non-demo beats without an explicit action simply wait.
                action = {"type": "wait", "duration": 1.5}

            # --- SKIP redundant demo actions ----------------------------------
            skipped = False
            if beat.kind == "demo" and action.get("type") != "wait":
                intended = self._intended_state_description(action)
                later_demos = [j for j in range(idx + 1, len(beats)) if beats[j].kind == "demo"]
                # Segmented typing builds the query clause-by-clause; each segment
                # must execute so the cumulative text is correct. Skipping a segment
                # would leave an incomplete query (e.g. SELECT without FROM).
                is_segmented = action.get("type") == "type_segments"
                can_skip = (executed_demo_count > 0 or len(later_demos) > 0) and not is_segmented
                skip_reason = (
                    f"can_skip={can_skip} (executed_demo_count={executed_demo_count}, "
                    f"later_demos={len(later_demos)})"
                )
                prev_summary = ""
                if previous_observed_state:
                    prev_summary = previous_observed_state.get("summary", "") or ""
                print(
                    f"  [skip-check] {beat.beat_id}: intended='{intended}' | "
                    f"previous_observed='{prev_summary[:80]}' | {skip_reason}",
                    file=sys.stderr,
                )
                if intended and can_skip:
                    try:
                        already_true, suggested = agent.is_end_state_already_present(
                            intended, previous_observed_state=previous_observed_state
                        )
                        verdict = "YES" if already_true else "NO"
                        print(
                            f"  [skip-check] {beat.beat_id}: VLM verdict={verdict} "
                            f"suggested='{suggested[:70]}'",
                            file=sys.stderr,
                        )
                        if already_true and suggested:
                            if not suggested.lower().startswith("we "):
                                suggested = f"We see that {suggested[0].lower()}{suggested[1:]}"
                            print(
                                f"  Skipping redundant action for {beat.beat_id}: {suggested[:70]}",
                                file=sys.stderr,
                            )
                            skipped = True
                            beat.kind = "state"
                            beat.text = suggested
                            beat.action = {"type": "wait", "duration": 1.5}
                            action = beat.action
                            beat.video_clip_path = None
                    except Exception as exc:
                        print(
                            f"Warning: skip check failed for {beat.beat_id}: {exc}",
                            file=sys.stderr,
                        )
                else:
                    why = "no intended state" if not intended else "would leave graph without demo edges"
                    print(f"  [skip-check] {beat.beat_id}: not checked because {why}", file=sys.stderr)

            clip_path = self.output_dir / f"{run_id}_{beat.beat_id}.mp4"
            beat_failed = False

            # Ceiling: MAX_BEAT_RETRIES (default WSDA_MAX_ATTEMPTS=3) per-beat retries.
            for retry in range(MAX_BEAT_RETRIES):
                beat_ok = False
                if retry > 0 and not skipped:
                    # C10 beat-scoped retry: restore the last known-good editor
                    # state without re-composing prior passed beats.
                    if last_good_editor:
                        print(
                            f"  [BEAT RETRY] restoring last good editor state "
                            f"({len(last_good_editor)} chars) for {beat.beat_id}",
                            file=sys.stderr,
                        )
                        was_recording = agent.recording
                        agent.recording = False
                        agent._clear_editor_accessibility()
                        agent._set_editor_text_accessibility(last_good_editor)
                        agent.recording = was_recording
                    else:
                        self._reset_editor_for_retry(agent, self.opening_state_history)

                recorder: Any = ScreenRecorder(str(clip_path), fps=10)
                if self.profile and self.profile.app_name:
                    recorder = _ScreenCaptureKitRecorder(
                        str(clip_path), fps=10, app_name=self.profile.app_name
                    )
                if not skipped:
                    # Stage prep before recording: clear editor for standalone typing
                    # beats, but keep the commented history in place for continuity
                    # videos. Dismiss any transient UI that would clutter the frame.
                    # Segmented typing builds the query incrementally, so the editor
                    # must keep whatever was typed in previous segments. Only clear for
                    # a monolithic type_block (or when there is no history to preserve).
                    if action.get("type") == "type_block" and not self.opening_state_history:
                        agent.prepare_sql_editor()
                    else:
                        agent.dismiss_transient_ui()
                    agent.recording = True
                    recorder.start()
                clip_start = time.time()

                try:
                    print(
                        f"  Executing beat {beat.beat_id} ({beat.kind}): {beat.text[:60]}",
                        file=sys.stderr,
                    )
                    if not skipped:
                        # Segmented beats get the intended cumulative block as a
                        # fallback so a failed segment recovers to the real target.
                        beat_fallback = (
                            segment_fallback_text
                            if action.get("type") == "type_segments"
                            else None
                        )
                        # C9: typing actions must not accumulate across internal
                        # attempts because each attempt appends to the live buffer.
                        # Rely on the outer retry loop, which resets the editor and
                        # starts a fresh clip, instead of re-typing inside the same
                        # recording.
                        is_typing_action = action.get("type") in (
                            "type_segments", "type_block", "append_block"
                        )
                        # Ceiling: 1 internal attempt for typing (handled by outer beat
                        # retry) or 3 attempts for other action types.
                        max_attempts = 1 if is_typing_action else 3
                        for attempt in range(max_attempts):
                            if agent.execute_beat(action, fallback_text=beat_fallback):
                                beat_ok = True
                                break
                            print(
                                f"  Beat {beat.beat_id} failed attempt {attempt + 1}; retrying...",
                                file=sys.stderr,
                            )
                            time.sleep(0.5)

                        # Programmatic fallback for row-count validations: the VLM
                        # sometimes focuses on the SQL editor instead of the result
                        # pane. If the beat asserts a specific row count, trust the
                        # actual result-pane summary when it matches.
                        if not beat_ok and beat.kind == "validation":
                            expected_rows = _extract_expected_row_count(beat.text)
                            if expected_rows is not None:
                                try:
                                    pane_summary = agent.summarize_result_pane()
                                    actual_rows = _parse_row_count(
                                        pane_summary.get("row_count")
                                    )
                                    if actual_rows is not None and actual_rows == expected_rows:
                                        print(
                                            f"  [VALIDATION] result pane confirms {actual_rows} rows; "
                                            "overriding VLM NO",
                                            file=sys.stderr,
                                        )
                                        beat_ok = True
                                        if beat.observed_state is None:
                                            beat.observed_state = {}
                                        beat.observed_state["query_result"] = pane_summary
                                except Exception as exc:
                                    print(
                                        f"  [VALIDATION] programmatic row-count check failed: {exc}",
                                        file=sys.stderr,
                                    )

                        if not beat_ok:
                            recovery = agent.ask_recovery(
                                f"Beat {beat.beat_id} ({beat.kind}): {beat.text}"
                            )
                            if recovery:
                                print(f"  Attempting recovery for {beat.beat_id}", file=sys.stderr)
                                if not agent.execute_beat(recovery):
                                    failed_reason = (
                                        f"Beat {beat.beat_id} failed and recovery did not succeed"
                                    )
                                    beat_failed = True
                                    break
                            else:
                                failed_reason = (
                                    f"Beat {beat.beat_id} failed and no recovery action was returned"
                                )
                                beat_failed = True
                                break

                        if beat_ok and beat.kind == "demo":
                            executed_demo_count += 1

                            # Persist the verified cumulative editor content so the
                            # final byte-for-byte canonical match sees the full block,
                            # not just the last segment.
                            if action.get("type") in ("type_segments", "type_block", "append_block"):
                                composed = agent._last_composed_text or ""
                                if not composed and action.get("type") in ("type_block", "append_block"):
                                    composed = action.get("text") or action.get("detail") or ""
                                if composed:
                                    if beat.observed_state is None:
                                        beat.observed_state = {}
                                    beat.observed_state["editor_content"] = composed
                                    last_good_editor = composed

                            # Stage prep after a successful action.
                            if action.get("type") == "run_query":
                                agent.scroll_result_pane_top()
                            agent.dismiss_transient_ui()
                        elif beat_ok and beat.kind in ("state", "explain", "concept"):
                            # Part E: fill explain/state beats with real cursor motion
                            # so the clip is not a static hold.
                            print(
                                f"  [EMPHASIS] {beat.beat_id}: performing emphasis actions",
                                file=sys.stderr,
                            )
                            agent.perform_emphasis_actions(beat)

                            # Persist the verified editor content for type_block beats.
                            if action.get("type") == "type_block":
                                intended = (
                                    action.get("text") or action.get("detail") or ""
                                ).strip()
                                if intended:
                                    if beat.observed_state is None:
                                        beat.observed_state = {}
                                    beat.observed_state["editor_content"] = intended
                                    last_good_editor = intended

                            # Keep recorder running while the UI settles so the clip
                            # captures the settled end state rather than cutting off
                            # while animations or loading are still in progress.
                            self._wait_for_visual_stability(
                                interval_seconds=0.4,
                                timeout_seconds=4.0,
                                frontmost_log_path=frontmost_log_path,
                            )

                            # --- SQL result grounding --------------------------------
                            # Only auto-run and ground when the beat actually contains a
                            # runnable SELECT statement. Comment blocks and prose beats
                            # should not trigger an execution.
                            sql_for_grounding = _extract_sql_query(beat)
                            if sql_for_grounding:
                                grounding_ok = False
                                try:
                                    agent.run_query()
                                    pane_summary = agent.summarize_result_pane()
                                    if beat.observed_state is None:
                                        beat.observed_state = {}
                                    beat.observed_state["query_result"] = pane_summary
                                    db_path_str = str(self.db_path) if self.db_path else None
                                    grounding_ok = _ground_query_result(
                                        beat, db_path_str, video_id=self.video_id
                                    )
                                except Exception as exc:
                                    print(
                                        f"[GROUNDING FAIL] SQL grounding failed for {beat.beat_id}: {exc}",
                                        file=sys.stderr,
                                    )
                                if not grounding_ok:
                                    failed_reason = (
                                        f"[GROUNDING] {beat.beat_id} failed SQL result grounding"
                                    )
                                    beat_failed = True
                                    break
                    else:
                        beat_ok = True
                finally:
                    if not skipped:
                        recorder.stop()
                        agent.recording = False
                clip_end = time.time()

                if skipped:
                    break

                if beat_ok:
                    if _clip_has_off_app_interval(
                        frontmost_log_path, clip_start, clip_end, self.profile.app_name
                    ):
                        print(
                            f"[OFF-APP] {beat.beat_id} attempt {retry + 1} recorded off-application frames",
                            file=sys.stderr,
                        )
                        if retry < 1:
                            beat_ok = False
                            continue
                        else:
                            failed_reason = (
                                f"[OFF-APP] {beat.beat_id} recorded off-application frames after retries"
                            )
                            beat_failed = True
                            break

                    # Drop any clip that shows the profile's error signature.
                    if self.profile.error_signature and clip_path.exists():
                        error_frame_count = count_error_signature_frames(
                            clip_path, self.profile, sample_fps=1
                        )
                        if error_frame_count > 0:
                            print(
                                f"[ERROR-SIG] {beat.beat_id} attempt {retry + 1} contains {error_frame_count} error-signature frame(s)",
                                file=sys.stderr,
                            )
                            try:
                                clip_path.unlink()
                            except Exception:
                                pass
                            if retry < 1:
                                beat_ok = False
                                continue
                            else:
                                failed_reason = (
                                    f"[ERROR-SIG] {beat.beat_id} still contains error signature after retries"
                                )
                                beat_failed = True
                                break

                    if clip_path.exists():
                        self._trim_clip_to_motion(clip_path)
                        beat.video_clip_path = str(clip_path.resolve())
                    break

            if beat_failed:
                break

            # --- OBSERVE after state/demo/validation beats --------------------
            if beat_ok and beat.kind in ("state", "demo", "validation"):
                # Dismiss any transient dropdown/modal before observing stable state.
                agent.dismiss_transient_ui()
                try:
                    observed = agent.summarize_observed_state()
                    # Preserve any SQL grounding data already attached to the beat.
                    prior_query_result = (beat.observed_state or {}).get("query_result")
                    # Preserve continuity-aware opening-state metadata set during stage prep.
                    prior_opening_strategy = (beat.observed_state or {}).get("opening_state_strategy")
                    prior_opening_log = (beat.observed_state or {}).get("opening_state_log")
                    prior_history_pasted = (beat.observed_state or {}).get("history_pasted")
                    beat.observed_state = observed
                    if prior_query_result:
                        beat.observed_state["query_result"] = prior_query_result
                    if prior_opening_strategy:
                        beat.observed_state["opening_state_strategy"] = prior_opening_strategy
                    if prior_opening_log:
                        beat.observed_state["opening_state_log"] = prior_opening_log
                    if prior_history_pasted:
                        beat.observed_state["history_pasted"] = prior_history_pasted
                    previous_observed_state = observed
                    summary = observed.get("summary", "")
                    if summary:
                        print(f"    Observed: {summary[:100]}", file=sys.stderr)
                except Exception as exc:
                    print(
                        f"Warning: could not summarize observed state for {beat.beat_id}: {exc}",
                        file=sys.stderr,
                    )

            if failed_reason:
                return self._make_result(success=False, reason=failed_reason)

        if failed_reason:
            return self._make_result(success=False, reason=failed_reason)

        # TIDY end state: dismiss any open dropdown/modal before final capture.
        try:
            agent.dismiss_transient_ui()
        except Exception as exc:
            print(f"Warning: end-state tidy check failed: {exc}", file=sys.stderr)

        # C10: authoritative editor read-back for canonical comparison.
        # Focus the editor first; the last action (e.g. clicking Execute) may have
        # moved keyboard focus to a button or the result pane, so a focus-less read
        # can return a non-editor value such as a button label.
        final_editor_content = ""
        try:
            agent._focus_editor()
            final_editor_content = agent._read_editor_content(focus=False) or ""
        except Exception as exc:
            print(f"Warning: could not read final editor content: {exc}", file=sys.stderr)

        # Capture and lock the final end state.
        try:
            b64, api_w, api_h, scale_to_logical, raw_img, raw_bytes = _capture_screenshot(
                self.output_dir
            )
        except Exception as exc:
            msg = f"Final screenshot capture failed: {exc}"
            print(f"Error: {msg}", file=sys.stderr)
            return self._make_result(success=False, reason=msg)

        screenshot_hash = hashlib.sha256(raw_bytes).hexdigest()
        if save_all_screenshots:
            hash_path = self.output_dir / f"{screenshot_hash}.png"
            raw_img.save(hash_path)

        # Execute SQL objectives must show a populated results grid.
        if _is_execute_query_objective(self.objective):
            if not self._results_grid_visible(b64):
                print(
                    "Warning: no results grid visible after query execution; "
                    "asking VLM to click the Result tab...",
                    file=sys.stderr,
                )
                result = agent._call_vlm(
                    "Return ONLY a JSON object with the coordinates of the 'Result' tab "
                    "in the lower results pane of DB Browser for SQLite: "
                    '{"action": "click", "point": {"x": int, "y": int}}. '
                    'If you do not see it, return {"action": "none"}.',
                    expect_json=True,
                    max_tokens=128,
                )
                action = result.action
                if action and action.get("action") == "click" and "point" in action:
                    point = action["point"]
                    lx = point["x"] * scale_to_logical
                    ly = point["y"] * scale_to_logical
                    _click(lx, ly)
                    print(
                        f"  Clicked Result tab at ({lx:.0f}, {ly:.0f})",
                        file=sys.stderr,
                    )
                    time.sleep(1.5)
                else:
                    print(
                        "  VLM did not find Result tab; continuing without click",
                        file=sys.stderr,
                    )

                try:
                    b64, api_w, api_h, scale_to_logical, raw_img, raw_bytes = _capture_screenshot(
                        self.output_dir
                    )
                except Exception as exc:
                    msg = f"Retry screenshot capture failed: {exc}"
                    print(f"Error: {msg}", file=sys.stderr)
                    return self._make_result(success=False, reason=msg)

                screenshot_hash = hashlib.sha256(raw_bytes).hexdigest()
                if save_all_screenshots:
                    hash_path = self.output_dir / f"{screenshot_hash}.png"
                    raw_img.save(hash_path)

                if not self._results_grid_visible(b64):
                    return self._make_result(
                        success=False,
                        reason="Execute SQL script did not display a results grid",
                    )

        self._log_attempt(
            run_id=run_id,
            attempt=len(beats) + 1,
            success=True,
            reason=f"Vision-agent script completed: {self.objective}",
            action=None,
            target=None,
            screenshot_hash=screenshot_hash,
            tokens=(0, 0),
            cost=0.0,
            latency=0.0,
            error=None,
            api_width_px=api_w,
            api_height_px=api_h,
            video_path=None,
        )

        state_id = f"{self.application}_{uuid.uuid4().hex[:12]}"
        screenshot_path = self.output_dir / f"{state_id}.png"
        raw_img.save(screenshot_path)

        if not visual_summary:
            visual_summary = _describe_recipe_outcome(self.objective)
        if not visual_summary:
            visual_summary = f"Vision-agent script completed: {self.objective}"

        screen_state = ScreenState(
            state_id=state_id,
            screenshot_path=str(screenshot_path.resolve()),
            timestamp=0.0,
            application=self.application,
            platform_snapshot={
                "api_width_px": api_w,
                "api_height_px": api_h,
                "scale_to_logical": scale_to_logical,
                "screenshot_sha256": screenshot_hash,
                "run_duration_seconds": round(time.time() - run_start, 3),
            },
            visual_summary=visual_summary,
        )

        print(f"End state discovered via vision agent in {len(beats)} beats: {self.objective}")
        return self._make_result(
            success=True,
            locked_state=screen_state,
            recipe=True,
            final_editor_content=final_editor_content,
        )

    def _intended_state_description(self, action: Dict[str, Any]) -> str:
        """Convert a demo action into a precise intended end-state description."""
        action_type = action.get("type")
        detail = action.get("action_detail") or action.get("detail") or ""
        if not detail and isinstance(action.get("target"), str):
            detail = action["target"]
        detail_lower = detail.lower()

        # Table-selection clicks (dropdown or list item) end with the table loaded.
        table_select_match = __import__("re").search(
            r"(\w+)\s+(?:table\s+)?(?:in\s+)?(?:the\s+)?(?:table\s+)?(?:dropdown|list|selector)",
            detail,
            __import__("re").IGNORECASE,
        )
        if table_select_match and "table" in detail_lower:
            return f"the {table_select_match.group(1)} table rows are visible in the Browse Data grid"

        if action_type == "sequence":
            sub_actions = action.get("actions", [])
            descriptions = [
                EndStateDiscovery._intended_state_description(sub)
                for sub in sub_actions
            ]
            descriptions = [d for d in descriptions if d]
            if descriptions:
                return descriptions[-1]
            return ""

        if action_type == "click":
            if "browse data" in detail_lower:
                return "the Browse Data tab is active and its content area is visible"
            if "table" in detail_lower and "dropdown" in detail_lower:
                return "the table dropdown is open"
            return f"{detail} is selected/active"
        if action_type == "browse_table":
            return f"the {action.get('table', detail)} table rows are visible in the Browse Data grid"
        if action_type == "sort_column":
            return f"the {action.get('table', '')} table is sorted by {action.get('column', '')}"
        if action_type == "filter_column":
            return f"the {action.get('table', '')} table is filtered by {action.get('column', '')} = {action.get('value', '')}"
        if action_type == "execute_query":
            execute_tab = self.profile.landmarks.get("execute_tab", "Execute SQL tab")
            return f"the {execute_tab} tab shows query results"
        if action_type == "type_segments":
            return "the SQL editor contains the typed query"
        if action_type == "type":
            return f"'{detail}' has been typed into the active input"
        if action_type == "key":
            return f"the '{detail}' key has been pressed"
        return detail or ""

    def _concatenate_clips_per_beat(
        self,
        run_id: str,
        script_beats: List[ScriptBeat],
        beat_for_action: List[Optional[ScriptBeat]],
    ) -> None:
        """
        Group the per-action video clips recorded in attempt_logs by demo beat
        and concatenate each group into a single clip stored on the beat.
        """
        from collections import defaultdict

        # Collect clip paths per beat id. The per-action clips are stored as
        # {run_id}_step_{i}.mp4 by _execute_action_script, so we can reference
        # them directly instead of parsing attempt_logs.
        clips_by_beat_id: Dict[str, List[str]] = defaultdict(list)
        beat_by_id: Dict[str, ScriptBeat] = {}
        for i, beat in enumerate(beat_for_action):
            if beat is None:
                continue
            beat_by_id[beat.beat_id] = beat
            step_path = self.output_dir / f"{run_id}_step_{i}.mp4"
            if step_path.exists():
                clips_by_beat_id[beat.beat_id].append(str(step_path.resolve()))

        for beat_id, clip_paths in clips_by_beat_id.items():
            beat = beat_by_id.get(beat_id)
            if beat is None:
                continue
            if not clip_paths:
                continue
            out_path = self.output_dir / f"{run_id}_beat_{beat.beat_id}.mp4"
            if len(clip_paths) == 1:
                # Fast path: copy the single clip to a deterministic beat path.
                import shutil
                shutil.copy(clip_paths[0], out_path)
            else:
                concat_list = self.output_dir / f"{run_id}_beat_{beat.beat_id}_concat.txt"
                concat_list.write_text(
                    "\n".join(f"file '{Path(p).resolve()}'" for p in clip_paths),
                    encoding="utf-8",
                )
                try:
                    subprocess.run(
                        [
                            "ffmpeg", "-y", "-f", "concat", "-safe", "0",
                            "-i", str(concat_list), "-c", "copy", str(out_path),
                        ],
                        check=True, capture_output=True, timeout=120,
                    )
                except subprocess.CalledProcessError as exc:
                    stderr = exc.stderr.decode("utf-8", errors="replace") if exc.stderr else ""
                    print(
                        f"Warning: could not concatenate clips for {beat.beat_id}: {stderr[:200]}",
                        file=sys.stderr,
                    )
                    continue
            if out_path.exists():
                self._trim_clip_to_motion(out_path)
            beat.video_clip_path = str(out_path.resolve())

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _wait_for_visual_stability(
        self,
        interval_seconds: float = 0.4,
        timeout_seconds: float = 4.0,
        stability_threshold: float = 1.0,
        stable_frames_required: int = 2,
        frontmost_log_path: Optional[Path] = None,
    ) -> None:
        """
        Poll screenshots until the UI stops changing or a timeout is reached.

        The recorder (if running) keeps capturing during this window so the
        resulting clip includes the settled end state rather than cutting off
        while animations or loading are still in progress. The frontmost app is
        logged at every poll so off-application intervals can be detected and cut.
        """
        start = time.time()
        prev_gray: Optional[np.ndarray] = None
        stable_frames = 0
        # Ceiling: timeout_seconds (default 4s) / interval_seconds (default 0.4s)
        # ~= 10 iterations.
        while time.time() - start < timeout_seconds:
            if frontmost_log_path is not None:
                _log_frontmost(frontmost_log_path)
            try:
                _, _, _, _, raw_img, _ = _capture_screenshot(self.output_dir)
                gray = np.array(raw_img.convert("L"))
                gray = cv2.resize(gray, (320, 180), interpolation=cv2.INTER_AREA)
                if prev_gray is not None:
                    diff = float(np.mean(np.abs(gray.astype(np.float32) - prev_gray.astype(np.float32))))
                    if diff < stability_threshold:
                        stable_frames += 1
                        if stable_frames >= stable_frames_required:
                            return
                    else:
                        stable_frames = 0
                prev_gray = gray
            except Exception as exc:
                print(f"Warning: stability screenshot failed: {exc}", file=sys.stderr)
            time.sleep(interval_seconds)

    def _trim_clip_to_motion(self, clip_path: Path) -> None:
        """Trim static head/tail from a recorded beat clip, keeping the action window.

        C9: conservative reference-based trim. Each frame is compared to the
        clip's first frame (to find when content leaves the static opening) and
        to the clip's last frame (to find when content last differed from the
        static closing). The full content-change window is preserved plus a
        small pad. This removes pre-focus and post-stability dead air without
        truncating the progressive line-paste composition the narration tracks.
        """
        clip_path = Path(clip_path)
        if not clip_path.exists():
            logger.warning("Trim skipped: clip does not exist: %s", clip_path)
            return

        cap = cv2.VideoCapture(str(clip_path))
        if not cap.isOpened():
            logger.warning("Trim skipped: could not open clip: %s", clip_path)
            return

        fps = cap.get(cv2.CAP_PROP_FPS) or 10.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        original_duration = total_frames / fps if fps > 0 else 0.0
        if original_duration <= MIN_CLIP_DURATION_SECONDS or total_frames <= 0:
            cap.release()
            logger.info("Trim skipped: clip too short: %s", clip_path.name)
            return

        sample_interval = max(1, int(round(fps / 2.0)))  # ~2 fps sampling
        samples: List[Tuple[int, np.ndarray]] = []
        frame_idx = 0
        # Ceiling: MAX_TRIM_FRAMES; abort loudly if decode runs away.
        while frame_idx < MAX_TRIM_FRAMES:
            ok, frame = cap.read()
            if not ok:
                break
            if frame_idx % sample_interval == 0:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                gray = cv2.resize(gray, (320, 180), interpolation=cv2.INTER_AREA)
                samples.append((frame_idx, gray))
            frame_idx += 1
        else:
            print(
                f"[CIRCUIT BREAKER] _trim_clip_to_motion hit frame ceiling "
                f"({MAX_TRIM_FRAMES}); stopping trim.",
                file=sys.stderr,
            )
        cap.release()

        if len(samples) < 2:
            logger.info("Trim skipped: not enough samples: %s", clip_path.name)
            return

        def _mean_diff(a: np.ndarray, b: np.ndarray) -> float:
            return float(np.mean(np.abs(a.astype(np.float32) - b.astype(np.float32))))

        head_ref = samples[0][1]
        tail_ref = samples[-1][1]

        # Find the first frame that departs from the static opening and the
        # last frame that departs from the static closing. The window between
        # them (plus pad) is the action we want to keep.
        first_change_idx: Optional[int] = None
        last_change_idx: Optional[int] = None
        for idx, (_frame_idx, gray) in enumerate(samples):
            if first_change_idx is None and _mean_diff(gray, head_ref) >= MOTION_DIFF_THRESHOLD:
                first_change_idx = idx
            if _mean_diff(gray, tail_ref) >= MOTION_DIFF_THRESHOLD:
                last_change_idx = idx

        if first_change_idx is None or last_change_idx is None:
            # No detectable content change: keep a short middle slice.
            keep_samples = max(
                1, int(round(NO_MOTION_KEEP_SECONDS * fps / sample_interval))
            )
            half = keep_samples // 2
            mid_sample = len(samples) // 2
            first_change_idx = max(0, mid_sample - half)
            last_change_idx = min(len(samples) - 1, mid_sample + half)

        pad_samples = max(1, int(round(MOTION_PAD_SECONDS * fps / sample_interval)))
        first_sample = max(0, first_change_idx - pad_samples)
        last_sample = min(len(samples) - 1, last_change_idx + pad_samples)

        start_frame = samples[first_sample][0]
        end_frame = samples[last_sample][0]
        start_sec = start_frame / fps
        # Add one sample interval to the end to include the sampled frame itself.
        end_sec = min(original_duration, (end_frame / fps) + (sample_interval / fps))
        duration = end_sec - start_sec

        if duration < MIN_CLIP_DURATION_SECONDS:
            end_sec = min(original_duration, start_sec + MIN_CLIP_DURATION_SECONDS)
            duration = end_sec - start_sec
            if duration < MIN_CLIP_DURATION_SECONDS and start_sec > 0:
                start_sec = max(0.0, end_sec - MIN_CLIP_DURATION_SECONDS)
                duration = end_sec - start_sec

        if duration <= 0:
            logger.warning("Trim skipped: computed duration <= 0 for %s", clip_path)
            return

        temp_path = clip_path.with_suffix(".trimmed.mp4")
        try:
            subprocess.run(
                [
                    "ffmpeg",
                    "-y",
                    "-ss",
                    str(start_sec),
                    "-i",
                    str(clip_path),
                    "-t",
                    str(duration),
                    "-c:v",
                    "libx264",
                    "-pix_fmt",
                    "yuv420p",
                    "-movflags",
                    "+faststart",
                    str(temp_path),
                ],
                check=True,
                capture_output=True,
                timeout=120,
            )
        except subprocess.CalledProcessError as exc:
            stderr = exc.stderr.decode("utf-8", errors="replace") if exc.stderr else ""
            logger.warning("Failed to trim %s: %s", clip_path, stderr[:200])
            if temp_path.exists():
                temp_path.unlink()
            return

        if not temp_path.exists() or temp_path.stat().st_size == 0:
            logger.warning("Trim produced empty output for %s", clip_path)
            if temp_path.exists():
                temp_path.unlink()
            return

        shutil.move(str(temp_path), str(clip_path))
        logger.info(
            "Trimmed %s: original=%.2fs -> trimmed=%.2fs (start=%.2f, duration=%.2f)",
            clip_path.name,
            original_duration,
            duration,
            start_sec,
            duration,
        )

    def _launch_app(self, db_path: Path) -> None:
        """Open the target application on the provided database file."""
        app_name = self.profile.app_name
        subprocess.run(
            ["open", "-a", app_name, str(db_path)],
            check=True,
            capture_output=True,
            timeout=30,
        )
        time.sleep(6)
        subprocess.run(
            ["osascript", "-e", f'tell application "{app_name}" to activate'],
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
                    tell application "{app_name}" to activate
                    delay 0.5
                    tell application "System Events"
                        tell process "{app_name}"
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
            # Window resizing is best-effort; don't fail discovery because of it.
            pass

    def _auto_fit_columns(self) -> None:
        """
        Setup step: ask the vision model to auto-fit DB Browser column widths
        when a table is already visible.  If no table is visible, the model
        should reply SETUP COMPLETE without taking any action.
        """
        try:
            b64, _, _, scale_to_logical, _, _ = _capture_screenshot(self.output_dir)
        except Exception as exc:
            print(
                f"Warning: column auto-fit setup skipped (screenshot failed): {exc}",
                file=sys.stderr,
            )
            return

        instruction = (
            f"Setup step for the recording: ensure all column headers in any visible "
            f"{self.profile.app_name} table are wide enough to show their full names. If a table with "
            "columns is visible, double-click the border between each pair of column "
            "headers to auto-fit widths. If no table is visible, do nothing. When "
            "finished (or if nothing needs to be done), reply exactly: SETUP COMPLETE."
        )
        messages: List[Dict[str, Any]] = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": b64}},
                    {"type": "text", "text": instruction},
                ],
            }
        ]

        # Ceiling: 4 vision-turn auto-fit attempts.
        for _ in range(4):
            try:
                response = tracked_create(
                    self.client, model=MODEL, max_tokens=512, messages=messages
                )
            except Exception as exc:
                print(
                    f"Warning: column auto-fit API call failed: {exc}", file=sys.stderr
                )
                break

            text_parts = [block.text for block in response.content if block.type == "text"]
            full_text = "\n".join(text_parts)
            if "SETUP COMPLETE" in full_text:
                break

            action = _extract_action(full_text)
            if action is None:
                break

            try:
                _execute_action(action, scale_to_logical)
                time.sleep(0.5)
            except Exception as exc:
                print(
                    f"Warning: column auto-fit action failed: {exc}", file=sys.stderr
                )
                break

            # Capture the result and continue the setup conversation.
            try:
                b64, _, _, scale_to_logical, _, _ = _capture_screenshot(self.output_dir)
            except Exception as exc:
                print(
                    f"Warning: column auto-fit setup skipped (screenshot failed): {exc}",
                    file=sys.stderr,
                )
                break

            messages.append({"role": "assistant", "content": response.content})
            messages.append(
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": b64}},
                        {"type": "text", "text": "Continue. Reply SETUP COMPLETE when done."},
                    ],
                }
            )

    def _auto_fit_if_truncated(self, b64: str, scale_to_logical: float) -> bool:
        """
        Check whether the current screenshot shows a table with truncated column
        headers. If so, ask the vision model to double-click a column-border to
        auto-fit, execute the action, and repeat until no truncation remains or
        a turn limit is reached. Returns True if any auto-fit was performed.
        """
        instruction = (
            f"Check this {self.profile.app_name} screenshot. If a table with column headers is visible "
            "and any header text appears truncated (cut off with '...' or not fully readable), "
            "return ONE JSON action to double-click the border between two column headers to auto-fit: "
            '{"action": "double_click", "bbox": {"x": int, "y": int, "w": int, "h": int}}. '
            "The bbox should tightly contain a visible column-header border. "
            "If no table is visible or no column headers are truncated, reply exactly: NO TRUNCATION."
        )
        messages: List[Dict[str, Any]] = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": b64}},
                    {"type": "text", "text": instruction},
                ],
            }
        ]

        acted = False
        # Ceiling: 3 vision-turn truncation auto-fit attempts.
        for _ in range(3):
            try:
                response = tracked_create(
                    self.client, model=MODEL, max_tokens=512, messages=messages
                )
            except Exception as exc:
                print(f"Warning: truncation-check API call failed: {exc}", file=sys.stderr)
                break

            text_parts = [block.text for block in response.content if block.type == "text"]
            full_text = "\n".join(text_parts)
            if "NO TRUNCATION" in full_text.upper():
                break

            action = _extract_action(full_text)
            if action is None:
                break

            try:
                _execute_action(action, scale_to_logical)
                acted = True
                time.sleep(0.5)
            except Exception as exc:
                print(f"Warning: auto-fit action failed: {exc}", file=sys.stderr)
                break

            try:
                b64, _, _, scale_to_logical, _, _ = _capture_screenshot(self.output_dir)
            except Exception as exc:
                print(f"Warning: post-auto-fit screenshot failed: {exc}", file=sys.stderr)
                break

            messages.append({"role": "assistant", "content": response.content})
            messages.append(
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": b64}},
                        {"type": "text", "text": "Check again. Reply NO TRUNCATION if all column headers are fully readable."},
                    ],
                }
            )

        return acted

    def _results_grid_visible(self, b64_image: str, max_retries: int = 2) -> bool:
        """
        Ask the vision model whether the execution tab shows a populated
        results grid. Retry a few times because the model can be inconsistent.
        """
        execute_tab = self.profile.landmarks.get("execute_tab", "Execute SQL tab")
        result_pane = self.profile.landmarks.get("result_pane", "the result pane")
        instruction = (
            f"Look at this {self.profile.app_name} screenshot. The {execute_tab} tab is active "
            "and a SQL query has just been executed. Look at the area below the editor. "
            f"Reply exactly YES if you see any of the following in {result_pane}: a table/grid of data rows, "
            "or text saying 'Result: N rows returned', or 'Execution finished without errors'. "
            "Reply exactly NO only if the lower pane is completely empty or shows only "
            "the placeholder text 'Results of the last executed statements' with no data."
        )
        messages: List[Dict[str, Any]] = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": b64_image}},
                    {"type": "text", "text": instruction},
                ],
            }
        ]
        # Ceiling: max_retries + 1 (default 3) results-grid verification calls.
        for attempt in range(max_retries + 1):
            try:
                response = tracked_create(
                    self.client, model=MODEL, max_tokens=32, messages=messages
                )
            except Exception as exc:
                print(f"Warning: results-grid verification API call failed: {exc}", file=sys.stderr)
                return False

            text_parts = [block.text for block in response.content if block.type == "text"]
            full_text = "\n".join(text_parts).strip().upper()
            if full_text.startswith("YES"):
                return True
            if attempt < max_retries:
                time.sleep(0.5)
        return False

    def _log_attempt(
        self,
        run_id: str,
        attempt: int,
        success: bool,
        reason: str,
        action: Optional[str],
        target: Optional[Dict[str, Any]],
        screenshot_hash: str,
        tokens: Tuple[int, int],
        cost: float,
        latency: float,
        error: Optional[str],
        api_width_px: int = 0,
        api_height_px: int = 0,
        video_path: Optional[str] = None,
    ) -> None:
        log: Dict[str, Any] = {
            "run_id": run_id,
            "attempt": attempt,
            "timestamp": time.time(),
            "success": success,
            "reason": reason,
            "action_taken": action,
            "target": target,
            "screenshot_sha256": screenshot_hash,
            "api_width_px": api_width_px,
            "api_height_px": api_height_px,
            "input_tokens": tokens[0],
            "output_tokens": tokens[1],
            "cost_usd": round(cost, 6),
            "latency_seconds": round(latency, 3),
            "error": error,
            "video_path": video_path,
        }
        self.attempt_logs.append(log)
        with open(self.telemetry_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(log) + "\n")

    def _assert_stage_resources(self) -> Dict[str, Any]:
        """
        Stage-prep resource assertion.

        Before any beat executes, confirm:
          - DB Browser for SQLite is the frontmost process.
          - The declared database file is open (file exists and app is running).
          - SELECT COUNT(*) FROM Customer returns exactly 60.

        Returns {"ok": True} on success or {"ok": False, "reason": str} on failure.
        """
        app_name = self.profile.app_name
        frontmost = _frontmost_app_name()
        if frontmost != app_name:
            return {
                "ok": False,
                "reason": f"[STAGE PREP] frontmost app is {frontmost!r}, expected {app_name!r}",
            }

        db_path = self.db_path
        if not db_path or not db_path.exists():
            return {
                "ok": False,
                "reason": f"[STAGE PREP] database not found: {db_path}",
            }

        try:
            conn = sqlite3.connect(str(db_path))
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM Customer")
            count = cursor.fetchone()[0]
            conn.close()
        except Exception as exc:
            return {
                "ok": False,
                "reason": f"[STAGE PREP] probe query failed: {exc}",
            }

        if count != 60:
            return {
                "ok": False,
                "reason": f"[STAGE PREP] Customer row count is {count}, expected 60",
            }

        print("[STAGE PREP] DB open and Customer count = 60", file=sys.stderr)
        return {"ok": True}

    def _make_result(
        self,
        success: bool,
        reason: str = "",
        locked_state: Optional[ScreenState] = None,
        recipe: bool = False,
        final_editor_content: Optional[str] = None,
    ) -> DiscoveryResult:
        attempts = len(self.attempt_logs)
        if recipe:
            # Deterministic recipes are fully reliable when they reach the end state.
            successful_attempts = attempts if success else 0
            reliability_score = 1.0 if success else 0.0
        else:
            successful_attempts = sum(1 for log in self.attempt_logs if log["success"])
            reliability_score = (successful_attempts / attempts) if attempts else 0.0
        costs = [log["cost_usd"] for log in self.attempt_logs]
        total = sum(costs)
        mean = total / attempts if attempts else 0.0
        std = 0.0
        if attempts > 1:
            variance = sum((c - mean) ** 2 for c in costs) / attempts
            std = math.sqrt(variance)

        return DiscoveryResult(
            success=success,
            locked_state=locked_state,
            attempts=attempts,
            successful_attempts=successful_attempts,
            reliability_score=reliability_score,
            mean_cost_usd=round(mean, 6),
            std_cost_usd=round(std, 6),
            attempt_logs=self.attempt_logs,
            final_editor_content=final_editor_content,
        )
