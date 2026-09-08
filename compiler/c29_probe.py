#!/usr/bin/env python3
"""
compiler/c29_probe.py

C29 STEP 1 diagnostic probe: what starves SCK delivery during demo beats?

Replays three 30-second action mixes against fresh production-configured SCK
streams (same recorder class the pipeline uses, same 1920x1200 capture config,
same 10fps wall-clock writer) and reports per-second delivery duty plus
per-frame convert/encode timing:

  (a) beat_003 mix: 5 production-cadence line pastes, then hover(0.7s tween)
      + 1.31s pause choreography for the remainder -- mirrors the starved beat.
  (b) beat_003 mix + a concurrent ``screencapture -x`` subprocess at every
      hover -- the governor's grounding-screenshot hypothesis. Production
      choreography resolves targets to fixed points (no VLM call), but VLM
      grounding elsewhere uses VisionAgent._capture_screen() == the
      screencapture CLI; this arm measures that capture path at production
      cadence against a live SCK stream.
  (c) beat_002 mix (control): production-cadence line pastes continuously for
      the full 30s -- mirrors the healthy beat.

Run:  python -m compiler.c29_probe
Needs DB Browser for SQLite open (pastes land in its SQL editor).
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
import time
from pathlib import Path

import pyautogui

from compiler.discovery import _ScreenCaptureKitRecorder

APP_NAME = "DB Browser for SQLite"
ARM_SECONDS = 30.0
OUT_DIR = Path(__file__).resolve().parent / "discovery_output"

# Production beat_003 content (comment header + SELECT clause, 5 lines).
BEAT003_LINES = [
    "/*",
    "Created By: WSDA Music Analyst",
    "Description: Contact list for management",
    "*/",
    "SELECT FirstName, LastName, Email FROM Customer;",
]

# Production beat_002 content (comment header, 2 segments -> sustained typing).
BEAT002_LINES = [
    "/*",
    "Created By: WSDA Music Analyst",
    "Create Date: 2026-09-08",
    "Description: Contact list query for management review",
    "*/",
]


def _activate_app() -> None:
    subprocess.run(
        ["osascript", "-e", f'tell application "{APP_NAME}" to activate'],
        capture_output=True, timeout=10,
    )
    time.sleep(1.0)


def _focus_editor() -> None:
    """Click the canonical SQL-editor point (matches choreography targets)."""
    w, h = pyautogui.size().width, pyautogui.size().height
    pyautogui.click(int(w * 0.50), int(h * 0.39))
    time.sleep(0.3)


def _paste_line(line: str) -> None:
    """Production paste cadence: clipboard set, cmd+v, flat 0.4s pace."""
    subprocess.run(["pbcopy"], input=line.encode(), check=True)
    subprocess.run(
        [
            "osascript", "-e",
            'tell application "System Events" to keystroke "v" using command down',
        ],
        capture_output=True, timeout=10,
    )
    time.sleep(0.4)


def _ax_read() -> None:
    """One System Events AX read (production read-back churn class)."""
    subprocess.run(
        [
            "osascript", "-e",
            f'tell application "System Events" to tell process "{APP_NAME}" '
            "to get value of attribute \"AXFocused\" of text area 1 of group 1 "
            "of splitter group 1 of window 1",
        ],
        capture_output=True, timeout=10,
    )


def _hover(fx: float, fy: float) -> None:
    """Production hover: 0.7s eased cursor tween to a fractional point."""
    w, h = pyautogui.size().width, pyautogui.size().height
    pyautogui.moveTo(int(w * fx), int(h * fy), duration=0.7,
                     tween=pyautogui.easeInOutQuad)
    time.sleep(0.05)


def _grounding_screenshot() -> None:
    """One VisionAgent._capture_screen-equivalent call (screencapture CLI)."""
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        path = tmp.name
    try:
        subprocess.run(["screencapture", "-x", path],
                       capture_output=True, timeout=10)
    finally:
        Path(path).unlink(missing_ok=True)


def _mix_beat003(with_grounding: bool) -> None:
    """5s of pastes, then hover+pause choreography (the starved beat's shape)."""
    for line in BEAT003_LINES:
        _paste_line(line)
        _ax_read()
    while time.monotonic() - _arm_t0 < ARM_SECONDS:
        _hover(0.50, 0.40)
        if with_grounding:
            _grounding_screenshot()
        time.sleep(1.31)


def _mix_beat002() -> None:
    """Continuous pastes for the whole arm (the healthy beat's shape)."""
    idx = 0
    while time.monotonic() - _arm_t0 < ARM_SECONDS:
        line = BEAT002_LINES[idx % len(BEAT002_LINES)]
        _paste_line(line)
        _ax_read()
        idx += 1


_arm_t0 = 0.0


def _run_arm(name: str, mix, path: Path) -> dict:
    global _arm_t0
    recorder = _ScreenCaptureKitRecorder(str(path), fps=10, app_name=APP_NAME)
    recorder.start()
    if recorder._fallback is not None:
        recorder.stop()
        raise RuntimeError("SCK stream fell back to MSS; probe invalid")
    _arm_t0 = time.monotonic()
    mix()
    recorder.stop()
    return recorder.delivery_summary or {}


def _duty_table(summary: dict) -> str:
    per_second = summary.get("per_second") or []
    delivered = [b.get("frames_delivered", 0) for b in per_second]
    span = summary.get("span_seconds", 0.0)
    expected = max(int(round(span * 10)), 1)
    total_delivered = sum(delivered)
    duty = 100.0 * total_delivered / expected
    return (
        f"delivered {total_delivered}/{expected} frames "
        f"({summary.get('delivered_fps', 0.0):.2f} fps, duty {duty:.0f}%) | "
        f"convert p50 {summary.get('convert_ms_p50', 0.0)}ms "
        f"max {summary.get('convert_ms_max', 0.0)}ms | "
        f"encode p50 {summary.get('encode_ms_p50', 0.0)}ms "
        f"max {summary.get('encode_ms_max', 0.0)}ms | "
        f"per-s {delivered}"
    )


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    _activate_app()
    _focus_editor()

    arms = [
        ("a) beat_003 mix (pastes + hover/pause choreo)", lambda: _mix_beat003(False)),
        ("b) beat_003 mix + screencapture per hover (grounding hypothesis)",
         lambda: _mix_beat003(True)),
        ("c) beat_002 mix (continuous pastes, control)", _mix_beat002),
    ]
    for name, mix in arms:
        path = OUT_DIR / f"c29_probe_{uuid4hex()}.mp4"
        summary = _run_arm(name, mix, path)
        path.unlink(missing_ok=True)
        print(f"wsda-probe: {name}\nwsda-probe:   {_duty_table(summary)}")


def uuid4hex() -> str:
    import uuid
    return uuid.uuid4().hex[:8]


if __name__ == "__main__":
    sys.exit(main())
