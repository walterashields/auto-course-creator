#!/usr/bin/env python3
"""
compiler/c34_probe.py

C34 STEP 3 probe: mss pull-based capture vs the beat_001 failure signature,
WITH TTS audio — the one production ingredient the C29/C32a probes never
replayed (afplay during capture).

Arms (30s each, production beat_001 mix: paste cadence + AX read-backs, then
hover(0.7s tween)+pause choreography; a real cached TTS mp3 loops via afplay):
  (a) _MssWindowRecorder (pull)  — report grab duty + timing
  (b) _ScreenCaptureKitRecorder (push) — report delivery duty, for the record

Hypothesis under test: the C31/C33 mid-clip stall (healthy seconds 0-9, zero
delivery seconds 10-13) is a push-delivery coalescing artifact; the pull
backend must hold full duty with audio playing.

Run:  python -m compiler.c34_probe
Needs DB Browser for SQLite open (pastes land in its SQL editor).
Probes leave the stage as found (C33): stray modals dismissed at exit.
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
import time
import uuid
from pathlib import Path

import pyautogui

from compiler import ax_pyobjc
from compiler.discovery import _MssWindowRecorder, _ScreenCaptureKitRecorder

APP_NAME = "DB Browser for SQLite"
ARM_SECONDS = 30.0
OUT_DIR = Path(__file__).resolve().parent / "discovery_output"
TTS_CACHE = Path(__file__).resolve().parent / "tts_cache"

BEAT001_LINES = [
    "/*",
    "Created By: WSDA Music Analyst",
    "Description: Contact list for management",
    "*/",
    "SELECT FirstName, LastName, Email FROM Customer;",
]


def _activate_app() -> None:
    subprocess.run(
        ["osascript", "-e", f'tell application "{APP_NAME}" to activate'],
        capture_output=True, timeout=10,
    )
    time.sleep(1.0)


def _focus_editor() -> None:
    w, h = pyautogui.size().width, pyautogui.size().height
    pyautogui.click(int(w * 0.50), int(h * 0.39))
    time.sleep(0.3)


def _paste_line(line: str) -> None:
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
    subprocess.run(
        [
            "osascript", "-e",
            f'tell application "System Events" to tell process "{APP_NAME}" '
            'to get value of attribute "AXFocused" of text area 1 of group 1 '
            "of splitter group 1 of window 1",
        ],
        capture_output=True, timeout=10,
    )


def _hover(fx: float, fy: float) -> None:
    w, h = pyautogui.size().width, pyautogui.size().height
    pyautogui.moveTo(int(w * fx), int(h * fy),
                     duration=0.7, tween=pyautogui.easeInOutQuad)
    time.sleep(0.05)


def _audio_path() -> Path:
    """One real cached TTS mp3 (any beat) to stand in for production audio."""
    mp3s = sorted(TTS_CACHE.glob("*.mp3"))
    if not mp3s:
        raise RuntimeError("no cached TTS mp3 in compiler/tts_cache")
    return mp3s[0]


def _beat_mix(t0: float, audio_proc: subprocess.Popen) -> None:
    """Production beat_001 shape: pastes + AX read-backs, then hover/pause
    choreography, with afplay running for the whole arm."""
    for line in BEAT001_LINES:
        _paste_line(line)
        _ax_read()
    while time.monotonic() - t0 < ARM_SECONDS:
        if audio_proc.poll() is not None:
            # Loop the narration like a longer beat would.
            audio_proc = subprocess.Popen(
                ["afplay", str(_audio_path())],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
        _hover(0.50, 0.40)
        time.sleep(1.31)


def _table(summary: dict) -> str:
    per_second = summary.get("per_second") or []
    delivered = [b.get("frames_delivered", 0) for b in per_second]
    failed = [b.get("grabs_failed", 0) for b in per_second]
    span = summary.get("span_seconds", 0.0)
    expected = max(int(round(span * 10)), 1)
    total = sum(delivered)
    return (
        f"grabbed {total}/{expected} ({summary.get('delivered_fps', 0.0):.2f} fps, "
        f"duty {100.0 * total / expected:.0f}%) | failed {sum(failed)} "
        f"(worst/s {max(failed) if failed else 0}) | convert p50 "
        f"{summary.get('convert_ms_p50', 0.0)}ms max "
        f"{summary.get('convert_ms_max', 0.0)}ms | encode p50 "
        f"{summary.get('encode_ms_p50', 0.0)}ms max "
        f"{summary.get('encode_ms_max', 0.0)}ms | per-s {delivered}"
    )


def _run_arm(name: str, recorder_cls) -> None:
    path = OUT_DIR / f"c34_probe_{uuid.uuid4().hex[:8]}.mp4"
    recorder = recorder_cls(str(path), fps=10, app_name=APP_NAME)
    audio_path = _audio_path()
    audio_proc = subprocess.Popen(
        ["afplay", str(audio_path)],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    t0 = time.monotonic()
    try:
        recorder.start()
        if isinstance(recorder, _ScreenCaptureKitRecorder) and recorder._fallback is not None:
            raise RuntimeError("SCK stream fell back to MSS; arm invalid")
        _beat_mix(t0, audio_proc)
        recorder.stop()
    finally:
        if audio_proc.poll() is None:
            audio_proc.terminate()
        path.unlink(missing_ok=True)
    summary = recorder.delivery_summary or {}
    print(f"wsda-probe: {name}\nwsda-probe:   {_table(summary)}")


def _dismiss_stray_modals() -> None:
    """C33 probe hygiene: leave the stage as found."""
    try:
        pid = ax_pyobjc.app_pid_for_name(APP_NAME)
        if pid is None:
            return
        app_el = ax_pyobjc.create_application(pid)
        for _ in range(3):
            modal = ax_pyobjc.frontmost_modal(app_el)
            if modal is None:
                return
            el, title = modal
            try:
                pressed = ax_pyobjc.press_button(el, "Cancel")
            except ax_pyobjc.AxCallError:
                pressed = False
            if not pressed:
                subprocess.run(
                    ["osascript", "-e",
                     'tell application "System Events" to key code 53'],
                    capture_output=True, timeout=5,
                )
            time.sleep(0.4)
            remaining = ax_pyobjc.frontmost_modal(app_el)
            if remaining is not None and remaining[1] == title:
                print(f"wsda-probe-modal-remaining:{title}")
                return
            print(f"wsda-probe-modal-dismissed:{title}")
    except ax_pyobjc.AxCallError as exc:
        print(f"wsda-probe-modal-cleanup:error {exc}")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    _activate_app()
    _focus_editor()
    arms = [
        ("a) mss pull backend + beat_001 mix + afplay TTS audio",
         lambda p, fps, app_name: _MssWindowRecorder(p, fps=fps, app_name=app_name)),
        ("b) SCK push backend + beat_001 mix + afplay TTS audio (record only)",
         lambda p, fps, app_name: _ScreenCaptureKitRecorder(p, fps=fps, app_name=app_name)),
    ]
    for name, cls in arms:
        try:
            _run_arm(name, cls)
        except Exception as exc:  # noqa: BLE001 — probe reports and continues
            print(f"wsda-probe: {name} FAILED: {exc}")
        time.sleep(2.0)  # SCK teardown settle between arms
    _dismiss_stray_modals()
    print("wsda-probe: done")


if __name__ == "__main__":
    sys.exit(main())
