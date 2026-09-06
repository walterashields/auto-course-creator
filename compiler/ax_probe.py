#!/usr/bin/env python3
"""
compiler/ax_probe.py

C21 isolation probe: measure the two AppleScript AX queries used by the focus
path — (1) the ``entire contents`` enumeration (find + focus the top-most
AXTextArea) and (2) the system-wide focused-element read — under four load
conditions, in order:

  A) idle
  B) ScreenCaptureKit recorder only (same recorder discovery uses)
  C) TTS audio playback only (afplay loop, same playback path discovery uses)
  D) SCK recorder + audio together

Each condition runs 60s with a probe every 2.0s. Every probe records its
marker and elapsed seconds. Results go to output/ax_probe_report.json and a
condition x marker counts table with p50/max elapsed is printed.

Run with DB Browser for SQLite open on the Execute SQL tab and Terminal
declared as the controlling terminal, exactly like a recording session, but
without the curriculum:

    WSDA_CONTROLLING_TERMINAL=Terminal python -m compiler.ax_probe
"""

from __future__ import annotations

import json
import statistics
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Dict, List

from .vision_agent import (
    _AX_ENUM_TIMEOUT_SECONDS,
    _AX_GUARD_TIMEOUT_SECONDS,
    _ax_enumeration_script,
    _ax_focused_element_script,
    _run_ax_script,
)

PROCESS_NAME = "DB Browser for SQLite"
PROBE_INTERVAL_SECONDS = 2.0
CONDITION_SECONDS = 60.0
REPORT_PATH = Path("output") / "ax_probe_report.json"

CONDITIONS = ["A_idle", "B_sck_recorder", "C_audio", "D_sck_plus_audio"]


def _find_cached_tts_mp3() -> Path:
    cache = Path("compiler") / "tts_cache"
    if cache.is_dir():
        mp3s = sorted(cache.glob("*.mp3"))
        if mp3s:
            return mp3s[0]
    raise SystemExit(
        f"[AX PROBE] no cached TTS mp3 found under {cache}; cannot run condition C/D"
    )


class _AudioLooper:
    """Loop an mp3 via afplay (the same playback path discovery uses)."""

    def __init__(self, mp3_path: Path):
        self._mp3_path = mp3_path
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _loop(self) -> None:
        while not self._stop.is_set():
            proc = subprocess.Popen(
                ["afplay", str(self._mp3_path)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            # afplay runs to the end of the file; stop early if told to.
            while proc.poll() is None and not self._stop.is_set():
                time.sleep(0.2)
            if proc.poll() is None:
                proc.terminate()

    def start(self) -> None:
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)


def _probe_once(enum_script: str, guard_script: str) -> Dict[str, Any]:
    enum_marker, enum_elapsed, enum_detail = _run_ax_script(
        enum_script, _AX_ENUM_TIMEOUT_SECONDS
    )
    guard_marker, guard_elapsed, guard_detail = _run_ax_script(
        guard_script, _AX_GUARD_TIMEOUT_SECONDS
    )
    return {
        "t": time.strftime("%H:%M:%S"),
        "enum": {"marker": enum_marker, "elapsed": round(enum_elapsed, 3),
                 "detail": enum_detail},
        "guard": {"marker": guard_marker, "elapsed": round(guard_elapsed, 3),
                  "detail": guard_detail},
    }


def _summarize(condition: str, probes: List[Dict[str, Any]]) -> Dict[str, Any]:
    summary: Dict[str, Any] = {"probes": len(probes), "enum": {}, "guard": {}}
    for key in ("enum", "guard"):
        markers: Dict[str, int] = {}
        elapseds: List[float] = []
        for p in probes:
            markers[p[key]["marker"]] = markers.get(p[key]["marker"], 0) + 1
            elapseds.append(p[key]["elapsed"])
        summary[key] = {
            "marker_counts": dict(sorted(markers.items())),
            "p50_elapsed": round(statistics.median(elapseds), 3) if elapseds else 0.0,
            "max_elapsed": round(max(elapseds), 3) if elapseds else 0.0,
        }
    return summary


def main() -> int:
    # Fail fast if the target app is not running — the probe is only
    # meaningful in the recording-session window configuration.
    check = subprocess.run(
        ["osascript", "-e",
         f'tell application "System Events" to count (every process whose name is '
         f'{json.dumps(PROCESS_NAME)})'],
        capture_output=True, text=True, timeout=10,
    )
    if check.returncode != 0 or (check.stdout or "").strip() not in ("1", "2", "3", "4", "5"):
        print(f"[AX PROBE] {PROCESS_NAME!r} is not running; open it on the "
              f"Execute SQL tab first.", file=sys.stderr)
        return 1

    mp3_path = _find_cached_tts_mp3()
    enum_script = _ax_enumeration_script(PROCESS_NAME)
    guard_script = _ax_focused_element_script(PROCESS_NAME)

    results: Dict[str, Any] = {
        "process": PROCESS_NAME,
        "probe_interval_seconds": PROBE_INTERVAL_SECONDS,
        "condition_seconds": CONDITION_SECONDS,
        "enum_timeout_seconds": _AX_ENUM_TIMEOUT_SECONDS,
        "guard_timeout_seconds": _AX_GUARD_TIMEOUT_SECONDS,
        "audio_file": str(mp3_path),
        "conditions": {},
    }

    for condition in CONDITIONS:
        probes: List[Dict[str, Any]] = []
        recorder = None
        audio = None
        scratch: tempfile.TemporaryDirectory | None = None
        if condition in ("B_sck_recorder", "D_sck_plus_audio"):
            from .discovery import _ScreenCaptureKitRecorder

            scratch = tempfile.TemporaryDirectory(prefix="wsda_ax_probe_")
            out = Path(scratch.name) / f"{condition}.mp4"
            recorder = _ScreenCaptureKitRecorder(
                str(out), fps=10, app_name=PROCESS_NAME
            )
            recorder.start()
            print(f"[AX PROBE] {condition}: SCK recorder -> {out}", file=sys.stderr)
        if condition in ("C_audio", "D_sck_plus_audio"):
            audio = _AudioLooper(mp3_path)
            audio.start()
            print(f"[AX PROBE] {condition}: afplay loop of {mp3_path.name}",
                  file=sys.stderr)

        deadline = time.time() + CONDITION_SECONDS
        next_probe = time.time()
        try:
            while time.time() < deadline:
                now = time.time()
                if now < next_probe:
                    time.sleep(next_probe - now)
                next_probe = time.time() + PROBE_INTERVAL_SECONDS
                probe = _probe_once(enum_script, guard_script)
                probes.append(probe)
                print(
                    f"[AX PROBE] {condition} {probe['t']} "
                    f"enum={probe['enum']['marker']} ({probe['enum']['elapsed']}s) "
                    f"guard={probe['guard']['marker']} ({probe['guard']['elapsed']}s)",
                    file=sys.stderr,
                )
        finally:
            if recorder is not None:
                recorder.stop()
            if audio is not None:
                audio.stop()
            if scratch is not None:
                scratch.cleanup()

        results["conditions"][condition] = {
            "probes": probes,
            "summary": _summarize(condition, probes),
        }

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(results, indent=2))
    print(f"[AX PROBE] wrote {REPORT_PATH}", file=sys.stderr)

    print("\ncondition | probe | marker counts | p50 elapsed | max elapsed")
    for condition in CONDITIONS:
        for key, label in (("enum", "enumeration"), ("guard", "focused-el")):
            s = results["conditions"][condition]["summary"][key]
            counts = ", ".join(f"{m}:{c}" for m, c in s["marker_counts"].items())
            print(f"{condition} | {label} | {counts} | "
                  f"{s['p50_elapsed']}s | {s['max_elapsed']}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
