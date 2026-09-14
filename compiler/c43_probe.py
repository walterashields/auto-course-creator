"""C43 STEP 1 probe — is the heartbeat/break-park motion VISIBLE in the mss recording?

Diagnosis target (C42 regression, B3 11.0s FAIL): the MSS writer composites a
magenta cursor sprite per grabbed frame. The C42 hypothesis was a cached
position channel. This probe establishes the ACTUAL channel and whether motion
that flows ONLY through the watchdog break-motion path shows up in the written
video at the B3 sampler's granularity (1fps, then consecutive-sample compare).

Method (no VLM, ~20s, real screen):
  1. Report the writer's cursor position source (read from the constructor).
  2. Stage DB Browser for SQLite (launch if needed), prime geometry.
  3. Park the physical cursor once (setup, before recorder start).
  4. Run the real _MssWindowRecorder at 10fps while a real ParkWatchdog with
     the real VisionAgent._watchdog_break_motion heartbeat fires every ~3s
     (registry target: the SQL editor body).
  5. Read the clip back; locate the magenta sprite centroid in every frame;
     downsample to 1fps; for every consecutive 1fps sample pair that overlaps
     a fire, assert sprite centroid displacement >= 40px.

Verdict: exits 0 when every fire-overlapping 1fps pair shows >= 40px sprite
displacement (motion VISIBLE at B3 granularity); exits 1 otherwise, printing
the per-pair table so the invisible-motion intervals are explicit.
"""

from __future__ import annotations

import inspect
import json
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pyautogui

from .choreo_runtime import HEARTBEAT_THRESHOLD, HEARTBEAT_TICK, ParkWatchdog
from .discovery import EndStateDiscovery, _MssWindowRecorder, _window_bounds
from .vision_agent import VisionAgent

APP_NAME = "DB Browser for SQLite"
MIN_VISIBLE_DISPLACEMENT_PX = 40.0
PROBE_SECONDS = 14.0
OUT_DIR = Path("compiler/discovery_output")


def report_position_source() -> str:
    """Read the actual cursor-position source the MSS writer uses per grab."""
    src = inspect.getsource(_MssWindowRecorder.__init__)
    live = "cursor_fn or pyautogui.position" in src
    kind = (
        "live pyautogui.position() read fresh at every grab"
        if live
        else "NON-LIVE/CACHED SOURCE: " + next(
            (ln.strip() for ln in src.splitlines() if "_cursor_fn" in ln), "?"
        )
    )
    print(f"[C43-PROBE] writer cursor source: {kind}")
    return kind


def ensure_stage() -> dict:
    bounds = _window_bounds(APP_NAME)
    if bounds is None:
        print("[C43-PROBE] launching DB Browser for SQLite ...", file=sys.stderr)
        subprocess.run(["open", "-a", APP_NAME], check=False)
        deadline = time.monotonic() + 25.0
        while time.monotonic() < deadline:
            bounds = _window_bounds(APP_NAME)
            if bounds is not None:
                break
            time.sleep(0.5)
    if bounds is None:
        raise RuntimeError("wsda-c43-probe: stage not ready (no window bounds)")
    print(f"[C43-PROBE] stage bounds: {bounds}")
    return bounds


def sprite_centroid(frame_bgr: np.ndarray):
    b = frame_bgr[:, :, 0].astype(int)
    g = frame_bgr[:, :, 1].astype(int)
    r = frame_bgr[:, :, 2].astype(int)
    mask = (b > 200) & (g < 80) & (r > 200)
    ys, xs = np.nonzero(mask)
    if len(xs) < 8:
        return None
    return float(xs.mean()), float(ys.mean())


def main() -> int:
    report_position_source()
    bounds = ensure_stage()

    profile = EndStateDiscovery._default_profile_for_app("db_browser_sqlite")
    agent = VisionAgent(profile=profile)
    try:
        agent.prime_choreography_geometry()
    except Exception as exc:  # best effort; motion falls back to drift
        print(f"[C43-PROBE] geometry priming failed (drift fallback): {exc}")

    fire_times: list[float] = []

    def motion(reason: str) -> bool:
        fire_times.append(time.monotonic())
        return agent._watchdog_break_motion(reason)

    watchdog = ParkWatchdog(
        "c43-probe",
        motion=motion,
        heartbeat_tick=HEARTBEAT_TICK,
        heartbeat_threshold=HEARTBEAT_THRESHOLD,
    )
    agent.watchdog = watchdog
    watchdog.register_targets(["sql-editor:body"])

    # Setup park (before recorder start; outside the assertion window).
    cx = bounds["x"] + bounds["w"] * 0.5
    cy = bounds["y"] + bounds["h"] * 0.45  # Quartz y is bottom-origin
    top_y = pyautogui.size().height - cy  # convert to top-origin points
    pyautogui.moveTo(cx, top_y, duration=0.2)
    time.sleep(0.4)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    clip = OUT_DIR / "c43_probe_sprite.mp4"
    if clip.exists():
        clip.unlink()
    recorder = _MssWindowRecorder(str(clip), fps=10, app_name=APP_NAME)
    start_mono = time.monotonic()
    recorder.start()
    watchdog.start_heartbeat()
    time.sleep(PROBE_SECONDS)
    watchdog.stop_heartbeat()
    recorder.stop()
    end_mono = time.monotonic()

    fires = watchdog.fires
    print(
        f"[C43-PROBE] fires={len(fires)} reasons="
        f"{[f['reason'] for f in fires]}"
    )
    for i, ft in enumerate(fire_times):
        print(f"[C43-PROBE] fire {i}: t={ft - start_mono:.2f}s")

    # Read the clip back: sprite centroid per frame at 10fps.
    cap = cv2.VideoCapture(str(clip))
    centroids: list = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        centroids.append(sprite_centroid(frame))
    cap.release()
    n = len(centroids)
    print(f"[C43-PROBE] clip frames={n} duration={n / 10.0:.1f}s")
    if n < 20:
        print("[C43-PROBE] FAIL: clip too short to evaluate")
        return 1

    # 1fps samples (every 10th frame) with fire-overlap annotation.
    pairs = []
    for s in range(0, n - 10, 10):
        t0 = s / 10.0
        t1 = (s + 10) / 10.0
        overlapping = [
            i for i, ft in enumerate(fire_times)
            if t0 <= (ft - start_mono) < t1
        ]
        a, b = centroids[s], centroids[s + 10]
        if a is None or b is None:
            disp = None
        else:
            disp = float(((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5)
        pairs.append((t0, t1, overlapping, disp))

    print("[C43-PROBE] 1fps sample pairs (B3 granularity):")
    invisible = 0
    for t0, t1, overlapping, disp in pairs:
        tag = ""
        if overlapping:
            ok_pair = disp is not None and disp >= MIN_VISIBLE_DISPLACEMENT_PX
            tag = "FIRE" + ("" if ok_pair else " *** INVISIBLE ***")
            if not ok_pair:
                invisible += 1
        print(
            f"  [{t0:5.1f}s -> {t1:5.1f}s] fires={overlapping} "
            f"displacement={disp if disp is not None else 'no-sprite'}px {tag}"
        )

    # Also report 10fps visibility (does the sprite move at full grab rate?).
    full_moves = 0
    for i in range(n - 1):
        a, b = centroids[i], centroids[i + 1]
        if a and b:
            d = ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5
            if d >= MIN_VISIBLE_DISPLACEMENT_PX:
                full_moves += 1
    print(
        f"[C43-PROBE] 10fps frame pairs with >= {MIN_VISIBLE_DISPLACEMENT_PX:.0f}px "
        f"sprite displacement: {full_moves}/{n - 1}"
    )

    report = {
        "cursor_source": report_position_source(),
        "fires": len(fires),
        "invisible_fire_pairs": invisible,
        "full_rate_moving_pairs": full_moves,
        "clip": str(clip),
        "wall_span_s": round(end_mono - start_mono, 2),
    }
    print(f"[C43-PROBE] report: {json.dumps(report)}")
    if invisible:
        print(
            f"[C43-PROBE] VERDICT: FAIL — {invisible} fire(s) produced NO "
            f">= {MIN_VISIBLE_DISPLACEMENT_PX:.0f}px sprite displacement at "
            f"1fps sampling (B3 cannot see the motion)"
        )
        return 1
    print("[C43-PROBE] VERDICT: PASS — every fire is visible at 1fps sampling")
    return 0


if __name__ == "__main__":
    sys.exit(main())
