#!/usr/bin/env python3
"""
compiler/c37_replan_proof.py

C37 STEP 1 / C38 STEP 2 — dry replan + park-cap proof (NO recording, NO VLM,
NO TTS API calls).

Loads the REAL baked video_1_1 manifest through the same code path run_course
uses (curriculum.load_manifest -> _dict_to_script_beat -> _validate_script_beats
-> replan_choreography hook), then schedules every beat against its real
TTS-derived window and prints:

  (a) whether replan_choreography() fired (the wsda-replan marker);
  (b) for every beat: each gesture's sentence index, semantic target, resolved
      point, and pixel distance to the previous gesture's point;
  (c) for every beat seam: previous beat's final rest point vs next beat's
      opener point, and the distance;
  (d) C38: per beat, a simulated timeline (gesture moves at planned speeds +
      pauses against the beat's TTS window): total motion / pause / glide
      seconds and the max contiguous stationary park (tail included).

Hard-fail assertions:
  - replan fired on the real manifest;
  - 100% of gestures carry a semantic target (no container-center fallbacks);
  - all consecutive intra-beat gesture distances >= 40px;
  - all seam distances >= 40px;
  - C38: max contiguous park <= 3.5s in EVERY beat, tails included;
  - zero VLM/Anthropic calls during the entire check (tracker + API spy).

Exits 0 only when every assertion passes.
"""

from __future__ import annotations

import contextlib
import hashlib
import io
import os
import re
import subprocess
import sys

from compiler.cost_tracker import get_tracker, reset_tracker
from compiler.curriculum import (
    _derive_sql_history,
    _dict_to_script_beat,
    load_manifest,
)
from compiler.discovery import (
    CHOREO_GESTURE_TYPES,
    CHOREO_PAUSE_CAP,
    _choreography_item_seconds,
    _schedule_choreography,
)
from compiler.lesson_builder import LessonBuilder
from compiler.target_resolver import (
    MIN_GESTURE_SEPARATION_PX,
    distance,
    nominal_geometry,
    resolve_semantic_target,
)

COURSE_ID = "sql_essential_training_ch4"
VIDEO_ID = "video_1_1"
TTS_CACHE_DIR = os.path.join(os.path.dirname(__file__), "tts_cache")
WORDS_PER_SECOND_FALLBACK = 3.0

_failures: list[str] = []


def check(condition: bool, label: str) -> None:
    status = "OK  " if condition else "FAIL"
    print(f"[ASSERT] {status} {label}")
    if not condition:
        _failures.append(label)


def _tts_seconds(beat_text: str) -> float:
    """Duration of the beat's cached TTS clip (afinfo probe; no network).

    Falls back to a words-per-second estimate when the cache has no entry.
    """
    voice = os.environ.get("ELEVENLABS_VOICE_ID", "")
    model = os.environ.get("ELEVENLABS_MODEL_ID", "eleven_turbo_v2_5")
    key = hashlib.sha256(f"{beat_text}|{voice}|{model}".encode("utf-8")).hexdigest()
    path = os.path.join(TTS_CACHE_DIR, f"{key}.mp3")
    if os.path.exists(path):
        try:
            out = subprocess.run(
                ["afinfo", path], capture_output=True, text=True, timeout=15
            ).stdout
            m = re.search(r"estimated duration:\s*([0-9.]+)\s*sec", out)
            if m:
                return float(m.group(1))
        except Exception:
            pass
    words = max(1, len(beat_text.split()))
    return words / WORDS_PER_SECOND_FALLBACK


def _reserved_seconds(beat: Any) -> float:
    """Mirror of the reserved-action-time math in discovery.execute_script."""
    action = beat.action or {}
    is_demo_action = beat.kind == "demo" and action.get("type") != "wait"
    if not is_demo_action:
        return 0.0
    action_type = action.get("type")
    if action_type == "type_segments":
        return 2.0 + len(action.get("segments") or []) * 2.0
    if action_type in ("type_block", "append_block", "run_query"):
        return 3.0
    return 2.0


def _move_seconds(item: dict) -> float:
    """Wall duration of one scheduled item — the executor-accurate model."""
    return _choreography_item_seconds(item)


def _simulate(plan: list[dict], window: float) -> dict:
    """Walk a scheduled plan against its window; report where the cursor is
    stationary. A park is any contiguous stationary stretch: a pause, the
    lead before the first gesture, and the tail after the plan ends."""
    motion = 0.0
    pause = 0.0
    glide = 0.0
    parks: list[float] = []
    for it in plan:
        if it.get("type") == "pause":
            d = float(it.get("duration", 0.5))
            pause += d
            parks.append(d)
        elif it.get("type") in CHOREO_GESTURE_TYPES:
            d = _move_seconds(it)
            motion += d
            if float(it.get("speed", 1.0) or 1.0) < 1.0:
                glide += d
    cost = motion + pause
    tail = max(0.0, window - cost)
    if tail > 0.0:
        parks.append(tail)
    return {
        "motion": motion,
        "pause": pause,
        "glide": glide,
        "cost": cost,
        "window": window,
        "tail": tail,
        "max_park": max(parks) if parks else 0.0,
    }


def main() -> int:
    # --- zero-API-call guards ---------------------------------------------
    reset_tracker()
    api_spy: list[str] = []
    try:
        from anthropic.resources.messages import Messages

        original_create = Messages.create

        def spy_create(self, *args, **kwargs):  # noqa: ANN001, ANN002, ANN003
            api_spy.append(str(kwargs.get("model", "unknown")))
            raise AssertionError("Anthropic API call during dry replan proof")

        Messages.create = spy_create
    except Exception as exc:  # pragma: no cover - anthropic layout drift
        print(f"Warning: could not install Messages.create spy: {exc}", file=sys.stderr)
        original_create = None

    captured = io.StringIO()
    try:
        with contextlib.redirect_stderr(captured):
            _run_proof()
    finally:
        if original_create is not None:
            Messages.create = original_create  # type: ignore[name-defined]

    check(
        "wsda-replan" in captured.getvalue(),
        "wsda-replan marker printed during replan",
    )
    seam_markers = captured.getvalue().count("wsda-seam:")
    check(seam_markers > 0, f"wsda-seam: markers printed ({seam_markers})")
    check(get_tracker().calls == 0, f"cost tracker calls == 0 (got {get_tracker().calls})")
    check(not api_spy, f"Anthropic Messages.create spy saw 0 calls (got {len(api_spy)})")

    if _failures:
        print(f"\nDRY REPLAN + PARK-CAP PROOF: FAIL ({len(_failures)} assertion(s))")
        for f in _failures:
            print(f"  - {f}")
        return 1
    print("\nDRY REPLAN + PARK-CAP PROOF: PASS (all assertions)")
    return 0


def _run_proof() -> None:
    # --- (0) load the REAL baked manifest exactly like run_course/main() ----
    manifest = load_manifest(COURSE_ID)
    assert manifest is not None, f"baked manifest {COURSE_ID!r} not found"
    video = next((v for v in manifest.videos if v.video_id == VIDEO_ID), None)
    assert video is not None, f"{VIDEO_ID!r} not found in {COURSE_ID!r}"

    lesson_builder = LessonBuilder()

    # Same path as curriculum.py run_course (Phase 2, baked-script branch).
    script_beats = [_dict_to_script_beat(b) for b in video.script_beats]
    script_beats = lesson_builder._validate_script_beats(script_beats, video)
    history, _new_query = _derive_sql_history(manifest, video)
    exercise = video.exercise_artifact or {}
    table = exercise.get("table_name", "Customer")
    facts = lesson_builder._db_facts(exercise.get("db_path"), table)
    replanned = lesson_builder.replan_choreography(
        script_beats,
        facts.get("columns", []),
        table,
        opening_history=history or "",
    )

    # --- (a) replan marker --------------------------------------------------
    print("=" * 78)
    print("(a) replan_choreography() fired?")
    print(f"  return value : {replanned}")
    print(f"  beats planned: {len(script_beats)}")
    check(bool(replanned), "replan_choreography() returned True on the real manifest")

    # --- geometry (plan-time nominal, same math the planner used) -----------
    try:
        import pyautogui

        size = pyautogui.size()
        screen = (float(size.width), float(size.height))
    except Exception:
        screen = (1440.0, 900.0)
    geo = nominal_geometry(screen)
    print(f"  screen={screen} line_height={geo.line_height}px park_cap={CHOREO_PAUSE_CAP}s")

    def resolver(name: str):
        return resolve_semantic_target(name, geo)

    # --- per-beat windows ----------------------------------------------------
    windows: dict[str, float] = {}
    tts_by_beat: dict[str, float] = {}
    reserved_by_beat: dict[str, float] = {}
    for beat in script_beats:
        tts = _tts_seconds(beat.text)
        reserved = _reserved_seconds(beat)
        tts_by_beat[beat.beat_id] = tts
        reserved_by_beat[beat.beat_id] = reserved
        windows[beat.beat_id] = max(0.0, tts - 1.0 - reserved)

    # --- (b) per-beat gesture table (real TTS windows) -----------------------
    print("=" * 78)
    print("(b) per-beat gestures (scheduled against the real TTS window)")
    header = (
        f"{'beat':<10} {'sidx':>4} {'type':<6} {'speed':>6} {'semantic':<38} "
        f"{'point':<16} {'dist_to_prev':>12}"
    )
    print(header)
    print("-" * len(header))

    gestures_total = 0
    gestures_with_semantic = 0
    intra_dist_failures: list[str] = []

    beat_final_rest: dict[str, tuple] = {}
    prev_rest_point = None  # like agent._last_rest_point at run start

    seam_rows: list[tuple] = []
    scheduled_plans: dict[str, list] = {}
    sims: dict[str, dict] = {}

    for beat in script_beats:
        plan = [dict(it) for it in (beat.choreography or [])]
        for it in plan:
            if it.get("type") in CHOREO_GESTURE_TYPES:
                gestures_total += 1
                if it.get("semantic"):
                    gestures_with_semantic += 1
        scheduled = _schedule_choreography(
            plan,
            tts_by_beat[beat.beat_id],
            reserved_seconds=reserved_by_beat[beat.beat_id],
            resolve_point=resolver,
            prev_rest_point=prev_rest_point,
            beat_id=beat.beat_id,
        )
        scheduled_plans[beat.beat_id] = scheduled
        sims[beat.beat_id] = _simulate(scheduled, windows[beat.beat_id])

        prev_gesture_point = None
        for it in scheduled:
            if it.get("type") not in CHOREO_GESTURE_TYPES:
                continue
            semantic = it.get("semantic") or ""
            point = resolver(semantic) if semantic else None
            dist = ""
            if point is not None and prev_gesture_point is not None:
                d = distance(point, prev_gesture_point)
                dist = f"{d:8.1f}px"
                if d < MIN_GESTURE_SEPARATION_PX:
                    intra_dist_failures.append(
                        f"{beat.beat_id} sidx {it.get('sentence_idx', 0)} "
                        f"{semantic} -> {point} is {d:.1f}px from previous gesture"
                    )
            speed = float(it.get("speed", 1.0) or 1.0)
            speed_str = f"{speed:.2f}" if speed != 1.0 else ""
            print(
                f"{beat.beat_id:<10} {it.get('sentence_idx', 0):>4} "
                f"{it.get('type', ''):<6} {speed_str:>6} {semantic:<38} "
                f"{str(point):<16} {dist:>12}"
            )
            if point is not None:
                prev_gesture_point = point
        if prev_gesture_point is not None:
            beat_final_rest[beat.beat_id] = prev_gesture_point

        # --- (c) seam row ------------------------------------------------
        if prev_rest_point is not None and prev_gesture_point is not None:
            opener = next(
                (it for it in scheduled if it.get("type") in CHOREO_GESTURE_TYPES),
                None,
            )
            if opener is not None:
                opener_semantic = opener.get("semantic") or ""
                opener_point = (
                    resolver(opener_semantic) if opener_semantic else None
                )
                seam_rows.append(
                    (
                        beat.beat_id,
                        prev_rest_point,
                        opener_semantic,
                        opener_point,
                        distance(opener_point, prev_rest_point)
                        if opener_point is not None
                        else None,
                    )
                )
        prev_rest_point = prev_gesture_point

    print("-" * len(header))
    print(
        f"gestures (raw replanned plans): {gestures_with_semantic}/{gestures_total} "
        f"carry a semantic target"
    )

    # --- (c) seam table ------------------------------------------------------
    print("=" * 78)
    print("(c) beat seams (prev beat final rest -> this beat opener)")
    seam_header = (
        f"{'beat':<10} {'prev_rest':<16} {'opener semantic':<38} "
        f"{'opener point':<16} {'distance':>12}"
    )
    print(seam_header)
    print("-" * len(seam_header))
    seam_failures: list[str] = []
    for beat_id, rest, opener_semantic, opener_point, dist in seam_rows:
        dist_str = f"{dist:8.1f}px" if dist is not None else "n/a"
        print(
            f"{beat_id:<10} {str(rest):<16} {opener_semantic:<38} "
            f"{str(opener_point):<16} {dist_str:>12}"
        )
        if dist is None or dist < MIN_GESTURE_SEPARATION_PX:
            seam_failures.append(
                f"seam into {beat_id}: {opener_semantic} at {opener_point} is "
                f"{dist if dist is not None else 'unresolvable'} from rest {rest}"
            )

    # --- (d) C38 temporal simulation -----------------------------------------
    print("=" * 78)
    print(
        f"(d) simulated timelines (window = tts - 1.0s tail - reserved; "
        f"park cap {CHOREO_PAUSE_CAP}s)"
    )
    sim_header = (
        f"{'beat':<10} {'tts':>6} {'resv':>5} {'window':>7} {'motion':>7} "
        f"{'pause':>7} {'glide':>7} {'plan':>7} {'tail':>6} {'max_park':>9}"
    )
    print(sim_header)
    print("-" * len(sim_header))
    park_failures: list[str] = []
    for beat in script_beats:
        s = sims[beat.beat_id]
        print(
            f"{beat.beat_id:<10} {tts_by_beat[beat.beat_id]:6.2f} "
            f"{reserved_by_beat[beat.beat_id]:5.2f} {s['window']:7.2f} "
            f"{s['motion']:7.2f} {s['pause']:7.2f} {s['glide']:7.2f} "
            f"{s['cost']:7.2f} {s['tail']:6.2f} {s['max_park']:8.2f}s"
        )
        if s["max_park"] > CHOREO_PAUSE_CAP + 0.05:
            park_failures.append(
                f"{beat.beat_id}: max contiguous park {s['max_park']:.2f}s "
                f"> cap {CHOREO_PAUSE_CAP}s"
            )

    # --- hard assertions -----------------------------------------------------
    print("=" * 78)
    check(
        gestures_total > 0 and gestures_with_semantic == gestures_total,
        f"100% gestures carry a semantic target "
        f"({gestures_with_semantic}/{gestures_total})",
    )
    check(
        not intra_dist_failures,
        f"all consecutive intra-beat gesture distances >= "
        f"{MIN_GESTURE_SEPARATION_PX:.0f}px"
        + ("" if not intra_dist_failures else f" | {intra_dist_failures}"),
    )
    check(
        not seam_failures,
        f"all seam distances >= {MIN_GESTURE_SEPARATION_PX:.0f}px"
        + ("" if not seam_failures else f" | {seam_failures}"),
    )
    check(
        not park_failures,
        f"max contiguous park <= {CHOREO_PAUSE_CAP}s in every beat (tails included)"
        + ("" if not park_failures else f" | {park_failures}"),
    )


if __name__ == "__main__":
    raise SystemExit(main())
