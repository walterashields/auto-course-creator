#!/usr/bin/env python3
"""
compiler/c37_replan_proof.py

C37 STEP 1 / C38 STEP 2 / C39 STEP 4 — dry replan + park-cap proof +
simulated-executor replay (NO recording, NO VLM, NO TTS API calls).

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
      seconds and the max contiguous stationary park (tail included);
  (e) C39: a simulated-executor replay of every beat against the MEASURED
      action durations from the C38 recording log (beat_002 = 17.61s, etc.)
      with the C39 budget guard and runtime park watchdog ACTIVE on a virtual
      clock: per beat it prints the planned window, the measured actions, the
      guard compressions applied, the watchdog fires, and the max contiguous
      park over the whole beat.

Hard-fail assertions:
  - replan fired on the real manifest;
  - 100% of gestures carry a semantic target (no container-center fallbacks);
  - all consecutive intra-beat gesture distances >= 40px;
  - all seam distances >= 40px;
  - C38: max contiguous park <= 3.5s in EVERY beat, tails included;
  - C39: under measured-action conditions, max contiguous park <= 3.5s in
    EVERY beat with the guard and watchdog active;
  - C39: no sentence loses its last gesture under any guard decision;
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

from compiler.choreo_runtime import (
    PARK_CAP,
    ParkWatchdog,
    compress_plan_for_budget,
    item_seconds,
)
from compiler.cost_tracker import get_tracker, reset_tracker
from compiler.curriculum import (
    _derive_sql_history,
    _dict_to_script_beat,
    load_manifest,
)
from compiler.discovery import (
    CHOREO_GESTURE_TYPES,
    CHOREO_PAUSE_CAP,
    RECORDER_TAIL_SECONDS,
    _choreography_item_seconds,
    _schedule_choreography,
    reserved_action_seconds,
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

# C39 STEP 4: MEASURED per-beat action windows from the C38 recording log
# (output/live_video_1_1_c38_20260912_161744.log, wsda-timeline "actions"
# column). These are the inputs the simulated executor replays against.
MEASURED_C38_ACTIONS = {
    "beat_001": 0.0,
    "beat_002": 17.61,
    "beat_003": 13.15,
    "beat_004": 11.46,
    "beat_005": 7.56,
    "beat_006": 0.0,
    "beat_007": 0.0,
    "beat_008": 0.0,
    "beat_009": 0.0,
}

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


# ---------------------------------------------------------------------------
# C39 STEP 4: simulated executor (virtual clock, real guard + watchdog)
# ---------------------------------------------------------------------------


class _VirtualClock:
    """Deterministic clock for the simulated executor."""

    def __init__(self) -> None:
        self.t = 0.0

    def time(self) -> float:
        return self.t

    def sleep(self, seconds: float) -> None:
        self.t += max(0.0, float(seconds))


class _ParkTracker:
    """Contiguous-still accounting over the virtual timeline. The tracker is
    injected as the watchdog's sleeper; the sim flips ``mode`` between still
    time (sleeps) and visible cursor motion so the max contiguous park is
    measured under B3 semantics: only real cursor motion closes a park."""

    def __init__(self, clock: _VirtualClock) -> None:
        self.clock = clock
        self.still = 0.0
        self.max_park = 0.0
        self.mode = "still"

    def sleep(self, seconds: float) -> None:
        seconds = max(0.0, float(seconds))
        if self.mode == "still":
            self.still += seconds
        self.clock.t += seconds

    def motion(self, seconds: float) -> None:
        self.max_park = max(self.max_park, self.still)
        self.still = 0.0
        self.clock.t += max(0.0, float(seconds))


def _sim_watchdog_motion(tracker: _ParkTracker):
    """The simulated watchdog's minimal visible motion: 0.25s of cursor
    travel (a fast hover or drift glide), deterministic, no VLM."""

    def motion(reason: str) -> bool:
        tracker.motion(0.25)
        return True

    return motion


def _replay_beat(
    beat: Any,
    scheduled: list[dict],
    tts: float,
    measured_actions: float,
) -> dict:
    """Replay one beat's scheduled plan against its MEASURED action duration
    with the C39 guard and watchdog active, exactly mirroring the executor's
    pipeline:

      1. actions phase: the measured window, subdivided into action steps
         (type_segments segments; run_query click + settle; otherwise one
         step). Step bodies sleep through ``watchdog.checked_sleep`` (the
         action-path sleeps: paste cadence, post-execute settles,
         verification polls) so an overrun can never park past the cap;
         between steps the watchdog checks. A type/click event resets the
         park clock (directive model).
      2. interleaved choreography: one sentence chunk per segment, then the
         remaining plan, both through the C39 budget guard (compress in the
         C35 order; never wholesale-skip; a sentence's last gesture always
         runs compressed).
      3. leftover rest until the TTS window ends, through checked_sleep;
         the recorder tail after the audio end is still time.

    Returns the replay report: guard decisions, watchdog fires, max park,
    and the executed gesture count per sentence (coverage check).
    """
    clock = _VirtualClock()
    tracker = _ParkTracker(clock)
    watchdog = ParkWatchdog(
        beat.beat_id,
        plan=[dict(it) for it in scheduled],
        clock=clock.time,
        sleeper=tracker.sleep,
        motion=_sim_watchdog_motion(tracker),
        log=lambda msg: print(msg, file=sys.stderr),
    )
    guard_decisions: list[str] = []
    covered: set[int] = set()
    compressed = False

    def execute_item(item: dict) -> None:
        if item.get("type") == "pause":
            tracker.mode = "still"
            watchdog.checked_sleep(
                float(item.get("duration", 0.5)), reason="choreo-pause"
            )
        elif item.get("type") in CHOREO_GESTURE_TYPES:
            tracker.motion(item_seconds(item))
            watchdog.note_motion()
            covered.add(item.get("sentence_idx", 0))
        # The glide/move itself is visible motion; the post-gesture settle
        # inside the executor is an action-path sleep already accounted by
        # the executor-accurate item_seconds model.

    def execute_guarded(items: list[dict]) -> None:
        nonlocal compressed
        work = [dict(it) for it in items]
        i = 0
        while i < len(work):
            remaining = max(0.0, tts - clock.t)
            if remaining <= 0.05 and not compressed:
                guard_decisions.append(
                    f"guard: time budget exhausted at t={clock.t:.2f}s; "
                    f"compressing {len(work) - i} remaining item(s) in C35 order"
                )
                tail, _decisions = compress_plan_for_budget(
                    work[i:],
                    remaining,
                    covered=sorted(covered),
                    log=guard_decisions.append,
                )
                compressed = True
                work = work[:i] + tail
                if not tail:
                    break
            item = work[i]
            watchdog.last_item = item
            execute_item(item)
            i += 1

    # --- 1. actions phase ---------------------------------------------------
    action = beat.action or {}
    action_type = action.get("type")
    if action_type == "type_segments":
        n_steps = max(1, len(action.get("segments") or []))
    elif action_type == "run_query":
        n_steps = 2  # click (motion) + execute/settle (still work)
    else:
        n_steps = 1
    step_dur = measured_actions / n_steps if n_steps else measured_actions

    choreo_by_sentence: dict[int, list[dict]] = {}
    for item in scheduled:
        choreo_by_sentence.setdefault(item.get("sentence_idx", 0), []).append(item)
    choreo_consumed = {s: 0 for s in choreo_by_sentence}
    interleaved = action_type == "type_segments" and bool(scheduled)

    for step in range(n_steps):
        # A type/click event at the step start resets the park clock (C39
        # directive model) and closes the B3 park. Only steps that actually
        # begin with typing/clicking get the reset: segmented typing steps
        # and the run_query click; a settle/verification step does not.
        has_type_event = action_type == "type_segments" or (
            action_type == "run_query" and step == 0
        )
        if action_type == "run_query" and step == 0:
            tracker.motion(0.5)  # the Execute-button click itself
        if has_type_event:
            tracker.motion(0.0)
            watchdog.note_motion()
        tracker.mode = "still"
        if step_dur > 0.0:
            watchdog.checked_sleep(step_dur, reason=f"action-step-{step + 1}")
        if step < n_steps - 1:
            watchdog.check(f"between-action-steps step={step + 1}")
        if interleaved:
            segs = action.get("segments") or []
            sidx = (
                segs[step].get("sentence_idx", 0)
                if step < len(segs) and isinstance(segs[step], dict)
                else 0
            )
            queue = choreo_by_sentence.get(sidx, [])
            ptr = choreo_consumed.get(sidx, 0)
            if ptr < len(queue):
                # One gesture per segment (mirrors the executor); the final
                # segment also runs the remainder of the sentence.
                chunk = queue[ptr : ptr + 1]
                if step == n_steps - 1:
                    chunk = queue[ptr:]
                choreo_consumed[sidx] = ptr + len(chunk)
                execute_guarded(chunk)

    # --- 2. remaining plan through the guard --------------------------------
    if not interleaved:
        execute_guarded(list(scheduled))
    else:
        remainder = []
        for sidx, queue in choreo_by_sentence.items():
            ptr = choreo_consumed.get(sidx, 0)
            remainder.extend(queue[ptr:])
        if remainder:
            execute_guarded(remainder)

    # --- 3. leftover rest + recorder tail ------------------------------------
    tracker.mode = "still"
    leftover = max(0.0, tts - clock.t)
    if leftover > 0.05:
        watchdog.checked_sleep(leftover, reason="leftover-rest")
    tracker.sleep(RECORDER_TAIL_SECONDS)  # stop at audio end + recorder tail

    return {
        "guard_decisions": guard_decisions,
        "watchdog_fires": list(watchdog.fires),
        "max_park": tracker.max_park,
        "covered_sentences": set(covered),
        "planned_sentences": {
            it.get("sentence_idx", 0)
            for it in scheduled
            if it.get("type") in CHOREO_GESTURE_TYPES
        },
        "sim_seconds": clock.t,
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
        # C39: single-source reservation helper (fallback estimate here — the
        # baked manifest carries no measured_action_seconds until a run
        # persists one).
        reserved = reserved_action_seconds(beat)
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

    # --- (e) C39 simulated-executor replay under MEASURED actions ------------
    print("=" * 78)
    print(
        "(e) C39 simulated-executor replay (measured C38 actions; "
        "budget guard + park watchdog active)"
    )
    sim_header = (
        f"{'beat':<10} {'tts':>6} {'resv':>6} {'meas':>6} {'window':>7} "
        f"{'guards':>7} {'fires':>6} {'max_park':>9} {'sim_s':>7}"
    )
    print(sim_header)
    print("-" * len(sim_header))
    sim_park_failures: list[str] = []
    sim_coverage_failures: list[str] = []
    sim_fire_lines: list[str] = []
    prev_rest_e = None  # executor-fresh seam chain for the measured replay
    for beat in script_beats:
        measured = MEASURED_C38_ACTIONS.get(beat.beat_id, 0.0)
        beat.measured_action_seconds = measured if measured > 0 else None
        reserved = reserved_action_seconds(beat)
        plan = [dict(it) for it in (beat.choreography or [])]
        scheduled = _schedule_choreography(
            plan,
            tts_by_beat[beat.beat_id],
            reserved_seconds=reserved,
            resolve_point=resolver,
            prev_rest_point=prev_rest_e,
            beat_id=beat.beat_id,
        )
        rep = _replay_beat(
            beat, scheduled, tts_by_beat[beat.beat_id], measured
        )
        for it in scheduled:
            if it.get("type") in CHOREO_GESTURE_TYPES:
                name = it.get("semantic") or it.get("target", "")
                pt = resolver(name) if name else None
                if pt is not None:
                    prev_rest_e = pt
        window_e = max(0.0, tts_by_beat[beat.beat_id] - 1.0 - reserved)
        print(
            f"{beat.beat_id:<10} {tts_by_beat[beat.beat_id]:6.2f} "
            f"{reserved:6.2f} {measured:6.2f} {window_e:7.2f} "
            f"{len(rep['guard_decisions']):7d} {len(rep['watchdog_fires']):6d} "
            f"{rep['max_park']:8.2f}s {rep['sim_seconds']:7.2f}"
        )
        for fire in rep["watchdog_fires"]:
            line = (
                f"  wsda-watchdog: fired beat={fire['beat_id']} "
                f"span={fire['span']:.2f}s reason={fire['reason']}"
            )
            print(line)
            sim_fire_lines.append(line)
        for d in rep["guard_decisions"]:
            print(f"  {d}")
        if rep["max_park"] > PARK_CAP + 0.05:
            sim_park_failures.append(
                f"{beat.beat_id}: max contiguous park {rep['max_park']:.2f}s "
                f"> cap {PARK_CAP}s under measured-action conditions"
            )
        uncovered = rep["planned_sentences"] - rep["covered_sentences"]
        if uncovered:
            sim_coverage_failures.append(
                f"{beat.beat_id}: sentence(s) {sorted(uncovered)} lost their "
                f"last gesture under a guard decision"
            )

    check(
        not sim_park_failures,
        f"C39: max contiguous park <= {PARK_CAP}s in every beat under "
        f"measured-action conditions (guard + watchdog active)"
        + ("" if not sim_park_failures else f" | {sim_park_failures}"),
    )
    check(
        not sim_coverage_failures,
        "C39: no sentence loses its last gesture under any guard decision"
        + ("" if not sim_coverage_failures else f" | {sim_coverage_failures}"),
    )


if __name__ == "__main__":
    raise SystemExit(main())
