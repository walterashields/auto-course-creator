"""compiler/choreo_runtime.py

C39 runtime choreography invariants — the single source of truth shared by
the planner (discovery), the executor (vision_agent), and the dry proof
(c37_replan_proof).

Leaf module: it imports nothing from the compiler package, so both
discovery and vision_agent can depend on it without an import cycle (the
CHOREO_PAUSE_CAP mirror in vision_agent existed precisely because
vision_agent could not import discovery).

Contents:
  - PARK_CAP / CHOREO_MAX_SPEED / the executor-accurate gesture wall-clock
    model (C38) used by both planner and executor;
  - max_contiguous_park(): the C38 still-block semantics over a plan;
  - ParkWatchdog: the C39 STEP 1 runtime park watchdog (deterministic, no
    VLM) — tracks last-motion time, truncates sleeps so no contiguous park
    exceeds the cap, and inserts a minimal visible motion when it fires;
  - compress_plan_for_budget(): the C39 STEP 2 executor budget guard — the
    C35 compression order (zero pauses, speed to the 2x cap, drop only
    non-sentence-last gestures, never drop a sentence's last/only gesture)
    replacing the pre-C39 wholesale "skipping redundant item(s)" break.
"""

from __future__ import annotations

import sys
import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple

# C38/C39 park cap: no contiguous stationary stretch may exceed 3.5s
# anywhere in a beat (the B3 detector samples at 1fps, so a 3.5s park
# measures ~4-5s, safely under the 6.0s anti-stall gate).
PARK_CAP = 3.5
# C35 compression order (b): gesture travel may be sped up to 2x and no
# further, so compressed motion stays deliberate.
CHOREO_MAX_SPEED = 2.0
# C38 executor-accurate per-gesture wall-clock model at speed 1.0: move base
# + settle time, mirroring VisionAgent.execute_choreography_item
# (_item_move_duration move plus post-gesture settle). Speeds below 1.0 are
# deliberate slow glides and INCREASE the duration.
GESTURE_COST = {"hover": 0.75, "click": 1.15, "scroll": 0.5, "drag": 1.2}
GESTURE_SETTLE = {"hover": 0.05, "click": 0.45, "scroll": 0.2, "drag": 0.4}
GESTURE_TYPES = ("hover", "click", "scroll", "drag")


def item_seconds(item: Dict[str, Any]) -> float:
    """Executor-accurate wall seconds of one scheduled plan item.

    Move time follows the executor (max(0.25, move_base / min(2, speed)));
    settle time (post-hover calm, post-click release, scroll/drag completion)
    is fixed. Speeds below 1.0 (C38 slow glides) stretch the move.
    """
    t = item.get("type")
    if t == "pause":
        return float(item.get("duration", 0.5))
    speed = float(item.get("speed", 1.0) or 1.0)
    if speed <= 0.0:
        speed = 1.0
    settle = GESTURE_SETTLE.get(t, 0.0)
    move_base = max(0.05, GESTURE_COST.get(t, 0.5) - settle)
    return max(0.25, move_base / min(speed, CHOREO_MAX_SPEED)) + settle


def plan_cost(plan: Sequence[Optional[Dict[str, Any]]]) -> float:
    """Estimated wall-clock cost of a scheduled choreography plan."""
    return sum(item_seconds(it) for it in plan if it is not None)


def plan_target(it: Dict[str, Any]) -> str:
    """C36: the identity of a plan item is its semantic target when present,
    else the legacy human description."""
    return it.get("semantic") or it.get("target", "")


def max_contiguous_park(
    plan: Sequence[Optional[Dict[str, Any]]], tail: float = 0.0
) -> float:
    """Longest contiguous stationary stretch of a plan under the C38
    still-block semantics: a block holds REST time only. A pause adds to the
    block; a gesture closes the block when it is a slow glide (speed < 1.0)
    or travels to a different target — same-target full-speed gestures are
    visually instant and continue the block. ``tail`` appends trailing still
    time (e.g. the rest after the plan ends) to the final block."""
    best = 0.0
    current = 0.0
    last_target: Optional[str] = None
    for it in plan:
        if it is None:
            continue
        if it.get("type") == "pause":
            current += float(it.get("duration", 0.5))
            continue
        if it.get("type") in GESTURE_TYPES:
            target = plan_target(it)
            speed = float(it.get("speed", 1.0) or 1.0)
            if speed < 1.0 or (target and target != last_target):
                best = max(best, current)
                current = 0.0
                last_target = target
    if tail > 0.0:
        current += tail
    return max(best, current)


class ParkWatchdog:
    """C39 STEP 1 runtime park watchdog (deterministic, no VLM, universal).

    Tracks ``last_motion_time`` — any cursor move, click, type, or on-screen
    state change resets it (the executor calls :meth:`note_motion`). Before
    executing any pause or sleep (:meth:`checked_sleep`) and between action
    steps (:meth:`check`), the watchdog verifies that the contiguous park
    stays within ``park_cap``; when a sleep or an overrun would breach the
    cap it truncates the sleep at the cap and inserts a minimal visible
    motion instead (the injected ``motion`` callable — a fast hover to the
    current sentence's resolved target, or a short drift glide to the beat's
    most-referenced alternative and back).

    The watchdog enforces the invariant REGARDLESS of cause: action overrun,
    skipped gestures, action-path sleeps (post-execute settles, verification
    polls), or planner error.

    One ``wsda-watchdog: armed beat=<id>`` line is printed at arm time (it
    proves the watchdog is active for the beat) and one
    ``wsda-watchdog: fired beat=<id> span=<s> reason=<...>`` line each time
    it acts. ``clock``/``sleeper``/``motion`` are injectable so the dry proof
    and the tests can replay beats on a virtual timeline.
    """

    def __init__(
        self,
        beat_id: str,
        plan: Optional[Sequence[Dict[str, Any]]] = None,
        park_cap: float = PARK_CAP,
        clock: Optional[Callable[[], float]] = None,
        sleeper: Optional[Callable[[float], None]] = None,
        motion: Optional[Callable[[str], bool]] = None,
        log: Optional[Callable[[str], None]] = None,
    ) -> None:
        self.beat_id = beat_id
        self.plan: List[Dict[str, Any]] = [dict(it) for it in (plan or [])]
        self.park_cap = float(park_cap)
        self._clock = clock or time.time
        self._sleep = sleeper or time.sleep
        self._motion = motion
        self._log = log or (lambda msg: print(msg, file=sys.stderr))
        # Item context for motion resolution: the executor updates
        # ``last_item`` as it works through the plan so a fire can hover to
        # the CURRENT sentence's resolved target first.
        self.last_item: Optional[Dict[str, Any]] = None
        self.last_motion_time = float(self._clock())
        self.fires: List[Dict[str, Any]] = []
        self._log(f"wsda-watchdog: armed beat={self.beat_id}")

    # -- motion tracking --------------------------------------------------
    def note_motion(self) -> None:
        """Reset the park clock: a cursor move, click, type, or on-screen
        state change happened."""
        self.last_motion_time = float(self._clock())

    def parked(self) -> float:
        """Seconds since the last registered motion."""
        return max(0.0, float(self._clock()) - self.last_motion_time)

    # -- enforcement ------------------------------------------------------
    def check(self, reason: str) -> bool:
        """Between action steps: if the cursor has been still past the cap,
        insert the minimal visible motion. Returns True when it fired."""
        span = self.parked()
        if span <= self.park_cap:
            return False
        self._fire(reason, span)
        return True

    def checked_sleep(self, seconds: float, reason: str) -> float:
        """Sleep ``seconds`` without ever letting the contiguous park exceed
        the cap. The sleep is processed in cap-sized chunks: whenever the
        accumulated park would reach the cap, the watchdog inserts its
        visible motion (resetting the park clock) and the remainder of the
        sleep continues on the fresh clock. A sleep longer than the cap
        therefore fires repeatedly, never parking past the cap."""
        seconds = max(0.0, float(seconds))
        remaining_sleep = seconds
        while remaining_sleep > 0.0:
            span = self.parked()
            if span + remaining_sleep <= self.park_cap:
                self._sleep(remaining_sleep)
                return seconds
            allowed = max(0.0, self.park_cap - span)
            if allowed > 0.0:
                self._sleep(allowed)
                remaining_sleep -= allowed
            # span is now exactly the cap (or already past it): break the park.
            self._fire(reason, self.parked())
        return seconds

    # -- internals --------------------------------------------------------
    def _fire(self, reason: str, span: float) -> None:
        self.fires.append(
            {"beat_id": self.beat_id, "span": round(span, 3), "reason": reason}
        )
        if self._motion is not None:
            try:
                self._motion(reason)
            except Exception as exc:  # motion must never kill the beat
                print(
                    f"wsda-watchdog: motion failed beat={self.beat_id}: {exc}",
                    file=sys.stderr,
                )
        self.note_motion()
        self._log(
            f"wsda-watchdog: fired beat={self.beat_id} "
            f"span={span:.2f}s reason={reason}"
        )

    # -- motion target resolution (executor-facing) ------------------------
    def resolve_motion_target(self) -> Tuple[Optional[str], str]:
        """Pick the target for the minimal visible motion, in order:
        (1) the current sentence's resolved target from the plan;
        (2) the beat's most-referenced alternative target;
        (3) None — the caller drifts relative to the current cursor point.
        Returns (target, source)."""
        if self.last_item is not None:
            sidx = self.last_item.get("sentence_idx", 0)
            for it in self.plan:
                if (
                    it.get("sentence_idx", 0) == sidx
                    and it.get("type") in GESTURE_TYPES
                    and plan_target(it)
                ):
                    return plan_target(it), "sentence"
        counts: Dict[str, int] = {}
        for it in self.plan:
            if it.get("type") in GESTURE_TYPES:
                t = plan_target(it)
                if t:
                    counts[t] = counts.get(t, 0) + 1
        if counts:
            best = max(sorted(counts), key=lambda t: (counts[t], t))
            return best, "alternative"
        return None, "none"


def compress_plan_for_budget(
    items: Sequence[Dict[str, Any]],
    budget: float,
    covered: Sequence[int] = (),
    park_cap: float = PARK_CAP,
    log: Optional[Callable[[str], None]] = None,
) -> Tuple[List[Dict[str, Any]], List[str]]:
    """C39 STEP 2: compress a plan tail that no longer fits the remaining
    time budget, in the C35 compression order — never a wholesale skip.

    Order:
      (a) drop pauses to zero;
      (b) compress gesture travel up to the 2x speed cap;
      (c) drop only gestures that are NOT any sentence's last gesture
          (a sentence already gestured earlier in the beat may lose all of
          its remaining gestures);
      (d) NEVER drop a sentence's last/only gesture — if the budget is
          exhausted the survivors run compressed, slightly past the soft
          budget.

    After every guard decision the park cap is re-checked: if a decision
    would create a contiguous park above ``park_cap`` (merging still time
    across a dropped gesture), the drop is reverted — the kept gesture is
    the cheapest motion that breaks the park.

    ``covered`` holds the sentence indices already gestured earlier in the
    beat; their remaining gestures are all droppable. Returns
    ``(plan, decisions)`` where ``decisions`` is the human-readable guard
    log. The string "redundant" is deliberately never applied to a
    sentence-covered gesture here.
    """
    decisions: List[str] = []

    def say(msg: str) -> None:
        decisions.append(msg)
        if log is not None:
            log(msg)

    plan: List[Optional[Dict[str, Any]]] = [dict(it) for it in items]
    covered_set: Set[int] = set(covered)

    def live_cost(pl: Sequence[Optional[Dict[str, Any]]]) -> float:
        return sum(item_seconds(it) for it in pl if it is not None)

    def live_plan() -> List[Dict[str, Any]]:
        return [it for it in plan if it is not None]

    def park_recheck() -> float:
        span = max_contiguous_park(live_plan())
        say(f"guard(park-recheck): max contiguous park {span:.2f}s <= {park_cap:.1f}s")
        return span

    needed = live_cost(plan) - max(0.0, float(budget))
    if needed <= 0.0:
        return live_plan(), decisions

    # (a) drop pauses to zero.
    pause_time = 0.0
    for it in plan:
        if it is not None and it.get("type") == "pause":
            pause_time += float(it.get("duration", 0.5))
            it["duration"] = 0.0
    if pause_time > 0.0:
        say(f"guard(a): pauses zeroed (freed {pause_time:.2f}s)")
    park_recheck()
    needed = live_cost(plan) - max(0.0, float(budget))

    # (b) compress gesture travel up to the 2x speed cap.
    if needed > 0.0:
        g_idx = [
            i
            for i, it in enumerate(plan)
            if it is not None and it.get("type") in GESTURE_TYPES
        ]
        if g_idx:
            g_total = sum(item_seconds(plan[i]) for i in g_idx)
            g_budget = max(0.0, g_total - needed)
            speed = (
                min(CHOREO_MAX_SPEED, g_total / g_budget)
                if g_budget > 0.05
                else CHOREO_MAX_SPEED
            )
            for i in g_idx:
                current = float(plan[i].get("speed", 1.0) or 1.0)
                plan[i]["speed"] = max(current, speed)
            say(f"guard(b): gesture travel at {speed:.2f}x (cap {CHOREO_MAX_SPEED:.1f}x)")
        park_recheck()
        needed = live_cost(plan) - max(0.0, float(budget))

    # (c) drop only gestures that are NOT any sentence's last gesture.
    if needed > 0.0:
        protected: Set[int] = set()
        while needed > 0.0:
            last_g: Dict[int, int] = {}
            for i, it in enumerate(plan):
                if it is not None and it.get("type") in GESTURE_TYPES:
                    last_g[it.get("sentence_idx", 0)] = i
            candidates: List[int] = []
            for i, it in enumerate(plan):
                if it is None or it.get("type") not in GESTURE_TYPES:
                    continue
                if i in protected:
                    continue
                s = it.get("sentence_idx", 0)
                # Droppable when the sentence keeps another gesture: either
                # one already executed earlier in the beat, or a later live
                # gesture in this tail.
                if s in covered_set or i != last_g.get(s):
                    candidates.append(i)
            if not candidates:
                break
            # Drop the latest gesture of the sentence with the most
            # droppable gestures (mirrors the plan-time compression order).
            per_sentence: Dict[int, List[int]] = {}
            for i in candidates:
                per_sentence.setdefault(plan[i].get("sentence_idx", 0), []).append(i)
            s = max(per_sentence, key=lambda k: (len(per_sentence[k]), k))
            drop = per_sentence[s][-1]
            dropped = plan[drop]
            saved = item_seconds(dropped)
            plan[drop] = None
            if max_contiguous_park(live_plan()) > park_cap:
                # The drop would merge still time into a park over the cap:
                # keep the gesture — it is the cheapest motion that breaks it.
                plan[drop] = dropped
                protected.add(drop)
                say(
                    f"guard(c): keeping {plan_target(dropped)[:50]!r} "
                    f"(drop would park past {park_cap:.1f}s)"
                )
                # Every remaining candidate may be tried, but if all are
                # protected there is nothing left to drop.
                if len(protected) >= len(candidates):
                    break
                continue
            needed -= saved
            say(
                f"guard(c): dropped non-last gesture {plan_target(dropped)[:50]!r} "
                f"(sentence {s} keeps a gesture; freed {saved:.2f}s)"
            )
            park_recheck()
        needed = live_cost(plan) - max(0.0, float(budget))

    # (d) never drop a sentence's last/only gesture: run compressed.
    if needed > 0.0:
        remaining_sentences = {
            it.get("sentence_idx", 0)
            for it in live_plan()
            if it.get("type") in GESTURE_TYPES
        }
        say(
            f"guard(d): budget exhausted; running {len(live_plan())} item(s) "
            f"compressed ({needed:.2f}s past soft budget); "
            f"{len(remaining_sentences)} sentence(s) keep their last gesture"
        )

    return (
        [
            it
            for it in live_plan()
            if not (it.get("type") == "pause" and float(it.get("duration", 0.5)) <= 0.0)
        ],
        decisions,
    )
