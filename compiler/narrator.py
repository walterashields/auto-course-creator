#!/usr/bin/env python3
"""
compiler/narrator.py

Script-first data model and quality helpers.

This module owns the ScriptBeat dataclass and the rule-based script quality
checks used by the lesson-first compiler. Script generation itself lives in
compiler.lesson_builder.LessonBuilder.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional


# ---------------------------------------------------------------------------
# Script-first data model
# ---------------------------------------------------------------------------


@dataclass
class ScriptBeat:
    """
    One line of narration in the lesson script, optionally paired with a
    concrete UI action for demo beats.
    """

    beat_id: str
    kind: Literal[
        "opening", "concept", "demo", "validation", "close", "recap", "preview", "state"
    ]
    text: str
    # For "demo" beats: a recipe-compatible action specification.
    action: Optional[Dict[str, Any]] = None
    # For state beats: what the viewer should see at this point.
    visual_check: Optional[str] = None
    # Filled during graph construction.
    attaches_to: Optional[Literal["state", "edge"]] = None
    target_id: Optional[str] = None
    # Filled during discovery: path to the recorded video clip for this beat.
    video_clip_path: Optional[str] = None
    # Filled during execution-time sight: structured observed UI state summary.
    observed_state: Optional[Dict[str, Any]] = None
    # Planned action duration used to budget narration length.
    planned_duration: Optional[float] = None
    # Narration-paced on-screen choreography: a list of small action specs that
    # fill the beat's spoken duration with deliberate cursor motion.
    choreography: Optional[List[Dict[str, Any]]] = None
    # True when this beat should be merged into the previous beat by the renderer.
    merge: bool = False


# ---------------------------------------------------------------------------
# Script quality helpers
# ---------------------------------------------------------------------------


# Per-beat minimum narration words (excluding markdown syntax).
MIN_WORDS = {
    "opening": 25,
    "state": 25,
    "explain": 25,
    "concept": 25,
    "demo": 15,
    "validation": 15,
    "close": 30,
    "recap": 25,
    "preview": 25,
}

# Final-sentence words that indicate truncation, not completion.
FUNCTION_WORDS = {
    "the", "a", "an", "our", "we", "and", "or", "to", "of", "with", "for", "in", "on",
}

_GENERIC_FILLER = {
    "basically", "essentially", "fundamentally",
    "it is important to note", "the fact that", "in order to",
    "as you can see", "as we can see",
    "the interface updates to show the change",
}

_ACTION_WORDS = {
    "click", "type", "press", "enter", "select", "choose", "hit", "tap",
    "double-click", "right-click", "drag", "scroll", "typed", "pressing",
    "confirm",
}

_STATE_SETUP_PHRASES = {
    "hasn't been applied", "has not been applied", "not applied yet",
    "narrowed down", "narrowed-down", "has been narrowed", "have been narrowed", "to narrow",
    "narrow down", "narrows", "narrowing",
    "since no value has been", "since no value",
    "still reads", "still shows", "still lists", "still visible", "still displaying",
    "still showing", "before moving on", "before we", "after typing", "after clicking",
    "after pressing", "we have typed", "we have clicked", "we have pressed",
    "we just typed", "we have entered", "now that we", "so we can", "so we",
    "we want", "we care about", "we need", "opposite of what we want",
    "want to match", "where we'll", "we'll narrow", "giving us", "leaving us",
    "results in", "resulting in", "this will", "which will", "will show",
    "will filter", "will sort", "will narrow", "will produce", "the filter will",
}

_RESULT_PHRASES = {
    "shows", "showing", "display", "displaying", "results in", "resulting in",
    "now we see", "we see that", "this will", "which will", "will show",
    "will filter", "will sort", "will narrow", "we just typed", "after clicking",
    "now that we", "we have entered", "we just", "you just", "we have",
    "the table now", "the data now", "so we can", "so we", "to narrow",
    "to show", "giving us", "leaving us", "now displays", "now shows",
    "narrow down", "narrows", "narrowing", "can't narrow", "cannot narrow",
    "we'll narrow", "where we'll", "want to match", "to match", "we want",
    "hasn't been applied", "has not been applied", "not applied yet",
    "narrowed down", "has been narrowed", "have been narrowed",
    "still reads", "still shows", "still lists", "still visible", "still displaying",
    "before moving on", "before we", "after typing", "after pressing",
    "we have typed", "we have clicked", "we have pressed",
}


def _format_action_sql(action: Dict[str, Any]) -> Dict[str, Any]:
    """Format SQL inside an execute_query action using the course SQL standard."""
    from .sql_formatter import format_sql_query

    if action.get("type") != "execute_query":
        return action
    query = action.get("query", "")
    if not query:
        return action
    formatted = format_sql_query(query)
    action = dict(action)
    action["query"] = formatted
    return action


def _contains_action_word(text: str) -> bool:
    lowered = text.lower()
    words = set(re.findall(r"[a-z]+(?:-[a-z]+)?", lowered))
    return bool(words & _ACTION_WORDS)


def _contains_result_phrase(text: str) -> bool:
    lowered = text.lower()
    return any(phrase in lowered for phrase in _RESULT_PHRASES)


def _contains_state_setup_phrase(text: str) -> bool:
    lowered = text.lower()
    return any(phrase in lowered for phrase in _STATE_SETUP_PHRASES)


def _contains_filler(text: str) -> bool:
    lowered = text.lower()
    return any(phrase in lowered for phrase in _GENERIC_FILLER)


def _extract_numbers(text: str) -> set[str]:
    return set(re.findall(r"\d+(?:\.\d+)?", text))
