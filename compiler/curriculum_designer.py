#!/usr/bin/env python3
"""
compiler/curriculum_designer.py

AI Curriculum Designer: takes a topic, audience, and depth, and produces a
CourseManifest that follows the content standard and reference curriculum
structures.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import re
import sqlite3
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import anthropic

from .curriculum import CourseManifest, VideoManifest
from .lesson_builder import LessonBuilder
from .lesson_standard import LessonStandard
from .sql_formatter import format_sql_in_text

DEFAULT_MODEL = os.environ.get("CURRICULUM_MODEL", "claude-sonnet-5")
_CACHE_VERSION = "20"  # Bump when prompt/schema changes to invalidate old caches

_DEPTH_REQUIRED_SLOTS = {
    "micro": "",
    "short": """Unit 1 — Foundation
1. orientation: Browse a table
2. demo: Sort a single column in the UI
3. demo: Filter by exact text match in the UI""",
    "mid": """Unit 1 — Foundation (UI)
1. orientation: Browse a table
2. demo: Sort a single column in the UI
3. demo: Filter by exact text match in the UI
4. concept: Why SQL beats point-and-click

Unit 2 — SQL Basics
5. demo: Run a first SELECT * query
6. demo: Filter rows with WHERE
7. demo: Count rows with COUNT

Unit 3 — Capstone
8. capstone: Top customers by total spend""",
    "long": """Unit 1 — Foundation (UI)
1. orientation: Browse a table
2. concept: Table relationships and foreign keys
3. demo: Sort a single column in the UI
4. demo: Filter by exact text match in the UI
5. demo: Filter by numeric comparison in the UI
6. concept: Why SQL beats point-and-click

Unit 2 — SQL Basics
7. demo: Run a first SELECT * query
8. demo: Filter rows with WHERE
9. demo: Sort results with ORDER BY in SQL
10. demo: Count rows with COUNT
11. demo: Group rows with GROUP BY
12. demo: Sum values with SUM

Unit 3 — Joins
13. concept: How tables connect using foreign keys in practice
14. demo: INNER JOIN two tables
15. demo: Filter joined results with WHERE

Unit 4 — Troubleshooting
16. anti-pattern: COUNT(*) vs COUNT(column)""",
    "full": """Unit 1 — Foundation (UI)
1. orientation: Browse a table
2. concept: Table relationships and foreign keys
3. demo: Sort a single column in the UI
4. demo: Filter by exact text match in the UI
5. demo: Filter by numeric comparison in the UI
6. concept: Why SQL beats point-and-click

Unit 2 — SQL Basics
7. demo: Run a first SELECT * query
8. demo: Filter rows with WHERE
9. demo: Sort results with ORDER BY in SQL
10. demo: Count rows with COUNT
11. demo: Group rows with GROUP BY
12. demo: Sum values with SUM

Unit 3 — Joins
13. concept: How tables connect using foreign keys in practice
14. demo: INNER JOIN two tables
15. demo: Filter joined results with WHERE

Unit 4 — Troubleshooting (anti-patterns)
16. anti-pattern: COUNT(*) vs COUNT(column)
17. anti-pattern: Why your GROUP BY is silently wrong
18. anti-pattern: Why your SQL SUM is wrong

Unit 5 — Capstone
19. capstone: Top customers by total spend""",
}


def _sanitize_strings(obj: Any) -> Any:
    """Recursively replace stray control characters in JSON string values."""
    if isinstance(obj, str):
        # Replace tabs/newlines/carriage returns with a single space.
        return re.sub(r"[\t\r\n]+", " ", obj).strip()
    if isinstance(obj, list):
        return [_sanitize_strings(item) for item in obj]
    if isinstance(obj, dict):
        return {k: _sanitize_strings(v) for k, v in obj.items()}
    return obj


SYSTEM_PROMPT = """You are a curriculum designer for professional data analytics video courses.
You design courses following a proven, six-part video taxonomy and strict structural rules.

{content_standard}

## Video Taxonomy (every video must have exactly one type)

- orientation: Tool introduction or UI navigation. Use sparingly (1-2 per course).
- concept: Abstract skill explanation BEFORE showing UI or SQL. Builds the "why".
- demo: Show exactly one new capability in action.
- exercise: Learner follows along with a provided artifact.
- anti-pattern: "Why your X is wrong" — show a common mistake, the surprising wrong result, the mechanism, the right approach, and proof comparing wrong vs right.
- capstone: Integrates 2-3 already-learned skills. ONLY at the end of a unit or course.

## Required Video Slots for SQL Fundamentals

{required_slots}

You may add extra concept/orientation videos, but never skip a required slot and never reorder them.

## Schema Utilization Rules

The running example must include at least:
- A Customers table and an Orders table.
- One-to-many relationship: Customers → Orders via customer_id foreign key.
- NULL values: some Orders with NULL status for filtering demos.
- Outliers: one customer with 10+ orders, others with 1-2.
- Date range: Orders spanning 12+ months.
- Multiple countries: at least 3 countries for GROUP BY demos.

## Anti-Pattern Structure (mandatory)

Every anti-pattern video must:
1. Show the WRONG query or approach first.
2. Show the SURPRISING result — concrete numbers that look right but are wrong.
3. Explain WHY it is wrong (the mechanism).
4. Show the RIGHT approach.
5. Close with proof numbers comparing wrong vs right.

## Strict One-Capability-Per-Video Rule

- Capstone videos are the ONLY exception.
- A title containing " and " or " & " is rejected unless the video_type is "capstone".
- A new_capability containing " and " or " & " is rejected unless the video_type is "capstone".
- If a video title or new_capability would naturally contain "and", the video is teaching TWO skills. Split it into two videos or pick a single focus.
- Example fixes (apply these exact patterns):
  - Bad title: "Browse the Customers and Orders Tables" → Good: "Browse the Customers Table"
  - Bad title: "Table Relationships and Foreign Keys" → Good: "How Foreign Keys Link Tables"
  - Bad title: "Sort and Filter a Table" → Good: "Sort a Single Column" (and make filtering a separate video)
- new_capability must be a single verb phrase describing ONE skill. Examples:
  - Bad: "Opening and browsing a table in the UI" → Good: "Open the Customers table in the Browse Data tab"
  - Bad: "Writing and running a SELECT * query" → Good: "Run a SELECT * query"
  - Bad: "Recognizing the difference between COUNT(*) and COUNT(column) with NULLs" → Good: "Diagnose the COUNT(*) vs COUNT(column) NULL discrepancy"

## Enhanced Recap / Preview

- Recap must reference CONCRETE numbers from the previous video, not vague concepts.
  Bad: "In the last video we learned about sorting."
  Good: "In the last video we sorted 250 orders by amount, with the smallest order — $36.24 — right at the top."
- Preview must state the EXACT next capability and must NOT use vague words such as "more", "next steps", "continue", "further", or "complex questions".
  Bad: "Next we'll do more filtering."
  Bad: "Next we'll discuss why SQL beats point-and-click filtering for more complex questions."
  Good: "Next we'll filter rows with a WHERE clause in the Execute SQL tab."

## SQL Query Formatting Standard

When any field (discovery_objective, recap, preview, etc.) includes an example SQL query, the query MUST be formatted exactly like this:

/*
Created By: Walter Shields
Create Date: mm/dd/yyyy
Description: [What the query does]
*/
SELECT
    field1,
    field2
FROM table
WHERE condition
GROUP BY field
ORDER BY field;

Rules:
- SQL keywords in ALL CAPS: SELECT, FROM, WHERE, JOIN, GROUP BY, ORDER BY, LIMIT, etc.
- One clause per line.
- Each SELECT field on its own indented line with trailing commas.
- Use AS for every alias. Multi-word aliases in double quotes.
- For joins, use table aliases: FROM orders AS o INNER JOIN customers AS c ON o.customer_id = c.customer_id.
- Never produce single-line queries. Never lowercase keywords. Never omit AS.

## General Rules

- ONE new capability per video (capstone excepted).
- STABLE running example across all videos.
- DEFERRED orientation: teach the abstract concept before the UI/SQL mechanics.
- CONCRETE proof numbers in every close.
- PROGRESSION: Browse → Sort → Filter (UI) → SELECT → WHERE → ORDER BY → COUNT → GROUP BY → SUM → JOIN → Anti-patterns → Capstone.
- Prerequisites form a DAG; first video has no prerequisites.
- Keep all string values concise (1-2 sentences each).

Design a course on: {topic}
Target audience: {target_audience}
Depth: {depth}
Application: {application}
"""

USER_PROMPT = """Return a compact structured JSON object (no markdown, no explanation) with this exact shape:

{
  "course_title": "string",
  "course_description": "string",
  "videos": [
    {
      "video_id": "string (e.g., video_1_1, video_1_2)",
      "title": "string",
      "video_type": "orientation|concept|demo|exercise|anti-pattern|capstone",
      "learning_objective": "string (1 sentence, pedagogical — the conceptual skill the learner gains)",
      "discovery_objective": "string (1 sentence, concrete screen state verifiable by a vision model looking at a screenshot)",
      "prerequisite_videos": ["video_id", ...],
      "format_tier": "micro|short|mid|long|full",
      "new_capability": "string (the ONE thing this video teaches)",
      "key_concept": "string (1 sentence — the single abstract idea the learner must understand)",
      "prerequisite_knowledge": "string (1 sentence — what the learner must already know from previous videos)",
      "running_example_usage": "string (one sentence)",
      "proof_numbers": "string (concrete numbers)",
      "estimated_word_count": int,
      "has_recap": boolean,
      "has_preview": boolean,
      "recap_text_hint": "string (1 concrete sentence; null if has_recap is false)",
      "preview_text_hint": "string (1 exact-capability sentence; null if has_preview is false)"
    }
  ],
  "running_example": {
    "name": "string",
    "description": "string (one sentence)",
    "tables": [
      {
        "name": "string",
        "columns": [
          {"name": "string", "type": "string"}
        ],
        "rows": integer
      }
    ]
  }
}

Output size constraints (STRICT — do not exceed the upper bound):
- Depth "micro" → exactly 1 video.
- Depth "short" → exactly 3 videos.
- Depth "mid" → exactly 8 videos.
- Depth "long" → exactly 15 videos.
- Depth "full" → exactly 19 videos (must cover all required SQL Fundamentals slots).

Discovery objective rules (CRITICAL):
- Mention a specific UI element/tab (Browse Data tab, Execute SQL tab, column header, filter box).
- Mention a specific table or view, OR state it is an Execute SQL tab query.
- Describe a visible state, not a concept.
- Achievable in 1-3 plain left-clicks.
- NEVER use "understand", "learn", "grasp", "comprehend".
- NEVER include exact numeric values from the data (those belong in proof_numbers).
- MUST be ATOMIC: exactly ONE end state, ONE skill, ONE direction.
  BAD: "Click the amount column header once to sort ascending, then click it again to sort descending..."
  GOOD: "Sort the Orders table by amount in ascending order"
  GOOD: "Sort the Orders table by amount in descending order"
- NEVER use sequencing words: then, after that, finally, and then, next, subsequently.
- NEVER use multiple verbs describing multiple actions: click again, first click, next click, observe the.
- Each video teaches ONE capability. One sort direction = one video. Ascending and descending are SEPARATE videos.

Interaction constraints:
- 1-3 plain left-clicks per discovery objective.
- NO shift-click, Ctrl-click, Option-click, drag, right-click menus, or multi-step typing.
- If a skill is too complex, decompose it or omit it.
- Prefer recipe-friendly phrasing so the discovery harness can use deterministic UI recipes:
  "Open the X table in the Browse Data tab"
  "Sort the X table by Y in ascending order"
  "Sort the X table by Y in descending order"

Recap/preview rules:
- First video: has_recap=false, has_preview=true.
- Last video: has_recap=true (unless only one video), has_preview=false.
- If the course contains only ONE video, set has_preview=false (there is no next video to preview).
- Middle videos: both true.
- Recap references concrete numbers from the previous video.
- Preview states the exact next capability and avoids vague words like "more", "next steps", "continue", "further", or "complex questions".

Orientation video rules:
- Orientation videos must use format_tier "short" (not "micro").
- The discovery_objective must include at least 3 concrete sub-steps, for example:
  "Open the Customers table in the Browse Data tab, identify the 5 columns by name,
   scroll through the first page of rows, and explain what each column represents."
- estimated_word_count should be at least 150 so the video runs 60+ seconds.
"""


USER_PROMPT_SKELETON = """Return a compact structured JSON object (no markdown, no explanation) with this exact shape:

{
  "course_title": "string",
  "course_description": "string",
  "videos": [
    {
      "video_id": "string (e.g., video_1_1, video_1_2)",
      "title": "string",
      "video_type": "orientation|concept|demo|exercise|anti-pattern|capstone",
      "learning_objective": "string (1 sentence, pedagogical)",
      "discovery_objective": "string (1 sentence, concrete screen state verifiable by screenshot)",
      "prerequisite_videos": ["video_id", ...],
      "format_tier": "micro|short|mid|long|full",
      "new_capability": "string (the ONE thing this video teaches)"
    }
  ],
  "running_example": {
    "name": "string",
    "description": "string (one sentence)",
    "tables": [
      {
        "name": "string",
        "columns": [{"name": "string", "type": "string"}],
        "rows": integer
      }
    ]
  }
}

Output size constraints (STRICT — do not exceed the upper bound):
- Depth "micro" → exactly 1 video.
- Depth "short" → exactly 3 videos.
- Depth "mid" → exactly 8 videos.
- Depth "long" → exactly 15 videos.
- Depth "full" → exactly 19 videos (must cover all required SQL Fundamentals slots).

Discovery objective rules (CRITICAL):
- Mention a specific UI element/tab (Browse Data tab, Execute SQL tab, column header, filter box).
- Mention a specific table or view, OR state it is an Execute SQL tab query.
- Describe a visible state, not a concept.
- Achievable in 1-3 plain left-clicks.
- NEVER use "understand", "learn", "grasp", "comprehend".
- NEVER include exact numeric values from the data.
- MUST be ATOMIC: exactly ONE end state, ONE skill, ONE direction.
  BAD: "Click the amount column header once to sort ascending, then click it again to sort descending..."
  GOOD: "Sort the Orders table by amount in ascending order"
- NEVER use sequencing words: then, after that, finally, and then, next, subsequently.
- NEVER use multiple verbs describing multiple actions: click again, first click, next click, observe the.

Interaction constraints:
- 1-3 plain left-clicks per discovery objective.
- NO shift-click, Ctrl-click, Option-click, drag, right-click menus, or multi-step typing.
- Prefer recipe-friendly phrasing: "Open the X table in the Browse Data tab", "Sort the X table by Y in ascending order".
"""


USER_PROMPT_ENRICH = """Given the following course skeleton, return a JSON object that adds rich metadata to every video.

Keep the exact same "videos" array order and video_ids. Add these fields to each video object:

{
  "videos": [
    {
      "video_id": "same as input",
      "key_concept": "string (1 sentence — the single abstract idea the learner must understand)",
      "prerequisite_knowledge": "string (1 sentence — what the learner must already know from previous videos; 'None' for the first video)",
      "running_example_usage": "string (one sentence)",
      "proof_numbers": "string (concrete, re-checkable numbers from the data that the close must reference)",
      "estimated_word_count": integer,
      "has_recap": boolean,
      "has_preview": boolean,
      "recap_text_hint": "string (1 concrete sentence; null if has_recap is false)",
      "preview_text_hint": "string (1 exact-capability sentence; null if has_preview is false)"
    }
  ]
}

Rules:
- First video: has_recap=false, has_preview=true (unless it is the only video).
- Last video: has_recap=true, has_preview=false.
- Middle videos: has_recap=true, has_preview=true.
- Recap must reference concrete numbers from the previous video.
- Preview must state the EXACT next capability and avoid vague words: "more", "next steps", "continue", "further", "complex questions".
- Orientation videos: estimated_word_count >= 150, format_tier will be forced to "short".
- estimated_word_count targets: micro=50-200, short=150-350, mid=300-700, long=500-1000, full=700-1500.
- Return ONLY the JSON object described above. Do not repeat the full skeleton.

Course skeleton:
__SKELETON_JSON__
"""


class CurriculumDesigner:
    """
    Takes a topic, target audience, and depth level, and produces a
    CourseManifest that follows the content standard and reference curriculum
    structures.
    """

    def __init__(self, content_standard_path: str = "LESSON_CONTENT_STANDARD.md"):
        self.client = anthropic.Anthropic()
        self.lesson_standard = LessonStandard(content_standard_path)
        self.content_standard = self.lesson_standard.curriculum_prompt_fragment()
        self.cache_dir = Path(__file__).resolve().parent / "courses" / ".cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def design(
        self,
        topic: str,
        target_audience: str,
        depth: str,
        reference_curriculum: Optional[str] = None,
        application: str = "db_browser_sqlite",
        running_example_hint: Optional[str] = None,
    ) -> CourseManifest:
        """
        Design a CourseManifest for the given topic.

        Steps:
        1. Call the LLM to generate the course structure.
        2. Validate the progression.
        3. Generate a seed SQLite database.
        4. Pre-validate each video objective against the seed data.
        5. Build and return a CourseManifest.
        """
        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise ValueError("ANTHROPIC_API_KEY environment variable is not set.")

        # 1. Generate structured course design from the LLM (with caching).
        # Use a compact skeleton prompt first so the response fits in the context window,
        # then enrich with metadata in a second, smaller call.
        design = self._generate_design(
            topic=topic,
            target_audience=target_audience,
            depth=depth,
            reference_curriculum=reference_curriculum,
            application=application,
            running_example_hint=running_example_hint,
            user_prompt=USER_PROMPT_SKELETON,
            phase="skeleton",
        )

        # 1b. Validate skeleton count before paying for enrichment.
        count_limits = {"micro": (1, 1), "short": (3, 3), "mid": (8, 8), "long": (15, 15), "full": (19, 19)}
        lo, hi = count_limits.get(depth, (1, 24))
        if not (lo <= len(design.get("videos", [])) <= hi):
            raise ValueError(
                f"Depth '{depth}' requires {lo}-{hi} videos, but skeleton generated {len(design.get('videos', []))}."
            )

        # 1c. Enrich the skeleton with key_concept, recap/preview hints, proof numbers, etc.
        enrichment = self._enrich_design(design)
        enrich_by_id = {v["video_id"]: v for v in enrichment.get("videos", [])}
        for video in design["videos"]:
            extra = enrich_by_id.get(video["video_id"], {})
            video.update(
                {
                    k: v
                    for k, v in extra.items()
                    if k != "video_id" and v is not None
                }
            )

        # 2. Validate progression.
        self._validate_progression(design)

        # 2b. Split any compound sort objectives into separate videos.
        design["videos"] = self._split_compound_sorts(design["videos"])
        self._validate_progression(design)

        # 2c. Ensure any SQL queries in objectives/hints follow the course format.
        self._format_sql_in_video_fields(design["videos"])

        # 3. Generate seed database.
        course_id = self._course_id_from_title(design["course_title"])
        seed_path = generate_seed_database(
            course_id=course_id,
            schema=design["running_example"],
            output_dir=str(Path(__file__).resolve().parent / "discovery_output"),
        )

        # 4. Pre-validate each discovery objective (concrete, screenshot-verifiable).
        for video in design["videos"]:
            if not validate_objective_against_db(
                objective=video["discovery_objective"],
                db_path=str(seed_path),
                application=application,
            ):
                raise ValueError(
                    f"Objective for {video['video_id']} cannot be achieved with seed DB: "
                    f"{video['discovery_objective']}"
                )

        # 5. Build CourseManifest.
        # Enforce orientation-video duration rules.
        for v in design["videos"]:
            if v.get("video_type") == "orientation":
                v["format_tier"] = "short"
                v["estimated_word_count"] = max(v.get("estimated_word_count", 0), 150)

        # Clean up vague previews by deriving concrete text from the next video.
        vague_preview_words = {"more", "next steps", "continue", "further", "complex questions", "precise"}
        for i, v in enumerate(design["videos"]):
            if not v.get("has_preview"):
                continue
            preview = (v.get("preview_text_hint") or "").lower()
            if any(word in preview for word in vague_preview_words):
                next_v = design["videos"][i + 1] if i + 1 < len(design["videos"]) else None
                if next_v:
                    v["preview_text_hint"] = (
                        f"Next we'll {next_v.get('new_capability') or next_v['title']}."
                    )

        lesson_builder = LessonBuilder()
        videos: List[VideoManifest] = []
        for v in design["videos"]:
            video = VideoManifest(
                video_id=v["video_id"],
                title=v["title"],
                video_type=v.get("video_type", "demo"),
                learning_objective=v["learning_objective"],
                discovery_objective=v["discovery_objective"],
                application=application,
                prerequisite_videos=v.get("prerequisite_videos", []),
                exercise_artifact={
                    "db_path": str(seed_path),
                    "table_name": design["running_example"]["tables"][0]["name"],
                    "description": design["running_example"]["description"],
                },
                format_tier=v["format_tier"],
                estimated_duration_seconds=self._estimate_duration(v),
                new_capability=v.get("new_capability"),
                key_concept=v.get("key_concept"),
                prerequisite_knowledge=v.get("prerequisite_knowledge"),
                running_example_usage=v.get("running_example_usage"),
                proof_numbers=v.get("proof_numbers"),
                estimated_word_count=v.get("estimated_word_count", 0),
                has_recap=v.get("has_recap", False),
                has_preview=v.get("has_preview", False),
                recap_text_hint=v.get("recap_text_hint"),
                preview_text_hint=v.get("preview_text_hint"),
            )
            # Path A: generate and validate the lesson script up front.
            try:
                beats = lesson_builder.generate_script(video)
                if beats:
                    ok, errors, warnings = lesson_builder.validate_script(beats, video)
                    for warning in warnings:
                        print(
                            f"Warning: script validation for {video.video_id}: {warning}",
                            file=sys.stderr,
                        )
                    if not ok:
                        print(
                            f"Warning: script validation failed for {video.video_id}; regenerating once.",
                            file=sys.stderr,
                        )
                        beats = lesson_builder.generate_script(video, fix_errors=errors)
                        if beats:
                            ok, errors, warnings = lesson_builder.validate_script(beats, video)
                            for warning in warnings:
                                print(
                                    f"Warning: script validation for {video.video_id}: {warning}",
                                    file=sys.stderr,
                                )
                            if not ok:
                                print(
                                    f"Warning: script validation still failing for {video.video_id}: {errors}",
                                    file=sys.stderr,
                                )
                    if beats:
                        video.script_beats = [
                            {
                                k: getattr(b, k)
                                for k in (
                                    "beat_id", "kind", "text", "action",
                                    "visual_check", "attaches_to", "target_id",
                                    "video_clip_path",
                                )
                                if getattr(b, k) is not None
                            }
                            for b in beats
                        ]
            except Exception as exc:
                print(
                    f"Warning: could not generate script for {video.video_id}: {exc}",
                    file=sys.stderr,
                )
            videos.append(video)

        manifest = CourseManifest(
            course_id=course_id,
            title=design["course_title"],
            description=design["course_description"],
            target_audience=target_audience,
            videos=videos,
            running_example=design["running_example"],
        )

        errors = self._validate_manifest(manifest, depth=depth)
        for fix_attempt in range(2):
            if not errors:
                break
            design = self._generate_design(
                topic=topic,
                target_audience=target_audience,
                depth=depth,
                reference_curriculum=reference_curriculum,
                application=application,
                running_example_hint=running_example_hint,
                fix_errors=errors,
                user_prompt=USER_PROMPT_SKELETON,
                phase="skeleton",
            )
            # Validate skeleton count before enrichment.
            if not (lo <= len(design.get("videos", [])) <= hi):
                raise ValueError(
                    f"Depth '{depth}' requires {lo}-{hi} videos, but skeleton generated {len(design.get('videos', []))}."
                )

            # Re-apply enrichment and post-processing, then rebuild manifest for re-validation.
            enrichment = self._enrich_design(design)
            enrich_by_id = {v["video_id"]: v for v in enrichment.get("videos", [])}
            for video in design["videos"]:
                extra = enrich_by_id.get(video["video_id"], {})
                video.update(
                    {
                        k: v
                        for k, v in extra.items()
                        if k != "video_id" and v is not None
                    }
                )
            design["videos"] = self._split_compound_sorts(design["videos"])
            self._validate_progression(design)
            self._format_sql_in_video_fields(design["videos"])
            course_id = self._course_id_from_title(design["course_title"])
            for v in design["videos"]:
                if v.get("video_type") == "orientation":
                    v["format_tier"] = "short"
                    v["estimated_word_count"] = max(v.get("estimated_word_count", 0), 150)
            vague_preview_words = {"more", "next steps", "continue", "further", "complex questions", "precise"}
            for i, v in enumerate(design["videos"]):
                if not v.get("has_preview"):
                    continue
                preview = (v.get("preview_text_hint") or "").lower()
                if any(word in preview for word in vague_preview_words):
                    next_v = design["videos"][i + 1] if i + 1 < len(design["videos"]) else None
                    if next_v:
                        v["preview_text_hint"] = (
                            f"Next we'll {next_v.get('new_capability') or next_v['title']}."
                        )
            videos = [
                VideoManifest(
                    video_id=v["video_id"],
                    title=v["title"],
                    video_type=v.get("video_type", "demo"),
                    learning_objective=v["learning_objective"],
                    discovery_objective=v["discovery_objective"],
                    application=application,
                    prerequisite_videos=v.get("prerequisite_videos", []),
                    exercise_artifact={
                        "db_path": str(seed_path),
                        "table_name": design["running_example"]["tables"][0]["name"],
                        "description": design["running_example"]["description"],
                    },
                    format_tier=v["format_tier"],
                    estimated_duration_seconds=self._estimate_duration(v),
                    new_capability=v.get("new_capability"),
                    key_concept=v.get("key_concept"),
                    prerequisite_knowledge=v.get("prerequisite_knowledge"),
                    running_example_usage=v.get("running_example_usage"),
                    proof_numbers=v.get("proof_numbers"),
                    estimated_word_count=v.get("estimated_word_count", 0),
                    has_recap=v.get("has_recap", False),
                    has_preview=v.get("has_preview", False),
                    recap_text_hint=v.get("recap_text_hint"),
                    preview_text_hint=v.get("preview_text_hint"),
                )
                for v in design["videos"]
            ]
            manifest = CourseManifest(
                course_id=course_id,
                title=design["course_title"],
                description=design["course_description"],
                target_audience=target_audience,
                videos=videos,
                running_example=design["running_example"],
            )
            errors = self._validate_manifest(manifest, depth=depth)

        if errors:
            raise ValueError("Manifest validation failed:\n" + "\n".join(errors))

        return manifest

    # ------------------------------------------------------------------
    # LLM interaction
    # ------------------------------------------------------------------

    def _generate_design(
        self,
        topic: str,
        target_audience: str,
        depth: str,
        reference_curriculum: Optional[str],
        application: str,
        running_example_hint: Optional[str],
        fix_errors: Optional[List[str]] = None,
        user_prompt: str = USER_PROMPT,
        phase: str = "full",
    ) -> Dict[str, Any]:
        """Call the LLM and return the parsed design JSON."""
        cache_key = self._cache_key(
            phase,
            topic, target_audience, depth, application, running_example_hint,
            *(fix_errors or []),
        )
        cache_path = self.cache_dir / f"{cache_key}.json"
        if cache_path.exists() and not fix_errors:
            return json.loads(cache_path.read_text(encoding="utf-8"))

        system = SYSTEM_PROMPT.format(
            topic=topic,
            target_audience=target_audience,
            depth=depth,
            application=application,
            content_standard=self.content_standard,
            required_slots=_DEPTH_REQUIRED_SLOTS.get(depth, _DEPTH_REQUIRED_SLOTS["full"]),
        )

        user = user_prompt
        if fix_errors:
            user += (
                "\n\nThe previous design failed validation with these errors. "
                "Fix them and return a corrected JSON object with no explanation:\n"
                + "\n".join(f"- {e}" for e in fix_errors)
            )
        if running_example_hint:
            user += f"\n\nRunning example hint: {running_example_hint}"
        if reference_curriculum:
            ref_path = Path(reference_curriculum)
            if ref_path.exists():
                user += f"\n\nReference curriculum:\n{ref_path.read_text(encoding='utf-8')[:4000]}"

        messages: List[Dict[str, Any]] = [{"role": "user", "content": user}]
        full_text = ""

        for attempt in range(3):
            response = self.client.messages.create(
                model=DEFAULT_MODEL,
                max_tokens=8192,
                system=system,
                messages=messages,
            )

            text_parts = [block.text for block in response.content if block.type == "text"]
            chunk = "\n".join(text_parts)
            full_text += chunk

            # Try to parse what we have so far.
            parse_text = full_text.strip()
            if parse_text.startswith("```"):
                parse_text = re.sub(
                    r"^```(?:json)?\s*|\s*```$", "", parse_text, flags=re.DOTALL
                ).strip()

            try:
                design = json.loads(parse_text, strict=False)
                # Sanitize any literal control characters that slipped into strings.
                design = _sanitize_strings(design)
                cache_path.write_text(json.dumps(design, indent=2), encoding="utf-8")
                return design
            except json.JSONDecodeError as exc:
                debug_dir = Path(__file__).resolve().parent / "courses" / ".cache" / "debug"
                debug_dir.mkdir(parents=True, exist_ok=True)
                debug_path = debug_dir / f"{cache_key}_attempt{attempt+1}.txt"
                debug_path.write_text(
                    f"Stop reason: {getattr(response, 'stop_reason', 'unknown')}\n"
                    f"JSON error: {exc}\n\n--- RAW RESPONSE ---\n{full_text}",
                    encoding="utf-8",
                )
                # If the model was cut off, ask it to continue.
                stop_reason = getattr(response, "stop_reason", None)
                if stop_reason == "max_tokens" or "Unterminated string" in str(exc):
                    messages.append({"role": "assistant", "content": chunk})
                    messages.append(
                        {
                            "role": "user",
                            "content": (
                                "You did not finish the JSON object. Continue from exactly "
                                "the next character after your last output, with no preamble. "
                                "Do not repeat what you already output."
                            ),
                        }
                    )
                    continue
                raise

        raise RuntimeError("Could not parse LLM design response as JSON after multiple attempts.")

    def _enrich_design(self, skeleton: Dict[str, Any]) -> Dict[str, Any]:
        """Add rich metadata (key_concept, recap/preview hints, etc.) to a skeleton."""
        cache_key = self._cache_key("enrich", json.dumps(skeleton, sort_keys=True))
        cache_path = self.cache_dir / f"{cache_key}.json"
        if cache_path.exists():
            return json.loads(cache_path.read_text(encoding="utf-8"))

        # Strip large fields from the skeleton copy to keep the prompt compact.
        compact_skeleton = {
            "course_title": skeleton.get("course_title"),
            "course_description": skeleton.get("course_description"),
            "videos": [
                {
                    "video_id": v.get("video_id"),
                    "title": v.get("title"),
                    "video_type": v.get("video_type"),
                    "learning_objective": v.get("learning_objective"),
                    "discovery_objective": v.get("discovery_objective"),
                    "prerequisite_videos": v.get("prerequisite_videos", []),
                    "format_tier": v.get("format_tier"),
                    "new_capability": v.get("new_capability"),
                }
                for v in skeleton.get("videos", [])
            ],
        }
        user = USER_PROMPT_ENRICH.replace(
            "__SKELETON_JSON__", json.dumps(compact_skeleton, indent=2)
        )
        system = "You are a curriculum assistant that adds rich metadata to a course skeleton. Return only compact, valid JSON."

        messages: List[Dict[str, Any]] = [{"role": "user", "content": user}]
        full_text = ""

        for attempt in range(3):
            response = self.client.messages.create(
                model=DEFAULT_MODEL,
                max_tokens=4096,
                system=system,
                messages=messages,
            )
            text_parts = [block.text for block in response.content if block.type == "text"]
            chunk = "\n".join(text_parts)
            full_text += chunk

            parse_text = full_text.strip()
            if parse_text.startswith("```"):
                parse_text = re.sub(
                    r"^```(?:json)?\s*|\s*```$", "", parse_text, flags=re.DOTALL
                ).strip()

            try:
                enrichment = json.loads(parse_text, strict=False)
                enrichment = _sanitize_strings(enrichment)
                cache_path.write_text(json.dumps(enrichment, indent=2), encoding="utf-8")
                return enrichment
            except json.JSONDecodeError as exc:
                debug_dir = Path(__file__).resolve().parent / "courses" / ".cache" / "debug"
                debug_dir.mkdir(parents=True, exist_ok=True)
                debug_path = debug_dir / f"{cache_key}_attempt{attempt+1}.txt"
                debug_path.write_text(
                    f"Stop reason: {getattr(response, 'stop_reason', 'unknown')}\n"
                    f"JSON error: {exc}\n\n--- RAW RESPONSE ---\n{full_text}",
                    encoding="utf-8",
                )
                stop_reason = getattr(response, "stop_reason", None)
                if stop_reason == "max_tokens" or "Unterminated string" in str(exc):
                    messages.append({"role": "assistant", "content": chunk})
                    messages.append(
                        {
                            "role": "user",
                            "content": (
                                "You did not finish the JSON object. Continue from exactly "
                                "the next character after your last output, with no preamble."
                            ),
                        }
                    )
                    continue
                raise RuntimeError(
                    f"Could not parse enrichment response as JSON: {exc}\nRaw:\n{full_text[:2000]}"
                )

        # Fallback: return the skeleton un-enriched rather than crashing.
        print("Warning: could not enrich design from LLM; using skeleton.", file=sys.stderr)
        return {"videos": skeleton.get("videos", [])}

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_progression(design: Dict[str, Any]) -> None:
        """Validate that the video progression follows the content standard."""
        videos = design.get("videos", [])
        if not videos:
            raise ValueError("Design contains no videos.")

        video_ids = {v["video_id"] for v in videos}

        # First video has no prerequisites.
        if videos[0].get("prerequisite_videos"):
            raise ValueError(
                f"First video {videos[0]['video_id']} must have no prerequisites."
            )

        # Check each video.
        seen: set = set()
        for video in videos:
            vid = video["video_id"]
            prereqs = video.get("prerequisite_videos", [])

            # No cycles, all prereqs must be earlier in the list.
            for prereq in prereqs:
                if prereq not in video_ids:
                    raise ValueError(f"Video {vid} references unknown prerequisite {prereq}")
                if prereq not in seen:
                    raise ValueError(
                        f"Video {vid} has out-of-order prerequisite {prereq}"
                    )

            # Exactly one new capability.
            new_capability = video.get("new_capability", "")
            if not new_capability:
                raise ValueError(
                    f"Video {vid} must declare a new_capability"
                )

            seen.add(vid)

        # Last video closes on proof numbers and has no preview.
        last = videos[-1]
        if not last.get("proof_numbers"):
            raise ValueError(f"Last video {last['video_id']} must close on proof numbers.")
        if last.get("has_preview"):
            raise ValueError(f"Last video {last['video_id']} must not have a preview.")

        # First video has no recap; last video has no preview.
        first = videos[0]
        if first.get("has_recap"):
            raise ValueError(f"First video {first['video_id']} must not have a recap.")

        # Middle videos must have both recap and preview.
        for video in videos[1:-1]:
            vid = video["video_id"]
            if video.get("has_recap") and not video.get("recap_text_hint"):
                raise ValueError(f"Video {vid} has_recap=true but missing recap_text_hint")
            if video.get("has_preview") and not video.get("preview_text_hint"):
                raise ValueError(f"Video {vid} has_preview=true but missing preview_text_hint")

        # Every video must have a concrete discovery_objective.
        for video in videos:
            vid = video["video_id"]
            discovery = video.get("discovery_objective", "")
            if not discovery:
                raise ValueError(f"Video {vid} must have a discovery_objective.")
            forbidden = {"understand", "learn", "grasp", "comprehend"}
            lowered = discovery.lower()
            found = {w for w in forbidden if w in lowered}
            if found:
                raise ValueError(
                    f"Video {vid} discovery_objective uses conceptual words {found}: {discovery!r}"
                )
            ui_elements = {"tab", "button", "header", "box", "slide", "screen", "chart", "diagram"}
            if not any(el in lowered for el in ui_elements):
                raise ValueError(
                    f"Video {vid} discovery_objective must mention a UI element such as tab, button, header, box, or slide: {discovery!r}"
                )
            # Browse Data / visual objectives must name a table or view.
            # Execute SQL tab objectives operate on tables via the query text, so
            # mentioning the Execute SQL tab and result columns is sufficient.
            if "execute sql" not in lowered and not re.search(r"\b(table|view)\b", lowered):
                raise ValueError(
                    f"Video {vid} discovery_objective must mention a specific table or view: {discovery!r}"
                )

    @staticmethod
    def _validate_discovery_objective(obj: str, video_type: str = "demo") -> bool:
        """Return True if the discovery objective is atomic (one end state, one action)."""
        if not obj:
            return False
        lowered = obj.lower()
        # Clear sequencing words that imply multiple distinct actions over time.
        forbidden = [
            " then ", "then click", "and then", "after that", "finally", "next click",
            "first click", "second click", "click again", "click it again", "and observe",
            "subsequently", "once to sort", "again to sort", "then sort",
        ]
        if any(word in lowered for word in forbidden):
            return False
        # Orientation videos may list multiple observable sub-steps in one screen.
        if video_type == "orientation":
            return True
        # For non-orientation videos, also reject "first"/"second" as action ordering.
        ordering = ["first ", "second ", "next ", "then "]
        return not any(word in lowered for word in ordering)

    @staticmethod
    def _split_compound_sorts(videos: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Post-process: split any video whose objective contains two sort directions."""
        result: List[Dict[str, Any]] = []
        id_mapping: Dict[str, str] = {}

        for video in videos:
            obj = video.get("discovery_objective", "").lower()
            has_asc = any(w in obj for w in ("ascending", " asc", "smallest"))
            has_desc = any(w in obj for w in ("descending", " desc", "largest"))
            if has_asc and has_desc and "sort" in obj:
                original_id = video["video_id"]
                first_id = f"{original_id}_a"
                second_id = f"{original_id}_b"
                id_mapping[original_id] = second_id

                first = dict(video)
                first["video_id"] = first_id
                first["title"] = re.sub(r"\s+(and|&)\s+.*", "", video["title"], flags=re.IGNORECASE) or f"{video['title']} (Ascending)"
                first["discovery_objective"] = re.sub(
                    r"(?:,\s*)?(?:and\s+)?(?:then\s+)?.*descending.*",
                    " in ascending order",
                    video["discovery_objective"],
                    flags=re.IGNORECASE,
                )
                first["new_capability"] = (
                    video.get("new_capability", "").replace("descending", "ascending").replace("desc", "asc")
                )
                first["has_preview"] = False

                second = dict(video)
                second["video_id"] = second_id
                second["title"] = re.sub(r".*(?:and|&)\s+", "", video["title"], flags=re.IGNORECASE) or f"{video['title']} (Descending)"
                second["discovery_objective"] = re.sub(
                    r".*ascending.*(?:,\s*)?(?:and\s+)?(?:then\s+)?",
                    "Sort the ",
                    video["discovery_objective"],
                    flags=re.IGNORECASE,
                )
                second["discovery_objective"] = re.sub(
                    r"\bin ascending order\b", "in descending order",
                    second["discovery_objective"], flags=re.IGNORECASE,
                )
                second["new_capability"] = (
                    video.get("new_capability", "").replace("ascending", "descending").replace("asc", "desc")
                )
                second["has_recap"] = False
                second["prerequisite_videos"] = [first_id]

                result.extend([first, second])
            else:
                result.append(video)

        # Remap any prerequisites that pointed to a split video to the last split part.
        for video in result:
            new_prereqs = []
            for prereq in video.get("prerequisite_videos", []):
                new_prereqs.append(id_mapping.get(prereq, prereq))
            video["prerequisite_videos"] = new_prereqs

        return result

    def _validate_manifest(self, manifest: CourseManifest, depth: str = "full") -> List[str]:
        """Return a list of errors. Empty list = valid."""
        errors: List[str] = []
        videos = manifest.videos
        if not videos:
            errors.append("Manifest contains no videos.")
            return errors

        by_id = {v.video_id: v for v in videos}

        # --- Required slots and order (only enforced for full-depth courses) ---
        if depth == "full":
            required_slots = [
                ("orientation", "browse"),
                ("concept", "relationship"),
                ("demo", "sort"),
                ("demo", "filter"),
                ("demo", "numeric"),
                ("concept", "sql beats"),
                ("demo", "select"),
                ("demo", "where"),
                ("demo", "order by"),
                ("demo", "count"),
                ("demo", "group by"),
                ("demo", "sum"),
                ("concept", "connect"),
                ("demo", "inner join"),
                ("demo", "joined"),
                ("anti-pattern", "count"),
                ("anti-pattern", "group by"),
                ("anti-pattern", "sum"),
                ("capstone", "top customers"),
            ]

            # Build a list of (index, video_type, lowered title/new_capability) for matching.
            matches = []
            for i, v in enumerate(videos):
                text = f"{v.title} {v.new_capability or ''}".lower()
                matches.append((i, v.video_type, text))

            required_index = 0
            for req_type, req_keyword in required_slots:
                found = False
                while required_index < len(matches):
                    idx, vtype, text = matches[required_index]
                    required_index += 1
                    if vtype == req_type and req_keyword in text:
                        found = True
                        break
                if not found:
                    errors.append(
                        f"Missing required slot: a '{req_type}' video about '{req_keyword}'."
                    )

        # --- One-capability-per-video (strict) ---
        for v in videos:
            if v.video_type == "capstone":
                continue
            title_lower = v.title.lower()
            cap_lower = (v.new_capability or "").lower()
            if " and " in title_lower or " & " in title_lower:
                errors.append(
                    f"Video {v.video_id} title '{v.title}' contains 'and' or '&' (only capstones may integrate skills)."
                )
            if " and " in cap_lower or " & " in cap_lower:
                errors.append(
                    f"Video {v.video_id} new_capability '{v.new_capability}' contains 'and' or '&' (only capstones may integrate skills)."
                )
            if not self._validate_discovery_objective(v.discovery_objective, video_type=v.video_type):
                errors.append(
                    f"Video {v.video_id} discovery_objective is compound or sequential: {v.discovery_objective!r}"
                )

        # --- Capstone rules (only enforced for full-depth courses) ---
        if depth == "full":
            capstones = [v for v in videos if v.video_type == "capstone"]
            if not capstones:
                errors.append("A full course must include at least one capstone video.")
            else:
                if capstones[-1].video_id != videos[-1].video_id:
                    errors.append("The final capstone must be the last video in the course.")
                if len(capstones) > 1:
                    errors.append("Only one capstone video is allowed.")

        # --- Video count matches requested depth ---
        count_limits = {
            "micro": (1, 2),
            "short": (3, 4),
            "mid": (8, 12),
            "long": (15, 18),
            "full": (19, 24),
        }
        lo, hi = count_limits.get(depth, (1, 24))
        if not (lo <= len(videos) <= hi):
            errors.append(
                f"Depth '{depth}' requires {lo}-{hi} videos, but {len(videos)} were generated."
            )

        # --- Orientation count and duration rules ---
        orientation_count = sum(1 for v in videos if v.video_type == "orientation")
        if orientation_count > 2:
            errors.append(f"Too many orientation videos ({orientation_count}); max is 2.")
        for v in videos:
            if v.video_type == "orientation":
                if v.format_tier != "short":
                    errors.append(
                        f"Orientation video {v.video_id} must have format_tier 'short', got '{v.format_tier}'."
                    )
                if v.estimated_word_count < 150:
                    errors.append(
                        f"Orientation video {v.video_id} estimated_word_count is too small ({v.estimated_word_count}); minimum is 150."
                    )

        # --- DAG / prerequisites ---
        visited: set = set()
        rec_stack: set = set()

        def has_cycle(vid: str, path: List[str]) -> bool:
            if vid in rec_stack:
                cycle_start = path.index(vid)
                errors.append(
                    "Prerequisite cycle detected: "
                    + " -> ".join(path[cycle_start:] + [vid])
                )
                return True
            if vid in visited:
                return False
            visited.add(vid)
            rec_stack.add(vid)
            path.append(vid)
            for prereq in by_id[vid].prerequisite_videos:
                if prereq not in by_id:
                    errors.append(f"Video {vid} references unknown prerequisite {prereq}")
                    return True
                if has_cycle(prereq, path):
                    return True
            path.pop()
            rec_stack.remove(vid)
            return False

        for v in videos:
            has_cycle(v.video_id, [])

        # --- Recap concrete numbers ---
        for v in videos:
            if v.has_recap and v.recap_text_hint:
                if not re.search(r"\d", v.recap_text_hint):
                    errors.append(
                        f"Video {v.video_id} recap must reference concrete numbers: {v.recap_text_hint!r}"
                    )

        # --- Preview exact capability ---
        for v in videos:
            if v.has_preview and v.preview_text_hint:
                lowered = v.preview_text_hint.lower()
                vague = ["more", "next steps", "continue", "further"]
                if any(word in lowered for word in vague):
                    errors.append(
                        f"Video {v.video_id} preview is too vague: {v.preview_text_hint!r}"
                    )

        return errors

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _load_content_standard(path: str) -> str:
        try:
            return Path(path).read_text(encoding="utf-8")
        except FileNotFoundError:
            return ""

    @staticmethod
    def _course_id_from_title(title: str) -> str:
        slug = re.sub(r"[^a-z0-9]+", "_", title.lower()).strip("_")
        return slug[:64] or "course"

    @staticmethod
    def _estimate_duration(video: Dict[str, Any]) -> int:
        words = video.get("estimated_word_count", 0)
        # 150 wpm ≈ 2.5 words per second.
        return int(round(words / 2.5)) if words else 0

    @staticmethod
    def _format_sql_in_video_fields(videos: List[Dict[str, Any]]) -> None:
        """Reformat any inline SQL queries in video text fields to course standard."""
        fields = [
            "discovery_objective",
            "learning_objective",
            "new_capability",
            "running_example_usage",
            "proof_numbers",
            "recap_text_hint",
            "preview_text_hint",
        ]
        for video in videos:
            for field in fields:
                value = video.get(field)
                if (
                    isinstance(value, str)
                    and "select" in value.lower()
                    and "from" in value.lower()
                ):
                    video[field] = format_sql_in_text(value)

    @staticmethod
    def _cache_key(*parts: str) -> str:
        hasher = hashlib.sha256()
        hasher.update(_CACHE_VERSION.encode("utf-8"))
        for part in parts:
            hasher.update(part.encode("utf-8"))
        return hasher.hexdigest()[:24]


# ---------------------------------------------------------------------------
# Seed database generation
# ---------------------------------------------------------------------------


def generate_seed_database(course_id: str, schema: Dict[str, Any], output_dir: str) -> Path:
    """
    Create a SQLite database with realistic sample data based on the provided
    running-example schema.

    When the schema contains both Customers and Orders tables, rows are generated
    relationally: one customer has 10+ orders, the rest have 1-2, dates span 12+
    months, statuses include NULLs, and at least 3 countries are represented.
    """
    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)
    db_path = output_dir_path / f"{course_id}_seed.db"

    if db_path.exists():
        db_path.unlink()

    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()

    tables = schema.get("tables", [])
    table_names = {t["name"] for t in tables}

    if "Customers" in table_names and "Orders" in table_names:
        _create_relational_tables(cursor, tables)
    else:
        for table in tables:
            _create_generic_table(cursor, table)

    conn.commit()
    conn.close()
    return db_path


def _create_relational_tables(cursor, tables: List[Dict[str, Any]]) -> None:
    """Create Customers and Orders tables with deterministic relational data."""
    by_name = {t["name"]: t for t in tables}
    customers_table = by_name["Customers"]
    orders_table = by_name["Orders"]

    # Create tables in dependency order.
    c_cols = customers_table["columns"]
    o_cols = orders_table["columns"]
    c_defs = ", ".join(f"{c['name']} {c['type']}" for c in c_cols)
    o_defs = ", ".join(f"{c['name']} {c['type']}" for c in o_cols)
    cursor.execute(f"CREATE TABLE Customers ({c_defs})")
    cursor.execute(f"CREATE TABLE Orders ({o_defs})")

    customer_rows, order_rows = _generate_customers_orders(customers_table, orders_table)

    c_placeholders = ", ".join("?" * len(c_cols))
    o_placeholders = ", ".join("?" * len(o_cols))
    cursor.executemany(f"INSERT INTO Customers VALUES ({c_placeholders})", customer_rows)
    cursor.executemany(f"INSERT INTO Orders VALUES ({o_placeholders})", order_rows)


def _create_generic_table(cursor, table: Dict[str, Any]) -> None:
    """Create and populate a single generic table using column/type heuristics."""
    table_name = table["name"]
    columns = table.get("columns", [])
    row_count = table.get("rows", 12)

    if not columns:
        return

    col_defs = ", ".join(f"{c['name']} {c['type']}" for c in columns)
    cursor.execute(f"CREATE TABLE {table_name} ({col_defs})")

    rows = _generate_rows(columns, row_count)
    placeholders = ", ".join("?" * len(columns))
    cursor.executemany(f"INSERT INTO {table_name} VALUES ({placeholders})", rows)


def _generate_customers_orders(
    customers_table: Dict[str, Any], orders_table: Dict[str, Any]
) -> tuple[List[tuple], List[tuple]]:
    """Generate related Customers and Orders rows enforcing schema-utilization rules."""
    rng = random.Random(42)

    c_cols = customers_table.get("columns", [])
    o_cols = orders_table.get("columns", [])
    c_count = max(customers_table.get("rows", 10), 3)
    o_count = max(orders_table.get("rows", 50), c_count + 9)

    countries = ["USA", "Canada", "UK", "Germany", "France"]
    first_names = [
        "Alice", "Bob", "Carol", "David", "Eva", "Frank", "Grace", "Henry", "Ivy", "Jack",
        "Karen", "Liam", "Mia", "Noah", "Olivia",
    ]
    last_names = [
        "Smith", "Jones", "Brown", "Taylor", "Anderson", "Thomas", "Jackson", "White",
        "Harris", "Martin", "Thompson", "Garcia",
    ]

    # --- Customers ---
    customer_ids = list(range(1, c_count + 1))
    customer_rows: List[tuple] = []
    customer_country: Dict[int, str] = {}
    for cid in customer_ids:
        country = rng.choice(countries)
        customer_country[cid] = country
        name = f"{rng.choice(first_names)} {rng.choice(last_names)}"
        customer_rows.append(_build_customer_row(c_cols, cid, name, country, rng))

    # --- Order counts: one whale, everyone else 1-2 orders ---
    other_counts = [rng.choice([1, 2]) for _ in range(c_count - 1)]
    other_total = sum(other_counts)
    whale_count = o_count - other_total
    if whale_count < 10:
        # Not enough orders requested; bump the whale until the rule is met.
        whale_count = 10
        o_count = whale_count + other_total

    order_counts: Dict[int, int] = {customer_ids[0]: whale_count}
    for cid, cnt in zip(customer_ids[1:], other_counts):
        order_counts[cid] = cnt

    # --- Orders spanning at least 12 months ---
    months: List[tuple] = []
    start_year, start_month = 2023, 6
    for m in range(12):
        year = start_year + (start_month + m - 1) // 12
        month = (start_month + m - 1) % 12 + 1
        months.append((year, month))

    statuses = ["Shipped", "Delivered", "Pending", "Cancelled", "Processing", None]
    status_weights = [30, 25, 15, 10, 10, 10]
    regions = ["North", "South", "East", "West"]

    order_rows: List[tuple] = []
    oid = 1
    for cid, count in order_counts.items():
        for _ in range(count):
            year, month = rng.choice(months)
            day = rng.randint(1, 28)
            order_date = f"{year}-{month:02d}-{day:02d}"
            amount = round(rng.uniform(20.0, 500.0), 2)
            status = rng.choices(statuses, weights=status_weights)[0]
            region = rng.choice(regions)
            order_rows.append(
                _build_order_row(o_cols, oid, cid, order_date, amount, status, region, rng)
            )
            oid += 1

    return customer_rows, order_rows


def _build_customer_row(
    columns: List[Dict[str, str]], cid: int, name: str, country: str, rng: random.Random
) -> tuple:
    """Map generated values onto the Customers schema columns."""
    row: List[Any] = []
    for col in columns:
        n = col["name"].lower()
        t = col["type"].upper()
        if ("id" in n and "PRIMARY" in t) or n in ("id", "customerid"):
            row.append(cid)
        elif "name" in n:
            row.append(name)
        elif "email" in n:
            parts = name.lower().split()
            row.append(f"{parts[0]}.{parts[1]}@example.com")
        elif "country" in n:
            row.append(country)
        elif "signup" in n or "date" in n:
            row.append(f"2023-{rng.randint(1, 12):02d}-{rng.randint(1, 28):02d}")
        elif "TEXT" in t:
            row.append(rng.choice(["Active", "Premium", "Standard"]))
        elif "INTEGER" in t:
            row.append(rng.randint(1, 100))
        elif "REAL" in t:
            row.append(round(rng.uniform(1.0, 100.0), 2))
        else:
            row.append(None)
    return tuple(row)


def _build_order_row(
    columns: List[Dict[str, str]],
    oid: int,
    cid: int,
    order_date: str,
    amount: float,
    status: Optional[str],
    region: str,
    rng: random.Random,
) -> tuple:
    """Map generated values onto the Orders schema columns."""
    row: List[Any] = []
    for col in columns:
        n = col["name"].lower()
        t = col["type"].upper()
        if n in ("id", "orderid", "order_id") or ("PRIMARY" in t and "id" in n):
            row.append(oid)
        elif n in ("customerid", "customer_id", "customer"):
            row.append(cid)
        elif "date" in n:
            row.append(order_date)
        elif "amount" in n or "total" in n or "price" in n or "cost" in n or "revenue" in n:
            row.append(amount)
        elif "status" in n:
            row.append(status)
        elif "region" in n:
            row.append(region)
        elif "product" in n or "item" in n:
            row.append(rng.choice(["Laptop", "Mouse", "Keyboard", "Monitor", "Headphones", "Webcam"]))
        elif "quantity" in n:
            row.append(rng.randint(1, 5))
        elif "TEXT" in t:
            row.append(rng.choice(["A", "B", "C"]))
        elif "INTEGER" in t:
            row.append(rng.randint(1, 100))
        elif "REAL" in t:
            row.append(round(rng.uniform(1.0, 100.0), 2))
        else:
            row.append(None)
    return tuple(row)


def _generate_rows(columns: List[Dict[str, str]], row_count: int) -> List[tuple]:
    """Generate realistic rows for a generic table based on column names/types."""
    regions = ["North", "South", "East", "West"]
    countries = ["USA", "Canada", "UK", "Germany", "France", "Japan", "Australia", "Brazil"]
    months = ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12"]
    first_names = ["Alice", "Bob", "Carol", "David", "Eva", "Frank", "Grace", "Henry", "Ivy", "Jack"]
    last_names = ["Smith", "Jones", "Brown", "Taylor", "Anderson", "Thomas", "Jackson", "White"]
    products = ["Laptop", "Mouse", "Keyboard", "Monitor", "Headphones", "Webcam", "Dock", "Cable"]
    rows: List[tuple] = []
    rng = random.Random(42)

    base_values: List[Dict[str, Any]] = []
    for i in range(1, row_count + 1):
        base = {
            "id": i * 3,
            "region": rng.choice(regions),
            "country": rng.choice(countries),
            "date": f"2024-{rng.choice(months)}-{rng.randint(1, 28):02d}",
            "customer": f"{rng.choice(first_names)} {rng.choice(last_names)}",
            "product": rng.choice(products),
            "quantity": rng.randint(1, 10),
            "unit_price": round(rng.uniform(15, 350), 2),
            "amount": round(rng.choice([85.0, 95.75, 120.5, 150.0, 210.25, 340.0]) + rng.uniform(-10, 10), 2),
        }
        base_values.append(base)

    for base in base_values:
        row = []
        for col in columns:
            name = col["name"].lower()
            col_type = col["type"].upper()

            if "PRIMARY KEY" in col_type or name in ("id", "orderid"):
                row.append(base["id"])
            elif "region" in name:
                row.append(base["region"])
            elif "country" in name:
                row.append(base["country"])
            elif "date" in name:
                row.append(base["date"])
            elif "customer" in name or "name" in name:
                row.append(base["customer"])
            elif "product" in name:
                row.append(base["product"])
            elif "status" in name:
                row.append(rng.choice(["Shipped", "Pending", "Delivered", "Cancelled", "Processing"]))
            elif "quantity" in name or "count" in name:
                row.append(base["quantity"])
            elif "unitprice" in name or "unit_price" in name:
                row.append(base["unit_price"])
            elif "totalamount" in name or "total_amount" in name or "amount" in name:
                row.append(round(base["quantity"] * base["unit_price"], 2))
            elif "price" in name or "cost" in name or "revenue" in name:
                row.append(base["amount"])
            elif "TEXT" in col_type:
                row.append(base["product"])
            elif "INTEGER" in col_type:
                row.append(base["id"])
            elif "REAL" in col_type:
                row.append(base["amount"])
            else:
                row.append(None)
        rows.append(tuple(row))

    return rows


# ---------------------------------------------------------------------------
# Objective validation
# ---------------------------------------------------------------------------


def validate_objective_against_db(objective: str, db_path: str, application: str) -> bool:
    """
    Quick sanity check: can this objective be achieved with this database?
    For DB Browser: check that the table and relevant columns exist and that
    the table has enough rows to be meaningful.
    """
    if application != "db_browser_sqlite":
        return True  # Only DB Browser validation is implemented.

    if not Path(db_path).exists():
        return False

    try:
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()

        # Discover tables.
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = {row[0] for row in cursor.fetchall()}

        # Find table mentioned in objective.
        mentioned_table = None
        for table in tables:
            if table.lower() in objective.lower():
                mentioned_table = table
                break

        if not mentioned_table:
            # Execute SQL tab objectives do not need to name a table; just verify
            # the database has usable tables.
            lowered = objective.lower()
            if "execute sql" in lowered or "query" in lowered:
                for table in tables:
                    cursor.execute(f"SELECT COUNT(*) FROM {table}")
                    if cursor.fetchone()[0] >= 5:
                        conn.close()
                        return True

            # Fallback: if only one table exists, assume it's the target.
            if len(tables) == 1:
                mentioned_table = next(iter(tables))
            else:
                conn.close()
                return False

        # Check row count.
        cursor.execute(f"SELECT COUNT(*) FROM {mentioned_table}")
        count = cursor.fetchone()[0]
        if count < 5:
            conn.close()
            return False

        # Check columns mentioned in objective.
        cursor.execute(f"PRAGMA table_info({mentioned_table})")
        columns = {row[1].lower() for row in cursor.fetchall()}

        # Extract likely column names from the objective (simple heuristic).
        words = re.findall(r"[a-zA-Z_][a-zA-Z0-9_]*", objective)
        for word in words:
            if word.lower() in columns and word.lower() != mentioned_table.lower():
                conn.close()
                return True

        conn.close()
        # If no column is explicitly mentioned, still pass if table has rows.
        return count >= 5
    except Exception:
        return False


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    designer = CurriculumDesigner()
    manifest = designer.design(
        topic="Sorting and filtering data in SQLite",
        target_audience="Beginner data analysts with no SQL experience",
        depth="short",
        application="db_browser_sqlite",
        running_example_hint="e-commerce orders",
    )

    print("Course:", manifest.title)
    print("Videos:", len(manifest.videos))
    for v in manifest.videos:
        print(f"  {v.video_id}: {v.title} ({v.format_tier})")
        print(f"    Learning objective: {v.learning_objective}")
        print(f"    Discovery objective: {v.discovery_objective}")
        print(f"    Prerequisites: {v.prerequisite_videos}")
        print(f"    Exercise DB: {v.exercise_artifact.get('db_path')}")
        print()
