#!/usr/bin/env python3
"""
compiler/lesson_builder.py

Lesson-first script generation, validation, action derivation, and graph
construction. This is the Path A orchestrator: it generates a narration script
before any screen action is taken, validates it against the SQL Essentials
standard, derives deterministic UI actions, executes them through the discovery
harness, and builds an ExecutionGraph from the resulting recorded clips.
"""

from __future__ import annotations

import json
import os
import re
import sqlite3
import subprocess
import sys
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Set, Tuple

import anthropic

from .discovery import DiscoveryRecipes, EndStateDiscovery
from .graph_store import GraphStore
from .lesson_standard import LessonStandard
from .narrator import (
    FUNCTION_WORDS,
    MIN_WORDS,
    ScriptBeat,
    _contains_action_word,
    _contains_filler,
    _contains_result_phrase,
    _contains_state_setup_phrase,
    _extract_numbers,
    _format_action_sql,
)
from .schemas import ActionEdge, DiscoveryResult, ExecutionGraph, NarrationBeat, ScreenState
from .sql_formatter import format_sql_query
from .cost_tracker import get_tracker, tracked_create

MODEL = os.environ.get("NARRATOR_MODEL", "claude-sonnet-5")

_APP_FRIENDLY_NAMES = {
    "db_browser_sqlite": "DB Browser for SQLite",
    "metabase": "Metabase",
    "excel": "Excel",
    "power_bi": "Power BI",
    "mysql_workbench": "MySQL Workbench",
}


def _friendly_app_name(application: str) -> str:
    return _APP_FRIENDLY_NAMES.get(application, application.replace("_", " ").title())


class LessonBuilder:
    """
    Generates a narration script from a VideoManifest, derives the concrete
    UI action sequence, executes it through the discovery harness, and builds
    an ExecutionGraph from the recorded demo-beat clips.
    """

    def __init__(self, content_standard_path: str = "LESSON_CONTENT_STANDARD.md"):
        self.client = anthropic.Anthropic()
        self.lesson_standard = LessonStandard(content_standard_path)
        self.style_guide_path = str(
            Path(__file__).resolve().parent / "style_guide.md"
        )

    # ------------------------------------------------------------------
    # Standard ingestion
    # ------------------------------------------------------------------

    def ingest_standard(self) -> dict:
        """
        Extract the lesson content standard into a structured dict for prompts
        and quality gates.
        """
        text = self.lesson_standard.raw_text
        return {
            "structure_pattern": self._extract_section(text, "The five rules", "Three more rules"),
            "voice": self._extract_section(text, "Sentence-level style", "Grounded in SQL Essential Training"),
            "pacing": self._extract_section(text, "Corollary: no click is fast", "The five rules"),
            "sql_style": self._extract_section(text, "SQL Query Formatting Standard", "## General Rules"),
            "validation_habit": self._extract_section(text, "### 8. Every explanation closes", "## Grounded"),
        }

    @staticmethod
    def _extract_section(text: str, start_marker: str, end_marker: str) -> str:
        """Return the text between two markdown headings, trimmed."""
        start = text.find(start_marker)
        if start == -1:
            return ""
        end = text.find(end_marker, start + len(start_marker))
        if end == -1:
            end = len(text)
        return re.sub(r"\n{2,}", "\n", text[start:end]).strip()

    # ------------------------------------------------------------------
    # Deterministic script generation helpers
    # ------------------------------------------------------------------

    _FORBIDDEN_VOICE_PATTERNS = [
        r"\byou'll\b",
        r"\byou need to\b",
        r"\byou need\b",
        r"\byour\b",
        r"\bimportant to note\b",
        r"\bbefore you\b",
        r"\bif you skip\b",
        r"\bif you\b",
        r"\bit is important\b",
        r"\bit's important\b",
        r"\bin order to\b",
        r"\bas you can see\b",
        r"\bbasically\b",
        r"\bessentially\b",
        r"\bvery\b",
        r"\breally\b",
        r"\bjust\b",
        r"\bsimply\b",
        r"\bunderstand\b",
        r"\bunderstanding\b",
        r"\blearn\b",
        r"\blearning\b",
        r"\bgrasp\b",
        r"\bcomprehend\b",
        r"\bconcept\b",
        r"\babstract\b",
        r"\bthe fact that\b",
        r"\bthis is because\b",
        r"\bwhich means\b",
        r"\btherefore\b",
    ]

    _SECOND_PERSON_PATTERN = re.compile(r"\byou\b|\byour\b|\byou'll\b|\byou're\b", re.IGNORECASE)

    @staticmethod
    def _word_count(text: str) -> int:
        return len(text.split())

    @staticmethod
    def _min_words_for_kind(kind: str) -> int:
        """Per-beat minimum narration words used by the integrity gate."""
        return MIN_WORDS.get(kind, 15)

    @staticmethod
    def _planned_action_seconds(action: Optional[Dict[str, Any]]) -> float:
        """Estimate how many seconds a demo action should take on screen."""
        if not isinstance(action, dict):
            return 2.0
        action_type = action.get("type")
        if action_type == "wait":
            return float(action.get("duration", 1.5))
        if action_type in ("type_block", "type"):
            text = str(action.get("text") or action.get("detail") or "")
            return max(1.5, 0.10 * len(text) + 0.5)
        if action_type == "type_segments":
            segments = action.get("segments") or []
            text = "".join(
                (s.get("text", "") if isinstance(s, dict) else str(s)) for s in segments
            )
            return max(1.5, 0.10 * len(text) + 0.5)
        if action_type in ("run_query", "execute_query"):
            return 2.0
        if action_type == "key":
            return 1.0
        if action_type == "click":
            return 1.5
        if action_type == "sequence":
            subs = action.get("actions") or []
            return sum(LessonBuilder._planned_action_seconds(sub) for sub in subs) or 2.0
        return 2.0

    @staticmethod
    def _split_sentences(text: str) -> List[str]:
        """Split text into sentences, keeping trailing fragments."""
        return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s.strip()]

    @staticmethod
    def _ends_with_terminal(text: str) -> bool:
        return bool(re.search(r"[.!?]$", text.strip()))

    def _beat_text_integrity_ok(self, text: str, kind: str) -> bool:
        """
        Return True when a beat text is a complete pedagogical sentence.

        Checks:
          - Ends with terminal punctuation.
          - Meets per-kind minimum word count.
          - Final sentence does not end on a function word (truncation-gaming guard).
        """
        text = text.strip()
        if not text or not self._ends_with_terminal(text):
            return False
        words = text.split()
        if len(words) < self._min_words_for_kind(kind):
            return False
        final_sentence = words[-1].rstrip(".!?").strip()
        if final_sentence and final_sentence.lower() in FUNCTION_WORDS:
            return False
        return True

    def script_integrity_ok(self, beats: List[ScriptBeat]) -> bool:
        """Harness-facing checker: every beat passes the integrity gate."""
        if not beats:
            return False
        for beat in beats:
            if not self._beat_text_integrity_ok(beat.text, beat.kind):
                return False
        return True

    def _enforce_sentence_integrity(self, beats: List[ScriptBeat]) -> None:
        """
        Rewrite any beat that fails the integrity gate.

        Rewriting must add whole sentences or call the LLM; truncating existing
        text is forbidden. After this method, every beat must end with terminal
        punctuation and meet its per-kind minimum word count.
        """
        for beat in beats:
            original = beat.text.strip()
            if self._beat_text_integrity_ok(original, beat.kind):
                continue
            fixed = self._complete_beat_text(beat)
            fixed = fixed.strip()
            if fixed and not self._ends_with_terminal(fixed):
                fixed += "."
            # If the deterministic completion still fails, ask the LLM to expand.
            if not self._beat_text_integrity_ok(fixed, beat.kind):
                fixed = self._llm_complete_beat(beat, fixed)
                fixed = fixed.strip()
                if fixed and not self._ends_with_terminal(fixed):
                    fixed += "."
            beat.text = fixed
            print(
                f"  [INTEGRITY] {beat.beat_id}: completed sentence ({len(beat.text.split())} words)",
                file=sys.stderr,
            )

    def _complete_beat_text(self, beat: ScriptBeat) -> str:
        """
        Complete a single beat by keeping all complete sentences and appending
        generic, non-fabricating completions until the per-kind minimum is met.
        """
        text = beat.text.strip().rstrip(",;:")
        kind = beat.kind
        min_words = self._min_words_for_kind(kind)

        sentences = self._split_sentences(text)
        complete = [s for s in sentences if self._ends_with_terminal(s)]
        if complete:
            candidate = " ".join(complete)
        else:
            candidate = self._rewrite_fragment(beat)

        # Generic completions that add words without inventing facts.
        completions = {
            "opening": "We will build this skill step by step in DB Browser for SQLite.",
            "state": "This sets up the next action clearly.",
            "explain": "This relationship is what makes the query useful.",
            "concept": "These two pieces work together to retrieve the right data.",
            "demo": "The interface updates to show the change.",
            "validation": "This confirms the outcome matches our goal.",
            "close": "We are ready to apply this pattern in the next video.",
        }
        completion = completions.get(kind, "This completes the thought.")
        # Ceiling: 20 completion-append iterations.
        for _ in range(20):
            if len(candidate.split()) >= min_words:
                break
            candidate = f"{candidate} {completion}".strip()

        if kind == "close":
            return self._finish_close_beat(candidate)
        return candidate

    def _finish_close_beat(self, recap: str) -> str:
        """Force a close beat to end with a complete preview sentence."""
        preview = "Next, we'll explore filtering with WHERE."
        preview_wc = len(preview.split())
        recap_words = recap.split()
        if len(recap_words) + preview_wc <= 70:
            return f"{recap} {preview}".strip()
        if " and " in recap:
            shorter = recap.split(" and ", 1)[0].strip()
            if shorter and not re.search(r"[.!?]$", shorter):
                shorter += "."
            if len(shorter.split()) + preview_wc <= 70:
                return f"{shorter} {preview}".strip()
        return preview

    def _rewrite_fragment(self, beat: ScriptBeat) -> str:
        """Rewrite a single-sentence fragment as a complete thought without inventing facts."""
        text = beat.text.strip().rstrip(",;:")
        kind = beat.kind
        action = beat.action or {}
        action_type = action.get("type") if isinstance(action, dict) else None
        lower = text.lower()

        if kind == "opening":
            if "select" in lower and "contact" in lower:
                return "In this video, we will write a SELECT query for the customer contact list."
            if "select" in lower:
                return "In this video, we will write a SELECT query."
            return text + "."

        if kind in ("concept", "explain"):
            if "select" in lower and "from" in lower:
                return "SELECT chooses columns and FROM chooses the table."
            return text + "."

        if kind == "close":
            return self._finish_close_beat(text)

        # Demo / validation fragments for the Phase 1 pilot.
        if "comment block" in lower:
            return "We type a comment block so we remember the query's purpose."
        if "result pane shows" in lower and "firstname" in lower:
            return "The result pane shows 60 rows with FirstName, LastName, and Email."
        if "first name" in lower:
            return "We type a query that asks for first name, last name, email."
        if "result pane fills" in lower or ("result pane" in lower and "fills" in lower):
            return "We run the query and the result pane fills."

        if action_type in ("type_block", "type"):
            return f"{text} and the text appears."

        if action_type == "run_query":
            return "We run the query and the result pane fills."

        if action_type == "click":
            return f"{text} and the view opens."

        return text + "."

    def _llm_complete_beat(self, beat: ScriptBeat, current: str) -> str:
        """Ask the LLM to expand a beat into a complete, minimum-length sentence."""
        min_words = self._min_words_for_kind(beat.kind)
        prompt = (
            "Rewrite the following narration beat so it is one or more complete "
            "sentences ending with terminal punctuation. Do not invent facts not "
            "implied by the original. Use at least {min_words} words.\n\n"
            "Original: {original}\n"
            "Current attempt: {current}\n\n"
            "Rewritten beat:"
        ).format(min_words=min_words, original=beat.text.strip(), current=current)
        try:
            response = tracked_create(
                self.client,
                model=MODEL,
                max_tokens=256,
                messages=[{"role": "user", "content": prompt}],
            )
            text = " ".join(
                block.text for block in response.content if block.type == "text"
            ).strip()
            if text:
                return text
        except Exception as exc:
            print(f"Warning: LLM beat completion failed: {exc}", file=sys.stderr)
        return current

    @staticmethod
    def _total_word_limit(format_tier: str) -> int:
        """Soft upper word limit used for warnings, not hard failures."""
        return {"micro": 50, "short": 80, "mid": 120, "long": 600, "full": 600}.get(format_tier, 80)

    @staticmethod
    def _db_facts(db_path: Optional[str], table_name: str) -> Dict[str, Any]:
        """Return row count and column list for a table."""
        facts = {"row_count": 0, "columns": []}
        if not db_path or not Path(db_path).exists():
            return facts
        try:
            with sqlite3.connect(db_path) as conn:
                cur = conn.cursor()
                cur.execute(f"SELECT COUNT(*) FROM {table_name}")
                facts["row_count"] = cur.fetchone()[0]
                cur.execute(f"PRAGMA table_info({table_name})")
                facts["columns"] = [row[1] for row in cur.fetchall()]
        except Exception as exc:
            print(f"Warning: could not read DB facts for {table_name}: {exc}", file=sys.stderr)
        return facts

    @staticmethod
    def _top_value(db_path: Optional[str], table_name: str, column: str, direction: str) -> str:
        """Return the top value after sorting a column."""
        if not db_path or not Path(db_path).exists():
            return ""
        try:
            with sqlite3.connect(db_path) as conn:
                cur = conn.cursor()
                order = "ASC" if direction == "asc" else "DESC"
                cur.execute(f"SELECT {column} FROM {table_name} ORDER BY {column} {order} LIMIT 1")
                row = cur.fetchone()
                return str(row[0]) if row and row[0] is not None else ""
        except Exception as exc:
            print(f"Warning: could not read top value: {exc}", file=sys.stderr)
            return ""

    @staticmethod
    def _filtered_count(db_path: Optional[str], table_name: str, column: str, value: str) -> int:
        """Return the number of rows matching a filter value."""
        if not db_path or not Path(db_path).exists():
            return 0
        try:
            with sqlite3.connect(db_path) as conn:
                cur = conn.cursor()
                cur.execute(f"SELECT COUNT(*) FROM {table_name} WHERE {column} = ?", (value,))
                return cur.fetchone()[0]
        except Exception as exc:
            print(f"Warning: could not read filtered count: {exc}", file=sys.stderr)
            return 0

    @staticmethod
    def _first_value(db_path: Optional[str], table: str, column: str, context: str = "") -> str:
        """Return a representative non-null value from a column."""
        if not db_path or not Path(db_path).exists():
            return ""
        try:
            with sqlite3.connect(db_path) as conn:
                cur = conn.cursor()
                cur.execute(
                    f"SELECT DISTINCT {column} FROM {table} WHERE {column} IS NOT NULL LIMIT 1"
                )
                row = cur.fetchone()
                return str(row[0]) if row and row[0] is not None else ""
        except Exception as exc:
            print(f"Warning: could not read first value ({context}, db={db_path}, table={table}, col={column}): {exc}", file=sys.stderr)
            return ""

    @staticmethod
    def _table_exists(db_path: Optional[str], table: str) -> bool:
        if not db_path or not Path(db_path).exists():
            return False
        try:
            with sqlite3.connect(db_path) as conn:
                cur = conn.cursor()
                cur.execute(
                    "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table,)
                )
                return cur.fetchone() is not None
        except Exception:
            return False

    @staticmethod
    def _load_canonical_reference() -> Dict[str, Any]:
        """Load the canonical SQL reference for chapter-4 videos."""
        path = Path(__file__).resolve().parent / "courses" / "ch4_canonical_reference.md"
        if not path.exists():
            return {}
        text = path.read_text(encoding="utf-8")
        # Parse video blocks and grounding table.
        reference: Dict[str, Any] = {"videos": {}}
        video_pattern = re.compile(
            r"## Video\s+(\d+)\s+—\s+(.+?)\nObjective:\s*(.+?)\n+```sql\n(.*?)```",
            re.DOTALL,
        )
        for match in video_pattern.finditer(text):
            video_num = match.group(1)
            objective = match.group(3).strip()
            sql = match.group(4).strip()
            reference["videos"][video_num] = {
                "title": match.group(2).strip(),
                "objective": objective,
                "sql": sql,
            }
        return reference

    @staticmethod
    def _assert_canonical_format(text: str) -> List[str]:
        """
        Check ``text`` against the C9 Editor Format Contract.

        Returns a list of violation strings. An empty list means the text follows
        the contract.

        Contract:
          - Content begins at line 1 (no leading blank lines).
          - Optional history appears first; every history line starts with '--'.
          - Exactly one blank line separates history from the current block.
          - Current block starts with a /* ... */ comment header.
          - Comment header is immediately followed by the query (no blank line).
          - SELECT list fields are indented exactly 2 spaces.
          - No tabs, no trailing whitespace, no literal escape sequences.
          - One uncommented statement per block, terminated with ';'.
        """
        violations: List[str] = []
        if not text:
            return violations

        lines = text.split("\n")

        # 1. Zero leading blank lines.
        if lines and lines[0].strip() == "":
            violations.append("leading blank line(s) before content")

        # 2. No tabs anywhere.
        if "\t" in text:
            violations.append("tab character found")

        # 3. No trailing whitespace.
        for i, line in enumerate(lines, start=1):
            if line != line.rstrip():
                violations.append(f"trailing whitespace on line {i}")

        # 4. No literal escape sequences as visible text.
        if re.search(r"(?<!\\)\\[tnr]", text):
            violations.append("literal escape sequence (\\t, \\n, \\r) visible in text")

        # Split into history and current block.
        history_end = -1
        comment_start = -1
        comment_end = -1
        for i, line in enumerate(lines):
            stripped = line.strip()
            if stripped.startswith("--"):
                history_end = i
            elif stripped.startswith("/*"):
                comment_start = i
                break
            elif stripped:
                # Non-history, non-comment content before /* is invalid.
                violations.append(f"line {i + 1} is not part of history or comment block: {line!r}")
                break

        history_lines = lines[: history_end + 1] if history_end >= 0 else []
        rest_lines = lines[history_end + 1 :] if history_end >= 0 else lines

        # 5. History format: every line starts with '--' and one blank line before current block.
        if history_lines:
            for i, line in enumerate(history_lines, start=1):
                stripped = line.strip()
                if stripped and not stripped.startswith("--"):
                    violations.append(f"history line {i} is not commented with '--': {line!r}")
            # After history, there must be exactly one blank line before the current block.
            blank_count = 0
            for line in rest_lines:
                if line.strip() == "":
                    blank_count += 1
                else:
                    break
            if blank_count != 1:
                violations.append(
                    f"expected exactly one blank line between history and current block, found {blank_count}"
                )
            current_block_lines = rest_lines[blank_count:]
        else:
            current_block_lines = rest_lines

        # 6. Current block: /* ... */ comment immediately followed by query.
        if current_block_lines:
            if not current_block_lines[0].strip().startswith("/*"):
                violations.append("current block must start with a /* ... */ comment header")
            else:
                # Find the matching */.
                for i, line in enumerate(current_block_lines):
                    if line.strip().startswith("*/"):
                        comment_end = i
                        break
                if comment_end < 0:
                    violations.append("current block comment header is not closed with */")
                elif comment_end + 1 < len(current_block_lines):
                    next_line = current_block_lines[comment_end + 1]
                    if next_line.strip() == "":
                        violations.append("blank line between comment block and query")

        # 7. Field indentation exactly 2 spaces inside SELECT clauses of the current block.
        in_select = False
        for i, line in enumerate(current_block_lines, start=1):
            stripped = line.strip()
            if re.search(r"^\bSELECT\b", stripped, re.IGNORECASE):
                in_select = True
                continue
            if in_select and re.search(r"^\bFROM\b", stripped, re.IGNORECASE):
                in_select = False
                continue
            if in_select and stripped and not stripped.startswith("--"):
                leading = line[: len(line) - len(line.lstrip())]
                if leading != "  ":
                    violations.append(f"field on line {i} not indented exactly 2 spaces: {line!r}")

        # 8. One uncommented statement per block, terminated with ';'.
        uncommented = [
            (i, line)
            for i, line in enumerate(current_block_lines, start=1)
            if line.strip()
            and not line.strip().startswith("--")
            and not line.strip().startswith("/*")
            and not line.strip().startswith("*/")
        ]
        if uncommented:
            last_idx, last_line = uncommented[-1]
            if not last_line.rstrip().endswith(";"):
                violations.append(f"last query line (line {last_idx}) is not terminated with ';'")
            statements = [line for _, line in uncommented if line.rstrip().endswith(";")]
            if len(statements) > 1:
                violations.append("more than one statement in the current block")

        return violations

    def _assert_canonical_grounding(
        self,
        video_id: str,
        query: str,
        db_path: Optional[str],
    ) -> List[str]:
        """
        Assert that running ``query`` against ``db_path`` matches the canonical
        reference expectations for ``video_id``.

        Returns a list of failure messages; empty list means grounding passed.
        """
        errors: List[str] = []
        reference = self._load_canonical_reference()
        video_ref = reference.get("videos", {}).get(video_id.lstrip("video_").split("_")[0])
        if not video_ref:
            return errors

        if not db_path or not Path(db_path).exists():
            errors.append(f"cannot ground {video_id}: database not found")
            return errors

        try:
            with sqlite3.connect(db_path) as conn:
                cur = conn.cursor()
                cur.execute(query)
                rows = cur.fetchall()
                columns = [desc[0] for desc in cur.description] if cur.description else []
                actual_row_count = len(rows)
                actual_first_row = list(rows[0]) if rows else []
        except Exception as exc:
            errors.append(f"grounding query failed for {video_id}: {exc}")
            return errors

        # Parse expected values from the reference markdown.
        ref_text = video_ref.get("objective", "") + "\n" + video_ref.get("sql", "")
        row_match = re.search(r"Expected result:\s*\*\*(\d+)\s+rows?\*\*", ref_text, re.IGNORECASE)
        expected_rows = int(row_match.group(1)) if row_match else None
        first_row_match = re.search(r"first row\s+\*\*(.+?)\*\*", ref_text, re.IGNORECASE)
        expected_first_row = first_row_match.group(1).strip() if first_row_match else None
        headers_match = re.search(r"columns?\s+(.+?)\.", ref_text, re.IGNORECASE)
        expected_headers = None
        if headers_match:
            header_text = headers_match.group(1)
            expected_headers = [h.strip() for h in re.split(r"[,|]", header_text) if h.strip()]

        if expected_rows is not None and actual_row_count != expected_rows:
            errors.append(
                f"{video_id} row count mismatch: expected {expected_rows}, got {actual_row_count}"
            )

        if expected_first_row and actual_first_row:
            first_row_str = " ".join(str(c) for c in actual_first_row[:3])
            if expected_first_row.lower() not in first_row_str.lower():
                errors.append(
                    f"{video_id} first row mismatch: expected {expected_first_row!r}, got {actual_first_row!r}"
                )

        if expected_headers and columns:
            if [c.strip() for c in columns] != [c.strip() for c in expected_headers]:
                errors.append(
                    f"{video_id} header mismatch: expected {expected_headers!r}, got {columns!r}"
                )

        return errors

    @staticmethod
    def _column_exists(db_path: Optional[str], table: str, column: str) -> bool:
        if not db_path or not Path(db_path).exists():
            return False
        try:
            with sqlite3.connect(db_path) as conn:
                cur = conn.cursor()
                cur.execute(f"PRAGMA table_info({table})")
                cols = [row[1] for row in cur.fetchall()]
                return column.lower() in (c.lower() for c in cols)
        except Exception:
            return False

    @staticmethod
    def _table_for_column(db_path: Optional[str], column: str) -> Optional[str]:
        """Return the name of a table that contains the given column."""
        if not db_path or not Path(db_path).exists():
            return None
        try:
            with sqlite3.connect(db_path) as conn:
                cur = conn.cursor()
                cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
                for (table,) in cur.fetchall():
                    try:
                        cur.execute(f"PRAGMA table_info({table})")
                        cols = [row[1] for row in cur.fetchall()]
                        if column.lower() in (c.lower() for c in cols):
                            return table
                    except Exception:
                        continue
        except Exception as exc:
            print(f"Warning: could not find table for column {column}: {exc}", file=sys.stderr)
        return None

    def _parse_objective(self, objective: str, db_path: Optional[str] = None, default_table: str = "Orders") -> Optional[Dict[str, Any]]:
        """Map a discovery objective to a recipe type and parameters."""
        lowered = objective.lower()

        # Browse table: open/browse/show/display the X table
        browse_match = re.search(
            r"(?:open|browse|show|display)\s+(?:the\s+)?(\w+)\s+(?:table)", lowered,
        )
        if browse_match:
            return {"type": "browse_table", "table": browse_match.group(1).capitalize()}

        # Sort: sort the X table by Y [direction]
        sort_match = re.search(
            r"sort\s+(?:the\s+)?(\w+)\s+(?:table\s+)?by\s+(?:the\s+)?(\w+)(?:\s+(?:column|header))?",
            lowered,
        )
        if sort_match:
            direction = "desc" if any(w in lowered for w in ("descending", "desc", "largest", "biggest", "highest")) else "asc"
            return {
                "type": "sort_column",
                "table": sort_match.group(1).capitalize(),
                "column": sort_match.group(2).capitalize(),
                "direction": direction,
            }

        # Filter with explicit value: filter X by typing Y into the Z column filter box
        filter_value_match = re.search(
            r"filter\s+(?:the\s+)?(\w+)\s+(?:table\s+)?(?:by\s+(?:typing|entering)\s+)['\"]?(.+?)['\"]?\s+(?:into|in)\s+(?:the\s+)?(\w+)\s+(?:column\s+)?filter",
            lowered,
        )
        if filter_value_match:
            value = filter_value_match.group(2).strip().strip("'\"")
            return {
                "type": "filter_column",
                "table": filter_value_match.group(1).capitalize(),
                "column": filter_value_match.group(3).capitalize(),
                "value": value,
            }

        # Filter without explicit value: filter X by/using/with the Y column filter box
        filter_col_match = re.search(
            r"filter\s+(?:the\s+)?(\w+)\s+(?:table\s+)?(?:by|using|with)\s+(?:the\s+)?(\w+)(?:\s+(?:column))?(?:\s+(?:filter\s+box))?(?:\s+(?:for\s+an\s+exact\s+text\s+match))?",
            lowered,
        )
        if filter_col_match:
            table = filter_col_match.group(1).capitalize()
            column = filter_col_match.group(2).capitalize()
            if not self._table_exists(db_path, table):
                table = default_table
            value = self._first_value(db_path, table, column, context=f"filter {objective[:60]}") or ""
            return {
                "type": "filter_column",
                "table": table,
                "column": column,
                "value": value,
            }

        # Execute query: try to extract the literal SELECT statement first.
        if any(phrase in lowered for phrase in ("execute sql", "select ", "run a query", "query in the execute sql", "join query")):
            table_match = re.search(r"(?:on|from)\s+(?:the\s+)?(\w+)\s+(?:table)?", lowered)
            table = table_match.group(1).capitalize() if table_match else "Orders"

            # Capture a SELECT ... clause up to a stopping word, but keep going
            # through "FROM table" so simple star queries stay complete.
            query_match = re.search(
                r"(SELECT\s+.+?(?:\s+FROM\s+\w+)?)(?:\s+(?:query|in the execute sql|in the|on the|and view|and see)|$)",
                objective,
                re.IGNORECASE,
            )
            if query_match:
                query = query_match.group(1).strip()
                query = re.sub(r"[;.]+$", "", query)
                # Reject placeholder captures like "SELECT query with a WHERE clause".
                looks_valid = (
                    re.search(r"\bFROM\b", query, re.IGNORECASE)
                    or re.search(r"\b(COUNT|SUM|AVG|MIN|MAX)\s*\(", query, re.IGNORECASE)
                )
                if looks_valid:
                    # Aggregates like COUNT(*) still need a source table.
                    if re.search(r"\b(COUNT|SUM|AVG|MIN|MAX)\s*\(", query, re.IGNORECASE) and not re.search(r"\bFROM\b", query, re.IGNORECASE):
                        query = f"{query}\nFROM {table}"
                    return {"type": "execute_query", "query": query}

            if "count" in lowered:
                return {"type": "execute_query", "query": f"SELECT\n    COUNT(*)\nFROM {table}"}

            if "where" in lowered:
                col_match = re.search(
                    r"where\s+clause\s+(?:on\s+(?:the\s+)?)?(?:\w+\s+table\s+)?(\w+)(?:\s+column)?",
                    lowered,
                    re.IGNORECASE,
                )
                column = col_match.group(1).capitalize() if col_match else "status"
                if not self._table_exists(db_path, table):
                    table = default_table
                # Make sure the chosen table actually has the target column.
                if db_path and not self._column_exists(db_path, table, column):
                    table = self._table_for_column(db_path, column) or table
                value_match = re.search(r"equal\s+to\s+['\"]?([\w-]+)", lowered, re.IGNORECASE)
                value = (value_match.group(1).strip() if value_match else self._first_value(db_path, table, column, context=f"where {objective[:60]}")) or ""
                value = value.replace("'", "''")
                return {
                    "type": "execute_query",
                    "query": f"SELECT\n    *\nFROM {table}\nWHERE {column} = '{value}'",
                }

            if "join" in lowered or ("top" in lowered and "spend" in lowered):
                join_table = "Customers" if table.lower() == "orders" else table
                return {
                    "type": "execute_query",
                    "query": (
                        f"SELECT\n"
                        f"    c.name,\n"
                        f"    SUM(o.amount) AS total_spend\n"
                        f"FROM {table.lower()} o\n"
                        f"JOIN {join_table.lower()} c ON o.customer_id = c.customer_id\n"
                        f"GROUP BY c.name\n"
                        f"ORDER BY total_spend DESC"
                    ),
                }

            return {"type": "execute_query", "query": f"SELECT\n    *\nFROM {table}"}

        return None

    @staticmethod
    def _verbalize_sql(query: str) -> str:
        """Convert a SQL query fragment to a single-line spoken form for narration."""
        spoken = query.strip()
        # Collapse whitespace and newlines to single spaces.
        spoken = re.sub(r"\s+", " ", spoken)
        spoken = re.sub(r"\bSELECT\b", "SELECT", spoken, flags=re.IGNORECASE)
        spoken = re.sub(r"\bFROM\b", "FROM", spoken, flags=re.IGNORECASE)
        spoken = re.sub(r"\bWHERE\b", "WHERE", spoken, flags=re.IGNORECASE)
        spoken = re.sub(r"\bORDER BY\b", "ORDER BY", spoken, flags=re.IGNORECASE)
        spoken = re.sub(r"\bGROUP BY\b", "GROUP BY", spoken, flags=re.IGNORECASE)
        spoken = re.sub(r"\bCOUNT\b", "COUNT", spoken, flags=re.IGNORECASE)
        spoken = re.sub(r"\bSUM\b", "SUM", spoken, flags=re.IGNORECASE)
        spoken = re.sub(r"\bAS\b", "AS", spoken, flags=re.IGNORECASE)
        spoken = re.sub(r"\bJOIN\b", "JOIN", spoken, flags=re.IGNORECASE)
        spoken = re.sub(r"\bINNER\b", "INNER", spoken, flags=re.IGNORECASE)
        spoken = re.sub(r"\bLEFT\b", "LEFT", spoken, flags=re.IGNORECASE)
        spoken = re.sub(r"\bRIGHT\b", "RIGHT", spoken, flags=re.IGNORECASE)
        spoken = re.sub(r"\bON\b", "ON", spoken, flags=re.IGNORECASE)
        spoken = re.sub(r"\*", "star", spoken)
        # Keep COUNT star natural (remove parentheses around the star substitution).
        spoken = re.sub(r"\(\s*star\s*\)", " star", spoken)
        return spoken.strip()

    @staticmethod
    def _wait_action() -> Dict[str, Any]:
        return {"type": "wait", "duration": 1.5}

    @staticmethod
    def _click_action(detail: str) -> Dict[str, Any]:
        return {"type": "click", "detail": detail}

    @staticmethod
    def _type_action(text: str, target: Optional[str] = None) -> Dict[str, Any]:
        action: Dict[str, Any] = {"type": "type", "detail": text}
        if target:
            action["target"] = target
        return action

    @staticmethod
    def _key_action(key: str) -> Dict[str, Any]:
        return {"type": "key", "detail": key}

    @staticmethod
    def _verify_action(detail: str) -> Dict[str, Any]:
        return {"type": "verify", "detail": detail}

    @staticmethod
    def _sequence_action(actions: List[Dict[str, Any]]) -> Dict[str, Any]:
        return {"type": "sequence", "actions": actions}

    @staticmethod
    def _parse_demo_to_action(text: str) -> List[Dict[str, Any]]:
        """
        Parse a demo-beat narration sentence into vision-agent action dicts.

        Used as a fallback when a demo beat does not already have an action.
        Examples:
          "We click the Browse Data tab and the table view opens."
            -> [{"type": "click", "detail": "Browse Data tab"}]
          "We type SELECT star FROM Orders and the query appears."
            -> [{"type": "type", "detail": "SELECT * FROM Orders"}]
          "We press F5 and the results grid populates."
            -> [{"type": "key", "detail": "F5"}]
        """
        actions: List[Dict[str, Any]] = []
        lowered = text.lower()

        # Explicit high-level action verbs.
        if re.search(r"\b(apply|activate)\s+(?:the\s+)?filter", lowered):
            actions.append({"type": "click", "detail": "filter box"})

        # SQL-specific demo verbs.
        if re.search(r"\b(open|click)\s+(?:the\s+)?execute\s+sql\s+tab", lowered):
            actions.append({"type": "click", "detail": "Execute SQL tab"})
        if re.search(r"\brun\s+(?:the\s+)?quer", lowered) or re.search(
            r"\bexecute\s+(?:the\s+)?quer", lowered
        ):
            actions.append({"type": "run_query"})

        # Sort: "sort X by Y" or "click the Y column header".
        sort_match = re.search(
            r"\bsort\s+(?:the\s+)?\w+\s+(?:table\s+)?by\s+(?:the\s+)?(\w+)", lowered
        )
        if sort_match:
            actions.append({"type": "click", "detail": f"{sort_match.group(1)} column header"})
        header_match = re.search(r"\bclick\s+(?:the\s+)?(\w+)\s+(?:column\s+)?header", lowered)
        if header_match:
            actions.append({"type": "click", "detail": f"{header_match.group(1)} column header"})

        # Find type_block actions for multi-line SQL.
        block_match = re.search(
            r"\btype\s+(?:the\s+)?(comment\s+block|formatted\s+query|query|sql)(?:\s+with)?\s*[:;]?\s*(.+?)(?=\s+(?:and|then|,)\s|\s*$)",
            lowered,
            re.DOTALL,
        )
        if block_match:
            payload = block_match.group(2).strip().rstrip(",.;:")
            payload = payload.replace("star", "*")
            if "\n" in payload or len(payload) > 80:
                actions.append({"type": "type_block", "text": payload})

        # Find type actions.
        for match in re.finditer(r"\btype\s+(.+?)(?=\s+(?:into|in|and|then|,)\s|\s*$)", lowered):
            detail = match.group(1).strip().rstrip(",.;:")
            # Convert spoken "star" back to the SQL asterisk.
            detail = detail.replace("star", "*")
            if "\n" in detail or len(detail) > 80:
                actions.append({"type": "type_block", "text": detail})
            else:
                actions.append({"type": "type", "detail": detail})

        # Find key actions.
        for match in re.finditer(r"\bpress(?:es)?\s+([a-z0-9_+]+)", lowered, re.IGNORECASE):
            actions.append({"type": "key", "detail": match.group(1)})

        # Find click actions.
        for match in re.finditer(
            r"\bclick(?:s)?\s+(?:the\s+)?(.+?)(?=\s+(?:and|then|,)\s|\s*$)", lowered
        ):
            detail = match.group(1).strip().rstrip(",.;:")
            actions.append({"type": "click", "detail": detail})

        if len(actions) > 1:
            return [{"type": "sequence", "actions": actions}]
        return actions

    @staticmethod
    def _normalize_action(action_spec: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert legacy recipe action specs and coordinate-based actions into the
        vision-agent action format (click/type/key/verify/wait/sequence).
        """
        if not isinstance(action_spec, dict):
            return {"type": "wait", "duration": 1.5}

        action_type = action_spec.get("type")

        if action_type == "browse_table":
            table = action_spec.get("table", "Orders")
            return {
                "type": "sequence",
                "actions": [
                    {"type": "click", "detail": "Browse Data tab"},
                    {"type": "click", "detail": f"{table} table in the table dropdown"},
                ],
            }

        if action_type == "sort_column":
            table = action_spec.get("table", "Orders")
            column = action_spec.get("column", "amount")
            direction = action_spec.get("direction", "asc")
            seq: List[Dict[str, Any]] = [
                {"type": "click", "detail": "Browse Data tab"},
                {"type": "click", "detail": f"{table} table in the table dropdown"},
            ]
            if direction == "desc":
                seq.append({"type": "click", "detail": f"{column} column header"})
            seq.append({"type": "click", "detail": f"{column} column header"})
            return {"type": "sequence", "actions": seq}

        if action_type == "filter_column":
            table = action_spec.get("table", "Orders")
            column = action_spec.get("column", "region")
            value = action_spec.get("value", "")
            return {
                "type": "sequence",
                "actions": [
                    {"type": "click", "detail": "Browse Data tab"},
                    {"type": "click", "detail": f"{table} table in the table dropdown"},
                    {"type": "click", "detail": f"{column} filter box"},
                    {"type": "type", "detail": str(value), "target": f"{column} filter box"},
                    {"type": "key", "detail": "Return"},
                ],
            }

        if action_type == "execute_query":
            query = action_spec.get("query", "SELECT * FROM Orders")
            return {
                "type": "sequence",
                "actions": [
                    {"type": "click", "detail": "Execute SQL tab"},
                    {"type": "click", "detail": "SQL editor text area"},
                    {"type": "type", "detail": query, "target": "SQL editor text area"},
                    {"type": "key", "detail": "F5"},
                ],
            }

        if action_type == "type_block":
            return {
                "type": "type_block",
                "text": action_spec.get("text") or action_spec.get("detail") or "",
            }

        if action_type == "type_segments":
            return {
                "type": "type_segments",
                "segments": action_spec.get("segments") or [],
            }

        if action_type == "run_query":
            return {"type": "run_query"}

        if action_type == "summarize_result_pane":
            return {"type": "summarize_result_pane"}

        # Coordinate-based click/type/key actions from older manifests.
        if action_type == "click":
            detail = action_spec.get("description") or action_spec.get("detail") or "UI element"
            target = action_spec.get("target")
            if not isinstance(target, dict):
                target = {}
            return {"type": "click", "detail": detail, "target": target}

        if action_type == "type":
            detail = action_spec.get("detail") or action_spec.get("text") or ""
            target = action_spec.get("target")
            if isinstance(target, dict):
                # Keep legacy coordinate dictionaries for the graph edge, but add
                # a human description for the vision agent if it is missing.
                if not target.get("description"):
                    target = {**target, "description": action_spec.get("description", "input field")}
            elif isinstance(target, str):
                # Vision-agent format uses a string target; keep it.
                pass
            else:
                target = action_spec.get("description", "input field")
            return {"type": "type", "detail": detail, "target": target}

        if action_type == "key":
            detail = action_spec.get("detail") or action_spec.get("text") or "Return"
            return {"type": "key", "detail": detail}

        if action_type == "wait":
            return {"type": "wait", "duration": action_spec.get("duration", 1.5)}

        # Already in vision-agent format.
        return action_spec

    def _build_script_beats(
        self,
        video: Any,
        parsed: Dict[str, Any],
        env_map: Optional[Dict[str, Any]] = None,
    ) -> List[ScriptBeat]:
        """Build SQL Essentials-quality beats from a parsed objective.

        Each non-wait beat gets a vision-agent action dict in ``beat.action`` so
        the discovery harness can drive the UI dynamically instead of using
        hard-coded coordinate recipes.
        """
        exercise = video.exercise_artifact or {}
        db_path = exercise.get("db_path")
        table = parsed.get("table", "Orders")
        facts = self._db_facts(db_path, table)
        row_count = facts.get("row_count", 0)
        columns = facts.get("columns", [])
        columns_text = ", ".join(columns) if columns else "the columns"
        tier = getattr(video, "format_tier", "short")
        compact = tier in {"micro"}

        # Ground against the scout pass if available.
        env_tables = []
        env_columns: Dict[str, List[str]] = {}
        env_row_counts: Dict[str, int] = {}
        env_default_table: Optional[str] = None
        env_active_tab: Optional[str] = None
        if env_map:
            env_tables = env_map.get("tables", []) or []
            env_columns = env_map.get("columns", {}) or {}
            env_row_counts = env_map.get("row_counts", {}) or {}
            ui = env_map.get("ui") or {}
            env_default_table = ui.get("browse_data_default_table")
            env_active_tab = ui.get("active_tab")
            # Prefer observed facts over raw DB facts when they conflict.
            if table in env_row_counts:
                row_count = env_row_counts[table]
            if table in env_columns:
                columns = env_columns[table]
                columns_text = ", ".join(columns)

        rows_word = "rows" if row_count != 1 else "row"
        browse_data_already_active = env_active_tab and "browse data" in env_active_tab.lower()
        target_table_already_open = env_default_table and env_default_table.lower() == table.lower()

        def _validation(text: str) -> str:
            """Ensure validation beats are 10-15 words and end with a period."""
            text = text.strip().rstrip(".")
            wc = len(text.split())
            pads = [
                ", confirming the result is correct",
                ", which confirms the operation succeeded",
                ", verifying the outcome matches our goal",
            ]
            # Ceiling: pads grow by ~3 words each iteration; wc starts >=0, so <=4 iterations.
            while wc < 10:
                text = text + pads[(wc // 3) % len(pads)]
                wc = len(text.split())
            if wc > 15:
                text = " ".join(text.split()[:15]).rstrip(",;:")
            return text + "."

        beats: List[ScriptBeat] = []

        if parsed["type"] == "browse_table":
            if compact:
                demo_actions: List[Dict[str, Any]] = []
                demo_texts: List[str] = []
                if not browse_data_already_active:
                    demo_actions.append(self._click_action("Browse Data tab"))
                    demo_texts.append("We click Browse Data")
                if not target_table_already_open:
                    demo_actions.append(self._click_action(f"{table} table in the table dropdown"))
                    demo_texts.append(f"and open the {table} table")
                if not demo_actions:
                    demo_actions = [self._wait_action()]
                    demo_texts.append(f"The {table} table is already open")
                demo_action = (
                    self._sequence_action(demo_actions)
                    if len(demo_actions) > 1
                    else demo_actions[0]
                )
                beats = [
                    ScriptBeat(
                        beat_id="beat_001",
                        kind="opening",
                        text=f"In this video, we will open the {table} table.",
                        action=self._wait_action(),
                    ),
                    ScriptBeat(
                        beat_id="beat_002",
                        kind="demo",
                        text=" ".join(demo_texts).strip() + ".",
                        action=demo_action,
                    ),
                    ScriptBeat(
                        beat_id="beat_003",
                        kind="validation",
                        text=_validation(
                            f"We see {len(columns)} columns and {row_count} {rows_word} in the grid"
                        ),
                        action=self._verify_action(
                            f"the {table} table is visible with its rows and columns"
                        ),
                    ),
                    ScriptBeat(
                        beat_id="beat_004",
                        kind="close",
                        text=f"We have opened the {table} table and confirmed its structure.",
                        action=self._wait_action(),
                    ),
                ]
            else:
                demo_actions = []
                demo_texts = []
                if not browse_data_already_active:
                    demo_actions.append(self._click_action("Browse Data tab"))
                    demo_texts.append("We click the Browse Data tab")
                if not target_table_already_open:
                    demo_actions.append(self._click_action(f"{table} table in the table dropdown"))
                    if demo_texts:
                        demo_texts.append(f"and select {table}")
                    else:
                        demo_texts.append(f"We select {table}")
                if not demo_actions:
                    demo_actions = [self._wait_action()]
                    demo_texts.append(f"The {table} table is already open")
                demo_action = (
                    self._sequence_action(demo_actions)
                    if len(demo_actions) > 1
                    else demo_actions[0]
                )
                beats = [
                    ScriptBeat(
                        beat_id="beat_001",
                        kind="opening",
                        text=f"In this video, we will open the {table} table in the Browse Data tab.",
                        action=self._wait_action(),
                    ),
                    ScriptBeat(
                        beat_id="beat_002",
                        kind="demo",
                        text=" ".join(demo_texts).strip() + ".",
                        action=demo_action,
                    ),
                    ScriptBeat(
                        beat_id="beat_003",
                        kind="validation",
                        text=_validation(
                            f"We see {len(columns)} columns in the grid: {columns_text}"
                        ),
                        action=self._verify_action(
                            f"the {table} table is visible with columns {columns_text}"
                        ),
                    ),
                    ScriptBeat(
                        beat_id="beat_004",
                        kind="close",
                        text=f"We have opened the {table} table and confirmed its structure.",
                        action=self._wait_action(),
                    ),
                ]

        elif parsed["type"] == "sort_column":
            column = parsed.get("column", "amount")
            direction = parsed.get("direction", "asc")
            direction_text = "ascending" if direction == "asc" else "descending"
            top_value = self._top_value(db_path, table, column, direction) or "the first value"
            if compact:
                beats = [
                    ScriptBeat(
                        beat_id="beat_001",
                        kind="opening",
                        text=f"In this video, we will sort by {column} in {direction_text} order.",
                        action=self._wait_action(),
                    ),
                    ScriptBeat(
                        beat_id="beat_002",
                        kind="demo",
                        text=f"We open the {table} table and click the {column} header.",
                        action=self._sequence_action([
                            self._click_action("Browse Data tab"),
                            self._click_action(f"{table} table in the table dropdown"),
                            self._click_action(f"{column} column header"),
                        ]),
                    ),
                    ScriptBeat(
                        beat_id="beat_003",
                        kind="validation",
                        text=_validation(
                            f"We see the {direction_text} sort with {top_value} at the top"
                        ),
                        action=self._verify_action(
                            f"the {table} rows are sorted by {column} in {direction_text} order"
                        ),
                    ),
                    ScriptBeat(
                        beat_id="beat_004",
                        kind="close",
                        text=f"We have sorted the {table} table by {column} in {direction_text} order.",
                        action=self._wait_action(),
                    ),
                ]
            else:
                beats = [
                    ScriptBeat(
                        beat_id="beat_001",
                        kind="opening",
                        text=f"In this video, we will sort the {table} table by {column} in {direction_text} order.",
                        action=self._wait_action(),
                    ),
                    ScriptBeat(
                        beat_id="beat_002",
                        kind="demo",
                        text=f"We open the {table} table and the rows appear.",
                        action=self._sequence_action([
                            self._click_action("Browse Data tab"),
                            self._click_action(f"{table} table in the table dropdown"),
                        ]),
                    ),
                    ScriptBeat(
                        beat_id="beat_003",
                        kind="demo",
                        text=f"We click the {column} column header and the rows reorder.",
                        action=self._click_action(f"{column} column header"),
                    ),
                    ScriptBeat(
                        beat_id="beat_004",
                        kind="validation",
                        text=_validation(
                            f"We see the {direction_text} sort, with {top_value} at the top"
                        ),
                        action=self._verify_action(
                            f"the {table} rows are sorted by {column} in {direction_text} order"
                        ),
                    ),
                    ScriptBeat(
                        beat_id="beat_005",
                        kind="close",
                        text=f"We have sorted the {table} table by {column} in {direction_text} order.",
                        action=self._wait_action(),
                    ),
                ]

        elif parsed["type"] == "filter_column":
            column = parsed.get("column", "region")
            value = parsed.get("value", "")
            filtered_count = self._filtered_count(db_path, table, column, value) if value else 0
            filtered_word = "rows" if filtered_count != 1 else "row"
            if compact:
                beats = [
                    ScriptBeat(
                        beat_id="beat_001",
                        kind="opening",
                        text=f"In this video, we will filter the {table} by {column}.",
                        action=self._wait_action(),
                    ),
                    ScriptBeat(
                        beat_id="beat_002",
                        kind="demo",
                        text=f"We click the filter box, type {value}, and press Return.",
                        action=self._sequence_action([
                            self._click_action("Browse Data tab"),
                            self._click_action(f"{table} table in the table dropdown"),
                            self._click_action(f"{column} filter box"),
                            self._type_action(str(value), target=f"{column} filter box"),
                            self._key_action("Return"),
                        ]),
                    ),
                    ScriptBeat(
                        beat_id="beat_003",
                        kind="validation",
                        text=_validation(
                            f"We see exactly {filtered_count} {filtered_word} where {column} equals {value}"
                        ),
                        action=self._verify_action(
                            f"only {filtered_count} {filtered_word} with {column} equal to {value} are visible"
                        ),
                    ),
                    ScriptBeat(
                        beat_id="beat_004",
                        kind="close",
                        text=f"We have filtered the {table} by {column} and isolated rows.",
                        action=self._wait_action(),
                    ),
                ]
            else:
                beats = [
                    ScriptBeat(
                        beat_id="beat_001",
                        kind="opening",
                        text=f"In this video, we will filter the {table} table to show only {value} rows.",
                        action=self._wait_action(),
                    ),
                    ScriptBeat(
                        beat_id="beat_002",
                        kind="demo",
                        text=f"We open the {table} table and the rows appear.",
                        action=self._sequence_action([
                            self._click_action("Browse Data tab"),
                            self._click_action(f"{table} table in the table dropdown"),
                        ]),
                    ),
                    ScriptBeat(
                        beat_id="beat_003",
                        kind="demo",
                        text=f"We click the {column} filter box and type {value}.",
                        action=self._click_action(f"{column} filter box"),
                    ),
                    ScriptBeat(
                        beat_id="beat_004",
                        kind="demo",
                        text="We press Return and the rows filter.",
                        action=self._type_action(str(value), target=f"{column} filter box"),
                    ),
                    ScriptBeat(
                        beat_id="beat_005",
                        kind="validation",
                        text=_validation(
                            f"We see {filtered_count} {filtered_word}, all with {column} equal to {value}"
                        ),
                        action=self._verify_action(
                            f"only {filtered_count} {filtered_word} with {column} equal to {value} are visible"
                        ),
                    ),
                    ScriptBeat(
                        beat_id="beat_006",
                        kind="close",
                        text=f"We have filtered the {table} table by {column} and narrowed the view.",
                        action=self._wait_action(),
                    ),
                ]

            # Compact the filter short version: the type action needs a Return keypress.
            if not compact:
                # Replace the Return press with a key action after the type beat.
                # Current beat_004 is type; add an explicit Return key beat.
                beats.insert(
                    5,
                    ScriptBeat(
                        beat_id="beat_005_key",
                        kind="demo",
                        text="We press Return and the rows filter.",
                        action=self._key_action("Return"),
                    ),
                )
                # Renumber beats.
                for i, beat in enumerate(beats, start=1):
                    beat.beat_id = f"beat_{i:03d}"

        elif parsed["type"] == "execute_query":
            query = parsed.get("query", "SELECT * FROM Orders")
            spoken_query = self._verbalize_sql(query)
            spoken_words = spoken_query.split()
            short_type_text = (
                f"We type {spoken_query} and the SQL appears."
                if len(spoken_words) <= 6
                else "We type the formatted query and the SQL appears."
            )
            if compact:
                beats = [
                    ScriptBeat(
                        beat_id="beat_001",
                        kind="opening",
                        text="In this video, we will run our first SELECT query.",
                        action=self._wait_action(),
                    ),
                    ScriptBeat(
                        beat_id="beat_002",
                        kind="demo",
                        text="We open Execute SQL, type the query, and run it.",
                        action=self._sequence_action([
                            self._click_action("Execute SQL tab"),
                            self._click_action("SQL editor text area"),
                            self._type_action(query, target="SQL editor text area"),
                            self._key_action("F5"),
                        ]),
                    ),
                    ScriptBeat(
                        beat_id="beat_003",
                        kind="validation",
                        text=_validation(
                            f"We see the results grid populate with the returned {rows_word}"
                        ),
                        action=self._verify_action(
                            "the Execute SQL tab shows a populated results grid below the query"
                        ),
                    ),
                    ScriptBeat(
                        beat_id="beat_004",
                        kind="close",
                        text=f"We have run the query and retrieved the {row_count} {rows_word}.",
                        action=self._wait_action(),
                    ),
                ]
            else:
                beats = [
                    ScriptBeat(
                        beat_id="beat_001",
                        kind="opening",
                        text="In this video, we will run a SELECT query in the Execute SQL tab.",
                        action=self._wait_action(),
                    ),
                    ScriptBeat(
                        beat_id="beat_002",
                        kind="demo",
                        text="We click the Execute SQL tab and the editor appears.",
                        action=self._click_action("Execute SQL tab"),
                    ),
                    ScriptBeat(
                        beat_id="beat_003",
                        kind="demo",
                        text="We click the SQL editor and type the formatted query.",
                        action=self._sequence_action([
                            self._click_action("SQL editor text area"),
                            self._type_action(query, target="SQL editor text area"),
                        ]),
                    ),
                    ScriptBeat(
                        beat_id="beat_004",
                        kind="demo",
                        text="We press F5 and the results grid populates.",
                        action=self._key_action("F5"),
                    ),
                    ScriptBeat(
                        beat_id="beat_005",
                        kind="validation",
                        text=_validation(
                            f"We see {row_count} {rows_word} returned, confirming the query executed successfully"
                        ),
                        action=self._verify_action(
                            "the Execute SQL tab shows a populated results grid below the query"
                        ),
                    ),
                    ScriptBeat(
                        beat_id="beat_006",
                        kind="close",
                        text=f"We have run our SELECT query and retrieved all {row_count} {rows_word}.",
                        action=self._wait_action(),
                    ),
                ]

        return beats

    @staticmethod
    def _make_comment_block(description: str) -> str:
        """Return a SQL Standard comment block for typing into the editor."""
        today = datetime.now().strftime("%Y-%m-%d")
        return (
            f"/*\n"
            f"Created By: WSDA Student\n"
            f"Create Date: {today}\n"
            f"Description: {description}\n"
            f"*/"
        )

    def _build_sql_script_beats(
        self,
        video: Any,
        env_map: Optional[Dict[str, Any]] = None,
    ) -> List[ScriptBeat]:
        """
        Deterministic SQL-arc script for the C4 Phase 2 chapter-4 videos.

        Each video is a self-contained SQL demo against the WSDA Music Customer
        table. The comment block and query are typed as SEGMENTS so the learner
        sees the query assemble clause-by-clause while hearing why each part is
        being typed.
        """
        exercise = video.exercise_artifact or {}
        db_path = exercise.get("db_path")
        video_id = getattr(video, "video_id", "video_1_1")

        planned = (
            video.planned_queries[0]
            if video.planned_queries
            else "SELECT FirstName, LastName, Email FROM Customer;"
        )
        query_results = (env_map or {}).get("query_results", {}) or {}
        ground = query_results.get(planned) or {}

        # Fallback to live sqlite3 if the scout pass did not include this query.
        if not ground and db_path and Path(db_path).exists():
            try:
                with sqlite3.connect(db_path) as conn:
                    cur = conn.cursor()
                    cur.execute(planned)
                    rows = cur.fetchall()
                    ground = {
                        "columns": [desc[0] for desc in cur.description] if cur.description else [],
                        "row_count": len(rows),
                        "first_rows": [list(row) for row in rows[:5]],
                    }
            except Exception as exc:
                print(f"Warning: could not ground pilot query: {exc}", file=sys.stderr)
                ground = {}

        row_count = ground.get("row_count")
        columns = ground.get("columns", [])
        columns_text = ", ".join(columns) if columns else "FirstName, LastName, Email"
        rows_word = "rows" if row_count != 1 else "row"
        first_rows = ground.get("first_rows", [])

        def _validation(text: str) -> str:
            """Ensure validation beats are at least 15 words and end with a period."""
            text = text.strip().rstrip(".")
            wc = len(text.split())
            pads = [
                ", confirming the result is correct",
                ", which confirms the operation succeeded",
                ", verifying the outcome matches our goal",
            ]
            # Ceiling: pads grow by ~3 words each iteration; wc starts >=0, so <=5 iterations.
            while wc < 15:
                text = text + pads[(wc // 3) % len(pads)]
                wc = len(text.split())
            return text + "."

        def _full_block(comment: str, query: str) -> str:
            return f"{comment}\n{query}"

        def _segment_action(text: str) -> Dict[str, Any]:
            return {
                "type": "type_segments",
                "segments": [{"text": text}],
            }

        def _segment_beat(beat_id: str, text: str, narration: str) -> ScriptBeat:
            action = _segment_action(text)
            return ScriptBeat(
                beat_id=beat_id,
                kind="demo",
                text=narration,
                action=action,
                planned_duration=LessonBuilder._planned_action_seconds(action),
            )

        beats: List[ScriptBeat] = []

        if video_id == "video_1_1":
            comment = self._make_comment_block("Customer contact list for management")
            select_clause = "\nSELECT\n  FirstName,\n  LastName,\n  Email"
            from_clause = "\nFROM Customer;"
            beats = [
                ScriptBeat(
                    beat_id="beat_001",
                    kind="opening",
                    text=(
                        "In this video, we will write our first SELECT query to pull a clean "
                        "customer contact list for WSDA Music management. The goal is to return "
                        "only the columns we need from the Customer table, so the result is "
                        "focused and immediately useful."
                    ),
                    action=self._wait_action(),
                ),
                ScriptBeat(
                    beat_id="beat_002",
                    kind="state",
                    text=(
                        "The editor is empty and the result pane below it is blank. "
                        "When we finish, the editor will hold a comment block followed by a "
                        "formatted SELECT statement, and the result pane will show the customer "
                        "contact list."
                    ),
                    action=self._wait_action(),
                ),
                ScriptBeat(
                    beat_id="beat_003",
                    kind="explain",
                    text=(
                        "SELECT is the SQL command that chooses which columns to return, and "
                        "FROM chooses which table holds those columns. Together they form the "
                        "simplest useful query pattern: ask for specific data from one table. "
                        "This keeps the result small and fast because the database does not waste "
                        "time returning columns we do not need."
                    ),
                    action=self._wait_action(),
                ),
                ScriptBeat(
                    beat_id="beat_004",
                    kind="state",
                    text=(
                        "Right now the editor is empty and the result pane below it is blank. "
                        "When we finish, the editor will hold a comment block followed by a "
                        "formatted SELECT statement, and the result pane will show the customer "
                        "contact list."
                    ),
                    action=self._wait_action(),
                ),
                _segment_beat(
                    "beat_005",
                    comment,
                    "We type a comment header that records who created the query, when it was written, and what problem it solves.",
                ),
                _segment_beat(
                    "beat_006",
                    select_clause,
                    "We type the SELECT clause, listing the columns FirstName, LastName, and Email.",
                ),
                _segment_beat(
                    "beat_007",
                    from_clause,
                    "We type FROM Customer to name the data source for the columns.",
                ),
                ScriptBeat(
                    beat_id="beat_008",
                    kind="demo",
                    text="We run the query and the result pane fills with the contact list.",
                    action={"type": "run_query"},
                ),
                ScriptBeat(
                    beat_id="beat_009",
                    kind="validation",
                    text=_validation(
                        f"We see {row_count} {rows_word} returned with FirstName, LastName, "
                        f"and Email for each customer, confirming the contact list is complete "
                        f"and the query ran exactly as intended"
                        if row_count is not None else "We see the contact list returned with the requested columns, confirming the query succeeded and the result is complete"
                    ),
                    action=self._verify_action(
                        "the Execute SQL tab shows a populated results grid below the query"
                    ),
                ),
                ScriptBeat(
                    beat_id="beat_010",
                    kind="explain",
                    text=(
                        "The result pane shows the complete customer contact list, with each "
                        "customer record appearing exactly once in the order it is stored. "
                        "Management now has a reliable set of first names, last names, and email "
                        "addresses without extra columns cluttering the view. Because the database "
                        "returned only the columns we requested, the output is compact and fast to "
                        "scan."
                    ),
                    action=self._wait_action(),
                ),
                ScriptBeat(
                    beat_id="beat_011",
                    kind="explain",
                    text=(
                        "Adding a comment header at the top of the query is a professional habit. "
                        "It records who created the query, when it was written, and what problem it "
                        "solves, which helps anyone who opens the file later. Good comments turn a "
                        "quick one-off query into documentation that the whole team can trust and "
                        "maintain."
                    ),
                    action=self._wait_action(),
                ),
                ScriptBeat(
                    beat_id="beat_012",
                    kind="close",
                    text=(
                        "We have written our first SELECT query and pulled a complete customer "
                        "contact list. The query returned only the columns we needed and included "
                        "a clear comment header. Next, we will make those column headers friendlier "
                        "for management reports by using aliases."
                    ),
                    action=self._wait_action(),
                ),
            ]

        elif video_id == "video_1_2":
            comment = self._make_comment_block("Readable customer contact headers")
            select_clause = (
                "\nSELECT\n"
                "  FirstName AS \"First Name\",\n"
                "  LastName AS \"Last Name\",\n"
                "  Email AS \"Email Address\""
            )
            from_clause = "\nFROM Customer;"
            beats = [
                ScriptBeat(
                    beat_id="beat_001",
                    kind="opening",
                    text=(
                        "In this video, we will use the AS keyword to give our query results "
                        "readable column headers that match management language. Last lesson we "
                        "pulled the raw contact list; now we want friendly labels like First Name "
                        "instead of FirstName in the report."
                    ),
                    action=self._wait_action(),
                ),
                ScriptBeat(
                    beat_id="beat_002",
                    kind="state",
                    text=(
                        "Our previous query sits above, commented out. The result pane below "
                        "still displays the raw headers. We can add a new query below the history "
                        "that replaces each raw column name with a friendly alias using the AS keyword."
                    ),
                    action=self._wait_action(),
                ),
                ScriptBeat(
                    beat_id="beat_003",
                    kind="explain",
                    text=(
                        "The AS keyword gives a column an alias, which is the name that appears "
                        "in the result header instead of the raw column name. Aliases make reports "
                        "easier to read without changing the underlying data or the table structure. "
                        "They are especially useful when managers want headers in plain English "
                        "while the database keeps its original column names."
                    ),
                    action=self._wait_action(),
                ),
                ScriptBeat(
                    beat_id="beat_004",
                    kind="state",
                    text=(
                        "Right now the result headers show the raw column names. After we run "
                        "the aliased query, those same three headers will display with spaces, "
                        "while the rows underneath stay exactly the same. This is a before-and-after "
                        "change we can verify at a glance."
                    ),
                    action=self._wait_action(),
                ),
                _segment_beat(
                    "beat_005",
                    comment,
                    "We type a comment header that explains this query makes the column headers readable.",
                ),
                _segment_beat(
                    "beat_006",
                    select_clause,
                    "We type the SELECT clause with AS aliases so First Name, Last Name, and Email Address appear as headers.",
                ),
                _segment_beat(
                    "beat_007",
                    from_clause,
                    "We type FROM Customer to keep the data source unchanged.",
                ),
                ScriptBeat(
                    beat_id="beat_008",
                    kind="demo",
                    text="We run the query and the result pane fills with the aliased list.",
                    action={"type": "run_query"},
                ),
                ScriptBeat(
                    beat_id="beat_009",
                    kind="validation",
                    text=_validation(
                        f"We see {row_count} {rows_word} returned with headers reading First Name, "
                        f"Last Name, and Email Address, confirming the AS aliases took effect "
                        f"and the report is readable for management"
                        if row_count is not None else "We see the aliased headers reading First Name, Last Name, and Email Address, confirming the query worked and the labels are readable"
                    ),
                    action=self._verify_action(
                        "the Execute SQL tab shows a populated results grid with aliased headers"
                    ),
                ),
                ScriptBeat(
                    beat_id="beat_010",
                    kind="explain",
                    text=(
                        "The headers now show the friendly aliases, giving management the "
                        "readable view they requested. The underlying values did not change; only "
                        "the labels at the top of each column did. This means the same query can "
                        "serve both technical users who know the original schema and managers who "
                        "need a polished report for presentations."
                    ),
                    action=self._wait_action(),
                ),
                ScriptBeat(
                    beat_id="beat_011",
                    kind="explain",
                    text=(
                        "Aliases also help when a column name is long or unclear. A short, "
                        "descriptive alias keeps the header visible in a narrow spreadsheet column "
                        "and makes formulas easier to write. Using spaces in quoted aliases is "
                        "common in reports, but the quotes are required so the database recognizes "
                        "the whole multi-word header."
                    ),
                    action=self._wait_action(),
                ),
                ScriptBeat(
                    beat_id="beat_012",
                    kind="close",
                    text=(
                        "We have used the AS keyword to create readable headers for the customer "
                        "contact list. The result now speaks management's language without changing "
                        "any data. Next, we will sort that list alphabetically by last name using "
                        "ORDER BY."
                    ),
                    action=self._wait_action(),
                ),
            ]

        elif video_id == "video_1_3":
            comment = self._make_comment_block("Customer contact list sorted by last name")
            select_clause = (
                "\nSELECT\n"
                "  FirstName AS \"First Name\",\n"
                "  LastName AS \"Last Name\",\n"
                "  Email AS \"Email Address\""
            )
            from_clause = "\nFROM Customer"
            order_clause = "\nORDER BY LastName;"
            top_last = ""
            if db_path and Path(db_path).exists():
                top_last = self._top_value(db_path, "Customer", "LastName", "asc") or ""
            beats = [
                ScriptBeat(
                    beat_id="beat_001",
                    kind="opening",
                    text=(
                        "In this video, we will sort the customer contact list alphabetically "
                        "by last name using the ORDER BY clause. Last lesson we made the headers "
                        "readable with aliases; now we control the order in which the rows appear "
                        "in the result pane."
                    ),
                    action=self._wait_action(),
                ),
                ScriptBeat(
                    beat_id="beat_002",
                    kind="state",
                    text=(
                        "Our previous queries sit above, commented out. The result pane below "
                        "shows the readable aliased headers with rows in their stored order. We can "
                        "add ORDER BY to tell the database exactly how to arrange the returned rows."
                    ),
                    action=self._wait_action(),
                ),
                ScriptBeat(
                    beat_id="beat_003",
                    kind="explain",
                    text=(
                        "ORDER BY tells the database how to sort the returned rows. It does "
                        "not change which rows are selected or the columns that come back; it only "
                        "rearranges the order in which they appear. By default, text columns sort "
                        "alphabetically from A to Z, which is what we want for a contact list that "
                        "managers can scan quickly."
                    ),
                    action=self._wait_action(),
                ),
                ScriptBeat(
                    beat_id="beat_004",
                    kind="state",
                    text=(
                        "Right now the rows appear in the order they are stored in the table. "
                        "After we add ORDER BY LastName, the same rows will reorder so the earliest "
                        "last name appears at the top and the rest follow alphabetically down the "
                        "result pane."
                    ),
                    action=self._wait_action(),
                ),
                _segment_beat(
                    "beat_005",
                    comment,
                    "We type a comment header that says this query sorts the contact list by last name.",
                ),
                _segment_beat(
                    "beat_006",
                    select_clause,
                    "We type the SELECT clause again with the friendly aliases so the headers stay readable.",
                ),
                _segment_beat(
                    "beat_007",
                    from_clause,
                    "We type FROM Customer so the data still comes from the same table.",
                ),
                _segment_beat(
                    "beat_008",
                    order_clause,
                    "We type ORDER BY LastName so the database returns the rows in A-to-Z order.",
                ),
                ScriptBeat(
                    beat_id="beat_009",
                    kind="demo",
                    text="We run the query and the rows reorder alphabetically in the result pane.",
                    action={"type": "run_query"},
                ),
                ScriptBeat(
                    beat_id="beat_010",
                    kind="validation",
                    text=_validation(
                        f"We see {row_count} {rows_word} returned, with {top_last} at the top "
                        f"of the Last Name column, confirming the ascending alphabetical sort is "
                        f"active and the rows follow A-to-Z order"
                        if row_count and top_last else "We see the rows arranged alphabetically by last name, confirming the ascending sort is active and the result is ready to scan"
                    ),
                    action=self._verify_action(
                        "the result pane rows are sorted by LastName in ascending order"
                    ),
                ),
                ScriptBeat(
                    beat_id="beat_011",
                    kind="explain",
                    text=(
                        "The rows are now arranged alphabetically by last name, with the "
                        "earliest last name at the top of the list. This makes it easy to scan "
                        "contacts from A to Z when looking for a specific person. The selected "
                        "columns and aliases did not change; only the sequence of rows changed to "
                        "match the ORDER BY clause."
                    ),
                    action=self._wait_action(),
                ),
                ScriptBeat(
                    beat_id="beat_012",
                    kind="explain",
                    text=(
                        "ORDER BY belongs at the end of the SELECT statement, after the column "
                        "list and FROM clause. Putting it last is a readability convention that "
                        "helps the team see the sort rule at a glance. It also makes the query "
                        "easier to edit when we later add filtering or limiting clauses before the "
                        "final sort."
                    ),
                    action=self._wait_action(),
                ),
                ScriptBeat(
                    beat_id="beat_013",
                    kind="close",
                    text=(
                        "We have sorted the customer contact list alphabetically by last name "
                        "with ORDER BY. The rows now appear in A-to-Z order, ready for scanning. "
                        "Next, we will limit the result to a small preview using LIMIT."
                    ),
                    action=self._wait_action(),
                ),
            ]

        elif video_id == "video_1_4":
            comment = self._make_comment_block("Preview of customer contacts")
            select_clause = (
                "\nSELECT\n"
                "  FirstName AS \"First Name\",\n"
                "  LastName AS \"Last Name\",\n"
                "  Email AS \"Email Address\""
            )
            from_clause = "\nFROM Customer"
            order_clause = "\nORDER BY LastName"
            limit_clause = "\nLIMIT 5;"
            beats = [
                ScriptBeat(
                    beat_id="beat_001",
                    kind="opening",
                    text=(
                        "In this video, we will limit the sorted customer contact list to a "
                        "small preview using the LIMIT clause. Last lesson we sorted the full "
                        "list alphabetically; now management only needs the first few rows for "
                        "a quick look."
                    ),
                    action=self._wait_action(),
                ),
                ScriptBeat(
                    beat_id="beat_002",
                    kind="state",
                    text=(
                        "Our previous queries sit above, commented out. The result pane below "
                        "shows the full alphabetical list by last name. We can add one more clause "
                        "at the end of the statement to control exactly how many rows the database returns."
                    ),
                    action=self._wait_action(),
                ),
                ScriptBeat(
                    beat_id="beat_003",
                    kind="explain",
                    text=(
                        "LIMIT returns only the requested number of rows from the result set. "
                        "It is useful when a full result is larger than we need, such as when a "
                        "manager asks for a quick preview before running the full report. The "
                        "database still sorts the rows first, because LIMIT comes after ORDER BY "
                        "in the statement."
                    ),
                    action=self._wait_action(),
                ),
                ScriptBeat(
                    beat_id="beat_004",
                    kind="state",
                    text=(
                        "Right now the result pane shows the full alphabetical list. After we "
                        "add a LIMIT clause, only the first few rows will remain visible, while "
                        "the sort order stays the same. This turns a long report into a manageable "
                        "preview."
                    ),
                    action=self._wait_action(),
                ),
                _segment_beat(
                    "beat_005",
                    comment,
                    "We type a comment header that says this query is a preview of customer contacts.",
                ),
                _segment_beat(
                    "beat_006",
                    select_clause,
                    "We type the SELECT clause with the friendly aliases so the headers stay readable.",
                ),
                _segment_beat(
                    "beat_007",
                    from_clause,
                    "We type FROM Customer so the data still comes from the same table.",
                ),
                _segment_beat(
                    "beat_008",
                    order_clause,
                    "We type ORDER BY LastName so the rows stay sorted alphabetically.",
                ),
                _segment_beat(
                    "beat_009",
                    limit_clause,
                    "We type LIMIT 5 so only the first five rows appear in the preview.",
                ),
                ScriptBeat(
                    beat_id="beat_010",
                    kind="demo",
                    text="We run the query and the result pane shows only the preview rows.",
                    action={"type": "run_query"},
                ),
                ScriptBeat(
                    beat_id="beat_011",
                    kind="validation",
                    text=_validation(
                        f"We see exactly {row_count} {rows_word} returned, confirming the LIMIT "
                        f"clause trimmed the sorted result to the requested preview size. The "
                        f"rows still appear in alphabetical order, so the preview is representative"
                        if row_count is not None else "We see only the preview rows returned, confirming the LIMIT clause trimmed the sorted result and the order is preserved"
                    ),
                    action=self._verify_action(
                        "the result pane shows exactly five rows from the sorted contact list"
                    ),
                ),
                ScriptBeat(
                    beat_id="beat_012",
                    kind="explain",
                    text=(
                        "LIMIT trimmed the sorted result to a small preview, and the rows "
                        "still appear in alphabetical order by last name. The database stopped "
                        "after the requested number of rows, so the result loads faster and fits "
                        "neatly on one screen. This is ideal for summaries and quick checks "
                        "before a manager requests the full data set."
                    ),
                    action=self._wait_action(),
                ),
                ScriptBeat(
                    beat_id="beat_013",
                    kind="explain",
                    text=(
                        "LIMIT always comes after ORDER BY in a SELECT statement. If it came "
                        "before the sort, the database would trim the rows first and then sort, "
                        "which could return the wrong rows for the preview. Keeping the order "
                        "SELECT, FROM, ORDER BY, LIMIT makes the query predictable and easy for "
                        "the team to maintain."
                    ),
                    action=self._wait_action(),
                ),
                ScriptBeat(
                    beat_id="beat_014",
                    kind="close",
                    text=(
                        "We have limited the result set with LIMIT, and the preview remains "
                        "sorted alphabetically by last name. Next, we will recap the chapter with "
                        "one clean, well-documented query that combines comments, aliases, ORDER "
                        "BY, and LIMIT."
                    ),
                    action=self._wait_action(),
                ),
            ]

        elif video_id == "video_1_5":
            comment = self._make_comment_block("Clean, documented customer contact preview")
            select_clause = (
                "\nSELECT\n"
                "  FirstName AS \"First Name\",\n"
                "  LastName AS \"Last Name\",\n"
                "  Email AS \"Email Address\""
            )
            from_clause = "\nFROM Customer"
            order_clause = "\nORDER BY LastName"
            limit_clause = "\nLIMIT 5;"
            beats = [
                ScriptBeat(
                    beat_id="beat_001",
                    kind="opening",
                    text=(
                        "In this video, we will combine comment headers, aliases, ORDER BY, "
                        "and LIMIT into one clean query. The previous lessons built each skill "
                        "step by step; now we use them together in a single professional "
                        "statement that is ready to share."
                    ),
                    action=self._wait_action(),
                ),
                ScriptBeat(
                    beat_id="beat_002",
                    kind="state",
                    text=(
                        "Our previous queries sit above, commented out. We can build the final "
                        "chapter query step by step below them: comment block first, then SELECT "
                        "with aliases, FROM, ORDER BY, and finally LIMIT. Each clause has a "
                        "specific job that we have already practiced."
                    ),
                    action=self._wait_action(),
                ),
                ScriptBeat(
                    beat_id="beat_003",
                    kind="explain",
                    text=(
                        "A comment header documents the query's purpose for anyone who opens it "
                        "later. Aliases make the headers readable, ORDER BY sorts the rows "
                        "alphabetically, and LIMIT keeps the output concise. Together these four "
                        "techniques turn a raw database query into a polished, shareable report "
                        "for management, showing both technical precision and professional presentation."
                    ),
                    action=self._wait_action(),
                ),
                ScriptBeat(
                    beat_id="beat_004",
                    kind="state",
                    text=(
                        "Right now the editor shows the commented history above and space below "
                        "for the new query. After we finish, it will hold one contiguous comment "
                        "block followed by a SELECT statement that uses aliases, ORDER BY LastName, "
                        "and a LIMIT clause. The result pane will then show the documented preview."
                    ),
                    action=self._wait_action(),
                ),
                _segment_beat(
                    "beat_005",
                    comment,
                    "We type a comment header that documents this clean, professional query.",
                ),
                _segment_beat(
                    "beat_006",
                    select_clause,
                    "We type the SELECT clause with AS aliases for readable headers.",
                ),
                _segment_beat(
                    "beat_007",
                    from_clause,
                    "We type FROM Customer to point the query at the right table.",
                ),
                _segment_beat(
                    "beat_008",
                    order_clause,
                    "We type ORDER BY LastName so the rows sort alphabetically.",
                ),
                _segment_beat(
                    "beat_009",
                    limit_clause,
                    "We type LIMIT 5 to keep the output concise and manageable.",
                ),
                ScriptBeat(
                    beat_id="beat_010",
                    kind="demo",
                    text="We run the query and the result pane shows the documented preview.",
                    action={"type": "run_query"},
                ),
                ScriptBeat(
                    beat_id="beat_011",
                    kind="validation",
                    text=_validation(
                        f"We see {row_count} {rows_word} returned with readable headers sorted "
                        f"alphabetically by last name, confirming the combined query works as one "
                        f"clean statement. The comment block at the top documents the purpose for "
                        f"anyone who opens the file later"
                        if row_count is not None else "We see the documented preview returned with readable headers sorted alphabetically, confirming the combined query works as one clean statement"
                    ),
                    action=self._verify_action(
                        "the SQL editor contains a comment block followed by a SELECT with aliases, ORDER BY, and LIMIT"
                    ),
                ),
                ScriptBeat(
                    beat_id="beat_012",
                    kind="explain",
                    text=(
                        "The result pane shows a professional preview with readable rows sorted "
                        "by last name, demonstrating all three presentation techniques in one "
                        "statement. The comment header makes the query's intent clear at a glance, "
                        "while aliases, ORDER BY, and LIMIT handle the headers, order, and size. "
                        "This is the kind of query a data analyst can confidently share with a "
                        "manager."
                    ),
                    action=self._wait_action(),
                ),
                ScriptBeat(
                    beat_id="beat_013",
                    kind="explain",
                    text=(
                        "Clean query etiquette matters when other people will read or maintain "
                        "the code. A comment header explains intent, aliases remove jargon, ORDER "
                        "BY removes ambiguity about row order, and LIMIT prevents accidental "
                        "overload of a report. These habits separate a quick scratch query from "
                        "production-ready SQL that a team can trust."
                    ),
                    action=self._wait_action(),
                ),
                ScriptBeat(
                    beat_id="beat_014",
                    kind="close",
                    text=(
                        "We have recapped the chapter with a clean, documented query that "
                        "combines comment headers, aliases, ORDER BY, and LIMIT. The result is a "
                        "readable, sorted, and appropriately sized preview. Next, we will filter "
                        "results with WHERE clauses so we can return only the rows that match "
                        "specific conditions."
                    ),
                    action=self._wait_action(),
                ),
            ]

        else:
            # Unknown SQL demo: fall back to a minimal deterministic script.
            query = (
                "SELECT\n"
                "  FirstName,\n"
                "  LastName,\n"
                "  Email\n"
                "FROM Customer;"
            )
            full_block = _full_block(
                self._make_comment_block("Customer contact list"),
                query,
            )
            beats = [
                ScriptBeat(
                    beat_id="beat_001",
                    kind="opening",
                    text="In this video, we will run a SELECT query in the Execute SQL tab.",
                    action=self._wait_action(),
                ),
                ScriptBeat(
                    beat_id="beat_002",
                    kind="demo",
                    text="We type the comment block and query into the SQL editor.",
                    action={"type": "type_block", "text": full_block},
                ),
                ScriptBeat(
                    beat_id="beat_003",
                    kind="demo",
                    text="We run the query and the result pane fills.",
                    action={"type": "run_query"},
                ),
                ScriptBeat(
                    beat_id="beat_004",
                    kind="validation",
                    text=_validation(
                        "The result pane is populated, confirming the query executed"
                    ),
                    action=self._verify_action(
                        "the Execute SQL tab shows a populated results grid"
                    ),
                ),
                ScriptBeat(
                    beat_id="beat_005",
                    kind="close",
                    text="We have run the query and reviewed the results.",
                    action=self._wait_action(),
                ),
            ]

        return beats

    def _enforce_word_limits(
        self, beats: List[ScriptBeat], video: Any
    ) -> List[ScriptBeat]:
        """
        Planning-only word budget. C6's truncation mechanism is REPEALED.

        This method now only:
          - Records the planned action duration on each beat.
          - Warns if the total script is outside the 400–700 word band.

        Words are never cut to fit the demo; deficits are filled with real
        action at record time (Part E).
        """
        for beat in beats:
            if beat.planned_duration is None:
                beat.planned_duration = self._planned_action_seconds(beat.action)

        total_words = sum(self._word_count(b.text) for b in beats)
        if total_words < 400:
            print(
                f"Warning: script is {total_words} words, below the 400-word minimum. "
                "Narration will be expanded, not cut.",
                file=sys.stderr,
            )
        elif total_words > 700:
            print(
                f"Warning: script is {total_words} words, above the 700-word maximum. "
                "Regenerate with tighter prompts instead of truncating.",
                file=sys.stderr,
            )

        return beats


    @staticmethod
    def _token_set(text: str) -> set[str]:
        """Return lowercase word tokens for overlap comparisons."""
        return set(re.findall(r"[a-z0-9_]+", text.lower()))

    def _validation_has_new_datum(
        self,
        beat: ScriptBeat,
        previous_beats: List[ScriptBeat],
    ) -> bool:
        """
        Return True if the validation beat asserts something not already present
        in the previous two beats. A new datum is a number, column name, or
        concrete value not found in the combined text of the previous two beats.
        Generic words like 'result', 'pane', or 'query' do not count as data.
        """
        if not previous_beats:
            return True

        prev_text = " ".join(b.text for b in previous_beats[-2:])
        prev_tokens = self._token_set(prev_text)
        beat_tokens = self._token_set(beat.text)
        new_tokens = beat_tokens - prev_tokens

        # Numbers count as new observable data.
        has_new_number = bool(
            re.search(r"\d+", beat.text)
            and set(re.findall(r"\d+", beat.text))
            - set(re.findall(r"\d+", prev_text))
        )

        # Column-like capitalized identifiers that are not generic UI words.
        generic_ui = {
            "execute", "sql", "tab", "editor", "query", "result", "pane",
            "table", "grid", "rows", "columns", "browse", "data", "window",
        }
        has_new_column = bool(
            any(
                re.search(r"\b[A-Z][a-zA-Z]+\b", t) and t.lower() not in generic_ui
                for t in new_tokens
            )
        )

        # Distinct concrete closure words that indicate a new verification claim.
        has_new_verification = bool(
            new_tokens & {"returned", "complete", "confirmed", "matches", "succeeded"}
        )
        return has_new_number or has_new_column or has_new_verification

    def _merge_validation_echoes(
        self,
        beats: List[ScriptBeat],
    ) -> List[ScriptBeat]:
        """
        Drop validation beats that merely restate the previous two beats.

        A validation beat is merged/dropped when it adds no new observable datum
        (no new numbers, column names, or factual claims) compared to the
        previous two beats. This prevents redundant "we see the same thing"
        validation echoes.
        """
        merged: List[ScriptBeat] = []
        for i, beat in enumerate(beats):
            if beat.kind == "validation":
                previous = beats[:i]
                if not self._validation_has_new_datum(beat, previous):
                    print(
                        f"  Merging redundant validation echo {beat.beat_id}: {beat.text[:60]}",
                        file=sys.stderr,
                    )
                    continue
            merged.append(beat)
        return merged

    def _extract_data_from_text(self, text: str) -> Set[str]:
        """
        Extract concrete, re-checkable data assertions from a beat.

        Captures:
          - number + noun phrases like "60 rows" or "3 columns"
          - raw numeric counts
          - comma/and/or separated column-like capitalized identifiers
          - table/view descriptors like "Customer table"
        """
        data: Set[str] = set()

        # Number + following noun phrase (keep the head noun only).
        for match in re.finditer(
            r"\b(\d+(?:\.\d+)?)\s+([a-z]{2,}(?:\s+[a-z]{2,}){0,2})\b",
            text,
            re.IGNORECASE,
        ):
            number = match.group(1)
            phrase = match.group(2).lower()
            data.add(f"{number} {phrase.split()[0]}")

        # Raw numeric counts.
        for number in re.findall(r"\b\d+\b", text):
            data.add(f"count:{number}")

        # Column lists: two or more capitalized identifiers separated by commas
        # or coordinating conjunctions, e.g. "FirstName, LastName, and Email".
        # Require at least one lowercase letter so all-caps SQL keywords are ignored.
        column_list_pattern = re.compile(
            r"\b([A-Z][a-z]*[a-z][a-zA-Z]*(?:\s+[A-Z][a-z]*[a-z][a-zA-Z]*)?"
            r"(?:,\s*(?:and\s+|&\s*)?[A-Z][a-z]*[a-z][a-zA-Z]*(?:\s+[A-Z][a-z]*[a-z][a-zA-Z]*)?)+)\b"
        )
        for match in column_list_pattern.finditer(text):
            parts = re.split(r"\s*,\s*|\s+\band\b\s+|\s+\bor\b\s+", match.group(1))
            columns = [re.sub(r"\s+", "", part.strip()) for part in parts if part.strip()]
            if columns:
                data.add("columns:" + ",".join(sorted(set(columns))))

        # Table/view descriptors.
        for match in re.finditer(
            r"\b([A-Z][a-zA-Z]+)\s+(table|view)\b", text, re.IGNORECASE
        ):
            data.add(f"{match.group(1).lower()} {match.group(2).lower()}")

        return data

    @staticmethod
    def _sanitize_repeated_data(text: str, repeated: Set[str]) -> str:
        """Remove repeated data tokens/phrases from a beat text."""
        phrase_datums = [
            d for d in repeated
            if re.match(r"^(\d+(?:\.\d+)?)\s+([a-z]+)$", d, re.IGNORECASE)
        ]
        count_datums = [d for d in repeated if d.startswith("count:")]
        other_datums = repeated - set(phrase_datums) - set(count_datums)

        # Number + noun phrase, e.g. "60 rows" -> "the rows".
        for datum in phrase_datums:
            phrase_match = re.match(r"^(\d+(?:\.\d+)?)\s+([a-z]+)$", datum, re.IGNORECASE)
            assert phrase_match is not None
            number = phrase_match.group(1)
            noun = phrase_match.group(2)
            text = re.sub(
                r"\b" + re.escape(number) + r"\s+" + re.escape(noun) + r"\b",
                f"the {noun}",
                text,
                flags=re.IGNORECASE,
                count=1,
            )

        # Raw numeric counts.
        for datum in count_datums:
            number = datum.split(":", 1)[1]
            text = re.sub(r"\b" + re.escape(number) + r"\b", "", text)

        # Column lists and table descriptors.
        for datum in other_datums:
            if datum.startswith("columns:"):
                columns = datum.split(":", 1)[1].split(",")
                alternatives = "|".join(
                    re.escape(c) for c in sorted(columns, key=len, reverse=True)
                )
                # replace the whole list with a generic placeholder
                text = re.sub(
                    r"\b(?:" + alternatives + r")"
                    r"(?:\s*,\s*(?:and\s+|&\s*)?(?:" + alternatives + r"))+\b",
                    "the columns",
                    text,
                    flags=re.IGNORECASE,
                )
                # strip any remaining individual column mentions
                for column in columns:
                    text = re.sub(r"\b" + re.escape(column) + r"\b", "", text, flags=re.IGNORECASE)
            elif datum.endswith(" table"):
                table = datum.replace(" table", "")
                text = re.sub(
                    r"\b" + re.escape(table) + r"\s+table\b",
                    "the table",
                    text,
                    flags=re.IGNORECASE,
                )

        # Tidy whitespace and punctuation.
        text = re.sub(r"\s+", " ", text).strip(" ,;:.:")
        if text:
            # Ensure the sentence starts with a capital letter.
            text = text[0].upper() + text[1:]
        if text and not re.search(r"[.!?]$", text):
            text += "."
        return text

    def _enforce_datum_uniqueness(self, beats: List[ScriptBeat]) -> None:
        """
        Ensure each concrete datum is asserted only once per script.

        After each beat is finalized, its data-like assertions are extracted. If a
        datum was already asserted in an earlier beat, the later beat is rewritten
        to remove the restatement. Validation beats must cite a new observation or
        reframe without restating numbers already given.

        Demo beats are left untouched: their narration names the columns and clauses
        being typed clause-by-clause, so stripping concrete identifiers would break
        the segmented typing explanation.
        """
        seen: Set[str] = set()
        for beat in beats:
            if beat.kind == "demo":
                continue
            datums = self._extract_data_from_text(beat.text)
            repeated = datums & seen
            if not repeated:
                seen.update(datums)
                continue

            original_text = beat.text
            sanitized = self._sanitize_repeated_data(original_text, repeated)

            # If sanitization failed or still repeats data, fall back to a generic,
            # kind-appropriate sentence that contains no repeated numbers.
            if not sanitized or (self._extract_data_from_text(sanitized) & seen):
                if beat.kind == "validation":
                    sanitized = "The result pane confirms the query ran successfully."
                elif beat.kind == "close":
                    sanitized = "We have completed the query and reviewed the results."
                elif beat.kind in ("explain", "concept"):
                    sanitized = "We see the expected result on screen."
                else:
                    sanitized = "We continue with the next step."
                if not sanitized.endswith("."):
                    sanitized += "."

            beat.text = sanitized
            for datum in sorted(repeated):
                print(
                    f"  [DATUM DEDUP] {beat.beat_id}: dropped repeated datum '{datum}'",
                    file=sys.stderr,
                )
            if beat.text != original_text:
                print(
                    f"  [DATUM DEDUP] {beat.beat_id}: rewritten as '{beat.text}'",
                    file=sys.stderr,
                )
            seen.update(self._extract_data_from_text(beat.text))

    # ------------------------------------------------------------------
    # Script generation
    # ------------------------------------------------------------------

    def generate_script(
        self,
        video: Any,
        fix_errors: Optional[List[str]] = None,
        max_attempts: int = 1,
        env_map: Optional[Dict[str, Any]] = None,
    ) -> List[ScriptBeat]:
        """
        Generate a SQL Essentials-quality narration script for the video.

        Known objectives are rendered deterministically from templates. Unknown
        objectives fall back to an LLM prompt. The validator emits warnings for
        most issues and only hard-fails on empty/missing beats. At most one
        regeneration attempt is made; after that the best-effort script is used.
        """
        exercise = video.exercise_artifact or {}
        db_path = exercise.get("db_path")
        default_table = (video.exercise_artifact or {}).get("table_name", "Orders")
        parsed = self._parse_objective(video.discovery_objective, db_path, default_table=default_table)
        beats: List[ScriptBeat] = []

        discovery_lower = video.discovery_objective.lower()
        is_sql_demo = (
            video.application == "db_browser_sqlite"
            and ("execute sql" in discovery_lower or "select" in discovery_lower)
        )

        if is_sql_demo:
            beats = self._build_sql_script_beats(video, env_map=env_map)
        elif parsed:
            beats = self._build_script_beats(video, parsed, env_map=env_map)
        else:
            if not os.environ.get("ANTHROPIC_API_KEY"):
                print("Error: ANTHROPIC_API_KEY environment variable is not set.", file=sys.stderr)
                return []
            # Ceiling: max_attempts (default 1) LLM script-generation attempts.
            for attempt in range(max_attempts):
                prompt = self._build_script_prompt(video, fix_errors=fix_errors, env_map=env_map)
                response = tracked_create(
                    self.client,
                    model=MODEL,
                    max_tokens=2048,
                    messages=[{"role": "user", "content": prompt}],
                )
                text_parts = [block.text for block in response.content if block.type == "text"]
                raw_text = "\n".join(text_parts).strip()
                script_data = self._parse_script_json(raw_text)
                if not script_data:
                    print(f"Warning: could not parse script JSON (attempt {attempt + 1}); retrying.", file=sys.stderr)
                    continue
                beats = [
                    ScriptBeat(
                        beat_id=item.get("beat_id") or f"beat_{i:03d}",
                        kind=item.get("kind", "state"),
                        text=item.get("text", "").strip(),
                        action=item.get("action"),
                        visual_check=item.get("visual_check"),
                    )
                    for i, item in enumerate(script_data, start=1)
                ]
                beats = self._validate_script_beats(beats, video)
                ok, errors, warnings = self.validate_script(beats, video)
                for warning in warnings:
                    print(f"Warning: {warning}", file=sys.stderr)
                if ok:
                    break
                print(f"Script quality gate failed (attempt {attempt + 1}); regenerating.", file=sys.stderr)
                fix_errors = errors

        beats = self._validate_script_beats(beats, video)
        beats = self._enforce_word_limits(beats, video)
        beats = self._merge_validation_echoes(beats)
        self._enforce_sentence_integrity(beats)
        self._enforce_datum_uniqueness(beats)
        ok, errors, warnings = self.validate_script(beats, video)
        for warning in warnings:
            print(f"Warning: {warning}", file=sys.stderr)
        if not ok:
            print(f"Warning: returning script despite hard failures: {errors}", file=sys.stderr)
        return beats

    @staticmethod
    def _parse_script_json(raw_text: str) -> List[Dict[str, Any]]:
        """Extract a JSON array of beats from the LLM response."""
        script_data: List[Dict[str, Any]] = []
        fenced = re.search(r"```(?:json)?\s*(\[.*\])\s*```", raw_text, re.DOTALL)
        if fenced:
            try:
                script_data = json.loads(fenced.group(1))
            except json.JSONDecodeError:
                pass

        if not script_data:
            try:
                script_data = json.loads(raw_text)
            except json.JSONDecodeError:
                return []

        if not isinstance(script_data, list):
            script_data = script_data.get("script", [])

        return script_data

    def _build_script_prompt(
        self,
        video: Any,
        fix_errors: Optional[List[str]] = None,
        env_map: Optional[Dict[str, Any]] = None,
    ) -> str:
        exercise = video.exercise_artifact or {}
        db_path = exercise.get("db_path", "")
        table_name = exercise.get("table_name", "")
        fix_section = ""
        if fix_errors:
            fix_section = (
                "\n\nThe previous script failed quality review with these errors. "
                "Fix them and return a corrected JSON array with no explanation:\n"
                + "\n".join(f"- {e}" for e in fix_errors)
            )

        env_section = ""
        if env_map:
            env_section = (
                "\n\nOBSERVED ENVIRONMENT (from a scout pass; treat as ground truth):\n"
                f"- Application: {env_map.get('application', video.application)}\n"
                f"- Tables in database: {', '.join(env_map.get('tables', []) or [])}\n"
            )
            row_counts = env_map.get("row_counts", {}) or {}
            if row_counts:
                env_section += "- Exact row counts: " + ", ".join(
                    f"{t}={row_counts.get(t, '?')}" for t in env_map.get("tables", [])
                ) + "\n"
            columns = env_map.get("columns", {}) or {}
            if columns:
                env_section += "- Columns per table:\n"
                for t in env_map.get("tables", []):
                    env_section += f"  - {t}: {', '.join(columns.get(t, []))}\n"
            query_results = env_map.get("query_results", {}) or {}
            if query_results:
                env_section += "- Verified query results:\n"
                for query, qresult in query_results.items():
                    cols = qresult.get("columns", [])
                    count = qresult.get("row_count", "?")
                    env_section += f"  - {query}\n"
                    env_section += f"    columns: {', '.join(cols)}\n"
                    env_section += f"    row_count: {count}\n"
            ui = env_map.get("ui") or {}
            env_section += (
                f"- Active tab on launch: {ui.get('active_tab')}\n"
                f"- Available tabs: {', '.join(ui.get('available_tabs', []) or [])}\n"
                f"- Browse Data default table: {ui.get('browse_data_default_table')}\n"
                f"- Notable UI state: {ui.get('notable', 'none')}\n"
            )

        style_guide = ""
        try:
            style_guide = Path(self.style_guide_path).read_text(encoding="utf-8")
        except Exception:
            style_guide = "Follow the WSDA delivery style: first-person plural present tense, no filler, narrate as it happens."

        return f"""You are writing narration for a short software-training video in the style of SQL Essentials.

Course context
- Topic: {getattr(video, 'title', '')}
- Tool: DB Browser for SQLite
- Learning objective: {video.learning_objective}
- Discovery objective: {video.discovery_objective}
- Running example: {table_name} table in {db_path}
{env_section}

DELIVERY STYLE GUIDE (apply verbatim):
{style_guide}

STRICT RULES (zero exceptions):
1. Total script length: 400-700 words. Add real explanation/state beats so the learner follows the reasoning, not just the clicks.
2. Beat kinds (in order): opening, state/explain (1-3 beats), demo (2-5 beats), validation, close.
3. Voice: first person plural, present tense. "We click...", "We type...", "We see..."
4. NEVER use: you'll, you need to, it's important to, before you, if you skip, understand, learn, concept, abstract.
5. Demo beats must combine action + immediate result in one sentence:
   GOOD: "We click the Browse Data tab and the table view opens."
   BAD: "Click the Browse Data tab." (robotic command)
   BAD: "The table view opens." (no action)
6. Every demo beat = exactly ONE atomic action with narration 15-80 words, written to be spoken WHILE the action happens.
7. Do NOT generate actions already satisfied by the observed default state.
8. Validation beats must reference facts in the EnvironmentMap verbatim (exact table names, column names, and row counts).
9. Only state numbers/names present in the EnvironmentMap. Never invent quantities, table names, or column names.
10. Close beat: "We have [skill]." Include a preview of the next topic.
11. Per-beat minimum words: opening/state/explain >= 25, demo/validation >= 15, close >= 30. Every beat must end with terminal punctuation and its final sentence must not end on a function word (the, a, an, our, we, and, or, to, of, with, for, in, on).
12. SQL keywords in narration stay uppercase: SELECT, FROM, WHERE. Use "star" for *.

Return ONLY a JSON array of beats like:
[
  {{"beat_id": "beat_001", "kind": "opening", "text": "In this video, we will open the Orders table.", "action": {{"type": "browse_table", "table": "Orders"}}}},
  {{"beat_id": "beat_002", "kind": "demo", "text": "We click the Browse Data tab and the table view opens."}},
  {{"beat_id": "beat_003", "kind": "demo", "text": "We select Orders from the dropdown and the rows appear."}},
  {{"beat_id": "beat_004", "kind": "validation", "text": "We see all rows and columns in the Orders table."}},
  {{"beat_id": "beat_005", "kind": "close", "text": "We have opened the Orders table and confirmed its structure."}}
]
{fix_section}
"""

    @staticmethod
    def _validate_script_beats(beats: List[ScriptBeat], video: Any) -> List[ScriptBeat]:
        """Normalize actions, split multi-step demo beats, and drop invalid beats."""
        valid_kinds = {"opening", "concept", "demo", "explain", "validation", "close", "recap", "preview", "state"}
        supported_actions = {
            "browse_table", "sort_column", "filter_column", "execute_query",
            "click", "type", "type_block", "type_segments", "key", "run_query", "wait", "verify", "sequence",
        }
        cleaned: List[ScriptBeat] = []
        for beat in beats:
            if beat.kind not in valid_kinds:
                print(f"Warning: dropping script beat with unknown kind {beat.kind!r}", file=sys.stderr)
                continue
            if not beat.text or not beat.text.strip():
                print(f"Warning: {beat.beat_id} has empty text; skipping.", file=sys.stderr)
                continue
            if beat.kind == "demo" and beat.action:
                action_type = beat.action.get("type")
                if action_type not in supported_actions:
                    print(
                        f"Warning: demo beat {beat.beat_id} uses unsupported action {action_type!r}; skipping.",
                        file=sys.stderr,
                    )
                    continue
                beat.action = _format_action_sql(beat.action)
                beat.action = LessonBuilder._normalize_action(beat.action)
                if beat.planned_duration is None:
                    beat.planned_duration = LessonBuilder._planned_action_seconds(beat.action)
            cleaned.append(beat)

        cleaned = LessonBuilder._split_atomic_demo_beats(cleaned)
        return cleaned

    @staticmethod
    def _split_atomic_demo_beats(beats: List[ScriptBeat]) -> List[ScriptBeat]:
        """Split any demo beat whose action contains >1 UI step into atomic demo beats."""
        result: List[ScriptBeat] = []
        for beat in beats:
            if beat.kind != "demo" or not beat.action:
                result.append(beat)
                continue

            sub_actions = LessonBuilder._atomic_sub_actions(beat.action)
            if len(sub_actions) <= 1:
                result.append(beat)
                continue

            texts = LessonBuilder._split_demo_text(beat.text, len(sub_actions))
            for i, (sub_action, sub_text) in enumerate(zip(sub_actions, texts), start=1):
                suffix = f"_{chr(ord('a') + i - 1)}"
                normalized = LessonBuilder._normalize_action(sub_action)
                result.append(
                    ScriptBeat(
                        beat_id=f"{beat.beat_id}{suffix}",
                        kind="demo",
                        text=sub_text,
                        action=normalized,
                        planned_duration=LessonBuilder._planned_action_seconds(normalized),
                    )
                )
        return result

    @staticmethod
    def _atomic_sub_actions(action: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Return a flat list of atomic UI actions from an action spec."""
        action_type = action.get("type")
        if action_type == "sequence":
            return [
                sub
                for sub in (action.get("actions") or [])
                if sub
            ]
        return [action]

    @staticmethod
    def _split_demo_text(text: str, n_parts: int) -> List[str]:
        """Split a demo narration into one phrase per atomic action."""
        text = text.strip().rstrip(".")
        # Try splitting on common conjunctions that join two action clauses.
        separators = [" and ", ", then ", ", and then ", "; "]
        lowered = text.lower()
        for sep in separators:
            if sep in lowered:
                raw_parts = [p.strip() for p in text.split(sep, n_parts - 1)]
                if len(raw_parts) == n_parts:
                    fixed: List[str] = []
                    for part in raw_parts:
                        part = part.rstrip(",.;:")
                        if not part.lower().startswith("we "):
                            # Fragment like "select Customers" -> "We select Customers."
                            part = f"We {part[0].lower()}{part[1:]}"
                        if part and not part.endswith("."):
                            part += "."
                        fixed.append(part)
                    return fixed

        # Fallback: generate concise narration for each implied action.
        words = text.split()
        chunk_size = max(1, len(words) // n_parts)
        parts: List[str] = []
        for i in range(n_parts):
            start = i * chunk_size
            end = len(words) if i == n_parts - 1 else (i + 1) * chunk_size
            chunk = " ".join(words[start:end]).rstrip(",.;:")
            if chunk and not chunk.lower().startswith("we "):
                chunk = f"We {chunk[0].lower()}{chunk[1:]}"
            if chunk and not chunk.endswith("."):
                chunk += "."
            parts.append(chunk)
        return parts

    def validate_script(
        self, beats: List[ScriptBeat], video: Any
    ) -> tuple[bool, List[str], List[str]]:
        """
        SQL Essentials quality gate. Returns (ok, hard_errors, warnings).

        Only empty text, missing required beat kinds, and a complete lack of demo
        beats are treated as hard failures. Everything else is a warning so the
        pipeline can use the best-effort script instead of looping forever.
        """
        errors: List[str] = []
        warnings: List[str] = []

        if not beats:
            errors.append("Script is empty.")
            return False, errors, warnings

        kinds = [b.kind for b in beats]
        required = {"opening", "demo", "validation", "close"}
        missing = required - set(kinds)
        if missing:
            errors.append(f"Missing required beat kinds: {sorted(missing)}")

        if "concept" in kinds:
            warnings.append("Concept beats are not preferred; consider replacing with demo/validation.")

        # Structure ordering warnings.
        try:
            first_demo = kinds.index("demo")
        except ValueError:
            first_demo = len(kinds)
        if "validation" in kinds and "close" in kinds:
            if kinds.index("validation") > kinds.index("close"):
                warnings.append("Validation beat should come before the close beat.")
        if "opening" in kinds and first_demo < kinds.index("opening"):
            warnings.append("Opening should come before demo beats.")

        demo_count = sum(1 for b in beats if b.kind == "demo")
        action_count = sum(1 for b in beats if b.kind == "demo" and b.action)
        if demo_count == 0:
            errors.append("Script must have at least one demo beat.")
        if action_count == 0:
            warnings.append("Demo beats lack concrete actions; discovery may not reach the objective.")

        # Total length (Part B gate: 400-700 words).
        total_words = sum(self._word_count(b.text) for b in beats)
        if not (400 <= total_words <= 700):
            errors.append(f"Script is {total_words} words; must be in [400, 700].")

        # Per-kind soft word limits, widened for 400-700-word chapter scripts.
        limits = {
            "opening": (25, 120),
            "concept": (25, 120),
            "state": (25, 100),
            "explain": (25, 120),
            "demo": (15, 80),
            "validation": (15, 80),
            "close": (30, 120),
        }
        for beat in beats:
            wc = self._word_count(beat.text)
            lo, hi = limits.get(beat.kind, (15, 120))
            if not (lo <= wc <= hi):
                warnings.append(f"{beat.beat_id} ({beat.kind}) has {wc} words; expected {lo}-{hi}.")

            # A3 integrity gate is a hard error.
            if not self._beat_text_integrity_ok(beat.text, beat.kind):
                errors.append(
                    f"{beat.beat_id} ({beat.kind}) fails integrity: terminal punctuation, "
                    f"min words, or function-word ending."
                )

            lowered = beat.text.lower()
            for pattern in self._FORBIDDEN_VOICE_PATTERNS:
                if re.search(pattern, lowered):
                    warnings.append(f"{beat.beat_id} contains filler phrase matching /{pattern}/.")
                    break

            if self._SECOND_PERSON_PATTERN.search(beat.text):
                warnings.append(f"{beat.beat_id} uses second-person voice; prefer 'we'.")

            if not beat.text.startswith("We ") and beat.kind in {"demo", "validation", "close"}:
                warnings.append(f"{beat.beat_id} should start with 'We '.")

            if beat.kind == "opening" and not re.search(
                r"^(in this video,\s+)?(we\s+will|this\s+video\s+shows|we\s+are\s+going)", lowered
            ):
                warnings.append(f"{beat.beat_id} opening should state the objective clearly.")

            if beat.kind == "close" and not re.search(r"^we\s+have", lowered):
                warnings.append(f"{beat.beat_id} close should recap the skill.")

            if beat.kind == "validation" and _contains_action_word(beat.text):
                warnings.append(f"{beat.beat_id} (validation) describes an action; prefer a visible fact.")

            if beat.kind == "demo":
                action_words = {"click", "type", "press", "enter", "select", "choose", "hit", "tap", "open", "run", "sort", "filter"}
                if not any(w in lowered for w in action_words):
                    warnings.append(f"{beat.beat_id} (demo) does not describe an action.")

        # C9 canonical format contract: any typed SQL block must follow the tight
        # editor standard. Segmented beats compose one cumulative block across
        # consecutive type_segments demo beats, so validate the concatenation,
        # not each segment in isolation.
        segment_buffer: List[str] = []

        def _flush_segment_buffer() -> None:
            if not segment_buffer:
                return
            cumulative = "".join(segment_buffer)
            if cumulative:
                violations = self._assert_canonical_format(cumulative)
                for violation in violations:
                    errors.append(
                        f"{buffer_beat_id} violates canonical format: {violation}"
                    )
            segment_buffer.clear()

        buffer_beat_id = ""
        for beat in beats:
            if beat.kind == "demo" and beat.action:
                action_type = beat.action.get("type")
                if action_type == "type_segments":
                    segments = beat.action.get("segments") or []
                    segment_text = "".join(
                        (s.get("text", "") if isinstance(s, dict) else str(s))
                        for s in segments
                    )
                    if segment_text:
                        if not segment_buffer:
                            buffer_beat_id = beat.beat_id
                        segment_buffer.append(segment_text)
                    continue

            # Any non-accumulating beat flushes the pending segmented block.
            _flush_segment_buffer()

            if beat.kind == "demo" and beat.action:
                action_type = beat.action.get("type")
                if action_type in ("type_block", "append_block"):
                    sql_text = beat.action.get("text") or beat.action.get("detail") or ""
                    if sql_text:
                        violations = self._assert_canonical_format(sql_text)
                        for violation in violations:
                            errors.append(
                                f"{beat.beat_id} violates canonical format: {violation}"
                            )

        _flush_segment_buffer()

        return not errors, errors, warnings
    # ------------------------------------------------------------------
    # Action derivation
    # ------------------------------------------------------------------

    def derive_actions(
        self, beats: List[ScriptBeat], db_path: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Convert script demo beats into concrete UI actions the discovery harness
        can execute. Recipe-friendly actions are expanded by DiscoveryRecipes;
        generic click/type/key actions pass through.
        """
        actions, _ = self._derive_actions_with_mapping(beats, db_path)
        return actions

    def _derive_actions_with_mapping(
        self, beats: List[ScriptBeat], db_path: Optional[str] = None
    ) -> Tuple[List[Dict[str, Any]], List[Optional[ScriptBeat]]]:
        """
        Like derive_actions, but also returns a parallel list mapping each
        concrete action back to its source demo beat (or None).
        """
        actions: List[Dict[str, Any]] = []
        beat_for_action: List[Optional[ScriptBeat]] = []
        for beat in beats:
            if beat.kind != "demo" or not beat.action:
                continue
            derived = self._derive_single_action(beat.action, db_path)
            actions.extend(derived)
            beat_for_action.extend([beat] * len(derived))
        return actions, beat_for_action

    @staticmethod
    def _derive_single_action(
        action_spec: Dict[str, Any], db_path: Optional[str]
    ) -> List[Dict[str, Any]]:
        action_type = action_spec.get("type")

        if action_type == "browse_table":
            return DiscoveryRecipes.browse_table(action_spec.get("table", "Orders"), db_path)

        if action_type == "sort_column":
            return DiscoveryRecipes.sort_column(
                action_spec.get("table", "Orders"),
                action_spec.get("column", "amount"),
                action_spec.get("direction", "asc"),
            )

        if action_type == "filter_column":
            return DiscoveryRecipes.filter_column(
                action_spec.get("table", "Orders"),
                action_spec.get("column", "region"),
                action_spec.get("value", ""),
            )

        if action_type == "execute_query":
            query = action_spec.get("query", "SELECT * FROM Orders")
            return DiscoveryRecipes.execute_query(query)

        if action_type == "click":
            return [
                {
                    "action": "click",
                    "target": action_spec.get("target", {"x": 0.5, "y": 0.5}),
                    "description": action_spec.get("description", "Click"),
                    "animate": True,
                }
            ]

        if action_type == "type":
            target = action_spec.get("target", {"x": 0.5, "y": 0.5})
            return [
                {
                    "action": "click",
                    "target": target,
                    "description": action_spec.get("description", "Focus input"),
                    "animate": True,
                },
                {
                    "action": "type",
                    "target": target,
                    "text": action_spec.get("text", ""),
                    "description": action_spec.get("description", "Type value"),
                    "click_first": False,
                },
            ]

        if action_type == "key":
            return [
                {
                    "action": "key",
                    "text": action_spec.get("text", "Return"),
                    "description": action_spec.get("description", "Press key"),
                }
            ]

        if action_type == "wait":
            return [
                {
                    "action": "wait",
                    "duration": action_spec.get("duration", 1.0),
                    "description": action_spec.get("description", "Wait"),
                }
            ]

        # Vision-agent native actions pass through unchanged.
        if action_type in ("type_block", "type_segments", "run_query", "summarize_result_pane"):
            return [dict(action_spec)]

        print(f"Warning: unsupported action type {action_type!r}", file=sys.stderr)
        return []

    # ------------------------------------------------------------------
    # Script execution
    # ------------------------------------------------------------------

    def execute_script(
        self,
        beats: List[ScriptBeat],
        discovery: EndStateDiscovery,
        db_path: Optional[str] = None,
        opening_state_query: Optional[str] = None,
        opening_state_history: Optional[str] = None,
        new_query: Optional[str] = None,
    ) -> DiscoveryResult:
        """
        Run the vision-agent script beats through the discovery harness.

        The harness records one video clip per beat (opening, demo, validation,
        close) and stores the path in ``beat.video_clip_path``.
        """
        # Back-fill demo actions using the text parser for any beat that is
        # missing an action specification, normalising multi-action parses to a
        # single sequence action dict.
        for beat in beats:
            if beat.kind == "demo" and not beat.action:
                parsed = self._parse_demo_to_action(beat.text)
                if len(parsed) == 1:
                    beat.action = parsed[0]
                elif len(parsed) > 1:
                    beat.action = {"type": "sequence", "actions": parsed}

        result = discovery.execute_script(
            beats=beats,
            visual_summary=discovery.objective,
            save_all_screenshots=True,
            opening_state_query=opening_state_query,
            opening_state_history=opening_state_history,
            new_query=new_query,
        )

        # ADAPT narration for validation/concept beats that conflict with observed facts.
        self._adapt_beats_to_observed_state(beats)
        beats = self._collapse_merge_beats(beats)

        return result

    def _adapt_beats_to_observed_state(self, beats: List[ScriptBeat]) -> None:
        """
        Rewrite beats whose claims conflict with observed facts or footage.

        Demo beats are preserved as demo beats so their recorded clips remain
        available to the renderer; only the narration text is rewritten when it
        conflicts with the observed state. Validation beats are preserved so the
        renderer can end on the last demo clip's final frame.

        Each adapted beat must add new information compared to all previously
        finalized beats; otherwise it is marked MERGE and collapsed later.
        """
        self._enforce_clip_truthfulness(beats)
        finalized_texts: List[str] = []
        for beat in beats:
            if beat.merge:
                continue
            if beat.kind == "concept" and beat.observed_state:
                if self._beat_conflicts_with_observed_state(beat):
                    self._rewrite_beat_from_observed(
                        beat, "state", previous_texts=finalized_texts
                    )
            # Continuity-aware rendering: opening state beats that could not be
            # established must be rewritten to describe the actual screen. When the
            # continuity history was successfully pasted, also rewrite the state beat
            # so it explicitly mentions the commented-out previous queries.
            if (
                beat.kind == "state"
                and beat.observed_state
                and beat.observed_state.get("opening_state_strategy") == "adapted"
            ):
                if self._beat_conflicts_with_observed_state(beat):
                    self._rewrite_beat_from_observed(
                        beat, "state", previous_texts=finalized_texts
                    )
            elif (
                beat.kind == "state"
                and beat.observed_state
                and beat.observed_state.get("opening_state_strategy") == "established"
                and beat.observed_state.get("history_pasted")
                and "commented out" not in beat.text.lower()
                and "commented-out" not in beat.text.lower()
            ):
                self._rewrite_beat_from_observed(
                    beat,
                    "state",
                    extra_instruction=(
                        "The editor contains the course's previous queries commented out "
                        "above the new query. Rewrite the state beat to mention that the "
                        "previous queries sit above, commented out, and that the new query "
                        "is being added below them."
                    ),
                    previous_texts=finalized_texts,
                )
            finalized_texts.append(beat.text)

    def _enforce_clip_truthfulness(self, beats: List[ScriptBeat]) -> List[ScriptBeat]:
        """
        Keep demo beats as demo beats and preserve their recorded clips.

        Converting a demo beat to a state beat drops its clip, which can cause
        the renderer to end on an earlier still frame instead of the final
        discovered state. Demo beats whose actions inherently change the UI
        (typing SQL, running a query, executing a recipe) are never rewritten,
        because the coarse observed-state summary can incorrectly report no
        visible change. Validation beats are always preserved.
        """
        previous_observed: Optional[Dict[str, Any]] = None
        finalized_texts: List[str] = []
        for beat in beats:
            if beat.kind == "validation":
                # Validation beats are allowed without clips or observed_state.
                finalized_texts.append(beat.text)
                continue
            if beat.kind == "demo" and beat.observed_state:
                if self._action_produces_visible_change(beat.action):
                    # The action itself guarantees visible change; keep the
                    # original narration and recorded clip.
                    previous_observed = beat.observed_state
                    finalized_texts.append(beat.text)
                    continue
                if previous_observed and self._observed_state_unchanged(
                    previous_observed, beat.observed_state
                ):
                    self._rewrite_beat_from_observed(
                        beat,
                        "demo",
                        extra_instruction=(
                            "The screen did not visibly change during this beat. "
                            "Rewrite the narration to describe the existing state while keeping the demo intent."
                        ),
                        previous_texts=finalized_texts,
                    )
            finalized_texts.append(beat.text)
            previous_observed = beat.observed_state or previous_observed
        return beats

    @staticmethod
    def _collapse_merge_beats(beats: List[ScriptBeat]) -> List[ScriptBeat]:
        """Join each beat marked merge into the previous beat and remove it."""
        collapsed: List[ScriptBeat] = []
        for beat in beats:
            if beat.merge and collapsed:
                prev = collapsed[-1]
                prev.text = f"{prev.text} {beat.text}".strip()
                # Inherit the latest clip if the merge beat carried one.
                if beat.video_clip_path:
                    prev.video_clip_path = beat.video_clip_path
                    prev.action = beat.action or prev.action
                print(f"  Merged {beat.beat_id} into {prev.beat_id}", file=sys.stderr)
                continue
            collapsed.append(beat)
        return collapsed

    @staticmethod
    def _action_produces_visible_change(action: Optional[Dict[str, Any]]) -> bool:
        """Return True for actions that always produce a visible UI change."""
        if not isinstance(action, dict):
            return False
        action_type = action.get("type")
        if action_type in ("type_block", "run_query", "execute_query"):
            return True
        if action_type == "type" and action.get("text"):
            return True
        if action_type == "key" and action.get("text") in {"F5", "Return", "Enter"}:
            return True
        if action_type == "sequence":
            return any(
                LessonBuilder._action_produces_visible_change(sub)
                for sub in (action.get("actions") or [])
            )
        return False

    @staticmethod
    def _extract_data_tokens(text: str) -> Set[str]:
        """Return numbers and column-like identifiers present in ``text``."""
        lowered = text.lower()
        tokens: Set[str] = set()
        # Numbers
        tokens.update(re.findall(r"\d+(?:\.\d+)?", lowered))
        # Column-like identifiers (underscore or capitalized camel/Pascal)
        tokens.update(re.findall(r"\b[a-z]+_[a-z_]+\b", lowered))
        tokens.update(re.findall(r"\b[a-z]+[A-Z][a-zA-Z]*\b", lowered))
        # Verification words
        tokens.update({w for w in {"confirms", "confirms", "verified", "validation", "returned"} if w in lowered})
        return tokens

    def _rewrite_beat_from_observed(
        self,
        beat: ScriptBeat,
        target_kind: str,
        extra_instruction: str = "",
        previous_texts: Optional[List[str]] = None,
    ) -> None:
        """Use a text-only LLM call to rewrite a beat from observed facts."""
        observed = beat.observed_state
        previous_texts = previous_texts or []
        previous_block = ""
        if previous_texts:
            previous_block = "\nPreviously finalized narration:\n" + "\n".join(
                f"- {t}" for t in previous_texts
            )
            previous_block += (
                "\n\nThe rewritten beat must add NEW concrete information "
                "(a number, column name, table name, or verification claim) "
                "that is not already in the previously finalized narration. "
                "If there is no new concrete information, reply with the literal token MERGE."
            )
        prompt = (
            "Rewrite this narration beat to describe ONLY the stable observed state. "
            "Do not invent numbers, column names, or table names. "
            "Do NOT mention dropdowns, menus, modals, popups, or anything transient that is open. "
            "Describe only what is persistently visible in the main window. "
            "Keep first person plural, present tense, and the original intent. "
            + extra_instruction
            + previous_block
            + "\n\n"
            f"Original beat: {beat.text}\n"
            f"Observed facts:\n"
            f"- Active tab: {observed.get('active_tab')}\n"
            f"- Visible table: {observed.get('visible_table')}\n"
            f"- Row range text: {observed.get('row_range_text')}\n"
            f"- Column headers: {', '.join(observed.get('column_headers', []) or [])}\n"
            f"- Summary: {observed.get('summary')}\n"
        )
        try:
            response = tracked_create(
                self.client,
                model=MODEL,
                max_tokens=256,
                messages=[{"role": "user", "content": prompt}],
            )
            text_parts = [block.text for block in response.content if block.type == "text"]
            rewritten = "\n".join(text_parts).strip().strip('"')
            if rewritten.upper() == "MERGE":
                print(f"  Adapted {beat.beat_id}: marked MERGE", file=sys.stderr)
                beat.merge = True
                return
            if rewritten:
                # Enforce new-information rule locally as well.
                new_data = self._extract_data_tokens(rewritten)
                previous_data = set().union(
                    *(self._extract_data_tokens(t) for t in previous_texts)
                )
                if previous_texts and not (new_data - previous_data):
                    print(f"  Adapted {beat.beat_id}: no new datum; marked MERGE", file=sys.stderr)
                    beat.merge = True
                    return
                print(
                    f"  Adapted {beat.beat_id}: '{beat.text[:50]}...' -> '{rewritten[:50]}...'",
                    file=sys.stderr,
                )
                beat.text = rewritten
                beat.kind = target_kind  # type: ignore[assignment]
                # Only drop the recorded clip when converting to a non-demo kind.
                # Demo beats must keep their clips so the renderer can play the
                # recorded action and end on the discovered state.
                if target_kind != "demo":
                    beat.video_clip_path = None
        except Exception as exc:
            print(f"Warning: could not adapt {beat.beat_id}: {exc}", file=sys.stderr)

    @staticmethod
    def _observed_state_unchanged(
        prev: Dict[str, Any], curr: Dict[str, Any]
    ) -> bool:
        """Return True if the UI state did not visibly change between observations."""
        keys = ["active_tab", "visible_table", "row_range_text"]
        for key in keys:
            if prev.get(key) != curr.get(key):
                return False
        prev_headers = set(prev.get("column_headers") or [])
        curr_headers = set(curr.get("column_headers") or [])
        if prev_headers != curr_headers:
            return False
        return True

    @staticmethod
    def _beat_conflicts_with_observed_state(beat: ScriptBeat) -> bool:
        """Detect obvious mismatches between beat text and observed state."""
        observed = beat.observed_state
        if not observed:
            return False
        text = beat.text.lower()
        headers = [h.lower() for h in (observed.get("column_headers") or [])]
        visible_table = (observed.get("visible_table") or "").lower()

        # Mentioned table not visible?
        for table in ["customers", "orders"]:
            if table in text and visible_table and table != visible_table:
                return True

        # Mentioned column not in the observed headers?
        text_words = set(__import__("re").findall(r"\b[a-z_]+\b", text))
        for word in text_words:
            if len(word) <= 3:
                continue
            if word in {"table", "rows", "grid", "columns", "browse", "data", "status"}:
                continue
            if word not in headers and any(h in word or word in h for h in headers):
                # Partial match to an observed header is OK.
                continue
            # If the word looks like a column name (contains underscore) and isn't observed.
            if "_" in word and word not in headers:
                return True

        # Numbers in text that don't appear in row_range_text?
        row_range = observed.get("row_range_text") or ""
        try:
            text_numbers = {int(n) for n in _extract_numbers(beat.text)}
            observed_numbers = {int(n) for n in _extract_numbers(row_range)}
        except Exception:
            text_numbers = set()
            observed_numbers = set()
        # If beat states a count that isn't in the observed range text, flag it.
        for num in text_numbers:
            if num >= 10 and num not in observed_numbers:
                return True

        # UI element count assertions (e.g. "two tabs", "three buttons").
        ui_counts = observed.get("ui_element_counts") or {}
        written_numbers = {
            "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
            "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
        }
        count_pattern = re.compile(
            r"\b(one|two|three|four|five|six|seven|eight|nine|ten|\d+)\s+([a-z]{2,})"
        )
        for match in count_pattern.finditer(text):
            count_str, noun = match.groups()
            count = int(count_str) if count_str.isdigit() else written_numbers.get(count_str)
            if count is None:
                continue
            noun_lower = noun.lower()
            # Allow singular/plural mismatch between text and map keys.
            keys_to_check = {noun_lower}
            if noun_lower.endswith("s"):
                keys_to_check.add(noun_lower[:-1])
            else:
                keys_to_check.add(noun_lower + "s")
            matched_key = next((k for k in keys_to_check if k in ui_counts), None)
            if matched_key is None:
                # A state beat asserting a UI element count that is not grounded
                # in the observed environment is treated as a conflict.
                if beat.kind == "state":
                    return True
                continue
            if ui_counts[matched_key] != count:
                return True

        return False

    # ------------------------------------------------------------------
    # Graph construction
    # ------------------------------------------------------------------

    def build_graph(
        self,
        video: Any,
        beats: List[ScriptBeat],
        discovery_result: DiscoveryResult,
    ) -> ExecutionGraph:
        """
        Build an ExecutionGraph from the script beats and the discovery result.

        Demo beats become edges whose video_path is the recorded clip for that
        beat. Non-demo beats attach to the surrounding states.
        """
        if not discovery_result.success or not discovery_result.locked_state:
            raise ValueError("DiscoveryResult must be successful to build a graph.")

        graph_id = f"{video.video_id}_{uuid.uuid4().hex[:8]}"
        application = video.application
        output_dir = Path(discovery_result.locked_state.screenshot_path).parent

        demo_beats = [b for b in beats if b.kind == "demo"]
        if not demo_beats:
            raise ValueError("Cannot build graph: script has no demo beats.")

        # Build states from frame extracts of demo-beat clips.
        states: List[ScreenState] = []
        start_screenshot = self._extract_first_frame(
            demo_beats[0].video_clip_path, output_dir / f"{graph_id}_start.png"
        )
        start_state = ScreenState(
            state_id="state_000",
            screenshot_path=str(start_screenshot.resolve()),
            timestamp=0.0,
            application=application,  # type: ignore[arg-type]
            platform_snapshot={},
            visual_summary="Start state",
        )

        prev_state = start_state
        edges: List[ActionEdge] = []

        for i, beat in enumerate(demo_beats, start=1):
            if not beat.video_clip_path:
                raise ValueError(f"Demo beat {beat.beat_id} has no recorded clip.")

            end_screenshot = self._extract_last_frame(
                beat.video_clip_path, output_dir / f"{graph_id}_state_{i:03d}.png"
            )
            state_id = f"state_{i:03d}"
            state = ScreenState(
                state_id=state_id,
                screenshot_path=str(end_screenshot.resolve()),
                timestamp=0.0,
                application=application,  # type: ignore[arg-type]
                platform_snapshot={},
                visual_summary=f"After {beat.beat_id}",
            )
            states.append(state)

            action_type: Literal["click", "type", "select", "scroll", "hotkey", "wait", "api_seed"] = "click"
            payload: Optional[str] = None
            target: Dict[str, Any] = {}
            if beat.action:
                raw_type = beat.action.get("type")
                if raw_type == "execute_query":
                    action_type = "type"
                    payload = beat.action.get("query")
                    target = {"x": 0.5, "y": 0.45, "w": 700, "h": 300}
                elif raw_type == "type":
                    action_type = "type"
                    # Vision-agent beats store the text in ``detail``; older recipe
                    # beats store it in ``text``.  Support both.
                    payload = beat.action.get("detail") or beat.action.get("text")
                    target = beat.action.get("target", {})
                elif raw_type == "click":
                    action_type = "click"
                    target = beat.action.get("target", {})
                elif raw_type == "key":
                    action_type = "hotkey"
                    payload = beat.action.get("detail") or beat.action.get("text")
                elif raw_type == "sequence":
                    # A sequence edge represents a multi-step demo beat; the renderer
                    # uses the recorded clip, so the edge action is a generic click.
                    action_type = "click"
                elif raw_type == "wait":
                    action_type = "wait"
                else:
                    target = beat.action.get("target", {})

            # ActionEdge.target must be a dict; vision-agent strings are descriptive.
            if isinstance(target, str):
                target = {"description": target}
            elif not isinstance(target, dict):
                target = {}

            edges.append(
                ActionEdge(
                    edge_id=f"edge_{i:03d}",
                    from_state_id=prev_state.state_id,
                    to_state_id=state.state_id,
                    action_type=action_type,
                    target=target,
                    payload=payload,
                    expected_duration=2.0,
                    video_path=beat.video_clip_path,
                )
            )
            prev_state = state

        end_state = discovery_result.locked_state
        end_state.state_id = f"state_{len(states) + 1:03d}"

        # Connect the last demo state to the locked end state if they differ.
        if prev_state.state_id != end_state.state_id:
            edges.append(
                ActionEdge(
                    edge_id=f"edge_{len(edges) + 1:03d}",
                    from_state_id=prev_state.state_id,
                    to_state_id=end_state.state_id,
                    action_type="wait",
                    target={},
                    payload=None,
                    expected_duration=1.0,
                    video_path=None,
                )
            )

        narration_beats = self._overlay_beats(beats, start_state, states, end_state, edges)

        graph = ExecutionGraph(
            graph_id=graph_id,
            learning_objective=video.learning_objective,
            application=application,
            start_state=start_state,
            end_state=end_state,
            states=states,
            edges=edges,
            narration_beats=narration_beats,
            generation_cost_usd=round(
                sum(log.get("cost_usd", 0.0) for log in discovery_result.attempt_logs), 6
            ),
            reliability_score=discovery_result.reliability_score,
        )

        store = GraphStore()
        store.save(graph)
        return graph

    @staticmethod
    def _extract_first_frame(video_path: Optional[str], out_path: Path) -> Path:
        """Extract the first frame of a video clip to a PNG."""
        if not video_path or not Path(video_path).exists():
            return LessonBuilder._blank_image(out_path)
        try:
            subprocess.run(
                [
                    "ffmpeg", "-y", "-i", video_path,
                    "-ss", "0", "-vframes", "1",
                    "-pix_fmt", "rgb24", str(out_path),
                ],
                check=True, capture_output=True, timeout=60,
            )
            return out_path
        except Exception as exc:
            print(f"Warning: could not extract first frame from {video_path}: {exc}", file=sys.stderr)
            return LessonBuilder._blank_image(out_path)

    @staticmethod
    def _extract_last_frame(video_path: Optional[str], out_path: Path) -> Path:
        """Extract the last frame of a video clip to a PNG."""
        if not video_path or not Path(video_path).exists():
            return LessonBuilder._blank_image(out_path)
        try:
            subprocess.run(
                [
                    "ffmpeg", "-y", "-sseof", "-0.5", "-i", video_path,
                    "-vframes", "1", "-pix_fmt", "rgb24", str(out_path),
                ],
                check=True, capture_output=True, timeout=60,
            )
            return out_path
        except Exception as exc:
            print(f"Warning: could not extract last frame from {video_path}: {exc}", file=sys.stderr)
            return LessonBuilder._blank_image(out_path)

    @staticmethod
    def _blank_image(out_path: Path) -> Path:
        """Create a small black placeholder PNG."""
        try:
            from PIL import Image
            img = Image.new("RGB", (1280, 720), color=(0, 0, 0))
            img.save(out_path)
        except Exception:
            # Absolute fallback: write a 1x1 transparent PNG header.
            out_path.write_bytes(
                b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\x0cIDATx\x9cc\xf8\x0f\x00\x00\x01\x01\x00\x05\x18\xd8N\x00\x00\x00\x00IEND\xaeB`\x82"
            )
        return out_path

    @staticmethod
    def _overlay_beats(
        beats: List[ScriptBeat],
        start_state: ScreenState,
        intermediate_states: List[ScreenState],
        end_state: ScreenState,
        edges: List[ActionEdge],
    ) -> List[NarrationBeat]:
        """
        Map script beats to graph states and edges.

        Demo beats consume edges in order. Non-demo state beats attach to the
        current state; multiple consecutive state beats can share a state.
        """
        states = [start_state] + intermediate_states + [end_state]
        edge_idx = 0
        state_idx = 0
        narration_beats: List[NarrationBeat] = []

        for beat in beats:
            if beat.kind == "demo":
                if edge_idx >= len(edges):
                    beat.attaches_to = "state"
                    beat.target_id = end_state.state_id
                    state_idx = len(states) - 1
                else:
                    beat.attaches_to = "edge"
                    beat.target_id = edges[edge_idx].edge_id
                    to_id = edges[edge_idx].to_state_id
                    edge_idx += 1
                    for i, s in enumerate(states):
                        if s.state_id == to_id:
                            state_idx = i
                            break
            else:
                beat.attaches_to = "state"
                beat.target_id = states[min(state_idx, len(states) - 1)].state_id

            narration_beats.append(
                NarrationBeat(
                    beat_id=beat.beat_id,
                    attaches_to=beat.attaches_to,  # type: ignore[arg-type]
                    target_id=beat.target_id or "",
                    text=beat.text,
                    word_count=len(beat.text.split()),
                    start_time=0.0,
                    end_time=0.0,
                    observed_state=beat.observed_state,
                )
            )

        return narration_beats