#!/usr/bin/env python3
"""
compiler/lesson_standard.py

Ingests LESSON_CONTENT_STANDARD.md and exposes it as structured prompt
fragments and a quality checklist for the lesson-first compiler pipeline.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import List, Optional


DEFAULT_STANDARD_PATH = "LESSON_CONTENT_STANDARD.md"


class LessonStandard:
    """
    Parses the SQL Essentials lesson content standard and produces prompt
    fragments that constrain curriculum design, script generation, and QA.
    """

    def __init__(self, path: str = DEFAULT_STANDARD_PATH):
        self.path = Path(path)
        self.raw_text = self._load()
        self.rules = self._extract_rules()
        self.checklist = self._extract_checklist()
        self.voice_notes = self._extract_voice_notes()

    # ------------------------------------------------------------------
    # Loading / parsing
    # ------------------------------------------------------------------

    def _load(self) -> str:
        if self.path.exists():
            return self.path.read_text(encoding="utf-8")
        return ""

    def _extract_rules(self) -> List[str]:
        """Extract the numbered rules and corollaries as a flat list."""
        if not self.raw_text:
            return []
        rules: List[str] = []
        # Match "## The five rules", "### 1. ...", "## Three more rules",
        # "### 6. ...", and corollary headings under the governing principle.
        for line in self.raw_text.splitlines():
            line = line.strip()
            if line.startswith("### ") and (
                line[4:5].isdigit()
                or line.startswith("### Corollary:")
            ):
                rules.append(line[4:].strip())
        return rules

    def _extract_checklist(self) -> List[str]:
        """Extract the authoring checklist items."""
        if not self.raw_text:
            return []
        items: List[str] = []
        in_checklist = False
        for line in self.raw_text.splitlines():
            if "## Authoring checklist" in line:
                in_checklist = True
                continue
            if in_checklist:
                if line.startswith("## "):
                    break
                stripped = line.strip()
                if stripped.startswith("- [ ]") or stripped.startswith("- [x]"):
                    items.append(re.sub(r"^- \[[ x]\]\s*", "", stripped))
        return items

    def _extract_voice_notes(self) -> List[str]:
        """Extract voice / sentence-level style guidance."""
        notes: List[str] = []
        capture = False
        for line in self.raw_text.splitlines():
            if "**Sentence-level style.**" in line:
                capture = True
                continue
            if capture:
                if line.startswith("## ") or line.startswith("### "):
                    break
                stripped = line.strip()
                if stripped:
                    notes.append(stripped)
        return notes

    # ------------------------------------------------------------------
    # Prompt fragments
    # ------------------------------------------------------------------

    def system_prompt_fragment(self) -> str:
        """
        Returns a concise system-prompt fragment describing the content standard.
        Suitable for injection into curriculum-design and script-generation prompts.
        """
        lines = [
            "You are authoring professional data-analytics training videos.",
            "Every lesson must obey the SQL Essentials content standard:",
            "",
        ]
        for rule in self.rules:
            lines.append(f"- {rule}")
        if self.checklist:
            lines.append("")
            lines.append("Authoring checklist (every item must be satisfied):")
            for item in self.checklist:
                lines.append(f"- {item}")
        if self.voice_notes:
            lines.append("")
            lines.append("Voice and style:")
            for note in self.voice_notes:
                lines.append(f"- {note}")
        return "\n".join(lines)

    def curriculum_prompt_fragment(self) -> str:
        """
        A shorter prompt fragment for curriculum design only.
        Keeps the rules that shape course structure without overwhelming the model.
        """
        return """\
Content-standard constraints for curriculum design:
- One new capability per video (capstone is the only exception).
- Stable running example across all videos in the course.
- Open with a concrete outcome statement before any hands-on action.
- Teach reasoning, not just mechanics; every step must explain why it exists.
- Connect every action to a real scenario with concrete stakes.
- Assume zero prior familiarity with the specific tool; name UI elements plainly.
- Close every explanation on a concrete, re-checkable number from the data.
- Narration must reference only what is visible and highlighted on screen.
- Environment orientation comes first, before any hands-on action, and only once.
- Recap must reference concrete numbers from the previous video.
- Preview must state the exact next capability; avoid vague words like "more" or "next steps".
"""

    def script_constraints(self) -> str:
        """
        Returns a stricter prompt fragment focused on script generation.
        """
        return """\
Script-generation constraints (from the SQL Essentials standard):

1. Open with a one-sentence outcome statement before any action.
2. Every step must explain WHY it exists, not just WHAT is clicked/typed.
3. Connect every action to a concrete scenario with real stakes.
4. Assume zero prior familiarity with the specific tool; name UI elements plainly.
5. Narration must reference only what is visible and highlighted on screen.
6. Close every explanation on a concrete, re-checkable number from the data.
7. One new capability per video; extend the same running example.
8. For first-time concepts, add extra hold time and explain what the resulting object IS and WHY it matters.

Voice: short, spoken sentences; contractions; direct second-person address; analogies for jargon; warm, professional tone.
"""

    def quality_checklist(self) -> List[str]:
        """Return the authoring checklist as a list of plain strings."""
        return list(self.checklist)


if __name__ == "__main__":
    standard = LessonStandard()
    print("Rules extracted:", len(standard.rules))
    for rule in standard.rules:
        print(f"  - {rule}")
    print("\nChecklist items:", len(standard.checklist))
    print("\nSystem prompt fragment preview (first 800 chars):")
    print(standard.system_prompt_fragment()[:800])
