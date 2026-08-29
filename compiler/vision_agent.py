#!/usr/bin/env python3
"""
compiler/vision_agent.py

Vision-Language Model (VLM) agent for dynamic UI interaction.

The agent captures screenshots, sends them to Claude (Anthropic API), and acts on
the returned coordinates. It replaces brittle hard-coded coordinate recipes with
a model that locates UI elements on the actual screen pixels.
"""

from __future__ import annotations

import base64
import io
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import anthropic
import cv2
import numpy as np
import pyautogui
from PIL import Image

from .schemas import EnvironmentProfile
from .frame_analysis import detect_error_signature

TARGET_LONG_EDGE = 1568
DEFAULT_MODEL = os.environ.get("DISCOVERY_MODEL", "claude-sonnet-5")


def _default_profile() -> EnvironmentProfile:
    """Default DB Browser for SQLite profile for callers that do not inject one."""
    return EnvironmentProfile(
        application="db_browser_sqlite",
        app_name="DB Browser for SQLite",
        focus_target="DB Browser for SQLite",
        window_title_hint="DB Browser for SQLite",
        landmarks={
            "editor": "the editable SQL text area in the Execute SQL tab",
            "run_button": "the Execute SQL toolbar button (blue play triangle / right-pointing arrow icon) above the SQL editor",
            "result_pane": "the lower result pane showing query output",
            "result_tab": "the Result tab in the lower results pane",
            "execute_tab": "Execute SQL tab",
            "browse_tab": "Browse Data tab",
        },
        grounding_channel={"type": "sqlite3"},
        action_vocabulary=[
            "click", "type", "type_block", "append_block", "type_segments",
            "key", "run_query", "wait", "verify", "scroll",
            "move_cursor", "select_text", "highlight",
        ],
        execute_scope="whole_script",
        comment_syntax={"line": "--", "block_start": "/*", "block_end": "*/"},
        error_signature={
            "status_region": {"x": 0.0, "y": 0.80, "w": 1.0, "h": 0.20},
            "color_ranges": [
                {"lower": [0, 100, 50], "upper": [10, 255, 255]},
                {"lower": [160, 100, 50], "upper": [180, 255, 255]},
            ],
            "min_area_ratio": 0.02,
            "text_hint": "red error band in the status region containing the text 'syntax error'",
        },
    )


@dataclass
class VisionAgentResult:
    """Structured result from a VLM call."""

    text: str = ""
    action: Optional[Dict[str, Any]] = None
    input_tokens: int = 0
    output_tokens: int = 0
    latency: float = 0.0


class FocusLostError(Exception):
    """Raised when DB Browser for SQLite cannot be kept frontmost."""


class VisionAgent:
    """
    Use a vision model to see the screen and produce UI actions.

    Coordinate convention:
    - Screenshots are resized so their longest edge is at most TARGET_LONG_EDGE.
    - The model returns points in this resized (API) coordinate space.
    - The agent scales those points back to macOS logical points for pyautogui.
    """

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        output_dir: Optional[str] = None,
        profile: Optional[EnvironmentProfile] = None,
    ):
        self.client = anthropic.Anthropic()
        self.model = model
        self.output_dir = output_dir
        self.profile = profile or _default_profile()
        self.scale_to_logical = 1.0
        self.last_api_size: Tuple[int, int] = (0, 0)
        self.last_raw_image: Optional[Image.Image] = None
        self.last_api_image: Optional[Image.Image] = None
        self.recording = False
        self._last_executed_statement: str = ""

    # ------------------------------------------------------------------
    # Core screenshot / scaling helpers
    # ------------------------------------------------------------------

    def screenshot(self) -> str:
        """Capture the screen, resize for the API, and return base64 PNG."""
        raw_img = pyautogui.screenshot()
        self.last_raw_image = raw_img
        raw_w, raw_h = raw_img.size

        long_edge = max(raw_w, raw_h)
        resize_scale = min(1.0, TARGET_LONG_EDGE / long_edge)
        api_w = int(raw_w * resize_scale)
        api_h = int(raw_h * resize_scale)
        api_img = raw_img.resize((api_w, api_h), Image.Resampling.LANCZOS)
        self.last_api_image = api_img
        self.last_api_size = (api_w, api_h)

        # Scale from API coordinates to macOS logical points for pyautogui.
        logical_w, _ = pyautogui.size()
        self.scale_to_logical = logical_w / api_w if api_w else 1.0

        buf = io.BytesIO()
        api_img.save(buf, format="PNG")
        return base64.standard_b64encode(buf.getvalue()).decode("utf-8")

    def save_screenshot(self, name: str) -> Optional[str]:
        """Save the last raw screenshot to disk if an output directory is set."""
        if not self.output_dir or self.last_raw_image is None:
            return None
        out_path = os.path.join(self.output_dir, name)
        self.last_raw_image.save(out_path)
        return out_path

    def _api_to_logical(self, x: float, y: float) -> Tuple[int, int]:
        """Convert API screenshot coordinates to macOS logical points."""
        return int(round(x * self.scale_to_logical)), int(round(y * self.scale_to_logical))

    def _scale_bbox(self, bbox: Dict[str, Any]) -> Dict[str, int]:
        """Scale a bounding box from API to logical coordinates."""
        return {
            "x": int(round(bbox["x"] * self.scale_to_logical)),
            "y": int(round(bbox["y"] * self.scale_to_logical)),
            "w": int(round(bbox.get("w", 0) * self.scale_to_logical)),
            "h": int(round(bbox.get("h", 0) * self.scale_to_logical)),
        }

    # ------------------------------------------------------------------
    # Focus discipline
    # ------------------------------------------------------------------

    @staticmethod
    def _frontmost_app_name() -> Optional[str]:
        """Return the name of the current frontmost process via System Events."""
        try:
            result = subprocess.run(
                [
                    "osascript",
                    "-e",
                    'tell application "System Events" to get name of first process whose frontmost is true',
                ],
                capture_output=True,
                text=True,
                timeout=5,
            )
            return result.stdout.strip() or None
        except Exception as exc:
            print(f"Warning: could not determine frontmost app: {exc}", file=sys.stderr)
            return None

    def _ensure_frontmost(self, max_attempts: int = 3) -> None:
        """
        Assert the target application is frontmost before any input action.

        If it is not, re-activate by application name only. After ``max_attempts``
        failed recovery attempts, raise ``FocusLostError`` so the caller can abort
        the beat. Recovery never clicks elsewhere on screen.
        """
        target = self.profile.focus_target
        for attempt in range(1, max_attempts + 1):
            frontmost = self._frontmost_app_name()
            if frontmost == target:
                return
            print(
                f"  [FOCUS] frontmost is {frontmost!r}, activating {target} "
                f"(attempt {attempt}/{max_attempts})",
                file=sys.stderr,
            )
            self._activate_target_app()
            time.sleep(0.3)
        raise FocusLostError(f"{target} could not be kept frontmost")

    def _activate_target_app(self) -> None:
        """Bring the target application to the foreground via AppleScript."""
        try:
            subprocess.run(
                ["osascript", "-e", f'tell application "{self.profile.focus_target}" to activate'],
                capture_output=True,
                timeout=3,
            )
            time.sleep(0.2)
        except Exception:
            pass

    # ------------------------------------------------------------------
    # VLM communication
    # ------------------------------------------------------------------

    def _call_vlm(
        self,
        prompt: str,
        expect_json: bool = False,
        max_tokens: int = 1024,
    ) -> VisionAgentResult:
        """Send the current screenshot plus prompt to Claude and parse the reply."""
        b64 = self.screenshot()
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": b64,
                        },
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        started = time.time()
        response = self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            messages=messages,
        )
        latency = time.time() - started

        text_parts = [block.text for block in response.content if block.type == "text"]
        full_text = "\n".join(text_parts).strip()

        usage = getattr(response, "usage", None) or {}
        input_tokens = int(
            getattr(usage, "input_tokens", None) or usage.get("input_tokens", 0)
        )
        output_tokens = int(
            getattr(usage, "output_tokens", None) or usage.get("output_tokens", 0)
        )

        action = None
        if expect_json:
            action = self._extract_json_action(full_text)

        return VisionAgentResult(
            text=full_text,
            action=action,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency=latency,
        )

    @staticmethod
    def _extract_json_action(text: str) -> Optional[Dict[str, Any]]:
        """Pull the first JSON action object out of the model response."""
        fenced = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, re.DOTALL)
        if fenced:
            try:
                parsed = json.loads(fenced.group(1))
                if isinstance(parsed, dict) and "action" in parsed:
                    return parsed
            except json.JSONDecodeError:
                pass

        for match in re.finditer(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", text):
            try:
                parsed = json.loads(match.group(0))
                if isinstance(parsed, dict) and "action" in parsed:
                    return parsed
            except json.JSONDecodeError:
                continue
        return None

    # ------------------------------------------------------------------
    # Public action methods
    # ------------------------------------------------------------------

    def find_and_click(self, instruction: str, element_description: str) -> bool:
        """
        Ask the VLM to locate an element and click its center.

        Args:
            instruction: High-level instruction (e.g., "Open the Customers table").
            element_description: What to click (e.g., "Browse Data tab").
        """
        self._ensure_frontmost()

        prompt = (
            f"You are a UI automation assistant controlling {self.profile.app_name}.\n"
            f"Task: {instruction}\n"
            f"Find and click the center of this element: {element_description}\n\n"
            "Return ONLY a JSON object with this exact shape:\n"
            '{"action": "click", "point": {"x": int, "y": int}, '
            '"element_type": "tab|button|column_header|table_cell|filter_box|menu_item|other", '
            '"description": "brief label"}\n\n"'
            "The point must be the center of the element in the screenshot coordinate space "
            "(top-left is 0,0; x increases right; y increases down). Do not add any other text."
        )

        result = self._call_vlm(prompt, expect_json=True)
        action = result.action
        if not action:
            print(
                f"Warning: VLM did not return a click action for '{element_description}'. "
                f"Response: {result.text[:200]}",
                file=sys.stderr,
            )
            return False

        point = action.get("point") or action
        if not isinstance(point, dict) or "x" not in point or "y" not in point:
            print(f"Warning: VLM click action missing point: {action}", file=sys.stderr)
            return False

        lx, ly = self._api_to_logical(point["x"], point["y"])
        # Reject coordinates at the extreme corners; they usually mean the VLM
        # could not find the element or DB Browser is not in focus.
        sw, sh = pyautogui.size()
        margin = 10
        if lx <= margin and ly <= margin:
            print(
                f"Warning: VLM returned corner coordinates for '{element_description}'; "
                "target application may not be in focus.",
                file=sys.stderr,
            )
            return False

        print(f"  VLM click '{element_description}' at logical ({lx}, {ly})", file=sys.stderr)

        # Animate cursor for visibility in recordings.
        pyautogui.moveTo(lx, ly, duration=0.5, tween=pyautogui.easeInOutQuad)
        pyautogui.click(lx, ly)
        time.sleep(0.5)
        return True

    def type_text(self, text: str) -> bool:
        """Type text at the current keyboard focus."""
        if not text:
            return True
        self._ensure_frontmost()
        print(f"  Typing: {text[:80]!r}", file=sys.stderr)
        # pyautogui handles newlines and special characters better than AppleScript.
        # ~0.10 s/char so learners can follow live typing.
        pyautogui.typewrite(text, interval=0.10)
        time.sleep(0.2)
        return True

    def _dismiss_character_viewer(self) -> None:
        """Dismiss the macOS Character Viewer / Dictation dialog if it opened."""
        for _ in range(5):
            pyautogui.press("esc")
            time.sleep(0.1)
        # Press Return to dismiss Dictation, then Escape again.
        pyautogui.press("return")
        time.sleep(0.1)
        pyautogui.press("esc")
        time.sleep(0.1)
        # Try to close any open Character Viewer window via AppleScript.
        try:
            subprocess.run(
                [
                    "osascript",
                    "-e",
                    'tell application "System Events" to if exists window "Character Viewer" of process "CharacterPalette" then click button 1 of window "Character Viewer" of process "CharacterPalette"',
                ],
                capture_output=True,
                timeout=3,
            )
        except Exception:
            pass
        # Re-activate the target app in case dismissal shifted focus.
        self._activate_target_app()

    def _focus_editor(self) -> None:
        """Click the editor, falling back to a normalized center click."""
        print("  [TYPE BLOCK] focusing editor", file=sys.stderr)
        self._ensure_frontmost()
        # The editor only exists on the execution tab; ensure it is active
        # before trying to focus the editor area.
        execute_tab = self.profile.landmarks.get("execute_tab", "Execute SQL tab")
        self.find_and_click(f"Click the {execute_tab} tab", execute_tab)
        if not self.find_and_click("Focus the SQL editor", "SQL editor text area"):
            logical_w, logical_h = pyautogui.size()
            fx, fy = int(logical_w * 0.5), int(logical_h * 0.45)
            print(f"  [TYPE BLOCK] fallback click editor at ({fx}, {fy})", file=sys.stderr)
            pyautogui.moveTo(fx, fy, duration=0.3, tween=pyautogui.easeInOutQuad)
            pyautogui.click(fx, fy)
            time.sleep(0.3)

    def _clear_editor(self) -> None:
        """
        Focus the SQL editor, select all, and delete any existing text.

        During recording, full clears are banned. Instead, move the cursor to
        the end of the document so the next input appends.
        """
        if self.recording:
            print(
                "  [TYPE BLOCK] recording active; appending instead of clearing",
                file=sys.stderr,
            )
            self._ensure_frontmost()
            self._dismiss_character_viewer()
            self._focus_editor()
            self.press_key("cmd+end")
            time.sleep(0.1)
            return
        print("  [TYPE BLOCK] clearing editor", file=sys.stderr)
        self._ensure_frontmost()
        self._dismiss_character_viewer()
        self._focus_editor()
        self.press_key("cmd+a")
        self.press_key("delete")
        time.sleep(0.2)

    def _type_visible(self, text: str) -> None:
        """
        Type text at ~0.03 s/character so a learner can follow along.

        Deprecated for SQL: ``type_block`` now pastes to avoid mangled characters.
        Kept for short non-SQL keystrokes and legacy callers.
        """
        print(f"  [TYPE BLOCK] typing {len(text)} characters", file=sys.stderr)
        self._ensure_frontmost()
        # Some SQLite editors drop newlines when they arrive too quickly via
        # typewrite, so type each line and press Return explicitly between them.
        lines = text.split("\n")
        for idx, line in enumerate(lines):
            if idx > 0:
                pyautogui.press("return")
                time.sleep(0.1)
            if line:
                # ~0.10 s/char so learners can follow live typing.
                pyautogui.typewrite(line, interval=0.10)
        time.sleep(0.5)
        self._dismiss_character_viewer()

    @staticmethod
    def _copy_to_clipboard(text: str) -> None:
        """Copy text to the macOS clipboard using pbcopy."""
        subprocess.run(
            ["pbcopy"],
            input=text.encode("utf-8"),
            check=True,
            capture_output=True,
            timeout=5,
        )

    @staticmethod
    def _read_clipboard() -> str:
        """Return the current macOS clipboard contents using pbpaste."""
        try:
            result = subprocess.run(
                ["pbpaste"],
                capture_output=True,
                timeout=5,
            )
            return result.stdout.decode("utf-8", errors="replace")
        except Exception:
            return ""

    def _paste_text(self, text: str) -> None:
        """Select all editor text and paste from the clipboard."""
        print("  [PASTE] selecting editor text and pasting from clipboard", file=sys.stderr)
        self._ensure_frontmost()
        self._dismiss_character_viewer()
        # Ensure the editor is focused and fully selected so the paste replaces
        # rather than appends to existing text.
        self._focus_editor()
        self.press_key("cmd+a")
        time.sleep(0.1)
        original_clipboard = self._read_clipboard()
        try:
            self._copy_to_clipboard(text)
            time.sleep(0.1)
            pyautogui.hotkey("command", "v")
            time.sleep(0.4)
        finally:
            try:
                self._copy_to_clipboard(original_clipboard)
            except Exception:
                pass

    @staticmethod
    def _ensure_leading_separator(prior: str, text: str) -> str:
        """Prepend a newline when appending to non-empty content that does not
        already start with whitespace. This keeps distinct SQL blocks on their
        own lines while letting inline continuations pass through unchanged."""
        if prior.strip() and text and not text[0].isspace():
            return "\n" + text
        return text

    def _append_text(self, text: str) -> str:
        """
        Append text at the end of the editor without clearing existing content.

        The cursor is moved to the end of the document via select-all + right
        arrow (a reliable cross-editor chord), then the new text is pasted at
        the end.  A newline is inserted automatically when the editor already
        contains non-whitespace content and the new text does not begin with
        whitespace, so distinct blocks (e.g. comment blocks) stay on separate
        lines. Returns the effective text that was pasted (may include the
        leading newline).
        """
        print("  [APPEND] appending text at end of buffer", file=sys.stderr)
        self._ensure_frontmost()
        self._dismiss_character_viewer()
        # Ensure the editor is focused, read the existing content (without
        # re-clicking, which would move the insertion point), then move the
        # cursor to the end of the document so the paste appends rather than
        # landing mid-document.
        self._focus_editor()
        prior = self._read_editor_content(focus=False) or ""
        text_to_append = self._ensure_leading_separator(prior, text)
        if text_to_append != text:
            print("  [APPEND] prepending newline separator", file=sys.stderr)
        self.press_key("cmd+a")
        time.sleep(0.1)
        pyautogui.keyDown("right")
        pyautogui.keyUp("right")
        time.sleep(0.1)
        original_clipboard = self._read_clipboard()
        try:
            self._copy_to_clipboard(text_to_append)
            time.sleep(0.1)
            pyautogui.hotkey("command", "v")
            time.sleep(0.4)
        finally:
            try:
                self._copy_to_clipboard(original_clipboard)
            except Exception:
                pass
        return text_to_append

    @staticmethod
    def _normalize_editor_text(text: str) -> str:
        """Collapse whitespace, strip, and lowercase for read-back comparison."""
        return re.sub(r"\s+", " ", text).strip().lower()

    @staticmethod
    def _read_focused_element_value(process_name: str) -> Optional[str]:
        """Use AppleScript/System Events to read the value of the focused UI element."""
        script = f"""\
tell application "System Events"
    tell process {json.dumps(process_name)}
        set focusedElement to value of attribute "AXFocusedUIElement"
        tell focusedElement
            return value
        end tell
    end tell
end tell
"""
        try:
            result = subprocess.run(
                ["osascript", "-e", script],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode == 0:
                return result.stdout
        except Exception as exc:
            print(f"  [READ-BACK] accessibility read failed: {exc}", file=sys.stderr)
        return None

    def _read_editor_content_vlm(self) -> str:
        """Fallback VLM transcription of the editor content."""
        editor = self.profile.landmarks.get("editor", "the editable text area")
        prompt = (
            f"Look at {editor} in {self.profile.app_name}. "
            "Transcribe the COMPLETE text currently visible in the editor, preserving line breaks. "
            "Return ONLY the editor text, no markdown or explanation."
        )
        result = self._call_vlm(prompt, expect_json=False, max_tokens=1024)
        return result.text

    def _read_editor_content(self, focus: bool = True) -> str:
        """
        Return the EXACT full text currently in the editor.

        First tries macOS accessibility (value of the focused UI element), which
        is fast, exact, and immune to clipboard pollution. Falls back to VLM
        transcription if accessibility fails.

        Args:
            focus: When False, skip the editor focus click. Callers that have
                already positioned the cursor (e.g. ``_append_text``) should pass
                False so the read does not move the insertion point.
        """
        self._ensure_frontmost()
        if focus:
            self._focus_editor()
        process_name = self.profile.focus_target or self.profile.app_name
        content = self._read_focused_element_value(process_name)
        if content is not None:
            # Normalize line endings.
            content = content.replace("\r\n", "\n").replace("\r", "\n")
            print(f"  [READ-BACK] accessibility read {len(content)} chars", file=sys.stderr)
            return content
        print("  [READ-BACK] accessibility failed; falling back to VLM", file=sys.stderr)
        return self._read_editor_content_vlm()

    def _verify_editor_layout(self, intended: str, actual: str) -> bool:
        """
        Verify that the editor contains the intended text and that the query
        immediately follows the comment block.

        Checks:
          - Whitespace-normalized intended full text equals read-back full text.
          - The query's first SELECT line appears right after the comment block's
            last line, with at most one blank line between them.
        """
        if self._normalize_editor_text(intended) != self._normalize_editor_text(actual):
            print(
                "  [TYPE BLOCK] content mismatch",
                file=sys.stderr,
            )
            print(
                f"  [TYPE BLOCK] read-back normalized: {self._normalize_editor_text(actual)!r}",
                file=sys.stderr,
            )
            print(
                f"  [TYPE BLOCK] intended normalized:  {self._normalize_editor_text(intended)!r}",
                file=sys.stderr,
            )
            return False

        intended_lines = [
            line.strip() for line in intended.splitlines() if line.strip() != ""
        ]
        actual_lines = [
            line.strip() for line in actual.splitlines() if line.strip() != ""
        ]

        # Locate the first query line in the intended text.
        query_line = next(
            (line for line in intended_lines if re.search(r"\bSELECT\b", line, re.I)),
            None,
        )
        if not query_line:
            # No separate query; content equality is sufficient.
            return True

        try:
            query_idx_intended = intended_lines.index(query_line)
        except ValueError:
            return False

        if query_idx_intended == 0:
            return True

        comment_end_intended = intended_lines[query_idx_intended - 1]

        # Find the same two lines in the actual editor content.
        try:
            comment_end_idx = next(
                i
                for i, line in enumerate(actual_lines)
                if self._normalize_editor_text(line)
                == self._normalize_editor_text(comment_end_intended)
            )
        except StopIteration:
            print(
                "  [TYPE BLOCK] could not locate comment end in read-back",
                file=sys.stderr,
            )
            return False

        try:
            query_idx_actual = next(
                i
                for i, line in enumerate(actual_lines)
                if self._normalize_editor_text(line)
                == self._normalize_editor_text(query_line)
            )
        except StopIteration:
            print(
                "  [TYPE BLOCK] could not locate query start in read-back",
                file=sys.stderr,
            )
            return False

        # With blank lines collapsed, "at most one blank line" means the query
        # line must be the next non-empty line after the comment end.
        if query_idx_actual == comment_end_idx + 1:
            return True

        print(
            f"  [TYPE BLOCK] query not adjacent to comment "
            f"(comment_end_idx={comment_end_idx}, query_idx={query_idx_actual})",
            file=sys.stderr,
        )
        return False

    @staticmethod
    def _is_shifted_char(char: str) -> bool:
        """Return True for characters that require holding Shift on a US keyboard."""
        if char.isupper():
            return True
        return char in {
            "~", "!", "@", "#", "$", "%", "^", "&", "*", "(", ")", "_", "+",
            "{", "}", "|", ":", "\"", "<", ">", "?",
        }

    def _type_segment_cadence(self, text: str) -> str:
        """
        Enter ``text`` as a single segment at the current cursor position.

        Character-level typing proved unreliable in DB Browser for SQLite (shifted
        characters mangled, keystrokes leaked to other apps on focus loss). Each
        segment is therefore pasted deterministically from the clipboard, keeping
        the segment bound to its narration beat while guaranteeing the exact text
        appears. Newlines inside the segment are preserved by the paste.

        Returns the effective text that was appended (may include a leading
        newline separator).
        """
        self._ensure_frontmost()
        print(f"  [SEGMENT] entering {len(text)} characters", file=sys.stderr)
        effective = self._append_text(text)
        time.sleep(0.3)
        self._dismiss_character_viewer()
        return effective

    def _undo_segment(self, segment_text: str) -> None:
        """Best-effort undo of the just-typed segment using Cmd+Z."""
        print("  [SEGMENT] undoing segment for retry", file=sys.stderr)
        self._ensure_frontmost()
        # One undo usually removes the last continuous text entry in DB Browser.
        self.press_key("cmd+z")
        time.sleep(0.2)

    def _editor_texts_match(self, expected: str, actual: str) -> bool:
        """
        Strict full-buffer comparison for VLM read-back of the SQL editor.

        Lenient suffix checks are banned: the VLM must return the entire editable
        content, not just the visible tail.
        """
        expected_norm = self._normalize_editor_text(expected)
        actual_norm = self._normalize_editor_text(actual)
        return expected_norm == actual_norm

    def _verify_buffer_exact(self, intended: str, label: str = "") -> bool:
        """
        Full-buffer read-back and diff after a mutation.

        Fails loudly if the normalized editor content does not match the
        intended content exactly.
        """
        read_back = self._read_editor_content()
        if self._editor_texts_match(intended, read_back):
            print(f"  [{label}] read-back OK", file=sys.stderr)
            return True
        print(f"  [{label}] read-back mismatch", file=sys.stderr)
        print(
            f"  [{label}] intended: {self._normalize_editor_text(intended)!r}",
            file=sys.stderr,
        )
        print(
            f"  [{label}] actual:   {self._normalize_editor_text(read_back)!r}",
            file=sys.stderr,
        )
        return False

    def type_segments(
        self,
        segments: List[Dict[str, Any]],
        fallback_text: Optional[str] = None,
    ) -> bool:
        """
        Type a list of segments into the editor, verifying after each one.

        Each segment dict has:
          - text: the text to type
          - narration (optional): the narration line for the segment

        On a segment mismatch the segment is undone and retyped (up to 2 tries).
        After 2 segment failures, the failed segment is pasted in place; the
        editor is NEVER fully cleared during recording.
        """
        if not segments:
            return True

        self._ensure_frontmost()
        # Capture any content already in the editor (e.g. commented history from
        # stage-prep) so segmented typing appends rather than replacing it.
        initial = self._read_editor_content() or ""
        # Build the fallback cumulative block using the same separator logic so
        # the nuclear paste fallback matches what the editor will actually hold.
        def _build_cumulative(base: str, segs: List[Any]) -> str:
            out = base
            for s in segs:
                t = s.get("text", "") if isinstance(s, dict) else str(s)
                if out.strip() and t and not t[0].isspace():
                    t = "\n" + t
                out += t
            return out

        full_text = _build_cumulative(initial, segments)
        # Use the caller-provided intended block if available; this guarantees the
        # fallback pastes the real target text even if the live editor is corrupted.
        intended_fallback = fallback_text if fallback_text is not None else full_text
        # Move the cursor to the end of the document before appending.
        pyautogui.keyDown("command")
        pyautogui.keyDown("end")
        pyautogui.keyUp("end")
        pyautogui.keyUp("command")
        time.sleep(0.1)
        expected_sofar = initial
        for seg_idx, segment in enumerate(segments):
            text = segment.get("text", "") if isinstance(segment, dict) else str(segment)
            if not text:
                continue
            # Predict the separator the append will insert so expected_sofar
            # matches the editor content after paste.
            if expected_sofar.strip() and text and not text[0].isspace():
                text = "\n" + text
            print(
                f"  [SEGMENTS] segment {seg_idx + 1}/{len(segments)} ({len(text)} chars)",
                file=sys.stderr,
            )
            segment_ok = False
            for attempt in range(1, 3):
                self._ensure_frontmost()
                self._type_segment_cadence(text)
                if self._verify_buffer_exact(expected_sofar + text, "SEGMENTS"):
                    segment_ok = True
                    break
                print(
                    f"  [SEGMENTS] segment {seg_idx + 1} mismatch, retry {attempt}/2",
                    file=sys.stderr,
                )
                # Undo the bad segment and retype it.
                self._undo_segment(text)

            if not segment_ok:
                print(
                    f"  [SEGMENTS] segment {seg_idx + 1} failed twice; pasting segment in place",
                    file=sys.stderr,
                )
                # Paste only the failed segment at the current cursor; never clear.
                self._append_text(text)
                if self._verify_buffer_exact(expected_sofar + text, "SEGMENTS"):
                    segment_ok = True
                else:
                    # Nuclear fallback only when not recording: paste full intended block.
                    if not self.recording:
                        self._clear_editor()
                        self._paste_text(intended_fallback)
                        if self._verify_buffer_exact(intended_fallback, "SEGMENTS"):
                            print("  [SEGMENTS] paste fallback OK", file=sys.stderr)
                            return True
                    print("  [SEGMENTS] in-place repair FAILED", file=sys.stderr)
                    return False

            expected_sofar = expected_sofar + text

        print("  [SEGMENTS] all segments typed and verified", file=sys.stderr)
        return True

    def type_block(self, text: str) -> bool:
        """
        Paste a multi-line SQL block into the active editor with full-buffer
        read-back verification.

        ``text`` is the FULL block: comment header and query together. During
        recording the block is appended at the end of the editor rather than
        replacing existing content. The VLM reads back the entire editor content
        and it must match the intended text exactly.
        """
        if not text:
            return True

        self._clear_editor()
        self._paste_text(text)

        for attempt in range(1, 3):
            if self._verify_buffer_exact(text, "TYPE BLOCK"):
                print("  [TYPE BLOCK] line-adjacency OK", file=sys.stderr)
                self.press_key("esc")
                print("  [TYPE BLOCK] dismissed autocomplete", file=sys.stderr)
                return True
            print(
                f"  [TYPE BLOCK] retry {attempt}/2",
                file=sys.stderr,
            )
            self._clear_editor()
            self._paste_text(text)

        print("  [TYPE BLOCK] paste verification FAILED", file=sys.stderr)
        return False

    def paste_history_block(self, text: str) -> bool:
        """
        Paste a long commented SQL history into the editor with full read-back.

        Full-buffer verification is now required after every mutation, including
        history pastes. During recording the editor is not cleared; the history
        is appended at the end.
        """
        if not text:
            return True
        self._ensure_frontmost()
        print("  [PASTE HISTORY] pasting commented history", file=sys.stderr)
        self._clear_editor()
        self._paste_text(text)
        # Scroll to the top so the VLM can read the full history if needed.
        self.press_key("cmd+home")
        time.sleep(0.2)
        if not self._verify_buffer_exact(text, "PASTE HISTORY"):
            print("  [PASTE HISTORY] verification FAILED", file=sys.stderr)
            return False
        # Move cursor to the end so the next append_block lands after the history.
        self.press_key("cmd+end")
        time.sleep(0.2)
        print("  [PASTE HISTORY] done", file=sys.stderr)
        return True

    def append_block(self, text: str) -> bool:
        """
        Append a SQL block at the end of the editor without clearing existing text.

        Used for continuity-by-design: the commented history is already in the
        editor, and this method pastes only the new query below it. The read-back
        must match the full intended editor content exactly. A newline separator
        is inserted automatically when the prior content does not end with
        whitespace and the new block does not begin with whitespace.
        """
        if not text:
            return True

        # The caller is responsible for tracking the full intended buffer if it
        # includes prior history; we verify the new text appears at the end.
        prior = self._read_editor_content() or ""
        text_to_append = self._ensure_leading_separator(prior, text)
        intended = prior + text_to_append
        self._append_text(text_to_append)

        for attempt in range(1, 3):
            if self._verify_buffer_exact(intended, "APPEND"):
                self.press_key("esc")
                return True
            print(
                f"  [APPEND] retry {attempt}/2",
                file=sys.stderr,
            )
            # Re-read prior in case it changed, then re-append.
            prior = self._read_editor_content() or ""
            text_to_append = self._ensure_leading_separator(prior, text)
            intended = prior + text_to_append
            self._append_text(text_to_append)

        print("  [APPEND] paste verification FAILED", file=sys.stderr)
        return False

    def prepare_sql_editor(self) -> bool:
        """Ensure the SQL editor is empty and focused before a typing beat."""
        print("  [STAGE PREP] SQL editor cleared", file=sys.stderr)
        self._clear_editor()
        return True

    def scroll_result_pane_top(self) -> bool:
        """Scroll the result pane to the top row."""
        print("  [STAGE PREP] result pane scrolled to top", file=sys.stderr)
        try:
            result_pane = self.profile.landmarks.get("result_pane", "the lower result pane")
            self.find_and_click("Focus the result pane", result_pane)
        except Exception:
            logical_w, logical_h = pyautogui.size()
            fx, fy = int(logical_w * 0.5), int(logical_h * 0.75)
            pyautogui.moveTo(fx, fy, duration=0.3, tween=pyautogui.easeInOutQuad)
            pyautogui.click(fx, fy)
            time.sleep(0.3)
        self.press_key("ctrl+home")
        time.sleep(0.3)
        return True

    def dismiss_transient_ui(self) -> bool:
        """Dismiss any open transient dropdown, modal, or character viewer before capturing state."""
        dismissed = False
        try:
            if self.is_modal_or_dropdown_open():
                print("  [STAGE PREP] dismissed transient UI", file=sys.stderr)
                self.press_key("esc")
                time.sleep(0.3)
                dismissed = True
        except Exception as exc:
            print(f"Warning: transient UI dismiss check failed: {exc}", file=sys.stderr)
        self._dismiss_character_viewer()
        return dismissed

    def _extract_uncommented_sql(self, content: str) -> Tuple[Optional[str], List[Tuple[int, str]]]:
        """
        Extract the single uncommented SQL block from ``content``.

        Returns (uncommented_text, uncommented_lines) where uncommented_lines is
        a list of (line_index, stripped_text) for diagnostics. Returns (None, [])
        if no uncommented text is found.

        Both line comments (e.g. '--') and block comments (e.g. '/*' ... '*/')
        declared in the EnvironmentProfile are respected.  Whitespace-only lines
        are ignored, so blank lines inside the SQL statement do not break the
        contiguous-block check.
        """
        comment_syntax = self.profile.comment_syntax or {}
        line_prefix = comment_syntax.get("line", "--")
        block_start = comment_syntax.get("block_start", "/*")
        block_end = comment_syntax.get("block_end", "*/")
        lines = content.splitlines()
        in_block_comment = False
        uncommented_lines: List[Tuple[int, str]] = []
        for i, line in enumerate(lines):
            if not line.strip():
                continue
            stripped = line.strip()
            if in_block_comment:
                if block_end and block_end in stripped:
                    in_block_comment = False
                continue
            if block_start and stripped.startswith(block_start):
                if block_end and block_end not in stripped:
                    in_block_comment = True
                continue
            if stripped.startswith(line_prefix):
                continue
            uncommented_lines.append((i, stripped))
        if not uncommented_lines:
            return None, uncommented_lines
        return "\n".join(text for _, text in uncommented_lines), uncommented_lines

    def _verify_statement_isolation(self, current_statement: str) -> bool:
        """
        Profile-driven pre-execution verifier.

        For execute_scope == 'whole_script', all uncommented text in the editor
        must be exactly the current statement. Any extra uncommented line or
        trailing fragment is treated as an orphan and fails loudly.
        """
        scope = getattr(self.profile, "execute_scope", "current_statement")
        if scope != "whole_script":
            return True
        if not current_statement or not current_statement.strip():
            return True
        content = self._read_editor_content()
        uncommented_text, uncommented_lines = self._extract_uncommented_sql(content)
        current_norm = self._normalize_editor_text(current_statement)

        if uncommented_text is None:
            print(
                "  [ISOLATION] no uncommented statement found; refusing to execute",
                file=sys.stderr,
            )
            return False

        # The uncommented text must form exactly one logical block: between the
        # first and last uncommented non-whitespace line there must be no
        # comment-prefixed or block-comment line.  Blank/whitespace-only lines
        # are allowed inside the statement.
        uncommented_indices = {i for i, _ in uncommented_lines}
        first_idx = uncommented_lines[0][0]
        last_idx = uncommented_lines[-1][0]
        comment_syntax = self.profile.comment_syntax or {}
        line_prefix = comment_syntax.get("line", "--")
        block_start = comment_syntax.get("block_start", "/*")
        block_end = comment_syntax.get("block_end", "*/")
        for i, line in enumerate(content.splitlines()):
            if first_idx < i < last_idx and line.strip():
                stripped = line.strip()
                if (
                    stripped.startswith(line_prefix)
                    or stripped.startswith(block_start)
                    or stripped == block_end
                ):
                    continue
                if i not in uncommented_indices:
                    print(
                        "  [ISOLATION] uncommented text is split into multiple blocks",
                        file=sys.stderr,
                    )
                    return False

        uncommented_norm = self._normalize_editor_text(uncommented_text)
        # Allow equality or current statement plus a trailing fragment that is
        # a continuation of the last line (no semicolon boundary).
        if uncommented_norm == current_norm:
            return True
        if uncommented_norm.startswith(current_norm):
            trailing = uncommented_norm[len(current_norm):].strip()
            if trailing:
                print(
                    f"  [ISOLATION] trailing uncommented fragment: {trailing!r}",
                    file=sys.stderr,
                )
                return False
            return True

        print(
            "  [ISOLATION] uncommented block does not match current statement",
            file=sys.stderr,
        )
        return False

    def _result_pane_shows_error(self) -> bool:
        """Pixel-check whether the profile's error_signature is visible on screen."""
        signature = getattr(self.profile, "error_signature", None)
        if not signature:
            return False
        # Use the most recent raw screenshot if available; otherwise capture one.
        raw = self.last_raw_image
        if raw is None:
            self.screenshot()
            raw = self.last_raw_image
        if raw is None:
            return False
        bgr = cv2.cvtColor(np.array(raw), cv2.COLOR_RGB2BGR)
        result = detect_error_signature(bgr, self.profile)
        if result:
            print("  [RUN QUERY] pixel error signature detected", file=sys.stderr)
        return result

    def _repair_editor_and_rerun(self, current_statement: str) -> bool:
        """Re-paste only the current statement and run the query again."""
        print("  [RUN QUERY] repairing buffer and re-running", file=sys.stderr)
        self._clear_editor()
        self._paste_text(current_statement)
        if not self._verify_buffer_exact(current_statement, "REPAIR"):
            return False
        return self.run_query()

    def _results_visible(self) -> bool:
        """Ask the VLM whether the result pane shows query output."""
        result_pane = self.profile.landmarks.get("result_pane", "the result pane")
        check = self._call_vlm(
            f"Look at {result_pane} in {self.profile.app_name}. Does it "
            "show a populated results grid, or text saying a query finished with "
            "a row count (e.g., 'Result: N rows returned'), or text saying "
            "'Execution finished without errors'? "
            "Reply exactly YES or NO, nothing else.",
            expect_json=False,
            max_tokens=32,
        )
        return check.text.strip().upper().startswith("YES")

    def run_query(self, current_statement: str = "") -> bool:
        """
        Execute the SQL in the active editor by clicking the Execute/Run toolbar button.

        No function keys are used. The button is located by the VLM using the same
        prompting and corner-rejection logic as ``find_and_click``. If the result
        pane does not populate, the VLM is asked to click the Result tab.

        Before execution, a profile-driven verifier ensures no orphan uncommented
        fragments are present. The exact statement to be executed is derived from
        the editor so that segmented beats that built up a query across multiple
        steps execute the full cumulative SQL. After execution, if the profile's
        error_signature appears, the buffer is repaired and the query re-run once.
        """
        self._ensure_frontmost()
        editor_statement, _ = self._extract_uncommented_sql(self._read_editor_content())
        passed_statement = current_statement or self._last_executed_statement
        statement: str = passed_statement or editor_statement or ""
        # When the caller's statement is only a segment, prefer the full uncommented
        # block currently in the editor.
        if (
            editor_statement
            and statement
            and self._normalize_editor_text(editor_statement)
            != self._normalize_editor_text(statement)
        ):
            print(
                "  [RUN QUERY] using full editor statement instead of passed segment",
                file=sys.stderr,
            )
            statement = editor_statement
        if statement and not self._verify_statement_isolation(statement):
            print("  [RUN QUERY] pre-execution isolation check FAILED", file=sys.stderr)
            return False

        print("  [RUN QUERY] locating Execute/Run toolbar button", file=sys.stderr)

        run_button = self.profile.landmarks.get(
            "run_button",
            "the Execute SQL toolbar button (blue play triangle / right-pointing arrow icon)",
        )
        clicked = self.find_and_click(
            "Execute the SQL query in the editor",
            run_button,
        )

        executed = False
        if clicked:
            # Give the app time to execute and render the result pane.
            time.sleep(2.5)
            executed = True
            if self._results_visible():
                if not self._result_pane_shows_error():
                    print("  [RUN QUERY] results visible", file=sys.stderr)
                    return True
                print("  [RUN QUERY] error signature detected after run", file=sys.stderr)

        if not executed:
            print("  [RUN QUERY] results not visible; clicking Result tab", file=sys.stderr)
            result_tab = self.profile.landmarks.get("result_tab", "the Result tab")
            if self.find_and_click(
                "Show the query results",
                f"{result_tab} in {self.profile.app_name}",
            ):
                time.sleep(1.0)
                if self._results_visible() and not self._result_pane_shows_error():
                    print("  [RUN QUERY] results visible after Result tab click", file=sys.stderr)
                    return True

        # Repair path: if we executed and saw an error signature, try once more.
        if executed and statement and self._result_pane_shows_error():
            return self._repair_editor_and_rerun(statement)

        print("  Warning: VLM did not find Execute/Run button or Result tab", file=sys.stderr)
        return False

    def summarize_result_pane(self) -> Dict[str, Any]:
        """
        Ask the VLM to summarize the result pane.

        Returns a dict with columns, row_count, first_rows, and a one-sentence
        summary. row_count may be an int, a string like 'N of M', or null.
        """
        result_pane = self.profile.landmarks.get("result_pane", "the result pane")
        prompt = (
            f"Look at {result_pane} in {self.profile.app_name} and return "
            "ONLY a JSON object with this exact shape:\n\n"
            "{\n"
            '  "columns": ["col1", "col2", ...],\n'
            '  "row_count": int | "N of M" | null,\n'
            '  "first_rows": [["val1", "val2", ...], ...],\n'
            '  "summary": "one sentence describing the result"\n'
            "}\n\n"
            "Use null for unknown values. Do not add any other text."
        )
        result = self._call_vlm(prompt, expect_json=False, max_tokens=512)
        data: Dict[str, Any] = {
            "columns": [],
            "row_count": None,
            "first_rows": [],
            "summary": "",
        }
        import re as _re

        fenced = _re.search(r"```(?:json)?\s*(\{.*\})\s*```", result.text, _re.DOTALL)
        payload = fenced.group(1) if fenced else result.text
        try:
            parsed = json.loads(payload)
            if isinstance(parsed, dict):
                data.update(parsed)
        except json.JSONDecodeError:
            print(
                f"Warning: could not parse result-pane summary as JSON: {result.text[:200]}",
                file=sys.stderr,
            )
        return data

    def press_key(self, key: str) -> bool:
        """Press a single key or key chord (e.g., 'Return', 'cmd+a')."""
        key = key.strip()
        print(f"  Pressing key: {key!r}", file=sys.stderr)

        # Normalize common names to pyautogui conventions.
        normalized = key.lower()
        if normalized in ("return", "enter"):
            pyautogui.press("return")
        elif normalized == "esc":
            pyautogui.press("esc")
        elif normalized == "space":
            pyautogui.press("space")
        elif normalized == "tab":
            pyautogui.press("tab")
        elif normalized in ("delete", "backspace"):
            pyautogui.press("backspace")
        elif "+" in key:
            parts = [p.strip().lower() for p in key.split("+")]
            modifiers = []
            base = parts[-1]
            for mod in parts[:-1]:
                if mod in ("cmd", "command", "super"):
                    modifiers.append("command")
                elif mod in ("ctrl", "control"):
                    modifiers.append("ctrl")
                elif mod == "shift":
                    modifiers.append("shift")
                elif mod in ("alt", "option"):
                    modifiers.append("option")
            pyautogui.keyDown(*modifiers)
            pyautogui.keyDown(base)
            pyautogui.keyUp(base)
            pyautogui.keyUp(*modifiers)
        else:
            pyautogui.press(key.lower())

        time.sleep(0.5)
        return True

    def verify_state(self, expected_description: str) -> bool:
        """Ask the VLM whether the screen matches the expected description."""
        result_pane = self.profile.landmarks.get("result_pane", "the result pane")
        prompt = (
            f"Look at this {self.profile.app_name} screenshot. "
            f"Check {result_pane}, the status bar, and any visible numbers. "
            f"Does the current screen show: {expected_description}?\n\n"
            "Respond in exactly this format on the first line:\n"
            "YES: <concise reason>\n"
            "or\n"
            "NO: <concise reason>\n\n"
            "Do not add any other text."
        )

        result = self._call_vlm(prompt, expect_json=False)
        text = result.text
        yes_no_line = next(
            (
                line.strip()
                for line in text.splitlines()
                if line.strip().upper().startswith("YES:")
                or line.strip().upper().startswith("NO:")
            ),
            "",
        )
        if yes_no_line:
            success = yes_no_line.upper().startswith("YES")
            print(
                f"  Verification: {yes_no_line} (success={success})", file=sys.stderr
            )
            return success

        print(
            f"Warning: VLM verification did not return YES/NO. Text: {text[:200]}",
            file=sys.stderr,
        )
        return False

    def summarize_observed_state(self) -> Dict[str, Any]:
        """Ask the VLM for a structured one-line summary of the current UI state."""
        prompt = (
            f"Look at this {self.profile.app_name} screenshot and return ONLY a JSON object "
            "with this exact shape:\n\n"
            "{\n"
            '  "active_tab": "current tab name",\n'
            '  "visible_table": "table whose grid is visible, or null",\n'
            '  "row_range_text": "e.g. 1 - 20 of 20, or null",\n'
            '  "column_headers": ["col1", "col2", ...],\n'
            '  "modal_or_dropdown_open": true|false,\n'
            '  "ui_element_counts": {"tabs": 2, "buttons": 5} or null,\n'
            '  "summary": "one concise sentence describing the visible state"\n'
            "}\n\n"
            "Use null for unknown values. Do not add any other text."
        )
        result = self._call_vlm(prompt, expect_json=False)
        text = result.text
        data: Dict[str, Any] = {
            "active_tab": None,
            "visible_table": None,
            "row_range_text": None,
            "column_headers": [],
            "modal_or_dropdown_open": False,
            "ui_element_counts": None,
            "summary": "",
        }
        import re as _re
        fenced = _re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, _re.DOTALL)
        payload = fenced.group(1) if fenced else text
        try:
            parsed = json.loads(payload)
            if isinstance(parsed, dict):
                data.update(parsed)
        except json.JSONDecodeError:
            print(
                f"Warning: could not parse observed-state summary as JSON: {text[:200]}",
                file=sys.stderr,
            )
        return data

    def is_end_state_already_present(
        self, intended_end_state: str, previous_observed_state: Optional[Dict[str, Any]] = None
    ) -> Tuple[bool, str]:
        """
        Ask the VLM whether the intended end state of an action is already visible.
        Returns (already_true, suggested_narration).
        """
        prev_summary = ""
        if previous_observed_state:
            prev_summary = previous_observed_state.get("summary", "") or ""
            if not prev_summary:
                prev_summary = (
                    f"active_tab={previous_observed_state.get('active_tab')}, "
                    f"visible_table={previous_observed_state.get('visible_table')}"
                )

        prompt = (
            "You are checking whether a planned UI action is redundant. "
            "Look at the CURRENT screenshot.\n\n"
            f"Previous observed state: {prev_summary or 'None'}\n"
            f"Planned action goal: {intended_end_state}\n\n"
            "Is this goal ALREADY achieved in the current screenshot? "
            "Be STRICT: answer YES only if the exact end state is clearly visible. "
            "If the previous state did not already show this, or if there is any doubt, answer NO.\n\n"
            "Reply in exactly this format:\n"
            "YES: <concise reason>\n"
            "or\n"
            "NO: <concise reason>\n\n"
            "If YES, also provide a one-line narration describing the existing state, "
            "in first person plural, present tense, ≤20 words, starting with 'We '. "
            "Put the narration on a second line after the reason, prefixed 'NARRATION: '."
        )
        result = self._call_vlm(prompt, expect_json=False)
        text = result.text
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        already_true = any(ln.upper().startswith("YES") for ln in lines)
        narration = ""
        for ln in lines:
            if ln.upper().startswith("NARRATION:"):
                narration = ln.split(":", 1)[1].strip()
                break
        if already_true and not narration:
            narration = f"We see that the {intended_end_state.strip('.')} is already visible."
        return already_true, narration

    def is_modal_or_dropdown_open(self) -> bool:
        """Ask the VLM whether a transient dropdown/modal is open and should be dismissed."""
        prompt = (
            f"Look at this {self.profile.app_name} screenshot. "
            "Is a transient dropdown menu, modal dialog, or popup currently open on top of the main window? "
            "Ignore the main application window, side panels, and table grids. "
            "Reply exactly YES or NO, nothing else."
        )
        result = self._call_vlm(prompt, expect_json=False)
        text = result.text.strip().upper()
        return text.startswith("YES")

    def ask_recovery(self, failed_action: str) -> Optional[Dict[str, Any]]:
        """Ask the VLM what to do after a failed action."""
        prompt = (
            f"The previous UI action failed: {failed_action}\n"
            "Look at the current screenshot and return a JSON action to recover.\n\n"
            "Return ONLY one JSON object with this shape:\n"
            '{"action": "click|type|key|wait", "point": {"x": int, "y": int}, '
            '"text": "only for type/key", "element_type": "...", "description": "..."}\n\n'
            "If no recovery is possible, return: {\"action\": \"wait\", \"duration\": 1}"
        )
        result = self._call_vlm(prompt, expect_json=True)
        return result.action

    def execute_beat(
        self,
        beat_dict: Dict[str, Any],
        fallback_text: Optional[str] = None,
    ) -> bool:
        """
        Execute a single vision-agent beat.

        Expected beat_dict keys:
          - action_type: "click" | "type" | "key" | "verify" | "wait" | "sequence"
          - action_detail: human description or text to type/press
          - target (optional for type): element to click before typing
          - actions (for "sequence"): list of sub-beat dicts

        ``fallback_text`` is passed through to segmented typing so that a failed
        segment can recover by pasting the intended cumulative block instead of
        whatever happens to be in the editor.
        """
        action_type = beat_dict.get("action_type") or beat_dict.get("type")
        detail = beat_dict.get("action_detail") or beat_dict.get("detail") or ""

        if action_type == "wait":
            duration = beat_dict.get("duration", 1.5)
            time.sleep(duration)
            return True

        if action_type == "click":
            return self.find_and_click(detail, detail)

        if action_type == "type":
            target = beat_dict.get("target")
            if target:
                # Target may be a plain string or a legacy coordinate dict with a
                # human description.  Use the description when available.
                if isinstance(target, dict):
                    target_label = target.get("description") or "input field"
                else:
                    target_label = str(target)
                if not self.find_and_click(f"Focus the {target_label}", target_label):
                    return False
            return self.type_text(detail)

        if action_type == "type_block":
            text = beat_dict.get("text") or beat_dict.get("detail") or ""
            self._last_executed_statement = text
            return self.type_block(text)

        if action_type == "type_segments":
            segments = beat_dict.get("segments") or []
            full = "".join(
                (s.get("text", "") if isinstance(s, dict) else str(s)) for s in segments
            )
            self._last_executed_statement = full
            return self.type_segments(segments, fallback_text=fallback_text)

        if action_type == "append_block":
            text = beat_dict.get("text") or beat_dict.get("detail") or ""
            self._last_executed_statement = text
            return self.append_block(text)

        if action_type == "run_query":
            statement = beat_dict.get("statement") or self._last_executed_statement
            return self.run_query(current_statement=statement)

        if action_type == "summarize_result_pane":
            # This action does not move the UI; it only populates observed_state.
            # Callers are responsible for storing the returned dict.
            self.summarize_result_pane()
            return True

        if action_type == "key":
            return self.press_key(detail)

        if action_type == "verify":
            return self.verify_state(detail)

        if action_type == "move_cursor":
            return self._move_cursor_to_text(detail)

        if action_type == "select_text":
            return self._select_text_in_editor(detail)

        if action_type == "highlight":
            return self._select_text_in_editor(detail)

        if action_type == "sequence":
            sub_actions = beat_dict.get("actions", [])
            for sub in sub_actions:
                if not self.execute_beat(sub):
                    return False
            return True

        print(f"Warning: unknown vision-agent action_type {action_type!r}", file=sys.stderr)
        return False

    def _locate_editor_text(self, text: str) -> Optional[Dict[str, int]]:
        """Ask the VLM for a bounding box of ``text`` inside the editor."""
        if not text:
            return None
        editor = self.profile.landmarks.get("editor", "the editable text area")
        prompt = (
            f"Find the first occurrence of the text {text!r} inside {editor} of "
            f"{self.profile.app_name}. Return ONLY a JSON object with this exact shape: "
            '{"x": int, "y": int, "w": int, "h": int}. '
            "Use coordinates in the screenshot's pixel space. Do not add any other text."
        )
        result = self._call_vlm(prompt, expect_json=True, max_tokens=128)
        action = result.action
        if not action:
            return None
        try:
            return {
                "x": int(action["x"]),
                "y": int(action["y"]),
                "w": int(action.get("w", 0)),
                "h": int(action.get("h", 0)),
            }
        except Exception:
            return None

    def _move_cursor_to_text(self, text: str) -> bool:
        """Move the animated cursor to the first occurrence of ``text``."""
        bbox = self._locate_editor_text(text)
        if not bbox:
            return False
        lx, ly = self._api_to_logical(
            bbox["x"] + bbox["w"] / 2, bbox["y"] + bbox["h"] / 2
        )
        self._ensure_frontmost()
        pyautogui.moveTo(lx, ly, duration=0.4, tween=pyautogui.easeInOutQuad)
        time.sleep(0.2)
        return True

    def _select_text_in_editor(self, text: str) -> bool:
        """Select/highlight the first occurrence of ``text`` in the editor."""
        bbox = self._locate_editor_text(text)
        if not bbox:
            return False
        x1, y1 = self._api_to_logical(bbox["x"], bbox["y"] + bbox["h"] / 2)
        x2, y2 = self._api_to_logical(
            bbox["x"] + bbox["w"], bbox["y"] + bbox["h"] / 2
        )
        self._ensure_frontmost()
        pyautogui.moveTo(x1, y1, duration=0.4, tween=pyautogui.easeInOutQuad)
        pyautogui.mouseDown()
        pyautogui.moveTo(x2, y2, duration=0.4, tween=pyautogui.easeInOutQuad)
        pyautogui.mouseUp()
        time.sleep(0.2)
        return True

    def emphasize_element(self, description: str, select: bool = False) -> bool:
        """
        Generic emphasis action: move the animated cursor to a UI element described
        by ``description`` and optionally highlight/select it.

        The description is grounded by the VLM in the live screenshot, so this works
        for any app whose landmarks are declared in the EnvironmentProfile.
        """
        if not description:
            return False
        self._ensure_frontmost()
        prompt = (
            f"You are controlling {self.profile.app_name}. "
            f"Move the cursor to this element: {description}. "
            "Return ONLY a JSON object with this exact shape: "
            '{"action": "click", "point": {"x": int, "y": int}, "element_type": "...", "description": "..."}\n'
            "The point must be the center of the element in the screenshot coordinate space."
        )
        result = self._call_vlm(prompt, expect_json=True, max_tokens=128)
        action = result.action
        if not action:
            return False
        point = action.get("point") or action
        if not isinstance(point, dict) or "x" not in point or "y" not in point:
            return False
        lx, ly = self._api_to_logical(point["x"], point["y"])
        print(f"  [EMPHASIS] move to '{description}' at ({lx}, {ly})", file=sys.stderr)
        pyautogui.moveTo(lx, ly, duration=0.5, tween=pyautogui.easeInOutQuad)
        time.sleep(0.2)
        if select and "highlight" in self.profile.action_vocabulary:
            pyautogui.mouseDown()
            time.sleep(0.1)
            pyautogui.mouseUp()
            time.sleep(0.1)
        return True

    def perform_emphasis_actions(self, beat: ScriptBeat) -> bool:
        """
        Parse a non-demo beat's narration into clauses and perform one emphasis
        action per clause so the cursor moves while the narration plays.
        """
        if not beat or not beat.text:
            return False
        text = beat.text.strip()
        # Split on sentence boundaries and coordinating conjunctions.
        clauses = [c.strip() for c in re.split(r"(?<=[.!?])\s+|\s+and\s+|\s+while\s+|\s+as\s+", text) if c.strip()]
        # Heuristic mapping from clause content to a generic UI description.
        descriptions: List[str] = []
        for clause in clauses:
            lowered = clause.lower()
            if "result pane" in lowered or "result" in lowered:
                descriptions.append("the result pane showing query output")
            elif "editor" in lowered or "query" in lowered:
                descriptions.append("the SQL editor text area")
            elif "header" in lowered:
                descriptions.append("the column headers in the result pane")
            elif "row" in lowered or "rows" in lowered:
                descriptions.append("the rows in the result grid")
            elif "column" in lowered:
                # Try to extract column name.
                col_match = re.search(r"([A-Z][a-zA-Z]+)\s+(?:column|header)", clause)
                if col_match:
                    descriptions.append(f"the {col_match.group(1)} column header")
                else:
                    descriptions.append("the relevant column in the result pane")
            elif "comment" in lowered:
                descriptions.append("the comment block in the SQL editor")
            elif "alias" in lowered or "as " in lowered:
                descriptions.append("the aliased column headers in the result pane")
            elif "limit" in lowered:
                descriptions.append("the LIMIT clause in the SQL editor")
            elif "order by" in lowered:
                descriptions.append("the ORDER BY clause in the SQL editor")

        if not descriptions:
            descriptions.append("the main UI element relevant to the narration")

        ok = True
        for description in descriptions[:3]:  # Cap at 3 emphasis actions per beat.
            if not self.emphasize_element(description, select=False):
                ok = False
            time.sleep(0.3)
        return ok

    def frame_shows_error_signature(self, frame_path: str) -> bool:
        """Return True if ``frame_path`` matches the profile's error_signature (pixel check)."""
        signature = getattr(self.profile, "error_signature", None)
        if not signature or not Path(frame_path).exists():
            return False
        try:
            bgr = cv2.imread(str(frame_path))
            if bgr is None:
                return False
            return detect_error_signature(bgr, self.profile)
        except Exception as exc:
            print(f"Warning: error-signature frame check failed: {exc}", file=sys.stderr)
            return False

    def verify_app_visible_in_frames(
        self,
        frame_paths: List[str],
        app_name: Optional[str] = None,
    ) -> List[bool]:
        """
        Return a list of booleans indicating whether ``app_name`` is the visible
        application in each frame path. Frames are processed in batches to keep
        each VLM call small.
        """
        if not frame_paths:
            return []

        app = app_name or self.profile.app_name
        results: List[bool] = []
        batch_size = 5
        prompt = (
            f"For each screenshot in order, answer YES if the {app} window "
            "is the visible application occupying the main area of the screen. "
            "Answer NO if another application, notification banner, overlay, or "
            "desktop is visible instead. "
            "Return ONLY a JSON list of booleans in the same order, e.g. [true, false, true]. "
            "Do not add any other text."
        )

        for i in range(0, len(frame_paths), batch_size):
            batch = frame_paths[i : i + batch_size]
            content: List[Dict[str, Any]] = []
            for path in batch:
                try:
                    b64 = base64.standard_b64encode(Path(path).read_bytes()).decode("utf-8")
                    content.append(
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": b64,
                            },
                        }
                    )
                except Exception as exc:
                    print(f"Warning: could not read frame {path}: {exc}", file=sys.stderr)
                    content.append(
                        {"type": "text", "text": "[frame unreadable; assume false]"}
                    )
            content.append({"type": "text", "text": prompt})

            try:
                response = self.client.messages.create(
                    model=self.model,
                    max_tokens=256,
                    messages=[{"role": "user", "content": content}],
                )
            except Exception as exc:
                print(f"Warning: VLM frame verification failed: {exc}", file=sys.stderr)
                results.extend([False] * len(batch))
                continue

            text_parts = [block.text for block in response.content if block.type == "text"]
            full_text = "\n".join(text_parts).strip()
            parsed: List[bool] = []
            # Try to pull a JSON list out of the reply.
            fenced = re.search(r"```(?:json)?\s*(\[.*\])\s*```", full_text, re.DOTALL)
            payload = fenced.group(1) if fenced else full_text
            try:
                parsed = json.loads(payload)
                if not isinstance(parsed, list):
                    parsed = []
            except json.JSONDecodeError:
                parsed = []

            # Pad/truncate to batch size.
            parsed = [bool(x) for x in parsed]
            if len(parsed) < len(batch):
                parsed.extend([False] * (len(batch) - len(parsed)))
            elif len(parsed) > len(batch):
                parsed = parsed[: len(batch)]
            results.extend(parsed)

        return results

    def total_cost_usd(self, result: VisionAgentResult) -> float:
        """Estimate the API cost of a VLM call."""
        input_price = 3.0 / 1_000_000  # Claude 3.5 Sonnet image/text input
        output_price = 15.0 / 1_000_000
        return (result.input_tokens * input_price) + (result.output_tokens * output_price)
