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
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

import anthropic
import pyautogui
from PIL import Image

TARGET_LONG_EDGE = 1568
DEFAULT_MODEL = os.environ.get("DISCOVERY_MODEL", "claude-sonnet-5")


@dataclass
class VisionAgentResult:
    """Structured result from a VLM call."""

    text: str = ""
    action: Optional[Dict[str, Any]] = None
    input_tokens: int = 0
    output_tokens: int = 0
    latency: float = 0.0


class VisionAgent:
    """
    Use a vision model to see the screen and produce UI actions.

    Coordinate convention:
    - Screenshots are resized so their longest edge is at most TARGET_LONG_EDGE.
    - The model returns points in this resized (API) coordinate space.
    - The agent scales those points back to macOS logical points for pyautogui.
    """

    def __init__(self, model: str = DEFAULT_MODEL, output_dir: Optional[str] = None):
        self.client = anthropic.Anthropic()
        self.model = model
        self.output_dir = output_dir
        self.scale_to_logical = 1.0
        self.last_api_size: Tuple[int, int] = (0, 0)
        self.last_raw_image: Optional[Image.Image] = None
        self.last_api_image: Optional[Image.Image] = None

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
        prompt = (
            f"You are a UI automation assistant controlling DB Browser for SQLite.\n"
            f"Task: {instruction}\n"
            f"Find and click the center of this element: {element_description}\n\n"
            "Return ONLY a JSON object with this exact shape:\n"
            '{"action": "click", "point": {"x": int, "y": int}, '
            '"element_type": "tab|button|column_header|table_cell|filter_box|menu_item|other", '
            '"description": "brief label"}\n\n'
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
        print(f"  Typing: {text[:80]!r}", file=sys.stderr)
        # pyautogui handles newlines and special characters better than AppleScript.
        pyautogui.typewrite(text, interval=0.005)
        time.sleep(0.2)
        return True

    def press_key(self, key: str) -> bool:
        """Press a single key or key chord (e.g., 'Return', 'F5', 'cmd+a')."""
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
        prompt = (
            f"Look at the screenshot. Does the current screen show: {expected_description}?\n\n"
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

    def execute_beat(self, beat_dict: Dict[str, Any]) -> bool:
        """
        Execute a single vision-agent beat.

        Expected beat_dict keys:
          - action_type: "click" | "type" | "key" | "verify" | "wait" | "sequence"
          - action_detail: human description or text to type/press
          - target (optional for type): element to click before typing
          - actions (for "sequence"): list of sub-beat dicts
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

        if action_type == "key":
            return self.press_key(detail)

        if action_type == "verify":
            return self.verify_state(detail)

        if action_type == "sequence":
            sub_actions = beat_dict.get("actions", [])
            for sub in sub_actions:
                if not self.execute_beat(sub):
                    return False
            return True

        print(f"Warning: unknown vision-agent action_type {action_type!r}", file=sys.stderr)
        return False

    def total_cost_usd(self, result: VisionAgentResult) -> float:
        """Estimate the API cost of a VLM call."""
        input_price = 3.0 / 1_000_000  # Claude 3.5 Sonnet image/text input
        output_price = 15.0 / 1_000_000
        return (result.input_tokens * input_price) + (result.output_tokens * output_price)
