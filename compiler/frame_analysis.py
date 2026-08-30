#!/usr/bin/env python3
"""
compiler/frame_analysis.py

Frame-level measurement primitives used by the pipeline and the harness.
All methods are app-agnostic: every app-specific cue comes from the
EnvironmentProfile so discovery.py / vision_agent.py / renderer.py stay clean.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image

from .schemas import EnvironmentProfile


# ---------------------------------------------------------------------------
# Video frame extraction
# ---------------------------------------------------------------------------


def _video_duration(path: Path) -> float:
    """Return duration in seconds using ffprobe, or 0.0 if unavailable."""
    if shutil.which("ffprobe") is None:
        return 0.0
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(path),
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    try:
        return float(result.stdout.strip())
    except Exception:
        return 0.0


def _extract_frames_at_fps(
    video_path: Path,
    sample_fps: int = 1,
) -> List[Tuple[float, np.ndarray]]:
    """
    Extract frames from ``video_path`` at ``sample_fps``.

    Returns a list of (timestamp_seconds, bgr_frame) tuples.
    """
    duration = _video_duration(video_path)
    if duration <= 0:
        return []

    total_frames = max(1, int(round(duration * sample_fps)))
    frames: List[Tuple[float, np.ndarray]] = []

    with tempfile.TemporaryDirectory(prefix="wsda_frames_") as tmpdir:
        tmp_path = Path(tmpdir)
        # Use ffmpeg fps filter to sample evenly.
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-i",
                str(video_path),
                "-vf",
                f"fps={sample_fps}",
                "-pix_fmt",
                "rgb24",
                f"{tmp_path}/frame_%04d.png",
            ],
            check=True,
            capture_output=True,
            timeout=120,
        )
        extracted = sorted(tmp_path.glob("frame_*.png"))
        for idx, frame_path in enumerate(extracted):
            t = idx / sample_fps
            bgr = cv2.imread(str(frame_path))
            if bgr is not None:
                frames.append((t, bgr))
    return frames


# ---------------------------------------------------------------------------
# A1 — Error-signature detection
# ---------------------------------------------------------------------------


def _normalized_region_to_pixels(
    frame_h: int, frame_w: int, region: Dict[str, float]
) -> Tuple[int, int, int, int]:
    """Convert a normalized {x,y,w,h} region to pixel (x, y, w, h)."""
    x = int(round(region.get("x", 0.0) * frame_w))
    y = int(round(region.get("y", 0.0) * frame_h))
    w = int(round(region.get("w", 1.0) * frame_w))
    h = int(round(region.get("h", 1.0) * frame_h))
    x = max(0, min(x, frame_w - 1))
    y = max(0, min(y, frame_h - 1))
    w = max(1, min(w, frame_w - x))
    h = max(1, min(h, frame_h - y))
    return x, y, w, h


def detect_error_signature(
    frame_bgr: np.ndarray, profile: EnvironmentProfile
) -> bool:
    """
    Return True if the frame contains the app's declared visual error signature.

    The profile's ``error_signature`` must be a dict with:
      - status_region: {x, y, w, h} normalized coordinates
      - color_ranges: list of {lower: [H,S,V], upper: [H,S,V]} HSV bounds
      - min_area_ratio: minimum fraction of the region that must match
    """
    sig = profile.error_signature if profile.error_signature else {}
    if not sig:
        return False

    region = sig.get("status_region")
    color_ranges = sig.get("color_ranges", [])
    min_area_ratio = float(sig.get("min_area_ratio", 0.01))
    if not region or not color_ranges:
        return False

    h, w = frame_bgr.shape[:2]
    x, y, rw, rh = _normalized_region_to_pixels(h, w, region)
    roi = frame_bgr[y : y + rh, x : x + rw]
    if roi.size == 0:
        return False

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    mask = np.zeros((rh, rw), dtype=np.uint8)
    for cr in color_ranges:
        lower = np.array(cr.get("lower", [0, 0, 0]), dtype=np.uint8)
        upper = np.array(cr.get("upper", [0, 0, 0]), dtype=np.uint8)
        mask = cv2.bitwise_or(mask, cv2.inRange(hsv, lower, upper))

    match_ratio = np.count_nonzero(mask) / (roi.shape[0] * roi.shape[1])
    return match_ratio >= min_area_ratio


def count_error_signature_frames(
    video_path: Path,
    profile: EnvironmentProfile,
    sample_fps: int = 1,
) -> int:
    """Count sampled frames that contain the profile's error signature."""
    frames = _extract_frames_at_fps(video_path, sample_fps)
    return sum(1 for _t, frame in frames if detect_error_signature(frame, profile))


# ---------------------------------------------------------------------------
# A2 — Frozen-share metric
# ---------------------------------------------------------------------------


def _resize_to_width(frame_gray: np.ndarray, width: int) -> np.ndarray:
    """Resize a grayscale frame to ``width`` px wide, keeping aspect ratio."""
    h, w = frame_gray.shape[:2]
    if w == width:
        return frame_gray
    scale = width / w
    new_h = max(1, int(round(h * scale)))
    return cv2.resize(frame_gray, (width, new_h), interpolation=cv2.INTER_AREA)


def frozen_share_percent(
    video_path: Path,
    sample_fps: int = 1,
    width: int = 320,
    mse_threshold: float = 0.5,
) -> float:
    """
    Sample ``video_path`` at ``sample_fps``; grayscale and resize to ``width``.

    A second is "frozen" when consecutive-frame MSE < ``mse_threshold``.
    Returns frozen_seconds / (total_sampled_seconds - 1) * 100.
    """
    frames = _extract_frames_at_fps(video_path, sample_fps)
    if len(frames) < 2:
        return 0.0

    gray_frames: List[np.ndarray] = []
    for _t, bgr in frames:
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        gray = _resize_to_width(gray, width)
        gray_frames.append(gray.astype(np.float32))

    frozen_pairs = 0
    for prev, curr in zip(gray_frames, gray_frames[1:]):
        mse = float(np.mean((prev - curr) ** 2))
        if mse < mse_threshold:
            frozen_pairs += 1

    return (frozen_pairs / (len(gray_frames) - 1)) * 100.0


# ---------------------------------------------------------------------------
# A4 — Computed metrics and reconciliation
# ---------------------------------------------------------------------------


def _strip_markdown_syntax(text: str) -> str:
    """Remove markdown table delimiters, headers, and extra formatting."""
    # Drop table separator lines like |------|------|... without consuming
    # surrounding newlines, so adjacent table rows stay on separate lines.
    text = re.sub(r"^\|[-:|\s]+\|.*$", "", text, flags=re.MULTILINE)
    # Drop markdown links, bold, code.
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    text = re.sub(r"[*_`#]+", "", text)
    return text


def word_count_from_reference(reference_md_path: Path) -> int:
    """Return word count of narration text parsed from a reference markdown file.

    Supports the table formats produced by the renderer:
      - | Beat | Kind | Words | Text |
      - | Beat | Time | Type | Target | Words | Text |
    Sums the numeric Words column when present; otherwise counts words in the
    Text column. Escaped pipes (\\|) inside narration text are handled.
    """
    if not reference_md_path.exists():
        return 0
    text = reference_md_path.read_text(encoding="utf-8")
    words = 0
    for line in text.splitlines():
        line = line.strip()
        if not line.startswith("|") or not line.endswith("|"):
            continue
        # Restore escaped pipes so they do not break the cell split.
        safe = line.replace("\\|", "\x00")
        cells = [c.strip().replace("\x00", "|") for c in safe.strip("|").split("|")]
        # Skip header rows and markdown separator rows.
        if not cells or cells[0] in ("Beat", "------") or all(c.replace("-", "") == "" for c in cells):
            continue
        # If a numeric Words column exists, use it (formats put Words before Text).
        if len(cells) >= 4:
            # Try the cell immediately before the Text column.
            candidate = cells[-2] if cells[-1] == "Text" else ""
            if candidate and candidate.replace("Text", "").strip().isdigit():
                words += int(candidate)
                continue
            # Renderer format: | Beat | Kind | Words | Text |
            if len(cells) == 4 and cells[2].replace("Words", "").strip().isdigit():
                words += int(cells[2])
                continue
            # Long format: | Beat | Time | Type | Target | Words | Text |
            if len(cells) == 6 and cells[4].replace("Words", "").strip().isdigit():
                words += int(cells[4])
                continue
        # Fallback: count words in the Text/last column.
        narration = cells[-1]
        if narration and narration not in ("Text", "Words"):
            words += len(narration.split())
    return words


def compute_video_metrics(
    final_path: Path,
    audio_path: Optional[Path],
    reference_md_path: Path,
    profile: EnvironmentProfile,
) -> Dict[str, Any]:
    """
    Compute all video metrics using the canonical harness-backed methods.

    Returns dict with:
      duration_seconds, audio_duration_seconds, word_count,
      frozen_pct, error_frames, final_beat_text
    """
    duration = _video_duration(final_path)
    audio_duration = _video_duration(audio_path) if audio_path and audio_path.exists() else 0.0
    word_count = word_count_from_reference(reference_md_path)
    frozen_pct = frozen_share_percent(final_path)
    error_frames = count_error_signature_frames(final_path, profile)

    final_beat_text = ""
    if reference_md_path.exists():
        text = reference_md_path.read_text(encoding="utf-8")
        text = _strip_markdown_syntax(text)
        table_rows = [ln for ln in text.splitlines() if ln.strip().startswith("|")]
        if table_rows:
            # Skip header and separator rows; last data row is the final beat.
            data_rows = [
                r
                for r in table_rows[2:]
                if not re.match(r"^\|[-:|\s]+\|$", r.strip()) and "Text" not in r
            ]
            if data_rows:
                cells = [c.strip() for c in data_rows[-1].strip("|").split("|")]
                final_beat_text = cells[-1] if cells else ""

    return {
        "duration_seconds": round(duration, 3),
        "audio_duration_seconds": round(audio_duration, 3),
        "word_count": word_count,
        "frozen_pct": round(frozen_pct, 2),
        "error_frames": error_frames,
        "final_beat_text": final_beat_text,
    }


def reconcile_summary(
    renderer_summary: Dict[str, Any],
    computed: Dict[str, Any],
    tolerances: Optional[Dict[str, float]] = None,
) -> List[str]:
    """
    Return a list of discrepancies between renderer-reported numbers and the
    harness-computed numbers. Empty list means reconciliation passed.
    """
    if tolerances is None:
        tolerances = {
            # Looped clip padding can shift the final muxed duration by a few
            # seconds relative to the renderer's planned sum.
            "duration_seconds": 3.0,
            "audio_duration_seconds": 0.5,
            "word_count": 5,
            "frozen_pct": 1.0,
            "error_frames": 0,
        }

    discrepancies: List[str] = []
    for key, tol in tolerances.items():
        reported = renderer_summary.get(key)
        actual = computed.get(key)
        if reported is None or actual is None:
            continue
        if abs(float(reported) - float(actual)) > tol:
            discrepancies.append(
                f"{key}: renderer reported {reported}, computed {actual}"
            )
    return discrepancies


# ---------------------------------------------------------------------------
# Part B — Acceptance gates
# ---------------------------------------------------------------------------


def run_acceptance_gates(
    final_path: Path,
    audio_path: Optional[Path],
    reference_md_path: Path,
    profile: EnvironmentProfile,
) -> Dict[str, Any]:
    """
    Evaluate Part B acceptance gates for a single rendered video.

    Returns a dict:
      {
        "passed": bool,
        "gates": [
          {"gate": "B1_words", "value": int, "threshold": "400-700", "passed": bool},
          {"gate": "B2_av_sync", "value": float, "threshold": "<=2.0", "passed": bool},
          {"gate": "B3_frozen", "value": float, "threshold": "<15%", "passed": bool},
          {"gate": "B4_errors", "value": int, "threshold": "0", "passed": bool},
          {"gate": "B5_terminal", "value": str, "threshold": "terminal punctuation", "passed": bool},
        ],
        "metrics": { ... compute_video_metrics ... }
      }
    """
    metrics = compute_video_metrics(final_path, audio_path, reference_md_path, profile)
    duration = metrics["duration_seconds"]
    audio_duration = metrics["audio_duration_seconds"]
    word_count = metrics["word_count"]
    frozen_pct = metrics["frozen_pct"]
    error_frames = metrics["error_frames"]
    final_beat_text = metrics["final_beat_text"]

    b1 = 400 <= word_count <= 700
    b2 = abs(duration - audio_duration) <= 2.0 if audio_duration > 0 else True
    b3 = frozen_pct < 15.0
    b4 = error_frames == 0
    b5 = bool(re.search(r"[.!?]$", final_beat_text.strip()))

    gates = [
        {"gate": "B1_words", "value": word_count, "threshold": "400-700", "passed": b1},
        {"gate": "B2_av_sync", "value": round(abs(duration - audio_duration), 3), "threshold": "<=2.0", "passed": b2},
        {"gate": "B3_frozen", "value": frozen_pct, "threshold": "<15%", "passed": b3},
        {"gate": "B4_errors", "value": error_frames, "threshold": "0", "passed": b4},
        {"gate": "B5_terminal", "value": final_beat_text[-30:] if final_beat_text else "", "threshold": "terminal punctuation", "passed": b5},
    ]

    return {"passed": all(g["passed"] for g in gates), "gates": gates, "metrics": metrics}


def format_gate_table(result: Dict[str, Any]) -> str:
    """Return a printable gate table."""
    lines = ["| Gate | Value | Threshold | Passed |", "|------|-------|-----------|--------|"]
    for g in result["gates"]:
        lines.append(
            f"| {g['gate']} | {g['value']} | {g['threshold']} | {'PASS' if g['passed'] else 'FAIL'} |"
        )
    return "\n".join(lines)
