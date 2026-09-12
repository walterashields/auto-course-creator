"""
C36: semantic sub-element choreography targets.

The planner (LessonBuilder) names the specific visible element a sentence
references — a clause line, the comment block, a results-grid region, a tab —
as a stable semantic name instead of collapsing everything to the container's
center. The environment profile plus cached AX geometry resolve those names
deterministically to logical screen points. No VLM is involved; a bounded,
cached VLM fallback remains available upstream for names the profile cannot
resolve.

Semantic name grammar (all coordinates are resolved, never stored):

    sql-editor:body                       editor text-area center
    sql-editor:line:<n>                   1-based editor line <n>
    sql-editor:comment-block:<a>-<b>      centroid of comment lines a..b (1-based)
    results-grid:body                     results pane center
    results-grid:header                   results grid header band
    results-grid:header:<col>             one column's header cell
    results-grid:row:<n>                  1-based data row <n>
    tab:execute-sql | tab:browse-data | tab:database-structure
    toolbar:execute-sql
    schema-tree:<table>
    schema-columns:<col>
    browse-grid:<table>

`TargetGeometry` carries everything resolution needs: the cached editor and
results rects (AXPosition/AXSize of the two text areas), the calibrated
line-height constant, and cached AX points for tabs/buttons. Missing geometry
falls back to the legacy fractional landmarks so resolution never raises.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

# Calibrated editor line height (logical points). DB Browser's default editor
# font measures ~19 pt per line on the reference display; the profile may
# override via ui["editor_line_height_px"]. Calibration happens once per run
# (the agent primes its geometry cache at stage prep) and is logged.
DEFAULT_EDITOR_LINE_HEIGHT_PX = 19.0

# x offset from the editor's left edge to the text start, plus a small pad so
# the cursor lands on the text rather than the gutter.
TEXT_START_OFFSET_PX = 28.0

# Minimum distance between a deliberate gesture and the cursor's current rest
# point. Anything closer is visually a no-op.
MIN_GESTURE_SEPARATION_PX = 40.0

SEMANTIC_PREFIXES = (
    "sql-editor:",
    "results-grid:",
    "tab:",
    "toolbar:",
    "schema-tree:",
    "schema-columns:",
    "browse-grid:",
)

# Legacy fractional landmarks (fraction of the full screen) used when the AX
# geometry for a region is unavailable. These mirror the pre-C36 table so the
# fallback behavior is unchanged.
_LEGACY_FRACTIONS = {
    "sql-editor:body": (0.50, 0.39),
    "sql-editor:comment-block": (0.50, 0.29),
    "results-grid:body": (0.50, 0.75),
    "results-grid:header": (0.065, 0.62),
    "results-grid:row:1": (0.18, 0.65),
    "tab:database-structure": (0.42, 0.11),
    "tab:execute-sql": (0.60, 0.11),
    "tab:browse-data": (0.72, 0.11),
    "toolbar:execute-sql": (0.10, 0.14),
}

_SCHEMA_COLUMN_X_FRACTION = 0.65
_SCHEMA_COLUMN_Y_BASE_FRACTION = 0.27
_SCHEMA_COLUMN_Y_OFFSETS = {"firstname": 0.00, "last": 0.04, "email": 0.08}
_RESULTS_HEADER_X_OFFSETS = {"first": 0.00, "last": 0.07, "email": 0.14}


def is_semantic_target(name: str) -> bool:
    """Return True when ``name`` is a C36 semantic target."""
    return bool(name) and name.startswith(SEMANTIC_PREFIXES)


def semantic_family(name: str) -> str:
    """Return the family prefix of a semantic target (e.g. ``sql-editor``)."""
    return name.split(":", 1)[0] if is_semantic_target(name) else ""


def describe_semantic_target(name: str) -> str:
    """Human description of a semantic target (for logs and VLM fallback)."""
    if not is_semantic_target(name):
        return name
    if name.startswith("sql-editor:line:"):
        n = name.rsplit(":", 1)[1]
        return f"line {n} of the SQL editor"
    if name.startswith("sql-editor:comment-block:"):
        return "the comment block in the SQL editor"
    if name == "sql-editor:body":
        return "the SQL editor text area"
    if name == "results-grid:body":
        return "the result pane showing query output"
    if name == "results-grid:header":
        return "the column headers in the result pane"
    if name.startswith("results-grid:header:"):
        col = name.rsplit(":", 1)[1]
        return f"the {col} column header in the result pane"
    if name.startswith("results-grid:row:"):
        n = name.rsplit(":", 1)[1]
        return f"row {n} of the result grid"
    if name.startswith("tab:"):
        return f"the {name.split(':', 1)[1].replace('-', ' ').title()} tab"
    if name.startswith("toolbar:"):
        return "the Execute SQL toolbar button"
    if name.startswith("schema-tree:"):
        return f"the {name.split(':', 1)[1]} table in the Database Structure tree"
    if name.startswith("schema-columns:"):
        return f"the {name.split(':', 1)[1]} column under its table"
    if name.startswith("browse-grid:"):
        return f"the {name.split(':', 1)[1]} rows in the Browse Data grid"
    return name


@dataclass
class TargetGeometry:
    """Cached geometry used to resolve semantic targets.

    ``editor_rect`` / ``results_rect`` are ``(x, y, w, h)`` in macOS logical
    points (AXPosition/AXSize of the top / lower text areas). ``tab_points``
    and ``toolbar_points`` cache AX element centers by semantic key.
    ``screen`` is ``(w, h)`` for fractional fallbacks. Any field may be None /
    empty; resolution degrades to the legacy fractions instead of failing.
    """

    editor_rect: Optional[Tuple[float, float, float, float]] = None
    results_rect: Optional[Tuple[float, float, float, float]] = None
    line_height: float = DEFAULT_EDITOR_LINE_HEIGHT_PX
    tab_points: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    toolbar_points: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    screen: Tuple[float, float] = (0.0, 0.0)

    def _fraction_point(self, fx: float, fy: float) -> Tuple[float, float]:
        w, h = self.screen
        return (w * fx, h * fy)


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _editor_line_point(
    rect: Tuple[float, float, float, float], line_height: float, n: int
) -> Tuple[float, float]:
    """Point for 1-based editor line ``n``: origin + line_height * (n - 0.5).

    x sits at the text start plus a small offset; y is clamped inside the rect
    so a line number beyond the visible area still lands on the editor.
    """
    x, y, w, h = rect
    lh = max(1.0, line_height)
    line_y = y + lh * (n - 0.5)
    line_y = _clamp(line_y, y + lh * 0.5, y + max(lh * 0.5, h - lh * 0.5))
    return (x + min(TEXT_START_OFFSET_PX, max(4.0, w * 0.2)), line_y)


def resolve_semantic_target(
    name: str, geo: TargetGeometry
) -> Optional[Tuple[float, float]]:
    """Resolve a semantic target to logical points. Never raises.

    Every resolution lands on the intended sub-element when the corresponding
    geometry is cached; otherwise the legacy fractional landmark keeps the
    cursor in the right region.
    """
    if not is_semantic_target(name):
        return None

    # --- sql-editor family ---------------------------------------------------
    if name == "sql-editor:body":
        if geo.editor_rect:
            x, y, w, h = geo.editor_rect
            return (x + w / 2.0, y + h / 2.0)
        return geo._fraction_point(*_LEGACY_FRACTIONS["sql-editor:body"])
    m = re.fullmatch(r"sql-editor:line:(\d+)", name)
    if m:
        n = max(1, int(m.group(1)))
        if geo.editor_rect:
            return _editor_line_point(geo.editor_rect, geo.line_height, n)
        return geo._fraction_point(*_LEGACY_FRACTIONS["sql-editor:body"])
    m = re.fullmatch(r"sql-editor:comment-block:(\d+)-(\d+)", name)
    if m:
        a, b = int(m.group(1)), int(m.group(2))
        if geo.editor_rect:
            mid = (a + b) / 2.0
            return _editor_line_point(geo.editor_rect, geo.line_height, max(1, int(round(mid))))
        return geo._fraction_point(*_LEGACY_FRACTIONS["sql-editor:comment-block"])

    # --- results-grid family ---------------------------------------------------
    if name == "results-grid:body":
        if geo.results_rect:
            x, y, w, h = geo.results_rect
            return (x + w / 2.0, y + h / 2.0)
        return geo._fraction_point(*_LEGACY_FRACTIONS["results-grid:body"])
    if name == "results-grid:header":
        if geo.results_rect:
            x, y, w, h = geo.results_rect
            return (x + w / 2.0, y + min(16.0, max(8.0, h * 0.08)))
        return geo._fraction_point(*_LEGACY_FRACTIONS["results-grid:header"])
    m = re.fullmatch(r"results-grid:header:(.+)", name)
    if m:
        col = m.group(1).lower()
        off = 0.0
        for key, val in _RESULTS_HEADER_X_OFFSETS.items():
            if key in col:
                off = val
                break
        if geo.results_rect:
            x, y, w, h = geo.results_rect
            return (x + w * (0.10 + off), y + min(16.0, max(8.0, h * 0.08)))
        fx, fy = _LEGACY_FRACTIONS["results-grid:header"]
        return geo._fraction_point(fx + off, fy)
    m = re.fullmatch(r"results-grid:row:(\d+)", name)
    if m:
        n = max(1, int(m.group(1)))
        if geo.results_rect:
            x, y, w, h = geo.results_rect
            header = min(16.0, max(8.0, h * 0.08))
            row_h = max(10.0, geo.line_height)
            row_y = y + header + row_h * (n - 0.5)
            row_y = _clamp(row_y, y + header + row_h * 0.5, y + max(row_h * 0.5, h - row_h * 0.5))
            return (x + min(120.0, w * 0.15), row_y)
        if n == 1:
            return geo._fraction_point(*_LEGACY_FRACTIONS["results-grid:row:1"])
        return geo._fraction_point(*_LEGACY_FRACTIONS["results-grid:body"])

    # --- tabs / toolbar --------------------------------------------------------
    if name.startswith("tab:"):
        key = name.split(":", 1)[1]
        if key in geo.tab_points:
            return geo.tab_points[key]
        return geo._fraction_point(*_LEGACY_FRACTIONS.get(f"tab:{key}", (0.60, 0.11)))
    if name == "toolbar:execute-sql":
        point = geo.toolbar_points.get("execute-sql")
        if point is not None:
            return point
        return geo._fraction_point(*_LEGACY_FRACTIONS["toolbar:execute-sql"])

    # --- schema tree / browse grid ---------------------------------------------
    if name.startswith("schema-tree:"):
        return geo._fraction_point(0.08, 0.27)
    if name.startswith("schema-columns:"):
        col = name.split(":", 1)[1].lower()
        off = 0.0
        for key, val in _SCHEMA_COLUMN_Y_OFFSETS.items():
            if key in col:
                off = val
                break
        return geo._fraction_point(_SCHEMA_COLUMN_X_FRACTION, _SCHEMA_COLUMN_Y_BASE_FRACTION + off)
    if name.startswith("browse-grid:"):
        return geo._fraction_point(0.50, 0.76)

    return None


def semantic_tab_name(human_tab: str) -> str:
    """Map a human tab description or tab key to its semantic name."""
    lowered = human_tab.lower().replace("_", " ")
    if "database structure" in lowered:
        return "tab:database-structure"
    if "browse data" in lowered:
        return "tab:browse-data"
    return "tab:execute-sql"


def _content_bounds_for(name: str) -> Optional[Tuple[str, int, int]]:
    """Family and line bounds used when refining a target to a distinct point.

    Returns (family, min_line, max_line) where the line range comes from the
    cached geometry (visible lines) for editor targets.
    """
    if name.startswith("sql-editor:line:"):
        return ("line", 1, 10_000)
    return (semantic_family(name), 0, 0)


def distinct_alternatives(
    name: str, max_alternatives: int = 8
) -> List[str]:
    """Candidate retargets for ``name``, most preferred first.

    Used by the seam contract and the planner's distinctness rule: when the
    resolved point of ``name`` would rest within MIN_GESTURE_SEPARATION_PX of
    the cursor, these are the next-best distinct sub-points of the same
    element family (a different line, a different region).
    """
    if not is_semantic_target(name):
        return []
    alts: List[str] = []
    m = re.fullmatch(r"sql-editor:line:(\d+)", name)
    if m:
        n = int(m.group(1))
        for delta in (1, -1, 2, -2, 3, -3, 4, -4, 5, -5, 6, -6):
            candidate = f"sql-editor:line:{max(1, n + delta)}"
            if candidate != name and candidate not in alts:
                alts.append(candidate)
            if len(alts) >= max_alternatives:
                break
        return alts
    family_order = [
        (
            "sql-editor:body",
            ["sql-editor:comment-block:1-5", "sql-editor:line:1"],
        ),
        (
            "sql-editor:comment-block",
            ["sql-editor:body", "sql-editor:line:6", "sql-editor:line:1"],
        ),
        (
            "results-grid:body",
            ["results-grid:header", "results-grid:row:1"],
        ),
        (
            "results-grid:header",
            ["results-grid:body", "results-grid:row:1"],
        ),
        (
            "results-grid:row",
            ["results-grid:body", "results-grid:header"],
        ),
    ]
    for prefix, candidates in family_order:
        if name == prefix or name.startswith(prefix + ":"):
            for candidate in candidates:
                if candidate != name and candidate not in alts:
                    alts.append(candidate)
            break
    return alts[:max_alternatives]


# --- Nominal planning geometry -----------------------------------------------

# Fraction-of-screen nominal editor/results rects for plan-time distinctness
# decisions. Only relative distances matter here; the runtime resolver uses the
# real AX rects, and both share the same line-height constant.
NOMINAL_EDITOR_RECT_FRACTIONS = (0.055, 0.225, 0.89, 0.33)
NOMINAL_RESULTS_RECT_FRACTIONS = (0.055, 0.60, 0.89, 0.28)


def nominal_geometry(screen: Tuple[float, float]) -> TargetGeometry:
    """Plan-time geometry: fractional rects on ``screen`` (w, h)."""
    w, h = screen
    ex, ey, ew, eh = NOMINAL_EDITOR_RECT_FRACTIONS
    rx, ry, rw, rh = NOMINAL_RESULTS_RECT_FRACTIONS
    return TargetGeometry(
        editor_rect=(w * ex, h * ey, w * ew, h * eh),
        results_rect=(w * rx, h * ry, w * rw, h * rh),
        line_height=DEFAULT_EDITOR_LINE_HEIGHT_PX,
        screen=screen,
    )


def distance(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    """Euclidean distance between two points."""
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5
