"""
compiler/schemas.py

The canonical data model for the reverse-engineered course compiler.
Every other module — discovery, pathfinding, narration, rendering —
reads and writes these shapes.
"""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field


class ScreenState(BaseModel):
    """
    A single verified screenshot of the application at a point in time.
    This is the atom of truth. Everything else is derived from it.
    """
    state_id: str
    screenshot_path: str          # Absolute path to the verified PNG
    timestamp: float              # Seconds from recording start (or 0.0 if pre-recording)
    application: Literal[
        "metabase",
        "db_browser_sqlite",
        "excel",
        "power_bi",
        "mysql_workbench",
    ]
    # Platform-specific metadata: DOM snapshot for Playwright, cursor coords for computer-use
    platform_snapshot: dict = Field(default_factory=dict)
    # Human-readable description of what is visible (for quick review)
    visual_summary: str = ""


class ActionEdge(BaseModel):
    """
    One atomic action that moves the application from one ScreenState to another.
    CRITICAL RULE: payload must be a single continuous value.
    Never split a SQL query, formula, or date range across multiple edges.
    """
    edge_id: str
    from_state_id: str
    to_state_id: str
    action_type: Literal[
        "click", "type", "select", "scroll", "hotkey", "wait", "api_seed"
    ]
    # Target: selector string for Playwright, or {x, y, w, h} for computer-use
    target: dict
    # The continuous value entered in this step (full query, full formula, etc.)
    payload: Optional[str] = None
    # How long this action should take on screen (for pacing and narration timing)
    expected_duration: float = 2.0
    # Path to a recorded video clip of this action (optional)
    video_path: Optional[str] = None
    # Highlight style burned in during render
    highlight_style: dict = Field(
        default_factory=lambda: {"color": "#FF2D95", "width": 3}
    )


class NarrationBeat(BaseModel):
    """
    One line of narration tied to a specific state or action.
    Generated from the verified screenshot, not imagined.
    """
    beat_id: str
    # Does this beat describe a state ("We see 47 rows") or an action ("Click Apply")?
    attaches_to: Literal["state", "edge"]
    target_id: str                # The state_id or edge_id this beat describes
    text: str                     # The literal narration line
    tts_text: Optional[str] = None  # Spoken-form text sent to TTS (normalized numbers/terms)
    word_count: int = 0
    # Timing on the video timeline
    start_time: float = 0.0
    end_time: float = 0.0
    # If TTS is used, the rendered audio file path
    audio_path: Optional[str] = None
    # Structured observed UI state summary captured during execution
    observed_state: Optional[dict] = None
    # True when this beat should be merged into the previous beat by the renderer
    merge: bool = False


class ExecutionGraph(BaseModel):
    """
    The complete lesson artifact.
    This is the single source of truth. Video, audio, and reference scripts are derived from it.
    """
    graph_id: str                 # e.g., "regional_order_analysis_v3"
    learning_objective: str       # The pedagogical contract: "After this video, the learner can..."
    application: str
    # The verified end state (the goal) and the blank starting state
    end_state: ScreenState
    start_state: ScreenState
    # All intermediate states and actions that form the path
    states: List[ScreenState] = Field(default_factory=list)
    edges: List[ActionEdge] = Field(default_factory=list)
    narration_beats: List[NarrationBeat] = Field(default_factory=list)
    # Telemetry: filled in by the discovery harness
    generation_cost_usd: Optional[float] = None
    reliability_score: Optional[float] = None  # 0.0 to 1.0
    render_status: Optional[str] = None


class DiscoveryResult(BaseModel):
    """
    Output of the EndStateDiscovery harness.
    Tells you whether the objective was reached reliably and how much it cost.
    """
    success: bool
    locked_state: Optional[ScreenState] = None
    attempts: int
    successful_attempts: int
    reliability_score: float      # successful_attempts / attempts
    mean_cost_usd: float
    std_cost_usd: float
    # The full log of every attempt (for debugging)
    attempt_logs: List[dict] = Field(default_factory=list)
    # Exact editor text read back via accessibility at the end of a successful run.
    final_editor_content: Optional[str] = None


class EnvironmentProfile(BaseModel):
    """
    Runtime description of the target application environment.

    All app-specific literals live in the profile so that discovery, vision, and
    rendering code can remain application-agnostic. The scout produces a profile
    for each (course, video) pair.
    """

    application: str                       # e.g. "db_browser_sqlite"
    app_name: str                          # Human-facing app name
    focus_target: str                      # AppleScript / activation target
    window_title_hint: str = ""
    # Generic UI landmark -> VLM prompt or human description
    landmarks: Dict[str, str] = Field(default_factory=dict)
    # How to obtain verified ground-truth facts for this app
    grounding_channel: Dict[str, Any] = Field(default_factory=dict)
    # Allowed action types for this environment
    action_vocabulary: List[str] = Field(
        default_factory=lambda: ["click", "type", "key", "scroll", "wait"]
    )
    # Execution semantics for this app
    execute_scope: Literal["whole_script", "current_statement", "none"] = "current_statement"
    comment_syntax: Dict[str, str] = Field(
        default_factory=lambda: {"line": "--", "block_start": "/*", "block_end": "*/"}
    )
    # Generic visual error signature. Dict keys:
    #   status_region: {x, y, w, h} normalized coords
    #   color_ranges: list of {lower: [H,S,V], upper: [H,S,V]} OpenCV HSV bounds
    #   min_area_ratio: minimum matching fraction of the region
    #   text_hint: optional human-readable hint for logs/VLM prompts
    error_signature: Optional[Dict[str, Any]] = None
    # Whitespace handling for editable regions. "exact" preserves leading
    # indentation byte-for-byte; "normalized" allows fuzzy whitespace compare.
    whitespace_policy: Literal["exact", "normalized"] = "normalized"
    # Observed facts populated by the scout
    tables: List[str] = Field(default_factory=list)
    columns: Dict[str, List[str]] = Field(default_factory=dict)
    row_counts: Dict[str, int] = Field(default_factory=dict)
    query_results: Dict[str, Any] = Field(default_factory=dict)
    ui: Dict[str, Any] = Field(default_factory=dict)
