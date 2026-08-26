#!/usr/bin/env python3
"""
compiler/pathfinder.py

ReversePathfinder: reconstructs an ExecutionGraph from a successful DiscoveryResult.
It reads the discovery telemetry JSONL, locates every intermediate screenshot by
its sha256 hash, and builds ScreenStates + ActionEdges from start state to end state.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .graph_store import GraphStore
from .schemas import ActionEdge, DiscoveryResult, ExecutionGraph, ScreenState


def _parse_click_coordinates(action_text: str) -> Tuple[int, int]:
    """Parse 'click at (758, 104)' into (758, 104)."""
    match = re.search(r"click at \((\d+(?:\.\d+)?),\s*(\d+(?:\.\d+)?)\)", action_text)
    if not match:
        raise ValueError(f"Could not parse click coordinates from: {action_text!r}")
    return int(round(float(match.group(1)))), int(round(float(match.group(2))))


def _normalize_target(target: Any) -> Optional[Dict[str, Any]]:
    """Convert a stored target into the ActionEdge target dict format."""
    if target is None:
        return None
    if isinstance(target, dict):
        return {k: int(round(float(v))) for k, v in target.items() if k in ("x", "y", "w", "h")}
    if isinstance(target, (list, tuple)) and len(target) >= 2:
        return {"x": int(round(float(target[0]))), "y": int(round(float(target[1])))}
    return None


def _parse_type_action(action_text: str) -> Tuple[str, Optional[Tuple[int, int]]]:
    """
    Parse 'type "West" at (100, 200)' into ("West", (100, 200)).
    The DOTALL flag allows multi-line SQL queries in the payload.
    Returns ("", None) if no text/coordinates found.
    """
    text_match = re.search(r"type\s+['\"](.+?)['\"]", action_text, re.DOTALL)
    payload = text_match.group(1) if text_match else ""
    coord_match = re.search(r"at\s+\((\d+(?:\.\d+)?),\s*(\d+(?:\.\d+)?)\)", action_text)
    coords = None
    if coord_match:
        coords = (
            int(round(float(coord_match.group(1)))),
            int(round(float(coord_match.group(2)))),
        )
    return payload, coords


def _action_to_edge(
    action_text: Optional[str], target: Optional[Any] = None
) -> Tuple[str, Dict[str, Any], Optional[str]]:
    """Return (action_type, target_dict, payload) for a single action."""
    normalized = _normalize_target(target)

    if not action_text:
        return "wait", {}, None

    action_text_lower = action_text.lower()

    if action_text_lower.startswith("type"):
        payload, coords = _parse_type_action(action_text)
        if normalized:
            return "type", normalized, payload
        if coords:
            return "type", {"x": coords[0], "y": coords[1]}, payload
        return "type", {}, payload

    if action_text_lower.startswith("press"):
        # "press 'Return'" -> treat as hotkey.
        payload = action_text.split(" ", 1)[1].strip().strip("'\"") if " " in action_text else ""
        return "hotkey", {}, payload

    if action_text_lower.startswith("click") or action_text_lower.startswith("double-click"):
        if normalized:
            return "click", normalized, None
        x, y = _parse_click_coordinates(action_text)
        return "click", {"x": x, "y": y}, None

    if action_text_lower.startswith("wait"):
        return "wait", {}, None

    # Default fallback: if a target was provided, treat it as a click.
    if normalized:
        return "click", normalized, None

    return "wait", {}, None


class ReversePathfinder:
    """
    Given a successful discovery result (locked end state + telemetry),
    derive the complete ExecutionGraph from start state to end state.
    """

    def __init__(self, discovery_result: DiscoveryResult):
        if not discovery_result.success:
            raise ValueError("DiscoveryResult must be successful to build a graph.")
        if not discovery_result.locked_state:
            raise ValueError("DiscoveryResult is missing a locked end state.")

        self.discovery_result = discovery_result
        self.end_state: ScreenState = discovery_result.locked_state
        self.output_dir: Path = Path(self.end_state.screenshot_path).parent
        self.telemetry_path: Path = self.output_dir / "telemetry.jsonl"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build_graph(self, graph_id: str, learning_objective: str) -> ExecutionGraph:
        """
        Parse discovery telemetry and construct an ExecutionGraph from start
        state to locked end state.  The graph is saved to the GraphStore.
        """
        run_logs = self._load_run_logs()
        if not run_logs:
            raise ValueError("No telemetry logs found for the successful discovery run.")

        application = self.end_state.application
        n = len(run_logs)

        # The first screenshot in the run is the start state.
        start_state = self._state_from_log(run_logs[0], "state_000", application)

        # Intermediate states are everything between start and end.
        intermediate_states: List[ScreenState] = []
        for i in range(1, n - 1):
            state_id = f"state_{i:03d}"
            intermediate_states.append(self._state_from_log(run_logs[i], state_id, application))

        # Build edges from one state to the next using the action recorded on the
        # *destination* log.  Log[i].action_taken produced the transition from
        # state[i-1] -> state[i].
        edges: List[ActionEdge] = []
        for i in range(1, n):
            from_state_id = f"state_{i - 1:03d}"
            to_state_id = (
                self.end_state.state_id if i == n - 1 else f"state_{i:03d}"
            )
            action_text = run_logs[i].get("action_taken")
            action_target = run_logs[i].get("target")
            action_type, target, payload = _action_to_edge(action_text, action_target)
            video_path = run_logs[i].get("video_path")

            edges.append(
                ActionEdge(
                    edge_id=f"edge_{i:03d}",
                    from_state_id=from_state_id,
                    to_state_id=to_state_id,
                    action_type=action_type,  # type: ignore[arg-type]
                    target=target,
                    payload=payload,
                    expected_duration=2.0,
                    video_path=video_path,
                )
            )

        # Assemble the graph.
        graph = ExecutionGraph(
            graph_id=graph_id,
            learning_objective=learning_objective,
            application=application,
            start_state=start_state,
            end_state=self.end_state,
            states=intermediate_states,
            edges=edges,
            narration_beats=[],
            generation_cost_usd=round(
                sum(log.get("cost_usd", 0.0) for log in run_logs), 6
            ),
            reliability_score=self.discovery_result.reliability_score,
        )

        self._verify_graph(graph)

        store = GraphStore()
        store.save(graph)

        return graph

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_run_logs(self) -> List[Dict[str, Any]]:
        """Read telemetry JSONL and return only the logs for the successful run."""
        if not self.telemetry_path.exists():
            # Fallback to the in-memory attempt logs if the file is missing.
            return list(self.discovery_result.attempt_logs)

        # Identify the run_id from the final (successful) attempt.
        final_log = self.discovery_result.attempt_logs[-1]
        run_id = final_log.get("run_id")

        run_logs: List[Dict[str, Any]] = []
        with open(self.telemetry_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    log = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if run_id is None or log.get("run_id") == run_id:
                    run_logs.append(log)

        # If filtering by run_id missed logs (e.g., older telemetry format), fall back
        # to using the attempt_logs from the DiscoveryResult directly.
        if not run_logs:
            return list(self.discovery_result.attempt_logs)

        return run_logs

    def _state_from_log(
        self, log: Dict[str, Any], state_id: str, application: str
    ) -> ScreenState:
        """Build a ScreenState from a telemetry log using its screenshot hash."""
        screenshot_hash = log.get("screenshot_sha256", "")
        if not screenshot_hash:
            raise ValueError(f"Telemetry log is missing screenshot_sha256: {log}")

        screenshot_path = (self.output_dir / f"{screenshot_hash}.png").resolve()
        if not screenshot_path.exists():
            raise FileNotFoundError(
                f"Screenshot for hash {screenshot_hash} not found at {screenshot_path}"
            )

        return ScreenState(
            state_id=state_id,
            screenshot_path=str(screenshot_path),
            timestamp=0.0,
            application=application,  # type: ignore[arg-type]
            platform_snapshot={
                "screenshot_sha256": screenshot_hash,
                "attempt": log.get("attempt"),
                "reason": log.get("reason", ""),
                "api_width_px": log.get("api_width_px"),
                "api_height_px": log.get("api_height_px"),
            },
            visual_summary=log.get("reason", ""),
        )

    @staticmethod
    def _verify_graph(graph: ExecutionGraph) -> None:
        """Sanity-check the constructed graph."""
        if graph.edges:
            if graph.edges[0].from_state_id != graph.start_state.state_id:
                raise ValueError(
                    "First edge does not originate from the start state: "
                    f"{graph.edges[0].from_state_id} != {graph.start_state.state_id}"
                )
            if graph.edges[-1].to_state_id != graph.end_state.state_id:
                raise ValueError(
                    "Last edge does not terminate at the end state: "
                    f"{graph.edges[-1].to_state_id} != {graph.end_state.state_id}"
                )

        state_ids = {graph.start_state.state_id, graph.end_state.state_id}
        state_ids.update(s.state_id for s in graph.states)
        if len(state_ids) != len(graph.states) + 2:
            raise ValueError("Duplicate state IDs detected in graph.")

        edge_ids = {e.edge_id for e in graph.edges}
        if len(edge_ids) != len(graph.edges):
            raise ValueError("Duplicate edge IDs detected in graph.")
