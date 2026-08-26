"""
compiler/graph_store.py

Simple JSON persistence for ExecutionGraphs.
Every graph is saved as a single JSON file under compiler/graphs/.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional

from compiler.schemas import ExecutionGraph


DEFAULT_STORE_DIR = Path(__file__).parent / "graphs"


class GraphStore:
    """
    Save and load ExecutionGraphs as JSON files.
    """

    def __init__(self, store_dir: Path = DEFAULT_STORE_DIR):
        self.store_dir = Path(store_dir)
        self.store_dir.mkdir(parents=True, exist_ok=True)

    def _path(self, graph_id: str) -> Path:
        safe_id = graph_id.replace("/", "_").replace("\\", "_")
        return self.store_dir / f"{safe_id}.json"

    def save(self, graph: ExecutionGraph) -> Path:
        file_path = self._path(graph.graph_id)
        raw_json = graph.model_dump_json(indent=2)
        file_path.write_text(raw_json, encoding="utf-8")
        return file_path

    def load(self, graph_id: str) -> Optional[ExecutionGraph]:
        file_path = self._path(graph_id)
        if not file_path.exists():
            return None
        raw_json = file_path.read_text(encoding="utf-8")
        data = json.loads(raw_json)
        return ExecutionGraph.model_validate(data)

    def list_graphs(self) -> List[str]:
        return sorted([p.stem for p in self.store_dir.glob("*.json")])

    def delete(self, graph_id: str) -> bool:
        file_path = self._path(graph_id)
        if file_path.exists():
            file_path.unlink()
            return True
        return False
