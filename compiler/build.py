#!/usr/bin/env python3
"""
compiler/build.py

Convenience script that runs the full course-compiler pipeline:
1. Load an ExecutionGraph from the GraphStore.
2. Render the silent video + reference script.
3. Generate TTS audio from narration beats.
4. Mux audio with the silent video to produce the final MP4.

Usage:
    ELEVENLABS_API_KEY=... ELEVENLABS_VOICE_ID=... python -m compiler.build orders_sort_desc_v4

The TTS step is optional: if the ElevenLabs env vars are not set, the script
still renders the silent video and reference script.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from .graph_store import GraphStore
from .renderer import GraphRenderer


def build(graph_id: str, output_dir: str = "output") -> dict:
    """Run the full pipeline for the given graph id."""
    store = GraphStore()
    graph = store.load(graph_id)

    renderer = GraphRenderer(output_dir=output_dir)

    # TTS is optional; only run if credentials are available.
    api_key = os.environ.get("ELEVENLABS_API_KEY", "").strip()
    voice_id = os.environ.get("ELEVENLABS_VOICE_ID", "").strip()
    if api_key and voice_id:
        # Two-pass render: actual TTS durations drive the video timeline.
        render_result = renderer.render_with_audio(graph, output_dir)
        if render_result is None:
            raise RuntimeError("Audio/video render failed.")

        result = {
            "graph_id": graph_id,
            "video_path": render_result["video_path"],
            "script_path": renderer.output_dir / f"{graph_id}_reference.md",
            "duration_seconds": render_result["duration"],
            "audio_path": render_result["audio_path"],
            "final_path": render_result["final_path"],
        }
    else:
        print(
            "Note: ELEVENLABS_API_KEY or ELEVENLABS_VOICE_ID not set; "
            "rendering silent video only.",
            file=sys.stderr,
        )
        render_result = renderer.render(graph)
        if render_result is None:
            raise RuntimeError("Video rendering failed.")

        result = {
            "graph_id": graph_id,
            "video_path": render_result["video_path"],
            "script_path": render_result["script_path"],
            "duration_seconds": render_result["duration_seconds"],
            "audio_path": None,
            "final_path": None,
        }

    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a final course video from an ExecutionGraph.")
    parser.add_argument("graph_id", help="Graph ID to build (e.g., orders_sort_desc_v4)")
    parser.add_argument(
        "--output-dir",
        default="output",
        help="Directory for rendered outputs (default: output)",
    )
    args = parser.parse_args()

    try:
        result = build(args.graph_id, output_dir=args.output_dir)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(f"Graph:      {result['graph_id']}")
    print(f"Silent:     {result['video_path']}")
    print(f"Script:     {result['script_path']}")
    print(f"Duration:   {result['duration_seconds']}s")
    if result["audio_path"]:
        print(f"Audio:      {result['audio_path']}")
    if result["final_path"]:
        print(f"Final MP4:  {result['final_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
