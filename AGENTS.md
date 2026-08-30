# WSDA Compiler — Project Guide for Kimi

## What this project is

A lesson-first course compiler for data-analytics training videos. Given a topic,
audience, and depth, it:

1. Designs a `CourseManifest` (`compiler/curriculum_designer.py`).
2. Generates a narration script for each video (`compiler/lesson_builder.py`).
3. Executes the script via a vision-language model agent that drives DB Browser for
   SQLite and records per-beat video clips (`compiler/discovery.py`).
4. Builds an `ExecutionGraph` and renders a final MP4 (`compiler/renderer.py`).

## Repository layout

- `compiler/` — all source code lives here.
  - `curriculum.py` — `CourseManifest`/`VideoManifest` schemas and `run_course()`.
  - `curriculum_designer.py` — LLM-based curriculum design + seed DB generation.
  - `lesson_builder.py` — script generation, validation, action derivation.
  - `discovery.py` — `EndStateDiscovery`, `VisionAgent`, screen recording.
  - `renderer.py` — video assembly, TTS muxing, highlights export.
  - `vision_agent.py` — VLM agent for dynamic UI interaction.
  - `narrator.py` — `ScriptBeat` data model and quality helpers.
  - `graph_store.py`, `schemas.py`, `sql_formatter.py`, `tts.py` — supporting modules.
- `LESSON_CONTENT_STANDARD.md` / `QA_CHECKLIST.md` — content and QA references.
- `requirements.txt` — Python dependencies.
- `output/` — rendered courses (ignored by Git).
- `compiler/discovery_output/` — screenshots, seed DBs, telemetry (ignored by Git).

## Common commands

```bash
# Design a curriculum
python -m compiler.curriculum_designer

# Run the default demo course
python -m compiler.curriculum

# Run a specific manifest
python -c "
from compiler.curriculum import load_manifest, run_course
m = load_manifest('sql_fundamentals_for_data_analysts')
results = run_course(m, output_dir='output/courses', output_mode='auto')
print(results)
"

# Compile check
python3 -m py_compile compiler/*.py
```

## Environment variables

- `ANTHROPIC_API_KEY` — required for curriculum design, script generation, and VLM discovery.
- `ELEVENLABS_API_KEY` / `ELEVENLABS_VOICE_ID` — optional, enables TTS narration.
- `DISCOVERY_MODEL` / `NARRATOR_MODEL` / `CURRICULUM_MODEL` — optional model overrides.

Never commit `.env` files or API keys.

## Conventions

- Keep all code changes inside `compiler/` unless explicitly asked otherwise.
- The pipeline is lesson-first: scripts are generated before screen actions.
- `DiscoveryRecipes` are deprecated; the VLM agent is the preferred execution path.
- Seed databases (`.db` files in `compiler/discovery_output/`) are preserved by the cleanup step.
- Render quality issues (clip/script duration mismatch, missing proof numbers, etc.) are warnings, not hard failures.

## Visibility, status, and executor stop rules

These rules are binding on the executor and must survive session resets.

1. **Status file.** During every run, update `output/STATUS.txt` at least every
   15 seconds with one line: `phase, beat x/y, elapsed, est. cost so far`.
   On completion write `DONE: <final mp4 path>`; on abort write
   `ABORTED: <gate + value>`. Fire a macOS notification (`osascript display
   notification`) and a `say` announcement at completion or abort. The user
   must always be able to answer “what is it doing” by reading one file.

2. **Terminal visibility.** Never hide the controlling terminal. Minimize it if
   screenshots require, but it stays in the Dock with the log tail visible
   (run `tail -f output/live_video_1_1_c11.log` in a second window).

3. **Executor stop rule.** When any run ends — pass OR fail — the executor
   STOPS. No autonomous re-runs, no “one more try”. Paste the raw output and
   wait for the governor’s next directive.

## Sync target

This codebase is periodically synced to `~/Desktop/wsda-video-engine/compiler/` for integration with the web console.
