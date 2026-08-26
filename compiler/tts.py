"""
compiler/tts.py

Text-to-speech generation using ElevenLabs API.
Generates one continuous audio file from all narration beats.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path
from typing import List, Optional

from pydub import AudioSegment

from compiler.schemas import ExecutionGraph, NarrationBeat
from compiler.speech_normalize import SpeechNormalizer


class TTSAuthError(RuntimeError):
    """Raised when ElevenLabs rejects the API key (HTTP 401/403)."""


class TTSGenerator:
    """
    Generate TTS audio for an ExecutionGraph using ElevenLabs API.
    Produces one continuous audio file with silence padding between beats.
    """

    def __init__(self, api_key: Optional[str] = None, voice_id: Optional[str] = None):
        self.api_key = (api_key or os.environ.get("ELEVENLABS_API_KEY", "")).strip()
        self.voice_id = (voice_id or os.environ.get("ELEVENLABS_VOICE_ID", "")).strip()
        # eleven_turbo_v2 is deprecated; v2.5 is the supported successor.
        # Override with ELEVENLABS_MODEL_ID if needed.
        self.model = os.environ.get("ELEVENLABS_MODEL_ID", "eleven_turbo_v2_5").strip()
        self.normalize = SpeechNormalizer.normalize

        if not self.api_key:
            raise ValueError("ElevenLabs API key required. Set ELEVENLABS_API_KEY env var.")
        if not self.voice_id:
            raise ValueError("ElevenLabs voice ID required. Set ELEVENLABS_VOICE_ID env var.")

        print(f"TTS using key prefix: {self.api_key[:10]}... voice: {self.voice_id[:8]}...")

    # ------------------------------------------------------------------
    # Credential preflight
    # ------------------------------------------------------------------

    def check_credentials(self) -> dict:
        """
        Verify the API key and voice ID with ElevenLabs BEFORE rendering.

        Returns the /v1/user payload on success. Raises TTSAuthError on
        401/403 (bad/revoked key) and RuntimeError for an unknown voice ID,
        with actionable messages. Run standalone via:
            python -m compiler.tts --check
        """
        user = self._api_get("https://api.elevenlabs.io/v1/user")
        try:
            self._api_get(f"https://api.elevenlabs.io/v1/voices/{self.voice_id}")
        except RuntimeError as exc:
            raise RuntimeError(
                f"ElevenLabs voice ID {self.voice_id!r} was not found on this account. "
                "Set ELEVENLABS_VOICE_ID to a voice from your Voice Library "
                "(https://elevenlabs.io/app/voice-library)."
            ) from exc
        return user

    def _api_get(self, url: str) -> dict:
        """GET an ElevenLabs endpoint; raise with a precise diagnosis on failure."""
        result = subprocess.run(
            [
                "curl", "-s", "-S", "-X", "GET", url,
                "-H", f"xi-api-key: {self.api_key}",
                "-w", "\n%{http_code}",
            ],
            capture_output=True, text=True, timeout=30,
        )
        body, _, status = result.stdout.rpartition("\n")
        status = status.strip()
        if status in ("401", "403"):
            raise TTSAuthError(
                f"ElevenLabs rejected the API key (HTTP {status}). The key is invalid, "
                "revoked, or lacks permission. Regenerate it at "
                "https://elevenlabs.io/app/settings/api-keys and update "
                f"ELEVENLABS_API_KEY. Response: {body[:300]}"
            )
        if not status.startswith("2"):
            raise RuntimeError(
                f"ElevenLabs GET {url} failed (HTTP {status or 'network error'}): "
                f"{body[:300] or result.stderr[:300]}"
            )
        try:
            return json.loads(body or "{}")
        except json.JSONDecodeError:
            return {}

    def generate(self, graph: ExecutionGraph, output_path: str) -> str:
        """
        Generate continuous audio for all narration beats in the graph.
        Returns path to the final MP3 file.
        """
        clips = self.generate_clips(graph)
        return self.assemble_clips(clips, output_path)

    def generate_clips(self, graph: ExecutionGraph, temp_dir: Optional[str] = None) -> List[dict]:
        """
        Generate individual TTS clips for each beat.
        Returns list of dicts: {"beat_id", "path", "duration_ms", "start_ms", "end_ms"}
        """
        clips = []
        temp_dir = Path(temp_dir) if temp_dir else Path(tempfile.gettempdir()) / "wsda_tts"
        temp_dir.mkdir(exist_ok=True)

        for i, beat in enumerate(graph.narration_beats):
            text = self.normalize(beat.text)
            beat.tts_text = text  # Store normalized text back on beat

            clip_path = temp_dir / f"{graph.graph_id}_beat_{i:03d}.mp3"

            if not self._clip_is_valid(clip_path):
                if clip_path.exists():
                    clip_path.unlink()
                self._synthesize_text(text, str(clip_path))

            # Measure actual duration with pydub
            audio = AudioSegment.from_mp3(str(clip_path))
            duration_ms = len(audio)

            clips.append((beat, str(clip_path), duration_ms))

        return clips

    @staticmethod
    def _clip_is_valid(clip_path: Path) -> bool:
        """Return True if the cached MP3 exists and is decodable."""
        if not clip_path.exists() or clip_path.stat().st_size < 1000:
            return False
        try:
            AudioSegment.from_mp3(str(clip_path))
            return True
        except Exception:
            return False

    def assemble_clips(self, clips: List[dict], output_path: str) -> str:
        """
        Concatenate clips with silence padding to match video timing.
        """
        final = AudioSegment.empty()
        position_ms = 0

        for beat, clip_path, duration_ms in clips:
            audio = AudioSegment.from_mp3(clip_path)

            # Calculate target start time in ms
            target_start_ms = int(beat.start_time * 1000)

            # If we are behind, pad with silence
            if position_ms < target_start_ms:
                silence_ms = target_start_ms - position_ms
                final += AudioSegment.silent(duration=silence_ms)
                position_ms += silence_ms

            # Add the audio clip
            final += audio
            position_ms += len(audio)

        # Export final
        final.export(output_path, format="mp3", bitrate="192k")
        return output_path

    def generate_text(self, text: str, output_path: str) -> str:
        """Generate TTS for a single text string."""
        normalized = self.normalize(text)
        self._synthesize_text(normalized, output_path)
        return output_path

    def _synthesize_text(self, text: str, output_path: str) -> str:
        """Route to the working synthesis method."""
        return self._synthesize_with_curl(text, output_path)

    def _synthesize_with_curl(self, text: str, output_path: str) -> str:
        """
        Shell out to curl since urllib.request fails with some keys.

        Captures the HTTP status explicitly so auth failures (401/403) are
        raised immediately as TTSAuthError with the API's own error detail,
        transient failures (429/5xx) are retried with backoff, and the old
        "--fail-with-body writes the JSON error into the MP3" trap is gone.
        """
        payload = json.dumps({
            "text": text,
            "model_id": self.model,
            "voice_settings": {
                "stability": 0.5,
                "similarity_boost": 0.75
            }
        })
        url = f"https://api.elevenlabs.io/v1/text-to-speech/{self.voice_id}"

        max_attempts = 3
        for attempt in range(1, max_attempts + 1):
            if os.path.exists(output_path):
                os.unlink(output_path)

            result = subprocess.run(
                [
                    "curl", "-s", "-S", "-X", "POST", url,
                    "-H", f"xi-api-key: {self.api_key}",
                    "-H", "Content-Type: application/json",
                    "-d", payload,
                    "-o", output_path,
                    "-w", "%{http_code}",
                ],
                capture_output=True, text=True, timeout=180,
            )
            status = (result.stdout or "").strip()

            if result.returncode == 0 and status.startswith("2"):
                break

            # On HTTP errors curl still wrote the JSON error body to the
            # output file — read it for diagnostics instead of hiding it.
            body = ""
            if os.path.exists(output_path):
                try:
                    body = Path(output_path).read_text(errors="replace")[:500]
                except Exception:
                    body = ""
                os.unlink(output_path)

            if status in ("401", "403"):
                raise TTSAuthError(
                    f"ElevenLabs rejected the API key (HTTP {status}). The key is "
                    "invalid, revoked, or over quota. Regenerate it at "
                    "https://elevenlabs.io/app/settings/api-keys and update "
                    f"ELEVENLABS_API_KEY. Response: {body}"
                )
            if status == "404":
                raise RuntimeError(
                    f"ElevenLabs voice ID {self.voice_id!r} not found (HTTP 404). "
                    "Set ELEVENLABS_VOICE_ID to a voice from your Voice Library. "
                    f"Response: {body}"
                )
            if (status == "429" or status.startswith("5")) and attempt < max_attempts:
                wait = 2 ** attempt
                print(
                    f"ElevenLabs HTTP {status}; retrying in {wait}s "
                    f"(attempt {attempt}/{max_attempts})...",
                    file=sys.stderr,
                )
                time.sleep(wait)
                continue

            raise RuntimeError(
                f"ElevenLabs TTS failed (HTTP {status or f'curl exit {result.returncode}'})"
                f": {body or result.stderr[:500]}"
            )

        if not os.path.exists(output_path) or os.path.getsize(output_path) < 1000:
            raise RuntimeError(
                f"TTS output file missing or too small: {output_path}"
            )

        return output_path


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--check":
        # Credential preflight: python -m compiler.tts --check
        try:
            tts = TTSGenerator()
            info = tts.check_credentials()
        except (ValueError, TTSAuthError, RuntimeError) as exc:
            print(f"TTS check FAILED: {exc}", file=sys.stderr)
            sys.exit(1)
        subscription = info.get("subscription", {}) if isinstance(info, dict) else {}
        tier = subscription.get("tier", "unknown")
        char_count = subscription.get("character_count", "?")
        char_limit = subscription.get("character_limit", "?")
        print(f"TTS check OK: tier={tier} characters={char_count}/{char_limit} "
              f"voice={tts.voice_id} model={tts.model}")
        sys.exit(0)

    if len(sys.argv) > 1:
        text = sys.argv[1]
    else:
        text = "Hello, this is a test of the TTS system."
    tts = TTSGenerator()
    path = tts.generate_text(text, "/tmp/tts_test.mp3")
    print(f"Saved to {path} ({os.path.getsize(path)} bytes)")
