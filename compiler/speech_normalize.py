#!/usr/bin/env python3
"""
compiler/speech_normalize.py

SpeechNormalizer: converts written text into natural spoken form for TTS.
Handles numbers, decimals, and technical terms in a teaching context without
over-normalizing.
"""

from __future__ import annotations

import re

from .schemas import NarrationBeat


class SpeechNormalizer:
    """
    Converts written text into natural spoken form for TTS.
    Handles numbers, decimals, and currency in a teaching context.
    """

    # Words that suggest a number is a currency/money amount.
    _MONEY_CONTEXT = (
        "amount",
        "cost",
        "price",
        "revenue",
        "money",
        "currency",
        "dollar",
        "dollars",
        "cent",
        "cents",
    )

    _DIGIT_WORDS = {
        "0": "zero",
        "1": "one",
        "2": "two",
        "3": "three",
        "4": "four",
        "5": "five",
        "6": "six",
        "7": "seven",
        "8": "eight",
        "9": "nine",
    }

    @classmethod
    def normalize(cls, text: str) -> str:
        """
        Apply conservative spoken-form conversions.

        Rules:
        1. Drop trailing .0 from decimals.
        2. In money context, convert simple decimals like 0.5 to "fifty cents".
        3. Leave complex decimals unchanged (TTS will read digits).
        4. Spell out technical abbreviations likely to be mispronounced.
        5. Preserve dates, ordinals, and other text.
        """
        result = text

        # Rule 4: technical abbreviations that TTS commonly mispronounces.
        result = cls._normalize_technical_terms(result)

        # Rule 1: drop trailing .0 (e.g., 340.0 -> 340, 85.0 -> 85).
        result = cls._drop_trailing_zero_decimals(result)

        # Rule 2: simple decimals like 0.5 -> "point five" by default, or
        # "fifty cents" in a clear money context.
        result = cls._normalize_simple_decimals(result)

        return result

    @classmethod
    def normalize_beat(cls, beat: NarrationBeat) -> str:
        """Normalize a single beat's text and return the normalized string."""
        return cls.normalize(beat.text)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @classmethod
    def _drop_trailing_zero_decimals(cls, text: str) -> str:
        """Convert 340.0, 85.0, etc. to 340, 85."""
        return re.sub(r"(\d+)\.0+(?![\d])", r"\1", text)

    @classmethod
    def _normalize_simple_decimals(cls, text: str) -> str:
        """
        Convert simple decimals like 0.5 to spoken form.
        - In a clear money context: "0.5" -> "fifty cents".
        - Otherwise: "0.5" -> "point five".
        Complex decimals (e.g., 210.25) are left for the TTS to read.
        """
        is_money = cls._is_money_context(text)

        def replace_simple_decimal(match: re.Match) -> str:
            number = match.group(1)
            integer_part, _, fractional_part = number.partition(".")
            if integer_part == "0" and len(fractional_part) == 1:
                digit_word = cls._DIGIT_WORDS.get(fractional_part, fractional_part)
                if is_money:
                    # e.g., 0.5 -> fifty cents
                    tens = int(fractional_part) * 10
                    return f"{tens} cents"
                # e.g., 0.5 -> point five
                return f"point {digit_word}"
            # Leave complex decimals untouched.
            return match.group(0)

        return re.sub(r"\b(\d+\.\d+)\b", replace_simple_decimal, text)

    @classmethod
    def _is_money_context(cls, text: str) -> bool:
        lower = text.lower()
        return any(word in lower for word in cls._MONEY_CONTEXT)

    @classmethod
    def _normalize_technical_terms(cls, text: str) -> str:
        """
        Spell out technical abbreviations that TTS often mispronounces.
        Uses word-boundary aware replacements to avoid mid-word matches.
        Spaces between letters force TTS engines to read them individually
        rather than as a single hyphenated word.
        """
        replacements = [
            (r"\bSQLite\b", "S Q L ite"),
            (r"\bSQL\b", "S Q L"),
            (r"\bDB\b", "D B"),
        ]
        for pattern, replacement in replacements:
            text = re.sub(pattern, replacement, text)
        return text


if __name__ == "__main__":
    tests = [
        ("340.0", "340"),
        ("85.0", "85"),
        ("150.0", "150"),
        ("The amount is 340.0", "The amount is 340"),
        ("SQL is powerful", "S Q L is powerful"),
        ("SQLite database", "S Q L ite database"),
        ("DB Browser", "D B Browser"),
        ("0.5", "point five"),
        ("The total is 210.25", "The total is 210.25"),
        ("The price is 0.5", "The price is 50 cents"),
        ("Q1 2024", "Q1 2024"),
        ("2024-01-22", "2024-01-22"),
    ]

    failed = 0
    for inp, expected in tests:
        result = SpeechNormalizer.normalize(inp)
        status = "OK" if result == expected else "FAIL"
        if result != expected:
            failed += 1
        print(f"  {inp!r:35} → {result!r:35} {status}")

    print()
    if failed == 0:
        print("All tests passed.")
    else:
        print(f"{failed} test(s) failed.")
        raise SystemExit(1)
