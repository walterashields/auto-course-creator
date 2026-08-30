#!/usr/bin/env python3
"""
compiler/cost_tracker.py

Global cost meter for VLM/API calls. Tracks running input/output token totals,
prints a live cost line after each call, and aborts the run if the spend
exceeds WSDA_MAX_RUN_COST_USD (default 5.0).
"""

from __future__ import annotations

import os
import sys
from typing import Any, Optional


# Per-model pricing (USD per token). Extend as new models are used.
MODEL_PRICING = {
    "claude-sonnet-5": {"input": 3.0 / 1_000_000, "output": 15.0 / 1_000_000},
    "claude-sonnet-4": {"input": 3.0 / 1_000_000, "output": 15.0 / 1_000_000},
    "claude-opus-4": {"input": 15.0 / 1_000_000, "output": 75.0 / 1_000_000},
    "claude-haiku-4": {"input": 0.25 / 1_000_000, "output": 1.25 / 1_000_000},
    "default": {"input": 3.0 / 1_000_000, "output": 15.0 / 1_000_000},
}


class CostExhaustedError(RuntimeError):
    """Raised when the run's API spend budget is exceeded."""

    pass


class CostTracker:
    """Singleton-style global cost tracker."""

    _instance: Optional["CostTracker"] = None

    def __new__(cls) -> "CostTracker":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.reset()
        return cls._instance

    def reset(self) -> None:
        self.input_tokens = 0
        self.output_tokens = 0
        self.calls = 0
        self.budget_usd = float(os.environ.get("WSDA_MAX_RUN_COST_USD", "5.0"))

    def check_budget(self) -> None:
        """Raise CostExhaustedError if the next API call would exceed the budget."""
        total = self.total_usd()
        if total > self.budget_usd:
            raise CostExhaustedError(
                f"Run cost ${total:.4f} already exceeded budget ${self.budget_usd:.2f}; "
                "aborting before next API call."
            )

    def add(self, input_tokens: int, output_tokens: int, model: str = "default") -> None:
        """Add a VLM/API call to the running totals and print the meter."""
        self.input_tokens += max(0, input_tokens)
        self.output_tokens += max(0, output_tokens)
        self.calls += 1
        cost = self._call_cost(input_tokens, output_tokens, model)
        print(
            f"[COST] call {self.calls}: +{input_tokens:,} in / +{output_tokens:,} out "
            f"(${cost:.4f}) | running ${self.total_usd():.4f} / ${self.budget_usd:.2f}",
            file=sys.stderr,
        )
        if self.total_usd() > self.budget_usd:
            raise CostExhaustedError(
                f"Run cost ${self.total_usd():.4f} exceeded budget ${self.budget_usd:.2f}; aborting."
            )

    @staticmethod
    def _call_cost(input_tokens: int, output_tokens: int, model: str) -> float:
        rates = MODEL_PRICING.get(model, MODEL_PRICING["default"])
        return (input_tokens * rates["input"]) + (output_tokens * rates["output"])

    def total_usd(self) -> float:
        """Return total estimated USD spent so far."""
        # We use the default rate for the running total because the meter is
        # called from wrappers that may not pass the exact model name.
        rates = MODEL_PRICING["default"]
        return (self.input_tokens * rates["input"]) + (self.output_tokens * rates["output"])

    def summary(self) -> dict:
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "calls": self.calls,
            "total_usd": round(self.total_usd(), 6),
            "budget_usd": round(self.budget_usd, 2),
        }


def get_tracker() -> CostTracker:
    return CostTracker()


def reset_tracker() -> None:
    get_tracker().reset()


def _extract_usage(response: Any) -> tuple[int, int]:
    """Return (input_tokens, output_tokens) from an Anthropic message response."""
    usage = getattr(response, "usage", None) or {}
    input_tokens = int(
        getattr(usage, "input_tokens", None) or usage.get("input_tokens", 0)
    )
    output_tokens = int(
        getattr(usage, "output_tokens", None) or usage.get("output_tokens", 0)
    )
    return input_tokens, output_tokens


def tracked_create(client: Any, *, model: str, **kwargs: Any) -> Any:
    """
    Wrapper around Anthropic ``client.messages.create``.

    Checks the run budget before the call, executes the call, then adds the
    token usage to the global CostTracker and prints the live meter.
    """
    tracker = get_tracker()
    tracker.check_budget()
    response = client.messages.create(model=model, **kwargs)
    input_tokens, output_tokens = _extract_usage(response)
    tracker.add(input_tokens, output_tokens, model=model)
    return response
