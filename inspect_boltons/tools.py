from __future__ import annotations

from typing import Callable

from inspect_ai.tool import Tool, tool
from inspect_ai.util import Limit, sample_limits


def _format_count(value: float) -> str:
    return f"{round(value):,}"


def _format_usd(value: float) -> str:
    return f"${value:,.2f}"


def _format_duration(seconds: float) -> str:
    """Render a number of seconds as a compact ``Xh Ym Zs`` string.

    Zero-valued leading units are dropped (so 129_600 -> ``"36h"``), and a flat
    ``"0s"`` is shown for zero.
    """
    total = round(seconds)
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    parts = []
    if hours:
        parts.append(f"{hours}h")
    if minutes:
        parts.append(f"{minutes}m")
    if secs or not parts:
        parts.append(f"{secs}s")
    return " ".join(parts)


def _resource_line(label: str, limit: Limit, fmt: Callable[[float], str]) -> str:
    used = fmt(limit.usage)
    if limit.limit is None:
        return f"- {label}: {used} used (no limit set)"
    # remaining is non-None whenever limit is non-None (Limit.remaining).
    assert limit.remaining is not None
    return (
        f"- {label}: {used} used, {fmt(limit.remaining)} remaining "
        f"(limit {fmt(limit.limit)})"
    )


@tool(name="resources")
def resources() -> Tool:
    """A tool that reports the agent's limits and how much of each remains."""

    async def execute() -> str:
        """Check your remaining limits (cost, tokens, messages, turns, time)."""
        limits = sample_limits()
        lines = [
            _resource_line("Token cost", limits.cost, _format_usd),
            _resource_line("Tokens", limits.token, _format_count),
            _resource_line("Messages", limits.message, _format_count),
            _resource_line("Turns", limits.turn, _format_count),
            # We surface the *working*-time limit plainly as "Time".
            # The working-time vs. clock-time distinction is not relevant to an agent.
            _resource_line("Time", limits.working, _format_duration),
        ]
        header = "Reaching any of the limits ends the task."
        return "\n".join([header, *lines])

    return execute
