from __future__ import annotations

from typing import Callable

from inspect_ai.tool import Tool, tool
from inspect_ai.util import Limit, sample_limits
from tabulate import tabulate


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


def _format_limit(limit: Limit, fmt: Callable[[float], str]) -> str:
    return "none" if limit.limit is None else fmt(limit.limit)


def _resource_row(label: str, limit: Limit, fmt: Callable[[float], str]) -> list[str]:
    return [label, fmt(limit.usage), _format_limit(limit, fmt)]


@tool(name="resources")
def resources() -> Tool:
    """A tool that reports the agent's limits and how much of each has been used."""

    async def execute() -> str:
        """Check your limits (cost, tokens, messages, turns, time) and how much of each you have used."""
        limits = sample_limits()
        rows = [
            _resource_row("Token cost", limits.cost, _format_usd),
            _resource_row("Tokens", limits.token, _format_count),
            # As of Inspect 0.3.266, reading `usage` on a live message limit raises
            # NotImplementedError: Inspect only tracks the message count on the task or
            # agent state, which a tool cannot reliably access (the sample's TaskState
            # is synced from the agent's state only after the agent finishes). So we
            # show the limit but not the usage.
            ["Messages", "?", _format_limit(limits.message, _format_count)],
            _resource_row("Turns", limits.turn, _format_count),
            # We surface the *working*-time limit plainly as "Time".
            # The working-time vs. clock-time distinction is not relevant to an agent.
            _resource_row("Time", limits.working, _format_duration),
        ]
        table = tabulate(
            rows,
            headers=["Resource", "Used", "Limit"],
            tablefmt="github",
            disable_numparse=True,
        )
        header = "Reaching any of the limits ends the task."
        return "\n".join([header, "", table])

    return execute
