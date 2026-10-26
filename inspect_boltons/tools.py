from __future__ import annotations

from typing import TYPE_CHECKING, Callable

from inspect_ai.tool import Tool, tool
from inspect_ai.util import Limit, sample_limits
from tabulate import tabulate

if TYPE_CHECKING:
    from inspect_swarm import BudgetUsage, SwarmBudget

try:
    from inspect_swarm import swarm_budget
except ImportError:
    swarm_budget = None


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


def _current_swarm_budget() -> SwarmBudget | None:
    if swarm_budget is None:
        return None
    try:
        return swarm_budget()
    except RuntimeError:
        return None


def _swarm_row(
    label: str, usage: BudgetUsage, fmt: Callable[[float], str]
) -> list[str]:
    if usage.share is not None and usage.share_used is not None:
        return [f"{label} (your share)", fmt(usage.share_used), fmt(usage.share)]
    return [f"{label} (whole swarm)", fmt(usage.used), fmt(usage.cap)]


@tool(name="resources")
def resources() -> Tool:
    """A tool that reports the agent's limits and how much of each has been used.

    Inside an inspect_swarm member, cost and tokens are the member's share of
    the swarm's caps when the budget is split, else the whole swarm's caps; a
    swarm time cap is shown alongside the sample's working-time limit.
    Elsewhere, every row is the sample's own limit.
    """

    async def execute() -> str:
        """Check your limits (cost, tokens, messages, turns, time) and how much of each you have used."""
        limits = sample_limits()
        budget = _current_swarm_budget()
        cost_row = (
            _swarm_row("Token cost", budget.cost, _format_usd)
            if budget is not None and budget.cost is not None
            else _resource_row("Token cost", limits.cost, _format_usd)
        )
        token_row = (
            _swarm_row("Tokens", budget.tokens, _format_count)
            if budget is not None and budget.tokens is not None
            else _resource_row("Tokens", limits.token, _format_count)
        )
        rows = [
            cost_row,
            token_row,
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
        if budget is not None and budget.time is not None:
            rows.append(
                [
                    "Time (whole swarm)",
                    _format_duration(budget.time.used),
                    _format_duration(budget.time.cap),
                ]
            )
        table = tabulate(
            rows,
            headers=["Resource", "Used", "Limit"],
            tablefmt="github",
            disable_numparse=True,
        )
        header = (
            "Reaching any of the limits ends the task."
            if budget is None
            else "Reaching any of the limits ends your work on the task."
        )
        return "\n".join([header, "", table])

    return execute
