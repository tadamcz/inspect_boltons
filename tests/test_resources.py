from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

from inspect_ai.model import ModelUsage
from inspect_ai.util import (
    cost_limit,
    message_limit,
    time_limit,
    token_limit,
    turn_limit,
    working_limit,
)
from inspect_ai.util._limit import record_model_cost, record_model_usage, record_turn

import inspect_boltons.tools as tools_mod
from inspect_boltons.tools import resources


@contextmanager
def _sample(
    *,
    token: int | None = None,
    cost: float | None = None,
    message: int | None = None,
    turn: int | None = None,
    working: float | None = None,
) -> Iterator[None]:
    """Open root limit nodes of every kind, as a running sample would."""
    with (
        token_limit(token),
        cost_limit(cost),
        message_limit(message),
        turn_limit(turn),
        working_limit(working),
        time_limit(None),
    ):
        yield


def test_format_duration_compact() -> None:
    assert tools_mod._format_duration(0) == "0s"
    assert tools_mod._format_duration(129_600) == "36h"  # 36h exactly, no m/s
    assert tools_mod._format_duration(45) == "45s"
    assert tools_mod._format_duration(3 * 3600 + 25 * 60) == "3h 25m"


async def test_resources_tool_tabulates_all_limits_with_header() -> None:
    # All limits are always listed under the "any one ends the task" header. Here
    # a token-limited run (no cost limit): Token cost's limit reads "none".
    with _sample(token=1_000_000, message=50, turn=25, working=129_600):
        record_model_usage(ModelUsage(total_tokens=10_000))
        record_model_cost(1.5)
        for _ in range(6):
            record_turn()
        output = await resources()()
    assert output == (
        "Reaching any of the limits ends the task.\n"
        "\n"
        "| Resource   | Used   | Limit     |\n"
        "|------------|--------|-----------|\n"
        "| Token cost | $1.50  | none      |\n"
        "| Tokens     | 10,000 | 1,000,000 |\n"
        "| Messages   | ?      | 50        |\n"
        "| Turns      | 6      | 25        |\n"
        "| Time       | 0s     | 36h       |"
    )


async def test_resources_tool_reports_cost_in_usd() -> None:
    # A cost-limited run (no token limit): Token cost shows USD, Tokens' limit "none".
    with _sample(cost=200.0, working=259_200):
        record_model_cost(1.5)
        output = await resources()()
    assert output == (
        "Reaching any of the limits ends the task.\n"
        "\n"
        "| Resource   | Used   | Limit   |\n"
        "|------------|--------|---------|\n"
        "| Token cost | $1.50  | $200.00 |\n"
        "| Tokens     | 0      | none    |\n"
        "| Messages   | ?      | none    |\n"
        "| Turns      | 0      | none    |\n"
        "| Time       | 0s     | 72h     |"
    )


async def test_resources_tool_handles_all_limits_unset() -> None:
    # With nothing configured, every limit reads "none" -- the header makes clear
    # that an unset dimension simply doesn't bound the run.
    with _sample():
        record_model_usage(ModelUsage(total_tokens=500))
        output = await resources()()
    assert output == (
        "Reaching any of the limits ends the task.\n"
        "\n"
        "| Resource   | Used   | Limit   |\n"
        "|------------|--------|---------|\n"
        "| Token cost | $0.00  | none    |\n"
        "| Tokens     | 500    | none    |\n"
        "| Messages   | ?      | none    |\n"
        "| Turns      | 0      | none    |\n"
        "| Time       | 0s     | none    |"
    )
