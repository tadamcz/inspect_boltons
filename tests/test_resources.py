from __future__ import annotations

import pytest

import inspect_boltons.tools as tools_mod
from inspect_boltons.tools import resources


class _FakeLimit:
    """Stand-in for inspect_ai.util.Limit with the fields the tool reads."""

    def __init__(self, *, usage: float, limit: float | None) -> None:
        self.usage = usage
        self.limit = limit


class _FakeSampleLimits:
    def __init__(
        self,
        *,
        working: _FakeLimit,
        token: _FakeLimit | None = None,
        cost: _FakeLimit | None = None,
        message: _FakeLimit | None = None,
        turn: _FakeLimit | None = None,
    ) -> None:
        # Default to "no limit set" for the limits not under test.
        self.token = token or _FakeLimit(usage=0, limit=None)
        self.cost = cost or _FakeLimit(usage=0, limit=None)
        self.message = message or _FakeLimit(usage=0, limit=None)
        self.turn = turn or _FakeLimit(usage=0, limit=None)
        self.working = working


def test_format_duration_compact() -> None:
    assert tools_mod._format_duration(0) == "0s"
    assert tools_mod._format_duration(129_600) == "36h"  # 36h exactly, no m/s
    assert tools_mod._format_duration(45) == "45s"
    assert tools_mod._format_duration(3 * 3600 + 25 * 60) == "3h 25m"


async def test_resources_tool_tabulates_all_limits_with_header(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # All limits are always listed under the "any one ends the task" header. Here
    # a token-limited run (no cost limit): Token cost's limit reads "none".
    monkeypatch.setattr(
        tools_mod,
        "sample_limits",
        lambda: _FakeSampleLimits(
            token=_FakeLimit(usage=10_000, limit=1_000_000),
            message=_FakeLimit(usage=12, limit=50),
            turn=_FakeLimit(usage=6, limit=25),
            working=_FakeLimit(usage=3600, limit=129_600),
        ),
    )
    output = await resources()()
    assert output == (
        "Reaching any of the limits ends the task.\n"
        "\n"
        "| Resource   | Used   | Limit     |\n"
        "|------------|--------|-----------|\n"
        "| Token cost | $0.00  | none      |\n"
        "| Tokens     | 10,000 | 1,000,000 |\n"
        "| Messages   | 12     | 50        |\n"
        "| Turns      | 6      | 25        |\n"
        "| Time       | 1h     | 36h       |"
    )


async def test_resources_tool_reports_cost_in_usd(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A cost-limited run (no token limit): Token cost shows USD, Tokens' limit "none".
    monkeypatch.setattr(
        tools_mod,
        "sample_limits",
        lambda: _FakeSampleLimits(
            cost=_FakeLimit(usage=1.5, limit=200.0),
            working=_FakeLimit(usage=3600, limit=259_200),
        ),
    )
    output = await resources()()
    assert output == (
        "Reaching any of the limits ends the task.\n"
        "\n"
        "| Resource   | Used   | Limit   |\n"
        "|------------|--------|---------|\n"
        "| Token cost | $1.50  | $200.00 |\n"
        "| Tokens     | 0      | none    |\n"
        "| Messages   | 0      | none    |\n"
        "| Turns      | 0      | none    |\n"
        "| Time       | 1h     | 72h     |"
    )


async def test_resources_tool_handles_all_limits_unset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # With nothing configured, every limit reads "none" -- the header makes clear
    # that an unset dimension simply doesn't bound the run.
    monkeypatch.setattr(
        tools_mod,
        "sample_limits",
        lambda: _FakeSampleLimits(
            token=_FakeLimit(usage=500, limit=None),
            working=_FakeLimit(usage=120, limit=None),
        ),
    )
    output = await resources()()
    assert output == (
        "Reaching any of the limits ends the task.\n"
        "\n"
        "| Resource   | Used   | Limit   |\n"
        "|------------|--------|---------|\n"
        "| Token cost | $0.00  | none    |\n"
        "| Tokens     | 500    | none    |\n"
        "| Messages   | 0      | none    |\n"
        "| Turns      | 0      | none    |\n"
        "| Time       | 2m     | none    |"
    )
