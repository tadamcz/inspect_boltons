from __future__ import annotations

from collections.abc import Iterator
from typing import Literal

import pytest

pytest.importorskip("inspect_swarm")

from inspect_ai import Task, eval
from inspect_ai.agent import Agent, AgentState, agent
from inspect_ai.dataset import Sample
from inspect_ai.model import (
    ModelCost,
    ModelInfo,
    ModelOutput,
    ModelUsage,
    get_model,
    set_model_info,
)
from inspect_ai.model._model_info import clear_model_info_cache
from inspect_swarm import Budget, member, swarm

from inspect_boltons.tools import resources

CALL_TOKENS = 100_000


@pytest.fixture(autouse=True)
def _priced_mockllm() -> Iterator[None]:
    set_model_info(
        "mockllm/model",
        ModelInfo(
            cost=ModelCost(
                input=10.0, output=10.0, input_cache_read=10.0, input_cache_write=10.0
            )
        ),
    )
    yield
    clear_model_info_cache()


@agent
def resources_reader(seen: list[str]) -> Agent:
    async def execute(state: AgentState) -> AgentState:
        await get_model().generate(state.messages)
        seen.append(str(await resources()()))
        return state

    return execute


def _run_swarm(
    split: Literal["pool", "equal"], time_limit: int | None = None
) -> list[str]:
    seen: list[str] = []
    output = ModelOutput.from_content("mockllm/model", "done")
    output.usage = ModelUsage(
        input_tokens=CALL_TOKENS, output_tokens=0, total_tokens=CALL_TOKENS
    )
    task = Task(
        dataset=[Sample(input="go")],
        solver=swarm(
            members=member(resources_reader, args={"seen": seen}, count=2),
            channels=[],
            budget=Budget(split=split),
        ),
        cost_limit=10.0,
        token_limit=1_000_000,
        time_limit=time_limit,
    )
    [log] = eval(
        task,
        model=get_model("mockllm/model", custom_outputs=[output, output]),
        display="none",
        log_dir=None,
    )
    assert log.status == "success", log.error
    return seen


def _rows(text: str) -> dict[str, list[str]]:
    rows: dict[str, list[str]] = {}
    for line in text.splitlines()[4:]:
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        rows[cells[0]] = cells[1:]
    return rows


def test_a_member_of_a_split_swarm_sees_its_own_share() -> None:
    seen = _run_swarm("equal")
    assert len(seen) == 2
    for text in seen:
        assert text.splitlines()[0] == (
            "Reaching any of the limits ends your work on the task."
        )
        rows = _rows(text)
        # 95% of each sample limit is the swarm's cap; each of 2 members gets half.
        assert rows["Token cost (your share)"] == ["$1.00", "$4.75"]
        assert rows["Tokens (your share)"] == ["100,000", "475,000"]
        assert "Time (whole swarm)" not in rows


def test_a_member_of_a_pooled_swarm_sees_the_whole_swarm_cap() -> None:
    seen = _run_swarm("pool", time_limit=600)
    for text in seen:
        rows = _rows(text)
        assert rows["Token cost (whole swarm)"][1] == "$9.50"
        assert rows["Tokens (whole swarm)"][1] == "950,000"
        assert rows["Time (whole swarm)"][1] == "9m 30s"
        assert "Token cost (your share)" not in rows
