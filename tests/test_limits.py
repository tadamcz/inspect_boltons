from __future__ import annotations

import itertools
from pathlib import Path

import pytest
from inspect_ai import Task, eval
from inspect_ai.agent import AgentState, react
from inspect_ai.dataset import Sample
from inspect_ai.event import SampleLimitEvent
from inspect_ai.log import transcript
from inspect_ai.model import (
    ChatMessageAssistant,
    ChatMessageTool,
    ChatMessageUser,
    ModelOutput,
    get_model,
)
from inspect_ai.tool import ToolCall, think
from inspect_ai.util import LimitExceededError

from inspect_boltons.limits import UnproductiveLoopLimit


def _text(content: str = "Thinking...") -> ChatMessageAssistant:
    return ChatMessageAssistant(content=content)


def _tool_call(function: str) -> list[ChatMessageAssistant | ChatMessageTool]:
    return [
        ChatMessageAssistant(
            content="",
            tool_calls=[ToolCall(id="c", function=function, arguments={})],
        ),
        ChatMessageTool(content="ok", tool_call_id="c", function=function),
    ]


def test_counts_trailing_turns_without_tool_call() -> None:
    limit = UnproductiveLoopLimit(turns=3)
    state = AgentState(
        messages=[
            ChatMessageUser(content="go"),
            *_tool_call("bash"),
            _text(),
            ChatMessageUser(content="continue"),
            _text(),
        ]
    )
    assert limit.unproductive_turns(state) == 2
    limit.check(state)

    state.messages.append(_text())
    with pytest.raises(LimitExceededError) as excinfo:
        limit.check(state)
    assert excinfo.value.type == "custom"
    assert excinfo.value.value == 3
    assert excinfo.value.limit == 3
    assert excinfo.value.message == (
        "Unproductive loop limit reached: 3 consecutive turns without a tool call; "
        "limit: 3"
    )
    event = transcript().events[-1]
    assert isinstance(event, SampleLimitEvent)
    assert event.type == "custom"
    assert event.limit == 3
    assert event.message == excinfo.value.message


def test_tool_call_resets_the_run() -> None:
    limit = UnproductiveLoopLimit(turns=2)
    state = AgentState(messages=[_text(), _text(), *_tool_call("bash"), _text()])
    assert limit.unproductive_turns(state) == 1
    limit.check(state)


def test_unproductive_tools_do_not_count_as_productive() -> None:
    limit = UnproductiveLoopLimit(turns=2, unproductive_tools=["think", "resources"])
    state = AgentState(messages=[_text(), *_tool_call("think")])
    with pytest.raises(LimitExceededError) as excinfo:
        limit.check(state)
    assert excinfo.value.message == (
        "Unproductive loop limit reached: 2 consecutive turns without a tool call "
        "other than resources, think; limit: 2"
    )

    state = AgentState(messages=[_text(), *_tool_call("think"), *_tool_call("python")])
    assert limit.unproductive_turns(state) == 0


def test_rejects_non_positive_turns() -> None:
    with pytest.raises(ValueError):
        UnproductiveLoopLimit(turns=0)


def test_react_agent_sample_ends_with_custom_limit(tmp_path: Path) -> None:
    limit = UnproductiveLoopLimit(turns=3, unproductive_tools=["think"])

    async def on_continue(state: AgentState) -> bool:
        limit.check(state)
        return True

    outputs = itertools.chain(
        [ModelOutput.for_tool_call("mockllm/model", "think", {"thought": "hm"})],
        itertools.repeat(ModelOutput.from_content("mockllm/model", "Still thinking.")),
    )
    model = get_model("mockllm/model", custom_outputs=outputs)
    task = Task(
        dataset=[Sample(input="Do the thing.")],
        solver=react(tools=[think()], on_continue=on_continue),
    )
    [log] = eval(task, model=model, log_dir=str(tmp_path), display="none")

    assert log.status == "success"
    assert log.samples is not None
    [sample] = log.samples
    assert sample.error is None
    assert sample.limit is not None
    assert sample.limit.type == "custom"
    assert sample.limit.limit == 3
    assistant = [m for m in sample.messages if m.role == "assistant"]
    assert len(assistant) == 3
    [event] = [e for e in sample.events if isinstance(e, SampleLimitEvent)]
    assert event.type == "custom"
    assert event.message == (
        "Unproductive loop limit reached: 3 consecutive turns without a tool call "
        "other than think; limit: 3"
    )
