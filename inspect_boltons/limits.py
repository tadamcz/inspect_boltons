"""Limits that stop a sample when the model is looping unproductively.

Each limit is a plain object with a `check(state)` method that takes an `AgentState`
or `TaskState`, inspects the trailing assistant turns in `state.messages`, and raises
`LimitExceededError` when its criterion is met. Inspect handles that error like any
other limit: the sample ends, is scored on its messages so far, and is recorded with
a limit of type "custom".

Call `check(state)` after each turn. With `react()`, do so from the `on_continue`
hook, which runs after each turn's generation and tool calls. Several limits can be
checked from the same hook:

    limits = [
        NoToolCallLimit(turns=100, unproductive_tools=["think"]),
        RepeatedTextLimit(turns=10),
    ]

    async def on_continue(state: AgentState) -> bool:
        for limit in limits:
            limit.check(state)
        return True

    agent = react(tools=[bash(), python(), think()], on_continue=on_continue)

In a custom solver, call it after each `generate()` and tool execution:

    @solver
    def my_solver() -> Solver:
        limit = NoToolCallLimit(turns=100)

        async def solve(state: TaskState, generate: Generate) -> TaskState:
            while not state.completed:
                state = await generate(state, tool_calls="single")
                limit.check(state)
            return state

        return solve

The limit objects hold only configuration, so one instance can be shared across
samples.
"""

from __future__ import annotations

from collections.abc import Collection

from inspect_ai.agent import AgentState
from inspect_ai.event import SampleLimitEvent
from inspect_ai.log import transcript
from inspect_ai.model import ChatMessageAssistant
from inspect_ai.solver import TaskState
from inspect_ai.util import LimitExceededError


def _assistant_turns(state: AgentState | TaskState) -> list[ChatMessageAssistant]:
    return [m for m in state.messages if isinstance(m, ChatMessageAssistant)]


def _raise_limit(count: int, limit: int, message: str) -> None:
    transcript()._event(SampleLimitEvent(type="custom", limit=limit, message=message))
    raise LimitExceededError("custom", value=count, limit=limit, message=message)


class NoToolCallLimit:
    """Stop a sample once the model has gone `turns` consecutive turns without a tool call.

    A turn is an assistant message. A turn counts as productive if it calls any tool
    other than those named in `unproductive_tools`; any productive turn resets the run.
    """

    def __init__(self, turns: int, *, unproductive_tools: Collection[str] = ()) -> None:
        if turns < 1:
            raise ValueError(f"turns must be a positive integer: {turns}")
        self.turns = turns
        self.unproductive_tools = frozenset(unproductive_tools)

    def _productive(self, message: ChatMessageAssistant) -> bool:
        return any(
            call.function not in self.unproductive_tools
            for call in message.tool_calls or []
        )

    def unproductive_turns(self, state: AgentState | TaskState) -> int:
        count = 0
        for message in reversed(_assistant_turns(state)):
            if self._productive(message):
                break
            count += 1
        return count

    def check(self, state: AgentState | TaskState) -> None:
        count = self.unproductive_turns(state)
        if count < self.turns:
            return
        what = (
            "a tool call"
            if not self.unproductive_tools
            else f"a tool call other than {', '.join(sorted(self.unproductive_tools))}"
        )
        _raise_limit(
            count,
            self.turns,
            f"Unproductive loop limit reached: {count:,} consecutive turns without "
            f"{what}; limit: {self.turns:,}",
        )


class RepeatedTextLimit:
    """Stop a sample after `turns` consecutive turns with no tool call and identical text.

    A turn is an assistant message. The limit fires when the last `turns` assistant
    messages all have no tool calls and the same visible text. Reasoning content is
    ignored, so a model whose visible reply repeats while its thinking varies still
    trips the limit. Any turn with a tool call, or with different text, resets the run.
    """

    def __init__(self, turns: int) -> None:
        if turns < 2:
            raise ValueError(f"turns must be at least 2: {turns}")
        self.turns = turns

    def repeated_turns(self, state: AgentState | TaskState) -> int:
        messages = _assistant_turns(state)
        if not messages or messages[-1].tool_calls:
            return 0
        text = messages[-1].text
        count = 0
        for message in reversed(messages):
            if message.tool_calls or message.text != text:
                break
            count += 1
        return count

    def check(self, state: AgentState | TaskState) -> None:
        count = self.repeated_turns(state)
        if count < self.turns:
            return
        _raise_limit(
            count,
            self.turns,
            f"Repeated text limit reached: {count:,} consecutive turns with no tool "
            f"call and identical text; limit: {self.turns:,}",
        )
