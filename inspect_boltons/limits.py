from __future__ import annotations

from collections.abc import Collection

from inspect_ai.agent import AgentState
from inspect_ai.event import SampleLimitEvent
from inspect_ai.log import transcript
from inspect_ai.model import ChatMessageAssistant
from inspect_ai.solver import TaskState
from inspect_ai.util import LimitExceededError


class UnproductiveLoopLimit:
    """Stop a sample once the model has gone `turns` consecutive turns without a tool call.

    A turn is an assistant message. A turn counts as productive if it calls any tool
    other than those named in `unproductive_tools`. Call `check(state)` after each
    turn (e.g. from a `react()` `on_continue` hook); it raises `LimitExceededError`
    once the trailing run of unproductive turns reaches `turns`.
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
        for message in reversed(state.messages):
            if not isinstance(message, ChatMessageAssistant):
                continue
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
        message = (
            f"Unproductive loop limit reached: {count:,} consecutive turns without "
            f"{what}; limit: {self.turns:,}"
        )
        transcript()._event(
            SampleLimitEvent(type="custom", limit=self.turns, message=message)
        )
        raise LimitExceededError(
            "custom", value=count, limit=self.turns, message=message
        )
