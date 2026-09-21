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
    turn; it raises `LimitExceededError` once the trailing run of unproductive turns
    reaches `turns`. Inspect handles that error like any other limit: the sample ends,
    is scored on its messages so far, and is recorded with a limit of type "custom".

    With `react()`, call it from the `on_continue` hook, which runs after each turn's
    generation and tool calls:

        limit = UnproductiveLoopLimit(turns=5, unproductive_tools=["think"])

        async def on_continue(state: AgentState) -> bool:
            limit.check(state)
            return True

        agent = react(tools=[bash(), python(), think()], on_continue=on_continue)

    In a custom solver, call it after each `generate()` and tool execution:

        @solver
        def my_solver() -> Solver:
            limit = UnproductiveLoopLimit(turns=5)

            async def solve(state: TaskState, generate: Generate) -> TaskState:
                while not state.completed:
                    state = await generate(state, tool_calls="single")
                    limit.check(state)
                return state

            return solve

    The instance holds only configuration, so one can be shared across samples.
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
