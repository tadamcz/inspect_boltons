"""Offline tests for the checked compaction strategies, driven by mockllm outputs."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest
from inspect_ai import Task, eval
from inspect_ai.agent import react
from inspect_ai.dataset import Sample
from inspect_ai.event import InfoEvent, ModelEvent
from inspect_ai.log import EvalSample, transcript
from inspect_ai.log._transcript import Transcript, init_transcript
from inspect_ai.model import (
    ChatMessage,
    ChatMessageAssistant,
    ChatMessageUser,
    CompactionAuto,
    CompactionSummary,
    ContentReasoning,
    GenerateConfig,
    Model,
    ModelOutput,
    compaction,
    get_model,
)
from inspect_ai.tool import ToolCall, ToolChoice, ToolInfo

from inspect_boltons.compaction import (
    MAX_SUMMARY_ATTEMPTS,
    MIN_SUMMARY_CHARS,
    REJECTED_SUMMARY_SOURCE,
    CheckedCompactionAuto,
    CheckedCompactionSummary,
    CompactionSummaryError,
)

MODEL = "mockllm/model"
# In tokens, as counted by `count_text_tokens` below. The conversation is above it, and
# the task prompt plus a good summary is below it.
THRESHOLD = 1_000
AGENT_TURN = "I read the next source file and ported one more function. " * 50
SUMMARY = "The agent ported the option parser. Two visible cases still fail. " * 20
GOOD_COMPLETION = (
    "<analysis>\nThe user asked for a port.\n</analysis>\n\n"
    f"<summary>\n{SUMMARY}\n</summary>"
)
# What a concise model writes when the task caps the summary's length.
SHORT_SUMMARY = "Stage 1 had 50 lines. Stage 2 is next."
SHORT_COMPLETION = (
    "<analysis>\nTurn 1 answered, turn 2 pending.\n</analysis>\n\n"
    f"<summary>\n{SHORT_SUMMARY}\n</summary>"
)
# The length of the block notice OpenAI returned as a summary in a real run.
BLOCK_NOTICE_CHARS = 311
BLOCK_NOTICE = ("The provider blocked this response under its usage policies. " * 6)[
    :BLOCK_NOTICE_CHARS
]
ANALYSIS_THEN_NOTHING = (
    "<analysis>\n" + "The agent ran the tests again. " * 200 + "\n</analysis>\n\n"
    "<summary>\n</summary>"
)

GOOD = ModelOutput.from_content(MODEL, GOOD_COMPLETION)
BLANK = ModelOutput.from_content(MODEL, "", stop_reason="unknown")
REJECTED = {
    "no text, stop reason unknown": BLANK,
    "reasoning only, max_tokens": ModelOutput.from_content(
        MODEL,
        [ContentReasoning(reasoning="Planning the summary.")],
        stop_reason="max_tokens",
    ),
    "block notice, content_filter": ModelOutput.from_content(
        MODEL, BLOCK_NOTICE, stop_reason="content_filter"
    ),
    "one character, stop": ModelOutput.from_content(MODEL, ".", stop_reason="stop"),
    "long analysis then an empty summary, stop": ModelOutput.from_content(
        MODEL, ANALYSIS_THEN_NOTHING, stop_reason="stop"
    ),
}
rejected_output = pytest.mark.parametrize(
    "rejected", REJECTED.values(), ids=REJECTED.keys()
)


@pytest.fixture(autouse=True)
def fresh_transcript() -> None:
    """Without this, a test sees the events of whichever test ran before it."""
    init_transcript(Transcript())


class ScriptedCalls:
    """Hands out the given outputs in order and keeps the input of every model call."""

    def __init__(self, outputs: Sequence[ModelOutput]) -> None:
        self.outputs = list(outputs)
        self.inputs: list[list[ChatMessage]] = []

    def __call__(
        self,
        input: list[ChatMessage],
        tools: list[ToolInfo],
        tool_choice: ToolChoice,
        config: GenerateConfig,
    ) -> ModelOutput:
        self.inputs.append(list(input))
        return self.outputs.pop(0)


def scripted_model(
    monkeypatch: pytest.MonkeyPatch, outputs: Sequence[ModelOutput]
) -> tuple[Model, ScriptedCalls]:
    calls = ScriptedCalls(outputs)
    model = get_model(MODEL, custom_outputs=calls, memoize=False)

    # Inspect's own counter downloads a tokenizer file.
    async def count_text_tokens(text: str) -> int:
        return len(text) // 4

    monkeypatch.setattr(model.api, "count_text_tokens", count_text_tokens)
    return model, calls


def conversation() -> list[ChatMessage]:
    return [
        ChatMessageUser(content="Port the program to the target language."),
        ChatMessageAssistant(content=AGENT_TURN),
        ChatMessageUser(content="Please continue."),
        ChatMessageAssistant(content=AGENT_TURN),
    ]


def rejected_summary_events() -> list[InfoEvent]:
    return [
        event
        for event in transcript().events
        if isinstance(event, InfoEvent) and event.source == REJECTED_SUMMARY_SOURCE
    ]


def contents(messages: Sequence[ChatMessage]) -> list[Any]:
    return [message.content for message in messages]


async def test_good_summary_is_used_as_the_stock_strategy_would(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model, calls = scripted_model(monkeypatch, [GOOD])
    stock_model, _ = scripted_model(monkeypatch, [GOOD])

    model_input, summary = await CheckedCompactionSummary(threshold=THRESHOLD).compact(
        model, conversation(), []
    )
    stock_input, _ = await CompactionSummary(threshold=THRESHOLD).compact(
        stock_model, conversation(), []
    )

    assert summary is not None and SUMMARY in summary.text
    assert "The user asked for a port." not in summary.text
    assert contents(model_input) == contents(stock_input)
    assert len(calls.inputs) == 1
    assert rejected_summary_events() == []


async def test_short_summary_that_says_something_is_accepted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model, calls = scripted_model(
        monkeypatch, [ModelOutput.from_content(MODEL, SHORT_COMPLETION)]
    )
    strategy = CheckedCompactionSummary(
        threshold=THRESHOLD, instructions="Keep the summary under 150 words."
    )

    _, summary = await strategy.compact(model, conversation(), [])

    assert summary is not None and SHORT_SUMMARY in summary.text
    assert len(calls.inputs) == 1
    assert rejected_summary_events() == []


@rejected_output
async def test_unusable_summary_is_rejected_and_the_call_made_again(
    monkeypatch: pytest.MonkeyPatch, rejected: ModelOutput
) -> None:
    model, calls = scripted_model(monkeypatch, [rejected, GOOD])

    model_input, summary = await CheckedCompactionSummary(threshold=THRESHOLD).compact(
        model, conversation(), []
    )

    assert summary is not None and SUMMARY in summary.text
    assert model_input[-1] is summary
    assert len(calls.inputs) == 2
    assert calls.inputs[0] == calls.inputs[1]
    assert len(rejected_summary_events()) == 1


async def test_summary_just_long_enough_is_accepted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    shortest = "s" * MIN_SUMMARY_CHARS
    model, calls = scripted_model(
        monkeypatch,
        [
            ModelOutput.from_content(MODEL, shortest[1:]),
            ModelOutput.from_content(MODEL, shortest),
        ],
    )

    _, summary = await CheckedCompactionSummary(threshold=THRESHOLD).compact(
        model, conversation(), []
    )

    assert summary is not None and shortest in summary.text
    assert len(calls.inputs) == 2


async def test_rejected_calls_are_recorded_in_the_transcript(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model, _ = scripted_model(
        monkeypatch,
        [
            REJECTED["no text, stop reason unknown"],
            REJECTED["block notice, content_filter"],
            REJECTED["long analysis then an empty summary, stop"],
            GOOD,
        ],
    )

    await CheckedCompactionSummary(threshold=THRESHOLD, max_attempts=4).compact(
        model, conversation(), []
    )

    limits = {"max_attempts": 4, "min_summary_chars": MIN_SUMMARY_CHARS}
    assert [event.data for event in rejected_summary_events()] == [
        {"attempt": 1, "stop_reason": "unknown", "summary_chars": 0, **limits},
        {
            "attempt": 2,
            "stop_reason": "content_filter",
            "summary_chars": BLOCK_NOTICE_CHARS,
            **limits,
        },
        {"attempt": 3, "stop_reason": "stop", "summary_chars": 0, **limits},
    ]


@rejected_output
async def test_compaction_fails_and_leaves_the_conversation_alone_when_every_call_is_rejected(
    monkeypatch: pytest.MonkeyPatch, rejected: ModelOutput
) -> None:
    model, calls = scripted_model(
        monkeypatch, [rejected] * MAX_SUMMARY_ATTEMPTS + [GOOD]
    )
    messages = conversation()
    messages_before = [message.model_copy(deep=True) for message in messages]
    compact = compaction(
        CheckedCompactionSummary(threshold=THRESHOLD), prefix=messages[:1], model=model
    )

    with pytest.raises(CompactionSummaryError) as error:
        await compact.compact_input(messages)

    assert messages == messages_before
    assert len(calls.inputs) == MAX_SUMMARY_ATTEMPTS
    assert f"stop reason '{rejected.stop_reason}'" in str(error.value)
    assert len(rejected_summary_events()) == MAX_SUMMARY_ATTEMPTS

    # The failure left nothing behind in the handler: the next compaction summarizes the
    # whole conversation.
    model_input, summary = await compact.compact_input(messages)
    assert summary is not None and SUMMARY in summary.text
    assert model_input == [messages[0], summary]
    assert all(message in calls.inputs[-1] for message in messages)


async def test_summary_call_that_overflows_the_context_window_is_not_repeated(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model, calls = scripted_model(
        monkeypatch,
        [ModelOutput.from_content(MODEL, "", stop_reason="model_length"), GOOD],
    )

    with pytest.raises(RuntimeError) as error:
        await CheckedCompactionSummary(threshold=THRESHOLD).compact(
            model, conversation(), []
        )

    assert not isinstance(error.value, CompactionSummaryError)
    assert len(calls.inputs) == 1


async def test_attempts_and_minimum_length_can_be_set_per_strategy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    short = ModelOutput.from_content(MODEL, SHORT_SUMMARY)
    model, calls = scripted_model(monkeypatch, [short, short, GOOD])
    strategy = CheckedCompactionSummary(
        threshold=THRESHOLD, max_attempts=2, min_summary_chars=len(SHORT_SUMMARY) + 1
    )

    with pytest.raises(CompactionSummaryError):
        await strategy.compact(model, conversation(), [])

    assert len(calls.inputs) == 2


def memory_conversation() -> list[ChatMessage]:
    """A conversation in which the agent saved a note with the memory tool."""
    save = ToolCall(
        id="call_1", function="memory", arguments={"command": "view", "path": "/"}
    )
    return [
        *conversation(),
        ChatMessageAssistant(content="Saving a note.", tool_calls=[save]),
    ]


@pytest.mark.parametrize(
    "arguments",
    [
        {"instructions": "Keep the summary under 150 words."},
        {"prompt": "Summarize the work so far in plain prose.\n\n{addendums}"},
        {
            "prompt": "Summarize the work so far in plain prose.\n\n{addendums}",
            "instructions": "List every file that was changed.",
        },
        {"memory": False},
        {"memory": True},
    ],
    ids=[
        "instructions",
        "prompt",
        "prompt and instructions",
        "memory off",
        "memory on",
    ],
)
async def test_summary_arguments_reach_the_summary_call_as_in_the_stock_strategy(
    monkeypatch: pytest.MonkeyPatch, arguments: dict[str, Any]
) -> None:
    model, calls = scripted_model(monkeypatch, [GOOD])
    stock_model, stock_calls = scripted_model(monkeypatch, [GOOD])
    checked = CheckedCompactionSummary(threshold=THRESHOLD, **arguments)
    stock = CompactionSummary(threshold=THRESHOLD, **arguments)

    model_input, _ = await checked.compact(model, memory_conversation(), [])
    stock_input, _ = await stock.compact(stock_model, memory_conversation(), [])

    assert contents(calls.inputs[0]) == contents(stock_calls.inputs[0])
    assert contents(model_input) == contents(stock_input)
    assert (checked.threshold, checked.memory) == (stock.threshold, stock.memory)


async def test_analysis_counts_as_summary_when_the_prompt_never_asks_for_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # With a prompt of its own, `CompactionSummary` keeps the whole completion.
    model, calls = scripted_model(
        monkeypatch, [ModelOutput.from_content(MODEL, ANALYSIS_THEN_NOTHING)]
    )
    strategy = CheckedCompactionSummary(
        threshold=10_000, prompt="Summarize the work so far."
    )

    _, summary = await strategy.compact(model, conversation(), [])

    assert summary is not None and ANALYSIS_THEN_NOTHING in summary.text
    assert len(calls.inputs) == 1


async def test_separate_summary_model_is_checked_too(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    agent_model, agent_calls = scripted_model(monkeypatch, [])
    summary_model, summary_calls = scripted_model(monkeypatch, [BLANK, GOOD])
    strategy = CheckedCompactionSummary(threshold=THRESHOLD, model=summary_model)

    _, summary = await strategy.compact(agent_model, conversation(), [])

    assert summary is not None and SUMMARY in summary.text
    assert len(summary_calls.inputs) == 2
    assert agent_calls.inputs == []


async def test_auto_strategy_checks_its_summary_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # mockllm has no native compaction, so the strategy falls back to a summary.
    model, calls = scripted_model(monkeypatch, [BLANK, GOOD])
    stock_model, stock_calls = scripted_model(monkeypatch, [GOOD])
    arguments: dict[str, Any] = {
        "threshold": THRESHOLD,
        "instructions": "List every file.",
    }

    model_input, summary = await CheckedCompactionAuto(**arguments).compact(
        model, conversation(), []
    )
    stock_input, _ = await CompactionAuto(**arguments).compact(
        stock_model, conversation(), []
    )

    assert summary is not None and SUMMARY in summary.text
    assert len(calls.inputs) == 2
    assert contents(calls.inputs[-1]) == contents(stock_calls.inputs[0])
    assert contents(model_input) == contents(stock_input)
    assert len(rejected_summary_events()) == 1


async def test_auto_strategy_fails_when_every_fallback_summary_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model, calls = scripted_model(monkeypatch, [BLANK, BLANK, GOOD])

    with pytest.raises(CompactionSummaryError):
        await CheckedCompactionAuto(THRESHOLD, max_attempts=2).compact(
            model, conversation(), []
        )

    assert len(calls.inputs) == 2


# One turn long enough to put a `react()` conversation over THRESHOLD.
LONG_TURN = ModelOutput.from_content(MODEL, AGENT_TURN * 2)
SUBMIT = ModelOutput.for_tool_call(MODEL, "submit", {"answer": "42"})


def run_react_agent(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, outputs: Sequence[ModelOutput]
) -> EvalSample:
    model, _ = scripted_model(monkeypatch, outputs)
    task = Task(
        dataset=[Sample(input="What is the answer?", target="42")],
        solver=react(compaction=CheckedCompactionSummary(threshold=THRESHOLD)),
    )
    [log] = eval(task, model=model, log_dir=str(tmp_path), display="none")
    assert log.samples is not None
    [sample] = log.samples
    return sample


def summaries(sample: EvalSample) -> list[str]:
    return [m.text for m in sample.messages if (m.metadata or {}).get("summary")]


def test_react_agent_asks_again_for_a_blank_summary(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    sample = run_react_agent(monkeypatch, tmp_path, [LONG_TURN, BLANK, GOOD, SUBMIT])

    assert sample.error is None
    assert sample.output.completion.endswith("42")
    [summary] = summaries(sample)
    assert SUMMARY in summary
    rejected = [
        event.data
        for event in sample.events
        if isinstance(event, InfoEvent) and event.source == REJECTED_SUMMARY_SOURCE
    ]
    assert rejected == [
        {
            "attempt": 1,
            "max_attempts": MAX_SUMMARY_ATTEMPTS,
            "stop_reason": "unknown",
            "summary_chars": 0,
            "min_summary_chars": MIN_SUMMARY_CHARS,
        }
    ]


def test_react_agent_sample_errors_with_its_history_intact_when_no_summary_is_usable(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    sample = run_react_agent(
        monkeypatch, tmp_path, [LONG_TURN] + [BLANK] * MAX_SUMMARY_ATTEMPTS + [SUBMIT]
    )

    assert sample.error is not None
    assert "CompactionSummaryError" in sample.error.message
    assert summaries(sample) == []
    assert AGENT_TURN * 2 in [m.text for m in sample.messages]
    model_events = [event for event in sample.events if isinstance(event, ModelEvent)]
    assert len(model_events) == 1 + MAX_SUMMARY_ATTEMPTS
