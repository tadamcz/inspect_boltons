"""Summary compaction that never replaces a conversation with a blank or broken summary.

`CheckedCompactionSummary` and `CheckedCompactionAuto` are drop-in replacements for
Inspect's `CompactionSummary` and `CompactionAuto`, and take the same arguments:

    agent = react(tools=[bash()], compaction=CheckedCompactionSummary())
"""

from __future__ import annotations

from typing import Any

from inspect_ai.log import transcript
from inspect_ai.model import (
    ChatMessage,
    ChatMessageUser,
    CompactionAuto,
    CompactionSummary,
    Model,
    ModelOutput,
)

# Inspect's own rule for which part of a completion it keeps as the summary. Imported,
# not copied, so the check always measures the text Inspect would keep.
from inspect_ai.model._compaction.summary import _ANALYSIS_OPEN, _summary_text
from inspect_ai.tool import ToolInfo

# Calls made for one summary before giving up, the first one included.
MAX_SUMMARY_ATTEMPTS = 3
# Fewest characters a summary can have and still be kept. The broken summaries seen in
# real runs had 0 or 1 characters, while a good summary can be a single sentence: one of
# 40 characters has carried an agent through its task. The floor sits just above "no
# summary at all" so that a short summary which does say something is never turned
# away: this check is for blank summaries, not thin ones.
MIN_SUMMARY_CHARS = 20
# `source` of the transcript info event written for every rejected summary call.
REJECTED_SUMMARY_SOURCE = "compaction_summary_rejected"

_SUMMARY_TAGS = ("<summary>", "</summary>")


class CompactionSummaryError(RuntimeError):
    """Every summary call was rejected, so the conversation was not compacted."""


class CheckedCompactionSummary(CompactionSummary):
    """`CompactionSummary` that only compacts with a summary worth keeping.

    A summary call can succeed and still return no usable summary: no text at all, an
    answer cut off before any text, or a provider's block notice. `CompactionSummary`
    builds the summary message from whatever came back, which leaves the agent with the
    task prompt and nothing of its own work.

    Here a summary counts only if the call ended with stop reason "stop" and the text
    that `CompactionSummary` would keep has at least `min_summary_chars` characters.
    Otherwise the call is made again, up to `max_attempts` times in total. Every
    rejected call is written to the transcript as an info event with source
    `REJECTED_SUMMARY_SOURCE`. If no call passes, `CompactionSummaryError` fails the
    sample with the conversation left as it was.

    Takes the same arguments as `CompactionSummary`, plus `max_attempts` and
    `min_summary_chars`.
    """

    def __init__(
        self,
        *,
        max_attempts: int = MAX_SUMMARY_ATTEMPTS,
        min_summary_chars: int = MIN_SUMMARY_CHARS,
        **compaction_summary_args: Any,
    ) -> None:
        if max_attempts < 1:
            raise ValueError(f"max_attempts must be at least 1, got {max_attempts}")
        super().__init__(**compaction_summary_args)
        self.max_attempts = max_attempts
        self.min_summary_chars = min_summary_chars
        # `CompactionSummary.compact` prefers this model over the one it is handed, so
        # the check has to sit on it too.
        if self.model is not None:
            self.model = _CheckedSummaryModel(self.model, self)

    def _repr_params_(self) -> dict[str, Any]:
        params = super()._repr_params_()
        params.update(
            {
                "max_attempts": self.max_attempts,
                "min_summary_chars": self.min_summary_chars,
            }
        )
        return params

    async def compact(
        self, model: Model, messages: list[ChatMessage], tools: list[ToolInfo]
    ) -> tuple[list[ChatMessage], ChatMessageUser | None]:
        return await super().compact(_CheckedSummaryModel(model, self), messages, tools)

    def summary_chars(self, completion: str) -> int:
        """Length of the summary `CompactionSummary` would keep from this completion.

        `CompactionSummary` drops the model's `<analysis>` scratchpad when its prompt
        asks for one, so a long scratchpad followed by an empty summary counts as empty
        here. The `<summary>` tags themselves are not counted either.
        """
        kept = (
            _summary_text(completion) if _ANALYSIS_OPEN in self.prompt else completion
        )
        for tag in _SUMMARY_TAGS:
            kept = kept.replace(tag, "")
        return len(kept.strip())


class CheckedCompactionAuto(CompactionAuto):
    """`CompactionAuto` whose summary fallback is `CheckedCompactionSummary`.

    Native compaction is tried first, exactly as in `CompactionAuto`. Takes the same
    arguments as `CompactionAuto`, plus `max_attempts` and `min_summary_chars` for the
    fallback.
    """

    def __init__(
        self,
        *compaction_auto_args: Any,
        max_attempts: int = MAX_SUMMARY_ATTEMPTS,
        min_summary_chars: int = MIN_SUMMARY_CHARS,
        **compaction_auto_kwargs: Any,
    ) -> None:
        super().__init__(*compaction_auto_args, **compaction_auto_kwargs)
        stock = self._summary
        self._summary = CheckedCompactionSummary(
            threshold=stock.threshold,
            memory=stock.memory,
            model=stock.model,
            instructions=stock.instructions,
            prompt=stock.prompt,
            max_attempts=max_attempts,
            min_summary_chars=min_summary_chars,
        )


class _CheckedSummaryModel(Model):
    """Stands in for the summary model. `generate` only returns a summary that passes."""

    def __init__(self, model: Model, strategy: CheckedCompactionSummary) -> None:
        super().__init__(model.api, model.config, model.model_args)
        self._model = model
        self._strategy = strategy

    async def generate(self, *args: Any, **kwargs: Any) -> ModelOutput:
        max_attempts = self._strategy.max_attempts
        min_summary_chars = self._strategy.min_summary_chars
        rejected: list[str] = []
        for attempt in range(1, max_attempts + 1):
            output = await self._model.generate(*args, **kwargs)
            # "model_length" means the input is too long, which another call cannot fix.
            # `CompactionSummary` raises its own error for it.
            if output.stop_reason == "model_length":
                return output
            summary_chars = self._strategy.summary_chars(output.completion)
            if output.stop_reason == "stop" and summary_chars >= min_summary_chars:
                return output
            rejected.append(
                f"stop reason {output.stop_reason!r}, {summary_chars} characters"
            )
            transcript().info(
                {
                    "attempt": attempt,
                    "max_attempts": max_attempts,
                    "stop_reason": output.stop_reason,
                    "summary_chars": summary_chars,
                    "min_summary_chars": min_summary_chars,
                },
                source=REJECTED_SUMMARY_SOURCE,
            )
        raise CompactionSummaryError(
            f"All {max_attempts} compaction summary calls were rejected "
            f"({'; '.join(rejected)}). A summary must end with stop reason 'stop' and "
            f"have at least {min_summary_chars} characters. The conversation was not "
            "compacted."
        )
