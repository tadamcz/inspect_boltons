"""Extract the agent transcript from an Inspect `.eval` log into plain text.

The transcript is reconstructed from the sample's `model` events (each carries its
turn's input messages plus the assistant output), deduped by message id in event
order, rather than taken from `sample.messages`. This works whether the agent loop
ran directly as the solver or inside its own `AgentState`.

If subagents were spawned, only the main loop is kept: a model event belongs to the
main loop iff its enclosing span chain contains the minimum number of `agent` spans.
Turns nested in a further `agent` span are dropped. For a plain `react` loop there
are no subagents, so everything is kept.

Each sample gets its own directory containing `info.json`, `messages.txt`,
`scores.txt`, `scores.json` and, if the transcript contains compaction summaries,
`compactions.txt`. The eval-level scores go in a `scores.json` alongside the sample
directories.

Usage:

    ibolt plain logs/run.eval
    ibolt plain logs/some-dir/ -o /tmp/out
    ibolt plain logs/run.eval --list-samples
    ibolt plain logs/run.eval -s some_sample_id
    ibolt plain logs/some-dir/ --parallel-evals --parallel-samples 4
"""

from __future__ import annotations

import json
import re
import sys
from collections import Counter
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import TextIO

from inspect_ai.log import (
    EvalSample,
    read_eval_log,
    read_eval_log_sample,
    read_eval_log_sample_summaries,
    resolve_sample_attachments,
)
from inspect_ai.model import (
    ChatMessage,
    ChatMessageAssistant,
    ChatMessageSystem,
    ChatMessageTool,
    ChatMessageUser,
    ContentImage,
    ContentReasoning,
    ContentText,
)
from inspect_ai.tool import ToolCall


def extract_text(msg: ChatMessage) -> str:
    if isinstance(msg.content, str):
        return msg.content
    parts = []
    for block in msg.content:
        if isinstance(block, ContentText):
            parts.append(block.text)
        elif isinstance(block, ContentReasoning):
            if block.redacted:
                parts.append(
                    block.summary
                    if block.summary is not None
                    else "[redacted reasoning]"
                )
            else:
                parts.append(block.reasoning)
        elif isinstance(block, ContentImage):
            parts.append("[image]")
        else:
            parts.append(f"[{type(block).__name__}]")
    return "\n".join(parts)


def format_tool_call(tc: ToolCall) -> str:
    """Render a tool call.

    Inspect's built-in `bash` and `text_editor` get compact renderings; anything
    else falls back to a JSON dump of the arguments.
    """
    fn = tc.function
    args = tc.arguments

    if fn == "bash" and "command" in args:
        return f">>> bash\n```\n{args['command']}\n```"

    if fn == "text_editor":
        cmd = args.get("command", "")
        path = args.get("path", "")
        parts = [f">>> text_editor {cmd} {path}".strip()]
        old = args.get("old_str", "")
        new = args.get("new_str", "")
        file_text = args.get("file_text", "")
        if cmd == "view":
            view_range = args.get("view_range")
            if view_range:
                parts.append(f"lines {view_range}")
        elif cmd == "str_replace" and old:
            parts.append(f"OLD:\n{old}\nNEW:\n{new}")
        elif cmd == "insert":
            parts.append(f"INSERT after line {args.get('insert_line', '')}:\n{new}")
        elif cmd == "create" and file_text:
            parts.append(file_text)
        else:
            remaining = {k: v for k, v in args.items() if k not in ("command", "path")}
            if remaining:
                parts.append(json.dumps(remaining, ensure_ascii=False))
        return "\n".join(parts)

    return f">>> {fn}({json.dumps(args, ensure_ascii=False)})"


def get_primary_model(messages: list[ChatMessage]) -> str | None:
    models = [
        msg.model
        for msg in messages
        if isinstance(msg, ChatMessageAssistant) and msg.model
    ]
    if not models:
        return None
    return Counter(models).most_common(1)[0][0]


def format_message(msg: ChatMessage, idx: int, primary_model: str | None = None) -> str:
    n = idx + 1
    lines: list[str] = []

    if isinstance(msg, ChatMessageSystem):
        lines.append(f"[{n}] === SYSTEM ===")
        lines.append(extract_text(msg))

    elif isinstance(msg, ChatMessageUser):
        source = msg.source or ""
        tag = f" ({source})" if source and source != "input" else ""
        lines.append(f"[{n}] --- USER{tag} ---")
        lines.append(extract_text(msg))

    elif isinstance(msg, ChatMessageAssistant):
        model = msg.model or ""
        model_tag = f" [{model}]" if model and model != primary_model else ""
        lines.append(f"[{n}] --- ASSISTANT{model_tag} ---")
        text = extract_text(msg)
        if text:
            lines.append(text)
        for tc in msg.tool_calls or []:
            lines.append(format_tool_call(tc))

    elif isinstance(msg, ChatMessageTool):
        lines.append(f"[{n}] --- TOOL ({msg.function}) ---")
        if msg.error:
            lines.append(f"[ERROR] {msg.error.message}")
        text = extract_text(msg)
        if text:
            lines.append(text)

    return "\n".join(lines)


def _is_compaction_summary(msg: ChatMessage) -> bool:
    return bool(msg.metadata and msg.metadata.get("summary"))


def _extract_summary_body(text: str) -> str:
    m = re.search(r"<summary>\s*\n?(.*?)\n?\s*</summary>", text, re.DOTALL)
    return m.group(1).strip() if m else text


def _agent_depth(
    span_id: str | None,
    parent: dict[str, str | None],
    span_type: dict[str, str | None],
) -> int:
    depth = 0
    sid = span_id
    while sid is not None:
        if span_type[sid] == "agent":
            depth += 1
        sid = parent[sid]
    return depth


def main_loop_messages(sample: EvalSample) -> list[ChatMessage]:
    """Reconstruct the main agent loop's conversation from the sample's events.

    See the module docstring.
    """
    spans = [e for e in sample.events if e.event == "span_begin"]
    parent = {s.id: s.parent_id for s in spans}
    span_type = {s.id: s.type for s in spans}

    model_events = [e for e in sample.events if e.event == "model"]
    if not model_events:
        return []
    depths = [_agent_depth(e.span_id, parent, span_type) for e in model_events]
    main_depth = min(depths)

    seen: set[str] = set()
    messages: list[ChatMessage] = []

    def add(m: ChatMessage) -> None:
        if m.id is not None:
            if m.id in seen:
                return
            seen.add(m.id)
        messages.append(m)

    for event, depth in zip(model_events, depths):
        if depth != main_depth:
            continue
        for m in event.input:
            add(m)
        if event.output.choices:
            add(event.output.choices[0].message)

    return messages


def _enumerate_messages(messages: list[ChatMessage]) -> list[tuple[int, ChatMessage]]:
    """Group tool messages under the assistant message that triggered them.

    Mirrors the Inspect UI's message grouping: a tool result shares the index of
    the preceding non-tool message rather than getting its own.
    """
    results = []
    index = -1
    for message in messages:
        if index == -1 or not isinstance(message, ChatMessageTool):
            index += 1
        results.append((index, message))
    return results


def write_transcript(messages: list[ChatMessage], out: TextIO) -> None:
    primary_model = get_primary_model(messages)
    for i, msg in _enumerate_messages(messages):
        out.write(format_message(msg, i, primary_model) + "\n\n")


def compaction_summaries(messages: list[ChatMessage]) -> list[tuple[int, ChatMessage]]:
    return [
        (i, msg)
        for i, msg in _enumerate_messages(messages)
        if _is_compaction_summary(msg)
    ]


def write_compactions(summaries: list[tuple[int, ChatMessage]], out: TextIO) -> None:
    for seq, (idx, msg) in enumerate(summaries, 1):
        body = _extract_summary_body(extract_text(msg))
        out.write(
            f"--- Compaction {seq}/{len(summaries)} (after message {idx + 1}) ---\n"
            f"{body}\n\n"
        )


def write_scores(sample: EvalSample, out: TextIO) -> None:
    if not sample.scores:
        return
    out.write("=== SCORES ===\n")
    for scorer_name, score in sample.scores.items():
        out.write(f"{scorer_name}: {score.value}\n")
        if score.explanation:
            for line in score.explanation.splitlines():
                out.write(f"  {line}\n")
    out.write("\n")


def _write_json(payload: object, out: TextIO) -> None:
    json.dump(payload, out, indent=2, ensure_ascii=False, default=str)
    out.write("\n")


def write_scores_json(sample: EvalSample, out: TextIO) -> None:
    _write_json(
        {
            name: {
                "value": score.value,
                "explanation": score.explanation,
                "metadata": score.metadata,
            }
            for name, score in (sample.scores or {}).items()
        },
        out,
    )


def write_info(sample: EvalSample, out: TextIO) -> None:
    dumped = sample.model_dump(include={"error", "limit", "model_usage"}, mode="json")
    _write_json(
        {
            "id": str(sample.id),
            "epoch": sample.epoch,
            "uuid": sample.uuid,
            "target": sample.target,
            "error": dumped["error"],
            "limit": dumped["limit"],
            "model_usage": dumped["model_usage"],
        },
        out,
    )


def list_samples(eval_path: Path) -> list[str]:
    summaries = read_eval_log_sample_summaries(str(eval_path))
    return sorted({str(s.id) for s in summaries})


def _sample_specs(
    eval_path: Path, sample_ids: set[str] | None
) -> list[tuple[str, str | int, int]]:
    """List the (output stem, sample id, epoch) of every sample to extract."""
    log = read_eval_log(str(eval_path), header_only=True)
    epochs = log.eval.config.epochs or 1
    specs: list[tuple[str, str | int, int]] = []
    for s in read_eval_log_sample_summaries(str(eval_path)):
        sid = str(s.id)
        if sample_ids is not None and sid not in sample_ids:
            continue
        stem = f"{sid}_ep{s.epoch:03d}" if epochs > 1 else sid
        specs.append((stem, s.id, s.epoch))
    return specs


def default_output_dir(eval_path: Path) -> Path:
    return eval_path.parent / (eval_path.stem + "_plaintext")


def _extract_sample(
    eval_path: Path,
    out_dir: Path,
    stem: str,
    sample_id: str | int,
    epoch: int,
) -> None:
    sample = resolve_sample_attachments(
        read_eval_log_sample(str(eval_path), id=sample_id, epoch=epoch)
    )

    sample_dir = out_dir / stem
    sample_dir.mkdir(parents=True, exist_ok=True)

    def write(name: str, writer: Callable[[TextIO], None]) -> None:
        path = sample_dir / name
        with open(path, "w") as f:
            writer(f)
        print(f"{stem}: {path.stat().st_size:,} bytes -> {path}", file=sys.stderr)

    write("info.json", lambda f: write_info(sample, f))

    messages = main_loop_messages(sample)
    write("messages.txt", lambda f: write_transcript(messages, f))
    summaries = compaction_summaries(messages)
    if summaries:
        write("compactions.txt", lambda f: write_compactions(summaries, f))

    write("scores.txt", lambda f: write_scores(sample, f))
    write("scores.json", lambda f: write_scores_json(sample, f))


def extract_eval_file(
    eval_path: Path,
    out_dir: Path,
    sample_ids: set[str] | None = None,
    *,
    sample_workers: int = 1,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    log = read_eval_log(str(eval_path), header_only=True)
    scores = log.results.model_dump(mode="json")["scores"] if log.results else []
    scores_path = out_dir / "scores.json"
    with open(scores_path, "w") as f:
        _write_json(scores, f)
    print(
        f"eval: {scores_path.stat().st_size:,} bytes -> {scores_path}", file=sys.stderr
    )

    specs = _sample_specs(eval_path, sample_ids)
    workers = min(sample_workers, len(specs))

    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = [
                pool.submit(
                    _extract_sample,
                    eval_path,
                    out_dir,
                    stem,
                    sid,
                    epoch,
                )
                for stem, sid, epoch in specs
            ]
            for future in futures:
                future.result()
        return

    for stem, sid, epoch in specs:
        _extract_sample(
            eval_path,
            out_dir,
            stem,
            sid,
            epoch,
        )
