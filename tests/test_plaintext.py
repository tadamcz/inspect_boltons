from __future__ import annotations

import itertools
import json
from pathlib import Path

import pytest
from click.testing import CliRunner
from inspect_ai import Task, eval
from inspect_ai.agent import react
from inspect_ai.dataset import Sample
from inspect_ai.model import ModelOutput, get_model
from inspect_ai.scorer import includes
from inspect_ai.tool import bash

from inspect_boltons.cli import main


@pytest.fixture
def eval_path(tmp_path: Path) -> Path:
    outputs = itertools.chain(
        [ModelOutput.for_tool_call("mockllm/model", "bash", {"command": "echo hi"})],
        itertools.repeat(
            ModelOutput.for_tool_call("mockllm/model", "submit", {"answer": "done"})
        ),
    )
    model = get_model("mockllm/model", custom_outputs=outputs)
    task = Task(
        dataset=[Sample(id="s1", input="Do the thing.", target="done")],
        solver=react(tools=[bash()]),
        scorer=includes(),
        sandbox="local",
    )
    [log] = eval(task, model=model, log_dir=str(tmp_path / "logs"), display="none")
    assert log.status == "success"
    return Path(log.location)


def test_list_samples(eval_path: Path) -> None:
    result = CliRunner().invoke(main, ["plaintext", str(eval_path), "--list-samples"])
    assert result.exit_code == 0, result.output
    assert result.output == "s1\n"


def test_extracts_transcript(eval_path: Path, tmp_path: Path) -> None:
    out = tmp_path / "out"
    result = CliRunner().invoke(main, ["plaintext", str(eval_path), "-o", str(out)])
    assert result.exit_code == 0, result.output

    assert json.loads((out / "scores.json").read_text())[0]["name"] == "includes"
    sample_dir = out / "s1"
    assert sorted(p.name for p in sample_dir.iterdir()) == [
        "compactions.txt",
        "info.json",
        "messages.txt",
        "scores.json",
        "scores.txt",
    ]
    messages = (sample_dir / "messages.txt").read_text()
    assert "=== SYSTEM ===" in messages
    assert "--- USER ---\nDo the thing." in messages
    assert ">>> bash\n```\necho hi\n```" in messages
    assert "--- TOOL (bash) ---\nhi" in messages
    assert '>>> submit({"answer": "done"})' in messages
    assert json.loads((sample_dir / "info.json").read_text())["id"] == "s1"
    assert (
        (sample_dir / "scores.txt")
        .read_text()
        .startswith("=== SCORES ===\nincludes: C")
    )


def test_directory_input_uses_subdirectories(eval_path: Path, tmp_path: Path) -> None:
    out = tmp_path / "out"
    result = CliRunner().invoke(
        main, ["plaintext", str(eval_path.parent), "-o", str(out), "--messages-only"]
    )
    assert result.exit_code == 0, result.output
    sample_dir = out / eval_path.stem / "s1"
    assert (sample_dir / "messages.txt").exists()
    assert not (sample_dir / "compactions.txt").exists()


def test_mode_flags_are_mutually_exclusive(eval_path: Path) -> None:
    result = CliRunner().invoke(
        main,
        ["plaintext", str(eval_path), "--messages-only", "--compaction-summaries"],
    )
    assert result.exit_code == 2
    assert "mutually exclusive" in result.output
