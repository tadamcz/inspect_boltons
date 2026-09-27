from __future__ import annotations

import itertools
from pathlib import Path

import pytest
from inspect_ai import Task, eval
from inspect_ai.agent import react
from inspect_ai.dataset import Sample
from inspect_ai.model import ModelOutput, get_model
from inspect_ai.scorer import includes
from inspect_ai.tool import bash


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
