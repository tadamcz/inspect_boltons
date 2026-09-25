from __future__ import annotations

import shutil
from pathlib import Path

import click
import pytest
from click.testing import CliRunner

from inspect_boltons import hawk
from inspect_boltons.cli import main
from inspect_boltons.hawk import EvalFile, read_eval_file, select_eval_excludes


def test_reads_header_from_eval_file(eval_path: Path) -> None:
    f = read_eval_file(eval_path)
    assert f.status == "success"
    assert f.task_id != ""
    assert f.tokens > 0


def test_unreadable_header_has_no_status(tmp_path: Path) -> None:
    path = tmp_path / "2026-01-01_task_abc123.eval"
    path.write_bytes(b"not a zip")
    f = read_eval_file(path)
    assert f == EvalFile(name=path.name, task_id="abc123", status=None)


def test_keeps_newest_valid_log_per_task() -> None:
    files = [
        EvalFile(name="a1.eval", task_id="a", status="error"),
        EvalFile(name="a2.eval", task_id="a", status="success"),
        EvalFile(name="a3.eval", task_id="a", status="error"),
        EvalFile(name="b1.eval", task_id="b", status="success"),
        EvalFile(name="b2.eval", task_id="b", status="started"),
    ]
    assert sorted(select_eval_excludes(files)) == ["a1.eval", "a3.eval", "b1.eval"]


def test_task_without_valid_log_raises_unless_forced() -> None:
    files = [
        EvalFile(name="a1.eval", task_id="a", status="error", tokens=10),
        EvalFile(name="a2.eval", task_id="a", status=None),
    ]
    with pytest.raises(click.ClickException, match="no started/success"):
        select_eval_excludes(files)
    assert select_eval_excludes(files, force_most_tokens_on_all_error=True) == [
        "a2.eval"
    ]


def test_dl_deletes_superseded_retries_and_extracts_plaintext(
    eval_path: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    task_id = read_eval_file(eval_path).task_id

    def fake_download(eval_set_id: str, dest_dir: Path, *, dry_run: bool) -> None:
        shutil.copy(eval_path, dest_dir / "2026-01-01_task_retry1.eval")
        (dest_dir / f"2027-01-01_task_{task_id}.eval").write_bytes(b"broken")
        shutil.copy(eval_path, dest_dir / eval_path.name)

    monkeypatch.setattr(hawk, "download_with_hawk", fake_download)
    root = tmp_path / "out"
    result = CliRunner().invoke(
        main, ["dl", "es", "--output-root", str(root), "--plain"]
    )
    assert result.exit_code == 0, result.output

    dest = root / "es"
    evals = sorted(p.name for p in dest.glob("*.eval"))
    assert evals == [max(eval_path.name, "2026-01-01_task_retry1.eval")]
    assert (
        dest / (Path(evals[0]).stem + "_plaintext") / "s1" / "messages.txt"
    ).exists()
