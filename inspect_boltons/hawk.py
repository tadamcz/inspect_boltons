"""Download the logs of a Hawk eval-set with `hawk download`.

Unless all files are requested, superseded retries are dropped, mirroring the log
viewer's "show retried logs" toggle: logs are grouped by task id and, per task, the
log whose status is started/success is kept, breaking ties by descending filename.
"""

from __future__ import annotations

import re
import shutil
import struct
import subprocess
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import click
import zstandard
from inspect_ai.log import read_eval_log

_VALID_STATUSES = ("started", "success")

_UUID_PATTERN = re.compile(r"_([^_]+?)(?:\.fast)?\.eval$")


@dataclass
class EvalFile:
    name: str
    task_id: str
    status: str | None
    tokens: int = 0


def hawk_command() -> str:
    found = shutil.which("hawk")
    if found is not None:
        return found
    sibling = Path(sys.executable).with_name("hawk")
    if sibling.exists():
        return str(sibling)
    return "hawk"


def _task_id_from_filename(name: str) -> str:
    m = _UUID_PATTERN.search(name)
    return m.group(1) if m else name


def read_eval_file(path: Path) -> EvalFile:
    name = path.name
    try:
        log = read_eval_log(path, header_only=True)
    except (ValueError, KeyError, struct.error, zstandard.ZstdError) as exc:
        click.echo(f"  Warning: could not read header from {name}: {exc}", err=True)
        return EvalFile(name=name, task_id=_task_id_from_filename(name), status=None)
    return EvalFile(
        name=name,
        task_id=log.eval.task_id,
        status=log.status,
        tokens=sum(u.total_tokens for u in log.stats.model_usage.values()),
    )


def select_eval_excludes(
    files: list[EvalFile], *, force_most_tokens_on_all_error: bool = False
) -> list[str]:
    """Return names of .eval files to exclude, keeping the viewer's winner per task.

    Raises if a task has no started/success log, since there is then no correct file
    to keep, unless `force_most_tokens_on_all_error` is set, in which case the log
    that used the most tokens is kept.
    """
    by_task: dict[str, list[EvalFile]] = defaultdict(list)
    for f in files:
        by_task[f.task_id].append(f)

    excludes: list[str] = []
    for task_id, group in by_task.items():
        ranked = sorted(group, key=lambda f: (f.status in _VALID_STATUSES, f.name))
        winner = ranked[-1]
        how = f"status={winner.status}"
        if winner.status not in _VALID_STATUSES:
            if not force_most_tokens_on_all_error:
                statuses = ", ".join(f"{f.name} (status={f.status})" for f in group)
                raise click.ClickException(
                    f"task {task_id} has no started/success .eval file; cannot "
                    f"determine which to keep. Candidates: {statuses}"
                )
            ranked = sorted(group, key=lambda f: (f.tokens, f.name))
            winner = ranked[-1]
            how = f"status={winner.status}, force-most-tokens"
            statuses = ", ".join(
                f"{f.name} (status={f.status}, tokens={f.tokens})" for f in group
            )
            click.echo(
                f"  Warning: task {task_id} has no started/success .eval file; "
                f"force-keeping candidate with most tokens {winner.name}. "
                f"Candidates: {statuses}",
                err=True,
            )
        elif winner.status == "started":
            click.echo(
                f"  Warning: keeping {winner.name} but its status is 'started' "
                f"(the run is incomplete); no success log exists for task {task_id}.",
                err=True,
            )
        excludes.extend(f.name for f in ranked[:-1])
        click.echo(
            f"Task {task_id}: keeping {winner.name} ({how}), "
            f"excluding {len(group) - 1} other file(s)"
        )

    return excludes


def delete_local_excludes(dest_dir: Path, excludes: list[str]) -> None:
    if not excludes:
        return
    click.echo(f"\nDeleting {len(excludes)} outdated local .eval file(s):")
    for name in excludes:
        path = dest_dir / name
        click.echo(f"  {path.name}")
        path.unlink(missing_ok=True)


def download_with_hawk(eval_set_id: str, dest_dir: Path, *, dry_run: bool) -> None:
    cmd = [hawk_command(), "download", eval_set_id]
    cmd += ["--list"] if dry_run else ["--output-dir", str(dest_dir)]
    click.echo(
        f"\n{'Dry run: listing' if dry_run else 'Downloading'} via hawk download ..."
    )
    result = subprocess.run(cmd)
    if result.returncode != 0:
        raise click.ClickException("hawk download failed.")


def download_eval_set(
    eval_set_id: str,
    dest_dir: Path,
    *,
    dry_run: bool,
    all_files: bool,
    force_most_tokens_on_all_error: bool,
) -> None:
    download_with_hawk(eval_set_id, dest_dir, dry_run=dry_run)
    if dry_run or all_files:
        return
    files = [read_eval_file(p) for p in sorted(dest_dir.glob("*.eval"))]
    excludes = select_eval_excludes(
        files, force_most_tokens_on_all_error=force_most_tokens_on_all_error
    )
    delete_local_excludes(dest_dir, excludes)
