import os
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path

import click

from inspect_boltons.hawk import download_eval_set
from inspect_boltons.plaintext import (
    default_output_dir,
    extract_eval_file,
    list_samples,
)

HALF_CPUS = max(1, (os.cpu_count() or 8) // 2)


@click.group()
def main() -> None:
    pass


@main.command()
@click.argument(
    "eval_files",
    nargs=-1,
    required=True,
    type=click.Path(exists=True, path_type=Path),
)
@click.option(
    "-o",
    "--output-dir",
    type=click.Path(path_type=Path),
    help=(
        "Output directory (default: <eval_stem>_plaintext next to the .eval file; "
        "with a directory or multiple eval files, each log gets its own "
        "<eval_stem>/ subdirectory here)."
    ),
)
@click.option(
    "-s",
    "--sample",
    "samples",
    multiple=True,
    help="Extract only these sample IDs (repeatable, e.g. -s foo -s bar).",
)
@click.option(
    "--list-samples", "list_only", is_flag=True, help="List sample IDs and exit."
)
@click.option(
    "--parallel-evals",
    type=click.IntRange(min=1),
    is_flag=False,
    flag_value=HALF_CPUS,
    default=1,
    metavar="N",
    help=(
        "Process eval files concurrently across N worker processes (default 1 = "
        "sequential; bare flag uses half the CPUs)."
    ),
)
@click.option(
    "--parallel-samples",
    type=click.IntRange(min=1),
    is_flag=False,
    flag_value=HALF_CPUS,
    default=1,
    metavar="N",
    help=(
        "Within each eval file, extract samples concurrently across N worker "
        "processes (default 1 = sequential; bare flag uses half the CPUs). "
        "Multiplies with --parallel-evals. Each worker holds one full sample in "
        "memory, so lower N if large runs exhaust RAM."
    ),
)
@click.option(
    "--compaction-summaries",
    is_flag=True,
    help="Extract only compaction summaries (condensed progress view).",
)
@click.option(
    "--messages-only",
    is_flag=True,
    help="Extract only full message transcripts (skip compaction summaries).",
)
def plaintext(
    eval_files: tuple[Path, ...],
    output_dir: Path | None,
    samples: tuple[str, ...],
    list_only: bool,
    parallel_evals: int,
    parallel_samples: int,
    compaction_summaries: bool,
    messages_only: bool,
) -> None:
    """Extract agent transcripts from Inspect .eval logs into plain text.

    EVAL_FILES are .eval files or directories of .eval files.
    """
    if compaction_summaries and messages_only:
        raise click.UsageError(
            "--compaction-summaries and --messages-only are mutually exclusive."
        )

    collection_mode = len(eval_files) > 1 or any(p.is_dir() for p in eval_files)

    eval_paths: list[Path] = []
    for input_path in eval_files:
        if input_path.is_dir():
            paths = sorted(p for p in input_path.glob("*.eval") if p.is_file())
            if not paths:
                raise click.UsageError(f"No .eval files found in {input_path}")
            eval_paths.extend(paths)
        else:
            eval_paths.append(input_path)

    if list_only:
        for eval_path in eval_paths:
            for sid in list_samples(eval_path):
                click.echo(f"{eval_path.name}: {sid}" if collection_mode else sid)
        return

    def out_dir_for(eval_path: Path) -> Path:
        if output_dir is None:
            return default_output_dir(eval_path)
        return output_dir / eval_path.stem if collection_mode else output_dir

    tasks = [(eval_path, out_dir_for(eval_path)) for eval_path in eval_paths]
    extract = partial(
        extract_eval_file,
        sample_ids=set(samples) if samples else None,
        compactions=not messages_only,
        messages=not compaction_summaries,
        sample_workers=parallel_samples,
    )
    eval_workers = min(parallel_evals, len(tasks))

    if eval_workers > 1:
        click.echo(
            f"Extracting {len(tasks)} eval file(s) across {eval_workers} workers",
            err=True,
        )
        with ProcessPoolExecutor(max_workers=eval_workers) as pool:
            futures = [
                pool.submit(extract, eval_path, out_dir) for eval_path, out_dir in tasks
            ]
            for future in futures:
                future.result()
        return

    for eval_path, out_dir in tasks:
        if collection_mode:
            click.echo(f"Extracting {eval_path} -> {out_dir}", err=True)
        extract(eval_path, out_dir)


@main.command()
@click.argument("eval_set_id")
@click.option(
    "--dry-run",
    is_flag=True,
    help="Show what would be downloaded without downloading.",
)
@click.option(
    "--output-root",
    type=click.Path(file_okay=False, path_type=Path),
    default=Path("logs"),
    show_default=True,
    help="Directory under which <eval-set-id>/ will be created.",
)
@click.option(
    "--plaintext",
    "extract_plaintext",
    is_flag=True,
    help="After downloading, run `ibolt plaintext` on the destination directory.",
)
@click.option(
    "--all",
    "all_files",
    is_flag=True,
    help="Download all .eval files, including superseded retries for the same task.",
)
@click.option(
    "--force-most-tokens-on-all-error",
    is_flag=True,
    help=(
        "If every retry for a task is status=error/unreadable, keep the .eval that "
        "used the most tokens instead of failing. Use only for intentional recovery "
        "of broken eval-sets."
    ),
)
@click.pass_context
def dl(
    ctx: click.Context,
    eval_set_id: str,
    dry_run: bool,
    output_root: Path,
    extract_plaintext: bool,
    all_files: bool,
    force_most_tokens_on_all_error: bool,
) -> None:
    """Download the logs of a Hawk eval-set.

    Superseded retries are dropped unless --all is given: per task, the log with
    status started/success is kept, as in the log viewer.
    """
    dest_dir = output_root / eval_set_id

    dest_dir.mkdir(parents=True, exist_ok=True)
    download_eval_set(
        eval_set_id,
        dest_dir,
        dry_run=dry_run,
        all_files=all_files,
        force_most_tokens_on_all_error=force_most_tokens_on_all_error,
    )

    if dry_run:
        if extract_plaintext:
            click.echo("Dry run: skipping plaintext extraction.")
        return

    click.echo(f"Done. Files saved to {dest_dir}")
    if extract_plaintext:
        click.echo(f"\nExtracting plaintext from .eval files in {dest_dir} ...")
        ctx.invoke(plaintext, eval_files=(dest_dir,), parallel_samples=HALF_CPUS)
