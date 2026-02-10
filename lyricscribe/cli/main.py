import logging
from pathlib import Path
from typing import List

import typer

from lyricscribe.cli.demucs import process_chunk, retry_failed, setup_job, show_stats

cli = typer.Typer(help="LyricScribe")
separate_app = typer.Typer(help="Audio source separation commands")
cli.add_typer(separate_app, name="separate")

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


@separate_app.command()
def setup(
    directories: List[Path] = typer.Argument(..., help="One or more directories containing subdirectories to process"),
    db_path: Path = typer.Option(..., "--db", help="Path to create SQLite database"),
    filename: str = typer.Option(..., "--filename", help="Audio filename to process within each subdirectory (e.g. mix.wav)"),
    model: str = typer.Option("htdemucs", "--model", help="Demucs model to use"),
    chunks: int = typer.Option(5, "--chunks", help="Number of chunks to split into"),
    stem: str | None = typer.Option(
        None,
        "--stem",
        help="Which stem to isolate (vocals, drums, bass, other). If not specified, all stems are saved.",
    ),
):
    """
    Initialize separation job by registering files into chunks.

    Stores all configuration (model, chunks, stem) in the database.
    Run this once before submitting SLURM jobs.
    """
    setup_job(
        directories=directories,
        db_path=db_path,
        filename=filename,
        model=model,
        num_chunks=chunks,
        stem=stem,
    )


@separate_app.command()
def run(
    db_path: Path = typer.Option(..., "--db", help="Path to job database"),
    chunk_id: int = typer.Option(
        ..., "--chunk-id", help="Chunk to process (1-based, for SLURM)"
    ),
):
    """
    Process one chunk (for SLURM workers).

    Reads configuration from database and processes assigned chunk.
    """
    process_chunk(db_path=db_path, chunk_id=chunk_id)


@separate_app.command()
def inspect(
    db_path: Path = typer.Option(..., "--db", help="Path to job database"),
):
    """
    Inspect job details and processing statistics.
    """
    show_stats(db_path)


@separate_app.command()
def retry(
    db_path: Path = typer.Option(..., "--db", help="Path to job database"),
):
    """
    Retry failed separations.
    """
    retry_failed(db_path)


if __name__ == "__main__":
    cli()
