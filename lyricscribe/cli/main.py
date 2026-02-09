"""
LyricScribe CLI for audio processing.
"""

from pathlib import Path
from typing import Optional

import typer

from lyricscribe.cli.separators.demucs import DemucsSeparator

cli = typer.Typer(help="LyricScribe - Audio processing toolkit")
separate_app = typer.Typer(help="Audio source separation commands")
cli.add_typer(separate_app, name="separate")

separator = DemucsSeparator()


@separate_app.command()
def setup(
    directory: Path = typer.Argument(..., help="Directory containing ISRC folders"),
    db_path: Path = typer.Option(..., "--db", help="Path to create SQLite database"),
    model: str = typer.Option("htdemucs", "--model", help="Demucs model to use"),
    device: str = typer.Option("cuda", "--device", help="Device to use (cuda or cpu)"),
    chunks: int = typer.Option(5, "--chunks", help="Number of chunks to split into"),
    stem: Optional[str] = typer.Option(None, "--stem", help="Which stem to isolate (vocals, drums, bass, other). If not specified, all stems are saved."),
):
    """
    Initialize separation job by registering files into chunks.

    Stores all configuration (model, device, chunks, stem) in the database.
    Run this once before submitting SLURM jobs.

    Example:
        lyricscribe separate setup /path/to/dataset --db job.db --model htdemucs_ft
        lyricscribe separate setup /path/to/dataset --db job.db --model htdemucs_ft --stem vocals
    """
    separator.setup_job(directory=directory, db_path=db_path, model=model, device=device, num_chunks=chunks, stem=stem)


@separate_app.command()
def run(
    db_path: Path = typer.Option(..., "--db", help="Path to job database"),
    chunk_id: int = typer.Option(..., "--chunk-id", help="Chunk to process (1-based, for SLURM)"),
):
    """
    Process one chunk (for SLURM workers).

    Reads configuration from database and processes assigned chunk.

    Example:
        lyricscribe separate run --db job.db --chunk-id 1
    """
    separator.process_chunk(db_path=db_path, chunk_id=chunk_id)


@separate_app.command()
def inspect(
    db_path: Path = typer.Option(..., "--db", help="Path to job database"),
):
    """
    Inspect job details and processing statistics.

    Example:
        lyricscribe separate inspect --db job.db
    """
    separator.show_stats(db_path)


@separate_app.command()
def retry(
    db_path: Path = typer.Option(..., "--db", help="Path to job database"),
):
    """
    Retry failed separations.

    Example:
        lyricscribe separate retry --db job.db
    """
    separator.retry_failed(db_path)


if __name__ == "__main__":
    cli()
