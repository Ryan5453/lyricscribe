import json
import logging
from pathlib import Path

import jiwer
import typer

from lyricscribe import demucs, jobs
from lyricscribe.dataset import download_jam_alt, download_musdb_alt
from lyricscribe.transcribe import job as transcribe_job

logger = logging.getLogger(__name__)

cli = typer.Typer(help="LyricScribe")
separate_app = typer.Typer(help="Audio source separation commands")
cli.add_typer(separate_app, name="separate")
dataset_app = typer.Typer(help="Dataset download commands")
cli.add_typer(dataset_app, name="dataset")
transcribe_app = typer.Typer(help="ASR transcription commands")
cli.add_typer(transcribe_app, name="transcribe")
evaluate_app = typer.Typer(help="ASR evaluation commands")
cli.add_typer(evaluate_app, name="evaluate")


@cli.callback()
def _setup_logging():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


@separate_app.command("setup")
def separate_setup(
    directories: list[Path] = typer.Argument(
        ..., help="One or more directories containing subdirectories to process"
    ),
    job_dir: Path = typer.Option(
        ..., "--job-dir", help="Directory to create for job files"
    ),
    filename: str = typer.Option(
        ...,
        "--filename",
        help="Audio filename to process within each subdirectory (e.g. mix.wav)",
    ),
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
    """
    demucs.setup_job(
        directories=directories,
        job_dir=job_dir,
        filename=filename,
        model=model,
        num_chunks=chunks,
        stem=stem,
    )


@separate_app.command("run")
def separate_run(
    job_dir: Path = typer.Option(..., "--job-dir", help="Path to job directory"),
    chunk_id: int = typer.Option(
        ..., "--chunk-id", help="Chunk to process (1-based, for SLURM)"
    ),
):
    """
    Process one chunk of the separation job.
    """
    demucs.process_chunk(job_dir=job_dir, chunk_id=chunk_id)


@separate_app.command("inspect")
def separate_inspect(
    job_dir: Path = typer.Option(..., "--job-dir", help="Path to job directory"),
):
    """
    Inspect separation job details and processing statistics.
    """
    with open(job_dir / "config.json") as f:
        config = json.load(f)

    logger.info(f"Stem: {config.get('stem') or 'all'}")
    jobs.show_stats(job_dir)


@transcribe_app.command("setup")
def transcribe_setup(
    directories: list[Path] = typer.Argument(
        ..., help="One or more directories containing subdirectories to process"
    ),
    job_dir: Path = typer.Option(
        ..., "--job-dir", help="Directory to create for job files"
    ),
    filename: str = typer.Option(
        ...,
        "--filename",
        help="Audio filename to transcribe within each subdirectory (e.g. vocals.wav)",
    ),
    model: str = typer.Option(
        ..., "--model", help="HuggingFace model ID (e.g. openai/whisper-large-v3)"
    ),
    chunks: int = typer.Option(1, "--chunks", help="Number of chunks to split into"),
    batch_size: int = typer.Option(
        0,
        "--batch-size",
        help="Batch size for inference. 0 = auto-calibrate from GPU memory.",
    ),
    vad: bool = typer.Option(
        False, "--vad", help="Enable VAD-based segmentation with Silero"
    ),
    lyrics_file: str | None = typer.Option(
        None,
        "--lyrics-file",
        help="JSON file in each subdirectory to read detected_language from (e.g. lyrics.json)",
    ),
):
    """
    Initialize a transcription job by registering files into chunks.
    """
    transcribe_job.setup_job(
        directories=directories,
        job_dir=job_dir,
        filename=filename,
        model=model,
        num_chunks=chunks,
        batch_size=batch_size,
        vad=vad,
        lyrics_filename=lyrics_file,
    )


@transcribe_app.command("run")
def transcribe_run(
    job_dir: Path = typer.Option(..., "--job-dir", help="Path to job directory"),
    chunk_id: int = typer.Option(
        ..., "--chunk-id", help="Chunk to process (1-based, for SLURM)"
    ),
):
    """
    Process one chunk of a transcription job.
    """
    transcribe_job.process_chunk(job_dir=job_dir, chunk_id=chunk_id)


@transcribe_app.command("inspect")
def transcribe_inspect(
    job_dir: Path = typer.Option(..., "--job-dir", help="Path to job directory"),
):
    """
    Inspect transcription job details and processing statistics.
    """
    with open(job_dir / "config.json") as f:
        config = json.load(f)
    logger.info(f"VAD: {'enabled' if config.get('vad') else 'disabled'}")
    logger.info(f"Batch size: {config.get('batch_size', 1)}")
    jobs.show_stats(job_dir)


@dataset_app.command("jam-alt")
def dataset_jam_alt(
    output_dir: Path = typer.Option(
        ..., "--output-dir", help="Directory to write the Jam-ALT dataset into"
    ),
):
    """
    Download the Jam-ALT dataset (79 songs, 4 languages).
    """
    download_jam_alt(output_dir)


@dataset_app.command("musdb-alt")
def dataset_musdb_alt(
    output_dir: Path = typer.Option(
        ..., "--output-dir", help="Directory to write the MUSDB-ALT dataset into"
    ),
):
    """
    Download the MUSDB-ALT dataset (39 English songs).
    """
    download_musdb_alt(output_dir)


@evaluate_app.command("run")
def evaluate(
    job_dir: Path = typer.Option(
        ..., "--job-dir", help="Path to job directory"
    ),
):
    """Evaluate transcription quality against ground-truth lyrics."""

    with open(job_dir / "config.json") as f:
        config = json.load(f)

    directories = [Path(d) for d in config["directories"]]

    references: dict[str, str] = {}
    for directory in directories:
        for song_dir in directory.iterdir():
            if not song_dir.is_dir():
                continue
            lyrics_path = song_dir / "lyrics.json"
            if lyrics_path.exists():
                with open(lyrics_path) as f:
                    lyrics_data = json.load(f)
                references[song_dir.name] = lyrics_data["unsynced"]["data"]

    if not references:
        logger.warning("No ground-truth lyrics found in dataset directories")
        return

    results = []
    for path in sorted(job_dir.glob("results_*.jsonl")):
        with open(path) as f:
            for line in f:
                results.append(json.loads(line))

    if not results:
        logger.warning("No results found in job directory")
        return

    totals = {"wer": [], "insertions": [], "deletions": [], "substitutions": []}

    for r in results:
        song_id = r["song_id"]
        hypothesis = r["transcription"]

        if hypothesis is None:
            logger.warning(f"{song_id}: skipped (no transcription)")
            continue

        if song_id not in references:
            logger.warning(f"{song_id}: skipped (no ground truth)")
            continue

        measures = jiwer.compute_measures(references[song_id], hypothesis)

        logger.debug(
            f"{song_id}: WER={measures['wer']:.2%}  "
            f"I={measures['insertions']}  "
            f"D={measures['deletions']}  "
            f"S={measures['substitutions']}"
        )

        totals["wer"].append(measures["wer"])
        totals["insertions"].append(measures["insertions"])
        totals["deletions"].append(measures["deletions"])
        totals["substitutions"].append(measures["substitutions"])

    n = len(totals["wer"])
    if n == 0:
        logger.warning("No songs could be evaluated")
        return

    logger.info(f"--- Summary ({n} songs) ---")
    logger.info(f"Mean WER: {sum(totals['wer']) / n:.2%}")
    logger.info(f"Total insertions: {sum(totals['insertions'])}")
    logger.info(f"Total deletions: {sum(totals['deletions'])}")
    logger.info(f"Total substitutions: {sum(totals['substitutions'])}")

if __name__ == "__main__":
    cli()
