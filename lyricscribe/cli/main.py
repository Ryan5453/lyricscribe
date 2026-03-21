import csv
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


@separate_app.command("reset")
def separate_reset(
    job_dir: Path = typer.Option(..., "--job-dir", help="Path to job directory"),
):
    """
    Reset a separation job so it can be re-run from scratch.

    Deletes separated output files and resets all chunk statuses to pending.
    """
    demucs.reset_job(job_dir)


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
        1,
        "--batch-size",
        help="Batch size for inference.",
    ),
    vad: bool = typer.Option(
        False, "--vad", help="Enable VAD-based segmentation with Silero"
    ),
    chunked: bool = typer.Option(
        False, "--chunked", help="Use fixed-length chunked inference instead of full-context"
    ),
    lyrics_file: str | None = typer.Option(
        None,
        "--lyrics-file",
        help="JSON file in each subdirectory to read detected_language from (e.g. lyrics.json)",
    ),
    vad_source_file: str | None = typer.Option(
        None,
        "--vad-source",
        help="Audio filename to use as VAD source (e.g. htdemucs_ft_vocals.wav). "
             "VAD runs on this file, transcription runs on --filename.",
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
        chunked=chunked,
        lyrics_filename=lyrics_file,
        vad_filename=vad_source_file,
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


@transcribe_app.command("reset")
def transcribe_reset(
    job_dir: Path = typer.Option(..., "--job-dir", help="Path to job directory"),
):
    """
    Reset a transcription job so it can be re-run from scratch.

    Deletes results files and resets all chunk statuses to pending.
    """
    # Delete results files
    deleted = 0
    for p in job_dir.glob("results*.jsonl"):
        p.unlink(missing_ok=True)
        deleted += 1
    logger.info(f"Deleted {deleted} results file(s)")

    # Reset chunk statuses
    reset_count = 0
    for chunk_path in sorted(job_dir.glob("chunk_*.json")):
        with open(chunk_path) as f:
            chunk_data = json.load(f)
        for entry in chunk_data["files"]:
            if entry["status"] != "pending":
                entry["status"] = "pending"
                entry["duration_seconds"] = None
                entry["error_message"] = None
                entry["processed_at"] = None
                reset_count += 1
        with open(chunk_path, "w") as f:
            json.dump(chunk_data, f, indent=2)

    logger.info(f"Reset {reset_count} entries to pending")


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


def _evaluate_job(job_dir: Path, verbose: bool = False) -> dict | None:
    config_path = job_dir / "config.json"
    if not config_path.exists():
        return None

    with open(config_path) as f:
        config = json.load(f)

    directories = [Path(d) for d in config.get("directories", [])]
    if not directories:
        if verbose:
            logger.warning("Job config has no directories")
        return None

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
        if verbose:
            logger.warning("No ground-truth lyrics found in dataset directories")
        return None

    results_map = {}
    # Read from either results.jsonl or results_*.jsonl
    for results_path in job_dir.glob("results*.jsonl"):
        with open(results_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                    results_map[r["song_id"]] = r
                except json.JSONDecodeError:
                    if verbose:
                        logger.warning(f"Skipping invalid JSON line in {results_path}")
                    continue

    results = list(results_map.values())

    if not results:
        if verbose:
            logger.warning("No results found in job directory")
        return None

    totals = {"wer": [], "insertions": [], "deletions": [], "substitutions": []}
    for r in results:
        song_id = r["song_id"]
        hypothesis = r["transcription"]

        if hypothesis is None:
            if verbose:
                logger.warning(f"{song_id}: skipped (no transcription)")
            continue

        if song_id not in references:
            if verbose:
                logger.warning(f"{song_id}: skipped (no ground truth)")
            continue

        measures = jiwer.compute_measures(references[song_id], hypothesis)

        if verbose:
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
        if verbose:
            logger.warning("No songs could be evaluated")
        return None

    return {
        "n_songs": n,
        "mean_wer": sum(totals["wer"]) / n,
        "insertions": sum(totals["insertions"]),
        "deletions": sum(totals["deletions"]),
        "substitutions": sum(totals["substitutions"]),
        "config": config,
    }


@evaluate_app.command("run")
def evaluate(
    job_dir: Path = typer.Option(
        ..., "--job-dir", help="Path to job directory"
    ),
):
    """Evaluate transcription quality against ground-truth lyrics."""
    stats = _evaluate_job(job_dir, verbose=True)
    if not stats:
        return

    logger.info(f"--- Summary ({stats['n_songs']} songs) ---")
    logger.info(f"Mean WER: {stats['mean_wer']:.2%}")
    logger.info(f"Total insertions: {stats['insertions']}")
    logger.info(f"Total deletions: {stats['deletions']}")
    logger.info(f"Total substitutions: {stats['substitutions']}")


@evaluate_app.command("summarize")
def evaluate_summarize(
    jobs_dir: Path = typer.Option(
        ..., "--jobs-dir", help="Path to base jobs directory containing model subdirectories"
    ),
    output: Path = typer.Option(
        "evaluation_summary.csv", "--output", help="Output CSV file path"
    ),
):
    """Aggregate evaluation results across all jobs into a CSV file."""
    all_stats = []
    
    for config_path in jobs_dir.glob("**/config.json"):
        job_dir = config_path.parent
        stats = _evaluate_job(job_dir, verbose=False)
        if stats:
            config = stats["config"]
            model = config.get("model", "unknown")
            dataset = ", ".join([Path(d).name for d in config.get("directories", [])])
            vad = config.get("vad", False)
            chunked = config.get("chunked", False)
            filename = config.get("filename", "")
            
            all_stats.append({
                "job_dir": str(job_dir.relative_to(jobs_dir)),
                "model": model,
                "dataset": dataset,
                "filename": filename,
                "vad": vad,
                "chunked": chunked,
                "mean_wer": stats["mean_wer"],
                "n_songs": stats["n_songs"],
                "insertions": stats["insertions"],
                "deletions": stats["deletions"],
                "substitutions": stats["substitutions"],
            })

    if not all_stats:
        logger.error("No successful evaluations found to summarize.")
        return

    all_stats.sort(key=lambda x: x["mean_wer"])

    with open(output, "w", newline="") as f:
        writer = csv.DictWriter(
            f, 
            fieldnames=[
                "job_dir", "model", "dataset", "filename", "vad", "chunked", 
                "mean_wer", "n_songs", "insertions", "deletions", "substitutions"
            ]
        )
        writer.writeheader()
        writer.writerows(all_stats)

    logger.info(f"Successfully summarized {len(all_stats)} jobs -> {output}")

if __name__ == "__main__":
    cli()
