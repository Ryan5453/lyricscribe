import csv
import json
import logging
import shutil
from pathlib import Path

import typer

from lyricscribe import demucs, jobs, plots
from lyricscribe.dataset import download_jam_alt, download_musdb_alt
from lyricscribe.evaluate import collect_evaluation_data, evaluate_job
from lyricscribe.finetune import data as finetune_data
from lyricscribe.finetune import jobs as finetune_jobs
from lyricscribe.finetune import trainer as finetune_trainer
from lyricscribe.finetune.config import (
    create_finetune_config,
    get_checkpoint_for_epoch,
    get_latest_checkpoint,
    list_checkpoints,
    load_finetune_config,
)
from lyricscribe.transcribe import job as transcribe_job
from lyricscribe.transcribe.artifacts import correlation, extractor, processor

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
artifacts_app = typer.Typer(help = "Artifacts feature extraction commands")
cli.add_typer(artifacts_app, name="artifacts")
finetune_app = typer.Typer(help="Model finetuning commands")
cli.add_typer(finetune_app, name="finetune")


def _collect_result_files(
    explicit_results: list[Path] | None,
    *,
    jobs_dir: Path | None = None,
    job_name: str | None = None,
) -> list[Path]:
    """
    Resolve one or more result files from explicit paths and/or a shared job name.

    When ``job_name`` is provided, discovers ``results*.jsonl`` files from
    ``jobs_dir/*/<job_name>/`` and returns them in sorted order.
    """
    result_files: list[Path] = list(explicit_results or [])

    if job_name:
        if jobs_dir is None:
            raise ValueError("--results-job-name requires --results-jobs-dir or --jobs-dir")
        discovered = sorted(jobs_dir.glob(f"*/{job_name}/results*.jsonl"))
        if not discovered:
            raise ValueError(
                f"No result files found for job '{job_name}' under {jobs_dir}"
            )
        result_files.extend(discovered)

    # Preserve order while deduplicating.
    unique: list[Path] = []
    seen: set[Path] = set()
    for path in result_files:
        resolved = path.expanduser().resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        unique.append(resolved)

    return unique


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


@evaluate_app.command("run")
def evaluate(
    job_dir: Path = typer.Option(
        ..., "--job-dir", help="Path to job directory"
    ),
):
    """Evaluate transcription quality against ground-truth lyrics."""
    stats = evaluate_job(job_dir, verbose=True)
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
    all_stats = collect_evaluation_data(jobs_dir)

    if not all_stats:
        logger.error("No successful evaluations found to summarize.")
        return

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


@evaluate_app.command("plot")
def evaluate_plot(
    jobs_dir: Path = typer.Option(
        ..., "--jobs-dir", help="Path to base jobs directory containing model subdirectories"
    ),
    output_dir: Path = typer.Option(
        ..., "--output-dir", help="Directory to save the generated PDF plots"
    ),
    alignments_dir: Path | None = typer.Option(
        None, "--alignments-dir", help="Directory of MFA alignment JSON files (enables artifact chart)"
    ),
    features_dir: Path | None = typer.Option(
        None, "--features-dir", help="Directory of artifact feature JSON files (enables artifact chart)"
    ),
    results_file: list[Path] = typer.Option(
        None, "--results-file", help="Path to results.jsonl with model transcriptions; repeat to include multiple models (enables artifact chart)"
    ),
    results_job_name: str | None = typer.Option(
        None,
        "--results-job-name",
        help="Job subdirectory name to auto-discover results*.jsonl across all model directories under --jobs-dir",
    ),
    musdb_dir: Path | None = typer.Option(
        None, "--musdb-dir", help="Root MUSDB directory for ground truth lyrics (enables artifact chart)"
    ),
):
    """Generate evaluation plots from job directories.

    To include the artifact quartile chart, pass --alignments-dir, --features-dir,
    --results-file, and --musdb-dir. Repeat --results-file to include multiple
    model result files. The word-level dataset is built in memory.
    """
    word_dataset = None
    result_files = _collect_result_files(
        results_file,
        jobs_dir=jobs_dir,
        job_name=results_job_name,
    )
    if alignments_dir is not None and features_dir is not None and musdb_dir is not None and result_files:
        word_dataset = correlation.build_dataset(
            alignments_dir, features_dir, result_files, musdb_dir
        )
    elif any(
        opt is not None for opt in [alignments_dir, features_dir, musdb_dir]
    ) or result_files or results_job_name:
        logger.warning(
            "Artifact chart requires --alignments-dir, --features-dir, --musdb-dir, "
            "and either one or more --results-file values or --results-job-name. "
            "Skipping artifact chart."
        )
    plots.generate_all_plots(jobs_dir, output_dir, word_dataset=word_dataset)


@artifacts_app.command("extract")
def artifact_extract(
    musdb_dir: Path = typer.Option(..., "--musdb-dir", help="Root MUSDB directory"),
    output_dir: Path = typer.Option(..., "--output-dir", help="Directory to write artifact feature JSON files"),
):
    """Extract per-frame artifact features for each song."""
    extractor.process_dataset(musdb_dir, output_dir)


@artifacts_app.command("align")
def artifact_align(
    musdb_dir: Path = typer.Option(
        ..., "--musdb-dir", help="Root MUSDB directory containing song subdirectories"
    ),
    output_dir: Path = typer.Option(
        ..., "--output-dir", help="Directory to write per-song alignment JSON files"
    ),
    container: str | None = typer.Option(
        None,
        "--container",
        "-c",
        envvar="LYRICSCRIBE_MFA_CONTAINER",
        help="Path to MFA .sif container image (or set LYRICSCRIBE_MFA_CONTAINER)",
    ),
    mfa_root: Path | None = typer.Option(
        None,
        "--mfa-root",
        help="Host directory for cached MFA pretrained models",
    ),
):
    """Run Montreal Forced Aligner (via Singularity/Apptainer) for word-level alignments."""
    processor.align(
        musdb_dir,
        output_dir,
        container=container,
        mfa_root=mfa_root,
    )


@artifacts_app.command("build")
def artifact_build(
    alignments_dir: Path = typer.Option(
        ..., "--alignments-dir", help="Directory of MFA alignment JSON files"
    ),
    features_dir: Path = typer.Option(
        ..., "--features-dir", help="Directory of artifact feature JSON files"
    ),
    results_file: list[Path] = typer.Option(
        ..., "--results-file", help="Path to results.jsonl with model transcriptions; repeat to include multiple models"
    ),
    results_jobs_dir: Path | None = typer.Option(
        None,
        "--results-jobs-dir",
        help="Jobs root used with --results-job-name to auto-discover results*.jsonl across model directories",
    ),
    results_job_name: str | None = typer.Option(
        None,
        "--results-job-name",
        help="Job subdirectory name to auto-discover results*.jsonl across model directories",
    ),
    musdb_dir: Path = typer.Option(
        ..., "--musdb-dir", help="Root MUSDB directory for ground truth lyrics"
    ),
    output: Path = typer.Option(
        ..., "--output", help="Path to write the word-level CSV dataset"
    ),
):
    """Build the word-level dataset combining alignments, artifacts, and errors."""
    result_files = _collect_result_files(
        results_file,
        jobs_dir=results_jobs_dir,
        job_name=results_job_name,
    )
    correlation.build_dataset(
        alignments_dir, features_dir, result_files, musdb_dir, csv_output=output
    )


@finetune_app.command("setup")
def finetune_setup(
    train_dir: Path = typer.Argument(
        ..., help="Directory containing training songs (subdirectories with lyrics.json and audio)"
    ),
    output_dir: Path = typer.Option(
        ..., "--output-dir", help="Directory to save experiment outputs"
    ),
    model: str = typer.Option(
        ..., "--model", help="Model identifier (e.g., nvidia/parakeet-tdt-0.6b-v3, openai/whisper-large-v3)"
    ),
    filename: list[str] = typer.Option(
        ..., "--filename", help="Audio filename(s) to train on (repeat for multi-file training)"
    ),
    val_dir: Path = typer.Option(
        None, "--val-dir", help="Directory containing validation songs (optional but recommended)"
    ),
    batch_size: int = typer.Option(
        8, "--batch-size", help="Training batch size"
    ),
    max_epochs: int = typer.Option(
        50, "--max-epochs", help="Maximum training epochs"
    ),
    epochs_per_job: int = typer.Option(
        5, "--epochs-per-job", help="Epochs to train per SLURM job (default: 5)"
    ),
    learning_rate: float = typer.Option(
        1e-5, "--learning-rate", help="Peak learning rate"
    ),
    no_augment: bool = typer.Option(
        False, "--no-augment", help="Disable SpecAugment"
    ),
):
    """
    Setup a finetuning experiment with epoch-level checkpointing.

    Creates job directory with manifests and chunk files for SLURM processing.
    Architecture is auto-detected from model name.

    Pass multiple --filename flags to train on a random mix of audio types.
    """
    try:
        config = create_finetune_config(
            base_model=model,
            train_dir=train_dir,
            output_dir=output_dir,
            filenames=filename,
            val_dir=val_dir,
            batch_size=batch_size,
            max_epochs=max_epochs,
            epochs_per_job=epochs_per_job,
            learning_rate=learning_rate,
            use_augmentation=not no_augment,
        )
    except ValueError as e:
        logger.error(f"Invalid configuration: {e}")
        return

    logger.info(f"Experiment: {config['exp_name']}")
    logger.info(f"Architecture: {config['architecture']} (auto-detected)")
    logger.info(f"Model: {config['base_model']}")
    logger.info(f"Filenames: {', '.join(config['filenames'])}")
    logger.info(f"Epochs: {config['max_epochs']} (per job: {config['epochs_per_job']})")

    train_dataset = finetune_data.LyricsDataset(
        dataset_dir=train_dir,
        filenames=config['filenames'],
    )

    if len(train_dataset) == 0:
        logger.error("No valid songs found in training directory!")
        return

    job_dir = Path(config['output_dir']) / config['exp_name']
    job_dir.mkdir(parents=True, exist_ok=True)
    train_manifest = job_dir / "train_manifest.jsonl"
    n_train = finetune_data.create_nemo_manifest(train_dataset, train_manifest)
    logger.info(f"Training: {n_train} songs")

    val_manifest = None
    if val_dir:
        val_dataset = finetune_data.LyricsDataset(
            dataset_dir=val_dir,
            filenames=config['filenames'],
        )
        val_manifest = job_dir / "val_manifest.jsonl"
        n_val = finetune_data.create_nemo_manifest(val_dataset, val_manifest)
        logger.info(f"Validation: {n_val} songs")
    
    finetune_jobs.setup_finetune_job(config, train_manifest, val_manifest)

    num_chunks = (config['max_epochs'] + config['epochs_per_job'] - 1) // config['epochs_per_job']
    logger.info(f"Job ready at {job_dir} ({num_chunks} chunks)")


@finetune_app.command("run")
def finetune_run(
    job_dir: Path = typer.Option(..., "--job-dir", help="Path to job directory"),
    chunk_id: int = typer.Option(..., "--chunk-id", help="Chunk to process (1-based, for SLURM)"),
):
    """
    Process one chunk (block of epochs) of a finetuning job.
    
    This is typically called by the SLURM script, not run directly.
    """
    config_path = job_dir / "config.json"
    if not config_path.exists():
        logger.error(f"No config found at {config_path}")
        return
    
    # status.json is the source of truth for current_epoch
    config = load_finetune_config(job_dir)
    status = finetune_jobs.load_job_status(job_dir)
    config["current_epoch"] = status["current_epoch"]

    chunk_data = finetune_jobs.load_chunk_status(job_dir, chunk_id)
    if chunk_data["status"] == "success":
        logger.info(f"Chunk {chunk_id} already complete, skipping")
        return

    logger.info(f"Processing chunk {chunk_id}: epochs {chunk_data['start_epoch']} to {chunk_data['end_epoch']}")

    train_manifest = job_dir / "train_manifest.jsonl"
    val_manifest = job_dir / "val_manifest.jsonl" if (job_dir / "val_manifest.jsonl").exists() else None

    try:
        result = finetune_trainer.run_training_job(
            config, train_manifest, val_manifest,
            chunk_end_epoch=chunk_data["end_epoch"],
        )
        
        if result["status"] == "complete":
            logger.info("Training already complete!")
            finetune_jobs.update_chunk_status(job_dir, chunk_id, "success")
        else:
            checkpoint_path = result.get("checkpoint_path")
            finetune_jobs.update_chunk_status(job_dir, chunk_id, "success")
            logger.info(f"Chunk {chunk_id} complete: checkpoint at {checkpoint_path}")
    
    except Exception as e:
        logger.error(f"Chunk {chunk_id} failed: {e}")
        finetune_jobs.update_chunk_status(job_dir, chunk_id, "failed")
        raise


@finetune_app.command("inspect")
def finetune_inspect(
    job_dir: Path = typer.Option(..., "--job-dir", help="Path to job directory"),
):
    """Inspect finetuning job details and processing statistics."""
    if not (job_dir / "status.json").exists():
        logger.error(f"No job found at {job_dir}")
        return
    
    finetune_jobs.show_job_stats(job_dir)


@finetune_app.command("reset")
def finetune_reset(
    job_dir: Path = typer.Option(..., "--job-dir", help="Path to job directory"),
):
    """
    Reset a finetuning job to start from scratch.
    
    Deletes all checkpoints and resets chunk statuses.
    """
    if not (job_dir / "status.json").exists():
        logger.error(f"No job found at {job_dir}")
        return
    
    finetune_jobs.reset_job(job_dir)
    logger.info(f"Job reset: {job_dir}")
    logger.info(f"Run training again with: sbatch scripts/slurm_finetune.sh {job_dir} 1")


@finetune_app.command("retry")
def finetune_retry(
    job_dir: Path = typer.Option(..., "--job-dir", help="Path to job directory"),
    chunk_id: int = typer.Option(..., "--chunk-id", help="Chunk to retry"),
):
    """
    Reset a single failed chunk back to pending so it can be resubmitted.

    Use this when a chunk fails (OOM, bug, timeout) and you want the
    orchestrator to pick it up again without resetting the whole experiment.
    """
    chunk_path = job_dir / "chunks" / f"chunk_{chunk_id}.json"
    if not chunk_path.exists():
        logger.error(f"Chunk {chunk_id} not found at {chunk_path}")
        return

    chunk_data = finetune_jobs.load_chunk_status(job_dir, chunk_id)
    if chunk_data["status"] == "success":
        logger.error(f"Chunk {chunk_id} already succeeded — use 'reset' to start over")
        return

    finetune_jobs.update_chunk_status(job_dir, chunk_id, "pending")
    logger.info(f"Chunk {chunk_id} reset to pending — orchestrator will resubmit it")


@finetune_app.command("export-model")
def finetune_export(
    job_dir: Path = typer.Option(..., "--job-dir", help="Path to job directory"),
    output_path: Path = typer.Option(..., "--output", help="Path to save exported model"),
    epoch: int = typer.Option(None, "--epoch", help="Epoch to export (default: latest)"),
):
    """
    Export a checkpoint for use in transcription.

    Defaults to the latest checkpoint. Use --epoch to pick a specific one.
    Use 'lyricscribe finetune inspect' to see available checkpoints and metrics.
    """
    config = load_finetune_config(job_dir)

    if epoch is not None:
        checkpoint = get_checkpoint_for_epoch(job_dir, config, epoch)
        if not checkpoint:
            logger.error(f"No checkpoint found for epoch {epoch}")
            available = list_checkpoints(job_dir, config)
            if available:
                epochs_str = ", ".join(str(e) for e, _ in available)
                logger.error(f"Available epochs: {epochs_str}")
            return
    else:
        checkpoint = get_latest_checkpoint(job_dir, config)
        if not checkpoint:
            logger.error("No checkpoints found!")
            return

    if output_path.exists():
        logger.error(f"Output path already exists: {output_path}")
        logger.error("Remove it manually before exporting.")
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)

    if checkpoint.suffix == ".nemo":
        shutil.copy(checkpoint, output_path)
    else:
        shutil.copytree(checkpoint, output_path)

    logger.info(f"Model exported: {checkpoint} -> {output_path}")
    logger.info(f"Use with: lyricscribe transcribe setup --model {output_path}")


if __name__ == "__main__":
    cli()
