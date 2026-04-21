import csv
import json
import logging
import shutil
from pathlib import Path

import typer

# Lightweight imports only — heavy modules (torch, transformers, nemo,
# demucs, librosa) are imported lazily inside the commands that need them
# so the CLI starts in well under a second for status/retry/inspect.
from lyricscribe.finetune import jobs as finetune_jobs
from lyricscribe.finetune.config import (
    create_finetune_config,
    get_checkpoint_for_epoch,
    get_latest_checkpoint,
    list_checkpoints,
    load_finetune_config,
)

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
    from lyricscribe import demucs

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
    from lyricscribe import demucs

    demucs.process_chunk(job_dir=job_dir, chunk_id=chunk_id)


@separate_app.command("inspect")
def separate_inspect(
    job_dir: Path = typer.Option(..., "--job-dir", help="Path to job directory"),
):
    """
    Inspect separation job details and processing statistics.
    """
    from lyricscribe import jobs

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
    from lyricscribe import demucs

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
    from lyricscribe.transcribe import job as transcribe_job

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
    from lyricscribe.transcribe import job as transcribe_job

    transcribe_job.process_chunk(job_dir=job_dir, chunk_id=chunk_id)


@transcribe_app.command("inspect")
def transcribe_inspect(
    job_dir: Path = typer.Option(..., "--job-dir", help="Path to job directory"),
):
    """
    Inspect transcription job details and processing statistics.
    """
    from lyricscribe import jobs

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
    from lyricscribe.dataset import download_jam_alt

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
    from lyricscribe.dataset import download_musdb_alt

    download_musdb_alt(output_dir)


@evaluate_app.command("run")
def evaluate(
    job_dir: Path = typer.Option(
        ..., "--job-dir", help="Path to job directory"
    ),
):
    """Evaluate transcription quality against ground-truth lyrics."""
    from lyricscribe.evaluate import evaluate_job

    stats = evaluate_job(job_dir, verbose=True)
    if not stats:
        return

    logger.info(f"--- Summary ({stats['n_songs']} songs) ---")
    logger.info(f"WER: {stats['wer']:.2%}")
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
    from lyricscribe.evaluate import collect_evaluation_data

    all_stats = collect_evaluation_data(jobs_dir)

    if not all_stats:
        logger.error("No successful evaluations found to summarize.")
        return

    with open(output, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "job_dir", "model", "dataset", "filename", "vad", "chunked",
                "wer", "n_songs", "insertions", "deletions", "substitutions", "hits"
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
        None, "--musdb-dir", help="Root MUSDB directory (alignments + ground-truth both come from each song's lyrics.json)"
    ),
):
    """Generate evaluation plots from job directories.

    To include the artifact quartile chart, pass --features-dir, --musdb-dir,
    and --results-file (repeat for multiple models). Alignments are read from
    the ``alignment`` field of each song's ``lyrics.json``, so run
    ``lyricscribe dataset align`` on the MUSDB directory first.
    """
    from lyricscribe import plots
    from lyricscribe.transcribe.artifacts import correlation

    word_dataset = None
    result_files = _collect_result_files(
        results_file,
        jobs_dir=jobs_dir,
        job_name=results_job_name,
    )
    if features_dir is not None and musdb_dir is not None and result_files:
        word_dataset = correlation.build_dataset(
            features_dir, result_files, musdb_dir
        )
    elif any(
        opt is not None for opt in [features_dir, musdb_dir]
    ) or result_files or results_job_name:
        logger.warning(
            "Artifact chart requires --features-dir, --musdb-dir, and either "
            "one or more --results-file values or --results-job-name. "
            "Skipping artifact chart."
        )
    plots.generate_all_plots(jobs_dir, output_dir, word_dataset=word_dataset)


@artifacts_app.command("extract")
def artifact_extract(
    musdb_dir: Path = typer.Option(..., "--musdb-dir", help="Root MUSDB directory"),
    output_dir: Path = typer.Option(..., "--output-dir", help="Directory to write artifact feature JSON files"),
):
    """Extract per-frame artifact features for each song."""
    from lyricscribe.transcribe.artifacts import extractor

    extractor.process_dataset(musdb_dir, output_dir)


@dataset_app.command("align")
def dataset_align(
    dataset_dir: Path = typer.Option(
        ..., "--dataset-dir", help="Root dataset directory containing song subdirectories"
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
    filename: str = typer.Option(
        "vocals.wav",
        "--filename",
        help="Audio filename inside each song subdirectory (e.g. htdemucs_ft_vocals.wav)",
    ),
    num_chunks: int = typer.Option(
        1,
        "--num-chunks",
        help="Total number of shards to partition the dataset into",
    ),
    chunk_id: int = typer.Option(
        0,
        "--chunk-id",
        help="0-indexed shard to process (requires --num-chunks > 1)",
    ),
    skip_existing: bool = typer.Option(
        True,
        "--skip-existing/--no-skip-existing",
        help="Skip songs whose lyrics.json already has a non-null alignment",
    ),
):
    """
    Run Montreal Forced Aligner on a dataset and write word-level alignments
    back into each song's lyrics.json (populates the `alignment` field).
    """
    from lyricscribe.transcribe.artifacts import processor

    processor.align(
        dataset_dir,
        container=container,
        mfa_root=mfa_root,
        filename=filename,
        num_chunks=num_chunks,
        chunk_id=chunk_id,
        skip_existing=skip_existing,
    )


@artifacts_app.command("build")
def artifact_build(
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
        ..., "--musdb-dir", help="Root MUSDB directory (alignments + ground truth both from each song's lyrics.json)"
    ),
    output: Path = typer.Option(
        ..., "--output", help="Path to write the word-level CSV dataset"
    ),
):
    """Build the word-level dataset combining alignments, artifacts, and errors.

    Alignments are read from the ``alignment`` field of each song's
    ``lyrics.json`` — run ``lyricscribe dataset align`` on the MUSDB
    directory first.
    """
    from lyricscribe.transcribe.artifacts import correlation

    result_files = _collect_result_files(
        results_file,
        jobs_dir=results_jobs_dir,
        job_name=results_job_name,
    )
    correlation.build_dataset(
        features_dir, result_files, musdb_dir, csv_output=output
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
        None, "--batch-size", help="Training batch size (default: arch-dependent)"
    ),
    max_epochs: int = typer.Option(
        None, "--max-epochs", help="Maximum training epochs (default: from config)"
    ),
    epochs_per_job: int = typer.Option(
        None, "--epochs-per-job", help="Epochs to train per SLURM job (default: from config)"
    ),
    learning_rate: float = typer.Option(
        None, "--learning-rate", help="Peak learning rate (default: arch-dependent)"
    ),
    no_augment: bool = typer.Option(
        False, "--no-augment", help="Disable SpecAugment"
    ),
    batch_duration: float = typer.Option(
        None,
        "--batch-duration",
        help="Canary-only: Lhotse batch duration in seconds (default: 600; lower for <24GB GPUs)",
    ),
    freeze_encoder: bool = typer.Option(
        False,
        "--freeze-encoder",
        help="Canary-only: freeze the encoder so only the decoder+head train (fits on 12-16 GB GPUs)",
    ),
    eval_subset_size: int = typer.Option(
        None,
        "--eval-subset-size",
        help="Fixed number of validation samples both Whisper and NeMo evaluate against each epoch (default: 200)",
    ),
    windows_per_song: int = typer.Option(
        3,
        "--windows-per-song",
        help="Random 30s windows sampled per song (per ~30s of song duration). Higher = more diverse coverage across epochs. Raise toward your total epoch count for approximate per-epoch randomization.",
    ),
    line_overlap_threshold: float = typer.Option(
        0.7,
        "--line-overlap-threshold",
        help="Minimum fraction of a synced line's duration that must fall inside a window for its text to be included in that window's label. Guards against partial-audio hallucination and deletion bias.",
    ),
):
    """
    Setup a finetuning experiment with epoch-level checkpointing.

    Creates job directory with manifests and chunk files for SLURM processing.
    Architecture is auto-detected from model name.

    Pass multiple --filename flags to train on a random mix of audio types.
    """
    # Lazy import: data.py pulls in transformers (WhisperTokenizer),
    # which is heavy. Only setup needs it.
    from lyricscribe.finetune import data as finetune_data

    overrides = {
        k: v
        for k, v in {
            "batch_size": batch_size,
            "max_epochs": max_epochs,
            "epochs_per_job": epochs_per_job,
            "learning_rate": learning_rate,
            "batch_duration": batch_duration,
            "freeze_encoder": freeze_encoder if freeze_encoder else None,
            "eval_subset_size": eval_subset_size,
        }.items()
        if v is not None
    }

    try:
        config = create_finetune_config(
            base_model=model,
            train_dir=train_dir,
            output_dir=output_dir,
            filenames=filename,
            val_dir=val_dir,
            use_augmentation=not no_augment,
            **overrides,
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

    n_train = finetune_data.create_manifest(
        train_dataset,
        train_manifest,
        architecture=config['architecture'],
        model_name=config['base_model'],
        windows_per_song_multiplier=windows_per_song,
        line_overlap_threshold=line_overlap_threshold,
    )
    logger.info(f"Training: {n_train} chunks (synced-line segments)")

    val_manifest = None
    if val_dir:
        val_dataset = finetune_data.LyricsDataset(
            dataset_dir=val_dir,
            filenames=config['filenames'],
        )
        val_manifest = job_dir / "val_manifest.jsonl"
        # Validation manifest stays at windows_per_song=1: eval is just a
        # sanity check, we don't need the same over-sampling as training.
        n_val = finetune_data.create_manifest(
            val_dataset,
            val_manifest,
            architecture=config['architecture'],
            model_name=config['base_model'],
            windows_per_song_multiplier=1,
            line_overlap_threshold=line_overlap_threshold,
        )
        logger.info(f"Validation: {n_val} chunks")

        # Write a fixed-size shuffled subset that both Whisper and NeMo
        # evaluate against each epoch. Without this each trainer picks
        # its own window (Subset vs. limit_val_batches) so cross-arch
        # WER comparisons are over different audio — see #2.
        subset_size = config.get("eval_subset_size", 200)
        val_subset_manifest = job_dir / "val_subset_manifest.jsonl"
        n_subset = finetune_data.write_subset_manifest(
            val_manifest, val_subset_manifest, size=subset_size, seed=42,
        )
        logger.info(f"Validation subset (used during training): {n_subset} chunks")

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
    # Lazy import: trainer pulls in NeMo + Lightning + transformers,
    # which takes a long time. Avoid that overhead for non-run commands.
    from lyricscribe.finetune import trainer as finetune_trainer

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
    # Prefer the fixed-size subset manifest for during-training eval
    # so Whisper and NeMo evaluate on identical samples. Fall back to
    # the full val manifest for older experiments that predate the
    # subset file.
    val_subset = job_dir / "val_subset_manifest.jsonl"
    val_full = job_dir / "val_manifest.jsonl"
    if val_subset.exists():
        val_manifest = val_subset
    elif val_full.exists():
        val_manifest = val_full
    else:
        val_manifest = None

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
    chunk_id: int = typer.Option(None, "--chunk-id", help="Specific chunk to retry (default: all failed chunks)"),
):
    """
    Reset failed chunks back to pending so the orchestrator resubmits them.

    Without ``--chunk-id``, resets every failed chunk in the experiment.
    With ``--chunk-id``, resets only that specific chunk.

    Use this after fixing a bug to re-run failed chunks without losing
    successful checkpoints.
    """
    if not (job_dir / "status.json").exists():
        logger.error(f"No job found at {job_dir}")
        return

    if chunk_id is not None:
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
    else:
        n = finetune_jobs.reset_failed_chunks(job_dir)
        if n == 0:
            logger.info(f"No failed chunks in {job_dir.name}")
        else:
            logger.info(f"Reset {n} failed chunks in {job_dir.name} — orchestrator will resubmit them")


@finetune_app.command("retry-all")
def finetune_retry_all(
    experiments_dir: Path = typer.Option(..., "--experiments-dir", help="Directory containing experiment subdirectories"),
):
    """
    Reset all failed chunks across every experiment in a directory.

    This is the "I just deployed a fix, retry everything" button. It
    leaves successful chunks and existing checkpoints alone.
    """
    if not experiments_dir.exists():
        logger.error(f"Directory not found: {experiments_dir}")
        return

    total_reset = 0
    total_experiments = 0
    for job_dir in sorted(experiments_dir.iterdir()):
        if not job_dir.is_dir():
            continue
        if not (job_dir / "status.json").exists():
            continue
        total_experiments += 1
        n = finetune_jobs.reset_failed_chunks(job_dir)
        if n > 0:
            logger.info(f"  {job_dir.name}: reset {n} chunks")
            total_reset += n

    logger.info(f"Reset {total_reset} chunks across {total_experiments} experiments")


@finetune_app.command("status")
def finetune_status(
    experiments_dir: Path = typer.Option(..., "--experiments-dir", help="Directory containing experiment subdirectories"),
):
    """
    Print a one-line summary of every experiment's progress.

    Shows chunk counts, last checkpoint, and last metric values for
    each experiment in the directory.
    """
    if not experiments_dir.exists():
        logger.error(f"Directory not found: {experiments_dir}")
        return

    rows = []
    for job_dir in sorted(experiments_dir.iterdir()):
        if not job_dir.is_dir():
            continue
        if not (job_dir / "status.json").exists():
            continue
        rows.append(finetune_jobs.gather_experiment_status(job_dir))

    if not rows:
        logger.info(f"No experiments found in {experiments_dir}")
        return

    name_width = max(len(r["exp_name"]) for r in rows)
    for r in rows:
        c = r["chunks"]
        ckpt = r["last_checkpoint"] or "none"
        chunks_str = f"{c['success']}✓ {c['failed']}✗ {c['running']}▶ {c['pending']}⏸"
        loss_str = ""
        if r["last_metrics"]:
            for k in ("train_loss", "loss", "eval_loss"):
                if k in r["last_metrics"]:
                    loss_str = f" {k}={r['last_metrics'][k]:.3f}"
                    break
        logger.info(
            f"  {r['exp_name']:<{name_width}}  "
            f"{r['current_epoch']:>2}/{r['max_epochs']} epochs  "
            f"chunks={chunks_str}  ckpt={ckpt}{loss_str}"
        )


@finetune_app.command("tune-batch")
def finetune_tune_batch(
    dataset_dir: Path = typer.Option(
        ...,
        "--dataset-dir",
        help="Folder of song subdirectories. The sweep reads the first "
             "bs samples of the requested filename per trial — no "
             "manifests, no job config files, just direct file reads.",
    ),
    filename: str = typer.Option(
        ...,
        "--filename",
        help="Which audio file to read in each song subdirectory "
             "(e.g. htdemucs_ft_vocals.wav, audio.mp3). Different "
             "filenames can have different VRAM footprints (stereo vs. "
             "mono source, different codec), so sweep on whichever "
             "you'll actually train against.",
    ),
    model: str = typer.Option(
        ..., "--model", help="Model identifier (e.g. nvidia/parakeet-tdt-0.6b-v3).",
    ),
    freeze_encoder: bool = typer.Option(
        False,
        "--freeze-encoder",
        help="Freeze the encoder — Canary/Parakeet only. MUST match "
             "production; freeze state shifts VRAM by ~5–10 GiB.",
    ),
    max_duration: float = typer.Option(
        30.0,
        "--max-duration",
        help="Audio clip length in seconds. Longer = more activations = "
             "more VRAM. Must match production.",
    ),
    start: int = typer.Option(
        1, "--start", help="Smallest batch size to try."
    ),
    max_bs: int = typer.Option(
        128, "--max", help="Largest batch size to try.",
    ),
    safety_margin_pct: float = typer.Option(
        15.0,
        "--safety-margin",
        help="Percent shaved off the VRAM ceiling when no sweet spot "
             "is found — covers DDP overhead + per-sample variance.",
    ),
    timeout_s: int = typer.Option(
        900,
        "--timeout",
        help="Kill any trial running longer than this (seconds). "
             "Usually indicates a hang on model download or first-step "
             "CUDA compilation.",
    ),
    min_gap: int = typer.Option(
        2,
        "--min-gap",
        help="After doubling hits OOM, bisect between last-success and "
             "OOM until gap ≤ this. 1 = exact ceiling, 2 = one less "
             "trial with 1-unit slop.",
    ),
):
    """
    Find the largest batch size that fits for this model on this hardware.

    Run as a SLURM job with the **same node shape** (GPU count, GPU type,
    DDP rank count) you'll actually train on — per-rank VRAM depends on
    the DDP bucket size, which depends on rank count.

    Each trial is a fresh ``multiprocessing.Process`` (spawn context) so
    an OOM cleanly crashes the worker instead of corrupting the parent's
    CUDA state. In-process measurement:

    1. Load the model.
    2. Read ``bs`` real audio files from ``--dataset-dir``, decode to
       16 kHz mono, pad/truncate to ``--max-duration``.
    3. Build the forward batch exactly as the model's training_step
       would see it.
    4. Run warmup + 4 training steps (forward + backward + optimizer)
       in bf16-mixed autocast with AdamW.
    5. Report peak reserved VRAM and median step time.

    No manifests are created. No config files are written. Nothing
    persists after the sweep.
    """
    from lyricscribe.finetune import tune_batch

    if not dataset_dir.is_dir():
        logger.error(f"Not a directory: {dataset_dir}")
        raise typer.Exit(code=1)

    tune_batch.tune_batch_size(
        dataset_dir=dataset_dir,
        filename=filename,
        model_name=model,
        freeze_encoder=freeze_encoder,
        max_duration_s=max_duration,
        start=start,
        max_bs=max_bs,
        min_gap=min_gap,
        timeout_s=timeout_s,
        safety_margin_pct=safety_margin_pct,
    )


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
