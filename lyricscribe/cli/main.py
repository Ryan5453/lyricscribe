import csv
import json
import logging
import random
import shutil
from pathlib import Path

import typer

from lyricscribe import jobs
from lyricscribe.finetune import domains as finetune_domains
from lyricscribe.finetune import jobs as finetune_jobs
from lyricscribe.finetune.config import (
    create_finetune_config,
    detect_architecture,
    list_exports,
    load_finetune_config,
    resolve_steps,
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

from lyricscribe.cli.analyze import analyze_app  # noqa: E402
from lyricscribe.cli.ensemble import ensemble_app  # noqa: E402

cli.add_typer(ensemble_app, name="ensemble")
cli.add_typer(analyze_app, name="analyze")


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

    :param explicit_results: Result files given directly.
    :param jobs_dir: Root holding one dir per model run.
    :param job_name: Job dir name to look for under each run.
    :return: Resolved result files, deduplicated, in input order.
    :raises ValueError: If *job_name* is given without *jobs_dir* or matches nothing.
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
    """
    Configure logging for every command.
    """
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
    model: str = typer.Option(
        "htdemucs",
        "--model",
        help="Unblend model name (see `unblend models list`). Custom "
        "checkpoints such as unwa Kim-FT2 are added with `unblend models "
        "import`.",
    ),
    chunks: int = typer.Option(
        5,
        "--chunks",
        min=1,
        help="Number of chunks to split into.",
    ),
    stem: str | None = typer.Option(
        None,
        "--stem",
        help="Which stem to isolate (vocals, drums, bass, other). If not specified, all stems are saved.",
    ),
    output_filename: str | None = typer.Option(
        None,
        "--output-filename",
        help="Output filename for the isolated stem (default "
             "{model}_{stem}.wav), e.g. vocals.wav.",
    ),
    shifts: int = typer.Option(
        1, "--shifts", min=0, max=20,
        help="Random time-shift passes to average (0 = deterministic).",
    ),
    split_overlap: float = typer.Option(
        0.25, "--split-overlap",
        help="Overlap between consecutive model segments, in [0, 1).",
    ),
    seed: int = typer.Option(0, "--seed", help="Seed for the shift offsets."),
    dtype: str = typer.Option(
        "auto", "--dtype",
        help="Compute precision: auto (FP16 on CUDA), float32, float16, bfloat16.",
    ),
):
    """
    Initialize separation job by registering files into chunks.
    """
    from lyricscribe import unblend

    unblend.setup_job(
        directories=directories,
        job_dir=job_dir,
        filename=filename,
        model=model,
        num_chunks=chunks,
        stem=stem,
        output_filename=output_filename,
        shifts=shifts,
        split_overlap=split_overlap,
        seed=seed,
        dtype=dtype,
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
    from lyricscribe import unblend

    unblend.process_chunk(job_dir=job_dir, chunk_id=chunk_id)


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
    from lyricscribe import unblend

    unblend.reset_job(job_dir)


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
    chunks: int = typer.Option(
        1,
        "--chunks",
        min=1,
        help="Number of chunks to split into",
    ),
    batch_size: int = typer.Option(
        1,
        "--batch-size",
        help="Batch size for inference.",
    ),
    vad: bool = typer.Option(
        False, "--vad", help="Enable VAD-based segmentation"
    ),
    vad_method: str = typer.Option(
        "silero",
        "--vad-method",
        help="VAD algorithm: 'silero' (neural, speech-trained) or 'rms' "
             "(RMS-amplitude on separated vocals, MSS-ALT-style). "
             "Only used when --vad is set.",
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
    repetition_penalty: float = typer.Option(
        1.0,
        "--repetition-penalty",
        help="Decoder repetition penalty on generated tokens (never the "
             "prompt). ``1.0`` disables. Every backend applies it.",
    ),
    no_repeat_ngram_size: int = typer.Option(
        0,
        "--no-repeat-ngram-size",
        help="Decoder ban on repeating a generated n-gram (never the prompt). "
             "``0`` disables. Every backend applies it.",
    ),
    mit: bool = typer.Option(
        False,
        "--mit",
        help="Shortcut: enable both ``--repetition-penalty 1.7`` and "
             "``--no-repeat-ngram-size 5`` together. Overrides explicit "
             "values for those two flags.",
    ),
    num_beams: int = typer.Option(
        1,
        "--num-beams",
        min=1,
        help="Beam width; 1 (default) is greedy. Every backend applies it (Parakeet: NeMo's TDT beam search).",
    ),
    condition_on_prev: bool = typer.Option(
        True,
        "--condition-on-prev/--no-condition-on-prev",
        help="Whisper long-form ``condition_on_prev_tokens``. Only affects "
             "the sequential (non-chunked) path. Whisper-only.",
    ),
    temperature_fallback: bool = typer.Option(
        False,
        "--temperature-fallback",
        help="Whisper temperature fallback (0.0..1.0) gated by the "
             "compression-ratio / logprob thresholds. Only affects the "
             "sequential (non-chunked) path. Whisper-only.",
    ),
    compression_ratio_threshold: float | None = typer.Option(
        None, "--compression-ratio-threshold",
        help="Fallback trigger (default 2.4 with --temperature-fallback).",
    ),
    logprob_threshold: float | None = typer.Option(
        None, "--logprob-threshold",
        help="Fallback trigger (default -1.0 with --temperature-fallback).",
    ),
    prompt_from_dirname: bool = typer.Option(
        False,
        "--prompt-from-dirname",
        help="Derive a per-song prompt from the song directory name "
             "(``A_-_B``; no artist/title order is assumed): Whisper's "
             "decoder prompt, or Qwen3-ASR's context. Ignored by other "
             "backends.",
    ),
    backend: str | None = typer.Option(
        None,
        "--backend",
        help="Force the backend: whisper, qwen3_asr, qwen3_omni, "
             "musicflamingo, cohere, granite, nemotron, parakeet, canary. "
             "Default: detected from the model id / checkpoint config.json.",
    ),
    max_new_tokens: int | None = typer.Option(
        None,
        "--max-new-tokens",
        help="Per-chunk generation cap for the autoregressive backends, "
             "within each decoder's position limit. Default 448 (Whisper's "
             "decoder limit), which Whisper can't exceed.",
    ),
    adapter: str | None = typer.Option(
        None,
        "--adapter",
        help="PEFT adapter dir merged into Qwen3-ASR, Qwen3-Omni (its thinker), Nemotron or Music Flamingo.",
    ),
    auto_language: bool = typer.Option(
        False,
        "--auto-language",
        help="Qwen3-ASR: don't force the --lyrics-file language; let the "
             "model identify it (qwen-asr's default).",
    ),
    deloop_max_run: int | None = typer.Option(
        None,
        "--deloop-max-run",
        min=1,
        help="Post-hoc repair: collapse any <=25-word cycle repeated more "
             "than this many times. Raw text is kept as transcription_raw. "
             "Prefer scoring-time de-looping, which runs on normalised words.",
    ),
    prompt_mode: str | None = typer.Option(
        None,
        "--prompt-mode",
        help="Prompt: shared (the one prompt every "
             "backend with a text input can send) or the backend's own: native "
             "(Whisper, Canary, Cohere, Qwen3-ASR: no prompt / empty context; "
             "their default), asr (Granite, Nemotron: the card's prompt; default), "
             "and for Omni / Music Flamingo legacy (default), asr or lyrics (the "
             "vendor's documented prompt for that task). Parakeet takes none.",
    ),
    dtype: str | None = typer.Option(
        None,
        "--dtype",
        help="Compute dtype override: Whisper (float16 default, bfloat16, float32), "
             "Parakeet, Canary and Music Flamingo (float32 default, bfloat16).",
    ),
):
    """
    Initialize a transcription job by registering files into chunks.
    """
    from lyricscribe.transcribe import job as transcribe_job

    if vad_method not in ("silero", "rms"):
        raise typer.BadParameter(
            f"--vad-method must be 'silero' or 'rms', got {vad_method!r}"
        )

    if mit:
        repetition_penalty = 1.7
        no_repeat_ngram_size = 5
    if vad and vad_method == "rms" and not vad_source_file:
        raise typer.BadParameter(
            "--vad-method=rms requires --vad-source pointing to a separated "
            "vocal track (e.g. htdemucs_ft_vocals.wav) or a true vocal stem "
            "(e.g. vocals.wav). RMS amplitude on a mixed track is dominated "
            "by accompaniment and produces garbage segments."
        )

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
        vad_method=vad_method,
        repetition_penalty=repetition_penalty,
        no_repeat_ngram_size=no_repeat_ngram_size,
        num_beams=num_beams,
        condition_on_prev=condition_on_prev,
        temperature_fallback=temperature_fallback,
        compression_ratio_threshold=compression_ratio_threshold,
        logprob_threshold=logprob_threshold,
        prompt_from_dirname=prompt_from_dirname,
        backend=backend,
        max_new_tokens=max_new_tokens,
        adapter=adapter,
        auto_language=auto_language,
        deloop_max_run=deloop_max_run,
        prompt_mode=prompt_mode,
        dtype=dtype,
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
    deleted = 0
    for p in job_dir.glob("results*.jsonl"):
        p.unlink(missing_ok=True)
        deleted += 1
    logger.info(f"Deleted {deleted} results file(s)")

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
def evaluate_run(
    jobs_dir: Path = typer.Option(
        ..., "--jobs-dir", help="Path to base jobs directory containing model subdirectories"
    ),
    output_dir: Path = typer.Option(
        ..., "--output-dir", help="Directory to write all outputs: CSV, LaTeX tables, and PDF plots"
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
    """
    Aggregate evaluation results into a CSV + LaTeX tables and generate analysis plots.

    All outputs land in --output-dir:
      - evaluation_summary.csv
      - tables/full_results.tex, tables/headline_results.tex
      - *.pdf plots

    To include the artifact quartile chart, pass --features-dir, --musdb-dir,
    and --results-file (repeat for multiple models). Alignments are read from
    the ``alignment`` field of each song's ``lyrics.json``, so run
    ``lyricscribe dataset align`` on the MUSDB directory first.
    """
    import pandas as pd

    from lyricscribe import plots
    from lyricscribe.evaluate import collect_evaluation_data
    from lyricscribe.latex_tables import (
        write_full_results_tex,
        write_headline_results_tex,
    )
    from lyricscribe.transcribe.artifacts import correlation

    all_stats = collect_evaluation_data(jobs_dir)

    if not all_stats:
        logger.error("No successful evaluations found to summarize.")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "evaluation_summary.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "job_dir", "model", "dataset", "filename", "vad", "vad_method",
                "vad_source", "chunked", "repetition_penalty", "no_repeat_ngram_size",
                "wer", "n_songs", "insertions", "deletions", "substitutions", "hits"
            ]
        )
        writer.writeheader()
        writer.writerows(all_stats)

    logger.info(f"Successfully summarized {len(all_stats)} jobs -> {csv_path}")

    tables_dir = output_dir / "tables"
    df = pd.DataFrame(all_stats)
    write_full_results_tex(df, tables_dir / "full_results.tex")
    write_headline_results_tex(df, tables_dir / "headline_results.tex")

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
    plots.generate_all_plots(jobs_dir, output_dir, word_dataset=word_dataset, df=df)


@artifacts_app.command("extract")
def artifact_extract(
    musdb_dir: Path = typer.Option(..., "--musdb-dir", help="Root MUSDB directory"),
    output_dir: Path = typer.Option(..., "--output-dir", help="Directory to write artifact feature JSON files"),
):
    """
    Extract per-frame artifact features for each song.
    """
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
        min=1,
        help="Total number of shards to partition the dataset into",
    ),
    chunk_id: int = typer.Option(
        0,
        "--chunk-id",
        min=0,
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
    """
    Build the word-level dataset combining alignments, artifacts, and errors.

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
    batch_size: int = typer.Option(None, "--batch-size", help="Windows per device per micro-batch (default: arch-dependent)."),
    global_batch: int = typer.Option(
        None, "--global-batch", help="Windows per optimizer step; gradient accumulation makes up the rest (default: 64).",
    ),
    learning_rate: float = typer.Option(None, "--learning-rate", help="Peak learning rate (default: arch-dependent)."),
    weight_decay: float = typer.Option(None, "--weight-decay", help="AdamW weight decay, not on biases or norms (default: 0.01)."),
    schedule: str = typer.Option(
        None, "--schedule",
        help="LR schedule after warmup: wsd (constant, then a (1 − √) cooldown; default), cosine or linear.",
    ),
    warmup_steps: int = typer.Option(None, "--warmup-steps", help="Linear warmup steps (default: 200)."),
    cooldown: float = typer.Option(
        None, "--cooldown",
        help="wsd: fraction of the run that cools down (default: 0.2; 0 = a constant-LR run to branch cooldowns from).",
    ),
    min_lr: float = typer.Option(None, "--min-lr", help="cosine / linear: final learning rate (default: 0)."),
    epochs: float = typer.Option(None, "--epochs", help="Run length in epochs, possibly fractional (default: 3)."),
    max_steps: int = typer.Option(None, "--max-steps", help="Run length in optimizer steps (overrides --epochs)."),
    keep: list[int] = typer.Option(None, "--keep", help="Step whose resumable checkpoint is kept (repeatable)."),
    export_every_epoch: bool = typer.Option(False, "--export-every-epoch", help="Export an inference model at every epoch end."),
    export_step: list[int] = typer.Option(None, "--export-step", help="Export an inference model at this step (repeatable)."),
    no_export_final: bool = typer.Option(False, "--no-export-final", help="Don't export at the final step."),
    resume_every: int = typer.Option(None, "--resume-every", help="Steps between rolling resume checkpoints (default: 500)."),
    seed: int = typer.Option(None, "--seed", help="Run seed: window order, dither, masks, LoRA init (default: 0)."),
    prompt_mode: str = typer.Option(
        None, "--prompt-mode",
        help="Training prompt, named as inference's --prompt-mode (e.g. shared, native, asr, lyrics); "
             "recorded with every export, which inference then refuses to decode with another prompt.",
    ),
    adaptation: str = typer.Option(
        None, "--adaptation",
        help="full (default up to 2B) or nvidia_adapter (full-rank encoder + projector, LoRA on the LLM; "
             "the default for Nemotron, Qwen3-Omni and Music Flamingo, and an option for Qwen3-ASR).",
    ),
    no_augment: bool = typer.Option(False, "--no-augment", help="Disable SpecAugment"),
    freeze_encoder: bool = typer.Option(
        False, "--freeze-encoder", help="Parakeet/Canary: freeze the encoder so only the decoder and joint/head train",
    ),
    eval_subset_size: int = typer.Option(
        None, "--eval-subset-size", help="Validation windows whose loss is computed at every export (default: 200)",
    ),
    windows_per_song: int = typer.Option(
        1,
        "--windows-per-song",
        help="Random 30s windows sampled per song (per ~30s of song duration). Higher = more diverse coverage across epochs. Raise toward your total epoch count for approximate per-epoch randomization.",
    ),
    line_overlap_threshold: float = typer.Option(
        0.7,
        "--line-overlap-threshold",
        help="Minimum fraction of a synced line's duration that must fall inside a window for its text to be included in that window's label. Guards against partial-audio hallucination and deletion bias.",
    ),
    whisper_timestamp_prob: float = typer.Option(
        None,
        "--whisper-timestamp-prob",
        help="Whisper only: fraction of samples trained with timestamp labels (0 breaks >30s long-form decoding).",
    ),
    dedup_max_run: int = typer.Option(
        0,
        "--dedup-max-run",
        help="Cap runs of consecutive identical words in labels (0 = off).",
    ),
    max_unlabeled_vocal_frac: float = typer.Option(
        None,
        "--max-unlabeled-vocal-frac",
        help="Reject windows where more than this fraction of sung time "
             "(synced-line spans) has no label text; offsets are re-drawn "
             "up to 8 times per window. Counters the deletion bias learned "
             "from labels that drop boundary words/lines. Recommended 0.2. "
             "Unset = off.",
    ),
    ref_selfwer_file: Path = typer.Option(
        None,
        "--ref-selfwer-file",
        help="Per-song reference self-WER artifact (JSONL with song_id + "
             "self_wer, from `lyricscribe analyze selfwer`). Used with "
             "--max-ref-self-wer to skip songs whose unsynced/synced "
             "references disagree heavily. Datasets are never modified.",
    ),
    max_ref_self_wer: float = typer.Option(
        None,
        "--max-ref-self-wer",
        help="Skip songs whose reference self-WER exceeds this value "
             "(requires --ref-selfwer-file). Recommended 0.3. Songs absent "
             "from the artifact (or with no self-WER) are kept.",
    ),
    no_mfa_alignment: bool = typer.Option(
        False,
        "--no-mfa-alignment",
        help="Ignore MFA word alignments and build labels from synced "
             "lines only (human word order, line-level timing).",
    ),
    audio_domain_mode: str = typer.Option(
        None,
        "--audio-domain-mode",
        help="Post-pass over the manifests when several --filename values "
             "are given. 'balanced': each song wholly uses the first "
             "filename or the rest (first existing), 50/50 by window count "
             "(e.g. --filename vocals.wav --filename audio.wav --filename "
             "audio.mp3). 'fallback': every "
             "song uses the first existing of the 2nd..Nth filenames. Unset "
             "= a random available file per song.",
    ),
    manifests_from: Path = typer.Option(
        None,
        "--manifests-from",
        help="Copy train/val/val-subset manifests from an existing "
             "experiment (or a prepare-manifests set) instead of generating them "
             "(manifests are model-agnostic). Manifest-construction flags are then ignored.",
    ),
    max_train_windows: int = typer.Option(
        None, "--max-train-windows", min=1,
        help="Train on a seeded random subset of this many windows. Unset = every window.",
    ),
    train_window_seed: int = typer.Option(
        0, "--train-window-seed", help="Seed for --max-train-windows sampling.",
    ),
    exp_name: str = typer.Option(
        None, "--exp-name",
        help="Experiment directory name (default: <arch>_<filenames>_<UTC "
             "timestamp>). Set it when launching setups in parallel.",
    ),
):
    """
    Set up a finetuning job: build (or copy) the manifests and fix the step plan.

    Architecture is auto-detected from model name. Pass multiple --filename flags to
    train on a random mix of audio types (or a controlled mix with --audio-domain-mode).
    Train with ``finetune run``; continue a constant-LR run with ``finetune extend`` and
    branch cooldowns off it with ``finetune branch``.
    """
    from lyricscribe.finetune import data as finetune_data

    if audio_domain_mode and audio_domain_mode not in finetune_domains.AUDIO_DOMAIN_MODES:
        logger.error(f"--audio-domain-mode must be one of {finetune_domains.AUDIO_DOMAIN_MODES}")
        return
    if prompt_mode is not None:
        from lyricscribe.transcribe.job import backend_prompt_modes

        modes = backend_prompt_modes(detect_architecture(model))
        if prompt_mode not in modes:
            logger.error(f"{model} has no prompt {prompt_mode!r}; expected one of {modes}")
            return

    skip_song_ids: set[str] | None = None
    if max_ref_self_wer is not None:
        if ref_selfwer_file is None:
            logger.error("--max-ref-self-wer requires --ref-selfwer-file")
            return
        skip_song_ids = set()
        with open(ref_selfwer_file) as sf_:
            for line in sf_:
                r = json.loads(line)
                sw = r.get("self_wer")
                if sw is not None and sw > max_ref_self_wer:
                    skip_song_ids.add(r["song_id"])
        logger.info(
            f"Ref-quality filter: skipping {len(skip_song_ids)} songs with "
            f"self-WER > {max_ref_self_wer}"
        )

    try:
        config = create_finetune_config(
            base_model=model,
            train_dir=train_dir,
            output_dir=output_dir,
            filenames=filename,
            val_dir=val_dir,
            batch_size=batch_size,
            global_batch=global_batch,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            schedule=schedule,
            warmup_steps=warmup_steps,
            cooldown=cooldown,
            min_lr=min_lr,
            epochs=epochs,
            max_steps=max_steps,
            keep_steps=keep or None,
            export_steps=export_step or None,
            export_every_epoch=export_every_epoch,
            export_final=not no_export_final,
            resume_every=resume_every,
            seed=seed,
            prompt_mode=prompt_mode,
            adaptation=adaptation,
            use_augmentation=not no_augment,
            freeze_encoder=freeze_encoder,
            eval_subset_size=eval_subset_size,
            whisper_timestamp_prob=whisper_timestamp_prob,
            dedup_max_run=dedup_max_run or None,
            max_unlabeled_vocal_frac=max_unlabeled_vocal_frac,
            ref_selfwer_file=str(ref_selfwer_file) if ref_selfwer_file else None,
            max_ref_self_wer=max_ref_self_wer,
            use_alignment=not no_mfa_alignment,
            windows_per_song=windows_per_song,
            line_overlap_threshold=line_overlap_threshold,
            audio_domain_mode=audio_domain_mode,
            manifests_from=str(manifests_from) if manifests_from else None,
            max_train_windows=max_train_windows,
            train_window_seed=train_window_seed,
            exp_name=exp_name,
        )
    except ValueError as e:
        logger.error(f"Invalid configuration: {e}")
        return

    logger.info(f"Experiment: {config['exp_name']}")
    logger.info(f"Architecture: {config['architecture']} (auto-detected), adaptation {config['adaptation']}")
    logger.info(f"Model: {config['base_model']}")
    logger.info(f"Filenames: {', '.join(config['filenames'])}")

    job_dir = Path(config['output_dir']) / config['exp_name']
    if (job_dir / "config.json").exists():
        logger.error(f"{job_dir} already holds an experiment; refusing to overwrite")
        return
    job_dir.mkdir(parents=True, exist_ok=True)
    train_manifest = job_dir / "train_manifest.jsonl"

    if manifests_from:
        for name in ("train_manifest.jsonl", "val_manifest.jsonl", "val_subset_manifest.jsonl"):
            if (manifests_from / name).exists():
                shutil.copy2(manifests_from / name, job_dir / name)
        if not train_manifest.exists():
            logger.error(f"No train_manifest.jsonl in {manifests_from}")
            return
        # The copied manifest's recipe, not this command's (unused) construction flags.
        source_keys = finetune_data.source_data_keys(manifests_from)
        for key in finetune_data.DATA_KEYS:
            if key in source_keys and key not in ("train_dir", "val_dir", "filenames"):
                config[key] = source_keys[key]
        config["manifest_data_keys"] = source_keys
        logger.info(f"Copied manifests from {manifests_from}")
    else:
        train_dataset = finetune_data.LyricsDataset(
            dataset_dir=train_dir,
            filenames=config['filenames'],
        )

        if len(train_dataset) == 0:
            logger.error("No valid songs found in training directory!")
            shutil.rmtree(job_dir, ignore_errors=True)
            return

        n_train = finetune_data.create_manifest(
            train_dataset,
            train_manifest,
            architecture=config['architecture'],
            model_name=config['base_model'],
            windows_per_song_multiplier=windows_per_song,
            line_overlap_threshold=line_overlap_threshold,
            dedup_max_run=config.get("dedup_max_run", 0),
            max_unlabeled_vocal_frac=config.get("max_unlabeled_vocal_frac"),
            skip_song_ids=skip_song_ids,
            use_alignment=not no_mfa_alignment,
        )
        logger.info(f"Training: {n_train} chunks (synced-line segments)")

        if val_dir:
            val_dataset = finetune_data.LyricsDataset(
                dataset_dir=val_dir,
                filenames=config['filenames'],
            )
            val_manifest = job_dir / "val_manifest.jsonl"
            n_val = finetune_data.create_manifest(
                val_dataset,
                val_manifest,
                architecture=config['architecture'],
                model_name=config['base_model'],
                windows_per_song_multiplier=1,
                line_overlap_threshold=line_overlap_threshold,
                dedup_max_run=config.get("dedup_max_run", 0),
                max_unlabeled_vocal_frac=config.get("max_unlabeled_vocal_frac"),
                skip_song_ids=skip_song_ids,
                use_alignment=not no_mfa_alignment,
            )
            logger.info(f"Validation: {n_val} chunks")

            val_subset_manifest = job_dir / "val_subset_manifest.jsonl"
            n_subset = finetune_data.write_subset_manifest(
                val_manifest, val_subset_manifest, size=config["eval_subset_size"], seed=42,
            )
            logger.info(f"Validation subset (scored at every export): {n_subset} chunks")

    if audio_domain_mode:
        config["audio_domain_windows"] = finetune_domains.apply_audio_domains(
            job_dir, config["filenames"], audio_domain_mode,
        )

    with open(train_manifest) as f:
        rows = [line for line in f if line.strip()]
    if config.get("max_train_windows") and config["max_train_windows"] < len(rows):
        rows = random.Random(config["train_window_seed"]).sample(rows, config["max_train_windows"])
        shutil.copy2(train_manifest, job_dir / "train_manifest_full.jsonl")
        with open(train_manifest, "w") as f:
            f.writelines(rows)
        logger.info(f"Training windows: {len(rows)} (max_train_windows={config['max_train_windows']})")

    try:
        resolve_steps(config, len(rows))
    except ValueError as e:
        logger.error(str(e))
        return
    finetune_jobs.setup_finetune_job(config)
    logger.info(
        f"Job ready at {job_dir}: {config['max_steps']} steps ({config['steps_per_epoch']} per epoch), "
        f"{config['schedule']} decaying from {config['decay_start']}, kept {config['keep_steps']}, "
        f"exports at {config['export_steps'] + ([config['max_steps']] if config['export_final'] else [])}"
    )


@finetune_app.command("run")
def finetune_run(
    job_dir: Path = typer.Option(..., "--job-dir", help="Path to job directory"),
):
    """
    Train a job to its final step, resuming from its latest checkpoint. Rerun it to
    continue after a time limit.
    """
    from lyricscribe.finetune import engine
    from lyricscribe.finetune import trainer as finetune_trainer

    if not (job_dir / "config.json").exists():
        logger.error(f"No config found at {job_dir / 'config.json'}")
        return
    config = load_finetune_config(job_dir)
    if finetune_jobs.load_job_status(job_dir)["status"] == "complete":
        logger.info(f"{job_dir} is already complete")
        return
    rank0 = engine.is_main_process()
    if rank0:
        finetune_jobs.set_status(job_dir, "running")
    try:
        result = finetune_trainer.run_job(config, job_dir)
    except Exception as e:
        logger.error(f"Training failed: {e}")
        if rank0:
            finetune_jobs.set_status(job_dir, "failed")
        raise
    if rank0:
        finetune_jobs.set_status(job_dir, "complete" if result["complete"] else "pending", result["global_step"])
    logger.info(f"At step {result['global_step']} of {config['max_steps']} (complete={result['complete']})")


@finetune_app.command("prepare-manifests")
def finetune_prepare_manifests(
    source: Path = typer.Argument(..., help="Experiment (or prepared) dir holding the manifests to copy."),
    output_dir: Path = typer.Option(..., "--output-dir", help="Where to write the prepared manifest set."),
    exclude_songs: Path = typer.Option(
        None, "--exclude-songs", help="File of song ids (one per line) removed from training."
    ),
    repoint: str = typer.Option(None, "--repoint", help="Audio filename replacing each row's file name."),
    whisper_segments: bool = typer.Option(
        False, "--whisper-segments", help="Rebuild Whisper timestamp segments from each song's lyrics."
    ),
    whisper_model: str = typer.Option(
        None, "--whisper-cap-model", help="Whisper tokenizer for dropping windows over its label cap (all families)."
    ),
    eval_exclude_songs: Path = typer.Option(
        None, "--eval-exclude-songs", help="Validation song ids the eval-loss subset must avoid (e.g. tune300)."
    ),
    eval_subset_size: int = typer.Option(200, "--eval-subset-size", min=1),
    eval_seed: int = typer.Option(42, "--eval-seed"),
    recipe: list[str] = typer.Option(
        None, "--recipe",
        help="key=value data key the source config doesn't record (JSON value), e.g. use_alignment=false.",
    ),
):
    """
    Prepare one family-agnostic manifest set.
    """
    from lyricscribe.finetune import data as finetune_data

    def read_ids(path: Path | None) -> set[str] | None:
        """
        Song ids listed one per line.

        :param path: File, or ``None``.
        :return: The ids, or ``None``.
        """
        return {line.strip() for line in path.read_text().splitlines() if line.strip()} if path else None

    report = finetune_data.prepare_manifests(
        source,
        output_dir,
        exclude_songs=read_ids(exclude_songs),
        repoint=repoint,
        whisper_segments=whisper_segments,
        whisper_model=whisper_model,
        eval_exclude_songs=read_ids(eval_exclude_songs),
        eval_subset_size=eval_subset_size,
        eval_seed=eval_seed,
        recipe={k: json.loads(v) for k, v in (item.split("=", 1) for item in recipe or [])},
    )
    typer.echo(json.dumps(report, indent=2))


@finetune_app.command("extend")
def finetune_extend(
    job_dir: Path = typer.Option(..., "--job-dir", help="Constant-LR job dir (setup with --cooldown 0)."),
    to_step: int = typer.Option(..., "--to", help="New final step."),
    keep: list[int] = typer.Option(None, "--keep", help="Extra kept checkpoint step (repeatable)."),
):
    """
    Continue a constant-LR run to a later final step.
    """
    from lyricscribe.finetune import engine

    config = engine.extend_trunk(job_dir, to_step, keep)
    finetune_jobs.write_status(job_dir, config, "pending", finetune_jobs.load_job_status(job_dir)["global_step"])
    typer.echo(f"Runs to step {config['max_steps']}, keeping {config['keep_steps']}")


@finetune_app.command("branch")
def finetune_branch(
    job_dir: Path = typer.Option(..., "--job-dir", help="Constant-LR job dir to branch from."),
    name: str = typer.Option(..., "--name", help="Branch name (a job dir under <job-dir>/branches)."),
    to_step: int = typer.Option(None, "--to", help="Final step of the branch."),
    epoch: float = typer.Option(None, "--epoch", help="End the branch after this many epochs (instead of --to)."),
    at_step: int = typer.Option(None, "--at", help="Kept checkpoint to branch from (default: --cooldown before the end)."),
    cooldown: float = typer.Option(0.2, "--cooldown", help="Fraction of the branch's end step spent cooling down."),
):
    """
    Branch a WSD cooldown off a kept checkpoint; the branch is a job of its own (train it with
    ``finetune run``) that exports at its end.
    """
    from lyricscribe.finetune import engine

    config = load_finetune_config(job_dir)
    if epoch is not None:
        to_step = engine.epochs_to_steps(epoch, config["steps_per_epoch"])
    if to_step is None:
        logger.error("give --to or --epoch")
        raise typer.Exit(1)
    at = at_step if at_step is not None else engine.cooldown_start(to_step, cooldown)
    typer.echo(str(engine.create_branch(job_dir, at, to_step, name)))


@finetune_app.command("throughput")
def finetune_throughput(
    model: str = typer.Option(..., "--model", help="Base model."),
    manifest: Path = typer.Option(..., "--manifest", help="Training manifest."),
    output_dir: Path = typer.Option(..., "--output-dir", help="Scratch dir for the probe."),
    batch_size: int = typer.Option(None, "--batch-size", help="Per-device windows (default: arch default)."),
    global_batch: int = typer.Option(None, "--global-batch", help="Windows per optimizer step (default: 64)."),
    steps: int = typer.Option(30, "--steps", min=10, help="Optimizer steps (the first 5 are not timed)."),
    adaptation: str = typer.Option(None, "--adaptation", help="full or nvidia_adapter (default: the model's)."),
):
    """
    Time training steps with the full training setup (optimizer, augmentation, precision).
    """
    import torch

    from lyricscribe.finetune import engine
    from lyricscribe.finetune import trainer as finetune_trainer

    config = create_finetune_config(
        base_model=model, train_dir=manifest.parent, output_dir=output_dir, filenames=["probe"],
        exp_name="throughput", batch_size=batch_size, global_batch=global_batch, max_steps=steps, adaptation=adaptation,
    )
    with open(manifest) as f:
        n_windows = sum(1 for line in f if line.strip())
    resolve_steps(config, n_windows)
    config["train_manifest"] = str(manifest)
    job_dir = output_dir / "throughput"
    job_dir.mkdir(parents=True, exist_ok=True)
    result = finetune_trainer.run_job(config, job_dir, save=False)
    sps = result["seconds_per_step"]
    gpus = engine.world_size()
    per_epoch = engine.steps_per_epoch(n_windows, config["global_batch"])
    summary = {
        "model": model,
        "per_device": config["batch_size"],
        "global_batch": config["global_batch"],
        "gpus": gpus,
        "seconds_per_step": sps,
        "steps_per_epoch": per_epoch,
        "gpu_hours_per_epoch": None if sps is None else sps * per_epoch * gpus / 3600,
        "peak_memory_gb": torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else None,
    }
    if engine.is_main_process():
        (job_dir / "throughput.json").write_text(json.dumps(summary, indent=2))
    typer.echo(json.dumps(summary, indent=2))


@finetune_app.command("inspect")
def finetune_inspect(
    job_dir: Path = typer.Option(..., "--job-dir", help="Path to job directory"),
):
    """
    Show a job's progress, exports and metrics.
    """
    if not (job_dir / "status.json").exists():
        logger.error(f"No job found at {job_dir}")
        return
    finetune_jobs.show_job_stats(job_dir)


@finetune_app.command("reset")
def finetune_reset(
    job_dir: Path = typer.Option(..., "--job-dir", help="Path to job directory"),
    force: bool = typer.Option(False, "--force", help="Confirm deleting the job's checkpoints, exports and state."),
):
    """
    Reset a finetuning job to start from scratch (refused while branches link to its checkpoints).
    """
    if not (job_dir / "status.json").exists():
        logger.error(f"No job found at {job_dir}")
        return
    if not force:
        logger.error(f"reset deletes {job_dir}'s checkpoints, exports and training state; pass --force to confirm")
        return
    try:
        finetune_jobs.reset_job(job_dir)
    except RuntimeError as e:
        logger.error(str(e))
        return
    logger.info(f"Run training again with: lyricscribe finetune run --job-dir {job_dir}")


@finetune_app.command("retry")
def finetune_retry(
    job_dir: Path = typer.Option(..., "--job-dir", help="Path to job directory"),
):
    """
    Mark a failed job pending again; its checkpoints are kept, so the next run resumes.
    """
    if not (job_dir / "status.json").exists():
        logger.error(f"No job found at {job_dir}")
        return
    if finetune_jobs.load_job_status(job_dir)["status"] != "failed":
        logger.info(f"{job_dir.name} has not failed")
        return
    finetune_jobs.set_status(job_dir, "pending")
    logger.info(f"{job_dir.name} reset to pending")


@finetune_app.command("retry-all")
def finetune_retry_all(
    experiments_dir: Path = typer.Option(..., "--experiments-dir", help="Directory containing experiment subdirectories"),
):
    """
    Mark every failed job in a directory pending again.
    """
    if not experiments_dir.exists():
        logger.error(f"Directory not found: {experiments_dir}")
        return
    n = 0
    for job_dir in sorted(p for p in experiments_dir.iterdir() if (p / "status.json").exists()):
        if finetune_jobs.load_job_status(job_dir)["status"] == "failed":
            finetune_jobs.set_status(job_dir, "pending")
            logger.info(f"  {job_dir.name}: reset")
            n += 1
    logger.info(f"Reset {n} failed jobs")


@finetune_app.command("status")
def finetune_status(
    experiments_dir: Path = typer.Option(..., "--experiments-dir", help="Directory containing experiment subdirectories"),
):
    """
    Print a one-line summary of every job in a directory.
    """
    if not experiments_dir.exists():
        logger.error(f"Directory not found: {experiments_dir}")
        return
    rows = [finetune_jobs.gather_experiment_status(p) for p in sorted(experiments_dir.iterdir())
            if (p / "status.json").exists()]
    if not rows:
        logger.info(f"No experiments found in {experiments_dir}")
        return
    width = max(len(r["exp_name"]) for r in rows)
    for r in rows:
        metrics = r["last_metrics"] or {}
        loss = next((f" {k}={metrics[k]:.3f}" for k in ("eval_loss", "train_loss") if k in metrics), "")
        logger.info(f"  {r['exp_name']:<{width}}  step {r['global_step']:>6}/{r['max_steps']:<6}  {r['status']:<8}  "
                    f"export={r['last_export'] or 'none'}{loss}")


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
        1,
        "--start",
        help="Smallest batch_size to try. Values outside the candidate "
             "list are filtered out.",
    ),
    max_bs: int = typer.Option(
        128,
        "--max",
        help="Largest batch_size to try.",
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
    output_path: Path = typer.Option(..., "--output", help="Path to copy the export to"),
    step: int = typer.Option(None, "--step", help="Export step to copy (default: latest)"),
):
    """
    Copy one of a job's inference exports (``exports/step_<N>``) elsewhere.

    Use 'lyricscribe finetune inspect' to see the available exports and metrics.
    """
    exports = dict(list_exports(job_dir))
    if not exports:
        logger.error(f"{job_dir} has no exports yet")
        return
    if step is None:
        step = max(exports)
    if step not in exports:
        logger.error(f"No export at step {step}; available: {sorted(exports)}")
        return
    if output_path.exists():
        logger.error(f"Output path already exists: {output_path}")
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(exports[step], output_path)
    logger.info(f"Model exported: {exports[step]} -> {output_path}")
    logger.info(f"Use with: lyricscribe transcribe setup --model {output_path}")


@finetune_app.command("eval-loss")
def finetune_eval_loss(
    base_model: str = typer.Option(
        ..., "--base-model",
        help="Base hub id/path (e.g. Qwen/Qwen3-ASR-1.7B-hf, Qwen/Qwen3-Omni-30B-A3B-Instruct)",
    ),
    checkpoint: list[Path] = typer.Option(
        ..., "--checkpoint",
        help="Checkpoint dir to compare (repeatable): a full model for "
             "Qwen3-ASR, a LoRA adapter for Qwen3-Omni",
    ),
    manifest: Path = typer.Option(
        ..., "--manifest", help="Manifest to sample held-out windows from"
    ),
    rows: int = typer.Option(32, "--rows", min=1, help="Windows to evaluate"),
    seed: int = typer.Option(1234, "--seed", help="Window sampling seed"),
    batch_size: int = typer.Option(2, "--batch-size", min=1),
    output: Path = typer.Option(None, "--output", help="Write results JSON here"),
):
    """
    Held-out mean per-token CE of a base Qwen model vs finetuned checkpoints.

    Measured in eval mode on the same seeded windows for every model. The
    Trainer's logged loss is accumulation-scaled and from train mode, so it is
    not comparable.
    """
    from lyricscribe.finetune.qwen import eval_loss

    architecture = detect_architecture(base_model)
    results = eval_loss(
        architecture, base_model, [str(c) for c in checkpoint], manifest,
        rows=rows, seed=seed, batch_size=batch_size,
    )
    base = results["base"]
    for name, ce in results.items():
        rel = "" if name == "base" else f"  ({100 * (ce - base) / base:+.1f}% vs base)"
        logger.info(f"{name}: CE {ce:.4f}{rel}")
    if output:
        output.write_text(json.dumps({"rows": rows, "seed": seed, "manifest": str(manifest),
                                      "ce": results}, indent=2) + "\n")


if __name__ == "__main__":
    cli()
