import json
import logging
import random
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import torch
from silero_vad import load_silero_vad
from torch.jit import ScriptModule

from lyricscribe.jobs import update_status
from lyricscribe.transcribe.base import Transcriber
from lyricscribe.transcribe.nemo import NemoTranscriber
from lyricscribe.transcribe.whisper import WhisperTranscriber

logger = logging.getLogger(__name__)


def _create_transcriber(model_name: str, batch_size: int) -> Transcriber:
    """
    Create the appropriate transcriber based on model name.

    Whisper models use HuggingFace Transformers, all others use NeMo.

    :param model_name: HuggingFace model identifier or path to local
        checkpoint directory.
    :param batch_size: Batch size for inference.
    :return: An unloaded :class:`Transcriber` instance.
    """
    if "whisper" in model_name.lower():
        return WhisperTranscriber(model_name, batch_size)

    # Local checkpoint dirs (e.g. finetune outputs) won't have "whisper" in
    # their path. Fall back to reading the HF ``config.json`` and routing
    # based on the architecture string so finetuned Whisper checkpoints
    # don't get mis-dispatched to the NeMo loader.
    ckpt_path = Path(model_name)
    if ckpt_path.is_dir():
        cfg_path = ckpt_path / "config.json"
        if cfg_path.exists():
            try:
                with open(cfg_path) as f:
                    cfg = json.load(f)
                arch = (cfg.get("architectures") or [""])[0].lower()
                if "whisper" in arch:
                    return WhisperTranscriber(model_name, batch_size)
            except Exception:
                pass

    return NemoTranscriber(model_name, batch_size)


def _append_result(
    output_path: Path,
    song_id: str,
    audio_file: str,
    transcription: str | None,
    model_name: str,
    duration_seconds: float,
    error: str | None,
) -> None:
    """
    Append a single transcription result as a JSON line to the output file.

    Uses append mode for atomic writes on networked filesystems.

    :param output_path: Path to the JSONL output file.
    :param song_id: Identifier for the song.
    :param audio_file: Absolute path to the audio file.
    :param transcription: Transcribed text, or ``None`` on failure.
    :param model_name: Model identifier used for transcription.
    :param duration_seconds: Time taken for transcription.
    :param error: Error message if failed, or ``None``.
    """
    result = {
        "song_id": song_id,
        "audio_file": audio_file,
        "transcription": transcription,
        "model_name": model_name,
        "duration_seconds": round(duration_seconds, 2),
        "error": error,
    }
    with open(output_path, "a") as f:
        f.write(json.dumps(result) + "\n")


def setup_job(
    directories: list[Path],
    job_dir: Path,
    filename: str,
    model: str,
    num_chunks: int,
    batch_size: int,
    vad: bool,
    chunked: bool = False,
    lyrics_filename: str | None = None,
    vad_filename: str | None = None,
    vad_method: str = "silero",
    repetition_penalty: float = 1.0,
    no_repeat_ngram_size: int = 0,
) -> None:
    """
    Initialize a transcription job by scanning directories and writing
    a config file plus per-chunk JSON manifests.

    :param directories: Root directories containing subdirectories to process.
    :param job_dir: Directory to create for job files.
    :param filename: Audio filename to target within each subdirectory.
    :param model: HuggingFace model identifier.
    :param num_chunks: Number of chunks to split the work into.
    :param batch_size: Batch size for inference.
    :param vad: Whether to enable VAD-based segmentation.
    :param chunked: Whether to use fixed-length chunked inference.
    :param lyrics_filename: Optional JSON filename (e.g. ``lyrics.json``)
        to read ``detected_language`` from in each subdirectory.
    :param vad_filename: Optional audio filename to use as VAD source
        (timestamps from this file, transcription from ``filename``).
    :param repetition_penalty: Whisper decode-time anti-stutter knob.
        ``1.0`` disables (vanilla decode).
    :param no_repeat_ngram_size: Whisper decode-time anti-loop knob.
        ``0`` disables.
    """
    subdirs = []
    for directory in directories:
        subdirs.extend(d for d in directory.iterdir() if d.is_dir())
    total = len(subdirs)

    if num_chunks < 1:
        raise ValueError(f"num_chunks must be >= 1, got {num_chunks}")

    if total == 0:
        logger.warning("No subdirectories found")
        return

    random.seed(42)
    random.shuffle(subdirs)

    job_dir.mkdir(parents=True, exist_ok=True)

    # Clean up old results when running setup to avoid duplicate appends during run.
    results_path = job_dir / "results.jsonl"
    if results_path.exists():
        results_path.unlink()

    config = {
        "directories": [str(d) for d in directories],
        "filename": filename,
        "model": model,
        "batch_size": batch_size,
        "vad": vad,
        "vad_method": vad_method,
        "chunked": chunked,
        "lyrics_filename": lyrics_filename,
        "vad_filename": vad_filename,
        "repetition_penalty": repetition_penalty,
        "no_repeat_ngram_size": no_repeat_ngram_size,
        "num_chunks": num_chunks,
        "total_files": total,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }

    with open(job_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    for chunk_id in range(1, num_chunks + 1):
        files = []
        for idx in range(chunk_id - 1, total, num_chunks):
            subdir = subdirs[idx]
            name = subdir.name
            input_file = subdir / filename

            language = None
            if lyrics_filename:
                lyrics_path = subdir / lyrics_filename
                if lyrics_path.exists():
                    try:
                        with open(lyrics_path) as lf:
                            lyrics_data = json.load(lf)
                        language = lyrics_data.get("detected_language")
                    except (json.JSONDecodeError, OSError) as e:
                        logger.warning(
                            f"{name}: failed to read {lyrics_filename}: {e}"
                        )

            vad_path = None
            if vad_filename:
                vad_file = subdir / vad_filename
                if vad_file.exists():
                    vad_path = str(vad_file)

            files.append(
                {
                    "name": name,
                    "input_path": str(input_file),
                    "vad_path": vad_path,
                    "language": language,
                    "status": "pending",
                    "duration_seconds": None,
                    "error_message": None,
                    "processed_at": None,
                }
            )

        chunk_data = {"chunk_id": chunk_id, "files": files}
        chunk_path = job_dir / f"chunk_{chunk_id}.json"
        with open(chunk_path, "w") as f:
            json.dump(chunk_data, f, indent=2)

    logger.info(f"Registered {total} files into {num_chunks} chunks in {job_dir}")


def _transcribe_with_oom_retry(
    transcriber: Transcriber,
    input_path: str,
    use_vad: bool,
    use_chunked: bool,
    vad_model: ScriptModule | None,
    language: str | None = None,
    vad_source: str | None = None,
    vad_method: str = "silero",
    repetition_penalty: float = 1.0,
    no_repeat_ngram_size: int = 0,
) -> str:
    """
    Attempt transcription with automatic OOM recovery.

    On ``torch.cuda.OutOfMemoryError``, halves the batch size, clears
    the CUDA cache, and retries. Keeps retrying until batch_size reaches
    1, at which point the OOM is re-raised.

    :param transcriber: Loaded transcriber instance.
    :param input_path: Path to the audio file.
    :param use_vad: Whether to use VAD-based transcription.
    :param use_chunked: Whether to use chunked inference.
    :param vad_model: Loaded Silero VAD model, or ``None``.
    :param language: Optional language code.
    :param vad_source: Optional path to VAD source audio file.
    :return: Transcribed text.
    :raises torch.cuda.OutOfMemoryError: If OOM persists at batch_size=1.
    """
    while True:
        try:
            return transcriber.transcribe(
                input_path,
                use_vad=use_vad,
                vad_model=vad_model,
                use_chunked=use_chunked,
                language=language,
                vad_source=vad_source,
                vad_method=vad_method,
                repetition_penalty=repetition_penalty,
                no_repeat_ngram_size=no_repeat_ngram_size,
            )
        except torch.cuda.OutOfMemoryError:
            if transcriber.batch_size <= 1:
                raise
            transcriber.halve_batch_size()


def process_chunk(job_dir: Path, chunk_id: int) -> None:
    """
    Process one chunk of the transcription job.

    Loads the appropriate model and transcribes all pending files
    assigned to the given chunk. Results are appended to a JSONL file
    in the job directory.

    On ``torch.cuda.OutOfMemoryError`` during transcription, the
    batch size is halved and the file is retried automatically.

    :param job_dir: Path to the job directory.
    :param chunk_id: 1-based chunk identifier.
    """
    with open(job_dir / "config.json") as f:
        config = json.load(f)
    model_name = config["model"]
    batch_size = config.get("batch_size", 1)
    if batch_size == 0:
        batch_size = 1  # Legacy: treat old "auto" (0) as default (1)
    use_vad = config.get("vad", False)
    use_chunked = config.get("chunked", False)
    vad_method = config.get("vad_method", "silero")
    vad_filename = config.get("vad_filename")
    repetition_penalty = float(config.get("repetition_penalty", 1.0) or 1.0)
    no_repeat_ngram_size = int(config.get("no_repeat_ngram_size", 0) or 0)

    if use_vad and vad_method == "rms" and not vad_filename:
        raise ValueError(
            "vad_method='rms' requires vad_filename to point at a separated "
            "vocal track or stem; RMS amplitude on a mixed track is dominated "
            "by accompaniment."
        )

    transcriber = _create_transcriber(model_name, batch_size)
    transcriber.load()

    logger.info(f"Using batch_size={transcriber.batch_size}")

    vad_model = None
    if use_vad and vad_method == "silero":
        vad_model = load_silero_vad()
        logger.info("Loaded Silero VAD model")
    elif use_vad and vad_method == "rms":
        logger.info("Using RMS-amplitude VAD on the (separated) vocal track")

    with open(job_dir / f"chunk_{chunk_id}.json") as f:
        chunk_data = json.load(f)
    pending = [f for f in chunk_data["files"] if f["status"] == "pending"]

    if not pending:
        logger.info(f"Chunk {chunk_id}: no pending files")
        return

    output_path = job_dir / "results.jsonl"

    success_count = 0
    failed_count = 0

    for entry in pending:
        name = entry["name"]
        input_path = entry["input_path"]

        if not Path(input_path).exists():
            logger.warning(f"{name}: input file not found at {input_path}")
            update_status(
                job_dir,
                chunk_id,
                chunk_data,
                name,
                "failed",
                0.0,
                f"File not found: {input_path}",
            )
            failed_count += 1
            continue

        start_time = time.time()

        try:
            text = _transcribe_with_oom_retry(
                transcriber,
                input_path,
                use_vad,
                use_chunked,
                vad_model,
                repetition_penalty=repetition_penalty,
                no_repeat_ngram_size=no_repeat_ngram_size,
                language=entry.get("language"),
                vad_source=entry.get("vad_path"),
                vad_method=vad_method,
            )

            duration = time.time() - start_time

            _append_result(
                output_path, name, input_path, text, model_name, duration, None
            )
            update_status(
                job_dir, chunk_id, chunk_data, name, "success", duration, None
            )
            logger.info(f"{name} completed in {duration:.2f}s")
            success_count += 1

        except Exception as e:
            duration = time.time() - start_time
            error_msg = str(e)

            _append_result(
                output_path,
                name,
                input_path,
                None,
                model_name,
                duration,
                error_msg,
            )
            update_status(
                job_dir, chunk_id, chunk_data, name, "failed", duration, error_msg
            )
            logger.error(f"{name} failed: {e}")
            traceback.print_exc()
            failed_count += 1

    logger.info(f"Chunk {chunk_id}: {success_count} success, {failed_count} failed")
