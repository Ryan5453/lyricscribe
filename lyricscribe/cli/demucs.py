import json
import logging
import random
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import torch
from demucs.api import Separator

logger = logging.getLogger(__name__)


def _load_model(model_name: str, stem: str | None) -> Separator:
    """
    Load a Demucs model with optional single-stem isolation.

    :param model_name: Name of the Demucs model to load.
    :param stem: Stem to isolate, or ``None`` for all stems.
    :return: A loaded :class:`Separator` instance.
    """
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    if stem:
        separator = Separator(model=model_name, only_load=stem)
    else:
        separator = Separator(model=model_name)
    logger.info(
        f"Loaded model {model_name} on {separator.device}, "
        f"isolating {stem or 'all stems'}"
    )
    return separator


def _load_config(job_dir: Path) -> dict:
    """
    Load job configuration from the config file.

    :param job_dir: Path to the job directory.
    :return: Configuration dictionary.
    """
    config_path = job_dir / "config.json"
    with open(config_path) as f:
        return json.load(f)


def _load_chunk(job_dir: Path, chunk_id: int) -> dict:
    """
    Load a chunk manifest from disk.

    :param job_dir: Path to the job directory.
    :param chunk_id: 1-based chunk identifier.
    :return: Chunk dictionary with ``chunk_id`` and ``files`` list.
    """
    chunk_path = job_dir / f"chunk_{chunk_id}.json"
    with open(chunk_path) as f:
        return json.load(f)


def _save_chunk(job_dir: Path, chunk_id: int, chunk_data: dict) -> None:
    """
    Write a chunk manifest back to disk.

    :param job_dir: Path to the job directory.
    :param chunk_id: 1-based chunk identifier.
    :param chunk_data: Chunk dictionary to serialize.
    """
    chunk_path = job_dir / f"chunk_{chunk_id}.json"
    with open(chunk_path, "w") as f:
        json.dump(chunk_data, f, indent=2)


def _update_status(
    job_dir: Path,
    chunk_id: int,
    chunk_data: dict,
    name: str,
    status: str,
    duration: float,
    error: str | None,
) -> None:
    """
    Update a file's status in its chunk manifest and write to disk.

    :param job_dir: Path to the job directory.
    :param chunk_id: 1-based chunk identifier.
    :param chunk_data: In-memory chunk data (mutated in place).
    :param name: Name of the entry to update.
    :param status: New status value (``'success'``, ``'failed'``).
    :param duration: Processing duration in seconds.
    :param error: Error message if failed, or ``None``.
    """
    for entry in chunk_data["files"]:
        if entry["name"] == name:
            entry["status"] = status
            entry["duration_seconds"] = round(duration, 2)
            entry["error_message"] = error
            entry["processed_at"] = datetime.now(timezone.utc).isoformat()
            break

    _save_chunk(job_dir, chunk_id, chunk_data)


def _process_file(
    job_dir: Path,
    chunk_id: int,
    chunk_data: dict,
    model: Separator,
    model_name: str,
    stem: str | None,
    name: str,
    input_path: str,
    output_path: str,
) -> str:
    """
    Separate a single audio file and update its status in the chunk manifest.

    If the output path already exists, the file is marked as skipped.

    :param job_dir: Path to the job directory.
    :param chunk_id: 1-based chunk identifier.
    :param chunk_data: In-memory chunk data (mutated in place).
    :param model: Loaded Demucs :class:`Separator` instance.
    :param model_name: Name of the model, used for output filenames.
    :param stem: Stem to isolate, or ``None`` for all stems.
    :param name: Identifier for this entry.
    :param input_path: Path to the input audio file.
    :param output_path: Path for the output file or directory.
    :return: One of ``'success'``, ``'failed'``, or ``'skipped'``.
    """
    if Path(output_path).exists():
        _update_status(job_dir, chunk_id, chunk_data, name, "success", 0.0, None)
        return "skipped"

    start_time = time.time()

    try:
        sources = model.separate(input_path)

        if stem:
            sources.export_stem(stem, output_path)
        else:
            output_dir = Path(output_path)
            output_dir.mkdir(parents=True, exist_ok=True)
            for stem_name in model._model.sources:
                stem_output = output_dir / f"{model_name}_{stem_name}.wav"
                sources.export_stem(stem_name, str(stem_output))

        duration = time.time() - start_time
        _update_status(job_dir, chunk_id, chunk_data, name, "success", duration, None)
        logger.info(f"{name} completed in {duration:.2f}s")
        return "success"

    except (torch.cuda.OutOfMemoryError, RuntimeError, OSError) as e:
        duration = time.time() - start_time
        _update_status(
            job_dir, chunk_id, chunk_data, name, "failed", duration, str(e)
        )
        logger.error(f"{name} failed: {e}")
        traceback.print_exc()
        return "failed"


def setup_job(
    directories: list[Path],
    job_dir: Path,
    filename: str,
    model: str,
    num_chunks: int,
    stem: str | None,
) -> None:
    """
    Initialize a separation job by scanning directories and writing
    a config file plus per-chunk JSON manifests.

    :param directories: Root directories containing subdirectories to process.
    :param job_dir: Directory to create for job files.
    :param filename: Audio filename to target within each subdirectory.
    :param model: Name of the Demucs model to use.
    :param num_chunks: Number of chunks to split the work into.
    :param stem: Stem to isolate, or ``None`` for all stems.
    """
    subdirs = []
    for directory in directories:
        subdirs.extend(d for d in directory.iterdir() if d.is_dir())
    total = len(subdirs)

    if total == 0:
        logger.warning("No subdirectories found")
        return

    random.seed(42)
    random.shuffle(subdirs)

    job_dir.mkdir(parents=True, exist_ok=True)

    config = {
        "directories": [str(d) for d in directories],
        "filename": filename,
        "model": model,
        "stem": stem,
        "num_chunks": num_chunks,
        "total_files": total,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }

    with open(job_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    chunk_size = total // num_chunks

    for chunk_id in range(1, num_chunks + 1):
        start = (chunk_id - 1) * chunk_size
        end = total if chunk_id == num_chunks else chunk_id * chunk_size

        files = []
        for subdir in subdirs[start:end]:
            name = subdir.name
            input_file = subdir / filename
            if stem:
                output_path = subdir / f"{model}_{stem}.wav"
            else:
                output_path = subdir / f"{model}_stems"

            files.append(
                {
                    "name": name,
                    "input_path": str(input_file),
                    "output_path": str(output_path),
                    "status": "pending",
                    "duration_seconds": None,
                    "error_message": None,
                    "processed_at": None,
                }
            )

        chunk_data = {"chunk_id": chunk_id, "files": files}
        _save_chunk(job_dir, chunk_id, chunk_data)

    logger.info(f"Registered {total} files into {num_chunks} chunks in {job_dir}")


def process_chunk(job_dir: Path, chunk_id: int) -> None:
    """
    Process one chunk of the separation job.

    Loads the model and configuration, then processes all pending files
    assigned to the given chunk.

    :param job_dir: Path to the job directory.
    :param chunk_id: 1-based chunk identifier.
    """
    config = _load_config(job_dir)
    model_name = config["model"]
    stem = config.get("stem")
    model = _load_model(model_name, stem)

    chunk_data = _load_chunk(job_dir, chunk_id)
    pending = [f for f in chunk_data["files"] if f["status"] == "pending"]

    if not pending:
        return

    success_count = 0
    failed_count = 0
    skipped_count = 0

    for entry in pending:
        result = _process_file(
            job_dir,
            chunk_id,
            chunk_data,
            model,
            model_name,
            stem,
            entry["name"],
            entry["input_path"],
            entry["output_path"],
        )
        if result == "success":
            success_count += 1
        elif result == "failed":
            failed_count += 1
        elif result == "skipped":
            skipped_count += 1

    logger.info(
        f"Chunk {chunk_id}: {success_count} success, {failed_count} failed, {skipped_count} skipped"
    )


def show_stats(job_dir: Path) -> None:
    """
    Display processing statistics by reading all chunk manifests.

    :param job_dir: Path to the job directory.
    """
    config = _load_config(job_dir)

    all_files = []
    for chunk_path in sorted(job_dir.glob("chunk_*.json")):
        with open(chunk_path) as f:
            chunk_data = json.load(f)
        all_files.extend(chunk_data["files"])

    total = len(all_files)

    if total == 0:
        logger.info("No files registered")
        return

    logger.info(f"Model: {config.get('model')}")
    logger.info(f"Stem: {config.get('stem') or 'all'}")
    logger.info(f"Directories: {', '.join(config.get('directories', []))}")

    counts: dict[str, int] = {}
    durations: dict[str, list[float]] = {}
    for entry in all_files:
        status = entry["status"]
        counts[status] = counts.get(status, 0) + 1
        if entry.get("duration_seconds") is not None:
            durations.setdefault(status, []).append(entry["duration_seconds"])

    for status in ["pending", "success", "failed"]:
        if status in counts:
            count = counts[status]
            pct = 100 * count / total
            avg = (
                sum(durations.get(status, [])) / len(durations[status])
                if durations.get(status)
                else 0
            )
            logger.info(f"  {status}: {count} ({pct:.1f}%) avg {avg:.1f}s")

    success_count = counts.get("success", 0)
    completion = 100 * success_count / total if total > 0 else 0
    logger.info(f"Completion: {completion:.1f}%")


def retry_failed(job_dir: Path) -> None:
    """
    Retry all failed separations across all chunks.

    :param job_dir: Path to the job directory.
    """
    config = _load_config(job_dir)
    model_name = config["model"]
    stem = config.get("stem")
    model = _load_model(model_name, stem)

    success_count = 0
    still_failed = 0

    for chunk_path in sorted(job_dir.glob("chunk_*.json")):
        with open(chunk_path) as f:
            chunk_data = json.load(f)

        chunk_id = chunk_data["chunk_id"]
        failed = [f for f in chunk_data["files"] if f["status"] == "failed"]

        if not failed:
            continue

        for entry in failed:
            result = _process_file(
                job_dir,
                chunk_id,
                chunk_data,
                model,
                model_name,
                stem,
                entry["name"],
                entry["input_path"],
                entry["output_path"],
            )
            if result == "success" or result == "skipped":
                success_count += 1
            elif result == "failed":
                still_failed += 1

    logger.info(f"Retry: {success_count} fixed, {still_failed} still failed")
