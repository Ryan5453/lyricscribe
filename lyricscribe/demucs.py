import json
import logging
import random
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import torch
from demucs.api import Separator

from lyricscribe.jobs import update_status

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
        separator = Separator(model=model_name, only_load=stem, device="cpu")
    else:
        separator = Separator(model=model_name, device="cpu")
    logger.info(
        f"Loaded model {model_name} on {separator.device}, "
        f"isolating {stem or 'all stems'}"
    )
    return separator


def _get_expected_output_paths(
    model_name: str,
    stem: str | None,
    output_path: str,
    model: Separator | None = None,
) -> list[Path]:
    """
    Return the output file paths expected for one separation entry.

    :param model_name: Name of the Demucs model used for filenames.
    :param stem: Stem to isolate, or ``None`` for all stems.
    :param output_path: Stored manifest output path.
    :param model: Loaded separator, required when ``stem`` is ``None``.
    :return: Expected output file paths.
    """
    output = Path(output_path)
    if stem:
        return [output]

    if model is None:
        model = _load_model(model_name, None)

    return [output / f"{model_name}_{stem_name}.wav" for stem_name in model.model.sources]


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
    expected_outputs = _get_expected_output_paths(
        model_name=model_name,
        stem=stem,
        output_path=output_path,
        model=model,
    )
    if expected_outputs and all(path.exists() for path in expected_outputs):
        update_status(job_dir, chunk_id, chunk_data, name, "success", 0.0, None)
        return "skipped"

    start_time = time.time()

    try:
        separated = model.separate(input_path)

        if stem:
            separated.export_stem(stem, output_path)
        else:
            output_dir = Path(output_path)
            for stem_name in separated.sources:
                stem_output = output_dir / f"{model_name}_{stem_name}.wav"
                separated.export_stem(stem_name, str(stem_output))

        duration = time.time() - start_time
        update_status(job_dir, chunk_id, chunk_data, name, "success", duration, None)
        logger.info(f"{name} completed in {duration:.2f}s")
        return "success"

    except Exception as e:
        duration = time.time() - start_time
        update_status(job_dir, chunk_id, chunk_data, name, "failed", duration, str(e))
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
                output_path = subdir

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
        chunk_path = job_dir / f"chunk_{chunk_id}.json"
        with open(chunk_path, "w") as f:
            json.dump(chunk_data, f, indent=2)

    logger.info(f"Registered {total} files into {num_chunks} chunks in {job_dir}")


def process_chunk(job_dir: Path, chunk_id: int) -> None:
    """
    Process one chunk of the separation job.

    Loads the model and configuration, then processes all pending files
    assigned to the given chunk.

    :param job_dir: Path to the job directory.
    :param chunk_id: 1-based chunk identifier.
    """
    with open(job_dir / "config.json") as f:
        config = json.load(f)
    model_name = config["model"]
    stem = config.get("stem")
    model = _load_model(model_name, stem)

    with open(job_dir / f"chunk_{chunk_id}.json") as f:
        chunk_data = json.load(f)
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


def reset_job(job_dir: Path) -> None:
    """
    Reset a separation job so it can be re-run from scratch.

    Deletes any existing output files tracked by the job config and resets all
    chunk entries to ``pending``.

    :param job_dir: Path to the job directory.
    """
    with open(job_dir / "config.json") as f:
        config = json.load(f)

    model_name = config["model"]
    stem = config.get("stem")
    model = None if stem else _load_model(model_name, None)

    deleted_outputs = 0
    reset_count = 0

    for chunk_path in sorted(job_dir.glob("chunk_*.json")):
        with open(chunk_path) as f:
            chunk_data = json.load(f)

        for entry in chunk_data["files"]:
            for path in _get_expected_output_paths(
                model_name=model_name,
                stem=stem,
                output_path=entry["output_path"],
                model=model,
            ):
                if path.exists():
                    path.unlink()
                    deleted_outputs += 1

            if entry["status"] != "pending":
                reset_count += 1

            entry["status"] = "pending"
            entry["duration_seconds"] = None
            entry["error_message"] = None
            entry["processed_at"] = None

        with open(chunk_path, "w") as f:
            json.dump(chunk_data, f, indent=2)

    logger.info(f"Deleted {deleted_outputs} output file(s)")
    logger.info(f"Reset {reset_count} entries to pending")
