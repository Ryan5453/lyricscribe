import json
import logging
import random
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import torch
from unblend import ModelRepository, Separator

from lyricscribe.jobs import update_status

logger = logging.getLogger(__name__)

# ``shifts`` > 0 draws random time offsets, so the seed is pinned.
DEFAULT_SHIFTS = 1
DEFAULT_SPLIT_OVERLAP = 0.25
DEFAULT_SEED = 0


def _load_model(
    model_name: str, stem: str | None, dtype: str = "auto"
) -> Separator:
    """
    Load an Unblend model with optional single-stem isolation.

    Any registered Unblend model works, including custom ones added via
    ``unblend models import``.

    :param model_name: Unblend model name (see ``unblend models list``).
    :param stem: Stem to isolate, or ``None`` for all stems.
    :param dtype: Compute precision: ``"auto"`` (FP16 on CUDA), ``"float32"``,
        ``"float16"`` or ``"bfloat16"``.
    :return: A loaded :class:`Separator` instance.
    """
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    torch_dtype = "auto" if dtype == "auto" else getattr(torch, dtype)
    separator = Separator(model=model_name, only_load=stem, dtype=torch_dtype)
    logger.info(
        f"Loaded model {model_name} on {separator.device} ({separator.dtype}), "
        f"isolating {stem or 'all stems'}"
    )
    return separator


def _model_sources(model_name: str) -> list[str]:
    """
    Stem names a registered Unblend model produces, without loading weights.

    :param model_name: Unblend model name.
    :return: Stem names in model order.
    """
    return list(ModelRepository().list_models()[model_name]["sources"])


def _get_expected_output_paths(
    model_name: str,
    stem: str | None,
    output_path: str,
) -> list[Path]:
    """
    Return the output file paths expected for one separation entry.

    :param model_name: Name of the model used for filenames.
    :param stem: Stem to isolate, or ``None`` for all stems.
    :param output_path: Stored manifest output path.
    :return: Expected output file paths.
    """
    output = Path(output_path)
    if stem:
        return [output]
    return [output / f"{model_name}_{s}.wav" for s in _model_sources(model_name)]


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
    separate_kwargs: dict,
) -> str:
    """
    Separate a single audio file and update its status in the chunk manifest.

    If the output path already exists, the file is marked as skipped.

    :param job_dir: Path to the job directory.
    :param chunk_id: 1-based chunk identifier.
    :param chunk_data: In-memory chunk data (mutated in place).
    :param model: Loaded Unblend :class:`Separator` instance.
    :param model_name: Name of the model, used for output filenames.
    :param stem: Stem to isolate, or ``None`` for all stems.
    :param name: Identifier for this entry.
    :param input_path: Path to the input audio file.
    :param output_path: Path for the output file or directory.
    :param separate_kwargs: ``shifts`` / ``split_overlap`` / ``seed`` for
        :meth:`Separator.separate`.
    :return: One of ``'success'``, ``'failed'``, or ``'skipped'``.
    """
    expected_outputs = _get_expected_output_paths(
        model_name=model_name,
        stem=stem,
        output_path=output_path,
    )
    if expected_outputs and all(path.exists() for path in expected_outputs):
        update_status(job_dir, chunk_id, chunk_data, name, "success", 0.0, None)
        return "skipped"

    start_time = time.time()

    try:
        separated = model.separate(input_path, **separate_kwargs)
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
    output_filename: str | None = None,
    shifts: int = DEFAULT_SHIFTS,
    split_overlap: float = DEFAULT_SPLIT_OVERLAP,
    seed: int = DEFAULT_SEED,
    dtype: str = "auto",
) -> None:
    """
    Initialize a separation job by scanning directories and writing
    a config file plus per-chunk JSON manifests.

    :param directories: Root directories containing subdirectories to process.
    :param job_dir: Directory to create for job files.
    :param filename: Audio filename to target within each subdirectory.
    :param model: Unblend model name.
    :param num_chunks: Number of chunks to split the work into.
    :param stem: Stem to isolate, or ``None`` for all stems.
    :param output_filename: Output filename for a single-stem job (default
        ``{model}_{stem}.wav``).
    :param shifts: Random time-shift passes to average (0 = deterministic).
    :param split_overlap: Overlap between consecutive model segments.
    :param seed: Seed for the shift offsets.
    :param dtype: Compute precision (see :func:`_load_model`).
    """
    if output_filename and not stem:
        raise ValueError("--output-filename requires --stem")
    if model not in ModelRepository().list_models():
        raise ValueError(
            f"Unknown Unblend model {model!r}; see `unblend models list` "
            "(custom checkpoints are added with `unblend models import`)"
        )

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

    config = {
        "directories": [str(d) for d in directories],
        "filename": filename,
        "model": model,
        "stem": stem,
        "shifts": shifts,
        "split_overlap": split_overlap,
        "seed": seed,
        "dtype": dtype,
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
            if stem:
                output_path = subdir / (output_filename or f"{model}_{stem}.wav")
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
    model = _load_model(model_name, stem, config.get("dtype", "auto"))
    separate_kwargs = {
        "shifts": config.get("shifts", DEFAULT_SHIFTS),
        "split_overlap": config.get("split_overlap", DEFAULT_SPLIT_OVERLAP),
        "seed": config.get("seed", DEFAULT_SEED),
    }

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
            separate_kwargs,
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
