import logging
import random
import sqlite3
import time
import traceback
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
    if stem:
        logger.info(f"Loading model {model_name}, isolating {stem}")
        return Separator(model=model_name, isolate_stem=stem)
    else:
        logger.info(f"Loading model {model_name}, all stems")
        return Separator(model=model_name)


def _load_config(db_path: Path) -> dict[str, str]:
    """
    Load job configuration from database.

    :param db_path: Path to the SQLite database.
    :return: Dictionary mapping config keys to their string values.
    """
    with sqlite3.connect(db_path) as conn:
        cursor = conn.execute("SELECT key, value FROM job_config")
        return {row[0]: row[1] for row in cursor.fetchall()}


def _update_status(
    conn: sqlite3.Connection,
    name: str,
    status: str,
    duration: float,
    error: str | None,
) -> None:
    """
    Update separation status in database.

    :param conn: Active SQLite connection.
    :param name: Name of the entry to update.
    :param status: New status value (e.g. ``'success'``, ``'failed'``).
    :param duration: Processing duration in seconds.
    :param error: Error message if failed, or ``None``.
    """
    conn.execute(
        """UPDATE separations
        SET status = ?, duration_seconds = ?, error_message = ?, processed_at = CURRENT_TIMESTAMP
        WHERE name = ?""",
        (status, duration, error, name),
    )


def _process_file(
    conn: sqlite3.Connection,
    model: Separator,
    model_name: str,
    stem: str | None,
    name: str,
    input_path: str,
    output_path: str,
) -> str:
    """
    Separate a single audio file and update its status in the database.

    If the output path already exists, the file is marked as skipped.

    :param conn: Active SQLite connection.
    :param model: Loaded Demucs :class:`Separator` instance.
    :param model_name: Name of the model, used for output filenames.
    :param stem: Stem to isolate, or ``None`` for all stems.
    :param name: Identifier for this entry in the database.
    :param input_path: Path to the input audio file.
    :param output_path: Path for the output file or directory.
    :return: One of ``'success'``, ``'failed'``, or ``'skipped'``.
    """
    if Path(output_path).exists():
        _update_status(conn, name, "success", 0.0, None)
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
        _update_status(conn, name, "success", duration, None)
        logger.info(f"{name} completed in {duration:.2f}s")
        return "success"

    except (torch.cuda.OutOfMemoryError, RuntimeError, OSError) as e:
        duration = time.time() - start_time
        _update_status(conn, name, "failed", duration, str(e))
        logger.error(f"{name} failed: {e}")
        traceback.print_exc()
        return "failed"


def setup_job(
    directory: Path,
    db_path: Path,
    filename: str,
    model: str,
    num_chunks: int,
    stem: str | None,
) -> None:
    """
    Initialize a separation job by scanning a directory and registering files
    into chunks in a SQLite database.

    :param directory: Root directory containing subdirectories to process.
    :param db_path: Path to create the SQLite database.
    :param filename: Audio filename to target within each subdirectory.
    :param model: Name of the Demucs model to use.
    :param num_chunks: Number of chunks to split the work into.
    :param stem: Stem to isolate, or ``None`` for all stems.
    """
    subdirs = [d for d in directory.iterdir() if d.is_dir()]
    total = len(subdirs)

    if total == 0:
        logger.warning("No subdirectories found")
        return

    random.seed(42)
    random.shuffle(subdirs)

    db_path.parent.mkdir(parents=True, exist_ok=True)

    with sqlite3.connect(db_path) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS job_config (
                key TEXT PRIMARY KEY,
                value TEXT
            )
        """)

        config = {
            "directory": str(directory),
            "filename": filename,
            "model": model,
            "stem": stem or "",
            "num_chunks": str(num_chunks),
            "total_files": str(total),
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }

        for key, value in config.items():
            conn.execute(
                "INSERT OR REPLACE INTO job_config (key, value) VALUES (?, ?)",
                (key, value),
            )

        conn.execute("""
            CREATE TABLE IF NOT EXISTS separations (
                name TEXT PRIMARY KEY,
                input_path TEXT NOT NULL,
                output_path TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'pending',
                chunk_id INTEGER NOT NULL,
                duration_seconds REAL,
                error_message TEXT,
                processed_at TIMESTAMP
            )
        """)

        conn.execute("DELETE FROM separations")

        chunk_size = total // num_chunks

        for chunk_id in range(1, num_chunks + 1):
            start = (chunk_id - 1) * chunk_size
            end = total if chunk_id == num_chunks else chunk_id * chunk_size

            for subdir in subdirs[start:end]:
                name = subdir.name
                input_file = subdir / filename
                if stem:
                    output_path = subdir / f"{model}_{stem}.wav"
                else:
                    output_path = subdir / f"{model}_stems"

                conn.execute(
                    "INSERT INTO separations (name, input_path, output_path, chunk_id) VALUES (?, ?, ?, ?)",
                    (name, str(input_file), str(output_path), chunk_id),
                )

    logger.info(f"Registered {total} files into {num_chunks} chunks")


def process_chunk(db_path: Path, chunk_id: int) -> None:
    """
    Process one chunk of the separation job.

    Loads the model and configuration from the database, then processes
    all pending files assigned to the given chunk.

    :param db_path: Path to the job database.
    :param chunk_id: 1-based chunk identifier.
    """
    config = _load_config(db_path)
    model_name = config["model"]
    stem = config["stem"] or None
    model = _load_model(model_name, stem)

    with sqlite3.connect(db_path) as conn:
        cursor = conn.execute(
            "SELECT name, input_path, output_path FROM separations WHERE chunk_id = ? AND status = 'pending'",
            (chunk_id,),
        )
        pending = cursor.fetchall()

        if not pending:
            return

        success_count = 0
        failed_count = 0
        skipped_count = 0

        for name, input_path, output_path in pending:
            result = _process_file(
                conn, model, model_name, stem, name, input_path, output_path
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


def show_stats(db_path: Path) -> None:
    """
    Display processing statistics from the database.

    Prints model configuration and per-status counts to stdout.

    :param db_path: Path to the job database.
    """
    config = _load_config(db_path)

    with sqlite3.connect(db_path) as conn:
        cursor = conn.execute(
            "SELECT status, COUNT(*), AVG(duration_seconds) FROM separations GROUP BY status"
        )
        stats = cursor.fetchall()

        cursor = conn.execute("SELECT COUNT(*) FROM separations")
        total = cursor.fetchone()[0]

    logger.info(f"Model: {config.get('model')}")
    logger.info(f"Stem: {config.get('stem') or 'all'}")
    logger.info(f"Directory: {config.get('directory')}")

    status_data = {row[0]: {"count": row[1], "avg": row[2] or 0} for row in stats}

    for status in ["pending", "processing", "success", "failed"]:
        if status in status_data:
            count = status_data[status]["count"]
            pct = 100 * count / total
            logger.info(f"  {status}: {count} ({pct:.1f}%)")

    success_count = status_data.get("success", {}).get("count", 0)
    completion = 100 * success_count / total if total > 0 else 0
    logger.info(f"Completion: {completion:.1f}%")


def retry_failed(db_path: Path) -> None:
    """
    Retry all failed separations.

    Loads the model and reprocesses every entry with ``'failed'`` status.

    :param db_path: Path to the job database.
    """
    config = _load_config(db_path)
    model_name = config["model"]
    stem = config["stem"] or None
    model = _load_model(model_name, stem)

    with sqlite3.connect(db_path) as conn:
        cursor = conn.execute(
            "SELECT name, input_path, output_path FROM separations WHERE status = 'failed'"
        )
        failed = cursor.fetchall()

        if not failed:
            return

        success_count = 0
        still_failed = 0

        for name, input_path, output_path in failed:
            result = _process_file(
                conn, model, model_name, stem, name, input_path, output_path
            )
            if result == "success" or result == "skipped":
                success_count += 1
            elif result == "failed":
                still_failed += 1

    logger.info(f"Retry: {success_count} fixed, {still_failed} still failed")
