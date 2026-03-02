import json
import logging
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)


def update_status(
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

    chunk_path = job_dir / f"chunk_{chunk_id}.json"
    with open(chunk_path, "w") as f:
        json.dump(chunk_data, f, indent=2)


def show_stats(job_dir: Path) -> None:
    """
    Display processing statistics by reading all chunk manifests.

    :param job_dir: Path to the job directory.
    """
    with open(job_dir / "config.json") as f:
        config = json.load(f)

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
