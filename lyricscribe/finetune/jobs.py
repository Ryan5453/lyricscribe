import fcntl
import json
import logging
import shutil
from pathlib import Path

from lyricscribe.finetune.config import list_exports
from lyricscribe.finetune.engine import STATE_FILE

logger = logging.getLogger(__name__)

STATUSES = ("pending", "running", "complete", "failed")


def write_status(job_dir: Path, config: dict, status: str = "pending", global_step: int = 0) -> None:
    """
    Write a job's ``status.json``.

    :param job_dir: Job directory.
    :param config: Job config.
    :param status: One of :data:`STATUSES`.
    :param global_step: Steps trained so far.
    """
    (job_dir / "status.json").write_text(json.dumps({
        "exp_name": config["exp_name"],
        "architecture": config["architecture"],
        "base_model": config["base_model"],
        "filenames": config["filenames"],
        "max_steps": config["max_steps"],
        "global_step": global_step,
        "status": status,
    }, indent=2))


def setup_finetune_job(config: dict) -> Path:
    """
    Save a new job's config and status.

    :param config: Job config (with its step plan resolved).
    :return: The job directory.
    """
    job_dir = Path(config["output_dir"]) / config["exp_name"]
    job_dir.mkdir(parents=True, exist_ok=True)
    (job_dir / "config.json").write_text(json.dumps(config, indent=2))
    write_status(job_dir, config)
    logger.info(f"Job initialized: {job_dir} ({config['max_steps']} steps, {config['steps_per_epoch']} per epoch)")
    return job_dir


def load_job_status(job_dir: Path) -> dict:
    """
    Load a job's ``status.json``.

    :param job_dir: Job directory.
    :return: Status dictionary.
    """
    return json.loads((job_dir / "status.json").read_text())


def set_status(job_dir: Path, status: str, global_step: int | None = None) -> None:
    """
    Update a job's status under a file lock.

    :param job_dir: Job directory.
    :param status: One of :data:`STATUSES`.
    :param global_step: Steps trained so far, if known.
    """
    with open(job_dir / ".status.lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            current = load_job_status(job_dir)
            current["status"] = status
            if global_step is not None:
                current["global_step"] = global_step
            (job_dir / "status.json").write_text(json.dumps(current, indent=2))
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def read_metrics(job_dir: Path) -> list[dict]:
    """
    The rows of a job's ``metrics.jsonl`` (one per export).

    :param job_dir: Job directory.
    :return: Rows, oldest first.
    """
    path = job_dir / "metrics.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def show_job_stats(job_dir: Path) -> None:
    """
    Log a job's progress, exports and metrics.

    :param job_dir: Job directory.
    """
    status = load_job_status(job_dir)
    logger.info(f"Experiment: {status['exp_name']}")
    logger.info(f"Model: {status['architecture']} / {status['base_model']}")
    logger.info(f"Filenames: {', '.join(status['filenames'])}")
    logger.info(f"Progress: step {status['global_step']}/{status['max_steps']} ({status['status']})")
    exports = list_exports(job_dir)
    if exports:
        logger.info(f"Exports: {', '.join(p.name for _, p in exports)}")
    for row in read_metrics(job_dir):
        logger.info("  " + ", ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in row.items()))


def gather_experiment_status(job_dir: Path) -> dict:
    """
    One-line summary of a job.

    :param job_dir: Job directory.
    :return: ``exp_name``, ``status``, ``global_step``, ``max_steps``, ``last_export`` and ``last_metrics``.
    """
    status = load_job_status(job_dir)
    exports = list_exports(job_dir)
    metrics = read_metrics(job_dir)
    return {
        "exp_name": status["exp_name"],
        "status": status["status"],
        "global_step": status["global_step"],
        "max_steps": status["max_steps"],
        "last_export": exports[-1][1].name if exports else None,
        "last_metrics": metrics[-1] if metrics else None,
    }


def reset_job(job_dir: Path) -> None:
    """
    Delete a job's checkpoints, exports, training state and metrics, and mark it pending.

    :param job_dir: Job directory.
    :raises RuntimeError: If branches still link to its checkpoints.
    """
    branches = sorted(p.name for p in (job_dir / "branches").glob("*")) if (job_dir / "branches").is_dir() else []
    if branches:
        raise RuntimeError(f"{job_dir} has branches {branches} linked to its checkpoints; remove them first")
    for name in ("hf_trainer", "exports"):
        shutil.rmtree(job_dir / name, ignore_errors=True)
    for name in (STATE_FILE, "metrics.jsonl"):
        (job_dir / name).unlink(missing_ok=True)
    set_status(job_dir, "pending", 0)
    logger.info(f"Job reset: {job_dir}")
