import fcntl
import json
import logging
import shutil
from pathlib import Path

logger = logging.getLogger(__name__)


def setup_finetune_job(config: dict, train_manifest: Path, val_manifest: Path | None) -> Path:
    """
    Initialize a finetuning job by saving config and creating job directory.
    
    :param config: Configuration dictionary
    :param train_manifest: Path to training manifest
    :param val_manifest: Path to validation manifest (optional)
    :return: Path to job directory
    """
    job_dir = Path(config["output_dir"]) / config["exp_name"]
    job_dir.mkdir(parents=True, exist_ok=True)
    
    with open(job_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)
    
    status = {
        "exp_name": config["exp_name"],
        "architecture": config["architecture"],
        "base_model": config["base_model"],
        "filenames": config["filenames"],
        "max_epochs": config["max_epochs"],
        "epochs_per_job": config["epochs_per_job"],
        "current_epoch": 0,
        "status": "pending",
    }
    
    with open(job_dir / "status.json", "w") as f:
        json.dump(status, f, indent=2)
    
    chunks_dir = job_dir / "chunks"
    chunks_dir.mkdir(exist_ok=True)
    
    num_chunks = (config["max_epochs"] + config["epochs_per_job"] - 1) // config["epochs_per_job"]
    
    for chunk_id in range(1, num_chunks + 1):
        start_epoch = (chunk_id - 1) * config["epochs_per_job"]
        end_epoch = min(start_epoch + config["epochs_per_job"], config["max_epochs"])
        
        chunk_data = {
            "chunk_id": chunk_id,
            "start_epoch": start_epoch,
            "end_epoch": end_epoch,
            "status": "pending",
        }
        
        with open(chunks_dir / f"chunk_{chunk_id}.json", "w") as f:
            json.dump(chunk_data, f, indent=2)
    
    logger.info(f"Job initialized: {config['exp_name']}")
    logger.info(f"Directory: {job_dir}")
    logger.info(f"Epochs: {config['max_epochs']} in {num_chunks} chunks")
    
    return job_dir


def load_job_status(job_dir: Path) -> dict:
    """
    Load the top-level job status from ``status.json``.

    :param job_dir: Path to the job directory.
    :return: Status dictionary.
    """
    with open(job_dir / "status.json") as f:
        return json.load(f)


def save_job_status(job_dir: Path, status: dict) -> None:
    """
    Write the top-level job status to ``status.json``.

    :param job_dir: Path to the job directory.
    :param status: Status dictionary to write.
    """
    with open(job_dir / "status.json", "w") as f:
        json.dump(status, f, indent=2)


def load_chunk_status(job_dir: Path, chunk_id: int) -> dict:
    """
    Load the status of a single chunk.

    :param job_dir: Path to the job directory.
    :param chunk_id: Chunk number.
    :return: Chunk status dictionary.
    """
    with open(job_dir / "chunks" / f"chunk_{chunk_id}.json") as f:
        return json.load(f)


def save_chunk_status(job_dir: Path, chunk_id: int, status: dict) -> None:
    """
    Write the status of a single chunk.

    :param job_dir: Path to the job directory.
    :param chunk_id: Chunk number.
    :param status: Chunk status dictionary to write.
    """
    with open(job_dir / "chunks" / f"chunk_{chunk_id}.json", "w") as f:
        json.dump(status, f, indent=2)


def update_chunk_status(
    job_dir: Path,
    chunk_id: int,
    status: str,
) -> None:
    """
    Update chunk status after processing.

    Uses a file lock to prevent concurrent writes from corrupting state.

    :param job_dir: Path to job directory
    :param chunk_id: Which chunk was processed
    :param status: New status (pending, success, failed)
    """
    lock_path = job_dir / ".status.lock"
    with open(lock_path, "w") as lock_file:
        fcntl.flock(lock_file, fcntl.LOCK_EX)
        try:
            chunk_data = load_chunk_status(job_dir, chunk_id)
            chunk_data["status"] = status
            save_chunk_status(job_dir, chunk_id, chunk_data)

            job_status = load_job_status(job_dir)

            if status == "success":
                job_status["current_epoch"] = chunk_data["end_epoch"]

            all_chunks = list((job_dir / "chunks").glob("chunk_*.json"))
            chunk_statuses = [
                load_chunk_status(job_dir, int(p.stem.split("_")[1]))["status"]
                for p in all_chunks
            ]
            all_done = all(s in ("success", "failed") for s in chunk_statuses)

            if all_done:
                if any(s == "failed" for s in chunk_statuses):
                    job_status["status"] = "complete_with_failures"
                else:
                    job_status["status"] = "complete"

            save_job_status(job_dir, job_status)
        finally:
            fcntl.flock(lock_file, fcntl.LOCK_UN)


def show_job_stats(job_dir: Path) -> None:
    status = load_job_status(job_dir)

    logger.info(f"Experiment: {status['exp_name']}")
    logger.info(f"Model: {status['architecture']} / {status['base_model']}")
    logger.info(f"Filenames: {', '.join(status['filenames'])}")
    logger.info(f"Progress: {status['current_epoch']}/{status['max_epochs']} epochs")
    logger.info(f"Status: {status['status']}")

    chunks_dir = job_dir / "chunks"
    chunk_files = list(chunks_dir.glob("chunk_*.json"))

    counts = {"pending": 0, "running": 0, "success": 0, "failed": 0}
    for chunk_file in chunk_files:
        chunk_id = int(chunk_file.stem.split("_")[1])
        chunk_data = load_chunk_status(job_dir, chunk_id)
        counts[chunk_data["status"]] = counts.get(chunk_data["status"], 0) + 1

    logger.info("Chunks:")
    for status_name, count in counts.items():
        if count > 0:
            logger.info(f"  {status_name}: {count}")

    # List checkpoints
    ckpt_dir = job_dir / "checkpoints"
    if ckpt_dir.exists():
        checkpoints = sorted(
            (p for p in ckpt_dir.iterdir() if p.name.startswith("epoch_")),
            key=lambda p: p.name,
        )
        if checkpoints:
            logger.info(f"Checkpoints: {len(checkpoints)}")
            for ckpt in checkpoints:
                logger.info(f"  {ckpt.name}")

    # Show training metrics
    metrics_path = job_dir / "metrics.jsonl"
    if metrics_path.exists():
        logger.info("Metrics:")
        with open(metrics_path) as f:
            for line in f:
                entry = json.loads(line)
                parts = [f"epoch {entry['epoch']}"]
                for key in sorted(entry):
                    if key == "epoch":
                        continue
                    val = entry[key]
                    if isinstance(val, float):
                        parts.append(f"{key}={val:.4f}")
                    else:
                        parts.append(f"{key}={val}")
                logger.info(f"  {', '.join(parts)}")


def reset_job(job_dir: Path) -> None:
    """
    Reset a finetuning job to start from scratch.
    
    Deletes all checkpoints and resets chunk statuses.
    
    :param job_dir: Path to job directory
    """
    status = load_job_status(job_dir)
    status["current_epoch"] = 0
    status["status"] = "pending"
    save_job_status(job_dir, status)
    
    chunks_dir = job_dir / "chunks"
    for chunk_file in chunks_dir.glob("chunk_*.json"):
        chunk_id = int(chunk_file.stem.split("_")[1])
        chunk_data = load_chunk_status(job_dir, chunk_id)
        chunk_data["status"] = "pending"
        save_chunk_status(job_dir, chunk_id, chunk_data)
    
    checkpoint_dir = job_dir / "checkpoints"
    if checkpoint_dir.exists():
        shutil.rmtree(checkpoint_dir)

    metrics_path = job_dir / "metrics.jsonl"
    if metrics_path.exists():
        metrics_path.unlink()

    logger.info(f"Job reset: {job_dir}")
