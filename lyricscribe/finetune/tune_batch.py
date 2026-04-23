"""
Batch-size sweep orchestrator.

Drives :mod:`lyricscribe.finetune.probe` to find the largest fitting
batch size for a given (model, audio-filename, GPU, DDP rank count)
combination. Builds one shared manifest from the real dataset, then
invokes each probe trial as a fresh ``torchrun`` subprocess so an OOM
kills the whole DDP group without poisoning the orchestrator's CUDA
state between trials.
"""

import json
import logging
import os
import subprocess
import sys
import tempfile
from pathlib import Path

logger = logging.getLogger(__name__)


# Whisper / Parakeet sweep batch_size (per-rank sample count).
# 24 is the Whisper default, 48 and 96 are common non-power-of-2 stops.
# Sweep stops at the first OOM; bisection fills in the gap afterwards.
_BATCH_SIZE_CANDIDATES: tuple[int, ...] = (1, 2, 4, 8, 16, 24, 32, 48, 64, 96, 128)

# Canary uses Lhotse duration batching — batch_size is ignored by its
# training dataloader. Sweep batch_duration (seconds of audio per batch)
# instead. Production default is 1440s; stop-points chosen to cover
# common GPU budgets from 40 GB up through H200-class.
_BATCH_DURATION_CANDIDATES: tuple[int, ...] = (60, 120, 300, 600, 900, 1200, 1440, 1800, 2400)


def _candidates_for(architecture: str) -> tuple[int, ...]:
    return _BATCH_DURATION_CANDIDATES if architecture == "canary" else _BATCH_SIZE_CANDIDATES


def _param_label(architecture: str) -> str:
    return "batch_duration (s)" if architecture == "canary" else "batch_size"


def _build_shared_manifest(
    dataset_dir: Path,
    filename: str,
    model_name: str,
    work_dir: Path,
) -> Path:
    """
    Build one NeMo manifest from real audio + ``lyrics.json`` under
    *dataset_dir*, shared across all trials. One window per song is
    enough for the probe — the manifest's only job is to supply the
    dataloader with representative real samples.

    Matches :func:`lyricscribe.finetune.data.create_manifest` exactly
    so the audio + text distribution the probe sees matches production.
    """
    from lyricscribe.finetune import data as finetune_data
    from lyricscribe.finetune.config import detect_architecture

    architecture = detect_architecture(model_name)
    manifest_path = work_dir / "probe_manifest.jsonl"
    dataset = finetune_data.LyricsDataset(dataset_dir, [filename])
    n = finetune_data.create_manifest(
        dataset,
        manifest_path,
        architecture=architecture,
        model_name=model_name,
        windows_per_song_multiplier=1,
    )
    logger.info(f"Built probe manifest with {n} entries at {manifest_path}")
    return manifest_path


def _run_trial(
    manifest: Path,
    model_name: str,
    bs: int,
    freeze_encoder: bool,
    max_duration_s: float,
    num_gpus: int,
    timeout_s: int,
    work_dir: Path,
    phase: str,
    n_steps: int = 5,
) -> dict:
    """
    Launch one probe trial as ``torchrun --nproc_per_node=num_gpus``
    subprocess. Exit code != 0 is treated as an OOM (the whole DDP
    group dies as a unit, so either all ranks succeeded or none did).

    :returns: Result dict including ``status`` ∈ {success, oom, timeout,
        crash} plus the fields the probe writes to its output file.
    """
    print(f"\n{'=' * 72}")
    print(f"  Trial ({phase}): bs={bs} (world_size={num_gpus})")
    print(f"{'=' * 72}")

    result_file = work_dir / f"probe_result_{bs}_{phase}.json"
    result_file.unlink(missing_ok=True)

    cmd = [
        "torchrun",
        f"--nproc_per_node={num_gpus}",
        "--standalone",
        "-m",
        "lyricscribe.finetune.probe",
        "--manifest",
        str(manifest),
        "--model",
        model_name,
        "--bs",
        str(bs),
        "--max-duration",
        str(max_duration_s),
        "--n-steps",
        str(n_steps),
        "--output",
        str(result_file),
    ]
    if freeze_encoder:
        cmd.append("--freeze-encoder")

    try:
        proc = subprocess.run(
            cmd,
            timeout=timeout_s,
            check=False,
            stdout=sys.stdout,
            stderr=sys.stderr,
        )
    except subprocess.TimeoutExpired:
        print(f"  [bs={bs}] TIMEOUT after {timeout_s}s")
        return {"bs": bs, "phase": phase, "status": "timeout"}

    # Success path: result file was written by rank 0 with all metrics.
    if result_file.exists():
        try:
            payload = json.loads(result_file.read_text())
        except json.JSONDecodeError:
            payload = {}
        if proc.returncode == 0 and "peak_reserved_gb" in payload:
            result = {"bs": bs, "phase": phase, "status": "success", **payload}
            print(
                f"  [bs={bs}] OK. "
                f"VRAM reserved peak: {payload['peak_reserved_gb']:.1f}/"
                f"{payload['total_gb']:.1f} GiB "
                f"({100 * payload['peak_reserved_gb'] / payload['total_gb']:.0f}%), "
                f"mean step: {payload.get('mean_step_ms', 0):.0f} ms"
            )
            return result

    # Non-zero exit code → DDP group died. Most common cause is OOM on
    # at least one rank. Distinguishing genuine crashes from OOMs from
    # the parent is unreliable — torchrun reports 1 for both. Treat as
    # OOM so the sweep correctly bisects below; a real crash will still
    # be visible in the stderr output above.
    print(f"  [bs={bs}] OOM (torchrun exit={proc.returncode})")
    return {"bs": bs, "phase": phase, "status": "oom", "exitcode": proc.returncode}


def _detect_num_gpus() -> int:
    """
    Best-effort GPU count. Prefers ``CUDA_VISIBLE_DEVICES`` (honors
    SLURM-managed device masking) and falls back to the device count
    ``torch`` reports on the orchestrator process.
    """
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
    if cvd:
        return len([x for x in cvd.split(",") if x.strip()])
    try:
        import torch

        return max(torch.cuda.device_count(), 1)
    except Exception:
        return 1


def tune_batch_size(
    dataset_dir: Path,
    filename: str,
    model_name: str,
    freeze_encoder: bool = False,
    max_duration_s: float = 30.0,
    start: int = 1,
    max_bs: int = 128,
    min_gap: int = 2,
    timeout_s: int = 900,
    safety_margin_pct: float = 15.0,
    n_steps: int = 5,
) -> list[dict]:
    """
    Find the largest fitting batch size for a given model / audio /
    DDP rank count.

    Two phases: a doubling sweep through :data:`DEFAULT_CANDIDATES`
    until the first OOM, then bisection between the last success and
    the first OOM until the gap is within ``min_gap``.

    :param dataset_dir: Directory of song subdirectories. The probe
        builds a single manifest from this and shares it across trials.
    :param filename: Audio filename inside each song subdirectory.
    :param model_name: Model identifier.
    :param freeze_encoder: Match what production training will use —
        shifts per-rank VRAM by many GiB.
    :param max_duration_s: Clip length. Memory scales with this.
    :param start: Lower bound of the sweep (inclusive).
    :param max_bs: Upper bound of the sweep (inclusive).
    :param min_gap: Bisection stops when ``(first_oom - last_success)
        <= min_gap``.
    :param timeout_s: Per-trial timeout (includes model download/load).
    :param safety_margin_pct: Fallback margin applied to the VRAM
        ceiling when no throughput sweet spot is found.
    :param n_steps: Training steps per trial (after PL's warmup).
    :return: Per-trial result dicts, one per attempted batch size.
    """
    from lyricscribe.finetune.config import detect_architecture

    architecture = detect_architecture(model_name)
    candidates = [b for b in _candidates_for(architecture) if start <= b <= max_bs]
    if not candidates:
        logger.error(
            f"No {_param_label(architecture)} candidates in [{start}, {max_bs}]"
        )
        return []

    num_gpus = _detect_num_gpus()

    print("Sweep config (must match production training):")
    print(f"  dataset_dir    = {dataset_dir}")
    print(f"  filename       = {filename}")
    print(f"  model          = {model_name}  ({architecture})")
    print(f"  sweeping       = {_param_label(architecture)}")
    print(f"  freeze_encoder = {freeze_encoder}")
    print(f"  max_duration_s = {max_duration_s}")
    print(f"  num_gpus (DDP) = {num_gpus}")
    print(f"  n_steps/trial  = {n_steps}")
    print(
        "\nCheck freeze_encoder + max_duration + num_gpus match your real\n"
        "training — each shifts VRAM by many GiB.\n"
    )

    results: list[dict] = []
    last_success: int | None = None
    first_oom: int | None = None

    with tempfile.TemporaryDirectory(prefix="lyricscribe_tune_") as tmp:
        work_dir = Path(tmp)
        manifest = _build_shared_manifest(
            dataset_dir, filename, model_name, work_dir
        )

        for bs in candidates:
            r = _run_trial(
                manifest, model_name, bs, freeze_encoder, max_duration_s,
                num_gpus, timeout_s, work_dir, phase="double", n_steps=n_steps,
            )
            results.append(r)
            if r["status"] == "success":
                last_success = bs
            else:
                if r["status"] == "oom":
                    first_oom = bs
                break

        if (
            last_success is not None
            and first_oom is not None
            and first_oom - last_success > min_gap
        ):
            print(
                f"\n--- bisecting between {last_success} (OK) and {first_oom} "
                f"(OOM), stop when gap ≤ {min_gap} ---"
            )
            lo, hi = last_success, first_oom
            while hi - lo > min_gap:
                mid = (lo + hi) // 2
                if mid == lo or mid == hi:
                    break
                r = _run_trial(
                    manifest, model_name, mid, freeze_encoder, max_duration_s,
                    num_gpus, timeout_s, work_dir, phase="bisect",
                    n_steps=n_steps,
                )
                results.append(r)
                if r["status"] == "success":
                    lo = mid
                elif r["status"] == "oom":
                    hi = mid
                else:
                    # Timeout or crash — can't tell which side, stop.
                    break

    _print_summary(results, safety_margin_pct)
    return results


def _find_throughput_sweet_spot(
    successes: list[dict], threshold: float
) -> dict | None:
    """
    Return the largest ``bs`` beyond which throughput plateaus.

    Walks the successes in order; returns the first point where the
    next trial gave less than ``threshold`` fractional gain in
    samples/sec. Past that point more ``bs`` just costs VRAM.

    :param successes: Successful trial dicts, ordered by increasing bs.
    :param threshold: Minimum acceptable fractional throughput gain
        between adjacent trials (e.g. ``0.10`` for 10%).
    :return: The plateau trial, or ``None`` if no plateau is found.
    """
    if len(successes) < 2:
        return None

    def tput(r: dict) -> float:
        step_ms = r.get("mean_step_ms", 0)
        return r["bs"] / (step_ms / 1000) if step_ms else 0

    for i in range(len(successes) - 1):
        cur, nxt = successes[i], successes[i + 1]
        cur_t, nxt_t = tput(cur), tput(nxt)
        if cur_t <= 0:
            continue
        if (nxt_t - cur_t) / cur_t < threshold:
            return cur
    return None


def _print_summary(results: list[dict], safety_margin_pct: float) -> None:
    print(f"\n{'=' * 72}")
    print("  SUMMARY")
    print(f"{'=' * 72}")
    sorted_r = sorted(results, key=lambda r: r["bs"])
    print(
        f"{'bs':>4}  {'phase':<7}  {'status':<9}  "
        f"{'VRAM':<22}  {'step ms':>8}  {'throughput':>14}"
    )
    for r in sorted_r:
        phase = r.get("phase", "-")
        if r["status"] == "success":
            pct = 100 * r["peak_reserved_gb"] / r["total_gb"]
            vram = f"{r['peak_reserved_gb']:.1f}/{r['total_gb']:.1f} GiB ({pct:.0f}%)"
            step_ms = r.get("mean_step_ms", 0)
            step = f"{step_ms:.0f}"
            tput = r["bs"] / (step_ms / 1000) if step_ms else 0
            tput_s = f"{tput:.1f} samp/s"
        else:
            vram = "-"
            step = "-"
            tput_s = r.get("error", "")[:14] if r.get("error") else "-"
        print(
            f"{r['bs']:>4}  {phase:<7}  {r['status']:<9}  "
            f"{vram:<22}  {step:>8}  {tput_s:>14}"
        )

    successes = sorted(
        [r for r in results if r["status"] == "success"],
        key=lambda r: r["bs"],
    )
    if not successes:
        print(
            "\nNo batch size completed. Model doesn't fit even at the smallest "
            "tried bs — enable --freeze-encoder, lower --max-duration, or "
            "switch to a bigger GPU."
        )
        return

    largest = successes[-1]
    sweet = _find_throughput_sweet_spot(successes, threshold=0.10)

    l_step = largest.get("mean_step_ms", 0)
    print(
        f"\nLargest fitting:      bs={largest['bs']}  "
        f"({largest['peak_reserved_gb']:.1f}/{largest['total_gb']:.1f} GiB, "
        f"{(largest['bs'] / (l_step / 1000)) if l_step else 0:.1f} samp/s)"
    )
    if sweet and sweet["bs"] < largest["bs"]:
        s_step = sweet.get("mean_step_ms", 0)
        s_tput = sweet["bs"] / (s_step / 1000) if s_step else 0
        l_tput = largest["bs"] / (l_step / 1000) if l_step else 0
        gain = 100 * (l_tput - s_tput) / s_tput if s_tput else 0
        print(
            f"Throughput sweet spot: bs={sweet['bs']}  "
            f"({sweet['peak_reserved_gb']:.1f}/{sweet['total_gb']:.1f} GiB, "
            f"{s_tput:.1f} samp/s)"
        )
        print(
            f"  → bs={sweet['bs']} → bs={largest['bs']} only gains {gain:.0f}% "
            f"throughput for "
            f"{largest['peak_reserved_gb']-sweet['peak_reserved_gb']:.1f} GiB more VRAM."
        )
        recommended = sweet["bs"]
        rationale = "throughput sweet spot — bigger wastes VRAM without speeding up"
    else:
        recommended = max(1, int(round(largest["bs"] * (1 - safety_margin_pct / 100))))
        rationale = (
            f"{safety_margin_pct:.0f}% margin on VRAM ceiling for "
            "per-sample length variance"
        )
    print(f"\nRecommended batch_size: {recommended}  ({rationale})")
