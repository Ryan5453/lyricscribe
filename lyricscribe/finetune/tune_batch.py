import logging
import multiprocessing as mp
from pathlib import Path

logger = logging.getLogger(__name__)


# 24 is the Whisper default, 48 and 96 are common non-power-of-2 stops.
# Sweep stops at the first OOM; bisection fills in the gap afterwards.
DEFAULT_CANDIDATES: tuple[int, ...] = (1, 2, 4, 8, 16, 24, 32, 48, 64, 96, 128)


def _worker(kwargs: dict, queue: "mp.Queue") -> None:
    """
    Multiprocessing target. Runs one VRAM probe, sends the result back
    through the queue, exits non-zero on OOM or crash so the parent can
    also detect failure via exit code.
    """
    import sys
    try:
        import torch
        from lyricscribe.finetune.probe import probe_batch_size
        result = probe_batch_size(**kwargs)
        queue.put({"status": "success", **result})
    except torch.cuda.OutOfMemoryError as e:
        queue.put({"status": "oom", "error": str(e)[:200]})
        sys.exit(1)
    except RuntimeError as e:
        # Numba / CUDA sometimes surfaces OOM as a generic RuntimeError
        # with "out of memory" in the message — treat the same.
        if "out of memory" in str(e).lower():
            queue.put({"status": "oom", "error": str(e)[:200]})
            sys.exit(1)
        queue.put({"status": "crash", "error": f"{type(e).__name__}: {e}"[:300]})
        sys.exit(2)
    except Exception as e:
        queue.put({"status": "crash", "error": f"{type(e).__name__}: {e}"[:300]})
        sys.exit(2)


def _run_trial(
    dataset_dir: Path,
    filename: str,
    model_name: str,
    bs: int,
    freeze_encoder: bool,
    max_duration_s: float,
    timeout_s: int,
    phase: str,
) -> dict:
    """
    Spawn a fresh worker process, run one probe, return the result.

    Uses ``spawn`` (not ``fork``) so the child doesn't inherit the
    parent's CUDA context — if it did, an OOM in the child would
    poison the parent and make subsequent trials unrecoverable.
    """
    print(f"\n{'=' * 72}")
    print(f"  Trial ({phase}): batch_size = {bs}")
    print(f"{'=' * 72}")

    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    p = ctx.Process(
        target=_worker,
        args=(
            dict(
                dataset_dir=dataset_dir,
                filename=filename,
                model_name=model_name,
                bs=bs,
                freeze_encoder=freeze_encoder,
                max_duration_s=max_duration_s,
            ),
            q,
        ),
    )
    p.start()
    p.join(timeout=timeout_s)

    if p.is_alive():
        # Hung — likely deadlocked on model download or dataloader init.
        p.kill()
        p.join(timeout=5)
        result = {"bs": bs, "phase": phase, "status": "timeout"}
        print(f"  [bs={bs}] TIMEOUT after {timeout_s}s")
        return result

    # Trust the queue over exit code when both exist — the worker may
    # have posted a result before exiting non-zero.
    msg = None
    try:
        if not q.empty():
            msg = q.get_nowait()
    except Exception:
        pass

    if msg is None:
        # Worker died before posting anything (segfault or OOM-kill).
        status = "oom" if p.exitcode != 0 else "no_result"
        msg = {"status": status, "exitcode": p.exitcode}

    result = {"bs": bs, "phase": phase, **msg}
    if result["status"] == "success":
        print(
            f"  [bs={bs}] OK. "
            f"VRAM reserved peak: {result['peak_reserved_gb']:.1f}/{result['total_gb']:.1f} GiB "
            f"({100*result['peak_reserved_gb']/result['total_gb']:.0f}%), "
            f"median step: {result['median_step_ms']:.0f} ms"
        )
    elif result["status"] == "oom":
        print(f"  [bs={bs}] OOM")
    elif result["status"] == "crash":
        print(f"  [bs={bs}] CRASH: {result.get('error', '?')}")
    else:
        print(f"  [bs={bs}] {result['status'].upper()}")
    return result


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
) -> list[dict]:
    """
    Find the largest fitting batch size for a given model / audio /
    GPU combination.

    Two phases: a doubling sweep through :data:`DEFAULT_CANDIDATES`
    until the first OOM, then bisection between the last success and
    the first OOM until the gap is within ``min_gap``.

    :param dataset_dir: Directory of song subdirectories to draw real
        audio from.
    :param filename: Audio filename inside each song subdirectory.
    :param model_name: Model identifier.
    :param freeze_encoder: Match what production training will use —
        shifts per-rank VRAM by many GiB.
    :param max_duration_s: Clip length. Memory scales with this.
    :param start: Lower bound of the sweep (inclusive).
    :param max_bs: Upper bound of the sweep (inclusive).
    :param min_gap: Bisection stops when ``(first_oom - last_success)
        <= min_gap``.
    :param timeout_s: Per-trial timeout. Hung trials are killed.
    :param safety_margin_pct: Fallback margin applied to the VRAM
        ceiling when no throughput sweet spot is found.
    :return: Per-trial result dicts, one per attempted batch size.
    """
    candidates = [b for b in DEFAULT_CANDIDATES if start <= b <= max_bs]
    if not candidates:
        logger.error(f"No sweep candidates in [{start}, {max_bs}]")
        return []

    print("Sweep config (must match production training):")
    print(f"  dataset_dir    = {dataset_dir}")
    print(f"  filename       = {filename}")
    print(f"  model          = {model_name}")
    print(f"  freeze_encoder = {freeze_encoder}")
    print(f"  max_duration_s = {max_duration_s}")
    print(
        "\nCheck freeze_encoder + max_duration match your real training —\n"
        "each shifts VRAM by many GiB.\n"
    )

    results: list[dict] = []
    last_success: int | None = None
    first_oom: int | None = None

    for bs in candidates:
        r = _run_trial(
            dataset_dir, filename, model_name, bs,
            freeze_encoder, max_duration_s, timeout_s, phase="double",
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
            f"\n--- bisecting between {last_success} (OK) and {first_oom} (OOM), "
            f"stop when gap ≤ {min_gap} ---"
        )
        lo, hi = last_success, first_oom
        while hi - lo > min_gap:
            mid = (lo + hi) // 2
            if mid == lo or mid == hi:
                break
            r = _run_trial(
                dataset_dir, filename, model_name, mid,
                freeze_encoder, max_duration_s, timeout_s, phase="bisect",
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
        return r["bs"] / (r["median_step_ms"] / 1000) if r["median_step_ms"] else 0

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
            step = f"{r['median_step_ms']:.0f}"
            tput = r["bs"] / (r["median_step_ms"] / 1000) if r["median_step_ms"] else 0
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

    print(
        f"\nLargest fitting:      bs={largest['bs']}  "
        f"({largest['peak_reserved_gb']:.1f}/{largest['total_gb']:.1f} GiB, "
        f"{largest['bs'] / (largest['median_step_ms'] / 1000):.1f} samp/s)"
    )
    if sweet and sweet["bs"] < largest["bs"]:
        s_tput = sweet["bs"] / (sweet["median_step_ms"] / 1000)
        l_tput = largest["bs"] / (largest["median_step_ms"] / 1000)
        gain = 100 * (l_tput - s_tput) / s_tput
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
            f"{safety_margin_pct:.0f}% margin on VRAM ceiling for DDP + "
            "per-sample length variance"
        )
    print(f"\nRecommended batch_size: {recommended}  ({rationale})")
