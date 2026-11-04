from __future__ import annotations

import json
import re
from pathlib import Path

DEFAULT_MAX_RUN = 5
DEFAULT_MAX_CYCLE = 25


def deloop(words: list[str], max_run: int = DEFAULT_MAX_RUN, max_cycle: int = DEFAULT_MAX_CYCLE) -> list[str]:
    """
    Collapse every over-long consecutive repeat of a short cycle.

    At each position the longest-covering loop (``repeats * cycle_len``) wins.

    :param words: Tokens.
    :param max_run: Repeats kept of a looping cycle.
    :param max_cycle: Longest cycle length considered, in words.
    :return: Repaired tokens.
    """
    out: list[str] = []
    i = 0
    n = len(words)
    while i < n:
        best = None
        for length in range(1, max_cycle + 1):
            if i + 2 * length > n:
                break
            cycle = words[i:i + length]
            k = 1
            while i + (k + 1) * length <= n and words[i + k * length:i + (k + 1) * length] == cycle:
                k += 1
            if k > max_run and (best is None or k * length > best[0] * best[1]):
                best = (k, length, cycle)
        if best:
            k, length, cycle = best
            out.extend(cycle * max_run)
            i += k * length
        else:
            out.append(words[i])
            i += 1
    return out


def deloop_job(src: Path, dst: Path, max_run: int = DEFAULT_MAX_RUN, max_cycle: int = DEFAULT_MAX_CYCLE) -> tuple[int, int]:
    """
    Write a de-looped copy of job dir *src* to *dst*.

    ``config.json`` is copied with a ``deloop`` provenance block added.

    :param src: Source job dir.
    :param dst: Destination job dir (created).
    :param max_run: See :func:`deloop`.
    :param max_cycle: See :func:`deloop`.
    :return: ``(songs changed, tokens removed)``.
    """
    src, dst = Path(src), Path(dst)
    if src.resolve() == dst.resolve():
        raise ValueError("deloop writes a copy; --out must differ from the source job dir")
    dst.mkdir(parents=True, exist_ok=True)
    cfg_path = src / "config.json"
    if cfg_path.exists():
        cfg = json.loads(cfg_path.read_text())
        cfg["deloop"] = {"source": str(src), "max_run": max_run, "max_cycle": max_cycle}
        (dst / "config.json").write_text(json.dumps(cfg, indent=2) + "\n")
    changed = removed = 0
    with (dst / "results.jsonl").open("w") as out:
        for line in (src / "results.jsonl").read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            toks = re.findall(r"\S+", row.get("transcription") or "")
            fixed = deloop(toks, max_run, max_cycle)
            if len(fixed) != len(toks):
                changed += 1
                removed += len(toks) - len(fixed)
            if row.get("transcription") is not None:
                row["transcription"] = " ".join(fixed)
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
    return changed, removed
