from __future__ import annotations

import json
import multiprocessing
import random
from pathlib import Path

import numpy as np

from lyricscribe.evaluate import _language_for, _normalized

Network = list[dict[str, str | None]]


def normalize(text: str, language: str) -> list[str]:
    """
    Tokenise *text* exactly as the evaluator does.

    :param text: Raw transcription or reference.
    :param language: ISO language code.
    :return: Normalised word tokens.
    """
    return _normalized(text or "", language).split()


def load_references(dataset_dir: Path) -> tuple[dict[str, list[str]], dict[str, str]]:
    """
    Load and normalise every song's ``unsynced`` reference.

    :param dataset_dir: Directory of song subdirectories with ``lyrics.json``.
    :return: ``(refs, languages)`` keyed by song id (directory name).
    """
    refs, langs = {}, {}
    for song_dir in Path(dataset_dir).iterdir():
        path = song_dir / "lyrics.json"
        if not song_dir.is_dir() or not path.exists():
            continue
        data = json.loads(path.read_text())
        lang = _language_for(data)
        langs[song_dir.name] = lang
        refs[song_dir.name] = normalize(data["unsynced"]["data"], lang)
    return refs, langs


def load_hypotheses(
    job_dir: Path, langs: dict[str, str], expected: int | None = None
) -> dict[str, list[str]]:
    """
    Load and normalise a job dir's ``results.jsonl``.

    :param job_dir: Transcription (or ROVER output) job directory.
    :param langs: Song id -> language, from :func:`load_references`.
    :param expected: Required number of songs; ``None`` skips the check.
    :return: Song id -> normalised tokens.
    :raises RuntimeError: If the song count differs from *expected*.
    """
    path = Path(job_dir) / "results.jsonl"
    hyps = {}
    with path.open() as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                sid = row["song_id"]
                hyps[sid] = normalize(row.get("transcription") or "", langs.get(sid, "en"))
    if expected is not None and len(hyps) != expected:
        raise RuntimeError(f"{path}: {len(hyps)} hyps, expected {expected}")
    return hyps


def merge_into_cn(
    cn: Network, system: str, tokens: list[str], prior: list[str]
) -> Network:
    """
    Align *tokens* into the network by edit distance.

    A token matches a slot at cost 0 if it equals any token already in that
    slot, else substitution cost 1. Rows are vectorised: the horizontal
    ``row[j-1]+1`` dependency is resolved with the shifted running-min
    identity rather than a Python inner loop.

    :param cn: Network built from the *prior* systems.
    :param system: Name of the system being merged.
    :param tokens: Its normalised hypothesis.
    :param prior: Systems already in the network (NULL-filled on insertions).
    :return: The extended network.
    """
    m, n = len(cn), len(tokens)
    if n == 0:
        return [{**slot, system: None} for slot in cn]
    if m == 0:
        return [{**{s: None for s in prior}, system: t} for t in tokens]

    slot_sets = [{v for v in slot.values() if v is not None} for slot in cn]
    token_arr = np.array(tokens, dtype=object)
    memo: dict[str, np.ndarray] = {}

    def equals(token: str) -> np.ndarray:
        cached = memo.get(token)
        if cached is None:
            cached = token_arr == token
            memo[token] = cached
        return cached

    idx = np.arange(n + 1, dtype=np.int32)
    dp = np.zeros((m + 1, n + 1), dtype=np.int32)
    dp[0] = idx
    for i in range(1, m + 1):
        candidates = slot_sets[i - 1]
        if candidates:
            match = np.zeros(n, dtype=bool)
            for token in candidates:
                match |= equals(token)
            cost = (~match).astype(np.int32)
        else:
            cost = np.ones(n, dtype=np.int32)
        prev = dp[i - 1]
        row = np.empty(n + 1, dtype=np.int32)
        row[0] = i
        row[1:] = np.minimum(prev[:-1] + cost, prev[1:] + 1)
        dp[i] = np.minimum.accumulate(row - idx) + idx

    merged: Network = []
    i, j = m, n
    while i > 0 or j > 0:
        if i > 0 and j > 0:
            cost = 0 if tokens[j - 1] in slot_sets[i - 1] else 1
            if dp[i, j] == dp[i - 1, j - 1] + cost:
                merged.append({**cn[i - 1], system: tokens[j - 1]})
                i -= 1
                j -= 1
                continue
        if i > 0 and dp[i, j] == dp[i - 1, j] + 1:
            merged.append({**cn[i - 1], system: None})
            i -= 1
            continue
        merged.append({**{s: None for s in prior}, system: tokens[j - 1]})
        j -= 1
    merged.reverse()
    return merged


def build_cn(systems: list[str], hyps: dict[str, dict[str, list[str]]], song: str) -> Network:
    """
    Build one song's confusion network by merging *systems* in order.

    :param systems: Members in alignment order (first = pivot).
    :param hyps: System -> song id -> tokens.
    :param song: Song id.
    :return: The network.
    """
    cn: Network = [{systems[0]: t} for t in hyps[systems[0]][song]]
    for index in range(1, len(systems)):
        cn = merge_into_cn(cn, systems[index], hyps[systems[index]][song], systems[:index])
    return cn


def select_vote(cn: Network, systems: list[str], null_weight: float = 1.0) -> list[str]:
    """
    Frequency vote over a network; ties break toward the earliest member.

    :param cn: Confusion network.
    :param systems: Members in priority order.
    :param null_weight: Weight of a NULL vote (``beta``). 1.0 is plain
        ROVER.
    :return: Selected tokens.
    """
    out = []
    for slot in cn:
        weights: dict[str | None, float] = {}
        for system in systems:
            token = slot.get(system)
            weights[token] = weights.get(token, 0.0) + (null_weight if token is None else 1.0)
        best = max(weights.values())
        tied = [t for t, w in weights.items() if w >= best - 1e-9]
        if len(tied) == 1:
            winner = tied[0]
        else:
            winner = next((slot.get(s) for s in systems if slot.get(s) in tied), tied[0])
        if winner is not None:
            out.append(winner)
    return out


def _oracle_choices(cn: Network) -> list[list[str | None]]:
    choices = []
    for slot in cn:
        candidates = {v for v in slot.values() if v is not None}
        has_null = any(slot.get(s) is None for s in slot)
        choices.append(sorted(candidates) + ([None] if has_null else []))
    return choices


def oracle_distance(cn: Network, ref: list[str]) -> int:
    """
    Edit distance of the best per-slot selection against *ref*.

    DP over (slot, reference position); reference deletions inside a row use
    the shifted running-min identity so the update stays vectorised.

    :param cn: Confusion network.
    :param ref: Normalised reference tokens.
    :return: Minimum achievable edit distance.
    """
    n = len(ref)
    ref_arr = np.array(ref, dtype=object)
    arange = np.arange(n + 1, dtype=np.int32)
    dp = arange.copy()
    for options in _oracle_choices(cn):
        best = np.full(n + 1, np.iinfo(np.int32).max // 4, dtype=np.int32)
        tokens = [c for c in options if c is not None]
        if None in options:
            best = np.minimum(best, dp)
        if tokens:
            best = np.minimum(best, dp + 1)
            if n:
                match = np.ones(n, dtype=np.int32)
                for token in tokens:
                    match = np.minimum(match, (ref_arr != token).astype(np.int32))
                best[1:] = np.minimum(best[1:], dp[:-1] + match)
        dp = np.minimum.accumulate(best - arange) + arange
    return int(dp[n])


def orderings(base: list[str], n: int, seed: int = 0) -> list[list[str]]:
    """
    *n* distinct member orderings: *base* first, then seeded random shuffles.

    :param base: Base member order.
    :param n: Number of orderings (including *base*).
    :param seed: RNG seed.
    :return: Orderings.
    """
    rng = random.Random(seed)
    out = [tuple(base)]
    seen = {tuple(base)}
    while len(out) < n:
        p = tuple(rng.sample(base, len(base)))
        if p not in seen:
            seen.add(p)
            out.append(p)
    return [list(o) for o in out]


# Populated before the fork pool is created so workers inherit it unpickled.
_STATE: dict = {}


def _song_task(song: str) -> tuple[str, list[list[str]], list[int | None]]:
    orders = _STATE["orders"]
    hyps = _STATE["hyps"]
    ref = _STATE["refs"][song]
    with_oracle = _STATE["oracle"]
    null_weight = _STATE["null_weight"]
    votes, oracles = [], []
    for order in orders:
        cn = build_cn(order, hyps, song)
        votes.append(select_vote(cn, order, null_weight))
        oracles.append(oracle_distance(cn, ref) if with_oracle else None)
    return song, votes, oracles


def run_orders(
    orders: list[list[str]],
    hyps: dict[str, dict[str, list[str]]],
    refs: dict[str, list[str]],
    oracle: bool = True,
    null_weight: float = 1.0,
    workers: int = 1,
) -> dict[str, tuple[list[list[str]], list[int | None]]]:
    """
    Vote (and optionally oracle) every song under every member ordering.

    :param orders: Member orderings.
    :param hyps: System -> song id -> tokens.
    :param refs: Song id -> reference tokens (defines the song set).
    :param oracle: Also compute per-order oracle distances.
    :param null_weight: NULL vote weight (see :func:`select_vote`).
    :param workers: Processes to use.
    :return: Song id -> (per-order vote tokens, per-order oracle distance).
    """
    _STATE.update(orders=orders, hyps=hyps, refs=refs, oracle=oracle, null_weight=null_weight)
    songs = sorted(refs)
    if workers > 1:
        ctx = multiprocessing.get_context("fork")
        with ctx.Pool(workers) as pool:
            results = pool.map(_song_task, songs, chunksize=8)
    else:
        results = [_song_task(s) for s in songs]
    return {song: (votes, oracles) for song, votes, oracles in results}


def combine_orderings(per_order_votes: list[list[str]]) -> list[str]:
    """
    ROVER-of-orderings: re-vote one song's per-ordering outputs.

    Ties break toward the first ordering's output.

    :param per_order_votes: Vote tokens, one list per ordering.
    :return: Combined tokens.
    """
    names = [f"o{i}" for i in range(len(per_order_votes))]
    hyps = {name: {"_": toks} for name, toks in zip(names, per_order_votes)}
    return select_vote(build_cn(names, hyps, "_"), names)


def write_vote_job(
    job_dir: Path,
    dataset_dir: Path,
    systems: list[str],
    texts: dict[str, str],
    extra_config: dict | None = None,
) -> None:
    """
    Write a ROVER output as a standard job dir (``config.json`` +
    ``results.jsonl``), consumable by ``evaluate_job`` and usable as a
    member of a further (meta) ensemble.

    :param job_dir: Output directory.
    :param dataset_dir: Reference dataset directory.
    :param systems: Members in order.
    :param texts: Song id -> combined transcription.
    :param extra_config: Additional keys recorded in ``config.json``.
    """
    job_dir = Path(job_dir)
    job_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "directories": [str(dataset_dir)],
        "filename": "rover",
        "model": f"rover-vote:{'+'.join(systems)}",
        "architecture": "rover",
        "rover_systems": systems,
        "total_files": len(texts),
        **(extra_config or {}),
    }
    (job_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    with (job_dir / "results.jsonl").open("w") as f:
        for song in sorted(texts):
            f.write(json.dumps({"song_id": song, "transcription": texts[song]}, ensure_ascii=False) + "\n")


def score(refs: dict[str, list[str]], hyps: dict[str, list[str]]) -> dict:
    """
    Corpus WER of token hypotheses, matching ``evaluate_job``.

    Songs with an empty reference are skipped (as the evaluator does).

    :param refs: Song -> reference tokens.
    :param hyps: Song -> hypothesis tokens.
    :return: ``{"wer", "I", "D", "S", "ref_words", "n_songs"}``.
    """
    import jiwer

    songs = [s for s in sorted(hyps) if refs.get(s)]
    m = jiwer.process_words([" ".join(refs[s]) for s in songs], [" ".join(hyps[s]) for s in songs])
    n = m.substitutions + m.deletions + m.hits
    return {"wer": (m.substitutions + m.deletions + m.insertions) / n if n else 0.0,
            "I": m.insertions, "D": m.deletions, "S": m.substitutions,
            "ref_words": n, "n_songs": len(songs)}
