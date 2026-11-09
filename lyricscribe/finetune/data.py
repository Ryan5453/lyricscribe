import functools
import json
import logging
import math
import random
from pathlib import Path
from typing import Iterator

import soundfile as sf
from transformers import WhisperTokenizer

logger = logging.getLogger(__name__)

_RNG = random.Random(42)


def _window_has_vocals(
    synced: list[dict], window_start_ms: float, window_end_ms: float
) -> bool:
    """
    True if any non-empty synced line has most of its duration inside
    the window.

    :param synced: Synced lyric lines (``start`` and ``duration`` in ms).
    :param window_start_ms: Window start.
    :param window_end_ms: Window end.
    :return: Whether the window has vocals.
    """
    for line in synced:
        if not line.get("text", "").strip():
            continue
        line_start = line.get("start", 0)
        line_dur = line.get("duration", 0)
        if line_dur <= 0:
            if window_start_ms <= line_start < window_end_ms:
                return True
            continue
        overlap = min(line_start + line_dur, window_end_ms) - max(
            line_start, window_start_ms
        )
        if overlap > 0.5 * line_dur:
            return True
    return False


def _unlabeled_vocal_fraction(
    synced: list[dict],
    segments_raw: list[list],
    window_start_ms: float,
    window_end_ms: float,
) -> float:
    """
    Fraction of sung time inside the window that has no label text.

    Sung time comes from the synced-line spans (every non-empty line,
    clipped to the window); labeled time from ``segments_raw`` spans.
    Windows with no sung time return 0.0 (purely instrumental windows
    are complete by definition).

    :param synced: Synced lyric lines (``start`` and ``duration`` in ms).
    :param segments_raw: Label segments ``(start_ms, end_ms, words)``.
    :param window_start_ms: Window start.
    :param window_end_ms: Window end.
    :return: The unlabeled fraction.
    """
    vocal: list[list[float]] = []
    for line in synced:
        if not line.get("text", "").strip():
            continue
        start = line.get("start", 0)
        dur = line.get("duration", 0)
        if dur <= 0:
            continue
        a = max(start, window_start_ms)
        b = min(start + dur, window_end_ms)
        if b > a:
            vocal.append([a, b])
    if not vocal:
        return 0.0

    vocal.sort()
    merged = [vocal[0]]
    for a, b in vocal[1:]:
        if a <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])

    labeled = [
        (max(s[0], window_start_ms), min(s[1], window_end_ms))
        for s in segments_raw
    ]
    total = 0.0
    unlabeled = 0.0
    for a, b in merged:
        total += b - a
        covered = sum(
            max(0.0, min(b, lb) - max(a, la)) for la, lb in labeled
        )
        unlabeled += max(0.0, (b - a) - covered)
    return unlabeled / total if total > 0 else 0.0


def _segments_for_window(
    synced: list[dict],
    aligned_words: list[dict] | None,
    window_start_ms: float,
    window_end_ms: float,
    line_overlap_threshold: float,
) -> tuple[list[list], bool]:
    """
    Build ``(start_ms, end_ms, [words])`` label segments for one window.

    Uses word-level MFA times when available (a word is kept only if it
    falls fully inside the window), otherwise the synced-line overlap
    heuristic. Returns ``(segments, mfa_gap)`` where ``mfa_gap`` is True
    when the window has vocals per the synced lines but no aligned words
    — an alignment failure, not silence.

    :param synced: Synced lyric lines (``start`` and ``duration`` in ms).
    :param aligned_words: MFA word timings, or ``None`` for the synced-line heuristic.
    :param window_start_ms: Window start.
    :param window_end_ms: Window end.
    :param line_overlap_threshold: Minimum fraction of a synced line inside the window for it to count.
    :return: ``(segments, mfa_gap)``.
    """
    segments_raw: list[list] = []
    if aligned_words is not None:
        kept = [
            w for w in aligned_words
            if w["start"] >= window_start_ms
            and w["start"] + w["duration"] <= window_end_ms
        ]
        for w in kept:
            w_start = w["start"]
            w_end = w["start"] + w["duration"]
            if segments_raw and w_start - segments_raw[-1][1] <= 1000:
                segments_raw[-1][1] = max(segments_raw[-1][1], w_end)
                segments_raw[-1][2].append(w["word"])
            else:
                segments_raw.append([w_start, w_end, [w["word"]]])

        if not segments_raw and _window_has_vocals(
            synced, window_start_ms, window_end_ms
        ):
            return [], True
    else:
        for line in synced:
            if not line.get("text", "").strip():
                continue
            line_start = line.get("start", 0)
            line_dur = line.get("duration", 0)
            if line_dur <= 0:
                if window_start_ms <= line_start < window_end_ms:
                    segments_raw.append(
                        [line_start, line_start,
                         line["text"].strip().split()]
                    )
                continue
            line_end = line_start + line_dur
            overlap_start = max(line_start, window_start_ms)
            overlap_end = min(line_end, window_end_ms)
            overlap = max(0, overlap_end - overlap_start)
            if overlap / line_dur >= line_overlap_threshold:
                segments_raw.append(
                    [max(line_start, window_start_ms),
                     min(line_end, window_end_ms),
                     line["text"].strip().split()]
                )
    return segments_raw, False


def _dedup_segments(segments_raw: list[list], dedup_max_run: int) -> list[list]:
    """
    Cap runs of consecutive identical words (case-insensitive) within each segment and
    drop segments left empty.

    :param segments_raw: ``(start_ms, end_ms, [words])`` segments (modified in place).
    :param dedup_max_run: Longest allowed run (``0`` = no cap).
    :return: The non-empty segments.
    """
    if dedup_max_run > 0:
        for seg in segments_raw:
            deduped: list[str] = []
            for w in seg[2]:
                wl = w.lower()
                if len(deduped) >= dedup_max_run and all(d.lower() == wl for d in deduped[-dedup_max_run:]):
                    continue
                deduped.append(w)
            seg[2] = deduped
    return [s for s in segments_raw if s[2]]


def _relative_segments(segments_raw: list[list], window_start_ms: float, duration: float) -> list[list]:
    """
    Whisper ``segments``: window-relative seconds, clamped and monotonic.

    :param segments_raw: ``(start_ms, end_ms, [words])`` segments.
    :param window_start_ms: Window start.
    :param duration: Window duration in seconds.
    :return: ``[start_s, end_s, text]`` rows.
    """
    segs = []
    prev_end = 0.0
    for s_ms, e_ms, words in segments_raw:
        s = min(max((s_ms - window_start_ms) / 1000.0, prev_end), duration)
        e = min(max((e_ms - window_start_ms) / 1000.0, s), duration)
        segs.append([round(s, 3), round(e, 3), " ".join(words)])
        prev_end = e
    return segs


class LyricsDataset:
    """
    Dataset that iterates over song directories, yielding audio paths
    and transcripts for finetuning.

    Each song directory must contain a ``lyrics.json`` file and at least one
    of the requested audio filenames.
    """

    def __init__(
        self,
        dataset_dir: Path,
        filenames: list[str],
    ):
        """
        Scan *dataset_dir* for songs with ``lyrics.json`` and at least one of *filenames*.

        :param dataset_dir: Root directory containing song subdirectories.
        :param filenames: Audio filenames to look for in each song directory.
            When multiple are provided, one is chosen at random per iteration.
        """
        self.dataset_dir = Path(dataset_dir)
        self.filenames = filenames

        self.songs = []
        for song_dir in sorted(self.dataset_dir.iterdir()):
            if not song_dir.is_dir():
                continue

            lyrics_path = song_dir / "lyrics.json"
            if not lyrics_path.exists():
                continue

            available = [f for f in filenames if (song_dir / f).exists()]
            if not available:
                continue

            self.songs.append({
                "id": song_dir.name,
                "dir": song_dir,
                "available": available,
            })

        logger.info(f"Loaded {len(self.songs)} songs from {dataset_dir} ({', '.join(filenames)})")

    def __len__(self) -> int:
        """
        Return the number of songs in the dataset.

        :return: Song count.
        """
        return len(self.songs)

    def __iter__(self) -> Iterator[dict]:
        """
        Yield dicts with ``song_id``, ``audio_path``, and ``transcript``
        for each song that has a non-empty unsynced transcript.

        :return: Iterator of song dicts.
        """
        for song in self.songs:
            lyrics_path = song["dir"] / "lyrics.json"

            try:
                with open(lyrics_path) as f:
                    lyrics_data = json.load(f)
                transcript = lyrics_data["unsynced"]["data"].strip()

                if not transcript:
                    continue

                audio_file = song["dir"] / _RNG.choice(song["available"])

                yield {
                    "song_id": song["id"],
                    "audio_path": str(audio_file),
                    "transcript": transcript,
                }

            except (json.JSONDecodeError, KeyError, OSError) as e:
                logger.warning(f"Failed to load {song['id']}: {e}")
                continue


def create_manifest(
    dataset: LyricsDataset,
    output_path: Path,
    architecture: str = "parakeet",
    model_name: str | None = None,
    window_seconds: float = 30.0,
    max_tokens: int = 440,
    windows_per_song_multiplier: int = 1,
    line_overlap_threshold: float = 0.7,
    dedup_max_run: int = 0,
    max_unlabeled_vocal_frac: float | None = None,
    max_offset_attempts: int = 8,
    skip_song_ids: set[str] | None = None,
    use_alignment: bool = True,
) -> int:
    """
    Create a manifest JSONL by sampling random fixed-length audio windows
    from each song. Synced lyric lines whose start time falls within a
    window become that window's text label; windows that land on purely
    instrumental regions get an empty text label and are kept as-is so
    the model learns to stay silent during non-vocal audio.

    This matches the inference-time distribution: HF's Whisper pipeline
    slices long songs into arbitrary 30s windows at inference, so
    training must expose the model to arbitrary boundaries too.
    Parakeet and Canary transcribe VAD chunks of up to 30 s, and still
    benefit from seeing a mix of vocal and instrumental contexts during
    training.

    The number of windows per song is proportional to song duration so
    longer songs contribute more samples. All offsets are drawn from the
    module-level ``_RNG`` for reproducibility.

    For Whisper, windows whose tokenized text exceeds *max_tokens* are
    dropped (the 448-token decoder cap). Parakeet/Canary skip this check.

    :param dataset: Dataset instance providing song directories with
        ``lyrics.json`` files.
    :param output_path: Path to write the manifest JSONL file.
    :param architecture: Model architecture name (``"whisper"``,
        ``"parakeet"``, or ``"canary"``).
    :param model_name: HuggingFace model identifier, required for
        ``"whisper"`` to load the tokenizer for token-count checking.
    :param window_seconds: Fixed window duration in seconds.
    :param max_tokens: Whisper decoder token cap. Only applied when
        ``architecture == "whisper"``.
    :param windows_per_song_multiplier: Multiplier on the base
        ``ceil(song_duration / window_seconds)`` window count. Default 1
        gives ~1 random window per ``window_seconds`` of song. Raising
        it toward the epoch count approximates per-epoch randomization
        (the model sees different slices of each song within the
        manifest, just in shuffled order across epochs).
    :param line_overlap_threshold: Minimum fraction of a synced line's
        duration that must fall inside a window for that line to be
        kept as part of the window's label. Used only when a song has
        no MFA alignment in its ``lyrics.json`` (fallback path).
        Guards against the model being asked to transcribe lyrics from
        partial audio (hallucination bias) or to stay silent on
        audible vocals that have no text label (deletion bias).
        Default 0.7 means we keep a line only if at least 70% of its
        duration is inside the window.
    :param dedup_max_run: If > 0, cap runs of consecutive identical
        words in the label at this length. MFA alignments of backing
        vocals can produce runs far longer than real lyrics repeat. Off by
        default; 8 is a reasonable value.
    :param max_unlabeled_vocal_frac: If set, reject windows where more
        than this fraction of sung time (per synced-line spans) has no
        label text — the boundary word/line drops that otherwise teach
        the model to delete audible vocals. Rejected offsets are
        re-drawn up to ``max_offset_attempts`` times so data volume is
        preserved rather than filtered away. ``None`` disables; 0.2 is a
        reasonable value.
    :param max_offset_attempts: Random offsets tried per window before
        giving up when ``max_unlabeled_vocal_frac`` is set.
    :param skip_song_ids: Song IDs (directory names) to exclude entirely,
        e.g. songs whose unsynced/synced references disagree heavily
        (untrustworthy labels). The datasets themselves are never modified;
        callers derive this set from an external artifact (see the
        ``--ref-selfwer-file`` CLI option).
    :param use_alignment: When False, ignore MFA word alignments and build
        labels from synced lines only (the no-alignment fallback path).
        Line text preserves the human word order, which MFA can jumble on
        fast or repetitive passages, at the cost of coarser (line-level)
        timing.

    When a song's ``lyrics.json`` carries an ``alignment`` field (from
    ``lyricscribe dataset align``), word-level MFA times are used
    directly: a word is kept only if its full ``[start, start+duration]``
    falls inside the window. Windows with vocals (per synced lines) but
    no aligned words are skipped — the gap is an alignment failure, not
    silence.

    For Whisper, each entry also carries ``segments``
    (``[start_s, end_s, text]`` relative to window start) for
    timestamp-mode labels.

    :return: Number of manifest entries written.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    tokenizer = None
    if architecture == "whisper":
        if model_name is None:
            raise ValueError(
                "model_name is required for whisper manifest creation"
            )
        logger.info(f"  Loading tokenizer: {model_name}")
        tokenizer = WhisperTokenizer.from_pretrained(model_name)

    window_ms = window_seconds * 1000

    count = 0
    skipped_no_sync = 0
    skipped_too_short = 0
    skipped_too_long = 0
    skipped_mfa_gap = 0
    skipped_incomplete = 0
    skipped_ref_quality = 0
    total = len(dataset)

    with open(output_path, "w") as f:
        for idx, song in enumerate(dataset.songs):
            if (idx + 1) % 500 == 0:
                logger.info(f"  Processing manifest: {idx + 1}/{total}")

            if skip_song_ids and song["id"] in skip_song_ids:
                skipped_ref_quality += 1
                continue

            lyrics_path = song["dir"] / "lyrics.json"
            try:
                with open(lyrics_path) as lf:
                    lyrics_data = json.load(lf)

                synced = lyrics_data.get("synced", {}).get("data", [])
                if not synced:
                    skipped_no_sync += 1
                    continue

                aligned_words = None
                if use_alignment:
                    alignment = lyrics_data.get("alignment")
                    if alignment:
                        aligned_words = alignment.get("words")

                detected_language = lyrics_data["detected_language"]
                audio_file = song["dir"] / _RNG.choice(song["available"])

                try:
                    info = sf.info(str(audio_file))
                    song_duration = info.duration
                except Exception as e:
                    logger.warning(
                        f"Failed to read audio info for {song['id']}: {e}"
                    )
                    continue

                if song_duration < 1.0:
                    skipped_too_short += 1
                    continue

                if song_duration <= window_seconds:
                    num_windows = 1
                    max_offset = 0.0
                else:
                    base_windows = math.ceil(song_duration / window_seconds)
                    num_windows = base_windows * windows_per_song_multiplier
                    max_offset = song_duration - window_seconds

                for _ in range(num_windows):
                    attempts = (
                        max_offset_attempts
                        if max_unlabeled_vocal_frac is not None
                        and max_offset > 0
                        else 1
                    )
                    offset = None
                    segments_raw: list[list] = []
                    fail_reason = None
                    for _attempt in range(attempts):
                        cand = (
                            _RNG.uniform(0.0, max_offset)
                            if max_offset > 0
                            else 0.0
                        )
                        window_start_ms = cand * 1000
                        window_end_ms = window_start_ms + window_ms
                        segments_raw, mfa_gap = _segments_for_window(
                            synced,
                            aligned_words,
                            window_start_ms,
                            window_end_ms,
                            line_overlap_threshold,
                        )
                        if mfa_gap:
                            fail_reason = "mfa_gap"
                            continue
                        if (
                            max_unlabeled_vocal_frac is not None
                            and _unlabeled_vocal_fraction(
                                synced,
                                segments_raw,
                                window_start_ms,
                                window_end_ms,
                            ) > max_unlabeled_vocal_frac
                        ):
                            fail_reason = "incomplete"
                            continue
                        offset = cand
                        break

                    if offset is None:
                        if fail_reason == "mfa_gap":
                            skipped_mfa_gap += 1
                        else:
                            skipped_incomplete += 1
                        continue

                    segments_raw = _dedup_segments(segments_raw, dedup_max_run)
                    text = " ".join(
                        w for seg in segments_raw for w in seg[2]
                    ).strip()

                    if tokenizer is not None and text:
                        token_count = len(tokenizer(text).input_ids)
                        if token_count > max_tokens:
                            skipped_too_long += 1
                            continue

                    duration = min(
                        window_seconds, song_duration - offset
                    )
                    if duration <= 0.1:
                        continue

                    entry = {
                        "audio_filepath": str(audio_file),
                        "offset": offset,
                        "duration": duration,
                        "text": text,
                        "language": detected_language,
                    }
                    if architecture == "whisper" and segments_raw:
                        entry["segments"] = _relative_segments(segments_raw, window_start_ms, duration)

                    f.write(json.dumps(entry) + "\n")
                    count += 1

            except (json.JSONDecodeError, KeyError, OSError) as e:
                logger.warning(f"Failed to process {song['id']}: {e}")
                continue

    logger.info(
        f"Created manifest with {count} windows "
        f"({skipped_ref_quality} songs skipped for untrusted references, "
        f"{skipped_no_sync} songs skipped for missing synced lyrics, "
        f"{skipped_too_short} songs skipped for being too short, "
        f"{skipped_too_long} windows dropped for exceeding {max_tokens} tokens, "
        f"{skipped_mfa_gap} windows dropped for MFA alignment gaps over vocals, "
        f"{skipped_incomplete} windows dropped for unlabeled vocals after "
        f"{max_offset_attempts} offset attempts)"
    )
    return count


def write_subset_manifest(
    source: Path,
    dest: Path,
    size: int,
    seed: int = 42,
) -> int:
    """
    Copy a deterministic shuffled subset of a JSONL manifest.

    Used for the during-training validation subset: we want every
    trainer to evaluate on the exact same set of samples each epoch, so we
    materialize one shared subset manifest at setup time instead of
    letting each trainer pick its own window (``torch.Subset`` vs
    ``limit_val_batches``, which produce different sizes *and*
    different contents).

    The shuffle is seeded so re-running setup on the same full manifest
    always produces the same subset — reproducible, and subsetting a
    subset is a no-op.

    :param source: Path to the full manifest (JSONL).
    :param dest: Path to write the subset to (JSONL).
    :param size: Maximum number of entries in the subset. If the source
        has fewer entries, the whole thing is copied.
    :param seed: Shuffle seed (default 42).
    :return: Number of entries written to the subset manifest.
    """
    with open(source) as f:
        lines = f.readlines()

    rng = random.Random(seed)
    rng.shuffle(lines)
    lines = lines[:size]

    dest.parent.mkdir(parents=True, exist_ok=True)
    with open(dest, "w") as f:
        f.writelines(lines)

    return len(lines)


# Data keys of the experiment that generated a manifest; recorded with every copy.
DATA_KEYS = (
    "filenames",
    "train_dir",
    "val_dir",
    "dedup_max_run",
    "max_unlabeled_vocal_frac",
    "use_alignment",
    "windows_per_song",
    "line_overlap_threshold",
    "ref_selfwer_file",
    "max_ref_self_wer",
    "audio_domain_mode",
)


def source_data_keys(source: Path) -> dict:
    """
    The data-construction keys recorded for a manifest directory: ``data_keys.json`` written by
    :func:`prepare_manifests`, else the generating experiment's ``config.json``.

    :param source: Experiment or prepared-manifest directory.
    :return: The keys (with their provenance under ``"source"``).
    """
    prepared = source / "data_keys.json"
    if prepared.exists():
        return json.loads(prepared.read_text())
    config = json.loads((source / "config.json").read_text())
    keys = {k: config.get(k) for k in DATA_KEYS}
    if config.get("manifests_from"):
        upstream = Path(config["manifests_from"])
        keys = source_data_keys(upstream) if (upstream / "config.json").exists() or (upstream / "data_keys.json").exists() else keys
    keys["source"] = str(source)
    return keys


def _song_id(row: dict) -> str:
    """
    Song directory name of a manifest row.

    :param row: Manifest row.
    :return: Song id.
    """
    return Path(row["audio_filepath"]).parent.name


@functools.lru_cache(maxsize=4096)
def _lyrics(path: str) -> dict:
    """
    A song's ``lyrics.json``, cached (manifests hold several windows per song).

    :param path: ``lyrics.json`` path.
    :return: Parsed lyrics.
    """
    return json.loads(Path(path).read_text())


def derive_whisper_segments(row: dict, keys: dict, window_seconds: float = 30.0) -> list[list] | None:
    """
    Rebuild a window's Whisper ``segments`` from its song's lyrics with the generating
    recipe, so a manifest built for another family gains timestamp labels with identical
    windows and text.

    :param row: Manifest row (``audio_filepath``, ``offset``, ``duration``, ``text``).
    :param keys: Source data keys (``use_alignment``, ``line_overlap_threshold``, ``dedup_max_run``).
    :param window_seconds: Window length the manifest was drawn with.
    :return: Segments, ``[]`` for an empty label, or ``None`` if the rebuilt text differs.
    """
    lyrics = _lyrics(str(Path(row["audio_filepath"]).parent / "lyrics.json"))
    synced = lyrics.get("synced", {}).get("data", [])
    aligned = None
    if keys["use_alignment"] and lyrics.get("alignment"):
        aligned = lyrics["alignment"].get("words")
    start_ms = float(row.get("offset", 0.0)) * 1000
    segments_raw, _ = _segments_for_window(
        synced, aligned, start_ms, start_ms + window_seconds * 1000, float(keys["line_overlap_threshold"])
    )
    segments_raw = _dedup_segments(segments_raw, int(keys["dedup_max_run"] or 0))
    text = " ".join(w for seg in segments_raw for w in seg[2]).strip()
    if text != row["text"]:
        return None
    return _relative_segments(segments_raw, start_ms, float(row["duration"]))


def whisper_label_tokens(row: dict, tokenizer) -> int:
    """
    Whisper decoder length of a window's no-timestamp label (prefix, text, end token).

    :param row: Manifest row.
    :param tokenizer: Whisper tokenizer.
    :return: Token count.
    """
    tokenizer.set_prefix_tokens(language=row["language"], task="transcribe", predict_timestamps=False)
    text = " " + row["text"] if row["text"] else row["text"]
    return len(tokenizer(text).input_ids)


def _read_rows(path: Path) -> list[dict]:
    """
    Read a JSONL manifest.

    :param path: Manifest path.
    :return: Rows.
    """
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def _write_rows(path: Path, rows: list[dict]) -> None:
    """
    Write a JSONL manifest.

    :param path: Manifest path.
    :param rows: Rows.
    """
    with open(path, "w") as f:
        f.writelines(json.dumps(r) + "\n" for r in rows)


def prepare_manifests(
    source: Path,
    output_dir: Path,
    exclude_songs: set[str] | None = None,
    repoint: str | None = None,
    whisper_segments: bool = False,
    whisper_model: str | None = None,
    whisper_cap: int = 448,
    eval_exclude_songs: set[str] | None = None,
    eval_subset_size: int = 200,
    eval_seed: int = 42,
    min_window_seconds: float = 0.5,
    recipe: dict | None = None,
) -> dict:
    """
    Build one shared, family-agnostic manifest set from an experiment's manifests: the same
    windows and text for every family, minus excluded songs, re-pointed at another audio file,
    with Whisper timestamp ``segments`` rebuilt from the lyrics, and with windows whose label
    exceeds Whisper's decoder cap dropped for every family. Writes ``train_manifest.jsonl``,
    ``val_manifest.jsonl``, ``val_subset_manifest.jsonl``, ``data_keys.json`` and
    ``prepare_report.json`` to *output_dir*.

    :param source: Experiment (or prepared) directory holding the manifests.
    :param output_dir: Where to write the prepared set.
    :param exclude_songs: Song ids removed from training (e.g. near-duplicates of eval songs).
    :param repoint: Audio filename replacing each row's ``audio_filepath`` basename.
    :param whisper_segments: Rebuild Whisper ``segments`` for every training row.
    :param whisper_model: Tokenizer for the Whisper label-cap check (``None`` skips it).
    :param whisper_cap: Whisper decoder positions.
    :param eval_exclude_songs: Validation songs the eval-loss subset must avoid (e.g. tune300).
    :param eval_subset_size: Eval-loss windows.
    :param eval_seed: Eval-loss subset seed.
    :param min_window_seconds: Windows with less audio than this after re-pointing are dropped.
    :param recipe: Data keys the source config doesn't record (e.g. from its launch script),
        recorded under ``recipe_overrides``.
    :return: The report (counts per step).
    :raises FileExistsError: If *output_dir* already holds a train manifest.
    :raises ValueError: If segment rebuilding needs a recipe key nobody recorded.
    """
    if (output_dir / "train_manifest.jsonl").exists():
        raise FileExistsError(f"{output_dir} already holds manifests")
    keys = source_data_keys(source)
    for key, value in (recipe or {}).items():
        if keys.get(key) is not None and keys[key] != value:
            raise ValueError(f"recipe {key}={value!r} contradicts the source's recorded {keys[key]!r}")
        keys[key] = value
    keys["recipe_overrides"] = recipe or {}
    if whisper_segments:
        missing = [k for k in ("use_alignment", "line_overlap_threshold", "dedup_max_run") if keys.get(k) is None]
        if missing:
            raise ValueError(f"{source} records no {missing}; pass them as --recipe key=value")
    output_dir.mkdir(parents=True, exist_ok=True)
    report: dict = {"source": str(source)}
    train = _read_rows(source / "train_manifest.jsonl")
    val = _read_rows(source / "val_manifest.jsonl") if (source / "val_manifest.jsonl").exists() else []
    report["train_in"] = len(train)
    report["val_in"] = len(val)

    if exclude_songs:
        before = len(train)
        dropped_songs = {_song_id(r) for r in train} & exclude_songs
        train = [r for r in train if _song_id(r) not in exclude_songs]
        report["excluded_songs"] = len(dropped_songs)
        report["excluded_windows"] = before - len(train)

    if repoint:
        info_cache: dict[str, float | None] = {}

        def duration(path: str) -> float | None:
            """
            Audio duration in seconds, cached.

            :param path: Audio path.
            :return: Duration, or ``None`` if unreadable.
            """
            if path not in info_cache:
                try:
                    info_cache[path] = sf.info(path).duration
                except Exception:
                    info_cache[path] = None
            return info_cache[path]

        missing, short_file, short_window = set(), set(), 0
        for rows, name in ((train, "train"), (val, "val")):
            kept = []
            for r in rows:
                old = r["audio_filepath"]
                new = str(Path(old).with_name(repoint))
                new_dur, old_dur = duration(new), duration(old)
                if new_dur is None:
                    missing.add(new)
                    continue
                if old_dur is not None and new_dur < old_dur:
                    short_file.add(new)
                    continue
                if new_dur - float(r.get("offset", 0.0)) < min_window_seconds:
                    short_window += 1
                    continue
                kept.append({**r, "audio_filepath": new})
            report[f"{name}_repointed"] = len(kept)
            rows[:] = kept
        report["repoint_missing_files"] = len(missing)
        report["repoint_shorter_than_source"] = sorted(short_file)[:50]
        report["repoint_shorter_count"] = len(short_file)
        report["repoint_short_windows"] = short_window

    if whisper_model:
        tokenizer = WhisperTokenizer.from_pretrained(whisper_model)
        over = [r for r in train if whisper_label_tokens(r, tokenizer) > whisper_cap]
        over_ids = {id(r) for r in over}
        train = [r for r in train if id(r) not in over_ids]
        report["whisper_cap"] = whisper_cap
        report["over_cap_windows"] = len(over)

    if whisper_segments:
        mismatched = 0
        for r in train:
            segs = derive_whisper_segments(r, keys)
            if segs is None:
                mismatched += 1
                r.pop("segments", None)
            elif segs:
                r["segments"] = segs
        report["segments_mismatch"] = mismatched
        report["segments_rows"] = sum(1 for r in train if r.get("segments"))

    pool = [r for r in val if _song_id(r) not in (eval_exclude_songs or set())]
    subset = random.Random(eval_seed).sample(pool, min(eval_subset_size, len(pool)))
    report["val_subset"] = len(subset)
    report["val_subset_songs"] = len({_song_id(r) for r in subset})
    report["train_out"] = len(train)
    report["train_songs"] = len({_song_id(r) for r in train})

    _write_rows(output_dir / "train_manifest.jsonl", train)
    if val:
        _write_rows(output_dir / "val_manifest.jsonl", val)
        _write_rows(output_dir / "val_subset_manifest.jsonl", subset)
    keys = {**keys, "prepared_from": str(source), "repoint": repoint, "excluded_songs": report.get("excluded_songs", 0)}
    if repoint:
        keys["filenames"] = [repoint]
    (output_dir / "data_keys.json").write_text(json.dumps(keys, indent=2))
    (output_dir / "prepare_report.json").write_text(json.dumps(report, indent=2))
    logger.info(f"Prepared manifests in {output_dir}: {json.dumps(report)}")
    return report
