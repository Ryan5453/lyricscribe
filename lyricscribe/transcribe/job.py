import fcntl
import hashlib
import importlib.metadata
import json
import logging
import os
import random
import re
import subprocess
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import torch
from silero_vad import load_silero_vad

from lyricscribe.ensemble.deloop import deloop
from lyricscribe.jobs import update_status
from lyricscribe.transcribe.adapter import TrainRecordMismatch
from lyricscribe.transcribe.audio_llm import AudioLLMTranscriber
from lyricscribe.transcribe.base import MAX_NEW_TOKENS, Decode, Transcriber
from lyricscribe.transcribe.canary import CanaryTranscriber
from lyricscribe.transcribe.cohere import CohereTranscriber
from lyricscribe.transcribe.granite import GraniteTranscriber
from lyricscribe.transcribe.nemotron import NemotronTranscriber
from lyricscribe.transcribe.parakeet import ParakeetTranscriber
from lyricscribe.transcribe.qwen_asr import Qwen3ASRTranscriber
from lyricscribe.transcribe.whisper import WhisperTranscriber

logger = logging.getLogger(__name__)


BACKENDS = (
    "whisper",
    "qwen3_asr",
    "qwen3_omni",
    "musicflamingo",
    "cohere",
    "granite",
    "nemotron",
    "parakeet",
    "canary",
)

# Substrings of hub ids / paths, checked in order.
_NAME_HINTS = (
    ("whisper", "whisper"),
    ("qwen3-asr", "qwen3_asr"),
    ("qwen3_asr", "qwen3_asr"),
    ("qwen3-omni", "qwen3_omni"),
    ("qwen3_omni", "qwen3_omni"),
    ("music-flamingo", "musicflamingo"),
    ("musicflamingo", "musicflamingo"),
    ("cohere-transcribe", "cohere"),
    ("granite-speech", "granite"),
    ("nemotron-3.5-transcribe", "nemotron"),
    ("parakeet", "parakeet"),
    ("canary", "canary"),
)
# ``model_type`` / architecture strings in a local checkpoint's config.json.
_CONFIG_HINTS = (
    ("whisper", "whisper"),
    ("qwen3_asr", "qwen3_asr"),
    ("qwen3omni", "qwen3_omni"),
    ("qwen3_omni", "qwen3_omni"),
    ("musicflamingo", "musicflamingo"),
    ("cohere_asr", "cohere"),
    ("granite_speech", "granite"),
    ("nemotron3_5transcribe", "nemotron"),
    ("parakeet", "parakeet"),
    ("canary", "canary"),
)


def detect_backend(model_name: str) -> str:
    """
    Pick the transcription backend for a model id or checkpoint path.

    Hub ids are matched by name; local checkpoint dirs (finetune outputs
    rarely carry the family in their path) by their ``config.json``
    ``model_type`` / ``architectures``.

    :param model_name: Hub id or local checkpoint dir.
    :return: One of :data:`BACKENDS`.
    :raises ValueError: For a ``.nemo`` file or a model no backend claims.
    """
    if model_name.endswith(".nemo"):
        raise ValueError(
            f"{model_name}: NeMo checkpoints are not supported; convert it to transformers format first"
        )
    lowered = model_name.lower()
    for hint, backend in _NAME_HINTS:
        if hint in lowered:
            return backend
    cfg_path = Path(model_name) / "config.json"
    if cfg_path.exists():
        try:
            with open(cfg_path) as f:
                cfg = json.load(f)
        except (json.JSONDecodeError, OSError):
            cfg = {}
        names = [str(cfg.get("model_type", ""))] + [
            str(a) for a in cfg.get("architectures") or []
        ]
        joined = " ".join(names).lower()
        for hint, backend in _CONFIG_HINTS:
            if hint in joined:
                return backend
    raise ValueError(f"Cannot detect the backend for {model_name!r}; pass one of {BACKENDS}")


def backend_prompt_modes(backend: str) -> tuple[str, ...]:
    """
    The prompt modes a backend's transcriber accepts, its default first.

    :param backend: One of :data:`BACKENDS`.
    :return: ``Transcriber.prompt_modes`` of that backend.
    """
    classes = {
        "whisper": WhisperTranscriber, "qwen3_asr": Qwen3ASRTranscriber, "qwen3_omni": AudioLLMTranscriber,
        "musicflamingo": AudioLLMTranscriber, "cohere": CohereTranscriber,
        "granite": GraniteTranscriber, "nemotron": NemotronTranscriber, "canary": CanaryTranscriber,
        "parakeet": ParakeetTranscriber,
    }
    return classes[backend].prompt_modes


def _create_transcriber(model_name: str, batch_size: int, config: dict | None = None) -> Transcriber:
    """
    Create the transcriber for a job.

    :param model_name: Hub id or local checkpoint path.
    :param batch_size: Batch size for inference.
    :param config: Job config; supplies ``backend`` (overrides detection)
        and backend options (``max_new_tokens``, ``adapter``,
        ``auto_language``, ``prompt_mode``, ``dtype``). ``prompt_mode`` is checked
        against the backend's ``prompt_modes``; Parakeet takes none.
    :return: An unloaded :class:`Transcriber` instance.
    """
    config = config or {}
    backend = config.get("backend") or detect_backend(model_name)
    max_new_tokens = int(config.get("max_new_tokens") or MAX_NEW_TOKENS)
    prompt_mode = config.get("prompt_mode")
    if backend == "whisper":
        return WhisperTranscriber(model_name, batch_size, dtype=config.get("dtype"), prompt_mode=prompt_mode)
    if backend == "qwen3_asr":
        return Qwen3ASRTranscriber(
            model_name,
            batch_size,
            max_new_tokens=max_new_tokens,
            auto_language=bool(config.get("auto_language", False)),
            adapter=config.get("adapter"),
            prompt_mode=prompt_mode,
        )
    if backend in ("qwen3_omni", "musicflamingo"):
        return AudioLLMTranscriber(
            model_name,
            backend,
            batch_size,
            max_new_tokens=max_new_tokens,
            adapter=config.get("adapter"),
            prompt_mode=prompt_mode or "legacy",
            dtype=config.get("dtype"),
        )
    if backend == "cohere":
        return CohereTranscriber(model_name, batch_size, max_new_tokens=max_new_tokens, prompt_mode=prompt_mode)
    if backend == "granite":
        return GraniteTranscriber(model_name, batch_size, max_new_tokens=max_new_tokens, prompt_mode=prompt_mode)
    if backend == "nemotron":
        return NemotronTranscriber(
            model_name, batch_size, max_new_tokens=max_new_tokens, adapter=config.get("adapter"), prompt_mode=prompt_mode
        )
    if backend == "canary":
        return CanaryTranscriber(
            model_name, batch_size, max_new_tokens=max_new_tokens, dtype=config.get("dtype"), prompt_mode=prompt_mode
        )
    if prompt_mode not in (None, "native"):
        raise ValueError(f"{backend} takes no prompt; got prompt_mode {prompt_mode!r}")
    if backend == "parakeet":
        return ParakeetTranscriber(model_name, batch_size, dtype=config.get("dtype"), max_new_tokens=max_new_tokens)
    raise ValueError(f"Unknown backend {backend!r}; expected one of {BACKENDS}")


def _append_result(
    output_path: Path,
    song_id: str,
    audio_file: str,
    transcription: str | None,
    model_name: str,
    duration_seconds: float,
    error: str | None,
    transcription_raw: str | None = None,
) -> None:
    """
    Append a single transcription result as a JSON line to the output file.

    Writes under an exclusive lock: concurrent chunks share the file and
    ``O_APPEND`` alone is not atomic on NFS.

    :param output_path: Path to the JSONL output file.
    :param song_id: Identifier for the song.
    :param audio_file: Absolute path to the audio file.
    :param transcription: Transcribed text, or ``None`` on failure.
    :param model_name: Model identifier used for transcription.
    :param duration_seconds: Time taken for transcription.
    :param error: Error message if failed, or ``None``.
    :param transcription_raw: Pre-repair text when a post-hoc repair
        (de-looping) changed it.
    """
    result = {
        "song_id": song_id,
        "audio_file": audio_file,
        "transcription": transcription,
        "model_name": model_name,
        "duration_seconds": round(duration_seconds, 2),
        "error": error,
    }
    if transcription_raw is not None:
        result["transcription_raw"] = transcription_raw
    with open(output_path, "a") as f:
        fcntl.lockf(f, fcntl.LOCK_EX)
        try:
            f.write(json.dumps(result) + "\n")
            f.flush()
            os.fsync(f.fileno())
        finally:
            fcntl.lockf(f, fcntl.LOCK_UN)


def parse_song_dirname(name: str) -> tuple[str, str]:
    """
    Split a ``A_-_B`` song directory name into its two cleaned halves.

    The halves are returned without assigning roles: MUSDB-ALT dirs are
    ``Artist_-_Title`` but Jam-ALT mixes both orders (``Avercage_-_Embers``
    vs ``10._Disparan_-_criatura``), so neither side can be assumed to be
    the artist. Leading track numbers (``10.``) are stripped.

    :param name: Song directory name.
    :return: ``(left, right)``; ``right`` is empty when there is no separator.
    """
    left, _, right = name.partition("_-_")

    def clean(s: str) -> str:
        """
        Replace underscores, strip a leading track number and collapse whitespace.

        :param s: One half of the directory name.
        :return: The cleaned half.
        """
        s = re.sub(r"^\d+\.?\s*", "", s.replace("_", " "))
        return re.sub(r"\s+", " ", s).strip()

    return clean(left), clean(right)


def metadata_prompt(name: str, backend: str = "whisper") -> str:
    """
    Role-free per-song prompt built from a song directory name.

    Whisper gets a decoder-prefix prompt; Qwen3-ASR gets a ``context``
    (system message) prompt.

    :param name: Song directory name.
    :param backend: Transcription backend (see :func:`detect_backend`).
    :return: Prompt text, e.g. ``Lyrics of "Avercage - Embers".``
    """
    left, right = parse_song_dirname(name)
    if backend == "qwen3_asr":
        if not right:
            return f"Song metadata: {left}. Transcribe the sung lyrics."
        return f"Song metadata: {left} - {right}. Transcribe the sung lyrics."
    return f'Lyrics of "{left} - {right}".' if right else f'Lyrics of "{left}".'


def setup_job(
    directories: list[Path],
    job_dir: Path,
    filename: str,
    model: str,
    num_chunks: int,
    batch_size: int,
    vad: bool,
    chunked: bool = False,
    lyrics_filename: str | None = None,
    vad_filename: str | None = None,
    vad_method: str = "silero",
    repetition_penalty: float = 1.0,
    no_repeat_ngram_size: int = 0,
    num_beams: int = 1,
    condition_on_prev: bool = True,
    temperature_fallback: bool = False,
    compression_ratio_threshold: float | None = None,
    logprob_threshold: float | None = None,
    prompt_from_dirname: bool = False,
    backend: str | None = None,
    max_new_tokens: int | None = None,
    adapter: str | None = None,
    auto_language: bool = False,
    deloop_max_run: int | None = None,
    prompt_mode: str | None = None,
    dtype: str | None = None,
) -> None:
    """
    Initialize a transcription job by scanning directories and writing
    a config file plus per-chunk JSON manifests.

    :param directories: Root directories containing subdirectories to process.
    :param job_dir: Directory to create for job files.
    :param filename: Audio filename to target within each subdirectory.
    :param model: Hub id or local checkpoint path.
    :param num_chunks: Number of chunks to split the work into.
    :param batch_size: Batch size for inference.
    :param vad: Whether to enable VAD-based segmentation.
    :param chunked: Whether to use fixed-length chunked inference.
    :param lyrics_filename: Optional JSON filename (e.g. ``lyrics.json``)
        to read ``detected_language`` from in each subdirectory.
    :param vad_filename: Optional audio filename to use as VAD source
        (timestamps from this file, transcription from ``filename``).
    :param vad_method: ``"silero"`` or ``"rms"``.
    :param repetition_penalty: Decoder repetition penalty on generated
        tokens. ``1.0`` disables.
    :param no_repeat_ngram_size: Decoder n-gram ban on generated tokens.
        ``0`` disables.
    :param num_beams: Beam width; ``1`` is greedy.
    :param condition_on_prev: Whisper long-form ``condition_on_prev_tokens``.
    :param temperature_fallback: Whisper temperature fallback.
    :param compression_ratio_threshold: Fallback trigger.
    :param logprob_threshold: Fallback trigger.
    :param prompt_from_dirname: Derive a per-song prompt from the song
        directory name (see :func:`metadata_prompt`): a decoder prompt for
        Whisper, the ``context`` for Qwen3-ASR.
    :param backend: Force a backend instead of :func:`detect_backend`.
    :param max_new_tokens: Per-chunk generation cap for the autoregressive
        backends (default :data:`MAX_NEW_TOKENS`).
    :param adapter: PEFT adapter dir merged into Qwen3-ASR, Qwen3-Omni, Nemotron or Music Flamingo.
    :param auto_language: Qwen3-ASR: ignore the provided language and let
        the model identify it.
    :param deloop_max_run: If set, collapse any short cycle repeated more
        than this many times in each transcription (post-hoc repair; pairs
        with ``no_repeat_ngram_size=0``). The raw text is kept as
        ``transcription_raw``. Prefer scoring-time de-looping
        (``evaluate_job(deloop_max_run=...)``), which runs on normalised words.
    :param prompt_mode: The backend's prompt (``Transcriber.prompt_modes``): ``native``
        or ``shared`` for every backend with a text input, plus ``asr``/``lyrics``/``legacy``
        for the audio-LLMs and ``asr`` for Granite and Nemotron. ``None`` is the backend default.
    :param dtype: Compute dtype override (Whisper, Parakeet, Canary, Music Flamingo).
    """
    if backend is not None and backend not in BACKENDS:
        raise ValueError(f"Unknown backend {backend!r}; expected one of {BACKENDS}")
    resolved_backend = backend or detect_backend(model)
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

    # The unloaded backend refuses a prompt_mode or dtype it doesn't have before anything is written.
    _create_transcriber(model, batch_size, {
        "backend": resolved_backend, "prompt_mode": prompt_mode, "dtype": dtype, "adapter": adapter,
        "auto_language": auto_language, "max_new_tokens": max_new_tokens,
    })

    job_dir.mkdir(parents=True, exist_ok=True)

    results_path = job_dir / "results.jsonl"
    if results_path.exists():
        results_path.unlink()

    config = {
        "directories": [str(d) for d in directories],
        "filename": filename,
        "model": model,
        "batch_size": batch_size,
        "vad": vad,
        "vad_method": vad_method,
        "chunked": chunked,
        "lyrics_filename": lyrics_filename,
        "vad_filename": vad_filename,
        "repetition_penalty": repetition_penalty,
        "no_repeat_ngram_size": no_repeat_ngram_size,
        "num_beams": num_beams,
        "condition_on_prev": condition_on_prev,
        "temperature_fallback": temperature_fallback,
        "compression_ratio_threshold": compression_ratio_threshold,
        "logprob_threshold": logprob_threshold,
        "prompt_from_dirname": prompt_from_dirname,
        "backend": resolved_backend,
        "max_new_tokens": max_new_tokens,
        "adapter": adapter,
        "auto_language": auto_language,
        "deloop_max_run": deloop_max_run,
        "prompt_mode": prompt_mode,
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

            language = None
            if lyrics_filename:
                lyrics_path = subdir / lyrics_filename
                if lyrics_path.exists():
                    try:
                        with open(lyrics_path) as lf:
                            lyrics_data = json.load(lf)
                        language = lyrics_data.get("detected_language")
                    except (json.JSONDecodeError, OSError) as e:
                        logger.warning(
                            f"{name}: failed to read {lyrics_filename}: {e}"
                        )

            vad_path = None
            if vad_filename:
                vad_file = subdir / vad_filename
                if vad_file.exists():
                    vad_path = str(vad_file)

            prompt = (
                metadata_prompt(name, resolved_backend) if prompt_from_dirname else None
            )

            files.append(
                {
                    "name": name,
                    "input_path": str(input_file),
                    "vad_path": vad_path,
                    "language": language,
                    "prompt": prompt,
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


def git_commit(root: Path) -> str | None:
    """
    The commit the code runs from, suffixed ``_dirty`` when the checkout has
    uncommitted changes.

    :param root: Repository root.
    :return: ``<sha>`` or ``<sha>_dirty``; ``None`` outside a git checkout.
    """
    try:
        sha = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
        status = subprocess.run(
            ["git", "-C", str(root), "status", "--porcelain", "--", "lyricscribe", "pyproject.toml", "uv.lock"],
            capture_output=True, text=True, check=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        return None
    return f"{sha}_dirty" if status.strip() else sha


def environment_record() -> dict:
    """
    Software and hardware a job runs on.

    :return: Code root, commit (:func:`git_commit`), ``uv.lock`` hash, package versions and GPU.
    """
    root = Path(__file__).resolve().parents[2]
    lock = root / "uv.lock"
    versions = {}
    for package in ("torch", "torchaudio", "transformers", "peft", "unblend", "alt-eval", "jiwer", "numpy"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    gpu = None
    if torch.cuda.is_available():
        gpu = {"name": torch.cuda.get_device_name(0), "capability": list(torch.cuda.get_device_capability(0))}
    return {
        "code_root": str(root),
        "commit": git_commit(root),
        "uv_lock_sha256": hashlib.sha256(lock.read_bytes()).hexdigest() if lock.exists() else None,
        "versions": versions,
        "gpu": gpu,
        "host": os.uname().nodename,
    }


def _write_effective(job_dir: Path, chunk_id: int, env: dict, transcriber: Transcriber, decode: Decode) -> None:
    """
    Record what the chunk actually ran with: environment (including the
    commit), backend state and the last ``generate()`` call's kwargs.

    :param job_dir: Job directory.
    :param chunk_id: 1-based chunk id.
    :param env: :func:`environment_record` output.
    :param transcriber: Loaded transcriber.
    :param decode: Decoding settings in force.
    """
    record = {
        "chunk_id": chunk_id,
        "written_at": datetime.now(timezone.utc).isoformat(),
        "environment": env,
        "decode": decode.as_dict(),
        "backend": transcriber.effective(),
    }
    path = job_dir / f"effective_chunk_{chunk_id}.json"
    path.write_text(json.dumps(record, indent=2, default=str) + "\n")


def process_chunk(job_dir: Path, chunk_id: int) -> None:
    """
    Process one chunk of the transcription job.

    Loads the appropriate model and transcribes all pending files
    assigned to the given chunk. Results are appended to a JSONL file
    in the job directory.

    A ``torch.cuda.OutOfMemoryError`` stops the chunk with the song still
    pending: the batch size is never changed silently, so rerun with a
    smaller ``batch_size``. ``effective_chunk_<id>.json`` records what ran.

    :param job_dir: Path to the job directory.
    :param chunk_id: 1-based chunk identifier.
    """
    with open(job_dir / "config.json") as f:
        config = json.load(f)
    env = environment_record()
    model_name = config["model"]
    batch_size = config.get("batch_size", 1)
    if batch_size == 0:
        batch_size = 1  # Legacy: treat old "auto" (0) as default (1)
    use_vad = config.get("vad", False)
    use_chunked = config.get("chunked", False)
    vad_method = config.get("vad_method", "silero")
    vad_filename = config.get("vad_filename")
    decode = Decode(
        num_beams=int(config.get("num_beams") or 1),
        repetition_penalty=float(config.get("repetition_penalty", 1.0) or 1.0),
        no_repeat_ngram_size=int(config.get("no_repeat_ngram_size", 0) or 0),
    )
    condition_on_prev = bool(config.get("condition_on_prev", True))
    temperature_fallback = bool(config.get("temperature_fallback", False))
    compression_ratio_threshold = config.get("compression_ratio_threshold")
    logprob_threshold = config.get("logprob_threshold")

    if use_vad and vad_method == "rms" and not vad_filename:
        raise ValueError(
            "vad_method='rms' requires vad_filename to point at a separated "
            "vocal track or stem; RMS amplitude on a mixed track is dominated "
            "by accompaniment."
        )

    deloop_max_run = config.get("deloop_max_run")
    transcriber = _create_transcriber(model_name, batch_size, config)
    transcriber.load()
    transcriber.check_decode(decode)
    _write_effective(job_dir, chunk_id, env, transcriber, decode)

    logger.info(f"Using batch_size={transcriber.batch_size}, decode={decode.as_dict()}")

    vad_model = None
    if use_vad and vad_method == "silero":
        vad_model = load_silero_vad()
        logger.info("Loaded Silero VAD model")
    elif use_vad and vad_method == "rms":
        logger.info("Using RMS-amplitude VAD on the (separated) vocal track")

    with open(job_dir / f"chunk_{chunk_id}.json") as f:
        chunk_data = json.load(f)
    pending = [f for f in chunk_data["files"] if f["status"] == "pending"]

    if not pending:
        logger.info(f"Chunk {chunk_id}: no pending files")
        return

    output_path = job_dir / "results.jsonl"

    success_count = 0
    failed_count = 0

    for entry in pending:
        name = entry["name"]
        input_path = entry["input_path"]

        if not Path(input_path).exists():
            logger.warning(f"{name}: input file not found at {input_path}")
            update_status(
                job_dir,
                chunk_id,
                chunk_data,
                name,
                "failed",
                0.0,
                f"File not found: {input_path}",
            )
            failed_count += 1
            continue

        start_time = time.time()

        try:
            text = transcriber.transcribe(
                input_path,
                use_vad=use_vad,
                vad_model=vad_model,
                use_chunked=use_chunked,
                decode=decode,
                condition_on_prev=condition_on_prev,
                temperature_fallback=temperature_fallback,
                compression_ratio_threshold=compression_ratio_threshold,
                logprob_threshold=logprob_threshold,
                language=entry.get("language"),
                vad_source=entry.get("vad_path"),
                vad_method=vad_method,
                initial_prompt=entry.get("prompt"),
            )

            raw = None
            if deloop_max_run:
                words = text.split()
                fixed = deloop(words, max_run=int(deloop_max_run))
                if fixed != words:
                    raw = text
                text = " ".join(fixed)

            duration = time.time() - start_time

            _append_result(
                output_path, name, input_path, text, model_name, duration, None, raw
            )
            update_status(
                job_dir, chunk_id, chunk_data, name, "success", duration, None
            )
            logger.info(f"{name} completed in {duration:.2f}s")
            if success_count == 0:
                _write_effective(job_dir, chunk_id, env, transcriber, decode)
            success_count += 1

        except torch.cuda.OutOfMemoryError:
            logger.error(f"{name}: CUDA OOM at batch_size={transcriber.batch_size}; left pending")
            raise
        except TrainRecordMismatch:
            logger.error(f"{name}: decode contradicts the checkpoint's training record; left pending")
            raise
        except Exception as e:
            duration = time.time() - start_time
            error_msg = str(e)

            _append_result(
                output_path,
                name,
                input_path,
                None,
                model_name,
                duration,
                error_msg,
            )
            update_status(
                job_dir, chunk_id, chunk_data, name, "failed", duration, error_msg
            )
            logger.error(f"{name} failed: {e}")
            traceback.print_exc()
            failed_count += 1

    logger.info(f"Chunk {chunk_id}: {success_count} success, {failed_count} failed")
