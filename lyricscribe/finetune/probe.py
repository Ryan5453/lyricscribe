import gc
import logging
import random
import statistics
import time
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import soxr
import torch

from lyricscribe.finetune.config import detect_architecture

logger = logging.getLogger(__name__)


def _load_audio_samples(
    dataset_dir: Path,
    filename: str,
    n: int,
    max_duration_s: float,
    seed: int = 42,
) -> list[np.ndarray]:
    """
    Return ``n`` audio samples as float32 numpy arrays at 16 kHz mono,
    each padded or truncated to a fixed length so every probe batch has
    the same ``(bs, target_len)`` shape — activation memory is dominated
    by this length, so we measure the worst case.

    :param dataset_dir: Directory of song subdirectories.
    :param filename: Audio filename inside each song subdirectory.
    :param n: Number of samples to return.
    :param max_duration_s: Target clip duration in seconds.
    :param seed: Shuffle seed so repeated probes at the same ``bs``
        measure the same memory shape.
    :return: List of ``n`` float32 arrays of length
        ``max_duration_s * 16000``.
    """
    target_len = int(max_duration_s * 16000)

    song_dirs = [
        d for d in sorted(Path(dataset_dir).iterdir())
        if d.is_dir() and (d / filename).exists()
    ]
    if len(song_dirs) < n:
        raise RuntimeError(
            f"Only {len(song_dirs)} songs in {dataset_dir} have {filename!r}, "
            f"need {n}. Lower --max or point at a larger dataset."
        )
    rng = random.Random(seed)
    rng.shuffle(song_dirs)

    samples: list[np.ndarray] = []
    for d in song_dirs:
        if len(samples) >= n:
            break
        try:
            audio, sr = sf.read(str(d / filename), always_2d=True, dtype="float32")
        except Exception as e:
            logger.warning(f"skipping {d.name}: {e}")
            continue
        mono = audio.mean(axis=1) if audio.shape[1] > 1 else audio[:, 0]
        if sr != 16000:
            mono = soxr.resample(mono, sr, 16000)
        if len(mono) > target_len:
            mono = mono[:target_len]
        elif len(mono) < target_len:
            mono = np.pad(mono, (0, target_len - len(mono)))
        samples.append(mono.astype(np.float32))
    return samples


def _load_model(architecture: str, model_name: str) -> tuple[Any, dict]:
    """
    Load the model on CPU.

    :param architecture: One of ``"whisper"``, ``"canary"``, ``"parakeet"``.
    :param model_name: Model identifier.
    :return: ``(model, aux)`` where ``aux`` holds architecture-specific
        helpers (e.g. Whisper's processor).
    """
    if architecture == "whisper":
        from transformers import WhisperForConditionalGeneration, WhisperProcessor
        model = WhisperForConditionalGeneration.from_pretrained(model_name)
        processor = WhisperProcessor.from_pretrained(model_name)
        return model, {"processor": processor}

    # map_location='cpu' so we don't auto-place on cuda:0 during load.
    import nemo.collections.asr as nemo_asr
    model = nemo_asr.models.ASRModel.from_pretrained(
        model_name=model_name, map_location="cpu"
    )
    return model, {}


def _whisper_batch(
    model, aux: dict, samples: list[np.ndarray], label_tokens: int
) -> dict:
    processor = aux["processor"]
    features = processor.feature_extractor(
        samples, sampling_rate=16000, return_tensors="pt"
    ).input_features.cuda()
    vocab = processor.tokenizer.vocab_size
    labels = torch.randint(
        0, vocab, (features.size(0), label_tokens), dtype=torch.long, device="cuda"
    )
    # Trailing -100s emulate label padding so the CE loss profile matches
    # production batches.
    if label_tokens > 4:
        labels[:, -4:] = -100
    return {"input_features": features, "labels": labels}


def _whisper_forward_loss(model, batch: dict) -> torch.Tensor:
    return model(**batch).loss


def _parakeet_batch(
    model, samples: list[np.ndarray], text_tokens: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    bs = len(samples)
    T = len(samples[0])
    signal = torch.from_numpy(np.stack(samples)).cuda()
    signal_len = torch.full((bs,), T, dtype=torch.int64, device="cuda")
    vocab = model.tokenizer.vocab_size
    transcript = torch.randint(
        0, vocab, (bs, text_tokens), dtype=torch.long, device="cuda"
    )
    transcript_len = torch.full((bs,), text_tokens, dtype=torch.int64, device="cuda")
    return signal, signal_len, transcript, transcript_len


def _parakeet_forward_loss(model, batch) -> torch.Tensor:
    """
    Replicate :meth:`EncDecRNNTBPEModel.training_step` minus the
    Lightning-only bookkeeping (logging, WER).
    """
    signal, signal_len, transcript, transcript_len = batch
    encoded, encoded_len = model.forward(
        input_signal=signal, input_signal_length=signal_len
    )
    decoder, target_length, _ = model.decoder(
        targets=transcript, target_length=transcript_len
    )
    if not model.joint.fuse_loss_wer:
        joint = model.joint(encoder_outputs=encoded, decoder_outputs=decoder)
        loss = model.loss(
            log_probs=joint,
            targets=transcript,
            input_lengths=encoded_len,
            target_lengths=target_length,
        )
    else:
        loss, _, _, _ = model.joint(
            encoder_outputs=encoded,
            decoder_outputs=decoder,
            encoder_lengths=encoded_len,
            transcripts=transcript,
            transcript_lengths=transcript_len,
            compute_wer=False,
        )
    if loss.ndim > 0:
        loss = loss.mean()
    return loss


def _canary_batch(model, samples: list[np.ndarray]):
    from nemo.collections.asr.data.audio_to_text_lhotse_prompted import (
        PromptedAudioToTextMiniBatch,
    )
    bs = len(samples)
    T = len(samples[0])
    audio = torch.from_numpy(np.stack(samples)).cuda()
    audio_lens = torch.full((bs,), T, dtype=torch.int64, device="cuda")
    vocab = model.tokenizer.vocab_size
    # Representative lengths for 30s of lyrics: ~12-token prompt (lang +
    # pnc + task), ~120-token answer. Decoder memory scales linearly
    # with prompted_transcript length.
    prompt_len = 12
    answer_len = 120
    prompted_len = prompt_len + answer_len
    prompt = torch.randint(0, vocab, (bs, prompt_len), dtype=torch.long, device="cuda")
    prompt_lens = torch.full((bs,), prompt_len, dtype=torch.int64, device="cuda")
    transcript = torch.randint(0, vocab, (bs, answer_len), dtype=torch.long, device="cuda")
    transcript_lens = torch.full((bs,), answer_len, dtype=torch.int64, device="cuda")
    prompted_transcript = torch.randint(
        0, vocab, (bs, prompted_len), dtype=torch.long, device="cuda"
    )
    prompted_transcript_lens = torch.full(
        (bs,), prompted_len, dtype=torch.int64, device="cuda"
    )
    return PromptedAudioToTextMiniBatch(
        audio=audio,
        audio_lens=audio_lens,
        transcript=transcript,
        transcript_lens=transcript_lens,
        prompt=prompt,
        prompt_lens=prompt_lens,
        prompted_transcript=prompted_transcript,
        prompted_transcript_lens=prompted_transcript_lens,
    )


def _canary_forward_loss(model, batch) -> torch.Tensor:
    """
    Replicate :meth:`EncDecMultiTaskModel.training_step` minus the
    logging / WER / BLEU metric updates.
    """
    input_ids, labels = batch.get_decoder_inputs_outputs()
    input_ids_lens = batch.prompted_transcript_lens - 1
    transf_log_probs, _, _, _ = model.forward(
        input_signal=batch.audio,
        input_signal_length=batch.audio_lens,
        transcript=input_ids,
        transcript_length=input_ids_lens,
    )
    loss = model.loss(log_probs=transf_log_probs, labels=labels, output_mask=None)
    if loss.ndim > 0:
        loss = loss.mean()
    return loss


def probe_batch_size(
    dataset_dir: Path,
    filename: str,
    model_name: str,
    bs: int,
    freeze_encoder: bool = False,
    max_duration_s: float = 30.0,
    n_steps: int = 4,
    learning_rate: float = 1e-5,
) -> dict:
    """
    Run ``n_steps`` training steps at batch size ``bs`` against real
    audio read from ``dataset_dir``, return peak VRAM plus median step
    time.

    Intended to be called inside a subprocess so a trial that OOMs
    cleanly crashes the worker without poisoning the parent's CUDA
    state.

    :param dataset_dir: Directory of song subdirectories.
    :param filename: Audio filename inside each song subdirectory.
    :param model_name: Model identifier (e.g. ``nvidia/canary-1b-v2``).
    :param bs: Batch size for this trial.
    :param freeze_encoder: Same semantics as the trainer — affects
        AdamW state size, so must match what production will use.
    :param max_duration_s: Clip length. Samples are padded or truncated
        to this so the batch tensor has a uniform shape.
    :param n_steps: Measurement steps after one warmup step.
    :param learning_rate: AdamW lr. Doesn't affect VRAM, kept
        configurable to match production exactly.
    :return: Dict with ``peak_reserved_gb``, ``peak_allocated_gb``,
        ``total_gb``, ``median_step_ms``, ``trainable_params_m``.
    """
    architecture = detect_architecture(model_name)
    logger.info(f"[probe] arch={architecture} model={model_name} bs={bs}")

    samples = _load_audio_samples(dataset_dir, filename, bs, max_duration_s)
    logger.info(
        f"[probe] loaded {len(samples)} audio samples, each {max_duration_s}s @ 16kHz"
    )

    model, aux = _load_model(architecture, model_name)
    model = model.cuda()
    model.train()

    if freeze_encoder and architecture in ("canary", "parakeet"):
        frozen = sum(1 for _ in model.encoder.parameters())
        for p in model.encoder.parameters():
            p.requires_grad = False
        logger.info(f"[probe] froze {frozen} encoder parameter tensors")

    trainable = [p for p in model.parameters() if p.requires_grad]
    n_trainable = sum(p.numel() for p in trainable)
    logger.info(f"[probe] trainable params: {n_trainable/1e6:.1f}M")

    optimizer = torch.optim.AdamW(trainable, lr=learning_rate, weight_decay=0.01)

    if architecture == "whisper":
        make_batch = lambda: _whisper_batch(model, aux, samples, label_tokens=300)
        fwd_loss = _whisper_forward_loss
    elif architecture == "parakeet":
        make_batch = lambda: _parakeet_batch(model, samples, text_tokens=30)
        fwd_loss = _parakeet_forward_loss
    else:
        make_batch = lambda: _canary_batch(model, samples)
        fwd_loss = _canary_forward_loss

    step_times: list[float] = []
    autocast = torch.amp.autocast("cuda", dtype=torch.bfloat16)

    total_steps = 1 + n_steps
    for step_i in range(total_steps):
        batch = make_batch()
        torch.cuda.synchronize()

        # Reset peak stats after the warmup step so the measurement only
        # captures steady-state allocations — the first step pays for
        # kernel JIT, cuDNN autotune, numba RNNT compile, etc.
        if step_i == 1:
            torch.cuda.reset_peak_memory_stats()

        t0 = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        with autocast:
            loss = fwd_loss(model, batch)
        loss.backward()
        optimizer.step()
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        if step_i >= 1:
            step_times.append(elapsed)
        logger.info(
            f"[probe] step {step_i}: {elapsed*1000:.0f} ms  "
            f"loss={loss.item():.3f}  "
            f"reserved={torch.cuda.memory_reserved()/1024**3:.2f} GiB"
        )

    peak_reserved = torch.cuda.max_memory_reserved()
    peak_allocated = torch.cuda.max_memory_allocated()
    total = torch.cuda.get_device_properties(0).total_memory
    median_step_ms = statistics.median(step_times) * 1000 if step_times else 0.0

    del model, optimizer, trainable
    gc.collect()
    torch.cuda.empty_cache()

    return {
        "peak_reserved_gb": peak_reserved / (1024**3),
        "peak_allocated_gb": peak_allocated / (1024**3),
        "total_gb": total / (1024**3),
        "median_step_ms": median_step_ms,
        "trainable_params_m": n_trainable / 1e6,
    }
