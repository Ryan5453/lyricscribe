import json
import logging
import tempfile
from pathlib import Path

import nemo.collections.asr as nemo_asr
import numpy as np
import torch
import torchaudio
from nemo.collections.asr.models import EncDecMultiTaskModel
from silero_vad import get_speech_timestamps
from torch.jit import ScriptModule

from lyricscribe.transcribe.base import Transcriber
from lyricscribe.transcribe.rms_vad import (
    RmsVadOptions,
    get_speech_timestamps_rms,
    merge_segments,
)

logger = logging.getLogger(__name__)


def _banned_ngram_next_tokens(ids: list[int], n: int) -> list[int]:
    """
    Tokens that would complete an already-generated n-gram if emitted next.

    Standard HF ``no_repeat_ngram_size`` semantics: collect every n-gram in
    *ids* and ban each token that previously followed the (n-1)-gram equal to
    the current suffix.

    :param ids: Generated token ids so far (including the prompt).
    :param n: N-gram size to block.
    :return: Token ids to ban at the next step.
    """
    if n < 1 or len(ids) < n:
        return []
    seen: dict[tuple[int, ...], set[int]] = {}
    for j in range(len(ids) - n + 1):
        seen.setdefault(tuple(ids[j : j + n - 1]), set()).add(ids[j + n - 1])
    return sorted(seen.get(tuple(ids[len(ids) - n + 1 :]), ()))


def _install_aed_logit_controls(generator) -> None:
    """
    Graft HF-style ``repetition_penalty`` / ``no_repeat_ngram_size`` onto a
    NeMo ``GreedySequenceGenerator`` by wrapping ``_one_step_forward`` and
    editing the last-position logits before the decode loop's argmax.

    NeMo's AED (Canary) decoder exposes no repetition control of its own.
    The decode loop passes ``decoder_mems_list=None`` exactly once per
    utterance batch (the prompt step) and then only the newest token, so
    mirroring those calls reconstructs the loop's full ``tgt`` prefix.

    Idempotent: the wrapper reads ``generator._ls_repetition_penalty`` /
    ``generator._ls_no_repeat_ngram_size`` on every call, so re-installs just
    refresh those attributes. With both knobs at defaults it is a no-op.
    """
    if getattr(generator, "_ls_logit_controls", False):
        return
    generator._ls_logit_controls = True
    original = generator._one_step_forward
    state: dict = {"ids": None}

    def wrapped(
        decoder_input_ids=None,
        encoder_hidden_states=None,
        encoder_input_mask=None,
        decoder_mems_list=None,
        pos=0,
        return_scores: bool = True,
    ):
        if decoder_mems_list is None:
            state["ids"] = decoder_input_ids.detach().clone()
        else:
            state["ids"] = torch.cat(
                (state["ids"], decoder_input_ids[:, -1:]), dim=-1
            )

        result = original(
            decoder_input_ids,
            encoder_hidden_states,
            encoder_input_mask,
            decoder_mems_list,
            pos,
            return_scores=return_scores,
        )
        penalty = getattr(generator, "_ls_repetition_penalty", 1.0) or 1.0
        ngram = int(getattr(generator, "_ls_no_repeat_ngram_size", 0) or 0)
        if penalty == 1.0 and ngram == 0:
            return result

        last = result[0][:, -1, :]
        for row, ids_row in enumerate(state["ids"]):
            if penalty != 1.0:
                prev = torch.unique(ids_row)
                scores = last[row, prev]
                last[row, prev] = torch.where(
                    scores < 0, scores * penalty, scores / penalty
                )
            if ngram > 0:
                banned = _banned_ngram_next_tokens(ids_row.tolist(), ngram)
                if banned:
                    last[row, banned] = float("-inf")
        return result

    generator._one_step_forward = wrapped


class NemoTranscriber(Transcriber):
    """
    Transcriber using NVIDIA NeMo ASR models.

    This should work with any NeMo-compatible model (Parakeet, Canary, etc.)
    loaded via ``ASRModel.from_pretrained()``. For VAD-based
    transcription, uses NeMo's native manifest format with
    ``offset`` and ``duration`` fields to avoid audio slicing.
    """

    def __init__(self, model_name: str, batch_size: int = 1) -> None:
        """
        Initialize the NeMo transcriber.

        :param model_name: HuggingFace model identifier (e.g.
            ``nvidia/parakeet-tdt-0.6b-v3``).
        :param batch_size: Batch size for NeMo's transcribe method.
        """
        super().__init__(model_name, batch_size)
        self.model = None
        self.is_multitask = False

    def load(self) -> None:
        """
        Load a NeMo ASR model from a pretrained checkpoint.

        The model is automatically placed on CUDA if available.
        """
        logger.info(f"CUDA available: {torch.cuda.is_available()}")
        logger.info(f"Loading NeMo model: {self.model_name}")
        # ``from_pretrained`` only accepts HF-hub repo IDs; local
        # ``.nemo`` checkpoints (e.g. finetune outputs) must be
        # loaded via ``restore_from``.
        from pathlib import Path
        if Path(self.model_name).is_file():
            # ``ASRModel.restore_from`` can't build multitask models; use their class.
            cfg = nemo_asr.models.ASRModel.restore_from(
                restore_path=self.model_name, return_config=True
            )
            model_cls = nemo_asr.models.ASRModel
            if "EncDecMultiTask" in str(cfg.get("target", "")):
                model_cls = EncDecMultiTaskModel
            self.model = model_cls.restore_from(restore_path=self.model_name)
        else:
            self.model = nemo_asr.models.ASRModel.from_pretrained(
                model_name=self.model_name
            )
        self.is_multitask = isinstance(self.model, EncDecMultiTaskModel)

        # Disable CUDA graphs for RNNT/TDT models (e.g. Parakeet).
        # CUDA graphs cause illegal memory access on H200 GPUs.
        decoding = getattr(self.model, "decoding", None)
        inner = getattr(decoding, "decoding", None)
        computer = getattr(inner, "decoding_computer", None)
        if computer is not None and hasattr(computer, "cuda_graphs_mode"):
            computer.cuda_graphs_mode = None

        logger.info(
            f"Loaded NeMo model on {self.model.device}"
            f"{' (multi-task)' if self.is_multitask else ''}"
        )

    def transcribe(
        self,
        audio_path: str,
        use_vad: bool = False,
        vad_model: ScriptModule | None = None,
        use_chunked: bool = False,
        language: str | None = None,
        vad_source: str | None = None,
        vad_method: str = "silero",
        repetition_penalty: float = 1.0,
        no_repeat_ngram_size: int = 0,
        num_beams: int = 5,
        condition_on_prev: bool = True,
        temperature_fallback: bool = False,
        compression_ratio_threshold: float | None = None,
        logprob_threshold: float | None = None,
        initial_prompt: str | None = None,
    ) -> str:
        """
        Transcribe a single audio file, optionally with VAD and/or chunking.
        """
        del num_beams, condition_on_prev, temperature_fallback
        del compression_ratio_threshold, logprob_threshold, initial_prompt
        # Repetition knobs apply only to multitask (AED) models; RNNT/TDT ignores them.
        if self.is_multitask:
            wants_controls = repetition_penalty != 1.0 or bool(no_repeat_ngram_size)
            inner = getattr(self.model.decoding, "decoding", None)
            if wants_controls and not hasattr(inner, "greedy_search"):
                # Controls need the greedy generator; same output as beam_size=1.
                from nemo.collections.asr.parts.submodules.multitask_decoding import (
                    MultiTaskDecodingConfig,
                )
                from omegaconf import OmegaConf

                self.model.change_decoding_strategy(
                    OmegaConf.structured(MultiTaskDecodingConfig(strategy="greedy"))
                )
                inner = getattr(self.model.decoding, "decoding", None)
                logger.info(
                    "Switched multitask decoding beam(1) -> greedy for logit controls"
                )
            generator = getattr(inner, "greedy_search", None)
            if generator is not None:
                _install_aed_logit_controls(generator)
                generator._ls_repetition_penalty = repetition_penalty
                generator._ls_no_repeat_ngram_size = no_repeat_ngram_size
                if wants_controls:
                    logger.info(
                        "AED logit controls active: repetition_penalty="
                        f"{repetition_penalty}, "
                        f"no_repeat_ngram_size={no_repeat_ngram_size}"
                    )
            elif wants_controls:
                logger.warning(
                    "Repetition controls requested but no greedy generator "
                    "found on the multitask decoder; knobs NOT applied."
                )
        wav, sr = torchaudio.load(audio_path)
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)
        if sr != 16000:
            wav = torchaudio.functional.resample(wav, sr, 16000)
            sr = 16000

        total_duration = wav.shape[1] / sr

        if use_vad:
            if vad_method == "silero" and vad_model is None:
                raise ValueError(
                    "vad_model must be provided when use_vad=True and vad_method='silero'"
                )
            if vad_method not in ("silero", "rms"):
                raise ValueError(f"Unknown vad_method: {vad_method!r}")

            if vad_source is not None:
                vad_wav, vad_sr = torchaudio.load(vad_source)
                if vad_wav.shape[0] > 1:
                    vad_wav = vad_wav.mean(dim=0, keepdim=True)
                if vad_sr != 16000:
                    vad_wav = torchaudio.functional.resample(vad_wav, vad_sr, 16000)
                vad_1d = vad_wav.squeeze(0)
            else:
                vad_1d = wav.squeeze(0)

            if vad_method == "silero":
                timestamps = get_speech_timestamps(
                    vad_1d,
                    vad_model,
                    sampling_rate=16000,
                    return_seconds=True,
                )
            else:
                # RMS-VAD returns sample indices; merge into <=30s chunks
                # then convert to seconds for NeMo's manifest format.
                rms_segments = get_speech_timestamps_rms(
                    vad_1d.numpy(),
                    vad_options=RmsVadOptions(),
                    window_size_samples=512,
                    sampling_rate=16000,
                )
                merged = merge_segments(
                    rms_segments, max_length_s=30, sampling_rate=16000
                )
                timestamps = [
                    {"start": seg["start"] / 16000.0, "end": seg["end"] / 16000.0}
                    for seg in merged
                ]
            if not timestamps:
                return ""
        else:
            timestamps = [{"start": 0.0, "end": total_duration}]

        with tempfile.TemporaryDirectory() as tmp_dir:
            mono_path = str(Path(tmp_dir) / "mono.wav")
            torchaudio.save(mono_path, wav, sr)

            manifest_path = Path(tmp_dir) / "manifest.json"
            with open(manifest_path, "w") as manifest_file:
                for seg in timestamps:
                    start = seg["start"]
                    end = seg["end"]

                    if use_chunked:
                        CHUNK_S = 30.0
                        OVERLAP_S = 5.0
                        STEP_S = CHUNK_S - OVERLAP_S

                        offset = start
                        while offset < end:
                            duration = min(CHUNK_S, end - offset)
                            entry = {
                                "audio_filepath": mono_path,
                                "offset": round(offset, 3),
                                "duration": round(duration, 3),
                            }
                            if self.is_multitask:
                                entry["taskname"] = "asr"
                                entry["pnc"] = "yes"
                                entry["source_lang"] = language or "en"
                                entry["target_lang"] = language or "en"
                            manifest_file.write(json.dumps(entry) + "\n")
                            offset += STEP_S
                    else:
                        entry = {
                            "audio_filepath": mono_path,
                            "offset": round(start, 3),
                            "duration": round(end - start, 3),
                        }
                        if self.is_multitask:
                            entry["taskname"] = "asr"
                            entry["pnc"] = "yes"
                            entry["source_lang"] = language or "en"
                            entry["target_lang"] = language or "en"
                        manifest_file.write(json.dumps(entry) + "\n")

            kwargs = {"batch_size": self.batch_size}
            if self.is_multitask:
                kwargs["source_lang"] = language or "en"
                kwargs["target_lang"] = language or "en"
            outputs = self.model.transcribe(
                str(manifest_path),
                **kwargs,
            )
            if not isinstance(outputs, list):
                outputs = [outputs]
            parts = [o.text.strip() for o in outputs if o.text.strip()]
            return " ".join(parts)
