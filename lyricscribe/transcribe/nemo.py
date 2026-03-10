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

logger = logging.getLogger(__name__)



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
    ) -> str:
        """
        Transcribe a single audio file, optionally with VAD and/or chunking.
        """
        wav, sr = torchaudio.load(audio_path)
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)
        if sr != 16000:
            wav = torchaudio.functional.resample(wav, sr, 16000)
            sr = 16000

        total_duration = wav.shape[1] / sr

        if use_vad:
            if vad_model is None:
                raise ValueError("vad_model must be provided when use_vad=True")
            wav_1d = wav.squeeze(0)
            timestamps = get_speech_timestamps(
                wav_1d,
                vad_model,
                sampling_rate=sr,
                return_seconds=True,
            )
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
