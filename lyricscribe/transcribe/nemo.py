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


def _load_mono(audio_path: str) -> tuple[np.ndarray, int]:
    """
    Load an audio file and average channels to mono.

    :param audio_path: Path to the audio file.
    :return: Tuple of (mono samples as float32 ndarray, sample rate).
    """
    wav, sr = torchaudio.load(audio_path)
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    return wav.squeeze(0).numpy(), sr


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

        # Disable CUDA graphs for RNNT/TDT models (e.g. Parakeet) to
        # work around a NeMo bug on newer GPUs (H200) where cu_call
        # returns fewer values than expected.
        if hasattr(self.model, "decoding") and hasattr(
            self.model.decoding, "decoding"
        ):
            self.model.decoding.decoding.use_cuda_graph_decoder = False

        logger.info(
            f"Loaded NeMo model on {self.model.device}"
            f"{' (multi-task)' if self.is_multitask else ''}"
        )

    def transcribe(self, audio_path: str, language: str | None = None) -> str:
        """
        Transcribe a single audio file.

        :param audio_path: Path to the audio file.
        :param language: Optional language code. Passed as
            ``source_lang``/``target_lang`` for multi-task models
            (Canary). Ignored for CTC/TDT models (Parakeet).
        :return: Transcribed text.
        """
        audio, _ = _load_mono(audio_path)
        kwargs: dict = {"batch_size": self.batch_size}
        if language and self.is_multitask:
            kwargs["source_lang"] = language
            kwargs["target_lang"] = language
        output = self.model.transcribe(audio, **kwargs)
        return output[0].text.strip()

    def transcribe_with_vad(
        self,
        audio_path: str,
        vad_model: ScriptModule,
        language: str | None = None,
    ) -> str:
        """
        Transcribe with VAD using NeMo's manifest-based offset/duration.

        Loads the audio, resamples to 16 kHz mono, runs Silero VAD to
        find speech regions, then writes a temporary NeMo manifest with
        one entry per segment. NeMo handles the seeking internally, so
        no audio slicing is needed.

        :param audio_path: Path to the audio file.
        :param vad_model: Loaded Silero VAD model.
        :param language: Optional language code.
        :return: Concatenated transcription of speech segments.
        """
        wav, sample_rate = torchaudio.load(audio_path)
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)
        if sample_rate != 16000:
            wav = torchaudio.functional.resample(wav, sample_rate, 16000)
            sample_rate = 16000

        wav_1d = wav.squeeze(0)
        timestamps = get_speech_timestamps(
            wav_1d,
            vad_model,
            sampling_rate=sample_rate,
            return_seconds=True,
        )

        if not timestamps:
            return ""

        with tempfile.TemporaryDirectory() as tmp_dir:
            # Save mono audio for the manifest to reference
            mono_path = str(Path(tmp_dir) / "mono.wav")
            torchaudio.save(mono_path, wav, sample_rate)

            manifest_path = Path(tmp_dir) / "manifest.json"
            with open(manifest_path, "w") as manifest_file:
                for seg in timestamps:
                    entry = {
                        "audio_filepath": mono_path,
                        "offset": round(seg["start"], 3),
                        "duration": round(seg["end"] - seg["start"], 3),
                    }
                    if language and self.is_multitask:
                        entry["source_lang"] = language
                        entry["target_lang"] = language
                        entry["taskname"] = "asr"
                        entry["pnc"] = "yes"
                    manifest_file.write(json.dumps(entry) + "\n")

            kwargs: dict = {"batch_size": self.batch_size}
            if language and self.is_multitask:
                kwargs["source_lang"] = language
                kwargs["target_lang"] = language
            outputs = self.model.transcribe(
                str(manifest_path),
                **kwargs,
            )
            parts = [o.text.strip() for o in outputs if o.text.strip()]
            return " ".join(parts)
