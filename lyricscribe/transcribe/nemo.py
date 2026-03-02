import json
import logging
import tempfile
from pathlib import Path

import nemo.collections.asr as nemo_asr
import torch
from torch.jit import ScriptModule
import torchaudio
from silero_vad import get_speech_timestamps

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
        logger.info(f"Loaded NeMo model on {self.model.device}")

    def transcribe(self, audio_path: str) -> str:
        """
        Transcribe a single audio file.

        :param audio_path: Path to the audio file.
        :return: Transcribed text.
        """
        output = self.model.transcribe([audio_path], batch_size=self.batch_size)
        return output[0].text.strip()

    def transcribe_with_vad(
        self,
        audio_path: str,
        vad_model: ScriptModule,
    ) -> str:
        """
        Transcribe with VAD using NeMo's manifest-based offset/duration.

        Loads the audio, resamples to 16 kHz mono, runs Silero VAD to
        find speech regions, then writes a temporary NeMo manifest with
        one entry per segment. NeMo handles the seeking internally, so
        no audio slicing is needed.

        :param audio_path: Path to the audio file.
        :param vad_model: Loaded Silero VAD model.
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
            manifest_path = Path(tmp_dir) / "manifest.json"
            with open(manifest_path, "w") as manifest_file:
                for seg in timestamps:
                    entry = {
                        "audio_filepath": audio_path,
                        "offset": round(seg["start"], 3),
                        "duration": round(seg["end"] - seg["start"], 3),
                    }
                    manifest_file.write(json.dumps(entry) + "\n")

            outputs = self.model.transcribe(
                str(manifest_path),
                batch_size=self.batch_size,
            )
            parts = [o.text.strip() for o in outputs if o.text.strip()]
            return " ".join(parts)
