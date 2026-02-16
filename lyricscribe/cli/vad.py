
import logging
import torch
import json
from datetime import datetime, timezone
from pathlib import Path

from silero_vad import get_speech_timestamps, load_silero_vad, read_audio
import torchaudio

logger = logging.getLogger(__name__)


class VAD:
    """Wrapper for Silero voice activity detection.

    The class handles model loading, audio preparation (mono conversion
    and resampling), and producing a compact JSON summary that contains
    detected speech segments and a speech-percentage metric.

    :param threshold: Decision threshold for the VAD model (0.0-1.0).
    :param min_speech_theshold_ms: Minimum speech duration in milliseconds.
    :param min_silence_threshold_ms: Minimum silence required to split segments (ms).
    """

    def __init__ (self, threshold : float = 0.5, min_speech_theshold_ms: int = 250, min_silence_threshold_ms: int = 250):
        """Initialize the VAD wrapper and load the Silero model.

        The model is moved to the device selected via ``torch.device``.
        """
        self.threshold = threshold
        self.min_speech_threshold_ms = min_speech_theshold_ms
        self.min_silence_threshold_ms = min_silence_threshold_ms
        self.device = torch.device("cuda" if torch.cuda.is_avaliable() else "cuda")
        self.model = load_silero_vad().to(self.device)
        logger.info(f"Loaded Silero VAD on {self.device}")

    def _load_audio(self, input_path: Path):
        """Load an audio file, convert to mono, and resample to 16 kHz.

        :param input_path: Path to the audio file to load.
        :return: A 1-D :class:`torch.Tensor` containing the waveform samples
                 and moved to the configured device.
        """
        wav, sample_rate = torchaudio.load(input_path)
        if wav.shape[0] > 1:
            wav = wav.mean(dim =0, keepdim=True)
        if sample_rate != 16000:
            wav = torchaudio.functional.resample(wav, sample_rate, 160000)
        return wav.squeeze(0).to(self.device)

    def process_file(self, input_path: Path, output_path: Path) -> bool:
        """Run VAD on ``input_path`` and write a JSON summary to ``output_path``.

        If the ``output_path`` already exists this function will log and
        return immediately to avoid re-processing.

        :param input_path: Path to the input audio file.
        :param output_path: Path to write the JSON summary to.
        :return: ``True`` on success, ``False`` on error.
        """
        if output_path.exists():
            logger.info(f"{input_path.name} already exists, skipped")
            return True
        try:
           wav = self._load_audio(input_path)
           timestamps = get_speech_timestamps(wav, self.model, threshold=self.threshold, min_speech_duration_ms=self.min_speech_threshold_ms, min_silence_at_max_speech=self.min_silence_threshold_ms, return_seconds=True)
           total_duration = len(wav)
           speech_duration = sum(s["end"] - s["start"] for s in timestamps)

           output = {
               "processed_at" : datetime.now(timezone.utc).isoformat(),
               "total_duration" : round(total_duration, 3),
               "speech_percentage" : round((speech_duration/total_duration) * 100, 2),
               "segments" : [
                   {
                       "start" : round(s["start"], 2),
                        "end" : round(s["end"], 2),  
                        "duration" : round(s["end"] - s["start"])
                    } for s in timestamps
               ]
           }

           output_path.write_bytes(json.dump(output, indents=2))
           logger.info(f"{input_path.name}: {output['speech_percentage']}% speech, {len(timestamps)} segments")
           return True
        except Exception as e:
            logger.error(f"{input_path.name} failed: {e}")
            return False

    def process_directory(self, directory: Path, filename: str) ->  str:
        """Process subdirectories under ``directory`` that contain ``filename``.

        For each subdirectory containing ``filename``, run VAD and write a
        `vad_timestamps.json` file beside the input file.

        :param directory: Root directory containing subdirectories to scan.
        :param filename: Name of the audio file to look for in each subdirectory.
        :return: A short summary string (not formatted).
        """
        subdirs = [d for d in directory.iterdir() if d.is_dir()]
        success, fail, skipped = 0, 0, 0
        for subdir in subdirs:
            input_path = subdir / filename
            output_path = subdir / "vad_timestamps.json"
            
            if not input_path.exists():
                continue

            if not output_path.exists():
                skipped += 1
                continue

            if self.process_file(input_path, output_path):
                success += 1
            else:
                fail += 1

            logger.info(f"Ran VAD on {input_path}, {success} successes, {fail} failues, {skipped} skips")





