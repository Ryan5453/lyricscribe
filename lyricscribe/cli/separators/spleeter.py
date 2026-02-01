"""
This assumes you have a directory structure like this:

/root
    /<ISRC>
        /audio.mp3  # Original audio

The output files will be in the same directory as the input files.
"""

import time

from spleeter.audio.adapter import AudioAdapter
from spleeter.separator import Separator as SpleeterModel  # Renamed

from .base import BaseSeparator  # Import the new base class


class SpleeterSeparator(BaseSeparator):
    def __init__(
        self,
        model_name: str,
        output_filename_suffix: str,
        directory: str,
        input_filename: str = "audio.mp3",
    ):
        # Spleeter might need AudioAdapter initialized before super().__init__ if _load_model uses it implicitly,
        # or _load_model should also handle adapter initialization.
        # For Spleeter, the model string (e.g., "spleeter:2stems") defines the model directly.
        self.audio_adapter = AudioAdapter.default()  # Spleeter needs this
        super().__init__(model_name, output_filename_suffix, directory, input_filename)

    def _load_model(self, model_name: str) -> SpleeterModel:
        # For Spleeter, model_name is something like "spleeter:2stems", "spleeter:4stems", "spleeter:5stems"
        # The Separator class itself handles the model loading based on this string.
        print(f"Spleeter will use model configuration: {model_name}")
        return SpleeterModel(model_name)

    def _get_implementation_name(self) -> str:
        return "Spleeter"

    def _separate_file(self, input_path: str, output_path: str) -> float:
        """
        Separates vocals using Spleeter and saves it.
        Uses self.model (SpleeterModel instance) and self.audio_adapter.
        """
        # Load waveform using the audio adapter
        waveform, sample_rate = self.audio_adapter.load(
            input_path,
            sample_rate=self.model._sample_rate,  # Use model's expected sample rate
        )

        start_time = time.time()
        # Perform separation
        sources = self.model.separate(
            waveform
        )  # This is a method of SpleeterModel instance
        separation_time = time.time() - start_time

        # Save the vocal stem
        # Spleeter typically returns a dict with keys like 'vocals', 'drums', etc.
        if "vocals" in sources:
            self.audio_adapter.save(
                output_path,
                sources["vocals"],
                sample_rate,  # Use the original sample rate for saving
                "wav",  # Output format
                "128k",  # Bitrate, though for wav, this is more like a placeholder from original code
                # For wav, quality is determined by sample rate and bit depth (implicitly 16-bit by Spleeter typically)
            )
        else:
            raise ValueError(
                f"'vocals' stem not found in Spleeter output for {input_path}. Available: {list(sources.keys())}"
            )

        return separation_time
