"""
This assumes you have a directory structure like this:

/root
    /<ISRC>
        /audio.mp3  # Original audio

The output files will be in the same directory as the input files.
"""

import time

import torch
from demucs.api import Separator as DemucsModel  # Renamed to avoid confusion
from demucs.audio import save_audio

from .base import BaseSeparator  # Import the new base class


class DemucsSeparator(BaseSeparator):
    def _load_model(self, model_name: str) -> DemucsModel:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Demucs will use device: {device}")
        return DemucsModel(model_name, device=device)

    def _get_implementation_name(self) -> str:
        return "Demucs"

    def _separate_file(self, input_path: str, output_path: str) -> float:
        """
        Extracts vocals from an audio file using Demucs and saves it.
        Uses the loaded self.model.
        """
        # The Demucs Separator class has its own _load_audio method.
        # We need an instance of the Demucs Separator model, which is self.model
        audio_tensor = self.model._load_audio(input_path)  # Use self.model here

        start_time = time.time()
        # The separate_tensor method is part of the Demucs Separator model instance
        _, sources = self.model.separate_tensor(audio_tensor, self.model.samplerate)
        separation_time = time.time() - start_time

        # Assuming "vocals" is the key for the vocal stem, which is common for Demucs
        if "vocals" in sources:
            save_audio(
                sources["vocals"].cpu(),
                output_path,
                samplerate=self.model.samplerate,
                bits_per_sample=16,  # Standard for WAV output
                as_float=False,  # Standard for WAV output
            )
        else:
            # Fallback or error if 'vocals' stem is not found. This depends on the Demucs model.
            # For now, let's try to save the first available stem if 'vocals' is not present,
            # or handle it as an error.
            # Most Demucs models (htdemucs, mdx_extra) provide 'vocals', 'drums', 'bass', 'other'.
            print(
                f"Warning: 'vocals' stem not found in Demucs output for {input_path}. Available stems: {list(sources.keys())}"
            )
            # As a fallback, could save all stems or the primary one if known, or raise an error.
            # For now, we just won't save anything if 'vocals' isn't there, and it will result in a missing file.
            # Or, more robustly, raise an exception.
            raise ValueError(
                f"'vocals' stem not found in Demucs output for {input_path}. Available: {list(sources.keys())}"
            )

        return separation_time
