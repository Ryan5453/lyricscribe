import logging
from abc import ABC, abstractmethod

import torch
from torch.jit import ScriptModule

logger = logging.getLogger(__name__)

MEMORY_SAFETY_FACTOR = 0.85
MAX_AUTO_BATCH_SIZE = 64


class Transcriber(ABC):
    """
    Abstract base class for ASR transcription models.
    """

    def __init__(self, model_name: str, batch_size: int = 0) -> None:
        """
        Initialize the transcriber with a model name and batch size.

        :param model_name: HuggingFace model identifier.
        :param batch_size: Batch size for inference. ``0`` means
            auto-calibrate based on available GPU memory.
        """
        self.model_name = model_name
        self.batch_size = batch_size

    @abstractmethod
    def load(self) -> None:
        """
        Load the model and any required processors.

        Called once before processing begins. Implementations should
        handle device placement and precision settings.
        """
        ...

    @abstractmethod
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

        :param audio_path: Absolute path to the audio file.
        :param use_vad: Whether to use VAD to segment the audio.
        :param vad_model: Loaded Silero VAD model instance (required if use_vad=True).
        :param use_chunked: Whether to use fixed-length chunked inference.
        :param language: Optional language code hint.
        :return: Transcribed text.
        """
        ...

    def calibrate_batch_size(
        self,
        audio_path: str,
        max_batch_size: int = MAX_AUTO_BATCH_SIZE,
        language: str | None = None,
    ) -> None:
        """
        Profile GPU memory to determine the optimal batch size.

        Inspired by vLLM's memory profiling approach: measures steady-state
        memory after model loading, runs a single inference to measure
        per-sample peak cost, then calculates how many samples fit in
        the remaining free memory with a safety margin.

        If ``self.batch_size`` is already set (> 0), this method is a
        no-op — explicit batch sizes are respected.

        On CPU this always sets ``batch_size = 1`` since memory is not
        the bottleneck.

        :param audio_path: Path to a representative audio file for
            profiling. Should be typical of the files to be processed.
        :param max_batch_size: Upper cap on the computed batch size.
        :param language: Optional language code, forwarded to
            :meth:`transcribe` during the profiling inference.
        """
        if self.batch_size > 0:
            logger.info(
                f"Using explicit batch_size={self.batch_size}, skipping calibration"
            )
            return

        if not torch.cuda.is_available():
            self.batch_size = 1
            logger.info("No CUDA available, setting batch_size=1")
            return

        device = torch.cuda.current_device()

        # Step 1: Measure baseline — memory used by the loaded model
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        baseline = torch.cuda.memory_allocated(device)

        # Step 2: Run a single inference to measure peak memory cost
        old_batch = self.batch_size
        self.batch_size = 1
        try:
            self.transcribe(audio_path, language=language)
        except Exception as e:
            logger.warning(
                f"Calibration inference failed ({e}), defaulting to batch_size=1"
            )
            self.batch_size = 1
            return
        finally:
            if self.batch_size == 1 and old_batch == 0:
                pass  # will be overwritten below
            else:
                self.batch_size = old_batch

        # Step 3: Calculate per-sample memory from peak
        peak = torch.cuda.max_memory_allocated(device)
        per_sample = peak - baseline

        if per_sample <= 0:
            logger.warning(
                "Could not measure per-sample memory, defaulting to batch_size=1"
            )
            self.batch_size = 1
            return

        # Step 4: Calculate optimal batch size from remaining free memory
        torch.cuda.empty_cache()
        free, total = torch.cuda.mem_get_info(device)
        available = int(free * MEMORY_SAFETY_FACTOR)
        optimal = max(1, available // per_sample)
        self.batch_size = min(optimal, max_batch_size)

        logger.info(
            f"Memory profile: "
            f"model={baseline / 1e6:.0f}MB, "
            f"per_sample={per_sample / 1e6:.0f}MB, "
            f"free={free / 1e6:.0f}MB, "
            f"available={available / 1e6:.0f}MB "
            f"(safety={MEMORY_SAFETY_FACTOR:.0%})"
        )
        logger.info(
            f"Calibrated batch_size={self.batch_size} "
            f"(max={max_batch_size}, optimal={optimal})"
        )

    def halve_batch_size(self) -> int:
        """
        Halve the current batch size as an OOM recovery strategy.

        Called by the job runner when a ``torch.cuda.OutOfMemoryError``
        is caught. Clears the CUDA cache before returning.

        :return: The new batch size after halving.
        """
        old = self.batch_size
        self.batch_size = max(1, self.batch_size // 2)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.warning(f"OOM recovery: batch_size {old} -> {self.batch_size}")
        return self.batch_size
