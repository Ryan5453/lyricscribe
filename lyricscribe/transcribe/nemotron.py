import logging

import numpy as np
import torch
from peft import PeftModel
from transformers import AutoProcessor, Nemotron3_5TranscribeForConditionalGeneration

from lyricscribe.transcribe.adapter import check_train_record, load_adapter
from lyricscribe.transcribe.base import (
    DECODE_KNOBS,
    MAX_NEW_TOKENS,
    ChunkTranscriber,
    Decode,
    describe_kwargs,
    generate_kwargs,
    shared_prompt,
)
from lyricscribe.transcribe.chunking import language_name

logger = logging.getLogger(__name__)

# How training records this backend's one prompt (the vendor ASR instruction).
PROMPT_MODE = "asr"
PROMPT = (
    "Audio transcription. Language: {language}. Wording: verbatim, including fillers and "
    "repetitions. Formatting: punctuation and capitalization. Output: transcript only."
)


def load_nemotron_adapter(model, adapter: str, is_trainable: bool = False) -> PeftModel:
    """
    Wrap a Nemotron 3.5 Transcribe model with a saved LoRA adapter.

    :param model: ``Nemotron3_5TranscribeForConditionalGeneration``.
    :param adapter: Adapter dir.
    :param is_trainable: Keep the adapter trainable.
    :return: The wrapped model.
    :raises RuntimeError: If any LoRA module or ``modules_to_save`` copy did not load.
    """
    return load_adapter(model, adapter, is_trainable=is_trainable)


class NemotronTranscriber(ChunkTranscriber):
    """
    NVIDIA Nemotron 3.5 Transcribe (audio encoder + hybrid Mamba-Transformer
    MoE LLM) with reasoning off.

    The prompt is the model card's verbatim-transcription instruction with the
    known language in place of ``as spoken`` (``prompt_mode="asr"``), or the
    shared prompt (``"shared"``).
    """

    supports = frozenset(DECODE_KNOBS)
    prompt_modes = (PROMPT_MODE, "shared")

    def __init__(
        self,
        model_name: str,
        batch_size: int = 8,
        max_new_tokens: int = MAX_NEW_TOKENS,
        adapter: str | None = None,
        prompt_mode: str | None = None,
    ) -> None:
        """
        :param model_name: Hub id or checkpoint dir.
        :param batch_size: Chunks decoded per ``generate`` call.
        :param max_new_tokens: Generation cap per chunk.
        :param adapter: LoRA adapter dir merged into the model, or ``None``.
        :param prompt_mode: ``asr`` (the card's prompt, default) or ``shared``.
        """
        super().__init__(model_name, batch_size, prompt_mode)
        self.max_new_tokens = max_new_tokens
        self.adapter = adapter
        self.model = None
        self.processor = None

    def load(self) -> None:
        """
        Load the model and processor onto the GPU (CPU without CUDA).

        :raises RuntimeError: If the adapter's training record has another prompt.
        """
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.bfloat16 if device == "cuda" else torch.float32
        logger.info(f"Loading Nemotron 3.5 Transcribe: {self.model_name} on {device}")
        # cuDNN SDPA rebuilds its graph per sequence shape, which stalls decoding.
        torch.backends.cuda.enable_cudnn_sdp(False)
        self.train_record = check_train_record(
            self.adapter or self.model_name, self.prompt_mode, self.model_name if self.adapter else None
        )
        self.processor = AutoProcessor.from_pretrained(self.model_name)
        self.model = Nemotron3_5TranscribeForConditionalGeneration.from_pretrained(
            self.model_name, dtype=dtype, device_map=device
        )
        if self.adapter:
            self.model = load_nemotron_adapter(self.model, self.adapter).merge_and_unload()
            logger.info(f"Merged LoRA adapter {self.adapter}")
        self.model.eval()

    @torch.inference_mode()
    def _decode(self, wavs: list[np.ndarray], language: str | None, decode: Decode) -> list[str]:
        """
        Decode chunks of one song in a single ``generate`` call.

        :param wavs: 16 kHz mono chunks.
        :param language: ``detected_language`` from ``lyrics.json``, or ``None``.
        :param decode: Decoding settings.
        :return: One transcription per chunk.
        """
        if self.prompt_mode == "shared":
            prompt = shared_prompt(language)
        else:
            prompt = PROMPT.format(language=language_name(language) or "as spoken")
        inputs = self.processor.apply_transcription_request(
            [np.asarray(w, dtype=np.float32) for w in wavs], prompt=prompt
        )
        inputs = inputs.to(self.model.device)
        kwargs = generate_kwargs(self.max_new_tokens, decode)
        self.last_generate_kwargs = {**describe_kwargs(kwargs), "prompt": prompt}
        ids = self.model.generate(**inputs, **kwargs)
        texts = self.processor.decode(
            ids[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True
        )
        return [texts] if isinstance(texts, str) else list(texts)
