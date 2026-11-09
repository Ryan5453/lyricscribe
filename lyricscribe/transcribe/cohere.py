import logging

import numpy as np
import torch
from transformers import AutoProcessor, CohereAsrForConditionalGeneration
from transformers.models.cohere_asr.processing_cohere_asr import LANGUAGES

from lyricscribe.transcribe.adapter import check_train_record
from lyricscribe.transcribe.base import (
    DECODE_KNOBS,
    MAX_NEW_TOKENS,
    ChunkTranscriber,
    Decode,
    describe_kwargs,
    generate_kwargs,
    shared_prompt,
)
from lyricscribe.transcribe.canary import insert_context
from lyricscribe.transcribe.chunking import SAMPLE_RATE, language_code

logger = logging.getLogger(__name__)


class CohereTranscriber(ChunkTranscriber):
    """
    Cohere Transcribe (Conformer encoder, Transformer decoder).

    The known language is passed as the decoder's language token (English when
    it is unknown or unsupported). ``prompt_mode="shared"`` puts the shared
    prompt in the decoder's context slot (Canary's layout).
    """

    supports = frozenset(DECODE_KNOBS)
    prompt_modes = ("native", "shared")

    def __init__(
        self, model_name: str, batch_size: int = 8, max_new_tokens: int = MAX_NEW_TOKENS, prompt_mode: str | None = None
    ) -> None:
        """
        :param model_name: Hub id or checkpoint dir.
        :param batch_size: Chunks decoded per ``generate`` call.
        :param max_new_tokens: Generation cap per chunk, clamped to the
            decoder's position limit.
        :param prompt_mode: ``native`` (empty context, default) or ``shared``.
        """
        super().__init__(model_name, batch_size, prompt_mode)
        self.max_new_tokens = max_new_tokens
        self.model = None
        self.processor = None

    def load(self) -> None:
        """
        Load the model and processor onto the GPU (CPU without CUDA).

        :raises TrainRecordMismatch: If a fine-tuned checkpoint was trained with another prompt.
        """
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.train_record = check_train_record(self.model_name, self.prompt_mode)
        dtype = torch.bfloat16 if device == "cuda" else torch.float32
        logger.info(f"Loading Cohere Transcribe: {self.model_name} on {device}")
        # cuDNN SDPA rebuilds its graph per sequence shape, which stalls decoding.
        torch.backends.cuda.enable_cudnn_sdp(False)
        self.processor = AutoProcessor.from_pretrained(self.model_name)
        self.model = CohereAsrForConditionalGeneration.from_pretrained(
            self.model_name, dtype=dtype, device_map=device
        ).eval()

    @torch.inference_mode()
    def _decode(self, wavs: list[np.ndarray], language: str | None, decode: Decode) -> list[str]:
        """
        Decode chunks of one song in a single ``generate`` call.

        :param wavs: 16 kHz mono chunks.
        :param language: ``detected_language`` from ``lyrics.json``, or ``None``.
        :param decode: Decoding settings.
        :return: One transcription per chunk.
        """
        code = language_code(language) or "en"
        if code not in LANGUAGES:
            logger.warning(f"Cohere Transcribe does not support {code!r}; using 'en'")
            code = "en"
        inputs = self.processor(
            wavs, sampling_rate=SAMPLE_RATE, return_tensors="pt", language=code
        )
        chunk_index = inputs.pop("audio_chunk_index", None)
        inputs = inputs.to(self.model.device, dtype=self.model.dtype)
        prompt = shared_prompt(language) if self.prompt_mode == "shared" else None
        if prompt:
            insert_context(inputs, self.processor.tokenizer, prompt)
        prompt_len = inputs["decoder_input_ids"].shape[1]
        room = self.model.config.max_position_embeddings - prompt_len
        kwargs = generate_kwargs(min(self.max_new_tokens, room), decode)
        self.last_generate_kwargs = {**describe_kwargs(kwargs), "prompt": prompt}
        ids = self.model.generate(**inputs, **kwargs)
        # Generated ids start with the decoder prompt; a context there isn't special tokens.
        texts = self.processor.decode(
            ids[:, prompt_len:], skip_special_tokens=True, audio_chunk_index=chunk_index, language=code
        )
        return [texts] if isinstance(texts, str) else list(texts)
