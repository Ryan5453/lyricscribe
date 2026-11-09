import logging

import numpy as np
import torch
from transformers import AutoProcessor, CanaryForConditionalGeneration
from transformers.models.canary.processing_canary import LANGUAGE_CODE_TO_NAME

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
from lyricscribe.transcribe.chunking import language_code

logger = logging.getLogger(__name__)


def add_nemo_prefix(decoder_input_ids: torch.Tensor, tokenizer) -> torch.Tensor:
    """
    Prepend the SentencePiece dummy-prefix token (``▁``) to Canary decoder prompts.

    NeMo's ``canary2`` prompt format encodes the prompt string with
    SentencePiece, which starts it with ``▁`` before ``<|startofcontext|>``;
    the transformers chat template drops it. The released checkpoints were
    trained and evaluated with it.

    :param decoder_input_ids: ``(batch, length)`` prompt ids.
    :param tokenizer: Canary tokenizer.
    :return: ``(batch, length + 1)`` ids.
    """
    prefix = torch.full(
        (decoder_input_ids.shape[0], 1),
        tokenizer.convert_tokens_to_ids("▁"),
        dtype=decoder_input_ids.dtype,
        device=decoder_input_ids.device,
    )
    return torch.cat([prefix, decoder_input_ids], dim=1)


def insert_context(inputs, tokenizer, text: str) -> None:
    """
    Put text in the decoder prompt's context slot, right after ``<|startofcontext|>``
    (NeMo's ``canary2`` ``decodercontext``), in place. Canary and Cohere Transcribe
    share this prompt layout; neither vendor documents the slot as a prompt.

    :param inputs: Processor outputs holding ``decoder_input_ids`` (and possibly
        ``decoder_attention_mask``), every row with the same prompt layout.
    :param tokenizer: The model's tokenizer.
    :param text: Context text.
    :raises ValueError: If the rows don't share one ``<|startofcontext|>`` position.
    """
    ids = inputs["decoder_input_ids"]
    soc = tokenizer.convert_tokens_to_ids("<|startofcontext|>")
    hits = (ids == soc).nonzero()
    if len(hits) != ids.shape[0] or len(set(hits[:, 1].tolist())) != 1:
        raise ValueError("expected one <|startofcontext|> per row at the same position")
    pos = int(hits[0, 1]) + 1
    ctx = torch.tensor(tokenizer(text, add_special_tokens=False)["input_ids"], dtype=ids.dtype, device=ids.device)
    ctx = ctx.expand(ids.shape[0], -1)
    inputs["decoder_input_ids"] = torch.cat([ids[:, :pos], ctx, ids[:, pos:]], dim=1)
    mask = inputs.get("decoder_attention_mask")
    if mask is not None:
        ones = torch.ones(ctx.shape, dtype=mask.dtype, device=mask.device)
        inputs["decoder_attention_mask"] = torch.cat([mask[:, :pos], ones, mask[:, pos:]], dim=1)


class CanaryTranscriber(ChunkTranscriber):
    """
    NVIDIA Canary (FastConformer encoder, Transformer decoder) in fp32 with
    punctuation on.

    The known language is the source and target language (English when it is
    unknown or unsupported). Output is capped at the shared per-chunk cap,
    within the decoder's positions; NeMo's own encoder-frames + 50 cap is not
    used, so the cap doesn't depend on which chunks share a batch.
    ``prompt_mode="shared"`` puts the shared prompt in the decoder's context slot.
    """

    supports = frozenset(DECODE_KNOBS)
    prompt_modes = ("native", "shared")

    def __init__(
        self,
        model_name: str,
        batch_size: int = 8,
        max_new_tokens: int = MAX_NEW_TOKENS,
        dtype: str | None = None,
        prompt_mode: str | None = None,
    ) -> None:
        """
        :param model_name: Hub id or checkpoint dir in transformers format.
        :param batch_size: Chunks decoded per ``generate`` call.
        :param max_new_tokens: Generation cap per chunk, clamped to the
            decoder's position limit.
        :param dtype: ``float32`` (default) or ``bfloat16``.
        :param prompt_mode: ``native`` (empty context, default) or ``shared``.
        """
        super().__init__(model_name, batch_size, prompt_mode)
        self.dtype = getattr(torch, dtype or "float32")
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
        logger.info(f"Loading Canary: {self.model_name} on {device}")
        # cuDNN SDPA rebuilds its graph per sequence shape, which stalls decoding.
        torch.backends.cuda.enable_cudnn_sdp(False)
        self.processor = AutoProcessor.from_pretrained(self.model_name)
        self.model = CanaryForConditionalGeneration.from_pretrained(
            self.model_name, dtype=self.dtype, device_map=device
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
        if code not in LANGUAGE_CODE_TO_NAME:
            logger.warning(f"Canary does not support {code!r}; using 'en'")
            code = "en"
        inputs = self.processor.apply_transcription_request(
            [np.asarray(w, dtype=np.float32) for w in wavs], source_language=code, punctuation=True
        ).to(self.model.device, dtype=self.model.dtype)
        inputs["decoder_input_ids"] = add_nemo_prefix(inputs["decoder_input_ids"], self.processor.tokenizer)
        if "decoder_attention_mask" in inputs:
            mask = inputs["decoder_attention_mask"]
            inputs["decoder_attention_mask"] = torch.cat([mask[:, :1], mask], dim=1)
        prompt = shared_prompt(language) if self.prompt_mode == "shared" else None
        if prompt:
            insert_context(inputs, self.processor.tokenizer, prompt)
        prompt_len = inputs["decoder_input_ids"].shape[1]
        room = self.model.config.decoder_config.max_position_embeddings - prompt_len
        kwargs = generate_kwargs(min(self.max_new_tokens, room), decode)
        self.last_generate_kwargs = {**describe_kwargs(kwargs), "prompt": prompt}
        ids = self.model.generate(**inputs, **kwargs)
        # Generated ids start with the decoder prompt; a context there isn't special tokens.
        return self.processor.batch_decode(ids[:, prompt_len:], skip_special_tokens=True)
