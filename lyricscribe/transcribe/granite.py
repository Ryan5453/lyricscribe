import inspect
import logging

import numpy as np
import torch
from transformers import AutoConfig, AutoModelForSpeechSeq2Seq, AutoProcessor

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
from lyricscribe.transcribe.chunking import language_name

logger = logging.getLogger(__name__)

ASR_PROMPT = "can you transcribe the speech into a written format?"
LANGUAGE_PROMPT = "The speech is in {language}. Transcribe it exactly; do not translate. " + ASR_PROMPT
PROMPT_LANGUAGES = ("English", "French", "German", "Spanish", "Portuguese", "Japanese")



def apply_logits_scaling(model) -> float | None:
    """
    Divide the speech LLM's logits by its text config's ``logits_scaling``.
    ``GraniteSpeechForConditionalGeneration`` (transformers 5.x) applies its own ``lm_head`` to the
    base text model and skips the division ``GraniteForCausalLM`` does, so its logits are
    ``logits_scaling`` times too sharp. Greedy argmax is unchanged; training losses, beam scores and
    log-probability repetition penalties are not. Registered as a forward hook, so exported
    checkpoints stay vendor-format and every load re-applies it.

    :param model: ``GraniteSpeechForConditionalGeneration``.
    :return: The factor divided out, or ``None`` when it is 1 or the model already applies it
        (transformers 4.x delegates to ``GraniteForCausalLM`` and has no top-level ``lm_head``).
    """
    scale = float(getattr(model.config.text_config, "logits_scaling", 1.0) or 1.0)
    head = getattr(model, "lm_head", None)
    if scale == 1.0 or head is None or "logits_scaling" in inspect.getsource(type(model).forward):
        return None
    if getattr(head, "_logits_scaling", None) is None:
        head.register_forward_hook(lambda module, args, out: out / scale)
        head._logits_scaling = scale
    return scale


class GraniteTranscriber(ChunkTranscriber):
    """
    IBM Granite Speech 5.0 2B, the speech LLM. It takes the known language through the model
    card's language-ID prompt. Its encoder has no padding mask, so it decodes one chunk at a time.
    """

    supports = frozenset(DECODE_KNOBS)
    prompt_modes = ("asr", "shared")

    def __init__(
        self, model_name: str, batch_size: int = 8, max_new_tokens: int = MAX_NEW_TOKENS, prompt_mode: str | None = None
    ) -> None:
        """
        :param model_name: Hub id or checkpoint dir.
        :param batch_size: Chunks decoded per call.
        :param max_new_tokens: Generation cap per chunk.
        :param prompt_mode: ``asr`` (the card's language-ID ASR prompt, default) or ``shared``.
        """
        super().__init__(model_name, batch_size, prompt_mode)
        self.max_new_tokens = max_new_tokens
        self.model = None
        self.processor = None
        self.logits_scaling = None

    def load(self) -> None:
        """
        Load the model and processor onto the GPU (CPU without CUDA).

        :raises ValueError: For a checkpoint that is not the Granite Speech LLM.
        :raises TrainRecordMismatch: If a fine-tuned checkpoint was trained with another prompt.
        """
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model_type = AutoConfig.from_pretrained(self.model_name).model_type
        if model_type != "granite_speech":
            raise ValueError(f"{self.model_name}: model_type {model_type!r} is not the Granite Speech LLM")
        self.train_record = check_train_record(self.model_name, self.prompt_mode)
        dtype = torch.bfloat16 if device == "cuda" else torch.float32
        logger.info(f"Loading Granite Speech: {self.model_name} on {device}")
        # cuDNN SDPA rebuilds its graph per sequence shape, which stalls decoding.
        torch.backends.cuda.enable_cudnn_sdp(False)
        if self.batch_size != 1:
            logger.warning(f"Granite speech LLM: batch_size {self.batch_size} -> 1 (unmasked encoder padding)")
            self.batch_size = 1
        self.processor = AutoProcessor.from_pretrained(self.model_name)
        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(self.model_name, dtype=dtype, device_map=device).eval()
        self.logits_scaling = apply_logits_scaling(self.model)

    @torch.inference_mode()
    def _decode(self, wavs: list[np.ndarray], language: str | None, decode: Decode) -> list[str]:
        """
        Decode chunks of one song in a single call.

        :param wavs: 16 kHz mono chunks.
        :param language: ``detected_language`` from ``lyrics.json``, or ``None``.
        :param decode: Decoding settings.
        :return: One transcription per chunk.
        """
        device = self.model.device
        audio = [torch.from_numpy(np.asarray(w, dtype=np.float32)) for w in wavs]
        if self.prompt_mode == "shared":
            text = shared_prompt(language)
        else:
            name = language_name(language)
            text = LANGUAGE_PROMPT.format(language=name) if name in PROMPT_LANGUAGES else ASR_PROMPT
        tokenizer = self.processor.tokenizer
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": "<|audio|>" + text}],
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self.processor([prompt] * len(audio), audio, device=device, return_tensors="pt")
        inputs = inputs.to(device)
        kwargs = generate_kwargs(self.max_new_tokens, decode)
        self.last_generate_kwargs = {**describe_kwargs(kwargs), "prompt": text, "logits_scaling": self.logits_scaling}
        ids = self.model.generate(**inputs, **kwargs)
        return tokenizer.batch_decode(
            ids[:, inputs["input_ids"].shape[1]:], add_special_tokens=False, skip_special_tokens=True
        )
