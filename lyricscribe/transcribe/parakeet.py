import logging

import numpy as np
import torch
from transformers import (
    AutoConfig,
    AutoProcessor,
    ParakeetForTDT,
)

from lyricscribe.transcribe.adapter import check_train_record
from lyricscribe.transcribe.base import ChunkTranscriber, Decode
from lyricscribe.transcribe.chunking import SAMPLE_RATE

logger = logging.getLogger(__name__)

class NeMoGreedyParakeetForTDT(ParakeetForTDT):
    """
    ``ParakeetForTDT`` whose greedy decoding applies NeMo's per-frame symbol
    cap: after ``max_symbols_per_step`` labels at the same encoder frame, the
    frame pointer is forced one frame on. transformers' TDT generation skips
    this guard (its RNN-T generation has it), so a run of duration-0 labels can
    grow unbounded at one frame.
    """

    def generate(self, *args, **kwargs):
        """
        Greedy TDT decoding with the per-frame symbol cap.

        :param args: ``generate`` positional arguments.
        :param kwargs: ``generate`` keyword arguments.
        :return: ``ParakeetRNNTGenerateOutput`` with ``sequences`` and ``durations``.
        """
        self._labels_at_frame = None
        try:
            return super().generate(*args, **kwargs)
        finally:
            del self._labels_at_frame

    def _update_model_kwargs_for_generation(self, outputs, *args, **kwargs):
        """
        Advance by the predicted duration, then by one more frame wherever the
        current frame has reached the symbol cap.

        :param outputs: Model outputs for this step.
        :param args: Base-class positional arguments.
        :param kwargs: Base-class keyword arguments.
        :return: The updated model kwargs.
        """
        model_kwargs = super()._update_model_kwargs_for_generation(outputs, *args, **kwargs)
        tokens = outputs.logits[:, -1, : self.config.vocab_size].argmax(dim=-1)
        stayed = (tokens != self.config.blank_token_id) & (self._step_durations[-1] == 0)
        if self._labels_at_frame is None:
            self._labels_at_frame = torch.zeros_like(tokens)
        self._labels_at_frame = torch.where(stayed, self._labels_at_frame + 1, torch.zeros_like(tokens))
        force = (self._labels_at_frame >= self.max_symbols_per_step).long()
        self._labels_at_frame = self._labels_at_frame * (1 - force)
        self._step_durations[-1] = self._step_durations[-1] + force
        model_kwargs["encoder_frame_idxs"] = model_kwargs["encoder_frame_idxs"] + force
        self._encoder_finished = model_kwargs["encoder_frame_idxs"] >= model_kwargs["encoder_valid_lengths"]
        return model_kwargs


MODEL_CLASSES = {"parakeet_tdt": NeMoGreedyParakeetForTDT}


class ParakeetTranscriber(ChunkTranscriber):
    """
    NVIDIA Parakeet TDT (FastConformer encoder, token-and-duration transducer decoder),
    greedy decoding in fp32.

    The model is language-agnostic, so the language is ignored. It has no beam
    search or repetition controls in transformers, so asking for them raises.
    """

    def __init__(self, model_name: str, batch_size: int = 8, dtype: str | None = None) -> None:
        """
        :param model_name: Hub id or checkpoint dir in transformers format.
        :param batch_size: Chunks decoded per ``generate`` call.
        :param dtype: ``float32`` (default) or ``bfloat16``.
        """
        super().__init__(model_name, batch_size)
        self.dtype = getattr(torch, dtype or "float32")
        self.model = None
        self.processor = None

    def load(self) -> None:
        """
        Load the model and processor onto the GPU (CPU without CUDA).

        :raises ValueError: For a checkpoint whose ``model_type`` is not Parakeet TDT.
        :raises TrainRecordMismatch: If a fine-tuned checkpoint was trained with another prompt.
        """
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.train_record = check_train_record(self.model_name, self.prompt_mode)
        model_type = AutoConfig.from_pretrained(self.model_name).model_type
        if model_type not in MODEL_CLASSES:
            raise ValueError(f"{self.model_name}: not a Parakeet TDT model (model_type {model_type!r})")
        logger.info(f"Loading Parakeet ({model_type}): {self.model_name} on {device}")
        # cuDNN SDPA rebuilds its graph per sequence shape, which stalls decoding.
        torch.backends.cuda.enable_cudnn_sdp(False)
        self.processor = AutoProcessor.from_pretrained(self.model_name)
        self.model = MODEL_CLASSES[model_type].from_pretrained(
            self.model_name, dtype=self.dtype, device_map=device
        ).eval()

    @torch.inference_mode()
    def _decode(self, wavs: list[np.ndarray], language: str | None, decode: Decode) -> list[str]:
        """
        Decode chunks of one song in a single ``generate`` call.

        :param wavs: 16 kHz mono chunks.
        :param language: Ignored.
        :param decode: Greedy settings (checked by the caller).
        :return: One transcription per chunk.
        """
        del language, decode
        inputs = self.processor(wavs, sampling_rate=SAMPLE_RATE, return_tensors="pt").to(self.model.device, dtype=self.model.dtype)
        out = self.model.generate(**inputs)
        ids = out.sequences if hasattr(out, "sequences") else out
        if getattr(out, "durations", None) is not None:
            # generate() only pads finished rows when the config has an eos_token_id (NeMo
            # conversions lack it), so drop tokens emitted past each chunk's own frames.
            frames = self.model.encoder._get_subsampling_output_length(inputs["attention_mask"].sum(-1))
            emitted_at = torch.cumsum(out.durations, dim=1) - out.durations
            ids = ids.masked_fill(emitted_at >= frames[:, None], self.model.config.pad_token_id)
        return self.processor.batch_decode(ids, skip_special_tokens=True)
