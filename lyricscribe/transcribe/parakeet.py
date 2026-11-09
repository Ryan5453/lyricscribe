import logging

import numpy as np
import torch
from transformers import AutoConfig, AutoProcessor, ParakeetForTDT

from lyricscribe.transcribe.adapter import check_train_record
from lyricscribe.transcribe.base import DECODE_KNOBS, MAX_NEW_TOKENS, ChunkTranscriber, Decode
from lyricscribe.transcribe.chunking import SAMPLE_RATE
from lyricscribe.transcribe.tdt import TDTDecoder

logger = logging.getLogger(__name__)


class ParakeetTranscriber(ChunkTranscriber):
    """
    NVIDIA Parakeet TDT (FastConformer encoder, token-and-duration transducer decoder) in fp32,
    decoded by :class:`~lyricscribe.transcribe.tdt.TDTDecoder`: greedy, or NeMo's recommended TDT
    beam search (``malsd_batch``), with the shared repetition controls on emitted labels.

    The model is language-agnostic, so the language is ignored.
    """

    supports = frozenset(DECODE_KNOBS)

    def __init__(
        self, model_name: str, batch_size: int = 8, dtype: str | None = None, max_new_tokens: int = MAX_NEW_TOKENS
    ) -> None:
        """
        :param model_name: Hub id or checkpoint dir in transformers format.
        :param batch_size: Chunks decoded together.
        :param dtype: ``float32`` (default) or ``bfloat16``.
        :param max_new_tokens: Labels allowed per chunk.
        """
        super().__init__(model_name, batch_size)
        self.dtype = getattr(torch, dtype or "float32")
        self.max_new_tokens = max_new_tokens
        self.model = None
        self.processor = None
        self.tdt = None

    def load(self) -> None:
        """
        Load the model and processor onto the GPU (CPU without CUDA).

        :raises ValueError: For a checkpoint whose ``model_type`` is not Parakeet TDT.
        :raises TrainRecordMismatch: If a fine-tuned checkpoint was trained with another prompt.
        """
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.train_record = check_train_record(self.model_name, self.prompt_mode)
        model_type = AutoConfig.from_pretrained(self.model_name).model_type
        if model_type != "parakeet_tdt":
            raise ValueError(f"{self.model_name}: not a Parakeet TDT model (model_type {model_type!r})")
        logger.info(f"Loading Parakeet TDT: {self.model_name} on {device}")
        # cuDNN SDPA rebuilds its graph per sequence shape, which stalls decoding.
        torch.backends.cuda.enable_cudnn_sdp(False)
        self.processor = AutoProcessor.from_pretrained(self.model_name)
        self.model = ParakeetForTDT.from_pretrained(self.model_name, dtype=self.dtype, device_map=device).eval()
        cfg, gen = self.model.config, self.model.generation_config
        self.tdt = TDTDecoder(
            self._predict, self._joint, vocab=cfg.vocab_size, blank=cfg.blank_token_id, durations=list(cfg.durations),
            start=gen.decoder_start_token_id, eos=gen.eos_token_id, max_symbols=self.model.max_symbols_per_step,
            max_new_tokens=self.max_new_tokens,
        )

    def _predict(self, labels: torch.Tensor, state: tuple | None):
        """
        One prediction-network step, as transformers' ``ParakeetRNNTDecoder`` runs it.

        :param labels: ``(N,)`` previous labels.
        :param state: LSTM ``(h, c)``, or ``None`` for the zero state.
        :return: ``(decoder output (N, H), state)``.
        """
        dec = self.model.decoder
        emb = dec.embedding(labels[:, None])
        if state is None:
            zeros = torch.zeros(dec.lstm.num_layers, labels.shape[0], dec.lstm.hidden_size, device=emb.device, dtype=emb.dtype)
            state = (zeros, zeros.clone())
        out, state = dec.lstm(emb, state)
        return dec.decoder_projector(out)[:, 0], state

    def _joint(self, enc: torch.Tensor, dec: torch.Tensor) -> torch.Tensor:
        """
        Token and duration logits.

        :param enc: ``(N, H)`` projected encoder frames.
        :param dec: ``(N, H)`` decoder outputs.
        :return: ``(N, V + D)`` logits.
        """
        return self.model.joint(decoder_hidden_states=dec, encoder_hidden_states=enc)

    @torch.inference_mode()
    def _decode(self, wavs: list[np.ndarray], language: str | None, decode: Decode) -> list[str]:
        """
        Encode the chunks together, then decode them greedily or by beam search.

        :param wavs: 16 kHz mono chunks.
        :param language: Ignored.
        :param decode: Decoding settings (checked by the caller).
        :return: One transcription per chunk.
        """
        del language
        inputs = self.processor(wavs, sampling_rate=SAMPLE_RATE, return_tensors="pt").to(self.model.device, dtype=self.model.dtype)
        encoded = self.model.get_audio_features(
            input_features=inputs["input_features"], attention_mask=inputs["attention_mask"], output_attention_mask=True,
        )
        enc, lengths = encoded.pooler_output, encoded.attention_mask.sum(-1)
        if decode.num_beams > 1:
            labels = self.tdt.beam(enc, lengths, decode.num_beams, decode)
        else:
            labels = self.tdt.greedy(enc, lengths, decode)
        self.last_generate_kwargs = {
            **decode.as_dict(),
            "search": "nemo malsd_batch (length-normalised)" if decode.num_beams > 1 else "greedy",
            "max_new_tokens": self.max_new_tokens,
            "max_symbols_per_step": self.tdt.max_symbols,
            "eos_token_id": self.tdt.eos,
        }
        return self.processor.batch_decode(labels, skip_special_tokens=True)
