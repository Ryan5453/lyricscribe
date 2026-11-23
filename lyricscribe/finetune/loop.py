import copy
import json
import re
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from safetensors.torch import load_file, save_file
from torch import nn
from transformers import (
    AutoProcessor,
    GenerationConfig,
    Qwen3ASRConfig,
    Qwen3ASRForConditionalGeneration,
)

LOOP_CONFIG_FILE = "loop_config.json"
LOOP_VARIANTS = ("inkling_add_2b", "enc_loop_2a", "double_loop_3", "pruned_ctrl")

_LATIN_KEEP_RE = re.compile(
    r"^[\x00-\x7F\u00C0-\u024F\u1E00-\u1EFF\u2018-\u201F\u2013\u2014\u2026\u00A1\u00BF\u00AB\u00BB]+$"
)


class DepthCursor:
    """
    Mutable virtual-pass cursor shared by all depth-conditioned leaves inside one looped block.

    :param n_passes: Number of virtual passes handled by the shared block.
    """

    def __init__(self, n_passes: int):
        self.n_passes = n_passes
        self.depth = 0


class LoopLinear(nn.Module):
    """
    Weight-shared linear projection with per-pass bias and SVD-initialised low-rank adapter:
    ``y = W x + b[d] + B[d](A[d] x)``.

    :param cursor: Shared :class:`DepthCursor`.
    :param n_passes: Number of virtual depths.
    :param in_features: Input dimension.
    :param out_features: Output dimension.
    :param rank: Per-pass adapter rank (``0`` disables adapters).
    :param bias: Whether to keep a per-pass bias.
    """

    def __init__(
        self,
        cursor: DepthCursor,
        n_passes: int,
        in_features: int,
        out_features: int,
        rank: int,
        bias: bool,
    ):
        super().__init__()
        object.__setattr__(self, "cursor", cursor)
        self.n_passes = n_passes
        self.in_features = in_features
        self.out_features = out_features
        self.rank = min(rank, in_features, out_features)
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(n_passes, out_features)) if bias else None
        if self.rank > 0:
            self.lora_A = nn.Parameter(torch.zeros(n_passes, self.rank, in_features))
            self.lora_B = nn.Parameter(torch.zeros(n_passes, out_features, self.rank))

    @torch.no_grad()
    def init_from_linears(self, linears: Sequence[nn.Linear], init_mode: str = "mean") -> None:
        """
        Initialise the shared weight from *linears* and fit each pass's residual via truncated SVD.

        :param linears: Original unshared ``nn.Linear`` layers in pass order.
        :param init_mode: ``"mean"``, ``"anchor"`` (middle layer) or ``"first"``.
        """
        if len(linears) != self.n_passes:
            raise ValueError(f"Expected {self.n_passes} linears, got {len(linears)}")
        dtype = linears[0].weight.dtype
        weights = torch.stack([lin.weight.detach().float() for lin in linears], dim=0)
        if init_mode == "mean":
            base_w = weights.mean(dim=0)
        elif init_mode == "anchor":
            base_w = weights[self.n_passes // 2]
        elif init_mode == "first":
            base_w = weights[0]
        else:
            raise ValueError(f"Unknown init_mode {init_mode!r}")
        self.weight.data.copy_(base_w.to(dtype))

        if self.bias is not None:
            biases = torch.stack([lin.bias.detach().float() for lin in linears], dim=0)
            self.bias.data.copy_(biases.to(dtype))

        if self.rank > 0:
            for d in range(self.n_passes):
                residual = weights[d] - base_w
                if residual.abs().max().item() < 1e-8:
                    self.lora_B.data[d].zero_()
                    self.lora_A.data[d].zero_()
                    continue
                u, s, vh = torch.linalg.svd(residual, full_matrices=False)
                r = self.rank
                sqrt_s = torch.sqrt(s[:r].clamp_min(0.0))
                self.lora_B.data[d].copy_((u[:, :r] * sqrt_s.unsqueeze(0)).to(dtype))
                self.lora_A.data[d].copy_((sqrt_s.unsqueeze(1) * vh[:r, :]).to(dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        d = self.cursor.depth
        b = None if self.bias is None else self.bias[d]
        y = F.linear(x, self.weight, b)
        if self.rank > 0:
            y = y + F.linear(F.linear(x, self.lora_A[d]), self.lora_B[d])
        return y


class LoopLayerNorm(nn.Module):
    """Per-pass LayerNorm paired with a shared :class:`DepthCursor`."""

    def __init__(self, cursor: DepthCursor, n_passes: int, normalized_shape: tuple[int, ...], eps: float):
        super().__init__()
        object.__setattr__(self, "cursor", cursor)
        self.n_passes = n_passes
        self.normalized_shape = tuple(normalized_shape)
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(n_passes, *self.normalized_shape))
        self.bias = nn.Parameter(torch.zeros(n_passes, *self.normalized_shape))

    @torch.no_grad()
    def init_from_norms(self, norms: Sequence[nn.LayerNorm]) -> None:
        dtype = norms[0].weight.dtype
        self.weight.data.copy_(torch.stack([n.weight.detach() for n in norms], dim=0).to(dtype))
        self.bias.data.copy_(torch.stack([n.bias.detach() for n in norms], dim=0).to(dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        d = self.cursor.depth
        return F.layer_norm(x, self.normalized_shape, self.weight[d], self.bias[d], self.eps)


class LoopRMSNorm(nn.Module):
    """Per-pass RMSNorm (covers Qwen3 layer norms as well as ``q_norm`` / ``k_norm``)."""

    def __init__(self, cursor: DepthCursor, n_passes: int, dim: int, eps: float):
        super().__init__()
        object.__setattr__(self, "cursor", cursor)
        self.n_passes = n_passes
        self.variance_epsilon = eps
        self.weight = nn.Parameter(torch.ones(n_passes, dim))

    @torch.no_grad()
    def init_from_norms(self, norms: Sequence[nn.Module]) -> None:
        dtype = norms[0].weight.dtype
        self.weight.data.copy_(torch.stack([n.weight.detach() for n in norms], dim=0).to(dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        d = self.cursor.depth
        orig_dtype = x.dtype
        h = x.to(torch.float32)
        h = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + self.variance_epsilon)
        return (self.weight[d].to(orig_dtype) * h.to(orig_dtype)).to(orig_dtype)


class LoopCall(nn.Module):
    """
    Parameter-free call site invoking a shared looped block at virtual depth ``pass_idx``,
    with an optional zero-initialised learned gate re-injecting the group's entry state.
    """

    def __init__(
        self,
        block: nn.Module,
        cursor: DepthCursor,
        pass_idx: int,
        kv_slot: int | None = None,
        entry_holder: list[torch.Tensor | None] | None = None,
        is_group_start: bool = False,
        use_entry_gate: bool = False,
    ):
        super().__init__()
        object.__setattr__(self, "_block", block)
        object.__setattr__(self, "_cursor", cursor)
        object.__setattr__(self, "_entry_holder", entry_holder)
        self.pass_idx = pass_idx
        self.kv_slot = kv_slot
        self.is_group_start = is_group_start
        self.use_entry_gate = use_entry_gate
        self.entry_gate = nn.Parameter(torch.zeros(1)) if (use_entry_gate and not is_group_start) else None

    def forward(self, hidden_states: torch.Tensor, *args, **kwargs):
        if self._entry_holder is not None:
            if self.is_group_start:
                self._entry_holder[0] = hidden_states
            elif self.entry_gate is not None and self._entry_holder[0] is not None:
                entry = self._entry_holder[0]
                if entry.shape == hidden_states.shape:
                    hidden_states = hidden_states + torch.tanh(self.entry_gate).to(hidden_states.dtype) * entry

        self._cursor.depth = self.pass_idx
        if self.kv_slot is not None and hasattr(self._block, "self_attn"):
            self._block.self_attn.layer_idx = self.kv_slot
        return self._block(hidden_states, *args, **kwargs)


class AdditiveLoopEntry(nn.Module):
    """Record the hidden state entering an additive-loop span (zero extra parameters)."""

    def __init__(self, layer: nn.Module, entry_holder: list[torch.Tensor | None]):
        super().__init__()
        self.layer = layer
        object.__setattr__(self, "_entry_holder", entry_holder)

    def forward(self, hidden_states: torch.Tensor, *args, **kwargs):
        self._entry_holder[0] = hidden_states
        return self.layer(hidden_states, *args, **kwargs)


class AdditiveLoopTail(nn.Module):
    """
    Inkling-2-style zero-initialised additive extra pass over an intact middle layer span.

    Installed at the final layer ``end - 1`` of ``[start, end)``:
      1. Runs the intact tail layer normally (yielding exact base hidden state ``h``).
      2. Re-runs the intact span layers for ``extra_passes`` additional passes (allocating
         dedicated KV-cache slots when ``past_key_values`` is active).
      3. Blends each extra pass via a zero-initialised gate:
         ``h = h + tanh(loop_gates[p]) * (h_loop - h)``.

    At initialisation (``loop_gates == 0``), outputs match the base model bit-for-bit
    while gradients into ``loop_gates`` are non-zero from step 1.
    """

    def __init__(
        self,
        tail_layer: nn.Module,
        span_layers: Sequence[nn.Module],
        extra_passes: int = 1,
        base_kv_offset: int | None = None,
        entry_holder: list[torch.Tensor | None] | None = None,
    ):
        super().__init__()
        self.tail_layer = tail_layer
        object.__setattr__(self, "_span_layers", list(span_layers))
        object.__setattr__(self, "_entry_holder", entry_holder)
        self.extra_passes = extra_passes
        self.base_kv_offset = base_kv_offset
        self.loop_gates = nn.Parameter(torch.zeros(extra_passes))
        self.entry_gates = nn.Parameter(torch.zeros(extra_passes))

    @staticmethod
    def _run_layer(layer: nn.Module, h: torch.Tensor, args: tuple, kwargs: dict) -> tuple[torch.Tensor, Any]:
        out = layer(h, *args, **kwargs)
        if isinstance(out, tuple):
            return out[0], out[1:]
        return out, None

    def forward(self, hidden_states: torch.Tensor, *args, **kwargs):
        h, rest = self._run_layer(self.tail_layer, hidden_states, args, kwargs)

        span_len = len(self._span_layers)
        pkv = kwargs.get("past_key_values")
        for p in range(self.extra_passes):
            h_in = h
            if self._entry_holder is not None and self._entry_holder[0] is not None:
                entry = self._entry_holder[0]
                if entry.shape == h_in.shape:
                    h_in = h_in + torch.tanh(self.entry_gates[p]).to(h_in.dtype) * entry

            h_loop = h_in
            for k, lay in enumerate(self._span_layers):
                orig_idx = None
                if self.base_kv_offset is not None and hasattr(lay, "self_attn"):
                    orig_idx = lay.self_attn.layer_idx
                    slot = self.base_kv_offset + p * span_len + k
                    lay.self_attn.layer_idx = slot
                    if pkv is not None and hasattr(pkv, "layers"):
                        while len(pkv.layers) <= slot:
                            layer_cls = type(pkv.layers[0]) if len(pkv.layers) > 0 else None
                            if layer_cls is None:
                                break
                            pkv.layers.append(layer_cls())
                try:
                    h_loop, _ = self._run_layer(lay, h_loop, args, kwargs)
                finally:
                    if orig_idx is not None:
                        lay.self_attn.layer_idx = orig_idx

            gate = torch.tanh(self.loop_gates[p]).to(h.dtype)
            h = h + gate * (h_loop - h)

        return (h, *rest) if rest is not None else h


def select_multilingual_latin_vocab(
    tokenizer,
    full_vocab_size: int,
    max_exact: int = 1024,
    extra_keep_ids: set[int] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Select token IDs covering Latin/European scripts (en/es/fr/de), byte tokens and all
    special/audio tokens while preserving original Qwen3 token IDs.

    :param tokenizer: Qwen3 tokenizer.
    :param full_vocab_size: Embedding matrix row count (e.g. ``151936``).
    :param max_exact: Number of frequent non-special tokens stored without rank factorisation.
    :param extra_keep_ids: Additional token IDs forced to stay exact.
    :return: ``(keep_ids, exact_mask_over_kept)``.
    """
    special = set(tokenizer.all_special_ids)
    if extra_keep_ids:
        special |= extra_keep_ids

    keep: list[int] = []
    exact_flags: list[bool] = []
    base_vocab = tokenizer.vocab_size
    for tid in range(full_vocab_size):
        if tid < 256 or tid >= base_vocab or tid in special:
            keep.append(tid)
            exact_flags.append(True)
            continue
        piece = tokenizer.decode([tid], skip_special_tokens=False)
        if piece and _LATIN_KEEP_RE.match(piece):
            keep.append(tid)
            exact_flags.append(len(keep) <= max_exact)

    return torch.tensor(keep, dtype=torch.long), torch.tensor(exact_flags, dtype=torch.bool)


class FactorizedTokenTable(nn.Module):
    """
    Tied pruned + rank-factorised embedding/LM head keeping original Qwen3 token IDs.

    Frequent/special rows are stored exactly; rare kept rows are factorised as ``U @ P``
    via truncated SVD. Non-kept tokens receive ``-inf`` logits in :class:`TableLMHead`.
    """

    def __init__(
        self,
        full_weight: torch.Tensor,
        keep_ids: torch.Tensor,
        exact_mask: torch.Tensor,
        rank: int = 384,
    ):
        super().__init__()
        n_full, dim = full_weight.shape
        n_kept = int(keep_ids.numel())
        exact_pos = torch.nonzero(exact_mask, as_tuple=False).squeeze(-1)
        rare_pos = torch.nonzero(~exact_mask, as_tuple=False).squeeze(-1)
        rank = min(rank, dim, max(int(rare_pos.numel()), 1))

        self.n_full = n_full
        self.embedding_dim = dim
        self.rank = rank

        lut = torch.zeros(n_full, dtype=torch.long)
        lut[keep_ids] = torch.arange(n_kept, dtype=torch.long)
        self.register_buffer("keep_ids", keep_ids.clone())
        self.register_buffer("lut", lut)
        self.register_buffer("exact_pos", exact_pos)
        self.register_buffer("rare_pos", rare_pos)

        dtype = full_weight.dtype
        kept_w = full_weight.detach()[keep_ids].float()
        self.exact_rows = nn.Parameter(kept_w[exact_pos].to(dtype))

        rare_w = kept_w[rare_pos]
        u, s, vh = torch.linalg.svd(rare_w, full_matrices=False)
        sqrt_s = torch.sqrt(s[:rank].clamp_min(0.0))
        self.U = nn.Parameter((u[:, :rank] * sqrt_s.unsqueeze(0)).to(dtype))
        self.P = nn.Parameter((sqrt_s.unsqueeze(1) * vh[:rank, :]).to(dtype))
        object.__setattr__(self, "_cached_w", None)

    @torch.no_grad()
    def merge_for_eval(self, dtype: torch.dtype | None = None) -> None:
        object.__setattr__(self, "_cached_w", None)
        object.__setattr__(self, "_cached_w", self.assembled_kept_weight(dtype).detach())

    def unmerge(self) -> None:
        object.__setattr__(self, "_cached_w", None)

    def assembled_kept_weight(self, dtype: torch.dtype | None = None) -> torch.Tensor:
        if self._cached_w is not None:
            if dtype is None or self._cached_w.dtype == dtype:
                return self._cached_w
            return self._cached_w.to(dtype)
        dtype = dtype or self.U.dtype
        w = torch.empty(
            (self.keep_ids.shape[0], self.embedding_dim),
            dtype=dtype,
            device=self.U.device,
        )
        w = w.index_copy(0, self.exact_pos, self.exact_rows.to(dtype))
        rare = (self.U.float() @ self.P.float()).to(dtype)
        return w.index_copy(0, self.rare_pos, rare)

    def embed(self, input_ids: torch.Tensor) -> torch.Tensor:
        return F.embedding(self.lut[input_ids], self.assembled_kept_weight())


class TableEmbedding(nn.Module):
    """Drop-in ``embed_tokens`` backed by :class:`FactorizedTokenTable`."""

    def __init__(self, table: FactorizedTokenTable):
        super().__init__()
        self.table = table
        self.embedding_dim = table.embedding_dim

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.table.embed(input_ids)


class TableLMHead(nn.Module):
    """Drop-in ``lm_head`` tied to :class:`FactorizedTokenTable` without duplicate parameters."""

    def __init__(self, table: FactorizedTokenTable):
        super().__init__()
        object.__setattr__(self, "_table", table)
        self.in_features = table.embedding_dim
        self.out_features = table.n_full

    def forward_kept(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return F.linear(hidden_states, self._table.assembled_kept_weight(hidden_states.dtype))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        kept_logits = self.forward_kept(hidden_states)
        out = torch.full(
            hidden_states.shape[:-1] + (self._table.n_full,),
            float("-inf"),
            dtype=kept_logits.dtype,
            device=kept_logits.device,
        )
        return out.index_copy_(-1, self._table.keep_ids, kept_logits)


def install_factorized_vocab(
    model: nn.Module,
    tokenizer,
    rank: int = 384,
    max_exact: int = 1024,
) -> dict[str, Any]:
    """
    Replace ``embed_tokens`` and ``lm_head`` with a tied :class:`FactorizedTokenTable`.

    :param model: Qwen3-ASR model.
    :param tokenizer: Matching tokenizer.
    :param rank: SVD factorisation rank for rare kept rows.
    :param max_exact: Exact frequent rows kept without factorisation.
    :return: Summary dictionary with kept token count and parameter count.
    """
    emb_w = model.model.language_model.embed_tokens.weight
    full_w = emb_w.detach().cpu()
    keep_ids, exact_mask = select_multilingual_latin_vocab(
        tokenizer, full_vocab_size=full_w.shape[0], max_exact=max_exact
    )
    table = FactorizedTokenTable(full_w, keep_ids, exact_mask, rank=rank).to(emb_w.device)
    model.model.language_model.embed_tokens = TableEmbedding(table)
    model.lm_head = TableLMHead(table)
    return {
        "kept_tokens": int(keep_ids.numel()),
        "exact_tokens": int(exact_mask.sum().item()),
        "vocab_params_M": sum(p.numel() for p in table.parameters()) / 1e6,
    }


def _fold_layer_subset(
    layers: Sequence[nn.Module],
    rank: int,
    init_mode: str = "mean",
    zero_lora: bool = False,
) -> tuple[nn.Module, DepthCursor]:
    """Collapse *layers* into one shared layer with per-pass SVD adapters and norms."""
    n_passes = len(layers)
    cursor = DepthCursor(n_passes)
    shared = copy.deepcopy(layers[0])

    for name in [n for n, _ in shared.named_modules() if n]:
        sub0 = shared.get_submodule(name)
        subs = [lay.get_submodule(name) for lay in layers]
        if isinstance(sub0, nn.Linear):
            replacement = LoopLinear(
                cursor,
                n_passes,
                sub0.in_features,
                sub0.out_features,
                rank=rank,
                bias=sub0.bias is not None,
            ).to(device=sub0.weight.device, dtype=sub0.weight.dtype)
            replacement.init_from_linears(subs, init_mode=init_mode)
            if zero_lora and replacement.rank > 0:
                replacement.lora_A.data.zero_()
                replacement.lora_B.data.zero_()
        elif isinstance(sub0, nn.LayerNorm):
            replacement = LoopLayerNorm(
                cursor, n_passes, sub0.normalized_shape, sub0.eps
            ).to(device=sub0.weight.device, dtype=sub0.weight.dtype)
            replacement.init_from_norms(subs)
        elif type(sub0).__name__.endswith("RMSNorm"):
            replacement = LoopRMSNorm(
                cursor, n_passes, sub0.weight.shape[0], sub0.variance_epsilon
            ).to(device=sub0.weight.device, dtype=sub0.weight.dtype)
            replacement.init_from_norms(subs)
        else:
            continue

        parent_name, _, child = name.rpartition(".")
        parent = shared.get_submodule(parent_name) if parent_name else shared
        setattr(parent, child, replacement)

    return shared, cursor


@dataclass
class LoopGroupSpec:
    """Fold original layers ``[start, end)`` into ``block_size`` physical layers looped R times."""

    start: int
    end: int
    block_size: int
    rank: int
    use_entry_gate: bool = True
    init_mode: str = "mean"
    zero_lora: bool = False


@dataclass
class AdditiveLoopSpec:
    """Run intact layers ``[start, end)`` for ``extra_passes`` additional zero-gated passes."""

    start: int
    end: int
    extra_passes: int = 1


def apply_surgical_loop_folding(
    model: nn.Module,
    enc_groups: Sequence[LoopGroupSpec],
    dec_groups: Sequence[LoopGroupSpec],
    in_place: bool = False,
) -> nn.Module:
    """
    Fold specified encoder/decoder spans into weight-shared looped blocks with per-pass
    SVD-initialised LoRA and per-pass norms.

    :param model: Base Qwen3-ASR model.
    :param enc_groups: Encoder fold specifications.
    :param dec_groups: Decoder fold specifications.
    :param in_place: Mutate *model* in place rather than deep-copying.
    :return: Folded model.
    """
    folded = model if in_place else copy.deepcopy(model)
    shared_registry = nn.ModuleList()

    def fold_stack(layer_list: nn.ModuleList, groups: Sequence[LoopGroupSpec], is_decoder: bool) -> None:
        orig_layers = list(layer_list)
        new_layers: list[nn.Module] = list(orig_layers)

        for g in groups:
            span = g.end - g.start
            if span <= 0 or span % g.block_size != 0:
                raise ValueError(
                    f"Span [{g.start}, {g.end}) len={span} must be divisible by block_size={g.block_size}"
                )
            n_loops = span // g.block_size
            entry_holder: list[torch.Tensor | None] = [None]

            for b in range(g.block_size):
                subset_indices = [g.start + r * g.block_size + b for r in range(n_loops)]
                subset_layers = [orig_layers[idx] for idx in subset_indices]
                shared_block, cursor = _fold_layer_subset(
                    subset_layers, rank=g.rank, init_mode=g.init_mode, zero_lora=g.zero_lora
                )
                shared_registry.append(shared_block)

                for r, orig_idx in enumerate(subset_indices):
                    new_layers[orig_idx] = LoopCall(
                        block=shared_block,
                        cursor=cursor,
                        pass_idx=r,
                        kv_slot=orig_idx if is_decoder else None,
                        entry_holder=entry_holder if b == 0 else None,
                        is_group_start=(r == 0 and b == 0),
                        use_entry_gate=(g.use_entry_gate and b == 0 and r > 0),
                    )

        for idx, mod in enumerate(new_layers):
            layer_list[idx] = mod

    fold_stack(folded.model.audio_tower.layers, enc_groups, is_decoder=False)
    fold_stack(folded.model.language_model.layers, dec_groups, is_decoder=True)
    folded._loop_shared_blocks = shared_registry
    return folded


def apply_additive_loop(
    model: nn.Module,
    enc_loops: Sequence[AdditiveLoopSpec],
    dec_loops: Sequence[AdditiveLoopSpec],
    in_place: bool = False,
) -> nn.Module:
    """
    Keep all base layers intact (exact ``KL = 0`` at step 0) and append zero-gated extra
    passes over the specified middle spans.

    :param model: Base Qwen3-ASR model.
    :param enc_loops: Encoder additive loop spans.
    :param dec_loops: Decoder additive loop spans.
    :param in_place: Mutate *model* in place rather than deep-copying.
    :return: Looped model.
    """
    looped = model if in_place else copy.deepcopy(model)

    def wrap_stack(layer_list: nn.ModuleList, specs: Sequence[AdditiveLoopSpec], is_decoder: bool) -> None:
        next_kv = len(layer_list) if is_decoder else None
        for sp in specs:
            if not (0 <= sp.start < sp.end <= len(layer_list)):
                raise ValueError(f"Invalid additive span [{sp.start}, {sp.end}) for {len(layer_list)} layers")
            entry_holder: list[torch.Tensor | None] = [None]
            orig_span = [layer_list[i] for i in range(sp.start, sp.end)]
            layer_list[sp.start] = AdditiveLoopEntry(orig_span[0], entry_holder)
            layer_list[sp.end - 1] = AdditiveLoopTail(
                tail_layer=orig_span[-1],
                span_layers=orig_span,
                extra_passes=sp.extra_passes,
                base_kv_offset=next_kv,
                entry_holder=entry_holder,
            )
            if next_kv is not None:
                next_kv += sp.extra_passes * len(orig_span)

    wrap_stack(looped.model.audio_tower.layers, enc_loops, is_decoder=False)
    wrap_stack(looped.model.language_model.layers, dec_loops, is_decoder=True)
    return looped


def prune_stack_control(
    model: nn.Module,
    enc_keep: Sequence[int],
    dec_keep: Sequence[int],
    in_place: bool = False,
) -> nn.Module:
    """
    Parameter-matched unlooped control keeping only the listed layer indices.

    :param model: Base Qwen3-ASR model.
    :param enc_keep: Encoder layer indices to keep.
    :param dec_keep: Decoder layer indices to keep.
    :param in_place: Mutate *model* in place rather than deep-copying.
    :return: Pruned model.
    """
    pruned = model if in_place else copy.deepcopy(model)
    enc_orig = list(pruned.model.audio_tower.layers)
    dec_orig = list(pruned.model.language_model.layers)

    pruned.model.audio_tower.layers = nn.ModuleList([enc_orig[i] for i in enc_keep])
    new_dec = []
    for new_idx, old_idx in enumerate(dec_keep):
        lay = dec_orig[old_idx]
        if hasattr(lay, "self_attn"):
            lay.self_attn.layer_idx = new_idx
        new_dec.append(lay)
    pruned.model.language_model.layers = nn.ModuleList(new_dec)
    pruned.config.audio_config.encoder_layers = len(enc_keep)
    pruned.config.text_config.num_hidden_layers = len(dec_keep)
    return pruned


def variant_presets(n_enc: int = 18, n_dec: int = 28) -> dict[str, dict[str, Any]]:
    """
    CKA-guided looping and pruning presets for Qwen3-ASR (0.6B ``18+28`` and 1.7B ``24+28``).

    :param n_enc: Encoder layer count.
    :param n_dec: Decoder layer count.
    :return: Mapping from variant name to specification dict.
    """
    if n_enc == 18:
        enc_add = [asdict(AdditiveLoopSpec(2, 12, extra_passes=1))]
        enc_fold = [
            asdict(LoopGroupSpec(2, 12, block_size=2, rank=32, init_mode="anchor", zero_lora=True)),
            asdict(LoopGroupSpec(12, 17, block_size=1, rank=32, init_mode="anchor", zero_lora=True)),
        ]
        enc_fold_mean = [
            asdict(LoopGroupSpec(2, 12, block_size=2, rank=32, init_mode="mean")),
            asdict(LoopGroupSpec(12, 17, block_size=1, rank=32, init_mode="mean")),
        ]
        enc_prune = [0, 1, 4, 7, 10, 13, 15, 17]
    elif n_enc == 24:
        enc_add = [asdict(AdditiveLoopSpec(4, 16, extra_passes=1))]
        enc_fold = [
            asdict(LoopGroupSpec(2, 18, block_size=4, rank=32, init_mode="anchor", zero_lora=True)),
            asdict(LoopGroupSpec(18, 23, block_size=1, rank=32, init_mode="anchor", zero_lora=True)),
        ]
        enc_fold_mean = [
            asdict(LoopGroupSpec(2, 18, block_size=4, rank=32, init_mode="mean")),
            asdict(LoopGroupSpec(18, 23, block_size=1, rank=32, init_mode="mean")),
        ]
        enc_prune = [0, 1, 4, 7, 10, 13, 16, 19, 21, 23]
    else:
        raise ValueError(f"No built-in loop preset for n_enc={n_enc} (expected 18 or 24)")

    if n_dec != 28:
        raise ValueError(f"No built-in loop preset for n_dec={n_dec} (expected 28)")

    dec_add = [asdict(AdditiveLoopSpec(6, 18, extra_passes=1))]
    dec_fold = [
        asdict(LoopGroupSpec(2, 18, block_size=2, rank=32, init_mode="mean")),
        asdict(LoopGroupSpec(18, 26, block_size=2, rank=32, init_mode="mean")),
    ]
    dec_prune = [0, 1, 4, 7, 10, 13, 16, 19, 22, 26, 27]

    return {
        "inkling_add_2b": {
            "kind": "additive",
            "enc_loops": enc_add,
            "dec_loops": dec_add,
            "vocab_rank": 0,
        },
        "enc_loop_2a": {
            "kind": "fold",
            "enc_groups": enc_fold,
            "dec_groups": [],
            "vocab_rank": 384,
        },
        "double_loop_3": {
            "kind": "fold",
            "enc_groups": enc_fold_mean,
            "dec_groups": dec_fold,
            "vocab_rank": 384,
        },
        "pruned_ctrl": {
            "kind": "prune",
            "enc_keep": enc_prune,
            "dec_keep": dec_prune,
            "vocab_rank": 384,
        },
    }


def build_loop_variant(
    base_model: nn.Module,
    variant_name: str,
    tokenizer=None,
    custom_spec: dict[str, Any] | None = None,
    in_place: bool = False,
) -> tuple[nn.Module, dict[str, Any]]:
    """
    Construct a named looping or pruning variant from a Qwen3-ASR checkpoint.

    :param base_model: Loaded ``Qwen3ASRForConditionalGeneration``.
    :param variant_name: One of :data:`LOOP_VARIANTS`.
    :param tokenizer: Required when the variant applies vocab factorisation.
    :param custom_spec: Explicit specification dict (used when reloading from ``loop_config.json``).
    :param in_place: Modify *base_model* in place instead of deep-copying.
    :return: ``(wrapped_model, spec)``.
    """
    n_enc = len(base_model.model.audio_tower.layers)
    n_dec = len(base_model.model.language_model.layers)
    if custom_spec is None:
        presets = variant_presets(n_enc=n_enc, n_dec=n_dec)
        if variant_name not in presets:
            raise ValueError(f"Unknown loop variant {variant_name!r}; expected one of {LOOP_VARIANTS}")
        spec = copy.deepcopy(presets[variant_name])
    else:
        spec = copy.deepcopy(custom_spec)
    spec["variant_name"] = variant_name
    spec.setdefault("orig_n_enc", n_enc)
    spec.setdefault("orig_n_dec", n_dec)
    kind = spec["kind"]

    if kind == "fold":
        enc_g = [LoopGroupSpec(**d) for d in spec["enc_groups"]]
        dec_g = [LoopGroupSpec(**d) for d in spec["dec_groups"]]
        out = apply_surgical_loop_folding(base_model, enc_g, dec_g, in_place=in_place)
    elif kind == "additive":
        enc_l = [AdditiveLoopSpec(**d) for d in spec["enc_loops"]]
        dec_l = [AdditiveLoopSpec(**d) for d in spec["dec_loops"]]
        out = apply_additive_loop(base_model, enc_l, dec_l, in_place=in_place)
    elif kind == "prune":
        out = prune_stack_control(base_model, spec["enc_keep"], spec["dec_keep"], in_place=in_place)
    else:
        raise ValueError(f"Unknown loop variant kind {kind!r}")

    vocab_rank = int(spec.get("vocab_rank", 0))
    if vocab_rank > 0:
        if tokenizer is None:
            raise ValueError(f"Variant {variant_name!r} requires a tokenizer for vocab_rank={vocab_rank}")
        spec["vocab_info"] = install_factorized_vocab(out, tokenizer, rank=vocab_rank, max_exact=1024)

    out._loop_variant_spec = spec
    return out, spec


def save_loop_checkpoint(model: nn.Module, processor, save_dir: Path) -> Path:
    """
    Save a looped/pruned Qwen3-ASR checkpoint with ``loop_config.json`` and deduplicated weights.

    :param model: Model built by :func:`build_loop_variant`.
    :param processor: Matching ``Qwen3ASRProcessor``.
    :param save_dir: Destination export directory.
    :return: *save_dir*.
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    spec = getattr(model, "_loop_variant_spec", None)
    if spec is None:
        raise ValueError("Model has no _loop_variant_spec; expected output of build_loop_variant()")

    model.config.save_pretrained(save_dir)
    if getattr(model, "generation_config", None) is not None:
        model.generation_config.save_pretrained(save_dir)
    processor.save_pretrained(save_dir)
    (save_dir / LOOP_CONFIG_FILE).write_text(json.dumps(spec, indent=2) + "\n")

    raw_sd = model.state_dict()
    sd: dict[str, torch.Tensor] = {}
    has_tied_embed = "model.language_model.embed_tokens.weight" in raw_sd
    for k, v in raw_sd.items():
        if k == "lm_head.weight" and has_tied_embed:
            continue
        sd[k] = v.detach().cpu().contiguous()
    save_file(sd, str(save_dir / "model.safetensors"), metadata={"format": "pt"})
    return save_dir


def load_loop_checkpoint(
    save_dir: Path,
    dtype: torch.dtype = torch.bfloat16,
    device: str = "cpu",
) -> tuple[nn.Module, Any, GenerationConfig]:
    """
    Reload a looped/pruned Qwen3-ASR checkpoint exported by :func:`save_loop_checkpoint`.

    :param save_dir: Checkpoint directory containing ``loop_config.json``.
    :param dtype: Target model dtype.
    :param device: Target torch device.
    :return: ``(model, processor, generation_config)``.
    """
    save_dir = Path(save_dir)
    spec = json.loads((save_dir / LOOP_CONFIG_FILE).read_text())
    config = Qwen3ASRConfig.from_pretrained(save_dir)
    config._attn_implementation = "sdpa"
    processor = AutoProcessor.from_pretrained(save_dir)

    if spec["kind"] == "prune":
        config.audio_config.encoder_layers = int(spec.get("orig_n_enc", max(spec["enc_keep"]) + 1))
        config.text_config.num_hidden_layers = int(spec.get("orig_n_dec", max(spec["dec_keep"]) + 1))

    with torch.device("cpu"):
        skeleton = Qwen3ASRForConditionalGeneration._from_config(config, dtype=dtype)
        model, _ = build_loop_variant(
            skeleton,
            spec["variant_name"],
            tokenizer=processor.tokenizer,
            custom_spec=spec,
            in_place=True,
        )

    sd = load_file(str(save_dir / "model.safetensors"))
    missing, unexpected = model.load_state_dict(sd, strict=False)
    real_missing = [k for k in missing if k != "lm_head.weight"]
    if real_missing or unexpected:
        raise ValueError(
            f"{save_dir}: state_dict mismatch (missing={real_missing[:5]}, unexpected={unexpected[:5]})"
        )
    if "lm_head.weight" in missing and hasattr(model.model.language_model.embed_tokens, "weight"):
        model.lm_head.weight = model.model.language_model.embed_tokens.weight

    # Move device only (do NOT cast buffers like rotary_emb.inv_freq out of fp32).
    for p in model.parameters():
        if p.is_floating_point() and p.dtype != dtype:
            p.data = p.data.to(dtype)
    model = model.to(device=device).eval()
    tbl = getattr(model.model.language_model.embed_tokens, "table", None)
    if tbl is not None:
        tbl.merge_for_eval(dtype)

    gen_cfg = (
        GenerationConfig.from_pretrained(save_dir)
        if (save_dir / "generation_config.json").exists()
        else model.generation_config
    )
    return model, processor, gen_cfg


def loop_parameter_summary(model: nn.Module) -> dict[str, float]:
    """
    Unique parameter counts (in millions), splitting embeddings and shared loop blocks.

    :param model: Model built by :func:`build_loop_variant`.
    :return: Parameter breakdown dictionary.
    """
    seen: set[int] = set()
    total = embed = enc = dec = other = 0
    embed_ids: set[int] = set()
    for p in model.model.language_model.embed_tokens.parameters():
        embed_ids.add(id(p))
    if hasattr(model, "lm_head"):
        for p in model.lm_head.parameters():
            embed_ids.add(id(p))

    for name, p in model.named_parameters():
        pid = id(p)
        if pid in seen:
            continue
        seen.add(pid)
        n = p.numel()
        total += n
        if pid in embed_ids:
            embed += n
        elif "audio_tower" in name or "multi_modal_projector" in name:
            enc += n
        elif "language_model" in name:
            dec += n
        else:
            other += n

    return {
        "total_M": round(total / 1e6, 2),
        "non_embed_M": round((total - embed) / 1e6, 2),
        "embed_M": round(embed / 1e6, 2),
        "enc_unshared_M": round(enc / 1e6, 2),
        "dec_unshared_M": round(dec / 1e6, 2),
        "shared_loop_M": round(other / 1e6, 2),
    }
