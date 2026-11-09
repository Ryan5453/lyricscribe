import torch

# Elements per fp32 chunk of the joint logits in :class:`_TokenLogProbs` (1 GiB).
_CHUNK_ELEMENTS = 2**28


class _TokenLogProbs(torch.autograd.Function):
    """
    Blank and next-label log-probabilities of the joint token logits,
    computed in chunks of frames so the full fp32 log-softmax is never
    materialized, forward or backward.
    """

    @staticmethod
    def forward(ctx, logits, labels, blank_token_id):
        """
        :param logits: ``(B, T, U+1, V)`` joint token logits, any float dtype.
        :param labels: ``(B, 1, U+1, 1)`` next-label id per lattice column.
        :param blank_token_id: Blank token id.
        :return: ``(blank, label)``, each ``(B, T, U+1)`` fp32 log-probs.
        """
        bsz, max_t, max_u, vocab = logits.shape
        chunk = max(1, _CHUNK_ELEMENTS // max(1, bsz * max_u * vocab))
        lse = torch.empty((bsz, max_t, max_u), device=logits.device, dtype=torch.float32)
        blank, label = torch.empty_like(lse), torch.empty_like(lse)
        for t0 in range(0, max_t, chunk):
            x = logits[:, t0:t0 + chunk].float()
            norm = torch.logsumexp(x, dim=-1)
            lse[:, t0:t0 + chunk] = norm
            blank[:, t0:t0 + chunk] = x[..., blank_token_id] - norm
            label[:, t0:t0 + chunk] = x.gather(-1, labels.expand(-1, x.shape[1], -1, -1)).squeeze(-1) - norm
        ctx.save_for_backward(logits, lse, labels)
        ctx.blank_token_id = blank_token_id
        ctx.chunk = chunk
        return blank, label

    @staticmethod
    def backward(ctx, g_blank, g_label):
        """
        Softmax-weighted gradient, chunk by chunk, written in the logits' dtype.

        :param g_blank: ``(B, T, U+1)`` upstream gradient of the blank log-probs.
        :param g_label: ``(B, T, U+1)`` upstream gradient of the label log-probs.
        :return: Gradient for ``logits`` (others ``None``).
        """
        logits, lse, labels = ctx.saved_tensors
        grad = torch.empty_like(logits)
        for t0 in range(0, logits.shape[1], ctx.chunk):
            span = slice(t0, t0 + ctx.chunk)
            gb, gl = g_blank[:, span], g_label[:, span]
            x = logits[:, span].float()
            gx = torch.exp(x - lse[:, span, :, None]) * -(gb + gl)[..., None]
            gx[..., ctx.blank_token_id] += gb
            gx.scatter_add_(-1, labels.expand(-1, x.shape[1], -1, -1), gl[..., None])
            grad[:, span] = gx
        return grad, None, None


def _alpha_torch(blank, label, duration, t_len, u_len, durations):
    """
    Forward variables by anti-diagonals ``t + u = n`` (duration-0 label arcs
    couple states within a frame), vectorized over each diagonal.

    :param blank: ``(B, T, U+1)`` blank arc scores.
    :param label: ``(B, T, U+1)`` label arc scores.
    :param duration: ``(B, T, U+1, D)`` duration log-probs.
    :param t_len: ``(B,)`` frames per sample.
    :param u_len: ``(B,)`` labels per sample.
    :param durations: Duration values.
    :return: ``(B, T, U+1)`` log alphas.
    """
    _, max_t, max_u = blank.shape
    dev = blank.device
    neg_inf = float("-inf")
    alpha = torch.full_like(blank, neg_inf)
    alpha[:, 0, 0] = 0.0
    for n in range(1, max_t + max_u - 1):
        u0, u1 = max(0, n - max_t + 1), min(n + 1, max_u)
        u_idx = torch.arange(u0, u1, device=dev)
        t_idx = n - u_idx
        cands = []
        for i, d in enumerate(durations):
            t_src = (t_idx - d).clamp(min=0)
            if d > 0 and u0 <= n - d:
                c = alpha[:, t_src, u_idx] + blank[:, t_src, u_idx] + duration[:, t_src, u_idx, i]
                cands.append(c.masked_fill(t_idx - d < 0, neg_inf))
            if max(u0, 1) <= min(u1 - 1, n - d):
                u_src = (u_idx - 1).clamp(min=0)
                c = alpha[:, t_src, u_src] + label[:, t_src, u_src] + duration[:, t_src, u_src, i]
                cands.append(c.masked_fill((t_idx - d < 0) | (u_idx == 0), neg_inf))
        if cands:
            alpha[:, t_idx, u_idx] = torch.logsumexp(torch.stack(cands), dim=0)
    return alpha


def _beta_torch(blank, label, duration, t_len, u_len, durations):
    """
    Backward variables by anti-diagonals, mirroring :func:`_alpha_torch`.

    :param blank: ``(B, T, U+1)`` blank arc scores.
    :param label: ``(B, T, U+1)`` label arc scores.
    :param duration: ``(B, T, U+1, D)`` duration log-probs.
    :param t_len: ``(B,)`` frames per sample.
    :param u_len: ``(B,)`` labels per sample.
    :param durations: Duration values.
    :return: ``(B, T, U+1)`` log betas, ``-inf`` outside each sample's lattice.
    """
    _, max_t, max_u = blank.shape
    dev = blank.device
    neg_inf = float("-inf")
    t_end, u_end = t_len.long()[:, None], u_len.long()[:, None]
    beta = torch.full_like(blank, neg_inf)
    for n in range(max_t + max_u - 2, -1, -1):
        u_idx = torch.arange(max(0, n - max_t + 1), min(n + 1, max_u), device=dev)
        t_idx = n - u_idx
        cands = []
        for i, d in enumerate(durations):
            t_dst = t_idx + d
            t_dst_c = t_dst.clamp(max=max_t - 1)
            if d > 0:
                inner = (t_dst[None] < t_end) & (u_idx[None] <= u_end)
                term = (t_dst[None] == t_end) & (u_idx[None] == u_end)
                nxt = torch.where(inner, beta[:, t_dst_c, u_idx], torch.where(term, 0.0, neg_inf))
                cands.append(blank[:, t_idx, u_idx] + duration[:, t_idx, u_idx, i] + nxt)
            if max_u > 1:
                u_dst = (u_idx + 1).clamp(max=max_u - 1)
                ok = (t_dst[None] < t_end) & (u_idx[None] < u_end)
                nxt = torch.where(ok, beta[:, t_dst_c, u_dst], neg_inf)
                cands.append(label[:, t_idx, u_idx] + duration[:, t_idx, u_idx, i] + nxt)
        beta[:, t_idx, u_idx] = torch.logsumexp(torch.stack(cands), dim=0)
    return beta


def _recursions(blank, label, duration, t_len, u_len, durations):
    """
    The Triton kernels on CUDA, the torch loops elsewhere.

    :param blank: ``(B, T, U+1)`` blank arc scores.
    :param label: ``(B, T, U+1)`` label arc scores.
    :param duration: ``(B, T, U+1, D)`` duration log-probs.
    :param t_len: ``(B,)`` frames per sample.
    :param u_len: ``(B,)`` labels per sample.
    :param durations: Duration values.
    :return: ``(alpha_fn, beta_fn)``.
    """
    if blank.is_cuda:
        from lyricscribe.finetune import tdt_triton

        if tdt_triton.supports(blank.shape[2]):
            return tdt_triton.alpha, tdt_triton.beta
    return _alpha_torch, _beta_torch


def _log_z(alpha, blank, duration, t_len, u_len, durations):
    """
    Total log-likelihood: a final blank from ``(T - d, U)`` landing exactly on ``T``.

    :param alpha: ``(B, T, U+1)`` log alphas.
    :param blank: ``(B, T, U+1)`` blank arc scores.
    :param duration: ``(B, T, U+1, D)`` duration log-probs.
    :param t_len: ``(B,)`` frames per sample.
    :param u_len: ``(B,)`` labels per sample.
    :param durations: Duration values.
    :return: ``(B,)`` log partition values.
    """
    b_idx = torch.arange(blank.shape[0], device=blank.device)
    t_end, u_end = t_len.long(), u_len.long()
    terms = []
    for i, d in enumerate(durations):
        if d == 0:
            continue
        t_src = (t_end - d).clamp(min=0)
        term = alpha[b_idx, t_src, u_end] + blank[b_idx, t_src, u_end] + duration[b_idx, t_src, u_end, i]
        terms.append(term.masked_fill(t_end - d < 0, float("-inf")))
    return torch.logsumexp(torch.stack(terms), dim=0)


def _arc_grads(alpha, beta, blank, label, duration, t_len, u_len, durations, log_z):
    """
    Gradients of the NLL w.r.t. the arc scores: minus each arc's occupancy
    ``exp(alpha + arc + beta_next - logZ)``, all lattice cells at once.

    :param alpha: ``(B, T, U+1)`` log alphas.
    :param beta: ``(B, T, U+1)`` log betas.
    :param blank: ``(B, T, U+1)`` blank arc scores.
    :param label: ``(B, T, U+1)`` label arc scores.
    :param duration: ``(B, T, U+1, D)`` duration log-probs.
    :param t_len: ``(B,)`` frames per sample.
    :param u_len: ``(B,)`` labels per sample.
    :param durations: Duration values.
    :param log_z: ``(B,)`` log partition values.
    :return: Gradients for ``blank``, ``label`` and ``duration``.
    """
    bsz, max_t, max_u = blank.shape
    neg_inf = float("-inf")
    nxt_label = torch.full((bsz, max_t + max(durations) + 1, max_u + 1), neg_inf, device=blank.device, dtype=blank.dtype)
    nxt_label[:, :max_t, :max_u] = beta
    nxt_blank = nxt_label.clone()
    nxt_blank[torch.arange(bsz, device=blank.device), t_len.long(), u_len.long()] = 0.0
    finite = torch.isfinite(log_z)[:, None, None]
    a = torch.where(finite, alpha - log_z[:, None, None], neg_inf)
    g_blank = torch.zeros_like(blank)
    g_label = torch.zeros_like(label)
    g_dur = torch.zeros_like(duration)
    for i, d in enumerate(durations):
        if d > 0:
            occ = torch.exp(a + blank + duration[..., i] + nxt_blank[:, d:d + max_t, :max_u])
            g_blank -= occ
            g_dur[..., i] -= occ
        occ = torch.exp(a + label + duration[..., i] + nxt_label[:, d:d + max_t, 1:max_u + 1])
        g_label -= occ
        g_dur[..., i] -= occ
    return g_blank, g_label, g_dur


class _TDTLattice(torch.autograd.Function):
    """
    TDT forward-backward over precomputed arc scores, with the analytic
    gradient (arc occupancies) instead of autograd through the recursion.
    """

    @staticmethod
    def forward(ctx, blank, label, duration, logit_lengths, target_lengths, durations):
        """
        :param blank: ``(B, T, U+1)`` blank log-probs.
        :param label: ``(B, T, U+1)`` log-prob of the next target label (last column unused).
        :param duration: ``(B, T, U+1, D)`` duration log-probs.
        :param logit_lengths: ``(B,)`` valid frames per sample.
        :param target_lengths: ``(B,)`` labels per sample.
        :param durations: Duration values, e.g. ``[0, 1, 2, 3, 4]``.
        :return: ``(B,)`` negative log-likelihoods.
        """
        blank, label, duration = blank.contiguous(), label.contiguous(), duration.contiguous()
        t_len = logit_lengths.to(device=blank.device, dtype=torch.int32).contiguous()
        u_len = target_lengths.to(device=blank.device, dtype=torch.int32).contiguous()
        alpha_fn, _ = _recursions(blank, label, duration, t_len, u_len, durations)
        alpha = alpha_fn(blank, label, duration, t_len, u_len, durations)
        log_z = _log_z(alpha, blank, duration, t_len, u_len, durations)
        ctx.save_for_backward(blank, label, duration, alpha, log_z, t_len, u_len)
        ctx.durations = list(durations)
        return -log_z

    @staticmethod
    def backward(ctx, grad_out):
        """
        Beta recursion, then the arc gradients scaled by the upstream gradient.

        :param grad_out: ``(B,)`` upstream gradient.
        :return: Gradients for ``blank``, ``label``, ``duration`` (others ``None``).
        """
        blank, label, duration, alpha, log_z, t_len, u_len = ctx.saved_tensors
        durations = ctx.durations
        _, beta_fn = _recursions(blank, label, duration, t_len, u_len, durations)
        beta = beta_fn(blank, label, duration, t_len, u_len, durations)
        g_blank, g_label, g_dur = _arc_grads(alpha, beta, blank, label, duration, t_len, u_len, durations, log_z)
        scale = grad_out[:, None, None]
        return g_blank * scale, g_label * scale, g_dur * scale[..., None], None, None, None


def tdt_loss(
    token_logits: torch.Tensor,
    duration_logits: torch.Tensor,
    targets: torch.Tensor,
    logit_lengths: torch.Tensor,
    target_lengths: torch.Tensor,
    blank_token_id: int,
    durations: list[int],
    sigma: float = 0.0,
) -> torch.Tensor:
    """
    Per-sample TDT negative log-likelihood (https://arxiv.org/abs/2304.06795),
    numerically equal to ``transformers.loss.loss_tdt.tdt_loss(...,
    reduction="none")`` but with an analytic backward instead of autograd
    through the lattice recursion.

    :param token_logits: ``(B, T, U+1, V)`` token logits (blank included).
    :param duration_logits: ``(B, T, U+1, D)`` duration logits.
    :param targets: ``(B, U)`` target labels.
    :param logit_lengths: ``(B,)`` encoder lengths.
    :param target_lengths: ``(B,)`` target lengths.
    :param blank_token_id: Blank token id.
    :param durations: Duration values.
    :param sigma: Logit undernormalization constant.
    :return: ``(B,)`` losses.
    """
    bsz, _, max_u, _ = token_logits.shape
    # The last column has no next label; its (unused) arcs read token 0.
    labels = torch.zeros((bsz, 1, max_u, 1), dtype=torch.long, device=token_logits.device)
    labels[:, 0, : max_u - 1, 0] = targets[:, : max_u - 1].long()
    blank, label = _TokenLogProbs.apply(token_logits, labels, blank_token_id)
    duration_lp = torch.log_softmax(duration_logits.float(), dim=-1)
    return _TDTLattice.apply(
        blank - sigma, label - sigma, duration_lp, logit_lengths, target_lengths, list(durations)
    )
