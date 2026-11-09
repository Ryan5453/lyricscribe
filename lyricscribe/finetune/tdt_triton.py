import torch
import triton
import triton.language as tl

MAX_BLOCK_U = 4096


@triton.jit
def _logaddexp(a, b):
    m = tl.maximum(a, b)
    safe = tl.where(m == float("-inf"), 0.0, m)
    out = safe + tl.log(tl.exp(a - safe) + tl.exp(b - safe))
    return tl.where(m == float("-inf"), float("-inf"), out)


@triton.jit
def _alpha_kernel(
    blank_ptr, label_ptr, dur_ptr, alpha_ptr, t_len_ptr, u_len_ptr, dvals_ptr,
    max_t, max_u, NUM_D: tl.constexpr, BLOCK_U: tl.constexpr,
):
    b = tl.program_id(0)
    t_len = tl.load(t_len_ptr + b)
    u_len = tl.load(u_len_ptr + b)
    base = b * max_t * max_u
    u = tl.arange(0, BLOCK_U)
    for n in range(1, t_len + u_len):
        t = n - u
        valid = (t >= 0) & (t < t_len) & (u <= u_len)
        acc = tl.full([BLOCK_U], float("-inf"), tl.float32)
        for i in tl.static_range(NUM_D):
            d = tl.load(dvals_ptr + i)
            src_ok = valid & (t - d >= 0)
            ok = src_ok & (d > 0)
            off = base + (t - d) * max_u + u
            a = tl.load(alpha_ptr + off, mask=ok, other=float("-inf"), cache_modifier=".cg")
            s = tl.load(blank_ptr + off, mask=ok, other=0.0) + tl.load(dur_ptr + off * NUM_D + i, mask=ok, other=0.0)
            acc = _logaddexp(acc, a + s)
            ok = src_ok & (u >= 1)
            off = off - 1
            a = tl.load(alpha_ptr + off, mask=ok, other=float("-inf"), cache_modifier=".cg")
            s = tl.load(label_ptr + off, mask=ok, other=0.0) + tl.load(dur_ptr + off * NUM_D + i, mask=ok, other=0.0)
            acc = _logaddexp(acc, a + s)
        tl.store(alpha_ptr + base + t * max_u + u, acc, mask=valid)
        tl.debug_barrier()


@triton.jit
def _beta_kernel(
    blank_ptr, label_ptr, dur_ptr, beta_ptr, t_len_ptr, u_len_ptr, dvals_ptr,
    max_t, max_u, NUM_D: tl.constexpr, BLOCK_U: tl.constexpr,
):
    b = tl.program_id(0)
    t_len = tl.load(t_len_ptr + b)
    u_len = tl.load(u_len_ptr + b)
    base = b * max_t * max_u
    u = tl.arange(0, BLOCK_U)
    for k in range(t_len + u_len):
        t = t_len - 1 + u_len - k - u
        valid = (t >= 0) & (t < t_len) & (u <= u_len)
        here = base + t * max_u + u
        blank = tl.load(blank_ptr + here, mask=valid, other=0.0)
        label = tl.load(label_ptr + here, mask=valid, other=0.0)
        acc = tl.full([BLOCK_U], float("-inf"), tl.float32)
        for i in tl.static_range(NUM_D):
            d = tl.load(dvals_ptr + i)
            dur = tl.load(dur_ptr + here * NUM_D + i, mask=valid, other=0.0)
            dst = here + d * max_u
            inner = valid & (d > 0) & (t + d < t_len)
            nxt = tl.load(beta_ptr + dst, mask=inner, other=float("-inf"), cache_modifier=".cg")
            nxt = tl.where(valid & (d > 0) & (t + d == t_len) & (u == u_len), 0.0, nxt)
            acc = _logaddexp(acc, tl.where(d > 0, blank + dur + nxt, float("-inf")))
            ok = valid & (t + d < t_len) & (u < u_len)
            nxt = tl.load(beta_ptr + dst + 1, mask=ok, other=float("-inf"), cache_modifier=".cg")
            acc = _logaddexp(acc, label + dur + nxt)
        tl.store(beta_ptr + here, acc, mask=valid)
        tl.debug_barrier()


def supports(max_u: int) -> bool:
    """
    Whether the kernels can hold a lattice this wide in one block.

    :param max_u: Lattice width ``U + 1``.
    :return: ``True`` if ``max_u`` fits.
    """
    return triton.next_power_of_2(max_u) <= MAX_BLOCK_U


def _launch(kernel, blank, label, duration, out, t_len, u_len, durations):
    """
    Run one recursion kernel, one program per sample.

    :param kernel: :func:`_alpha_kernel` or :func:`_beta_kernel`.
    :param blank: ``(B, T, U+1)`` blank arc scores.
    :param label: ``(B, T, U+1)`` label arc scores.
    :param duration: ``(B, T, U+1, D)`` duration log-probs.
    :param out: ``(B, T, U+1)`` lattice, pre-filled (written in place).
    :param t_len: ``(B,)`` frames per sample.
    :param u_len: ``(B,)`` labels per sample.
    :param durations: Duration values.
    :return: *out*.
    """
    bsz, max_t, max_u = blank.shape
    block_u = triton.next_power_of_2(max_u)
    dvals = torch.tensor(durations, dtype=torch.int32, device=blank.device)
    kernel[(bsz,)](
        blank, label, duration, out, t_len, u_len, dvals, max_t, max_u,
        NUM_D=len(durations), BLOCK_U=block_u, num_warps=min(16, max(1, block_u // 64)),
    )
    return out


def alpha(blank, label, duration, t_len, u_len, durations):
    """
    Forward variables of the TDT lattice.

    :param blank: ``(B, T, U+1)`` contiguous fp32 blank arc scores.
    :param label: ``(B, T, U+1)`` contiguous fp32 label arc scores.
    :param duration: ``(B, T, U+1, D)`` contiguous fp32 duration log-probs.
    :param t_len: ``(B,)`` int32 frames per sample.
    :param u_len: ``(B,)`` int32 labels per sample.
    :param durations: Duration values.
    :return: ``(B, T, U+1)`` log alphas, ``-inf`` outside each sample's lattice.
    """
    out = torch.full_like(blank, float("-inf"))
    out[:, 0, 0] = 0.0
    return _launch(_alpha_kernel, blank, label, duration, out, t_len, u_len, durations)


def beta(blank, label, duration, t_len, u_len, durations):
    """
    Backward variables of the TDT lattice.

    :param blank: ``(B, T, U+1)`` contiguous fp32 blank arc scores.
    :param label: ``(B, T, U+1)`` contiguous fp32 label arc scores.
    :param duration: ``(B, T, U+1, D)`` contiguous fp32 duration log-probs.
    :param t_len: ``(B,)`` int32 frames per sample.
    :param u_len: ``(B,)`` int32 labels per sample.
    :param durations: Duration values.
    :return: ``(B, T, U+1)`` log betas, ``-inf`` outside each sample's lattice.
    """
    out = torch.full_like(blank, float("-inf"))
    return _launch(_beta_kernel, blank, label, duration, out, t_len, u_len, durations)
