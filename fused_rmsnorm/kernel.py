"""Fused residual-add + RMSNorm in Triton.

Why fuse: both ops are memory-bound (a few FLOPs per byte), so runtime is set by
bytes moved through HBM, not by arithmetic. Eager PyTorch launches separate
kernels and writes/reads the intermediate `x + residual` (and the fp32 copies)
through global memory. This kernel loads x, residual and weight once, keeps the
row in registers, and writes y and residual_out once:

    bytes = (2 reads + 2 writes) * M * N * elem_size + N * elem_size

One program instance handles one row (one token). The row is reduced in fp32.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _fused_add_rmsnorm_kernel(
    X, R, W, Y, R_OUT,
    stride_x, stride_r, stride_y, stride_ro,
    N, eps,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N

    x = tl.load(X + row * stride_x + cols, mask=mask, other=0.0).to(tl.float32)
    r = tl.load(R + row * stride_r + cols, mask=mask, other=0.0).to(tl.float32)

    # Round the sum to the storage dtype first, exactly like the reference,
    # then normalise the rounded value.
    h_stored = (x + r).to(R_OUT.dtype.element_ty)
    tl.store(R_OUT + row * stride_ro + cols, h_stored, mask=mask)
    h = h_stored.to(tl.float32)

    # Masked lanes are 0, so they add nothing to the sum; divide by the true N.
    var = tl.sum(h * h, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + eps)

    w = tl.load(W + cols, mask=mask, other=0.0).to(tl.float32)
    y = h * rstd * w
    tl.store(Y + row * stride_y + cols, y.to(Y.dtype.element_ty), mask=mask)


def _pick_num_warps(block_n):
    # More warps for wider rows so each thread holds a manageable slice.
    if block_n >= 8192:
        return 16
    if block_n >= 2048:
        return 8
    return 4


def fused_add_rmsnorm(x, residual, weight, eps=1e-6):
    """x, residual: (..., N); weight: (N,). Returns (y, residual_out)."""
    assert x.shape == residual.shape, "x and residual must have the same shape"
    assert x.dtype == residual.dtype == weight.dtype, "dtypes must match"
    N = x.shape[-1]
    assert weight.shape == (N,), "weight must have shape (hidden_size,)"

    x2 = x.reshape(-1, N)
    r2 = residual.reshape(-1, N)
    # The kernel assumes unit stride along the hidden dim.
    if x2.stride(-1) != 1:
        x2 = x2.contiguous()
    if r2.stride(-1) != 1:
        r2 = r2.contiguous()
    weight = weight.contiguous()
    M = x2.shape[0]

    y = torch.empty_like(x2)
    r_out = torch.empty_like(r2)
    if M == 0:
        return y.reshape(x.shape), r_out.reshape(x.shape)

    BLOCK_N = triton.next_power_of_2(N)
    max_block = 65536 // x.element_size()
    if BLOCK_N > max_block:
        raise ValueError(f"hidden size {N} too large for single-block kernel (max {max_block})")

    _fused_add_rmsnorm_kernel[(M,)](
        x2, r2, weight, y, r_out,
        x2.stride(0), r2.stride(0), y.stride(0), r_out.stride(0),
        N, eps,
        BLOCK_N=BLOCK_N,
        num_warps=_pick_num_warps(BLOCK_N),
    )
    return y.reshape(x.shape), r_out.reshape(x.shape)
