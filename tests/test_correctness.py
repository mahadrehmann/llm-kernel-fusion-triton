"""Correctness harness: Triton kernel vs PyTorch reference.

Covers dtypes (fp32/fp16/bf16), power-of-2 and odd hidden sizes, single-token
and large batches, non-contiguous inputs, and extreme input scales.
Runs on GPU if available; otherwise in Triton's CPU interpreter
(TRITON_INTERPRET=1, set in conftest.py).
"""
import pytest
import torch

from fused_rmsnorm import fused_add_rmsnorm, fused_add_rmsnorm_ref

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Tolerances are set by the output dtype's precision, not loosened to pass.
TOL = {
    torch.float32: dict(atol=1e-5, rtol=1e-5),
    torch.float16: dict(atol=2e-3, rtol=2e-3),
    torch.bfloat16: dict(atol=1.6e-2, rtol=1.6e-2),
}
DTYPES = list(TOL)

# Hidden sizes: tiny, odd, non-power-of-2, and real model widths
# (Llama-3-8B 4096, Qwen-3.5-27B-class 5120, 8192).
SHAPES = [(1, 1), (1, 7), (3, 33), (17, 1000), (64, 4096), (5, 5120), (128, 8192)]
if DEVICE == "cpu":
    SHAPES = [s for s in SHAPES if s[0] * s[1] <= 70_000]  # keep the interpreter fast


def _make(M, N, dtype, scale=1.0, seed=0):
    g = torch.Generator(device="cpu").manual_seed(seed)
    x = (torch.randn(M, N, generator=g) * scale).to(dtype).to(DEVICE)
    r = (torch.randn(M, N, generator=g) * scale).to(dtype).to(DEVICE)
    w = (1 + 0.1 * torch.randn(N, generator=g)).to(dtype).to(DEVICE)
    return x, r, w


def _check(x, r, w, eps=1e-6):
    y, ro = fused_add_rmsnorm(x, r, w, eps)
    y_ref, ro_ref = fused_add_rmsnorm_ref(x, r, w, eps)
    assert y.dtype == x.dtype and ro.dtype == x.dtype
    assert not torch.isnan(y).any()
    torch.testing.assert_close(ro, ro_ref, **TOL[x.dtype])
    torch.testing.assert_close(y, y_ref, **TOL[x.dtype])


@pytest.mark.parametrize("dtype", DTYPES, ids=str)
@pytest.mark.parametrize("M,N", SHAPES)
def test_matches_reference(M, N, dtype):
    _check(*_make(M, N, dtype))


@pytest.mark.parametrize("dtype", DTYPES, ids=str)
@pytest.mark.parametrize("scale", [1e-3, 1e2])
def test_extreme_scales(dtype, scale):
    # Small values stress eps handling; large ones stress fp16 sum-of-squares overflow
    # (accumulation is fp32, so this must not produce inf/nan).
    _check(*_make(8, 1024, dtype, scale=scale))


@pytest.mark.parametrize("dtype", DTYPES, ids=str)
def test_3d_input(dtype):
    x, r, w = _make(2 * 9, 256, dtype)
    _check(x.view(2, 9, 256), r.view(2, 9, 256), w)


def test_non_contiguous_rows():
    # Row stride != N: a slice of a wider buffer.
    x, r, w = _make(16, 512, torch.float16)
    xb = torch.zeros(16, 768, dtype=x.dtype, device=DEVICE)
    xb[:, :512] = x
    _check(xb[:, :512], r, w)


def test_constant_rows():
    # Zero rows: output must be 0 (eps prevents divide-by-zero), not nan.
    x = torch.zeros(4, 128, dtype=torch.float16, device=DEVICE)
    r = torch.zeros_like(x)
    w = torch.ones(128, dtype=torch.float16, device=DEVICE)
    y, _ = fused_add_rmsnorm(x, r, w)
    assert torch.all(y == 0)


def test_empty_batch():
    x, r, w = _make(1, 64, torch.float32)
    y, ro = fused_add_rmsnorm(x[:0], r[:0], w)
    assert y.shape == (0, 64) and ro.shape == (0, 64)


def test_does_not_modify_inputs():
    x, r, w = _make(4, 300, torch.bfloat16)
    x0, r0 = x.clone(), r.clone()
    fused_add_rmsnorm(x, r, w)
    assert torch.equal(x, x0) and torch.equal(r, r0)
