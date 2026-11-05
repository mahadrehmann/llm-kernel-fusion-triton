"""PyTorch reference: the semantics the Triton kernel must match.

Same contract as vLLM's `fused_add_rms_norm` used in every decoder layer of
Llama/Qwen-style models:

    residual_out = x + residual
    y            = residual_out / sqrt(mean(residual_out^2) + eps) * weight

The math is done in fp32 and the outputs are cast back to the input dtype.
"""
import torch


def fused_add_rmsnorm_ref(x, residual, weight, eps=1e-6):
    h = x.float() + residual.float()
    residual_out = h.to(x.dtype)
    # Normalise the value that is actually stored, so rounding matches the kernel.
    h = residual_out.float()
    rstd = torch.rsqrt(h.pow(2).mean(dim=-1, keepdim=True) + eps)
    y = (h * rstd * weight.float()).to(x.dtype)
    return y, residual_out


def eager_add_rmsnorm(x, residual, weight, eps=1e-6):
    """What a plain PyTorch model does: separate ops, each a full pass over memory."""
    residual_out = x + residual
    var = residual_out.float().pow(2).mean(-1, keepdim=True)
    y = (residual_out.float() * torch.rsqrt(var + eps)).to(x.dtype) * weight
    return y, residual_out
