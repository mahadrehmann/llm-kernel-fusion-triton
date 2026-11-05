# Fused Residual-Add + RMSNorm in Triton

A single Triton kernel that replaces the `x + residual` → `RMSNorm` pair that runs
**twice in every decoder layer** of Llama/Qwen-style LLMs (before attention and
before the MLP). Same contract as vLLM's `fused_add_rms_norm`:

```
residual_out = x + residual
y            = residual_out * rsqrt(mean(residual_out²) + eps) * weight
```

## Results

> Run `python bench.py` on a GPU and paste `results.md` here. bf16, hidden sizes
> 4096 / 5120 / 8192, 1 to 8192 tokens.

<!-- RESULTS:START -->
_Not yet measured._
<!-- RESULTS:END -->

## Where the speedup comes from

Both ops are **memory-bound**: about 4 FLOPs per element against 2–8 bytes moved
per element, far below the ridge point of any modern GPU (A100 ≈ 150 FLOP/byte in bf16).
Runtime is therefore set by bytes moved through HBM, not arithmetic.

| | HBM traffic per token row (N = hidden, b = bytes/elem) |
|---|---|
| **Fused (this kernel)** | read x, residual, weight; write y, residual_out ≈ **4·N·b** |
| Eager PyTorch | `add` (3·N·b) + fp32 upcast, `pow`, `mean`, `rsqrt`, `mul`, downcast, `mul` (each a separate kernel with its own read/write) ≈ **several times more**, plus one launch per op |

So the expected gain is the ratio of bytes moved, and it is largest where launch
overhead dominates (decode: 1–16 tokens). `torch.compile` already fuses most of
this, so the honest comparison is against compile. The kernel should be roughly on
par with compile at large batch and only win where Inductor's generated kernel
reaches less bandwidth. `bench.py` reports **effective GB/s against the same ideal
byte count for every implementation**, plus % of peak HBM bandwidth. That number,
not the speedup, is the evidence: near 80–90 % of peak means there is little left
to gain.

Design choices:
- **One program per row (token).** A row of ≤ 8192 bf16 values (16 KB) fits in
  registers, so the sum of squares and the normalisation share one load.
- **fp32 accumulation.** In fp16, `sum(h²)` over 8192 elements overflows (max 65504);
  `test_extreme_scales` covers this.
- **Round, then normalise.** The sum is cast to the storage dtype before it is
  squared, so `y` is computed from exactly the `residual_out` the next layer sees.
- **Masked tail.** `BLOCK_N` is the next power of 2; masked lanes load 0 and the
  mean divides by the true `N`, not `BLOCK_N` (the mutation test for this bug
  fails 11 of 28 cases).
- `num_warps` scales with row width (4 / 8 / 16).

## Correctness

`tests/test_correctness.py`, 28 cases, compared against an fp32 PyTorch reference:
- dtypes fp32 / fp16 / bf16, tolerance set by each dtype's precision
- hidden sizes 1, 7, 33, 1000, 4096, 5120, 8192 (odd and non-power-of-2 included)
- 1 to 128 rows, 3-D input, non-contiguous row stride, empty batch
- tiny (1e-3) and large (1e2) magnitudes, all-zero rows (eps path, no NaN)
- inputs left unmodified

`bench.py` also re-checks correctness at every benchmarked shape before timing.

## Run it

```bash
pip install -r requirements.txt
pytest tests -q                  # GPU, or CPU via Triton's interpreter (auto)
python bench.py                  # GPU only: writes results.md / results.json
bash scripts/profile_ncu.sh      # Nsight Compute: DRAM throughput for one shape
```

Free GPU: Google Colab → Runtime → T4 GPU, then
`!git clone <repo> && cd <repo> && pip install -q pytest && pytest tests -q && python bench.py`.

## Limitations / next steps

- Single-block rows only (hidden ≤ 32768 in bf16). Wider rows need a two-pass split reduction.
- Forward only; no backward pass.
- Not autotuned. An `@triton.autotune` sweep over `num_warps` is the obvious next experiment.
- Possible extension: fuse FP8 output quantisation (per-token scale) into the same pass, as vLLM does for FP8 models.
