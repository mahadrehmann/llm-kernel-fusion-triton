# Task: Fused Residual-Add + RMSNorm for LLM Decode and Prefill

An agent-facing optimisation task. The agent gets the starter code and the
evaluator; this repo's kernel is the reference solution.

## Objective

Replace the eager PyTorch `x + residual` -> RMSNorm pair (`fused_rmsnorm/reference.py`,
`eager_add_rmsnorm`) with a single GPU kernel that returns `(y, residual_out)`
matching the reference within dtype tolerance, and is faster.

## Workload

- Hidden sizes: 4096, 5120, 8192 (Llama-3-8B, Qwen-3.5-27B-class, 70B-class widths)
- Tokens: 1, 16 (decode), 128 (mixed), 512, 2048, 8192 (prefill)
- dtype for timing: fp16. bf16 and fp32 must be correct but are not timed on T4,
because T4 (sm_75) has no native bf16 and `torch.compile` does not optimise it there.
- Reference hardware: NVIDIA Tesla T4 (320 GB/s peak HBM). Thresholds below are
calibrated on T4 and must be re-calibrated before use on other GPUs.

## Starting code

- `fused_rmsnorm/reference.py`: correctness oracle (fp32 math) + eager baseline
- `fused_add_rmsnorm(x, residual, weight, eps)`: stub to implement, same signature
- Allowed: Triton, CUDA via `torch.utils.cpp_extension`. Not allowed: calling PyTorch
compute ops (add, pow, mean, rsqrt, mul) on the input tensors inside the function.

## Success criteria (all required)

1. `pytest tests -q`: all correctness cases pass (dtypes, odd/non-power-of-2 widths,
non-contiguous rows, 3-D input, empty batch, extreme magnitudes, zero rows,
inputs unmodified).
2. Hidden correctness set (evaluator only): fresh random seeds, plus widths 3, 127, 6144.
3. Performance, fp16, `triton.testing.do_bench` (warm-up, CUDA sync, median),
best of 2 full sweeps after a 2-second GPU warm-up:

- >= 3.5x faster than eager at every shape
- >= 0.85x vs `torch.compile` at every shape, and >= 1.25x at decode shapes
(<= 16 tokens) with hidden >= 5120
- >= 65% of peak HBM bandwidth at every prefill shape (>= 512 tokens),
ideal bytes = (4*M*N + N)*elem_size

### How the thresholds were set

Each threshold sits about 10% below the reference solution's worst result across
two independent T4 runs, so an equivalent solution passes despite run-to-run noise:

| Criterion | Reference worst case (2 runs) | Threshold |
|---|---:|---:|
| Speedup vs eager, all shapes | 3.93x | 3.5x |
| vs torch.compile, all shapes | 0.89x (1x4096) | 0.85x |
| vs torch.compile, decode, hidden >= 5120 | 1.37x | 1.25x |
| % of peak, prefill | 69.5% | 65% |

Small-batch hidden-4096 shapes varied by up to +/-20% between runs (they are timed
first, while the GPU leaves its idle clock state), which is why the evaluator warms
the GPU and takes the best of 2 sweeps.

## Intended failure modes (what a weak solution gets wrong)

- Dividing the mean by the padded block size instead of N -> wrong at odd widths
- Accumulating sum-of-squares in fp16 -> inf/NaN at large magnitudes
- Normalising the unrounded fp32 sum instead of the stored `residual_out` -> bf16 mismatch
- Assuming contiguous rows -> wrong on sliced inputs
- Unmasked tail loads -> out-of-bounds reads at non-power-of-2 widths

## Shortcuts the evaluator must reject

| Shortcut | Detection |
|---|---|
| Wrapping the PyTorch reference (no real kernel) | `torch.profiler` trace must show exactly 1 kernel launch, with no aten add/mean/rsqrt kernels |
| Caching outputs by input pointer or shape | Inputs re-randomised in place between timed runs; outputs re-checked after timing |
| Only handling power-of-2 widths | Hidden widths 3, 127, 6144 in the held-out set |
| Timing launch instead of completion | Evaluator times; agent code is never trusted to time itself |
| Writing into the input buffers to save memory | Inputs compared against pre-call clones |
| Loosening tolerances / editing tests | Evaluator ships its own copy of the tests and reference |

## Reference result

See `README.md` -> Results (fp16, Tesla T4, two runs).