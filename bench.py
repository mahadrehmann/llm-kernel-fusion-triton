"""Benchmark: eager PyTorch vs torch.compile vs Triton fused add+RMSNorm.

Needs a CUDA GPU (Colab T4, RunPod A100, ...). Reports median latency and
effective bandwidth = ideal bytes / time, where ideal bytes is the minimum
traffic for this op (read x, residual, weight; write y, residual_out).
All three implementations are scored against the SAME ideal byte count, so
GB/s shows how close each gets to the memory-bound limit.

    python bench.py                       # bf16, default shapes
    python bench.py --dtype fp16 --peak-gbps 320
"""
import argparse
import json
import platform

import torch
import triton
import triton.testing

from fused_rmsnorm import fused_add_rmsnorm, fused_add_rmsnorm_ref
from fused_rmsnorm.reference import eager_add_rmsnorm

# Spec-sheet peak HBM bandwidth (GB/s). Override with --peak-gbps for other GPUs.
PEAK_GBPS = {
    "Tesla T4": 320,
    "NVIDIA A100-SXM4-80GB": 2039,
    "NVIDIA A100 80GB PCIe": 1935,
    "NVIDIA A100-SXM4-40GB": 1555,
    "NVIDIA A100-PCIE-40GB": 1555,
    "NVIDIA L4": 300,
    "NVIDIA H100 80GB HBM3": 3350,
    "NVIDIA H100 PCIe": 2000,
    "NVIDIA GeForce RTX 4090": 1008,
    "NVIDIA GeForce RTX 3090": 936,
}
DTYPES = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}
TOKENS = [1, 16, 128, 512, 2048, 8192]
HIDDEN = [4096, 5120, 8192]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dtype", default="bf16", choices=DTYPES)
    p.add_argument("--peak-gbps", type=float, default=None)
    p.add_argument("--no-compile", action="store_true", help="skip torch.compile baseline")
    p.add_argument("--out", default="results")
    args = p.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("bench.py needs a CUDA GPU. Run tests with `pytest tests` for CPU correctness checks.")

    dtype = DTYPES[args.dtype]
    gpu = torch.cuda.get_device_name()
    peak = args.peak_gbps or PEAK_GBPS.get(gpu)
    compiled = None if args.no_compile else torch.compile(eager_add_rmsnorm)

    impls = {"eager": eager_add_rmsnorm, "triton": fused_add_rmsnorm}
    if compiled is not None:
        impls["torch.compile"] = compiled

    rows = []
    for N in HIDDEN:
        w = (1 + 0.1 * torch.randn(N, device="cuda")).to(dtype)
        for M in TOKENS:
            x = torch.randn(M, N, device="cuda", dtype=dtype)
            r = torch.randn(M, N, device="cuda", dtype=dtype)

            # Correctness gate: never report a speedup for a wrong answer.
            y, ro = fused_add_rmsnorm(x, r, w)
            y_ref, ro_ref = fused_add_rmsnorm_ref(x, r, w)
            torch.testing.assert_close(y, y_ref, atol=1.6e-2, rtol=1.6e-2)
            torch.testing.assert_close(ro, ro_ref, atol=1.6e-2, rtol=1.6e-2)

            ideal_bytes = (4 * M * N + N) * x.element_size()
            row = {"tokens": M, "hidden": N}
            for name, fn in impls.items():
                ms = triton.testing.do_bench(lambda: fn(x, r, w), warmup=25, rep=100)
                row[f"{name}_us"] = ms * 1e3
                row[f"{name}_gbps"] = ideal_bytes / (ms * 1e-3) / 1e9
            row["speedup_vs_eager"] = row["eager_us"] / row["triton_us"]
            if "torch.compile_us" in row:
                row["speedup_vs_compile"] = row["torch.compile_us"] / row["triton_us"]
            if peak:
                row["triton_pct_peak"] = 100 * row["triton_gbps"] / peak
            rows.append(row)
            print(row)

    meta = {"gpu": gpu, "peak_gbps": peak, "dtype": args.dtype,
            "torch": torch.__version__, "triton": triton.__version__, "python": platform.python_version()}
    with open(f"{args.out}.json", "w") as f:
        json.dump({"meta": meta, "rows": rows}, f, indent=2)

    has_c = compiled is not None
    lines = [f"**{gpu}** · {args.dtype} · torch {torch.__version__} · triton {triton.__version__}"
             + (f" · peak {peak:.0f} GB/s" if peak else ""), "",
             "| tokens × hidden | eager µs | " + ("compile µs | " if has_c else "") + "triton µs | "
             "× vs eager | " + ("× vs compile | " if has_c else "") + "triton GB/s | % peak |",
             "|---|---|" + ("---|" if has_c else "") + "---|---|" + ("---|" if has_c else "") + "---|---|"]
    for r_ in rows:
        cells = [f"{r_['tokens']} × {r_['hidden']}", f"{r_['eager_us']:.1f}"]
        if has_c:
            cells.append(f"{r_['torch.compile_us']:.1f}")
        cells += [f"{r_['triton_us']:.1f}", f"{r_['speedup_vs_eager']:.2f}"]
        if has_c:
            cells.append(f"{r_['speedup_vs_compile']:.2f}")
        cells += [f"{r_['triton_gbps']:.0f}", f"{r_['triton_pct_peak']:.0f}%" if peak else "–"]
        lines.append("| " + " | ".join(cells) + " |")
    with open(f"{args.out}.md", "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nWrote {args.out}.json and {args.out}.md")


if __name__ == "__main__":
    main()
