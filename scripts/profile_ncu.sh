#!/usr/bin/env bash
# Nsight Compute profile of the Triton kernel at one real shape (2048 tokens x 5120 hidden, bf16).
# Look at: "Memory Throughput" (% of peak DRAM) and "DRAM Bytes" — the evidence for the
# memory-bound claim in the README. Requires `ncu` (preinstalled on most CUDA images; not on Colab).
set -euo pipefail
cat > /tmp/one_shot.py <<'PY'
import torch, sys
sys.path.insert(0, ".")
from fused_rmsnorm import fused_add_rmsnorm
x = torch.randn(2048, 5120, device="cuda", dtype=torch.bfloat16)
r = torch.randn_like(x); w = torch.ones(5120, device="cuda", dtype=torch.bfloat16)
for _ in range(3): fused_add_rmsnorm(x, r, w)
torch.cuda.synchronize()
PY
ncu --kernel-name regex:_fused_add_rmsnorm_kernel --launch-skip 2 --launch-count 1 \
    --section SpeedOfLight --section MemoryWorkloadAnalysis \
    -o fused_add_rmsnorm_profile python /tmp/one_shot.py
echo "Open fused_add_rmsnorm_profile.ncu-rep in Nsight Compute, or: ncu --import fused_add_rmsnorm_profile.ncu-rep --page details"
