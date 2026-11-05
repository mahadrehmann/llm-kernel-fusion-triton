#!/usr/bin/env bash
# Serving benchmark for the TelecomGPT-R1 (27B, BF16) deployment on one A100 80GB.
# Run on the RunPod pod while the server is up (same config as production):
#   vllm serve KU-DFI/TelecomGPT-R1 --max-model-len 8192 --gpu-memory-utilization 0.90
# Produces throughput, TTFT, TPOT and ITL (mean/median/p99) per concurrency level.
# ~20-30 min total on an A100 => well under $1 of pod time.
set -euo pipefail
MODEL=${MODEL:-KU-DFI/TelecomGPT-R1}
BASE_URL=${BASE_URL:-http://localhost:8000}
mkdir -p serve_results
for C in 1 4 16 32; do
  vllm bench serve \
    --backend vllm --base-url "$BASE_URL" --model "$MODEL" \
    --dataset-name random --random-input-len 1024 --random-output-len 512 \
    --ignore-eos \
    --num-prompts $(( C * 8 > 32 ? C * 8 : 32 )) --max-concurrency "$C" \
    --percentile-metrics ttft,tpot,itl,e2el --metric-percentiles 50,99 \
    --save-result --result-dir serve_results --result-filename "c${C}.json"
done
echo "Done. Per-concurrency JSON in serve_results/. Paste the printed summary tables into the README / resume."
