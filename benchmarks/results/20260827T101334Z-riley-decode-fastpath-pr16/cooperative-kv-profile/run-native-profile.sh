#!/usr/bin/env bash
# Replays the final AB/BA cooperative shared-KV GQA campaign on ai-assistant.
set -euo pipefail

readonly profile=/input/riley-profile
readonly output_dir=/evidence/raw

mkdir -p "$output_dir"

common=(
  --model /model
  --prompts /workspace/benchmarks/prompts.jsonl
  --git-commit 436dad64526f228292c24cec73a652a72e0b1e38
  --git-dirty false
  --executable-sha256 a6870817235c80342878258a689973d979ebdcd0e04bc2638235a82a135b1ffc
  --runtime-flag-name decode_fast_path
  --semantic-class E0
  --correctness-gate-id pr16-decode-fast-path-exact-v1
  --correctness-report-sha256 61fade7f28c2459bf1e4fd8f9c5392b62d6672d14563afdb54aee975a6cd8c76
  --gpu-model 'NVIDIA GeForce RTX 4090'
  --gpu-uuid GPU-9087e425-6aca-b722-b8c9-cc0423b39fb0
  --device-index 0
  --gpu-pci-bus-id 00000000:01:00.0
  --gpu-compute-capability 8.9
  --gpu-vram-bytes 25757220864
  --environment-id ai-assistant-rtx4090-pr16-cooperative-kv-v1
  --cpu-model '13th Gen Intel(R) Core(TM) i7-13700K'
  --physical-core-count 16
  --logical-core-count 24
  --ram-bytes 67185598464
  --os-release 'Ubuntu 22.04.5 LTS'
  --kernel-release 6.8.0-138-generic
  --architecture x86_64
  --nvidia-driver-version 580.173.02
  --cuda-runtime-version 12.8.1
  --cuda-toolkit-version 12.8.93
  --cublas-version 12.8.4.1
  --container-image-sha256 f51d74009d8a5abd2aa0115ab51967aca200f99c0c0ffbafcff603212af258c1
  --workload-id smollm2-c1-p128-o32-greedy-v1
  --model-id HuggingFaceTB/SmolLM2-135M
  --model-revision 93efa2f097d58c2a74874c7e644dbc9b0cee75a2
  --weights-sha256 80521b40281d6ce74e35c9282c22539e75aa0ac8578892b2a59955ef78d55da1
  --tokenizer-sha256 9ca9acddb6525a194ec8ac7a87f24fbba7232a9a15ffa1af0c1224fcd888e47c
  --dtype bf16
  --concurrency 1
  --prompt-tokens 128
  --output-tokens 32
  --warmups 5
  --measured-iterations 30
  --sampling-id greedy
  --seed none
)

run_profile() {
  local role=$1
  local pair_index=$2
  local mode=$3
  local implementation_id=$4

  "$profile" \
    --output "${output_dir}/${role}-${pair_index}.json" \
    --role "$role" \
    --pair-index "$pair_index" \
    --run-id "decode-attention.436dad6.${role}${pair_index}" \
    --recorded-at-utc "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
    --implementation-id "$implementation_id" \
    --runtime-flag-value "$mode" \
    "${common[@]}"
}

# B,C,C,B,B,C,C,B,B,C keeps the two modes interleaved without changing their
# source, executable, model, or workload identity.
run_profile baseline 1 fixed-sync-cpu riley-decode-fixed-sync-cpu-v2
run_profile candidate 1 bucket-packed-gpu riley-decode-bucket-packed-gpu-cooperative-shared-kv-heads-v4
run_profile candidate 2 bucket-packed-gpu riley-decode-bucket-packed-gpu-cooperative-shared-kv-heads-v4
run_profile baseline 2 fixed-sync-cpu riley-decode-fixed-sync-cpu-v2
run_profile baseline 3 fixed-sync-cpu riley-decode-fixed-sync-cpu-v2
run_profile candidate 3 bucket-packed-gpu riley-decode-bucket-packed-gpu-cooperative-shared-kv-heads-v4
run_profile candidate 4 bucket-packed-gpu riley-decode-bucket-packed-gpu-cooperative-shared-kv-heads-v4
run_profile baseline 4 fixed-sync-cpu riley-decode-fixed-sync-cpu-v2
run_profile baseline 5 fixed-sync-cpu riley-decode-fixed-sync-cpu-v2
run_profile candidate 5 bucket-packed-gpu riley-decode-bucket-packed-gpu-cooperative-shared-kv-heads-v4
