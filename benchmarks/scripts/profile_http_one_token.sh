#!/usr/bin/env bash

# Remote-only Nsight Compute attribution sentinel for a PR 15 implementation.
# The script is intended to run inside the pinned CUDA container. It uses only
# loopback networking and never writes prompt or generated text to the profiler
# output; the HTTP response is retained separately for correctness inspection.

set -euo pipefail
set -o noclobber

output_dir=${RUSTINFER_PROFILE_OUTPUT_DIR:-/out}
model_dir=${RUSTINFER_PROFILE_MODEL_DIR:-/model}
binary=${RUSTINFER_PROFILE_BINARY:-/usr/local/bin/rustinfer}
bind_port=${RUSTINFER_PROFILE_PORT:-18080}
residual_rmsnorm=${RUSTINFER_PROFILE_RESIDUAL_RMSNORM:-separate}
output_tokens=${RUSTINFER_PROFILE_OUTPUT_TOKENS:-2}
source_revision=${RUSTINFER_PROFILE_SOURCE_REVISION:?source revision is required}
source_archive_sha256=${RUSTINFER_PROFILE_SOURCE_ARCHIVE_SHA256:?source archive SHA-256 is required}
container_image_sha256=${RUSTINFER_PROFILE_CONTAINER_IMAGE_SHA256:?container image SHA-256 is required}
correctness_report_sha256=${RUSTINFER_PROFILE_CORRECTNESS_REPORT_SHA256:?correctness report SHA-256 is required}
profile_pid=
profile_status=0

stop_profile() {
  local pid=$profile_pid
  if [[ -z "$pid" ]]; then
    return
  fi
  if kill -0 "$pid" 2>/dev/null; then
    kill -INT "$pid" 2>/dev/null || true
    for _interrupt_attempt in $(seq 1 100); do
      if ! kill -0 "$pid" 2>/dev/null; then
        break
      fi
      sleep 0.1
    done
  fi
  if kill -0 "$pid" 2>/dev/null; then
    kill -TERM "$pid" 2>/dev/null || true
    for _terminate_attempt in $(seq 1 50); do
      if ! kill -0 "$pid" 2>/dev/null; then
        break
      fi
      sleep 0.1
    done
  fi
  if kill -0 "$pid" 2>/dev/null; then
    kill -KILL "$pid" 2>/dev/null || true
  fi
  if wait "$pid" 2>/dev/null; then
    profile_status=0
  else
    profile_status=$?
  fi
  profile_pid=
}

cleanup() {
  local exit_status=$?
  trap - EXIT INT TERM
  stop_profile
  exit "$exit_status"
}

trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

case "$residual_rmsnorm" in
  separate|fused) ;;
  *)
    echo "profile sentinel: residual RMSNorm must be separate or fused" >&2
    exit 2
    ;;
esac
case "$output_tokens" in
  1|2) ;;
  *)
    echo "profile sentinel: output tokens must be 1 (prefill) or 2 (prefill plus decode)" >&2
    exit 2
    ;;
esac
if [[ ! "$source_revision" =~ ^([0-9a-f]{40}|[0-9a-f]{64})$ ]] || \
  [[ ! "$source_archive_sha256" =~ ^[0-9a-f]{64}$ ]] || \
  [[ ! "$container_image_sha256" =~ ^sha256:[0-9a-f]{64}$ ]] || \
  [[ ! "$correctness_report_sha256" =~ ^[0-9a-f]{64}$ ]]; then
  echo "profile sentinel: provenance digests have an invalid format" >&2
  exit 2
fi

mkdir -p "$output_dir"
for evidence_name in \
  ncu.csv server.stdout server.stderr completion.response runtime-flag.txt \
  provenance.txt model-SHA256SUMS environment.txt SHA256SUMS; do
  if [[ -e "$output_dir/$evidence_name" ]]; then
    echo "profile sentinel: refusing to overwrite $evidence_name" >&2
    exit 2
  fi
done
printf '%s\n' "$residual_rmsnorm" >"$output_dir/runtime-flag.txt"
for model_name in rustinfer-checkpoint.json config.json tokenizer.json model.safetensors; do
  if [[ ! -f "$model_dir/$model_name" ]]; then
    echo "profile sentinel: required model artifact $model_name is absent" >&2
    exit 2
  fi
done
sha256sum \
  "$model_dir/rustinfer-checkpoint.json" \
  "$model_dir/config.json" \
  "$model_dir/tokenizer.json" \
  "$model_dir/model.safetensors" \
  >"$output_dir/model-SHA256SUMS"
{
  printf 'source_revision=%s\n' "$source_revision"
  printf 'source_archive_sha256=%s\n' "$source_archive_sha256"
  printf 'container_image_sha256=%s\n' "$container_image_sha256"
  printf 'correctness_report_sha256=%s\n' "$correctness_report_sha256"
  printf 'runtime_flag=residual_rmsnorm=%s\n' "$residual_rmsnorm"
  printf 'output_tokens=%s\n' "$output_tokens"
  printf 'command_contract=smollm2-c1-seq256-output%s-batch128-prefill128\n' \
    "$output_tokens"
} >"$output_dir/provenance.txt"
{
  nvidia-smi --query-gpu=name,uuid,pci.bus_id,compute_cap,memory.total,driver_version \
    --format=csv,noheader,nounits
  nvcc --version
} >"$output_dir/environment.txt"

ncu \
  --target-processes all \
  --clock-control none \
  --cache-control none \
  --csv \
  --page raw \
  --metrics gpu__time_duration.sum,launch__registers_per_thread,launch__occupancy_limit_registers,sm__warps_active.avg.pct_of_peak_sustained_active,dram__bytes_read.sum,dram__bytes_write.sum \
  --log-file "$output_dir/ncu.csv" \
  "$binary" serve \
  --model "$model_dir" \
  --model-id SmolLM2-135M \
  --bind "127.0.0.1:${bind_port}" \
  --max-active-sequences 1 \
  --max-waiting-requests 1 \
  --max-sequence-tokens 256 \
  --max-output-tokens 8 \
  --batch-token-budget 128 \
  --prefill-chunk-tokens 128 \
  --residual-rmsnorm "$residual_rmsnorm" \
  >"$output_dir/server.stdout" \
  2>"$output_dir/server.stderr" &
profile_pid=$!

ready=0
for _attempt in $(seq 1 600); do
  if { exec 4<>"/dev/tcp/127.0.0.1/${bind_port}"; } 2>/dev/null; then
    exec 4>&-
    ready=1
    break
  fi
  sleep 0.1
done
if [[ "$ready" -ne 1 ]]; then
  echo "profile sentinel: server did not become ready" >&2
  exit 1
fi

body=$(printf '{"model":"SmolLM2-135M","prompt":"Profile deterministic native tokens.","max_tokens":%s,"temperature":0.0,"top_p":1.0,"seed":7,"stream":false}' "$output_tokens")
exec 5<>"/dev/tcp/127.0.0.1/${bind_port}"
printf 'POST /v1/completions HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: %s\r\nConnection: close\r\n\r\n%s' \
  "${#body}" "$body" >&5
timeout 300 cat <&5 >"$output_dir/completion.response"
exec 5>&-
status_line=$(head -n 1 "$output_dir/completion.response" | tr -d '\r')
if [[ "$status_line" != "HTTP/1.1 200 OK" ]] || \
  ! grep -Fq "\"completion_tokens\":${output_tokens}" "$output_dir/completion.response" || \
  ! grep -Fq '"finish_reason":"length"' "$output_dir/completion.response"; then
  echo "profile sentinel: completion response failed the fixed-token HTTP contract" >&2
  exit 1
fi

# Nsight Compute has no service-aware shutdown hook. Interrupt the profiler
# only after the close-delimited completion has been received. Production
# graceful shutdown is covered by the PR 14/16 lifecycle gates, not this
# one-iteration attribution sentinel.
stop_profile
status=$profile_status
if [[ "$status" -ne 0 && "$status" -ne 9 && "$status" -ne 130 ]]; then
  echo "profile sentinel: ncu exited with status $status" >&2
  exit "$status"
fi
if [[ ! -s "$output_dir/ncu.csv" ]] || ! grep -Eq '^"[0-9]+"' "$output_dir/ncu.csv"; then
  echo "profile sentinel: NCU output is empty, truncated, or lacks the requested metric" >&2
  exit 1
fi
for metric_name in \
  gpu__time_duration.sum \
  launch__registers_per_thread \
  launch__occupancy_limit_registers \
  sm__warps_active.avg.pct_of_peak_sustained_active \
  dram__bytes_read.sum \
  dram__bytes_write.sum; do
  if ! grep -Fq "\"${metric_name}\"" "$output_dir/ncu.csv"; then
    echo "profile sentinel: NCU output lacks ${metric_name}" >&2
    exit 1
  fi
done
case "$residual_rmsnorm" in
  separate)
    if ! grep -Fq 'residual_add_kernel<' "$output_dir/ncu.csv" || \
      ! grep -Fq '::rms_norm_kernel<' "$output_dir/ncu.csv"; then
      echo "profile sentinel: separate arm lacks residual-add or standalone RMSNorm rows" >&2
      exit 1
    fi
    ;;
  fused)
    if ! grep -Fq 'residual_rms_norm_kernel<' "$output_dir/ncu.csv"; then
      echo "profile sentinel: fused arm lacks fused residual-RMSNorm rows" >&2
      exit 1
    fi
    ;;
esac

sha256sum "$binary" \
  "$output_dir/runtime-flag.txt" \
  "$output_dir/provenance.txt" \
  "$output_dir/model-SHA256SUMS" \
  "$output_dir/environment.txt" \
  "$output_dir/ncu.csv" \
  "$output_dir/completion.response" \
  "$output_dir/server.stdout" \
  "$output_dir/server.stderr" \
  >"$output_dir/SHA256SUMS"

trap - EXIT INT TERM
