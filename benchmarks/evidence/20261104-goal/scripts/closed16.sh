#!/bin/bash
# closed16.sh GPU VLLM_MEM MODELS...: once the GPU has had no other process for 10 minutes, closed-c8/c16/c32 Riley lp16 vs vLLM (rvvr). Rechecks the TPOT cells above 1.03.
W=/mnt/efs/nlp/psyche/riley-gap-20261102; cd $W; G=$1; VM=$2; shift 2; mkdir -p evidence/closed16
UUID=$(nvidia-smi -i $G --query-gpu=uuid --format=csv,noheader)
quiet=0
while [ $quiet -lt 20 ]; do
  if nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader | grep -q "$UUID"; then quiet=0; else quiet=$((quiet+1)); fi
  sleep 30
done
echo "$(date +%m-%d\ %H:%M) GPU $G quiet"
for M in "$@"; do
  echo "$(date +%H:%M) $M"; python3 src/benchmarks/live/compare.py $W closed16/$M $M $G lp16 --order rvvr --scenarios closed-c8,closed-c16,closed-c32 --port 1838$G --vllm-memory $VM | tail -n 1
done
echo CLOSED16_${G}_DONE
