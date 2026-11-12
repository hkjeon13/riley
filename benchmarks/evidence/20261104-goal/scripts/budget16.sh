#!/bin/bash
# budget16.sh GPU LABEL BUDGET: Coder-14B closed cells on lp16 with a wider step budget (bytes unchanged), Riley vs vLLM rvvr, after 10 quiet minutes.
W=/mnt/efs/nlp/psyche/riley-gap-20261102; cd $W; G=$1; L=$2; N=$3; M=Qwen2.5-Coder-14B-Instruct
BK="1,2,4,8,16,24,32,48,64,80,96,112,128,144,160,176,192,208,224,240,256,512"; [ $N = 1024 ] && BK="$BK,1024"
UUID=$(nvidia-smi -i $G --query-gpu=uuid --format=csv,noheader); quiet=0
while [ $quiet -lt 4 ]; do
  if nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader | grep -q "$UUID"; then quiet=0; else quiet=$((quiet+1)); fi
  sleep 30
done
echo "$(date +%m-%d\ %H:%M) GPU $G quiet"
python3 src/benchmarks/live/compare.py $W $L/$M $M $G lp16 --order rvvr --scenarios closed-c1,closed-c8,closed-c16,closed-c32 --port 1839$G --vllm-memory 0.9 --riley-short "--batch-token-budget $N --prefill-chunk-tokens $N --batch-shape-buckets $BK" | tail -n 1
echo BUDGET16_${G}_DONE
