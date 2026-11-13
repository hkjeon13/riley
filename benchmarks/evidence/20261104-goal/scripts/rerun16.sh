#!/bin/bash
# rerun16.sh GPU MODEL: repeat one rest16 model after an OOM caused by another tenant.
W=/mnt/efs/nlp/psyche/riley-gap-20261102; cd $W; G=$1; M=$2
U=$(nvidia-smi -i $G --query-gpu=uuid --format=csv,noheader); q=0
while [ $q -lt 4 ]; do q=$((q+1)); nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader | grep -q "$U" && q=0
  [ $(nvidia-smi -i $G --query-gpu=memory.used --format=csv,noheader,nounits) -gt 1000 ] && q=0; sleep 30; done
echo "$(date +%m-%d\ %H:%M) $M"
python3 src/benchmarks/live/compare.py $W rest16/$M $M $G lp16 --order rvvr --scenarios closed-c1,logprobs-c8,logprobs-c32 --port 1860$G --vllm-memory 0.8 | tail -n 1
echo RERUN_DONE
