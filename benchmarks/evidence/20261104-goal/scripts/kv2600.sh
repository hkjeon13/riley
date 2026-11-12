#!/bin/bash
W=/mnt/efs/nlp/psyche/riley-gap-20261102; cd $W
for M in Qwen2.5-14B-Instruct DeepSeek-R1-Distill-Qwen-14B; do
echo "$(date +%H:%M) $M"; python3 src/benchmarks/live/compare.py $W kv2600/$M $M 7 lp12 --order rvvr --scenarios reason-c32,reason-c8 --port 18397 --vllm-memory 0.9 --riley-long "--kv-blocks 2600" | tail -1
done
echo KV2600_DONE
