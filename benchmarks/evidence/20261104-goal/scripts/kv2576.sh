#!/bin/bash
W=/mnt/efs/nlp/psyche/riley-gap-20261102; cd $W
for M in Qwen2.5-Coder-14B-Instruct; do K=$(python3 kvb.py models/$M)
echo "$(date +%H:%M) $M $K"; python3 src/benchmarks/live/compare.py $W kv2600/$M $M 7 lp12 --order rvvr --scenarios reason-c32,reason-c8 --port 18397 --vllm-memory 0.9 --riley-long "$K" | tail -1
done
echo KV2576_DONE
