#!/bin/bash
# recheck6.sh: GPU 6, repeat the narrow misses with fresh rvvr runs.
W=/mnt/efs/nlp/psyche/riley-gap-20261102; cd $W
python3 src/benchmarks/live/compare.py $W recheck/Qwen2.5-Coder-14B-Instruct Qwen2.5-Coder-14B-Instruct 6 lp11 --order rvvr --scenarios closed-c32 --port 18345 --vllm-memory 0.9 | tail -1
B="--batch-token-budget 1024 --prefill-chunk-tokens 1024 --batch-shape-buckets 1,2,4,8,16,24,32,48,64,80,96,112,128,144,160,176,192,208,224,240,256,512,1024 --prefill-burst-steady-budget 256 --prefill-burst-max-decode-rows 32 --prefill-burst-min-prompt-tokens 512"
python3 src/benchmarks/live/compare.py $W recheck/Qwen2.5-7B-Instruct Qwen2.5-7B-Instruct 6 lp11 --order rvvr --scenarios rag-c8 --port 18345 --riley-long "$B" | tail -1
python3 src/benchmarks/live/compare.py $W recheck/Qwen2.5-14B-Instruct Qwen2.5-14B-Instruct 6 lp11 --order rvvr --scenarios rag-c8 --port 18345 --vllm-memory 0.9 --riley-long "$B --kv-blocks 1859" | tail -1
echo RECHECK_DONE
