#!/bin/bash
# tp32.sh: Qwen2.5-32B on GPUs 6+7 with tensor parallelism 2, Riley lp10 vs vLLM (rvvr), closed c1..c32.
W=/mnt/efs/nlp/psyche/riley-gap-20261102; cd $W
until grep -q SHORT_DONE logs/short.log && grep -q STAGE1_14B_DONE logs/stage1_14b.log && grep -q LP11_DONE logs/lp11.log; do sleep 60; done; sleep 20
python3 src/benchmarks/live/compare.py $W tp32/Qwen2.5-32B-Instruct Qwen2.5-32B-Instruct 6,7 lp10 --tp 2 --order rvvr --scenarios closed-c1,closed-c8,closed-c16,closed-c32 --port 18350 --vllm-memory 0.9 | tail -1
python3 src/benchmarks/live/summarize.py evidence/tp32/Qwen2.5-32B-Instruct
echo TP32_DONE
