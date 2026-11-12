#!/bin/bash
# shared13.sh: shared-c8/c32 on Qwen2.5-7B with lp13 (ID-based tokenizer), prefix cache 4096, rvvr, GPU 6, after stage1t on GPU 6.
W=/mnt/efs/nlp/psyche/riley-gap-20261102; cd $W
until grep -q STAGE1_DONE logs/stage1t-6.log; do sleep 30; done; sleep 10
python3 src/benchmarks/live/compare.py $W shared13/Qwen2.5-7B-Instruct Qwen2.5-7B-Instruct 6 lp13 --order rvvr --scenarios shared-c8,shared-c32 --port 18336 --prefix-cache 4096 | tail -1
echo SHARED13_DONE
