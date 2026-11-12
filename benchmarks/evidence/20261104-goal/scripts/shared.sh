#!/bin/bash
cd /mnt/efs/nlp/psyche/riley-gap-20261102; until grep -q HN_DONE logs/hn4b.log; do sleep 15; done; sleep 10
python3 src/benchmarks/live/compare.py /mnt/efs/nlp/psyche/riley-gap-20261102 shared/Qwen2.5-7B-Instruct Qwen2.5-7B-Instruct 7 lp8 --order rvvr --scenarios shared-c8,shared-c32 --port 18330 --prefix-cache 4096 | tail -1
echo SHARED_DONE
