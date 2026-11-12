#!/bin/bash
W=/mnt/efs/nlp/psyche/riley-gap-20261102; cd $W
until grep -q STAGE1_DONE logs/stage1.log; do sleep 60; done; sleep 10
sed -i "s/budget = (0.9 \* 39.4 \* GIB - w - 4.5 \* GIB) if w > 20 \* GIB/budget = (0.9 * 39.4 * GIB - w - 2.5 * GIB) if w > 20 * GIB/" stage1.py
for M in Qwen2.5-14B-Instruct Qwen2.5-Coder-14B-Instruct DeepSeek-R1-Distill-Qwen-14B Qwen3-14B; do
  [ -d evidence/stage1/$M ] && mv evidence/stage1/$M evidence/stage1-lp8-oom-$M
done
python3 stage1.py $W lp10 6 stage1 Qwen2.5-14B-Instruct,Qwen2.5-Coder-14B-Instruct,DeepSeek-R1-Distill-Qwen-14B,Qwen3-14B
echo STAGE1_14B_DONE
