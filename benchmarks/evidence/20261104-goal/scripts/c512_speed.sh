#!/bin/bash
# c512_speed.sh: GPU 7, reason and rag-c1 cells, vLLM vs Riley lp11 at chunk 1024 and 512, more models.
W=/mnt/efs/nlp/psyche/riley-gap-20261102; cd $W
for M in Qwen3-4B Meta-Llama-3.1-8B-Instruct Qwen2.5-1.5B-Instruct; do
  K=""; [ $M = Qwen3-4B ] && K="--kv-blocks 10050"; [ $M = Meta-Llama-3.1-8B-Instruct ] && K="--kv-blocks 7484"
  python3 src/benchmarks/live/compare.py $W chunk/$M/base $M 7 lp11 --order rvvr --scenarios reason-c8,reason-c32,rag-c1 --port 18344 --riley-long "$K" | tail -1
  for rep in 1 2; do
    python3 src/benchmarks/live/compare.py $W chunk/$M/c512-$rep $M 7 lp11 --order r --scenarios reason-c8,reason-c32,rag-c1 --port 18344 --riley-long "--attention-kv-chunk-tokens 512 $K" | tail -1
  done
done
echo C512S_DONE
