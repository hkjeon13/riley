#!/bin/bash
# chunk_exp.sh: GPU 7, reason and rag-c1 cells: vLLM vs Riley lp11 with KV chunk 1024 (default), 512, 256.
W=/mnt/efs/nlp/psyche/riley-gap-20261102; cd $W
for M in Qwen2.5-7B-Instruct Qwen2.5-0.5B-Instruct; do
  python3 src/benchmarks/live/compare.py $W chunk/$M/base $M 7 lp11 --order rvvr --scenarios reason-c8,reason-c32,rag-c1 --port 18344 | tail -1
  for c in 512 256; do
    for rep in 1 2; do
      python3 src/benchmarks/live/compare.py $W chunk/$M/c$c-$rep $M 7 lp11 --order r --scenarios reason-c8,reason-c32,rag-c1 --port 18344 --riley-long "--attention-kv-chunk-tokens $c" | tail -1
    done
  done
done
echo CHUNK_DONE
