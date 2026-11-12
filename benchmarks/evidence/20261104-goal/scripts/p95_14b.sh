#!/bin/bash
# p95_14b.sh: Qwen2.5-14B rag-c8/rag-r3 with the burst at prefill chunk 1024 vs 512, lp11, GPU 6 (vLLM from recheck/base0).
W=/mnt/efs/nlp/psyche/riley-gap-20261102; cd $W; M=Qwen2.5-14B-Instruct
BK="--batch-token-budget 1024 --batch-shape-buckets 1,2,4,8,16,24,32,48,64,80,96,112,128,144,160,176,192,208,224,240,256,512,1024 --prefill-burst-steady-budget 256 --prefill-burst-max-decode-rows 32 --prefill-burst-min-prompt-tokens 512 --kv-blocks 1859"
for rep in 1 2; do for c in 512 1024; do
  python3 src/benchmarks/live/compare.py $W p95/$M/c$c-$rep $M 6 lp11 --order r --scenarios rag-c8,rag-r3,rag-c1 --port 18348 --riley-long "$BK --prefill-chunk-tokens $c" | tail -1
done; done
echo P95_DONE
