#!/bin/bash
# exp_burst.sh: Riley-only on GPU 7: default steps vs a 1024-row burst for long prompts (steady 256).
W=/mnt/efs/nlp/psyche/riley-gap-20261102; cd $W; M=Qwen2.5-7B-Instruct
B="--batch-token-budget 1024 --prefill-chunk-tokens 1024 --batch-shape-buckets 1,2,4,8,16,24,32,48,64,80,96,112,128,144,160,176,192,208,224,240,256,512,1024 --prefill-burst-steady-budget 256 --prefill-burst-max-decode-rows 32 --prefill-burst-min-prompt-tokens 512"
SC=closed-c8,closed-c32,mid-r16,rag-c1,rag6k-c1,rag-c8,rag-r3,reason-c8,reason-c32
for rep in 1 2; do
  python3 src/benchmarks/live/compare.py $W burst/burst-$rep $M 7 lp4 --order r --scenarios $SC --port 18330 --riley-long "$B" --riley-short "$B" | tail -1
  python3 src/benchmarks/live/compare.py $W burst/base-$rep $M 7 lp4 --order r --scenarios $SC --port 18330 | tail -1
done
echo BURST_DONE
