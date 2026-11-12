#!/bin/bash
# budget14.sh: Qwen2.5-Coder-14B closed-c8/c16/c32, Riley lp11 with step budgets 256 (default), 192, 128 on GPU 6; vLLM from recheck/short.
W=/mnt/efs/nlp/psyche/riley-gap-20261102; cd $W; M=Qwen2.5-Coder-14B-Instruct
for rep in 1 2; do for b in 192 128; do
  X=""; [ $b != 256 ] && X="--batch-token-budget $b --prefill-chunk-tokens $b --prefill-plateau-rows $([ $b = 192 ] && echo 128 || echo 96) --batch-shape-buckets $(python3 -c "print(','.join(str(x) for x in [1,2,4,8,16,24,32,48,64,80,96,112,128,144,160,176,192] if x<=$b))")"
  python3 src/benchmarks/live/compare.py $W budget14/b$b-$rep $M 6 lp11 --order r --scenarios closed-c8,closed-c16,closed-c32 --port 18347 --riley-short "$X" | tail -1
done; done
echo B14_DONE
